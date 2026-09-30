"""h3t writer: lay ComfyUI-layout tensors out block-contiguous and aligned (format described in store.py)."""
from __future__ import annotations

import json
import os
import re
import struct
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .config import H3Config
from .store import (
    _BLOCK_RE, _QUANT_PARTS, BLOCK_ALIGN, FORMAT, PAD_PREFIX, TENSOR_ALIGN, VERSION, BlockEntry, H3TError, _align, _typed,
    nbytes_of, quant_linear_names, st_dtype,
)

# --------------------------------------------------------------------------------------------- sources to write


class TensorSource:
    """Raw little-endian bytes of one tensor, produced lazily so a writer never holds more than one chunk."""

    dtype: str
    shape: Tuple[int, ...]
    nbytes: int

    def chunks(self, chunk_bytes: int) -> Iterator[memoryview]:
        raise NotImplementedError

    def read(self) -> torch.Tensor:
        """The whole tensor (for small ones: codebooks, scales, JSON)."""
        data = bytearray(self.nbytes)
        pos = 0
        for c in self.chunks(32 << 20):
            data[pos : pos + len(c)] = c
            pos += len(c)
        buf = torch.frombuffer(data, dtype=torch.uint8) if self.nbytes else torch.empty(0, dtype=torch.uint8)
        return _typed(buf, 0, self.dtype, self.shape)


class TorchSource(TensorSource):
    def __init__(self, t: torch.Tensor):
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        self._t = t.contiguous()
        self.dtype, self.shape = st_dtype(t.dtype), tuple(t.shape)
        self.nbytes = self._t.numel() * self._t.element_size()

    def chunks(self, chunk_bytes: int) -> Iterator[memoryview]:
        if self.nbytes == 0:
            return
        mv = memoryview(self._t.reshape(-1).view(torch.uint8).numpy())
        for a in range(0, self.nbytes, chunk_bytes):
            yield mv[a : a + chunk_bytes]


class LazySource(TensorSource):
    """A tensor produced by `fn()` only when it is written (synthetic and streaming-conversion sources)."""

    def __init__(self, dtype: str, shape: Sequence[int], fn: Callable[[], torch.Tensor]):
        self.dtype, self.shape, self._fn = dtype, tuple(shape), fn
        self.nbytes = nbytes_of(dtype, shape)

    def chunks(self, chunk_bytes: int) -> Iterator[memoryview]:
        src = TorchSource(self._fn())
        if (src.dtype, src.shape) != (self.dtype, self.shape):
            raise H3TError(f"lazy tensor produced {src.dtype}{src.shape}, declared {self.dtype}{self.shape}")
        yield from src.chunks(chunk_bytes)


def as_source(x: Union[TensorSource, torch.Tensor]) -> TensorSource:
    if isinstance(x, TensorSource):
        return x
    if isinstance(x, torch.Tensor):
        return TorchSource(x)
    raise TypeError(f"expected a Tensor or TensorSource, got {type(x).__name__}")


_BLOCK_CANON = (
    "adaln_proj.linear.weight", "adaln_proj.linear.bias", "norm1.weight", "attn.q_norm.weight", "attn.k_norm.weight",
    *(f"attn.qkv_proj.{p}" for p in _QUANT_PARTS), *(f"attn.out_proj.{p}" for p in _QUANT_PARTS),
    "norm2.weight", *(f"mlp.fc1.{p}" for p in _QUANT_PARTS), *(f"mlp.fc2.{p}" for p in _QUANT_PARTS),
)
_GLOBAL_CANON = (
    "video_patch_proj.weight", "video_patch_proj.bias", "audio_patch_proj.weight", "audio_patch_proj.bias",
    "condition_proj.weight", "condition_proj.bias", "adaln_t_table", "rope.inv_freq", "final_layer.norm.weight",
    "final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias", "final_layer.video_out.weight",
    "final_layer.video_out.bias", "final_layer.audio_out.weight", "final_layer.audio_out.bias",
)
_REFINER_RE = re.compile(r"^token_refiner\.blocks\.(\d+)\.(.+)$")


def _order(names: Iterable[str], canon: Sequence[str]) -> List[str]:
    s = set(names)
    head = [n for n in canon if n in s]
    return head + sorted(s.difference(head))


def classify(names: Iterable[str]):
    """Split names into (globals, {refiner block: rel names}, other token_refiner.* names, {block: rel names})."""
    glob: List[str] = []
    ref_blocks: Dict[int, List[str]] = {}
    ref_other: List[str] = []
    blocks: Dict[int, List[str]] = {}
    for n in names:
        if n.startswith(PAD_PREFIX):
            raise H3TError(f"tensor name {n!r} uses the reserved padding prefix {PAD_PREFIX!r}")
        m = _BLOCK_RE.match(n)
        if m:
            blocks.setdefault(int(m.group(1)), []).append(m.group(2))
            continue
        m = _REFINER_RE.match(n)
        if m:
            ref_blocks.setdefault(int(m.group(1)), []).append(m.group(2))
        elif n.startswith("token_refiner."):
            ref_other.append(n)
        else:
            glob.append(n)
    return glob, ref_blocks, ref_other, blocks


def _place(entries: Sequence[Tuple[str, str, Tuple[int, ...]]]) -> Tuple[List[BlockEntry], int]:
    """Assign TENSOR_ALIGN-aligned offsets (from 0) in the given order; returns (entries, end offset)."""
    out: List[BlockEntry] = []
    cur = 0
    for name, dt, shape in entries:
        cur = _align(cur, TENSOR_ALIGN)
        nb = nbytes_of(dt, shape)
        out.append(BlockEntry(name, dt, tuple(shape), cur, nb))
        cur += nb
    return out, cur


# --------------------------------------------------------------------------------------------- writer
def write_h3t(
    dst: Union[str, os.PathLike],
    cfg: H3Config,
    tensors: Mapping[str, Union[TensorSource, torch.Tensor]],
    *,
    quant: Optional[Mapping[str, object]] = None,
    source_name: str = "",
    source_size: int = 0,
    source_sha256: str = "",
    extra_meta: Optional[Mapping[str, str]] = None,
    chunk_bytes: int = 32 << 20,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, int]:
    """Write `tensors` (ComfyUI-layout names) as an h3t file. Peak memory is one chunk plus the header.

    The write goes to `<dst>.part` and is renamed on success, so an interrupted run never leaves a
    truncated file that looks valid. Returns a small summary dict.
    """
    if chunk_bytes < 4096 or chunk_bytes % 8:
        raise ValueError("chunk_bytes must be a multiple of 8 and at least 4096")
    src = {k: as_source(v) for k, v in tensors.items()}
    glob, ref_blocks, ref_other, blocks = classify(src)
    n_blocks = cfg.layers
    if n_blocks < 1 or sorted(blocks) != list(range(n_blocks)):
        raise H3TError(f"config says {n_blocks} blocks but the tensors carry block indices {sorted(blocks)[:8]}...")

    # every block must have the identical tensor set, dtypes and shapes: that is what lets one layout describe them all
    rel0 = _order(blocks[0], _BLOCK_CANON)
    sig0 = {r: (src[f"blocks.0.{r}"].dtype, tuple(src[f"blocks.0.{r}"].shape)) for r in rel0}
    for i in range(1, n_blocks):
        got = set(blocks[i])
        if got != set(rel0):
            raise H3TError(f"block {i} tensor set differs from block 0: missing {sorted(set(rel0) - got)[:4]}, extra {sorted(got - set(rel0))[:4]}")
        for r in rel0:
            s = src[f"blocks.{i}.{r}"]
            if (s.dtype, tuple(s.shape)) != sig0[r]:
                raise H3TError(f"blocks.{i}.{r} is {s.dtype}{tuple(s.shape)}, block 0 has {sig0[r][0]}{sig0[r][1]}")
    blk_layout, blk_end = _place([(r, *sig0[r]) for r in rel0])
    block_nbytes = _align(blk_end, BLOCK_ALIGN)

    g_names = _order(glob, _GLOBAL_CANON)
    for j in sorted(ref_blocks):
        g_names += [f"token_refiner.blocks.{j}.{r}" for r in _order(ref_blocks[j], _BLOCK_CANON)]
    g_names += _order(ref_other, ("token_refiner.final_norm.weight",))
    g_layout, g_end = _place([(n, src[n].dtype, tuple(src[n].shape)) for n in g_names])
    blocks_start = _align(g_end, BLOCK_ALIGN)

    is_quant = bool(quant_linear_names(rel0))
    qmeta = dict(quant) if quant else ({"format": "asym_w4a8_int8", "group_size": cfg.quant_group, "convrot_groupsize": cfg.quant_convrot} if is_quant else {})
    data_end = blocks_start + n_blocks * block_nbytes

    # flat write plan: (offset, name or None for padding, nbytes, source)
    plan: List[Tuple[int, Optional[str], int, Optional[TensorSource]]] = [(e.offset, e.name, e.nbytes, src[e.name]) for e in g_layout]
    for i in range(n_blocks):
        base = blocks_start + i * block_nbytes
        plan += [(base + e.offset, f"blocks.{i}.{e.name}", e.nbytes, src[f"blocks.{i}.{e.name}"]) for e in blk_layout]
    plan.sort(key=lambda p: p[0])
    flat: List[Tuple[int, Optional[str], int, Optional[TensorSource]]] = []
    cur, npad = 0, 0
    for off, name, nb, s in plan:
        if off < cur:
            raise H3TError(f"internal layout overlap at {name!r}")
        if off > cur:
            flat.append((cur, f"{PAD_PREFIX}{npad}", off - cur, None))
            npad += 1
        flat.append((off, name, nb, s))
        cur = off + nb
    if cur < data_end:
        flat.append((cur, f"{PAD_PREFIX}{npad}", data_end - cur, None))
        npad += 1

    meta = {
        "format": FORMAT, "version": str(VERSION), "config": cfg.to_json(), "n_blocks": str(n_blocks),
        "tensor_align": str(TENSOR_ALIGN), "block_align": str(BLOCK_ALIGN), "block_nbytes": str(block_nbytes),
        "blocks_start": str(blocks_start),
        "block_ranges": json.dumps([[blocks_start + i * block_nbytes, blocks_start + (i + 1) * block_nbytes] for i in range(n_blocks)], separators=(",", ":")),
        "block_layout": json.dumps([[e.name, e.dtype, list(e.shape), e.offset, e.nbytes] for e in blk_layout], separators=(",", ":")),
        "globals_layout": json.dumps([[e.name, e.dtype, list(e.shape), e.offset, e.nbytes] for e in g_layout], separators=(",", ":")),
        "quant": json.dumps(qmeta, sort_keys=True), "source_name": source_name, "source_size": str(source_size),
        "source_header_sha256": source_sha256,
    }
    meta.update({str(k): str(v) for k, v in (extra_meta or {}).items()})
    header: Dict[str, object] = {"__metadata__": meta}
    for off, name, nb, s in flat:
        header[name] = {
            "dtype": s.dtype if s else "U8", "shape": list(s.shape) if s else [nb], "data_offsets": [off, off + nb],
        }
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("ascii")
    data_start = _align(8 + len(hdr_bytes), BLOCK_ALIGN)
    hdr_bytes += b" " * (data_start - 8 - len(hdr_bytes))  # JSON whitespace padding: aligns the data section

    dst = os.fspath(dst)
    tmp = dst + ".part"
    total = sum(nb for _, name, nb, _ in flat if not name.startswith(PAD_PREFIX))
    done = 0
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", data_start - 8))
            f.write(hdr_bytes)
            pos = 0
            for off, name, nb, s in flat:
                assert off == pos, (name, off, pos)
                if s is None:
                    f.write(bytes(nb))
                else:
                    wrote = 0
                    for chunk in s.chunks(chunk_bytes):
                        f.write(chunk)
                        wrote += len(chunk)
                    if wrote != nb:
                        raise H3TError(f"{name}: source produced {wrote} bytes, expected {nb}")
                    done += nb
                    if progress and nb >= (1 << 20):
                        progress(name, done, total)
                pos += nb
            f.flush()
            os.fsync(f.fileno())  # the rename below must not be able to outrun the data
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return {"file_size": data_start + data_end, "data_start": data_start, "blocks_start": blocks_start,
            "block_nbytes": block_nbytes, "n_blocks": n_blocks, "n_pads": npad, "n_tensors": len(flat) - npad}


