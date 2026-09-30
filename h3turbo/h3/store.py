"""h3t: a safetensors file laid out for streaming the official H3 DiT block by block.

Why a custom layout: the source checkpoint stores tensors alphabetically, so one block's ~30 tensors are
scattered over the file and a block cannot be moved with a single host->device copy. An h3t file is still a
plain safetensors file (`safetensors.safe_open` reads it) but is ordered globals, token refiner, block 0..N-1,
every block is ONE contiguous BLOCK_ALIGN-aligned byte range with an identical tensor layout in every block, and
every tensor starts on a TENSOR_ALIGN boundary. safetensors forbids holes, so alignment gaps are explicit
`__pad.*` U8 tensors. The streaming engine copies the raw block bytes into a device slot and takes typed
zero-copy views (`H3TFile.block_weights_from`) of them.

The tensor names, dtypes and bytes are exactly those of the ComfyUI-layout source (nothing is re-encoded);
only the order and the padding differ. This module is the format and the reader; the writer is in writer.py
(`write_h3t` and friends are re-exported here).
"""
from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import re
import struct
import sys
import warnings
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import H3Config
from .types import BlockWeights, GlobalWeights, W4A8Weight, Weight

if sys.byteorder != "little":  # pragma: no cover - safetensors is little-endian by definition
    raise ImportError("h3t files are little-endian; big-endian hosts are unsupported")

FORMAT = "h3turbo-h3"
VERSION = 1
TENSOR_ALIGN = 256  # every tensor start (kernels and vector loads like 16 B; 256 keeps a wide margin)
BLOCK_ALIGN = 4096  # every block start/size (page-aligned so a block range can be page-locked in place)
PAD_PREFIX = "__pad."
_MAX_HEADER = 100_000_000  # same cap as safetensors itself

_DTYPES: Dict[str, Tuple[torch.dtype, int]] = {
    "F64": (torch.float64, 8), "F32": (torch.float32, 4), "F16": (torch.float16, 2), "BF16": (torch.bfloat16, 2),
    "I64": (torch.int64, 8), "I32": (torch.int32, 4), "I16": (torch.int16, 2), "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1), "BOOL": (torch.bool, 1),
    "F8_E4M3": (torch.float8_e4m3fn, 1), "F8_E5M2": (torch.float8_e5m2, 1),
}
_ST_NAME = {v[0]: k for k, v in _DTYPES.items()}


class H3TError(ValueError):
    """The file is not a valid h3t container (or a write request is inconsistent)."""


def itemsize(dtype: str) -> int:
    try:
        return _DTYPES[dtype][1]
    except KeyError:
        raise H3TError(f"unsupported safetensors dtype {dtype!r}") from None


def torch_dtype(dtype: str) -> torch.dtype:
    itemsize(dtype)
    return _DTYPES[dtype][0]


def st_dtype(t: torch.dtype) -> str:
    try:
        return _ST_NAME[t]
    except KeyError:
        raise H3TError(f"torch dtype {t} has no safetensors equivalent here") from None


def nbytes_of(dtype: str, shape: Sequence[int]) -> int:
    return math.prod(shape) * itemsize(dtype)


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# --------------------------------------------------------------------------------------------- safetensors header
class Header(NamedTuple):
    tensors: Dict[str, dict]  # name -> {"dtype", "shape", "data_offsets"} (offsets relative to data_start)
    meta: Dict[str, str]
    data_start: int
    file_size: int
    raw: bytes  # the header JSON bytes as stored (for hashing)


def read_header(path: Union[str, os.PathLike]) -> Header:
    """Parse a safetensors header without touching tensor data. Checks every entry lies inside the file."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise H3TError(f"{path}: file too small to be safetensors ({size} bytes)")
        (n,) = struct.unpack("<Q", head)
        if n > _MAX_HEADER or 8 + n > size:
            raise H3TError(f"{path}: implausible safetensors header size {n} (file is {size} bytes)")
        raw = f.read(n)
    if not raw.lstrip(b" ").startswith(b"{"):
        raise H3TError(f"{path}: safetensors header does not start with '{{'")
    try:
        hdr = json.loads(raw)
    except ValueError as e:
        raise H3TError(f"{path}: unreadable safetensors header: {e}") from None
    meta = hdr.pop("__metadata__", None) or {}
    data_len = size - 8 - n
    for name, e in hdr.items():
        try:
            a, b = e["data_offsets"]
            dtype, shape = e["dtype"], e["shape"]
        except (KeyError, TypeError, ValueError):
            raise H3TError(f"{path}: malformed header entry for {name!r}") from None
        if not (0 <= a <= b <= data_len):
            raise H3TError(f"{path}: tensor {name!r} range [{a}, {b}) is outside the data section ({data_len} bytes); truncated file?")
        if dtype in _DTYPES and b - a != nbytes_of(dtype, shape):
            raise H3TError(f"{path}: tensor {name!r} declares {b - a} bytes but {dtype}{shape} needs {nbytes_of(dtype, shape)}")
    return Header(hdr, meta, 8 + n, size, raw)


def header_sha256(path: Union[str, os.PathLike]) -> str:
    return hashlib.sha256(read_header(path).raw).hexdigest()


# --------------------------------------------------------------------------------------------- layout
class BlockEntry(NamedTuple):
    name: str  # relative to the block (or absolute for the global region)
    dtype: str
    shape: Tuple[int, ...]
    offset: int  # relative to the block start (or to the data section for the global region)
    nbytes: int


class TensorInfo(NamedTuple):
    dtype: str
    shape: Tuple[int, ...]
    start: int  # absolute file offset
    nbytes: int


_LINEARS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
_QUANT_PARTS = ("weight", "weight_s_rel", "weight_s_channel", "weight_codebook", "comfy_quant")
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
def quant_linear_names(layout_names: Iterable[str]) -> List[str]:
    """Prefixes ('attn.qkv_proj', ...) of the W4A8 linears present in a block layout."""
    s = set(layout_names)
    return [p for p in _LINEARS if f"{p}.weight_codebook" in s]


# --------------------------------------------------------------------------------------------- typed views
def _typed(buf: torch.Tensor, off: int, dtype: str, shape: Sequence[int]) -> torch.Tensor:
    """Zero-copy typed view of `shape`/`dtype` at byte offset `off` of a 1-D uint8 tensor."""
    tdt = torch_dtype(dtype)
    v = buf[off : off + nbytes_of(dtype, shape)]
    if tdt is not torch.uint8:
        v = v.view(tdt)
    return v.view(tuple(shape))


def build_block_weights(get: Callable[[str], torch.Tensor], has: Callable[[str], bool], group: int, convrot: int, adaln: bool = True) -> BlockWeights:
    """Assemble BlockWeights from a name -> tensor accessor (names relative to the block)."""

    def lin(p: str) -> Weight:
        if has(f"{p}.weight_codebook"):
            return W4A8Weight(get(f"{p}.weight"), get(f"{p}.weight_s_rel"), get(f"{p}.weight_s_channel"), get(f"{p}.weight_codebook"), group, convrot)
        return get(f"{p}.weight")

    with_adaln = adaln and has("adaln_proj.linear.weight")
    return BlockWeights(
        qkv=lin("attn.qkv_proj"), out=lin("attn.out_proj"), fc1=lin("mlp.fc1"), fc2=lin("mlp.fc2"),
        norm1=get("norm1.weight"), norm2=get("norm2.weight"), q_norm=get("attn.q_norm.weight"), k_norm=get("attn.k_norm.weight"),
        adaln_w=get("adaln_proj.linear.weight") if with_adaln else None,
        adaln_b=get("adaln_proj.linear.bias") if with_adaln else None,
    )


# --------------------------------------------------------------------------------------------- reader
class H3TFile:
    """Read-only memory-mapped h3t file. All tensor/bytes accessors are zero-copy views of the mapping.

    The views are backed by a read-only mapping: never write into them (that faults the process). Close the
    file only after everything derived from it (providers, page-locked ranges) is closed; if views are still
    alive `close()` cannot unmap and the mapping is released when they are garbage collected.
    """

    def __init__(self, path: Union[str, os.PathLike]):
        self.path = os.fspath(path)
        h = read_header(self.path)
        m = h.meta
        if m.get("format") != FORMAT:
            raise H3TError(f"{self.path}: not an h3t file (metadata format={m.get('format')!r}, expected {FORMAT!r})")
        try:
            version = int(m.get("version", "0"))
        except (ValueError, TypeError):
            version = -1
        if version != VERSION:
            raise H3TError(f"{self.path}: unsupported h3t version {m.get('version')!r} (this reader supports {VERSION})")
        try:
            self.cfg = H3Config.from_json(m["config"])
            self.n_blocks = int(m["n_blocks"])
            self.block_nbytes = int(m["block_nbytes"])
            self.blocks_start = int(m["blocks_start"])  # relative to data_start
            self.block_layout = [BlockEntry(n, d, tuple(s), int(o), int(b)) for n, d, s, o, b in json.loads(m["block_layout"])]
            self.quant = json.loads(m.get("quant") or "{}")
            self.meta = dict(m)
        except (KeyError, ValueError, TypeError) as e:
            raise H3TError(f"{self.path}: corrupt h3t metadata: {e!r}") from None
        self.data_start, self.file_size = h.data_start, h.file_size
        self.source_name = m.get("source_name", "")
        self.source_size = int(m.get("source_size") or 0)
        try:
            self._validate(h)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            raise H3TError(f"{self.path}: corrupt h3t metadata: {e!r}") from None
        self.quant_group = int(self.quant.get("group_size", self.cfg.quant_group))
        self.quant_convrot = int(self.quant.get("convrot_groupsize", self.cfg.quant_convrot))
        self._lay = {e.name: e for e in self.block_layout}
        self._tensors: Dict[str, TensorInfo] = {
            n: TensorInfo(e["dtype"], tuple(e["shape"]), self.data_start + e["data_offsets"][0], e["data_offsets"][1] - e["data_offsets"][0])
            for n, e in h.tensors.items() if not n.startswith(PAD_PREFIX)
        }
        with open(self.path, "rb") as fh:
            self._mm: Optional[mmap.mmap] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._np: Optional[np.ndarray] = np.frombuffer(self._mm, dtype=np.uint8)
        with warnings.catch_warnings():  # torch warns that the (read-only) buffer is not writable; we never write
            warnings.simplefilter("ignore")
            self._u8: Optional[torch.Tensor] = torch.frombuffer(self._mm, dtype=torch.uint8)

    # ---- validation
    def _validate(self, h: Header) -> None:
        p, cfg = self.path, self.cfg
        if self.n_blocks != cfg.layers or self.n_blocks < 1:
            raise H3TError(f"{p}: n_blocks={self.n_blocks} disagrees with config.layers={cfg.layers}")
        if self.data_start % BLOCK_ALIGN or self.blocks_start % BLOCK_ALIGN or self.block_nbytes % BLOCK_ALIGN:
            raise H3TError(f"{p}: data/block offsets are not {BLOCK_ALIGN}-aligned")
        expect = self.data_start + self.blocks_start + self.n_blocks * self.block_nbytes
        if h.file_size != expect:
            raise H3TError(f"{p}: file is {h.file_size} bytes, layout needs exactly {expect} (truncated or trailing data?)")
        ranges = json.loads(h.meta["block_ranges"])
        for i, (a, b) in enumerate(ranges):
            if a != self.blocks_start + i * self.block_nbytes or b - a != self.block_nbytes:
                raise H3TError(f"{p}: block_ranges[{i}]={a, b} inconsistent with block_nbytes/blocks_start")
        if len(ranges) != self.n_blocks:
            raise H3TError(f"{p}: {len(ranges)} block ranges for {self.n_blocks} blocks")
        for i in range(self.n_blocks):  # the safetensors header must agree with the block layout, tensor by tensor
            base = self.blocks_start + i * self.block_nbytes
            for e in self.block_layout:
                got = h.tensors.get(f"blocks.{i}.{e.name}")
                if got is None or got["dtype"] != e.dtype or tuple(got["shape"]) != e.shape or got["data_offsets"] != [base + e.offset, base + e.offset + e.nbytes]:
                    raise H3TError(f"{p}: blocks.{i}.{e.name} does not match the block layout")
        cur = 0
        for n, e in sorted(h.tensors.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = e["data_offsets"]
            if a != cur:
                raise H3TError(f"{p}: hole or overlap before {n!r}")
            if not n.startswith(PAD_PREFIX) and a % TENSOR_ALIGN:
                raise H3TError(f"{p}: {n!r} starts at {a}, not {TENSOR_ALIGN}-aligned")
            cur = b
        if cur != h.file_size - self.data_start:
            raise H3TError(f"{p}: tensors cover {cur} bytes, data section has {h.file_size - self.data_start}")
        have = {e.name for e in self.block_layout}
        for q in quant_linear_names(have):
            missing = [x for x in _QUANT_PARTS if f"{q}.{x}" not in have]
            if missing:
                raise H3TError(f"{p}: W4A8 linear {q} lacks {missing}")

    # ---- raw bytes
    def block_range(self, i: int) -> Tuple[int, int]:
        """Absolute [start, end) file offsets of block i."""
        if not 0 <= i < self.n_blocks:
            raise IndexError(f"block {i} out of range [0, {self.n_blocks})")
        a = self.data_start + self.blocks_start + i * self.block_nbytes
        return a, a + self.block_nbytes

    def _live(self) -> None:
        if self._mm is None:
            raise ValueError("H3TFile is closed")

    def block_bytes(self, i: int) -> np.ndarray:
        """Read-only uint8 numpy view of block i in the mapping (no copy)."""
        self._live()
        a, b = self.block_range(i)
        return self._np[a:b]

    def block_tensor(self, i: int) -> torch.Tensor:
        """uint8 torch view of block i in the mapping (no copy); the source of a host->device copy."""
        self._live()
        a, b = self.block_range(i)
        return self._u8[a:b]

    @property
    def block_layout_size(self) -> int:
        return self.block_layout[-1].offset + self.block_layout[-1].nbytes

    # ---- typed access
    def keys(self) -> List[str]:
        return list(self._tensors)

    def info(self, name: str) -> TensorInfo:
        try:
            return self._tensors[name]
        except KeyError:
            raise KeyError(f"{name!r} is not in {self.path}") from None

    def tensor(self, name: str) -> torch.Tensor:
        """Zero-copy CPU view of one named tensor."""
        self._live()
        t = self.info(name)
        return _typed(self._u8, t.start, t.dtype, t.shape)

    def block_weights_from(self, buf: torch.Tensor) -> BlockWeights:
        """Typed zero-copy views of a block held in `buf` (1-D uint8, block_nbytes long, any device)."""
        if buf.dtype != torch.uint8 or buf.dim() != 1 or buf.numel() != self.block_nbytes:
            raise ValueError(f"expected a 1-D uint8 buffer of {self.block_nbytes} bytes, got {tuple(buf.shape)} {buf.dtype}")
        lay = self._lay
        return build_block_weights(lambda r: _typed(buf, lay[r].offset, lay[r].dtype, lay[r].shape), lay.__contains__, self.quant_group, self.quant_convrot)

    def block_weights_cpu(self, i: int) -> BlockWeights:
        return self.block_weights_from(self.block_tensor(i))

    def globals_nbytes(self, refiner: bool = True) -> int:
        """Bytes of the non-block tensors (the token refiner alone is ~1.5 GB for the real checkpoint)."""
        return sum(t.nbytes for n, t in self._tensors.items() if not _BLOCK_RE.match(n) and (refiner or not n.startswith("token_refiner.")))

    def load_globals(self, device="cpu", *, refiner_device=None) -> GlobalWeights:
        """GlobalWeights on `device` (refiner on `refiner_device`, default the same). On CPU the tensors are
        zero-copy views of the mapping; pass refiner_device='cpu' to keep the 1.5 GB refiner out of VRAM."""
        dev = torch.device(device)
        rdev = torch.device(refiner_device) if refiner_device is not None else dev

        def get(name: str, d: torch.device = dev, view: bool = False) -> torch.Tensor:
            t = self.tensor(name)
            if d.type != "cpu":
                return t.to(d)
            return t if view else t.clone()  # only the refiner (view=True) is worth keeping as a read-only mapping

        def opt(name: str) -> Optional[torch.Tensor]:
            return get(name) if name in self._tensors else None

        need = ["video_patch_proj.weight", "video_patch_proj.bias", "audio_patch_proj.weight", "audio_patch_proj.bias", "condition_proj.weight",
                "condition_proj.bias", "rope.inv_freq", "final_layer.norm.weight", "final_layer.adaln_proj.linear.weight",
                "final_layer.adaln_proj.linear.bias", "final_layer.video_out.weight", "final_layer.video_out.bias", "final_layer.audio_out.weight",
                "final_layer.audio_out.bias"]
        missing = [n for n in need if n not in self._tensors]
        if missing:
            raise H3TError(f"{self.path}: missing global tensors {missing}")
        refiner: List[BlockWeights] = []
        for j in range(self.cfg.refiner_layers):
            pre = f"token_refiner.blocks.{j}."
            refiner.append(build_block_weights(lambda r, pre=pre: get(pre + r, rdev, True), lambda r, pre=pre: pre + r in self._tensors, self.quant_group, self.quant_convrot, adaln=False))
        return GlobalWeights(
            video_patch_w=get("video_patch_proj.weight"), video_patch_b=get("video_patch_proj.bias"),
            audio_patch_w=get("audio_patch_proj.weight"), audio_patch_b=get("audio_patch_proj.bias"),
            condition_w=get("condition_proj.weight"), condition_b=get("condition_proj.bias"),
            adaln_t_table=opt("adaln_t_table"), rope_inv_freq=get("rope.inv_freq"), final_norm=get("final_layer.norm.weight"),
            final_adaln_w=get("final_layer.adaln_proj.linear.weight"), final_adaln_b=get("final_layer.adaln_proj.linear.bias"),
            video_out_w=get("final_layer.video_out.weight"), video_out_b=get("final_layer.video_out.bias"),
            audio_out_w=get("final_layer.audio_out.weight"), audio_out_b=get("final_layer.audio_out.bias"),
            refiner=refiner,
            refiner_final_norm=get("token_refiner.final_norm.weight", rdev, True) if "token_refiner.final_norm.weight" in self._tensors else None,
        )

    # ---- lifecycle
    def close(self) -> None:
        if self._mm is None:
            return
        mm, self._mm, self._np, self._u8 = self._mm, None, None, None
        try:
            mm.close()
        except BufferError:  # views handed out earlier are still alive; the mapping goes away with them
            pass

    def __enter__(self) -> "H3TFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"H3TFile({self.path!r}, blocks={self.n_blocks}, block={self.block_nbytes / 2**20:.1f} MiB)"


_WRITER_API = ("write_h3t", "TensorSource", "TorchSource", "LazySource", "as_source", "classify")


def __getattr__(name: str):
    """PEP 562: the writer lives in writer.py (which imports this module); re-export it lazily to avoid a cycle."""
    if name in _WRITER_API:
        from . import writer

        return getattr(writer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
