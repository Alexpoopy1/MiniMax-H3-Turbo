"""ComfyUI-layout H3 checkpoint (W4A8/ConvRot, or dense) -> h3t, lossless and low-RAM.

The converter never re-encodes anything: every source tensor is copied byte for byte (including the
`comfy_quant` JSON tensors), only reordered and padded (see store.py). It reads one tensor at a time in
bounded chunks, so peak memory is a few chunks regardless of the 12.5 GB checkpoint size. Before writing it
validates every quantised layer; after writing it re-reads the destination with plain file reads (not the
mapping, so the process working set stays small) and compares every tensor byte for byte.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .config import H3Config
from .store import BLOCK_ALIGN, PAD_PREFIX, H3TError, H3TFile, header_sha256, read_header, torch_dtype
from .writer import LazySource, TensorSource, TorchSource, classify, write_h3t

QUANT_FORMAT = "asym_w4a8_int8"
_FLOATS = ("F32", "BF16", "F16")
_LINEARS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
_CHUNK = 32 << 20
_SAFE_OPEN_MAX = 2 << 30


class ConvertError(H3TError):
    """The source checkpoint is not a valid ComfyUI-layout H3 checkpoint."""


class FileSource(TensorSource):
    """One tensor of a safetensors file, read from disk in chunks (never mapped)."""

    def __init__(self, path: str, entry: dict, data_start: int):
        self.path, self.dtype, self.shape = path, entry["dtype"], tuple(entry["shape"])
        a, b = entry["data_offsets"]
        self._off, self.nbytes = data_start + a, b - a

    def chunks(self, chunk_bytes: int):
        buf = bytearray(max(1, min(chunk_bytes, self.nbytes)))
        with open(self.path, "rb") as f:
            f.seek(self._off)
            left = self.nbytes
            while left:
                mv = memoryview(buf)[: min(left, len(buf))]
                if f.readinto(mv) != len(mv):
                    raise ConvertError(f"{self.path}: unexpected end of file")
                yield mv
                left -= len(mv)


@dataclass
class ConvertReport:
    src: str = ""
    dst: str = ""
    cfg: Optional[H3Config] = None
    n_tensors: int = 0
    n_blocks: int = 0
    block_nbytes: int = 0
    src_bytes: int = 0
    dst_bytes: int = 0
    quant: Dict[str, object] = field(default_factory=dict)
    n_quant_layers: int = 0
    n_dense_linears: int = 0
    codebook_sorted: int = 0
    codebook_distinct: int = 0
    codebook_range: Tuple[float, float] = (0.0, 0.0)
    s_ch_min: float = 0.0
    s_rel_range: Tuple[float, float] = (0.0, 0.0)
    extras: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    verified: bool = False
    verify_detail: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def summary(self) -> str:
        c = self.cfg
        lines = [
            f"source      {self.src} ({self.src_bytes / 2**30:.3f} GiB, {self.n_tensors} tensors)",
            f"destination {self.dst} ({self.dst_bytes / 2**30:.3f} GiB)",
            f"model       {c.layers} blocks x {self.block_nbytes / 2**20:.2f} MiB, hidden {c.hidden}, heads {c.heads}x{c.head_dim}, ffn {c.ffn}, refiner {c.refiner_layers}",
            f"linears     {self.n_quant_layers} W4A8 ({self.quant}), {self.n_dense_linears} dense",
        ]
        if self.n_quant_layers:
            lines += [
                f"codebooks   {self.n_quant_layers} layers, {self.codebook_distinct} distinct, {self.codebook_sorted} sorted ascending, range {self.codebook_range}",
                f"scales      s_channel min {self.s_ch_min:.3g}, s_rel range {self.s_rel_range}",
            ]
        if self.extras:
            lines.append(f"kept as-is  {len(self.extras)} tensors the runtime does not read: {', '.join(self.extras[:6])}")
        lines += [f"warning     {w}" for w in self.warnings]
        lines.append(f"verified    {self.verified} {self.verify_detail if self.verified else ''}".rstrip())
        lines.append(f"time        {self.seconds:.1f} s")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------- validation
def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise ConvertError(msg)


def _shape(src: Mapping[str, TensorSource], name: str, dtypes: Sequence[str], shape: Optional[Sequence[int]] = None) -> None:
    _expect(name in src, f"missing tensor {name!r}")
    s = src[name]
    _expect(s.dtype in dtypes, f"{name}: dtype {s.dtype}, expected one of {list(dtypes)}")
    if shape is not None:
        _expect(tuple(s.shape) == tuple(shape), f"{name}: shape {tuple(s.shape)}, expected {tuple(shape)}")


def validate(src: Mapping[str, TensorSource], cfg: Optional[H3Config], sigma_shifts, report: ConvertReport, log: Callable[[str], None]) -> Tuple[H3Config, Dict[str, object]]:
    """Check the whole checkpoint against the architecture; returns (config, quant meta)."""
    if cfg is None:
        try:
            cfg = H3Config.from_shapes({n: s.shape for n, s in src.items()}, sigma_shifts)
        except (KeyError, IndexError) as e:
            raise ConvertError(f"cannot infer the model config: tensor {e} not found; is this a ComfyUI-layout H3 checkpoint?") from None
    glob, ref_blocks, ref_other, blocks = classify(src)
    _expect(sorted(blocks) == list(range(cfg.layers)), f"block indices {sorted(blocks)[:6]}... do not form 0..{cfg.layers - 1}")
    _expect(sorted(ref_blocks) == list(range(cfg.refiner_layers)), f"token_refiner blocks {sorted(ref_blocks)} != 0..{cfg.refiner_layers - 1}")
    h, inner = cfg.hidden, cfg.inner
    dims = {"attn.qkv_proj": (3 * inner, h), "attn.out_proj": (h, inner), "mlp.fc1": (2 * cfg.ffn, h), "mlp.fc2": (h, cfg.ffn)}

    # globals
    for n, dt, shp in (
        ("video_patch_proj.weight", _FLOATS, (h, cfg.video_patch_dim)), ("video_patch_proj.bias", _FLOATS, (h,)),
        ("audio_patch_proj.weight", _FLOATS, (h, cfg.audio_channels)), ("audio_patch_proj.bias", _FLOATS, (h,)),
        ("condition_proj.weight", _FLOATS, (h, cfg.text_dim)), ("condition_proj.bias", _FLOATS, (h,)),
        ("rope.inv_freq", _FLOATS, (cfg.rope_inv_freq_len,)), ("final_layer.norm.weight", _FLOATS, (h,)),
        ("final_layer.adaln_proj.linear.weight", _FLOATS, (2 * h, cfg.t_dim)), ("final_layer.adaln_proj.linear.bias", _FLOATS, (2 * h,)),
        ("final_layer.video_out.weight", _FLOATS, (cfg.video_patch_dim, h)), ("final_layer.video_out.bias", _FLOATS, (cfg.video_patch_dim,)),
        ("final_layer.audio_out.weight", _FLOATS, (cfg.audio_channels, h)), ("final_layer.audio_out.bias", _FLOATS, (cfg.audio_channels,)),
    ):
        _shape(src, n, dt, shp)
    if cfg.curve_grid:
        _shape(src, "adaln_t_table", _FLOATS, (cfg.curve_grid, cfg.t_dim))
    if cfg.refiner_layers:
        _shape(src, "token_refiner.final_norm.weight", _FLOATS, (h,))
    known = {"adaln_t_table", "video_patch_proj.weight", "video_patch_proj.bias", "audio_patch_proj.weight", "audio_patch_proj.bias", "condition_proj.weight",
             "condition_proj.bias", "rope.inv_freq", "final_layer.norm.weight", "final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias",
             "final_layer.video_out.weight", "final_layer.video_out.bias", "final_layer.audio_out.weight", "final_layer.audio_out.bias"}
    report.extras = sorted(set(glob) - known) + sorted(set(ref_other) - {"token_refiner.final_norm.weight"})

    # refiner: dense only (the runtime applies it once per prompt in bf16)
    for j, rels in sorted(ref_blocks.items()):
        pre = f"token_refiner.blocks.{j}."
        for p, (n_, k_) in dims.items():
            _shape(src, f"{pre}{p}.weight", _FLOATS, (n_, k_))
            _expect(f"{pre}{p}.weight_codebook" not in src, f"{pre}{p}: a quantised token-refiner linear is not supported")
        for n, shp in (("norm1.weight", (h,)), ("norm2.weight", (h,)), ("attn.q_norm.weight", (cfg.head_dim,)), ("attn.k_norm.weight", (cfg.head_dim,))):
            _shape(src, pre + n, _FLOATS, shp)
        _expect(set(rels) == {f"{p}.weight" for p in dims} | {"norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight"}, f"{pre}: unexpected tensors {sorted(rels)}")

    # blocks: uniform layout, every linear either dense or a complete W4A8 set
    qmeta: Optional[dict] = None
    cbs, ranges_rel, ranges_ch = [], [], []
    for i in range(cfg.layers):
        pre = f"blocks.{i}."
        for n, shp in (("norm1.weight", (h,)), ("norm2.weight", (h,)), ("attn.q_norm.weight", (cfg.head_dim,)), ("attn.k_norm.weight", (cfg.head_dim,))):
            _shape(src, pre + n, _FLOATS, shp)
        has_adaln = f"{pre}adaln_proj.linear.weight" in src
        _expect(has_adaln == ("blocks.0.adaln_proj.linear.weight" in src), f"block {i}: adaln_proj presence differs from block 0")
        if has_adaln:
            _shape(src, pre + "adaln_proj.linear.weight", _FLOATS, (18 * h, cfg.t_dim))
            _shape(src, pre + "adaln_proj.linear.bias", _FLOATS, (18 * h,))
        for p, (n_, k_) in dims.items():
            w = f"{pre}{p}"
            if f"{w}.weight_codebook" not in src:
                _expect(not any(f"{w}.{x}" in src for x in ("weight_s_rel", "weight_s_channel", "comfy_quant")), f"{w}: quantisation tensors without weight_codebook")
                _shape(src, f"{w}.weight", _FLOATS, (n_, k_))
                report.n_dense_linears += 1
                continue
            _shape(src, f"{w}.weight", ("I8",), (n_, k_ // 2))
            _shape(src, f"{w}.weight_s_channel", ("F32",), (n_,))
            _shape(src, f"{w}.weight_codebook", ("F32",), (16,))
            _expect(f"{w}.comfy_quant" in src and src[f"{w}.comfy_quant"].dtype == "U8", f"{w}.comfy_quant missing or not U8")
            try:
                q = json.loads(src[f"{w}.comfy_quant"].read().numpy().tobytes())
            except ValueError:
                raise ConvertError(f"{w}.comfy_quant is not valid JSON") from None
            _expect(isinstance(q, dict) and q.get("format") == QUANT_FORMAT, f"{w}.comfy_quant format {q.get('format') if isinstance(q, dict) else q!r}, expected {QUANT_FORMAT!r}")
            g, c = q.get("group_size"), q.get("convrot_groupsize")
            _expect(isinstance(g, int) and isinstance(c, int) and g > 0 and c > 0, f"{w}.comfy_quant lacks integer group_size/convrot_groupsize: {q}")
            if qmeta is None:
                qmeta = dict(q)
            _expect(q == qmeta, f"{w}.comfy_quant {q} differs from the first quantised layer {qmeta}")
            _expect(k_ % g == 0 and k_ % c == 0, f"{w}: K={k_} not divisible by group_size {g} / convrot {c}")
            _shape(src, f"{w}.weight_s_rel", ("F8_E4M3",), (n_, k_ // g))
            cb = src[f"{w}.weight_codebook"].read()
            sc = src[f"{w}.weight_s_channel"].read()
            _expect(bool(torch.isfinite(cb).all()), f"{w}.weight_codebook has non-finite values")
            _expect(bool(torch.isfinite(sc).all()) and float(sc.min()) >= 0, f"{w}.weight_s_channel has non-finite or negative values")
            cbs.append(cb)
            ranges_ch.append(float(sc.min()))
            lo, hi = np.inf, -np.inf
            for chunk in src[f"{w}.weight_s_rel"].chunks(_CHUNK):
                f = torch.frombuffer(chunk, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
                _expect(not bool(torch.isnan(f).any()), f"{w}.weight_s_rel contains NaN")
                lo, hi = min(lo, float(f.min())), max(hi, float(f.max()))
            ranges_rel.append((lo, hi))
            report.n_quant_layers += 1
        log(f"validated block {i + 1}/{cfg.layers}")
    if cbs:
        allcb = torch.stack(cbs)
        report.codebook_sorted = int((allcb[:, 1:] >= allcb[:, :-1]).all(dim=1).sum())
        report.codebook_distinct = len({tuple(c.tolist()) for c in cbs})
        report.codebook_range = (float(allcb.min()), float(allcb.max()))
        report.s_ch_min = min(ranges_ch)
        report.s_rel_range = (min(r[0] for r in ranges_rel), max(r[1] for r in ranges_rel))
        if report.codebook_sorted != len(cbs):
            report.warnings.append(f"{len(cbs) - report.codebook_sorted} codebooks are not sorted ascending (decode does not need sorting)")
        if (qmeta["group_size"], qmeta["convrot_groupsize"]) != (cfg.quant_group, cfg.quant_convrot):  # checkpoint is authoritative
            report.warnings.append(f"config quant params {(cfg.quant_group, cfg.quant_convrot)} replaced by the checkpoint's {(qmeta['group_size'], qmeta['convrot_groupsize'])}")
            cfg = dataclasses.replace(cfg, quant_group=qmeta["group_size"], quant_convrot=qmeta["convrot_groupsize"])
    report.quant = dict(qmeta) if qmeta else {}
    return cfg, report.quant


# --------------------------------------------------------------------------------------------- verification
def _compare_file(src: TensorSource, dst_path: str, off: int, chunk_bytes: int) -> None:
    buf = bytearray(min(chunk_bytes, max(1, src.nbytes)))
    with open(dst_path, "rb") as f:
        f.seek(off)
        for c in src.chunks(chunk_bytes):
            got = memoryview(buf)[: len(c)]
            if f.readinto(got) != len(c) or not np.array_equal(np.frombuffer(c, np.uint8), np.frombuffer(got, np.uint8)):
                raise ConvertError("byte mismatch")


def verify_h3t(src: Mapping[str, TensorSource], dst: str, cfg: H3Config, *, chunk_bytes: int = _CHUNK, log: Callable[[str], None] = lambda s: None) -> Dict[str, float]:
    """Re-read `dst` and prove it holds exactly the source tensors, byte for byte, with a valid layout."""
    t0 = time.time()
    h = read_header(dst)
    names = {n for n in h.tensors if not n.startswith(PAD_PREFIX)}
    if names != set(src):
        raise ConvertError(f"tensor sets differ: only in source {sorted(set(src) - names)[:5]}, only in destination {sorted(names - set(src))[:5]}")
    total = 0
    for k, n in enumerate(sorted(src, key=lambda n: h.tensors[n]["data_offsets"][0])):
        e, s = h.tensors[n], src[n]
        if e["dtype"] != s.dtype or tuple(e["shape"]) != tuple(s.shape):
            raise ConvertError(f"{n}: destination is {e['dtype']}{e['shape']}, source {s.dtype}{tuple(s.shape)}")
        try:
            _compare_file(s, dst, h.data_start + e["data_offsets"][0], chunk_bytes)
        except ConvertError:
            raise ConvertError(f"{n}: destination bytes differ from the source") from None
        total += s.nbytes
        if k % 200 == 199:
            log(f"verified {k + 1}/{len(src)} tensors, {total / 2**30:.2f} GiB")
    # an independent reader agrees: safetensors' own parser sees the same names and small tensors. It maps the
    # whole file WITH commit charge on Windows (12 GB for the real checkpoint), so only small files get this check
    # (the real 12.5 GB file passed it once; see the report).
    if h.file_size <= _SAFE_OPEN_MAX:
        from safetensors import safe_open

        with safe_open(dst, "pt") as f:
            _expect(set(f.keys()) == set(h.tensors), "safetensors.safe_open key set differs from the header")
            for n, s in src.items():
                if 0 < s.nbytes <= (1 << 20):
                    got = f.get_tensor(n).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                    _expect(got == b"".join(bytes(c) for c in s.chunks(chunk_bytes)), f"{n}: safe_open bytes differ from the source")
    # the store's own view: config round trip and the block -> BlockWeights mapping on the first and last block
    with H3TFile(dst) as st:
        _expect(st.cfg == cfg, f"config round trip failed: {st.cfg} != {cfg}")
        inferred = H3Config.from_shapes({n: tuple(e["shape"]) for n, e in h.tensors.items() if not n.startswith(PAD_PREFIX)})
        _expect(all(getattr(inferred, k) == getattr(cfg, k) for k in ("hidden", "layers", "refiner_layers", "heads", "head_dim", "ffn", "video_channels", "audio_channels", "text_dim", "rope_inv_freq_len", "curve_grid", "t_dim")), "shapes of the destination do not reproduce the config")
        for i in sorted({0, st.n_blocks - 1}):
            bw = st.block_weights_cpu(i)
            pairs = [("norm1.weight", bw.norm1), ("norm2.weight", bw.norm2), ("attn.q_norm.weight", bw.q_norm), ("attn.k_norm.weight", bw.k_norm)]
            for p, w in zip(_LINEARS, (bw.qkv, bw.out, bw.fc1, bw.fc2)):
                if isinstance(w, torch.Tensor):
                    pairs.append((f"{p}.weight", w))
                else:
                    pairs += [(f"{p}.weight", w.q), (f"{p}.weight_s_rel", w.s_rel), (f"{p}.weight_s_channel", w.s_ch), (f"{p}.weight_codebook", w.codebook)]
            if bw.adaln_w is not None:
                pairs += [("adaln_proj.linear.weight", bw.adaln_w), ("adaln_proj.linear.bias", bw.adaln_b)]
            for rel, t in pairs:
                s = src[f"blocks.{i}.{rel}"]
                _expect((t.dtype, tuple(t.shape)) == (torch_dtype(s.dtype), tuple(s.shape)), f"block {i} {rel}: view is {t.dtype}{tuple(t.shape)}, source {s.dtype}{tuple(s.shape)}")
                _expect(t.reshape(-1).view(torch.uint8).numpy().tobytes() == b"".join(bytes(c) for c in s.chunks(chunk_bytes)), f"block {i} {rel}: typed view differs from the source")
    return {"tensors": len(src), "bytes": total, "seconds": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------------------------- convert
def _open_source(src) -> Tuple[Dict[str, TensorSource], str, int, str]:
    if isinstance(src, Mapping):
        return {k: (v if isinstance(v, TensorSource) else TorchSource(v)) for k, v in src.items()}, "<state_dict>", 0, ""
    path = os.fspath(src)
    h = read_header(path)
    if not h.tensors:
        raise ConvertError(f"{path}: no tensors")
    return {n: FileSource(path, e, h.data_start) for n, e in h.tensors.items()}, os.path.basename(path), h.file_size, header_sha256(path)


def convert(src, dst, *, verify: bool = True, cfg: Optional[H3Config] = None, sigma_shifts: Optional[Tuple[float, float]] = None,
            overwrite: bool = False, chunk_bytes: int = _CHUNK, progress: Optional[Callable[[str], None]] = None) -> ConvertReport:
    """Convert `src` (safetensors path, or a {name: Tensor} ComfyUI-layout state dict) into the h3t file `dst`."""
    log = progress or (lambda s: None)
    t0 = time.time()
    dst = os.fspath(dst)
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"{dst} exists (pass overwrite=True / --force)")
    if not isinstance(src, Mapping) and os.path.exists(dst) and os.path.samefile(os.fspath(src), dst):
        raise ConvertError("source and destination are the same file")
    sources, name, size, sha = _open_source(src)
    rep = ConvertReport(src=name if isinstance(src, Mapping) else os.fspath(src), dst=dst, src_bytes=sum(s.nbytes for s in sources.values()), n_tensors=len(sources))
    cfg, qmeta = validate(sources, cfg, sigma_shifts, rep, log)
    rep.cfg = cfg
    last = [0.0]

    def prog(n: str, done: int, total: int) -> None:
        if time.time() - last[0] > 2.0:
            last[0] = time.time()
            log(f"wrote {done / 2**30:.2f}/{total / 2**30:.2f} GiB ({n})")

    info = write_h3t(dst, cfg, sources, quant=qmeta or None, source_name=name, source_size=size, source_sha256=sha, chunk_bytes=chunk_bytes, progress=prog)
    rep.n_blocks, rep.block_nbytes, rep.dst_bytes = info["n_blocks"], info["block_nbytes"], info["file_size"]
    if verify:
        rep.verify_detail = verify_h3t(sources, dst, cfg, chunk_bytes=chunk_bytes, log=log)
        rep.verified = True
    rep.seconds = time.time() - t0
    return rep


# --------------------------------------------------------------------------------------------- synthetic checkpoints
def _gen(name: str, seed: int) -> torch.Generator:
    return torch.Generator().manual_seed((zlib.crc32(name.encode()) ^ (seed * 2654435761)) & 0x7FFFFFFF)


def synthetic_state_dict(cfg: H3Config, *, quant: bool = True, seed: int = 0, refiner: bool = True, lazy: bool = False) -> Dict[str, Union[torch.Tensor, TensorSource]]:
    """Random ComfyUI-layout tensors with the real dtypes/shapes for `cfg` (tests, GPU checks). Deterministic per name."""
    sd: Dict[str, Union[torch.Tensor, TensorSource]] = {}
    qjson = json.dumps({"format": QUANT_FORMAT, "group_size": cfg.quant_group, "convrot_groupsize": cfg.quant_convrot}, separators=(",", ":")).encode()
    h, inner = cfg.hidden, cfg.inner

    def add(name: str, dtype: str, shape: Sequence[int], make: Callable[[torch.Generator], torch.Tensor]) -> None:
        fn = lambda: make(_gen(name, seed))  # noqa: E731
        sd[name] = LazySource(dtype, shape, fn) if lazy else fn()

    def randn(dt: torch.dtype, shape, scale=0.05, off=0.0):
        return lambda g: (torch.randn(*shape, generator=g) * scale + off).to(dt)

    dims = {"attn.qkv_proj": (3 * inner, h), "attn.out_proj": (h, inner), "mlp.fc1": (2 * cfg.ffn, h), "mlp.fc2": (h, cfg.ffn)}

    def block(pre: str, quantised: bool, adaln: bool) -> None:
        for p, (n, k) in dims.items():
            if quantised:
                add(f"{pre}{p}.weight", "I8", (n, k // 2), lambda g, n=n, k=k: torch.randint(-128, 128, (n, k // 2), dtype=torch.int8, generator=g))
                add(f"{pre}{p}.weight_s_rel", "F8_E4M3", (n, k // cfg.quant_group), lambda g, n=n, k=k: (torch.rand(n, k // cfg.quant_group, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn))
                add(f"{pre}{p}.weight_s_channel", "F32", (n,), lambda g, n=n: torch.rand(n, generator=g) * 0.01 + 0.001)
                add(f"{pre}{p}.weight_codebook", "F32", (16,), lambda g: torch.linspace(-1.0, 1.0, 16) + torch.rand(16, generator=g) * 0.01)
                add(f"{pre}{p}.comfy_quant", "U8", (len(qjson),), lambda g: torch.tensor(list(qjson), dtype=torch.uint8))
            else:
                add(f"{pre}{p}.weight", "BF16", (n, k), randn(torch.bfloat16, (n, k), 0.02))
        for nm in ("norm1.weight", "norm2.weight"):
            add(pre + nm, "BF16", (h,), randn(torch.bfloat16, (h,), 0.05, 1.0))
        for nm in ("attn.q_norm.weight", "attn.k_norm.weight"):
            add(pre + nm, "BF16", (cfg.head_dim,), randn(torch.bfloat16, (cfg.head_dim,), 0.05, 1.0))
        if adaln:
            add(pre + "adaln_proj.linear.weight", "BF16", (18 * h, cfg.t_dim), randn(torch.bfloat16, (18 * h, cfg.t_dim), 0.05))
            add(pre + "adaln_proj.linear.bias", "F32", (18 * h,), randn(torch.float32, (18 * h,), 0.05))

    for i in range(cfg.layers):
        block(f"blocks.{i}.", quant, True)
    if refiner:
        for j in range(cfg.refiner_layers):
            block(f"token_refiner.blocks.{j}.", False, False)
        add("token_refiner.final_norm.weight", "BF16", (h,), randn(torch.bfloat16, (h,), 0.05, 1.0))
    for nm, dt, shp in (
        ("video_patch_proj.weight", "F32", (h, cfg.video_patch_dim)), ("video_patch_proj.bias", "F32", (h,)),
        ("audio_patch_proj.weight", "F32", (h, cfg.audio_channels)), ("audio_patch_proj.bias", "F32", (h,)),
        ("condition_proj.weight", "BF16", (h, cfg.text_dim)), ("condition_proj.bias", "BF16", (h,)),
        ("final_layer.adaln_proj.linear.weight", "BF16", (2 * h, cfg.t_dim)), ("final_layer.adaln_proj.linear.bias", "F32", (2 * h,)),
        ("final_layer.video_out.weight", "F32", (cfg.video_patch_dim, h)), ("final_layer.video_out.bias", "F32", (cfg.video_patch_dim,)),
        ("final_layer.audio_out.weight", "F32", (cfg.audio_channels, h)), ("final_layer.audio_out.bias", "F32", (cfg.audio_channels,)),
        ("final_layer.norm.weight", "BF16", (h,)), ("adaln_t_table", "F32", (cfg.curve_grid, cfg.t_dim)),
        ("rope.inv_freq", "F32", (cfg.rope_inv_freq_len,)), ("adaln_basis", "F32", (cfg.t_dim, h // 2)), ("adaln_mean", "F32", (h // 2,)),
    ):
        add(nm, dt, shp, randn(torch_dtype(dt), shp, 0.05))
    return sd


# --------------------------------------------------------------------------------------------- CLI
def describe(path: str) -> str:
    """Human-readable summary of an h3t file (h3-info)."""
    with H3TFile(path) as st:
        c = st.cfg
        lines = [
            f"{st.path}", f"  size        {st.file_size / 2**30:.3f} GiB, data section at {st.data_start} (aligned {BLOCK_ALIGN})",
            f"  source      {st.source_name or '?'} ({st.source_size / 2**30:.3f} GiB)",
            f"  model       {c.layers} blocks, hidden {c.hidden}, heads {c.heads}x{c.head_dim}, ffn {c.ffn}, refiner {c.refiner_layers}, curve {c.curve_grid}x{c.t_dim}",
            f"  quant       {st.quant or 'none (dense)'}",
            f"  block       {st.block_nbytes} bytes ({st.block_nbytes / 2**20:.2f} MiB), payload {st.block_layout_size}, {len(st.block_layout)} tensors, starts at {st.blocks_start}",
            f"  globals     {st.globals_nbytes(True) / 2**20:.1f} MiB total, {st.globals_nbytes(False) / 2**20:.1f} MiB without the token refiner",
            "  block layout (offset, bytes, dtype, shape, name):",
        ]
        lines += [f"    {e.offset:>10} {e.nbytes:>10} {e.dtype:<8} {str(list(e.shape)):<16} {e.name}" for e in st.block_layout]
        return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """h3-convert SRC DST [--no-verify] [--force]"""
    ap = argparse.ArgumentParser(prog="h3-convert", description="Convert a ComfyUI-layout H3 checkpoint into the streaming h3t format (lossless).")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--no-verify", action="store_true", help="skip the byte-for-byte re-read of the destination")
    ap.add_argument("--force", action="store_true", help="overwrite an existing destination")
    a = ap.parse_args(argv)
    try:
        rep = convert(a.src, a.dst, verify=not a.no_verify, overwrite=a.force, progress=lambda s: print(s, file=sys.stderr, flush=True))
    except H3TError as e:  # an invalid checkpoint is a user error, not a crash
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(rep.summary())
    return 0


def info_main(argv: Optional[Sequence[str]] = None) -> int:
    """h3-info PATH"""
    ap = argparse.ArgumentParser(prog="h3-info", description="Describe an h3t file.")
    ap.add_argument("path")
    print(describe(ap.parse_args(argv).path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
