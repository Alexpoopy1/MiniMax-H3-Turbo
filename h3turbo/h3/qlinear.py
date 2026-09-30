"""ConvRot W4A8 linear ops: decode, rotation, activation quantisation and the GEMM backends.

Layout of a W4A8 layer (see types.W4A8Weight): 4-bit codebook indices with fp8 group scales decode
LOSSLESSLY onto an int8 grid; that grid times a per-channel fp32 scale is the weight in a
ConvRot-rotated basis (block-wise regular Hadamard along K, block 256).  The activation is rotated
with the same (symmetric, involutory) matrix, quantised to int8 per token, multiplied in int8 with
int32 accumulation and rescaled.  All integer work is exact, so the only numerics that matter are the
activation rotation/quantiser and the epilogue; they follow comfy_kitchen 0.2.35 (Apache-2.0), the
reference ComfyUI itself runs.

Two backends: "ck" hands the whole op to comfy_kitchen's CUDA kernels; "torch" is portable (CPU ok)
and uses `torch._int_mm` on CUDA (with the weight passed TRANSPOSED, ~2.3x faster than a row-major
[K, N] copy) or an exact fp32 emulation elsewhere.
"""
from __future__ import annotations

import math
import sys
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .types import W4A8Weight, Weight

if sys.byteorder != "little":  # the int8-pair decode table reinterprets int16 as two little-endian int8
    raise ImportError("h3turbo.h3.qlinear needs a little-endian platform")

CONVROT = 256
_H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
_HAD_CACHE: Dict[Tuple[int, str, int, torch.dtype], torch.Tensor] = {}
_LOWP = (torch.bfloat16, torch.float16)
_ROT_TILE = 4096  # rows per rotation GEMM call on CUDA
_EPS_SCALE = 1e-30  # kitchen clamps the per-row scale here (before the x.dtype "tiny" guard)
_EXACT_K = 1024  # 1024 * 128 * 127 < 2^24: every fp32 partial sum of int8 products is an exact integer
_ACC_BYTES = 128 << 20  # budget for one int32 accumulator [rows, N]; rows per pass derive from it
_DECODE_ELEMS = 1 << 24  # int8 grid elements decoded per slab (bounds the int32 index temporary)
_CPU_INT_MM = False  # torch._int_mm on CPU is ~50x slower than the exact fp32 emulation
_STATE: Dict[str, Any] = {}  # cached capability probes and the comfy_kitchen import result


# --------------------------------------------------------------------------------------------- Hadamard
def hadamard_regular(size: int, device="cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalised regular Hadamard H = kron(H4, ..., H4) / sqrt(size); symmetric, H @ H = I.

    Entries are +-1/sqrt(size) = +-2^-j, exact in every float dtype.  Cached: do not mutate the result.
    """
    n = size
    while n > 1 and n % 4 == 0:
        n //= 4
    if size < 4 or n != 1:
        raise ValueError(f"regular Hadamard size must be a power of 4 (>= 4), got {size}")
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    key = (size, dev.type, -1 if dev.index is None else dev.index, dtype)
    h = _HAD_CACHE.get(key)
    if h is None:
        h4 = torch.tensor(_H4, dtype=torch.float32)
        full = h4
        for _ in range(round(math.log(size, 4)) - 1):
            full = torch.kron(full, h4)
        h = (full / math.sqrt(size)).to(device=dev, dtype=dtype).contiguous()
        _HAD_CACHE[key] = h
    return h


def _probe(key: str, fn) -> bool:
    """Run a tiny capability check once (torch versions/architectures differ) and cache the answer."""
    if key not in _STATE:
        try:
            _STATE[key] = bool(fn())
        except (RuntimeError, TypeError):
            _STATE[key] = False
    return _STATE[key]


def _mm_out_dtype_ok() -> bool:
    """torch.mm(bf16, bf16, out_dtype=float32): tensor-core GEMM with an fp32 result (torch >= 2.8, CUDA)."""
    return torch.cuda.is_available() and _probe("mm_out_dtype", lambda: torch.mm(*[torch.ones(16, 16, device="cuda", dtype=torch.bfloat16)] * 2, out_dtype=torch.float32).dtype == torch.float32)


def _mm_rot_tiled(x2: torch.Tensor, size: int) -> torch.Tensor:
    """CUDA rotation with a FIXED GEMM shape: [_ROT_TILE, size] @ [size, size] per call, the last tile zero-padded.
    cuBLAS picks kernels (and so fp32 accumulation orders) by problem size; with one shape every row is summed
    identically whatever the batch, so results are bit-stable across chunk sizes and batch compositions.
    bf16/fp16: tensor cores with fp32 output (the +-2^-j entries make every product exact); fp32: sgemm."""
    m, k = x2.shape
    lowp = x2.dtype in _LOWP and _mm_out_dtype_ok()
    rows = (x2 if lowp else x2.float()).reshape(-1, size)
    n = rows.shape[0]
    h = hadamard_regular(size, x2.device, rows.dtype)
    kw = {"out_dtype": torch.float32} if lowp else {}
    out = torch.empty(n, size, dtype=torch.float32, device=x2.device)
    for a in range(0, n, _ROT_TILE):
        b = min(a + _ROT_TILE, n)
        if b - a == _ROT_TILE:
            torch.mm(rows[a:b], h, out=out[a:b], **kw)
        else:
            out[a:b] = torch.mm(F.pad(rows[a:b], (0, 0, 0, _ROT_TILE - (b - a))), h, **kw)[: b - a]
    return out.reshape(m, k)


def _rot32(x2: torch.Tensor, size: int) -> torch.Tensor:
    """x2 [M, K] -> fp32 [M, K] rotated per `size` block, accumulated in fp32 (CUDA) or fp64 (CPU).

    CPU: a low-precision GEMM there is ~150x slower, and fp64 sums of <= 256 short-mantissa terms are exact, so the
    result is the correctly rounded rotation and does not depend on the row count or the BLAS blocking.
    """
    m, k = x2.shape
    if x2.is_cuda:
        return _mm_rot_tiled(x2, size)
    h = hadamard_regular(size, x2.device, torch.float64)
    return torch.matmul(x2.double().reshape(-1, k // size, size), h).reshape(m, k).float()


def rotate(x: torch.Tensor, size: int = CONVROT) -> torch.Tensor:
    """x[..., K] -> per-`size`-block x @ H, in x.dtype (accumulated in fp32 or wider, one rounding to x.dtype)."""
    if not x.is_floating_point():
        raise ValueError(f"rotate expects a floating tensor, got {x.dtype}")
    k = x.shape[-1]
    if k % size:
        raise ValueError(f"last dim {k} is not a multiple of the rotation block {size}")
    if x.numel() == 0:
        return x.clone()
    x2 = x.reshape(-1, k)
    if x.dtype == torch.float64:
        y = torch.matmul(x2.reshape(-1, k // size, size), hadamard_regular(size, x.device, x.dtype)).reshape(-1, k)
    else:
        y = _rot32(x2, size).to(x.dtype)
    return y.reshape(x.shape)


# --------------------------------------------------------------------------------------------- weights
def _fp8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.float8_e4m3fn) if t.dtype == torch.uint8 else t


def _check_weight(w: W4A8Weight) -> Tuple[int, int]:
    """Kitchen's validate_w4a8_operands, plus dtype/shape checks for the pieces we index directly."""
    if w.q.dim() != 2 or w.q.dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"W4A8 q must be a 2D int8 tensor, got {tuple(w.q.shape)} {w.q.dtype}")
    n, k = w.q.shape[0], w.q.shape[1] * 2
    gs, cr = w.group_size, w.convrot
    if gs < 4 or k % 16 or k % gs or k % cr or (16 % gs and gs % 16):
        raise ValueError(f"K={k} must be divisible by 16, group_size={gs} and convrot={cr}; group_size >= 4 dividing 16 or a multiple of 16")
    if tuple(w.s_rel.shape) != (n, k // gs):
        raise ValueError(f"s_rel must have shape {(n, k // gs)}, got {tuple(w.s_rel.shape)}")
    if tuple(w.s_ch.shape) != (n,):
        raise ValueError(f"s_ch must have shape {(n,)}, got {tuple(w.s_ch.shape)}")
    if tuple(w.codebook.shape) != (16,):
        raise ValueError(f"codebook must have shape (16,), got {tuple(w.codebook.shape)}")
    return n, k


def _pair_lut(codebook: torch.Tensor) -> torch.Tensor:
    """int16 [65536]: entry (s << 8 | byte) holds the two int8 levels (low nibble first, little endian) of a packed
    byte under fp8 scale byte s: round(codebook[nibble] * scale).clamp(-127, 127), exactly kitchen's formula."""
    dev = codebook.device
    scale = torch.arange(256, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn).float()
    lv = torch.nan_to_num((codebook.float()[None, :] * scale[:, None]).round_().clamp_(-127, 127))  # NaN scale bytes never occur in files
    lv8 = lv.to(torch.int8)  # [256 scale, 16 code]
    byte = torch.arange(256, device=dev)
    pair = torch.stack([lv8[:, byte & 15], lv8[:, byte >> 4]], dim=-1)  # [256, 256, 2]
    return pair.contiguous().view(torch.int16).reshape(-1)


def _decode_rows(w: W4A8Weight, r0: int, r1: int, lut: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """int8 grid rows [r0, r1) -> int8 [r1 - r0, K] (written into `out` when given, a contiguous slab)."""
    k = w.in_features
    n = r1 - r0
    q = w.q[r0:r1].contiguous().view(torch.uint8)
    res = out if out is not None else torch.empty(n, k, dtype=torch.int8, device=q.device)
    if w.s_rel.dtype in (torch.float8_e4m3fn, torch.uint8):
        gb = w.group_size // 2  # packed bytes per scale group
        if lut is None:
            lut = _pair_lut(w.codebook)
        s = _fp8(w.s_rel[r0:r1]).contiguous().view(torch.uint8)
        # fp8 scale byte in the high half of a 16-bit key; the add promotes to int32 in one pass
        key = torch.add(s.view(n, -1, 1).to(torch.int32).mul_(256), q.view(n, -1, gb))
        torch.index_select(lut, 0, key.reshape(-1), out=res.view(torch.int16).reshape(-1))
        return res
    # generic scale dtype (fp32/bf16/fp16): kitchen's formula verbatim
    p = q.to(torch.int32)
    code = torch.empty(n, k, dtype=torch.int32, device=q.device)
    code[:, 0::2] = p & 0xF
    code[:, 1::2] = p >> 4
    vals = w.codebook.float()[code].view(n, -1, w.group_size) * w.s_rel[r0:r1].float().unsqueeze(-1)
    res.copy_(vals.view(n, k).round_().clamp_(-127, 127))
    return res


@torch.no_grad()
def decode_int8_grid(w: W4A8Weight) -> torch.Tensor:
    """int8 [N, K]: the exact grid the int8 GEMM consumes (bit-identical to kitchen's _dequant_int4_grouped_to_int8)."""
    n, k = _check_weight(w)
    out = torch.empty(n, k, dtype=torch.int8, device=w.q.device)
    rows = max(1, _DECODE_ELEMS // k)
    lut = _pair_lut(w.codebook) if w.s_rel.dtype in (torch.float8_e4m3fn, torch.uint8) else None
    for a in range(0, n, rows):
        b = min(a + rows, n)
        _decode_rows(w, a, b, lut, out[a:b])
    return out


@torch.no_grad()
def dequantize(w: W4A8Weight, dtype: torch.dtype = torch.float32, rotated: bool = False, *, rows: Optional[slice] = None) -> torch.Tensor:
    """Dense [N, K] weight.  rotated=True: the stored ConvRot basis (grid * s_ch); False: the original basis
    (W_rot @ H per block).  The rotation is done in fp32 and rounded once to `dtype` (kitchen rounds twice
    for bf16).  `rows` restricts to a row slice so callers can bound memory."""
    n, k = _check_weight(w)
    if rows is not None and rows.step not in (None, 1):
        raise ValueError("rows must be a contiguous slice")
    r0, r1, _ = (rows or slice(0, n)).indices(n)
    out = torch.empty(max(r1 - r0, 0), k, dtype=dtype, device=w.q.device)
    lut = _pair_lut(w.codebook) if w.s_rel.dtype in (torch.float8_e4m3fn, torch.uint8) else None
    h = None if rotated else hadamard_regular(w.convrot, w.q.device, torch.float32)
    step = max(1, _DECODE_ELEMS // k)
    for a in range(r0, r1, step):
        b = min(a + step, r1)
        wr = _decode_rows(w, a, b, lut).float().mul_(w.s_ch[a:b].float().unsqueeze(1))
        if h is not None:
            wr = torch.matmul(wr.view(b - a, k // w.convrot, w.convrot), h).view(b - a, k)  # H symmetric+involutory: W = W_rot @ H
        out[a - r0 : b - r0] = wr
    return out


# --------------------------------------------------------------------------------------------- int8 GEMM
def _int_mm_ok(dev: torch.device) -> bool:
    if dev.type != "cuda" or not hasattr(torch, "_int_mm"):
        return False
    return _probe(f"int_mm:{dev}", lambda: (torch._int_mm(torch.ones(32, 16, device=dev, dtype=torch.int8), torch.ones(16, 8, device=dev, dtype=torch.int8)) == 16).all())


def _int8_mm(a: torch.Tensor, w8: torch.Tensor) -> torch.Tensor:
    """Exact int32 a[M, K] @ w8[N, K].T for int8 operands."""
    m, k = a.shape
    n = w8.shape[0]
    if m == 0 or n == 0 or k == 0:
        return torch.zeros(m, n, dtype=torch.int32, device=a.device)
    if a.is_cuda and _int_mm_ok(a.device) and k % 8 == 0:
        pm = 32 - m if m <= 16 else 0  # cuBLASLt int8 wants M > 16
        pn = -n % 8
        ap = F.pad(a, (0, 0, 0, pm)) if pm else a
        wp = F.pad(w8, (0, 0, 0, pn)) if pn else w8
        return torch._int_mm(ap, wp.t())[:m, :n]  # w8.t() stays a view: cuBLASLt takes the natural TN layout
    if not a.is_cuda and _CPU_INT_MM and hasattr(torch, "_int_mm"):
        return torch._int_mm(a, w8.t().contiguous())
    acc = None  # exact emulation: K chunks keep every partial sum an integer below 2^24
    for k0 in range(0, k, _EXACT_K):
        part = torch.mm(a[:, k0 : k0 + _EXACT_K].float(), w8[:, k0 : k0 + _EXACT_K].float().t()).to(torch.int32)
        acc = part if acc is None else acc.add_(part)
    return acc


# --------------------------------------------------------------------------------------------- activation quantiser
def _rotate_literal(x2: torch.Tensor, size: int) -> torch.Tensor:
    """Kitchen eager's rotation: one batched matmul in x.dtype (CUDA).  CPU uses the canonical rotation instead
    (a low-precision CPU GEMM is ~150x slower and its accumulation order is not a contract)."""
    if not x2.is_cuda:
        return _rot32(x2, size).to(x2.dtype)
    m, k = x2.shape
    return torch.matmul(x2.reshape(-1, k // size, size), hadamard_regular(size, x2.device, x2.dtype)).reshape(m, k)


def _quantize_rows(x2: torch.Tensor, size: int, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate + per-row int8 quantisation.  Returns (int8 [M, K], fp32 scale [M, 1]).

    Both modes divide the x.dtype rotation by the x.dtype-rounded scale and round half-to-even (as kitchen's
    eager `quantize_int8_rowwise`).  They differ in where the absmax comes from:
      "ck"    absmax of the UNROUNDED fp32 rotation; measured bit-identical (0 of 7.2M int8 values differ, bf16
              rows) to the fused quantiser comfy_kitchen's CUDA linear runs.  Row-independent: chunk-invariant;
      "eager" kitchen's portable path, literally: rotate in x.dtype, absmax of the ROUNDED rotation (differs from
              "ck" in ~1.6% of the int8 values for bf16, identical for fp32).
    """
    dt = x2.dtype
    if mode == "ck":
        y32 = _rot32(x2, size)
        mn, mx = torch.aminmax(y32, dim=-1, keepdim=True)
        amax = torch.maximum(mx, -mn)
        y = y32.to(dt) if dt != torch.float32 else y32
        del y32
    else:
        y = _rotate_literal(x2, size)
        mn, mx = torch.aminmax(y, dim=-1, keepdim=True)
        amax = torch.maximum(mx, -mn).float()
    xs = (amax / 127.0).clamp_(min=_EPS_SCALE)
    s = xs.to(dt)
    s = torch.where(s == 0, torch.full_like(s, torch.finfo(dt).tiny), s)
    y.div_(s).round_().clamp_(-128, 127)
    return y.to(torch.int8), xs


def quantize_activation(x: torch.Tensor, *, size: int = CONVROT, quant_mode: str = "ck") -> Tuple[torch.Tensor, torch.Tensor]:
    """x[..., K] -> (int8 [..., K], fp32 scale [..., 1]) of the rotated activation; exposed for analysis and tests."""
    _check_mode(quant_mode)
    k = x.shape[-1]
    if k % size:
        raise ValueError(f"last dim {k} is not a multiple of the rotation block {size}")
    with torch.no_grad():
        q, s = _quantize_rows(x.reshape(math.prod(x.shape[:-1]), k), size, quant_mode)
    return q.reshape(x.shape), s.reshape(*x.shape[:-1], 1)


def _check_mode(mode: str) -> None:
    if mode not in ("ck", "eager"):
        raise ValueError(f"quant_mode must be 'ck' or 'eager', got {mode!r}")


# --------------------------------------------------------------------------------------------- backends
def _ck():
    """comfy_kitchen's W4A8 op or None (cached), with the reason recorded."""
    if "ck" not in _STATE:
        try:
            import comfy_kitchen as ck
            from comfy_kitchen.tensor.w4a8_int8 import w4a8_int8_linear

            cuda = bool(ck.list_backends().get("cuda", {}).get("available"))
            _STATE["ck"] = (w4a8_int8_linear if cuda else None, None if cuda else "comfy_kitchen has no CUDA backend", getattr(ck, "__version__", None))
        except Exception as e:  # ImportError, or a broken compiled extension
            _STATE["ck"] = (None, f"{type(e).__name__}: {e}", None)
    return _STATE["ck"]


def available_backends() -> Dict[str, Dict[str, Any]]:
    """{"ck": {...}, "torch": {...}} with availability and the reason a backend is missing."""
    fn, why, ver = _ck()
    cuda = torch.cuda.is_available()
    return {
        "ck": {"available": fn is not None and cuda, "reason": why if fn is None else (None if cuda else "no CUDA device"), "version": ver},
        "torch": {"available": True, "int_mm_cuda": bool(cuda and hasattr(torch, "_int_mm")), "mm_out_dtype": bool(cuda and _mm_out_dtype_ok())},
    }


def resolve_backend(x: torch.Tensor, w: Weight, backend: str = "auto", precision: str = "a8") -> str:
    """The backend `linear` will actually use ("dense" for a dense weight)."""
    if backend not in ("auto", "ck", "torch"):
        raise ValueError(f"backend must be 'auto', 'ck' or 'torch', got {backend!r}")
    if precision not in ("a8", "a16"):
        raise ValueError(f"precision must be 'a8' or 'a16', got {precision!r}")
    if not isinstance(w, W4A8Weight):
        return "dense"
    if precision == "a16" or backend == "torch":
        return "torch"  # kitchen has no activation-unquantised mode
    fn, why, _ = _ck()
    usable = fn is not None and x.is_cuda and w.q.is_cuda
    if backend == "ck" and not usable:
        raise RuntimeError(f"backend 'ck' unavailable: {why or ('needs CUDA tensors' if not x.is_cuda else 'weights not on CUDA')}")
    if not usable and backend == "auto" and x.is_cuda and w.q.is_cuda:
        _warn_no_ck(why)
    return "ck" if usable else "torch"


_WARNED_NO_CK = False


def _warn_no_ck(why: Optional[str]) -> None:
    """The torch fallback is ~2x slower on the linears (measured), so say so once instead of degrading silently."""
    global _WARNED_NO_CK
    if not _WARNED_NO_CK:
        _WARNED_NO_CK = True
        warnings.warn(f"comfy_kitchen is unusable ({why or 'unknown reason'}); W4A8 linears fall back to the portable torch path, "
                      "about 2x slower on CUDA", RuntimeWarning, stacklevel=3)


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    """silu(gate) * up on [gate | up] halves, in x.dtype: the eager form ComfyUI applies before a W4A8 fc2."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


class _TorchRunner:
    """Portable W4A8 forward for one weight: decode the int8 grid once, then run token chunks into `out` slices."""

    def __init__(self, w: W4A8Weight, bias: Optional[torch.Tensor], out_dtype: torch.dtype, precision: str, quant_mode: str):
        self.w, self.out_dtype, self.precision, self.mode = w, out_dtype, precision, quant_mode
        self.n, self.k = _check_weight(w)
        self.w8 = decode_int8_grid(w)
        self.ws = w.s_ch.float().reshape(1, -1)
        self.bias = None if bias is None else bias.to(device=w.q.device)
        self.max_rows = max(32, _ACC_BYTES // (4 * self.n)) if precision == "a8" else 1 << 30

    def __call__(self, x2: torch.Tensor, out: torch.Tensor) -> None:
        (self._a8 if self.precision == "a8" else self._a16)(x2, out)

    def _a8(self, x2: torch.Tensor, out: torch.Tensor) -> None:
        xq, xs = _quantize_rows(x2, self.w.convrot, self.mode)
        acc = _int8_mm(xq, self.w8)
        # int32 * fp32 promotes to fp32 exactly like kitchen's `acc.float() * scale`, and out= rounds once to out_dtype
        if self.mode == "ck":
            torch.mul(torch.mul(acc, xs), self.ws, out=out)  # the CUDA epilogue is (acc * x_scale) * w_scale: 0 of 5.4M outputs differ given its scales
        else:
            torch.mul(acc, xs * self.ws, out=out)  # kitchen eager: acc * (x_scale * w_scale)
        if self.bias is not None:
            out.add_(self.bias.to(self.out_dtype))  # after the output rounding, as in kitchen's eager path

    def _a16(self, x2: torch.Tensor, out: torch.Tensor) -> None:
        """Unquantised activation.  The int8 levels are exact in bf16/fp16 (<= 8 significant bits), so the GEMM runs on
        (rotated x, levels) with fp32 accumulation and the per-channel scale is applied to the fp32 result: the weights
        are never rounded and the output is rounded once.  Rounding errors left: rotated x -> x.dtype, the output."""
        dt = x2.dtype
        xr = rotate(x2, self.w.convrot)
        tc = xr.is_cuda and dt in _LOWP and _mm_out_dtype_ok()  # tensor cores with fp32 output; CPU/fp32/fp64: plain GEMM in a wide dtype
        cdt = dt if tc else (torch.float64 if dt == torch.float64 else torch.float32)
        xg = xr.to(cdt)
        step = max(1, _DECODE_ELEMS // self.k)
        for a in range(0, self.n, step):
            b = min(a + step, self.n)
            lv = self.w8[a:b].to(cdt).t()
            y = torch.mm(xg, lv, out_dtype=torch.float32) if tc else torch.mm(xg, lv)
            y.mul_(self.w.s_ch[a:b].to(y.dtype))
            if self.bias is not None:
                y.add_(self.bias[a:b].to(y.dtype))
            out[:, a:b] = y


def _run_ck(x2: torch.Tensor, w: W4A8Weight, bias: Optional[torch.Tensor], out_dtype: torch.dtype) -> torch.Tensor:
    fn = _ck()[0]
    return fn(x2, w.q, _fp8(w.s_rel), w.s_ch, codebook=w.codebook, bias=bias, group_size=w.group_size, convrot_groupsize=w.convrot, out_dtype=out_dtype)


@torch.no_grad()
def _linear_w4a8(x2, w, bias, out_dtype, backend, precision, swiglu, chunk_tokens, quant_mode):
    n, k = _check_weight(w)
    kx = x2.shape[-1] // (2 if swiglu else 1)
    if kx != k:
        raise ValueError(f"input features {kx}{' (after swiglu)' if swiglu else ''} do not match the weight's K={k}")
    if x2.device != w.q.device:
        raise ValueError(f"input on {x2.device} but weight on {w.q.device}")
    if bias is not None and tuple(bias.shape) != (n,):
        raise ValueError(f"bias must have shape {(n,)}, got {tuple(bias.shape)}")
    m = x2.shape[0]
    if m == 0:
        return torch.empty(0, n, dtype=out_dtype, device=x2.device)
    kind = resolve_backend(x2, w, backend, precision)
    if kind != "torch" and (chunk_tokens or m) >= m:
        # one ck call: hand back its output as is instead of copying it into a second [m, n] buffer (a device copy per linear)
        out = _run_ck(_swiglu(x2) if swiglu else x2, w, bias, out_dtype)
        if tuple(out.shape) != (m, n) or out.dtype != out_dtype or not out.is_contiguous():
            raise RuntimeError(f"comfy_kitchen returned {tuple(out.shape)} {out.dtype}, expected ({m}, {n}) {out_dtype}")
        return out
    out = torch.empty(m, n, dtype=out_dtype, device=x2.device)
    runner = _TorchRunner(w, bias, out_dtype, precision, quant_mode) if kind == "torch" else None
    step = min(chunk_tokens or m, runner.max_rows if runner else m)
    for a in range(0, m, step):
        xc = x2[a : a + step]
        if swiglu:
            xc = _swiglu(xc)
        if runner:
            runner(xc, out[a : a + step])
        else:
            out[a : a + step] = _run_ck(xc, w, bias, out_dtype)
    return out


def linear(
    x: torch.Tensor,
    w: Weight,
    bias: Optional[torch.Tensor] = None,
    *,
    out_dtype: Optional[torch.dtype] = None,
    backend: str = "auto",
    precision: str = "a8",
    input_act: Optional[str] = None,
    chunk_tokens: Optional[int] = None,
    quant_mode: str = "ck",
) -> torch.Tensor:
    """out = act(x) @ W.T + bias, over the last dim of x.

    w: dense [N, K] tensor (plain F.linear in x.dtype) or W4A8Weight.
    backend: "ck" comfy_kitchen CUDA kernels (RuntimeError if unavailable or x is not on CUDA), "torch" portable,
        "auto" = ck when usable else torch.  precision "a16" never quantises activations: rotated x (rounded to x.dtype)
        times the exact int8 levels, fp32 accumulation, per-channel scale applied in fp32 (weights never rounded);
        it always runs on the torch code path.
    input_act="swiglu": x is [gate | up] (last dim 2*K); silu(gate) * up in x.dtype is applied first, separately
        (comfy_kitchen's W4A8 op has no fused activation; ComfyUI does the same).
    chunk_tokens bounds peak memory.  a8 is token-wise (per-row quantiser, integer GEMM, elementwise epilogue), so
        chunking is exact: bit-identical on CPU and, for quant_mode="ck", on CUDA (fixed-shape rotation GEMM);
        quant_mode="eager" on CUDA inherits cuBLAS's batch-dependent bf16 rotation (last-bit ties may differ).
    quant_mode: see _quantize_rows ("ck" reproduces the CUDA quantiser, "eager" kitchen's portable one).
    """
    if not x.is_floating_point():
        raise ValueError(f"x must be floating point, got {x.dtype}")
    if input_act not in (None, "none", "swiglu"):
        raise ValueError(f"input_act must be None or 'swiglu', got {input_act!r}")
    if chunk_tokens is not None and chunk_tokens < 1:
        raise ValueError("chunk_tokens must be >= 1")
    _check_mode(quant_mode)
    swiglu = input_act == "swiglu"
    if swiglu and x.shape[-1] % 2:
        raise ValueError(f"swiglu needs an even last dim, got {x.shape[-1]}")
    out_dtype = out_dtype or x.dtype
    kind = resolve_backend(x, w, backend, precision)
    lead = x.shape[:-1]
    x2 = x.reshape(math.prod(lead), x.shape[-1])  # explicit: reshape(-1, K) is ambiguous for zero tokens
    if kind != "dense":
        y = _linear_w4a8(x2, w, bias, out_dtype, backend, precision, swiglu, chunk_tokens, quant_mode)
        return y.reshape(*lead, y.shape[-1])
    if x2.shape[-1] // (2 if swiglu else 1) != w.shape[-1]:
        raise ValueError(f"input features {x2.shape[-1]} do not match the dense weight {tuple(w.shape)}")
    step = chunk_tokens or max(x2.shape[0], 1)
    cpu_lowp = x.device.type == "cpu" and x.dtype in _LOWP  # low-precision GEMM is ~150x slower on CPU; fp32 sums + one rounding are equivalent
    wd = w.to(dtype=torch.float32 if cpu_lowp else x.dtype)
    bd = None if bias is None else bias.to(dtype=wd.dtype)
    outs = []
    for a in range(0, max(x2.shape[0], 1), step):
        xc = x2[a : a + step]
        if swiglu:
            xc = _swiglu(xc)
        y = F.linear(xc.to(wd.dtype), wd, bd)
        outs.append(y.to(out_dtype))
    y = outs[0] if len(outs) == 1 else torch.cat(outs, 0)
    return y.reshape(*lead, y.shape[-1])
