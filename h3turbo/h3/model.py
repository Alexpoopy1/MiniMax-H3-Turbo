"""Forward pass of the official H3 audio-video DiT (50 blocks, hidden 5376) over pluggable block weights.

Same network as ComfyUI's native H3 (curve-form AdaLN checkpoints), written for streaming: the block loop only touches
weights through a `BlockProvider`, every large linear goes through `qlinear.linear` (W4A8 or dense), and the hot path works
in place / in token chunks so a 6 GB card can hold the activations of a ~5k-token clip next to a few streamed blocks.

Numerics follow the reference op by op: in-place bf16 modulation, fp32 head/AdaLN math, RMSNorm via F.rms_norm, rope tables
rounded to the compute dtype, q/k norm+rope through comfy_kitchen's fused op or a torch reproduction of that kernel
(layout.rms_norm_fused_order), and ComfyUI's SDPA backend priority. The checkpoint stores some tensors in fp32 (patch
projections, output heads, AdaLN biases); ComfyUI's mixed-precision loader keeps every parameter in the compute dtype, so
natively they are bf16-rounded (measured on the real file). By default this model does the same; `fp32_islands=True` keeps
the checkpoint's fp32 values.
"""
from __future__ import annotations

import inspect
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import H3Config
from .layout import (AUDIO_COND_TIMESTEP, VISUAL_COND_TIMESTEP, PackedLayout, TimePlan, bank_head_weights, curve_lerp,
                     head_bank_range, layout_key, norm_rope_, pack_audio, pad_to_patch, patchify_video, plan_time,
                     resolve_rope_impl, rope_angles, rope_ck_table, time_shift_sigma, unpack_audio, unpatchify_video)
from .types import BlockProvider, BlockWeights, GlobalWeights, W4A8Weight, Weight

_VIDEO_ROWS = ("cond", "ref_img", "video")
_COND_VIDEO, _COND_AUDIO = ("cond", "ref_img"), ("cond_audio", "ref_audio")
_ROW_CHUNK = 4096  # rows per modulation gather
# The fp32 patch projection and the output heads run as ONE GEMM up to this many rows: cuBLAS picks its tiling from the row count,
# so splitting a 14k-row projection (a 832x480, 5 s clip) changed the last bit against ComfyUI and that amplified over 8 steps
# (measured: relL2 4e-5 per forward, video PSNR 29 dB after sampling). 65536 rows of fp32 hidden is ~1.4 GB, far above any clip a 6 GB card runs.
_PROJ_ROWS = 1 << 16


def _qlinear():
    from . import qlinear  # only needed once a layer is W4A8

    return qlinear


class ResidentProvider(BlockProvider):
    """All blocks already on the compute device."""

    def __init__(self, blocks: Sequence[BlockWeights]):
        self.blocks = list(blocks)

    def begin_forward(self) -> None:
        pass

    def acquire(self, i: int) -> BlockWeights:
        return self.blocks[i]

    def release(self, i: int) -> None:
        pass

    def end_forward(self) -> None:
        pass


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight if weight.dtype == x.dtype else weight.to(x.dtype), eps)


def _comfy_sdpa_priority():
    """ComfyUI's SDPA backend order for big calls; PyTorch's default picks mem-efficient where flash is missing (Windows)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return None
    if "set_priority" not in inspect.signature(sdpa_kernel).parameters:
        return None
    return sdpa_kernel, [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def _int8_attention_fn(device):
    """comfy_kitchen's INT8 SDPA (int8 Q/K/V/P with a Hadamard rotation of Q and K): about 2x faster than cuDNN attention on
    an RTX 3050 but NOT the reference numerics (measured 1.6% rel-L2 per call on Gaussian data), so it is opt-in."""
    try:
        from comfy_kitchen import sage_attention as sa
    except Exception as e:
        raise RuntimeError(f"attn_impl='int8' needs comfy_kitchen ({e!r})") from None
    if not sa.is_available(torch.device(device)):
        raise RuntimeError("attn_impl='int8' needs comfy_kitchen's CUDA extension on an sm75+ GPU")
    return sa.int8_attention


def _swiglu(z: torch.Tensor) -> torch.Tensor:
    gate, up = z.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


def _pieces(segments):
    """(start, stop, row) pieces; per-token row tensors are split so their [n, hidden] gathers stay small."""
    for a, b, row in segments:
        if isinstance(row, torch.Tensor):
            for c in range(0, b - a, _ROW_CHUNK):
                yield a + c, min(b, a + c + _ROW_CHUNK), row[c:c + _ROW_CHUNK]
        elif b > a:
            yield a, b, row


def _scale_shift_(h: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, segments) -> torch.Tensor:
    for a, b, row in _pieces(segments):
        h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _gate_(x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, segments) -> torch.Tensor:
    for a, b, row in _pieces(segments):
        x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
    return x


class H3Model:
    """H3 DiT over a BlockProvider. backend/precision: qlinear backend and "a8" (native W4A8) or "a16" (activations unquantised).
    mlp_chunk/attn_chunk/rope_chunk: token/query/row chunks bounding peak memory, exact
    up to GEMM tiling. rope_impl (auto|ck|torch|eager), rope_dtype, attn_backend (auto = ComfyUI's SDPA order | default),
    fp32_islands: see the module docstring."""

    def __init__(self, cfg: H3Config, glob: GlobalWeights, provider: BlockProvider, *, device,
                 dtype: torch.dtype = torch.bfloat16, backend: str = "auto", precision: str = "a8",
                 mlp_chunk: Optional[int] = None, attn_chunk: Optional[int] = None,
                 rope_dtype: Optional[torch.dtype] = None, rope_chunk: int = 1024, rope_impl: str = "auto",
                 fp32_islands: bool = False, attn_backend: str = "auto", attn_impl: str = "sdpa"):
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"compute dtype must be bfloat16 or float32, got {dtype}")
        if backend not in ("auto", "ck", "torch") or precision not in ("a8", "a16") or attn_backend not in ("auto", "default"):
            raise ValueError(f"bad backend/precision/attn_backend: {backend!r}/{precision!r}/{attn_backend!r}")
        if attn_impl not in ("sdpa", "int8"):
            raise ValueError(f"attn_impl must be 'sdpa' (exact, the reference) or 'int8', got {attn_impl!r}")
        if cfg.curve_grid < 2 or glob.adaln_t_table is None:
            raise NotImplementedError("only curve-form checkpoints (adaln_t_table, no time embedder) are supported")
        if tuple(cfg.patch[1:]) != (2, 2):
            raise NotImplementedError("the packed layout assumes 2x2 spatial patches")
        if 6 * cfg.rope_inv_freq_len > cfg.head_dim:
            raise ValueError(f"rotary width {6 * cfg.rope_inv_freq_len} exceeds head_dim {cfg.head_dim}")
        for name, v in (("mlp_chunk", mlp_chunk), ("attn_chunk", attn_chunk), ("rope_chunk", rope_chunk)):
            if v is not None and v < 1:
                raise ValueError(f"{name} must be >= 1")
        self.cfg, self.provider, self.device, self.dtype = cfg, provider, torch.device(device), dtype
        self.backend, self.precision = backend, precision
        self.mlp_chunk, self.attn_chunk, self.rope_chunk = mlp_chunk, attn_chunk, rope_chunk
        self._int8_attention = _int8_attention_fn(device) if attn_impl == "int8" else None
        self.attn_impl = attn_impl
        self.rope_dtype = rope_dtype or dtype
        self.rope_impl = resolve_rope_impl(rope_impl, device, dtype, self.rope_dtype)
        self.glob = glob
        f32 = torch.float32
        self.island_dtype = f32 if fp32_islands else dtype
        self._sdpa_prio = _comfy_sdpa_priority() if attn_backend == "auto" and self.device.type == "cuda" else None
        d = lambda t, dt=None: t.to(device=self.device, dtype=dt or t.dtype)  # noqa: E731
        isl = lambda t: self._island(d(t))  # noqa: E731
        self.vpw, self.vpb = isl(glob.video_patch_w), isl(glob.video_patch_b)
        self.apw, self.apb = isl(glob.audio_patch_w), isl(glob.audio_patch_b)
        self.cond_w, self.cond_b = d(glob.condition_w, dtype), d(glob.condition_b, dtype)
        self.table, self.inv_freq = d(glob.adaln_t_table, f32), d(glob.rope_inv_freq, f32)
        self.final_norm = d(glob.final_norm, dtype)
        self.final_adaln_w, self.final_adaln_b = isl(glob.final_adaln_w), isl(glob.final_adaln_b)
        self.vow, self.vob = isl(glob.video_out_w), isl(glob.video_out_b)
        self.aow, self.aob = isl(glob.audio_out_w), isl(glob.audio_out_b)
        self.refiner_norm = None if glob.refiner_final_norm is None else d(glob.refiner_final_norm, dtype)
        pd, ac = cfg.video_patch_dim, cfg.audio_channels
        if self.vow.shape[0] % pd or self.aow.shape[0] % ac or self.vow.shape[0] // pd != self.aow.shape[0] // ac:
            raise ValueError(f"output heads {tuple(self.vow.shape)}/{tuple(self.aow.shape)} do not tile {pd}/{ac} rows")
        self.head_bank = self.vow.shape[0] // pd
        self._layouts: Dict[tuple, PackedLayout] = {}
        self._rope: Optional[tuple] = None

    # ------------------------------------------------------------------ primitives

    def _island(self, t: torch.Tensor) -> torch.Tensor:
        """fp32 view of a tensor that natively lives in `island_dtype` (rounded through it), for fp32 math."""
        return t.to(self.island_dtype).to(torch.float32)

    def _matmul(self, x: torch.Tensor, w: Weight, input_act: Optional[str] = None) -> torch.Tensor:
        if isinstance(w, torch.Tensor):  # dense: the reference's eager path
            if input_act == "swiglu":
                x = _swiglu(x)
            return F.linear(x, w if w.dtype == x.dtype else w.to(x.dtype))
        return _qlinear().linear(x, w, backend=self.backend, precision=self.precision, input_act=input_act)

    def _mlp(self, h: torch.Tensor, w: BlockWeights) -> torch.Tensor:
        n = h.shape[0]
        step = self.mlp_chunk or n
        if step >= n:
            return self._matmul(self._matmul(h, w.fc1), w.fc2, "swiglu")
        out = torch.empty(n, self.cfg.hidden, dtype=h.dtype, device=h.device)
        for a in range(0, n, step):  # token-wise ops only, so chunking is exact and never builds [n, 2*ffn]
            out[a:a + step] = self._matmul(self._matmul(h[a:a + step], w.fc1), w.fc2, "swiglu")
        return out

    def _attention(self, h: torch.Tensor, w: BlockWeights, rope: Optional[tuple]) -> torch.Tensor:
        cfg = self.cfg
        s, heads, hd = h.shape[0], cfg.heads, cfg.head_dim
        inner = heads * hd
        qkv = self._matmul(h, w.qkv)
        q, k, v = (t.view(s, heads, hd) for t in qkv.split(inner, dim=-1))
        if rope is None:  # token refiner: plain per-head norm (fresh q/k like the reference), no positions
            q, k = _rms(q, w.q_norm, cfg.qk_norm_eps), _rms(k, w.k_norm, cfg.qk_norm_eps)
        else:
            norm_rope_(q, k, w.q_norm, w.k_norm, cfg.qk_norm_eps, *rope[:2], impl=self.rope_impl, table=rope[2],
                       chunk=self.rope_chunk)
        qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # [1, heads, S, hd] strided views
        if self._int8_attention is not None and rope is not None:  # main blocks only; the short token refiner stays exact
            o = self._int8_attention(qh.contiguous(), kh.contiguous(), vh.contiguous()).transpose(1, 2).reshape(s, inner)
            del qkv, q, k, v, qh, kh, vh
            return self._matmul(o, w.out)
        step = self.attn_chunk or s
        prio = self._sdpa_prio if s * inner >= 1024 * 128 else None  # ComfyUI's size rule for using the priority order

        def sdpa(qc):
            if prio is None:
                return F.scaled_dot_product_attention(qc, kh, vh)
            with prio[0](prio[1], set_priority=True):
                return F.scaled_dot_product_attention(qc, kh, vh)

        if step >= s:
            o = sdpa(qh).transpose(1, 2).reshape(s, inner)
        else:  # query rows are independent, so chunking is exact
            o = torch.empty(s, inner, dtype=h.dtype, device=h.device)
            for a in range(0, s, step):
                o[a:a + step] = sdpa(qh[:, :, a:a + step])[0].transpose(0, 1).reshape(-1, inner)
        del qkv, q, k, v, qh, kh, vh
        return self._matmul(o, w.out)

    def _adaln(self, t_emb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, parts: int, mods: int):
        z = F.linear(t_emb, self._island(weight), self._island(bias))
        return z.view(t_emb.shape[0] * mods, -1).chunk(parts, dim=-1)

    def _block(self, x: torch.Tensor, w: BlockWeights, t_emb: torch.Tensor, segments, rope) -> torch.Tensor:
        if w.adaln_w is None or w.adaln_b is None:
            raise ValueError("DiT block weights need adaln_w/adaln_b")
        eps = self.cfg.norm_eps
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self._adaln(t_emb, w.adaln_w, w.adaln_b, 6, 3)
        h = _scale_shift_(_rms(x, w.norm1, eps), sh_a, sc_a, segments)
        _gate_(x, g_a, self._attention(h, w, rope), segments)
        h = _scale_shift_(_rms(x, w.norm2, eps), sh_m, sc_m, segments)
        return _gate_(x, g_m, self._mlp(h, w), segments)

    def _refiner_block(self, x: torch.Tensor, w: BlockWeights) -> torch.Tensor:
        eps = self.cfg.norm_eps
        x = self._attention(_rms(x, w.norm1, eps), w, None).add_(x)
        return self._mlp(_rms(x, w.norm2, eps), w).add_(x)

    # ------------------------------------------------------------------ text

    @torch.no_grad()
    def encode_text(self, text_states: torch.Tensor) -> torch.Tensor:
        """[1, L, text_dim] Qwen states -> [1, L, hidden] refined embeddings (idempotent on already-refined input)."""
        cfg = self.cfg
        if text_states.ndim != 3 or text_states.shape[0] != 1:
            raise ValueError(f"text_states must be [1, L, C], got {tuple(text_states.shape)}")
        if text_states.shape[-1] == cfg.hidden:
            return text_states.to(self.device, self.dtype)
        if text_states.shape[-1] != cfg.text_dim or len(self.glob.refiner) != cfg.refiner_layers or self.refiner_norm is None:
            raise ValueError(f"text_states width {text_states.shape[-1]} is neither hidden nor text_dim {cfg.text_dim}, or the refiner is missing")
        x = F.linear(text_states[0].to(self.device, self.dtype), self.cond_w, self.cond_b)
        for blk in self.glob.refiner:
            on_device = BlockWeights(**{k: None if v is None else v.to(self.device) for k, v in vars(blk).items()})
            x = self._refiner_block(x, on_device)  # the refiner may live off-device: one block at a time
        return _rms(x, self.refiner_norm, cfg.final_norm_eps).unsqueeze(0)

    # ------------------------------------------------------------------ forward

    def _layout(self, payload, text_len, shape, audio_t) -> PackedLayout:
        layout = payload.get("layout")
        sig = (text_len, *shape, audio_t)
        if layout is not None and tuple(layout.signature) == sig:
            return layout
        key = layout_key(text_len, (*shape, audio_t), payload.get("keyframes"), payload.get("refs"))
        if key not in self._layouts:
            if len(self._layouts) >= 4:
                self._layouts.pop(next(iter(self._layouts)))
            self._layouts[key] = PackedLayout.build(text_len, *shape, audio_t, keyframes=payload.get("keyframes"),
                                                    refs=payload.get("refs"))
        return self._layouts[key]

    def _rope_tables(self, layout) -> tuple:
        """(cos, sin, kitchen table or None) for a layout, cached while the same layout object keeps being used."""
        if self._rope is None or self._rope[0] is not layout:
            ang = rope_angles(layout.position_ids, self.inv_freq, self.device)
            cos, sin = torch.cos(ang).to(self.rope_dtype), torch.sin(ang).to(self.rope_dtype)
            self._rope = (layout, cos, sin, rope_ck_table(cos, sin) if self.rope_impl == "ck" else None)
        return self._rope[1:]

    def _cond_rows(self, latents, pack, aug: float, seed: int) -> Optional[torch.Tensor]:
        rows = []
        for z in latents or ():
            r = pack(z.to(torch.float32))
            if aug < 1.0:  # every condition restarts the same noise stream on purpose
                noise = torch.randn(r.shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(self.device))
        return torch.cat(rows, dim=0) if rows else None

    def _project(self, rows: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """fp32 patch projection into the compute dtype, one GEMM up to _PROJ_ROWS rows (see there)."""
        step, out = _PROJ_ROWS, torch.empty(rows.shape[0], w.shape[0], dtype=self.dtype, device=rows.device)
        for a in range(0, rows.shape[0], step):
            out[a:a + step] = F.linear(rows[a:a + step], w, b)
        return out

    def _embed(self, layout, ctx, video_x, audio_x, payload, v_aug: float, a_aug: float) -> torch.Tensor:
        cfg, patch, seed = self.cfg, tuple(self.cfg.patch), int(payload.get("seed") or 0)
        cond_v = self._cond_rows(payload.get("cond_video_latents"), lambda z: patchify_video(z, patch), v_aug, seed)
        cond_a = self._cond_rows(payload.get("cond_audio_latents"), pack_audio, a_aug, seed + 1)
        target_v = patchify_video(video_x.to(self.device, torch.float32), patch)
        target_a = pack_audio(audio_x.to(self.device, torch.float32))
        img, aud, ci, ca = [], [], 0, 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "video":
                img.append(target_v)
            elif kind == "audio":
                aud.append(target_a)
            elif kind in _COND_VIDEO or kind in _COND_AUDIO:
                video = kind in _COND_VIDEO
                src, at = (cond_v, ci) if video else (cond_a, ca)
                if src is None or at + n > src.shape[0]:
                    raise ValueError(f"payload has too few conditioning {'video' if video else 'audio'} rows for segment {kind}")
                (img if video else aud).append(src[at:at + n])
                ci, ca = (ci + n, ca) if video else (ci, ca + n)
        if (cond_v is not None and ci != cond_v.shape[0]) or (cond_a is not None and ca != cond_a.shape[0]):
            raise ValueError("payload conditioning rows do not match the layout (keyframes/refs vs cond latents)")
        img_e, aud_e = self._project(torch.cat(img), self.vpw, self.vpb), self._project(torch.cat(aud), self.apw, self.apb)
        h = torch.empty(layout.seq_len, cfg.hidden, dtype=self.dtype, device=self.device)
        io = ao = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = ctx[0]
            elif kind in _VIDEO_ROWS:
                h[a:b] = img_e[io:io + n]
                io += n
            else:
                h[a:b] = aud_e[ao:ao + n]
                ao += n
        return h

    def _final(self, h, t_emb, plan: TimePlan, sigma, sample_sigmas, shifts):
        cfg = self.cfg
        shift, scale = self._adaln(t_emb, self.final_adaln_w, self.final_adaln_b, 2, 1)

        def head(seg, w, b):  # token-wise; one GEMM up to _PROJ_ROWS rows like the reference, chunked only beyond that
            a0, b0, row = seg
            row = row.to(self.device) if isinstance(row, torch.Tensor) else row
            outs = []
            for a in range(a0, b0, _PROJ_ROWS):
                e = min(b0, a + _PROJ_ROWS)
                r = row[a - a0:e - a0] if isinstance(row, torch.Tensor) else row
                outs.append(F.linear(_rms(h[a:e], self.final_norm, cfg.final_norm_eps) * (1.0 + scale[r]) + shift[r], w, b))
            return torch.cat(outs)

        n, vw, vb, aw, ab = self.head_bank, self.vow, self.vob, self.aow, self.aob
        if n > 1:
            if sample_sigmas is None:
                raise ValueError("this checkpoint has a multi-head output bank; pass the sampler's sigma schedule as sample_sigmas")
            start, stop = head_bank_range(sample_sigmas, sigma, shifts[0], n)
            (vw, vb), (aw, ab) = bank_head_weights(vw, vb, n, start, stop, shifts[0]), bank_head_weights(aw, ab, n, start, stop, shifts[1])
        return head(plan.video_seg, vw, vb), head(plan.audio_seg, aw, ab)

    @torch.no_grad()
    def forward(self, x: Sequence[torch.Tensor], sigma, text_states: Optional[torch.Tensor], *,
                payload: Optional[Mapping[str, Any]] = None, denoise_mask: Optional[torch.Tensor] = None,
                audio_denoise_mask: Optional[torch.Tensor] = None, sample_sigmas=None,
                refined_text: Optional[torch.Tensor] = None,
                shifts: Optional[Tuple[float, float]] = None) -> List[torch.Tensor]:
        """One velocity prediction. x = [video [1,C,T,H,W], audio [1,Ca,2,Ta]] (each stream's own latent), sigma = the
        video sigma (float or fp32 scalar tensor). Returns [-video_v, -audio_v] like the reference, times the denoise masks.
        Per-step callers should pass `refined_text=encode_text(...)` computed once: with the refiner off the GPU, recomputing it
        every call copies ~1.5 GB of refiner weights per step."""
        cfg, payload = self.cfg, payload or {}
        if len(x) != 2:
            raise ValueError("x must be [video, audio]")
        video_x, audio_x = x
        if video_x.ndim != 5 or video_x.shape[0] != 1 or video_x.shape[1] != cfg.video_channels:
            raise ValueError(f"video latent must be [1, {cfg.video_channels}, T, H, W], got {tuple(video_x.shape)}")
        if audio_x.ndim != 4 or audio_x.shape[0] != 1 or audio_x.shape[1] != cfg.audio_channels or audio_x.shape[2] != 2:
            raise ValueError(f"audio latent must be [1, {cfg.audio_channels}, 2, Ta], got {tuple(audio_x.shape)}")
        for name, m, ref_shape in (("denoise_mask", denoise_mask, video_x.shape[2:]), ("audio_denoise_mask", audio_denoise_mask, audio_x.shape[2:])):
            if m is not None and (m.shape[0] != 1 or tuple(m.shape[2:]) != tuple(ref_shape) or m.ndim != len(ref_shape) + 2):
                raise ValueError(f"{name} must be [1, C, {', '.join(map(str, ref_shape))}], got {tuple(m.shape)}")
        sigma_v = (sigma.detach().flatten()[0].to("cpu", torch.float32) if isinstance(sigma, torch.Tensor)
                   else torch.tensor(float(sigma), dtype=torch.float32))
        if not math.isfinite(float(sigma_v)):
            raise ValueError("sigma must be finite")
        sigma_v = sigma_v.clamp(min=1e-6)  # same clamp as the reference
        shifts = (float(shifts[0]), float(shifts[1])) if shifts is not None else (cfg.sigma_shift_video, cfg.sigma_shift_audio)

        # the sampler may carry audio scaled onto the video schedule: undo it here, redo it on the velocity
        scale = float(1.0 if payload.get("audio_scale") is None else payload["audio_scale"])
        audio_src = audio_x
        if scale != 1.0:
            sigma_a = time_shift_sigma(sigma_v, *shifts)
            carry = (sigma_a / sigma_v).to(audio_src.dtype)
            audio_x = audio_src * carry

        out_v, out_a = self._forward(video_x, audio_x, sigma_v, shifts, text_states, refined_text, payload,
                                     denoise_mask, audio_denoise_mask, sample_sigmas)
        if denoise_mask is not None:  # masked rows predict at mask * sigma: scale their velocity to match
            out_v = out_v * denoise_mask.to(out_v.device, out_v.dtype)
        if audio_denoise_mask is not None:
            out_a = out_a * audio_denoise_mask.to(out_a.device, out_a.dtype)
        if scale != 1.0:
            out_a = ((1.0 - scale) * (audio_src.to(out_a.device) * carry.to(out_a.device))
                     + (1.0 + (scale - 1.0) * sigma_a).to(out_a.dtype) * out_a)
        return [out_v, out_a]

    def _forward(self, video_x, audio_x, sigma, shifts, text_states, refined_text, payload, denoise_mask,
                 audio_denoise_mask, sample_sigmas) -> List[torch.Tensor]:
        cfg, in_device = self.cfg, video_x.device
        orig_t, orig_h, orig_w = video_x.shape[2:]
        padded = pad_to_patch(video_x.to(self.device), cfg.patch)
        lat_t, lat_h, lat_w = padded.shape[2:]
        src = refined_text if refined_text is not None else text_states
        if src is None:
            raise ValueError("pass text_states or refined_text")
        ctx = self.encode_text(src)  # no-op when already hidden-wide
        layout = self._layout(payload, ctx.shape[1], (lat_t, lat_h, lat_w), audio_x.shape[-1])
        aug = lambda k, default: default if payload.get(k) is None else float(payload[k])  # noqa: E731
        v_aug, a_aug = aug("visual_cond_noise_aug", VISUAL_COND_TIMESTEP), aug("audio_cond_noise_aug", AUDIO_COND_TIMESTEP)
        plan = plan_time(layout, sigma, shifts, visual_aug=v_aug, audio_aug=a_aug,
                         text_tags=payload.get("text_token_tags"), denoise_mask=denoise_mask,
                         audio_denoise_mask=audio_denoise_mask)
        h = self._embed(layout, ctx, padded, audio_x, payload, v_aug, a_aug)
        t_emb = curve_lerp(self.table, plan.t_values)
        segments = [(a, b, r.to(self.device) if isinstance(r, torch.Tensor) else r) for a, b, r in plan.mod_segments]
        rope = self._rope_tables(layout)
        prov = self.provider
        prov.begin_forward()
        try:
            for i in range(cfg.layers):
                w = prov.acquire(i)
                try:
                    h = self._block(h, w, t_emb, segments, rope)
                finally:
                    prov.release(i)
        finally:
            prov.end_forward()
        v, a = self._final(h, t_emb, plan, sigma, sample_sigmas, shifts)
        video = unpatchify_video(v, lat_t, lat_h // 2, lat_w // 2, cfg.video_channels, cfg.patch)
        video = video[:, :, :orig_t, :orig_h, :orig_w]
        return [(-video).to(in_device, video_x.dtype), (-unpack_audio(a)).to(in_device, audio_x.dtype)]


def weights_from_state_dict(sd: Mapping[str, torch.Tensor], cfg: H3Config, device, dtype: torch.dtype,
                            *, prefix: str = "") -> Tuple[GlobalWeights, List[BlockWeights]]:
    """Build model weights from a ComfyUI-layout state dict (dense tensors and/or W4A8 layers).

    Dense linears and norms are cast to the compute `dtype`; the fp32 islands (patch projections, output
    heads, AdaLN tables/biases) stay fp32. Shapes are checked against `cfg`; every missing key is reported at once.
    """
    f32, missing = torch.float32, []
    hid, inner, hd = cfg.hidden, cfg.inner, cfg.head_dim

    def fetch(name: str, shape: Optional[Tuple[int, ...]] = None):
        t = sd.get(prefix + name)
        if t is None:
            missing.append(prefix + name)
            return None
        if shape is not None and tuple(t.shape) != tuple(shape):
            raise ValueError(f"{prefix + name}: expected shape {tuple(shape)}, got {tuple(t.shape)}")
        return t

    def cast(name, shape, dt):
        t = fetch(name, shape)
        return None if t is None else t.to(device=device, dtype=dt or t.dtype)

    def linear(name: str, n: int, k: int) -> Weight:
        q = fetch(name + ".weight")
        if q is None:
            return None
        if (prefix + name + ".weight_s_rel") not in sd:
            if tuple(q.shape) != (n, k):
                raise ValueError(f"{prefix + name}.weight: expected {(n, k)}, got {tuple(q.shape)}")
            return q.to(device=device, dtype=dtype)
        raw = sd.get(prefix + name + ".comfy_quant")
        conf = json.loads(raw.cpu().numpy().tobytes()) if raw is not None else {}
        if conf.get("format", "asym_w4a8_int8") != "asym_w4a8_int8" or conf.get("full_precision_matrix_mult"):
            raise NotImplementedError(f"{prefix + name}: unsupported quantization config {conf}")
        params = conf.get("params") if isinstance(conf.get("params"), dict) else {}
        group = int(conf.get("group_size", params.get("group_size", cfg.quant_group)))
        rot = int(conf.get("convrot_groupsize", params.get("convrot_groupsize", cfg.quant_convrot)))
        s_rel, s_ch, book = fetch(name + ".weight_s_rel"), fetch(name + ".weight_s_channel", (n,)), fetch(name + ".weight_codebook", (16,))
        if any(t is None for t in (s_rel, s_ch, book)):
            return None
        if s_rel.dtype == torch.uint8:
            s_rel = s_rel.view(torch.float8_e4m3fn)
        if q.ndim != 2 or (q.shape[0], q.shape[1] * 2) != (n, k) or k % group or k % rot or tuple(s_rel.shape) != (n, k // group):
            raise ValueError(f"{prefix + name}: W4A8 shapes q{tuple(q.shape)} s_rel{tuple(s_rel.shape)} do not match {(n, k)}/g{group}/r{rot}")
        return W4A8Weight(q.to(device=device, dtype=torch.int8), s_rel.to(device), s_ch.to(device=device, dtype=f32),
                          book.to(device=device, dtype=f32), group, rot)

    def block(base: str, adaln: bool) -> BlockWeights:
        return BlockWeights(
            qkv=linear(base + ".attn.qkv_proj", 3 * inner, hid), out=linear(base + ".attn.out_proj", hid, inner),
            fc1=linear(base + ".mlp.fc1", 2 * cfg.ffn, hid), fc2=linear(base + ".mlp.fc2", hid, cfg.ffn),
            norm1=cast(base + ".norm1.weight", (hid,), dtype), norm2=cast(base + ".norm2.weight", (hid,), dtype),
            q_norm=cast(base + ".attn.q_norm.weight", (hd,), dtype), k_norm=cast(base + ".attn.k_norm.weight", (hd,), dtype),
            adaln_w=cast(base + ".adaln_proj.linear.weight", (18 * hid, cfg.t_dim), None) if adaln else None,
            adaln_b=cast(base + ".adaln_proj.linear.bias", (18 * hid,), f32) if adaln else None)

    blocks = [block(f"blocks.{i}", True) for i in range(cfg.layers)]
    refiner = [block(f"token_refiner.blocks.{j}", False) for j in range(cfg.refiner_layers)]
    pd, ac = cfg.video_patch_dim, cfg.audio_channels
    glob = GlobalWeights(
        video_patch_w=cast("video_patch_proj.weight", (hid, pd), f32), video_patch_b=cast("video_patch_proj.bias", (hid,), f32),
        audio_patch_w=cast("audio_patch_proj.weight", (hid, ac), f32), audio_patch_b=cast("audio_patch_proj.bias", (hid,), f32),
        condition_w=cast("condition_proj.weight", (hid, cfg.text_dim), dtype), condition_b=cast("condition_proj.bias", (hid,), dtype),
        adaln_t_table=cast("adaln_t_table", (cfg.curve_grid, cfg.t_dim), f32),
        rope_inv_freq=cast("rope.inv_freq", (cfg.rope_inv_freq_len,), f32),
        final_norm=cast("final_layer.norm.weight", (hid,), dtype),
        final_adaln_w=cast("final_layer.adaln_proj.linear.weight", (2 * hid, cfg.t_dim), f32),
        final_adaln_b=cast("final_layer.adaln_proj.linear.bias", (2 * hid,), f32),
        video_out_w=cast("final_layer.video_out.weight", None, f32), video_out_b=cast("final_layer.video_out.bias", None, f32),
        audio_out_w=cast("final_layer.audio_out.weight", None, f32), audio_out_b=cast("final_layer.audio_out.bias", None, f32),
        refiner=refiner, refiner_final_norm=cast("token_refiner.final_norm.weight", (hid,), dtype))
    if missing:
        raise KeyError(f"state dict is missing {len(missing)} tensors, e.g. {missing[:6]}")
    return glob, blocks
