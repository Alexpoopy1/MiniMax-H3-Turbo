"""Sizing and benchmarking.

`estimate()` is analytic and needs no GPU or weights: parameter counts, weight memory
per precision, an activation-memory estimate, and FLOPs for a request. It answers
"does tier X fit on card Y" and "how much work is one generation".

`measure()` actually runs a loaded pipeline and reports wall time and, on CUDA, peak
memory. Numbers from `estimate()` are a plan, not a measurement: run `measure()` (or
`h3turbo bench --ckpt ... --measure`) on your card for real timings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.flop_counter import FlopCounterMode

from .config import H3TurboConfig, VIDEO_SPATIAL
from .layout import Geometry, snap_frames, snap_size
from .model import H3TurboTransformer
from .video_vae import VideoVAE

# VRAM in GB. (Memory sizes only: throughput varies too much by SKU/clocks/drivers to hardcode.)
GPUS = {
    "rtx3050-4gb": 4, "rtx3050-6gb": 6, "rtx3050-8gb": 8, "rtx3060-12gb": 12, "rtx3060ti-8gb": 8,
    "rtx3070-8gb": 8, "rtx3080-10gb": 10, "rtx3090-24gb": 24, "rtx4060-8gb": 8, "rtx4070-12gb": 12,
    "rtx4080-16gb": 16, "rtx4090-24gb": 24,
}
GB = 1024**3


@dataclass
class Estimate:
    tier: str
    params: Dict[str, float]
    weights_gb: Dict[str, float]  # GPU-resident weights per precision
    activations_gb: float
    tokens: Dict[str, int]
    forward_tflop: float
    generation_tflop: float
    vae_decode_tflop: float

    def fits(self, vram_gb: float, precision: str, headroom_gb: float = 0.8) -> bool:
        return self.weights_gb[precision] + self.activations_gb + headroom_gb <= vram_gb


def _meta_params(cfg: H3TurboConfig):
    with torch.device("meta"):
        tr = H3TurboTransformer(cfg.transformer)
    blocks = sum(p.numel() for p in tr.blocks.parameters())
    adaln = sum(p.numel() for p in tr.adaln.parameters())
    io = sum(p.numel() for n, p in tr.named_parameters() if n.startswith(("in_proj", "out_proj")))
    extra = tr.num_params() - blocks - adaln - io
    return tr, blocks, adaln, io, extra


def _text_params(cfg: H3TurboConfig) -> int:
    from .text import build_text_encoder

    with torch.device("meta"):
        te = build_text_encoder(cfg.text)
    return sum(p.numel() for p in te.parameters())


def _vae_decode_flops(cfg: H3TurboConfig, geom: Geometry) -> float:
    with torch.device("meta"):
        vae = VideoVAE(cfg.video_vae)
        z = torch.empty(1, cfg.video_vae.latent_ch, geom.lat_t, geom.height // VIDEO_SPATIAL, geom.width // VIDEO_SPATIAL)
        with FlopCounterMode(display=False) as fc:
            vae.decode_raw(z)
    return float(fc.get_total_flops())


def forward_flops(cfg: H3TurboConfig, n_tokens: int) -> float:
    """Matmul FLOPs of one transformer forward: block GEMMs + attention (QK^T and AV)."""
    t = cfg.transformer
    H = t.hidden
    per_layer_params = 4 * H * H + 3 * H * t.ffn_dim
    gemm = 2 * per_layer_params * n_tokens * t.layers
    attn = 4 * n_tokens * n_tokens * H * t.layers
    return float(gemm + attn)


def estimate(
    cfg: H3TurboConfig,
    width: int = 512,
    height: int = 320,
    seconds: float = 4.0,
    steps: int = 4,
    text_tokens: int = 64,
    context_tokens: int = 0,
    with_audio: bool = True,
    vae_dtype_bytes: int = 2,
) -> Estimate:
    fps = cfg.fps
    geom = Geometry(snap_size(width), snap_size(height), snap_frames(round(seconds * fps)), fps)
    tr, blocks, adaln, io, extra = _meta_params(cfg)
    te = _text_params(cfg)
    vae_v = _vae_params(cfg)
    n_tok = geom.video_tokens + (geom.audio_tokens if with_audio else 0) + text_tokens + context_tokens
    H, F_ = cfg.transformer.hidden, cfg.transformer.ffn_dim

    # AdaLN modulation is cached per schedule and its bank stays off the GPU at inference.
    fixed = (io + extra + te + vae_v) * 2  # fp16 bytes
    weights = {
        "fp16": (fixed + blocks * 2) / GB,
        "int8": (fixed + blocks * 1.02) / GB,
        "int4": (fixed + blocks * 0.53) / GB,
    }
    # peak per-layer activations without grad: x, qkv, attn out, gated ffn hidden, gathered modulation
    act = n_tok * (14 * H + 2 * F_) * 2 / GB
    fwd = forward_flops(cfg, n_tok)
    vae = _vae_decode_flops(cfg, geom)
    return Estimate(
        tier=cfg.name,
        params={
            "transformer_total_M": tr.num_params() / 1e6,
            "on_gpu_M (no adaln)": (blocks + io + extra) / 1e6,
            "adaln_M (kept on CPU)": adaln / 1e6,
            "text_encoder_M": te / 1e6,
            "video_vae_M": vae_v / 1e6,
        },
        weights_gb=weights,
        activations_gb=act,
        tokens={"video": geom.video_tokens, "audio": geom.audio_tokens if with_audio else 0, "text": text_tokens, "total": n_tok},
        forward_tflop=fwd / 1e12,
        generation_tflop=(fwd * steps + vae) / 1e12,
        vae_decode_tflop=vae / 1e12,
    )


def _vae_params(cfg: H3TurboConfig) -> int:
    with torch.device("meta"):
        return sum(p.numel() for p in VideoVAE(cfg.video_vae).parameters())


def format_report(e: Estimate, steps: int, tflops: Optional[float] = None) -> str:
    lines = [f"== {e.tier} ==", "parameters:"]
    lines += [f"  {k:24s} {v:10.1f}" for k, v in e.params.items()]
    lines.append(f"tokens: {e.tokens}")
    lines.append("GPU weights (GB):  " + "  ".join(f"{k}={v:.2f}" for k, v in e.weights_gb.items()) + f"   activations~{e.activations_gb:.2f}")
    lines.append(f"work: {e.forward_tflop:.2f} TFLOP/forward, {e.generation_tflop:.2f} TFLOP per generation ({steps} steps + VAE decode {e.vae_decode_tflop:.2f})")
    if tflops:
        lines.append(f"if sustained {tflops:g} TFLOPS (your assumption, unmeasured): ~{e.generation_tflop / tflops:.1f} s per generation")
    fit = []
    for name, gb in GPUS.items():
        best = next((p for p in ("fp16", "int8", "int4") if e.fits(gb, p)), None)
        fit.append(f"{name}:{best or 'needs offload'}")
    lines.append("fits (lowest precision needed): " + "  ".join(fit))
    return "\n".join(lines)


@torch.no_grad()
def measure(pipe, prompt: str = "a red square moving left", runs: int = 3, **kwargs) -> dict:
    """Wall time per generation (median of `runs` after one warmup) and peak CUDA memory."""
    dev = pipe.device
    is_cuda = dev.type == "cuda"
    pipe(prompt, seed=0, **kwargs)  # warmup / kernel selection
    if is_cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    times = []
    for i in range(runs):
        t = time.perf_counter()
        pipe(prompt, seed=i, **kwargs)
        if is_cuda:
            torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - t)
    out = {"device": str(dev), "median_s": sorted(times)[len(times) // 2], "runs_s": times}
    if is_cuda:
        out["peak_gb"] = torch.cuda.max_memory_allocated(dev) / GB
    return out
