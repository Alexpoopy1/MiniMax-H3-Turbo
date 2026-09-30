"""`h3turbo h3-bench`: time real DiT forwards of an .h3t file on this machine (random latents/text, so no encoders needed)."""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional, Sequence

import torch

from .engine import H3Engine


def latent_shape(width: int, height: int, seconds: float, fps: int = 24) -> tuple:
    """(T, H, W, Ta): latent frames (4x temporal, first frame alone), rows/cols (16x), audio latent frames (40/s)."""
    if width % 32 or height % 32:
        raise ValueError("width and height must be multiples of 32 (16x VAE x 2x patch)")
    frames = max(1, round(seconds * fps))
    return (frames - 1) // 4 + 1, height // 16, width // 16, max(1, round(seconds * 40))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="h3-bench", description="Time DiT forwards of an .h3t on this GPU.")
    ap.add_argument("h3t")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--seconds", type=float, default=1.4)
    ap.add_argument("--text-len", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3, help="timed forwards after one warm-up")
    ap.add_argument("--precision", default="a8", choices=["a8", "a16"])
    ap.add_argument("--resident", default="auto", help="blocks kept on the GPU ('auto' or a count)")
    ap.add_argument("--reserve-gb", type=float, default=1.5, help="VRAM kept free for activations when resident='auto'")
    ap.add_argument("--mlp-chunk", type=int, default=None)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "int8"], help="int8 = comfy_kitchen INT8 attention (faster, approximate)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    t, h, w, ta = latent_shape(a.width, a.height, a.seconds)
    g = torch.Generator("cpu").manual_seed(0)
    with H3Engine.from_h3t(a.h3t, a.device, resident=a.resident if a.resident == "auto" else int(a.resident), precision=a.precision,
                           reserve_gb=a.reserve_gb, mlp_chunk=a.mlp_chunk, attn_impl=a.attn) as eng:
        dev, cfg = eng.model.device, eng.cfg
        xv = torch.randn(1, cfg.video_channels, t, h, w, generator=g).bfloat16().to(dev)
        xa = torch.randn(1, cfg.audio_channels, 2, ta, generator=g).bfloat16().to(dev)
        refined = eng.encode_text(torch.randn(1, a.text_len, cfg.text_dim, generator=g).bfloat16().to(dev))
        tokens = a.text_len + 2 * ta + t * (h // 2) * (w // 2)
        times = []
        for i in range(a.reps + 1):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            eng.velocity([xv, xa], 0.6, refined_text=refined)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            times.append(time.perf_counter() - t0)
        steady = times[1:]
        print(json.dumps({
            "tokens": tokens, "latent_thwTa": [t, h, w, ta], "first_forward_s": round(times[0], 2),
            "steady_forward_s": [round(x, 2) for x in steady], "mean_steady_s": round(sum(steady) / len(steady), 2),
            "peak_vram_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == "cuda" else None,
            "precision": a.precision, "linear_backend": eng.linear_backend(), "stream": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in eng.stats().items()},
        }, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
