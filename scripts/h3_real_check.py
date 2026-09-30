"""Real-checkpoint parity + timing: native ComfyUI H3 vs the H3-Turbo engine, same inputs, on the same GPU.

Run under the ComfyUI venv (torch cu130 + comfy_kitchen), one mode per process so the two 12 GB models never coexist in RAM:

  python scripts/h3_real_check.py native --src <comfy .safetensors> --out ref.pt
  python scripts/h3_real_check.py engine --h3t <file.h3t> --ref ref.pt [--precision a16] [--shape 9,20,32,100]

`--shape T,H,W,Ta` is latent frames, latent rows, latent cols (x16 px), audio latent frames; 9,20,32,100 = 512x320, ~33 frames.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

SIGMAS = (0.95, 0.6, 0.2)  # overridden by --sigmas


def make_inputs(shape, seed=0, text_len=128, text_dim=5120):
    t, h, w, ta = shape
    g = torch.Generator("cpu").manual_seed(seed)
    xv = torch.randn(1, 24, t, h, w, generator=g).bfloat16()
    xa = torch.randn(1, 32, 2, ta, generator=g).bfloat16()
    ctx = torch.randn(1, text_len, text_dim, generator=g).bfloat16()
    return xv, xa, ctx


def native(a):
    sys.path.insert(0, r"D:\ComfyUIBig\ComfyUI\ComfyUI")
    sys.argv = [sys.argv[0]]
    import comfy.model_management as mm
    import comfy.sd

    patcher = comfy.sd.load_diffusion_model(a.src)
    dm = patcher.model.diffusion_model
    mm.load_models_gpu([patcher])
    dev = mm.get_torch_device()
    xv, xa, ctx = make_inputs(a.shape)
    out, times = {}, []
    for s in SIGMAS:
        for rep in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                o = dm([xv.to(dev), xa.to(dev)], torch.tensor([s * 1000.0], device=dev), ctx.to(dev), transformer_options={})
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        times.append(dt)
        out[s] = [t.float().cpu() for t in o]
        print(f"native sigma={s}: {dt:.2f}s (steady)", flush=True)
    torch.save({"out": out, "shape": a.shape, "times": times}, a.out)
    print(f"native mean steady forward {sum(times) / len(times):.2f}s; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


def engine(a):
    from h3turbo.h3.engine import H3Engine

    ref = torch.load(a.ref) if a.ref else None
    xv, xa, ctx = make_inputs(a.shape)
    with H3Engine.from_h3t(a.h3t, precision=a.precision, resident=a.resident if a.resident == "auto" else int(a.resident),
                           reserve_gb=a.reserve_gb, mlp_chunk=a.mlp_chunk, attn_impl=a.attn,
                           backend=a.backend, dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]) as eng:
        dev = eng.model.device
        md = eng.model.dtype
        refined = eng.encode_text(ctx.to(dev, md))
        times = []
        for s in SIGMAS:
            for rep in range(a.reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                o = eng.velocity([xv.to(dev, md), xa.to(dev, md)], (torch.tensor([s * 1000.0], device=dev) / 1000.0).float()[0].cpu(), refined_text=refined)  # sigma exactly as native computes it on the GPU
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            times.append(dt)
            msg = f"engine[{a.precision}] sigma={s}: {dt:.2f}s (last of {a.reps})"
            if ref is not None:
                for name, r, e in zip(("video", "audio"), ref["out"][s], o):
                    e = e.float().cpu()
                    rel = ((e - r).norm() / r.norm()).item()
                    cos = torch.nn.functional.cosine_similarity(e.flatten().double(), r.flatten().double(), dim=0).item()
                    msg += f" | {name}: relL2={rel:.2e} cos={cos:.6f} maxabs={(e - r).abs().max().item():.3g}"
            print(msg, flush=True)
        print(f"engine mean forward {sum(times) / len(times):.2f}s; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB; stats {eng.stats()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["native", "engine"])
    ap.add_argument("--src")
    ap.add_argument("--out", default="ref.pt")
    ap.add_argument("--h3t")
    ap.add_argument("--ref")
    ap.add_argument("--precision", default="a8")
    ap.add_argument("--resident", default="auto")
    ap.add_argument("--reserve-gb", type=float, default=1.5)
    ap.add_argument("--mlp-chunk", type=int, default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "ck", "torch"], help="torch = portable int8 path (what runs without comfy_kitchen, e.g. on Colab)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="fp32 = what the ComfyUI node picks on pre-Ampere GPUs (T4)")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "int8"], help="int8 = comfy_kitchen INT8 attention (faster, approximate)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--shape", type=lambda s: tuple(int(v) for v in s.split(",")), default=(9, 20, 32, 100))
    ap.add_argument("--sigmas", type=lambda s: tuple(float(v) for v in s.split(",")), default=SIGMAS)
    args = ap.parse_args()
    SIGMAS = args.sigmas
    (native if args.mode == "native" else engine)(args)
