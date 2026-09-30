"""Block 0 of the REAL 4-bit checkpoint: ComfyUI's quantized block vs h3turbo's H3Model block through qlinear.

Reads only block-0 and the small global tensors with safe_open (~0.3 GB), builds ComfyUI's model with 1 block from that
partial state dict, and compares one block forward at a realistic token count on random hidden states. Reports
rel-L2/max-abs of h3turbo (backend ck / torch, precision a8 / a16) against ComfyUI's own W4A8 block, plus time and
peak VRAM. Companion of h3_check_model_vs_comfy.py (which covers the whole network at tiny scale with dense weights).

  set PYTHONPATH=C:/Users/Alexp/MiniMax-H3-Turbo
  D:/ComfyUIBig/ComfyUI/ComfyUI/.venv/Scripts/python.exe scripts/h3_check_block0_real.py [--tokens 1768]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

COMFY = r"D:\ComfyUIBig\ComfyUI\ComfyUI"
CKPT = COMFY + r"\models\diffusion_models\minimax_h3_fused_refdelta_r1024_turbo8_mystic07_w4a8_convrot.safetensors"
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import comfy.model_management as mm  # noqa: E402
import comfy.sd  # noqa: E402
sys.argv = _argv
from h3turbo.h3.config import H3Config  # noqa: E402
from h3turbo.h3.layout import PackedLayout, curve_lerp, plan_time, rope_angles, rope_ck_table  # noqa: E402
from h3turbo.h3.model import H3Model, ResidentProvider, weights_from_state_dict  # noqa: E402


def err(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return f"rel_l2={((a - b).norm() / b.norm()).item():.3e} max_abs={(a - b).abs().max().item():.3e}"


def timed(fn, n=5):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def refiner_check(args) -> int:
    """condition_proj + token refiner (1 real dense block, ~0.8 GB) : H3Model.encode_text with the refiner left on the CPU
    (moved to the GPU block by block) vs ComfyUI's preprocess_text_embeds."""
    keep = ("blocks.0.", "video_patch_proj.", "audio_patch_proj.", "condition_proj.", "adaln_t_table", "rope.inv_freq", "final_layer.",
            "token_refiner.final_norm", "token_refiner.blocks.0.")
    with safe_open(CKPT, "pt") as f:
        sd = {k: f.get_tensor(k) for k in f.keys() if k.startswith(keep)}
    cfg = H3Config.from_shapes({k: tuple(v.shape) for k, v in sd.items()})
    print(f"refiner check: {len(sd)} tensors, {sum(t.numel() * t.element_size() for t in sd.values()) / 2**30:.2f} GiB, refiner layers {cfg.refiner_layers}")
    text = torch.randn(1, args.text_len, cfg.text_dim, generator=torch.Generator().manual_seed(1)).to("cuda", torch.bfloat16)
    patcher = comfy.sd.load_diffusion_model_state_dict(dict(sd))
    mm.load_models_gpu([patcher])
    with torch.inference_mode():
        want = patcher.model.diffusion_model.preprocess_text_embeds(text).cpu()
    del patcher
    mm.unload_all_models()
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    glob, blocks = weights_from_state_dict(sd, cfg, "cpu", torch.bfloat16)  # everything on the CPU; H3Model moves what it needs
    m = H3Model(cfg, glob, ResidentProvider(blocks), device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    got = m.encode_text(text).cpu()
    print(f"encode_text vs preprocess_text_embeds: {err(got, want)} | peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB | bit-identical: {torch.equal(got, want)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1768, help="1768 = 128 text + 200 audio + 9x10x16 video rows")
    ap.add_argument("--refiner", action="store_true", help="check condition_proj + token refiner instead (loads ~0.8 GB more)")
    ap.add_argument("--text-len", type=int, default=128)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs CUDA")
        return 2
    if args.refiner:
        return refiner_check(args)
    keep = ("blocks.0.", "video_patch_proj.", "audio_patch_proj.", "condition_proj.", "adaln_t_table", "rope.inv_freq", "final_layer.",
            "token_refiner.final_norm")
    with safe_open(CKPT, "pt") as f:
        sd = {k: f.get_tensor(k) for k in f.keys() if k.startswith(keep)}
    print(f"read {len(sd)} tensors, {sum(t.numel() * t.element_size() for t in sd.values()) / 2**20:.0f} MiB")
    cfg = H3Config.from_shapes({k: tuple(v.shape) for k, v in sd.items()})
    print(f"config: hidden {cfg.hidden}, heads {cfg.heads}x{cfg.head_dim}, ffn {cfg.ffn}, layers {cfg.layers}, rope {cfg.rope_inv_freq_len}")

    patcher = comfy.sd.load_diffusion_model_state_dict(dict(sd))
    mm.load_models_gpu([patcher])
    native = patcher.model.diffusion_model.blocks[0]
    dev, dt = torch.device("cuda"), torch.bfloat16
    print("native fc1 weight:", type(native.mlp.fc1.weight).__name__, "| vram after native load", f"{torch.cuda.memory_allocated() / 2**20:.0f} MiB")

    glob, blocks = weights_from_state_dict(sd, cfg, dev, dt)
    # a layout of ~args.tokens rows: 128 text, 100 audio frames, video frames of 10x16 patch rows
    text, ta = 128, 100
    per_frame = 10 * 16
    frames = max(1, (args.tokens - text - 2 * ta) // per_frame)
    layout = PackedLayout.build(text, frames, 20, 32, ta)
    plan = plan_time(layout, torch.tensor(0.8, dtype=torch.float32), (cfg.sigma_shift_video, cfg.sigma_shift_audio))
    g = torch.Generator().manual_seed(0)
    x0 = (torch.randn(layout.seq_len, cfg.hidden, generator=g) * 1.5).to(dev, dt)
    t_emb = curve_lerp(glob.adaln_t_table.to(dev), plan.t_values)
    segs = [(a, b, r.to(dev) if isinstance(r, torch.Tensor) else r) for a, b, r in plan.mod_segments]
    ang = rope_angles(layout.position_ids, glob.rope_inv_freq, dev)
    cos, sin = torch.cos(ang).to(dt), torch.sin(ang).to(dt)
    table = rope_ck_table(cos, sin)
    print(f"tokens {layout.seq_len}, distinct timesteps {len(plan.t_values)}")

    with torch.inference_mode():
        def run_native():
            return native(x0.clone(), t_emb, segs, table, transformer_options={})

        want = run_native()
        results = {}
        for backend, precision in (("ck", "a8"), ("torch", "a8"), ("torch", "a16")):
            m = H3Model(cfg, glob, ResidentProvider(blocks), device=dev, dtype=dt, backend=backend, precision=precision)
            torch.cuda.reset_peak_memory_stats()
            out = m._block(x0.clone(), blocks[0], t_emb, segs, (cos, sin, table))
            results[(backend, precision)] = out
            ms = timed(lambda: m._block(x0.clone(), blocks[0], t_emb, segs, (cos, sin, table)))
            print(f"h3turbo backend={backend:5s} precision={precision}: vs ComfyUI block {err(out, want)} | {ms:.1f} ms/block | peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
        print(f"ComfyUI native block: {timed(run_native):.1f} ms/block")
        d8 = results[("ck", "a8")].float() - want.float()
        print(f"native output rms {want.float().pow(2).mean().sqrt().item():.3f}; ck-a8 vs native differing elements: {(d8 != 0).sum().item()} of {d8.numel()}")
        print(f"a8 (ck) vs a16 (torch): {err(results[('ck', 'a8')], results[('torch', 'a16')])}  <- activation-quantisation noise of the W4A8 path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
