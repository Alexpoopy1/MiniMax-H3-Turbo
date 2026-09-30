"""Smoke test of the ComfyUI H3-Turbo Fast UNET Loader without starting a server (run under the ComfyUI venv).

Builds the MODEL patcher for an .h3t, loads it through ComfyUI's own model management, and checks:
  * outputs equal a saved native-ComfyUI reference (from `scripts/h3_real_check.py native --out ref.pt`, same inputs) exactly,
  * partially_unload frees the GPU weights and a second load reproduces the same outputs,
  * run-time LoRA patches are refused, and a clone shares the loaded engine,
  * unload_all_models leaves no engine memory behind.

  python scripts/comfy_h3_node_smoke.py --h3t model.h3t --ref ref_1768.pt [--comfy D:\\ComfyUIBig\\ComfyUI\\ComfyUI]

Time nothing while another GPU job is running: the copy stream and the GEMMs share the card.
"""
import argparse
import logging
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMAS = (0.95, 0.6, 0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3t", required=True)
    ap.add_argument("--ref", required=True, help="reference outputs saved by h3_real_check.py native (shape 9,20,32,100, sigmas 0.95,0.6,0.2)")
    ap.add_argument("--comfy", default=r"D:\ComfyUIBig\ComfyUI\ComfyUI")
    a = ap.parse_args()
    sys.path[:0] = [a.comfy, os.path.dirname(HERE), HERE]
    sys.argv = [sys.argv[0]]
    logging.basicConfig(level=logging.INFO)
    import comfy.model_management as mm
    import comfy_h3_nodes as nodes
    from h3_real_check import make_inputs

    ref = torch.load(a.ref)
    patcher = nodes.build_model_patcher(a.h3t)
    dev = mm.get_torch_device()
    dm = patcher.model.diffusion_model
    xv, xa, ctx = make_inputs(tuple(ref["shape"]))
    failures = []

    def check(tag):
        for s in SIGMAS:
            ts = torch.tensor([s * 1000.0], device=dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = dm([xv.to(dev), xa.to(dev)], ts, ctx.to(dev), transformer_options={})
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rel = [((o.float().cpu() - r).norm() / r.norm()).item() for o, r in zip(out, ref["out"][s])]
            print(f"[{tag}] sigma={s}: {dt:.2f}s rel-L2 vs native {rel}", flush=True)
            if any(r != 0.0 for r in rel):
                failures.append(f"{tag} sigma={s} differs from native: {rel}")

    mm.load_models_gpu([patcher], memory_required=1 << 30)
    print(f"loaded {patcher.loaded_size() / 2**30:.2f} GiB of {patcher.model_size() / 2**30:.2f}", flush=True)
    check("first load")
    freed = patcher.partially_unload(patcher.offload_device, 1 << 30)
    print(f"partially_unload freed {freed / 2**30:.2f} GiB; torch allocated {torch.cuda.memory_allocated() / 2**30:.3f} GiB", flush=True)
    if patcher.loaded_size() != 0 or torch.cuda.memory_allocated() > (256 << 20):
        failures.append("unload left engine memory on the GPU")
    mm.load_models_gpu([patcher], memory_required=1 << 30)
    check("reload")
    try:
        patcher.add_patches({"diffusion_model.blocks.0.attn.qkv_proj.weight": ("lora", None)})
        failures.append("a LoRA patch was accepted")
    except RuntimeError as e:
        print("LoRA refused:", str(e)[:70], flush=True)
    clone = patcher.clone()
    if clone.model is not patcher.model or not clone.is_clone(patcher):
        failures.append("clone does not share the model")
    mm.load_models_gpu([clone], memory_required=1 << 30)
    check("clone")
    mm.unload_all_models()
    if patcher.loaded_size() != 0 or torch.cuda.memory_allocated() > (256 << 20):
        failures.append("unload_all_models left engine memory behind")
    print("RESULT:", "PASS" if not failures else "FAIL: " + "; ".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
