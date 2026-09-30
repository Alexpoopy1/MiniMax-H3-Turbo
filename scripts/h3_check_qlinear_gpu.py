"""GPU check of h3turbo.h3.qlinear on the REAL block-0 tensors of the W4A8 checkpoint.

Run with the ComfyUI venv (has CUDA torch and comfy_kitchen):
    set PYTHONPATH=C:\\Users\\Alexp\\MiniMax-H3-Turbo
    D:\\ComfyUIBig\\ComfyUI\\ComfyUI\\.venv\\Scripts\\python.exe scripts\\h3_check_qlinear_gpu.py

Reads one layer at a time with safetensors.safe_open (never the whole file), caps this process at 1.5 GB of
VRAM, and prints: decode exactness, ck-vs-torch agreement, error against an fp64 reference of the dequantised
weight (rel-L2, max abs), how much of the A8 error is activation quantisation, the weight-side error against the
int8 sibling checkpoint (block-0 tensors only), chunk invariance and ms per layer for M in {512, 1768, 4226}.
Activations are synthetic (block-0 inputs need the whole network): gaussian rows and a heavy-tailed variant.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from h3turbo.h3 import qlinear as ql  # noqa: E402
from h3turbo.h3.types import W4A8Weight  # noqa: E402

CKPT_DIR = r"D:\ComfyUIBig\ComfyUI\ComfyUI\models\diffusion_models"
W4A8 = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_w4a8_convrot.safetensors"
INT8 = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors"
LAYERS = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]
BF = torch.bfloat16


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def maxrel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


def load_layer(f, name, dev) -> W4A8Weight:
    p = f"blocks.0.{name}"
    return W4A8Weight(f.get_tensor(p + ".weight").to(dev), f.get_tensor(p + ".weight_s_rel").to(dev),
                      f.get_tensor(p + ".weight_s_channel").to(dev), f.get_tensor(p + ".weight_codebook").to(dev))


def make_input(kind, m, k, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(m, k, device=dev, generator=g)
    if kind == "outlier":  # a few massive channels and a few loud tokens, as in transformer residual streams
        ch = torch.randperm(k, device=dev, generator=g)[:8]
        x[:, ch] *= 30.0
        x[torch.randperm(m, device=dev, generator=g)[: max(1, m // 32)]] *= 5.0
    return x.to(BF)


def swiglu(h):
    gate, up = h.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


def timed(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def slab_reference(x_act, w, i8=None, i8_scale=None, slab=1024):
    """fp64 references, slab by slab over output rows: (y_ref with the dequantised W4 weight in the original
    basis, y_rot_act with the QUANTISED activation against the same weight in the rotated basis, y_w8 with the int8 sibling)."""
    n = w.out_features
    xd = x_act.double()
    xq, xs = ql.quantize_activation(x_act, quant_mode="ck")
    xdq = (xq.float() * xs).double()
    y_ref = torch.empty(x_act.shape[0], n, dtype=torch.float64, device=x_act.device)
    y_act = torch.empty_like(y_ref)
    y_w8 = torch.empty_like(y_ref) if i8 is not None else None
    h = ql.hadamard_regular(256, x_act.device, torch.float32)
    for a in range(0, n, slab):
        b = min(a + slab, n)
        y_ref[:, a:b] = xd @ ql.dequantize(w, torch.float32, rotated=False, rows=slice(a, b)).double().T
        y_act[:, a:b] = xdq @ ql.dequantize(w, torch.float32, rotated=True, rows=slice(a, b)).double().T
        if i8 is not None:
            rot = i8[a:b].to(w.q.device).float() * i8_scale[a:b].to(w.q.device).float()
            orig = torch.matmul(rot.view(b - a, -1, 256), h).view(b - a, -1)
            y_w8[:, a:b] = xd @ orig.double().T
    return y_ref, y_act, y_w8


def weight_error(w, i8, i8s, slab=1024):
    """rel-L2 between the W4 weight and the int8 sibling weight, both in the (shared) ConvRot-rotated basis, slab by slab."""
    num = den = 0.0
    for a in range(0, w.out_features, slab):
        b = min(a + slab, w.out_features)
        w4 = ql.dequantize(w, torch.float32, rotated=True, rows=slice(a, b)).double()
        w8 = (i8[a:b].to(w.q.device).float() * i8s[a:b].to(w.q.device).float()).double()
        num += (w4 - w8).pow(2).sum().item()
        den += w8.pow(2).sum().item()
    return (num / den) ** 0.5


def decode_section(f, dev, ck, rows):
    print("\n== 1. decode of the int4 codebook / fp8-scale format onto the int8 grid (bit-exactness) ==")
    for name in LAYERS:
        w = load_layer(f, name, dev)
        mine = ql.decode_int8_grid(w)
        res = {"layer": name, "N": w.out_features, "K": w.in_features, "ms": round(timed(lambda: ql.decode_int8_grid(w)), 2)}
        s_bytes = w.s_rel.view(torch.uint8)
        res["nan_scale_bytes"] = int(((s_bytes & 0x7F) == 0x7F).sum())
        res["codebook_sorted"] = bool((w.codebook[1:] > w.codebook[:-1]).all())
        if ck:
            from comfy_kitchen.backends import cuda as ckc
            from comfy_kitchen.backends.eager.w4a8_int8 import _dequant_int4_grouped_to_int8

            # kitchen's eager decode builds several fp32/int32 [N, K] temporaries: compare in row slabs to stay in the VRAM budget
            res["equal_kitchen_eager"] = all(torch.equal(mine[a : a + 1024], _dequant_int4_grouped_to_int8(w.q[a : a + 1024], w.s_rel[a : a + 1024], w.codebook, 16))
                                             for a in range(0, w.out_features, 1024))
            try:
                out = torch.empty_like(mine)
                ckc._C.dequant_int4_grouped_to_int8_e4m3(ckc._wrap_for_dlpack(w.q), ckc._wrap_for_dlpack(s_bytes), ckc._wrap_for_dlpack(w.codebook),
                                                         ckc._wrap_for_dlpack(out), 16, torch.cuda.current_stream().cuda_stream)
                res["equal_kitchen_cuda_kernel"] = bool(torch.equal(mine, out))
            except Exception as e:  # internal API
                res["equal_kitchen_cuda_kernel"] = f"n/a ({type(e).__name__})"
        print(" ", json.dumps(res))
        rows.append(("decode", res))
        del w, mine


def rotation_section(dev, ck):
    print("\n== 2. activation rotation vs kitchen's _rotate_activation (bf16 GEMM) ==")
    if not ck:
        print("  comfy_kitchen not importable: skipped")
        return
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    for k in (5376, 7168, 14336):
        x = torch.randn(1000, k, device=dev, dtype=BF)
        theirs = _rotate_activation(x, _build_hadamard(256, device=dev, dtype=BF), 256)
        mine = ql.rotate(x)
        print(f"  K={k}: differing elements {(mine != theirs).sum().item()} of {mine.numel()}  rel-L2 {rel(mine, theirs):.2e}  (H equal: {torch.equal(ql.hadamard_regular(256, dev, BF), _build_hadamard(256, device=dev, dtype=BF))})")


def accuracy_section(f, f8, dev, ck, m, rows):
    print(f"\n== 3. accuracy at M={m}: rel-L2 against an fp64 reference (dequantised W4 weight, unquantised activation) ==")
    if ck:
        import comfy_kitchen as ckm
        from comfy_kitchen.tensor.w4a8_int8 import w4a8_int8_linear as kit
    for name in LAYERS:
        w = load_layer(f, name, dev)
        n, k = w.out_features, w.in_features
        swi = name == "mlp.fc2"
        i8 = i8s = None
        if f8 is not None:
            p = f"blocks.0.{name}"
            i8, i8s = f8.get_tensor(p + ".weight"), f8.get_tensor(p + ".weight_scale")
            w_err = weight_error(w, i8, i8s)
        for kind in ("gauss", "outlier"):
            x = make_input(kind, m, 2 * k if swi else k, dev, seed=zlib.crc32(f"{name}/{kind}".encode()) % 1000)
            xa = swiglu(x) if swi else x
            kw = {"input_act": "swiglu"} if swi else {}
            y_ref, y_act, y_w8 = slab_reference(xa, w, i8, i8s)
            out = {}
            out["torch_ck"] = ql.linear(x, w, backend="torch", quant_mode="ck", **kw)
            out["torch_eager"] = ql.linear(x, w, backend="torch", quant_mode="eager", **kw)
            out["a16"] = ql.linear(x, w, backend="torch", precision="a16", **kw)
            if ck:
                out["ck"] = ql.linear(x, w, backend="ck", **kw)
                with ckm.use_backend("eager"):  # per-output-row independent, so slab the weight to fit the VRAM budget
                    out["kitchen_eager"] = torch.cat([kit(xa, w.q[a : a + 2048], w.s_rel[a : a + 2048], w.s_ch[a : a + 2048], codebook=w.codebook, out_dtype=BF)
                                                      for a in range(0, n, 2048)], dim=1)
            main = out.get("ck", out["torch_ck"])
            e8, e16, eact = rel(main, y_ref), rel(out["a16"], y_ref), rel(y_act, y_ref)
            r = {"layer": name, "input": kind, "M": m,
                 "rel_ck": round(e8, 5) if ck else None, "rel_torch_ck": round(rel(out["torch_ck"], y_ref), 5),
                 "rel_torch_eager": round(rel(out["torch_eager"], y_ref), 5), "rel_a16": round(e16, 5),
                 "rel_act_only_fp32": round(eact, 5), "max_abs_over_max_ref_a8": round(maxrel(main, y_ref), 5), "max_abs_over_max_ref_a16": round(maxrel(out["a16"], y_ref), 5),
                 "act_quant_share_of_a8_error": round(min(1.0, eact**2 / e8**2), 3), "act_quant_share_via_a16": round(max(0.0, 1 - (e16 / e8) ** 2), 3)}
            if ck:
                r["ck_vs_torch_ck_equal_frac"] = round((out["ck"] == out["torch_ck"]).float().mean().item(), 7)
                r["ck_vs_torch_ck_rel"] = float(f"{rel(out['torch_ck'], out['ck']):.2e}")
                r["torch_eager_bit_equal_kitchen_eager"] = bool(torch.equal(out["torch_eager"], out["kitchen_eager"]))
                r["rel_kitchen_eager"] = round(rel(out["kitchen_eager"], y_ref), 5)
            if y_w8 is not None:
                r["weight_rel_W4_vs_W8_rotated"] = round(w_err, 4)
                r["out_rel_ref_vs_W8"] = round(rel(y_ref, y_w8), 5)
                r["out_rel_a8_vs_W8"] = round(rel(main, y_w8), 5)
                r["out_rel_a16_vs_W8"] = round(rel(out["a16"], y_w8), 5)
            print(" ", json.dumps(r))
            rows.append(("accuracy", r))
            del x, xa, out, y_ref, y_act, y_w8
            torch.cuda.empty_cache()
        del w, i8, i8s


def token_count_section(f, dev, ck):
    print("\n== 4. token counts that are not multiples of anything and chunk invariance (out_proj) ==")
    if not ck:
        print("  comfy_kitchen not importable: skipped")
        return
    w = load_layer(f, "attn.out_proj", dev)
    for m in (1, 2, 7, 8, 9, 17, 257, 1000):
        x = make_input("gauss", m, w.in_features, dev, seed=m)
        a, b = ql.linear(x, w, backend="ck"), ql.linear(x, w, backend="torch")
        print(f"  M={m:5d}: torch(ck profile) vs ck: bit-equal={torch.equal(a, b)} differing elements={(a != b).sum().item()} of {a.numel()}")
    x = make_input("gauss", 1000, w.in_features, dev, seed=5)
    for backend in ("ck", "torch"):
        full = ql.linear(x, w, backend=backend)
        for c in (1, 100, 333):
            ch = ql.linear(x, w, backend=backend, chunk_tokens=c)
            print(f"  backend={backend} chunk_tokens={c}: bit-equal to unchunked={torch.equal(full, ch)} differing={(full != ch).sum().item()}")


def timing_section(f, dev, ck, sizes, rows):
    print("\n== 5. ms per layer (bf16 activations, CUDA events, mean of 5 after 2 warm-ups; fc2 includes the swiglu) ==")
    tot = {(b, m): 0.0 for b in ("ck", "torch_a8", "torch_a16") for m in sizes}
    for name in LAYERS:
        w = load_layer(f, name, dev)
        n, k = w.out_features, w.in_features
        swi = name == "mlp.fc2"
        kw = {"input_act": "swiglu"} if swi else {}
        for m in sizes:
            x = make_input("gauss", m, 2 * k if swi else k, dev, seed=m)
            ops = 2.0 * m * k * n
            r = {"layer": name, "M": m}
            fns = {"torch_a8": lambda: ql.linear(x, w, backend="torch", **kw), "torch_a16": lambda: ql.linear(x, w, backend="torch", precision="a16", **kw)}
            if ck:
                fns = {"ck": lambda: ql.linear(x, w, backend="ck", **kw), **fns}
            for key, fn in fns.items():
                ms = timed(fn, iters=3 if key == "torch_a16" and m > 2000 else 5)
                r[key + "_ms"] = round(ms, 2)
                r[key + "_TOPS"] = round(ops / ms / 1e9, 1)
                tot[(key, m)] += ms
            print(" ", json.dumps(r))
            rows.append(("timing", r))
            del x
            torch.cuda.empty_cache()
        del w
    print("  block linears total (qkv+out+fc1+fc2), ms / x50 blocks in s:")
    for (key, m), v in tot.items():
        if v > 0:
            print(f"    M={m:5d} {key:10s}: {v:8.1f} ms  -> {v * 50 / 1000:6.2f} s per forward (linears only)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc-m", type=int, default=512, help="tokens for the accuracy section")
    ap.add_argument("--sizes", default="512,1768,4226")
    ap.add_argument("--skip", default="", help="comma list of sections to skip: decode,rotation,accuracy,tokens,timing")
    ap.add_argument("--no-sibling", action="store_true", help="skip the int8 sibling checkpoint comparison")
    ap.add_argument("--json", default="", help="write all result rows to this file")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(1.5 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    try:
        import comfy_kitchen  # noqa: F401

        ck = ql.available_backends()["ck"]["available"]
    except Exception:
        ck = False
    try:
        from importlib.metadata import version

        ckver = version("comfy_kitchen")
    except Exception:
        ckver = "not installed"
    print("torch", torch.__version__, torch.cuda.get_device_name(0), "| comfy_kitchen", ckver, "| backends:", json.dumps(ql.available_backends()))
    f = safe_open(os.path.join(CKPT_DIR, W4A8), "pt")
    f8 = None
    if not args.no_sibling and os.path.exists(os.path.join(CKPT_DIR, INT8)):
        f8 = safe_open(os.path.join(CKPT_DIR, INT8), "pt")
    rows = []
    sizes = [int(s) for s in args.sizes.split(",")]
    if "decode" not in skip:
        decode_section(f, dev, ck, rows)
    if "rotation" not in skip:
        rotation_section(dev, ck)
    if "accuracy" not in skip:
        accuracy_section(f, f8, dev, ck, args.acc_m, rows)
    if "tokens" not in skip:
        token_count_section(f, dev, ck)
    if "timing" not in skip:
        timing_section(f, dev, ck, sizes, rows)
    print(f"\npeak VRAM allocated by this process: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
