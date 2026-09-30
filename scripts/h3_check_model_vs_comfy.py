"""Parity of h3turbo.h3 (PackedLayout, time plan, H3Model) against ComfyUI's own MiniMaxH3Model.

Builds ComfyUI's model with a TINY random config (dense weights, comfy.ops.disable_weight_init), loads the same tensors
into H3Model and compares outputs for 19 scenarios (t2va, keyframes video+audio, refs of every kind, denoise masks, text
token tags, odd latents, prebuilt layout, audio carry), a sigma sweep, a 3-head output bank and chunking/backend options,
in fp32 and bf16 (with fp32 or ComfyUI-style bf16-rounded islands), on CPU and CUDA, plus the reference's own bf16-vs-fp32
spread as the noise floor. Internals (block-0 input, AdaLN rows, modulation segments, rope tables, every block's output)
are captured through ComfyUI's block patch hook; negative controls perturb our call and must be detected. Also fuzzes
PackedLayout against the reference and checks the q/k norm+rope ops against the kitchen op at the real head size.

Run with the ComfyUI venv (Windows paths):
  set PYTHONPATH=C:/Users/Alexp/MiniMax-H3-Turbo
  D:/ComfyUIBig/ComfyUI/ComfyUI/.venv/Scripts/python.exe scripts/h3_check_model_vs_comfy.py --device both
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

COMFY = r"D:\ComfyUIBig\ComfyUI\ComfyUI"
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_argv, sys.argv = sys.argv, [sys.argv[0]]  # comfy parses the command line at import
import torch  # noqa: E402

import comfy.ops  # noqa: E402
import comfy.ldm.minimax.model as ref  # noqa: E402

sys.argv = _argv
from h3turbo.h3.config import H3Config  # noqa: E402
from h3turbo.h3.layout import PackedLayout, norm_rope_, rope_angles, rope_ck_table, token_grid_masks  # noqa: E402
from h3turbo.h3.model import H3Model, ResidentProvider, weights_from_state_dict  # noqa: E402

TINY = dict(hidden=256, layers=3, refiner_layers=2, heads=4, head_dim=64, ffn=192, text_dim=160, t_dim=8,
            curve_grid=65, rope_inv_freq_len=8)
T, H, W, TA, L = 3, 6, 8, 9, 12  # target latent frames/h/w, audio frames, text tokens
FAILS: list = []


# ----------------------------------------------------------------------------- random network


def make_state_dict(cfg: H3Config, seed: int, head_bank: int = 1) -> dict:
    """Random weights with realistic scales; linears/norms are bf16-rounded so fp32 and bf16 models share values."""
    g = torch.Generator().manual_seed(seed)
    hid, inner, ffn = cfg.hidden, cfg.inner, cfg.ffn
    bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
    rnd = lambda *shape, std: torch.randn(*shape, generator=g) * std  # noqa: E731
    sd = {}

    def block(p, adaln):
        sd[p + ".attn.qkv_proj.weight"] = bf(rnd(3 * inner, hid, std=hid ** -0.5))
        sd[p + ".attn.out_proj.weight"] = bf(rnd(hid, inner, std=0.7 * inner ** -0.5))
        sd[p + ".mlp.fc1.weight"] = bf(rnd(2 * ffn, hid, std=hid ** -0.5))
        sd[p + ".mlp.fc2.weight"] = bf(rnd(hid, ffn, std=0.7 * ffn ** -0.5))
        for n, d in (("norm1", hid), ("norm2", hid), ("attn.q_norm", cfg.head_dim), ("attn.k_norm", cfg.head_dim)):
            sd[f"{p}.{n}.weight"] = bf(1.0 + rnd(d, std=0.2))
        if adaln:
            sd[p + ".adaln_proj.linear.weight"] = bf(rnd(18 * hid, cfg.t_dim, std=0.35))
            sd[p + ".adaln_proj.linear.bias"] = rnd(18 * hid, std=0.4)

    for i in range(cfg.layers):
        block(f"blocks.{i}", True)
    for j in range(cfg.refiner_layers):
        block(f"token_refiner.blocks.{j}", False)
    sd["token_refiner.final_norm.weight"] = bf(1.0 + rnd(hid, std=0.2))
    sd["video_patch_proj.weight"], sd["video_patch_proj.bias"] = rnd(hid, cfg.video_patch_dim, std=0.15), rnd(hid, std=0.1)
    sd["audio_patch_proj.weight"], sd["audio_patch_proj.bias"] = rnd(hid, cfg.audio_channels, std=0.15), rnd(hid, std=0.1)
    sd["condition_proj.weight"], sd["condition_proj.bias"] = bf(rnd(hid, cfg.text_dim, std=cfg.text_dim ** -0.5)), bf(rnd(hid, std=0.1))
    grid = torch.linspace(0, 1, cfg.curve_grid)
    sd["adaln_t_table"] = torch.stack([0.5 * 0.4 ** j * torch.sin(2 * math.pi * (1 + j) * grid + 1.3 * j) for j in range(cfg.t_dim)], dim=1)
    sd["rope.inv_freq"] = 10.0 ** (-torch.arange(cfg.rope_inv_freq_len, dtype=torch.float32) / 4.0)
    sd["final_layer.norm.weight"] = bf(1.0 + rnd(hid, std=0.2))
    sd["final_layer.adaln_proj.linear.weight"] = bf(rnd(2 * hid, cfg.t_dim, std=0.35))
    sd["final_layer.adaln_proj.linear.bias"] = rnd(2 * hid, std=0.3)
    sd["final_layer.video_out.weight"] = rnd(head_bank * cfg.video_patch_dim, hid, std=hid ** -0.5)
    sd["final_layer.video_out.bias"] = rnd(head_bank * cfg.video_patch_dim, std=0.1)
    sd["final_layer.audio_out.weight"] = rnd(head_bank * cfg.audio_channels, hid, std=hid ** -0.5)
    sd["final_layer.audio_out.bias"] = rnd(head_bank * cfg.audio_channels, std=0.1)
    return sd


ISLANDS = ("video_patch_proj.", "audio_patch_proj.", "final_layer.video_out.", "final_layer.audio_out.", "adaln_proj.linear.")


def native_islands(sd: dict) -> dict:
    """What ComfyUI's mixed-precision loader does to the checkpoint's fp32 tensors: round them through bf16."""
    return {k: (v.bfloat16().float() if any(i in k for i in ISLANDS) else v) for k, v in sd.items()}


def build_reference(cfg: H3Config, sd: dict, dtype, device, head_bank: int = 1):
    m = ref.MiniMaxH3Model(
        hidden_size=cfg.hidden, num_layers=cfg.layers, token_refiner_num_layers=cfg.refiner_layers,
        num_attention_heads=cfg.heads, attention_head_dim=cfg.head_dim, ffn_hidden_size=cfg.ffn,
        latents_dim=cfg.video_channels, audio_latents_dim=cfg.audio_channels, patch_size=cfg.patch, text_dim=cfg.text_dim,
        time_embed_dim=cfg.t_dim, rope_inv_freq_len=cfg.rope_inv_freq_len, adaln_curve_grid=cfg.curve_grid,
        dtype=dtype, device="cpu", operations=comfy.ops.disable_weight_init)
    if head_bank > 1:  # the reference infers the bank from the weight shape; give it the bank tensors directly
        for name, out in (("video_out", cfg.video_patch_dim), ("audio_out", cfg.audio_channels)):
            lin = getattr(m.final_layer, name)
            lin.weight = torch.nn.Parameter(torch.empty(head_bank * out, cfg.hidden), requires_grad=False)
            lin.bias = torch.nn.Parameter(torch.empty(head_bank * out), requires_grad=False)
    m.load_state_dict(sd, strict=True)
    m.requires_grad_(False)  # ComfyUI loads inference weights without grad; the kitchen in-place rope requires it
    return m.to(device).eval()


def build_mine(cfg: H3Config, sd: dict, dtype, device, **kw) -> H3Model:
    glob, blocks = weights_from_state_dict(sd, cfg, device, dtype)
    return H3Model(cfg, glob, ResidentProvider(blocks), device=device, dtype=dtype, **kw)


# ----------------------------------------------------------------------------- scenarios


def _lat(g, *shape):
    return torch.randn(*shape, generator=g)


def scenario_inputs(name: str, cfg: H3Config, seed: int) -> dict:
    """Latents, text, payload, masks and sigma for one scenario (fp32 CPU; cast/moved by the caller)."""
    g = torch.Generator().manual_seed(seed)
    t, h, w, ta = T, H, W, TA
    if name == "t2va_odd":
        t, h, w = 2, 5, 7
    elif name == "t2va_big":  # S ~ 630 rows: big enough for ComfyUI's SDPA backend priority order to apply
        t, h, w, ta = 12, 12, 16, 20
    hp, wp = (h + 1) // 2 * 2, (w + 1) // 2 * 2
    sc = dict(x=[_lat(g, 1, 24, t, h, w), _lat(g, 1, 32, 2, ta)], text=_lat(g, 1, L, cfg.text_dim), sigma=0.8,
              payload={"seed": 7}, denoise_mask=None, audio_denoise_mask=None, prebuild_layout=False)
    p = sc["payload"]
    kf1 = {"resolved_frame_index": 0, "latent": _lat(g, 1, 24, 1, hp, wp), "audio_latent": _lat(g, 1, 32, 2, 4)}
    kf2 = {"resolved_frame_index": 12, "latent": _lat(g, 1, 24, 2, hp, wp)}
    kf3 = {"resolved_frame_index": 5, "audio_latent": _lat(g, 1, 32, 2, 3)}
    r_img = {"kind": "image", "latent_h": 4, "latent_w": 6, "latent": _lat(g, 1, 24, 1, 4, 6)}
    r_aud = {"kind": "audio", "ref_audio_t": 5, "audio_latent": _lat(g, 1, 32, 2, 5)}
    r_vid = {"kind": "video", "latent_t": 3, "latent_h": 4, "latent_w": 6, "ref_audio_t": 0, "latent": _lat(g, 1, 24, 3, 4, 6), "audio_latent": None}
    r_vau = {"kind": "video_audio", "latent_t": 2, "latent_h": 6, "latent_w": 4, "ref_audio_t": 6, "latent": _lat(g, 1, 24, 2, 6, 4),
             "audio_latent": _lat(g, 1, 32, 2, 6)}
    tags = torch.tensor([[1, 1, 0, 0, 0, 1, 2, 2, 1, 0, 1, 1]])

    def set_kf(kfs):
        p["keyframes"] = kfs
        p["cond_video_latents"] = [k["latent"] for k in kfs if k.get("latent") is not None]
        p["cond_audio_latents"] = [k["audio_latent"] for k in kfs if k.get("audio_latent") is not None]

    def set_refs(refs):
        p["refs"] = refs
        p["cond_video_latents"] = p.get("cond_video_latents", []) + [r["latent"] for r in refs if "latent" in r]
        p["cond_audio_latents"] = p.get("cond_audio_latents", []) + [r["audio_latent"] for r in refs if r.get("audio_latent") is not None]

    if name == "t2va_tags":
        p["text_token_tags"] = tags
    elif name in ("fl2va", "fl2va_aug", "fl2va_noaug"):
        set_kf([kf1, kf2] if name == "fl2va_noaug" else [kf1, kf2, kf3])
        if name != "fl2va":
            p["visual_cond_noise_aug"], p["audio_cond_noise_aug"] = (0.9, 0.7) if name == "fl2va_aug" else (1.0, 1.0)
    elif name in ("ref_image", "ref_audio", "ref_video", "ref_video_audio"):
        set_refs([{"ref_image": r_img, "ref_audio": r_aud, "ref_video": r_vid, "ref_video_audio": r_vau}[name]])
    elif name in ("ref_mix", "ref_mix_prebuilt"):
        set_kf([kf1, kf3])
        set_refs([r_img, r_vau, r_aud, r_vid])
        p["text_token_tags"] = tags
        sc["prebuild_layout"] = name.endswith("prebuilt")
    elif name in ("mask_video", "mask_both", "mask_video_fl2va"):
        m = torch.ones(1, 1, t, h, w)
        m[:, :, 0] = 0.0
        m[:, :, 1, :2, :3] = 0.5
        m[:, :, 2, 3:, 5:] = 0.21875
        sc["denoise_mask"] = token_grid_masks(m)[0]
        if name == "mask_both":
            am = torch.ones(1, 32, 2, ta)
            am[:, :, :, :3] = 0.0
            am[:, :, :, 5] = 0.4
            sc["audio_denoise_mask"] = token_grid_masks(m, am)[1][:1].amax(dim=1, keepdim=True)
        if name == "mask_video_fl2va":
            set_kf([kf1])
    elif name == "mask_uniform":
        sc["denoise_mask"], sc["audio_denoise_mask"] = torch.full((1, 1, t, h, w), 0.4), torch.full((1, 1, 2, ta), 0.25)
    elif name == "mask_audio":
        am = torch.ones(1, 1, 2, ta)
        am[..., :4] = 0.0
        sc["audio_denoise_mask"] = am
    elif name == "audio_scale":
        p["audio_scale"] = 1.37
    if sc["prebuild_layout"]:  # what MiniMaxH3.extra_conds does once per sampling run
        p["layout"] = ref.PackedLayout(L, t, hp, wp, ta, keyframes=p.get("keyframes"), refs=p.get("refs"))
    return sc


SCENARIOS = ["t2va", "t2va_odd", "t2va_big", "t2va_tags", "fl2va", "fl2va_aug", "fl2va_noaug", "ref_image", "ref_audio", "ref_video",
             "ref_video_audio", "ref_mix", "ref_mix_prebuilt", "mask_video", "mask_both", "mask_uniform", "mask_audio",
             "mask_video_fl2va", "audio_scale"]
SIGMAS = [1.0, 0.95, 0.5, 0.1, 0.004, 1e-7]
sc_seed = lambda name: 11 + SCENARIOS.index(name)  # noqa: E731


# ----------------------------------------------------------------------------- running and comparing


def err(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().cpu(), b.double().cpu()
    d = a - b
    return {"max_abs": d.abs().max().item(), "rel_l2": (d.norm() / b.norm().clamp_min(1e-30)).item(), "ref_rms": b.pow(2).mean().sqrt().item()}


def fmt(e: dict) -> str:
    return f"max_abs={e['max_abs']:.2e} rel_l2={e['rel_l2']:.2e}"


def check(cond: bool, what: str) -> None:
    if not cond:
        FAILS.append(what)
        print("  FAIL:", what)


def tol_for(dtype) -> float:
    return 2e-5 if dtype == torch.float32 else 3e-2


def _cast(sc, dtype, device):
    x = [t.to(device=device, dtype=dtype) for t in sc["x"]]
    dm, am = (None if m is None else m.to(device=device, dtype=dtype) for m in (sc["denoise_mask"], sc["audio_denoise_mask"]))
    return x, torch.tensor([sc["sigma"] * 1000.0], device=device), sc["text"].to(device=device, dtype=dtype), dm, am


def run_reference(model, sc, dtype, device, hooks=None, sample_sigmas=None):
    x, ts, text, dm, am = _cast(sc, dtype, device)
    opts = {}
    if hooks is not None:
        opts["patches_replace"] = {"dit": {("double_block", i): hooks(i) for i in range(len(model.blocks))}}
    if sample_sigmas is not None:
        opts["sample_sigmas"] = sample_sigmas.to(device)
    with torch.inference_mode():  # the kitchen in-place rope refuses tensors that require grad
        return model(x, ts, text, transformer_options=opts, minimax_payload=sc["payload"], denoise_mask=dm, audio_denoise_mask=am)


def run_mine(model: H3Model, sc, dtype, device, capture=None, sample_sigmas=None):
    x, ts, text, dm, am = _cast(sc, dtype, device)
    sigma = (ts.flatten()[0] / 1000.0).float()  # same fp32 arithmetic as the reference
    if capture is not None:
        orig = model._block

        def wrapped(xx, w, t_emb, segments, rope):
            if "h0" not in capture:
                capture.update(h0=xx.clone(), t_emb=t_emb.clone(), segments=segments, rope=rope, blocks=[])
            out = orig(xx, w, t_emb, segments, rope)
            capture["blocks"].append(out.clone())
            return out

        model._block = wrapped
    try:
        return model.forward(x, sigma, text, payload=sc["payload"], denoise_mask=dm, audio_denoise_mask=am, sample_sigmas=sample_sigmas)
    finally:
        if capture is not None:
            del model._block


def make_hooks(store):
    def hooks(i):
        def hook(args, ctx):
            if i == 0:
                store.update(h0=args["img"].clone(), t_emb=args["t_emb"].clone(), segments=args["mod_segments"],
                             rope=args["rope_freqs"].clone(), layout=args["layout"], blocks=[])
            out = ctx["original_block"](args)
            store["blocks"].append(out["img"].clone())
            return out
        return hook
    return hooks


def compare_internals(name, mine_cap, ref_cap):
    e0 = err(mine_cap["h0"], ref_cap["h0"])
    check(e0["max_abs"] == 0.0, f"{name}: block-0 input differs ({fmt(e0)})")
    check(torch.equal(mine_cap["t_emb"].cpu(), ref_cap["t_emb"].cpu()), f"{name}: AdaLN input rows differ")
    same = lambda r1, r2: torch.equal(r1.cpu(), r2.cpu()) if isinstance(r1, torch.Tensor) and isinstance(r2, torch.Tensor) else r1 == r2  # noqa: E731
    ms, rs = mine_cap["segments"], ref_cap["segments"]
    check(len(ms) == len(rs) and all(a == c and b == d and same(r1, r2) for (a, b, r1), (c, d, r2) in zip(ms, rs)), f"{name}: modulation segments differ")
    cos, sin = mine_cap["rope"][:2]
    rt = ref_cap["rope"]
    check(torch.equal(cos.cpu(), rt[0, :, 0, :, 0, 0].cpu()) and torch.equal(sin.cpu(), rt[0, :, 0, :, 1, 0].cpu()), f"{name}: rope tables differ")
    return e0, [err(a, b)["rel_l2"] for a, b in zip(mine_cap["blocks"], ref_cap["blocks"])]


def compare_layouts(a, b, tag) -> bool:
    ok = a.seq_len == b.seq_len and tuple(a.signature) == tuple(b.signature) and [tuple(s) for s in a.segments] == [tuple(s) for s in b.segments]
    ok = ok and all(torch.equal(getattr(a, f), getattr(b, f)) for f in ("position_ids", "img_pos", "img_update", "audio_pos", "audio_update"))
    if not ok:
        FAILS.append(f"layout mismatch {tag}")
    return ok


def fuzz_layouts(n: int, seed: int) -> None:
    rng = random.Random(seed)
    bad = 0
    for it in range(n):
        text_len, t, ta = rng.randint(0, 30), rng.randint(1, 8), rng.randint(1, 20)
        h, w = 2 * rng.randint(1, 9), 2 * rng.randint(1, 9)
        kfs, refs = [], []
        for _ in range(rng.randint(0, 3)):
            kf = {"resolved_frame_index": rng.randint(0, 60)}
            if rng.random() < 0.8:
                kf["latent"] = torch.zeros(1, 24, rng.randint(1, 4), h, w)
            if rng.random() < 0.5:
                kf["audio_latent"] = torch.zeros(1, 32, 2, rng.randint(1, 9))
            kfs.append(kf)
        for _ in range(rng.randint(0, 4)):
            kind = rng.choice(["image", "audio", "video", "video_audio"])
            rt = {"video": 0, "video_audio": rng.randint(1, 12)}.get(kind, rng.randint(0, 12))
            refs.append({"kind": kind, "latent_h": 2 * rng.randint(1, 8), "latent_w": 2 * rng.randint(1, 8), "latent_t": rng.randint(1, 6), "ref_audio_t": rt})
        a = PackedLayout.build(text_len, t, h, w, ta, keyframes=kfs or None, refs=refs or None)
        b = ref.PackedLayout(text_len, t, h, w, ta, keyframes=kfs or None, refs=refs or None)
        bad += not compare_layouts(a, b, f"fuzz#{it}")
    print(f"layout fuzz: {n} random signatures, {bad} mismatches (exact float64 equality of all tables)")


def mask_helper_check() -> None:
    import types  # token_grid_masks vs MiniMaxH3._pool_masks_to_token_grid + rounding (model_base)
    import comfy.model_base as mb
    fake = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(patch_size=(1, 2, 2)))
    g = torch.Generator().manual_seed(4)
    for (t, h, w, ta) in ((3, 6, 8, 9), (2, 5, 7, 4), (1, 3, 3, 2)):
        v, a = torch.rand(1, 1, t, h, w, generator=g), torch.rand(1, 32, 2, ta, generator=g)
        want = [torch.ceil(m * 256.0) / 256.0 for m in mb.MiniMaxH3._pool_masks_to_token_grid(fake, [v, a])]
        got = token_grid_masks(v, a)
        check(torch.equal(want[0], got[0]) and torch.equal(want[1], got[1]), f"token_grid_masks {(t, h, w, ta)}")
    print("token_grid_masks: exact match with model_base pooling on 3 shapes")


def rope_unit_check(device, dtype) -> None:
    """q/k norm+rope on views of one packed qkv buffer at the real head geometry (128-dim, 96 rotated): every impl vs the kitchen op."""
    import comfy.quant_ops as qo
    g = torch.Generator().manual_seed(3)
    s, heads, hd, nf, eps = 2048, 8, 128, 16, 1e-5
    inner = heads * hd
    ang = rope_angles(torch.rand(s, 3, generator=g, dtype=torch.float64) * 40, 10.0 ** (-torch.arange(nf, dtype=torch.float32) / 4.0), device)
    table = ref.rope_rotation_table(torch.cat([ang, ang], -1), dtype)
    cos, sin = torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)
    qkv = (2 * torch.randn(s, 3 * inner, generator=g)).to(device=device, dtype=dtype)
    w = (1.0 + 0.2 * torch.randn(hd, generator=g)).to(device=device, dtype=dtype)
    rq, rk = qkv[:, :inner].clone().view(1, s, heads, hd), qkv[:, inner: 2 * inner].clone().view(1, s, heads, hd)
    with torch.inference_mode():
        qo.ck.rms_rope_split_half_(rq, rk, table, w, w, epsilon=eps, rot_dim=2 * cos.shape[1])
    for impl in ["torch", "eager"] + (["ck"] if dtype == torch.bfloat16 and device == "cuda" else []):
        buf = qkv.clone()  # q and k as strided views of ONE buffer, like the model
        q, k = buf[:, :inner].view(s, heads, hd), buf[:, inner: 2 * inner].view(s, heads, hd)
        norm_rope_(q, k, w, w, eps, cos, sin, impl=impl, table=rope_ck_table(cos, sin), chunk=700)
        n_diff = int((q != rq[0]).sum() + (k != rk[0]).sum())
        print(f"  rope unit [{device},{str(dtype)[6:]}] impl={impl:6s} vs kitchen op: q {fmt(err(q, rq[0]))}; differing elements {n_diff} of {2 * q.numel()}")
        if impl == "ck" or (impl == "torch" and device == "cuda") or dtype == torch.float32:
            check(n_diff == 0, f"rope unit {device} {dtype} impl {impl}: {n_diff} differing elements")


# ----------------------------------------------------------------------------- the parity run


def sweep(model: H3Model, ref_out: dict, dtype, device) -> tuple:
    exact, wv, wa = 0, 0.0, 0.0  # -> (#bit-identical scenarios, worst video rel_l2, worst audio rel_l2)
    for name in SCENARIOS:
        o = run_mine(model, scenario_inputs(name, model.cfg, sc_seed(name)), dtype, device)
        ev, ea = err(o[0], ref_out[name][0]), err(o[1], ref_out[name][1])
        exact += int(ev["max_abs"] == 0 and ea["max_abs"] == 0)
        wv, wa = max(wv, ev["rel_l2"]), max(wa, ea["rel_l2"])
    return exact, wv, wa


def negative_controls(cfg, m_ref, m_mine, dtype, device) -> None:
    print("-- negative controls: perturb one input of OUR call only, the harness must notice")
    want = run_reference(m_ref, scenario_inputs("fl2va", cfg, 3), dtype, device)
    tol = 1e-5 if dtype == torch.float32 else 2e-4
    for label, edit in (("sigma + 1e-3", lambda c: c.update(sigma=c["sigma"] + 1e-3)),
                        ("keyframe index + 1", lambda c: c["payload"]["keyframes"][1].update(resolved_frame_index=13)),
                        ("cond noise seed + 1", lambda c: c["payload"].update(seed=8)),
                        ("one text token tag", lambda c: c["payload"].update(text_token_tags=torch.tensor([[1] * (L - 1) + [0]])))):
        c2 = scenario_inputs("fl2va", cfg, 3)
        c2["payload"] = dict(c2["payload"], keyframes=[dict(k) for k in c2["payload"]["keyframes"]])
        edit(c2)
        e = err(run_mine(m_mine, c2, dtype, device)[0], want[0])
        print(f"{label:22s} video rel_l2={e['rel_l2']:.2e}")
        check(e["rel_l2"] > tol, f"negative control '{label}' not detected (rel_l2 {e['rel_l2']:.2e})")


def parity(cfg, sd, device, dtype, native: bool) -> None:
    """native=True: reference weights carry ComfyUI's bf16-rounded islands and H3Model uses its default (rounds them too);
    native=False: reference keeps fp32 islands and H3Model is built with fp32_islands=True."""
    tag = f"{device}/{str(dtype)[6:]}/{'native bf16 islands' if native else 'fp32 islands'}"
    print(f"\n=== parity {tag} ===")
    sd = native_islands(sd) if native else sd
    kw = {"fp32_islands": not native}
    m_ref, m_mine = build_reference(cfg, sd, dtype, device), build_mine(cfg, sd, dtype, device, **kw)
    floor = build_reference(cfg, sd, torch.float32, device) if dtype != torch.float32 else None
    tol = tol_for(dtype)
    print(f"rope_impl on this device/dtype: {m_mine.rope_impl}; sdpa priority order active: {m_mine._sdpa_prio is not None}")
    ref_out, worst = {}, 0.0
    for name in SCENARIOS:
        sc = scenario_inputs(name, cfg, sc_seed(name))
        rcap, mcap = {}, {}
        o_ref = ref_out[name] = run_reference(m_ref, sc, dtype, device, hooks=make_hooks(rcap))
        o_mine = run_mine(m_mine, sc, dtype, device, capture=mcap)
        ev, ea = err(o_mine[0], o_ref[0]), err(o_mine[1], o_ref[1])
        e0, growth = compare_internals(name, mcap, rcap)
        if not sc["prebuild_layout"]:
            t, h, w = sc["x"][0].shape[2:]
            check(compare_layouts(m_mine._layout(sc["payload"], L, (t, (h + 1) // 2 * 2, (w + 1) // 2 * 2), sc["x"][1].shape[-1]), rcap["layout"], name), f"{name}: layout tables differ")
        check(o_mine[0].dtype == o_ref[0].dtype and o_mine[0].shape == o_ref[0].shape and o_mine[1].shape == o_ref[1].shape, f"{name}: output shape/dtype")
        line = f"{name:18s} video {fmt(ev)} | audio {fmt(ea)} | h0 {e0['max_abs']:.0e} | per-block rel_l2 " + ",".join(f"{v:.0e}" for v in growth)
        if floor is not None:
            o_fl = run_reference(floor, sc, torch.float32, device)
            line += f"\n{'':18s} noise floor (ref bf16 vs ref fp32) video rel_l2={err(o_ref[0], o_fl[0])['rel_l2']:.2e} audio {err(o_ref[1], o_fl[1])['rel_l2']:.2e}"
        print(line)
        check(ev["rel_l2"] < tol and ea["rel_l2"] < tol, f"{tag} {name}: rel_l2 video {ev['rel_l2']:.2e} audio {ea['rel_l2']:.2e} > {tol}")
        worst = max(worst, ev["rel_l2"], ea["rel_l2"])
    print("-- rope implementations x all scenarios (bit-identical outputs / worst rel_l2)")
    for impl in ("torch", "eager") + (("ck",) if dtype == torch.bfloat16 and device == "cuda" else ()):
        exact, wv, wa = sweep(build_mine(cfg, sd, dtype, device, rope_impl=impl, **kw), ref_out, dtype, device)
        print(f"rope_impl={impl:6s}: {exact}/{len(SCENARIOS)} bit-identical to the reference, worst rel_l2 video {wv:.2e} audio {wa:.2e}")
        check(max(wv, wa) < tol, f"{tag} rope_impl {impl}")
    negative_controls(cfg, m_ref, m_mine, dtype, device)
    print("-- sigma sweep (t2va)")
    for s in SIGMAS:
        sc = scenario_inputs("t2va", cfg, 5)
        sc["sigma"] = s
        o_ref, o = run_reference(m_ref, sc, dtype, device), run_mine(m_mine, sc, dtype, device)
        ev, ea = err(o[0], o_ref[0]), err(o[1], o_ref[1])
        print(f"sigma={s:<8g} video {fmt(ev)} | audio {fmt(ea)}")
        check(max(ev["rel_l2"], ea["rel_l2"]) < tol, f"{tag} sigma {s}")
    txt = torch.randn(1, 17, cfg.text_dim, generator=torch.Generator().manual_seed(2)).to(device=device, dtype=dtype)
    with torch.no_grad():
        e = err(m_mine.encode_text(txt), m_ref.preprocess_text_embeds(txt))
    print(f"-- encode_text vs preprocess_text_embeds: {fmt(e)}")
    check(e["rel_l2"] < tol, f"{tag} encode_text")
    print("-- options on mine, ref_mix scenario, vs reference")
    sc = scenario_inputs("ref_mix", cfg, 99)
    o_ref, base = run_reference(m_ref, sc, dtype, device), run_mine(m_mine, sc, dtype, device)
    for opt in ({"mlp_chunk": 7}, {"attn_chunk": 13}, {"rope_chunk": 5}, {"mlp_chunk": 9, "attn_chunk": 11, "rope_chunk": 3},
                {"attn_backend": "default"}, {"rope_dtype": torch.float32}):
        o = run_mine(build_mine(cfg, sd, dtype, device, **kw, **opt), sc, dtype, device)
        print(f"{str(opt):58s} vs ref rel_l2 video {err(o[0], o_ref[0])['rel_l2']:.2e} audio {err(o[1], o_ref[1])['rel_l2']:.2e} | vs default mine {err(o[0], base[0])['rel_l2']:.2e}")
        check(err(o[0], o_ref[0])["rel_l2"] < tol, f"{tag} option {opt}")
    sc = scenario_inputs("t2va_big", cfg, 98)
    o_ref, base = run_reference(m_ref, sc, dtype, device), run_mine(m_mine, sc, dtype, device)
    for opt in ({}, {"attn_backend": "default"}, {"attn_chunk": 100}, {"mlp_chunk": 64}, {"rope_chunk": 100}):
        o = run_mine(build_mine(cfg, sd, dtype, device, **kw, **opt), sc, dtype, device)
        print(f"t2va_big {str(opt):49s} vs ref rel_l2 video {err(o[0], o_ref[0])['rel_l2']:.2e} audio {err(o[1], o_ref[1])['rel_l2']:.2e}")
        check(err(o[0], o_ref[0])["rel_l2"] < tol, f"{tag} big option {opt}")
    print(f"worst rel_l2 over the scenarios [{tag}]: {worst:.3e}")


def head_bank_check(cfg, device, dtype) -> None:
    print(f"\n=== 3-head output bank {device}/{str(dtype)[6:]} ===")
    sd = make_state_dict(cfg, 21, head_bank=3)
    m_ref = build_reference(cfg, sd, dtype, device, head_bank=3)
    m_mine = build_mine(cfg, sd, dtype, device, fp32_islands=True)
    sched = torch.tensor([1.0, 0.9, 0.7, 0.45, 0.2, 0.0])
    for s in (1.0, 0.9, 0.7, 0.45, 0.2):
        sc = scenario_inputs("t2va", cfg, 8)
        sc["sigma"] = s
        o_ref, o = run_reference(m_ref, sc, dtype, device, sample_sigmas=sched), run_mine(m_mine, sc, dtype, device, sample_sigmas=sched)
        ev, ea = err(o[0], o_ref[0]), err(o[1], o_ref[1])
        print(f"sigma={s:<5g} video rel_l2={ev['rel_l2']:.2e} audio rel_l2={ea['rel_l2']:.2e}")
        check(max(ev["rel_l2"], ea["rel_l2"]) < tol_for(dtype), f"head bank sigma {s}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="both", choices=["cpu", "cuda", "both"])
    ap.add_argument("--dtypes", default="fp32,bf16")
    ap.add_argument("--fuzz", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    cfg = H3Config(**TINY)
    devices = ["cpu"] * (args.device in ("cpu", "both")) + ["cuda"] * (args.device in ("cuda", "both") and torch.cuda.is_available())
    dtypes = [{"fp32": torch.float32, "bf16": torch.bfloat16}[d] for d in args.dtypes.split(",")]
    print("torch", torch.__version__, "| devices", devices, "| comfy attention:", ref.optimized_attention.__name__)
    fuzz_layouts(args.fuzz, args.seed)
    mask_helper_check()
    sd = make_state_dict(cfg, args.seed)
    for device in devices:
        for dtype in dtypes:
            if device == "cuda":
                rope_unit_check(device, dtype)
            parity(cfg, sd, device, dtype, native=False)
            if dtype == torch.bfloat16:  # ComfyUI keeps every parameter in the compute dtype: the checkpoint's fp32 islands become bf16
                parity(cfg, sd, device, dtype, native=True)
            head_bank_check(cfg, device, dtype)
        if device == "cuda":
            torch.cuda.empty_cache()
    print("\nRESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    for f in FAILS:
        print("  -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
