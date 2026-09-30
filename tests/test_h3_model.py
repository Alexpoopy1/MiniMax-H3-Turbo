"""H3Model on a tiny random network (CPU): fp64 cross-check, chunk exactness, provider protocol, payloads, errors.

The bit-level proof against ComfyUI's own model is scripts/h3_check_model_vs_comfy.py; layout/plan/rope-op tests are in
test_h3_layout.py. The independent fp64 implementation below is written from the spec, not from the library.
"""
from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch

from h3turbo.h3.config import H3Config
from h3turbo.h3.layout import (PackedLayout, curve_lerp, pack_audio, pad_to_patch, patchify_video, plan_time, rope_angles,
                               time_shift_sigma, unpack_audio, unpatchify_video)
import h3turbo.h3.model as model_mod
from h3turbo.h3.model import H3Model, ResidentProvider, weights_from_state_dict
from h3turbo.h3.types import W4A8Weight

CFG = H3Config(hidden=64, layers=2, refiner_layers=1, heads=2, head_dim=32, ffn=48, text_dim=40, t_dim=8, curve_grid=17,
               rope_inv_freq_len=4)
T, H, W, TA, L = 2, 4, 6, 4, 5


def make_sd(cfg: H3Config, seed: int = 0, head_bank: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, std: torch.randn(*s, generator=g) * std  # noqa: E731
    hid, inner, ffn = cfg.hidden, cfg.inner, cfg.ffn
    sd = {}

    def block(p, adaln):
        sd[p + ".attn.qkv_proj.weight"] = r(3 * inner, hid, std=hid ** -0.5)
        sd[p + ".attn.out_proj.weight"] = r(hid, inner, std=0.7 * inner ** -0.5)
        sd[p + ".mlp.fc1.weight"] = r(2 * ffn, hid, std=hid ** -0.5)
        sd[p + ".mlp.fc2.weight"] = r(hid, ffn, std=0.7 * ffn ** -0.5)
        for n, d in (("norm1", hid), ("norm2", hid), ("attn.q_norm", cfg.head_dim), ("attn.k_norm", cfg.head_dim)):
            sd[f"{p}.{n}.weight"] = 1.0 + r(d, std=0.2)
        if adaln:
            sd[p + ".adaln_proj.linear.weight"] = r(18 * hid, cfg.t_dim, std=0.35)
            sd[p + ".adaln_proj.linear.bias"] = r(18 * hid, std=0.4)

    for i in range(cfg.layers):
        block(f"blocks.{i}", True)
    for j in range(cfg.refiner_layers):
        block(f"token_refiner.blocks.{j}", False)
    sd["token_refiner.final_norm.weight"] = 1.0 + r(hid, std=0.2)
    sd["video_patch_proj.weight"], sd["video_patch_proj.bias"] = r(hid, cfg.video_patch_dim, std=0.15), r(hid, std=0.1)
    sd["audio_patch_proj.weight"], sd["audio_patch_proj.bias"] = r(hid, cfg.audio_channels, std=0.15), r(hid, std=0.1)
    sd["condition_proj.weight"], sd["condition_proj.bias"] = r(hid, cfg.text_dim, std=cfg.text_dim ** -0.5), r(hid, std=0.1)
    grid = torch.linspace(0, 1, cfg.curve_grid)
    sd["adaln_t_table"] = torch.stack([0.5 * 0.4 ** j * torch.sin(6.28 * (1 + j) * grid + j) for j in range(cfg.t_dim)], dim=1)
    sd["rope.inv_freq"] = 10.0 ** (-torch.arange(cfg.rope_inv_freq_len, dtype=torch.float32) / 4.0)
    sd["final_layer.norm.weight"] = 1.0 + r(hid, std=0.2)
    sd["final_layer.adaln_proj.linear.weight"], sd["final_layer.adaln_proj.linear.bias"] = r(2 * hid, cfg.t_dim, std=0.35), r(2 * hid, std=0.3)
    sd["final_layer.video_out.weight"] = r(head_bank * cfg.video_patch_dim, hid, std=hid ** -0.5)
    sd["final_layer.video_out.bias"] = r(head_bank * cfg.video_patch_dim, std=0.1)
    sd["final_layer.audio_out.weight"] = r(head_bank * cfg.audio_channels, hid, std=hid ** -0.5)
    sd["final_layer.audio_out.bias"] = r(head_bank * cfg.audio_channels, std=0.1)
    return sd


def make_model(cfg=CFG, sd=None, dtype=torch.float32, provider=None, **kw) -> H3Model:
    glob, blocks = weights_from_state_dict(sd or make_sd(cfg), cfg, "cpu", dtype)
    return H3Model(cfg, glob, provider(blocks) if provider else ResidentProvider(blocks), device="cpu", dtype=dtype, **kw)


def inputs(seed=1, t=T, h=H, w=W, ta=TA, cfg=CFG):
    g = torch.Generator().manual_seed(seed)
    return ([torch.randn(1, 24, t, h, w, generator=g), torch.randn(1, 32, 2, ta, generator=g)],
            torch.randn(1, L, cfg.text_dim, generator=g))


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


# ------------------------------------------------------------------------------------------------ model


def _naive(sd, cfg, x, sigma, text, payload=None, layout=None):
    """fp64 forward written straight from the spec; only the token->AdaLN-row assignment comes from the library."""
    d = {k: v.double() for k, v in sd.items()}
    hid, hd, heads = cfg.hidden, cfg.head_dim, cfg.heads
    rms = lambda v, w, e: v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + e) * w  # noqa: E731
    silu = lambda v: v * torch.sigmoid(v)  # noqa: E731

    def attn(h, p, ang):
        s = h.shape[0]
        q, k, v = (h @ d[p + ".attn.qkv_proj.weight"].T).reshape(s, 3, heads, hd).unbind(1)
        q, k = rms(q, d[p + ".attn.q_norm.weight"], cfg.qk_norm_eps), rms(k, d[p + ".attn.k_norm.weight"], cfg.qk_norm_eps)
        if ang is not None:
            half = ang.shape[1]
            c, sn = torch.cos(ang)[:, None], torch.sin(ang)[:, None]
            rot = lambda t: torch.cat([t[..., :half] * c - t[..., half:2 * half] * sn, t[..., :half] * sn + t[..., half:2 * half] * c, t[..., 2 * half:]], -1)  # noqa: E731
            q, k = rot(q), rot(k)
        p_ = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd), -1)
        return torch.einsum("hqk,khd->qhd", p_, v).reshape(s, -1) @ d[p + ".attn.out_proj.weight"].T

    def mlp(h, p):
        g, u = (h @ d[p + ".mlp.fc1.weight"].T).chunk(2, -1)
        return (silu(g) * u) @ d[p + ".mlp.fc2.weight"].T

    ctx = d["condition_proj.bias"] + text[0].double() @ d["condition_proj.weight"].T
    for j in range(cfg.refiner_layers):
        p = f"token_refiner.blocks.{j}"
        ctx = ctx + attn(rms(ctx, d[p + ".norm1.weight"], cfg.norm_eps), p, None)
        ctx = ctx + mlp(rms(ctx, d[p + ".norm2.weight"], cfg.norm_eps), p)
    ctx = rms(ctx, d["token_refiner.final_norm.weight"], cfg.final_norm_eps)
    video, audio = x
    lat_t, lat_h, lat_w = video.shape[2:]
    lay = layout or PackedLayout.build(ctx.shape[0], lat_t, lat_h, lat_w, audio.shape[-1])
    plan = plan_time(lay, torch.tensor(sigma, dtype=torch.float32), (cfg.sigma_shift_video, cfg.sigma_shift_audio))
    vr = patchify_video(video.double()) @ d["video_patch_proj.weight"].T + d["video_patch_proj.bias"]
    ar = pack_audio(audio.double()) @ d["audio_patch_proj.weight"].T + d["audio_patch_proj.bias"]
    h = torch.cat([ctx, ar, vr])  # t2va order: text | audio | video
    row = torch.zeros(h.shape[0], dtype=torch.long)
    for a, b, r in plan.mod_segments:
        row[a:b] = r
    ang = rope_angles(lay.position_ids, sd["rope.inv_freq"], "cpu").double()
    t_emb = curve_lerp(sd["adaln_t_table"], plan.t_values).double()
    for i in range(cfg.layers):
        p = f"blocks.{i}"
        m = (t_emb @ d[p + ".adaln_proj.linear.weight"].T + d[p + ".adaln_proj.linear.bias"]).reshape(-1, 6, hid)[row]  # [tokens, 6, hid]
        sh1, sc1, g1, sh2, sc2, g2 = m.unbind(1)
        h = h + g1 * attn(rms(h, d[p + ".norm1.weight"], cfg.norm_eps) * (1 + sc1) + sh1, p, ang)
        h = h + g2 * mlp(rms(h, d[p + ".norm2.weight"], cfg.norm_eps) * (1 + sc2) + sh2, p)
    fa = (t_emb @ d["final_layer.adaln_proj.linear.weight"].T + d["final_layer.adaln_proj.linear.bias"])
    fsh, fsc = fa.chunk(2, -1)
    va, vb, vrow = plan.video_seg
    aa, ab, arow = plan.audio_seg
    fin = lambda seg_h, r, w, b: (rms(seg_h, d["final_layer.norm.weight"], cfg.final_norm_eps) * (1 + fsc[r]) + fsh[r]) @ d[w].T + d[b]  # noqa: E731
    v = fin(h[va:vb], vrow, "final_layer.video_out.weight", "final_layer.video_out.bias")
    a = fin(h[aa:ab], arow, "final_layer.audio_out.weight", "final_layer.audio_out.bias")
    return [-unpatchify_video(v, lat_t, lat_h // 2, lat_w // 2, 24), -unpack_audio(a)]


def test_forward_matches_independent_fp64_implementation():
    sd = make_sd(CFG, 3)
    m = make_model(CFG, sd)
    x, text = inputs(2)
    out = m.forward(x, 0.7, text)
    want = _naive(sd, CFG, x, 0.7, text)
    assert out[0].shape == x[0].shape and out[1].shape == x[1].shape
    assert rel(out[0], want[0]) < 5e-6 and rel(out[1], want[1]) < 5e-6  # measured 2.6e-7 (fp32 vs fp64)
    assert out[0].abs().max() > 0.05  # the comparison is not vacuous
    out2 = m.forward(x, 0.7, text)
    assert torch.equal(out[0], out2[0]) and torch.equal(out[1], out2[1])


def test_forward_is_sensitive_to_every_input():
    m = make_model()
    x, text = inputs(2)
    base = m.forward(x, 0.7, text)
    assert rel(m.forward(x, 0.71, text)[0], base[0]) > 1e-4
    assert rel(m.forward(x, 0.7, text + 0.1)[0], base[0]) > 1e-4
    xa = [x[0], x[1] + 0.1]
    assert rel(m.forward(xa, 0.7, text)[0], base[0]) > 1e-5  # audio reaches the video stream (joint attention)


def test_chunking_and_rope_impl_do_not_change_results():
    sd = make_sd(CFG, 4)
    x, text = inputs(5)
    base = make_model(CFG, sd).forward(x, 0.6, text)
    for kw in ({"mlp_chunk": 3}, {"attn_chunk": 7}, {"rope_chunk": 4}, {"mlp_chunk": 5, "attn_chunk": 11, "rope_chunk": 2},
               {"rope_impl": "eager"}):
        out = make_model(CFG, sd, **kw).forward(x, 0.6, text)
        assert rel(out[0], base[0]) < 1e-6 and rel(out[1], base[1]) < 1e-6, kw
    with pytest.raises(ValueError):
        make_model(CFG, sd, mlp_chunk=0)
    with pytest.raises(ValueError):
        make_model(CFG, sd, rope_impl="magic")


def test_bf16_stays_close_to_fp32():
    sd = make_sd(CFG, 6)
    x, text = inputs(7)
    ref32 = make_model(CFG, sd).forward(x, 0.5, text)
    xb = [t.bfloat16() for t in x]
    out = make_model(CFG, sd, dtype=torch.bfloat16).forward(xb, 0.5, text.bfloat16())
    assert out[0].dtype == torch.bfloat16 and out[1].dtype == torch.bfloat16
    assert rel(out[0].float(), ref32[0]) < 5e-2 and rel(out[1].float(), ref32[1]) < 5e-2


class Recorder(ResidentProvider):
    def __init__(self, blocks, fail_at=None):
        super().__init__(blocks)
        self.log, self.fail_at = [], fail_at

    def begin_forward(self):
        self.log.append("begin")

    def acquire(self, i):
        self.log.append(f"acquire{i}")
        if i == self.fail_at:
            raise RuntimeError("copy failed")
        return super().acquire(i)

    def release(self, i):
        self.log.append(f"release{i}")

    def end_forward(self):
        self.log.append("end")


def test_provider_protocol_order_and_abort_cleanup():
    sd = make_sd(CFG, 8)
    x, text = inputs(9)
    glob, blocks = weights_from_state_dict(sd, CFG, "cpu", torch.float32)
    rec = Recorder(blocks)
    m = H3Model(CFG, glob, rec, device="cpu", dtype=torch.float32)
    m.forward(x, 0.5, text)
    m.forward(x, 0.4, text)
    one = ["begin", "acquire0", "release0", "acquire1", "release1", "end"]
    assert rec.log == one + one
    bad = Recorder(blocks, fail_at=1)
    mb = H3Model(CFG, glob, bad, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="copy failed"):
        mb.forward(x, 0.5, text)
    assert bad.log == ["begin", "acquire0", "release0", "acquire1", "end"]  # end_forward always runs; acquire that failed is not released
    bad.fail_at = None
    assert rel(mb.forward(x, 0.5, text)[0], make_model(CFG, sd).forward(x, 0.5, text)[0]) < 1e-6  # usable after the abort


def test_conditioning_payloads_run_and_masks_zero_the_velocity():
    m = make_model()
    x, text = inputs(3)
    g = torch.Generator().manual_seed(4)
    kfs = [{"resolved_frame_index": 0, "latent": torch.randn(1, 24, 1, H, W, generator=g), "audio_latent": torch.randn(1, 32, 2, 3, generator=g)},
           {"resolved_frame_index": 9, "latent": torch.randn(1, 24, 2, H, W, generator=g)}]
    refs = [{"kind": "image", "latent_h": 4, "latent_w": 2, "latent": torch.randn(1, 24, 1, 4, 2, generator=g)},
            {"kind": "video_audio", "latent_t": 2, "latent_h": 2, "latent_w": 4, "ref_audio_t": 3, "latent": torch.randn(1, 24, 2, 2, 4, generator=g),
             "audio_latent": torch.randn(1, 32, 2, 3, generator=g)},
            {"kind": "audio", "ref_audio_t": 4, "audio_latent": torch.randn(1, 32, 2, 4, generator=g)}]
    payload = {"keyframes": kfs, "refs": refs, "seed": 5,
               "cond_video_latents": [kfs[0]["latent"], kfs[1]["latent"], refs[0]["latent"], refs[1]["latent"]],
               "cond_audio_latents": [kfs[0]["audio_latent"], refs[1]["audio_latent"], refs[2]["audio_latent"]],
               "text_token_tags": torch.tensor([[1, 1, 0, 0, 2]])}
    out = m.forward(x, 0.6, text, payload=payload)
    assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()
    assert rel(out[0], m.forward(x, 0.6, text)[0]) > 1e-4  # conditioning changes the result
    # a different seed changes the noise mixed into the conditions (aug < 1); aug = 1 makes the seed irrelevant
    assert rel(m.forward(x, 0.6, text, payload={**payload, "seed": 6})[0], out[0]) > 1e-6
    p1 = {**payload, "visual_cond_noise_aug": 1.0, "audio_cond_noise_aug": 1.0}
    assert torch.equal(m.forward(x, 0.6, text, payload={**p1, "seed": 1})[0], m.forward(x, 0.6, text, payload={**p1, "seed": 2})[0])
    # a prebuilt layout is honoured; a stale one (other signature) is ignored
    assert torch.equal(m.forward(x, 0.6, text, payload={**payload, "layout": PackedLayout.build(L, T, H, W, TA, kfs, refs)})[0], out[0])
    assert torch.equal(m.forward(x, 0.6, text, payload={**payload, "layout": PackedLayout.build(L + 1, T, H, W, TA)})[0], out[0])
    # denoise masks scale the velocity: zero where preserved, untouched where generated
    dm = torch.ones(1, 1, T, H, W)
    dm[:, :, 0] = 0.0
    adm = torch.ones(1, 1, 2, TA)
    adm[..., :1] = 0.0
    mv, ma = m.forward(x, 0.6, text, denoise_mask=dm, audio_denoise_mask=adm)
    assert not mv[:, :, 0].any() and not ma[..., 0].any() and mv[:, :, 1:].abs().sum() > 0


def test_audio_scale_carry_is_undone_and_redone():
    m = make_model()
    x, text = inputs(3)
    sigma, scale = torch.tensor(0.5, dtype=torch.float32), 1.25
    sa = time_shift_sigma(sigma, 12.0, 3.0)
    carry = sa / sigma
    inner = m.forward([x[0], x[1] * carry], 0.5, text)  # the raw network sees audio * carry
    out = m.forward(x, 0.5, text, payload={"audio_scale": scale})
    assert rel(out[0], inner[0]) < 1e-6  # the video stream is untouched by the carry
    want_a = (1.0 - scale) * (x[1] * carry) + (1.0 + (scale - 1.0) * sa) * inner[1]
    assert rel(out[1], want_a) < 1e-6


def test_odd_latent_dims_are_circularly_padded_then_cropped():
    m = make_model()
    x, text = inputs(3, t=2, h=3, w=5)
    out = m.forward(x, 0.5, text)
    assert out[0].shape == x[0].shape
    padded = [pad_to_patch(x[0]), x[1]]
    full = m.forward(padded, 0.5, text)
    assert torch.equal(out[0], full[0][:, :, :, :3, :5]) and torch.equal(out[1], full[1])


def test_text_paths_and_input_validation():
    m = make_model()
    x, text = inputs(3)
    refined = m.encode_text(text)
    assert refined.shape == (1, L, CFG.hidden) and torch.equal(m.encode_text(refined), refined)
    a = m.forward(x, 0.5, text)
    b = m.forward(x, 0.5, None, refined_text=refined)
    c = m.forward(x, 0.5, refined)
    assert torch.equal(a[0], b[0]) and torch.equal(a[0], c[0])
    bad = [
        lambda: m.forward(x, 0.5, None),
        lambda: m.forward(x, float("nan"), text),
        lambda: m.forward([x[0].repeat(2, 1, 1, 1, 1), x[1]], 0.5, text),
        lambda: m.forward([x[0], x[1][:, :, :1]], 0.5, text),
        lambda: m.forward(x[:1], 0.5, text),
        lambda: m.encode_text(torch.randn(1, L, 11)),
        lambda: m.encode_text(torch.randn(2, L, CFG.text_dim)),
        lambda: m.forward(x, 0.5, text, denoise_mask=torch.ones(1, 1, T, H, W + 2)),
        lambda: m.forward(x, 0.5, text, audio_denoise_mask=torch.ones(1, 1, 2, TA + 1)),
        lambda: m.forward(x, 0.5, text, payload={"text_token_tags": torch.tensor([[1, 0]])}),
        lambda: m.forward(x, 0.5, text, payload={"cond_video_latents": [torch.randn(1, 24, 1, H, W)]}),  # conditions without layout rows
        lambda: m.forward(x, 0.5, text, payload={"keyframes": [{"resolved_frame_index": 0, "latent": torch.randn(1, 24, 1, H, W)}]}),  # rows without latents
    ]
    for i, f in enumerate(bad):
        with pytest.raises(ValueError):
            f()


def test_head_bank_needs_schedule_and_reduces_to_single_head():
    sd1 = make_sd(CFG, 11)
    sd2 = dict(sd1)
    for name in ("video_out", "audio_out"):
        w, b = sd1[f"final_layer.{name}.weight"], sd1[f"final_layer.{name}.bias"]
        sd2[f"final_layer.{name}.weight"] = torch.cat([w, torch.zeros_like(w), torch.zeros_like(w)])  # offsets of zero
        sd2[f"final_layer.{name}.bias"] = torch.cat([b, torch.zeros_like(b), torch.zeros_like(b)])
    m1, m3 = make_model(CFG, sd1), make_model(CFG, sd2)
    assert m3.head_bank == 3 and m1.head_bank == 1
    x, text = inputs(3)
    with pytest.raises(ValueError):
        m3.forward(x, 0.6, text)
    sched = torch.tensor([1.0, 0.8, 0.6, 0.3, 0.0])
    for s in (0.8, 0.6, 0.3):
        a, b = m3.forward(x, s, text, sample_sigmas=sched), m1.forward(x, s, text)
        assert rel(a[0], b[0]) < 1e-6 and rel(a[1], b[1]) < 1e-6


def test_unsupported_configurations_fail_loudly():
    sd = make_sd(CFG)
    glob, blocks = weights_from_state_dict(sd, CFG, "cpu", torch.float32)
    glob.adaln_t_table = None
    with pytest.raises(NotImplementedError):
        H3Model(CFG, glob, ResidentProvider(blocks), device="cpu", dtype=torch.float32)
    glob, blocks = weights_from_state_dict(sd, CFG, "cpu", torch.float32)
    with pytest.raises(ValueError):
        H3Model(CFG, glob, ResidentProvider(blocks), device="cpu", dtype=torch.float16)
    with pytest.raises(ValueError):
        H3Model(dataclasses.replace(CFG, rope_inv_freq_len=6), glob, ResidentProvider(blocks), device="cpu", dtype=torch.float32)


# ------------------------------------------------------------------------------------------------ weights


def test_weights_from_state_dict_validation_and_prefix():
    sd = make_sd(CFG)
    glob, blocks = weights_from_state_dict({"model.dm." + k: v for k, v in sd.items()}, CFG, "cpu", torch.bfloat16, prefix="model.dm.")
    assert len(blocks) == CFG.layers and len(glob.refiner) == CFG.refiner_layers
    assert blocks[0].qkv.dtype == torch.bfloat16 and glob.video_patch_w.dtype == torch.float32 and glob.final_norm.dtype == torch.bfloat16
    assert blocks[0].adaln_w.shape == (18 * CFG.hidden, CFG.t_dim) and glob.refiner[0].adaln_w is None
    broken = dict(sd)
    del broken["blocks.1.mlp.fc2.weight"], broken["rope.inv_freq"]
    with pytest.raises(KeyError, match="2 tensors"):
        weights_from_state_dict(broken, CFG, "cpu", torch.float32)
    wrong = dict(sd)
    wrong["blocks.0.attn.qkv_proj.weight"] = torch.zeros(5, 5)
    with pytest.raises(ValueError, match="expected"):
        weights_from_state_dict(wrong, CFG, "cpu", torch.float32)


W4CFG = H3Config(hidden=256, layers=1, refiner_layers=1, heads=4, head_dim=64, ffn=256, text_dim=40, t_dim=8, curve_grid=17,
                 rope_inv_freq_len=4)


def _w4a8_entries(n, k, g, prefix):
    packed = torch.randint(-128, 128, (n, k // 2), generator=g, dtype=torch.int64).to(torch.int8)
    s_rel = (20 + 100 * torch.rand(n, k // 16, generator=g)).to(torch.float8_e4m3fn)
    conf = json.dumps({"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}).encode()
    return {prefix + ".weight": packed, prefix + ".weight_s_rel": s_rel, prefix + ".weight_s_channel": 0.0012 + 0.0004 * torch.rand(n, generator=g),
            prefix + ".weight_codebook": torch.linspace(-1, 1, 16) + 0.02 * torch.randn(16, generator=g),
            prefix + ".comfy_quant": torch.tensor(list(conf), dtype=torch.uint8)}


def test_w4a8_layers_load_and_run_through_qlinear():
    qlinear = pytest.importorskip("h3turbo.h3.qlinear")
    cfg = W4CFG
    sd = make_sd(cfg, 12)
    g = torch.Generator().manual_seed(13)
    for name, (n, k) in {"attn.qkv_proj": (3 * cfg.inner, cfg.hidden), "attn.out_proj": (cfg.hidden, cfg.inner),
                         "mlp.fc1": (2 * cfg.ffn, cfg.hidden), "mlp.fc2": (cfg.hidden, cfg.ffn)}.items():
        del sd[f"blocks.0.{name}.weight"]
        sd.update(_w4a8_entries(n, k, g, f"blocks.0.{name}"))
    glob, blocks = weights_from_state_dict(sd, cfg, "cpu", torch.float32)
    assert all(isinstance(getattr(blocks[0], f), W4A8Weight) for f in ("qkv", "out", "fc1", "fc2")) and blocks[0].qkv.convrot == 256
    with pytest.raises(ValueError, match="W4A8"):
        bad = dict(sd)
        bad["blocks.0.mlp.fc2.weight_s_rel"] = bad["blocks.0.mlp.fc2.weight_s_rel"][:, :-1]
        weights_from_state_dict(bad, cfg, "cpu", torch.float32)
    x, text = inputs(14, cfg=cfg)
    # the same network with dense weights = dequantised originals: a16 must agree to fp32 rounding, a8 to activation-quantisation noise
    dense = dict(sd)
    for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
        p = f"blocks.0.{name}"
        w = blocks[0]
        wq = {"attn.qkv_proj": w.qkv, "attn.out_proj": w.out, "mlp.fc1": w.fc1, "mlp.fc2": w.fc2}[name]
        for suffix in ("weight_s_rel", "weight_s_channel", "weight_codebook", "comfy_quant"):
            del dense[f"{p}.{suffix}"]
        dense[p + ".weight"] = qlinear.dequantize(wq, torch.float32)
    want = make_model(cfg, dense).forward(x, 0.6, text)
    a16 = H3Model(cfg, glob, ResidentProvider(blocks), device="cpu", dtype=torch.float32, precision="a16").forward(x, 0.6, text)
    a8 = H3Model(cfg, glob, ResidentProvider(blocks), device="cpu", dtype=torch.float32, precision="a8", backend="torch").forward(x, 0.6, text)
    a8c = H3Model(cfg, glob, ResidentProvider(blocks), device="cpu", dtype=torch.float32, precision="a8", backend="torch", mlp_chunk=3).forward(x, 0.6, text)
    assert rel(a16[0], want[0]) < 1e-5 and rel(a16[1], want[1]) < 1e-5  # measured 3e-7
    assert 1e-4 < rel(a8[0], want[0]) < 5e-2 and 1e-4 < rel(a8[1], want[1]) < 5e-2  # activation quantisation noise, measured ~0.4-0.6%
    assert rel(a8c[0], a8[0]) < 1e-6 and rel(a8c[1], a8[1]) < 1e-6  # token chunking is exact for the quantised path


def test_islands_are_bf16_rounded_by_default_and_fp32_on_request():
    sd = make_sd(CFG, 15)
    x, text = inputs(16)
    native = make_model(CFG, sd, dtype=torch.bfloat16)  # ComfyUI keeps the checkpoint's fp32 tensors in the compute dtype
    exact = make_model(CFG, sd, dtype=torch.bfloat16, fp32_islands=True)
    w = sd["video_patch_proj.weight"]
    assert torch.equal(native.vpw, w.bfloat16().float()) and torch.equal(exact.vpw, w)
    assert torch.equal(native.vob, sd["final_layer.video_out.bias"].bfloat16().float())
    xb = [t.bfloat16() for t in x]
    a, b = native.forward(xb, 0.6, text.bfloat16()), exact.forward(xb, 0.6, text.bfloat16())
    assert rel(a[0].float(), b[0].float()) > 1e-4  # different weights, bf16-level effect
    assert rel(a[0].float(), b[0].float()) < 5e-2
    f32 = make_model(CFG, sd, fp32_islands=False)  # fp32 compute: islands are fp32 either way
    assert torch.equal(f32.vpw, w) and torch.equal(f32.final_adaln_b, sd["final_layer.adaln_proj.linear.bias"])


def test_row_chunking_of_modulation_embedding_and_head_is_exact(monkeypatch):
    sd = make_sd(CFG, 17)
    x, text = inputs(18)
    dm = torch.ones(1, 1, T, H, W)
    dm[:, :, 0] = 0.0
    dm[:, :, 1, :2, :3] = 0.5  # per-token modulation rows
    adm = torch.ones(1, 1, 2, TA)
    adm[..., 1] = 0.3
    base = make_model(CFG, sd).forward(x, 0.6, text, denoise_mask=dm, audio_denoise_mask=adm)
    monkeypatch.setattr(model_mod, "_ROW_CHUNK", 3)
    small = make_model(CFG, sd).forward(x, 0.6, text, denoise_mask=dm, audio_denoise_mask=adm)
    tiny_head = make_model(CFG, sd, mlp_chunk=2).forward(x, 0.6, text, denoise_mask=dm, audio_denoise_mask=adm)
    for out in (small, tiny_head):
        assert rel(out[0], base[0]) < 1e-6 and rel(out[1], base[1]) < 1e-6


def test_option_validation():
    for kw in ({"attn_backend": "flash"}, {"backend": "cuda"}, {"precision": "a4"}, {"attn_chunk": -1}, {"rope_chunk": 0}):
        with pytest.raises(ValueError):
            make_model(**kw)
