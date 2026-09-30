"""H3 packed layout, timestep/modulation plan, mask helpers and the q/k norm+rope ops (pure torch, CPU).

Positions and AdaLN rows are compared with hand-derived values here; the bit-level proof against ComfyUI's own
PackedLayout / model is scripts/h3_check_model_vs_comfy.py (random-signature fuzz + 18 scenarios).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from h3turbo.h3.layout import (FRAME_RESCALE, PackedLayout, bank_head_weights, curve_lerp, head_bank_range, mask_row_values, norm_rope_, pack_audio,
                               pad_to_patch, patchify_video, plan_time, rms_norm_fused_order, rope_angles, time_shift_sigma,
                               token_grid_masks, unpack_audio, unpatchify_video)

T, H, W, TA, L = 2, 4, 6, 4, 5


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


# ------------------------------------------------------------------------------------------------ layout


def test_time_shift_sigma_inverts_and_keeps_dtype():
    s = torch.tensor([0.05, 0.3, 0.8, 1.0])
    back = time_shift_sigma(time_shift_sigma(s, 12.0, 3.0), 3.0, 12.0)
    assert torch.allclose(back, s, atol=1e-6)
    assert torch.allclose(time_shift_sigma(s, 5.0, 5.0), s, atol=1e-6)
    assert time_shift_sigma(s, 12.0, 3.0).dtype == torch.float32
    assert time_shift_sigma(1.0, 12.0, 3.0) == pytest.approx(1.0)


def test_patchify_roundtrip_and_row_order():
    x = torch.randn(1, 24, 3, 6, 8)
    rows = patchify_video(x)
    assert rows.shape == (3 * 3 * 4, 96)
    assert torch.equal(unpatchify_video(rows, 3, 3, 4, 24), x)
    # row (t=1, h=2, w=3) holds the 2x2 pixel block at h 4..5, w 6..7, features ordered (c, pt, ph, pw)
    row = rows[(1 * 3 + 2) * 4 + 3].reshape(24, 1, 2, 2)
    assert torch.equal(row, x[0, :, 1:2, 4:6, 6:8])
    with pytest.raises(ValueError):
        patchify_video(torch.randn(1, 24, 1, 5, 6))


def test_pad_to_patch_is_circular_and_only_pads_when_needed():
    x = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(1, 1, 2, 3, 5)
    y = pad_to_patch(x)
    assert y.shape == (1, 1, 2, 4, 6)
    assert torch.equal(y[..., :3, :5], x) and torch.equal(y[..., 3, :5], x[..., 0, :]) and torch.equal(y[..., 5], y[..., 0])
    assert pad_to_patch(y) is y or torch.equal(pad_to_patch(y), y)


def test_audio_pack_is_channel_major_and_invertible():
    a = torch.randn(1, 32, 2, 5)
    rows = pack_audio(a)
    assert rows.shape == (10, 32)
    assert torch.equal(rows[:5], a[0, :, 0].T) and torch.equal(rows[5:], a[0, :, 1].T)
    assert torch.equal(unpack_audio(rows), a)


def test_layout_t2va_positions():
    lay = PackedLayout.build(L, 6, 4, 8, 3)
    frame_rows = 2 * 4
    assert [tuple(s) for s in lay.segments] == [(0, L, "text"), (L, L + 6, "audio"), (L + 6, L + 6 + 6 * frame_rows, "video")]
    assert lay.seq_len == L + 6 + 6 * frame_rows and lay.position_ids.dtype == torch.float64
    pos = lay.position_ids
    assert torch.equal(pos[:L, 0], torch.arange(L, dtype=torch.float64)) and not pos[:L, 1:].any()
    origin = float(L)
    a = pos[L:L + 6]
    assert torch.allclose(a[:, 0], origin + torch.tensor([0, 1, 2, 0, 1, 2.0], dtype=torch.float64))
    assert not a[:, 1].any()
    area = math.sqrt(4 * 8)
    w_axis = [(i * (8 / area / 4) + (1 - 8 / area) / 2) * 32 for i in range(4)]
    assert a[0, 2].item() == pytest.approx(w_axis[0]) and a[3, 2].item() == pytest.approx(w_axis[-1])
    v = pos[L + 6:].reshape(6, frame_rows, 3)
    spans = [FRAME_RESCALE * k for k in (1, 4, 4, 4, 4, 1)]
    starts = [origin + sum(spans[:i]) for i in range(6)]
    for f in range(6):
        assert torch.allclose(v[f, :, 0], torch.full((frame_rows,), starts[f], dtype=torch.float64))
    h_axis = [(i * (4 / area / 2) + (1 - 4 / area) / 2) * 32 for i in range(2)]
    assert v[0, 0, 1].item() == pytest.approx(h_axis[0]) and v[0, -1, 1].item() == pytest.approx(h_axis[1])
    assert v[0, 0, 2].item() == pytest.approx(w_axis[0]) and v[0, -1, 2].item() == pytest.approx(w_axis[-1])
    assert lay.img_update.all() and lay.audio_update.all()


def _full_layout():
    kfs = [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, H, W), "audio_latent": torch.zeros(1, 32, 2, 3)},
           {"resolved_frame_index": 9, "latent": torch.zeros(1, 24, 2, H, W)}]
    refs = [{"kind": "image", "latent_h": 4, "latent_w": 2, "latent": torch.zeros(1, 24, 1, 4, 2)},
            {"kind": "video_audio", "latent_t": 2, "latent_h": 2, "latent_w": 4, "ref_audio_t": 3},
            {"kind": "audio", "ref_audio_t": 4},
            {"kind": "video", "latent_t": 1, "latent_h": 2, "latent_w": 2, "ref_audio_t": 0}]
    return PackedLayout.build(L, T, H, W, TA, keyframes=kfs, refs=refs), kfs, refs


def test_layout_with_keyframes_and_refs_tiles_the_sequence():
    lay, kfs, refs = _full_layout()
    kinds = [s.kind for s in lay.segments]
    assert kinds == ["text", "cond", "cond_audio", "cond", "ref_img", "ref_audio", "ref_img", "ref_audio", "ref_img", "audio", "video"]
    assert lay.segments[0].start == 0 and lay.segments[-1].stop == lay.seq_len
    assert all(a.stop == b.start for a, b in zip(lay.segments, lay.segments[1:]))
    rows = torch.cat([torch.arange(L), lay.img_pos, lay.audio_pos]).sort().values
    assert torch.equal(rows, torch.arange(lay.seq_len))  # every row is text, image-like or audio-like, exactly once
    assert lay.img_update.sum() == T * (H // 2) * (W // 2) and lay.audio_update.sum() == 2 * TA
    assert not lay.img_update[:-T * (H // 2) * (W // 2)].any()  # only the target video rows are generated
    # target origin = text + image span 1 + video_audio max(3, 5/3*(1+4)) + audio 4 + video max(0, 5/3)
    origin = L + 1.0 + max(3.0, FRAME_RESCALE * 5) + 4.0 + FRAME_RESCALE
    tgt = lay.position_ids[lay.span("audio").start:lay.span("audio").stop]
    assert tgt[0, 0].item() == pytest.approx(origin)
    kf_pos = lay.position_ids[lay.segments[1].start:lay.segments[1].stop]
    assert kf_pos[0, 0].item() == pytest.approx(origin + 0.0)
    last_kf = lay.position_ids[lay.segments[3].start:lay.segments[3].stop]
    assert last_kf[0, 0].item() == pytest.approx(origin + FRAME_RESCALE * 9)
    ref_img = lay.position_ids[lay.segments[4].start:lay.segments[4].stop]
    assert ref_img[:, 0].eq(L).all()  # first reference block starts at the text end


def test_layout_rejects_bad_input():
    with pytest.raises(ValueError):
        PackedLayout.build(L, 2, 5, 6, 3)
    with pytest.raises(ValueError):
        PackedLayout.build(L, 0, 4, 6, 3)
    with pytest.raises(ValueError):
        PackedLayout.build(L, 2, 4, 6, 3, refs=[{"kind": "hologram"}])


# ------------------------------------------------------------------------------------------------ plan


def _sigma(v):
    return torch.tensor(v, dtype=torch.float32)


def test_plan_time_rows_tags_and_pins():
    lay, _, _ = _full_layout()
    plan = plan_time(lay, _sigma(0.6), (12.0, 3.0))
    tv = float(1 - _sigma(0.6))
    ta = float(1 - time_shift_sigma(_sigma(0.6), 12.0, 3.0))
    assert plan.t_values == sorted(plan.t_values) and tv in plan.t_values and ta in plan.t_values and 0.999 in plan.t_values
    assert 1.0 in plan.t_values  # audio conditioning pins at 1.0
    rows = {s.kind: r for s, (_, _, r) in zip(lay.segments, plan.mod_segments)}
    tr = {t: i for i, t in enumerate(plan.t_values)}
    assert rows["text"] == tr[tv] * 3 + 1 and rows["video"] == tr[tv] * 3 + 0 and rows["audio"] == tr[ta] * 3 + 2
    assert rows["cond"] == tr[0.999] * 3 + 0 and rows["ref_img"] == tr[0.999] * 3 and rows["cond_audio"] == tr[1.0] * 3 + 2
    assert plan.video_seg[2] == tr[tv] and plan.audio_seg[2] == tr[ta]
    assert [(a, b) for a, b, _ in plan.mod_segments] == [(s.start, s.stop) for s in lay.segments]


def test_plan_time_conditioning_labels_follow_the_clock():
    lay, _, _ = _full_layout()
    plan = plan_time(lay, _sigma(0.0005), (12.0, 3.0), visual_aug=0.5)
    assert plan.t_video == pytest.approx(0.9995, abs=1e-6) and plan.t_audio < 1.0
    assert 0.5 not in plan.t_values and max(plan.t_values) == 1.0  # an image label below the clock is lifted to it; audio pins at 1


def test_plan_time_text_tag_runs():
    lay = PackedLayout.build(7, T, H, W, TA)
    plan = plan_time(lay, _sigma(0.5), (12.0, 3.0), text_tags=torch.tensor([[1, 1, 0, 0, 0, 2, 1]]))
    text = [seg for seg in plan.mod_segments if seg[1] <= 7]
    assert [(a, b) for a, b, _ in text] == [(0, 2), (2, 5), (5, 6), (6, 7)]
    base = plan.mod_segments[0][2] - 1
    assert [r - base for _, _, r in text] == [1, 0, 2, 1]
    for bad in (torch.tensor([[1, 1, 0]]), torch.tensor([[1, 1, 0, 0, 0, 2, 3]])):
        with pytest.raises(ValueError):
            plan_time(lay, _sigma(0.5), (12.0, 3.0), text_tags=bad)


def test_plan_time_denoise_masks():
    lay = PackedLayout.build(L, T, H, W, TA)
    sig = _sigma(0.8)
    m = torch.ones(1, 1, T, H, W)
    m[:, :, 0] = 0.0
    m[:, :, 1, :2, :2] = 0.5
    plan = plan_time(lay, sig, (12.0, 3.0), denoise_mask=m)
    va, vb, rows = plan.video_seg
    assert isinstance(rows, torch.Tensor) and rows.shape == (vb - va,)
    levels = sorted(plan.t_values[i] for i in rows.unique().tolist())
    assert levels[0] == pytest.approx(0.2, abs=1e-6)  # unmasked rows: 1 - sigma
    assert levels[1] == pytest.approx(1 - 0.5 * 0.8, abs=1e-6)  # mask value m puts a row at sigma = m * sigma
    assert levels[-1] == pytest.approx(0.999, abs=1e-6)  # fully preserved rows are pinned at the conditioning label
    per = (H // 2) * (W // 2)  # frame 0 (mask 0): pinned label; frame 1: first patch at 0.5, the rest generating
    assert plan.t_values[rows[0].item()] == pytest.approx(0.999, abs=1e-6)
    assert plan.t_values[rows[per].item()] == pytest.approx(0.6, abs=1e-6) and plan.t_values[rows[per + 5].item()] == pytest.approx(0.2, abs=1e-6)
    mod_video = [r for a, b, r in plan.mod_segments if a == va][0]
    assert isinstance(mod_video, torch.Tensor) and (mod_video % 3 == 0).all()
    # a uniform mask collapses to one row with the modified label
    plan_u = plan_time(lay, sig, (12.0, 3.0), denoise_mask=torch.full((1, 1, T, H, W), 0.4))
    assert isinstance(plan_u.video_seg[2], int) and plan_u.t_values[plan_u.video_seg[2]] == pytest.approx(1 - 0.4 * 0.8, abs=1e-6)
    # audio mask: per-frame rows, both channels alike
    am = torch.ones(1, 1, 2, TA)
    am[..., :2] = 0.0
    plan_a = plan_time(lay, sig, (12.0, 3.0), audio_denoise_mask=am)
    assert plan_a.audio_seg[2].shape == (2 * TA,) and (plan_a.audio_seg[2][:2] != plan_a.audio_seg[2][2]).all()


def test_mask_helpers():
    m = torch.ones(3, 5, 7)
    assert mask_row_values(m, 3, 6, 8) is None
    m[1, 3, 2] = 0.25  # one preserved pixel next to generating ones: amax pooling keeps the patch generating
    assert mask_row_values(m, 3, 6, 8) is None
    m[1, 2:4, 2:4] = 0.0  # a fully preserved 2x2 patch (frame 1, patch row 1, patch col 1)
    v = mask_row_values(m, 3, 6, 8)
    assert v.shape == (3 * 3 * 4,) and v[(1 * 3 + 1) * 4 + 1] == 0 and v.sum() == v.numel() - 1
    assert mask_row_values(torch.zeros(3, 5, 7), 3, 6, 8).eq(0).all()
    assert mask_row_values(torch.full((3, 5, 7), 0.7), 3, 6, 8) is not None  # anything visibly below 1 counts as masked
    assert mask_row_values(torch.full((3, 5, 7), 0.9995), 3, 6, 8) is None  # within 1e-3 of 1 does not
    vm = torch.rand(1, 1, 2, 5, 7)
    am = torch.rand(1, 32, 2, 3)
    gv, ga = token_grid_masks(vm, am)
    assert gv.shape == vm.shape and ga.shape == am.shape
    assert torch.equal(gv, torch.ceil(gv * 256) / 256) and (gv >= vm).all()  # rounded up, patch-max pooled
    assert torch.equal(ga[:, 0], ga[:, 5]) and (ga >= am).all()


def test_curve_lerp():
    table = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)
    out = curve_lerp(table, [0.0, 1.0, 0.5, 0.125, 2.0, -1.0])
    assert torch.equal(out[0], table[0]) and torch.equal(out[1], table[4]) and torch.equal(out[2], table[2])
    assert torch.allclose(out[3], 0.5 * (table[0] + table[1])) and torch.equal(out[4], table[4]) and torch.equal(out[5], table[0])


# ------------------------------------------------------------------------------------------------ rope ops


def test_norm_rope_impls_and_geometry():
    torch.manual_seed(0)
    s, heads, hd, half = 37, 3, 64, 12
    pos = torch.rand(s, 3, dtype=torch.float64) * 20
    ang = rope_angles(pos, 10.0 ** (-torch.arange(4, dtype=torch.float32) / 4), "cpu")
    assert ang.shape == (s, half)
    w = 1 + 0.2 * torch.randn(hd)
    x0 = torch.randn(s, heads, hd)

    def naive(x):  # fp64: rmsnorm, then rotate channel j with channel j + half inside the first 2*half dims
        x = x.double()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w.double()
        c, sn = torch.cos(ang.double())[:, None], torch.sin(ang.double())[:, None]
        a, b = x[..., :half], x[..., half:2 * half]
        return torch.cat([a * c - b * sn, a * sn + b * c, x[..., 2 * half:]], -1)

    for impl in ("torch", "eager"):
        q, k = x0.clone(), x0.clone().flip(0)
        norm_rope_(q, k, w, w, 1e-5, torch.cos(ang), torch.sin(ang), impl=impl, chunk=10)
        assert rel(q, naive(x0)) < 1e-6 and rel(k, naive(x0.flip(0))) < 1e-6
    # bf16: the kernel-order norm stays within one bf16 ulp of F.rms_norm and is deterministic; impls agree to bf16 noise
    xb = (3 * torch.randn(s, heads, hd)).bfloat16()
    a, b = rms_norm_fused_order(xb, w.bfloat16(), 1e-5), F.rms_norm(xb, (hd,), w.bfloat16(), 1e-5)
    assert rel(a, b) < 8e-3 and torch.equal(a, rms_norm_fused_order(xb, w.bfloat16(), 1e-5))
    outs = []
    for impl in ("torch", "eager"):
        q, k = xb.clone(), xb.clone()
        norm_rope_(q, k, w.bfloat16(), w.bfloat16(), 1e-5, torch.cos(ang).bfloat16(), torch.sin(ang).bfloat16(), impl=impl)
        outs.append(q)
    assert rel(outs[0], outs[1]) < 3e-2 and rel(outs[0], naive(xb.float())) < 3e-2


def test_head_bank_range_and_weights():
    sched = torch.tensor([1.0, 0.8, 0.6, 0.3, 0.0])
    for sigma, want in ((1.0, (0, 2)), (0.8, (2, 3)), (0.3, (2, 3))):  # start/stop = round((1 - unshifted sigma) * n)
        assert head_bank_range(sched, torch.tensor(sigma), 12.0, 3) == want
    assert head_bank_range(sched, torch.tensor(0.0), 12.0, 3) == (2, 3)  # last sigma: no next step, one head at least
    w, b = torch.randn(6, 5, dtype=torch.float64), torch.randn(6, dtype=torch.float64)  # 3 heads of 2 outputs
    we, be = bank_head_weights(w, b, 3, 0, 1, 12.0)  # one head spanned: the base head itself
    assert torch.equal(we, w[:2]) and torch.equal(be, b[:2])
    we, be = bank_head_weights(w, b, 3, 1, 3, 12.0)  # heads 1 and 2: base + dt-weighted offsets, weights sum to one
    grid = torch.linspace(1.0, 0.0, 4, dtype=torch.float64)
    dt = (1.0 - 12.0 * grid / (1.0 + 11.0 * grid)).diff()[1:3]
    wt = dt / dt.sum()
    assert torch.allclose(we, w[:2] + wt[0] * w[2:4] + wt[1] * w[4:6]) and torch.allclose(be, b[:2] + wt[0] * b[2:4] + wt[1] * b[4:6])
