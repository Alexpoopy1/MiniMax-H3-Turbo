import pytest
import torch

from h3turbo.config import H3TurboConfig, ModalitySpec, make_config, tier_names
from h3turbo.model import H3TurboTransformer, Segment


def _randomised(cfg):
    torch.manual_seed(0)
    m = H3TurboTransformer(cfg)
    for p in m.parameters():  # undo the zero-init so paths are exercised
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.02)
    return m


def _segments(cfg, B=2):
    def seg(mod, n, dim, gi, emit):
        return Segment(mod, torch.randn(B, n, dim), torch.rand(1, n, 3) * 5, torch.full((B, n), gi), emit)

    segs = [
        seg("text", 7, cfg.text_dim, 0, False),
        seg("video", 12, cfg.video_in_dim, 1, True),
        seg("video", 4, cfg.video_in_dim, 2, False),
        seg("audio", 9, cfg.audio_in_dim, 3, True),
    ]
    groups = ["text", "video", "video", "audio", "audio"]
    gt = torch.tensor([[0, 700.0, 0, 700.0, 0], [0, 300.0, 0, 300.0, 0]])[:B]
    return segs, groups, gt


def test_all_tiers_are_valid_and_ordered():
    sizes = []
    for t in tier_names():
        cfg = make_config(t)
        cfg.transformer.validate()
        with torch.device("meta"):
            sizes.append(H3TurboTransformer(cfg.transformer).num_params())
    assert sizes == sorted(sizes)


def test_config_roundtrip():
    cfg = make_config("small")
    cfg.transformer.extra_modalities.append(ModalitySpec("depth", 16, 16))
    assert H3TurboConfig.from_json(cfg.to_json()) == cfg


def test_output_shapes_and_emit():
    cfg = make_config("nano").transformer
    m = _randomised(cfg)
    segs, groups, gt = _segments(cfg)
    outs = m(segs, groups, gt)
    assert [o is None for o in outs] == [True, False, True, False]
    assert outs[1].shape == (2, 12, cfg.video_in_dim) and outs[3].shape == (2, 9, cfg.audio_in_dim)


def test_zero_init_outputs_zero():
    cfg = make_config("nano").transformer
    m = H3TurboTransformer(cfg)
    segs, groups, gt = _segments(cfg)
    outs = m(segs, groups, gt)
    assert outs[1].abs().max() == 0


def test_cached_modulation_matches_direct():
    cfg = make_config("nano").transformer
    m = _randomised(cfg).eval()
    segs, groups, gt = _segments(cfg)
    direct = m(segs, groups, gt)
    cached = m(segs, groups, gt, mods=m.compute_mods(groups, gt))
    for a, b in zip(direct, cached):
        if a is not None:
            assert torch.equal(a, b)


def test_adaln_bank_can_live_elsewhere():
    """Inference layout: bank in fp32 on its own device, blocks in compute dtype."""
    cfg = make_config("nano").transformer
    m = _randomised(cfg).eval()
    segs, groups, gt = _segments(cfg)
    ref = m(segs, groups, gt)
    m.to_inference("cpu", torch.bfloat16, adaln_device="cpu")
    assert next(m.adaln.parameters()).dtype == torch.float32
    assert m.blocks[0].attn.qkv.weight.dtype == torch.bfloat16
    segs16 = [s._replace(x=s.x.bfloat16()) for s in segs]
    out = m(segs16, groups, gt)
    for a, b in zip(ref, out):
        if a is not None:
            assert (a - b.float()).abs().max() < 0.1 * a.abs().max().clamp(min=1e-3) + 0.05


def test_group_changes_output_for_same_tokens():
    """Clean-vs-noisy group must change what the model does with identical tokens."""
    cfg = make_config("nano").transformer
    m = _randomised(cfg).eval()
    segs, groups, gt = _segments(cfg)
    a = m(segs, groups, gt)[1]
    segs2 = [s if i != 1 else s._replace(group=torch.full_like(s.group, 2)) for i, s in enumerate(segs)]
    b = m(segs2, groups, gt)[1]
    assert not torch.allclose(a, b)


def test_add_modality_trains_alone():
    cfg = make_config("nano").transformer
    m = _randomised(cfg)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    m.add_modality(ModalitySpec("depth", 16, 16))
    trainable = m.freeze_shared()
    assert trainable and all("depth" in n for n in trainable)
    segs, groups, gt = _segments(cfg)
    depth = Segment("depth", torch.randn(2, 5, 16), torch.rand(1, 5, 3), torch.full((2, 5), 4), True)
    groups = groups[:4] + ["depth"]
    gt = torch.cat([gt[:, :4], torch.zeros(2, 1)], 1)
    outs = m(segs + [depth], groups, gt)
    outs[-1].pow(2).mean().backward()
    assert all(p.grad is None for n, p in m.named_parameters() if n in before and not p.requires_grad)
    assert all(m.get_parameter(n).grad is not None for n in trainable)
    for n, p in before.items():
        assert torch.equal(p, m.get_parameter(n))


def test_context_resampler_fixed_budget():
    cfg = make_config("nano", resampler_queries=8).transformer
    m = _randomised(cfg)
    ctx = Segment("video", torch.randn(2, 50, cfg.video_in_dim), torch.rand(1, 50, 3), torch.full((2, 50), 2))
    r = m.resample(ctx, torch.full((2, 8), 2), torch.rand(1, 8, 3))
    assert r.x.shape == (2, 8, cfg.hidden) and r.embedded
    segs, groups, gt = _segments(cfg)
    outs = m(segs + [r], groups, gt)
    assert torch.isfinite(outs[1]).all()


def test_gradient_checkpointing_matches():
    cfg = make_config("nano").transformer
    m = _randomised(cfg).train()
    segs, groups, gt = _segments(cfg)
    def grads():
        m.zero_grad()
        m(segs, groups, gt)[1].pow(2).mean().backward()
        return torch.cat([p.grad.flatten() for p in m.blocks.parameters()])
    g0 = grads()
    m.gradient_checkpointing = True
    g1 = grads()
    assert torch.allclose(g0, g1, atol=1e-5)
