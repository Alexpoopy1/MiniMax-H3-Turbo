import numpy as np
import pytest
import torch

from h3turbo.config import make_config
from h3turbo.io import build_modules
from h3turbo.training import EMA, PATTERNS, STRUCTURES, Geo, cosine_lr, flow_loss, sample_pins

GEO = Geo(lat_t=3, hp=3, wp=3, fps=8, lo_hp=2, lo_wp=2)


def _setup():
    torch.manual_seed(0)
    cfg = make_config("nano")
    tr, _, _, te = build_modules(cfg)
    for p in tr.parameters():  # leave zero-init so gradients are non-trivial
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.02)
    B = 6
    batch = {
        "prompts": ["a red square moving left", "blue square going up", "x", "", "green", "yellow square moving down"],
        "video": torch.randn(B, 27, 96),
        "audio": torch.randn(B, 45, 32),
        "ref": torch.randn(B, 9, 96),
        "video_lo": torch.randn(B, 12, 96),
    }
    return tr, te, batch


@pytest.mark.parametrize("structure", STRUCTURES)
def test_every_structure_gives_a_finite_loss_and_gradients(structure):
    tr, te, batch = _setup()
    rng, gen = np.random.default_rng(0), torch.Generator().manual_seed(0)
    loss, stats = flow_loss(tr, te, batch, structure, GEO, rng, gen)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tr.blocks.parameters())
    assert te.embed.weight.grad is not None  # the text encoder trains through the transformer


def test_pinned_modality_is_excluded_from_the_loss():
    tr, te, batch = _setup()
    rng, gen = np.random.default_rng(0), torch.Generator().manual_seed(0)
    _, stats = flow_loss(tr, te, batch, "full", GEO, rng, gen, patterns={"video_full": 1.0})
    assert set(stats["pattern"]) == {"video_full"}
    assert float(stats["video"].abs().sum()) == 0.0  # all video tokens pinned -> nothing to score
    assert float(stats["audio"].abs().sum()) > 0.0


def test_text_is_dropped_far_more_often_when_the_other_modality_is_pinned():
    """Regression: with the prompt (which names the direction) almost always present, the model
    never learned to read direction from a pinned video; V2A direction sat at chance."""
    tr, te, batch = _setup()
    seen = []
    orig = te.tokenizer.batch
    te.tokenizer.batch = lambda prompts: (seen.extend(prompts), orig(prompts))[1]
    rng, gen = np.random.default_rng(0), torch.Generator().manual_seed(0)
    for _ in range(40):
        flow_loss(tr, te, batch, "full", GEO, rng, gen, patterns={"video_full": 1.0}, text_drop=0.0, cross_modal_text_drop=0.7)
    frac_empty_pinned = np.mean([p == "" for p in seen])
    seen.clear()
    for _ in range(40):
        flow_loss(tr, te, batch, "full", GEO, rng, gen, patterns={"none": 1.0}, text_drop=0.0, cross_modal_text_drop=0.7)
    empty_when_prompt_is_only_source = sum(p == "" for p in seen) - 40 * 1  # batch already has one "" prompt
    assert frac_empty_pinned > 0.6
    assert empty_when_prompt_is_only_source == 0


def test_pinned_noise_is_used_verbatim_for_reflow_pairs():
    tr, te, batch = _setup()
    B = 6
    eps_v, eps_a = torch.randn(B, 27, 96), torch.randn(B, 45, 32)
    gens = [torch.Generator().manual_seed(s) for s in (1, 2)]
    losses = []
    for g in gens:  # different generators, same supplied noise and sigma stream -> same loss
        rng = np.random.default_rng(0)
        g.manual_seed(5)  # sigma is drawn from `gen`; noise is not
        l, _ = flow_loss(tr, te, {**batch, "eps_video": eps_v, "eps_audio": eps_a}, "video", GEO, rng, g)
        losses.append(l.item())
    assert losses[0] == losses[1]


def test_sample_pins_patterns_have_the_right_shape():
    rng = np.random.default_rng(0)
    pv, pa, names = sample_pins(200, GEO, 45, rng)
    assert pv.shape == (200, 27) and pa.shape == (200, 45)
    for i, n in enumerate(names):
        if n == "video_full":
            assert pv[i].all() and not pa[i].any()
        if n == "audio_full":
            assert pa[i].all() and not pv[i].any()
        if n == "first_last":
            f = pv[i].view(3, 9)
            assert f[0].all() and f[2].all() and not f[1].any()
    assert {n for n in names} == set(PATTERNS)


def test_ema_tracks_and_lr_schedule_shape():
    p = torch.nn.Parameter(torch.zeros(3))
    ema = EMA([p], decay=0.9)
    for _ in range(200):
        p.data.fill_(1.0)
        ema.update()
    assert torch.allclose(ema.shadow[0], torch.ones(3), atol=1e-3)
    ema.copy_to()
    lrs = [cosine_lr(i, 1000, 1e-3, warmup=100) for i in range(1000)]
    assert lrs[0] < lrs[99] and lrs[100] == pytest.approx(1e-3, rel=0.01) and lrs[-1] < 0.2e-3
