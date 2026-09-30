"""H3Engine glue, sigma schedule, bench shapes and CLI dispatch on a tiny synthetic h3t (CPU)."""
import pytest
import torch

from h3turbo.h3 import convert as cv
from h3turbo.h3.bench import latent_shape
from h3turbo.h3.config import H3Config
from h3turbo.h3.engine import H3Engine, h3_sigmas
from h3turbo.h3.layout import time_shift_sigma

CFG = H3Config(hidden=64, layers=2, refiner_layers=1, heads=2, head_dim=32, ffn=48, text_dim=40, t_dim=8, curve_grid=17,
               rope_inv_freq_len=4)
VIDEO, AUDIO_LEN, TEXT_LEN = (2, 4, 6), 4, 5


@pytest.fixture(scope="module")
def h3t(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("h3") / "tiny.h3t")
    cv.convert(cv.synthetic_state_dict(CFG, quant=False, seed=3), path, cfg=CFG, verify=False)
    return path


@pytest.fixture()
def engine(h3t):
    with H3Engine.from_h3t(h3t, "cpu", resident=1, pin=False) as eng:
        yield eng


def text(seed=0):
    return torch.randn(1, TEXT_LEN, CFG.text_dim, generator=torch.Generator().manual_seed(seed))


def test_sigmas_endpoints_and_monotone():
    s = h3_sigmas(8, 12.0)
    assert s.shape == (9,) and s[0] == 1.0 and s[-1] == 0.0
    assert bool((s[1:] < s[:-1]).all())
    # shift 1 is the plain linear schedule; a bigger shift keeps sigma higher for longer
    assert torch.allclose(h3_sigmas(4, 1.0), torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
    assert bool((h3_sigmas(8, 12.0)[1:-1] > h3_sigmas(8, 1.0)[1:-1]).all())
    with pytest.raises(ValueError):
        h3_sigmas(0, 12.0)


def test_latent_shape():
    assert latent_shape(512, 320, 4.0) == (24, 20, 32, 160)  # 96 frames: 1 + 95 // 4 latent frames; 16x spatial; 40 audio frames/s
    assert latent_shape(32, 32, 0.0)[0] == 1  # a single image frame
    with pytest.raises(ValueError):
        latent_shape(500, 320, 1.0)


def test_velocity_shapes_and_finite(engine):
    g = torch.Generator().manual_seed(1)
    xv = torch.randn(1, CFG.video_channels, *VIDEO, generator=g)
    xa = torch.randn(1, CFG.audio_channels, 2, AUDIO_LEN, generator=g)
    vv, va = engine.velocity([xv.bfloat16(), xa.bfloat16()], 0.7, text().bfloat16())
    assert vv.shape == xv.shape and va.shape == xa.shape
    assert bool(torch.isfinite(vv.float()).all() and torch.isfinite(va.float()).all())


def test_refined_text_matches_raw_text(engine):
    g = torch.Generator().manual_seed(2)
    x = [torch.randn(1, CFG.video_channels, *VIDEO, generator=g).bfloat16(), torch.randn(1, CFG.audio_channels, 2, AUDIO_LEN, generator=g).bfloat16()]
    st = text().bfloat16()
    a = engine.velocity(x, 0.5, st)
    b = engine.velocity(x, 0.5, refined_text=engine.encode_text(st))
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_sample_is_deterministic_and_seeded(engine):
    a = engine.sample(VIDEO, AUDIO_LEN, text(), steps=2, seed=5)
    b = engine.sample(VIDEO, AUDIO_LEN, text(), steps=2, seed=5)
    c = engine.sample(VIDEO, AUDIO_LEN, text(), steps=2, seed=6)
    assert a[0].shape == (1, CFG.video_channels, *VIDEO) and a[1].shape == (1, CFG.audio_channels, 2, AUDIO_LEN)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[0], c[0])
    assert all(bool(torch.isfinite(x).all()) for x in a)


def test_sampler_integrates_both_streams_to_sigma_zero(engine, monkeypatch):
    """With a constant velocity of +1 both streams must move by exactly -1 (sigma 1 -> 0), whatever their own schedules do."""
    calls = []

    def fake(x, sigma, text_states, **kw):
        calls.append(float(sigma))
        return [torch.ones_like(x[0]), torch.ones_like(x[1])]

    monkeypatch.setattr(engine.model, "forward", fake)
    g = torch.Generator().manual_seed(9)
    xv0 = torch.randn(1, CFG.video_channels, *VIDEO, generator=g)
    xa0 = torch.randn(1, CFG.audio_channels, 2, AUDIO_LEN, generator=g)
    seen = []
    xv, xa = engine.sample(VIDEO, AUDIO_LEN, text(), steps=6, video_noise=xv0, audio_noise=xa0,
                           callback=lambda i, v, a: seen.append((v.clone(), a.clone())))
    assert torch.allclose(xv, xv0 - 1.0, atol=1e-5) and torch.allclose(xa, xa0 - 1.0, atol=1e-5)
    sv = h3_sigmas(6, CFG.sigma_shift_video)
    assert calls == pytest.approx([float(s) for s in sv[:-1]])
    # the audio clock is the video clock re-expressed on the audio shift, and each stream steps on its OWN clock
    sa = time_shift_sigma(sv, CFG.sigma_shift_video, CFG.sigma_shift_audio)
    assert sa[0] == pytest.approx(1.0) and sa[-1] == pytest.approx(0.0)
    pv, pa = xv0, xa0
    for i, (v, a) in enumerate(seen):
        assert torch.allclose(v - pv, torch.full_like(v, float(sv[i + 1] - sv[i])), atol=1e-5)
        assert torch.allclose(a - pa, torch.full_like(a, float(sa[i + 1] - sa[i])), atol=1e-5)
        pv, pa = v, a
    assert not torch.allclose(sa[1:-1], sv[1:-1])  # the two clocks really differ, so the check above discriminates


def test_cli_dispatches_h3_info(h3t, capsys):
    from h3turbo.cli import main

    with pytest.raises(SystemExit) as e:
        main(["h3-info", h3t])
    assert e.value.code == 0
    assert "hidden" in capsys.readouterr().out.lower()


def test_audio_scale_zero_is_not_treated_as_one(engine):
    g = torch.Generator().manual_seed(4)
    x = [torch.randn(1, CFG.video_channels, *VIDEO, generator=g).bfloat16(), torch.randn(1, CFG.audio_channels, 2, AUDIO_LEN, generator=g).bfloat16()]
    refined = engine.encode_text(text().bfloat16())
    base = engine.velocity(x, 0.5, refined_text=refined)
    zero = engine.velocity(x, 0.5, refined_text=refined, payload={"audio_scale": 0.0})
    one = engine.velocity(x, 0.5, refined_text=refined, payload={"audio_scale": 1.0})
    assert torch.equal(base[1], one[1]) and not torch.equal(base[1], zero[1])


def test_provider_state_dump_and_backend_report(engine):
    assert engine.provider._dump().startswith("slots ")  # f-string nested quotes here were a SyntaxError before Python 3.12
    assert engine.linear_backend() in ("ck", "torch")


def test_patch_projection_is_one_gemm_for_a_15k_token_clip(engine, monkeypatch):
    """Splitting the fp32 projection changed cuBLAS tiling, hence the last bit versus ComfyUI, and sampling amplified it (video PSNR 29 dB)."""
    import torch.nn.functional as F

    import h3turbo.h3.model as mod

    calls = []
    real = F.linear
    monkeypatch.setattr(mod.F, "linear", lambda x, w, b=None: (calls.append(x.shape[0]), real(x, w, b))[1])
    m = engine.model
    rows = torch.randn(15000, CFG.video_patch_dim)
    out = m._project(rows, m.vpw, m.vpb)
    assert calls == [15000] and out.shape == (15000, CFG.hidden)


def test_int8_attention_is_opt_in_and_validated(h3t):
    """The default stays the reference SDPA; 'int8' needs comfy_kitchen's CUDA extension and says so instead of silently falling back."""
    from h3turbo.h3.store import H3TFile

    with H3Engine.from_h3t(h3t, "cpu", resident=1, pin=False) as eng:
        assert eng.model.attn_impl == "sdpa" and eng.model._int8_attention is None
    with pytest.raises(ValueError, match="attn_impl"):
        H3Engine.from_h3t(h3t, "cpu", resident=1, pin=False, attn_impl="fp4")
    with pytest.raises(RuntimeError, match="attn_impl='int8'"):  # CPU / no comfy_kitchen CUDA extension
        H3Engine.from_h3t(h3t, "cpu", resident=1, pin=False, attn_impl="int8")
