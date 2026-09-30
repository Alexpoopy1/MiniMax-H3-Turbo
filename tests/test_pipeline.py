import pytest
import torch

from h3turbo.config import make_config
from h3turbo.io import build_pipeline, load_checkpoint
from h3turbo.layout import Geometry, patchify_video, unpatchify_video
from h3turbo.pipeline import AudioCond, OmniContext, VideoCond


def _randomise(pipe):
    torch.manual_seed(0)
    for m in (pipe.model,):
        for p in m.parameters():
            if p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.02)
    return pipe


@pytest.fixture(scope="module")
def pipe():
    cfg = make_config("nano")
    cfg.fps = 8
    torch.manual_seed(0)
    return _randomise(build_pipeline(cfg, "cpu", torch.float32))


KW = dict(width=64, height=64, num_frames=9, steps=2, seed=3)


def _clip(pipe, T, value):
    return torch.full((T, 3, 64, 64), value)


def test_geometry_rules():
    g = Geometry(64, 96, 9, 8)
    assert (g.lat_t, g.hp, g.wp, g.video_tokens) == (3, 3, 2, 18)
    assert g.audio_tokens == 45 and g.audio_samples == 36000
    with pytest.raises(ValueError):
        Geometry(60, 64, 9, 8)
    with pytest.raises(ValueError):
        Geometry(64, 64, 8, 8)


def test_patchify_roundtrip():
    z = torch.randn(2, 24, 3, 6, 8)
    assert torch.equal(unpatchify_video(patchify_video(z), 24, 3, 6, 8), z)


def test_text_to_video_audio(pipe):
    g = pipe("a red square moving left", **KW)
    assert g.video.shape == (9, 64, 64, 3) and g.video.dtype == torch.uint8
    assert g.audio.shape == (36000,) and g.audio.abs().max() <= 1
    again = pipe("a red square moving left", **KW)
    assert torch.equal(g.video, again.video) and torch.equal(g.audio, again.audio)


def test_generate_subset(pipe):
    only_audio = pipe("x", generate="audio", **KW)
    assert only_audio.video is None and only_audio.audio is not None
    only_video = pipe("x", generate=("video",), **KW)
    assert only_video.audio is None and only_video.video is not None
    with pytest.raises(ValueError):
        pipe("x", generate=(), **KW)
    with pytest.raises(ValueError):
        pipe("x", generate=("smell",), **KW)


def _timeline(pipe, ctx, generate=("video", "audio")):
    geom = Geometry(64, 64, 9, 8)
    return geom, pipe._build_timeline(geom, ctx, set(generate), None)


def test_image_to_video_pins_first_latent_frame_only(pipe):
    ctx = OmniContext(video=[VideoCond(_clip(pipe, 1, 0.5), frame_index=0)])
    geom, (tracks, refs, _) = _timeline(pipe, ctx)
    pin = tracks["video"].pinned.view(geom.lat_t, -1)
    assert pin[0].all() and not pin[1:].any()
    assert not tracks["audio"].pinned.any() and not refs


def test_first_last_frame_pins_both_ends(pipe):
    ctx = OmniContext(
        video=[VideoCond(_clip(pipe, 1, -0.5), 0), VideoCond(_clip(pipe, 1, 0.5), frame_index=8)]
    )
    geom, (tracks, _, _) = _timeline(pipe, ctx)
    pin = tracks["video"].pinned.view(geom.lat_t, -1)
    assert pin[0].all() and pin[-1].all() and not pin[1].any()
    assert not torch.equal(tracks["video"].x.view(geom.lat_t, -1, 96)[0], tracks["video"].x.view(geom.lat_t, -1, 96)[-1])


def test_video_to_audio_keeps_video_as_clean_context(pipe):
    ctx = OmniContext(video=[VideoCond(_clip(pipe, 9, 0.1), 0)])
    geom, (tracks, _, _) = _timeline(pipe, ctx, generate=("audio",))
    assert not tracks["video"].emit and tracks["video"].pinned.all()
    assert tracks["video"].x.shape[1] == geom.video_tokens
    assert tracks["audio"].emit
    g = pipe("", generate="audio", context=ctx, **KW)
    assert g.video is None and g.audio.shape == (36000,)


def test_audio_to_video_and_partial_audio_pin(pipe):
    wave = torch.randn(8000).clamp(-1, 1)
    ctx = OmniContext(audio=[AudioCond(wave, start_sec=0.25)])
    geom, (tracks, _, _) = _timeline(pipe, ctx, generate=("video",))
    assert not tracks["audio"].emit and tracks["audio"].x.shape[1] == 10
    assert tracks["audio"].pos[0, 0, 0] == pytest.approx(10 * 8 / 160)  # starts at token 10
    g = pipe("", generate="video", context=ctx, **KW)
    assert g.audio is None and g.video.shape == (9, 64, 64, 3)


def test_context_only_modality_is_omitted_when_nothing_pinned(pipe):
    geom, (tracks, _, _) = _timeline(pipe, OmniContext(), generate=("video",))
    assert "audio" not in tracks


def test_pinned_tokens_survive_denoising_bit_exact(pipe):
    ctx = OmniContext(
        video=[VideoCond(_clip(pipe, 1, 0.3), 0)], audio=[AudioCond(torch.randn(8000).clamp(-1, 1), 0.0)]
    )
    geom, (tracks, refs, res) = _timeline(pipe, ctx)
    g = torch.Generator().manual_seed(0)
    for k, t in tracks.items():
        t.x = torch.where(t.pinned[..., None], t.x, torch.randn(t.x.shape, generator=g))
    before = {k: t.x.clone() for k, t in tracks.items()}
    from h3turbo.sampler import flow_sigmas

    sig = flow_sigmas(3)
    out = pipe._denoise(pipe.encode_text("hi"), None, tracks, refs, res, sig, sig, 1.0, None, None)
    for k, t in tracks.items():
        assert torch.equal(out[k][t.pinned], before[k][t.pinned]), k
        assert not torch.equal(out[k][~t.pinned], before[k][~t.pinned]), k


def test_inpaint_mask_pins_only_masked_tokens(pipe):
    mask = torch.zeros(64, 64)
    mask[:, :32] = 1  # left half kept -> left token column pinned
    ctx = OmniContext(video=[VideoCond(_clip(pipe, 9, 0.2), 0, mask=mask)])
    geom, (tracks, _, _) = _timeline(pipe, ctx)
    pin = tracks["video"].pinned.view(geom.lat_t, geom.hp, geom.wp)
    assert pin[:, :, 0].all() and not pin[:, :, 1].any()


def test_extension_pins_prefix(pipe):
    ctx = OmniContext(video=[VideoCond(_clip(pipe, 5, 0.0), 0)])  # 5 frames = 2 latent frames
    geom, (tracks, _, _) = _timeline(pipe, ctx)
    pin = tracks["video"].pinned.view(geom.lat_t, -1)
    assert pin[:2].all() and not pin[2:].any()


def test_references_are_clean_and_off_timeline(pipe):
    ctx = OmniContext(ref_video=[_clip(pipe, 1, 0.4)], ref_audio=[torch.randn(8000).clamp(-1, 1)])
    geom, (tracks, refs, _) = _timeline(pipe, ctx)
    assert [r.modality for r in refs] == ["video", "audio"]
    assert refs[0].pos[..., 0].min() > geom.lat_t  # beyond the target timeline
    g = pipe("hi", context=ctx, **KW)
    assert g.video is not None


def test_cfg_path_runs_and_differs(pipe):
    a = pipe("a", negative_prompt="b", guidance=1.0, **KW)
    b = pipe("a", negative_prompt="b", guidance=4.0, **KW)
    assert not torch.equal(a.video, b.video)


def test_step_count_is_flexible(pipe):
    for steps in (1, 2, 5):
        pipe("x", **{**KW, "steps": steps})


def test_refine_upscales_and_keeps_audio(pipe):
    g = pipe("x", **KW)
    r = pipe.refine(g, "x", scale=1.5, steps=2, seed=0)
    assert r.video.shape == (9, 96, 96, 3)
    assert torch.equal(r.audio, g.audio)


def test_guidance_embedding_and_resampler_paths():
    cfg = make_config("nano", guidance_embed=True, resampler_queries=4)
    cfg.fps = 8
    p = _randomise(build_pipeline(cfg, "cpu", torch.float32))
    ctx = OmniContext(ref_video=[_clip(p, 1, 0.4), _clip(p, 5, -0.3)])
    g = p("x", context=ctx, guidance=2.0, ref_budget=2, **KW)
    assert g.video.shape == (9, 64, 64, 3)
    a = p("x", guidance=1.0, **KW)
    b = p("x", guidance=6.0, **KW)
    assert not torch.equal(a.video, b.video)


def test_external_text_encoder():
    cfg = make_config("nano")
    cfg.text.kind, cfg.text.ext_dim = "external", 48
    p = _randomise(build_pipeline(cfg, "cpu", torch.float32))
    with pytest.raises(ValueError):
        p("x", **KW)
    g = p(text_embeds=torch.randn(5, 48), **KW)
    assert g.video is not None


def test_save_load_roundtrip_is_exact_in_fp32(pipe, tmp_path):
    path = str(tmp_path / "m.safetensors")
    from h3turbo.io import save_checkpoint

    save_checkpoint(path, pipe, dtype=torch.float32)
    loaded = load_checkpoint(path, "cpu", torch.float32)
    a, b = pipe("hi", **KW), loaded("hi", **KW)
    assert torch.equal(a.video, b.video) and torch.equal(a.audio, b.audio)


def test_fp16_checkpoint_loads_and_is_half_the_size(pipe, tmp_path):
    from h3turbo.io import save_checkpoint
    import os

    f32, f16 = str(tmp_path / "a.safetensors"), str(tmp_path / "b.safetensors")
    save_checkpoint(f32, pipe, dtype=torch.float32)
    save_checkpoint(f16, pipe, dtype=torch.float16)
    assert os.path.getsize(f16) < 0.7 * os.path.getsize(f32)
    load_checkpoint(f16, "cpu", torch.float32)("hi", **KW)


@pytest.mark.parametrize("mode,tol", [("int8", 0.002), ("int4", 0.01)])
def test_quantised_pipeline_runs_and_tracks_fp(pipe, tmp_path, mode, tol):
    from h3turbo.io import save_checkpoint
    from h3turbo.quant import quantized_bytes

    path = str(tmp_path / "m.safetensors")
    save_checkpoint(path, pipe, dtype=torch.float32)
    fp = load_checkpoint(path, "cpu", torch.float32)
    q = load_checkpoint(path, "cpu", torch.float32, quant=mode)
    assert quantized_bytes(q.model.blocks) < 0.6 * quantized_bytes(fp.model.blocks)
    a, b = fp("hi", **KW), q("hi", **KW)
    err = (a.video.float() - b.video.float()).abs().mean() / 255
    control = (a.video.float() - fp("hi", **{**KW, "seed": 99}).video.float()).abs().mean() / 255
    assert err < tol, err
    assert control > 10 * tol, "test cannot tell same from different"  # measured: control ~0.15
    # a quantised checkpoint round-trips as quantised
    qpath = str(tmp_path / "q.safetensors")
    save_checkpoint(qpath, q, dtype=torch.float32)
    q2 = load_checkpoint(qpath, "cpu", torch.float32)
    assert q2.quant == mode
    assert torch.equal(q("hi", **KW).video, q2("hi", **KW).video)


@pytest.mark.skipif(__import__("shutil").which("g++") is None, reason="torch.compile on CPU needs a C++ compiler")
def test_compile_keeps_keys_and_matches_eager(pipe):
    import copy

    p = copy.deepcopy(pipe)
    ref = p("hi", **KW)
    keys = sorted(p.model.state_dict())
    p.compile()
    out = p("hi", **KW)
    assert sorted(p.model.state_dict()) == keys
    assert (out.video.int() - ref.video.int()).abs().max() <= 1
    assert (out.audio - ref.audio).abs().max() < 1e-4


def test_single_frame_generation_is_text_to_image(pipe):
    g = pipe("x", generate="video", width=64, height=96, num_frames=1, steps=2, seed=0)
    assert g.video.shape == (1, 96, 64, 3)


def test_requested_size_and_frames_are_snapped(pipe):
    g = pipe("x", generate="video", width=70, height=50, num_frames=10, steps=1, seed=0)
    assert g.video.shape == (9, 64, 64, 3)  # 70->64, 50->64, 10 -> 1+4k = 9


def test_conditioning_past_the_end_is_an_error_not_silently_dropped(pipe):
    with pytest.raises(ValueError, match="past the end"):
        pipe("x", context=OmniContext(video=[VideoCond(_clip(pipe, 1, 0.0), frame_index=40)]), **KW)
    with pytest.raises(ValueError, match="past the end"):
        pipe("x", context=OmniContext(audio=[AudioCond(torch.zeros(8000), start_sec=9.0)]), **KW)


def test_snap_size_rounds_halves_up_and_has_a_floor():
    from h3turbo.layout import snap_size

    assert [snap_size(v) for v in (144, 112, 80, 70, 50, 16, 1)] == [160, 128, 96, 64, 64, 32, 32]
