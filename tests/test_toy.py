import numpy as np
import torch

from h3turbo import toy


def _u8(scene, side=96, T=9):
    return ((toy.render(scene, side, T).permute(0, 2, 3, 1) + 1) * 127.5).round().byte()


def test_analyzers_recover_ground_truth_from_clean_data():
    """The eval numbers in the README rest on this: on clean renders/audio the analyzers
    must be right every time, so any shortfall on model output is the model's."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = toy.sample_scene(rng)
        v = toy.analyze_video(_u8(s))
        a = toy.analyze_audio(toy.synth_audio(s, 45 * 800))
        assert v == {"color": s.color, "direction": s.direction}
        assert a == {"color": s.color, "direction": s.direction}


def test_analyzers_are_not_trivially_constant():
    """Wrong answers must be reachable: a different scene must give a different reading."""
    a = toy.Scene(0, 0, 0.25, 0.7, 0.5)
    b = toy.Scene(2, 3, 0.25, 0.5, 0.3)
    assert toy.analyze_video(_u8(a)) != toy.analyze_video(_u8(b))
    assert toy.analyze_audio(toy.synth_audio(a, 36000)) != toy.analyze_audio(toy.synth_audio(b, 36000))
    assert toy.analyze_video(torch.full((9, 96, 96, 3), 20, dtype=torch.uint8)) == {"color": None, "direction": None}
    assert toy.analyze_audio(torch.zeros(36000)) == {"color": None, "direction": None}


def test_chance_level_is_25_percent_on_noise():
    """Random garbage should score near chance against random scenes, or the metric would
    flatter a broken model."""
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(1)
    n = 240
    hits_c = hits_d = 0
    for _ in range(n):
        s = toy.sample_scene(rng)
        v = toy.analyze_audio(torch.randn(36000, generator=g) * 0.3)
        hits_c += v["color"] == s.color
        hits_d += v["direction"] == s.direction
    assert 0.1 < hits_c / n < 0.45 and 0.1 < hits_d / n < 0.45, (hits_c / n, hits_d / n)


def test_scene_from_prompt_matches_templates():
    for t in range(len(toy.TEMPLATES)):
        s = toy.Scene(3, 2, 0.25, 0.5, 0.5, template=t)
        assert toy.scene_from_prompt(s.prompt) == (3, 2)
