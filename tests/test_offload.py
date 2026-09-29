import torch

from h3turbo.config import make_config
from h3turbo.io import build_pipeline
from h3turbo.offload import BlockSwapper


def _pipe():
    torch.manual_seed(0)
    cfg = make_config("nano")
    cfg.fps = 8
    pipe = build_pipeline(cfg, "cpu", torch.float32)
    for p in pipe.model.parameters():
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.02)
    return pipe


KW = dict(width=64, height=64, num_frames=9, steps=2, seed=3)


def test_block_swap_matches_and_bounds_residency():
    pipe = _pipe()
    ref = pipe("hi", **KW)
    live = {"max": 0}

    def fake_device_copy(t, nb):
        return t.clone()  # a distinct tensor, like a real host->device copy

    sw = BlockSwapper(pipe.model.blocks, "cpu", resident=2, prefetch=False, transfer=fake_device_copy)
    per_block = [sum(t.numel() for t in ts) for ts in sw._cpu]
    peak = 0

    def spy(module, args):
        nonlocal peak
        resident = sum(1 for g in sw._gpu if g is not None)
        peak = max(peak, resident)

    for blk in pipe.model.blocks:
        blk.register_forward_pre_hook(spy)
    out = pipe("hi", **KW)
    assert torch.equal(out.video, ref.video) and torch.equal(out.audio, ref.audio)
    n = len(pipe.model.blocks)
    assert 2 <= peak <= 2 + 2 < n + 1, peak  # pinned + current + next
    assert sum(1 for g in sw._gpu if g is not None) == 2  # only the pinned ones remain
    sw.remove()
    assert all(g is not None for g in sw._gpu)
    assert torch.equal(pipe("hi", **KW).video, ref.video)


def test_zero_resident_wraps_prefetch():
    pipe = _pipe()
    ref = pipe("hi", **KW)
    sw = BlockSwapper(pipe.model.blocks, "cpu", resident=0, prefetch=False, transfer=lambda t, nb: t.clone())
    assert torch.equal(pipe("hi", **KW).video, ref.video)
    sw.remove()
