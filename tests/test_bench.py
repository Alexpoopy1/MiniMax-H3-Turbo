import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from h3turbo.bench import estimate, forward_flops
from h3turbo.config import make_config, tier_names
from h3turbo.layout import GROUPS
from h3turbo.model import H3TurboTransformer, Segment


def test_analytic_flops_match_pytorch_counter():
    cfg = make_config("nano")
    m = H3TurboTransformer(cfg.transformer).eval()
    B, n = 1, 150
    seg = Segment("video", torch.randn(B, n, cfg.transformer.video_in_dim), torch.rand(1, n, 3), torch.full((B, n), 1), True)
    gt = torch.tensor([[0, 500.0, 0, 500.0, 0]])
    mods = m.compute_mods(GROUPS, gt)
    # the default CPU flash-attention op is not registered with the counter (it would
    # silently drop the attention term), so force the decomposed math path it can see
    with FlopCounterMode(display=False) as fc, sdpa_kernel(SDPBackend.MATH):
        m([seg], GROUPS, gt, mods=mods)
    measured = fc.get_total_flops()
    io = 2 * cfg.transformer.video_in_dim * cfg.transformer.hidden * n * 2  # in_proj + out_proj
    analytic = forward_flops(cfg, n) + io
    assert measured == analytic, (measured, analytic)


def test_estimate_orders_tiers_and_precisions():
    ests = [estimate(make_config(t)) for t in tier_names()]
    assert [e.params["transformer_total_M"] for e in ests] == sorted(e.params["transformer_total_M"] for e in ests)
    for e in ests:
        w = e.weights_gb
        assert w["int4"] < w["int8"] < w["fp16"]
        assert e.generation_tflop > e.forward_tflop
    # AdaLN is kept off the GPU
    e = ests[2]
    assert e.params["on_gpu_M (no adaln)"] < e.params["transformer_total_M"]
