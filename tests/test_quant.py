import torch

from h3turbo.quant import Int4Linear, Int8Linear, quantize_


def _lin(i=128, o=96, bias=True):
    torch.manual_seed(0)
    return torch.nn.Linear(i, o, bias=bias)


def test_int8_error_is_small():
    lin = _lin()
    q = Int8Linear.from_linear(lin)
    x = torch.randn(4, 128)
    rel = (q(x) - lin(x)).norm() / lin(x).norm()
    assert rel < 0.01
    assert q.qweight.dtype == torch.int8


def test_int4_error_is_bounded_and_packs_two_per_byte():
    lin = _lin()
    q = Int4Linear.from_linear(lin)
    assert q.qweight.shape == (96, 64) and q.qweight.dtype == torch.uint8
    x = torch.randn(4, 128)
    rel = (q(x) - lin(x)).norm() / lin(x).norm()
    assert rel < 0.15


def test_int4_dequantise_matches_reference_rounding():
    lin = _lin(64, 8, bias=False)
    q = Int4Linear.from_linear(lin, group=64)
    w = lin.weight.detach()
    scale = w.abs().amax(1, keepdim=True) / 7
    ref = (w / scale).round().clamp(-8, 7) * scale
    assert torch.allclose(q.dequantize(), ref, atol=2e-3)


def test_quantize_swaps_only_linears_and_supports_no_bias():
    net = torch.nn.Sequential(torch.nn.Linear(64, 64, bias=False), torch.nn.SiLU(), torch.nn.Linear(64, 8))
    quantize_(net, "int8")
    assert isinstance(net[0], Int8Linear) and isinstance(net[2], Int8Linear)
    assert net[0].bias is None and net[2].bias is not None
    assert net(torch.randn(2, 64)).shape == (2, 8)


def test_int4_skips_layers_not_divisible_by_group():
    net = torch.nn.Sequential(torch.nn.Linear(100, 8))
    quantize_(net, "int4")
    assert isinstance(net[0], torch.nn.Linear)
