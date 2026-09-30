"""W4A8 (ConvRot) linear ops: decode, rotation, quantiser, GEMM backends.  CPU only, seeded, independent references."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from h3turbo.h3 import qlinear as ql
from h3turbo.h3.types import W4A8Weight

# the 16 Lloyd-Max levels of a real checkpoint layer (blocks.0.attn.out_proj.weight_codebook)
REAL_CB = [-0.9807561635971069, -0.7949668765068054, -0.6386534571647644, -0.5035541653633118, -0.38370054960250854,
           -0.2726648449897766, -0.1664646714925766, -0.06367518752813339, 0.03837905079126358, 0.1426069736480713,
           0.2516845762729645, 0.3661285936832428, 0.4894854724407196, 0.6280644536018372, 0.787971019744873, 0.979333221912384]


def fp8_e4m3fn_value(b: int) -> float:
    """OCP E4M3FN by hand: bias 7, subnormals below 2^-6, no infinities, S.1111.111 is NaN."""
    sign = -1.0 if b & 0x80 else 1.0
    e, m = (b >> 3) & 0xF, b & 7
    if e == 0xF and m == 7:
        return float("nan")
    return sign * (m / 8.0) * 2.0**-6 if e == 0 else sign * (1 + m / 8.0) * 2.0 ** (e - 7)


def spec_hadamard(size: int) -> np.ndarray:
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64)
    h = h4
    while h.shape[0] < size:
        h = np.kron(h, h4)
    return h / np.sqrt(size)


def spec_decode(q: np.ndarray, s_bytes: np.ndarray, cb: np.ndarray, group: int = 16) -> np.ndarray:
    """Element loops straight from the format description: low nibble first, fp32 product, half-even, clamp +-127."""
    n, kh = q.shape
    out = np.zeros((n, kh * 2), dtype=np.int8)
    for i in range(n):
        for j in range(kh * 2):
            byte = int(q[i, j // 2]) & 0xFF
            nib = (byte >> 4) if j % 2 else (byte & 0xF)
            s = np.float32(fp8_e4m3fn_value(int(s_bytes[i, j // group])))
            out[i, j] = int(np.clip(np.rint(np.float32(cb[nib]) * s), -127, 127))
    return out


def make_weight(n, k, seed=0, group=16) -> W4A8Weight:
    convrot = 256
    while k % convrot:  # small test widths use a smaller (still power-of-4) rotation block
        convrot //= 4
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-128, 128, (n, k // 2), generator=g, dtype=torch.int8)
    s = (torch.rand(n, k // group, generator=g) * 180 + 20).to(torch.float8_e4m3fn)  # group scales 20..200: exercises the clamp
    return W4A8Weight(q, s, torch.rand(n, generator=g) * 0.02 + 0.005, torch.tensor(REAL_CB), group_size=group, convrot=convrot)


def literal_int8_linear(x, w8, s_ch, convrot=256):
    """kitchen eager int8_linear(convrot=True), re-typed op by op: rotation in x.dtype, absmax, bf16-rounded scale, int32 acc."""
    k = x.shape[-1]
    h = torch.tensor(spec_hadamard(convrot), dtype=torch.float32).to(x.dtype)
    xr = torch.matmul(x.reshape(-1, k // convrot, convrot), h).reshape(-1, k)
    scale = (xr.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    sm = scale.to(x.dtype)
    sm = torch.where(sm == 0, torch.full_like(sm, torch.finfo(x.dtype).tiny), sm)
    xq = (xr / sm).round().clamp(-128, 127).to(torch.int8)
    acc = (xq.long() @ w8.long().T).to(torch.int32)
    return (acc.float() * (scale * s_ch.float().reshape(1, -1))).to(x.dtype)


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def ref_fp32(x, w):  # unquantised activation against the dequantised original-basis weight, in float64
    return x.double() @ ql.dequantize(w, torch.float64, rotated=False).T


@pytest.mark.parametrize("size", [4, 16, 64, 256, 1024])
def test_hadamard_matches_spec_and_is_symmetric_involutory(size):
    h = ql.hadamard_regular(size, "cpu", torch.float32)
    ref = spec_hadamard(size)
    assert h.shape == (size, size) and torch.equal(h, h.T)
    assert np.array_equal(h.double().numpy(), ref)  # entries +-2^-j: the fp32 cast is exact
    assert torch.equal(h @ h, torch.eye(size))  # sums of dyadic rationals: exact in fp32
    assert set(h.abs().unique().tolist()) == {1.0 / np.sqrt(size)}
    assert ql.hadamard_regular(size, "cpu", torch.float32) is h  # cached
    hb = ql.hadamard_regular(size, "cpu", torch.bfloat16)
    assert hb.dtype == torch.bfloat16 and torch.equal(hb.float(), h)  # exact in bf16 too


@pytest.mark.parametrize("bad", [0, 1, 2, 3, 8, 12, 32, 100, 128, 512, -4])
def test_hadamard_rejects_non_powers_of_four(bad):
    with pytest.raises(ValueError, match="power of 4"):
        ql.hadamard_regular(bad)


def test_rotate_blockwise_reference_involution_and_norm():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(5, 3, 512, generator=g)
    y = ql.rotate(x)
    h = spec_hadamard(256)
    xn = x.double().numpy().reshape(15, 2, 256)
    ref = (xn @ h).reshape(5, 3, 512)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert np.allclose(y.double().numpy(), ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ql.rotate(y), x, atol=1e-5)  # involution
    blocks = lambda t: t.reshape(15, 2, 256).norm(dim=-1)
    assert torch.allclose(blocks(y), blocks(x), rtol=1e-5)  # orthogonal per block
    # blocks do not mix: zeroing one block of the input leaves the other block's rotation untouched
    x2 = x.clone()
    x2[..., :256] = 0
    assert torch.equal(ql.rotate(x2)[..., 256:], y[..., 256:])
    assert torch.count_nonzero(ql.rotate(x2)[..., :256]) == 0


def test_rotate_low_precision_and_errors():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(6, 512, generator=g).to(torch.bfloat16)
    y = ql.rotate(x)
    ref = torch.from_numpy((x.double().numpy().reshape(6, 2, 256) @ spec_hadamard(256)).reshape(6, 512))
    assert y.dtype == torch.bfloat16
    assert ((y.double() - ref).abs() <= ref.abs() * 2.0**-8 + 1e-30).all()  # one bf16 rounding of the exact value
    with pytest.raises(ValueError, match="multiple"):
        ql.rotate(torch.zeros(3, 250))
    with pytest.raises(ValueError, match="multiple"):
        ql.quantize_activation(torch.zeros(2, 100))
    with pytest.raises(ValueError, match="floating"):
        ql.rotate(torch.zeros(3, 256, dtype=torch.int8))
    assert ql.rotate(torch.zeros(0, 256)).shape == (0, 256)
    x64 = torch.randn(3, 128, generator=torch.Generator().manual_seed(4))
    ref = torch.from_numpy((x64.double().numpy().reshape(3, 2, 64) @ spec_hadamard(64)).reshape(3, 128))
    assert torch.allclose(ql.rotate(x64, size=64).double(), ref, rtol=1e-5, atol=1e-6)  # the size argument is honoured


def test_rotate_and_linear_do_not_mutate_their_inputs():
    wq_ = make_weight(8, 512, seed=1)
    for dt in (torch.float32, torch.bfloat16):
        x = torch.randn(5, 512, generator=torch.Generator().manual_seed(3)).to(dt)
        h = torch.randn(5, 1024, generator=torch.Generator().manual_seed(4)).to(dt)
        x0, h0 = x.clone(), h.clone()
        ql.rotate(x)
        ql.quantize_activation(x)
        for kw in ({}, {"precision": "a16"}, {"quant_mode": "eager"}):
            ql.linear(x, wq_, **kw)
            ql.linear(h, wq_, input_act="swiglu", **kw)
        assert torch.equal(x, x0) and torch.equal(h, h0)


def test_fp8_reference_matches_torch_for_every_non_nan_byte():
    raw = torch.arange(256, dtype=torch.uint8)
    ours = np.array([fp8_e4m3fn_value(b) for b in range(256)])
    theirs = raw.view(torch.float8_e4m3fn).float().numpy()
    nan = np.isnan(ours)
    assert nan.sum() == 2 and np.array_equal(nan, np.isnan(theirs))
    assert np.array_equal(ours[~nan], theirs[~nan])


@pytest.mark.parametrize("group", [16, 4, 32])
def test_decode_matches_element_loops(group):
    k = 128
    w = make_weight(7, k, seed=3, group=group)
    s_bytes = w.s_rel.view(torch.uint8).numpy()
    assert not np.isnan(w.s_rel.float().numpy()).any()
    ref = spec_decode(w.q.numpy(), s_bytes, w.codebook.numpy(), group)
    got = ql.decode_int8_grid(w)
    assert got.dtype == torch.int8 and got.shape == (7, k)
    assert np.array_equal(got.numpy(), ref)
    assert ref.min() < -100 and ref.max() > 100  # the sampled scales actually exercise the range
    assert np.array_equal(ql.decode_int8_grid(W4A8Weight(w.q, w.s_rel.view(torch.uint8), w.s_ch, w.codebook, group, w.convrot)).numpy(), ref)  # raw fp8 bytes accepted


def test_decode_ties_round_half_even_and_clamp_symmetric():
    cb = [0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # (code, fp8 scale, expected level): 0.5*3=1.5 -> 2 ; 0.5*5=2.5 -> 2 (half-even) ; -0.5*5=-2.5 -> -2 ;
    # 1.0*448 -> clamp 127 ; -1.0*448 -> -127 (never -128) ; 0.75*7=5.25 -> 5 ; 0.5*7=3.5 -> 4
    cases = [(0, 3.0, 2), (0, 5.0, 2), (1, 5.0, -2), (2, 448.0, 127), (3, 448.0, -127), (4, 6.0, 2), (5, 6.0, -2), (6, 7.0, 5), (8, 448.0, 0), (0, 7.0, 4)]
    codes = torch.tensor([c[0] for c in cases])
    byte = (codes | (codes << 4)).to(torch.int32)
    q = ((byte + 128) % 256 - 128).to(torch.int8)[:, None].expand(-1, 8).contiguous()  # both nibbles of every byte carry the code
    s = torch.tensor([[c[1]] for c in cases])
    w = W4A8Weight(q, s.to(torch.float8_e4m3fn), torch.ones(len(cases)), torch.tensor(cb), group_size=16, convrot=16)
    assert w.s_rel.float().flatten().tolist() == [c[1] for c in cases]  # the scales are exactly representable in fp8
    got = ql.decode_int8_grid(w)
    assert got.tolist() == [[c[2]] * 16 for c in cases]
    assert np.array_equal(got.numpy(), spec_decode(q.numpy(), w.s_rel.view(torch.uint8).numpy(), np.array(cb, dtype=np.float32)))


def test_decode_slabs_generic_path_and_padding(monkeypatch):
    w = make_weight(37, 128, seed=5)
    full = ql.decode_int8_grid(w)
    monkeypatch.setattr(ql, "_DECODE_ELEMS", 128 * 5 + 3)  # 5 rows per slab, 7 slabs, remainder of 2
    assert torch.equal(ql.decode_int8_grid(w), full)
    wf = W4A8Weight(w.q, w.s_rel.float(), w.s_ch, w.codebook, 16, w.convrot)  # fp32 group scales take the arithmetic path
    assert torch.equal(ql.decode_int8_grid(wf), full)
    big = torch.cat([torch.zeros(3, 64, dtype=torch.int8), w.q, torch.zeros(2, 64, dtype=torch.int8)])
    sub = W4A8Weight(big[3:-2], w.s_rel, w.s_ch, w.codebook, 16, w.convrot)  # a view into a larger buffer, as the streaming slots hand out
    assert torch.equal(ql.decode_int8_grid(sub), full)


def test_weight_validation_errors():
    w = make_weight(4, 256)
    bad = [(W4A8Weight(w.q, w.s_rel[:, :-1], w.s_ch, w.codebook), "s_rel"), (W4A8Weight(w.q, w.s_rel, w.s_ch[:-1], w.codebook), "s_ch"),
           (W4A8Weight(w.q, w.s_rel, w.s_ch, w.codebook[:8]), "codebook"), (W4A8Weight(w.q.float(), w.s_rel, w.s_ch, w.codebook), "int8"),
           (W4A8Weight(torch.zeros(2, 100, dtype=torch.int8), torch.zeros(2, 12).to(torch.float8_e4m3fn), torch.ones(2), w.codebook), "divisible")]
    for weight, msg in bad:
        with pytest.raises(ValueError, match=msg):
            ql.decode_int8_grid(weight)


def test_dequantize_bases_rows_and_rotation_consistency():
    w = make_weight(20, 512, seed=6)
    grid = ql.decode_int8_grid(w).double()
    rot = ql.dequantize(w, torch.float64, rotated=True)
    assert torch.equal(rot, (grid.float() * w.s_ch[:, None]).double())  # grid times the per-channel scale, one fp32 product
    orig = ql.dequantize(w, torch.float64, rotated=False)
    ref = (rot.numpy().reshape(20, 2, 256) @ spec_hadamard(256)).reshape(20, 512)
    assert np.abs(orig.numpy() - ref).max() < 2e-6 * np.abs(ref).max()  # fp32 rotation: rounding error is relative to the block's magnitude
    assert torch.equal(ql.dequantize(w, rows=slice(5, 11)), ql.dequantize(w)[5:11])
    with pytest.raises(ValueError, match="contiguous"):
        ql.dequantize(w, rows=slice(0, 10, 2))
    x = torch.randn(9, 512, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(ql.rotate(x) @ ql.dequantize(w, rotated=True).T, x @ ql.dequantize(w, rotated=False).T, rtol=1e-4, atol=1e-5)
    b = ql.dequantize(w, torch.bfloat16)
    assert b.dtype == torch.bfloat16 and rel(b, orig) < 2.0**-8  # a single rounding of the fp32-rotated weight


@pytest.mark.parametrize("k", [8, 256, 1000, 1024, 1032, 3000, 4096])
def test_int8_mm_is_exact_including_worst_case_magnitudes(k):
    # |a| <= 128, |w| <= 127: a chunk of _EXACT_K terms sums to at most _EXACT_K * 16256 < 2^24, where every integer is an fp32
    assert ql._EXACT_K * 128 * 127 < 2**24 <= (ql._EXACT_K + 8) * 128 * 127 * 2
    g = torch.Generator().manual_seed(k)
    a = torch.randint(-128, 128, (5, k), generator=g, dtype=torch.int8)
    w = torch.randint(-127, 128, (9, k), generator=g, dtype=torch.int8)
    assert torch.equal(ql._int8_mm(a, w), (a.long() @ w.long().T).to(torch.int32))
    # odd products (16129 = 127 * 127) that pass 2^24 after 1040 terms: an fp32 accumulation over the whole K would round
    a = torch.full((3, k), 127, dtype=torch.int8)
    w = torch.full((2, k), 127, dtype=torch.int8)
    assert ql._int8_mm(a, w)[2, 1].item() == 16129 * k
    a = torch.full((3, k), -128, dtype=torch.int8)  # the extreme product 16256 is even: mix in odd rows so sums stay odd
    a[:, 1::2] = -127
    w = torch.full((2, k), -127, dtype=torch.int8)
    assert torch.equal(ql._int8_mm(a, w), (a.long() @ w.long().T).to(torch.int32))
    w = torch.full((2, k), 127, dtype=torch.int8)
    assert torch.equal(ql._int8_mm(a, w), (a.long() @ w.long().T).to(torch.int32))


def test_int8_mm_native_cpu_kernel_agrees_and_empty_shapes(monkeypatch):
    g = torch.Generator().manual_seed(3)
    a, w = torch.randint(-128, 128, (20, 64), generator=g, dtype=torch.int8), torch.randint(-127, 128, (24, 64), generator=g, dtype=torch.int8)
    emu = ql._int8_mm(a, w)
    monkeypatch.setattr(ql, "_CPU_INT_MM", True)
    assert torch.equal(ql._int8_mm(a, w), emu)
    assert ql._int8_mm(a[:0], w).shape == (0, 24)
    assert ql._int8_mm(a, w[:0]).shape == (20, 0)


def test_quantize_fp32_both_modes_equal_literal_port_and_bounds():
    g = torch.Generator().manual_seed(8)
    x = torch.randn(11, 512, generator=g) * torch.exp(torch.randn(11, 512, generator=g))
    xr = ql.rotate(x)
    scale = (xr.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
    want = (xr / scale).round().clamp(-128, 127).to(torch.int8)
    for mode in ("ck", "eager"):
        q, s = ql.quantize_activation(x, quant_mode=mode)
        assert q.dtype == torch.int8 and s.dtype == torch.float32 and s.shape == (11, 1)
        assert torch.equal(q, want) and torch.equal(s, scale)
        assert q.min() >= -128 and q.max() <= 127
        assert (q.abs().amax(-1) >= 126).all()  # the absmax element sits on the grid end
        assert (q.float() * s - xr).abs().max() <= (s.max() * 0.5 + 1e-6)  # within half a step


def test_quantize_bf16_modes_differ_only_in_where_the_absmax_comes_from():
    g = torch.Generator().manual_seed(9)
    x = torch.randn(64, 512, generator=g).to(torch.bfloat16)
    x32 = ql._rot32(x, 256)
    amax_unrounded = x32.abs().amax(-1, keepdim=True)
    amax_rounded = x32.to(torch.bfloat16).abs().amax(-1, keepdim=True).float()
    qc, sc = ql.quantize_activation(x, quant_mode="ck")
    qe, se = ql.quantize_activation(x, quant_mode="eager")
    assert torch.equal(sc, amax_unrounded / 127.0) and torch.equal(se, amax_rounded / 127.0)
    assert not torch.equal(sc, se)  # bf16 rounding of the max moves the scale
    y = x32.to(torch.bfloat16)
    for q, s in ((qc, sc), (qe, se)):
        s16 = s.to(torch.bfloat16)
        assert torch.equal(q, (y / s16).round().clamp(-128, 127).to(torch.int8))  # divide in bf16 by the bf16-rounded scale
    assert (qc != qe).float().mean() < 0.1
    # both are within ~1.5 steps of the exact rotation (kitchen's quantiser divides in bf16)
    for q, s in ((qc, sc), (qe, se)):
        assert ((q.float() * s - x32).abs() <= s * 1.5).all()  # half a step + bf16 rounding of the input, the quotient and the scale


def test_quantize_degenerate_rows_do_not_produce_nan_or_overflow():
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        x = torch.zeros(4, 256, dtype=dt)
        x[1, 3], x[2], x[3, 0] = 1e-3, 3.0, (6e4 if dt == torch.float16 else 1e30)
        for mode in ("ck", "eager"):
            q, s = ql.quantize_activation(x, quant_mode=mode)
            assert torch.isfinite(s).all() and (s > 0).all()
            assert q[0].abs().sum() == 0 and s[0].item() == pytest.approx(1e-30, rel=1e-6)
            assert q.int().abs().max() <= 128
            assert q[3].int().abs().max() >= 120 and q[2].int().abs().max() >= 120


N, K = 40, 512  # the W4A8 layer used by the linear tests


@pytest.fixture(scope="module")
def wq():
    return make_weight(N, K, seed=11)


def _int_valued(m, seed):
    """Small integers: every rotation sum is exact in fp32, so any correct implementation agrees bit for bit."""
    return torch.randint(-9, 10, (m, K), generator=torch.Generator().manual_seed(seed)).float()


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("m", [1, 7, 17, 257])
def test_linear_a8_eager_mode_equals_literal_port_bit_for_bit(wq, m, dt):
    x = _int_valued(m, m).to(dt)
    want = literal_int8_linear(x, ql.decode_int8_grid(wq), wq.s_ch)
    got = ql.linear(x, wq, backend="torch", quant_mode="eager")
    assert got.dtype == dt and torch.equal(got, want)
    # the "ck" profile differs from eager only through the absmax source (bf16) and the epilogue order
    ck = ql.linear(x, wq, backend="torch", quant_mode="ck")
    assert rel(ck, want) < (1e-6 if dt == torch.float32 else 0.02)  # bf16: independent int8 rounding noise, ~half the total error


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_linear_a8_eager_mode_matches_literal_port_on_gaussian_rows(wq, dt):
    x = torch.randn(64, K, generator=torch.Generator().manual_seed(99)).to(dt)
    want = literal_int8_linear(x, ql.decode_int8_grid(wq), wq.s_ch)
    # the port rotates with the platform GEMM, we with a correctly rounded rotation: they differ in the last bit of
    # some absmax scales and, rarely, by one int8 step in a tie case
    assert rel(ql.linear(x, wq, backend="torch", quant_mode="eager"), want) < 3e-4


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_ck_profile_epilogue_and_absmax_source(wq, dt):
    """quant_mode='ck' = the CUDA kernel's numerics: absmax of the unrounded rotation, then (acc * x_scale) * w_scale."""
    x = _int_valued(33, 5).to(dt)
    q, xs = ql.quantize_activation(x, quant_mode="ck")
    xr = torch.from_numpy((x.double().numpy().reshape(33, 2, 256) @ spec_hadamard(256)).reshape(33, K))  # exact in float64
    assert torch.equal(xs, (xr.abs().amax(-1, keepdim=True) / 127.0).float())  # from the UNROUNDED rotation
    acc = (q.long() @ ql.decode_int8_grid(wq).long().T).to(torch.int32)
    want = ((acc.float() * xs) * wq.s_ch.float().reshape(1, -1)).to(dt)
    assert torch.equal(ql.linear(x, wq, backend="torch", quant_mode="ck"), want)
    eager_order = (acc.float() * (xs * wq.s_ch.float().reshape(1, -1))).to(dt)
    if dt == torch.float32:
        assert not torch.equal(want, eager_order)  # the two orders really differ in fp32, so this test pins the ck one


def test_linear_a8_error_is_the_predicted_activation_quantisation_noise(wq):
    m = 64
    x = torch.randn(m, K, generator=torch.Generator().manual_seed(12))
    y = ql.linear(x, wq, backend="torch")
    ref = ref_fp32(x, wq)
    q, s = ql.quantize_activation(x)
    xr = ql.rotate(x)
    # uniform rounding noise: std = step / sqrt(12) per element, independent of the weights, so the
    # relative output error equals the relative input noise (both rms over the same rows)
    predicted = (s.double() ** 2 / 12).mean().sqrt() / xr.double().pow(2).mean().sqrt()
    measured = rel(y, ref)
    assert 0.8 * predicted < measured < 1.25 * predicted, (measured, predicted.item())
    assert measured < 0.015  # ~0.9% for gaussian rows of this width
    # and a16 removes it
    y16 = ql.linear(x, wq, backend="torch", precision="a16")
    assert rel(y16, ref) < 2e-6


def test_linear_a16_bf16_is_bf16_noise_only(wq):
    x = torch.randn(48, K, generator=torch.Generator().manual_seed(13)).to(torch.bfloat16)
    ref = ref_fp32(x.float(), wq)
    y16 = ql.linear(x, wq, precision="a16")
    y8 = ql.linear(x, wq)
    assert y16.dtype == torch.bfloat16
    # two bf16 roundings (rotated x, output) at ~0.16% rms each; the weights are not rounded
    assert rel(y16, ref) < 0.004 and rel(y16, ref) < 0.5 * rel(y8, ref)
    assert torch.equal(ql.linear(x, wq, backend="ck", precision="a16"), y16)  # ck has no unquantised mode: runs the torch path


def test_a16_applies_the_channel_scale_to_the_fp32_accumulator(wq):
    """rotated x times the EXACT int8 levels, then s_ch in fp32, rounded once (the weights are never rounded to bf16)."""
    x = _int_valued(9, 6).to(torch.bfloat16)
    xr = ql.rotate(x).double()
    y = xr @ ql.decode_int8_grid(wq).double().T * wq.s_ch.double()
    got = ql.linear(x, wq, precision="a16")
    assert rel(got, y) < 2.0**-9  # one bf16 rounding of the exact value (an fp32 accumulate may tie-flip a few)


def test_chunking_is_exact_for_a8(wq, monkeypatch):
    for dt in (torch.float32, torch.bfloat16):
        x = torch.randn(37, K, generator=torch.Generator().manual_seed(14)).to(dt)
        full = ql.linear(x, wq)
        for c in (1, 2, 5, 36, 37, 1000):
            assert torch.equal(ql.linear(x, wq, chunk_tokens=c), full), (dt, c)
    monkeypatch.setattr(ql, "_ACC_BYTES", 4 * N * 32)  # accumulator budget of one 32-row pass -> internal chunking of a 100-row input
    x = torch.randn(100, K, generator=torch.Generator().manual_seed(15))
    small = ql.linear(x, wq)
    monkeypatch.setattr(ql, "_ACC_BYTES", 128 << 20)
    assert torch.equal(small, ql.linear(x, wq))


def test_chunking_a16_and_swiglu(wq):
    x = torch.randn(23, K, generator=torch.Generator().manual_seed(16))
    assert torch.allclose(ql.linear(x, wq, precision="a16", chunk_tokens=4), ql.linear(x, wq, precision="a16"), rtol=1e-5, atol=1e-6)
    h = torch.randn(23, 2 * K, generator=torch.Generator().manual_seed(17)).to(torch.bfloat16)
    assert torch.equal(ql.linear(h, wq, input_act="swiglu", chunk_tokens=6), ql.linear(h, wq, input_act="swiglu"))


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_swiglu_is_silu_gate_times_up_then_linear(wq, dt):
    h = torch.randn(9, 2 * K, generator=torch.Generator().manual_seed(18)).to(dt)
    gate, up = h[:, :K], h[:, K:]
    act = F.silu(gate).mul_(up)  # gate is the FIRST half
    for kw in ({}, {"precision": "a16"}, {"quant_mode": "eager"}):
        assert torch.equal(ql.linear(h, wq, input_act="swiglu", **kw), ql.linear(act, wq, **kw))
    dense = torch.randn(30, K, generator=torch.Generator().manual_seed(19)).to(dt)
    got = ql.linear(h, dense, input_act="swiglu")
    assert torch.equal(got, F.linear(act, dense)) if dt == torch.float32 else rel(got, F.linear(act, dense)) < 3e-3  # bf16: fp32 sums, ties may flip
    with pytest.raises(ValueError, match="even"):
        ql.linear(torch.zeros(2, 5), wq, input_act="swiglu")
    with pytest.raises(ValueError, match="do not match"):
        ql.linear(torch.zeros(2, K), wq, input_act="swiglu")  # swiglu halves the width: K/2 != K


@pytest.mark.parametrize("shape", [(1, K), (7, K), (17, K), (257, K), (2, 5, K), (3, 1, 4, K)])
def test_token_counts_and_leading_dims(wq, shape):
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(sum(shape)))
    y = ql.linear(x, wq)
    assert y.shape == (*shape[:-1], N) and y.dtype == torch.float32
    assert torch.equal(y.reshape(-1, N), ql.linear(x.reshape(-1, K), wq))
    row = ql.linear(x.reshape(-1, K)[:1], wq)  # token-wise: a row's result does not depend on its neighbours
    assert torch.equal(y.reshape(-1, N)[:1], row)


def test_noncontiguous_and_empty_inputs(wq):
    x = torch.randn(K, 13, generator=torch.Generator().manual_seed(20)).T  # [13, K] view with transposed strides
    assert not x.is_contiguous() and torch.equal(ql.linear(x, wq), ql.linear(x.contiguous(), wq))
    assert ql.linear(torch.zeros(0, K), wq).shape == (0, N) and ql.linear(torch.zeros(2, 0, K), wq).shape == (2, 0, N)


def test_bias_is_added_after_output_rounding(wq):
    b = torch.randn(N, generator=torch.Generator().manual_seed(21))
    for dt in (torch.float32, torch.bfloat16):
        x = torch.randn(9, K, generator=torch.Generator().manual_seed(22)).to(dt)
        for mode in ("ck", "eager"):
            plain = ql.linear(x, wq, quant_mode=mode)
            with_b = ql.linear(x, wq, b, quant_mode=mode)
            assert torch.equal(with_b, plain + b.to(dt))
    x = torch.randn(9, K, generator=torch.Generator().manual_seed(23))
    assert torch.allclose(ql.linear(x, wq, b, precision="a16"), ql.linear(x, wq, precision="a16") + b, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="bias"):
        ql.linear(x, wq, b[:-1])


def test_out_dtype_and_input_dtypes(wq):
    x = torch.randn(5, K, generator=torch.Generator().manual_seed(24)).to(torch.bfloat16)
    y = ql.linear(x, wq, out_dtype=torch.float32)
    assert y.dtype == torch.float32 and torch.equal(y.to(torch.bfloat16), ql.linear(x, wq))  # one rounding either way
    xh = x.to(torch.float16)
    yh = ql.linear(xh, wq)
    assert yh.dtype == torch.float16 and torch.isfinite(yh).all()
    assert rel(yh, ref_fp32(xh.float(), wq)) < 0.02
    with pytest.raises(ValueError, match="floating"):
        ql.linear(torch.zeros(2, K, dtype=torch.int32), wq)


def test_dense_weight_path():
    g = torch.Generator().manual_seed(26)
    x, w, b = torch.randn(6, 48, generator=g), torch.randn(10, 48, generator=g), torch.randn(10, generator=g)
    assert torch.equal(ql.linear(x, w), F.linear(x, w))
    assert torch.equal(ql.linear(x, w, b), F.linear(x, w, b))
    xb = x.to(torch.bfloat16)
    y = ql.linear(xb, w, b)  # fp32 weights are cast to the activation dtype, like ComfyUI's cast_bias_weight
    assert y.dtype == torch.bfloat16 and rel(y, F.linear(xb.float(), w.to(torch.bfloat16).float(), b.to(torch.bfloat16).float())) < 5e-3
    assert ql.linear(xb, w, out_dtype=torch.float32).dtype == torch.float32
    assert torch.equal(ql.linear(x, w, chunk_tokens=4), F.linear(x, w))
    assert ql.linear(torch.zeros(0, 48), w).shape == (0, 10)
    with pytest.raises(ValueError, match="do not match"):
        ql.linear(x, w[:, :40])


def test_backend_selection_and_validation(wq):
    info = ql.available_backends()
    assert set(info) == {"ck", "torch"} and info["torch"]["available"] is True
    assert info["ck"]["available"] or info["ck"]["reason"]
    x = torch.randn(3, K)
    assert [ql.resolve_backend(x, wq, "auto"), ql.resolve_backend(x, torch.zeros(4, K)), ql.resolve_backend(x, wq, "ck", "a16")] == ["torch", "dense", "torch"]  # CPU tensors never use ck
    with pytest.raises(RuntimeError, match="ck"):
        ql.linear(x, wq, backend="ck")
    for kw, msg in (({"backend": "cuda"}, "backend"), ({"precision": "a4"}, "precision"), ({"quant_mode": "x"}, "quant_mode"),
                    ({"input_act": "gelu"}, "input_act"), ({"chunk_tokens": 0}, "chunk_tokens")):
        with pytest.raises(ValueError, match=msg):
            ql.linear(x, wq, **kw)
    with pytest.raises(ValueError, match="do not match"):
        ql.linear(torch.zeros(2, K // 2), wq)
    with pytest.raises(ValueError, match="floating"):
        ql.linear(torch.zeros(2, K, dtype=torch.int8), wq)


def test_a16_multi_slab_bias_scale_and_rows(monkeypatch):
    """Real layers decode in several slabs; the slab-only slices (s_ch, bias, output columns, row offsets) must line up."""
    w = make_weight(40, 512, seed=21)
    monkeypatch.setattr(ql, "_DECODE_ELEMS", 3 * 512)  # 3 rows per slab -> 14 slabs, remainder of 1
    g = torch.Generator().manual_seed(22)
    x, b = torch.randn(7, 512, generator=g), torch.randn(40, generator=g)
    got = ql.linear(x, w, b, precision="a16")
    ref = ref_fp32(x, w) + b.double()
    assert rel(got, ref) < 1e-5
    full = ql.dequantize(w, torch.float32)
    part = ql.dequantize(w, torch.float32, rows=slice(3, 17))
    assert torch.allclose(part, full[3:17], rtol=1e-6, atol=1e-7)


def test_missing_comfy_kitchen_warns_once(monkeypatch):
    monkeypatch.setattr(ql, "_WARNED_NO_CK", False)
    with pytest.warns(RuntimeWarning, match="comfy_kitchen is unusable"):
        ql._warn_no_ck("boom")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ql._warn_no_ck("boom")  # second time: silent
