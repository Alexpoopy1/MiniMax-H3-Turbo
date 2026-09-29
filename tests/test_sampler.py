import pytest
import torch

from h3turbo.sampler import euler_step, flow_sigmas


@pytest.mark.parametrize("shift", [1.0, 3.0, 7.0])
@pytest.mark.parametrize("start", [1.0, 0.6, 0.25])
def test_schedule_starts_where_the_sample_really_is_and_ends_at_zero(shift, start):
    s = flow_sigmas(4, shift, start)
    assert s[0].item() == pytest.approx(start, abs=1e-6)
    assert s[-1].item() == 0.0
    assert len(s) == 5 and bool((s[:-1] > s[1:]).all())


def test_refine_schedule_regression():
    """Refine used to lay a linear grid at `strength` and then shift it, so it began at
    ~0.82 for strength 0.6 while the video had been noised to 0.6."""
    assert flow_sigmas(2, 3.0, start=0.6)[0].item() == pytest.approx(0.6, abs=1e-6)


def test_shift_pushes_time_towards_high_noise():
    lo, hi = flow_sigmas(4, 1.0), flow_sigmas(4, 5.0)
    assert bool((hi[1:-1] > lo[1:-1]).all())


def test_euler_recovers_x0_exactly_for_a_straight_flow():
    """If v = noise - x0 is exact, Euler over any schedule lands on x0."""
    x0, eps = torch.randn(8), torch.randn(8)
    sig = flow_sigmas(3, 3.0)
    x = eps.clone()
    for i in range(3):
        x = euler_step(x, eps - x0, float(sig[i]), float(sig[i + 1]))
    assert torch.allclose(x, x0, atol=1e-5)
    with pytest.raises(ValueError):
        flow_sigmas(0)
