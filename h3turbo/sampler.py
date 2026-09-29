"""Rectified-flow sampling.

x_sigma = (1 - sigma) * x0 + sigma * noise, and the model predicts v = noise - x0.
An Euler step is x <- x + (sigma_next - sigma) * v. With a Turbo (step-distilled)
model 2-4 steps is the intended operating point; a plain model wants 20+.
"""
from __future__ import annotations

import torch


def shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    """Bias the schedule towards high noise (larger shift = more time at high sigma)."""
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def flow_sigmas(steps: int, shift: float = 3.0, start: float = 1.0) -> torch.Tensor:
    """steps + 1 sigmas from exactly `start` (the noise level the sample really has) down
    to exactly 0. For start < 1 (img2img-style refinement) the linear grid is laid out
    before the shift is applied, so the shift is inverted to land on `start`."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0.0 < start <= 1.0:
        raise ValueError("start must be in (0, 1]")
    u0 = start / (shift - (shift - 1) * start)
    return shift_sigmas(torch.linspace(u0, 0.0, steps + 1), shift)


def euler_step(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    return x + (sigma_next - sigma) * v
