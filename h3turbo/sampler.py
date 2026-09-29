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
    """steps + 1 values from `start` down to exactly 0."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    return shift_sigmas(torch.linspace(start, 0.0, steps + 1), shift)


def euler_step(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    return x + (sigma_next - sigma) * v
