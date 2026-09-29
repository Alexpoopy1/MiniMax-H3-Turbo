"""Synthetic omni dataset used to train and *measure* the shipped demo checkpoint.

A coloured square moves in one of four directions. The soundtrack ties the two
modalities together: the amplitude level encodes the colour, the frequency encodes the
direction.

The "audio" is a 3-12 Hz waveform, i.e. infrasound you cannot hear. That is deliberate:
a 40 Hz latent rate represents such a signal exactly, so a CPU-trained toy audio VAE can
learn it in minutes. Noise bursts and audible tones were tried first and a tiny VAE
learns neither (noise cannot be reconstructed from a compressed latent, so waveform
losses reward silence; a tone needs a phase-exact oscillator). The toy audio exists to
test cross-modal conditioning and the audio path end to end, not to sound like anything.
That makes every omni task checkable from its output alone:

    text  -> video+audio   colour/direction from the prompt, audio level consistent with colour
    image -> video         direction follows the prompt, colour matches the image
    first+last -> video    the motion agrees with the two pinned frames
    video -> audio         pitch/tremolo recovered from the picture
    audio -> video         colour/direction recovered from the sound
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import AUDIO_SAMPLE_RATE

COLORS = ["red", "green", "blue", "yellow"]
DIRECTIONS = ["left", "right", "up", "down"]
RGB = torch.tensor([[0.9, -0.8, -0.8], [-0.8, 0.9, -0.8], [-0.8, -0.8, 0.9], [0.9, 0.9, -0.8]])
BG = -0.7
LEVELS = [0.2, 0.4, 0.6, 0.8]  # colour -> peak amplitude
LFO_HZ = [3.0, 6.0, 9.0, 12.0]  # direction -> frequency
DIR_VEC = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]  # (dx, dy)
TRAVEL = 0.3  # fraction of the frame the square travels over the clip

TEMPLATES = ["a {c} square moving {d}", "{c} square going {d}"]


@dataclass
class Scene:
    color: int
    direction: int
    size: float  # square side as a fraction of the frame
    x0: float  # start centre, fraction of the frame
    y0: float
    template: int = 0

    @property
    def prompt(self) -> str:
        return TEMPLATES[self.template].format(c=COLORS[self.color], d=DIRECTIONS[self.direction])


def sample_scene(rng: np.random.Generator) -> Scene:
    color, direction = int(rng.integers(4)), int(rng.integers(4))
    size = float(rng.uniform(0.36, 0.48))
    dx, dy = DIR_VEC[direction]
    lo, hi = 0.5 * size + 0.04, 1 - 0.5 * size - 0.04
    x0 = float(rng.uniform(lo, hi)) if dx == 0 else float(rng.uniform(lo + TRAVEL, hi)) if dx < 0 else float(rng.uniform(lo, hi - TRAVEL))
    y0 = float(rng.uniform(lo, hi)) if dy == 0 else float(rng.uniform(lo + TRAVEL, hi)) if dy < 0 else float(rng.uniform(lo, hi - TRAVEL))
    return Scene(color, direction, size, x0, y0, int(rng.integers(len(TEMPLATES))))


def render(scene: Scene, side: int, T: int) -> torch.Tensor:
    """-> [T, 3, side, side] in [-1, 1]."""
    frames = torch.full((T, 3, side, side), BG)
    ys = torch.arange(side, dtype=torch.float32)[:, None] + 0.5
    xs = torch.arange(side, dtype=torch.float32)[None, :] + 0.5
    dx, dy = DIR_VEC[scene.direction]
    for t in range(T):
        f = t / max(T - 1, 1)
        cx, cy = (scene.x0 + dx * TRAVEL * f) * side, (scene.y0 + dy * TRAVEL * f) * side
        half = scene.size * side / 2
        m = ((xs - cx).abs() <= half) & ((ys - cy).abs() <= half)
        frames[t][:, m] = RGB[scene.color][:, None]
    return frames


def synth_audio(scene: Scene, samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sine at LFO_HZ[direction] with peak LEVELS[color] and a random phase."""
    if generator is None:  # deterministic per scene, so tests and caches are reproducible
        generator = torch.Generator().manual_seed(int(scene.x0 * 1e6) * 7919 + int(scene.y0 * 1e6) + scene.color * 13 + scene.direction)
    phase = float(torch.rand(1, generator=generator)) * 2 * math.pi
    t = torch.arange(samples, dtype=torch.float32) / AUDIO_SAMPLE_RATE
    return LEVELS[scene.color] * torch.sin(2 * math.pi * LFO_HZ[scene.direction] * t + phase)


# --------------------------------------------------------------------------- analysis
def analyze_video(video: torch.Tensor) -> Dict[str, Optional[int]]:
    """uint8 [T,H,W,3] -> {"color": idx, "direction": idx} estimated from pixels (None if no square)."""
    x = video.float() / 127.5 - 1.0  # [T,H,W,3]
    T, H, W, _ = x.shape
    mask = (x - BG).abs().amax(-1) > 0.6
    if mask.sum() < 4:
        return {"color": None, "direction": None}
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    cents = []
    for t in range(T):
        m = mask[t]
        cents.append((xs[m].mean().item(), ys[m].mean().item()) if m.sum() >= 4 else None)
    valid = [c for c in cents if c is not None]
    color = int(((x[mask].mean(0)[None] - RGB).abs().sum(-1)).argmin())
    k = max(1, len(valid) // 3)
    a = np.mean(valid[:k], axis=0)
    b = np.mean(valid[-k:], axis=0)
    dx, dy = b - a
    if abs(dx) < 0.02 * W and abs(dy) < 0.02 * H:
        return {"color": color, "direction": None}
    if abs(dx) >= abs(dy):
        direction = 1 if dx > 0 else 0
    else:
        direction = 3 if dy > 0 else 2
    return {"color": color, "direction": direction}


def analyze_audio(wave: torch.Tensor) -> Dict[str, Optional[int]]:
    """float [S] -> {"color": nearest amplitude level, "direction": nearest LFO frequency}."""
    w = wave.double()
    if w.abs().max() < 1e-3:
        return {"color": None, "direction": None}
    amp = w.pow(2).mean().sqrt().item() * math.sqrt(2)
    color = int(np.abs(np.array(LEVELS) - amp).argmin())
    n = len(w)
    spec = torch.fft.rfft((w - w.mean()) * torch.hann_window(n, dtype=torch.float64)).abs()
    freqs = torch.fft.rfftfreq(n, 1 / AUDIO_SAMPLE_RATE)
    band = (freqs > 1.5) & (freqs < 16)
    f = freqs[band][spec[band].argmax()].item()
    direction = int(np.abs(np.array(LFO_HZ) - f).argmin())
    return {"color": color, "direction": direction}


def scene_from_prompt(prompt: str) -> Tuple[int, int]:
    words = prompt.split()
    return next(i for i, c in enumerate(COLORS) if c in words), next(i for i, d in enumerate(DIRECTIONS) if d in words)
