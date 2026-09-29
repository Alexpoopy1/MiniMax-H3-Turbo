"""Training utilities: task-mixed flow-matching batches for the omni transformer.

Batches are built with the same `build_segments` the pipeline uses, so what the model
is trained on is exactly the layout it sees at inference. Each step draws one
*structure* (which token groups exist); inside the "full" structure every sample gets
its own pin pattern, which is how one model learns text/image/first-last/extension/
video->audio/audio->video at once.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .layout import (
    GROUPS,
    RefTokens,
    Track,
    audio_positions,
    build_segments,
    group_times,
    video_positions,
)

STRUCTURES = ("full", "video", "audio", "ref", "refine")
# pin patterns inside "full": name -> probability
PATTERNS = {"none": 0.35, "first": 0.15, "first_last": 0.12, "prefix": 0.06, "video_full": 0.16, "audio_full": 0.16}
REFINE_MAX_SIGMA = 0.7


@dataclass
class Geo:
    """Token-grid geometry of one training batch."""

    lat_t: int
    hp: int
    wp: int
    fps: int
    lo_hp: int = 0  # low-res grid, for the refine structure
    lo_wp: int = 0


def sample_sigma(B: int, shift: float, gen: torch.Generator) -> torch.Tensor:
    u = torch.sigmoid(torch.randn(B, generator=gen))  # logit-normal
    return (shift * u / (1 + (shift - 1) * u)).clamp(1e-3, 1.0)


def sample_pins(B: int, geo: Geo, Na: int, rng: np.random.Generator):
    names = list(PATTERNS)
    choice = rng.choice(len(names), size=B, p=list(PATTERNS.values()))
    per_frame = geo.hp * geo.wp
    pin_v = torch.zeros(B, geo.lat_t, per_frame, dtype=torch.bool)
    pin_a = torch.zeros(B, Na, dtype=torch.bool)
    for b, c in enumerate(choice):
        n = names[c]
        if n == "first":
            pin_v[b, 0] = True
        elif n == "first_last":
            pin_v[b, 0] = pin_v[b, -1] = True
        elif n == "prefix":
            pin_v[b, : min(2, geo.lat_t - 1)] = True
        elif n == "video_full":
            pin_v[b] = True
        elif n == "audio_full":
            pin_a[b] = True
    return pin_v.view(B, -1), pin_a, [names[c] for c in choice]


def flow_loss(
    model,
    text_encoder,
    batch: Dict,
    structure: str,
    geo: Geo,
    rng: np.random.Generator,
    gen: torch.Generator,
    shift: float = 3.0,
    text_drop: float = 0.1,
    w_audio: float = 1.0,
):
    """batch: prompts (List[str]), video [B,Nv,D], audio [B,Na,D], optionally ref [B,Nr,D]
    (a clean reference image) and video_lo [B,Nl,D] (low-res clip, for "refine").
    `eps_video` / `eps_audio` pin the noise instead of drawing it: pairing each noise
    with the teacher's own output for it is what reflow (step distillation) trains on.
    Returns (loss, per-sample stats dict)."""
    dev = next(model.parameters()).device
    vid, aud = batch["video"].to(dev), batch["audio"].to(dev)
    B = vid.shape[0]
    prompts = ["" if rng.random() < text_drop else p for p in batch["prompts"]]
    ids, mask = text_encoder.tokenizer.batch(prompts)
    ids, mask = ids.to(dev), mask.to(dev)
    text = text_encoder(ids, mask)
    L = ids.shape[1]
    lens = mask.sum(1)
    tpos = torch.zeros(B, L, 3, device=dev)
    tpos[..., 0] = torch.arange(L, device=dev)[None] - lens[:, None]

    sigma = sample_sigma(B, shift, gen).to(dev)
    if structure == "refine":
        sigma = sigma * REFINE_MAX_SIGMA
    s = sigma[:, None, None]

    def noisy(x0, key):
        given = batch.get("eps_" + key)
        eps = given.to(dev) if given is not None else torch.randn(x0.shape, generator=gen).to(dev)
        return (1 - s) * x0 + s * eps, eps - x0

    Nv, Na = vid.shape[1], aud.shape[1]
    pin_v = torch.zeros(B, Nv, dtype=torch.bool, device=dev)
    pin_a = torch.zeros(B, Na, dtype=torch.bool, device=dev)
    names = ["-"] * B
    if structure == "full":
        pv, pa, names = sample_pins(B, geo, Na, rng)
        pin_v, pin_a = pv.to(dev), pa.to(dev)

    pos_v = video_positions(geo.lat_t, geo.hp, geo.wp, device=dev)
    pos_a = audio_positions(Na, geo.fps, device=dev)
    xv, tv = noisy(vid, "video")
    xa, ta = noisy(aud, "audio")
    xv = torch.where(pin_v[..., None], vid, xv)
    xa = torch.where(pin_a[..., None], aud, xa)

    refs: List[RefTokens] = []
    video = Track(xv, pos_v, pin_v, True) if structure != "audio" else None
    audio = Track(xa, pos_a, pin_a, True) if structure not in ("video", "refine") else None
    if structure == "ref":
        r = batch["ref"].to(dev)
        rp = video_positions(1, geo.hp, geo.wp, t0=geo.lat_t + 1.0, device=dev)
        refs.append(RefTokens("video", r, rp))
    if structure == "refine":
        lo = batch["video_lo"].to(dev)
        rp = video_positions(geo.lat_t, geo.lo_hp, geo.lo_wp, device=dev)
        rp[..., 1] *= geo.hp / geo.lo_hp
        rp[..., 2] *= geo.wp / geo.lo_wp
        refs.append(RefTokens("video", lo, rp))
        audio = Track(aud, pos_a, torch.ones(B, Na, dtype=torch.bool, device=dev), False)

    segs, index = build_segments(text, video, audio, refs, text_pos=tpos)
    key_mask = None
    if not bool(mask.all()):
        n_total = sum(sg.x.shape[1] for sg in segs)
        key_mask = torch.ones(B, n_total, dtype=torch.bool, device=dev)
        key_mask[:, :L] = mask
    gt = group_times(B, sigma, sigma, device=dev)
    outs = model(segs, GROUPS, gt, key_mask=key_mask)

    losses, per = [], {}
    for name, tr, target, weight in (("video", video, tv, 1.0), ("audio", audio, ta, w_audio)):
        if tr is None or not tr.emit:
            continue
        err = (outs[index[name]].float() - target).pow(2).mean(-1)  # [B, N]
        free = (~tr.pinned).float()
        per[name] = (err * free).sum(1) / free.sum(1).clamp(min=1)
        losses.append(weight * (err * free).sum() / free.sum().clamp(min=1))
    return sum(losses), {"pattern": names, **{k: v.detach().cpu() for k, v in per.items()}}


# --------------------------------------------------------------------------- optimisation helpers
class EMA:
    def __init__(self, params: Sequence[torch.nn.Parameter], decay: float = 0.995):
        self.params = list(params)
        self.shadow = [p.detach().clone() for p in self.params]
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for s, p in zip(self.shadow, self.params):
            s.lerp_(p.detach(), 1 - d)

    @torch.no_grad()
    def copy_to(self):
        for s, p in zip(self.shadow, self.params):
            p.copy_(s)


def cosine_lr(step: int, total: int, base: float, warmup: int = 100, floor: float = 0.1) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


# --------------------------------------------------------------------------- VAE losses
def stft_loss(pred: torch.Tensor, target: torch.Tensor, sizes=(256, 1024, 2048)) -> torch.Tensor:
    """Multi-resolution STFT loss: log-magnitude L1 plus spectral convergence. The log term
    alone averages over mostly-silent bins and is happy with quiet noise; spectral
    convergence (relative Frobenius error of the magnitudes) weights the bins that hold
    the energy. pred/target [B, S]."""
    loss = 0.0
    for n in sizes:
        win = torch.hann_window(n, device=pred.device)
        a = torch.stft(pred, n, n // 4, window=win, return_complex=True).abs()
        b = torch.stft(target, n, n // 4, window=win, return_complex=True).abs()
        log_term = (torch.log(a + 1e-3) - torch.log(b + 1e-3)).abs().mean()
        conv = (a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp(min=1e-6)
        loss = loss + log_term + conv.mean()
    return loss / len(sizes)


def kl_term(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar).mean()
