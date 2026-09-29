"""Token layout shared by training and inference.

Everything the model sees is described by a few `Track`s (target-timeline tokens per
modality, with a pin mask) plus context tokens. Every omni task is a choice of pin
mask, so there is one code path for all of them:

    text -> video+audio      nothing pinned
    image -> video           first latent frame pinned
    first+last -> video      first and last latent frames pinned
    video -> audio           whole video pinned, audio free
    audio -> video           whole audio pinned, video free
    extension / continuation prefix frames pinned
    inpainting / editing     pin mask over part of the frame
    reference -> video       clean reference tokens appended, off the timeline
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .config import (
    AUDIO_HOP,
    AUDIO_LATENT_RATE,
    PIXELS_PER_TOKEN,
    VIDEO_PATCH,
    VIDEO_TEMPORAL,
)
from .model import Segment

# fixed group table: (modality, clean?) -> index. Noisy groups get the sampler's t,
# clean groups (pinned / reference tokens) always t=0.
GROUPS = ["text", "video", "video", "audio", "audio"]
G_TEXT, G_VIDEO, G_VIDEO_CLEAN, G_AUDIO, G_AUDIO_CLEAN = range(5)


# --------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    num_frames: int  # pixel frames, 1 + 4k
    fps: int

    def __post_init__(self):
        if self.width % PIXELS_PER_TOKEN or self.height % PIXELS_PER_TOKEN:
            raise ValueError(f"width/height must be multiples of {PIXELS_PER_TOKEN}, got {self.width}x{self.height}")
        if (self.num_frames - 1) % VIDEO_TEMPORAL:
            raise ValueError(f"num_frames must be 1 + {VIDEO_TEMPORAL}k, got {self.num_frames}")

    @property
    def lat_t(self) -> int:
        return 1 + (self.num_frames - 1) // VIDEO_TEMPORAL

    @property
    def hp(self) -> int:
        return self.height // PIXELS_PER_TOKEN

    @property
    def wp(self) -> int:
        return self.width // PIXELS_PER_TOKEN

    @property
    def video_tokens(self) -> int:
        return self.lat_t * self.hp * self.wp

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    @property
    def audio_tokens(self) -> int:
        return max(1, round(self.duration * AUDIO_LATENT_RATE))

    @property
    def audio_samples(self) -> int:
        return self.audio_tokens * AUDIO_HOP


def snap_frames(n: int) -> int:
    """Nearest valid frame count 1 + 4k (>= 1)."""
    return 1 + VIDEO_TEMPORAL * max(0, round((n - 1) / VIDEO_TEMPORAL))


def snap_size(v: int) -> int:
    return max(PIXELS_PER_TOKEN, round(v / PIXELS_PER_TOKEN) * PIXELS_PER_TOKEN)


def latent_index_of_frame(frame: int) -> int:
    """Latent frame holding pixel frame `frame` (frame 0 is latent 0 on its own)."""
    return 0 if frame <= 0 else math.ceil(frame / VIDEO_TEMPORAL)


# --------------------------------------------------------------------------- patchify
def patchify_video(z: torch.Tensor) -> torch.Tensor:
    """[B, C, T, H, W] latent -> [B, T*(H/2)*(W/2), C*pt*ph*pw] tokens, order (t, h, w)."""
    pt, ph, pw = VIDEO_PATCH
    B, C, T, H, W = z.shape
    z = z.view(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
    z = z.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return z.reshape(B, (T // pt) * (H // ph) * (W // pw), C * pt * ph * pw)


def unpatchify_video(x: torch.Tensor, C: int, T: int, H: int, W: int) -> torch.Tensor:
    pt, ph, pw = VIDEO_PATCH
    B = x.shape[0]
    x = x.view(B, T // pt, H // ph, W // pw, C, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(B, C, T, H, W)


def video_positions(T: int, hp: int, wp: int, t0: float = 0.0, hw_scale: float = 1.0, device=None) -> torch.Tensor:
    t = torch.arange(T, dtype=torch.float32, device=device) + t0
    h = torch.arange(hp, dtype=torch.float32, device=device) * hw_scale
    w = torch.arange(wp, dtype=torch.float32, device=device) * hw_scale
    grid = torch.stack(torch.meshgrid(t, h, w, indexing="ij"), dim=-1)
    return grid.reshape(1, -1, 3)


def audio_positions(n: int, fps: int, t0: float = 0.0, device=None) -> torch.Tensor:
    """Audio tokens live on the video time axis: token i is at i/40 s, and one latent
    video frame spans 4/fps s, so t = i * fps / (4 * 40)."""
    t = torch.arange(n, dtype=torch.float32, device=device) * (fps / (VIDEO_TEMPORAL * AUDIO_LATENT_RATE)) + t0
    return torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1)[None]


def text_positions(n: int, device=None) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32, device=device) - n
    return torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], dim=-1)[None]


# --------------------------------------------------------------------------- tracks
@dataclass
class Track:
    """Tokens of one modality on the target timeline.

    x       [B, N, D]  current tokens; free ones noisy, pinned ones already clean
    pos     [1, N, 3]
    pinned  [B, N] bool
    emit    tokens are being generated (the model output is used)
    """

    x: torch.Tensor
    pos: torch.Tensor
    pinned: torch.Tensor
    emit: bool = True


@dataclass
class RefTokens:
    modality: str  # "video" | "audio"
    x: torch.Tensor  # [B, n, D] clean tokens
    pos: torch.Tensor  # [1, n, 3]


def build_segments(
    text: Optional[torch.Tensor],
    video: Optional[Track],
    audio: Optional[Track],
    refs: Sequence[RefTokens] = (),
    resampled: Sequence[Segment] = (),
) -> Tuple[List[Segment], dict]:
    """-> (segments, index) where index maps 'video'/'audio' to their segment position."""
    segs: List[Segment] = []
    index = {}
    if text is not None:
        segs.append(
            Segment("text", text, text_positions(text.shape[1], text.device), torch.full(text.shape[:2], G_TEXT, device=text.device))
        )
    for r in refs:
        g = G_VIDEO_CLEAN if r.modality == "video" else G_AUDIO_CLEAN
        segs.append(Segment(r.modality, r.x, r.pos, torch.full(r.x.shape[:2], g, device=r.x.device)))
    segs.extend(resampled)
    for name, tr, (gfree, gclean) in (("video", video, (G_VIDEO, G_VIDEO_CLEAN)), ("audio", audio, (G_AUDIO, G_AUDIO_CLEAN))):
        if tr is None:
            continue
        group = torch.where(tr.pinned, gclean, gfree)
        index[name] = len(segs)
        segs.append(Segment(name, tr.x, tr.pos, group, emit=tr.emit))
    return segs, index


def group_times(B: int, sigma_video, sigma_audio, device=None) -> torch.Tensor:
    """[B, 5] timesteps (0..1000) for GROUPS; clean and text groups are always 0.
    Sigmas may be floats or [B] tensors."""
    t = torch.zeros(B, len(GROUPS), device=device)
    t[:, G_VIDEO] = torch.as_tensor(sigma_video, dtype=torch.float32, device=device) * 1000.0
    t[:, G_AUDIO] = torch.as_tensor(sigma_audio, dtype=torch.float32, device=device) * 1000.0
    return t
