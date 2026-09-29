"""Configs for every H3-Turbo component, plus the size tiers.

The layout mirrors what is public about MiniMax H3: a single-stream dense
transformer whose attention/FFN are modality-agnostic, a causal video VAE with
16x spatial / 4x temporal compression and 24 latent channels, and an audio VAE
that emits 40 latent tokens per second of 32 kHz audio. The tiers only differ in
width/depth, so a tier is a one-line choice.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Tuple

VIDEO_SPATIAL = 16  # VAE spatial compression
VIDEO_TEMPORAL = 4  # VAE temporal compression
VIDEO_PATCH = (1, 2, 2)  # transformer patchify over (t, h, w) of the latent
PIXELS_PER_TOKEN = VIDEO_SPATIAL * VIDEO_PATCH[1]  # 32: frame sides must be multiples of this
AUDIO_SAMPLE_RATE = 32000
AUDIO_LATENT_RATE = 40  # latent tokens per second
AUDIO_HOP = AUDIO_SAMPLE_RATE // AUDIO_LATENT_RATE  # 800 samples per token


@dataclass
class ModalitySpec:
    """One token type. Registering a new one adds an input layer, an output layer
    (if out_dim > 0) and an AdaLN branch; attention and FFN are shared."""

    name: str
    in_dim: int
    out_dim: int = 0


@dataclass
class TransformerConfig:
    hidden: int
    layers: int
    heads: int
    ffn_dim: int
    rope_axes: Tuple[int, int, int]  # rotary dims for (t, h, w); must sum to head_dim
    text_dim: int
    video_in_dim: int = 24 * VIDEO_PATCH[0] * VIDEO_PATCH[1] * VIDEO_PATCH[2]
    audio_in_dim: int = 32
    rope_theta: float = 10000.0
    adaln_rank: int = 256  # 0 = full-rank modulation
    qk_norm: bool = True
    guidance_embed: bool = False  # True: CFG is distilled in, one forward per step
    extra_modalities: List[ModalitySpec] = field(default_factory=list)
    resampler_queries: int = 0  # >0 adds the context resampler
    resampler_layers: int = 2

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def modalities(self) -> List[ModalitySpec]:
        base = [
            ModalitySpec("text", self.text_dim, 0),
            ModalitySpec("video", self.video_in_dim, self.video_in_dim),
            ModalitySpec("audio", self.audio_in_dim, self.audio_in_dim),
        ]
        return base + list(self.extra_modalities)

    def validate(self) -> None:
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")
        if sum(self.rope_axes) != self.head_dim:
            raise ValueError(f"rope_axes {self.rope_axes} must sum to head_dim {self.head_dim}")
        if any(a % 2 for a in self.rope_axes):
            raise ValueError("rope axes must be even")


@dataclass
class VideoVAEConfig:
    channels: Tuple[int, ...] = (32, 64, 128, 256, 256)  # 5 entries = 4 downsamples = 16x
    blocks: int = 2
    latent_ch: int = 24
    temporal_down: Tuple[bool, ...] = (False, False, True, True)  # 4x temporal overall
    t_kernel: Tuple[int, ...] = (1, 1, 3, 3, 3)  # temporal kernel per stage (cheap at high res)


@dataclass
class AudioVAEConfig:
    channels: Tuple[int, ...] = (32, 64, 128, 256, 256)
    strides: Tuple[int, ...] = (2, 4, 5, 5, 4)  # product must equal AUDIO_HOP
    latent_ch: int = 32
    blocks: int = 1


@dataclass
class TextEncoderConfig:
    kind: str = "byte"  # "byte": built-in, no tokenizer download; "external": adapter over your own embeddings
    dim: int = 768
    layers: int = 8
    heads: int = 12
    max_len: int = 256
    ext_dim: int = 0  # embedding size of the external encoder (kind == "external")


@dataclass
class H3TurboConfig:
    name: str
    transformer: TransformerConfig
    video_vae: VideoVAEConfig
    audio_vae: AudioVAEConfig
    text: TextEncoderConfig
    fps: int = 24

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "H3TurboConfig":
        t = dict(d["transformer"])
        t["rope_axes"] = tuple(t["rope_axes"])
        t["extra_modalities"] = [ModalitySpec(**m) for m in t.get("extra_modalities", [])]
        v = {k: tuple(x) if isinstance(x, list) else x for k, x in d["video_vae"].items()}
        a = {k: tuple(x) if isinstance(x, list) else x for k, x in d["audio_vae"].items()}
        return cls(
            name=d["name"],
            transformer=TransformerConfig(**t),
            video_vae=VideoVAEConfig(**v),
            audio_vae=AudioVAEConfig(**a),
            text=TextEncoderConfig(**d["text"]),
            fps=d.get("fps", 24),
        )

    @classmethod
    def from_json(cls, s: str) -> "H3TurboConfig":
        return cls.from_dict(json.loads(s))


# (hidden, layers, heads, ffn, rope_axes, text_enc(dim, layers, heads))
_TIERS = {
    # test-scale; the shipped demo checkpoint is this tier
    "nano": (192, 6, 6, 512, (8, 12, 12), (192, 3, 6)),
    # ~0.3B: 4 GB cards, near-instant
    "small": (1024, 16, 16, 2816, (16, 24, 24), (512, 6, 8)),
    # ~0.8B: 6 GB cards (RTX 3050 laptop)
    "base": (1536, 24, 24, 4096, (16, 24, 24), (768, 8, 12)),
    # ~2.3B: 8 GB cards (RTX 3050 desktop) with int8, or 12 GB+ in fp16
    "large": (2304, 32, 18, 6144, (32, 48, 48), (1024, 12, 16)),
    # ~4.3B: 12-16 GB cards, or 8 GB with int4
    "xl": (3072, 36, 24, 8192, (32, 48, 48), (1280, 16, 20)),
}


def tier_names() -> List[str]:
    return list(_TIERS)


def make_config(tier: str, **overrides) -> H3TurboConfig:
    """Build the config for a tier. Keyword overrides go to TransformerConfig."""
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {tier_names()}")
    hidden, layers, heads, ffn, axes, (tdim, tlayers, theads) = _TIERS[tier]
    small = tier == "nano"
    tcfg = TransformerConfig(
        hidden=hidden,
        layers=layers,
        heads=heads,
        ffn_dim=ffn,
        rope_axes=axes,
        text_dim=tdim,
        adaln_rank=64 if small else 256,
    )
    for k, v in overrides.items():
        setattr(tcfg, k, v)
    tcfg.validate()
    vv = VideoVAEConfig(channels=(16, 32, 48, 64, 64)) if small else VideoVAEConfig()
    av = AudioVAEConfig(channels=(16, 32, 48, 64, 64)) if small else AudioVAEConfig()
    text = TextEncoderConfig(dim=tdim, layers=tlayers, heads=theads)
    return H3TurboConfig(name=f"h3turbo-{tier}", transformer=tcfg, video_vae=vv, audio_vae=av, text=text)
