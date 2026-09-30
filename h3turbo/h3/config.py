"""Architecture description of the official H3 DiT, inferred from checkpoint tensor shapes."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class H3Config:
    hidden: int = 5376
    layers: int = 50
    refiner_layers: int = 2
    heads: int = 56
    head_dim: int = 128
    ffn: int = 14336  # SwiGLU width; fc1 emits 2 * ffn
    video_channels: int = 24
    audio_channels: int = 32
    patch: Tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    t_dim: int = 8  # width of the AdaLN input (curve coordinates for curve-form checkpoints)
    curve_grid: int = 1025  # rows of adaln_t_table; 0 = the checkpoint carries a time embedder instead
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0
    quant_group: int = 16  # W4A8: weights per fp8 group scale
    quant_convrot: int = 256  # W4A8: ConvRot (regular Hadamard) block along K

    @property
    def video_patch_dim(self) -> int:
        return self.video_channels * self.patch[0] * self.patch[1] * self.patch[2]

    @property
    def inner(self) -> int:
        return self.heads * self.head_dim

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "H3Config":
        d = json.loads(s)
        d["patch"] = tuple(d["patch"])
        return cls(**d)

    @classmethod
    def from_shapes(cls, shapes: Mapping[str, Sequence[int]], sigma_shifts: Optional[Tuple[float, float]] = None) -> "H3Config":
        """Infer the config from {tensor name: shape} of a ComfyUI-layout H3 checkpoint.

        Row counts are used (they are unaffected by 4-bit packing, which only halves K).
        """

        def count(prefix: str) -> int:
            idx = {int(m.group(1)) for k in shapes for m in [re.match(re.escape(prefix) + r"(\d+)\.", k)] if m}
            return (max(idx) + 1) if idx else 0

        head_dim = shapes["blocks.0.attn.q_norm.weight"][0]
        vpp = shapes["video_patch_proj.weight"]
        patch = (1, 2, 2)
        table = shapes.get("adaln_t_table")
        kw = dict(
            hidden=vpp[0],
            layers=count("blocks."),
            refiner_layers=count("token_refiner.blocks."),
            heads=shapes["blocks.0.attn.qkv_proj.weight"][0] // (3 * head_dim),
            head_dim=head_dim,
            ffn=shapes["blocks.0.mlp.fc1.weight"][0] // 2,
            video_channels=shapes["final_layer.video_out.weight"][0] // (patch[0] * patch[1] * patch[2]),
            audio_channels=shapes["final_layer.audio_out.weight"][0],
            patch=patch,
            text_dim=shapes["condition_proj.weight"][1],
            rope_inv_freq_len=shapes["rope.inv_freq"][0],
        )
        if table is not None:
            kw["curve_grid"], kw["t_dim"] = table[0], table[1]
        else:
            kw["curve_grid"] = 0
            kw["t_dim"] = shapes["time_embedder.proj_out.weight"][0]
        if sigma_shifts:
            kw["sigma_shift_video"], kw["sigma_shift_audio"] = sigma_shifts
        return cls(**kw)
