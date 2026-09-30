"""Shared contracts between the W4A8 ops, the model, the store/converter and the streaming engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Union

import torch


@dataclass
class W4A8Weight:
    """One ConvRot W4A8 linear weight (out = x @ W.T), all tensors on the same device.

    Physical weight (in the ConvRot-rotated basis) is
        W_rot[n, k] = round(codebook[nibble[n, k]] * s_rel[n, k // group_size]).clamp(-127, 127) * s_ch[n]
    where nibble[n, 2j] is the LOW nibble and nibble[n, 2j+1] the HIGH nibble of q[n, j].
    The activation is rotated by the same regular-Hadamard block (size `convrot`) before the matmul.
    """

    q: torch.Tensor  # int8 [N, K // 2]
    s_rel: torch.Tensor  # float8_e4m3fn [N, K // group_size]
    s_ch: torch.Tensor  # float32 [N]
    codebook: torch.Tensor  # float32 [16]
    group_size: int = 16
    convrot: int = 256

    @property
    def out_features(self) -> int:
        return self.q.shape[0]

    @property
    def in_features(self) -> int:
        return self.q.shape[1] * 2

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.q, self.s_rel, self.s_ch, self.codebook))

    def to(self, device, non_blocking: bool = False) -> "W4A8Weight":
        return W4A8Weight(
            self.q.to(device, non_blocking=non_blocking),
            self.s_rel.to(device, non_blocking=non_blocking),
            self.s_ch.to(device, non_blocking=non_blocking),
            self.codebook.to(device, non_blocking=non_blocking),
            self.group_size,
            self.convrot,
        )


# a linear weight is either W4A8 or a dense [N, K] tensor (bf16/fp32)
Weight = Union[W4A8Weight, torch.Tensor]


@dataclass
class BlockWeights:
    """Everything one DiT block needs, on the compute device."""

    qkv: Weight  # [3*inner, hidden]
    out: Weight  # [hidden, inner]
    fc1: Weight  # [2*ffn, hidden]
    fc2: Weight  # [hidden, ffn]
    norm1: torch.Tensor  # [hidden]
    norm2: torch.Tensor  # [hidden]
    q_norm: torch.Tensor  # [head_dim]
    k_norm: torch.Tensor  # [head_dim]
    adaln_w: Optional[torch.Tensor] = None  # [6*hidden*3, t_dim]  (None for refiner blocks)
    adaln_b: Optional[torch.Tensor] = None  # [6*hidden*3]


@dataclass
class GlobalWeights:
    """Non-block tensors (small, kept resident) plus the token refiner."""

    video_patch_w: torch.Tensor  # fp32 [hidden, video_patch_dim]
    video_patch_b: torch.Tensor
    audio_patch_w: torch.Tensor  # fp32 [hidden, audio_channels]
    audio_patch_b: torch.Tensor
    condition_w: torch.Tensor  # bf16 [hidden, text_dim]
    condition_b: torch.Tensor
    adaln_t_table: Optional[torch.Tensor]  # fp32 [grid, t_dim]  (curve form)
    rope_inv_freq: torch.Tensor  # fp32 [rope_inv_freq_len]
    final_norm: torch.Tensor  # bf16 [hidden]
    final_adaln_w: torch.Tensor  # bf16 [2*hidden, t_dim]
    final_adaln_b: torch.Tensor  # fp32 [2*hidden]
    video_out_w: torch.Tensor  # fp32 [video_patch_dim * n_heads_bank, hidden]
    video_out_b: torch.Tensor
    audio_out_w: torch.Tensor  # fp32 [audio_channels * n, hidden]
    audio_out_b: torch.Tensor
    refiner: List[BlockWeights] = field(default_factory=list)  # adaln_* = None
    refiner_final_norm: Optional[torch.Tensor] = None


class BlockProvider(Protocol):
    """Hands the model the weights of block i on the compute device.

    `acquire` may block until an async copy has finished. The model calls, per forward:
    begin_forward(); for i in range(layers): w = acquire(i); ...use w...; release(i); end_forward().
    A resident provider just returns the same objects every time.
    """

    def begin_forward(self) -> None: ...

    def acquire(self, i: int) -> BlockWeights: ...

    def release(self, i: int) -> None: ...

    def end_forward(self) -> None: ...
