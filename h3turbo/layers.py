"""Shared building blocks: RMSNorm, 3D RoPE, attention, SwiGLU, AdaLN bank."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(x.dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


# --------------------------------------------------------------------------- RoPE
def rope_cos_sin(pos: torch.Tensor, axes: Sequence[int], theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """pos: [B, N, 3] float (t, h, w). Returns cos/sin of shape [B, N, 1, head_dim/2].

    Positions are floats on purpose: audio tokens sit at fractional video-time
    coordinates and low-res context tokens are rescaled onto the hi-res grid."""
    angles = []
    for a, d in enumerate(axes):
        if d == 0:
            continue
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=pos.device, dtype=torch.float32) / d))
        angles.append(pos[..., a : a + 1].float() * inv)
    ang = torch.cat(angles, dim=-1)
    return ang.cos()[:, :, None, :], ang.sin()[:, :, None, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, N, H, D] -> same, rotating interleaved pairs."""
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.to(x.dtype)


# --------------------------------------------------------------------------- attention / FFN
class Attention(nn.Module):
    """Fused-QKV self attention with optional QK-norm, run through SDPA (flash on Ampere)."""

    def __init__(self, hidden: int, heads: int, qk_norm: bool = True):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, cos, sin, key_mask: Optional[torch.Tensor] = None):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask = None if key_mask is None else key_mask[:, None, None, :]
        o = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        )
        return self.proj(o.transpose(1, 2).reshape(B, N, -1))


class CrossAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.kv = nn.Linear(hidden, 2 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, ctx):
        B, N, _ = x.shape
        M = ctx.shape[1]
        q = self.q(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(ctx).view(B, M, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, -1))


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, ffn_dim: int):
        super().__init__()
        self.w13 = nn.Linear(hidden, 2 * ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x):
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


class Block(nn.Module):
    """DiT block. Modulation (shift/scale/gate x2) is passed in, per token, so the
    block itself holds no modality-specific weights."""

    def __init__(self, hidden: int, heads: int, ffn_dim: int, qk_norm: bool):
        super().__init__()
        self.norm1 = RMSNorm(hidden, affine=False)
        self.attn = Attention(hidden, heads, qk_norm)
        self.norm2 = RMSNorm(hidden, affine=False)
        self.ffn = SwiGLU(hidden, ffn_dim)

    def forward(self, x, mod, cos, sin, key_mask=None):
        s1, c1, g1, s2, c2, g2 = mod.unbind(2)
        x = x + g1 * self.attn(modulate(self.norm1(x), s1, c1), cos, sin, key_mask)
        x = x + g2 * self.ffn(modulate(self.norm2(x), s2, c2))
        return x


class PlainBlock(nn.Module):
    """Un-modulated pre-norm block, used by the text encoder and the resampler."""

    def __init__(self, hidden: int, heads: int, ffn_dim: int, qk_norm: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = Attention(hidden, heads, qk_norm)
        self.norm2 = RMSNorm(hidden)
        self.ffn = SwiGLU(hidden, ffn_dim)

    def forward(self, x, cos, sin, key_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, key_mask)
        return x + self.ffn(self.norm2(x))


# --------------------------------------------------------------------------- AdaLN
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[..., None].float() * freqs
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class AdaLNBranch(nn.Module):
    """cond vector -> n_out modulation vectors, optionally through a low-rank bottleneck.
    Zero-initialised output (adaLN-Zero) so every block starts as the identity."""

    def __init__(self, hidden: int, n_out: int, rank: int):
        super().__init__()
        self.n_out = n_out
        self.hidden = hidden
        self.down = nn.Linear(hidden, rank) if rank > 0 else None
        self.up = nn.Linear(rank if rank > 0 else hidden, n_out * hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        c = F.silu(c)
        if self.down is not None:
            c = F.silu(self.down(c))
        return self.up(c).unflatten(-1, (self.n_out, self.hidden))


class AdaLNBank(nn.Module):
    """All modulation parameters: time/guidance embedders and one branch per
    (layer, modality). Because modulation depends only on (modality, timestep,
    guidance) it can be computed once per sampling schedule and cached, after which
    none of these weights need to live on the GPU (they hold ~30% of the params)."""

    def __init__(self, hidden: int, layers: int, rank: int, modalities: List[str], guidance: bool):
        super().__init__()
        self.hidden, self.rank = hidden, rank
        self.time = TimestepEmbedder(hidden)
        self.guidance = TimestepEmbedder(hidden) if guidance else None
        self.layers = nn.ModuleList(
            nn.ModuleDict({m: AdaLNBranch(hidden, 6, rank) for m in modalities}) for _ in range(layers)
        )
        self.final = nn.ModuleDict({m: AdaLNBranch(hidden, 2, rank) for m in modalities})

    def add_modality(self, name: str, like: str) -> None:
        """New branch initialised as a copy of an existing one."""
        import copy

        for layer in self.layers:
            layer[name] = copy.deepcopy(layer[like])
        self.final[name] = copy.deepcopy(self.final[like])

    def cond(self, group_t: torch.Tensor, guidance: Optional[torch.Tensor]) -> torch.Tensor:
        """group_t: [B, G] in [0, 1000]; guidance: [B] or None -> [B, G, H]."""
        c = self.time(group_t)
        if self.guidance is not None and guidance is not None:
            c = c + self.guidance(guidance * 1000.0)[:, None, :]
        return c

    def forward(self, group_mods: Sequence[str], group_t, guidance=None):
        """-> (mods [B, G, L, 6, H], final [B, G, 2, H])"""
        c = self.cond(group_t, guidance)
        mods = torch.stack(
            [
                torch.stack([layer[m](c[:, g]) for layer in self.layers], dim=1)
                for g, m in enumerate(group_mods)
            ],
            dim=1,
        )
        fin = torch.stack([self.final[m](c[:, g]) for g, m in enumerate(group_mods)], dim=1)
        return mods, fin


def init_linear_(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
