"""H3-Turbo omni transformer.

Single stream: every modality is a segment of tokens in one sequence, all attending
to each other. Attention and FFN are shared; what is modality-specific is confined
to the input projection, the output projection and the AdaLN branch, so a new
modality is a small addition (`add_modality`) rather than a new network.

Per-token conditioning is expressed through *groups*. A group is a (modality,
timestep) pair, e.g. "noisy video at t=0.7" or "clean video" (t=0). Every token
points at one group. Pinned/reference tokens sit in the clean group, so the model
can tell clean context from tokens it must denoise.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import ModalitySpec, TransformerConfig
from .layers import (
    AdaLNBank,
    Block,
    CrossAttention,
    PlainBlock,
    RMSNorm,
    SwiGLU,
    init_linear_,
    modulate,
    rope_cos_sin,
)


class Segment(NamedTuple):
    modality: str
    x: torch.Tensor  # [B, n, in_dim] raw features (or [B, n, hidden] if embedded)
    pos: torch.Tensor  # [B or 1, n, 3] float (t, h, w)
    group: torch.Tensor  # [B, n] long, index into the forward's groups
    emit: bool = False  # produce an output for these tokens
    embedded: bool = False  # x is already in hidden space (resampled context)


class ContextResampler(nn.Module):
    """Perceiver-style compression of an arbitrary number of context tokens into a
    fixed budget. This is the Turbo analogue of H3's contextual omni representation:
    long multi-reference contexts stay a constant, short number of tokens."""

    def __init__(self, hidden: int, heads: int, ffn_dim: int, n_queries: int, layers: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, hidden) * 0.02)
        self.norm_ctx = RMSNorm(hidden)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                dict(
                    n1=RMSNorm(hidden),
                    xattn=CrossAttention(hidden, heads),
                    n2=RMSNorm(hidden),
                    ffn=SwiGLU(hidden, ffn_dim),
                )
            )
            for _ in range(layers)
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        q = self.queries[None].expand(ctx.shape[0], -1, -1).to(ctx.dtype)
        ctx = self.norm_ctx(ctx)
        for l in self.layers:
            q = q + l["xattn"](l["n1"](q), ctx)
            q = q + l["ffn"](l["n2"](q))
        return q


class H3TurboTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        H = cfg.hidden
        mods = cfg.modalities()
        self.in_proj = nn.ModuleDict({m.name: nn.Linear(m.in_dim, H) for m in mods})
        self.out_proj = nn.ModuleDict({m.name: nn.Linear(H, m.out_dim) for m in mods if m.out_dim > 0})
        self.blocks = nn.ModuleList(Block(H, cfg.heads, cfg.ffn_dim, cfg.qk_norm) for _ in range(cfg.layers))
        self.final_norm = RMSNorm(H, affine=False)
        self.adaln = AdaLNBank(H, cfg.layers, cfg.adaln_rank, [m.name for m in mods], cfg.guidance_embed)
        self.resampler = (
            ContextResampler(H, cfg.heads, cfg.ffn_dim, cfg.resampler_queries, cfg.resampler_layers)
            if cfg.resampler_queries > 0
            else None
        )
        self.gradient_checkpointing = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_linear_(self.blocks)
        init_linear_(self.in_proj)
        for m in self.out_proj.values():  # DiT-style zero head: initial velocity is 0
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        if self.resampler is not None:
            init_linear_(self.resampler)

    # ------------------------------------------------------------------ modalities
    def add_modality(self, spec: ModalitySpec, like: str = "video") -> None:
        """Register a new token type after the fact (depth, pose, mask, 3D, ...).
        Only new IO layers and an AdaLN branch are created; shared weights are reused,
        so it can be trained by itself with everything else frozen."""
        if spec.name in self.in_proj:
            raise ValueError(f"modality {spec.name!r} already exists")
        H = self.cfg.hidden
        dev = self.in_proj["video"].weight.device
        self.in_proj[spec.name] = nn.Linear(spec.in_dim, H).to(dev)
        nn.init.xavier_uniform_(self.in_proj[spec.name].weight)
        nn.init.zeros_(self.in_proj[spec.name].bias)
        if spec.out_dim > 0:
            head = nn.Linear(H, spec.out_dim).to(dev)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.out_proj[spec.name] = head
        bank_dev = next(self.adaln.parameters()).device
        self.adaln.add_modality(spec.name, like)
        self.adaln.to(bank_dev)
        self.cfg.extra_modalities.append(spec)

    def freeze_shared(self) -> List[str]:
        """Freeze everything except modality-specific weights; returns trainable names."""
        for p in self.parameters():
            p.requires_grad = False
        trainable = []
        extras = {m.name for m in self.cfg.extra_modalities}
        for name, p in self.named_parameters():
            if any(f".{e}." in f".{name}." for e in extras):
                p.requires_grad = True
                trainable.append(name)
        return trainable

    # ------------------------------------------------------------------ modulation
    def compute_mods(self, groups: Sequence[str], group_t: torch.Tensor, guidance=None, dtype=None, device=None):
        """Run the AdaLN bank. The bank may live on a different device than the
        blocks (CPU by default at inference); results are moved/cast to match them."""
        bank_dev = next(self.adaln.parameters()).device
        ref = self.in_proj["video"].weight  # IO layers are never quantised, so this is reliable
        dtype = dtype or ref.dtype
        device = device or ref.device
        g = None if guidance is None else guidance.to(bank_dev)
        mods, fin = self.adaln(groups, group_t.to(bank_dev), g)
        return mods.to(device=device, dtype=dtype), fin.to(device=device, dtype=dtype)

    def resample(self, seg: Segment, group: torch.Tensor, pos: torch.Tensor) -> Segment:
        """Compress a context segment to `resampler_queries` tokens (hidden space)."""
        if self.resampler is None:
            raise RuntimeError("model was built without a context resampler")
        h = seg.x if seg.embedded else self.in_proj[seg.modality](seg.x)
        return Segment(seg.modality, self.resampler(h), pos, group, emit=False, embedded=True)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        segments: Sequence[Segment],
        groups: Sequence[str],
        group_t: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
        mods: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> List[Optional[torch.Tensor]]:
        """
        segments : token segments, concatenated in order.
        groups   : modality name of each group (G,).
        group_t  : [B, G] timestep of each group on a 0..1000 scale.
        mods     : optional precomputed `compute_mods(...)` result (inference fast path).
        key_mask : optional [B, N_total] bool, False = ignore (padding).
        Returns one entry per segment: predicted velocity if `emit`, else None.
        """
        B = segments[0].x.shape[0]
        hs = [s.x if s.embedded else self.in_proj[s.modality](s.x) for s in segments]
        x = torch.cat(hs, dim=1)
        pos = torch.cat([s.pos.expand(B, -1, -1) for s in segments], dim=1)
        gidx = torch.cat([s.group for s in segments], dim=1)
        cos, sin = rope_cos_sin(pos, self.cfg.rope_axes, self.cfg.rope_theta)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        if mods is None:
            mods = self.compute_mods(groups, group_t, guidance, dtype=x.dtype, device=x.device)
        mods_all, fin = mods
        bidx = torch.arange(B, device=x.device)[:, None]
        for l, blk in enumerate(self.blocks):
            m = mods_all[:, :, l][bidx, gidx]  # [B, N, 6, H]
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, m, cos, sin, key_mask, use_reentrant=False)
            else:
                x = blk(x, m, cos, sin, key_mask)
        outs: List[Optional[torch.Tensor]] = []
        off = 0
        for s in segments:
            n = s.x.shape[1]
            if s.emit:
                f = fin[bidx, s.group]  # [B, n, 2, H]
                h = modulate(self.final_norm(x[:, off : off + n]), f[:, :, 0], f[:, :, 1])
                outs.append(self.out_proj[s.modality](h))
            else:
                outs.append(None)
            off += n
        return outs

    # ------------------------------------------------------------------ inference placement
    def to_inference(self, device, dtype, adaln_device="cpu"):
        """Blocks/IO on `device` in `dtype`; the AdaLN bank stays in fp32 on `adaln_device`
        (its outputs are cached per schedule, so it never needs GPU memory)."""
        self.adaln.to(device=adaln_device, dtype=torch.float32)
        for name, mod in self.named_children():
            if name != "adaln":
                mod.to(device=device, dtype=dtype)
        return self

    def num_params(self, include_adaln: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not include_adaln:
            total -= sum(p.numel() for p in self.adaln.parameters())
        return total
