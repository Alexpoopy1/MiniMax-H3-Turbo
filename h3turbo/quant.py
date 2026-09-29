"""Weight-only int8 / int4 quantisation for the transformer blocks.

Why weight-only: a DiT runs thousands of tokens per forward, so it is compute-bound
and the dequantise-then-matmul overhead is tiny next to the GEMM, while VRAM drops 2x
(int8) or ~4x (int4). It is also the right choice for Ampere (RTX 30xx): those cards
have no FP8 tensor cores, and weight-only avoids quantising activations.

Rounding is plain round-to-nearest (per-output-channel for int8, per-group for int4);
there is no calibration, so int4 costs measurable quality on a trained model.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

INT4_GROUP = 64


class Int8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "Int8Linear":
        q = cls(lin.in_features, lin.out_features, lin.bias is not None)
        w = lin.weight.detach().float()
        scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
        q.qweight.copy_((w / scale).round().clamp(-127, 127).to(torch.int8))
        q.scale.copy_(scale.to(torch.float16))
        if lin.bias is not None:
            q.bias.data.copy_(lin.bias.detach())
        return q

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        return self.qweight.to(dtype) * self.scale.to(dtype)

    def forward(self, x):
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.dequantize(x.dtype), b)


class Int4Linear(nn.Module):
    """Symmetric 4-bit, group size 64, two values packed per byte (offset 8)."""

    def __init__(self, in_features: int, out_features: int, bias: bool, group: int = INT4_GROUP):
        super().__init__()
        if in_features % group:
            raise ValueError(f"in_features {in_features} not divisible by group {group}")
        self.in_features, self.out_features, self.group = in_features, out_features, group
        self.register_buffer("qweight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("scale", torch.ones(out_features, in_features // group, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear, group: int = INT4_GROUP) -> "Int4Linear":
        q = cls(lin.in_features, lin.out_features, lin.bias is not None, group)
        w = lin.weight.detach().float().view(lin.out_features, -1, group)
        scale = (w.abs().amax(dim=2, keepdim=True) / 7.0).clamp(min=1e-8)
        codes = (w / scale).round().clamp(-8, 7).add(8).to(torch.uint8).view(lin.out_features, -1)
        q.qweight.copy_(codes[:, 0::2] | (codes[:, 1::2] << 4))
        q.scale.copy_(scale.squeeze(-1).to(torch.float16))
        if lin.bias is not None:
            q.bias.data.copy_(lin.bias.detach())
        return q

    def dequantize(self, dtype=torch.float32) -> torch.Tensor:
        lo, hi = self.qweight & 0xF, self.qweight >> 4
        codes = torch.stack([lo, hi], dim=-1).view(self.out_features, -1, self.group)
        w = (codes.to(dtype) - 8) * self.scale.to(dtype).unsqueeze(-1)
        return w.view(self.out_features, self.in_features)

    def forward(self, x):
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.dequantize(x.dtype), b)


_MODES = {"int8": Int8Linear, "int4": Int4Linear}


def _replace(parent: nn.Module, mode: str, skip: Optional[Callable[[str], bool]], prefix: str) -> None:
    for name, child in list(parent.named_children()):
        full = f"{prefix}{name}"
        if isinstance(child, nn.Linear) and not (skip and skip(full)):
            if mode == "int4" and child.in_features % INT4_GROUP:
                continue  # leave odd-sized layers in high precision
            setattr(parent, name, _MODES[mode].from_linear(child))
        else:
            _replace(child, mode, skip, full + ".")


def quantize_(model: nn.Module, mode: str, skip: Optional[Callable[[str], bool]] = None) -> nn.Module:
    """In-place: swap nn.Linear for quantised layers. Apply to `transformer.blocks`."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {list(_MODES)}")
    _replace(model, mode, skip, "")
    return model


def quantize_structure_(model: nn.Module, mode: str) -> nn.Module:
    """Same swap, for building an empty model to load a pre-quantised checkpoint into.
    Works on meta tensors: the values are replaced by load_state_dict(assign=True)."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            if mode == "int4" and child.in_features % INT4_GROUP:
                continue
            cls = _MODES[mode]
            with torch.device("meta"):
                setattr(model, name, cls(child.in_features, child.out_features, child.bias is not None))
        else:
            quantize_structure_(child, mode)
    return model


def quantized_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters()) + sum(
        b.numel() * b.element_size() for b in model.buffers()
    )
