"""Text conditioning.

Built-in: a byte-level tokenizer + small transformer encoder. No vocabulary file,
no download, any language. For stronger semantics plug in your own encoder (a Qwen3
family LLM/VLM, T5, ...) through `ExternalTextAdapter` and pass `text_embeds`.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .config import TextEncoderConfig
from .layers import PlainBlock, RMSNorm, init_linear_, rope_cos_sin

PAD, BOS, EOS = 256, 257, 258
VOCAB = 259


class ByteTokenizer:
    def __init__(self, max_len: int = 256):
        self.max_len = max_len

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))[: self.max_len - 2]
        return [BOS] + ids + [EOS]

    def batch(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = [self.encode(t) for t in texts]
        n = max(len(e) for e in enc)
        ids = torch.full((len(enc), n), PAD, dtype=torch.long)
        mask = torch.zeros(len(enc), n, dtype=torch.bool)
        for i, e in enumerate(enc):
            ids[i, : len(e)] = torch.tensor(e)
            mask[i, : len(e)] = True
        return ids, mask


class ByteTextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = ByteTokenizer(cfg.max_len)
        self.embed = nn.Embedding(VOCAB, cfg.dim, padding_idx=PAD)
        self.blocks = nn.ModuleList(PlainBlock(cfg.dim, cfg.heads, cfg.dim * 8 // 3 // 8 * 8) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.dim)
        self.axes = (cfg.dim // cfg.heads, 0, 0)  # 1D rope on the "t" axis
        nn.init.normal_(self.embed.weight, std=0.02)
        init_linear_(self.blocks)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N = ids.shape
        pos = torch.zeros(B, N, 3, device=ids.device)
        pos[..., 0] = torch.arange(N, device=ids.device)
        cos, sin = rope_cos_sin(pos, self.axes, 10000.0)
        x = self.embed(ids)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        # padded queries attend to something valid, so no NaNs; their outputs are ignored
        km = mask if not bool(mask.all()) else None
        for b in self.blocks:
            x = b(x, cos, sin, km)
        return self.norm(x)

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str], device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.embed.weight.device
        ids, mask = self.tokenizer.batch(texts)
        ids, mask = ids.to(device), mask.to(device)
        return self(ids, mask), mask


class ExternalTextAdapter(nn.Module):
    """Project hidden states of any external encoder into the transformer's text space."""

    def __init__(self, ext_dim: int, text_dim: int):
        super().__init__()
        self.norm = RMSNorm(ext_dim)
        self.proj = nn.Linear(ext_dim, text_dim)
        init_linear_(self.proj)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(embeds))


def build_text_encoder(cfg: TextEncoderConfig) -> nn.Module:
    if cfg.kind == "byte":
        return ByteTextEncoder(cfg)
    if cfg.kind == "external":
        return ExternalTextAdapter(cfg.ext_dim, cfg.dim)
    raise ValueError(f"unknown text encoder kind {cfg.kind!r}")
