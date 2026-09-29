"""Causal audio VAE: 32 kHz mono -> 40 latent tokens/s (hop 800), 32 channels."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AUDIO_HOP, AudioVAEConfig


class ChannelRMSNorm1d(nn.Module):
    """Per-timestep RMSNorm over channels: causal, and keeps activations at a stable scale
    through the deep stack (without it the input signal has all but vanished by the latent
    and the encoder learns to ignore it)."""

    def __init__(self, ch: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ch, 1))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, kernel, stride=1, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation + 1 - stride  # left-only => causal, T_out = T/stride
        self.conv = nn.Conv1d(cin, cout, kernel, stride, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class ResUnit(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.n1, self.n2 = ChannelRMSNorm1d(ch), ChannelRMSNorm1d(ch)
        self.c1 = CausalConv1d(ch, ch, 3, dilation=dilation)
        self.c2 = CausalConv1d(ch, ch, 1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class AudioVAE(nn.Module):
    def __init__(self, cfg: AudioVAEConfig):
        super().__init__()
        assert math.prod(cfg.strides) == AUDIO_HOP, f"strides must multiply to {AUDIO_HOP}"
        self.cfg = cfg
        ch, st = cfg.channels, cfg.strides
        dil = (1, 3, 9)[: min(cfg.blocks + 1, 3)]

        enc = [CausalConv1d(1, ch[0], 7)]
        for i, s in enumerate(st):
            enc += [ResUnit(ch[i], d) for d in dil]
            enc += [ChannelRMSNorm1d(ch[i]), nn.SiLU(), CausalConv1d(ch[i], ch[i + 1] if i + 1 < len(ch) else ch[-1], 2 * s, stride=s)]
        self.encoder = nn.Sequential(*enc)
        self.enc_norm = ChannelRMSNorm1d(ch[-1])
        self.enc_out = CausalConv1d(ch[-1], 2 * cfg.latent_ch, 3)

        dec = [CausalConv1d(cfg.latent_ch, ch[-1], 7)]
        for i in reversed(range(len(st))):
            cin = ch[i + 1] if i + 1 < len(ch) else ch[-1]
            dec += [
                ChannelRMSNorm1d(cin),
                nn.SiLU(),
                nn.Upsample(scale_factor=st[i], mode="nearest"),
                CausalConv1d(cin, ch[i], 2 * st[i] + 1),
            ]
            dec += [ResUnit(ch[i], d) for d in dil]
        self.decoder = nn.Sequential(*dec)
        self.dec_norm = ChannelRMSNorm1d(ch[0])
        self.dec_out = CausalConv1d(ch[0], 1, 7)
        with torch.no_grad():  # start as a near-deterministic autoencoder; the KL term relaxes it
            self.enc_out.conv.bias[cfg.latent_ch :].fill_(-6.0)

        self.register_buffer("latent_mean", torch.zeros(cfg.latent_ch))
        self.register_buffer("latent_std", torch.ones(cfg.latent_ch))

    def encode_dist(self, wave: torch.Tensor):
        """wave: [B, 1, S] in [-1, 1], S a multiple of 800 -> raw (mean, logvar) [B, C, S/800]."""
        if wave.shape[-1] % AUDIO_HOP:
            raise ValueError(f"audio length must be a multiple of {AUDIO_HOP} samples")
        h = self.enc_out(F.silu(self.enc_norm(self.encoder(wave))))
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar.clamp(-20, 4)

    def normalize(self, z):
        mean, std = self.latent_mean.to(z.dtype), self.latent_std.to(z.dtype)
        return (z - mean.view(1, -1, 1)) / std.view(1, -1, 1)

    def denormalize(self, z):
        mean, std = self.latent_mean.to(z.dtype), self.latent_std.to(z.dtype)
        return z * std.view(1, -1, 1) + mean.view(1, -1, 1)

    @torch.no_grad()
    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.encode_dist(wave)[0])

    def decode_raw(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_out(F.silu(self.dec_norm(self.decoder(z))))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_raw(self.denormalize(z))

    def forward(self, wave, sample: bool = True):
        mean, logvar = self.encode_dist(wave)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        return self.decode_raw(z), mean, logvar
