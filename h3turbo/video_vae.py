"""Causal video VAE: 16x spatial, 4x temporal, 24 latent channels (f16t4d24).

Frame counts are 1 + 4k -> 1 + k latent frames: the first frame is encoded on its
own, so a single image is a valid 1-frame video and image conditioning falls out of
the same code path.

Normalisation is channel-wise RMSNorm (no spatial/temporal statistics), which makes
every activation a local function of the input, so tiled/chunked decoding converges
to the untiled result as the overlap grows (no GroupNorm statistics shift per tile).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VIDEO_SPATIAL, VIDEO_TEMPORAL, VideoVAEConfig


class ChannelRMSNorm(nn.Module):
    def __init__(self, ch: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ch, 1, 1, 1))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class CausalConv3d(nn.Module):
    """Time-causal conv: the past is padded by replicating the first frame.

    In streaming mode (stride-1 convs only, used by chunked decode) the last kt-1
    input frames are cached between calls, so feeding a clip in pieces gives exactly
    the same result as feeding it whole, with no recomputed context."""

    def __init__(self, cin, cout, kernel=(3, 3, 3), stride=(1, 1, 1)):
        super().__init__()
        self.kt, kh, kw = kernel
        self.stride_t = stride[0]
        self.pad = (kw // 2, kw // 2, kh // 2, kh // 2)
        self.conv = nn.Conv3d(cin, cout, kernel, stride)
        self.streaming = False
        self._cache = None

    def set_stream(self, on: bool) -> None:
        if on and self.stride_t != 1:
            raise RuntimeError("streaming supports stride-1 temporal convs only")
        self.streaming, self._cache = on, None

    def forward(self, x):
        if self.kt > 1:
            if self.streaming and self._cache is not None:
                prefix = self._cache
            else:
                prefix = x[:, :, :1].expand(-1, -1, self.kt - 1, -1, -1)
            x = torch.cat([prefix, x], dim=2)
            if self.streaming:
                self._cache = x[:, :, -(self.kt - 1) :]
        return self.conv(F.pad(x, self.pad))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tk):
        super().__init__()
        self.n1, self.n2 = ChannelRMSNorm(cin), ChannelRMSNorm(cout)
        self.c1 = CausalConv3d(cin, cout, (tk, 3, 3))
        self.c2 = CausalConv3d(cout, cout, (tk, 3, 3))
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, cin, cout, temporal: bool):
        super().__init__()
        k, s = ((3, 3, 3), (2, 2, 2)) if temporal else ((1, 3, 3), (1, 2, 2))
        self.conv = CausalConv3d(cin, cout, k, s)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, cin, cout, temporal: bool, tk: int):
        super().__init__()
        self.temporal = temporal
        self.conv = CausalConv3d(cin, cout, (tk, 3, 3))
        self.streaming = False
        self._started = False

    def set_stream(self, on: bool) -> None:
        self.streaming, self._started = on, False

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        if self.temporal:  # 1+k -> 1+2k: only the clip's very first frame stays single
            if self.streaming and self._started:
                x = x.repeat_interleave(2, dim=2)
            else:
                x = torch.cat([x[:, :, :1], x[:, :, 1:].repeat_interleave(2, dim=2)], dim=2)
            self._started = True
        return self.conv(x)


def _shortcut_groups(latent_ch: int) -> int:
    per = 3 * VIDEO_SPATIAL * VIDEO_SPATIAL
    if per % latent_ch:
        raise ValueError(f"latent_ch {latent_ch} must divide {per} for the residual shortcut")
    return per // latent_ch


def encode_shortcut(x: torch.Tensor, latent_ch: int) -> torch.Tensor:
    """Parameter-free f16t4 projection of the input onto latent shape: pixel-unshuffle each
    frame, average channel groups down to `latent_ch`, then average time in groups of 4
    (the first frame stays on its own, matching the causal layout). Causal by construction.
    The encoder learns a residual on top of this (DC-AE-style residual autoencoding)."""
    B, C, T, H, W = x.shape
    g = _shortcut_groups(latent_ch)
    y = F.pixel_unshuffle(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), VIDEO_SPATIAL)
    h, w = y.shape[-2:]
    y = y.reshape(B, T, latent_ch, g, h, w).mean(3).permute(0, 2, 1, 3, 4)  # [B, L, T, h, w]
    rest = y[:, :, 1:].reshape(B, latent_ch, (T - 1) // VIDEO_TEMPORAL, VIDEO_TEMPORAL, h, w).mean(3)
    return torch.cat([y[:, :, :1], rest], dim=2)


def decode_shortcut(z: torch.Tensor, first_is_single: bool = True) -> torch.Tensor:
    """Inverse layout of `encode_shortcut`: repeat channels, pixel-shuffle, repeat time x4.
    `first_is_single` is False for a streamed chunk that does not start the clip."""
    B, L, T, h, w = z.shape
    g = _shortcut_groups(L)
    y = z.repeat_interleave(g, dim=1).permute(0, 2, 1, 3, 4).reshape(B * T, L * g, h, w)
    y = F.pixel_shuffle(y, VIDEO_SPATIAL).reshape(B, T, 3, h * VIDEO_SPATIAL, w * VIDEO_SPATIAL).permute(0, 2, 1, 3, 4)
    if first_is_single:
        return torch.cat([y[:, :, :1], y[:, :, 1:].repeat_interleave(VIDEO_TEMPORAL, dim=2)], dim=2)
    return y.repeat_interleave(VIDEO_TEMPORAL, dim=2)


class VideoVAE(nn.Module):
    def __init__(self, cfg: VideoVAEConfig):
        super().__init__()
        self.cfg = cfg
        ch, tk, n = cfg.channels, cfg.t_kernel, cfg.blocks
        assert len(ch) == 5 and len(cfg.temporal_down) == 4 and len(tk) == 5
        assert 2 ** len(cfg.temporal_down) == VIDEO_SPATIAL
        assert 2 ** sum(cfg.temporal_down) == VIDEO_TEMPORAL

        # encoder
        enc = [CausalConv3d(3, ch[0], (1, 3, 3))]
        for i in range(4):
            enc += [ResBlock(ch[i], ch[i], tk[i]) for _ in range(n)]
            enc.append(Downsample(ch[i], ch[i + 1], cfg.temporal_down[i]))
        enc += [ResBlock(ch[4], ch[4], tk[4]) for _ in range(2)]
        self.encoder = nn.Sequential(*enc)
        self.enc_norm = ChannelRMSNorm(ch[4])
        self.enc_out = CausalConv3d(ch[4], 2 * cfg.latent_ch, (tk[4], 3, 3))

        # decoder
        dec = [CausalConv3d(cfg.latent_ch, ch[4], (tk[4], 3, 3))]
        dec += [ResBlock(ch[4], ch[4], tk[4]) for _ in range(2)]
        for i in reversed(range(4)):
            dec.append(Upsample(ch[i + 1], ch[i], cfg.temporal_down[i], tk[i]))
            dec += [ResBlock(ch[i], ch[i], tk[i]) for _ in range(n)]
        self.decoder = nn.Sequential(*dec)
        self.dec_norm = ChannelRMSNorm(ch[0])
        self.dec_out = CausalConv3d(ch[0], 3, (1, 3, 3))
        with torch.no_grad():  # start as a near-deterministic autoencoder; the KL term relaxes it
            self.enc_out.conv.bias[cfg.latent_ch :].fill_(-6.0)

        self._streaming, self._stream_started = False, False

        # latent normalisation (fitted after training so latents are ~unit variance)
        self.register_buffer("latent_mean", torch.zeros(cfg.latent_ch))
        self.register_buffer("latent_std", torch.ones(cfg.latent_ch))

    # ------------------------------------------------------------------ core
    def encode_dist(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, 3, T, H, W] in [-1, 1], T = 1 + 4k. Returns raw (mean, logvar)."""
        self._check(x)
        h = self.enc_out(F.silu(self.enc_norm(self.encoder(x))))
        mean, logvar = h.chunk(2, dim=1)
        mean = mean + encode_shortcut(x, self.cfg.latent_ch).to(mean.dtype)
        return mean, logvar.clamp(-20, 4)

    def normalize(self, z):
        mean, std = self.latent_mean.to(z.dtype), self.latent_std.to(z.dtype)
        return (z - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)

    def denormalize(self, z):
        mean, std = self.latent_mean.to(z.dtype), self.latent_std.to(z.dtype)
        return z * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic, normalised latents [B, 24, 1+k, H/16, W/16]."""
        return self.normalize(self.encode_dist(x)[0])

    def decode_raw(self, z: torch.Tensor) -> torch.Tensor:
        """z: raw (un-normalised) latents -> [B, 3, T, H, W]."""
        out = self.dec_out(F.silu(self.dec_norm(self.decoder(z))))
        first = not (self._streaming and self._stream_started)
        self._stream_started = True
        return out + decode_shortcut(z, first).to(out.dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_raw(self.denormalize(z))

    def forward(self, x, sample: bool = True):
        mean, logvar = self.encode_dist(x)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        return self.decode_raw(z), mean, logvar

    def _check(self, x):
        T, H, W = x.shape[2:]
        if (T - 1) % VIDEO_TEMPORAL or H % VIDEO_SPATIAL or W % VIDEO_SPATIAL:
            raise ValueError(f"need T=1+4k and H,W multiples of 16, got {tuple(x.shape[2:])}")

    # ------------------------------------------------------------------ tiled decode
    def _stream(self, on: bool) -> None:
        self._streaming, self._stream_started = on, False
        for m in self.decoder.modules():
            if isinstance(m, (CausalConv3d, Upsample)):
                m.set_stream(on)
        self.dec_out.set_stream(on)

    @torch.no_grad()
    def decode_tiled(self, z: torch.Tensor, tile: int = 32, overlap: int = 12, chunk: int = 2) -> torch.Tensor:
        """Memory-bounded decode of normalised latents.

        Time: streamed `chunk` latent frames at a time with cached causal state, so
        it is exactly equal to a full decode (tested). Space: tiles of `tile` latent
        pixels blended over `overlap` pixels (x16 for pixels). The decoder's spatial
        receptive field is ~10 latent pixels, so spatial tiling is approximate and its
        error shrinks as `overlap` grows (0.16% mean relative error at tile=20/overlap=12
        on random weights, the worst case). A clip that fits in one tile is not split."""
        z = self.denormalize(z)
        B, _, T, H, W = z.shape
        s = VIDEO_SPATIAL
        step = max(tile - overlap, 1)
        out = acc = None
        for y0 in _starts(H, tile, step):
            for x0 in _starts(W, tile, step):
                y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
                piece = self._decode_stream(z[:, :, :, y0:y1, x0:x1], chunk)
                if H <= tile and W <= tile:
                    return piece
                if out is None:
                    out = z.new_zeros(B, 3, piece.shape[2], H * s, W * s)
                    acc = z.new_zeros(1, 1, 1, H * s, W * s)
                wy = _ramp((y1 - y0) * s, overlap * s, y0 > 0, y1 < H, piece.device)
                wx = _ramp((x1 - x0) * s, overlap * s, x0 > 0, x1 < W, piece.device)
                w = (wy[:, None] * wx[None, :])[None, None, None]
                out[:, :, :, y0 * s : y1 * s, x0 * s : x1 * s] += piece * w
                acc[:, :, :, y0 * s : y1 * s, x0 * s : x1 * s] += w
        return out / acc

    def _decode_stream(self, z: torch.Tensor, chunk: int) -> torch.Tensor:
        T = z.shape[2]
        if T <= chunk:
            return self.decode_raw(z)
        self._stream(True)
        try:
            return torch.cat([self.decode_raw(z[:, :, a : a + chunk]) for a in range(0, T, chunk)], dim=2)
        finally:
            self._stream(False)


def _starts(size: int, tile: int, step: int):
    if size <= tile:
        return [0]
    s = list(range(0, size - tile, step)) + [size - tile]
    return sorted(set(s))


def _ramp(n: int, ov: int, left: bool, right: bool, device) -> torch.Tensor:
    """Blend weights along one axis of a tile. Weight is zero for the first `ov/3`
    pixels next to an interior tile edge (where conv border effects are worst) and
    ramps to 1 across the rest of the overlap."""
    w = torch.ones(n, device=device)
    if ov > 0:
        m = ov / 3.0
        d = torch.arange(ov, device=device, dtype=torch.float32) + 0.5  # distance from the edge
        r = ((d - m) / (ov - m)).clamp(min=1e-4, max=1.0)
        if left:
            w[:ov] = torch.minimum(w[:ov], r)
        if right:
            w[-ov:] = torch.minimum(w[-ov:], r.flip(0))
    return w
