"""High-level entry point: an h3t file -> a streaming, 4-bit H3 DiT that predicts velocities and runs a sampler.

Text states (Qwen3-VL hidden states) and the VAEs are inputs/outputs of this engine, not part of it: feed it the
[1, L, 5120] states from any encoder and decode the returned latents with the H3 VAEs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .config import H3Config
from .layout import time_shift_sigma
from .model import H3Model
from .store import H3TFile
from .stream import StreamingProvider

_GIB = 1 << 30


def h3_sigmas(steps: int, shift: float) -> torch.Tensor:
    """Flow-matching sigma schedule with time shift `shift` (H3: 12 for video), steps + 1 values from 1 down to 0."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return (shift * t / (1.0 + (shift - 1.0) * t)).to(torch.float32)


class H3Engine:
    def __init__(self, store: H3TFile, provider: StreamingProvider, model: H3Model):
        self.store, self.provider, self.model, self.cfg = store, provider, model, store.cfg

    @classmethod
    def from_h3t(cls, path: str, device: Optional[str] = None, *, resident="auto", precision: str = "a8", backend: str = "auto",
                 reserve_gb: float = 1.5, prefetch: int = 2, ring: int = 3, pin="auto", mlp_chunk: Optional[int] = None,
                 attn_chunk: Optional[int] = None, refiner_on_gpu: bool = False, **model_kw) -> "H3Engine":
        """resident="auto" keeps as many blocks on the GPU as fit after `reserve_gb` for activations (all of them on a
        big card, ~13 of 50 on a 6 GB one); the rest stream from RAM behind compute. precision "a16" skips activation
        quantisation (slower, slightly closer to the un-quantised function)."""
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        store = H3TFile(path)
        provider = None
        try:
            glob = store.load_globals(dev, refiner_device=dev if refiner_on_gpu else "cpu")
            provider = StreamingProvider(store, dev, resident=resident, prefetch=prefetch, pin=pin, ring=ring,
                                         reserve_bytes=int(reserve_gb * _GIB))
            model = H3Model(store.cfg, glob, provider, device=dev, backend=backend, precision=precision,
                            mlp_chunk=mlp_chunk, attn_chunk=attn_chunk, **model_kw)
        except BaseException:
            if provider is not None:
                provider.close()
            store.close()
            raise
        return cls(store, provider, model)

    # ---------------------------------------------------------------- inference
    def encode_text(self, text_states: torch.Tensor) -> torch.Tensor:
        """Run condition_proj + token refiner once; pass the result as `refined_text` to every step."""
        return self.model.encode_text(text_states)

    @torch.inference_mode()
    def velocity(self, x: Sequence[torch.Tensor], sigma, text_states: Optional[torch.Tensor] = None, **kw) -> List[torch.Tensor]:
        return self.model.forward(x, sigma, text_states, **kw)

    @torch.inference_mode()
    def sample(self, video_shape: Tuple[int, int, int], audio_len: int, text_states: torch.Tensor, *, steps: int = 8, seed: int = 0,
               payload: Optional[Mapping[str, Any]] = None, callback=None, video_noise: Optional[torch.Tensor] = None,
               audio_noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler sampler on each stream's own sigma (video shift 12, audio shift 3). video_shape = (T, H, W) latent frames/rows/cols,
        audio_len = latent audio frames Ta (40 per second). Returns (video [1,C,T,H,W], audio [1,Ca,2,Ta]) latents at sigma 0."""
        cfg, dev = self.cfg, self.model.device
        g = torch.Generator("cpu").manual_seed(seed)
        t, h, w = video_shape
        xv = video_noise if video_noise is not None else torch.randn(1, cfg.video_channels, t, h, w, generator=g)
        xa = audio_noise if audio_noise is not None else torch.randn(1, cfg.audio_channels, 2, audio_len, generator=g)
        xv, xa = xv.to(dev, torch.float32), xa.to(dev, torch.float32)
        refined = self.encode_text(text_states.to(dev, self.model.dtype))
        sv = h3_sigmas(steps, cfg.sigma_shift_video)
        sa = time_shift_sigma(sv, cfg.sigma_shift_video, cfg.sigma_shift_audio)
        for i in range(steps):
            vv, va = self.model.forward([xv.to(self.model.dtype), xa.to(self.model.dtype)], sv[i], None, payload=payload,
                                        sample_sigmas=sv, refined_text=refined)
            xv = xv + vv.float() * float(sv[i + 1] - sv[i])
            xa = xa + va.float() * float(sa[i + 1] - sa[i])
            if callback is not None:
                callback(i, xv, xa)
        return xv, xa

    def linear_backend(self) -> str:
        """Which kernels run the 4-bit linears: "ck" (comfy_kitchen CUDA, fast) or "torch" (portable, ~2x slower)."""
        from .qlinear import available_backends

        m = self.model
        ck_ok = available_backends()["ck"]["available"] and m.device.type == "cuda"
        return "ck" if ck_ok and m.backend != "torch" and m.precision == "a8" else "torch"

    # ---------------------------------------------------------------- lifecycle
    def stats(self) -> Dict[str, object]:
        return self.provider.stats()

    def close(self) -> None:
        self.provider.close()
        self.store.close()

    def __enter__(self) -> "H3Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
