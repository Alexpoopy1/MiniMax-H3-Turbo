"""Omni generation pipeline: text / image / video / audio in, video and/or audio out.

One denoise loop serves every task; a task is just which tokens are pinned clean
(see layout.py). Sampling uses cached AdaLN modulation and, for step-distilled
models, no CFG, so a generation is `steps` transformer forwards plus the VAE decode.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .audio_vae import AudioVAE
from .config import (
    AUDIO_HOP,
    AUDIO_LATENT_RATE,
    AUDIO_SAMPLE_RATE,
    PIXELS_PER_TOKEN,
    VIDEO_SPATIAL,
    VIDEO_TEMPORAL,
    H3TurboConfig,
)
from .layout import (
    GROUPS,
    G_AUDIO_CLEAN,
    G_VIDEO_CLEAN,
    Geometry,
    RefTokens,
    Track,
    audio_positions,
    build_segments,
    group_times,
    latent_index_of_frame,
    patchify_video,
    snap_frames,
    snap_size,
    unpatchify_video,
    video_positions,
)
from .model import H3TurboTransformer, Segment
from .sampler import euler_step, flow_sigmas, shift_sigmas
from .text import ByteTextEncoder, ExternalTextAdapter
from .video_vae import VideoVAE


# --------------------------------------------------------------------------- inputs / outputs
@dataclass
class VideoCond:
    """Pin part of the output video. `frames` [T,3,H,W] in [-1,1] (T=1: an image).
    `frame_index` is the output pixel frame where frames[0] lands (0 = start; pass
    num_frames-1 to pin the last frame). `mask` [H,W] in [0,1]: 1 keeps the content,
    0 regenerates it (inpainting / editing); default keeps everything."""

    frames: torch.Tensor
    frame_index: int = 0
    mask: Optional[torch.Tensor] = None


@dataclass
class AudioCond:
    """Pin part of the output audio. `wave` [S] float in [-1,1] at 32 kHz."""

    wave: torch.Tensor
    start_sec: float = 0.0


@dataclass
class OmniContext:
    video: List[VideoCond] = field(default_factory=list)  # pinned on the timeline
    audio: List[AudioCond] = field(default_factory=list)
    ref_video: List[torch.Tensor] = field(default_factory=list)  # [T,3,H,W]; off-timeline references
    ref_audio: List[torch.Tensor] = field(default_factory=list)  # [S]; e.g. a voice sample


@dataclass
class Generation:
    video: Optional[torch.Tensor]  # uint8 [T, H, W, 3]
    audio: Optional[torch.Tensor]  # float32 [S] mono in [-1, 1]
    fps: int
    sample_rate: int = AUDIO_SAMPLE_RATE


def frames_from_uint8(video: torch.Tensor) -> torch.Tensor:
    """uint8 [T,H,W,3] -> float [T,3,H,W] in [-1,1]."""
    return video.permute(0, 3, 1, 2).float() / 127.5 - 1.0


def frames_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """float [B?,3,T,H,W] or [3,T,H,W] in [-1,1] -> uint8 [T,H,W,3]."""
    if x.dim() == 5:
        x = x[0]
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu()


# --------------------------------------------------------------------------- pipeline
class H3TurboPipeline:
    def __init__(
        self,
        transformer: H3TurboTransformer,
        video_vae: VideoVAE,
        audio_vae: AudioVAE,
        text_encoder: torch.nn.Module,
        config: H3TurboConfig,
    ):
        self.model, self.video_vae, self.audio_vae = transformer.eval(), video_vae.eval(), audio_vae.eval()
        self.text_encoder = text_encoder.eval()
        self.cfg = config
        self.quant: Optional[str] = None
        self.vae_tile = dict(tile=32, overlap=12, chunk=2)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "H3TurboPipeline":
        """path: a .safetensors file or a directory holding model.safetensors."""
        from .io import load_checkpoint

        return load_checkpoint(path, **kwargs)

    def save_pretrained(self, path: str, dtype: torch.dtype = torch.float16) -> None:
        from .io import save_checkpoint

        save_checkpoint(path, self, dtype)

    # ------------------------------------------------------------------ placement
    @property
    def device(self) -> torch.device:
        return self.model.in_proj["video"].weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.in_proj["video"].weight.dtype

    @property
    def vae_dtype(self) -> torch.dtype:
        return next(self.video_vae.parameters()).dtype

    # ------------------------------------------------------------------ text
    @torch.no_grad()
    def encode_text(self, prompt: str = "", text_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[1, L, text_dim] in the transformer's compute dtype."""
        if isinstance(self.text_encoder, ExternalTextAdapter):
            if text_embeds is None:
                raise ValueError("this model uses an external text encoder: pass text_embeds [L, ext_dim]")
            e = text_embeds if text_embeds.dim() == 3 else text_embeds[None]
            enc = self.text_encoder(e.to(next(self.text_encoder.parameters()).device, next(self.text_encoder.parameters()).dtype))
        else:
            enc, _ = self.text_encoder.encode_text([prompt])
        return enc.to(self.device, self.dtype)

    # ------------------------------------------------------------------ VAE helpers
    def _video_tensor(self, frames: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = frames.to(self.device, torch.float32)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False).clamp(-1, 1)
        T = 1 + VIDEO_TEMPORAL * ((x.shape[0] - 1) // VIDEO_TEMPORAL)
        return x[:T].permute(1, 0, 2, 3)[None].to(self.vae_dtype)

    @torch.no_grad()
    def encode_video_tokens(self, frames: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int]:
        """frames [T,3,h,w] -> (tokens [1, T'*hp*wp, D] float32, T')."""
        z = self.video_vae.encode(self._video_tensor(frames, H, W))
        return patchify_video(z.float()), z.shape[2]

    @torch.no_grad()
    def encode_audio_tokens(self, wave: torch.Tensor) -> torch.Tensor:
        w = wave.to(self.device, torch.float32).reshape(1, 1, -1)
        w = w[..., : (w.shape[-1] // AUDIO_HOP) * AUDIO_HOP]
        if w.shape[-1] == 0:
            raise ValueError(f"audio shorter than one token ({AUDIO_HOP} samples)")
        return self.audio_vae.encode(w.to(next(self.audio_vae.parameters()).dtype)).float().transpose(1, 2)

    @torch.no_grad()
    def decode_video_tokens(self, tokens: torch.Tensor, geom: Geometry) -> torch.Tensor:
        C = self.cfg.video_vae.latent_ch
        z = unpatchify_video(tokens.to(self.vae_dtype), C, geom.lat_t, geom.height // VIDEO_SPATIAL, geom.width // VIDEO_SPATIAL)
        return frames_to_uint8(self.video_vae.decode_tiled(z, **self.vae_tile))

    @torch.no_grad()
    def decode_audio_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        z = tokens.transpose(1, 2).to(next(self.audio_vae.parameters()).dtype)
        return self.audio_vae.decode(z)[0, 0].float().clamp(-1, 1).cpu()

    # ------------------------------------------------------------------ timeline
    def _build_timeline(self, geom: Geometry, ctx: OmniContext, generate: Sequence[str], ref_budget: Optional[int]):
        dev = self.device
        D_v, D_a = self.cfg.transformer.video_in_dim, self.cfg.transformer.audio_in_dim
        hp, wp, T = geom.hp, geom.wp, geom.lat_t
        H, W = geom.height, geom.width

        # -- video ----------------------------------------------------------------
        pos_v = video_positions(T, hp, wp, device=dev)
        clean_v = torch.zeros(1, T, hp * wp, D_v, device=dev)
        pin_v = torch.zeros(1, T, hp * wp, dtype=torch.bool, device=dev)
        for vc in ctx.video:
            tok, Tc = self.encode_video_tokens(vc.frames, H, W)
            j0 = latent_index_of_frame(vc.frame_index)
            n = min(Tc, T - j0)
            if n <= 0:
                continue
            tok = tok.view(1, Tc, hp * wp, D_v)[:, :n]
            if vc.mask is None:
                m = torch.ones(hp * wp, dtype=torch.bool, device=dev)
            else:
                mm = vc.mask.to(dev, torch.float32)[None, None]
                m = (F.adaptive_avg_pool2d(mm, (hp, wp)).flatten() >= 0.5)
            clean_v[:, j0 : j0 + n] = torch.where(m[None, None, :, None], tok, clean_v[:, j0 : j0 + n])
            pin_v[:, j0 : j0 + n] |= m[None, None, :]
        clean_v, pin_v = clean_v.view(1, -1, D_v), pin_v.view(1, -1)

        # -- audio ----------------------------------------------------------------
        Na = geom.audio_tokens
        pos_a = audio_positions(Na, geom.fps, device=dev)
        clean_a = torch.zeros(1, Na, D_a, device=dev)
        pin_a = torch.zeros(1, Na, dtype=torch.bool, device=dev)
        for ac in ctx.audio:
            tok = self.encode_audio_tokens(ac.wave)
            a0 = max(0, round(ac.start_sec * AUDIO_LATENT_RATE))
            n = min(tok.shape[1], Na - a0)
            if n <= 0:
                continue
            clean_a[:, a0 : a0 + n] = tok[:, :n]
            pin_a[:, a0 : a0 + n] = True

        tracks = {}
        for name, gen, pos, clean, pin in (("video", "video" in generate, pos_v, clean_v, pin_v), ("audio", "audio" in generate, pos_a, clean_a, pin_a)):
            if gen:
                tracks[name] = Track(clean, pos, pin, emit=True)
            elif bool(pin.any()):  # context only: keep just the pinned tokens, at their true positions
                idx = pin[0].nonzero().squeeze(-1)
                tracks[name] = Track(clean[:, idx], pos[:, idx], torch.ones(1, len(idx), dtype=torch.bool, device=dev), emit=False)

        # -- references (off the timeline) ------------------------------------------
        refs, resampled = [], []
        t_ref = float(T + 1)
        for frames in ctx.ref_video:
            h, w = snap_size(frames.shape[-2]), snap_size(frames.shape[-1])
            tok, Tr = self.encode_video_tokens(frames, h, w)
            pos = video_positions(Tr, h // PIXELS_PER_TOKEN, w // PIXELS_PER_TOKEN, t0=t_ref, device=dev)
            self._add_ref(refs, resampled, "video", tok, pos, t_ref, Tr, ref_budget)
            t_ref += Tr + 1
        for wave in ctx.ref_audio:
            tok = self.encode_audio_tokens(wave)
            pos = audio_positions(tok.shape[1], geom.fps, t0=t_ref, device=dev)
            self._add_ref(refs, resampled, "audio", tok, pos, t_ref, tok.shape[1] * geom.fps / 160.0, ref_budget)
            t_ref += tok.shape[1] * geom.fps / 160.0 + 1
        return tracks, refs, resampled

    def _add_ref(self, refs, resampled, modality, tok, pos, t_ref, span, budget):
        tok = tok.to(self.dtype)
        if budget is not None and self.model.resampler is not None and tok.shape[1] > budget:
            K = self.model.cfg.resampler_queries
            g = G_VIDEO_CLEAN if modality == "video" else G_AUDIO_CLEAN
            rpos = torch.zeros(1, K, 3, device=tok.device)
            rpos[..., 0] = t_ref + torch.linspace(0, float(span), K, device=tok.device)
            seg = Segment(modality, tok, pos, torch.full(tok.shape[:2], g, device=tok.device))
            resampled.append(self.model.resample(seg, torch.full((1, K), g, device=tok.device), rpos))
        else:
            refs.append(RefTokens(modality, tok, pos))

    # ------------------------------------------------------------------ denoising
    @torch.no_grad()
    def _denoise(
        self,
        text_c: torch.Tensor,
        text_u: Optional[torch.Tensor],
        tracks: dict,
        refs: Sequence[RefTokens],
        resampled: Sequence[Segment],
        sigmas_v: torch.Tensor,
        sigmas_a: torch.Tensor,
        cfg_scale: float,
        distilled_guidance: Optional[float],
        callback: Optional[Callable[[int, int], None]],
    ) -> dict:
        dev, dt = self.device, self.dtype
        steps = len(sigmas_v) - 1
        gvec = None
        if self.model.cfg.guidance_embed:
            gvec = torch.full((1,), distilled_guidance if distilled_guidance is not None else 3.5)
        state = {k: t.x.clone().float() for k, t in tracks.items()}
        gts = [group_times(1, float(sigmas_v[i]), float(sigmas_a[i])) for i in range(steps)]
        # AdaLN modulation for the whole schedule, computed once (the bank stays off the GPU)
        mods = [self.model.compute_mods(GROUPS, gt, gvec, dtype=dt, device=dev) for gt in gts]
        use_cfg = text_u is not None and cfg_scale != 1.0 and gvec is None

        def forward(text, i):
            live = {k: Track(state[k].to(dt), t.pos, t.pinned, t.emit) for k, t in tracks.items()}
            segs, index = build_segments(text, live.get("video"), live.get("audio"), refs, resampled)
            outs = self.model(segs, GROUPS, gts[i], guidance=gvec, mods=mods[i])
            return {k: outs[index[k]] for k in tracks if tracks[k].emit}

        for i in range(steps):
            v = forward(text_c, i)
            if use_cfg:
                vu = forward(text_u, i)
                v = {k: vu[k] + cfg_scale * (v[k] - vu[k]) for k in v}
            for k, vk in v.items():
                sig = sigmas_v if k == "video" else sigmas_a
                new = euler_step(state[k], vk.float(), float(sig[i]), float(sig[i + 1]))
                state[k] = torch.where(tracks[k].pinned[..., None], state[k], new)
            if callback:
                callback(i + 1, steps)
        return state

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        *,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 320,
        num_frames: Optional[int] = None,
        duration: Optional[float] = None,
        fps: Optional[int] = None,
        generate: Union[str, Iterable[str]] = ("video", "audio"),
        context: Optional[OmniContext] = None,
        steps: int = 4,
        guidance: Optional[float] = None,
        shift: float = 3.0,
        seed: Optional[int] = None,
        text_embeds: Optional[torch.Tensor] = None,
        negative_embeds: Optional[torch.Tensor] = None,
        ref_budget: Optional[int] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Generation:
        """Generate video and/or audio.

        generate   subset of {"video", "audio"}. Generating only audio from a pinned
                   video is video-to-audio; only video from pinned audio is audio-to-video.
        context    pinned frames / audio and off-timeline references (see OmniContext).
        steps      2-4 for Turbo checkpoints; 20+ for an undistilled one.
        guidance   CFG scale for undistilled models (1 = off, the default); the guidance
                   value fed to the model when it has distilled guidance.
        """
        gen = {generate} if isinstance(generate, str) else set(generate)
        if not gen or not gen <= {"video", "audio"}:
            raise ValueError("generate must be a non-empty subset of {'video', 'audio'}")
        fps = fps or self.cfg.fps
        if num_frames is None:
            num_frames = round((duration if duration else 2.0) * fps)
        geom = Geometry(snap_size(width), snap_size(height), snap_frames(num_frames), fps)
        ctx = context or OmniContext()

        seed = random.randrange(2**31) if seed is None else seed
        g = torch.Generator().manual_seed(seed)
        tracks, refs, resampled = self._build_timeline(geom, ctx, gen, ref_budget)
        for k in gen:  # start from noise wherever a track is generated and not pinned
            t = tracks[k]
            noise = torch.randn(t.x.shape, generator=g).to(self.device)
            t.x = torch.where(t.pinned[..., None], t.x, noise)

        text_c = self.encode_text(prompt, text_embeds)
        cfg_scale = 1.0 if guidance is None else float(guidance)
        distilled = guidance
        text_u = None
        if not self.model.cfg.guidance_embed and cfg_scale != 1.0:
            text_u = self.encode_text(negative_prompt, negative_embeds)

        sig = flow_sigmas(steps, shift)
        state = self._denoise(text_c, text_u, tracks, refs, resampled, sig, sig, cfg_scale, distilled, callback)

        video = audio = None
        if "video" in gen:
            video = self.decode_video_tokens(state["video"], geom)
        if "audio" in gen:
            audio = self.decode_audio_tokens(state["audio"])
        return Generation(video, audio, fps)

    @torch.no_grad()
    def refine(
        self,
        gen: Generation,
        prompt: str = "",
        *,
        scale: float = 2.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 2,
        strength: float = 0.6,
        shift: float = 3.0,
        seed: Optional[int] = None,
        text_embeds: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> Generation:
        """In-context regeneration (replaces a separate super-resolution network).

        The low-res result is upsampled, partially re-noised (`strength`), and the
        *same* model regenerates it at the new size while re-reading the original
        low-res video as clean context, whose positions are rescaled onto the hi-res
        grid. Audio, if present, stays pinned so it remains in sync."""
        if gen.video is None:
            raise ValueError("nothing to refine: generation has no video")
        T, H, W, _ = gen.video.shape
        W2, H2 = snap_size(width or round(W * scale)), snap_size(height or round(H * scale))
        geom = Geometry(W2, H2, T, gen.fps)
        frames = frames_from_uint8(gen.video)
        dev = self.device

        lo_tok, Tl = self.encode_video_tokens(frames, H, W)
        hi_tok, _ = self.encode_video_tokens(frames, H2, W2)
        hp_l, wp_l = H // PIXELS_PER_TOKEN, W // PIXELS_PER_TOKEN
        ref_pos = video_positions(Tl, hp_l, wp_l, device=dev)
        ref_pos[..., 1] *= geom.hp / hp_l
        ref_pos[..., 2] *= geom.wp / wp_l
        refs = [RefTokens("video", lo_tok.to(self.dtype), ref_pos)]

        seed = random.randrange(2**31) if seed is None else seed
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(hi_tok.shape, generator=g).to(dev)
        x = (1 - strength) * hi_tok + strength * noise
        tracks = {
            "video": Track(x, video_positions(geom.lat_t, geom.hp, geom.wp, device=dev), torch.zeros(x.shape[:2], dtype=torch.bool, device=dev), True)
        }
        if gen.audio is not None:
            tok = self.encode_audio_tokens(gen.audio)[:, : geom.audio_tokens]
            pos = audio_positions(geom.audio_tokens, geom.fps, device=dev)[:, : tok.shape[1]]
            tracks["audio"] = Track(tok, pos, torch.ones(tok.shape[:2], dtype=torch.bool, device=dev), False)

        sig = shift_sigmas(torch.linspace(strength, 0.0, steps + 1), shift)
        text_c = self.encode_text(prompt, text_embeds)
        state = self._denoise(text_c, None, tracks, refs, [], sig, sig, 1.0, None, callback)
        return Generation(self.decode_video_tokens(state["video"], geom), gen.audio, gen.fps)
