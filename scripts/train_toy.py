#!/usr/bin/env python
"""Train the demo checkpoint on the synthetic omni dataset (CPU-friendly).

Stages (each resumable by re-running with the same --workdir):
  vae_video  ->  vae_audio  ->  cache latents  ->  dit  ->  export

This is the reference for how a real run is wired; only the data source changes.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from h3turbo import toy
from h3turbo.config import AUDIO_HOP, AUDIO_SAMPLE_RATE, make_config
from h3turbo.io import build_modules, build_pipeline, save_checkpoint
from h3turbo.layout import patchify_video
from h3turbo.pipeline import H3TurboPipeline
from h3turbo.training import EMA, Geo, cosine_lr, flow_loss, kl_term, stft_l1

HI, LO, T_FRAMES, FPS = 96, 64, 9, 8
AUDIO_SAMPLES = 45 * AUDIO_HOP  # 9 frames @ 8 fps = 1.125 s


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def scene_frames(scene, side, T=T_FRAMES):
    return toy.render(scene, side, T).permute(1, 0, 2, 3)  # [3,T,H,W]


# --------------------------------------------------------------------------- VAEs
def train_video_vae(vae, steps, bs, lr, seed):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr, warmup=50)
        side = int(rng.choice([LO, HI]))
        x = torch.stack([scene_frames(toy.sample_scene(rng), side, 5) for _ in range(bs)])
        rec, mean, logvar = vae(x)
        l1 = (rec - x).abs().mean()
        loss = l1 + (rec - x).pow(2).mean() + 1e-6 * kl_term(mean, logvar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            log(f"vae_video {step}/{steps} l1={l1.item():.4f}")


def train_audio_vae(vae, steps, bs, lr, seed):
    rng = np.random.default_rng(seed + 1)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
    crop = 20 * AUDIO_HOP
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr, warmup=50)
        waves = []
        for _ in range(bs):
            w = toy.synth_audio(toy.sample_scene(rng), AUDIO_SAMPLES)
            o = int(rng.integers(0, AUDIO_SAMPLES - crop + 1))
            waves.append(w[o : o + crop])
        x = torch.stack(waves)[:, None]
        rec, mean, logvar = vae(x)
        l1 = (rec - x).abs().mean()
        spec = stft_l1(rec[:, 0], x[:, 0])
        # spectral loss leads: it is phase-invariant, and pitch/envelope are what matter here
        loss = 0.1 * l1 + spec + 1e-6 * kl_term(mean, logvar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            log(f"vae_audio {step}/{steps} l1={l1.item():.4f} stft={spec.item():.4f}")


# --------------------------------------------------------------------------- latent cache
@torch.no_grad()
def fit_latent_stats(vv, av, seed, n=192):
    rng = np.random.default_rng(seed + 7)
    zs, za = [], []
    for i in range(0, n, 16):
        sc = [toy.sample_scene(rng) for _ in range(16)]
        zs.append(vv.encode_dist(torch.stack([scene_frames(s, HI) for s in sc]))[0])
        za.append(av.encode_dist(torch.stack([toy.synth_audio(s, AUDIO_SAMPLES) for s in sc])[:, None])[0])
    z, a = torch.cat(zs), torch.cat(za)
    vv.latent_mean.copy_(z.mean((0, 2, 3, 4)))
    vv.latent_std.copy_(z.std((0, 2, 3, 4)).clamp(min=1e-3))
    av.latent_mean.copy_(a.mean((0, 2)))
    av.latent_std.copy_(a.std((0, 2)).clamp(min=1e-3))


@torch.no_grad()
def build_cache(vv, av, n, seed):
    rng = np.random.default_rng(seed + 100)
    out = {k: [] for k in ("prompts", "video", "video_lo", "audio", "ref")}
    for i in range(0, n, 32):
        sc = [toy.sample_scene(rng) for _ in range(min(32, n - i))]
        hi = torch.stack([scene_frames(s, HI) for s in sc])
        lo = torch.stack([scene_frames(s, LO) for s in sc])
        ref_t = rng.integers(0, T_FRAMES, size=len(sc))
        ref = torch.stack([hi[j, :, int(ref_t[j]) : int(ref_t[j]) + 1] for j in range(len(sc))])
        out["prompts"] += [s.prompt for s in sc]
        out["video"].append(patchify_video(vv.encode(hi)))
        out["video_lo"].append(patchify_video(vv.encode(lo)))
        out["ref"].append(patchify_video(vv.encode(ref)))
        out["audio"].append(av.encode(torch.stack([toy.synth_audio(s, AUDIO_SAMPLES) for s in sc])[:, None]).transpose(1, 2))
        if (i // 32) % 20 == 0:
            log(f"cache {i}/{n}")
    return {k: (torch.cat(v) if k != "prompts" else v) for k, v in out.items()}


# --------------------------------------------------------------------------- DiT
def train_dit(pipe, cache, steps, bs, lr, seed, mix):
    model, te = pipe.model, pipe.text_encoder
    params = list(model.parameters()) + list(te.parameters())
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    ema = EMA(params, 0.995)
    rng, gen = np.random.default_rng(seed + 5), torch.Generator().manual_seed(seed + 6)
    n = len(cache["prompts"])
    geo = Geo(lat_t=3, hp=HI // 32, wp=HI // 32, fps=FPS, lo_hp=LO // 32, lo_wp=LO // 32)
    names, probs = list(mix), np.array(list(mix.values())) / sum(mix.values())
    model.train(); te.train()
    run, t0 = {}, time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, steps, lr, warmup=150)
        idx = rng.integers(0, n, size=bs)
        batch = {k: ([cache[k][i] for i in idx] if k == "prompts" else cache[k][idx]) for k in cache}
        structure = names[int(rng.choice(len(names), p=probs))]
        loss, st = flow_loss(model, te, batch, structure, geo, rng, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ema.update()
        run.setdefault(structure, []).append(loss.item())
        if step % 100 == 0 or step == steps - 1:
            summary = " ".join(f"{k}={np.mean(v[-50:]):.3f}" for k, v in sorted(run.items()))
            log(f"dit {step}/{steps} {summary}  ({(time.time()-t0)/(step+1):.2f}s/step)")
    ema.copy_to()
    model.eval(); te.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", default="weights/h3turbo-nano-toy.safetensors")
    ap.add_argument("--vae-video-steps", type=int, default=1200)
    ap.add_argument("--vae-audio-steps", type=int, default=1200)
    ap.add_argument("--cache", type=int, default=4000)
    ap.add_argument("--dit-steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="all", choices=["all", "vae_video", "vae_audio", "dit"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(a.workdir, exist_ok=True)
    cfg = make_config("nano")
    cfg.fps = FPS
    _, vv, av, _ = build_modules(cfg)

    if a.stage in ("all", "vae_video"):
        p = os.path.join(a.workdir, "vae_video.pt")
        if not os.path.exists(p):
            train_video_vae(vv, a.vae_video_steps, 8, 2e-3, a.seed)
            torch.save(vv.state_dict(), p)
    if a.stage in ("all", "vae_audio"):
        p = os.path.join(a.workdir, "vae_audio.pt")
        if not os.path.exists(p):
            train_audio_vae(av, a.vae_audio_steps, 16, 1e-3, a.seed)
            torch.save(av.state_dict(), p)
    if a.stage != "all" and a.stage != "dit":
        return

    vv.load_state_dict(torch.load(os.path.join(a.workdir, "vae_video.pt")))
    av.load_state_dict(torch.load(os.path.join(a.workdir, "vae_audio.pt")))
    vv.eval(); av.eval()
    fit_latent_stats(vv, av, a.seed)
    log("latent std video", vv.latent_std.mean().item(), "audio", av.latent_std.mean().item())
    cp = os.path.join(a.workdir, "cache.pt")
    if os.path.exists(cp):
        cache = torch.load(cp)
    else:
        cache = build_cache(vv, av, a.cache, a.seed)
        torch.save(cache, cp)
    log("cache ready", {k: (len(v) if isinstance(v, list) else tuple(v.shape)) for k, v in cache.items()})

    pipe = build_pipeline(cfg, "cpu", torch.float32)
    pipe.video_vae.load_state_dict(vv.state_dict())
    pipe.audio_vae.load_state_dict(av.state_dict())
    mix = {"full": 0.60, "video": 0.06, "audio": 0.06, "ref": 0.14, "refine": 0.14}
    train_dit(pipe, cache, a.dit_steps, a.bs, a.lr, a.seed, mix)
    save_checkpoint(a.out, pipe, dtype=torch.float32, extra_meta={"trained_on": "h3turbo.toy synthetic dataset", "dit_steps": a.dit_steps})
    log("saved", a.out)


if __name__ == "__main__":
    main()
