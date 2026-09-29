#!/usr/bin/env python
"""Step distillation by reflow (rectified-flow "2-rectification").

The teacher integrates its full ODE from noise; each (noise, output) pair is a straight
target, and the student is fine-tuned with the ordinary flow-matching loss on those
coupled pairs. Trajectories get straighter, so a few Euler steps land near the teacher's
many-step result. Original-data batches are mixed in so the other structures (audio-only,
reference, refine) are not forgotten.

    python scripts/reflow.py --ckpt teacher.safetensors --cache cache.pt --out student.safetensors

`--cache` is the latent cache written by train_toy.py (prompts + tokens). For a real run
the same script works with your own prompt list and cache; the teacher can be any
H3-Turbo checkpoint.
"""
import argparse
import time

import numpy as np
import torch

from h3turbo.io import load_checkpoint, save_checkpoint
from h3turbo.training import EMA, Geo, cosine_lr, flow_loss

HI, LO, FPS = 96, 64, 8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@torch.no_grad()
def make_pairs(teacher, prompts, steps, seed, n):
    rng = np.random.default_rng(seed)
    out = {"prompts": [], "video": [], "audio": [], "eps_video": [], "eps_audio": []}
    for i in range(n):
        p = prompts[int(rng.integers(len(prompts)))]
        g = teacher(p, width=HI, height=HI, num_frames=9, fps=FPS, steps=steps, seed=seed * 100003 + i, return_latents=True)
        out["prompts"].append(p)
        out["video"].append(g.latents["video"][0]); out["eps_video"].append(g.noise["video"][0])
        out["audio"].append(g.latents["audio"][0]); out["eps_audio"].append(g.noise["audio"][0])
        if i % 200 == 0:
            log(f"pairs {i}/{n}")
    return {k: (v if k == "prompts" else torch.stack(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher-steps", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=3000)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--replay", type=float, default=0.3, help="fraction of steps on original data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    student = load_checkpoint(a.ckpt, "cpu", torch.float32)
    teacher = load_checkpoint(a.ckpt, "cpu", torch.float32)
    cache = torch.load(a.cache)
    log("sampling teacher pairs with", a.teacher_steps, "steps")
    pairs = make_pairs(teacher, cache["prompts"], a.teacher_steps, a.seed, a.pairs)

    model, te = student.model, student.text_encoder
    params = list(model.parameters()) + list(te.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.01)
    ema = EMA(params, 0.995)
    rng, gen = np.random.default_rng(a.seed + 1), torch.Generator().manual_seed(a.seed + 2)
    geo = Geo(lat_t=3, hp=HI // 32, wp=HI // 32, fps=FPS, lo_hp=LO // 32, lo_wp=LO // 32)
    model.train(); te.train()
    n_pairs, n_cache = len(pairs["prompts"]), len(cache["prompts"])
    for step in range(a.iters):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, a.iters, a.lr, warmup=50)
        if rng.random() > a.replay:
            idx = rng.integers(0, n_pairs, size=a.bs)
            batch = {k: ([pairs[k][i] for i in idx] if k == "prompts" else pairs[k][idx]) for k in pairs}
            structure = "full"
        else:
            idx = rng.integers(0, n_cache, size=a.bs)
            batch = {k: ([cache[k][i] for i in idx] if k == "prompts" else cache[k][idx]) for k in cache}
            structure = ["full", "video", "audio", "ref", "refine"][int(rng.choice(5, p=[0.3, 0.1, 0.1, 0.25, 0.25]))]
        loss, _ = flow_loss(model, te, batch, structure, geo, rng, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ema.update()
        if step % 100 == 0 or step == a.iters - 1:
            log(f"reflow {step}/{a.iters} {structure} loss={loss.item():.4f}")
    ema.copy_to()
    model.eval(); te.eval()
    save_checkpoint(a.out, student, dtype=torch.float32, extra_meta={"distilled": f"reflow from {a.teacher_steps}-step teacher"})
    log("saved", a.out)


if __name__ == "__main__":
    main()
