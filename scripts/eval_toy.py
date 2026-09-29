#!/usr/bin/env python
"""Measure every omni task of a toy-trained checkpoint against ground truth.

Chance level for colour / direction is 25% (4 classes). The analyzers are validated in
tests: they recover ground truth 100% from clean renders/audio, so a shortfall here is
the model's, not the metric's.
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from h3turbo import toy
from h3turbo.io import load_checkpoint
from h3turbo.pipeline import AudioCond, Generation, OmniContext, VideoCond, frames_from_uint8

HI, LO, T = 96, 64, 9


def clip(scene, side=HI):
    return toy.render(scene, side, T)  # [T,3,H,W] in [-1,1]


def wave(scene):
    return toy.synth_audio(scene, 45 * 800)


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return 10 * np.log10(255.0**2 / max(mse, 1e-9))


def run(pipe, n, steps, seed):
    rng = np.random.default_rng(seed)
    scenes = [toy.sample_scene(rng) for _ in range(n)]
    kw = dict(width=HI, height=HI, num_frames=T, steps=steps)
    acc = {}

    def add(task, key, ok):
        acc.setdefault(task, {}).setdefault(key, []).append(bool(ok))

    for i, sc in enumerate(scenes):
        c, d = sc.color, sc.direction
        prompt = sc.prompt
        sd = seed * 1000 + i

        g = pipe(prompt, seed=sd, **kw)
        v, a = toy.analyze_video(g.video), toy.analyze_audio(g.audio)
        add("t2va", "video colour", v["color"] == c); add("t2va", "video direction", v["direction"] == d)
        add("t2va", "audio pitch=colour", a["color"] == c); add("t2va", "audio tremolo=direction", a["direction"] == d)
        add("t2va", "audio/video agree on colour", a["color"] == v["color"] and v["color"] is not None)

        g = pipe(prompt, generate="video", seed=sd, **kw)
        v = toy.analyze_video(g.video)
        add("t2v (no audio)", "video colour", v["color"] == c); add("t2v (no audio)", "video direction", v["direction"] == d)

        first = clip(sc)[:1]
        g = pipe(prompt, generate="video", context=OmniContext(video=[VideoCond(first, 0)]), seed=sd, **kw)
        v = toy.analyze_video(g.video)
        add("i2v", "video colour", v["color"] == c); add("i2v", "video direction", v["direction"] == d)
        add("i2v", "first frame kept (MAE<0.08)", (frames_from_uint8(g.video)[0] - first[0]).abs().mean() < 0.08 * 2)

        fl = clip(sc)
        g = pipe("", generate="video", context=OmniContext(video=[VideoCond(fl[:1], 0), VideoCond(fl[-1:], T - 1)]), seed=sd, **kw)
        v = toy.analyze_video(g.video)
        add("first+last frame (no prompt)", "video colour", v["color"] == c); add("first+last frame (no prompt)", "video direction", v["direction"] == d)

        g = pipe("", generate="audio", context=OmniContext(video=[VideoCond(clip(sc), 0)]), seed=sd, **kw)
        a = toy.analyze_audio(g.audio)
        add("video->audio", "audio pitch=colour", a["color"] == c); add("video->audio", "audio tremolo=direction", a["direction"] == d)

        g = pipe("", generate="video", context=OmniContext(audio=[AudioCond(wave(sc), 0.0)]), seed=sd, **kw)
        v = toy.analyze_video(g.video)
        add("audio->video (no prompt)", "video colour", v["color"] == c); add("audio->video (no prompt)", "video direction", v["direction"] == d)

        ref = clip(sc)[3:4]
        g = pipe("", generate="video", context=OmniContext(ref_video=[ref]), seed=sd, **kw)
        v = toy.analyze_video(g.video)
        add("reference image (no prompt)", "video colour", v["color"] == c)

        lo = clip(sc, LO)
        lo_gen = Generation(((lo.permute(0, 2, 3, 1) + 1) * 127.5).round().byte(), wave(sc), 8)
        r = pipe.refine(lo_gen, prompt, width=HI, height=HI, steps=2, seed=sd)
        truth = ((clip(sc).permute(0, 2, 3, 1) + 1) * 127.5).round().byte()
        up = F.interpolate(lo, size=(HI, HI), mode="bicubic", align_corners=False).clamp(-1, 1)
        base = ((up.permute(0, 2, 3, 1) + 1) * 127.5).round().byte()
        add("refine 64->96", "PSNR refined (dB)", psnr(r.video, truth))
        add("refine 64->96", "PSNR bicubic (dB)", psnr(base, truth))
        v = toy.analyze_video(r.video)
        add("refine 64->96", "video colour", v["color"] == c); add("refine 64->96", "video direction", v["direction"] == d)
    out = {}
    for t, m in acc.items():
        out[t] = {k: (float(np.mean(x)) if not k.startswith("PSNR") else float(np.mean(x))) for k, x in m.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sweep", action="store_true", help="also sweep step counts on text->video+audio")
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pipe = load_checkpoint(a.ckpt, "cpu", torch.float32)
    res = {"steps": a.steps, "n": a.n, "chance": 0.25, "tasks": run(pipe, a.n, a.steps, a.seed)}
    if a.sweep:
        res["step_sweep"] = {}
        for s in (1, 2, 4, 8, 16):
            r = run_t2va(pipe, a.n, s, a.seed)
            res["step_sweep"][s] = r
            print(f"steps={s:2d}  video colour {r['video colour']:.2f} direction {r['video direction']:.2f} | audio pitch {r['audio pitch=colour']:.2f} tremolo {r['audio tremolo=direction']:.2f}", flush=True)
    print(f"\nn={a.n} per task, steps={a.steps}, chance=0.25")
    for t, m in res["tasks"].items():
        print(f"  {t}")
        for k, v in m.items():
            print(f"      {k:34s} {v:6.2f}" if not k.startswith("PSNR") else f"      {k:34s} {v:6.2f} dB")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def run_t2va(pipe, n, steps, seed):
    rng = np.random.default_rng(seed)
    hits = {"video colour": [], "video direction": [], "audio pitch=colour": [], "audio tremolo=direction": []}
    for i in range(n):
        sc = toy.sample_scene(rng)
        g = pipe(sc.prompt, width=HI, height=HI, num_frames=T, steps=steps, seed=seed * 1000 + i)
        v, a = toy.analyze_video(g.video), toy.analyze_audio(g.audio)
        hits["video colour"].append(v["color"] == sc.color); hits["video direction"].append(v["direction"] == sc.direction)
        hits["audio pitch=colour"].append(a["color"] == sc.color); hits["audio tremolo=direction"].append(a["direction"] == sc.direction)
    return {k: float(np.mean(x)) for k, x in hits.items()}


if __name__ == "__main__":
    main()
