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
from h3turbo.layout import Geometry
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
        acc.setdefault(task, {}).setdefault(key, []).append(float(ok))  # bools count as 0/1; PSNR stays a float

    for i, sc in enumerate(scenes):
        c, d = sc.color, sc.direction
        prompt = sc.prompt
        sd = seed * 1000 + i

        g = pipe(prompt, seed=sd, **kw)
        v, a = toy.analyze_video(g.video), toy.analyze_audio(g.audio)
        add("t2va", "video colour", v["color"] == c); add("t2va", "video direction", v["direction"] == d)
        add("t2va", "audio level=colour", a["color"] == c); add("t2va", "audio frequency=direction", a["direction"] == d)
        add("t2va", "audio level agrees with video colour", a["color"] == v["color"] and v["color"] is not None)

        g = pipe(prompt, generate="video", seed=sd, **kw)
        v = toy.analyze_video(g.video)
        add("t2v (no audio)", "video colour", v["color"] == c); add("t2v (no audio)", "video direction", v["direction"] == d)

        first = clip(sc)[:1]
        g = pipe(prompt, generate="video", context=OmniContext(video=[VideoCond(first, 0)]), seed=sd, **kw)
        v = toy.analyze_video(g.video)
        add("i2v", "video colour", v["color"] == c); add("i2v", "video direction", v["direction"] == d)
        # decoded frame 0 depends only on latent 0 (causal decoder), so a correctly pinned image must
        # equal the VAE's own roundtrip of it; this isolates pin correctness from VAE blur
        tok, _ = pipe.encode_video_tokens(first, HI, HI)
        rt = pipe.decode_video_tokens(tok, Geometry(HI, HI, 1, 8))[0]
        add("i2v", "first frame == VAE roundtrip of input (MAE<2/255)", (g.video[0].float() - rt.float()).abs().mean() < 2.0)

        fl = clip(sc)
        g = pipe("", generate="video", context=OmniContext(video=[VideoCond(fl[:1], 0), VideoCond(fl[-1:], T - 1)]), seed=sd, **kw)
        v = toy.analyze_video(g.video)
        add("first+last frame (no prompt)", "video colour", v["color"] == c); add("first+last frame (no prompt)", "video direction", v["direction"] == d)
        # the end frames are pinned, so their direction is given; what generation must get right is
        # the middle: it should sit halfway along the path (a static square would be ~14 px off)
        truth_c = toy.centroids(((fl.permute(0, 2, 3, 1) + 1) * 127.5).round().byte())
        got_c = toy.centroids(g.video)
        expected = (truth_c[0] + truth_c[-1]) / 2
        err = float(np.linalg.norm(got_c[T // 2] - expected)) if not np.isnan(got_c[T // 2]).any() else 99.0
        add("first+last frame (no prompt)", "middle frame within 8 px of the midpoint", err < 8.0)
        add("first+last frame (no prompt)", "mean middle-frame error (px; static square = 14)", err)
        # floor: the VAE's own reconstruction of the true clip, measured the same way
        tok_t, _ = pipe.encode_video_tokens(fl, HI, HI)
        rt_c = toy.centroids(pipe.decode_video_tokens(tok_t, Geometry(HI, HI, T, 8)))[T // 2]
        floor = float(np.linalg.norm(rt_c - expected)) if not np.isnan(rt_c).any() else 99.0
        add("first+last frame (no prompt)", "floor: VAE(true clip) middle-frame error (px)", floor)

        g = pipe("", generate="audio", context=OmniContext(video=[VideoCond(clip(sc), 0)]), seed=sd, **kw)
        a = toy.analyze_audio(g.audio)
        add("video->audio", "audio level=colour", a["color"] == c); add("video->audio", "audio frequency=direction", a["direction"] == d)

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
        # every decoded output is capped by the VAE's own reconstruction quality, so bicubic
        # (which never touches the VAE) is not a fair baseline. Also report the same bicubic
        # input run through the VAE with no regeneration, and the ceiling: VAE(truth).
        g96 = Geometry(HI, HI, T, 8)
        def through_vae(u8):
            tok, _ = pipe.encode_video_tokens(frames_from_uint8(u8), HI, HI)
            return pipe.decode_video_tokens(tok, g96)
        add("refine 64->96", "PSNR refined (dB)", psnr(r.video, truth))
        add("refine 64->96", "PSNR bicubic, no VAE (dB)", psnr(base, truth))
        add("refine 64->96", "PSNR bicubic through VAE, no regeneration (dB)", psnr(through_vae(base), truth))
        add("refine 64->96", "PSNR ceiling: VAE(truth) (dB)", psnr(through_vae(truth), truth))
        v = toy.analyze_video(r.video)
        add("refine 64->96", "video colour", v["color"] == c); add("refine 64->96", "video direction", v["direction"] == d)
    out = {}
    for t, m in acc.items():
        out[t] = {k: float(np.mean(x)) for k, x in m.items()}
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
    ap.add_argument("--quant", default=None, choices=[None, "int8", "int4"], help="weight-only quantisation of the transformer blocks")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pipe = load_checkpoint(a.ckpt, "cpu", torch.float32, quant=a.quant)
    res = {"steps": a.steps, "n": a.n, "chance": 0.25, "tasks": run(pipe, a.n, a.steps, a.seed)}
    if a.sweep:
        res["step_sweep"] = {}
        for s in (1, 2, 4, 8, 16):
            r = run_t2va(pipe, a.n, s, a.seed)
            res["step_sweep"][s] = r
            print(f"steps={s:2d}  video colour {r['video colour']:.2f} direction {r['video direction']:.2f} | audio level {r['audio level=colour']:.2f} frequency {r['audio frequency=direction']:.2f}", flush=True)
    print(f"\nn={a.n} per task, steps={a.steps}, chance=0.25")
    for t, m in res["tasks"].items():
        print(f"  {t}")
        for k, v in m.items():
            print(f"      {k:34s} {v:6.2f}" if not k.startswith("PSNR") else f"      {k:34s} {v:6.2f} dB")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def run_t2va(pipe, n, steps, seed):
    rng = np.random.default_rng(seed)
    hits = {"video colour": [], "video direction": [], "audio level=colour": [], "audio frequency=direction": []}
    for i in range(n):
        sc = toy.sample_scene(rng)
        g = pipe(sc.prompt, width=HI, height=HI, num_frames=T, steps=steps, seed=seed * 1000 + i)
        v, a = toy.analyze_video(g.video), toy.analyze_audio(g.audio)
        hits["video colour"].append(v["color"] == sc.color); hits["video direction"].append(v["direction"] == sc.direction)
        hits["audio level=colour"].append(a["color"] == sc.color); hits["audio frequency=direction"].append(a["direction"] == sc.direction)
    return {k: float(np.mean(x)) for k, x in hits.items()}


if __name__ == "__main__":
    main()
