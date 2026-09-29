"""`h3turbo` command line: generate / bench / init / quantize / info."""
from __future__ import annotations

import argparse
import json
import sys

import torch

from .config import make_config, tier_names


def _load(a):
    from .io import load_checkpoint

    dt = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = load_checkpoint(a.ckpt, device=dev, dtype=dt, quant=a.quant)
    if getattr(a, "offload", 0):
        from .offload import enable_block_swap

        enable_block_swap(pipe, resident=a.offload)
    return pipe


def _add_model_args(p):
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    p.add_argument("--quant", default=None, choices=[None, "int8", "int4"])
    p.add_argument("--offload", type=int, default=0, help="keep N blocks on the GPU, stream the rest (0 = off)")


def cmd_generate(a):
    from .media import load_image, read_wav, save_video
    from .pipeline import AudioCond, OmniContext, VideoCond

    pipe = _load(a)
    ctx = OmniContext()
    W, H = a.width, a.height
    n = a.frames or round(a.seconds * (a.fps or pipe.cfg.fps))
    if a.image:
        ctx.video.append(VideoCond(load_image(a.image, (H, W)), 0))
    if a.last_image:
        ctx.video.append(VideoCond(load_image(a.last_image, (H, W)), frame_index=n - 1))
    if a.audio_in:
        ctx.audio.append(AudioCond(read_wav(a.audio_in), 0.0))
    gen = ("video",) if a.no_audio else ("video", "audio")
    if a.audio_in and not a.audio_out:
        gen = ("video",)
    out = pipe(
        a.prompt, negative_prompt=a.negative, width=W, height=H, num_frames=a.frames, duration=None if a.frames else a.seconds,
        fps=a.fps, generate=gen, context=ctx, steps=a.steps, guidance=a.guidance, seed=a.seed,
    )
    if a.refine:
        out = pipe.refine(out, a.prompt, scale=a.refine, steps=2, seed=a.seed)
    print("wrote", save_video(a.out, out.video, out.fps, out.audio))


def cmd_bench(a):
    from .bench import estimate, format_report, measure

    if a.ckpt:
        from .io import read_metadata
        from .config import H3TurboConfig

        cfg = H3TurboConfig.from_json(read_metadata(a.ckpt)["h3turbo_config"])
    else:
        cfg = make_config(a.tier)
    print(format_report(estimate(cfg, a.width, a.height, a.seconds, a.steps), a.steps, a.tflops))
    if a.measure:
        if not a.ckpt:
            sys.exit("--measure needs --ckpt")
        pipe = _load(a)
        print(json.dumps(measure(pipe, width=a.width, height=a.height, duration=a.seconds, steps=a.steps), indent=1))


def cmd_init(a):
    from .io import build_pipeline, save_checkpoint

    cfg = make_config(a.tier)
    pipe = build_pipeline(cfg, "cpu", torch.float32)
    save_checkpoint(a.out, pipe, dtype=torch.float16, extra_meta={"trained": "no (random init)"})
    print(f"wrote {a.out}\nWARNING: randomly initialised, UNTRAINED. It has the right architecture and I/O but produces noise until trained.")


def cmd_quantize(a):
    from .io import load_checkpoint, save_checkpoint

    pipe = load_checkpoint(a.ckpt, device="cpu", dtype=torch.float32, quant=a.mode)
    save_checkpoint(a.out, pipe, dtype=torch.float16)
    print("wrote", a.out)


def cmd_info(a):
    from .io import read_metadata

    m = read_metadata(a.ckpt)
    print(json.dumps({k: (json.loads(v) if k == "h3turbo_config" else v) for k, v in m.items()}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="h3turbo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="text/image/audio -> video and/or audio")
    _add_model_args(g)
    g.add_argument("prompt", nargs="?", default="")
    g.add_argument("--negative", default="")
    g.add_argument("--out", default="out.mp4")
    g.add_argument("--width", type=int, default=512)
    g.add_argument("--height", type=int, default=320)
    g.add_argument("--seconds", type=float, default=2.0)
    g.add_argument("--frames", type=int, default=None)
    g.add_argument("--fps", type=int, default=None)
    g.add_argument("--steps", type=int, default=4)
    g.add_argument("--guidance", type=float, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--image", help="pin as the first frame (image-to-video)")
    g.add_argument("--last-image", help="pin as the last frame (with --image: first/last-frame-to-video)")
    g.add_argument("--audio-in", help="wav to condition on (audio-to-video)")
    g.add_argument("--audio-out", action="store_true", help="also generate audio when --audio-in is given")
    g.add_argument("--no-audio", action="store_true")
    g.add_argument("--refine", type=float, default=0, help="in-context regeneration upscale factor, e.g. 2")
    g.set_defaults(fn=cmd_generate)

    b = sub.add_parser("bench", help="size/FLOP estimate; --measure runs it for real")
    b.add_argument("--tier", default="base", choices=tier_names())
    b.add_argument("--ckpt")
    b.add_argument("--device", default=None)
    b.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    b.add_argument("--quant", default=None, choices=[None, "int8", "int4"])
    b.add_argument("--offload", type=int, default=0)
    b.add_argument("--width", type=int, default=512)
    b.add_argument("--height", type=int, default=320)
    b.add_argument("--seconds", type=float, default=4.0)
    b.add_argument("--steps", type=int, default=4)
    b.add_argument("--tflops", type=float, default=None, help="your assumed sustained TFLOPS, for a time estimate")
    b.add_argument("--measure", action="store_true")
    b.set_defaults(fn=cmd_bench)

    i = sub.add_parser("init", help="write an UNTRAINED checkpoint of a tier")
    i.add_argument("--tier", required=True, choices=tier_names())
    i.add_argument("--out", required=True)
    i.set_defaults(fn=cmd_init)

    q = sub.add_parser("quantize", help="write an int8/int4 copy of a checkpoint")
    q.add_argument("--ckpt", required=True)
    q.add_argument("--mode", required=True, choices=["int8", "int4"])
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_quantize)

    n = sub.add_parser("info", help="print a checkpoint's metadata/config")
    n.add_argument("--ckpt", required=True)
    n.set_defaults(fn=cmd_info)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
