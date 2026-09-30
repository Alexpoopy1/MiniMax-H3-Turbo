# H3-Turbo

A compact, fast, **omni-modal** generator in the [MiniMax H3](https://www.minimax.io/news/minimax-h3-open-source) mould, designed for consumer GPUs (RTX 3050 and up): text, images, video and audio in; video and/or audio out, from one model, in a few sampling steps.

It is a **new small architecture that follows H3's design**, not a compressed copy of H3's weights. Read [What is and isn't verified](#what-is-and-isnt-verified) before relying on any claim here.

## Why a separate small model

Official H3 is a 33B dense transformer plus a Qwen3-VL-32B text encoder. Even with 4-bit weights that is far beyond a 6-8 GB card, so on an RTX 3050 it can only run by streaming weights from system RAM, which is slow by construction. ComfyUI already supports official H3 (with community GGUF/INT4 quants) for cards that can take it. H3-Turbo is for the other end: an architecture small enough to sit entirely in a 3050's VRAM and sample in 2-4 steps.

## What it takes from H3

| H3 idea | H3-Turbo |
|---|---|
| Single-stream dense transformer, no modality-specific attention/FFN | Same. Modality-specific weights are confined to input/output layers and AdaLN branches |
| AdaLN modulation depends only on (modality, timestep), so it can be cached and its weights dropped at inference | Same: modulation is precomputed once per sampling schedule and the AdaLN bank stays on the CPU (13-31% of a tier's parameters, 23% for `base`, never touch the GPU; official H3 puts about 13 B of its 33 B there) |
| 3D RoPE over (time, height, width) | Same; audio tokens sit on the video time axis at fractional positions |
| Video VAE f16t4d24 (16x spatial, 4x temporal, 24 channels, causal), patch 1x2x2 | Same layout; first frame is encoded alone so an image is a valid 1-frame video |
| Audio VAE at 40 latent tokens/s, 32 kHz | Same rate |
| In-context regeneration instead of a separate super-resolution network | `pipe.refine()`: the same model regenerates at higher resolution while re-reading the low-res result as clean context |
| Compressing long multimodal context to a short token budget | Optional Perceiver resampler (architecture + tests; not trained in the demo checkpoint) |
| Omni tasks (text/first-last-frame/reference to video+audio, video to audio, ...) | One mechanism: which tokens are pinned clean. See below |

## Omni tasks

Every task is a choice of pin mask over one token sequence, so there is one code path and one model:

| task | how |
|---|---|
| text → video + audio | nothing pinned |
| image → video | `VideoCond(image, frame_index=0)` |
| first + last frame → video | two `VideoCond`s (`frame_index=0` and `num_frames-1`) |
| video extension / continuation | pin the leading frames |
| inpainting / editing | `VideoCond(..., mask=...)` pins only the kept region |
| video → audio | pin the whole video, `generate="audio"` |
| audio → video | `AudioCond(wave)`, `generate="video"` |
| reference (identity / style / voice) | `OmniContext(ref_video=[...], ref_audio=[...])`, kept off the timeline |
| upscale | `pipe.refine(result, scale=2)` |
| new modality (depth, pose, mask, ...) | `model.add_modality(ModalitySpec(...))`, see [Extending](#adding-a-modality) |

## Quickstart

```bash
pip install -e .            # torch, safetensors, numpy
pip install av pillow       # optional: mp4 output, image input
```

A trained demo checkpoint is included: [`weights/h3turbo-nano-toy.safetensors`](weights/README.md). **It is a 5 M-parameter model trained on a synthetic dataset of a coloured square moving in four directions, and it can do nothing else.** It exists so the whole stack (every omni task, the ComfyUI nodes, quantisation, refinement) is exercised end to end by something that has actually learned, and so the results below are measurable against ground truth.

```python
import torch
from h3turbo.pipeline import H3TurboPipeline, OmniContext, VideoCond, AudioCond
from h3turbo.media import save_video, load_image

pipe = H3TurboPipeline.from_pretrained("weights/h3turbo-nano-toy.safetensors", device="cpu")  # "cuda" on a GPU

# text -> video + audio
out = pipe("a red square moving left", width=96, height=96, num_frames=9, fps=8, steps=4, seed=0)
save_video("t2va.mp4", out.video, out.fps, out.audio)

# image -> video: pin frame 0 (any IMAGE; here first.png)
ctx = OmniContext(video=[VideoCond(load_image("first.png", (96, 96)), frame_index=0)])
out = pipe("a blue square moving right", context=ctx, generate="video", width=96, height=96, num_frames=9, fps=8, seed=1)

# first + last frame -> video (pin both ends)
first, last = load_image("first.png", (96, 96)), load_image("last.png", (96, 96))
ctx = OmniContext(video=[VideoCond(first, 0), VideoCond(last, frame_index=8)])
out = pipe("", context=ctx, generate="video", width=96, height=96, num_frames=9, fps=8, seed=2)

# video -> audio: pin the whole clip, generate only the sound
clip = torch.cat([first] * 9)                                   # [T, 3, H, W] in [-1, 1]
out = pipe("", context=OmniContext(video=[VideoCond(clip, 0)]), generate="audio", width=96, height=96, num_frames=9, fps=8, seed=3)

# in-context regeneration at a higher resolution
big = pipe.refine(pipe("a green square moving up", width=64, height=64, num_frames=9, fps=8, seed=4), scale=1.5)
print(big.video.shape)  # torch.Size([9, 96, 96, 3])
```

Or from the shell: `h3turbo generate "a red square moving left" --ckpt weights/h3turbo-nano-toy.safetensors --width 96 --height 96 --frames 9 --fps 8 --out out.mp4` (also `--image`, `--last-image`, `--audio-in`, `--refine 2`, `--quant int8`, `--offload N`).

Untrained tiers: `h3turbo init --tier base --out base.safetensors`, then `h3turbo quantize` / `h3turbo bench`.

## Sizes

Five tiers share one design and differ only in width and depth (`h3turbo init --tier <name>` builds any of them; configs are in [`configs/`](configs)). The table is an **analytic plan**, generated by `scripts/gen_docs.py`: parameter counts and FLOPs are exact counts of the code, GPU memory is weights plus an activation estimate plus 0.8 GB headroom. No GPU was available when it was produced, so it contains **no timings**. See [`docs/SIZING.md`](docs/SIZING.md) for more resolutions.

512x320, 4 s (97 frames at 24 fps), 4 sampling steps, audio on:

| tier | params on GPU¹ | GPU weights fp16 / int8 / int4 | TFLOP per generation² | fits (estimate): 4 GB / 6 GB / 8 GB card |
|---|---|---|---|---|
| nano | 3 M | 0.01 GB | 2 | yes (this is the trained demo) |
| small | 206 M | 0.49 / 0.30 / 0.21 GB | 23 | fp16 / fp16 / fp16 |
| base | 681 M | 1.45 / 0.83 / 0.52 GB | 45 | fp16 / fp16 / fp16 |
| large | 2.04 B | 4.16 / 2.29 / 1.36 GB | 101 | int8 / fp16 / fp16 |
| xl | 4.08 B | 8.26 / 4.54 / 2.68 GB | 181 | offload / int8 / int8 |

¹ Excludes the AdaLN bank (31% of `small`, 23% of `base`, 17% of `large`, 13% of `xl`), which is cached per sampling schedule and stays on the CPU. ² Transformer forwards plus the VAE decode; the formula matches PyTorch's own FLOP counter exactly (tested).

Why this should be fast, from the design rather than from a measurement: a forward pass costs roughly what its active parameters cost, and `base` has about 0.7 B on the GPU against roughly 20 B active in official H3; sampling is 2-4 steps with no CFG doubling; attention runs through PyTorch SDPA (flash on Ampere); the VAE decodes in exact streamed chunks. Turn FLOPs into seconds with your card's real throughput: `h3turbo bench --tier base --tflops <sustained TFLOPS you measured>`, or better, `h3turbo init --tier base --out base.safetensors` then `h3turbo bench --ckpt base.safetensors --measure` on the card itself.

**RTX 3050 specifics.** Ampere has bf16 and int8 tensor cores but **no FP8**, so the low-precision path here is weight-only int8/int4 (`--quant int8`), not FP8/NVFP4 (which need Ada/Blackwell and are not implemented). Cards come with 4, 6 or 8 GB: `base` fp16 is the comfortable choice at 6 GB, `large` int8 at 6-8 GB. If a tier does not fit, `--offload N` keeps N blocks resident and streams the rest from system RAM.

## ComfyUI

Clone the repo into `ComfyUI/custom_nodes/` (no pip install needed) and put checkpoints in `ComfyUI/models/h3turbo/`.

| node | purpose |
|---|---|
| **H3-Turbo Loader** | checkpoint → pipeline; precision, `int8`/`int4`, block offload |
| **H3-Turbo Pin Frames** | an IMAGE (one frame = image conditioning, a batch = a clip) pinned at a frame index; optional MASK for inpainting |
| **H3-Turbo Pin Audio** | AUDIO pinned at a start time |
| **H3-Turbo Reference** | IMAGE / AUDIO references kept off the timeline (identity, style, voice) |
| **H3-Turbo Generate** | prompt (+ context) → IMAGE, AUDIO, fps; toggles for generating video and/or audio |
| **H3-Turbo Refine** | in-context regeneration upscale of a Generate result |

`Pin Frames`, `Pin Audio` and `Reference` chain through `omni_context`, so adding a new condition is one more node. Wire `images`, `audio` and `fps` into ComfyUI's core **Create Video** and **Save Video**.

This is tested against a real ComfyUI (see [verification](#what-is-and-isnt-verified)): `python scripts/comfyui_e2e.py --comfy /path/to/ComfyUI --ckpt weights/h3turbo-nano-toy.safetensors` starts the server headless, submits text→video+audio→mp4, image→video→refine, and audio-only workflows through its HTTP API, and checks the outputs.

The nodes are ComfyUI's stable `NODE_CLASS_MAPPINGS` style, and the pipeline is plain PyTorch, so the same checkpoints work from Python, the CLI, or any other front end you put around `H3TurboPipeline`.

## Training and distillation

Everything needed to train is in the package; `scripts/train_toy.py` is a complete, runnable example of how the pieces fit, and only the data source changes for a real run.

* `h3turbo.training.flow_loss` builds task-mixed batches through the same `build_segments` the pipeline uses, so training and inference cannot drift apart. Each step draws a *structure* (which token groups exist: text+video+audio, video only, audio only, +reference, +low-res context for refine); inside the main structure every sample gets its own pin pattern (none / first frame / first+last / prefix / whole video / whole audio). That is how one model learns text-to-video, image-to-video, first/last-frame, extension, video-to-audio and audio-to-video together. Text is dropped 10% of the time.
* `scripts/reflow.py` is step distillation: the teacher integrates its full ODE, each (noise, output) pair is a straight target, and the student is fine-tuned on those pairs so a few Euler steps land near the teacher's result.
* Guidance can be distilled in (`guidance_embed=True`, one forward per step instead of two), but there is **no** guidance-distillation script yet.

**To get a strong model** you need to train one of the larger tiers on real data at real scale; that takes GPU-days to GPU-weeks and is not something this repository ships. The most direct route to quality is distillation from official H3 (open weights): sample (prompt, noise, latents) pairs from it and train a tier on them with `flow_loss`. That loader/teacher bridge is **not implemented here**: official-checkpoint loading needs the exact H3 key layout, which I could not read from this environment.

`h3turbo init --tier base --out base.safetensors` writes an **untrained** checkpoint of any tier. It produces noise, but it has the right shapes, so `h3turbo bench --ckpt base.safetensors --measure` gives you real speed and VRAM numbers for your GPU before you spend anything on training (speed does not depend on the weight values).

### Adding a modality

Modality-specific weights are only the input projection, output projection and AdaLN branch, so a new token type is a small addition, trainable on its own with everything else frozen:

```python
from h3turbo.config import ModalitySpec

model = pipe.model
model.add_modality(ModalitySpec("depth", in_dim=16, out_dim=16), like="video")
trainable = model.freeze_shared()   # only depth's IO layers and AdaLN branch require grad
# run the model with an extra group: groups=GROUPS + ["depth"], group_t with one more column,
# and a Segment("depth", tokens, pos, group_index, emit=True); see tests/test_model.py
# (test_add_modality_trains_alone) for a complete, runnable example.
```

## What is and isn't verified

### Verified, by running it here (CPU only)

* **90 tests pass** (`pytest`, ~25-45 s). They check the invariants that matter rather than just shapes: cached AdaLN modulation equals the direct path bit for bit; adding a modality trains only its own weights; gradient checkpointing gives identical gradients; the video VAE is causal, and streamed decode equals whole-clip decode to 1e-4 at every chunk size (including ragged ones); every omni task's pin mask pins exactly the intended tokens and leaves pinned tokens untouched through denoising; a checkpoint round-trips bit-exact in fp32; quantised checkpoints round-trip; the analytic FLOP formula equals PyTorch's own counter exactly; the sampling schedule starts where the sample really is.
* **Real ComfyUI, end to end** (`scripts/comfyui_e2e.py`): the server is started headless, all six nodes register, and workflows submitted through its HTTP API produce a correct result. Text→video+audio muxed by core *Create Video* / *Save Video* to an mp4 (H.264 + AAC, audio and video the same length) that reads back as the requested colour and direction; a pinned image → video → *Refine* whose frames read back as the pinned colour moving the prompted way; audio-only generation.
* **The demo checkpoint does what it was trained to do**, measured against ground truth ([full tables](weights/README.md)): text → video+audio 1.00 colour and direction in the video, 0.99 / 0.94 in the audio; image → video 1.00, with the pinned frame reproduced exactly; video → audio 0.99 / 0.93; audio → video 0.90 / 1.00; reference → video 0.98 (chance is 0.25). Sampling steps: video is right at 1 step, audio needs about 4, and reflow distillation buys a real improvement at 1-2 steps (audio level 0.48 → 0.69 at 1 step).
* **Quantisation, measured on the trained model** (same evaluation, n=96): int8 is indistinguishable from fp32 (every metric within 0.01, identical interpolation error); int4, plain round-to-nearest with no calibration, costs a little (video → audio level 0.99 → 0.90, text → audio level 0.99 → 0.96, first+last-frame middle-frame error 8.2 → 8.8 px) and leaves the rest unchanged. This is a 5 M-parameter model, and I did not measure the larger tiers, so do not extrapolate either way.

### Not verified, or not done

* **No GPU was available, so nothing here has run on CUDA and there are no RTX 3050 timings.** I do not claim any speed number. The design choices that should make it fast (a small model, 2-4 steps, cached AdaLN, SDPA/flash, weight-only quantisation, streamed VAE decode) are real, and `h3turbo bench --measure` will give you the actual figures on your card, but until someone runs it, "very fast on an RTX 3050" is an intent, not a result. The CUDA-only paths (block-swap stream prefetch, bf16/fp16 flash attention, `torch.compile` on GPU) are written but unexercised; on CPU the tests cover the bookkeeping and the numerics, not stream overlap.
* **There is no strong, general model here.** Only the 5 M `nano` demo is trained, and only on a toy dataset (its "audio" is inaudible infrasound). The `small` … `xl` tiers are architecture and shape-correct random initialisations. Making one of them good means training it on real data at real scale, which this repository provides the code for but did not do.
* **Official MiniMax H3 weights are not loaded or distilled from.** I could not reach Hugging Face from this environment, so I could not read the checkpoint layout, and I did not guess at it. Official H3 already runs in ComfyUI natively; this project is a separate, small model.
* **Refinement has no measured benefit yet.** On the toy it keeps colour and direction (1.00) but is indistinguishable from a no-op in PSNR because the VAE (19.3 dB roundtrip) is the ceiling. Whether in-context regeneration improves real content is untested. The optional context resampler is implemented and shape-tested but untrained.
* **First + last frame interpolation is only partial** on the toy: the middle frame lands 8.2 px from the ideal against 14.4 for a static square (the VAE alone allows 2.3).
* int4 is plain round-to-nearest with no calibration; FP8 / NVFP4 are not implemented (Ampere has no FP8 tensor cores; Ada/Blackwell would). No guidance-distillation script. The pipeline generates one video at a time. The mp4 writer needs PyAV.
* No licence file is included; that choice is yours. If you use official H3 weights or code alongside this, its own licence applies to them.
