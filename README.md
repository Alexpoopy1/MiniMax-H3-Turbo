# H3-Turbo

A compact, fast, **omni-modal** generator in the [MiniMax H3](https://www.minimax.io/news/minimax-h3-open-source) mould, designed for consumer GPUs (RTX 3050 and up): text, images, video and audio in; video and/or audio out, from one model, in a few sampling steps.

It is a **new small architecture that follows H3's design**, not a compressed copy of H3's weights. Read [What is and isn't verified](#what-is-and-isnt-verified) before relying on any claim here.

## Why a separate small model

Official H3 is a 33B dense transformer plus a Qwen3-VL-32B text encoder. Even with 4-bit weights that is far beyond a 6-8 GB card, so on an RTX 3050 it can only run by streaming weights from system RAM, which is slow by construction. ComfyUI already supports official H3 (with community GGUF/INT4 quants) for cards that can take it. H3-Turbo is for the other end: an architecture small enough to sit entirely in a 3050's VRAM and sample in 2-4 steps.

## What it takes from H3

| H3 idea | H3-Turbo |
|---|---|
| Single-stream dense transformer, no modality-specific attention/FFN | Same. Modality-specific weights are confined to input/output layers and AdaLN branches |
| AdaLN modulation depends only on (modality, timestep), so it can be cached and its weights dropped at inference | Same: modulation is precomputed once per sampling schedule and the AdaLN bank stays on the CPU (about 25-40% of the parameters never touch the GPU) |
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

TBD_QUICKSTART

## Sizes

TBD_SIZES

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

TBD_STATUS
