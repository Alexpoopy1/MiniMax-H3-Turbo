# H3-Turbo

A compact, fast, **omni-modal** generator in the [MiniMax H3](https://www.minimax.io/news/minimax-h3-open-source) mould, designed for consumer GPUs (RTX 3050 and up): text, images, video and audio in; video and/or audio out, from one model, in a few sampling steps.

The small tiers below are a **new architecture that follows H3's design**, not a compressed copy of H3's weights. Separately, [`h3turbo/h3/`](#official-h3-4-bit-checkpoint) runs the **official H3 DiT** from a 4-bit W4A8/ConvRot checkpoint on a 6 GB card. Read [What is and isn't verified](#what-is-and-isnt-verified) before relying on any claim here.

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

## Official H3 (4-bit checkpoint)

`h3turbo/h3/` runs the real 50-block, hidden-5376 H3 DiT from a ComfyUI-layout **W4A8 + ConvRot** checkpoint (4-bit codebook weights, fp8 group scales, int8 activations on int8 tensor cores), streaming blocks from system RAM behind compute. No training or requantisation is involved: the conversion is a lossless re-layout.

```bash
h3turbo h3-convert  minimax_h3_..._w4a8_convrot.safetensors  model.h3t      # 12.55 GB -> 12.55 GB, verified byte for byte
h3turbo h3-info     model.h3t
h3turbo h3-bench    model.h3t --width 512 --height 320 --seconds 1.4       # real forwards on this GPU
```

```python
from h3turbo.h3.engine import H3Engine
with H3Engine.from_h3t("model.h3t") as eng:                  # keeps as many blocks resident as fit, streams the rest
    refined = eng.encode_text(qwen_states)                   # [1, L, 5120] Qwen3-VL states from your own encoder
    v_video, v_audio = eng.velocity([video_latent, audio_latent], sigma, refined_text=refined)
    video, audio = eng.sample((T, H, W), audio_frames, qwen_states, steps=8, seed=0)   # Euler, H3 shifts 12 / 3
```

**Measured** (RTX 3050 6 GB, about 4.9 GB free, PCIe Gen3 x8 at 6.5 GB/s, real checkpoint, random latents and text, bf16 compute, `scripts/h3_real_check.py`):

| tokens | example | native ComfyUI 0.37.4 + comfy_kitchen 0.2.35 | this engine | output vs native |
|---|---|---|---|---|
| 1,768 | 512x320, 33 frames | 3.39 s / forward | 2.55 s | bit-identical (max abs 0) at 6 sigmas |
| 4,448 | 512x320, 97 frames | 9.29 s | 8.60 s | bit-identical |
| 8,512 | 768x448, 4 s | not run | 25.5 s (peak VRAM 4.78 GiB) | not compared |
| 14,972 | 832x480, 5 s | 61.8 s (standalone native forward) | 44-46 s | bit-identical |

Each forward is one sampling step; the checkpoint is an 8-step turbo model, so a clip costs about 8x the figure. 13 of 50 blocks stay on the GPU, 37 are copied in behind compute with no stalls (`stats()['waits'] == 0`); the gain over native is the hidden copy time and it shrinks as sequences get longer and compute dominates. The linears run at 25-39 TOPS through comfy_kitchen's CUDA kernels; without comfy_kitchen a portable torch path is used (about 2x slower per block, measured by the qlinear check script).

**Quality.** The engine is the same network as ComfyUI's, so it has the checkpoint's quality, no more. Against the sibling int8 file, the 4-bit weights differ by 7.3% relative L2 per layer (checked on blocks 0, 25, 49); int8 activation quantisation adds about 1% per layer against an fp64 reference, and skipping it (`precision="a16"`) only moves the error against the int8 function from about 7.4% to 7.3%. Weight precision, not activations, is what limits fidelity. If you have the VRAM/RAM, the int8 sibling is the higher-quality file; this engine does not load it.

**Not done / not verified.** Text encoding (Qwen3-VL) and the VAEs are inputs and outputs of the engine, not part of it, and no clip has been generated through it, so there is no end-to-end visual quality check. Only the RTX 3050 was available: the "everything resident on a big card" plan is arithmetic, not a run. Parity holds only for identical inputs: with random inputs the network is chaotic (a 1-ulp change in sigma moved the video output by 6-12% in this test), so compare same-sigma outputs, and note ComfyUI computes `timestep / 1000` on the GPU (1 ulp off the exact value). No fp8/fp4 path, no multi-GPU. comfy_kitchen's INT8 attention is available as an opt-in (`attn_impl="int8"`, node option `int8_fast`); see the ComfyUI section for what it does to speed and to the output.

### In ComfyUI: H3-Turbo Fast UNET Loader

A node (`H3TurboFastUNetLoader`, category `H3-Turbo`) that replaces the core **UNETLoader** in an H3 workflow. It returns an ordinary MODEL, so your text encoder, `MiniMaxH3ImageToVideo`, `BasicScheduler`, `SamplerCustomAdvanced` and VAE nodes stay as they are; only the DiT forward runs through this engine.

1. Convert once: `h3turbo h3-convert <...>_w4a8_convrot.safetensors <ComfyUI>/models/h3turbo/<name>.h3t`
2. Install the node: clone this repo into `ComfyUI/custom_nodes/`, or link it (`mklink /J <ComfyUI>\custom_nodes\MiniMax-H3-Turbo <this repo>`) and restart ComfyUI.
3. In the workflow, delete UNETLoader, add **H3-Turbo Fast UNET Loader (h3t)**, pick the `.h3t`, and connect its MODEL where UNETLoader's went. Options: `precision` (a8 = native, a16 = unquantised activations, about 2x slower), `resident_blocks` (0 = whatever VRAM ComfyUI offers), `mlp_chunk` (0 = automatic), `attention` (`exact` = reference SDPA, identical output to UNETLoader; `int8_fast` = INT8 attention, faster but not identical, see below).

It cooperates with ComfyUI's memory manager: ComfyUI gives the model a VRAM budget, the engine fits its resident blocks and a 3-slot streaming ring inside it, and it hands all of it back (including the page-locked host copies) when the VAE or text encoder needs the card. Run-time LoRAs cannot be applied to a streamed 4-bit model, so adding one raises an error instead of being ignored; merge it into the checkpoint. Block-level model patches are not honoured.

**Measured** through a real ComfyUI server with your workflow (text to video, seed 42, `res_multistep`, 8 steps, RTX 3050 6 GB), `scripts/comfy_h3_e2e.py`:

| clip | | sampler | per step | wall total | decoded video and audio |
|---|---|---|---|---|---|
| 512x320, 22 frames | UNETLoader | 117-128 s | 2.0-2.7 s | 269-291 s | reference |
| | Fast loader | 88 s | 1.95 s | 196 s | identical, max diff 0 |
| 832x480, 124 frames (about 15k tokens) | UNETLoader | 406 s | 45.7 s | 614 s | reference |
| | Fast loader | 411 s | 44.8 s | 589 s | identical, max diff 0 |
| | Fast loader, `attention=int8_fast` | 321 s | 33.2 s | 496 s | different sample, see below |

So the loader pays off on short clips, where hiding weight copies matters, and is a wash at your 832x480, 5 s setting, where the 3050's compute is the limit (about half of each step is attention, the rest int8 GEMMs). Wall totals include text-encoder and VAE loading from a slow disk and vary by tens of seconds between runs; the sampler column is the reliable one. Single runs, not averages (the two native small runs agreed with each other bit for bit).

**`int8_fast` attention** halves the attention kernel time (comfy_kitchen's INT8 SDPA: int8 Q/K/V/P with a Hadamard rotation of Q and K) and cut the sampler 22% at 832x480x124, but it is not the reference numerics, so the result is a different sample rather than the same video: 26.1 dB PSNR from the native video, against 28.9 dB for a run that differed from native only in the last bit (this network amplifies tiny differences over 8 steps, so PSNR between two valid samples is not a quality score). On the one clip I compared, three frames side by side looked equally sharp and coherent. The audio deviates about 4x more than in the last-bit case (mean absolute difference 0.030 vs 0.008) and I did not listen to it. It also raised peak GPU memory to 5.8 GB in `nvidia-smi` (5.4 GB exact). One prompt and seed only: treat it as an option to try, not a proven equivalent. Fixing an earlier mismatch mattered here: at 15k tokens the patch-embedding GEMM must not be split into row chunks, or the last bit differs from ComfyUI and eight sampling steps amplify it into a visibly different video (PSNR 29 dB); it is now one GEMM and the videos are identical.

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

* **286 tests pass** (`pytest`, ~45 s; 5 CUDA-only tests skip on CPU). One test, `test_pipeline.py::test_compile_keeps_keys_and_matches_eager`, fails on this Windows machine because torch.compile finds no MSVC `cl`; it failed before the official-H3 engine was added and is unrelated to it. They check the invariants that matter rather than just shapes: cached AdaLN modulation equals the direct path bit for bit; adding a modality trains only its own weights; gradient checkpointing gives identical gradients; the video VAE is causal, and streamed decode equals whole-clip decode to 1e-4 at every chunk size (including ragged ones); every omni task's pin mask pins exactly the intended tokens and leaves pinned tokens untouched through denoising; a checkpoint round-trips bit-exact in fp32; quantised checkpoints round-trip; the analytic FLOP formula equals PyTorch's own counter exactly; the sampling schedule starts where the sample really is.
* **Real ComfyUI, end to end** (`scripts/comfyui_e2e.py`): the server is started headless, all six nodes register, and workflows submitted through its HTTP API produce a correct result. Text→video+audio muxed by core *Create Video* / *Save Video* to an mp4 (H.264 + AAC, audio and video the same length) that reads back as the requested colour and direction; a pinned image → video → *Refine* whose frames read back as the pinned colour moving the prompted way; audio-only generation.
* **The demo checkpoint does what it was trained to do**, measured against ground truth ([full tables](weights/README.md)): text → video+audio 1.00 colour and direction in the video, 0.99 / 0.94 in the audio; image → video 1.00, with the pinned frame reproduced exactly; video → audio 0.99 / 0.93; audio → video 0.90 / 1.00; reference → video 0.98 (chance is 0.25). Sampling steps: video is right at 1 step, audio needs about 4, and reflow distillation buys a real improvement at 1-2 steps (audio level 0.48 → 0.69 at 1 step).
* **Quantisation, measured on the trained model** (same evaluation, n=96): int8 is indistinguishable from fp32 (every metric within 0.01, identical interpolation error); int4, plain round-to-nearest with no calibration, costs a little (video → audio level 0.99 → 0.90, text → audio level 0.99 → 0.96, first+last-frame middle-frame error 8.2 → 8.8 px) and leaves the rest unchanged. This is a 5 M-parameter model, and I did not measure the larger tiers, so do not extrapolate either way.

### Not verified, or not done

* **No GPU was available, so nothing here has run on CUDA and there are no RTX 3050 timings.** I do not claim any speed number. The design choices that should make it fast (a small model, 2-4 steps, cached AdaLN, SDPA/flash, weight-only quantisation, streamed VAE decode) are real, and `h3turbo bench --measure` will give you the actual figures on your card, but until someone runs it, "very fast on an RTX 3050" is an intent, not a result. The CUDA-only paths (block-swap stream prefetch, bf16/fp16 flash attention, `torch.compile` on GPU) are written but unexercised; on CPU the tests cover the bookkeeping and the numerics, not stream overlap.
* **There is no strong, general model here.** Only the 5 M `nano` demo is trained, and only on a toy dataset (its "audio" is inaudible infrasound). The `small` … `xl` tiers are architecture and shape-correct random initialisations. Making one of them good means training it on real data at real scale, which this repository provides the code for but did not do.
* **The small tiers do not use official H3 weights** and are not distilled from them. Official H3 runs separately through [`h3turbo/h3/`](#official-h3-4-bit-checkpoint), which reads the real checkpoint layout.
* **Refinement has no measured benefit yet.** On the toy it keeps colour and direction (1.00) but is indistinguishable from a no-op in PSNR because the VAE (19.3 dB roundtrip) is the ceiling. Whether in-context regeneration improves real content is untested. The optional context resampler is implemented and shape-tested but untrained.
* **First + last frame interpolation is only partial** on the toy: the middle frame lands 8.2 px from the ideal against 14.4 for a static square (the VAE alone allows 2.3).
* int4 is plain round-to-nearest with no calibration; FP8 / NVFP4 are not implemented (Ampere has no FP8 tensor cores; Ada/Blackwell would). No guidance-distillation script. The pipeline generates one video at a time. The mp4 writer needs PyAV.
* No licence file is included; that choice is yours. If you use official H3 weights or code alongside this, its own licence applies to them.
