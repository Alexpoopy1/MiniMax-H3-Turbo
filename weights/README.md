# Demo checkpoints

| file | what it is |
|---|---|
| `h3turbo-nano-toy.safetensors` | 5 M-parameter `nano` tier, trained on the synthetic toy dataset, plain flow matching (best at 4+ steps) |
| `h3turbo-nano-toy-turbo.safetensors` | the same model after `scripts/reflow.py` step distillation (better at 1-2 steps) |

Both are single-file checkpoints (transformer, both VAEs, text encoder; config in the file metadata), 37 MB each in fp32. `h3turbo info --ckpt <file>` prints the metadata.

**These are not general models.** They were trained on `h3turbo.toy`: a coloured square (red / green / blue / yellow) moving left / right / up / down, with prompts like "a blue square moving up". The "audio" in that dataset is a **3-12 Hz sine, i.e. infrasound you cannot hear**: amplitude encodes the colour, frequency encodes the direction. (Audible tones and noise bursts were tried first; a tiny CPU-trained audio VAE learns neither. See `h3turbo/toy.py`.) They exist so that every omni task, the ComfyUI nodes, quantisation and refinement are exercised end to end by something that has genuinely learned, and so quality can be scored against ground truth. They say nothing about how a larger model would do on real video or sound.

## Results

`scripts/eval_toy.py --n 96 --steps 4`, CPU, 96 fresh scenes per task, chance is 0.25. Colour and direction are read back from the generated pixels / waveform by analyzers that recover the ground truth **100%** of the time from clean data (tested in `tests/test_toy.py`), so a shortfall below is the model's, not the metric's. Raw numbers: `eval_nano_toy.json`, `eval_nano_toy_turbo.json`. Repeat runs of the same weights differ by about 0.01 (floating-point nondeterminism, one sample of 96).

| task | metric | toy | turbo |
|---|---|---|---|
| text → video + audio | video colour / direction | 1.00 / 1.00 | 1.00 / 1.00 |
| | audio level (=colour) / frequency (=direction) | 0.99 / 0.94 | 0.98 / 0.93 |
| | audio agrees with the video's colour | 0.99 | 0.98 |
| text → video only | colour / direction | 1.00 / 1.00 | 1.00 / 1.00 |
| image → video | colour / direction | 1.00 / 1.00 | 1.00 / 1.00 |
| | decoded first frame == VAE roundtrip of the input image (the pin is exact) | 1.00 | 1.00 |
| first + last frame → video (no prompt) | middle frame within 8 px of the true midpoint | 0.55 | 0.59 |
| | mean middle-frame error, px (static square: 14.4; VAE-only floor: 2.3) | 8.2 | 7.4 |
| video → audio (no prompt) | level (=colour) / frequency (=direction) | 0.99 / 0.93 | 1.00 / 0.92 |
| audio → video (no prompt) | colour / direction | 0.90 / 1.00 | 0.90 / 0.98 |
| reference image → video (no prompt) | colour | 0.98 | 0.93 |
| refine 64 → 96 px | colour / direction preserved | 1.00 / 1.00 | 1.00 / 1.00 |

Step count, text → video + audio (video colour/direction | audio level/frequency):

| steps | toy | turbo |
|---|---|---|
| 1 | 1.00/1.00 \| 0.48/0.86 | 1.00/1.00 \| 0.69/0.86 |
| 2 | 1.00/1.00 \| 0.88/0.92 | 1.00/1.00 \| 0.92/0.90 |
| 4 | 1.00/1.00 \| 0.99/0.94 | 1.00/1.00 \| 0.98/0.93 |
| 8 | 1.00/1.00 \| 1.00/0.94 | 1.00/1.00 \| 1.00/0.93 |
| 16 | 1.00/1.00 \| 1.00/0.96 | 1.00/1.00 \| 1.00/0.93 |

Weight-only quantisation of the transformer blocks, same evaluation (`--quant int8|int4`; raw numbers in `eval_nano_toy_int8.json`, `eval_nano_toy_int4.json`):

| metric | fp32 | int8 | int4 |
|---|---|---|---|
| text → video+audio: audio level / frequency | 0.99 / 0.94 | 1.00 / 0.93 | 0.96 / 0.95 |
| video → audio: level / frequency | 0.99 / 0.93 | 0.98 / 0.93 | 0.90 / 0.92 |
| audio → video: colour / direction | 0.90 / 1.00 | 0.90 / 1.00 | 0.91 / 1.00 |
| reference → video: colour | 0.98 | 0.98 | 0.99 |
| image → video: pinned frame exact | 1.00 | 1.00 | 1.00 |
| first+last frame middle-frame error, px | 8.2 | 8.2 | 8.8 |

int8 is indistinguishable from fp32; int4 (round-to-nearest, no calibration) costs a little on the audio side. This is a 5 M-parameter model; larger tiers were not measured.

What to take from it:

* Text conditioning, image / first-last-frame pinning, and cross-modal conditioning (video → audio, audio → video) all work, on this toy.
* The video is right even at 1 step; the audio needs about 4. Reflow buys a real improvement at 1-2 steps (audio level 0.48 → 0.69 at 1 step, 0.88 → 0.92 at 2) for a small loss at 4 (0.99 → 0.98) and in reference colour (0.98 → 0.93).
* **First + last frame interpolation is only partial:** the middle frame lands 8.2 px from the ideal against 14.4 px for a square that does not move, but the VAE alone would land within 2.3 px.
* **Refinement shows no measurable benefit here.** Decoded output is capped by the VAE (roundtrip PSNR of the true frames: 19.3 dB); the refined video scores 19.1 dB, a no-op through the same VAE scores 19.2 dB. It preserves colour and direction, but the toy is too simple, and the VAE too lossy, to show whether in-context regeneration improves real content. Plain bicubic upsampling scores 29.6 dB only because it never touches the VAE.
* The video VAE is soft: about 19.3 dB PSNR at f16.

## How they were made

CPU only (4 cores): about 1.5 hours of training for the final recipe (audio VAE ~4 min, video VAE ~9 min, latent cache ~9 min, stage 1 ~33 min, stage 2 ~18 min, reflow ~15 min), not counting the failed first attempts described below.

```bash
# stage 1: both VAEs, latent cache, DiT from scratch
python scripts/train_toy.py --workdir run --vae-audio-steps 500 --vae-video-steps 800 \
    --dit-steps 6000 --bs 32 --cache 4000 --out stage1.safetensors

# stage 2: fine-tune with the cross-modal pin patterns weighted up (copy vae_*.pt and cache.pt
# from run/ into run2/ first); this fixed video -> audio direction, 0.28 -> 0.93
python scripts/train_toy.py --workdir run2 --stage dit --init-from stage1.safetensors \
    --cross-modal-boost --dit-steps 3000 --lr 4e-4 --bs 32 --out h3turbo-nano-toy.safetensors

# turbo: reflow step distillation
python scripts/reflow.py --ckpt h3turbo-nano-toy.safetensors --cache run2/cache.pt \
    --out h3turbo-nano-toy-turbo.safetensors --teacher-steps 8 --pairs 1500 --iters 1500 --lr 3e-4
```

Reproducibility caveat: stage 1 was trained before the `cross_modal_text_drop` change to `flow_loss` (it also reordered the RNG draws), so re-running stage 1 today gives a slightly different model. Stage 2 and reflow used the final code.

## Things that went wrong on the way (and are fixed in the code)

Recorded because they are the kind of failure that a low loss hides:

* Both VAEs first collapsed to trivial solutions (video: paint the background; audio: emit the same signal for every input) while their loss looked fine. A reconstruction gate that reads colour/direction back out of decoded output caught it before any DiT time was spent.
* The video VAE only converged once it got a parameter-free residual shortcut (pixel-unshuffle into the latent, pixel-shuffle back out); a run without it had stopped improving by step ~300 and scored 0.27 colour accuracy at step 800 (I stopped it there), while 300 steps with it reached 1.00.
* Video → audio direction sat at chance because the prompt, which names the direction, was present 90% of the time, so the model never had to read the motion out of the pinned video. The linear-probe check showed the information was fully available (1.00). Dropping the prompt 60% of the time when the other modality is fully given fixed it.
* `refine` noised the video to sigma = `strength` but told the model the schedule started at a different sigma. Fixed and covered by `tests/test_sampler.py`.
