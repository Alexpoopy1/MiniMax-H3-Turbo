"""Writes colab/H3_Turbo_Colab.ipynb from the cells below (edit here, then rerun: python colab/make_notebook.py).

The form options (tasks, presets) come from h3_colab.py, so the notebook and the helper cannot drift apart."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_colab  # noqa: E402  (standard library only)

MD, CODE = "markdown", "code"


def _opts(values) -> str:
    return json.dumps(list(values), ensure_ascii=False)


TASK_OPTS = _opts(h3_colab.TASKS)
PRESET_OPTS = _opts(list(h3_colab.PRESETS) + ["Custom"])
DEFAULT_PRESET = next(iter(h3_colab.PRESETS))

CELLS = [
    (MD, """# MiniMax H3 on Google Colab, with the H3-Turbo engine

Text, a first frame (or first + last frame), or **omni references** (images, videos with their sound, audio) to **video with
sound**, using your merged MiniMax H3 8-step turbo model (official W4A8 4-bit format) through ComfyUI v0.37.4, headless: no web
UI, the video is shown right here.

**Runtime → Change runtime type → a GPU.** The notebook reads the GPU and picks a profile:

| GPU | profile | what it does |
|---|---|---|
| **G4** (RTX PRO 6000 Blackwell, 96 GB), H100, A100, L4 | `big` | one pass; every model stays loaded on the GPU between generations (`--highvram`); bf16; text encoder **Q4_K_M** on >= 40 GB VRAM (higher quality; Q2_K = same as the PC is selectable) |
| **T4** (free tier, 15 GB VRAM, ~12.7 GB RAM) | `t4` | two phases so the 32B text encoder and the 33B DiT never share the RAM: phase 1 encodes the prompt (and frames/references) and drops the encoder, phase 2 samples and decodes; the DiT computes in fp32 (a T4 has no bf16; the 4-bit weights are unchanged) |

Quality is the model's own on every profile: the same 4-bit DiT, the fp16 video VAE and fp32 audio VAE (not the int8 VAE), 8
steps, `res_multistep`, exact attention by default.

**Status, honestly:** this notebook and its helper (`colab/h3_colab.py`) are tested offline (graph building checked against
ComfyUI 0.37.4's node definitions, the websocket/timing code against a fake server, file staging with ffmpeg). It has **not been
run on a Colab GPU yet**, so there are no Colab timings here; every run prints its own measured breakdown.

Run the cells top to bottom. All files are public: no token needed (an optional `HF_TOKEN` Colab secret only raises Hugging
Face's rate limit)."""),
    (CODE, """#@title 1. Setup: check the GPU, install ComfyUI v0.37.4, ComfyUI-GGUF and H3-Turbo (3-8 min) { display-mode: "form" }
H3_REPO = "https://github.com/Alexpoopy1/MiniMax-H3-Turbo.git"  #@param {type:"string"}
H3_BRANCH = "claude/funny-ride-htsqza"  #@param {type:"string"}
#@markdown **Profile**: `auto` = `big` on G4 / H100 / A100 / L4 (bf16, >= 20 GB VRAM, >= 30 GB RAM), `t4` otherwise.
PROFILE = "auto"  #@param ["auto", "big", "t4"]
#@markdown **PyTorch**: `auto` adds the CUDA 13 build (comfy_kitchen's fast W4A8 kernels) for ComfyUI on sm80+ GPUs with a
#@markdown driver >= 580, in its own folder, and only after it ran on the GPU. Colab's own PyTorch is never replaced.
TORCH_CU130 = "auto"  #@param ["auto", "yes", "no"]
import importlib, os, subprocess, sys
COMFY = "/content/ComfyUI"
H3_DIR = f"{COMFY}/custom_nodes/MiniMax-H3-Turbo"

def _sh(cmd):
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.returncode:
        print((r.stdout + r.stderr)[-4000:])
        raise RuntimeError(f"failed: {cmd}")
    return r.stdout.strip()

if not os.path.isdir(f"{COMFY}/.git"):
    _sh(f"git clone --depth 1 --branch v0.37.4 https://github.com/Comfy-Org/ComfyUI.git {COMFY}")
if not os.path.isdir(f"{H3_DIR}/.git"):
    _sh(f"git clone --depth 1 --branch {H3_BRANCH} {H3_REPO} {H3_DIR}")
else:  # always take the branch's latest code, so re-running this cell picks up fixes
    _sh(f"git -C {H3_DIR} fetch --depth 1 {H3_REPO} {H3_BRANCH} && git -C {H3_DIR} reset --hard FETCH_HEAD")
print(_sh(f"git -C {H3_DIR} log -1 --format='H3-Turbo %h %s'"))
if f"{H3_DIR}/colab" not in sys.path:
    sys.path.insert(0, f"{H3_DIR}/colab")
import h3_colab
importlib.reload(h3_colab)
HW = h3_colab.detect_hardware()
print(HW.describe())
PROF = h3_colab.choose_profile(HW, PROFILE)
print(PROF.describe())
TORCH = h3_colab.install(COMFY, hw=HW, torch_mode=TORCH_CU130)"""),
    (CODE, """#@title 2. Download the models (skips files that are already there) { display-mode: "form" }
#@markdown **Text encoder** (Qwen3-VL-32B for H3, GGUF): `auto` = Q4_K_M on GPUs with >= 40 GB VRAM, Q2_K otherwise.
TEXT_ENCODER = "auto"  #@param ["auto", "Q4_K_M (higher quality, 14.6 GB)", "Q2_K (same as the PC, 8.5 GB)"]
#@markdown **Your model**: a `.h3t` is used as is; a ComfyUI `_w4a8_convrot.safetensors` is converted here (lossless, ~3 min).
MODEL_SOURCE = "huggingface"  #@param ["huggingface", "google_drive"]
MODEL_HF_REPO = "Alexpoopy21/h3-private"  #@param {type:"string"}
MODEL_FILE = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_w4a8_convrot.h3t"  #@param {type:"string"}
MODEL_DRIVE_PATH = "/content/drive/MyDrive/h3/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_w4a8_convrot.h3t"  #@param {type:"string"}
TE_FILE = h3_colab.pick_text_encoder(TEXT_ENCODER, big=PROF.name == "big", vram_gib=HW.vram_gib)
PROF.text_encoder = TE_FILE
if TE_FILE == h3_colab.TE_Q4 and HW.vram_gib < h3_colab.Q4_MIN_VRAM_GIB:
    print("WARNING: Q4_K_M (14.6 GB) barely fits this GPU on its own; Q2_K is the safe choice here")
H3T_NAME = h3_colab.download_models(COMFY, te_files=[TE_FILE], model_source=MODEL_SOURCE, model_repo=MODEL_HF_REPO,
                                    model_file=MODEL_FILE, drive_path=MODEL_DRIVE_PATH)
print("ready:", H3T_NAME, "+", TE_FILE)"""),
    (CODE, """#@title 3. Start ComfyUI (headless; keep it running between generations) { display-mode: "form" }
#@markdown **Warm-up** (`big` profile only): loads the text encoder, the DiT and the VAEs once with a tiny clip, so the first
#@markdown real generation does not wait for the disk. On the `t4` profile models are reloaded per prompt anyway.
WARMUP = True  #@param {type:"boolean"}
try:
    srv.stop()
except NameError:
    pass
srv = h3_colab.ComfyServer(COMFY, profile=PROF, env=TORCH.env, extra_args=TORCH.comfy_args).start()
if WARMUP and PROF.name == "big":
    _w = h3_colab.warmup(srv, h3t_name=H3T_NAME, text_encoder=TE_FILE)
    print(f"warm-up done in {_w.total:.0f} s")"""),
    (CODE, """#@title 4. Upload images / videos / audio (optional) { display-mode: "form" }
#@markdown Opens a file picker; files are saved in `/content/uploads`. In step 5 use their names (e.g. `cat.png`), full paths
#@markdown (also `/content/drive/MyDrive/...`), or type `upload` in a field to pick the file right there.
MOUNT_GOOGLE_DRIVE = False  #@param {type:"boolean"}
if MOUNT_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
h3_colab.upload_files()
h3_colab.list_uploads()"""),
    (MD, """### Tasks
* **text -> video + audio**: the prompt only. End it with what you want to hear, e.g. `Audio: waves, soft wind.`
* **image -> video (first frame)**: `FIRST_FRAME` is the opening frame. With `MATCH_IMAGE_ASPECT` the canvas takes the image's
  aspect at the preset's pixel count (the node stretches the frame to the canvas, so this avoids distortion).
* **first + last frame -> video**: `FIRST_FRAME` and `LAST_FRAME`.
* **omni reference**: up to 9 reference images (identity, objects, style), 3 reference videos (motion, scene; their soundtrack is
  used when `REF_VIDEO_SOUNDTRACK` is on and the file has one) and 3 reference audio clips (voice, music). Mention them in the
  prompt as `<Picture 1>`, `<Video 1>`, `<Audio 1>`; the cell prints the numbering. `FIRST_FRAME` / `LAST_FRAME` may be added as
  keyframes. `REF_IMAGE_SIZE = max` keeps references at up to 2048 px (best identity, several times slower).

Files are staged without loss: images are copied unchanged; reference videos are converted to 24 fps (the node assumes 24) at
exactly the size the node resizes them to, trimmed to the output length (the node uses no more), as lossless H.264 + ALAC;
reference audio is decoded to 32-bit WAV (first 15 s). Re-running with only a new seed reuses the stored encode."""),
    (CODE, f"""#@title 5. Generate {{ display-mode: "form" }}
TASK = "text -> video + audio"  #@param {TASK_OPTS}
PROMPT = "Cinematic close-up of ocean waves rolling onto a golden beach at sunset, warm backlight, soft foam, gentle camera drift. Audio: waves washing the shore, soft wind."  #@param {{type:"string"}}
#@markdown **Size**: a preset, or `Custom` for WIDTH / HEIGHT (multiples of 32) and LENGTH (frames at 24 fps, snapped to
#@markdown 17k+5: 22 = 0.9 s, 73 = 3 s, 124 = 5.2 s).
PRESET = "{DEFAULT_PRESET}"  #@param {PRESET_OPTS}
WIDTH = 640  #@param {{type:"integer"}}
HEIGHT = 384  #@param {{type:"integer"}}
LENGTH = 22  #@param {{type:"integer"}}
STEPS = 8  #@param {{type:"integer"}}
SEED = 0  #@param {{type:"integer"}}
RANDOM_SEED = False  #@param {{type:"boolean"}}
#@markdown `exact` = reference attention, identical to ComfyUI's own loader. `int8_fast` = comfy_kitchen INT8 attention:
#@markdown faster on long clips but a different (not bit-identical) sample; needs the CUDA 13 PyTorch (else exact is used).
ATTENTION = "exact"  #@param ["exact", "int8_fast"]
#@markdown **Frames** (image -> video, first + last; optional keyframes for omni): file name, path, or `upload`.
FIRST_FRAME = ""  #@param {{type:"string"}}
LAST_FRAME = ""  #@param {{type:"string"}}
MATCH_IMAGE_ASPECT = True  #@param {{type:"boolean"}}
#@markdown **Omni references**: comma-separated file names / paths, or `upload`.
REF_IMAGES = ""  #@param {{type:"string"}}
REF_VIDEOS = ""  #@param {{type:"string"}}
REF_VIDEO_SOUNDTRACK = True  #@param {{type:"boolean"}}
REF_AUDIOS = ""  #@param {{type:"string"}}
REF_IMAGE_SIZE = "match"  #@param ["match", "max"]
RESULT = h3_colab.run_task(srv, TASK, PROMPT, preset=PRESET, width=WIDTH, height=HEIGHT, length=LENGTH, steps=STEPS, seed=SEED,
                           random_seed=RANDOM_SEED, attention=ATTENTION, first_frame=FIRST_FRAME, last_frame=LAST_FRAME,
                           match_image_aspect=MATCH_IMAGE_ASPECT, ref_images=REF_IMAGES, ref_videos=REF_VIDEOS,
                           ref_video_soundtrack=REF_VIDEO_SOUNDTRACK, ref_audios=REF_AUDIOS, ref_image_size=REF_IMAGE_SIZE,
                           h3t_name=H3T_NAME, text_encoder=TE_FILE)"""),
    (CODE, """#@title 6. Download the last video
from google.colab import files
files.download(RESULT.path)"""),
    (MD, """### If something fails
* Errors end with the last lines of ComfyUI's log; the full log is `/content/ComfyUI/h3turbo_server.log`.
* **"ComfyUI died ... SIGKILL"** means the machine ran out of RAM. The server restarts by itself on the next run. On a T4 use
  the fast preview first, fewer or shorter references, or a High-RAM runtime. Each run prints its peak RAM and VRAM.
* **Missing nodes** at start: a custom node failed to import; search the log for `H3-Turbo` or `GGUF`.
* **No run-time LoRAs** with this loader: merge them into the model before converting.

### What the profiles change in ComfyUI
* `big`: `--highvram` (models stay on the GPU; this also turns off ComfyUI's dynamic VRAM). The text encoder is kept loaded on
  hosts with >= 40 GB RAM, so a new prompt only re-runs the encoder, and a new seed only re-runs the sampler and the decode.
* `t4` (< 30 GB RAM): `--disable-dynamic-vram --disable-pinned-memory`. The text encoder node runs with
  `release=after_encode`, then `POST /free {"unload_models": true}`; ComfyUI's node cache keeps the encode, so phase 2 does not
  load the encoder again. Set the environment variable `H3_COLAB_FREE_MEMORY=1` before step 3 to also reset ComfyUI's node
  cache between the phases. `H3_COLAB_UNET_DTYPE` (auto / fp32 / bf16 / fp16) and `H3_COLAB_COMFY_ARGS` override the rest."""),
]


def build_notebook() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}
        if kind == CODE:
            cell.update(execution_count=None, outputs=[])
            cell["metadata"] = {"cellView": "form"}
        cells.append(cell)
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "H3_Turbo_Colab.ipynb")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(build_notebook(), f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
