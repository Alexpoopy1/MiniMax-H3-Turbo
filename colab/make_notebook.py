"""Writes colab/H3_Turbo_Colab.ipynb from the cells below (edit here, then rerun: python colab/make_notebook.py)."""
import json
import os

MD, CODE = "markdown", "code"

CELLS = [
    (MD, """# MiniMax H3 on Google Colab, with the H3-Turbo engine

Text (or image) to video **with sound**, using the official MiniMax H3 4-bit (W4A8) model through ComfyUI, headless: no web UI is
exposed, the video is shown right here.

**Runtime → Change runtime type → GPU.** What to expect (not measured on Colab yet, only on an RTX 3050):
* **L4 / A100** (Colab Pro): the right choice. bf16, enough VRAM to keep the whole model on the GPU.
* **T4** (free tier): works, but slow. No bf16 on a T4, so the model computes in fp32. The free tier's ~12.7 GB of RAM is tight
  for the text encoder plus the VAE, so the runtime may be killed on long clips. Use **High-RAM** if you have it.

**Before the first run:**
1. Your merged model is **not public**. Upload your `.h3t` file (or the `_w4a8_convrot.safetensors` it came from) once, either
   to a **private Hugging Face repo** (fastest to download here) or to **Google Drive**. For Hugging Face, run
   `hf upload YOUR_NAME/h3-private model.h3t --private` on your PC (already done for `Alexpoopy21/h3-private`).
2. Add **Colab secrets** (key icon on the left): `HF_TOKEN` (read access to your private repo) and, while the H3-Turbo GitHub
   repo is private, `GITHUB_TOKEN` (a read-only token). Turn on notebook access for both. Tokens are never printed.

The text encoder (`realrebelai/MiniMax-H3_GGUFs`) and the VAEs (`Comfy-Org/MiniMax-H3`) download from public repos."""),
    (CODE, """#@title 1. Settings { display-mode: "form" }
H3_REPO = "https://github.com/Alexpoopy1/MiniMax-H3-Turbo.git"  #@param {type:"string"}
H3_BRANCH = "claude/funny-ride-htsqza"  #@param {type:"string"}
#@markdown **Your model**: a `.h3t` (used as is) or a ComfyUI `_w4a8_convrot.safetensors` (converted here, ~3 min).
MODEL_SOURCE = "huggingface"  #@param ["huggingface", "google_drive"]
MODEL_HF_REPO = "Alexpoopy21/h3-private"  #@param {type:"string"}
MODEL_FILE = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_w4a8_convrot.h3t"  #@param {type:"string"}
MODEL_DRIVE_PATH = "/content/drive/MyDrive/h3/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_w4a8_convrot.h3t"  #@param {type:"string"}
#@markdown **Text encoder** (the same Q2_K GGUF as the RTX 3050 workflow).
TE_HF_REPO = "realrebelai/MiniMax-H3_GGUFs"  #@param {type:"string"}
TE_FILE = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"  #@param {type:"string"}
#@markdown **PyTorch**: `auto` installs the CUDA 13 build (enables comfy_kitchen's fast W4A8 kernels) only when the GPU is sm80+
#@markdown and the driver supports CUDA 13; otherwise it keeps Colab's PyTorch and uses the portable int8 path.
TORCH_CU130 = "auto"  #@param ["auto", "yes", "no"]
COMFY = "/content/ComfyUI"
print("ok")"""),
    (CODE, """#@title 2. Check the GPU
import shutil, subprocess
assert shutil.which("nvidia-smi"), "No GPU: Runtime -> Change runtime type -> a GPU"
q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"],
                   capture_output=True, text=True).stdout.strip().split(", ")
GPU_NAME, GPU_MEM, GPU_CC, DRIVER = q[0], q[1], float(q[2]), q[3]
import os
RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
print(f"GPU {GPU_NAME}, {GPU_MEM}, sm{GPU_CC:.1f}, driver {DRIVER}; RAM {RAM_GB:.0f} GB; free disk {shutil.disk_usage('/content').free / 2**30:.0f} GB")
if GPU_CC < 8.0:
    print("-> pre-Ampere GPU (e.g. T4): no bf16, the model computes in fp32 and comfy_kitchen's W4A8 kernels are unavailable. It works, slowly.")
if RAM_GB < 20:
    print("-> under 20 GB of RAM: keep clips short (e.g. 640x384, 22 frames); the runtime can be killed if RAM runs out.")"""),
    (CODE, """#@title 3. Install ComfyUI v0.37.4, ComfyUI-GGUF and H3-Turbo (~3-5 min)
import os, subprocess, sys
def sh(cmd, secret=None):
    shown = cmd if secret is None else cmd.replace(secret, "***")
    print("$", shown)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    out = (r.stdout + r.stderr)
    if secret:
        out = out.replace(secret, "***")
    if r.returncode:
        print(out[-4000:])
        raise RuntimeError(f"failed: {shown}")
    return out

def secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)

if not os.path.isdir(COMFY):
    sh(f"git clone --depth 1 --branch v0.37.4 https://github.com/Comfy-Org/ComfyUI.git {COMFY}")
drv = int(DRIVER.split(".")[0])
want_cu130 = TORCH_CU130 == "yes" or (TORCH_CU130 == "auto" and GPU_CC >= 8.0 and drv >= 580)
if want_cu130:  # the exact build the RTX 3050 setup was verified with; the notebook kernel never imports torch, so no restart needed
    sh("pip install -q torch==2.12.1 torchvision==0.27.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130")
sh(f"pip install -q -r {COMFY}/requirements.txt")
nodes = f"{COMFY}/custom_nodes"
if not os.path.isdir(f"{nodes}/ComfyUI-GGUF"):
    sh(f"git clone --depth 1 https://github.com/city96/ComfyUI-GGUF {nodes}/ComfyUI-GGUF")
sh(f"pip install -q -r {nodes}/ComfyUI-GGUF/requirements.txt")
tok = secret("GITHUB_TOKEN")
url = H3_REPO.replace("https://", f"https://x-access-token:{tok}@") if tok else H3_REPO
h3 = f"{nodes}/MiniMax-H3-Turbo"
if not os.path.isdir(h3):
    sh(f"git clone --depth 1 --branch {H3_BRANCH} {url} {h3}", secret=tok)
else:  # always update to the latest code of the branch (re-running this cell picks up fixes)
    sh(f"git -C {h3} fetch --depth 1 {url} {H3_BRANCH} && git -C {h3} reset --hard FETCH_HEAD", secret=tok)
sh(f"git -C {h3} remote set-url origin {H3_REPO}")  # never leave the token in .git/config
print(sh(f"git -C {h3} log -1 --format='H3-Turbo at %h %s'"))
sh("pip install -q psutil av huggingface_hub safetensors")
print(sh(f"{sys.executable} -c \\"import torch;print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name())\\""))"""),
    (CODE, """#@title 4. Download the models (text encoder 8.5 GB, VAEs 5.8 GB, your model 12.5 GB)
import os, subprocess, sys, shutil
from huggingface_hub import hf_hub_download
M = f"{COMFY}/models"
for d in ("vae", "clip", "h3turbo"):
    os.makedirs(f"{M}/{d}", exist_ok=True)
for f in ("vae/minimax_h3_video_vae_fp16.safetensors", "vae/minimax_h3_audio_vae_fp32.safetensors"):
    hf_hub_download("Comfy-Org/MiniMax-H3", f, local_dir=M)
hf_hub_download(TE_HF_REPO, TE_FILE, local_dir=f"{M}/clip")

name = os.path.basename(MODEL_FILE if MODEL_SOURCE == "huggingface" else MODEL_DRIVE_PATH)
h3t = os.path.splitext(name)[0] + ".h3t"
dst = f"{M}/h3turbo/{h3t}"
if not os.path.exists(dst):
    if MODEL_SOURCE == "huggingface":
        src = hf_hub_download(MODEL_HF_REPO, MODEL_FILE, local_dir="/content/src_model", token=secret("HF_TOKEN"))
    else:
        from google.colab import drive
        drive.mount("/content/drive")
        src = "/content/src_model/" + name  # copy first: streaming weights over the Drive mount would be far too slow
        os.makedirs("/content/src_model", exist_ok=True)
        shutil.copyfile(MODEL_DRIVE_PATH, src)
    if src.endswith(".h3t"):
        shutil.move(src, dst)
    else:  # lossless re-layout, verified byte for byte
        sh(f"cd {COMFY}/custom_nodes/MiniMax-H3-Turbo && {sys.executable} -m h3turbo h3-convert '{src}' '{dst}'")
        os.remove(src)
H3T_NAME = h3t
print("model:", dst, round(os.path.getsize(dst) / 1e9, 2), "GB")"""),
    (CODE, """#@title 5. Start ComfyUI (headless; keep it running between generations so models stay loaded)
import os, sys
_helper = f"{COMFY}/custom_nodes/MiniMax-H3-Turbo/colab"
if not os.path.exists(f"{_helper}/h3_colab.py"):
    raise RuntimeError("The H3-Turbo code in custom_nodes is older than this notebook. Re-run step 3 (it updates the clone).")
sys.path.insert(0, _helper)
from h3_colab import ComfyServer, generate, show
try:
    srv.stop()
except NameError:
    pass
srv = ComfyServer(COMFY).start()"""),
    (CODE, """#@title 6. Generate
PROMPT = "Cinematic close-up of ocean waves rolling onto a golden beach at sunset, warm backlight, soft foam, gentle camera drift. Audio: waves washing the shore, soft wind."  #@param {type:"string"}
WIDTH = 640  #@param {type:"integer"}
HEIGHT = 384  #@param {type:"integer"}
#@markdown Frames at 24 fps, on the model's 17k+5 grid: 22 = 0.9 s, 73 = 3 s, 124 = 5.2 s (trained range ~124-362).
LENGTH = 22  #@param [5, 22, 39, 56, 73, 124, 175, 226] {type:"raw"}
STEPS = 8  #@param {type:"integer"}
SEED = 0  #@param {type:"integer"}
#@markdown `exact` = the reference attention (same output as ComfyUI's own loader). `int8_fast` = faster on long clips, a
#@markdown different sample, and it needs comfy_kitchen's CUDA extension (the cu130 PyTorch from step 3).
ATTENTION = "exact"  #@param ["exact", "int8_fast"]
#@markdown Optional first frame (image-to-video): path of an image uploaded to /content, or leave empty.
FIRST_FRAME = ""  #@param {type:"string"}
mp4 = generate(srv, PROMPT, width=WIDTH, height=HEIGHT, length=LENGTH, steps=STEPS, seed=SEED, attention=ATTENTION,
               h3t_name=H3T_NAME, text_encoder=TE_FILE, image=FIRST_FRAME or None)
show(mp4)"""),
    (CODE, """#@title 7. Download the last video
from google.colab import files
files.download(mp4)"""),
    (MD, """### If something fails
* The error message ends with the last lines of ComfyUI's log; the full log is `/content/ComfyUI/h3turbo_server.log`.
* **Runtime killed / out of memory**: shorten the clip or lower the resolution, or use a High-RAM runtime.
* **`H3TurboFastUNetLoader` missing**: the H3-Turbo clone failed (private repo without `GITHUB_TOKEN`?) or its import failed:
  search the log for `H3-Turbo`.
* **No run-time LoRAs** with this loader: merge them into the model before converting.
* Re-running step 6 with only a new seed reuses the cached text encoding and the loaded models (much faster than the first run)."""),
]


def main():
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": [ln + "\n" for ln in lines[:-1]] + [lines[-1]]}
        if kind == CODE:
            cell.update(execution_count=None, outputs=[])
            cell["metadata"] = {"cellView": "form"}
        cells.append(cell)
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "H3_Turbo_Colab.ipynb")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
