"""MiniMax H3 on Google Colab (or any Linux machine with an NVIDIA GPU) through a headless ComfyUI with the H3-Turbo nodes.

Everything the notebook does lives here, standard library only at import time (so the graph tests run anywhere):

* hardware detection and the two GPU profiles with their ComfyUI launch flags (`choose_profile`):
    big  G4 (RTX PRO 6000, 96 GB), H100, A100, L4: one pass, every model stays loaded (--highvram), bf16 compute.
    t4   T4 and any GPU with < 20 GB VRAM, no bf16, or < 30 GB RAM: on hosts with < 30 GB RAM the text encoder runs in its
         own prompt first ("phase 1"), its weights are dropped, and the DiT then samples in a second prompt that reuses the
         stored conditioning ("phase 2"), so the 32B encoder and the 33B DiT never share the ~12.7 GB of RAM.
* installation (`install`): ComfyUI's requirements, ComfyUI-GGUF pinned to the code the RTX 3050 PC runs, and optionally an
  isolated CUDA 13 PyTorch (comfy_kitchen's CUDA kernels) that is only used after it passed a GPU test; Colab's own PyTorch is
  never replaced, so a failed install cannot break the runtime.
* model downloads (`download_models`), idempotent, with sizes.
* API graphs (`build_graph`): text -> video+audio, image -> video (first frame, first + last frame), omni reference
  (reference images, reference videos with or without their soundtrack, reference audio), and the encode-only graph.
* `ComfyServer`: starts/attaches ComfyUI, websocket progress, timing breakdown, RAM/VRAM peaks, restart after a crash.
* `generate` / `run_task`: stage uploads (lossless), run one or two phases, return the mp4.

    hw = detect_hardware(); prof = choose_profile(hw)
    ts = install("/content/ComfyUI", hw=hw)
    h3t = download_models("/content/ComfyUI", te_files=[prof.text_encoder])
    srv = ComfyServer("/content/ComfyUI", profile=prof, env=ts.env).start()
    res = generate(srv, "a red fox running through snow", h3t_name=h3t, width=640, height=384, length=22, seed=1)
    show(res.path)

Environment overrides (read when the profile is chosen):
    H3_COLAB_UNET_DTYPE   auto (default) | fp32 | bf16 | fp16: ComfyUI's --fp32-unet / --bf16-unet / --fp16-unet. The H3-Turbo
                          loader computes in bf16 on Ampere+ and fp32 on GPUs without bf16 (T4); it currently maps fp16 to fp32.
    H3_COLAB_COMFY_ARGS   extra ComfyUI flags, appended (e.g. "--reserve-vram 1").
    H3_COLAB_FREE_MEMORY  1 = two-phase also resets ComfyUI's node cache between the phases (POST /free free_memory=true).
"""
from __future__ import annotations

import ast
import base64
import glob
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ----------------------------------------------------------------------------------------------- pinned versions and files
COMFY_URL = "https://github.com/Comfy-Org/ComfyUI.git"
COMFY_TAG = "v0.37.4"
COMFY_COMMIT = "8ff6dc384ba5c410266b40e137799e049459d4f2"
GGUF_URL = "https://github.com/city96/ComfyUI-GGUF"
GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"  # loader/nodes/ops/dequant identical to ComfyUI-GGUF 1.1.10 on the RTX 3050 PC
TORCH_CU130 = ("torch==2.12.1", "torchvision==0.27.1", "torchaudio==2.11.0")  # the build the RTX 3050 setup was verified with
TORCH_CU130_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_CU130_DIR = "/content/torch_cu130"

VAE_REPO = "Comfy-Org/MiniMax-H3"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"  # not the int8 VAE: that one loses quality
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TE_REPO = "realrebelai/MiniMax-H3_GGUFs"
TE_Q2 = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"  # 8.49 GB, the RTX 3050 workflow's encoder
TE_Q4 = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"  # 14.58 GB, higher quality
MODEL_REPO = "Alexpoopy21/h3-private"
MODEL_FILE = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_w4a8_convrot.h3t"

FPS = 24
MAX_REF_IMAGES, MAX_REF_VIDEOS, MAX_REF_AUDIOS = 9, 3, 3  # MiniMaxH3ReferenceToVideo's Autogrow limits
PRESETS = {
    "Fast preview 640x384, 22 frames (0.9 s)": (640, 384, 22),
    "Fast preview portrait 384x640, 22 frames (0.9 s)": (384, 640, 22),
    "Full 832x480, 124 frames (5 s)": (832, 480, 124),
    "Full portrait 480x832, 124 frames (5 s)": (480, 832, 124),
}
TASKS = {
    "text -> video + audio": "t2v",
    "image -> video (first frame)": "i2v",
    "first + last frame -> video": "flf2v",
    "omni reference (images / videos / audio)": "omni",
}
UPLOAD_DIR = "/content/uploads"
REQUIRED_NODES = ("H3TurboFastUNetLoader", "H3TurboCachedTextEncoder", "H3TurboConditioningWarmup", "CLIPLoaderGGUF",
                  "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "MiniMaxH3AddGuide", "LoadImage", "LoadVideo",
                  "GetVideoComponents", "LoadAudio", "CreateVideo", "SaveVideo")

BIG_MIN_VRAM_GIB = 20.0  # L4 reports ~22.5 GiB
BIG_MIN_RAM_GIB = 30.0
Q4_MIN_VRAM_GIB = 39.0  # A100 40 GB reports 40 GiB
LOW_RAM_GIB = 30.0  # below this the t4 profile encodes and samples in separate prompts


def _gib(nbytes: float) -> float:
    return nbytes / 2**30


# ----------------------------------------------------------------------------------------------- geometry (mirrors ComfyUI)
def snap_length(length: int) -> int:
    """Frames on H3's 17k+5 grid, rounded up exactly like the H3 nodes do (22 = 0.9 s, 124 = 5.2 s at 24 fps)."""
    n = max(5, int(length))
    while n % 17 != 5:
        n += 1
    return n


def fit32(v: float) -> int:
    return max(32, int(round(v / 32)) * 32)


def adapt_canvas(width: int, height: int) -> Tuple[int, int]:
    """comfy_extras/nodes_minimax_h3.py adapt_canvas: 768-short-edge canvas, 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    nom_w, nom_h = (768 * ratio, 768) if ratio >= 1.0 else (768, 768 / ratio)
    if nom_w * nom_h > 768 * 1344:
        s = math.sqrt(768 * 1344 / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return max(32, round(nom_w / 32) * 32), max(32, round(nom_h / 32) * 32)


def ref_video_canvas(vw: int, vh: int) -> Tuple[int, int]:
    """The size MiniMaxH3ReferenceToVideo resizes a reference video to (never upscaled past the source)."""
    cw, ch = adapt_canvas(vw, vh)
    if vw * vh < cw * ch:
        cw, ch = max(32, round(vw / 32) * 32), max(32, round(vh / 32) * 32)
    return cw, ch


def match_aspect(img_w: int, img_h: int, width: int, height: int) -> Tuple[int, int]:
    """A canvas with the pixel budget of width x height and the aspect of the image (the first frame is stretched to the
    canvas by MiniMaxH3ImageToVideo, so a matching aspect avoids distorting it)."""
    area, r = width * height, img_w / img_h
    return fit32(math.sqrt(area * r)), fit32(math.sqrt(area / r))


# ----------------------------------------------------------------------------------------------- hardware and profiles
@dataclass
class Hardware:
    gpu: str
    vram_gib: float
    cc: Tuple[int, int]
    driver: str
    ram_gib: float
    disk_free_gib: float = 0.0

    @property
    def bf16(self) -> bool:
        return self.cc >= (8, 0)

    @property
    def driver_major(self) -> int:
        try:
            return int(self.driver.split(".")[0])
        except ValueError:
            return 0

    def describe(self) -> str:
        return (f"GPU {self.gpu}, {self.vram_gib:.1f} GiB VRAM, sm{self.cc[0]}{self.cc[1]}, driver {self.driver}; "
                f"RAM {self.ram_gib:.1f} GiB; free disk {self.disk_free_gib:.0f} GiB")


def parse_gpu_query(line: str) -> Tuple[str, float, Tuple[int, int], str]:
    """One line of `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits`."""
    name, mem, cc, drv = (p.strip() for p in line.strip().rsplit(",", 3))
    major, minor = (int(x) for x in cc.split("."))
    return name, float(mem) / 1024.0, (major, minor), drv


def _ram_bytes() -> Tuple[int, int]:
    """(total, available) bytes; (0, 0) if unknown."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info["MemTotal"], info.get("MemAvailable", 0)
    except Exception:
        return 0, 0


def detect_hardware(disk_path: str = "/content") -> Hardware:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("No NVIDIA GPU: Runtime -> Change runtime type -> a GPU (T4, L4, A100, H100 or G4)")
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    name, vram, cc, drv = parse_gpu_query(out)
    total, _ = _ram_bytes()
    if not total:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    disk = shutil.disk_usage(disk_path if os.path.isdir(disk_path) else "/").free
    return Hardware(name, vram, cc, drv, _gib(total), _gib(disk))


@dataclass
class Profile:
    name: str  # "big" | "t4"
    two_phase: bool
    comfy_args: List[str]
    text_encoder: str
    te_release: str
    free_payload: Dict[str, bool]
    env: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"profile {self.name}: {'two-phase (encode, free, then sample)' if self.two_phase else 'single pass'}; "
                 f"ComfyUI flags: {' '.join(self.comfy_args) or '(defaults)'}; text encoder {self.text_encoder}; "
                 f"release={self.te_release}"]
        lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def pick_text_encoder(choice: str, *, big: bool, vram_gib: float) -> str:
    c = (choice or "auto").strip()
    if c.lower() in ("auto", ""):
        return TE_Q4 if big and vram_gib >= Q4_MIN_VRAM_GIB else TE_Q2
    if c.upper().startswith("Q4"):
        return TE_Q4
    if c.upper().startswith("Q2"):
        return TE_Q2
    return c  # an explicit file name


def choose_profile(hw: Hardware, profile: str = "auto", text_encoder: str = "auto", unet_dtype: Optional[str] = None,
                   extra_args: Sequence[str] = ()) -> Profile:
    """BIG when the GPU has bf16, >= 20 GiB VRAM and the host >= 30 GiB RAM; T4 profile otherwise."""
    name = (profile or "auto").lower()
    if name == "auto":
        name = "big" if (hw.bf16 and hw.vram_gib >= BIG_MIN_VRAM_GIB and hw.ram_gib >= BIG_MIN_RAM_GIB) else "t4"
    if name not in ("big", "t4"):
        raise ValueError(f"profile must be auto, big or t4, got {profile!r}")
    te = pick_text_encoder(text_encoder, big=name == "big", vram_gib=hw.vram_gib)
    dtype = (unet_dtype or os.environ.get("H3_COLAB_UNET_DTYPE") or "auto").strip().lower()
    dtype_flags = {"auto": [], "fp32": ["--fp32-unet"], "bf16": ["--bf16-unet"], "fp16": ["--fp16-unet"]}
    if dtype not in dtype_flags:
        raise ValueError(f"H3_COLAB_UNET_DTYPE must be auto, fp32, bf16 or fp16, got {dtype!r}")
    notes: List[str] = []
    env: Dict[str, str] = {}
    if name == "big":
        # --highvram: models stay on the GPU between generations (unet offload device = GPU, nothing pushed back to RAM unless
        # VRAM runs out). It also turns ComfyUI's dynamic VRAM off, so all models use the classic estimate-based loader.
        args, two_phase, release = ["--highvram"], False, "auto"  # auto keeps the text encoder loaded on >= 40 GB RAM hosts
        notes.append("every model stays loaded between generations; a new prompt only re-runs the text encoder")
    else:
        two_phase = hw.ram_gib < LOW_RAM_GIB
        # classic loader instead of comfy-aimdo's dynamic VRAM (only validated on the Windows PC; with it, weights may stay
        # mirrored in RAM): a model then lives either on the GPU or in RAM, and the T4 has more VRAM (15 GB) than RAM (12.7 GB)
        args = ["--disable-dynamic-vram"]
        if two_phase:
            # page-locked RAM cannot be reclaimed; the H3 engine pins its own streaming buffers within its RAM budget
            args.append("--disable-pinned-memory")
            env["MALLOC_ARENA_MAX"] = "2"  # fewer glibc arenas: less RSS bloat from ComfyUI's worker threads
            notes.append("phase 1 encodes the prompt and drops the text encoder; phase 2 samples (the encoder is not loaded again)")
        release = "after_encode" if two_phase else "auto"
        if not hw.bf16:
            notes.append("no bf16 on this GPU: the DiT computes in fp32 (4-bit weights unchanged); "
                         "comfy_kitchen's W4A8 CUDA kernels need sm80+, so the portable int8 path is used")
        if te == TE_Q4 and hw.vram_gib < Q4_MIN_VRAM_GIB:
            notes.append("Q4_K_M (14.6 GB) barely fits this GPU's VRAM on its own; expect a slow or failing encode, Q2_K is the safe choice")
    args = args + dtype_flags[dtype] + shlex.split(os.environ.get("H3_COLAB_COMFY_ARGS", "")) + list(extra_args)
    if dtype == "fp16":
        notes.append("fp16 requested: the H3-Turbo loader currently only computes in bf16 or fp32 and maps fp16 to fp32")
    free_payload = {"unload_models": True, "free_memory": os.environ.get("H3_COLAB_FREE_MEMORY", "0") == "1"}
    return Profile(name, two_phase, args, te, release, free_payload, env, notes)


# ----------------------------------------------------------------------------------------------- install
def _run(cmd: Union[str, List[str]], *, env=None, cwd=None, check=True, quiet=True) -> str:
    shown = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
    print("$", shown[:300], flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), text=True, capture_output=True, env=env, cwd=cwd)
    out = r.stdout + r.stderr
    if r.returncode and check:
        print(out[-4000:])
        raise RuntimeError(f"command failed ({r.returncode}): {shown[:300]}")
    if not quiet:
        print(out[-2000:])
    return out


def _pip(args: List[str], *, check=True) -> str:
    return _run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args], check=check)


def git_checkout(url: str, path: str, commit: str) -> None:
    """Shallow checkout of one exact commit (idempotent)."""
    if os.path.isdir(os.path.join(path, ".git")):
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == commit:
            return
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    _run(["git", "-C", path, "init", "-q"])
    _run(["git", "-C", path, "fetch", "-q", "--depth", "1", url, commit])
    _run(["git", "-C", path, "checkout", "-q", "FETCH_HEAD"])


_CHECK_TORCH = r"""
import json
r = {"ok": False}
try:
    import torch
    r["torch"], r["cuda"] = torch.__version__, torch.version.cuda
    r["cuda_available"] = torch.cuda.is_available()
    if r["cuda_available"]:
        cc = torch.cuda.get_device_capability()
        r["cc"], r["arch_list"] = list(cc), torch.cuda.get_arch_list()
        x = torch.randn(512, 512, device="cuda")
        (x @ x).sum().item()
        h = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16 if cc >= (8, 0) else torch.float16)
        (h @ h).float().sum().item()
        torch.cuda.synchronize()
        r["kernel_ok"] = True
    import torchvision, torchaudio
    r["torchvision"], r["torchaudio"] = torchvision.__version__, torchaudio.__version__
    try:
        import comfy_kitchen as ck
        b = ck.list_backends().get("cuda", {})
        r["kitchen_cuda_ext"] = bool(b.get("available")) and "w4a8_int8_linear" in (b.get("capabilities") or [])
    except Exception as e:
        r["kitchen_cuda_ext"], r["kitchen_error"] = False, repr(e)[:300]
    major = int(str(r["cuda"] or "0").split(".")[0])
    r["kitchen_cuda"] = bool(r.get("kitchen_cuda_ext")) and major >= 13  # ComfyUI disables kitchen's CUDA backend below cu130
    r["ok"] = bool(r.get("kernel_ok"))
except Exception as e:
    r["error"] = repr(e)[:800]
print("H3JSON" + json.dumps(r))
"""


def check_torch(env: Optional[Dict[str, str]] = None) -> dict:
    """Import torch in a fresh interpreter (with `env`, e.g. the isolated cu130 path) and run real kernels on the GPU."""
    full = dict(os.environ, **(env or {}))
    r = subprocess.run([sys.executable, "-c", _CHECK_TORCH], capture_output=True, text=True, env=full, timeout=600)
    for line in r.stdout.splitlines():
        if line.startswith("H3JSON"):
            return json.loads(line[6:])
    return {"ok": False, "error": (r.stdout + r.stderr)[-800:]}


@dataclass
class TorchSetup:
    env: Dict[str, str]  # for the ComfyUI process (PYTHONPATH of the isolated cu130 build), empty = Colab's PyTorch
    info: dict
    summary: str
    comfy_args: List[str] = field(default_factory=list)


_SHADOW_REMOVE = ("numpy", "PIL", "pillow")  # compiled packages Colab's other compiled packages are built against


def _prune_shadowing(target: str) -> None:
    """Remove numpy/Pillow that pip put next to the cu130 torch when Colab already has them, so ComfyUI keeps Colab's builds
    (scipy, av, opencv there are compiled against them). Pure-Python helpers (sympy, ...) stay: shadowing them is harmless."""
    import importlib.util

    for entry in os.listdir(target):
        base = re.split(r"[-.]", entry, maxsplit=1)[0]
        for mod in _SHADOW_REMOVE:
            if base.lower() == mod.lower() and importlib.util.find_spec(mod if mod != "pillow" else "PIL") is not None:
                p = os.path.join(target, entry)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
                break


def setup_torch(hw: Hardware, mode: str = "auto", target: str = TORCH_CU130_DIR) -> TorchSetup:
    """Keep Colab's PyTorch unless (a) it cannot run kernels on this GPU, or (b) mode allows the CUDA 13 build that enables
    comfy_kitchen's W4A8 CUDA kernels (auto: sm80+ and driver >= 580). The cu130 build goes into its own directory and is put
    on ComfyUI's PYTHONPATH only after it ran real kernels on the GPU; otherwise it is deleted and Colab's build is used."""
    mode = (mode or "auto").lower()
    sys_info = check_torch()
    print(f"Colab PyTorch: {sys_info.get('torch')} (CUDA {sys_info.get('cuda')}), GPU kernels "
          f"{'ok' if sys_info.get('ok') else 'FAILED: ' + str(sys_info.get('error', ''))[:300]}")
    need = not sys_info.get("ok")
    want = mode == "yes" or (mode == "auto" and hw.cc >= (8, 0) and hw.driver_major >= 580 and not sys_info.get("kitchen_cuda"))
    if not (need or want):
        why = ("driver older than 580 (CUDA 13 needs >= 580)" if hw.driver_major < 580 else
               "pre-Ampere GPU" if hw.cc < (8, 0) else "kept by choice" if mode == "no" else "already has kitchen CUDA")
        return TorchSetup({}, sys_info, f"using Colab's PyTorch {sys_info.get('torch')} ({why}); "
                                        f"comfy_kitchen CUDA kernels {'on' if sys_info.get('kitchen_cuda') else 'off (portable int8 path)'}")
    pp = target + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    env = {"PYTHONPATH": pp}
    marker = os.path.join(target, ".h3turbo_ok")
    info = None
    if os.path.exists(marker):
        info = check_torch(env)
    if not (info and info.get("ok")):
        shutil.rmtree(target, ignore_errors=True)
        print(f"installing PyTorch {TORCH_CU130[0]} (CUDA 13) into {target} (isolated; Colab's PyTorch stays as it is) ...")
        try:
            _pip(["--target", target, "--index-url", TORCH_CU130_INDEX, "--extra-index-url", "https://pypi.org/simple", *TORCH_CU130])
            _prune_shadowing(target)
            info = check_torch(env)
        except Exception as e:
            info = {"ok": False, "error": repr(e)[:800]}
    if info.get("ok") and (info.get("kitchen_cuda") or need):
        with open(marker, "w") as f:
            json.dump(info, f)
        # an xformers build that came with Colab would not match this torch; ComfyUI then uses PyTorch attention
        return TorchSetup(env, info, f"using PyTorch {info.get('torch')} (CUDA {info.get('cuda')}) from {target} for ComfyUI; "
                                     f"comfy_kitchen CUDA kernels {'on' if info.get('kitchen_cuda') else 'off'}", ["--disable-xformers"])
    shutil.rmtree(target, ignore_errors=True)
    if need:
        raise RuntimeError("Colab's PyTorch cannot run on this GPU and the CUDA 13 build did not pass its GPU test either:\n"
                           f"Colab: {sys_info}\ncu130: {info}")
    return TorchSetup({}, sys_info, f"CUDA 13 PyTorch rejected ({info.get('error') or 'no comfy_kitchen CUDA kernels with it'}); "
                                    f"using Colab's PyTorch {sys_info.get('torch')}")


def install(comfy_dir: str, *, hw: Optional[Hardware] = None, torch_mode: str = "auto") -> TorchSetup:
    """Everything after the ComfyUI and H3-Turbo clones: Python requirements, ComfyUI-GGUF, PyTorch choice. Idempotent."""
    hw = hw or detect_hardware()
    head = subprocess.run(["git", "-C", comfy_dir, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMFY_COMMIT:
        print(f"WARNING: ComfyUI is at {head or '?'}, not {COMFY_TAG} ({COMFY_COMMIT[:10]}); the H3-Turbo nodes were tested on {COMFY_TAG}")
    _pip(["-r", os.path.join(comfy_dir, "requirements.txt")])
    gguf = os.path.join(comfy_dir, "custom_nodes", "ComfyUI-GGUF")
    git_checkout(GGUF_URL, gguf, GGUF_COMMIT)
    _pip(["-r", os.path.join(gguf, "requirements.txt")])
    _pip(["psutil", "huggingface_hub", "safetensors"])
    _pip(["hf_xet"], check=False)  # faster Hugging Face downloads where available; optional
    _pip(["hf_transfer"], check=False)
    ts = setup_torch(hw, torch_mode)
    print(ts.summary)
    return ts


# ----------------------------------------------------------------------------------------------- downloads
def colab_secret(name: str) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore

        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name) or None


def _fast_hf() -> None:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        import hf_transfer  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        if "huggingface_hub" in sys.modules:
            import huggingface_hub.constants as c

            if hasattr(c, "HF_HUB_ENABLE_HF_TRANSFER"):
                c.HF_HUB_ENABLE_HF_TRANSFER = True
    except ImportError:
        pass


def remote_size(repo: str, filename: str, token: Optional[str] = None) -> Optional[int]:
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        return int(get_hf_file_metadata(hf_hub_url(repo, filename), token=token).size)
    except Exception:
        return None


def hf_fetch(repo: str, filename: str, local_dir: str, token: Optional[str] = None) -> str:
    """hf_hub_download into local_dir (no second copy in ~/.cache); skips a file that is already complete."""
    final = os.path.join(local_dir, filename)
    size = remote_size(repo, filename, token)
    if os.path.exists(final) and (size is None or os.path.getsize(final) == size):
        print(f"  {filename}: present ({os.path.getsize(final) / 1e9:.2f} GB)")
        return final
    _fast_hf()
    from huggingface_hub import hf_hub_download

    print(f"  {filename}: downloading {'?' if size is None else f'{size / 1e9:.2f}'} GB from {repo} ...", flush=True)
    t0 = time.time()
    path = hf_hub_download(repo, filename, local_dir=local_dir, token=token)
    dt = max(time.time() - t0, 1e-3)
    print(f"  {filename}: {os.path.getsize(path) / 1e9:.2f} GB in {dt:.0f} s ({os.path.getsize(path) / 1e6 / dt:.0f} MB/s)")
    return path


def download_models(comfy_dir: str, *, te_files: Iterable[str] = (TE_Q2,), model_source: str = "huggingface",
                    model_repo: str = MODEL_REPO, model_file: str = MODEL_FILE, drive_path: str = "",
                    token: Optional[str] = None) -> str:
    """VAEs, text encoder(s) and the DiT into ComfyUI/models; returns the .h3t name for the loader node."""
    token = token or colab_secret("HF_TOKEN")  # optional: only raises the anonymous rate limit
    m = os.path.join(comfy_dir, "models")
    for d in ("vae", "clip", "h3turbo"):
        os.makedirs(os.path.join(m, d), exist_ok=True)
    print("VAEs:")
    for f in (VIDEO_VAE, AUDIO_VAE):
        hf_fetch(VAE_REPO, f"vae/{f}", m, token)
    print("text encoder:")
    for f in te_files:
        hf_fetch(TE_REPO, f, os.path.join(m, "clip"), token)
    print("DiT:")
    h3t_dir = os.path.join(m, "h3turbo")
    if model_source == "huggingface" and model_file.endswith(".h3t"):  # used as is (sub-folders stay part of the name)
        hf_fetch(model_repo, model_file, h3t_dir, token)
        return model_file.replace("\\", "/")
    name = os.path.basename(model_file if model_source == "huggingface" else drive_path)
    h3t = os.path.splitext(name)[0] + ".h3t"
    dst = os.path.join(h3t_dir, h3t)
    if os.path.exists(dst):
        print(f"  {h3t}: present ({os.path.getsize(dst) / 1e9:.2f} GB)")
        return h3t
    if model_source == "huggingface":
        src = hf_fetch(model_repo, model_file, "/content/src_model", token)
    else:
        if not os.path.exists(drive_path) and drive_path.startswith("/content/drive"):
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        os.makedirs("/content/src_model", exist_ok=True)
        src = os.path.join("/content/src_model", name)  # copy first: streaming weights over the Drive mount would be far too slow
        print(f"  copying {drive_path} from Google Drive ...")
        shutil.copyfile(drive_path, src)
    if src.endswith(".h3t"):
        shutil.move(src, dst)
    else:  # ComfyUI _w4a8_convrot.safetensors -> .h3t, a lossless re-layout verified byte for byte
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _run([sys.executable, "-m", "h3turbo", "h3-convert", src, dst], cwd=repo_dir)
        os.remove(src)
    print(f"  {h3t}: {os.path.getsize(dst) / 1e9:.2f} GB")
    return h3t


# ----------------------------------------------------------------------------------------------- API graphs
def _node(cls: str, **inputs) -> dict:
    return {"class_type": cls, "inputs": inputs}


def _ref_videos(ref_videos) -> List[Tuple[str, bool]]:
    out = []
    for v in ref_videos or ():
        out.append((v, False) if isinstance(v, str) else (str(v[0]), bool(v[1])))
    return out


def resolve_task(first_frame=None, last_frame=None, ref_images=(), ref_videos=(), ref_audios=()) -> str:
    if ref_images or ref_videos or ref_audios:
        return "omni"
    if last_frame:
        return "flf2v"
    return "i2v" if first_frame else "t2v"


def build_graph(prompt: str, *, width: int, height: int, length: int, h3t_name: str, text_encoder: str, steps: int = 8,
                seed: int = 0, attention: str = "exact", precision: str = "a8", first_frame: Optional[str] = None,
                last_frame: Optional[str] = None, ref_images: Sequence[str] = (), ref_videos: Sequence = (),
                ref_audios: Sequence[str] = (), ref_image_size: str = "match", te_loader: str = "cached",
                te_release: str = "auto", te_cache: bool = True, video_vae: str = VIDEO_VAE, audio_vae: str = AUDIO_VAE,
                fps: int = FPS, prefix: str = "video/H3Turbo", sampler: str = "res_multistep", scheduler: str = "simple",
                phase: str = "full") -> dict:
    """The user's H3 workflow as a ComfyUI API graph, with the H3-Turbo loader in place of UNETLoader.

    Files (first_frame, last_frame, ref_*) are names inside ComfyUI's input folder. ref_videos items are a name, or
    (name, use_its_soundtrack). Without references the conditioning node is MiniMaxH3ImageToVideo (text, first frame,
    first + last frame); with any reference it is MiniMaxH3ReferenceToVideo, and first/last frames become MiniMaxH3AddGuide
    keyframes. phase="encode" returns only the conditioning part plus H3TurboConditioningWarmup (phase 1 of the two-phase
    run); its nodes are identical to the full graph's, so phase 2 reuses ComfyUI's cached outputs instead of re-running them.
    """
    if width % 32 or height % 32 or width < 32 or height < 32:
        raise ValueError(f"width and height must be multiples of 32 (got {width}x{height})")
    if length < 1 or steps < 1:
        raise ValueError("length and steps must be >= 1")
    if attention not in ("exact", "int8_fast") or precision not in ("a8", "a16") or ref_image_size not in ("match", "max"):
        raise ValueError("attention: exact|int8_fast, precision: a8|a16, ref_image_size: match|max")
    if te_release not in ("auto", "after_encode", "keep") or te_loader not in ("cached", "stock") or phase not in ("full", "encode"):
        raise ValueError("te_release: auto|after_encode|keep, te_loader: cached|stock, phase: full|encode")
    vids = _ref_videos(ref_videos)
    if len(ref_images) > MAX_REF_IMAGES or len(vids) > MAX_REF_VIDEOS or len(ref_audios) > MAX_REF_AUDIOS:
        raise ValueError(f"MiniMax H3 takes at most {MAX_REF_IMAGES} reference images, {MAX_REF_VIDEOS} videos and {MAX_REF_AUDIOS} audio clips")
    task = resolve_task(first_frame, last_frame, ref_images, vids, ref_audios)

    g: Dict[str, dict] = {}
    if te_loader == "cached":
        g["te"] = _node("H3TurboCachedTextEncoder", clip_name=text_encoder, release=te_release, cache=bool(te_cache))
    elif text_encoder.endswith(".gguf"):
        g["te"] = _node("CLIPLoaderGGUF", clip_name=text_encoder, type="minimax")
    else:
        g["te"] = _node("CLIPLoader", clip_name=text_encoder, type="minimax")
    g["vvae"] = _node("VAELoader", vae_name=video_vae)
    g["avae"] = _node("VAELoader", vae_name=audio_vae)
    encode_nodes = ["te", "vvae"]
    if first_frame:
        g["img_first"] = _node("LoadImage", image=first_frame)
        encode_nodes.append("img_first")
    if last_frame:
        g["img_last"] = _node("LoadImage", image=last_frame)
        encode_nodes.append("img_last")

    if task == "omni":
        cond = _node("MiniMaxH3ReferenceToVideo", clip=["te", 0], vae=["vvae", 0], prompt=prompt, width=width, height=height,
                     length=length, ref_image_size=ref_image_size)
        uses_audio = bool(ref_audios) or any(a for _, a in vids)
        if uses_audio:
            cond["inputs"]["audio_vae"] = ["avae", 0]
            encode_nodes.append("avae")
        for i, name in enumerate(ref_images):
            g[f"ref_img{i}"] = _node("LoadImage", image=name)
            cond["inputs"][f"ref_images.ref_image_{i}"] = [f"ref_img{i}", 0]
            encode_nodes.append(f"ref_img{i}")
        for k, (name, with_audio) in enumerate(vids):
            g[f"ref_vid{k}"] = _node("LoadVideo", file=name)
            g[f"ref_vid{k}_parts"] = _node("GetVideoComponents", video=[f"ref_vid{k}", 0])
            cond["inputs"][f"ref_videos.ref_video_{k}"] = [f"ref_vid{k}_parts", 0]
            if with_audio:  # ref_video_audio_N is the soundtrack of ref_video_N
                cond["inputs"][f"ref_video_audios.ref_video_audio_{k}"] = [f"ref_vid{k}_parts", 1]
            encode_nodes += [f"ref_vid{k}", f"ref_vid{k}_parts"]
        for j, name in enumerate(ref_audios):
            g[f"ref_aud{j}"] = _node("LoadAudio", audio=name)
            cond["inputs"][f"ref_audios.ref_audio_{j}"] = [f"ref_aud{j}", 0]
            encode_nodes.append(f"ref_aud{j}")
        g["cond"] = cond
        encode_nodes.append("cond")
        positive = ["cond", 0]
        for key, img, idx in (("guide_first", "img_first", 0), ("guide_last", "img_last", -1)):
            if img in g:
                g[key] = _node("MiniMaxH3AddGuide", positive=positive, vae=["vvae", 0], latent=["cond", 1], image=[img, 0], frame_idx=idx)
                positive = [key, 0]
                encode_nodes.append(key)
    else:
        cond = _node("MiniMaxH3ImageToVideo", clip=["te", 0], vae=["vvae", 0], prompt=prompt, width=width, height=height, length=length)
        if first_frame:
            cond["inputs"]["first_frame"] = ["img_first", 0]
        if last_frame:
            cond["inputs"]["last_frame"] = ["img_last", 0]
        g["cond"] = cond
        encode_nodes.append("cond")
        positive = ["cond", 0]

    if phase == "encode":
        out = {k: g[k] for k in encode_nodes}
        out["warm"] = _node("H3TurboConditioningWarmup", conditioning=positive)
        return out
    g["unet"] = _node("H3TurboFastUNetLoader", h3t_name=h3t_name, precision=precision, resident_blocks=0, mlp_chunk=0, attention=attention)
    g["noise"] = _node("RandomNoise", noise_seed=int(seed))
    g["ksel"] = _node("KSamplerSelect", sampler_name=sampler)
    g["sigmas"] = _node("BasicScheduler", scheduler=scheduler, steps=int(steps), denoise=1.0, model=["unet", 0])
    g["guider"] = _node("BasicGuider", model=["unet", 0], conditioning=positive)
    g["sample"] = _node("SamplerCustomAdvanced", noise=["noise", 0], guider=["guider", 0], sampler=["ksel", 0], sigmas=["sigmas", 0],
                        latent_image=["cond", 1])
    g["dec"] = _node("VAEDecode", samples=["sample", 0], vae=["vvae", 0])
    g["adec"] = _node("VAEDecodeAudio", samples=["sample", 0], vae=["avae", 0])
    g["video"] = _node("CreateVideo", fps=fps, bit_depth=8, color_space="sRGB", images=["dec", 0], audio=["adec", 0])
    g["save"] = _node("SaveVideo", filename_prefix=prefix, format="auto", **{"format.codec": "auto"}, video=["video", 0])
    return g


def reference_tags(ref_images: Sequence[str] = (), ref_videos: Sequence = (), ref_audios: Sequence[str] = ()) -> List[str]:
    """How MiniMaxH3ReferenceToVideo numbers references for the prompt: images, then videos (a video's soundtrack takes the
    <Audio j> label right before its <Video k>), then standalone audio; 1-based per type."""
    tags, a, v = [], 0, 0
    for i, name in enumerate(ref_images):
        tags.append(f"<Picture {i + 1}> = {os.path.basename(name)}")
    for name, with_audio in _ref_videos(ref_videos):
        if with_audio:
            a += 1
            tags.append(f"<Audio {a}> = soundtrack of <Video {v + 1}>")
        v += 1
        tags.append(f"<Video {v}> = {os.path.basename(name)}")
    for name in ref_audios:
        a += 1
        tags.append(f"<Audio {a}> = {os.path.basename(name)}")
    return tags


# ----------------------------------------------------------------------------------------------- minimal websocket client
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocket:
    """Just enough RFC 6455 for ComfyUI's /ws progress stream (text frames, fragmentation, ping/pong, close)."""

    def __init__(self, sock: socket.socket, buf: bytes = b""):
        self.sock, self._buf = sock, bytearray(buf)
        self._frag, self._frag_op = bytearray(), None

    @classmethod
    def connect(cls, host: str, port: int, path: str, timeout: float = 10.0) -> "WebSocket":
        sock = socket.create_connection((host, port), timeout=timeout)
        try:
            key = base64.b64encode(os.urandom(16)).decode()
            sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                          f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("websocket handshake: connection closed")
                head += chunk
                if len(head) > 65536:
                    raise ConnectionError("websocket handshake: header too long")
            head, rest = head.split(b"\r\n\r\n", 1)
            lines = head.decode("latin-1").split("\r\n")
            parts = lines[0].split()
            if len(parts) < 2 or parts[1] != "101":
                raise ConnectionError(f"websocket handshake refused: {lines[0]}")
            hdrs = {k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
            want = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
            if hdrs.get("sec-websocket-accept") != want:
                raise ConnectionError("websocket handshake: wrong Sec-WebSocket-Accept")
            return cls(sock, rest)
        except BaseException:
            sock.close()
            raise

    @staticmethod
    def build_frame(opcode: int, payload: bytes, mask: bool = True, fin: bool = True) -> bytes:
        hdr = bytearray([(0x80 if fin else 0) | opcode])
        n, mbit = len(payload), 0x80 if mask else 0
        if n < 126:
            hdr.append(mbit | n)
        elif n < 65536:
            hdr.append(mbit | 126)
            hdr += struct.pack(">H", n)
        else:
            hdr.append(mbit | 127)
            hdr += struct.pack(">Q", n)
        if not mask:
            return bytes(hdr) + payload
        key = os.urandom(4)
        return bytes(hdr) + key + bytes(b ^ key[i % 4] for i, b in enumerate(payload))

    @staticmethod
    def parse_frame(buf) -> Optional[Tuple[bool, int, bytes, int]]:
        """(fin, opcode, payload, bytes consumed), or None if `buf` does not hold a whole frame yet."""
        if len(buf) < 2:
            return None
        fin, op, masked, n, i = bool(buf[0] & 0x80), buf[0] & 0x0F, bool(buf[1] & 0x80), buf[1] & 0x7F, 2
        if n == 126:
            if len(buf) < 4:
                return None
            n, i = struct.unpack(">H", bytes(buf[2:4]))[0], 4
        elif n == 127:
            if len(buf) < 10:
                return None
            n, i = struct.unpack(">Q", bytes(buf[2:10]))[0], 10
        key = None
        if masked:
            if len(buf) < i + 4:
                return None
            key, i = bytes(buf[i:i + 4]), i + 4
        if len(buf) < i + n:
            return None
        payload = bytes(buf[i:i + n])
        if key:
            payload = bytes(b ^ key[k % 4] for k, b in enumerate(payload))
        return fin, op, payload, i + n

    def recv(self, timeout: Optional[float] = None) -> Optional[Union[str, bytes]]:
        """Next message (str for text, bytes for binary), or None when `timeout` passes first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            fr = self.parse_frame(self._buf)
            if fr is not None:
                fin, op, payload, used = fr
                del self._buf[:used]
                if op == 0x9:
                    self.sock.sendall(self.build_frame(0xA, payload))
                elif op == 0x8:
                    raise ConnectionError("websocket closed by the server")
                elif op in (0x1, 0x2):
                    if fin:
                        return payload.decode("utf-8", "replace") if op == 0x1 else payload
                    self._frag, self._frag_op = bytearray(payload), op
                elif op == 0x0:
                    self._frag += payload
                    if fin:
                        data, op0 = bytes(self._frag), self._frag_op
                        self._frag, self._frag_op = bytearray(), None
                        return data.decode("utf-8", "replace") if op0 == 0x1 else data
                continue  # pong / unknown opcodes are ignored
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                return None
            self.sock.settimeout(left)
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                return None
            if not chunk:
                raise ConnectionError("websocket connection closed")
            self._buf += chunk

    def close(self) -> None:
        try:
            self.sock.sendall(self.build_frame(0x8, b""))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# ----------------------------------------------------------------------------------------------- RAM / VRAM monitor
def _tree_rss(pid: Optional[int]) -> int:
    if not pid:
        return 0
    try:
        import psutil

        p = psutil.Process(pid)
        return int(sum(q.memory_info().rss for q in [p] + p.children(recursive=True)))
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def gpu_memory_mib() -> Tuple[Optional[float], Optional[float]]:
    """(used, total) MiB of GPU 0 from nvidia-smi, or (None, None)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        used, total = (float(x) for x in out.split(","))
        return used, total
    except Exception:
        return None, None


class Monitor:
    """Samples system RAM in use, ComfyUI's RSS (with children) and GPU memory while a prompt runs; reports the peaks."""

    def __init__(self, pid: Optional[int], interval: float = 0.5, gpu_interval: float = 2.0):
        self.pid, self.interval, self.gpu_interval = pid, interval, gpu_interval
        self.peak_used = self.peak_rss = 0
        self.peak_vram = None
        self.ram_total = self.vram_total = None
        self.min_available = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "Monitor":
        self._thread.start()
        return self

    def _loop(self) -> None:
        last_gpu = 0.0
        while not self._stop.is_set():
            total, avail = _ram_bytes()
            if total:
                self.ram_total = total
                self.peak_used = max(self.peak_used, total - avail)
                self.min_available = avail if self.min_available is None else min(self.min_available, avail)
            self.peak_rss = max(self.peak_rss, _tree_rss(self.pid))
            now = time.monotonic()
            if now - last_gpu >= self.gpu_interval:
                used, tot = gpu_memory_mib()
                last_gpu = now
                if used is not None:
                    self.vram_total = tot
                    self.peak_vram = used if self.peak_vram is None else max(self.peak_vram, used)
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=15)
        return {"ram_used_gib": _gib(self.peak_used), "ram_total_gib": _gib(self.ram_total or 0),
                "ram_min_available_gib": None if self.min_available is None else _gib(self.min_available),
                "comfy_rss_gib": _gib(self.peak_rss),
                "vram_used_gib": None if self.peak_vram is None else self.peak_vram / 1024,
                "vram_total_gib": None if self.vram_total is None else self.vram_total / 1024}


# ----------------------------------------------------------------------------------------------- progress / timing
CATEGORY = {
    "H3TurboCachedTextEncoder": "text encode", "CLIPLoaderGGUF": "text encode", "CLIPLoader": "text encode",
    "MiniMaxH3ImageToVideo": "text encode", "MiniMaxH3ReferenceToVideo": "text encode", "MiniMaxH3AddGuide": "text encode",
    "H3TurboConditioningWarmup": "text encode",
    "LoadImage": "load inputs", "LoadVideo": "load inputs", "GetVideoComponents": "load inputs", "LoadAudio": "load inputs",
    "VAELoader": "load models", "H3TurboFastUNetLoader": "load models",
    "RandomNoise": "sampler", "KSamplerSelect": "sampler", "BasicScheduler": "sampler", "BasicGuider": "sampler",
    "SamplerCustomAdvanced": "sampler",
    "VAEDecode": "decode", "VAEDecodeAudio": "decode",
    "CreateVideo": "save", "SaveVideo": "save",
}
CATEGORY_ORDER = ("load inputs", "load models", "text encode", "sampler", "decode", "save", "other")
CATEGORY_LABEL = {"text encode": "text encode (+ VAE encode of frames/refs)", "sampler": "sampler (DiT)",
                  "decode": "decode (video + audio VAE)", "load models": "load VAE files", "load inputs": "load input files",
                  "save": "mux + save mp4", "other": "other"}


_QUIET = {"RandomNoise", "KSamplerSelect", "BasicScheduler", "BasicGuider", "H3TurboConditioningWarmup"}


class Tracker:
    """Turns ComfyUI websocket events for one prompt into node timings and progress lines."""

    def __init__(self, prompt_id: str, graph: dict, t0: float, label: str = "", echo: bool = True):
        self.pid, self.graph, self.t0, self.label, self.echo = prompt_id, graph, t0, label, echo
        self.node_seconds: Dict[str, float] = {}
        self.node_start: Dict[str, float] = {}
        self.steps: Dict[str, List[Tuple[float, int, int]]] = {}
        self.cached: List[str] = []
        self.current: Optional[Tuple[str, float]] = None
        self.done = False
        self.error: Optional[dict] = None
        self.t_end: Optional[float] = None

    def _say(self, t: float, text: str) -> None:
        if self.echo:
            print(f"[{t - self.t0:7.1f} s] {self.label}{text}", flush=True)

    def _close(self, t: float) -> None:
        if self.current:
            node, ts = self.current
            self.node_seconds[node] = self.node_seconds.get(node, 0.0) + (t - ts)
            self.current = None

    def feed(self, msg: dict, t: float) -> None:
        typ, data = msg.get("type"), msg.get("data") or {}
        if not isinstance(data, dict) or data.get("prompt_id", self.pid) != self.pid:
            return
        if typ == "execution_cached":
            self.cached = [str(n) for n in data.get("nodes") or []]
            if self.cached and self.echo:
                self._say(t, f"cached, not re-run: {', '.join(self.cached)}")
        elif typ == "executing":
            node = data.get("node")
            self._close(t)
            if node is None:
                if "prompt_id" in data:
                    self.done, self.t_end = True, self.t_end or t
                return
            node = str(node)
            self.current = (node, t)
            self.node_start.setdefault(node, t - self.t0)
            cls = self.graph.get(node, {}).get("class_type", "?")
            if cls not in _QUIET:
                self._say(t, f"{node} ({cls})")
        elif typ == "progress":
            node = str(data.get("node"))
            value, mx = int(data.get("value") or 0), int(data.get("max") or 0)
            seq = self.steps.setdefault(node, [])
            if seq and seq[-1][1] == value:
                return
            seq.append((t - self.t0, value, mx))
            if self.graph.get(node, {}).get("class_type") == "SamplerCustomAdvanced":
                start = self.node_start.get(node, seq[0][0])
                if len(seq) == 1:
                    self._say(t, f"sampling step {value}/{mx} (first step {seq[0][0] - start:.1f} s, includes putting the DiT on the GPU)")
                else:
                    self._say(t, f"sampling step {value}/{mx} ({seq[-1][0] - seq[-2][0]:.2f} s)")
        elif typ == "execution_success":
            self._close(t)
            self.done, self.t_end = True, t
        elif typ in ("execution_error", "execution_interrupted"):
            self._close(t)
            self.done, self.t_end, self.error = True, t, dict(data, type=typ)


@dataclass
class PromptRun:
    label: str
    prompt_id: str
    graph: dict
    wall: float
    node_seconds: Dict[str, float]
    node_start: Dict[str, float]
    cached: List[str]
    steps: Dict[str, List[Tuple[float, int, int]]]
    history: dict
    peaks: dict
    log: str = ""

    def categories(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for node, s in self.node_seconds.items():
            cat = CATEGORY.get(self.graph.get(node, {}).get("class_type", ""), "other")
            out[cat] = out.get(cat, 0.0) + s
        return out

    def sampler_detail(self) -> str:
        for node, seq in self.steps.items():
            if self.graph.get(node, {}).get("class_type") != "SamplerCustomAdvanced" or not seq:
                continue
            start = self.node_start.get(node, seq[0][0])
            first = seq[0][0] - start
            if len(seq) > 1:
                rest = (seq[-1][0] - seq[0][0]) / (len(seq) - 1)
                return f"{seq[-1][1]} steps: first {first:.1f} s (includes putting the DiT on the GPU), then {rest:.2f} s/step"
            return f"{seq[-1][1]} step(s), first {first:.1f} s"
        return ""

    def outputs(self, output_dir: str) -> List[str]:
        files = [os.path.join(output_dir, it.get("subfolder", ""), it["filename"])
                 for out in (self.history.get("outputs") or {}).values() for it in out.get("images", []) + out.get("videos", [])
                 if isinstance(it, dict) and "filename" in it]
        return [f for f in files if f.lower().endswith((".mp4", ".webm", ".mkv", ".mov"))]


def _fmt_peaks(p: dict) -> str:
    if not p:
        return ""
    s = f"peak RAM in use {p['ram_used_gib']:.1f} of {p['ram_total_gib']:.1f} GiB (ComfyUI {p['comfy_rss_gib']:.1f} GiB)"
    if p.get("vram_used_gib") is not None:
        s += f", peak VRAM {p['vram_used_gib']:.1f} of {p['vram_total_gib']:.1f} GiB"
    return s


@dataclass
class Result:
    path: str
    runs: List[PromptRun]
    free_seconds: Optional[float]
    total: float
    settings: dict

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path

    def report(self) -> str:
        lines = ["timing, measured from ComfyUI's websocket events during this run:"]
        for run in self.runs:
            lines.append(f"  {run.label or 'generate':<34} {run.wall:7.1f} s")
            cats = run.categories()
            for cat in CATEGORY_ORDER:
                if cats.get(cat, 0) >= 0.05:
                    extra = f"   {run.sampler_detail()}" if cat == "sampler" else ""
                    lines.append(f"    {CATEGORY_LABEL[cat]:<32} {cats[cat]:7.1f} s{extra}")
            if run.cached:
                lines.append(f"    cached, not re-run: {', '.join(run.cached)}")
            for ln in run.log.splitlines():
                lines.append(f"    log: {ln.strip()[-160:]}")
            if run.peaks:
                lines.append(f"    {_fmt_peaks(run.peaks)}")
            if run is self.runs[0] and self.free_seconds is not None and len(self.runs) > 1:
                lines.append(f"  {'free memory between the phases':<34} {self.free_seconds:7.1f} s")
        lines.append(f"  {'total':<34} {self.total:7.1f} s")
        return "\n".join(lines)


class ServerDied(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- the ComfyUI server
def parse_startup_log(text: str) -> dict:
    info: Dict[str, object] = {}
    m = re.search(r"pytorch version: (\S+)", text)
    info["torch"] = m.group(1) if m else None
    m = re.search(r"Set vram state to: (\S+)", text)
    info["vram_state"] = m.group(1) if m else None
    info["dynamic_vram"] = "DynamicVRAM support detected and enabled" in text
    info["kitchen_cuda"] = None
    m = re.search(r"Found comfy_kitchen backend cuda: (\{.*\})", text)
    if m:
        try:
            d = ast.literal_eval(m.group(1))
            info["kitchen_cuda"] = bool(d.get("available")) and not d.get("disabled") and "w4a8_int8_linear" in (d.get("capabilities") or [])
        except (ValueError, SyntaxError):
            pass
    return info


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ComfyServer:
    """A headless ComfyUI on 127.0.0.1. Keep one alive between generations so models (and ComfyUI's node cache) stay warm."""

    def __init__(self, comfy_dir: str, port: Optional[int] = None, extra_args: Sequence[str] = (), profile: Optional[Profile] = None,
                 env: Optional[Dict[str, str]] = None):
        self.comfy_dir, self.port, self.extra_args = comfy_dir, port or _free_port(), list(extra_args)
        self.profile, self.env = profile, dict(env or {})
        self.proc: Optional[subprocess.Popen] = None
        self.log_path = os.path.join(comfy_dir, "h3turbo_server.log")
        self._log = None
        self._log_start = 0
        self._attached = False
        self.info: dict = {}
        self.last_cond_sig: Optional[str] = None

    @property
    def output_dir(self) -> str:
        return os.path.join(self.comfy_dir, "output")

    @property
    def input_dir(self) -> str:
        return os.path.join(self.comfy_dir, "input")

    @property
    def kitchen_cuda(self) -> Optional[bool]:
        return self.info.get("kitchen_cuda")

    def command(self) -> List[str]:
        flags = list(self.profile.comfy_args) if self.profile else []
        return [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port), "--disable-auto-launch",
                "--disable-api-nodes", "--preview-method", "none", *flags, *self.extra_args]

    def start(self, timeout: float = 900) -> "ComfyServer":
        if self.alive():
            return self
        if self._log:
            self._log.close()
        self._log_start = self._log_size()
        self._log = open(self.log_path, "ab")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env.update(self.profile.env if self.profile else {})
        env.update(self.env)
        cmd = self.command()
        print("starting ComfyUI:", " ".join(cmd[1:]))
        self.proc = subprocess.Popen(cmd, cwd=self.comfy_dir, stdout=self._log, stderr=subprocess.STDOUT, env=env)
        try:  # if RAM runs out, the kernel should kill ComfyUI (restartable), not the notebook
            with open(f"/proc/{self.proc.pid}/oom_score_adj", "w") as f:
                f.write("500")
        except OSError:
            pass
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise ServerDied(f"ComfyUI exited with code {self.proc.returncode}; last log lines:\n{self.tail(40)}")
            try:
                self.request("/system_stats", timeout=5)
                break
            except (OSError, RuntimeError):
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"ComfyUI did not start in {timeout:.0f} s; last log lines:\n{self.tail()}")
                time.sleep(2)
        missing = set(REQUIRED_NODES) - set(self.request("/object_info", timeout=300))
        if missing:
            raise RuntimeError(f"ComfyUI is up but these nodes are missing: {sorted(missing)} (a custom node failed to import; "
                               f"search the log for 'H3-Turbo' or 'GGUF'); last log lines:\n{self.tail(60)}")
        self.info = parse_startup_log(self.log_since(self._log_start))
        self.last_cond_sig = None
        print(f"ComfyUI ready on port {self.port} after {time.time() - t0:.0f} s (log: {self.log_path}); "
              f"torch {self.info.get('torch')}, vram state {self.info.get('vram_state')}, "
              f"dynamic VRAM {'on' if self.info.get('dynamic_vram') else 'off'}, "
              f"comfy_kitchen CUDA kernels {'on' if self.kitchen_cuda else 'off' if self.kitchen_cuda is False else 'unknown'}")
        return self

    @classmethod
    def attach(cls, comfy_dir: str, port: int = 8188, profile: Optional[Profile] = None) -> "ComfyServer":
        """Use a ComfyUI that is already running instead of starting one; stop() leaves it alone."""
        srv = cls(comfy_dir, port, profile=profile)
        srv._attached = True
        srv.request("/system_stats", timeout=10)
        return srv

    def alive(self) -> bool:
        if self._attached:
            try:
                self.request("/system_stats", timeout=10)
                return True
            except (OSError, RuntimeError):
                return False
        return self.proc is not None and self.proc.poll() is None

    def ensure_running(self) -> "ComfyServer":
        if self.alive():
            return self
        if self._attached:
            raise ServerDied(f"the attached ComfyUI on port {self.port} is not answering")
        if self.proc is not None:
            rc = self.proc.returncode
            hint = " (SIGKILL: almost certainly out of RAM)" if rc in (-9, 137) else ""
            print(f"ComfyUI is not running (exit code {rc}{hint}); restarting it. Last log lines:\n{self.tail(15)}")
        return self.start()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None and not self._attached:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log:
            self._log.close()
            self._log = None

    def _log_size(self) -> int:
        try:
            return os.path.getsize(self.log_path)
        except OSError:
            return 0

    def log_since(self, offset: int, limit: int = 4 << 20) -> str:
        try:
            with open(self.log_path, "rb") as f:
                size = f.seek(0, 2)
                f.seek(max(0, offset, size - limit))
                return f.read().decode("utf-8", "replace").replace("\r", "\n")
        except OSError:
            return ""

    def tail(self, n: int = 30) -> str:
        lines = self.log_since(self._log_size() - (256 << 10)).splitlines()
        return "\n".join(line for line in lines[-n:] if line.strip())

    def request(self, path: str, payload=None, timeout: float = 60):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:  # ComfyUI explains a rejected prompt in the body
            raise RuntimeError(f"{path}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:3000]}") from None
        try:
            return json.loads(body) if body else {}
        except ValueError:
            return {}

    def free(self, unload_models: bool = True, free_memory: bool = False, settle: float = 10.0) -> float:
        """POST /free and wait until it has been applied (the worker applies it as soon as it is idle, so we wait a moment and
        then for GPU memory to settle before queueing the next prompt). Returns the seconds spent."""
        t0 = time.perf_counter()
        self.request("/free", {"unload_models": bool(unload_models), "free_memory": bool(free_memory)})
        time.sleep(1.0)
        prev, _ = gpu_memory_mib()
        while time.perf_counter() - t0 < settle:
            time.sleep(0.5)
            used, _ = gpu_memory_mib()
            if used is None or prev is None or abs(used - prev) < 64:
                break
            prev = used
        return time.perf_counter() - t0

    def run(self, graph: dict, *, label: str = "", timeout: float = 7200, echo: bool = True) -> PromptRun:
        """Queue one prompt, follow it over the websocket (polling /history if that fails), return its timings and history."""
        self.ensure_running()
        cid = uuid.uuid4().hex
        ws = None
        try:
            ws = WebSocket.connect("127.0.0.1", self.port, f"/ws?clientId={cid}", timeout=10)
        except Exception as e:
            print(f"(no websocket progress: {e!r}; polling instead)")
        log_off = self._log_size()
        mon = Monitor(self.proc.pid if self.proc else None).start()
        t0 = time.perf_counter()
        tracker = None
        try:
            r = self.request("/prompt", {"prompt": graph, "client_id": cid})
            if r.get("node_errors"):
                raise RuntimeError("ComfyUI rejected the graph: " + json.dumps(r["node_errors"])[:3000])
            pid = r["prompt_id"]
            tracker = Tracker(pid, graph, t0, f"{label}: " if label else "", echo)
            last_poll = time.perf_counter()
            while not tracker.done:
                if ws is not None:
                    try:
                        msg = ws.recv(timeout=1.0)
                    except (OSError, ConnectionError) as e:
                        print(f"(websocket lost: {e!r}; polling instead)")
                        ws = None
                        msg = None
                    if isinstance(msg, str):
                        try:
                            tracker.feed(json.loads(msg), time.perf_counter())
                        except ValueError:
                            pass
                else:
                    time.sleep(1.0)
                now = time.perf_counter()
                if not tracker.done and (ws is None or now - last_poll > 15):  # safety net for a missed websocket event
                    last_poll = now
                    h = self.request(f"/history/{pid}").get(pid)
                    if h:
                        tracker.done, tracker.t_end = True, tracker.t_end or now
                if tracker.done:
                    break
                if not self.alive():
                    rc = self.proc.returncode if self.proc else None
                    hint = " (killed with SIGKILL: almost certainly out of RAM)" if rc in (-9, 137) else ""
                    self.last_cond_sig = None
                    raise ServerDied(f"ComfyUI died during '{label or 'generate'}'{hint}; last log lines:\n{self.tail(40)}")
                if now - t0 > timeout:
                    self.request("/interrupt", {})
                    raise TimeoutError(f"no result after {timeout:.0f} s")
        except KeyboardInterrupt:  # the cell was stopped: stop ComfyUI's run too, or it keeps the GPU busy
            try:
                self.request("/interrupt", {}, timeout=10)
                print("interrupted ComfyUI's run")
            except Exception:
                pass
            raise
        finally:
            if ws is not None:
                ws.close()
            peaks = mon.stop()
        hist = {}
        for _ in range(100):  # /history can lag a moment behind the websocket
            hist = self.request(f"/history/{tracker.pid}").get(tracker.pid) or {}
            if hist:
                break
            time.sleep(0.2)
        status = (hist.get("status") or {})
        if tracker.error or status.get("status_str") not in (None, "success"):
            msgs = [m for m in status.get("messages", []) if m and m[0] in ("execution_error", "execution_interrupted")]
            err = tracker.error or msgs
            raise RuntimeError(f"'{label or 'generate'}' failed: {json.dumps(err, default=str)[-3000:]}\nlast log lines:\n{self.tail(40)}")
        wall = (tracker.t_end or time.perf_counter()) - t0
        h3_lines = [ln for ln in self.log_since(log_off).splitlines()
                    if "H3-Turbo" in ln and ("text encoder" in ln or "engine loaded" in ln or "compute dtype" in ln or "back on GPU" in ln)]
        return PromptRun(label, tracker.pid, graph, wall, tracker.node_seconds, tracker.node_start, tracker.cached, tracker.steps,
                         hist, peaks, "\n".join(h3_lines[-8:]))


# ----------------------------------------------------------------------------------------------- inputs: uploads and staging
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif")
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus")


def upload_files(dest: str = UPLOAD_DIR) -> List[str]:
    """Colab's file picker; files are saved in `dest` and their paths returned."""
    from google.colab import files  # type: ignore

    os.makedirs(dest, exist_ok=True)
    picked = files.upload()
    paths = []
    for name, data in picked.items():
        p = os.path.join(dest, os.path.basename(name))
        with open(p, "wb") as f:
            f.write(data)
        paths.append(p)
        print(f"saved {p} ({len(data) / 1e6:.1f} MB)")
    return paths


def list_uploads(dest: str = UPLOAD_DIR) -> List[str]:
    files = sorted(glob.glob(os.path.join(dest, "*")))
    if files:
        print(f"files in {dest} (use the name or the full path in the generate cell):")
        for p in files:
            print(f"  {os.path.basename(p)}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    else:
        print(f"{dest} is empty")
    return files


def resolve_files(spec: Union[str, Sequence[str], None], uploads: str = UPLOAD_DIR) -> List[str]:
    """Comma/newline-separated file list: absolute paths, names in /content/uploads, globs, Google Drive paths (mounted on
    demand), or the word `upload` (opens the file picker)."""
    if not spec:
        return []
    items = spec if isinstance(spec, (list, tuple)) else [s for s in re.split(r"[,\n]", spec)]
    out: List[str] = []
    for raw in items:
        s = str(raw).strip().strip('"').strip("'")
        if not s:
            continue
        if s.lower() == "upload":
            out += upload_files(uploads)
            continue
        if s.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
            try:
                from google.colab import drive  # type: ignore

                drive.mount("/content/drive")
            except Exception as e:
                raise FileNotFoundError(f"{s}: could not mount Google Drive ({e!r})") from None
        cands = [s, os.path.join(uploads, s), os.path.join("/content", s)]
        hit = next((c for c in cands if os.path.isfile(c)), None)
        if hit:
            out.append(hit)
            continue
        globbed = sorted(p for c in cands if any(ch in c for ch in "*?[") for p in glob.glob(c) if os.path.isfile(p))
        if globbed:
            out += globbed
            continue
        have = ", ".join(os.path.basename(p) for p in sorted(glob.glob(os.path.join(uploads, "*")))) or "nothing"
        raise FileNotFoundError(f"{s!r} not found (looked in {uploads} and /content; uploaded so far: {have})")
    return out


def _digest(path: str, *extra) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(repr(extra).encode())
    return h.hexdigest()[:16]


def probe_media(path: str) -> dict:
    """ffprobe summary: width, height (display orientation), fps, duration, has_video, has_audio."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffprobe could not read {path}: {r.stderr[-500:]}")
    d = json.loads(r.stdout)
    info = {"has_video": False, "has_audio": False, "duration": float((d.get("format") or {}).get("duration") or 0)}
    for s in d.get("streams", []):
        if s.get("codec_type") == "video" and not info["has_video"] and not (s.get("disposition") or {}).get("attached_pic"):
            w, h = int(s.get("width") or 0), int(s.get("height") or 0)
            rot = 0
            for sd in s.get("side_data_list") or []:
                if "rotation" in sd:
                    rot = int(float(sd["rotation"]))
            rot = int(float((s.get("tags") or {}).get("rotate", rot)))
            if rot % 180:
                w, h = h, w
            num, _, den = str(s.get("avg_frame_rate") or "0/1").partition("/")
            fps = float(num) / float(den or 1) if float(den or 1) else 0.0
            info.update(has_video=True, width=w, height=h, fps=fps)
        elif s.get("codec_type") == "audio":
            info["has_audio"] = True
    return info


def stage_image(src: str, input_dir: str) -> str:
    """Copy an image into ComfyUI's input folder under a content hash name (identical file = identical name, so ComfyUI's
    node cache and the text-encoder cache hit). Unusual formats are converted to PNG losslessly."""
    os.makedirs(input_dir, exist_ok=True)
    ext = os.path.splitext(src)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        name = f"h3c_{_digest(src)}{ext}"
        dst = os.path.join(input_dir, name)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        return name
    from PIL import Image  # type: ignore

    name = f"h3c_{_digest(src, 'png')}.png"
    dst = os.path.join(input_dir, name)
    if not os.path.exists(dst):
        with Image.open(src) as im:
            im.save(dst)
    return name


def stage_ref_video(src: str, input_dir: str, frames: int, keep_audio: bool = True) -> Tuple[str, bool]:
    """Prepare a reference video the way MiniMaxH3ReferenceToVideo will use it, without loss: 24 fps (the node assumes 24),
    its own canvas size (the node resizes to exactly this), at most `frames` frames (the node keeps no more than the output
    length), lossless H.264 + ALAC. Avoids ComfyUI decoding a long full-resolution clip into RAM. Returns (name, has_audio)."""
    os.makedirs(input_dir, exist_ok=True)
    if not shutil.which("ffmpeg"):
        name = f"h3c_{_digest(src)}{os.path.splitext(src)[1].lower()}"
        if not os.path.exists(os.path.join(input_dir, name)):
            shutil.copyfile(src, os.path.join(input_dir, name))
        print(f"WARNING: no ffmpeg; {os.path.basename(src)} is used as is (its frame rate is assumed to be 24 fps)")
        return name, keep_audio
    info = probe_media(src)
    if not info["has_video"]:
        raise ValueError(f"{src} has no video stream")
    if info.get("duration") and info["duration"] * FPS < 5:
        raise ValueError(f"{os.path.basename(src)}: reference videos need at least 5 frames (~0.2 s)")
    cw, ch = ref_video_canvas(info["width"], info["height"])
    with_audio = keep_audio and info["has_audio"]
    name = f"h3c_{_digest(src, cw, ch, frames, with_audio, 'v1')}.mp4"
    dst = os.path.join(input_dir, name)
    if not os.path.exists(dst):
        tmp = dst + ".tmp.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-t", f"{frames / FPS:.4f}", "-map", "0:v:0",
               "-vf", f"fps={FPS},scale={cw}:{ch}:flags=lanczos", "-frames:v", str(frames),
               "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p"]
        cmd += ["-map", "0:a:0", "-c:a", "alac"] if with_audio else ["-an"]
        r = subprocess.run(cmd + [tmp], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ffmpeg failed on {src}: {r.stderr[-800:]}")
        os.replace(tmp, dst)
    return name, with_audio


def stage_audio(src: str, input_dir: str, max_seconds: float = 15.0) -> str:
    """Decode to 32-bit float WAV (lossless for the decoded samples), first `max_seconds` only."""
    os.makedirs(input_dir, exist_ok=True)
    if not shutil.which("ffmpeg"):
        name = f"h3c_{_digest(src)}{os.path.splitext(src)[1].lower()}"
        if not os.path.exists(os.path.join(input_dir, name)):
            shutil.copyfile(src, os.path.join(input_dir, name))
        return name
    name = f"h3c_{_digest(src, max_seconds, 'a1')}.wav"
    dst = os.path.join(input_dir, name)
    if not os.path.exists(dst):
        tmp = dst + ".tmp.wav"
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-t", f"{max_seconds:.3f}", "-vn", "-map", "0:a:0",
                            "-c:a", "pcm_f32le", tmp], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ffmpeg could not read audio from {src}: {r.stderr[-800:]}")
        os.replace(tmp, dst)
    return name


def image_size(path: str) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return im.size
    except Exception:
        info = probe_media(path)
        return info["width"], info["height"]


# ----------------------------------------------------------------------------------------------- generation
def _default_profile() -> Profile:
    return Profile("big", False, [], TE_Q2, "auto", {"unload_models": True, "free_memory": False})


def generate(srv: ComfyServer, prompt: str, *, h3t_name: str = MODEL_FILE, text_encoder: Optional[str] = None, width: int = 640,
             height: int = 384, length: int = 22, steps: int = 8, seed: int = 0, attention: str = "exact", precision: str = "a8",
             first_frame: Optional[str] = None, last_frame: Optional[str] = None, image: Optional[str] = None,
             ref_images: Sequence[str] = (), ref_videos: Sequence[str] = (), ref_audios: Sequence[str] = (),
             ref_video_audio: bool = True, ref_image_size: str = "match", match_image_aspect: bool = True,
             max_ref_audio_seconds: float = 15.0, te_cache: bool = True, timeout: float = 7200, prefix: str = "video/H3Turbo",
             echo: bool = True) -> Result:
    """One generation. Files are local paths (they are staged into ComfyUI's input folder). Returns a Result whose `path` is the
    mp4; `print(result.report())` shows the timing breakdown."""
    srv.ensure_running()
    prof = srv.profile or _default_profile()
    te = text_encoder or prof.text_encoder
    first_frame = first_frame or image
    t_start = time.perf_counter()
    w, h = fit32(width), fit32(height)
    if (w, h) != (width, height):
        print(f"canvas {width}x{height} -> {w}x{h} (multiples of 32)")
    frames = snap_length(length)
    if frames != length:
        print(f"length {length} -> {frames} frames (H3's 17k+5 grid)")
    refs = bool(ref_images or ref_videos or ref_audios)
    if first_frame and match_image_aspect and not refs:
        iw, ih = image_size(first_frame)
        mw, mh = match_aspect(iw, ih, w, h)
        if (mw, mh) != (w, h):
            print(f"canvas {w}x{h} -> {mw}x{mh} to match the first frame's aspect ({iw}x{ih}); turn off match_image_aspect to keep it")
            w, h = mw, mh
    if attention == "int8_fast" and srv.kitchen_cuda is False:
        print("int8_fast attention needs comfy_kitchen's CUDA kernels (CUDA 13 PyTorch on sm80+), which this ComfyUI does not "
              "have: using exact attention")
        attention = "exact"
    inp = srv.input_dir
    ff = stage_image(first_frame, inp) if first_frame else None
    lf = stage_image(last_frame, inp) if last_frame else None
    imgs = [stage_image(p, inp) for p in ref_images]
    vids = [stage_ref_video(p, inp, frames, ref_video_audio) for p in ref_videos]
    auds = [stage_audio(p, inp, max_ref_audio_seconds) for p in ref_audios]
    if refs:
        print("reference tags for the prompt: " + "; ".join(reference_tags(ref_images, [(p, a) for p, (_, a) in zip(ref_videos, vids)], ref_audios)))
    kw = dict(width=w, height=h, length=frames, h3t_name=h3t_name, text_encoder=te, steps=steps, seed=seed, attention=attention,
              precision=precision, first_frame=ff, last_frame=lf, ref_images=imgs, ref_videos=vids, ref_audios=auds,
              ref_image_size=ref_image_size, te_release=prof.te_release, te_cache=te_cache, prefix=prefix)
    full = build_graph(prompt, **kw)
    runs: List[PromptRun] = []
    free_s = None
    if prof.two_phase:
        enc = build_graph(prompt, phase="encode", **kw)
        sig = json.dumps(enc, sort_keys=True)
        if sig != srv.last_cond_sig:
            if echo:
                print("phase 1/2: text encoding (the DiT is not loaded in this prompt)")
            runs.append(srv.run(enc, label="phase 1 encode", timeout=timeout, echo=echo))
            free_s = srv.free(**prof.free_payload)
            srv.last_cond_sig = sig
            if echo:
                print(f"freed memory between the phases in {free_s:.1f} s; phase 2/2: sampling + decode")
        elif echo:
            print("same conditioning as the last generation: single pass (ComfyUI reuses the stored encode)")
    try:
        runs.append(srv.run(full, label="phase 2 generate" if prof.two_phase else "generate", timeout=timeout, echo=echo))
    except Exception:
        srv.last_cond_sig = None
        raise
    files = runs[-1].outputs(srv.output_dir)
    if not files:
        raise RuntimeError("the run finished but produced no video: " + json.dumps(runs[-1].history.get("outputs"))[:1000])
    res = Result(files[0], runs, free_s, time.perf_counter() - t_start,
                 dict(kw, prompt=prompt, profile=prof.name, seconds=frames / FPS))
    if echo:
        print(f"done: {res.path} ({w}x{h}, {frames} frames = {frames / FPS:.2f} s)")
        print(res.report())
    return res


def warmup(srv: ComfyServer, *, h3t_name: str, text_encoder: Optional[str] = None) -> Optional[Result]:
    """Load the text encoder, DiT and VAEs once with a tiny clip (256x256, 5 frames, 1 step), so the first real generation does
    not pay the disk reads. Only useful on the big profile, where everything then stays loaded."""
    print("warm-up: loading every model once with a tiny clip ...")
    return generate(srv, "warm-up", h3t_name=h3t_name, text_encoder=text_encoder, width=256, height=256, length=5, steps=1,
                    seed=0, prefix="warmup/H3Turbo", echo=False)


def run_task(srv: ComfyServer, task: str, prompt: str, *, preset: str = "", width: int = 640, height: int = 384, length: int = 22,
             steps: int = 8, seed: int = 0, random_seed: bool = False, attention: str = "exact", first_frame: str = "",
             last_frame: str = "", match_image_aspect: bool = True, ref_images: str = "", ref_videos: str = "",
             ref_video_soundtrack: bool = True, ref_audios: str = "", ref_image_size: str = "match", h3t_name: str = MODEL_FILE,
             text_encoder: Optional[str] = None, display: bool = True) -> Result:
    """The notebook's generate cell: task label from the form, file fields as text (see resolve_files), a preset or custom size."""
    kind = TASKS.get(task, task)
    if kind not in ("t2v", "i2v", "flf2v", "omni"):
        raise ValueError(f"unknown task {task!r}; one of {list(TASKS)}")
    if preset in PRESETS:
        width, height, length = PRESETS[preset]
    if random_seed:
        seed = random.randint(0, 2**31 - 1)
        print(f"seed {seed}")
    ff = resolve_files(first_frame) if kind in ("i2v", "flf2v", "omni") else []
    lf = resolve_files(last_frame) if kind in ("flf2v", "omni") else []
    if kind in ("i2v", "flf2v") and not ff:
        raise ValueError("this task needs FIRST_FRAME (a file name in /content/uploads, a path, or 'upload')")
    if kind == "flf2v" and not lf:
        raise ValueError("first + last frame needs LAST_FRAME too")
    imgs = resolve_files(ref_images) if kind == "omni" else []
    vids = resolve_files(ref_videos) if kind == "omni" else []
    auds = resolve_files(ref_audios) if kind == "omni" else []
    if kind == "omni" and not (imgs or vids or auds):
        raise ValueError("omni reference needs at least one of REF_IMAGES, REF_VIDEOS, REF_AUDIOS")
    res = generate(srv, prompt, h3t_name=h3t_name, text_encoder=text_encoder, width=width, height=height, length=length,
                   steps=steps, seed=seed, attention=attention, first_frame=ff[0] if ff else None, last_frame=lf[0] if lf else None,
                   ref_images=imgs, ref_videos=vids, ref_audios=auds, ref_video_audio=ref_video_soundtrack,
                   ref_image_size=ref_image_size, match_image_aspect=match_image_aspect)
    if display:
        show(res.path)
    return res


def show(path, width: int = 640):
    """Display an mp4 inline in a notebook (embedded, so it survives a runtime restart in the saved notebook)."""
    from IPython.display import Video, display  # type: ignore

    display(Video(os.fspath(path), embed=True, width=width))
