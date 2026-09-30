"""Run the official MiniMax H3 through ComfyUI headless, with the H3-Turbo Fast UNET Loader: the helpers the Colab notebook uses.

No web UI is exposed: the notebook starts ComfyUI as a local server, submits an API graph, waits, and shows the mp4. Standard
library only, so it also runs outside Colab (any machine with a ComfyUI install that has the H3-Turbo and ComfyUI-GGUF nodes).

    from h3_colab import ComfyServer, generate, show
    srv = ComfyServer("/content/ComfyUI").start()
    mp4 = generate(srv, "a red fox running through snow", width=640, height=384, length=22, seed=1,
                   h3t_name="model.h3t", text_encoder="qwen3vl-32B-MiniMax-H3-Q2_K.gguf")
    show(mp4)
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Optional

VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ComfyServer:
    """A ComfyUI server process on 127.0.0.1. Keep one alive between generations so models stay loaded (warm runs)."""

    def __init__(self, comfy_dir: str, port: Optional[int] = None, extra_args=()):
        self.comfy_dir, self.port, self.extra_args = comfy_dir, port or _free_port(), list(extra_args)
        self.proc: Optional[subprocess.Popen] = None
        self.log_path = os.path.join(comfy_dir, "h3turbo_server.log")
        self._log = None

    @property
    def output_dir(self) -> str:
        return os.path.join(self.comfy_dir, "output")

    @property
    def input_dir(self) -> str:
        return os.path.join(self.comfy_dir, "input")

    def start(self, timeout: float = 600) -> "ComfyServer":
        if self.alive():
            return self
        self._log = open(self.log_path, "ab")
        cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port), "--disable-auto-launch",
               "--disable-api-nodes", "--preview-method", "none", *self.extra_args]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(cmd, cwd=self.comfy_dir, stdout=self._log, stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited with code {self.proc.returncode}; last log lines:\n{self.tail()}")
            try:
                self.request("/system_stats", timeout=5)
                break
            except OSError:
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"ComfyUI did not start in {timeout:.0f} s; last log lines:\n{self.tail()}")
                time.sleep(2)
        missing = {"H3TurboFastUNetLoader", "CLIPLoaderGGUF", "MiniMaxH3ImageToVideo"} - set(self.request("/object_info", timeout=300))
        if missing:
            raise RuntimeError(f"ComfyUI is up but these nodes are missing: {sorted(missing)} (a custom node failed to import); "
                               f"last log lines:\n{self.tail(60)}")
        print(f"ComfyUI ready on port {self.port} after {time.time() - t0:.0f} s (log: {self.log_path})")
        return self

    @classmethod
    def attach(cls, comfy_dir: str, port: int = 8188) -> "ComfyServer":
        """Use a ComfyUI that is already running (e.g. a desktop install) instead of starting one; stop() leaves it alone."""
        srv = cls(comfy_dir, port)
        srv._attached = True
        srv.request("/system_stats", timeout=10)
        return srv

    def alive(self) -> bool:
        if getattr(self, "_attached", False):
            try:
                self.request("/system_stats", timeout=10)
                return True
            except OSError:
                return False
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log:
            self._log.close()
            self._log = None

    def tail(self, n: int = 30) -> str:
        try:
            with open(self.log_path, "rb") as f:
                lines = f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
            return "\n".join(line for line in lines[-n:] if line.strip())
        except OSError:
            return ""

    def request(self, path: str, payload=None, timeout: float = 60):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:  # ComfyUI explains a rejected prompt in the body
            raise RuntimeError(f"{path}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:3000]}") from None
        return json.loads(body) if body else {}


def build_graph(prompt: str, *, width: int, height: int, length: int, steps: int = 8, seed: int = 0, h3t_name: str,
                text_encoder: str, attention: str = "exact", precision: str = "a8", first_frame: Optional[str] = None,
                video_vae: str = VIDEO_VAE, audio_vae: str = AUDIO_VAE, fps: int = 24, prefix: str = "video/H3Turbo") -> dict:
    """The user's H3 workflow as an API graph, with the H3-Turbo loader in place of UNETLoader.

    text_encoder: a .gguf goes through ComfyUI-GGUF's CLIPLoaderGGUF, anything else through the core CLIPLoader (type minimax).
    first_frame: an image file name already in ComfyUI's input folder (image-to-video), or None for text-to-video."""
    if width % 32 or height % 32:
        raise ValueError("width and height must be multiples of 32")
    te = ({"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": text_encoder, "type": "minimax"}} if text_encoder.endswith(".gguf")
          else {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "minimax"}})
    g = {
        "unet": {"class_type": "H3TurboFastUNetLoader", "inputs": {"h3t_name": h3t_name, "precision": precision, "resident_blocks": 0,
                                                                   "mlp_chunk": 0, "attention": attention}},
        "te": te,
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": video_vae}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae}},
        "i2v": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"prompt": prompt, "width": width, "height": height, "length": length,
                                                                  "clip": ["te", 0], "vae": ["vvae", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": steps, "denoise": 1.0, "model": ["unet", 0]}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["unet", 0], "conditioning": ["i2v", 0]}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
                                                                     "sigmas": ["sigmas", 0], "latent_image": ["i2v", 1]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vvae", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"fps": fps, "bit_depth": 8, "color_space": "sRGB", "images": ["dec", 0],
                                                          "audio": ["adec", 0]}},
        "save": {"class_type": "SaveVideo", "inputs": {"filename_prefix": prefix, "format": "auto", "format.codec": "auto", "video": ["video", 0]}},
    }
    if first_frame:
        g["img"] = {"class_type": "LoadImage", "inputs": {"image": first_frame}}
        g["i2v"]["inputs"]["first_frame"] = ["img", 0]
    return g


def generate(srv: ComfyServer, prompt: str, *, image: Optional[str] = None, timeout: float = 7200, **kw) -> str:
    """Queue one generation, print progress, and return the path of the finished mp4. `image`: a local file for image-to-video."""
    first = None
    if image:
        first = f"h3turbo_{uuid.uuid4().hex[:8]}{os.path.splitext(image)[1] or '.png'}"
        os.makedirs(srv.input_dir, exist_ok=True)
        shutil.copyfile(image, os.path.join(srv.input_dir, first))
    graph = build_graph(prompt, first_frame=first, **kw)
    r = srv.request("/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
    if r.get("node_errors"):
        raise RuntimeError("ComfyUI rejected the graph: " + json.dumps(r["node_errors"])[:3000])
    pid, t0, last = r["prompt_id"], time.time(), ""
    while True:
        if not srv.alive():
            raise RuntimeError(f"ComfyUI died during the run; last log lines:\n{srv.tail(60)}")
        hist = srv.request(f"/history/{pid}").get(pid)
        if hist:
            break
        line = next((ln for ln in reversed(srv.tail(8).splitlines()) if "%|" in ln or "H3-Turbo" in ln or "Requested to load" in ln), "")
        if line and line != last:
            print(f"[{time.time() - t0:6.0f} s] {line.strip()[-150:]}", flush=True)
            last = line
        if time.time() - t0 > timeout:
            srv.request("/interrupt", {})
            raise TimeoutError(f"no result after {timeout:.0f} s")
        time.sleep(2)
    if hist["status"]["status_str"] != "success":
        msgs = [m for m in hist["status"].get("messages", []) if m[0] == "execution_error"]
        raise RuntimeError("generation failed: " + json.dumps(msgs)[-3000:] + f"\nlast log lines:\n{srv.tail(40)}")
    files = [os.path.join(srv.output_dir, im.get("subfolder", ""), im["filename"])
             for out in hist["outputs"].values() for im in out.get("images", []) + out.get("videos", [])]
    files = [f for f in files if f.lower().endswith((".mp4", ".webm", ".mkv"))]
    if not files:
        raise RuntimeError("the run finished but produced no video: " + json.dumps(hist["outputs"])[:1000])
    print(f"done in {time.time() - t0:.0f} s -> {files[0]}")
    return files[0]


def show(path: str, width: int = 640):
    """Display an mp4 inline in a notebook (embedded, so it survives a runtime restart in the saved notebook)."""
    from IPython.display import Video, display

    display(Video(path, embed=True, width=width))
