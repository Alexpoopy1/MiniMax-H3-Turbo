#!/usr/bin/env python
"""End-to-end check against a real ComfyUI: start the server headless on CPU, submit
workflows through its HTTP API, verify the outputs it wrote. Nothing is mocked.

    python scripts/comfyui_e2e.py --comfy /path/to/ComfyUI --ckpt weights/h3turbo-nano-toy.safetensors
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def call(port, path, payload=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path} -> HTTP {e.code}: {e.read().decode()[:2000]}")


def run_graph(port, graph, timeout=600):
    pid = call(port, "/prompt", {"prompt": graph})["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = call(port, f"/history/{pid}")
        if pid in h:
            st = h[pid]["status"]
            if st["status_str"] != "success":
                raise RuntimeError(json.dumps(st["messages"])[:3000])
            return h[pid]["outputs"]
        time.sleep(0.5)
    raise TimeoutError(pid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    node_dir = os.path.join(a.comfy, "custom_nodes", "MiniMax-H3-Turbo")
    if not os.path.exists(node_dir):
        os.symlink(REPO, node_dir)
    models = os.path.join(a.comfy, "models", "h3turbo")
    os.makedirs(models, exist_ok=True)
    shutil.copy(a.ckpt, os.path.join(models, "e2e.safetensors"))

    sys.path.insert(0, REPO)
    from h3turbo import toy
    from h3turbo.media import write_png

    sc = toy.Scene(color=2, direction=1, size=0.25, x0=0.3, y0=0.5)
    first = ((toy.render(sc, 96, 9)[0].permute(1, 2, 0) + 1) * 127.5).round().byte()
    os.makedirs(os.path.join(a.comfy, "input"), exist_ok=True)
    write_png(os.path.join(a.comfy, "input", "e2e_first.png"), first)

    log = open(os.path.join(a.comfy, "e2e_server.log"), "w")
    srv = subprocess.Popen([sys.executable, "main.py", "--cpu", "--port", str(a.port), "--disable-auto-launch",
                            "--disable-api-nodes", "--whitelist-custom-nodes", "MiniMax-H3-Turbo", "--dont-print-server"],
                           cwd=a.comfy, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            try:
                info = call(a.port, "/object_info/H3TurboGenerate")
                break
            except Exception:
                if srv.poll() is not None:
                    raise RuntimeError("ComfyUI exited early; see e2e_server.log")
                time.sleep(1)
        else:
            raise TimeoutError("ComfyUI did not come up")
        names = [n for n in call(a.port, "/object_info") if n.startswith("H3Turbo")]
        print("registered nodes:", sorted(names))
        assert len(names) == 6, names
        loader = {"class_type": "H3TurboLoader", "inputs": {"ckpt_name": "e2e.safetensors", "precision": "fp32", "quantize": "none", "offload_blocks": 0}}

        # 1) text -> video + audio, muxed to mp4 by core nodes
        g1 = {"1": loader,
              "2": {"class_type": "H3TurboGenerate", "inputs": {"pipe": ["1", 0], "prompt": "a red square moving left", "width": 96, "height": 96, "frames": 9, "fps": 8,
                                                                  "steps": 4, "guidance": 1.0, "seed": 1, "generate_video": True, "generate_audio": True}},
              "3": {"class_type": "CreateVideo", "inputs": {"images": ["2", 0], "fps": ["2", 2], "audio": ["2", 1]}},
              "4": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0], "filename_prefix": "h3turbo_e2e/t2va", "format": "mp4", "codec": "h264"}}}
        out = run_graph(a.port, g1)
        print("t2va outputs:", json.dumps(out)[:300])

        # 2) image -> video via Pin Frames, then in-context refine, saved as PNG frames
        g2 = {"1": loader,
              "5": {"class_type": "LoadImage", "inputs": {"image": "e2e_first.png"}},
              "6": {"class_type": "H3TurboPinFrames", "inputs": {"image": ["5", 0], "frame_index": 0}},
              "2": {"class_type": "H3TurboGenerate", "inputs": {"pipe": ["1", 0], "prompt": "a blue square moving right", "width": 96, "height": 96, "frames": 9, "fps": 8,
                                                                  "steps": 4, "guidance": 1.0, "seed": 2, "generate_video": True, "generate_audio": True, "omni_context": ["6", 0]}},
              "7": {"class_type": "H3TurboRefine", "inputs": {"pipe": ["1", 0], "generation": ["2", 3], "prompt": "a blue square moving right", "scale": 1.5, "steps": 2, "strength": 0.6, "seed": 3}},
              "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "h3turbo_e2e/i2v_refined"}}}
        out = run_graph(a.port, g2)
        imgs = out["8"]["images"]
        print("i2v+refine frames written:", len(imgs))
        assert len(imgs) == 9

        # 3) audio-only generation from the same checkpoint
        g3 = {"1": loader,
              "2": {"class_type": "H3TurboGenerate", "inputs": {"pipe": ["1", 0], "prompt": "a green square moving up", "width": 96, "height": 96, "frames": 9, "fps": 8,
                                                                  "steps": 2, "guidance": 1.0, "seed": 4, "generate_video": False, "generate_audio": True}},
              "9": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 1], "filename_prefix": "audio/h3turbo_e2e"}}}
        out = run_graph(a.port, g3)
        print("audio-only outputs:", json.dumps(out)[:200])

        outdir = os.path.join(a.comfy, "output", "h3turbo_e2e")
        files = sorted(os.listdir(outdir))
        print("output files:", files[:12], "...")
        mp4 = [f for f in files if f.endswith(".mp4")]
        assert mp4 and os.path.getsize(os.path.join(outdir, mp4[0])) > 1000
        print("E2E OK")
    finally:
        srv.terminate()
        try:
            srv.wait(15)
        except Exception:
            srv.kill()
        if not a.keep:
            try:
                os.remove(node_dir)
            except OSError:
                pass


if __name__ == "__main__":
    main()
