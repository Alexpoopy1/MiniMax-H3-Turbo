#!/usr/bin/env python
"""Headless end-to-end harness: run the user's real MiniMax H3 W4A8 workflow through a real ComfyUI
(native UNETLoader vs. the H3TurboFastUNetLoader node), then compare the two mp4 results bit for bit.

Use the ComfyUI venv python for both commands (compare needs PyAV + numpy):

    PY=D:/ComfyUIBig/ComfyUI/ComfyUI/.venv/Scripts/python.exe
    $PY scripts/comfy_h3_e2e.py run --variant native --width 512 --height 320 --length 22 --steps 8 --seed 42 --out-dir OUT_A
    $PY scripts/comfy_h3_e2e.py run --variant fast   --width 512 --height 320 --length 22 --steps 8 --seed 42 --out-dir OUT_B
    $PY scripts/comfy_h3_e2e.py compare --a OUT_A/result_native.mp4 --b OUT_B/result_fast.mp4 [--tol 0]

Nothing under the ComfyUI install is modified: the server gets its own output/temp/input/user dirs, and the repo is added as an
extra custom-node location through a directory junction + a `custom_nodes:` entry in a generated extra_model_paths yaml.
"""
import argparse, base64, itertools, json, os, shutil, socket, struct, subprocess, sys, threading, time, uuid  # noqa: E401
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = r"D:\ComfyUIBig\ComfyUI\ComfyUI"
PY = os.path.join(COMFY, ".venv", "Scripts", "python.exe")
WORKFLOW = os.path.join(COMFY, "user", "default", "workflows", "MiniMax H3 W4A8 - RTX 3050.json")
H3T = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_w4a8_convrot.h3t"
DEFAULT_PROMPT = "A slow cinematic shot of ocean waves rolling onto a sandy beach at sunset, gentle ambient music."
# widget values the frontend stores in widgets_values that are NOT plain entries of the node's `inputs` list
EXTRA_WIDGETS = {"RandomNoise": [None], "SaveVideo": ["format", "format.codec"]}  # None = frontend-only (control_after_generate)


# ----------------------------------------------------------------------------------------------- small helpers
def http(port, path, payload=None, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:3000]}")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tail(path, n=30):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log)"


def gpu_used_mib():
    """Total GPU memory in use. Per-process numbers are N/A under WDDM, so the total is the only usable busy signal."""
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return int(o.split()[0])
    except Exception:
        return None


def wait_gpu(limit_mib, max_wait=900):
    t0 = time.time()
    while True:
        used = gpu_used_mib()
        if used is None or used <= limit_mib:
            return
        if time.time() - t0 > max_wait:
            raise RuntimeError(f"GPU still busy after {max_wait}s ({used} MiB used > {limit_mib}); another job holds it (use --no-gpu-wait to ignore)")
        print(f"[gpu] {used} MiB in use (> {limit_mib}), another job is running; retry in 60 s", flush=True)
        time.sleep(60)


def kill_tree(proc):
    if proc.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def is_junction(p):
    return getattr(os.path, "isjunction", lambda _: False)(p)


def make_junction(link, target):
    if os.path.lexists(link):
        if not is_junction(link):
            raise RuntimeError(f"{link} exists and is not a junction")
        os.rmdir(link)  # removes only the junction, never the target
    r = subprocess.run(["cmd", "/c", "mklink", "/J", link, target], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isdir(link):
        raise RuntimeError(f"mklink /J failed: {r.stdout} {r.stderr}")


# ------------------------------------------------------------------------- minimal websocket client (stdlib only)
class WS:
    def __init__(self, port, path):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=15)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = self.s.recv(4096)
            if not d:
                raise ConnectionError("websocket handshake closed")
            buf += d
        head, self.buf = buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise ConnectionError("websocket upgrade refused: " + head.decode(errors="replace")[:200])
        self.s.settimeout(None)

    def _read(self, n):
        while len(self.buf) < n:
            d = self.s.recv(65536)
            if not d:
                raise EOFError
            self.buf += d
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self):  # -> (opcode, payload) of one complete message
        msg, op0 = b"", 1
        while True:
            b0, b1 = self._read(2)
            op, ln = b0 & 15, b1 & 127
            if ln == 126:
                ln = struct.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read(8))[0]
            data = self._read(ln)
            if op == 9:  # ping -> masked pong
                mask = os.urandom(4)
                self.s.sendall(bytes([0x8A, 0x80 | len(data)]) + mask + bytes(c ^ mask[i % 4] for i, c in enumerate(data)))
            elif op == 8:
                raise EOFError
            elif op in (1, 2):
                op0, msg = op, data
            else:
                msg += data
            if op in (0, 1, 2) and b0 & 0x80:
                return op0, msg

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


# ------------------------------------------------------------------------------ workflow (UI json) -> API json
def workflow_to_api(wf, variant, ov):
    """Convert the saved UI workflow to API format. Bypassed (mode 4) / muted (mode 2) nodes and Notes are dropped, links to them removed."""
    nodes = {n["id"]: n for n in wf["nodes"]}
    dropped = {i for i, n in nodes.items() if n.get("mode") in (2, 4) or n["type"] in ("Note", "MarkdownNote")}
    link = {l[0]: (l[1], l[2]) for l in wf["links"]}
    graph = {}
    for nid, n in nodes.items():
        if nid in dropped:
            continue
        ins = {}
        names = [i["name"] for i in n.get("inputs", []) if i.get("widget")] + EXTRA_WIDGETS.get(n["type"], [])
        for name, val in zip(names, n.get("widgets_values") or []):
            if name:
                ins[name] = val
        for i in n.get("inputs", []):
            if i.get("link") is not None:
                src, slot = link[i["link"]]
                if src not in dropped:
                    ins[i["name"]] = [str(src), slot]
        graph[str(nid)] = {"class_type": n["type"], "inputs": ins, "_meta": {"title": n.get("title") or n["type"]}}

    def find(cls):
        return [g for g in graph.values() if g["class_type"] == cls]

    if variant == "fast":
        (u,) = find("UNETLoader")
        u.update(class_type="H3TurboFastUNetLoader", inputs={"h3t_name": H3T, "precision": "a8", "resident_blocks": 0, "mlp_chunk": 0})
    (i2v,) = find("MiniMaxH3ImageToVideo")
    for k in ("prompt", "width", "height", "length"):
        if ov.get(k) is not None:
            i2v["inputs"][k] = ov[k]
    if ov.get("steps") is not None:
        find("BasicScheduler")[0]["inputs"]["steps"] = ov["steps"]
    if ov.get("seed") is not None:
        find("RandomNoise")[0]["inputs"]["noise_seed"] = ov["seed"]
    find("SaveVideo")[0]["inputs"]["filename_prefix"] = f"video/h3e2e_{variant}"
    return graph


# ------------------------------------------------------------------------------------------------------- timing
def report_timing(events, graph, out_json):
    ex = [(t, d["node"]) for t, ty, d in events if ty == "executing" and d.get("node")]
    done = next((t for t, ty, d in events if ty in ("execution_success", "execution_error")), None)
    print("\n== per-node timing (server-reported order; model loads happen lazily inside the node that first uses them) ==")
    rows = []
    for i, (t, n) in enumerate(ex):
        end = ex[i + 1][0] if i + 1 < len(ex) else done
        if end is not None:
            rows.append({"node": n, "class": graph[n]["class_type"], "sec": round(end - t, 2), "t_start": round(t, 2)})
            print(f"  #{n:>4} {graph[n]['class_type']:<26} {end - t:8.1f} s")
    info = {"per_node": rows, "total_wall_s": round(done, 2) if done is not None else None}
    for r in rows:
        if r["class"] == "SamplerCustomAdvanced":
            info["sampler_s"] = r["sec"]
            pts = [t for t, ty, d in events if ty == "progress" and d.get("node") == r["node"]]
            if pts:
                info["sampler_steps"] = len(pts)
                info["sampler_first_step_after_node_start_s"] = round(pts[0] - r["t_start"], 2)  # includes UNet load + first step
                if len(pts) > 1:
                    info["sampler_mean_step_s_after_first"] = round((pts[-1] - pts[0]) / (len(pts) - 1), 2)
    print("\n".join(f"  {k}: {v}" for k, v in info.items() if k != "per_node"))
    with open(out_json, "w") as f:
        json.dump(info, f, indent=1)
    return info


# ------------------------------------------------------------------------------------------------ run command
def cmd_run(a):
    out = os.path.abspath(a.out_dir)
    for d in ("output", "temp", "input", "user", "custom_nodes"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    with open(WORKFLOW, encoding="utf-8") as f:
        wf = json.load(f)
    graph = workflow_to_api(wf, a.variant, dict(prompt=a.prompt, width=a.width, height=a.height, length=a.length, steps=a.steps, seed=a.seed))
    with open(os.path.join(out, f"prompt_{a.variant}.json"), "w") as f:
        json.dump(graph, f, indent=1)

    whitelist = ["ComfyUI-GGUF"]  # the only custom node the workflow needs (CLIPLoaderGGUF); the rest of custom_nodes is not loaded
    port = free_port()
    cmd = [PY, "main.py", "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch", "--disable-api-nodes",
           "--output-directory", os.path.join(out, "output"), "--temp-directory", os.path.join(out, "temp"),
           "--input-directory", os.path.join(out, "input"), "--user-directory", os.path.join(out, "user"),
           "--database-url", "sqlite:///:memory:"]  # else ComfyUI renames the real user/comfyui.db under D:\ComfyUIBig to .bak (legacy-db migration)
    junction = None
    if a.variant == "fast":
        junction = os.path.join(out, "custom_nodes", "MiniMax-H3-Turbo")
        make_junction(junction, REPO)
        yml = os.path.join(out, "extra_model_paths.yaml")
        with open(yml, "w") as f:  # extra custom-node dir (folder_paths "custom_nodes" list); models stay in ComfyUI's own models dir
            f.write("h3e2e:\n  custom_nodes: '" + os.path.join(out, "custom_nodes").replace("\\", "/") + "'\n")
        cmd += ["--extra-model-paths-config", yml]
        whitelist.append("MiniMax-H3-Turbo")
    cmd += ["--disable-all-custom-nodes", "--whitelist-custom-nodes", *whitelist]

    if not a.no_gpu_wait:
        wait_gpu(a.gpu_busy_mib)
    logp = os.path.join(out, f"server_{a.variant}.log")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")  # no .pyc writes into the install
    log = open(logp, "wb")
    proc = subprocess.Popen(cmd, cwd=COMFY, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"[run] {a.variant}: server pid {proc.pid} port {port}, log {logp}", flush=True)
    events, stop, gpu_peak, ws, ok = [], threading.Event(), [0], None, False
    try:
        t0 = time.time()
        while True:  # wait for the server
            if proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited early with code {proc.returncode}")
            try:
                http(port, "/system_stats", timeout=5)
                break
            except (OSError, RuntimeError):
                if time.time() - t0 > a.startup_timeout:
                    raise TimeoutError("server did not come up")
                time.sleep(2)
        print(f"[run] server up after {time.time() - t0:.0f} s", flush=True)

        info = http(port, "/object_info", timeout=180)
        missing = sorted({g["class_type"] for g in graph.values()} - set(info))
        if missing:
            raise RuntimeError(f"node class(es) not registered in this ComfyUI: {missing}"
                               + ("  (the custom node failed to import; see the log tail below / grep 'H3-Turbo' in the server log)" if a.variant == "fast" else ""))
        loader = next(g for g in graph.values() if g["class_type"] in ("UNETLoader", "H3TurboFastUNetLoader"))
        spec = {**info[loader["class_type"]]["input"].get("optional", {}), **info[loader["class_type"]]["input"]["required"]}
        for name, val in loader["inputs"].items():
            s = spec.get(name) or [None]
            opts = s[0] if isinstance(s[0], list) else (s[1].get("options") if len(s) > 1 and isinstance(s[1], dict) else None)
            if isinstance(opts, list) and val not in opts:
                raise RuntimeError(f"{loader['class_type']}.{name}={val!r} not in server options {opts}")

        if a.check_nodes:
            print(f"[run] check-nodes OK: all {len(graph)} nodes registered, loader options valid ({loader['class_type']})")
            ok = True
            return 0
        cid = uuid.uuid4().hex
        ws = WS(port, f"/ws?clientId={cid}")
        t_sub = time.perf_counter()  # provisional (the server sends a status frame on connect); reset right before the POST

        def reader():
            try:
                while True:
                    op, data = ws.recv()
                    if op == 1:
                        m = json.loads(data)
                        events.append((time.perf_counter() - t_sub, m.get("type"), m.get("data") or {}))
            except Exception:
                pass

        def gpu_poll():
            while not stop.wait(2):
                gpu_peak[0] = max(gpu_peak[0], gpu_used_mib() or 0)

        threading.Thread(target=reader, daemon=True).start()
        threading.Thread(target=gpu_poll, daemon=True).start()
        t_sub = time.perf_counter()
        r = http(port, "/prompt", {"prompt": graph, "client_id": cid})
        if r.get("node_errors"):
            raise RuntimeError("node_errors: " + json.dumps(r["node_errors"])[:3000])
        pid = r["prompt_id"]
        print(f"[run] queued prompt {pid}", flush=True)
        hist, last_print = None, 0
        while hist is None:
            if proc.poll() is not None:
                raise RuntimeError(f"ComfyUI died mid-run (code {proc.returncode})")
            if time.perf_counter() - t_sub > a.timeout:
                http(port, "/interrupt", {})
                raise TimeoutError(f"prompt did not finish in {a.timeout}s")
            h = http(port, f"/history/{pid}")
            hist = h.get(pid)
            if hist is None:
                time.sleep(1)
                if time.time() - last_print > 60:
                    last_print = time.time()
                    prog = [d for _, ty, d in events if ty == "progress"]
                    print(f"[run] {time.perf_counter() - t_sub:6.0f} s  running; last progress {prog[-1] if prog else '-'}", flush=True)
        time.sleep(0.5)  # let the final ws frames land
        wall = time.perf_counter() - t_sub
        if hist["status"]["status_str"] != "success":
            raise RuntimeError("execution failed: " + json.dumps(hist["status"]["messages"])[-3000:])
        report_timing(events, graph, os.path.join(out, f"timing_{a.variant}.json"))
        print(f"  wall time as seen by the harness: {wall:.1f} s;  GPU memory peak (nvidia-smi total, incl. desktop): {gpu_peak[0]} MiB")
        vids = [os.path.join(out, "output", im.get("subfolder", ""), im["filename"]) for o in hist["outputs"].values() for im in o.get("images", [])]
        vids = [v for v in vids if v.lower().endswith((".mp4", ".webm", ".mkv"))]
        if not vids:
            raise RuntimeError("no video in outputs: " + json.dumps(hist["outputs"])[:1000])
        dst = os.path.join(out, f"result_{a.variant}.mp4")
        shutil.copyfile(vids[0], dst)
        print(f"[run] OK -> {dst} ({os.path.getsize(dst) / 1e6:.2f} MB)")
        ok = True
    except BaseException as e:
        print(f"[run] FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    finally:
        stop.set()
        if ws:
            ws.close()
        kill_tree(proc)
        log.close()
        if junction and is_junction(junction):
            os.rmdir(junction)
        if not ok:
            print(f"--- last 30 lines of {logp} ---\n{tail(logp)}", file=sys.stderr)
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------- compare command
def cmd_compare(a):
    import av
    import numpy as np

    def video(p):
        with av.open(p) as c:
            s = c.streams.video[0]
            meta = (s.codec_context.width, s.codec_context.height, float(s.average_rate or 0))
            for f in c.decode(video=0):
                yield meta, f.to_ndarray(format="rgb24")

    def audio(p):
        with av.open(p) as c:
            if not c.streams.audio:
                return None, np.zeros((0, 0), np.float32)
            s = c.streams.audio[0]
            arr = [f.to_ndarray().astype(np.float32) for f in c.decode(audio=0)]
            return (s.rate, s.channels), (np.concatenate(arr, axis=1) if arr else np.zeros((0, 0), np.float32))

    n, na, nb, meta_a, meta_b, mx, tot, sq, npx, bad = 0, 0, 0, None, None, 0, 0.0, 0.0, 0, []
    shape_ok = True
    for x, y in itertools.zip_longest(video(a.a), video(a.b)):
        na, nb = na + (x is not None), nb + (y is not None)
        meta_a, meta_b = x[0] if x else meta_a, y[0] if y else meta_b
        if x is None or y is None or x[1].shape != y[1].shape:
            shape_ok = shape_ok and (x is None or y is None)  # both present but different shape -> False
            continue
        d = np.abs(x[1].astype(np.int16) - y[1].astype(np.int16))
        m = int(d.max())
        if m > a.tol:
            bad.append(n)
        mx, tot, sq, npx, n = max(mx, m), tot + float(d.sum()), sq + float((d.astype(np.float64) ** 2).sum()), npx + d.size, n + 1
    (ra, aa), (rb, ab) = audio(a.a), audio(a.b)
    atol = a.audio_tol if a.audio_tol is not None else a.tol / 255.0
    audio_ok, amax, amean = True, 0.0, 0.0
    if aa.shape != ab.shape:
        audio_ok = False
    elif aa.size:
        d = np.abs(aa - ab)
        amax, amean = float(d.max()), float(d.mean())
        audio_ok = amax <= atol
    print(f"A: {a.a}\n   frames={na} res={meta_a[:2] if meta_a else None} fps={meta_a[2] if meta_a else None} audio(sr,ch)={ra} samples/ch={aa.shape[-1] if aa.size else 0}")
    print(f"B: {a.b}\n   frames={nb} res={meta_b[:2] if meta_b else None} fps={meta_b[2] if meta_b else None} audio(sr,ch)={rb} samples/ch={ab.shape[-1] if ab.size else 0}")
    vid_ok = shape_ok and na == nb and n > 0 and not bad
    mse = sq / max(npx, 1)
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"video: compared {n} frames  max|diff|={mx}  mean|diff|={tot / max(npx, 1):.6f}  PSNR={psnr:.2f} dB  "
          f"frames over tol({a.tol}): {len(bad)} {bad[:8]}{'...' if len(bad) > 8 else ''}  same_shape_and_count={shape_ok and na == nb}")
    print(f"audio: max|diff|={amax:.3e}  mean|diff|={amean:.3e}  same_shape={aa.shape == ab.shape}  (tol {atol:g})")
    ident = vid_ok and audio_ok and mx == 0 and amax == 0
    print("RESULT:", "IDENTICAL" if ident else ("WITHIN TOLERANCE" if vid_ok and audio_ok else "DIFFERENT"))
    return 0 if vid_ok and audio_ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the workflow through a headless ComfyUI")
    r.add_argument("--variant", choices=["native", "fast"], required=True)
    r.add_argument("--width", type=int, default=512)
    r.add_argument("--height", type=int, default=320)
    r.add_argument("--length", type=int, default=22, help="frames; snaps up to the 17k+5 grid (5, 22, 39, 56, 73, ..., 124)")
    r.add_argument("--steps", type=int, default=8)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--prompt", default=DEFAULT_PROMPT)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--timeout", type=int, default=7200, help="max seconds for the prompt itself")
    r.add_argument("--startup-timeout", type=int, default=900)
    r.add_argument("--gpu-busy-mib", type=int, default=2500, help="wait (1-min polls, 15 min max) while total GPU memory in use exceeds this")
    r.add_argument("--no-gpu-wait", action="store_true")
    r.add_argument("--check-nodes", action="store_true", help="start the server, verify every node class/loader option is registered, exit (no GPU work)")
    c = sub.add_parser("compare", help="decode two mp4 files and diff video + audio")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--tol", type=float, default=0, help="max allowed abs pixel diff (0-255 scale)")
    c.add_argument("--audio-tol", type=float, default=None, help="max allowed abs sample diff (float, default tol/255)")
    a = ap.parse_args()
    sys.exit(cmd_run(a) if a.cmd == "run" else cmd_compare(a))


if __name__ == "__main__":
    main()
