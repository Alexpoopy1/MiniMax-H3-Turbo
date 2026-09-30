"""GPU check for the h3t streaming engine (run with the ComfyUI venv python; never touches the real checkpoint).

Builds a SYNTHETIC store with real-size blocks (~219 MB: hidden 5376, W4A8 shapes), then measures and verifies:
  1. raw host->device throughput of one block for each host source (page-locked file mapping, page-locked copy,
     pageable mapping),
  2. bytes on the device == bytes in the file, for every block, after streaming,
  3. per-block kernel ordering: a fingerprint of each slot is read by the LAST kernel of a fake compute block, so a
     copy that overwrote a live slot early would show up as a wrong fingerprint,
  4. overlap efficiency: streaming with a fake matmul workload vs compute alone vs copies alone,
  5. abort mid-forward and recovery.

    set PYTHONPATH=C:\\Users\\Alexp\\MiniMax-H3-Turbo
    D:\\ComfyUIBig\\ComfyUI\\ComfyUI\\.venv\\Scripts\\python.exe scripts\\h3_check_stream_gpu.py [--blocks 6] [--compute-ms 70]

Stays inside ~1.2 GB of VRAM (--vram-limit-mb) and ~1.5 GB of RAM with the defaults.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from h3turbo.h3.config import H3Config  # noqa: E402
from h3turbo.h3.convert import synthetic_state_dict  # noqa: E402
from h3turbo.h3.store import H3TFile, write_h3t  # noqa: E402
from h3turbo.h3.stream import StreamingProvider, free_ram_bytes  # noqa: E402

MB = 1 << 20


def sync() -> None:
    torch.cuda.synchronize()


def fingerprint(w) -> torch.Tensor:
    """Small, position-spread sample of the block's tensors as int32 (identical CPU/GPU computation)."""
    parts = [w.qkv.q.reshape(-1)[::131071], w.out.q.reshape(-1)[::65537], w.fc1.q.reshape(-1)[::262139], w.fc2.q.reshape(-1)[::65521],
             w.fc1.s_rel.reshape(-1)[::9973].contiguous().view(torch.int8), w.adaln_w.reshape(-1)[::4093].contiguous().view(torch.int16)]
    return torch.cat([p.to(torch.int32) for p in parts])


def build_store(path: str, blocks: int) -> H3TFile:
    cfg = H3Config(layers=blocks, refiner_layers=0)
    sd = synthetic_state_dict(cfg, quant=True, seed=3, refiner=False, lazy=True)
    t0 = time.time()
    info = write_h3t(path, cfg, sd, chunk_bytes=16 << 20)
    dt = time.time() - t0
    print(f"synthetic store: {info['file_size'] / MB:.0f} MiB, {blocks} blocks x {info['block_nbytes'] / MB:.2f} MiB, written in {dt:.1f}s ({info['file_size'] / MB / dt:.0f} MiB/s incl. random generation)")
    return H3TFile(path)


def raw_throughput(st: H3TFile, dev: torch.device) -> None:
    print("\n[1] raw host->device throughput, one block per copy")
    blk = st.block_tensor(1)
    dst = torch.empty(st.block_nbytes, dtype=torch.uint8, device=dev)
    rt = torch.cuda.cudart()

    def gbps(src, nb, reps=6):
        dst.copy_(src, non_blocking=nb)
        sync()
        t = time.perf_counter()
        for _ in range(reps):
            dst.copy_(src, non_blocking=nb)
        sync()
        return reps * st.block_nbytes / (time.perf_counter() - t) / 1e9

    print(f"    pageable mapping           {gbps(blk, False):6.2f} GB/s")
    t = time.perf_counter()
    ok = int(rt.cudaHostRegister(blk.data_ptr(), blk.numel(), 8)) == 0
    reg_s = time.perf_counter() - t
    if ok:
        print(f"    page-locked mapping (RO)   {gbps(blk, True):6.2f} GB/s   (register took {reg_s * 1e3:.0f} ms)")
        rt.cudaHostUnregister(blk.data_ptr())
    else:
        print("    page-locked mapping (RO)   not supported here")
    buf = torch.empty(st.block_nbytes, dtype=torch.uint8)
    assert int(rt.cudaHostRegister(buf.data_ptr(), buf.numel(), 0)) == 0
    buf.copy_(blk)
    print(f"    page-locked copy           {gbps(buf, True):6.2f} GB/s")
    rt.cudaHostUnregister(buf.data_ptr())
    del dst
    torch.cuda.empty_cache()


def run_forwards(p, n, compute, forwards):
    times = []
    for _ in range(forwards):
        sync()
        t = time.perf_counter()
        p.begin_forward()
        try:
            for i in range(n):
                w = p.acquire(i)
                compute(i, w)
                p.release(i)
        finally:
            p.end_forward()
        sync()
        times.append(time.perf_counter() - t)
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--dir", default=r"D:\h3turbo_scratch")
    ap.add_argument("--compute-ms", type=float, default=70.0, help="fake compute per block (real model: ~70 ms at 1768 tokens)")
    ap.add_argument("--forwards", type=int, default=4)
    ap.add_argument("--resident", type=int, default=1)
    ap.add_argument("--ring", type=int, default=3)
    ap.add_argument("--vram-limit-mb", type=int, default=1500)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required")
        return 2
    dev = torch.device("cuda", torch.cuda.current_device())
    blk_mb = 218.96
    need = (a.resident + min(a.ring, a.blocks - a.resident)) * blk_mb + 220
    if need > a.vram_limit_mb:
        print(f"refusing: needs ~{need:.0f} MiB VRAM > --vram-limit-mb {a.vram_limit_mb}")
        return 2
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(dev)}, free VRAM {torch.cuda.mem_get_info(dev)[0] / MB:.0f} MiB, free RAM {free_ram_bytes() / MB:.0f} MiB")
    print("note: the GPU is shared with other processes; timings are indicative, byte/order checks are exact")
    made_dir = not os.path.isdir(a.dir)
    os.makedirs(a.dir, exist_ok=True)
    path = os.path.join(a.dir, "synthetic_stream_check.h3t")
    ok = True
    st = build_store(path, a.blocks)
    try:
        raw_throughput(st, dev)
        n = st.n_blocks
        expect = [fingerprint(st.block_weights_cpu(i)) for i in range(n)]

        # fake compute: matmuls sized to ~compute_ms, then a fingerprint read of the slot as the LAST kernel of the block
        x = torch.randn(1024, 5376, device=dev, dtype=torch.bfloat16)
        wm = torch.randn(5376, 7168, device=dev, dtype=torch.bfloat16)
        out = torch.empty(1024, 7168, device=dev, dtype=torch.bfloat16)
        for _ in range(20):  # warm up clocks before calibrating
            torch.mm(x, wm, out=out)
        sync()
        cal = []
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(10):
                torch.mm(x, wm, out=out)
            sync()
            cal.append((time.perf_counter() - t) / 10 * 1e3)
        t_mm = min(cal)
        results = {}

        def make_compute(ms):
            k = max(1, round(ms / t_mm))
            fps = {}

            def compute(i, w):
                for _ in range(k):
                    torch.mm(x, wm, out=out)
                fps[i] = fingerprint(w)  # reads the slot after the busy work

            return compute, fps, k

        def check_fps(fps):
            bad = [i for i in range(n) if not torch.equal(fps[i].cpu(), expect[i])]
            return bad

        def compute_alone(k):
            sync()
            t = time.perf_counter()
            for _ in range(n):
                for _ in range(k):
                    torch.mm(x, wm, out=out)
            sync()
            return time.perf_counter() - t

        print(f"\n[2-4] fake compute: one 1024x5376x7168 bf16 matmul = {t_mm:.1f} ms; {a.blocks} blocks, resident {a.resident}, ring {a.ring}, prefetch 2")
        print(f"      each cell is the minimum of {a.forwards} interleaved rounds (compute alone / copies alone / both); the GPU is shared")
        for label, ms in (("compute-bound", a.compute_ms), ("copy-bound", max(5.0, a.compute_ms / 6))):
            compute, fps, k = make_compute(ms)
            print(f"\n  -- {label}: {k} matmuls/block = {k * t_mm:.0f} ms of compute per block")
            print(f"     {'host source':<22}{'construct':>10}{'compute':>10}{'copies':>9}{'both':>8}{'efficiency':>12}{'copy hidden':>13}  order/bytes")
            for name, kw in (("page-locked mapping", dict(pin="auto", pin_blocks=99)), ("page-locked copies", dict(pin=True, pin_mode="copy")),
                             ("staging ring", dict(pin="auto", pin_mode="stage")), ("pageable mapping", dict(pin=False))):
                t0 = time.time()
                with StreamingProvider(st, dev, resident=a.resident, prefetch=2, ring=a.ring, **kw) as p:
                    t_build = time.time() - t0
                    kinds = p.stats()["host_kinds"]
                    p.wait_prefetch()
                    run_forwards(p, n, compute, 1)  # warm-up
                    tc, tm, tb = [], [], []
                    for _ in range(a.forwards):
                        tc.append(compute_alone(k))
                        tm.append(run_forwards(p, n, lambda i, w: None, 1)[0])  # copies with no compute to hide behind
                        tb.append(run_forwards(p, n, compute, 1)[0])
                    t_comp, t_copy, t_ov = min(tc), min(tm), min(tb)
                    bad = check_fps(fps)
                    eff = max(t_comp, t_copy) / t_ov
                    hid = (t_comp + t_copy - t_ov) / min(t_comp, t_copy) if min(t_comp, t_copy) > 0.02 else float("nan")
                    hidden = "%.0f%%" % (hid * 100) if hid == hid else "n/a"
                    good = "ok" if not bad else "BAD blocks %s" % bad
                    ok &= not bad
                    print(f"     {name:<22}{t_build:>9.2f}s{t_comp * 1e3:>8.0f}ms{t_copy * 1e3:>7.0f}ms{t_ov * 1e3:>6.0f}ms{eff:>11.0%}{hidden:>13}   {good}  {kinds}")
                    results[(label, name)] = (t_comp, t_copy, t_ov)

        print("\n[5] bytes on device == bytes in the file, every block (D2H compare)")
        with StreamingProvider(st, dev, resident=a.resident, prefetch=2, ring=a.ring) as p:
            for rnd in range(2):
                p.begin_forward()
                for i in range(n):
                    w = p.acquire(i)
                    buf = p.buffer_of(i)
                    sync()
                    same = np.array_equal(buf.cpu().numpy(), st.block_bytes(i))
                    same &= bool(torch.equal(fingerprint(w).cpu(), expect[i]))
                    ok &= same
                    if not same:
                        print(f"    MISMATCH block {i} (round {rnd})")
                    p.release(i)
                p.end_forward()
            print(f"    {2 * n} block checks done; peak VRAM allocated {torch.cuda.max_memory_allocated() / MB:.0f} MiB")

            print("\n[6] abort mid-forward, then recover")
            try:
                p.begin_forward()
                for i in range(n):
                    p.acquire(i)
                    if i == n - 2:
                        raise KeyboardInterrupt("simulated abort")
                    p.release(i)
            except KeyboardInterrupt:
                pass
            finally:
                p.end_forward()
            compute, fps, _ = make_compute(5.0)
            run_forwards(p, n, compute, 2)
            bad = check_fps(fps)
            ok &= not bad
            print(f"    recovered: {'ok' if not bad else f'BAD blocks {bad}'}; stats {p.stats()}")
    finally:
        st.close()
        if not a.keep:
            try:
                os.remove(path)
                if made_dir:
                    shutil.rmtree(a.dir, ignore_errors=True)
            except OSError as e:
                print(f"could not delete {path}: {e}")
        else:
            print(f"kept {path}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
