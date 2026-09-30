"""Block streaming engine: slot ring, prefetch, hazards, error paths. CPU by default; the CUDA tests run when a GPU is present."""
import dataclasses
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from h3turbo.h3 import convert as cv
from h3turbo.h3.config import H3Config
from h3turbo.h3.store import H3TFile
from h3turbo.h3.stream import ACQUIRED, PENDING, StreamingProvider, free_ram_bytes, plan_pinned_blocks, plan_residency
from h3turbo.h3.types import BlockWeights, W4A8Weight

LAYERS = 6
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny_cfg(layers: int = LAYERS) -> H3Config:
    return H3Config(hidden=64, layers=layers, refiner_layers=1, heads=2, head_dim=32, ffn=48, video_channels=4, audio_channels=8,
                    text_dim=32, t_dim=8, curve_grid=17, rope_inv_freq_len=16, quant_group=16, quant_convrot=16)


def raw(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensors_of(w: BlockWeights):
    out = []
    for f in dataclasses.fields(w):
        v = getattr(w, f.name)
        out += [v.q, v.s_rel, v.s_ch, v.codebook] if isinstance(v, W4A8Weight) else [v]
    return out


def same(a: BlockWeights, b: BlockWeights) -> bool:
    for x, y in zip(tensors_of(a), tensors_of(b)):
        if x is None or y is None:
            if x is not y:
                return False
        elif (x.dtype, tuple(x.shape)) != (y.dtype, tuple(y.shape)) or raw(x) != raw(y):
            return False
    return True


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    d = tmp_path_factory.mktemp("h3t")
    path = str(d / "m.h3t")
    cv.convert(cv.synthetic_state_dict(tiny_cfg(), quant=True, seed=7), path, verify=False)
    st = H3TFile(path)
    yield st
    st.close()


def forward(p, store, compute=None, check=True):
    p.begin_forward()
    try:
        for i in range(p.n_blocks):
            w = p.acquire(i)
            if check:
                assert same(w, store.block_weights_cpu(i)), f"block {i} differs from the file"
            if compute:
                compute(i, w)
            p.release(i)
    finally:
        p.end_forward()


def no_slot_held(p):
    return all(s.state != ACQUIRED for s in p._slots)


# ------------------------------------------------------------------ planning
def test_plan_residency():
    st = SimpleNamespace(block_nbytes=100, n_blocks=50)
    assert plan_residency(st, 5000, 0, 3) == 50  # all resident: no ring needed
    assert plan_residency(st, 4999, 0, 3) == 46  # 46 resident + 3 ring slots = 49 blocks <= 49.99
    assert plan_residency(st, 5000, 1000, 3) == 37  # 37 + 3 = 40 blocks <= 40
    assert plan_residency(st, 5000, 0, 1) == 50 and plan_residency(st, 4999, 0, 1) == 48
    assert plan_residency(st, 250, 0, 3) == 0  # not even the ring fits: 0, the allocation reports the OOM
    assert plan_residency(st, 100_000, 0, 3) == 50
    two = SimpleNamespace(block_nbytes=100, n_blocks=2)
    assert plan_residency(two, 150, 0, 3) == 0  # 2 blocks of ring do not fit
    assert plan_residency(two, 200, 0, 3) == 2  # a remainder that fits the ring is better kept resident: same memory
    assert plan_residency(SimpleNamespace(block_nbytes=100, n_blocks=10), 700, 0, 3) == 4  # 4 resident + 3 slots
    with pytest.raises(ValueError):
        plan_residency(st, 1, 0, 0)


def test_plan_pinned_blocks_respects_free_ram():
    st = SimpleNamespace(block_nbytes=100)
    assert plan_pinned_blocks(st, 10, 1000, 0.5, 0) == 3  # budget min(500, 1000) - 2 staging buffers = 300
    assert plan_pinned_blocks(st, 2, 1000, 0.5, 0) == 2  # never more than the streamed blocks
    assert plan_pinned_blocks(st, 10, 1000, 0.5, 900) == 0  # headroom leaves nothing
    assert plan_pinned_blocks(st, 10, 100, 0.5, 0) == 0
    assert free_ram_bytes() > 0


# ------------------------------------------------------------------ equality with the file
@pytest.mark.parametrize("resident", [0, 1, 3, 6])
@pytest.mark.parametrize("ring,prefetch", [(1, 0), (1, 2), (2, 1), (3, 2), (4, 5)])
def test_streaming_equals_resident_over_many_forwards(store, resident, ring, prefetch):
    with StreamingProvider(store, "cpu", resident=resident, prefetch=prefetch, pin=False, ring=ring) as p:
        assert p.stats()["resident"] == resident and p.stats()["slots"] == min(ring, LAYERS - resident)
        for _ in range(3):
            forward(p, store)
            assert no_slot_held(p)


@pytest.mark.parametrize("kwargs,kinds", [
    (dict(pin=False), {"pageable": 4}),
    (dict(pin=True), {"pinned": 4}),
    (dict(pin="auto", pin_blocks=1), {"pinned": 1, "staged": 3}),
    (dict(pin="auto", pin_blocks=9), {"pinned": 4}),
    (dict(pin="auto", pin_mode="stage"), {"staged": 4}),
    (dict(pin="auto"), {"pageable": 4}),  # nothing to page-lock on CPU
])
def test_host_source_modes_are_all_exact(store, kwargs, kinds):
    with StreamingProvider(store, "cpu", resident=2, prefetch=2, ring=3, **kwargs) as p:
        assert p.stats()["host_kinds"] == kinds
        for _ in range(2):
            forward(p, store)


def test_device_bytes_equal_file_bytes(store):
    with StreamingProvider(store, "cpu", resident=2, prefetch=2, ring=3, pin=True) as p:
        forward(p, store)
        p.wait_prefetch()
        seen = 0
        for i in range(LAYERS):
            buf = p.buffer_of(i)
            if buf is not None:
                seen += 1
                assert buf.dtype == torch.uint8 and raw(buf) == bytes(store.block_bytes(i)), i
        assert seen >= 2 + 2  # the resident blocks plus at least the prefetched window


def test_cross_forward_prefetch_warms_the_next_forward(store):
    with StreamingProvider(store, "cpu", resident=2, prefetch=2, ring=3, pin=True) as p:
        forward(p, store)
        p.wait_prefetch()
        assert p.buffer_of(2) is not None and p.buffer_of(3) is not None  # first streamed blocks are already loaded
        c0 = p.stats()["copies"]
        forward(p, store)
        p.wait_prefetch()
        assert p.stats()["copies"] - c0 == LAYERS - 2  # each streamed block copied exactly once per forward, no re-copies


def test_ring_that_holds_every_streamed_block_copies_once(store):
    with StreamingProvider(store, "cpu", resident=3, prefetch=2, ring=3, pin=True) as p:
        for _ in range(4):
            forward(p, store)
        p.wait_prefetch()
        assert p.stats()["copies"] == 3  # working set fits the ring: later forwards are cache hits


def test_resident_auto_uses_the_plan(store):
    blk = store.block_nbytes
    for free_blocks, ring in ((4.5, 3), (100, 3), (2.5, 2)):
        want = plan_residency(store, int(free_blocks * blk), 0, ring)
        with StreamingProvider(store, "cpu", resident="auto", free_vram_bytes=int(free_blocks * blk), reserve_bytes=0, ring=ring, pin=False) as p:
            assert p.stats()["resident"] == want
            forward(p, store)


# ------------------------------------------------------------------ hazards and ordering
def test_no_slot_is_written_while_the_model_holds_it(store):
    holder = {}
    log = []
    ready = threading.Event()  # the worker starts prefetching inside the constructor, before `p` exists

    def hook(dst, src, nb):
        assert ready.wait(10)
        slot = next(s for s in holder["p"]._slots if s.buf is dst)
        assert slot.state == PENDING, f"copy into a slot in state {slot.state}"  # a held slot must never be a copy target
        log.append(slot.idx)
        time.sleep(0.003)
        dst.copy_(src)

    with StreamingProvider(store, "cpu", resident=0, prefetch=3, ring=3, pin=False, transfer=hook) as p:
        holder["p"] = p
        ready.set()
        p.begin_forward()
        held = [p.acquire(0), p.acquire(1)]
        expect = [raw(t) for w in held for t in tensors_of(w)]
        time.sleep(0.15)  # the worker is free to prefetch; it must use the third slot only
        assert [raw(t) for w in held for t in tensors_of(w)] == expect
        assert sum(s.state == ACQUIRED for s in p._slots) == 2
        with pytest.raises(RuntimeError, match="already acquired"):
            p.acquire(0)
        third = p.acquire(2)
        with pytest.raises(RuntimeError, match="ring slots are held"):
            p.acquire(3)  # ring=3 and all three are held: a clear error, not a hang
        for i in (0, 1, 2):
            p.release(i)
        assert same(third, store.block_weights_cpu(2))
        for i in range(3, LAYERS):
            assert same(p.acquire(i), store.block_weights_cpu(i))
            p.release(i)
        p.end_forward()
    assert len(log) >= LAYERS


def test_acquire_waits_for_a_slow_copy(store):
    def slow(dst, src, nb):
        time.sleep(0.05)
        dst.copy_(src)

    with StreamingProvider(store, "cpu", resident=0, prefetch=1, ring=2, pin=False, transfer=slow) as p:
        t0 = time.perf_counter()
        forward(p, store)  # `same` inside proves acquire never returned before its bytes were complete
        assert time.perf_counter() - t0 >= 0.05 * LAYERS * 0.9
        assert p.stats()["waits"] >= 1 and p.stats()["wait_s"] > 0


def test_copies_overlap_compute(store):
    def slow(dst, src, nb):
        time.sleep(0.04)
        dst.copy_(src)

    def compute(i, w):
        time.sleep(0.04)

    def timed(**kw):
        with StreamingProvider(store, "cpu", resident=0, pin=False, transfer=slow, **kw) as p:
            forward(p, store, compute, check=False)
            p.wait_prefetch()
            t0 = time.perf_counter()
            forward(p, store, compute, check=False)
            return time.perf_counter() - t0

    serial = timed(prefetch=0, ring=1)  # every acquire waits for its own copy, then computes
    overlapped = timed(prefetch=2, ring=3)
    assert serial > 0.04 * 2 * LAYERS * 0.9
    assert overlapped < 0.8 * serial, (overlapped, serial)


@pytest.mark.parametrize("call_end", [True, False])
def test_exception_mid_forward_resets_state(store, call_end):
    with StreamingProvider(store, "cpu", resident=1, prefetch=2, ring=3, pin=True) as p:
        forward(p, store)
        for stop_at in (0, 1, 4):  # before any streamed block, after the first, deep in the ring
            with pytest.raises(ZeroDivisionError):
                p.begin_forward()
                for i in range(LAYERS):
                    p.acquire(i)
                    if i == stop_at:
                        1 / 0  # the block is held and never released
                    p.release(i)
            if call_end:
                p.end_forward()
                assert no_slot_held(p)
            forward(p, store)  # begin_forward recovers even when end_forward was skipped
            assert no_slot_held(p)
        p.wait_prefetch()


def test_out_of_order_and_repeated_access(store):
    with StreamingProvider(store, "cpu", resident=1, prefetch=2, ring=2, pin=False) as p:
        p.begin_forward()
        for i in (5, 1, 1, 4, 2, 5, 3, 3):
            assert same(p.acquire(i), store.block_weights_cpu(i)), i
            p.release(i)
        with pytest.raises(RuntimeError, match="not acquired"):
            p.release(2)
        with pytest.raises(IndexError):
            p.acquire(LAYERS)
        p.end_forward()
        forward(p, store)


def test_worker_error_surfaces_and_provider_still_closes(store):
    n = {"calls": 0}

    def flaky(dst, src, nb):
        n["calls"] += 1
        if n["calls"] == 3:
            raise OSError("disk fell off")
        dst.copy_(src)

    p = StreamingProvider(store, "cpu", resident=0, prefetch=1, ring=2, pin=False, transfer=flaky)
    with pytest.raises(RuntimeError, match="disk fell off"):
        for _ in range(3):
            forward(p, store, check=False)
    p.close()
    assert not p._thread.is_alive()


def test_wait_timeout_reports_state(store):
    gate = threading.Event()

    def stuck(dst, src, nb):
        gate.wait(10)
        dst.copy_(src)

    p = StreamingProvider(store, "cpu", resident=0, prefetch=1, ring=2, pin=False, transfer=stuck, wait_timeout=0.3)
    try:
        p.begin_forward()
        with pytest.raises(TimeoutError, match="block 0"):
            p.acquire(0)
    finally:
        gate.set()
        p.close()


def test_close_is_idempotent_joins_and_blocks_further_use(store):
    p = StreamingProvider(store, "cpu", resident=1, prefetch=2, ring=2, pin=True)
    forward(p, store)
    th = p._thread
    p.close()
    p.close()
    assert not th.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        p.acquire(3)
    with pytest.raises(RuntimeError, match="closed"):
        p.acquire(0)


def test_all_resident_needs_no_worker(store):
    with StreamingProvider(store, "cpu", resident=LAYERS) as p:
        assert p._thread is None and p.stats()["slots"] == 0
        forward(p, store)
        forward(p, store)


def test_stats_counters(store):
    with StreamingProvider(store, "cpu", resident=2, prefetch=2, ring=3, pin=True) as p:
        p.reset_stats()
        forward(p, store)
        p.wait_prefetch()
        s = p.stats()
        assert s["acquires"] == LAYERS - 2 and s["bytes"] == s["copies"] * store.block_nbytes and s["copies"] >= LAYERS - 2
        assert s["gpu_bytes"] == (2 + 3) * store.block_nbytes


def test_constructor_validation(store):
    for kw in (dict(ring=0), dict(prefetch=-1), dict(pin="maybe"), dict(pin_mode="x"), dict(resident=-1), dict(resident="some")):
        with pytest.raises(ValueError):
            StreamingProvider(store, "cpu", **kw)


# ------------------------------------------------------------------ CUDA (skipped without a GPU; run via the ComfyUI venv)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("resident,kw", [
    (0, dict(pin="auto", pin_blocks=99)),  # page-lock the file mapping in place
    (2, dict(pin=True, pin_mode="copy")),  # page-locked copies
    (1, dict(pin="auto", pin_mode="stage")),  # staging ring
    (0, dict(pin=False)),  # straight from the mapping
    (LAYERS, dict()),
])
def test_cuda_streaming_matches_file_and_orders_kernels(store, resident, kw):
    dev = torch.device("cuda")
    with StreamingProvider(store, dev, resident=resident, prefetch=2, ring=3, **kw) as p:
        want = {i: sum(int(t.to(torch.int64).sum()) for t in (store.block_weights_cpu(i).qkv.q, store.block_weights_cpu(i).fc2.q)) for i in range(LAYERS)}
        big = torch.randn(1024, 1024, device=dev)
        for _ in range(4):
            sums = []

            def compute(i, w):
                x = big
                for _ in range(20):  # keep the GPU busy so the host runs ahead of the device
                    x = torch.tanh(x @ big)
                sums.append((w.qkv.q.to(torch.int64).sum() + w.fc2.q.to(torch.int64).sum(), x))  # reads the slot AFTER the busy work

            forward(p, store, compute, check=False)
            got = [int(s) for s, _ in sums]
            assert got == [want[i] for i in range(LAYERS)]
        torch.cuda.synchronize()
        for i in range(LAYERS):
            buf = p.buffer_of(i)
            if buf is not None:
                assert raw(buf) == bytes(store.block_bytes(i))
        if resident < LAYERS:
            kinds = p.stats()["host_kinds"]
            assert sum(kinds.values()) == LAYERS - resident
