"""Block streaming for the official H3 DiT: ONE contiguous host->device copy per block into a ring of device slots.

Design (measured on the RTX 3050 6 GB, PCIe 3.0 x8: ~6.6 GB/s, i.e. ~33 ms per 219 MB block, against 55-70 ms of
compute per block. With real H3 forwards no acquire() ever waited (stats()["waits"] == 0); on synthetic compute
82-95% of the copy time was hidden):

* The first `resident` blocks live on the device for good; the rest ("streamed") cycle through `ring` slots. A
  slot is one uint8 buffer of `block_nbytes`; the typed tensors of a `BlockWeights` are zero-copy views of it.
* A worker thread issues the copies. Reuse of a slot is ordered on the device, not by blocking the host: the
  copy stream waits on the slot's `done` event (recorded by release() on the compute stream) and the compute
  stream waits on the slot's `ready` event in acquire(). The model thread only waits for a copy to be *issued*,
  so the CPU can run ahead of the GPU without ever overwriting weights a queued kernel still reads.
* The lookahead window is cyclic: after block N-1 the worker already prefetches the first streamed blocks of
  the next forward while the last blocks still compute (and while the resident blocks 0..R-1 compute).
* Host side, per streamed block: page-lock the file mapping in place (`cudaHostRegister` read-only: zero copy,
  no second RAM copy), or keep a page-locked copy, or stage through two page-locked buffers filled from the
  mapping by the worker, or (pin=False) copy straight from the mapping. `pin="auto"` bounds all of this by free RAM.

CPU mode (device="cpu") runs the identical state machine with plain memory copies and no events, so tests
exercise the slot bookkeeping, ordering and error paths without CUDA.
"""
from __future__ import annotations

import atexit
import os
import threading
import time
import weakref
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from .store import H3TFile
from .types import BlockWeights

_GIB = 1 << 30
_HOST_REGISTER_READ_ONLY = 8  # cudaHostRegisterReadOnly: needed to page-lock a read-only file mapping
FREE, PENDING, READY, ACQUIRED, IDLE = range(5)


def free_ram_bytes() -> int:
    """Physical memory available without swapping (includes reclaimable file cache)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    raise RuntimeError("cannot determine free RAM (install psutil, or pass pin_blocks explicitly)")


def plan_residency(store: H3TFile, free_vram_bytes: int, reserve_bytes: int = 0, ring: int = 3) -> int:
    """How many leading blocks to keep on the device: the most that still leaves room for the streaming ring
    (`min(ring, streamed)` slots) inside `free_vram_bytes - reserve_bytes`. All blocks resident needs no ring.
    Returns 0 when even the ring does not fit (the caller then gets a clear OOM when slots are allocated)."""
    if ring < 1:
        raise ValueError("ring must be >= 1")
    avail, blk, n = free_vram_bytes - reserve_bytes, store.block_nbytes, store.n_blocks
    for r in range(n, -1, -1):
        if (r + min(ring, n - r)) * blk <= avail:
            return r
    return 0


def plan_pinned_blocks(store: H3TFile, streamed: int, free_ram: int, pin_fraction: float = 0.5, headroom: int = 4 * _GIB) -> int:
    """How many streamed blocks to page-lock in host RAM: at most `pin_fraction` of the free RAM, and never
    dipping below `headroom` bytes of free RAM (two staging buffers are reserved out of the budget)."""
    budget = min(free_ram * pin_fraction, free_ram - headroom) - 2 * store.block_nbytes
    return int(max(0, min(streamed, budget // store.block_nbytes)))


_LIVE: "weakref.WeakSet[StreamingProvider]" = weakref.WeakSet()


@atexit.register
def _close_all() -> None:  # a forgotten close() must not leave the worker thread inside CUDA at interpreter exit
    for p in list(_LIVE):
        try:
            p.close()
        except Exception:
            pass


def _cuda_ok(r) -> bool:
    return int(r) == 0  # torch's cudaError enum does not compare with ints


def _clear_cuda_error(device: torch.device) -> None:
    """A failed cudart call leaves a sticky last-error that the next kernel launch would raise; consume it."""
    for _ in range(2):
        try:
            torch.zeros(1, device=device).add_(1)
            torch.cuda.synchronize(device)
        except Exception:
            pass


class _Slot:
    __slots__ = ("idx", "buf", "weights", "state", "block", "ready", "done", "stamp")

    def __init__(self, idx: int, buf: torch.Tensor, weights: BlockWeights):
        self.idx, self.buf, self.weights = idx, buf, weights
        self.state, self.block, self.ready, self.done, self.stamp = FREE, None, None, None, 0


class StreamingProvider:
    """BlockProvider over an H3TFile: resident blocks + a prefetching ring for the rest. Use as a context
    manager or call close(); the worker thread and page-locked host memory live until then."""

    def __init__(
        self,
        store: H3TFile,
        device: Union[str, torch.device] = "cuda",
        resident: Union[int, str] = "auto",
        prefetch: int = 2,
        pin: Union[bool, str] = "auto",
        ring: int = 3,
        *,
        reserve_bytes: int = 1536 << 20,
        free_vram_bytes: Optional[int] = None,
        pin_fraction: float = 0.5,
        ram_headroom_bytes: int = 4 * _GIB,
        pin_blocks: Optional[int] = None,
        pin_mode: str = "auto",
        transfer: Optional[Callable[[torch.Tensor, torch.Tensor, bool], None]] = None,
        wait_timeout: float = 600.0,
    ):
        if ring < 1 or prefetch < 0 or pin not in (True, False, "auto") or pin_mode not in ("auto", "register", "copy", "stage"):
            raise ValueError("ring >= 1, prefetch >= 0, pin in (True, False, 'auto'), pin_mode in ('auto', 'register', 'copy', 'stage')")
        if not (resident == "auto" or (isinstance(resident, int) and resident >= 0)):
            raise ValueError("resident must be 'auto' or a non-negative int")
        dev = torch.device(device)
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.store, self.device, self._cuda = store, dev, dev.type == "cuda"
        self.n_blocks, self._prefetch, self._timeout = store.n_blocks, prefetch, wait_timeout
        self._transfer = transfer or (lambda dst, src, nb: dst.copy_(src, non_blocking=nb))
        self._cv = threading.Condition()
        self._slots: List[_Slot] = []
        self._by_block: Dict[int, _Slot] = {}
        self._resident: Dict[int, tuple] = {}
        self._kind: Dict[int, str] = {}
        self._host: Dict[int, torch.Tensor] = {}
        self._staging: List[torch.Tensor] = []
        self._stage_ev: List[object] = []
        self._stage_next = 0
        self._reg_ptrs: List[int] = []
        self._streamed: List[int] = []
        self._pos, self._demand, self._tick = -1, None, 0
        self._stop = self._closed = self._in_forward = False
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._st = dict(acquires=0, waits=0, wait_s=0.0, copies=0, bytes=0, busy_s=0.0)
        self._copy_stream = torch.cuda.Stream(dev) if self._cuda else None
        _LIVE.add(self)
        try:
            self._build(resident, ring, pin, pin_mode, reserve_bytes, free_vram_bytes, pin_fraction, ram_headroom_bytes, pin_blocks)
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ construction
    def _alloc(self) -> torch.Tensor:
        return torch.empty(self.store.block_nbytes, dtype=torch.uint8, device=self.device)

    def _build(self, resident, ring, pin, pin_mode, reserve, free_vram, pin_fraction, headroom, pin_blocks) -> None:
        st, n = self.store, self.store.n_blocks
        auto = resident == "auto"
        if auto:
            free = free_vram if free_vram is not None else (torch.cuda.mem_get_info(self.device)[0] if self._cuda else free_ram_bytes() // 2)
            resident = plan_residency(st, free, reserve, ring)
        for b in range(min(int(resident), n)):
            try:
                buf = self._alloc()
            except torch.cuda.OutOfMemoryError:
                if auto:
                    break  # the plan was optimistic; the rest streams
                raise
            buf.copy_(st.block_tensor(b))  # synchronous; the mapping is the source of truth
            self._resident[b] = (buf, st.block_weights_from(buf))
        self._streamed = list(range(len(self._resident), n))
        for k in range(min(ring, len(self._streamed))):
            try:
                buf = self._alloc()
            except torch.cuda.OutOfMemoryError as e:
                raise torch.cuda.OutOfMemoryError(f"cannot allocate streaming slot {k + 1} of {min(ring, len(self._streamed))} ({st.block_nbytes / 2**20:.0f} MiB): lower resident/ring or free VRAM") from e
            self._slots.append(_Slot(k, buf, st.block_weights_from(buf)))
        if self._streamed:
            self._plan_host(pin, pin_mode, pin_fraction, headroom, pin_blocks)
            self._thread = threading.Thread(target=self._run, name="h3-stream", daemon=True)
            self._thread.start()

    def _fill(self, dst: torch.Tensor, b: int) -> None:
        """Host copy of block b out of the mapping (numpy's memcpy: ~2x torch's copy_ here, releases the GIL)."""
        np.copyto(dst.numpy(), self.store.block_bytes(b))

    def _pin_buffer(self) -> Optional[torch.Tensor]:
        """Exact-size page-locked host buffer (torch's own pinned allocator would round 219 MB up to 256 MiB)."""
        t = torch.empty(self.store.block_nbytes, dtype=torch.uint8)
        if not self._cuda:
            return t
        if not _cuda_ok(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0)):
            _clear_cuda_error(self.device)
            return None
        self._reg_ptrs.append(t.data_ptr())
        return t

    def _plan_host(self, pin, pin_mode, pin_fraction, headroom, pin_blocks) -> None:
        st, S = self.store, self._streamed
        if pin is False or pin_mode == "stage":
            want = 0
        elif pin is True:
            want = len(S)
        elif pin_blocks is not None:
            want = max(0, min(len(S), pin_blocks))
        else:
            avail = free_ram_bytes() if self._cuda else 0
            self._plan_info = {"ram_available_gib": round(avail / _GIB, 2)}
            want = plan_pinned_blocks(st, len(S), avail, pin_fraction, headroom) if self._cuda else 0
        self._plan_info = {**getattr(self, "_plan_info", {}), "pinned_wanted": want}
        mode = "copy" if pin_mode == "stage" else pin_mode
        for b in S[:want]:
            t = None
            if self._cuda and mode in ("auto", "register"):
                view = st.block_tensor(b)  # page-lock the file pages themselves: no copy, no second RAM copy
                if _cuda_ok(torch.cuda.cudart().cudaHostRegister(view.data_ptr(), view.numel(), _HOST_REGISTER_READ_ONLY)):
                    self._reg_ptrs.append(view.data_ptr())
                    t = view
                else:
                    _clear_cuda_error(self.device)
                    if pin_mode == "register":
                        break
                    mode = "copy"
            if t is None:
                t = self._pin_buffer() if mode != "register" else None
                if t is None:
                    break  # out of lockable memory: everything from here on is staged
                self._fill(t, b)
                self._kind[b] = "pinned"
            else:
                self._kind[b] = "registered"
            self._host[b] = t
        rest = [b for b in S if b not in self._kind]
        # A 2-buffer page-locked staging ring (2 blocks, ~0.44 GB for H3) is always worth it on CUDA: copies from pageable memory
        # block the copy thread and barely overlap with compute (measured in ComfyUI with RAM nearly full: every block fell back
        # to pageable and a 640x384x22 step went from 2.78 s to 3.82 s). Only pin=False opts out.
        if rest and pin is not False and (pin is True or want > 0 or pin_mode == "stage" or (pin == "auto" and self._cuda)):
            for _ in range(2):
                t = self._pin_buffer()
                if t is None:
                    break
                self._staging.append(t)
            self._stage_ev = [None] * len(self._staging)
            if self._staging:
                self._kind.update({b: "staged" for b in rest})
        for b in rest:
            self._kind.setdefault(b, "pageable")  # copy straight from the mapping (synchronous, worker thread)

    # ------------------------------------------------------------------ worker
    def _targets(self) -> List[int]:
        S = self._streamed
        out = [self._demand] if self._demand is not None else []
        out += [S[(self._pos + 1 + k) % len(S)] for k in range(min(self._prefetch, len(S)))]
        return out

    def _pick_slot(self, protect) -> Optional[_Slot]:
        free = [s for s in self._slots if s.state == FREE]
        if free:
            return free[0]
        for state in (IDLE, READY):  # evict released blocks first, then prefetched blocks nobody asked for yet
            cand = [s for s in self._slots if s.state == state and s.block not in protect]
            if cand:
                return min(cand, key=lambda s: s.stamp)
        return None

    def _next_job(self, commit: bool = True):
        """The next (slot, block) copy to issue, or None. Marks the slot PENDING unless commit=False (dry run)."""
        targets = self._targets()
        for k, t in enumerate(targets):
            if t in self._by_block:
                continue
            slot = self._pick_slot(() if (k == 0 and t == self._demand) else targets)
            if slot is None:
                return None
            if commit:
                if slot.block is not None:
                    del self._by_block[slot.block]
                self._tick += 1
                slot.state, slot.block, slot.stamp = PENDING, t, self._tick
                self._by_block[t] = slot
            return slot, t
        return None

    def _run(self) -> None:
        # inference mode is thread-local: a host that builds the provider under torch.inference_mode() (ComfyUI runs every
        # node that way) owns inference-tensor slot buffers, and copy_ into those from a thread outside the mode is an error
        with torch.inference_mode():
            self._run_loop()

    def _run_loop(self) -> None:
        if self._cuda:
            torch.cuda.set_device(self.device)
        try:
            while True:
                with self._cv:
                    while True:
                        if self._stop:
                            return
                        job = self._next_job()
                        if job:
                            break
                        self._cv.wait()
                self._issue(*job)
                with self._cv:
                    job[0].state = READY
                    self._cv.notify_all()
        except BaseException as e:  # surfaces in the next acquire()
            with self._cv:
                self._error = e
                self._cv.notify_all()

    def _issue(self, slot: _Slot, b: int) -> None:
        t0, kind, stage = time.perf_counter(), self._kind[b], None
        if kind == "staged":
            stage = self._stage_next
            self._stage_next = (stage + 1) % len(self._staging)
            if self._stage_ev[stage] is not None:
                self._stage_ev[stage].synchronize()  # the previous copy out of this buffer must have finished
            self._fill(self._staging[stage], b)
            src = self._staging[stage]
        elif kind == "pageable":
            src = self.store.block_tensor(b)
        else:
            src = self._host[b]
        if self._cuda:
            cs = self._copy_stream
            with torch.cuda.stream(cs):
                if slot.done is not None:
                    cs.wait_event(slot.done)  # device-side: the last reader of this slot is finished
                self._transfer(slot.buf, src, kind != "pageable")
                ev = torch.cuda.Event()
                ev.record(cs)
            slot.ready = ev
            if stage is not None:
                self._stage_ev[stage] = ev
        else:
            self._transfer(slot.buf, src, False)
        with self._cv:
            self._st["copies"] += 1
            self._st["bytes"] += self.store.block_nbytes
            self._st["busy_s"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ BlockProvider
    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("StreamingProvider is closed")
        if self._error is not None:
            raise RuntimeError(f"H3 streaming worker failed: {self._error!r}") from self._error

    def begin_forward(self) -> None:
        if self._in_forward:  # the previous forward never reached end_forward (e.g. an exception without finally)
            self.end_forward()
        self._in_forward = True

    def acquire(self, i: int) -> BlockWeights:
        if not 0 <= i < self.n_blocks:
            raise IndexError(f"block {i} out of range [0, {self.n_blocks})")
        if self._closed:
            raise RuntimeError("StreamingProvider is closed")
        if i in self._resident:
            return self._resident[i][1]
        t0 = time.perf_counter()
        deadline = t0 + self._timeout
        waited = False
        with self._cv:
            self._check()
            while True:
                slot = self._by_block.get(i)
                if slot is None:
                    if all(s.state == ACQUIRED for s in self._slots):
                        raise RuntimeError(f"all {len(self._slots)} ring slots are held; release() a block before acquiring block {i} (ring too small?)")
                    self._demand = i
                    self._cv.notify_all()
                elif slot.state == ACQUIRED:
                    raise RuntimeError(f"block {i} is already acquired")
                elif slot.state in (READY, IDLE):
                    break
                waited = True
                left = deadline - time.perf_counter()
                if left <= 0:
                    raise TimeoutError(f"timed out after {self._timeout:.0f}s waiting for block {i}: {self._dump()}")
                self._cv.wait(min(left, 1.0))
                self._check()
            slot.state = ACQUIRED
            self._tick += 1
            slot.stamp = self._tick
            if self._demand == i:
                self._demand = None
            self._pos = self._streamed.index(i)
            self._st["acquires"] += 1
            if waited:
                self._st["waits"] += 1
                self._st["wait_s"] += time.perf_counter() - t0
            self._cv.notify_all()  # the lookahead window moved
        if self._cuda and slot.ready is not None:
            torch.cuda.current_stream(self.device).wait_event(slot.ready)
        return slot.weights

    def release(self, i: int) -> None:
        if i in self._resident:
            return
        ev = None
        if self._cuda:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))  # every kernel using the slot is already enqueued
        with self._cv:
            slot = self._by_block.get(i)
            if slot is None or slot.state != ACQUIRED:
                raise RuntimeError(f"release({i}) but block {i} is not acquired")
            slot.done = ev
            slot.state = IDLE
            self._tick += 1
            slot.stamp = self._tick
            self._cv.notify_all()

    def end_forward(self) -> None:
        """Reset for the next forward (safe after an exception): held slots are released behind an event on the
        current stream, the lookahead restarts at the first streamed block."""
        ev = None
        if self._cuda and not self._closed:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
        with self._cv:
            for s in self._slots:
                if s.state == ACQUIRED:
                    s.state, s.done = IDLE, ev
            self._pos, self._demand, self._in_forward = -1, None, False
            self._cv.notify_all()

    # ------------------------------------------------------------------ introspection / lifecycle
    def buffer_of(self, i: int) -> Optional[torch.Tensor]:
        """The device buffer currently holding block i (resident, or a valid ring slot), else None."""
        if i in self._resident:
            return self._resident[i][0]
        with self._cv:
            s = self._by_block.get(i)
            return s.buf if s is not None and s.state in (READY, IDLE, ACQUIRED) else None

    def wait_prefetch(self, timeout: float = 60.0) -> None:
        """Block until the worker has nothing left to issue (the lookahead window is fully loaded)."""
        deadline = time.perf_counter() + timeout
        with self._cv:
            while self._thread is not None and not self._stop and self._error is None and (
                any(s.state == PENDING for s in self._slots) or self._next_job(commit=False) is not None
            ):
                left = deadline - time.perf_counter()
                if left <= 0:
                    raise TimeoutError(f"prefetch did not settle: {self._dump()}")
                self._cv.wait(min(left, 0.5))
            self._check()

    def _dump(self) -> str:
        names = "free pending ready acquired idle".split()
        return "slots " + ", ".join(f"{s.idx}:{names[s.state]}:{s.block}" for s in self._slots) + f"; pos={self._pos} demand={self._demand}"

    def stats(self) -> Dict[str, object]:
        with self._cv:
            kinds: Dict[str, int] = {}
            for k in self._kind.values():
                kinds[k] = kinds.get(k, 0) + 1
            blk = self.store.block_nbytes
            return dict(self._st, resident=len(self._resident), streamed=len(self._streamed), slots=len(self._slots), host_kinds=kinds,
                        gpu_bytes=(len(self._resident) + len(self._slots)) * blk, locked_host_bytes=(len(self._reg_ptrs)) * blk,
                        plan=dict(getattr(self, "_plan_info", {})), state=self._dump())

    def reset_stats(self) -> None:
        with self._cv:
            self._st = dict(acquires=0, waits=0, wait_s=0.0, copies=0, bytes=0, busy_s=0.0)

    def close(self) -> None:
        if self._closed:
            return
        with self._cv:
            self._closed = self._stop = True
            self._cv.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=120)
        if self._cuda:
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
            rt = torch.cuda.cudart()
            for ptr in self._reg_ptrs:
                rt.cudaHostUnregister(ptr)
        self._reg_ptrs = []
        self._slots, self._by_block, self._resident, self._host, self._staging = [], {}, {}, {}, []
        self._kind = {}
        if self._cuda:
            torch.cuda.empty_cache()

    def __enter__(self) -> "StreamingProvider":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
