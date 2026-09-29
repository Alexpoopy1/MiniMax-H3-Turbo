"""Block-swap offload: keep only a few transformer blocks resident on the GPU and
stream the rest from CPU memory, prefetching the next block on a side stream while
the current one computes. This is what lets a model bigger than VRAM run on a small
card, at the cost of PCIe bandwidth (so prefer a smaller tier or int8/int4 first).

Weights are read-only at inference, so a block is "unloaded" by pointing its
parameters back at the pinned CPU copy; nothing is ever copied off the GPU.

The CUDA stream/prefetch path can only be exercised on a CUDA machine. On CPU the
same hooks run with the "gpu" device set to CPU, which verifies the load/unload
bookkeeping and output equality (see tests) but not stream overlap.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn


class BlockSwapper:
    def __init__(
        self,
        blocks: nn.ModuleList,
        device,
        resident: int = 2,
        prefetch: bool = True,
        transfer: Optional[Callable[[torch.Tensor, bool], torch.Tensor]] = None,
    ):
        """transfer(cpu_tensor, non_blocking) -> device tensor; injectable for tests."""
        self.blocks = list(blocks)
        self._transfer = transfer or (lambda t, nb: t.to(self.device, non_blocking=nb))
        self.device = torch.device(device)
        self.n = len(self.blocks)
        self.resident = max(0, min(resident, self.n))
        self.use_stream = prefetch and self.device.type == "cuda"
        self.stream = torch.cuda.Stream(self.device) if self.use_stream else None
        self._events = {}
        # CPU master copies (pinned when a CUDA transfer will read them)
        self._cpu: List[List[torch.Tensor]] = []
        self._gpu: List[Optional[List[torch.Tensor]]] = [None] * self.n
        self._tensors = []
        for blk in self.blocks:
            ts = [t for t in list(blk.parameters()) + list(blk.buffers())]
            self._tensors.append(ts)
            master = [t.data.detach().cpu() for t in ts]
            if self.use_stream:
                master = [m.pin_memory() for m in master]
            self._cpu.append(master)
        self._hooks = []
        for i, blk in enumerate(self.blocks):
            self._unload(i)
            self._hooks.append(blk.register_forward_pre_hook(self._pre(i)))
            self._hooks.append(blk.register_forward_hook(self._post(i)))
        for i in range(self.resident):  # first blocks stay put
            self._load(i, sync=True)
        self._pinned_resident = set(range(self.resident))

    # ------------------------------------------------------------------ transfers
    def _load(self, i: int, sync: bool = False) -> None:
        if self._gpu[i] is not None:
            return
        stream = self.stream
        ctx = torch.cuda.stream(stream) if stream is not None and not sync else _null()
        with ctx:
            moved = [self._transfer(m, stream is not None and not sync) for m in self._cpu[i]]
        if stream is not None and not sync:
            ev = torch.cuda.Event()
            ev.record(stream)
            self._events[i] = ev
        self._gpu[i] = moved
        for t, m in zip(self._tensors[i], moved):
            t.data = m

    def _unload(self, i: int) -> None:
        for t, m in zip(self._tensors[i], self._cpu[i]):
            t.data = m
        self._gpu[i] = None
        self._events.pop(i, None)

    # ------------------------------------------------------------------ hooks
    def _pre(self, i):
        def hook(module, args):
            self._load(i)
            ev = self._events.pop(i, None)
            if ev is not None:
                cur = torch.cuda.current_stream(self.device)
                cur.wait_event(ev)
                for m in self._gpu[i]:  # allocated on the side stream, consumed on this one
                    m.record_stream(cur)
            nxt = (i + 1) % self.n
            if nxt not in self._pinned_resident:
                self._load(nxt)  # async on the side stream when CUDA
        return hook

    def _post(self, i):
        def hook(module, args, output):
            if i not in self._pinned_resident:
                self._unload(i)
        return hook

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        for i in range(self.n):
            self._load(i, sync=True)

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for g in self._gpu if g for t in g)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def enable_block_swap(pipe, resident: int = 2, prefetch: bool = True) -> BlockSwapper:
    """Attach block swapping to a loaded pipeline's transformer blocks."""
    sw = BlockSwapper(pipe.model.blocks, pipe.device, resident=resident, prefetch=prefetch)
    pipe.block_swapper = sw
    return sw
