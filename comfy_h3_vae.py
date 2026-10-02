"""Keep the MiniMax H3 video and audio VAEs' decoder weights page-locked in RAM, so every decode uploads them by DMA instead of
re-reading them from disk.

Why: the H3 video VAE has a 2.4 B-parameter transformer decoder (4.85 GB fp16). With ComfyUI's dynamic VRAM its weights stay
in host memory and are copied to the GPU on each decode, through page-locked "pins" that ComfyUI evicts whenever another model
asks for pinned RAM (the DiT loader does at every run, for its whole size). On a 32 GB PC the weights' own pages are then pushed
out of RAM, and each decode re-reads ~4.5 GB from disk: measured 13-15 s per decode instead of a few seconds.

What: at a VAE's first decode (after the DiT engine has locked its streamed blocks), the decoder-side weights that ComfyUI streams
to the GPU (weight and bias of its cast layers above 16 KiB) are copied once into a single page-locked RAM arena, as many layers
as free RAM allows above a 2 GiB floor, and those layers are excluded from ComfyUI's own pinning. Every later decode copies them
to the GPU by DMA from the arena. The values and the kernels are unchanged; only where the bytes come from changes.

The arena is given back (the layers return to their file-backed weights, ComfyUI's default path) when ComfyUI frees pinned
memory because of real RAM pressure after a node, or when free RAM drops below 1 GiB; not when ComfyUI only makes room to pin
another model (the H3 DiT loader asks for that at every run although its engine never uses ComfyUI's pins). It is also freed
with the VAE object. Requires ComfyUI's dynamic VRAM (the default with an NVIDIA GPU).

Disable with an empty file named NO_VAE_LOCK next to this file.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
import weakref

import torch

LOG = logging.getLogger("h3turbo")
_DISABLED = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "NO_VAE_LOCK"))
_PENDING: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()  # H3 VAEs not page-locked yet -> their decoder bytes
_LOCKED: "weakref.WeakSet" = weakref.WeakSet()  # H3 VAEs holding an arena
_LOCK_FLOOR = 2 << 30  # free RAM a lock must leave: the floor ComfyUI's own pin budget keeps (the arena replaces those pins)
_EMERGENCY = 1 << 30  # below this much free RAM an arena is given back whatever ComfyUI frees pins for
_MIN_LOCK = 256 << 20  # smaller partial arenas are not worth holding
_STREAMED_MIN = 16 * 1024  # ComfyUI force-loads smaller layers to the GPU and streams only the larger ones (model_patcher.py)
_BOUNCE_EVERY = 512 << 20  # while copying, let go of the checkpoint's mapped pages every this many bytes
_GC_EVERY_S = 10.0
_DECODE_PREFIXES = {  # submodules a decode reads; the encoder side is left to ComfyUI (T2V never uses it)
    "MiniMaxH3VideoVAE": ("post_quant_conv", "decoder"),
    "MiniMaxH3AudioVAE": ("dec_in_proj", "decoder"),
}
_tls = threading.local()
_last_gc = [0.0]


def ram_headroom() -> int:
    """Free RAM to leave untouched: ComfyUI's RAM-pressure cache evicts cached outputs (our engine included) below its headroom."""
    try:
        import comfy.memory_management

        base = int(getattr(comfy.memory_management, "RAM_CACHE_HEADROOM", 0))
    except Exception:
        base = 0
    return max(2 << 30, base + (1 << 30))


def pending_reserve_bytes() -> int:
    """RAM the H3 VAE(s) will want to page-lock at their first decode (the DiT leaves it free when parking)."""
    return int(sum(_PENDING.values()))


def _available() -> int:
    try:
        import comfy.system_memory

        return int(comfy.system_memory.virtual_memory_available())
    except Exception:
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:
            return 1 << 62


def _lockable() -> bool:
    try:
        import comfy.memory_management

        aimdo = bool(getattr(comfy.memory_management, "aimdo_enabled", False))
    except Exception:
        aimdo = False
    return not _DISABLED and aimdo and torch.cuda.is_available()


def _unregister(ptrs):
    try:
        rt = torch.cuda.cudart()
        for p in ptrs:
            rt.cudaHostUnregister(p)
    except Exception:
        pass


def _nbytes(t) -> int:
    return -(-(t.numel() * t.element_size()) // 256) * 256


def _module_size(mod) -> int:
    try:
        import comfy.model_management

        return int(comfy.model_management.module_size(mod))
    except Exception:
        return sum(t.nbytes for t in mod.state_dict().values())


def _decode_layers(model):
    """[(module, [(module, name, source tensor)])] for the layers ComfyUI streams to the GPU at decode time: decoder-side cast
    layers (comfy.ops, `comfy_cast_weights`) above 16 KiB with plain CPU weight/bias. Smaller layers, other parameters and
    buffers are force-loaded to the GPU by ComfyUI (and kept in its backups), so they never belong in the arena."""
    prefixes = _DECODE_PREFIXES.get(type(model).__name__)
    out = []
    for name, mod in model.named_modules():
        if prefixes is not None and not any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        if not hasattr(mod, "comfy_cast_weights") or _module_size(mod) <= _STREAMED_MIN:
            continue
        ents = []
        for pname in ("weight", "bias"):
            t = mod._parameters.get(pname)
            if t is None or t.numel() == 0:
                continue
            src = t.data if isinstance(t, torch.nn.Parameter) else t
            if type(src) is not torch.Tensor or src.device.type != "cpu":  # e.g. QuantizedTensor (int8 VAE): not ours
                ents = []
                break
            ents.append((mod, pname, src))
        if ents:
            out.append((mod, ents))
    return out


def _layers_bytes(layers) -> int:
    seen, total = set(), 0
    for _, ents in layers:
        for _, _, src in ents:
            key = (src.untyped_storage().data_ptr(), src.storage_offset(), tuple(src.shape))
            if key not in seen:  # a tensor shared by several layers is counted (and copied) once
                seen.add(key)
                total += _nbytes(src)
    return total


def _collect_dead_vaes() -> None:
    """comfy.sd.VAE objects are reference cycles, so a VAE ComfyUI dropped keeps its arena until the cyclic GC runs."""
    now = time.monotonic()
    if now - _last_gc[0] >= _GC_EVERY_S:
        _last_gc[0] = now
        gc.collect()


def lock_vae_weights(vae):
    """Copy an H3 VAE's streamed decoder weights into one page-locked RAM arena (same values) and keep ComfyUI from staging them
    through its own evictable pins. Returns the bytes locked, 0 when free RAM is short right now (retried at a later decode),
    or None when it can never apply to this VAE."""
    import comfy.model_management as mm

    if not _lockable():
        return None
    model = getattr(vae, "first_stage_model", None)
    if model is None or getattr(vae, "_h3turbo_arena", None) is not None:
        return None
    layers = _decode_layers(model)
    want = _layers_bytes(layers)
    if want == 0:
        return None
    a0 = _available()
    if a0 - want < _LOCK_FLOOR and len(_LOCKED):
        _collect_dead_vaes()  # an unreachable H3 VAE may still hold its arena
        a0 = _available()
    budget, total, chosen = a0 - _LOCK_FLOOR, 0, []
    for mod, ents in layers:  # whole layers only (ComfyUI pins per layer), in decode order, as many as fit
        n = sum(_nbytes(src) for _, _, src in ents)
        if total + n <= budget:
            chosen.append((mod, ents))
            total += n
    if total < min(want, _MIN_LOCK):
        LOG.info("H3-Turbo VAE lock: skipped (%s needs %.2f GiB of RAM, %.2f GiB free, %.2f GiB must stay free); decodes read it "
                 "from disk. Close other apps to free RAM for the fast path.", type(model).__name__, want / 2**30, a0 / 2**30,
                 _LOCK_FLOOR / 2**30)
        return 0
    # Phase 1: copy everything; nothing is re-pointed until every copy has succeeded.
    arena = torch.empty(total, dtype=torch.uint8)
    off, since_bounce, views, locked = 0, 0, {}, []
    try:
        with torch.no_grad():
            for mod, ents in chosen:
                for _, pname, src in ents:
                    key = (src.untyped_storage().data_ptr(), src.storage_offset(), tuple(src.shape))
                    view = views.get(key)
                    if view is None:  # a tensor shared by several layers gets one copy, so the tie is kept
                        n = src.numel() * src.element_size()
                        view = arena[off:off + n].view(src.dtype).view(src.shape)
                        view.copy_(src)  # read once from the checkpoint
                        views[key] = view
                        off += _nbytes(src)
                        since_bounce += n
                        refs = getattr(src.untyped_storage(), "_comfy_tensor_mmap_refs", None)
                        if refs is not None:
                            mm.mark_mmap_dirty(src.untyped_storage())  # ComfyUI unmaps these pages after the node
                            if since_bounce >= _BOUNCE_EVERY:  # and so do we, so the copy does not hold the VAE twice
                                since_bounce = 0
                                try:
                                    refs[0].bounce()
                                except Exception:
                                    pass
                    locked.append((mod, pname, src, view))
    except Exception:
        del views, locked, arena
        raise
    # Phase 2: point the layers at the arena.
    for mod, pname, src, view in locked:
        mod._parameters[pname].data = view
        mod._h3turbo_locked = True
    rt = torch.cuda.cudart()
    if int(rt.cudaHostRegister(arena.data_ptr(), total, 0)) != 0:
        try:
            mm.discard_cuda_async_error()
        except Exception:
            pass
        LOG.info("H3-Turbo VAE lock: page-locking failed; the weights stay in (pageable) RAM")
    else:
        weakref.finalize(arena, _unregister, [arena.data_ptr()])
    vae._h3turbo_arena, vae._h3turbo_locked_entries = arena, locked
    _LOCKED.add(vae)
    LOG.info("H3-Turbo VAE lock: %.2f of %.2f GiB of %s decoder weights held page-locked in RAM (free RAM %.2f GiB before)%s",
             total / 2**30, want / 2**30, type(model).__name__, a0 / 2**30,
             "" if total == want else "; the rest is read from the checkpoint at each decode (free more RAM to lock it all)")
    return total


def _arena_in_use(vae) -> bool:
    """False when ComfyUI has swapped most of the arena's bytes for other tensors (the arena would then be RAM held for nothing)."""
    locked = getattr(vae, "_h3turbo_locked_entries", None) or []
    total = live = 0
    for mod, pname, _, view in locked:
        n = view.numel() * view.element_size()
        total += n
        cur = mod._parameters.get(pname)
        if cur is not None and cur.data_ptr() == view.data_ptr():
            live += n
    return total > 0 and live * 2 >= total


def release_vae_arena(vae, retry: bool = True) -> int:
    """Give an arena back: the layers return to their file-backed weights, ComfyUI's default path. Returns the bytes freed.
    retry=False keeps this VAE on the default path for good (its arena was not being used)."""
    arena = getattr(vae, "_h3turbo_arena", None)
    if arena is None:
        return 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # no copy out of the arena may still be in flight when it is freed
    with torch.no_grad():
        for mod, pname, src, view in vae._h3turbo_locked_entries:
            cur = mod._parameters.get(pname)
            if cur is not None and cur.data_ptr() == view.data_ptr():  # leave alone what ComfyUI has moved elsewhere
                cur.data = src
            mod.__dict__.pop("_h3turbo_locked", None)
    n = arena.numel()
    vae._h3turbo_arena = vae._h3turbo_locked_entries = None
    _LOCKED.discard(vae)
    del arena
    if retry:
        _PENDING[vae] = n
    return n


def _install_pin_hooks() -> None:
    """ComfyUI must not stage the arena's weights through its own pins (a second copy it evicts), and frees pinned memory
    for two reasons: real RAM pressure after a node, and making room to pin a model it is about to load. The arena is given
    back for the first (or when RAM is critically low), never for the second: the H3 DiT loader asks for room at every run,
    and giving the arena back then would re-read the VAE from disk at every decode."""
    import comfy.model_management as mm
    import comfy.pinned_memory as pm

    if getattr(pm, "_h3turbo_pin_skip", False):
        return
    orig_pin_memory, orig_free_pins, orig_budget = pm.pin_memory, mm.free_pins, mm.ensure_pin_budget

    def pin_memory(module, subset="weights", size=None):
        if getattr(module, "_h3turbo_locked", False):
            return None
        return orig_pin_memory(module, subset=subset, size=size)

    def ensure_pin_budget(*args, **kwargs):
        _tls.budget = getattr(_tls, "budget", 0) + 1
        try:
            return orig_budget(*args, **kwargs)
        finally:
            _tls.budget -= 1

    def free_pins(size, evict_active=False, loaded=False):
        freed = orig_free_pins(size, evict_active=evict_active, loaded=loaded)
        if freed < size and len(_LOCKED):
            _collect_dead_vaes()
            making_room = getattr(_tls, "budget", 0) > 0
            if len(_LOCKED) and (not making_room or _available() < _EMERGENCY):
                for vae in sorted(list(_LOCKED), key=lambda v: -v._h3turbo_arena.numel()):
                    if freed >= size:
                        break
                    n = release_vae_arena(vae)
                    freed += n
                    LOG.info("H3-Turbo VAE lock: %.2f GiB arena released under RAM pressure (locked again at a later decode)",
                             n / 2**30)
        return freed

    pm.pin_memory = pin_memory
    mm.ensure_pin_budget = ensure_pin_budget
    mm.free_pins = free_pins
    pm._h3turbo_pin_skip = True


def _before_decode(vae) -> None:
    if vae in _LOCKED and not _arena_in_use(vae):
        n = release_vae_arena(vae, retry=False)
        LOG.warning("H3-Turbo VAE lock: ComfyUI replaced the VAE's weights; %.2f GiB arena released, this VAE uses the default path",
                    n / 2**30)
    if vae in _PENDING:
        try:
            r = lock_vae_weights(vae)
        except Exception as e:
            r = None
            LOG.warning("H3-Turbo VAE lock failed: %r", e)
        if r != 0:  # locked, or never applicable: stop reserving RAM for it
            _PENDING.pop(vae, None)


def install() -> None:
    """Register H3 VAEs when created and page-lock them lazily at their first decode, i.e. after the DiT engine has locked its
    streamed blocks: those come from the hard disk at every step, so they get RAM first."""
    import comfy.sd

    if getattr(comfy.sd.VAE, "_h3turbo_lock_hook", False):
        return
    _install_pin_hooks()
    orig_init, orig_decode, orig_decode_tiled = comfy.sd.VAE.__init__, comfy.sd.VAE.decode, comfy.sd.VAE.decode_tiled
    h3_types = []
    for mod_name, cls_name in (("comfy.ldm.minimax.vae", "MiniMaxH3VideoVAE"), ("comfy.ldm.minimax.audio_vae", "MiniMaxH3AudioVAE")):
        try:
            h3_types.append(getattr(__import__(mod_name, fromlist=[cls_name]), cls_name))
        except Exception as e:  # an older ComfyUI without one of them: lock the other
            LOG.info("H3-Turbo VAE lock: %s.%s unavailable (%r)", mod_name, cls_name, e)
    h3_types = tuple(h3_types)

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            fsm = getattr(self, "first_stage_model", None)
            if h3_types and isinstance(fsm, h3_types) and _lockable():  # no dynamic VRAM: nothing to lock, reserve nothing
                n = _layers_bytes(_decode_layers(fsm))
                if n:
                    _PENDING[self] = n
        except Exception as e:  # never break VAE loading
            LOG.warning("H3-Turbo VAE lock: registration failed: %r", e)

    def decode(self, *args, **kwargs):
        _before_decode(self)
        return orig_decode(self, *args, **kwargs)

    def decode_tiled(self, *args, **kwargs):
        _before_decode(self)
        return orig_decode_tiled(self, *args, **kwargs)

    comfy.sd.VAE.__init__ = __init__
    comfy.sd.VAE.decode = decode
    comfy.sd.VAE.decode_tiled = decode_tiled
    comfy.sd.VAE._h3turbo_lock_hook = True
