"""Opt-in timing log for the H3 decode path (create an empty file named PROFILE next to this file and restart ComfyUI).

Logs, with CUDA synchronisation, the wall time of every MiniMax H3 video-VAE decoder call and of whole VAE decodes, plus the
free VRAM at the start. Off by default: the synchronisation itself costs a little.
"""
import functools
import logging
import os
import time

import torch

LOG = logging.getLogger("h3turbo")
ENABLED = os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROFILE"))


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _wrap(owner, name, label):
    fn = getattr(owner, name)
    if getattr(fn, "_h3_prof", False):
        return

    @functools.wraps(fn)
    def timed(*a, **k):
        _sync()
        free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
        t0 = time.perf_counter()
        out = fn(*a, **k)
        _sync()
        shape = next((tuple(x.shape) for x in a[1:] if isinstance(x, torch.Tensor)), None)
        LOG.info("[h3prof] %s %s %.3f s (free VRAM at start %.2f GiB)", label, shape, time.perf_counter() - t0, free)
        return out

    timed._h3_prof = True
    setattr(owner, name, timed)


if ENABLED:
    try:
        import comfy.ldm.minimax.vae as _vae
        import comfy.sd as _sd

        _wrap(_vae.MiniMaxH3VideoVAE, "_decode_pixels", "vae.decoder_call")
        _wrap(_vae.MiniMaxH3VideoVAE, "decode", "vae.decode_total")
        _wrap(_sd.VAE, "decode", "VAE.decode(wrapper)")

        import collections
        import comfy.pinned_memory as _pm
        import comfy.model_management as _mm

        _acc = collections.defaultdict(lambda: [0, 0.0, 0])  # calls, seconds, bytes

        def _count(owner, name):
            fn = getattr(owner, name)

            @functools.wraps(fn)
            def counted(*a, **k):
                t0 = time.perf_counter()
                out = fn(*a, **k)
                e = _acc[name]
                e[0] += 1
                e[1] += time.perf_counter() - t0
                return out

            setattr(owner, name, counted)

        for _n in ("get_pin", "pin_memory"):
            _count(_pm, _n)
        for _n in ("free_pins", "free_registrations", "ensure_pin_budget", "load_models_gpu", "free_memory"):
            _count(_mm, _n)

        _vae_decode = _sd.VAE.decode

        @functools.wraps(_vae_decode)
        def _decode_with_counters(*a, **k):
            _acc.clear()
            out = _vae_decode(*a, **k)
            LOG.info("[h3prof] counters during VAE.decode: %s", {n: (c, round(s, 3)) for n, (c, s, _) in _acc.items()})
            return out

        _sd.VAE.decode = _decode_with_counters
        LOG.info("[h3prof] H3 decode profiling enabled")
    except Exception as e:  # pragma: no cover
        LOG.warning("[h3prof] could not enable profiling: %r", e)

    def _memdiag():
        """Where the process RAM goes: CPU tensor storages by kind (pinned / mmap-backed / private), loaded models, pins."""
        import gc
        import collections

        import psutil
        import comfy.model_management as mm

        seen, kinds, big = set(), collections.Counter(), []
        for o in gc.get_objects():
            try:
                if not isinstance(o, torch.Tensor) or o.device.type != "cpu":
                    continue
                st = o.untyped_storage()
                key = st.data_ptr()
                if key in seen or st.nbytes() == 0:
                    continue
                seen.add(key)
                kind = "pinned" if o.is_pinned() else ("hostbuf" if getattr(st, "_comfy_hostbuf", None) is not None else "cpu")
                kinds[kind] += st.nbytes()
                big.append((st.nbytes(), kind, tuple(o.shape), str(o.dtype)))
            except Exception:
                continue
        big.sort(reverse=True)
        p = psutil.Process()
        mi = p.memory_info()
        vm = psutil.virtual_memory()
        models = []
        for lm in mm.current_loaded_models:
            m = lm.model
            if m is None:
                continue
            models.append({"model": type(m.model).__name__, "size_gib": round(m.model_size() / 2**30, 2), "loaded_gib": round(m.loaded_size() / 2**30, 2)})
        return {
            "process": {"rss_gib": round(mi.rss / 2**30, 2), "private_gib": round(getattr(mi, "private", 0) / 2**30, 2)},
            "system": {"available_gib": round(vm.available / 2**30, 2), "total_gib": round(vm.total / 2**30, 2)},
            "cpu_tensor_storages_gib": {k: round(v / 2**30, 2) for k, v in kinds.items()},
            "largest": [(round(b / 2**20), k, s, d) for b, k, s, d in big[:15]],
            "comfy_total_pinned_gib": round(getattr(mm, "TOTAL_PINNED_MEMORY", 0) / 2**30, 2),
            "loaded_models": models,
            "cuda": {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2), "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
                     "free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2)} if torch.cuda.is_available() else {},
        }

    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get("/h3turbo/memdiag")
        async def _memdiag_route(request):
            return web.json_response(_memdiag())
    except Exception as e:  # pragma: no cover
        LOG.warning("[h3prof] memdiag route unavailable: %r", e)
