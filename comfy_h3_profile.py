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
        LOG.info("[h3prof] H3 decode profiling enabled")
    except Exception as e:  # pragma: no cover
        LOG.warning("[h3prof] could not enable profiling: %r", e)
