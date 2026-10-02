"""ComfyUI node: H3-Turbo Cached Text Encoder (Qwen3-VL for MiniMax H3).

A drop-in replacement for CLIPLoaderGGUF / CLIPLoader in H3 workflows. It returns a CLIP that the core H3 nodes
(MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, CLIPTextEncode, ...) use unchanged, but every encode is cached on
disk, keyed by the exact token stream (prompt text, picture/video pixels, reference layout) and the text-encoder file:

* A cache hit returns the stored conditioning without touching the 32B text encoder: no GPU load, no RAM for its
  weights, no disk read. The values are the ones the encoder produced, so hit and miss give identical videos.
* A miss runs the real encoder (the same ComfyUI code path as the stock loader), stores the result, and then, by
  default on machines with less than 40 GB of RAM, drops the encoder's weights from RAM and VRAM. Its 9 GB would
  otherwise push the DiT's streamed weights out of the page cache, and the next sampling run would re-read them from
  disk (measured on a 32 GB PC with the models on a hard disk: +70 s on the first sampling step).

The cache lives in <ComfyUI user dir>/h3turbo_cond_cache (a few MB per prompt). Delete the folder to clear it.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import os
import struct
import threading
import time

import torch

import comfy.model_management as mm
import folder_paths

LOG = logging.getLogger("h3turbo")
_FORMAT = b"h3turbo-cond-v1"
_LOW_RAM_BYTES = 40 << 30


def _cache_dir() -> str:
    try:
        base = folder_paths.get_user_directory()
    except Exception:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    d = os.path.join(base, "h3turbo_cond_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _total_ram() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return 1 << 40


def _te_names():
    names = set()
    for folder in ("clip", "text_encoders", "clip_gguf"):
        try:
            names.update(folder_paths.get_filename_list(folder))
        except Exception:
            pass
    return sorted(names)


def _te_path(name: str) -> str:
    for folder in ("clip", "text_encoders", "clip_gguf"):
        try:
            p = folder_paths.get_full_path(folder, name)
        except Exception:
            p = None
        if p:
            return p
    raise FileNotFoundError(f"text encoder {name!r} not found in models/clip or models/text_encoders")


# ------------------------------------------------------------------------------------------- key of a token stream
def _feed(h, obj) -> None:
    """Canonical, type-tagged serialisation of tokenizer output into a hash: ints, floats, strings, tensors, nested."""
    if obj is None:
        h.update(b"N")
    elif isinstance(obj, bool):
        h.update(b"B1" if obj else b"B0")
    elif isinstance(obj, int):
        h.update(b"I" + str(obj).encode())
    elif isinstance(obj, float):
        h.update(b"F" + struct.pack("<d", obj))
    elif isinstance(obj, str):
        b = obj.encode("utf-8")
        h.update(b"S" + str(len(b)).encode() + b":" + b)
    elif isinstance(obj, torch.Tensor):
        t = obj.detach().to("cpu").contiguous()
        h.update(b"T" + str(tuple(t.shape)).encode() + str(t.dtype).encode())
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        h.update(t.numpy().tobytes())
    elif isinstance(obj, dict):
        h.update(b"D" + str(len(obj)).encode())
        for k in sorted(obj, key=str):
            _feed(h, str(k))
            _feed(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update((b"L" if isinstance(obj, list) else b"U") + str(len(obj)).encode())
        for v in obj:
            _feed(h, v)
    else:
        h.update(b"O" + type(obj).__name__.encode() + repr(obj).encode())


def token_key(te_identity: str, tokens, extra=None) -> str:
    h = hashlib.sha256(_FORMAT)
    _feed(h, te_identity)
    _feed(h, tokens)
    _feed(h, extra)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------- the CLIP stand-in
class CachedTextEncoder:
    """Duck-types comfy.sd.CLIP for the methods the H3 nodes call; anything else is forwarded to the real CLIP."""

    def __init__(self, clip_name: str, release: str = "auto", cache: bool = True):
        self.clip_name, self.release, self.cache = clip_name, release, cache
        self.path = _te_path(clip_name)
        st = os.stat(self.path)
        self.identity = f"{os.path.basename(self.path)}|{st.st_size}|{int(st.st_mtime)}"
        self._real = None
        self._lock = threading.RLock()

    # -------------------------------------------------------------- real encoder lifecycle
    def _load_real(self):
        with self._lock:
            if self._real is None:
                import nodes

                t0 = time.perf_counter()
                if self.path.lower().endswith(".gguf"):
                    cls = nodes.NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")
                    if cls is None:
                        raise RuntimeError("a .gguf text encoder needs the ComfyUI-GGUF custom node (CLIPLoaderGGUF)")
                    self._real = cls().load_clip(self.clip_name, "minimax")[0]
                else:
                    self._real = nodes.NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(self.clip_name, type="minimax")[0]
                LOG.info("H3-Turbo text encoder: opened %s in %.1f s", self.clip_name, time.perf_counter() - t0)
            return self._real

    def release_weights(self) -> None:
        """Unload the real encoder from VRAM and drop it from RAM; it is re-opened (lazily) on the next cache miss."""
        with self._lock:
            real, self._real = self._real, None
        if real is None:
            return
        patcher = getattr(real, "patcher", None)
        for i in range(len(mm.current_loaded_models) - 1, -1, -1):
            lm = mm.current_loaded_models[i]
            m = lm.model
            if m is not None and patcher is not None and (m is patcher or getattr(m, "parent", None) is patcher):
                try:
                    lm.model_unload()
                except Exception as e:  # pragma: no cover
                    LOG.warning("H3-Turbo text encoder: unload failed: %r", e)
                mm.current_loaded_models.pop(i)
        del real, patcher
        gc.collect()
        mm.soft_empty_cache()
        LOG.info("H3-Turbo text encoder: weights released from RAM and VRAM")

    def _should_release(self) -> bool:
        if self.release == "keep":
            return False
        if self.release == "after_encode":
            return True
        return _total_ram() < _LOW_RAM_BYTES

    # -------------------------------------------------------------- the CLIP surface the H3 nodes use
    def tokenize(self, text, return_word_ids=False, **kwargs):
        return self._load_real().tokenize(text, return_word_ids=return_word_ids, **kwargs)

    def _cached(self, tokens, extra, encode):
        key = token_key(self.identity, tokens, extra) if self.cache else None
        path = os.path.join(_cache_dir(), key + ".pt") if key else None
        if path and os.path.exists(path):
            try:
                out = torch.load(path, map_location="cpu", weights_only=True)
                LOG.info("H3-Turbo text encoder: cache hit %s (encoder not loaded)", key[:12])
                return out
            except Exception as e:
                LOG.warning("H3-Turbo text encoder: unreadable cache entry %s (%r); re-encoding", key[:12], e)
        t0 = time.perf_counter()
        out = encode(self._load_real())
        LOG.info("H3-Turbo text encoder: encoded in %.1f s%s", time.perf_counter() - t0, f" (cached as {key[:12]})" if key else "")
        if path:
            tmp = path + f".{os.getpid()}.tmp"
            try:
                torch.save(_to_cpu(out), tmp)
                os.replace(tmp, path)
            except Exception as e:  # a read-only or full disk must not fail the generation
                LOG.warning("H3-Turbo text encoder: could not write cache %s: %r", path, e)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        if self._should_release():
            self.release_weights()
        return out

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
        return self._cached(tokens, ("scheduled", bool(unprojected), add_dict),
                            lambda real: real.encode_from_tokens_scheduled(tokens, unprojected=unprojected, add_dict=add_dict, show_pbar=show_pbar))

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        return self._cached(tokens, ("plain", bool(return_pooled), bool(return_dict)),
                            lambda real: real.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=return_dict))

    def clone(self):
        return self  # stateless apart from the lazily opened encoder; patches (LoRA, clip skip) go to the real CLIP below

    def __getattr__(self, name):
        if name.startswith("_") or name in ("clip_name", "release", "cache", "path", "identity"):
            raise AttributeError(name)
        return getattr(self._load_real(), name)


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class H3TurboCachedTextEncoder:
    CATEGORY = "H3-Turbo"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    DESCRIPTION = ("Qwen3-VL text encoder for MiniMax H3 (.gguf via ComfyUI-GGUF, or .safetensors) with an on-disk cache of every "
                   "encode. Repeating a prompt (same text and same images) skips the 32B encoder entirely; results are identical.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip_name": (_te_names(), {"tooltip": "The H3 text encoder, e.g. qwen3vl-32B-MiniMax-H3-Q2_K.gguf"}),
            "release": (["auto", "after_encode", "keep"], {"default": "auto", "tooltip": "After encoding a NEW prompt: auto = drop the encoder from RAM/VRAM on PCs with < 40 GB RAM (keeps the DiT and VAE in RAM, so sampling never re-reads from disk); after_encode = always; keep = never (fastest prompt changes when RAM is plentiful)."}),
            "cache": ("BOOLEAN", {"default": True, "tooltip": "Store encodes on disk (user/h3turbo_cond_cache). Off = always run the encoder."}),
        }}

    def load(self, clip_name, release="auto", cache=True):
        return (CachedTextEncoder(clip_name, release, cache),)


class H3TurboConditioningWarmup:
    """Output node that only forces the encode to run (and land in the cache): low-RAM hosts (Colab T4) encode in one prompt,
    free everything, then sample in a second prompt that hits the cache."""

    CATEGORY = "H3-Turbo"
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    def run(self, conditioning):
        return {}


NODE_CLASS_MAPPINGS = {"H3TurboCachedTextEncoder": H3TurboCachedTextEncoder, "H3TurboConditioningWarmup": H3TurboConditioningWarmup}
NODE_DISPLAY_NAME_MAPPINGS = {"H3TurboCachedTextEncoder": "H3-Turbo Cached Text Encoder (Qwen3-VL)",
                              "H3TurboConditioningWarmup": "H3-Turbo Encode Only (cache warm-up)"}
