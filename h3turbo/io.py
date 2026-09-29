"""Single-file checkpoints: one .safetensors holding transformer, both VAEs and the
text encoder, with the config in the file metadata. Drop it in ComfyUI's models
folder or point the CLI at it; no sidecar files.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .audio_vae import AudioVAE
from .config import H3TurboConfig
from .model import H3TurboTransformer
from .pipeline import H3TurboPipeline
from .quant import KEEP_PRECISION, cast_, quantize_, quantize_structure_
from .text import build_text_encoder
from .video_vae import VideoVAE

FORMAT = "h3turbo-1"
PARTS = ("transformer", "video_vae", "audio_vae", "text_encoder")


def auto_dtype(device) -> torch.dtype:
    if torch.device(device).type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def build_modules(cfg: H3TurboConfig):
    return (
        H3TurboTransformer(cfg.transformer),
        VideoVAE(cfg.video_vae),
        AudioVAE(cfg.audio_vae),
        build_text_encoder(cfg.text),
    )


def build_pipeline(cfg: H3TurboConfig, device="cpu", dtype: Optional[torch.dtype] = None) -> H3TurboPipeline:
    """Freshly initialised (untrained) pipeline."""
    dtype = dtype or auto_dtype(device)
    tr, vv, av, te = build_modules(cfg)
    pipe = H3TurboPipeline(tr, vv, av, te, cfg)
    place(pipe, device, dtype)
    return pipe


def place(pipe: H3TurboPipeline, device, dtype, vae_dtype=None, adaln_device="cpu") -> None:
    vae_dtype = vae_dtype or dtype
    pipe.model.to_inference(device, dtype, adaln_device)
    cast_(pipe.model, dtype)
    pipe.model.adaln.to(dtype=torch.float32)  # the bank always stays fp32
    for m, dt in ((pipe.video_vae, vae_dtype), (pipe.audio_vae, vae_dtype), (pipe.text_encoder, dtype)):
        m.to(device)
        cast_(m, dt)


def save_checkpoint(path: str, pipe: H3TurboPipeline, dtype: torch.dtype = torch.float16, extra_meta: Optional[dict] = None) -> None:
    tensors = {}
    for part in PARTS:
        module = {
            "transformer": pipe.model,
            "video_vae": pipe.video_vae,
            "audio_vae": pipe.audio_vae,
            "text_encoder": pipe.text_encoder,
        }[part]
        for k, v in module.state_dict().items():
            keep = v.dtype in (torch.int8, torch.uint8) or k.endswith(KEEP_PRECISION) or k.startswith("adaln.")
            if v.is_floating_point() and not keep:
                v = v.to(dtype)
            tensors[f"{part}.{k}"] = v.detach().cpu().contiguous()
    meta = {
        "format": FORMAT,
        "h3turbo_config": pipe.cfg.to_json(),
        "quant": getattr(pipe, "quant", None) or "none",
    }
    meta.update({k: str(v) for k, v in (extra_meta or {}).items()})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(tensors, path, metadata=meta)


def read_metadata(path: str) -> dict:
    with safe_open(path, "pt") as f:
        return dict(f.metadata() or {})


def load_checkpoint(
    path: str,
    device="cpu",
    dtype: Optional[torch.dtype] = None,
    vae_dtype: Optional[torch.dtype] = None,
    quant: Optional[str] = None,
    adaln_device="cpu",
) -> H3TurboPipeline:
    """quant: None / "int8" / "int4" applied to the transformer blocks on load (ignored
    if the checkpoint is already quantised)."""
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    meta = read_metadata(path)
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not an {FORMAT} checkpoint")
    cfg = H3TurboConfig.from_json(meta["h3turbo_config"])
    ckpt_quant = None if meta.get("quant", "none") == "none" else meta["quant"]

    with torch.device("meta"):
        tr, vv, av, te = build_modules(cfg)
    if ckpt_quant:
        quantize_structure_(tr.blocks, ckpt_quant)
    mods = {"transformer": tr, "video_vae": vv, "audio_vae": av, "text_encoder": te}
    with safe_open(path, "pt", device="cpu") as f:
        for part, module in mods.items():
            prefix = part + "."
            sd = {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
            module.load_state_dict(sd, strict=True, assign=True)
    pipe = H3TurboPipeline(tr, vv, av, te, cfg)
    pipe.quant = ckpt_quant
    if quant and not ckpt_quant:
        cast_(pipe.model, torch.float32)
        quantize_(pipe.model.blocks, quant)
        pipe.quant = quant
    dtype = dtype or auto_dtype(device)
    place(pipe, device, dtype, vae_dtype, adaln_device)
    return pipe
