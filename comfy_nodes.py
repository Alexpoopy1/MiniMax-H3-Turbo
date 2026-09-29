"""ComfyUI nodes for H3-Turbo.

Drop this repo into ComfyUI/custom_nodes and put checkpoints in ComfyUI/models/h3turbo.

  H3-Turbo Loader        checkpoint -> pipeline (precision, int8/int4, block offload)
  H3-Turbo Pin Frames    IMAGE (one frame or a clip) -> pinned at a frame index, optional inpaint mask
  H3-Turbo Pin Audio     AUDIO pinned at a start time
  H3-Turbo Reference     IMAGE / AUDIO references kept off the timeline (identity, style, voice)
  H3-Turbo Generate      prompt (+ context) -> IMAGE, AUDIO, fps
  H3-Turbo Refine        in-context regeneration upscale of a Generate result

Feed IMAGE + AUDIO + fps into ComfyUI's core Create Video / Save Video nodes.
"""
import copy
import os

import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from h3turbo.io import load_checkpoint
from h3turbo.pipeline import AudioCond, Generation, OmniContext, VideoCond, frames_from_uint8

FOLDER = "h3turbo"
folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER))
_paths, _ = folder_paths.folder_names_and_paths[FOLDER]
folder_paths.folder_names_and_paths[FOLDER] = (_paths, {".safetensors"})

CATEGORY = "H3-Turbo"
_loaded = {}  # one resident pipeline per (file, options)


def _image_to_frames(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE [T,H,W,3] in 0..1 -> [T,3,H,W] in -1..1."""
    return image[..., :3].permute(0, 3, 1, 2).float() * 2.0 - 1.0


def _audio_to_wave(audio: dict, rate: int) -> torch.Tensor:
    wave = audio["waveform"][0].float().mean(0)  # mono
    if audio["sample_rate"] != rate:
        n = round(len(wave) * rate / audio["sample_rate"])
        wave = torch.nn.functional.interpolate(wave[None, None], size=n, mode="linear", align_corners=False)[0, 0]
    return wave


def _to_comfy(gen: Generation, size_hw):
    if gen.video is not None:
        image = gen.video.float() / 255.0
    else:
        image = torch.zeros(1, size_hw[0], size_hw[1], 3)
    if gen.audio is not None:
        audio = {"waveform": gen.audio[None, None].float(), "sample_rate": gen.sample_rate}
    else:
        audio = {"waveform": torch.zeros(1, 1, 800), "sample_rate": gen.sample_rate}
    return image, audio


class H3TurboLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list(FOLDER),),
                "precision": (["auto", "fp16", "bf16", "fp32"],),
                "quantize": (["none", "int8", "int4"], {"tooltip": "weight-only; ~2x / ~4x less VRAM for the transformer"}),
                "offload_blocks": ("INT", {"default": 0, "min": 0, "max": 128, "tooltip": "0 = all on GPU. N = keep N blocks resident and stream the rest from CPU RAM (for models bigger than VRAM)"}),
            }
        }

    RETURN_TYPES = ("H3TURBO_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, ckpt_name, precision, quantize, offload_blocks):
        key = (ckpt_name, precision, quantize, offload_blocks)
        if key not in _loaded:
            _loaded.clear()  # never hold two pipelines: small cards
            mm.unload_all_models()
            mm.soft_empty_cache()
            dtype = {"auto": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
            pipe = load_checkpoint(
                folder_paths.get_full_path_or_raise(FOLDER, ckpt_name),
                device=mm.get_torch_device(),
                dtype=dtype,
                quant=None if quantize == "none" else quantize,
            )
            if offload_blocks:
                from h3turbo.offload import enable_block_swap

                enable_block_swap(pipe, resident=offload_blocks)
            _loaded[key] = pipe
        return (_loaded[key],)


class H3TurboPinFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "one frame = image conditioning; a batch = a clip (length 1+4k)"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "output frame where the first pinned frame lands; use frames-1 for the last frame"}),
            },
            "optional": {
                "omni_context": ("H3TURBO_CTX",),
                "mask": ("MASK", {"tooltip": "white = keep, black = regenerate (inpainting / editing)"}),
            },
        }

    RETURN_TYPES = ("H3TURBO_CTX",)
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(self, image, frame_index, omni_context=None, mask=None):
        ctx = copy.copy(omni_context) if omni_context else OmniContext()
        ctx.video = list(ctx.video)
        m = None if mask is None else mask[0].float()
        ctx.video.append(VideoCond(_image_to_frames(image), frame_index, m))
        return (ctx,)


class H3TurboPinAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio": ("AUDIO",), "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.05})},
            "optional": {"omni_context": ("H3TURBO_CTX",)},
        }

    RETURN_TYPES = ("H3TURBO_CTX",)
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(self, audio, start_seconds, omni_context=None):
        ctx = copy.copy(omni_context) if omni_context else OmniContext()
        ctx.audio = list(ctx.audio)
        ctx.audio.append(AudioCond(_audio_to_wave(audio, 32000), start_seconds))
        return (ctx,)


class H3TurboReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {"omni_context": ("H3TURBO_CTX",), "image": ("IMAGE",), "audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("H3TURBO_CTX",)
    FUNCTION = "add"
    CATEGORY = CATEGORY

    def add(self, omni_context=None, image=None, audio=None):
        ctx = copy.copy(omni_context) if omni_context else OmniContext()
        ctx.ref_video, ctx.ref_audio = list(ctx.ref_video), list(ctx.ref_audio)
        if image is not None:
            ctx.ref_video.append(_image_to_frames(image))
        if audio is not None:
            ctx.ref_audio.append(_audio_to_wave(audio, 32000))
        return (ctx,)


class H3TurboGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("H3TURBO_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 320, "min": 32, "max": 4096, "step": 32}),
                "frames": ("INT", {"default": 49, "min": 1, "max": 1025, "step": 4, "tooltip": "snapped to 1+4k"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100, "tooltip": "2-4 for Turbo checkpoints"}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "1 = off. CFG for undistilled models; guidance value for distilled ones"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "generate_video": ("BOOLEAN", {"default": True}),
                "generate_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "omni_context": ("H3TURBO_CTX",),
                "text_embeds": ("CONDITIONING", {"tooltip": "only for checkpoints built around an external text encoder"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "H3TURBO_GEN")
    RETURN_NAMES = ("images", "audio", "fps", "generation")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, pipe, prompt, width, height, frames, fps, steps, guidance, seed, generate_video, generate_audio,
                 negative_prompt="", omni_context=None, text_embeds=None):
        what = tuple(k for k, on in (("video", generate_video), ("audio", generate_audio)) if on)
        if not what:
            raise ValueError("enable generate_video and/or generate_audio")
        bar = comfy.utils.ProgressBar(steps)

        def tick(i, total):
            mm.throw_exception_if_processing_interrupted()
            bar.update_absolute(i, total)

        gen = pipe(
            prompt, negative_prompt=negative_prompt, width=width, height=height, num_frames=frames, fps=fps,
            generate=what, context=omni_context, steps=steps, guidance=guidance if guidance != 1.0 else None,
            seed=seed % (2**63), text_embeds=None if text_embeds is None else text_embeds[0][0], callback=tick,
        )
        image, audio = _to_comfy(gen, (height, width))
        return (image, audio, float(gen.fps), gen)


class H3TurboRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("H3TURBO_PIPE",),
                "generation": ("H3TURBO_GEN",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25}),
                "steps": ("INT", {"default": 2, "min": 1, "max": 20}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "how much of the upsampled video is re-noised and regenerated"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "fps")
    FUNCTION = "refine"
    CATEGORY = CATEGORY

    def refine(self, pipe, generation, prompt, scale, steps, strength, seed):
        bar = comfy.utils.ProgressBar(steps)

        def tick(i, total):
            mm.throw_exception_if_processing_interrupted()
            bar.update_absolute(i, total)

        out = pipe.refine(generation, prompt, scale=scale, steps=steps, strength=strength, seed=seed % (2**63), callback=tick)
        h, w = out.video.shape[1:3]
        image, audio = _to_comfy(out, (h, w))
        return (image, audio, float(out.fps))


NODE_CLASS_MAPPINGS = {
    "H3TurboLoader": H3TurboLoader,
    "H3TurboPinFrames": H3TurboPinFrames,
    "H3TurboPinAudio": H3TurboPinAudio,
    "H3TurboReference": H3TurboReference,
    "H3TurboGenerate": H3TurboGenerate,
    "H3TurboRefine": H3TurboRefine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TurboLoader": "H3-Turbo Loader",
    "H3TurboPinFrames": "H3-Turbo Pin Frames",
    "H3TurboPinAudio": "H3-Turbo Pin Audio",
    "H3TurboReference": "H3-Turbo Reference",
    "H3TurboGenerate": "H3-Turbo Generate",
    "H3TurboRefine": "H3-Turbo Refine",
}
