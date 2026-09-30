"""ComfyUI node: H3-Turbo Fast UNET Loader (.h3t).

A drop-in replacement for the core UNETLoader for the official MiniMax H3 4-bit (W4A8/ConvRot) checkpoint, converted with
`h3turbo h3-convert`. It returns an ordinary MODEL, so ComfyUI's own conditioning, samplers and VAE nodes are unchanged; only the
DiT forward is replaced by the h3turbo streaming engine (bit-identical outputs, block copies hidden behind compute).

How it plugs into ComfyUI's memory management: the MODEL's diffusion model is a stub without weights, and the ModelPatcher lets
ComfyUI treat the engine like any partially loaded model. ComfyUI hands `partially_load` a VRAM budget for weights (free memory
minus the activation estimate it reserves); the engine builds itself inside that budget (resident blocks + a 3-slot streaming
ring). When ComfyUI needs the memory back (VAE decode, the text encoder), `partially_unload` frees all of it, including the
page-locked host copies; the next sampling run rebuilds it.

Limits: run-time LoRAs cannot be applied to a streamed 4-bit model (they raise a clear error instead of being ignored: merge them
into the checkpoint first), and block-level model patches (attention replacement, Fun ControlNet block patches) are not honoured.
"""
import gc
import logging
import os
import threading

import torch

import comfy.model_base
import comfy.model_management as mm
import comfy.model_patcher
import comfy.patcher_extension
import comfy.supported_models
import folder_paths

from h3turbo.h3.engine import H3Engine
from h3turbo.h3.store import H3TFile

LOG = logging.getLogger("h3turbo")
FOLDER = "h3turbo_h3t"  # same directory as the small-tier checkpoints, but only *.h3t
folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, "h3turbo"))
_paths, _ = folder_paths.folder_names_and_paths[FOLDER]
folder_paths.folder_names_and_paths[FOLDER] = (_paths, {".h3t"})

_ACT_MIB_PER_TOKEN = 0.18  # measured unchunked activation peak of the DiT: 0.15-0.18 MiB per token (bf16)
_MIN_BUDGET = 1 << 30  # never plan the weights into less than this when ComfyUI offers a token amount


class _LoraKeyStub(torch.nn.Module):
    """Zero-element parameter named like a real linear so ComfyUI's LoRA key mapping finds keys, and add_patches can refuse them."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj, self.out_proj = _LoraKeyStub(), _LoraKeyStub()


class _Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = _LoraKeyStub(), _LoraKeyStub()


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.mlp = _Attn(), _Mlp()


class H3TurboDiT(torch.nn.Module):
    """Stands in for comfy.ldm.minimax.model.MiniMaxH3Model: same call signature, engine inside, no weights of its own."""

    def __init__(self, path: str, precision: str, resident_blocks: int, mlp_chunk: int, attention: str = "exact"):
        super().__init__()
        self.path, self.precision, self.resident_blocks, self.mlp_chunk = path, precision, resident_blocks, mlp_chunk
        self.attn_impl = "int8" if attention == "int8_fast" else "sdpa"
        self.store = H3TFile(path)
        cfg = self.store.cfg
        self.cfg = cfg
        self.dtype = torch.bfloat16
        self.patch_size = tuple(cfg.patch)
        self.hidden_size, self.latents_dim, self.audio_latents_dim = cfg.hidden, cfg.video_channels, cfg.audio_channels
        self.sigma_shift_video, self.sigma_shift_audio = cfg.sigma_shift_video, cfg.sigma_shift_audio
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(cfg.layers)])
        self.engine = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ engine lifecycle
    def model_bytes(self) -> int:
        return int(self.store.file_size)

    def gpu_bytes(self) -> int:
        e = self.engine
        return e.gpu_bytes() if e is not None else 0

    def ensure_loaded(self, budget=None) -> int:
        """Build the engine inside `budget` bytes of VRAM (default: what is free now). Returns bytes newly placed on the GPU."""
        with self._lock:
            if self.engine is not None:
                return 0
            dev = mm.get_torch_device()
            free = int(mm.get_free_memory(dev))
            budget = free if budget is None else int(min(max(budget, _MIN_BUDGET), free))
            e = H3Engine.from_store(self.store, str(dev), owns_store=False, precision=self.precision,
                                    resident="auto" if not self.resident_blocks else self.resident_blocks,
                                    reserve_gb=0.25, free_vram_bytes=budget, attn_impl=self.attn_impl)
            self.engine = e
            st = e.stats()
            LOG.info("H3-Turbo engine loaded: %d/%d blocks resident, %d streamed, %.2f GiB on GPU (budget %.2f GiB, %s host copies)",
                     st.get("resident", 0), self.cfg.layers, st.get("streamed", 0), e.gpu_bytes() / 2**30, budget / 2**30,
                     st.get("host_kinds"))
            return e.gpu_bytes()

    def unload(self) -> int:
        """Free every byte the engine holds on the GPU (and its page-locked host ranges); the file stays mapped."""
        with self._lock:
            e, self.engine = self.engine, None
            if e is None:
                return 0
            freed = e.gpu_bytes()
            e.close()
            del e
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOG.info("H3-Turbo engine unloaded (%.2f GiB of VRAM released)", freed / 2**30)
        return freed

    def close(self) -> None:
        self.unload()
        self.store.close()

    # ------------------------------------------------------------------ what the sampler calls
    def preprocess_text_embeds(self, text_states: torch.Tensor) -> torch.Tensor:
        """[B, L, text_dim] Qwen states -> [B, L, hidden] (condition_proj + token refiner); ComfyUI calls this once per sampling run."""
        self.ensure_loaded()
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self.engine.encode_text(text_states)

    def _pick_mlp_chunk(self, x, context) -> "int | None":
        if self.mlp_chunk:
            return self.mlp_chunk
        t, h, w = x[0].shape[2:]
        tokens = context.shape[1] + t * ((h + 1) // 2) * ((w + 1) // 2) + 2 * x[1].shape[-1]
        free = torch.cuda.mem_get_info(x[0].device)[0] if x[0].is_cuda else 1 << 62
        return 4096 if tokens * _ACT_MIB_PER_TOKEN * 2**20 > 0.75 * free else None  # chunking is exact for W4A8 (per-row quantiser)

    def forward(self, x, timestep, context, control=None, transformer_options=None, minimax_payload=None, denoise_mask=None,
                audio_denoise_mask=None, **kwargs):
        to = transformer_options or {}
        self.ensure_loaded()
        if getattr(x, "is_nested", False):
            x = list(x.unbind())
        sigma = (timestep.flatten()[0] / 1000.0).float()  # computed on the device exactly like the reference (1 ulp differs on CPU)
        shifts = (float(to.get("minimax_h3_sigma_shift_video", self.sigma_shift_video)),
                  float(to.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio)))
        # ComfyUI's prebuilt PackedLayout is dropped: the engine builds (and caches) its own from the same keyframes/refs
        payload = {k: v for k, v in (minimax_payload or {}).items() if k != "layout"}
        model = self.engine.model
        model.mlp_chunk = self._pick_mlp_chunk(x, context)
        with torch.inference_mode():
            return model.forward(list(x), sigma, None, payload=payload, denoise_mask=denoise_mask,
                                 audio_denoise_mask=audio_denoise_mask, sample_sigmas=to.get("sample_sigmas"),
                                 refined_text=context, shifts=shifts)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class H3TurboBaseModel(comfy.model_base.MiniMaxH3):
    """ComfyUI's own MiniMaxH3 model object (conditioning, latent scaling, denoise masks, layout payload) around the stub DiT."""

    def __init__(self, model_config, dit, device=None):
        # skip MiniMaxH3.__init__ (it would build the 33B-parameter reference network) and hand BaseModel our stub instead
        comfy.model_base.BaseModel.__init__(self, model_config, comfy.model_base.ModelType.FLOW_AV, device=device,
                                            unet_model=lambda **_: dit)


class H3TurboPatcher(comfy.model_patcher.ModelPatcher):
    """ModelPatcher whose 'weights' are the engine's GPU state: built on partially_load, freed on partially_unload/detach."""

    def _dit(self) -> H3TurboDiT:
        return self.model.diffusion_model

    def model_size(self):
        return self._dit().model_bytes()

    def loaded_size(self):
        return self._dit().gpu_bytes()

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        if patches:
            raise RuntimeError("H3-Turbo Fast UNET Loader cannot apply LoRAs or weight patches at run time (the 4-bit weights are "
                               "streamed, never patched). Merge the LoRA into the checkpoint, or use the core UNETLoader for this workflow.")
        return []

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            self.unpatch_hooks()
            self.model.device = device_to
            self.model.current_weight_patches_uuid = self.patches_uuid
            self.model.model_loaded_weight_memory = self.loaded_size()
            for callback in self.get_all_callbacks(comfy.patcher_extension.CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)
            self.apply_hooks(self.forced_hooks, force_apply=True)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        with self.use_ejected(skip_and_inject_on_exit_only=True):
            self.unpatch_model(self.offload_device, unpatch_weights=False)
            self.patch_model(load_weights=False)  # object patches (e.g. the sigma-shift model_sampling)
            placed = self._dit().ensure_loaded(extra_memory if extra_memory and extra_memory < 1e30 else None)
            self.load(device_to, lowvram_model_memory=extra_memory, force_patch_weights=force_patch_weights)
            return placed

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        freed = self._dit().unload()
        self.model.model_loaded_weight_memory = 0
        return freed

    def detach(self, unpatch_all=True):
        if unpatch_all:  # detach(False) only swaps this patcher for a clone that shares the same engine
            self._dit().unload()
        return super().detach(unpatch_all)


def _unet_config(cfg) -> dict:
    """The dict comfy.model_detection derives from a curve-form H3 checkpoint (see detect_unet_config)."""
    return {
        "image_model": "minimax_h3", "num_layers": cfg.layers, "token_refiner_num_layers": cfg.refiner_layers,
        "hidden_size": cfg.hidden, "latents_dim": cfg.video_channels, "audio_latents_dim": cfg.audio_channels,
        "attention_head_dim": cfg.head_dim, "num_attention_heads": cfg.heads, "ffn_hidden_size": cfg.ffn,
        "text_dim": cfg.text_dim, "adaln_curve_grid": cfg.curve_grid, "time_embed_dim": cfg.t_dim,
        "rope_inv_freq_len": cfg.rope_inv_freq_len, "gate_compress": False,
    }


def build_model_patcher(path: str, precision: str = "a8", resident_blocks: int = 0, mlp_chunk: int = 0,
                        attention: str = "exact") -> H3TurboPatcher:
    dit = H3TurboDiT(path, precision, resident_blocks, mlp_chunk, attention)
    load_device, offload_device = mm.get_torch_device(), mm.unet_offload_device()
    model_config = comfy.supported_models.MiniMaxH3(_unet_config(dit.cfg))
    manual_cast = mm.unet_manual_cast(torch.bfloat16, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(torch.bfloat16, manual_cast, device=load_device)
    model = H3TurboBaseModel(model_config, dit, device=offload_device)
    return H3TurboPatcher(model, load_device=load_device, offload_device=offload_device)


class H3TurboFastUNetLoader:
    CATEGORY = "H3-Turbo"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    DESCRIPTION = ("Loads a converted official-H3 4-bit checkpoint (.h3t) and streams it through the H3-Turbo engine. Same MODEL "
                   "output and results as UNETLoader, with weight copies hidden behind compute. No run-time LoRAs.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "h3t_name": (folder_paths.get_filename_list(FOLDER),),
            "precision": (["a8", "a16"], {"default": "a8", "tooltip": "a8 = the checkpoint's native int8 activations (fast). a16 = unquantised activations: about 2x slower, closer to the un-quantised function by a small margin."}),
            "resident_blocks": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "Blocks kept on the GPU. 0 = as many as ComfyUI's VRAM budget allows."}),
            "mlp_chunk": ("INT", {"default": 0, "min": 0, "max": 65536, "tooltip": "Rows per MLP pass. 0 = automatic (chunks only when a long clip would not fit)."}),
            "attention": (["exact", "int8_fast"], {"default": "exact", "tooltip": "exact = the reference SDPA, output identical to UNETLoader. int8_fast = comfy_kitchen INT8 attention: about a quarter faster per step on long clips, but NOT bit-identical (a different, equally plausible sample)."}),
        }}

    def load(self, h3t_name, precision, resident_blocks, mlp_chunk, attention="exact"):
        path = folder_paths.get_full_path_or_raise(FOLDER, h3t_name)
        return (build_model_patcher(path, precision, resident_blocks, mlp_chunk, attention),)


NODE_CLASS_MAPPINGS = {"H3TurboFastUNetLoader": H3TurboFastUNetLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"H3TurboFastUNetLoader": "H3-Turbo Fast UNET Loader (h3t)"}
