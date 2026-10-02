"""ComfyUI entry point. Cloning this repo into ComfyUI/custom_nodes is enough: the
h3turbo package next to this file is put on sys.path, no pip install required."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

if __package__:  # loaded by ComfyUI as a package
    from .comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    try:  # the official-H3 4-bit engine loader; a failure here must not take the small-tier nodes down with it
        from .comfy_h3_nodes import NODE_CLASS_MAPPINGS as _h3_nodes, NODE_DISPLAY_NAME_MAPPINGS as _h3_names

        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_h3_nodes}
        NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_h3_names}
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger("h3turbo").warning("H3-Turbo Fast UNET Loader unavailable: %r", e)

    try:  # cached Qwen3-VL text encoder for H3 workflows
        from .comfy_h3_te import NODE_CLASS_MAPPINGS as _te_nodes, NODE_DISPLAY_NAME_MAPPINGS as _te_names

        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_te_nodes}
        NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_te_names}
    except Exception as e:  # pragma: no cover
        import logging

        logging.getLogger("h3turbo").warning("H3-Turbo Cached Text Encoder unavailable: %r", e)
    try:  # opt-in decode timing log (an empty file named PROFILE next to this file enables it)
        from . import comfy_h3_profile  # noqa: F401
    except Exception:  # pragma: no cover
        pass
else:  # imported as a bare module (pytest collecting the repo root): nothing to register
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
