"""ComfyUI entry point. Cloning this repo into ComfyUI/custom_nodes is enough: the
h3turbo package next to this file is put on sys.path, no pip install required."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

if __package__:  # loaded by ComfyUI as a package
    from .comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:  # imported as a bare module (pytest collecting the repo root): nothing to register
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
