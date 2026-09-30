"""The Colab helper's API graph: every link points at an existing node output, and the options wire the right nodes."""
import importlib.util
import os

import pytest

_spec = importlib.util.spec_from_file_location("h3_colab", os.path.join(os.path.dirname(__file__), "..", "colab", "h3_colab.py"))
h3_colab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h3_colab)

OUTPUTS = {"H3TurboFastUNetLoader": 1, "CLIPLoaderGGUF": 1, "CLIPLoader": 1, "VAELoader": 1, "MiniMaxH3ImageToVideo": 2, "RandomNoise": 1,
           "KSamplerSelect": 1, "BasicScheduler": 1, "BasicGuider": 1, "SamplerCustomAdvanced": 2, "VAEDecode": 1, "VAEDecodeAudio": 1,
           "CreateVideo": 1, "SaveVideo": 0, "LoadImage": 2}


def links(g):
    for node in g.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                yield v


def graph(**kw):
    base = dict(width=640, height=384, length=22, seed=3, h3t_name="m.h3t", text_encoder="te.gguf")
    return h3_colab.build_graph("a fox", **{**base, **kw})


def test_every_link_targets_an_existing_output():
    for g in (graph(), graph(text_encoder="te.safetensors", first_frame="x.png")):
        for src, slot in links(g):
            assert src in g and slot < OUTPUTS[g[src]["class_type"]], (src, slot)
        assert {n["class_type"] for n in g.values()} <= set(OUTPUTS)


def test_options_reach_the_nodes():
    g = graph(steps=6, attention="int8_fast", precision="a16")
    assert g["unet"]["inputs"] == {"h3t_name": "m.h3t", "precision": "a16", "resident_blocks": 0, "mlp_chunk": 0, "attention": "int8_fast"}
    assert g["sigmas"]["inputs"]["steps"] == 6 and g["noise"]["inputs"]["noise_seed"] == 3
    assert g["i2v"]["inputs"]["prompt"] == "a fox" and (g["i2v"]["inputs"]["width"], g["i2v"]["inputs"]["length"]) == (640, 22)
    assert g["te"]["class_type"] == "CLIPLoaderGGUF" and graph(text_encoder="te.safetensors")["te"]["class_type"] == "CLIPLoader"
    assert "first_frame" not in g["i2v"]["inputs"] and graph(first_frame="x.png")["i2v"]["inputs"]["first_frame"] == ["img", 0]


def test_canvas_must_be_a_multiple_of_32():
    with pytest.raises(ValueError):
        graph(width=500)
