"""The Colab helper (colab/h3_colab.py) and notebook generator, tested without a GPU or ComfyUI.

* API graphs, for every task (text, first frame, first + last frame, omni references with images / videos / soundtracks /
  audio, omni + keyframes): validated against the node schemas read from a ComfyUI 0.37.4 server with the H3-Turbo and
  ComfyUI-GGUF nodes (colab/comfy_node_schemas_v0.37.4.json: required inputs, Autogrow/DynamicCombo names, combo values,
  ranges, link types); options reach the right inputs;
  the encode-only graph of the two-phase run is node-for-node identical to the full graph (that is what lets phase 2 reuse
  ComfyUI's cached conditioning instead of loading the text encoder again).
* GPU profiles and launch flags for T4 / L4 / A100 / G4, environment overrides.
* websocket client, progress tracker and timing breakdown, against a fake ComfyUI (HTTP + websocket) that also emulates the
  node cache, so generate()'s one- and two-phase orchestration runs end to end.
* file staging with ffmpeg (skipped without ffmpeg), the notebook's cells compile and match the helper's options.
"""
import base64
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_COLAB = os.path.join(_HERE, "..", "colab")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_COLAB, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


h3_colab = _load("h3_colab")

# outputs per node class in ComfyUI 0.37.4 (+ ComfyUI-GGUF, H3-Turbo); read from GET /object_info of a real 0.37.4 server
OUTPUTS = {"H3TurboFastUNetLoader": 1, "H3TurboCachedTextEncoder": 1, "H3TurboConditioningWarmup": 0, "CLIPLoaderGGUF": 1,
           "CLIPLoader": 1, "VAELoader": 1, "MiniMaxH3ImageToVideo": 2, "MiniMaxH3ReferenceToVideo": 2, "MiniMaxH3AddGuide": 1,
           "RandomNoise": 1, "KSamplerSelect": 1, "BasicScheduler": 1, "BasicGuider": 1, "SamplerCustomAdvanced": 2,
           "VAEDecode": 1, "VAEDecodeAudio": 1, "CreateVideo": 1, "SaveVideo": 1, "LoadImage": 2, "LoadVideo": 1,
           "GetVideoComponents": 5, "LoadAudio": 1}
# output types, to check that each link carries what the input expects
OUT_TYPES = {"H3TurboFastUNetLoader": ["MODEL"], "H3TurboCachedTextEncoder": ["CLIP"], "CLIPLoaderGGUF": ["CLIP"],
             "CLIPLoader": ["CLIP"], "VAELoader": ["VAE"], "MiniMaxH3ImageToVideo": ["CONDITIONING", "LATENT"],
             "MiniMaxH3ReferenceToVideo": ["CONDITIONING", "LATENT"], "MiniMaxH3AddGuide": ["CONDITIONING"],
             "RandomNoise": ["NOISE"], "KSamplerSelect": ["SAMPLER"], "BasicScheduler": ["SIGMAS"], "BasicGuider": ["GUIDER"],
             "SamplerCustomAdvanced": ["LATENT", "LATENT"], "VAEDecode": ["IMAGE"], "VAEDecodeAudio": ["AUDIO"],
             "CreateVideo": ["VIDEO"], "LoadImage": ["IMAGE", "MASK"], "LoadVideo": ["VIDEO"],
             "GetVideoComponents": ["IMAGE", "AUDIO", "FLOAT", "COMBO", "COMBO"], "LoadAudio": ["AUDIO"]}
IN_TYPES = {"clip": "CLIP", "vae": "VAE", "audio_vae": "VAE", "model": "MODEL", "conditioning": "CONDITIONING",
            "positive": "CONDITIONING", "latent": "LATENT", "latent_image": "LATENT", "samples": "LATENT", "noise": "NOISE",
            "guider": "GUIDER", "sampler": "SAMPLER", "sigmas": "SIGMAS", "images": "IMAGE", "image": "IMAGE",
            "first_frame": "IMAGE", "last_frame": "IMAGE", "audio": "AUDIO", "video": "VIDEO"}
BASE = dict(width=640, height=384, length=22, seed=3, h3t_name="m.h3t", text_encoder="te.gguf")
VARIANTS = {
    "t2v": {},
    "i2v": dict(first_frame="f.png"),
    "flf2v": dict(first_frame="f.png", last_frame="l.png"),
    "omni": dict(ref_images=["a.png", "b.png"], ref_videos=[("v0.mp4", True), "v1.mp4"], ref_audios=["voice.wav"]),
    "omni_img_only": dict(ref_images=["a.png"]),
    "omni_keyframes": dict(ref_images=["a.png"], first_frame="f.png", last_frame="l.png"),
    "stock_te": dict(te_loader="stock", text_encoder="te.safetensors"),
}


def graph(phase="full", **kw):
    return h3_colab.build_graph("a fox", phase=phase, **{**BASE, **kw})


def links(g):
    for nid, node in g.items():
        for name, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                yield nid, name, v


# ----------------------------------------------------------------------------------------------- graphs
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("phase", ["full", "encode"])
def test_every_link_targets_an_existing_output_of_the_right_type(variant, phase):
    g = graph(phase, **VARIANTS[variant])
    assert {n["class_type"] for n in g.values()} <= set(OUTPUTS)
    for nid, name, (src, slot) in links(g):
        assert src in g, (nid, name, src)
        cls = g[src]["class_type"]
        assert slot < OUTPUTS[cls], (nid, name, src, slot)
        want = IN_TYPES.get(name)
        if name.startswith("ref_images."):
            want = "IMAGE"
        elif name.startswith("ref_videos."):
            want = "IMAGE"  # GetVideoComponents' frames
        elif name.startswith(("ref_video_audios.", "ref_audios.")):
            want = "AUDIO"
        if want:
            assert OUT_TYPES[cls][slot] == want, (nid, name, cls, slot)


SCHEMAS = json.load(open(os.path.join(_COLAB, "comfy_node_schemas_v0.37.4.json"), encoding="utf-8"))["nodes"]
FILE_COMBOS = {"image", "file", "audio", "clip_name", "vae_name", "h3t_name"}  # their options are the files present


def _spec_inputs(spec_inputs, values, prefix=""):
    """name -> (type, options, required), with Autogrow inputs and the selected DynamicCombo children expanded and named the
    way ComfyUI's backend names them ("ref_images.ref_image_0", "format.codec")."""
    out = {}
    for sect in ("required", "optional"):
        for name, s in (spec_inputs.get(sect) or {}).items():
            t, opts, full = s[0], (s[1] if len(s) > 1 and isinstance(s[1], dict) else {}), prefix + name
            if t == "COMFY_AUTOGROW_V3":
                tpl = opts["template"]
                (tspec,) = [v for d in tpl["input"].values() for v in d.values()]
                for i in range(tpl["max"]):
                    out[f"{full}.{tpl['prefix']}{i}"] = (tspec[0], tspec[1] if len(tspec) > 1 else {}, False)
            elif t == "COMFY_DYNAMICCOMBO_V3":
                out[full] = ("DYNCOMBO", opts, sect == "required")
                for o in opts.get("options", []):
                    if o["key"] == values.get(full):
                        out.update(_spec_inputs(o.get("inputs") or {}, values, full + "."))
            else:
                out[full] = (t, opts, sect == "required")
    return out


def schema_errors(g):
    errs = []
    for nid, node in g.items():
        cls, vals = node["class_type"], node["inputs"]
        spec = _spec_inputs(SCHEMAS[cls]["input"], vals)
        errs += [f"{nid}: missing required {k}" for k, (_, _, req) in spec.items() if req and k not in vals]
        for k, v in vals.items():
            if k not in spec:
                errs.append(f"{nid}: {cls} has no input {k}")
                continue
            t, opts, _ = spec[k]
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                got = SCHEMAS[g[v[0]]["class_type"]]["output"][v[1]]
                if got != t:
                    errs.append(f"{nid}.{k}: expects {t}, linked to {got}")
            elif t in ("INT", "FLOAT"):
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not opts.get("min", v) <= v <= opts.get("max", v):
                    errs.append(f"{nid}.{k}: {v!r} not a valid {t}")
            elif t == "BOOLEAN" and not isinstance(v, bool) or t == "STRING" and not isinstance(v, str):
                errs.append(f"{nid}.{k}: {v!r} is not a {t}")
            elif (t == "COMBO" or isinstance(t, list)) and k not in FILE_COMBOS and v not in (t if isinstance(t, list) else opts["options"]):
                errs.append(f"{nid}.{k}: {v!r} is not an option")
            elif t == "DYNCOMBO" and v not in [o["key"] for o in opts["options"]]:
                errs.append(f"{nid}.{k}: {v!r} is not an option")
    return errs


@pytest.mark.parametrize("variant", sorted(VARIANTS) + ["omni_max", "full_832"])
@pytest.mark.parametrize("phase", ["full", "encode"])
def test_graphs_validate_against_comfyui_0_37_4_node_schemas(variant, phase):
    extra = {"omni_max": dict(ref_images=["a.png"] * 9, ref_videos=[("v.mp4", True)] * 3, ref_audios=["x.wav"] * 3),
             "full_832": dict(width=832, height=480, length=124, attention="int8_fast", precision="a16")}
    assert schema_errors(graph(phase, **{**VARIANTS.get(variant, {}), **extra.get(variant, {})})) == []


def test_schema_check_catches_mistakes():
    g = graph(ref_images=["a.png"])
    g["ksel"]["inputs"]["sampler_name"] = "nope"
    g["cond"]["inputs"]["ref_images.ref_image_9"] = ["ref_img0", 0]
    g["dec"]["inputs"]["vae"] = ["te", 0]
    del g["cond"]["inputs"]["length"]
    del g["save"]["inputs"]["format.codec"]
    errs = schema_errors(g)
    assert len(errs) == 5, errs


def test_text_to_video_matches_the_users_workflow():
    g = graph(steps=6, attention="int8_fast", precision="a16")
    assert g["unet"]["inputs"] == {"h3t_name": "m.h3t", "precision": "a16", "resident_blocks": 0, "mlp_chunk": 0, "attention": "int8_fast"}
    assert g["te"] == {"class_type": "H3TurboCachedTextEncoder", "inputs": {"clip_name": "te.gguf", "release": "auto", "cache": True}}
    assert g["cond"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert {k: g["cond"]["inputs"][k] for k in ("prompt", "width", "height", "length")} == {"prompt": "a fox", "width": 640, "height": 384, "length": 22}
    assert "first_frame" not in g["cond"]["inputs"] and "last_frame" not in g["cond"]["inputs"]
    assert g["ksel"]["inputs"]["sampler_name"] == "res_multistep"
    assert g["sigmas"]["inputs"] == {"scheduler": "simple", "steps": 6, "denoise": 1.0, "model": ["unet", 0]}
    assert g["noise"]["inputs"]["noise_seed"] == 3
    assert g["guider"]["inputs"]["conditioning"] == ["cond", 0] and g["sample"]["inputs"]["latent_image"] == ["cond", 1]
    assert g["vvae"]["inputs"]["vae_name"] == "minimax_h3_video_vae_fp16.safetensors"
    assert g["avae"]["inputs"]["vae_name"] == "minimax_h3_audio_vae_fp32.safetensors"
    assert g["video"]["inputs"]["fps"] == 24 and g["video"]["inputs"]["audio"] == ["adec", 0]
    assert g["save"]["inputs"]["format"] == "auto" and g["save"]["inputs"]["format.codec"] == "auto"


def test_frames_and_text_encoder_options():
    g = graph(first_frame="f.png", last_frame="l.png", te_release="after_encode", te_cache=False)
    assert g["cond"]["inputs"]["first_frame"] == ["img_first", 0] and g["cond"]["inputs"]["last_frame"] == ["img_last", 0]
    assert g["img_first"]["inputs"] == {"image": "f.png"} and g["img_last"]["inputs"] == {"image": "l.png"}
    assert g["te"]["inputs"]["release"] == "after_encode" and g["te"]["inputs"]["cache"] is False
    assert graph(te_loader="stock")["te"]["class_type"] == "CLIPLoaderGGUF"
    assert graph(te_loader="stock", text_encoder="x.safetensors")["te"] == {"class_type": "CLIPLoader", "inputs": {"clip_name": "x.safetensors", "type": "minimax"}}


def test_omni_reference_autogrow_inputs_and_soundtrack_pairing():
    g = graph(**VARIANTS["omni"], ref_image_size="max")
    c = g["cond"]
    assert c["class_type"] == "MiniMaxH3ReferenceToVideo" and c["inputs"]["ref_image_size"] == "max"
    assert c["inputs"]["vae"] == ["vvae", 0] and c["inputs"]["audio_vae"] == ["avae", 0]
    assert c["inputs"]["ref_images.ref_image_0"] == ["ref_img0", 0] and c["inputs"]["ref_images.ref_image_1"] == ["ref_img1", 0]
    assert g["ref_img1"]["inputs"] == {"image": "b.png"}
    assert c["inputs"]["ref_videos.ref_video_0"] == ["ref_vid0_parts", 0] and c["inputs"]["ref_videos.ref_video_1"] == ["ref_vid1_parts", 0]
    # the node pairs ref_video_audio_N with ref_video_N: only video 0 asked for its soundtrack
    assert c["inputs"]["ref_video_audios.ref_video_audio_0"] == ["ref_vid0_parts", 1]
    assert "ref_video_audios.ref_video_audio_1" not in c["inputs"]
    assert g["ref_vid0"] == {"class_type": "LoadVideo", "inputs": {"file": "v0.mp4"}}
    assert g["ref_vid0_parts"] == {"class_type": "GetVideoComponents", "inputs": {"video": ["ref_vid0", 0]}}
    assert c["inputs"]["ref_audios.ref_audio_0"] == ["ref_aud0", 0] and g["ref_aud0"]["inputs"] == {"audio": "voice.wav"}
    # no audio reference -> the audio VAE is not an input of the conditioning node
    assert "audio_vae" not in graph(ref_images=["a.png"])["cond"]["inputs"]


def test_omni_with_keyframes_chains_add_guide():
    g = graph(**VARIANTS["omni_keyframes"])
    assert g["guide_first"]["inputs"] == {"positive": ["cond", 0], "vae": ["vvae", 0], "latent": ["cond", 1], "image": ["img_first", 0], "frame_idx": 0}
    assert g["guide_last"]["inputs"]["positive"] == ["guide_first", 0] and g["guide_last"]["inputs"]["frame_idx"] == -1
    assert g["guider"]["inputs"]["conditioning"] == ["guide_last", 0]
    assert "first_frame" not in g["cond"]["inputs"]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_encode_graph_is_an_identical_subset_plus_warmup(variant):
    full, enc = graph("full", **VARIANTS[variant]), graph("encode", **VARIANTS[variant])
    warm = enc.pop("warm")
    assert warm["class_type"] == "H3TurboConditioningWarmup"
    assert warm["inputs"]["conditioning"] == full["guider"]["inputs"]["conditioning"]  # the final positive, guides included
    for nid, node in enc.items():
        assert full[nid] == node, nid  # same id, same class, same inputs: ComfyUI's cache serves it in phase 2
    sampling = {"unet", "noise", "ksel", "sigmas", "guider", "sample", "dec", "adec", "video", "save"}
    assert not sampling & set(enc)
    # everything the conditioning needs is in the encode graph (closed under links)
    for nid, _, (src, _) in links(enc):
        assert src in enc, (nid, src)


def test_validation():
    with pytest.raises(ValueError):
        graph(width=500)
    with pytest.raises(ValueError):
        graph(ref_images=[f"{i}.png" for i in range(10)])
    with pytest.raises(ValueError):
        graph(ref_videos=["a.mp4"] * 4)
    with pytest.raises(ValueError):
        graph(attention="fast")
    with pytest.raises(ValueError):
        graph(phase="half")
    assert len([k for k in graph(ref_images=[f"{i}.png" for i in range(9)])["cond"]["inputs"] if k.startswith("ref_images.")]) == 9


def test_geometry_mirrors_the_h3_nodes():
    assert [h3_colab.snap_length(n) for n in (1, 5, 6, 22, 23, 124, 125)] == [5, 5, 22, 22, 39, 124, 141]
    assert h3_colab.adapt_canvas(1920, 1080) == (1344, 768) and h3_colab.adapt_canvas(1080, 1920) == (768, 1344)
    assert h3_colab.adapt_canvas(768, 768) == (768, 768)
    assert h3_colab.ref_video_canvas(1920, 1080) == (1344, 768)  # downscaled to the canvas
    assert h3_colab.ref_video_canvas(640, 360) == (640, 352)  # smaller than the canvas: never upscaled, rounded to 32
    assert h3_colab.fit32(500) == 512 and h3_colab.fit32(10) == 32
    assert h3_colab.match_aspect(1080, 1920, 640, 384) == (384, 672)  # portrait image, about the same pixel budget


def test_reference_tags_follow_the_nodes_numbering():
    tags = h3_colab.reference_tags(["/u/cat.png", "dog.png"], [("/u/dance.mp4", True), "walk.mp4", ("talk.mp4", True)], ["voice.wav"])
    assert tags == ["<Picture 1> = cat.png", "<Picture 2> = dog.png",
                    "<Audio 1> = soundtrack of <Video 1>", "<Video 1> = dance.mp4", "<Video 2> = walk.mp4",
                    "<Audio 2> = soundtrack of <Video 3>", "<Video 3> = talk.mp4", "<Audio 3> = voice.wav"]


def test_resolve_task():
    assert h3_colab.resolve_task() == "t2v" and h3_colab.resolve_task("f") == "i2v" and h3_colab.resolve_task("f", "l") == "flf2v"
    assert h3_colab.resolve_task(ref_audios=["a"]) == "omni" and h3_colab.resolve_task("f", ref_images=["x"]) == "omni"


# ----------------------------------------------------------------------------------------------- hardware and profiles
T4 = h3_colab.Hardware("Tesla T4", 15.0, (7, 5), "550.54.15", 12.7, 80)
T4_HIGHRAM = h3_colab.Hardware("Tesla T4", 15.0, (7, 5), "550.54.15", 51.0, 80)
L4 = h3_colab.Hardware("NVIDIA L4", 22.5, (8, 9), "550.54.15", 53.0, 200)
A100_40 = h3_colab.Hardware("NVIDIA A100-SXM4-40GB", 40.0, (8, 0), "580.65.06", 83.5, 200)
G4 = h3_colab.Hardware("NVIDIA RTX PRO 6000 Blackwell Server Edition", 95.6, (12, 0), "580.82.07", 176.0, 300)


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("H3_COLAB_UNET_DTYPE", "H3_COLAB_COMFY_ARGS", "H3_COLAB_FREE_MEMORY"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_nvidia_smi_parsing():
    assert h3_colab.parse_gpu_query("Tesla T4, 15360, 7.5, 550.54.15") == ("Tesla T4", 15.0, (7, 5), "550.54.15")
    name, vram, cc, drv = h3_colab.parse_gpu_query("NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0, 580.82.07\n")
    assert name.startswith("NVIDIA RTX PRO 6000") and cc == (12, 0) and abs(vram - 95.6) < 0.1 and drv == "580.82.07"


def test_profiles(clean_env):
    p = h3_colab.choose_profile(T4)
    assert p.name == "t4" and p.two_phase and p.te_release == "after_encode" and p.text_encoder == h3_colab.TE_Q2
    assert p.comfy_args == ["--disable-dynamic-vram", "--disable-pinned-memory"] and "--highvram" not in p.comfy_args
    assert p.free_payload == {"unload_models": True, "free_memory": False} and p.env == {"MALLOC_ARENA_MAX": "2"}
    p = h3_colab.choose_profile(T4_HIGHRAM)
    assert p.name == "t4" and not p.two_phase and p.te_release == "auto" and p.comfy_args == ["--disable-dynamic-vram"]
    for hw, te in ((L4, h3_colab.TE_Q2), (A100_40, h3_colab.TE_Q4), (G4, h3_colab.TE_Q4)):
        p = h3_colab.choose_profile(hw)
        assert p.name == "big" and not p.two_phase and p.comfy_args == ["--highvram"] and p.text_encoder == te, hw.gpu
    assert h3_colab.choose_profile(G4, text_encoder="Q2_K (same as the PC, 8.5 GB)").text_encoder == h3_colab.TE_Q2
    assert h3_colab.choose_profile(T4, text_encoder="Q4_K_M").text_encoder == h3_colab.TE_Q4
    assert h3_colab.choose_profile(G4, profile="t4").two_phase is False  # 176 GB RAM: no need to split
    assert h3_colab.choose_profile(T4, profile="big").comfy_args == ["--highvram"]
    with pytest.raises(ValueError):
        h3_colab.choose_profile(T4, profile="huge")


def test_profile_overrides(clean_env):
    clean_env.setenv("H3_COLAB_UNET_DTYPE", "fp16")
    clean_env.setenv("H3_COLAB_COMFY_ARGS", "--reserve-vram 1.5")
    clean_env.setenv("H3_COLAB_FREE_MEMORY", "1")
    p = h3_colab.choose_profile(T4)
    assert p.comfy_args[-3:] == ["--fp16-unet", "--reserve-vram", "1.5"] and p.free_payload["free_memory"] is True
    assert any("fp16" in n for n in p.notes)
    assert "--fp32-unet" in h3_colab.choose_profile(G4, unet_dtype="fp32").comfy_args
    with pytest.raises(ValueError):
        h3_colab.choose_profile(G4, unet_dtype="int4")


def test_pick_text_encoder():
    assert h3_colab.pick_text_encoder("auto", big=True, vram_gib=95) == h3_colab.TE_Q4
    assert h3_colab.pick_text_encoder("auto", big=True, vram_gib=22.5) == h3_colab.TE_Q2
    assert h3_colab.pick_text_encoder("auto", big=False, vram_gib=95) == h3_colab.TE_Q2
    assert h3_colab.pick_text_encoder("my.gguf", big=True, vram_gib=95) == "my.gguf"


def test_startup_log_parsing():
    log = ("[INFO] pytorch version: 2.12.1+cu130\n[INFO] Set vram state to: HIGH_VRAM\n"
           "[INFO] Found comfy_kitchen backend cuda: {'available': True, 'disabled': False, 'unavailable_reason': None, "
           "'capabilities': ['int8_linear', 'w4a8_int8_linear']}\n")
    assert h3_colab.parse_startup_log(log) == {"torch": "2.12.1+cu130", "vram_state": "HIGH_VRAM", "dynamic_vram": False, "kitchen_cuda": True}
    off = h3_colab.parse_startup_log("Found comfy_kitchen backend cuda: {'available': True, 'disabled': True, 'capabilities': ['w4a8_int8_linear']}\n"
                                     "DynamicVRAM support detected and enabled")
    assert off["kitchen_cuda"] is False and off["dynamic_vram"] is True
    assert h3_colab.parse_startup_log("")["kitchen_cuda"] is None


# ----------------------------------------------------------------------------------------------- websocket
def _accept(key):
    return base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()


def test_websocket_frames_roundtrip():
    WS = h3_colab.WebSocket
    for n in (0, 5, 125, 126, 300, 65535, 65536, 70000):
        payload = os.urandom(n)
        for mask in (False, True):
            fr = WS.build_frame(0x2, payload, mask=mask)
            assert WS.parse_frame(fr) == (True, 0x2, payload, len(fr))
            assert WS.parse_frame(fr[:-1] if n else fr[:1]) is None  # incomplete


def test_websocket_client_against_a_raw_server():
    WS = h3_colab.WebSocket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    got = {}

    def serve():
        c, _ = srv.accept()
        req = b""
        while b"\r\n\r\n" not in req:
            req += c.recv(4096)
        key = next(ln.split(":", 1)[1].strip() for ln in req.decode().split("\r\n") if ln.lower().startswith("sec-websocket-key"))
        c.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Accept: {_accept(key)}\r\n\r\n").encode() + WS.build_frame(0x1, b'{"a": 1}', mask=False))
        big = json.dumps({"x": "y" * 1000}).encode()
        c.sendall(WS.build_frame(0x1, big, mask=False))
        c.sendall(WS.build_frame(0x9, b"hi", mask=False))  # ping: the client must answer with a masked pong
        c.sendall(WS.build_frame(0x1, b"frag", mask=False, fin=False) + WS.build_frame(0x0, b"mented", mask=False))
        c.sendall(WS.build_frame(0x2, b"\x00\x01", mask=False))
        buf = b""
        while True:
            fr = WS.parse_frame(buf)
            if fr:
                got["pong"] = fr
                break
            buf += c.recv(4096)
        time.sleep(0.3)
        c.sendall(WS.build_frame(0x8, b"", mask=False))
        time.sleep(0.2)
        c.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    ws = WS.connect("127.0.0.1", port, "/ws?clientId=x", timeout=5)
    assert ws.recv(5) == '{"a": 1}'
    assert json.loads(ws.recv(5))["x"] == "y" * 1000
    assert ws.recv(5) == "fragmented"
    assert ws.recv(5) == b"\x00\x01"
    with pytest.raises(ConnectionError):
        ws.recv(5)
    t.join(5)
    srv.close()
    assert got["pong"][1] == 0xA and got["pong"][2] == b"hi"


# ----------------------------------------------------------------------------------------------- tracker / timings
def test_tracker_and_breakdown():
    g = graph()
    tr = h3_colab.Tracker("p1", g, 0.0, echo=False)
    ev = [(0.1, "execution_start", {}), (0.1, "execution_cached", {"nodes": ["te", "vvae"]}),
          (0.2, "executing", {"node": "cond"}), (5.2, "executing", {"node": "unet"}), (5.3, "executing", {"node": "sample"}),
          (13.3, "progress", {"node": "sample", "value": 1, "max": 3}), (16.3, "progress", {"node": "sample", "value": 2, "max": 3}),
          (19.3, "progress", {"node": "sample", "value": 3, "max": 3}), (19.4, "executing", {"node": "dec"}),
          (31.4, "executing", {"node": "adec"}), (32.4, "executing", {"node": "video"}), (32.5, "executing", {"node": "save"}),
          (33.5, "execution_success", {}), (33.6, "executing", {"node": None})]
    tr.feed({"type": "executing", "data": {"node": "cond", "prompt_id": "other"}}, 0.15)  # another prompt: ignored
    for t, typ, data in ev:
        tr.feed({"type": typ, "data": dict(data, prompt_id="p1")}, t)
    assert tr.done and tr.error is None and tr.cached == ["te", "vvae"]
    run = h3_colab.PromptRun("generate", "p1", g, 33.5, tr.node_seconds, tr.node_start, tr.cached, tr.steps, {}, {})
    cats = run.categories()
    assert cats["text encode"] == pytest.approx(5.0) and cats["sampler"] == pytest.approx(14.1)
    assert cats["decode"] == pytest.approx(13.0) and cats["save"] == pytest.approx(1.1)
    assert run.sampler_detail() == "3 steps: first 8.0 s (includes putting the DiT on the GPU), then 3.00 s/step"
    err = h3_colab.Tracker("p2", g, 0.0, echo=False)
    err.feed({"type": "execution_error", "data": {"prompt_id": "p2", "exception_message": "boom"}}, 1.0)
    assert err.done and err.error["exception_message"] == "boom"


# ----------------------------------------------------------------------------------------------- fake ComfyUI
class FakeComfy(ThreadingHTTPServer):
    """HTTP + websocket stand-in for ComfyUI 0.37.4: /prompt, /history, /free, /system_stats, /ws. Emulates the node cache
    (a node whose id, inputs and upstream nodes match an earlier run is reported as cached and not 'executed')."""

    daemon_threads = True

    def __init__(self, comfy_dir):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.comfy_dir, self.clients, self.history, self.cache = comfy_dir, {}, {}, {}
        self.prompts, self.frees, self.lock, self.counter = [], [], threading.Lock(), 0

    def send(self, cid, msg):
        c = self.clients.get(cid)
        if c:
            conn, lock = c
            with lock:
                try:
                    conn.sendall(h3_colab.WebSocket.build_frame(0x1, json.dumps(msg).encode(), mask=False))
                except OSError:
                    pass

    def execute(self, pid, cid, g):
        def send(typ, **data):
            self.send(cid, {"type": typ, "data": dict(data, prompt_id=pid)})

        sig = {}

        def signature(n):  # like ComfyUI's input signature: class + constants + the signatures of linked nodes
            if n not in sig:
                d = g[n]
                ins = {k: (["link", signature(v[0]), v[1]] if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) else v)
                       for k, v in d["inputs"].items()}
                sig[n] = json.dumps([n, d["class_type"], ins], sort_keys=True)
            return sig[n]

        for n in g:
            signature(n)
        cached = [n for n in g if self.cache.get(n) == sig[n]]
        send("execution_start")
        send("execution_cached", nodes=cached)
        fail = "FAIL" in json.dumps(g)
        for n, d in g.items():
            if n in cached:
                continue
            send("executing", node=n, display_node=n)
            time.sleep(0.01)
            if d["class_type"] == "SamplerCustomAdvanced":
                if fail:
                    self.history[pid] = {"status": {"status_str": "error", "completed": False,
                                                    "messages": [["execution_error", {"exception_message": "boom"}]]}, "outputs": {}}
                    send("execution_error", node_id=n, exception_message="boom")
                    return
                steps = g["sigmas"]["inputs"]["steps"]
                for i in range(1, steps + 1):
                    time.sleep(0.01)
                    send("progress", value=i, max=steps, node=n)
            self.cache[n] = sig[n]
        outputs = {}
        if "save" in g:
            sub = os.path.join(self.comfy_dir, "output", "video")
            os.makedirs(sub, exist_ok=True)
            self.counter += 1
            name = f"H3Turbo_{self.counter:05d}_.mp4"
            open(os.path.join(sub, name), "wb").write(b"\0" * 16)
            outputs["save"] = {"images": [{"filename": name, "subfolder": "video", "type": "output"}], "animated": [True]}
        self.history[pid] = {"status": {"status_str": "success", "completed": True, "messages": []}, "outputs": outputs}
        send("execution_success")
        send("executing", node=None)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/ws":
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", _accept(self.headers["Sec-WebSocket-Key"]))
            self.end_headers()
            self.wfile.flush()
            cid = parse_qs(u.query)["clientId"][0]
            self.server.clients[cid] = (self.connection, threading.Lock())
            buf = b""
            try:
                while True:
                    chunk = self.connection.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    fr = h3_colab.WebSocket.parse_frame(buf)
                    if fr and fr[1] == 0x8:
                        break
            except OSError:
                pass
            self.server.clients.pop(cid, None)
            self.close_connection = True
        elif u.path == "/system_stats":
            self._json({"system": {}, "devices": []})
        elif u.path.startswith("/history/"):
            pid = u.path.rsplit("/", 1)[1]
            self._json({pid: self.server.history[pid]} if pid in self.server.history else {})
        else:
            self._json({}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/prompt":
            pid = f"p{len(self.server.prompts)}"
            self.server.prompts.append(body["prompt"])
            self._json({"prompt_id": pid, "number": len(self.server.prompts), "node_errors": {}})
            threading.Thread(target=self.server.execute, args=(pid, body["client_id"], body["prompt"]), daemon=True).start()
        elif self.path == "/free":
            self.server.frees.append(body)
            if body.get("free_memory"):
                self.server.cache.clear()
            self._json({})
        elif self.path == "/interrupt":
            self._json({})
        else:
            self._json({}, 404)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(h3_colab, "gpu_memory_mib", lambda: (None, None))  # never query the GPU from the tests
    srv = FakeComfy(str(tmp_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _client(fake, prof):
    return h3_colab.ComfyServer.attach(fake.comfy_dir, fake.server_address[1], profile=prof)


def test_generate_two_phase_on_the_t4_profile(fake, clean_env, capsys):
    srv = _client(fake, h3_colab.choose_profile(T4))
    res = srv_gen = h3_colab.generate(srv, "a fox", h3t_name="m.h3t", seed=1, steps=3)
    assert os.path.exists(res.path) and res.path.endswith(".mp4") and os.fspath(res) == res.path
    enc, full = fake.prompts
    assert "warm" in enc and "sample" not in enc and enc["te"]["inputs"]["release"] == "after_encode"
    assert fake.frees == [{"unload_models": True, "free_memory": False}]
    assert [r.label for r in res.runs] == ["phase 1 encode", "phase 2 generate"]
    p2 = res.runs[1]
    assert set(p2.cached) >= {"te", "vvae", "cond"}  # phase 2 reuses phase 1's nodes: the encoder is not loaded again
    assert "te" not in p2.node_seconds and "sample" in p2.node_seconds and p2.sampler_detail().startswith("3 steps")
    rep = srv_gen.report()
    assert "phase 1 encode" in rep and "free memory between the phases" in rep and "sampler (DiT)" in rep
    # same conditioning, new seed: one pass
    res2 = h3_colab.generate(srv, "a fox", h3t_name="m.h3t", seed=2, steps=3)
    assert len(fake.prompts) == 3 and len(res2.runs) == 1 and len(fake.frees) == 1
    assert "same conditioning" in capsys.readouterr().out
    # new prompt: two phases again
    h3_colab.generate(srv, "a cat", h3t_name="m.h3t", seed=2, steps=3)
    assert len(fake.prompts) == 5 and len(fake.frees) == 2


def test_generate_single_pass_on_the_big_profile_and_failures(fake, clean_env, monkeypatch):
    srv = _client(fake, h3_colab.choose_profile(G4))
    res = h3_colab.generate(srv, "a fox", h3t_name="m.h3t", width=630, length=20, steps=2, echo=False)
    (g,) = fake.prompts
    assert len(res.runs) == 1 and not fake.frees and g["te"]["inputs"]["clip_name"] == h3_colab.TE_Q4
    assert (g["cond"]["inputs"]["width"], g["cond"]["inputs"]["length"]) == (640, 22)  # snapped to 32 and to 17k+5
    srv.info["kitchen_cuda"] = False
    h3_colab.generate(srv, "a fox", h3t_name="m.h3t", attention="int8_fast", steps=2, echo=False)
    assert fake.prompts[-1]["unet"]["inputs"]["attention"] == "exact"  # no kitchen kernels: falls back instead of failing
    with pytest.raises(RuntimeError, match="boom"):
        h3_colab.generate(srv, "FAIL", h3t_name="m.h3t", steps=2, echo=False)
    # websocket unavailable: polling /history still completes the run
    monkeypatch.setattr(h3_colab.WebSocket, "connect", classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(OSError("no ws"))))
    res = h3_colab.generate(srv, "a dog", h3t_name="m.h3t", steps=2, echo=False)
    assert os.path.exists(res.path)


def _make_png(path, size=(64, 96)):
    pil = pytest.importorskip("PIL.Image")
    pil.new("RGB", size, (200, 30, 30)).save(path)
    return path


def test_generate_image_to_video_and_omni_with_staged_files(fake, clean_env, tmp_path):
    srv = _client(fake, h3_colab.choose_profile(G4))
    img = _make_png(str(tmp_path / "first.png"), (1080, 1920))
    res = h3_colab.generate(srv, "a fox", h3t_name="m.h3t", image=img, steps=2, echo=False)
    g = fake.prompts[-1]
    assert (g["cond"]["inputs"]["width"], g["cond"]["inputs"]["height"]) == (384, 672)  # canvas follows the portrait frame
    staged = g["img_first"]["inputs"]["image"]
    assert staged.startswith("h3c_") and os.path.exists(os.path.join(srv.input_dir, staged))
    assert res.settings["first_frame"] == staged
    res = h3_colab.generate(srv, "<Picture 1> walks", h3t_name="m.h3t", ref_images=[img, img], steps=2, echo=False)
    g = fake.prompts[-1]
    assert g["cond"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert g["ref_img0"]["inputs"]["image"] == g["ref_img1"]["inputs"]["image"] == staged  # identical file, identical name


def test_resolve_files(tmp_path):
    up = tmp_path / "uploads"
    up.mkdir()
    (up / "a.png").write_bytes(b"x")
    (up / "b.png").write_bytes(b"y")
    other = tmp_path / "c.wav"
    other.write_bytes(b"z")
    got = h3_colab.resolve_files(f"a.png, {other}\n b.png", uploads=str(up))
    assert got == [str(up / "a.png"), str(other), str(up / "b.png")]
    assert h3_colab.resolve_files("*.png", uploads=str(up)) == [str(up / "a.png"), str(up / "b.png")]
    assert h3_colab.resolve_files("", uploads=str(up)) == []
    with pytest.raises(FileNotFoundError, match="a.png, b.png"):
        h3_colab.resolve_files("missing.png", uploads=str(up))


# ----------------------------------------------------------------------------------------------- staging with ffmpeg
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg/ffprobe")


@needs_ffmpeg
def test_stage_reference_video_and_audio(tmp_path):
    src = str(tmp_path / "clip.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", src], check=True)
    inp = str(tmp_path / "input")
    name, has_audio = h3_colab.stage_ref_video(src, inp, frames=22)
    assert has_audio and name.endswith(".mp4")
    info = h3_colab.probe_media(os.path.join(inp, name))
    assert (info["width"], info["height"]) == h3_colab.ref_video_canvas(1280, 720) == (1280, 704)
    assert abs(info["fps"] - 24) < 1e-6 and info["has_audio"]
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", os.path.join(inp, name)], capture_output=True, text=True).stdout.strip())
    assert n == 22
    assert h3_colab.stage_ref_video(src, inp, frames=22) == (name, True)  # idempotent, same name
    silent, a = h3_colab.stage_ref_video(src, inp, frames=22, keep_audio=False)
    assert not a and not h3_colab.probe_media(os.path.join(inp, silent))["has_audio"]
    wav = h3_colab.stage_audio(src, inp, max_seconds=1.5)
    winfo = h3_colab.probe_media(os.path.join(inp, wav))
    assert wav.endswith(".wav") and winfo["has_audio"] and not winfo["has_video"] and 1.4 < winfo["duration"] < 1.6


# ----------------------------------------------------------------------------------------------- notebook
def test_notebook_cells_compile_and_match_the_helper():
    mk = _load("make_notebook")
    nb = mk.build_notebook()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        compile(src, "<cell>", "exec")
    gen = next(s for s in code if "run_task(" in s)
    for task in h3_colab.TASKS:
        assert json.dumps(task) in gen
    for preset in h3_colab.PRESETS:
        assert json.dumps(preset) in gen
    assert nb["metadata"]["accelerator"] == "GPU"
    with open(os.path.join(_COLAB, "H3_Turbo_Colab.ipynb"), encoding="utf-8") as f:
        assert json.load(f) == nb, "regenerate the notebook: python colab/make_notebook.py"
