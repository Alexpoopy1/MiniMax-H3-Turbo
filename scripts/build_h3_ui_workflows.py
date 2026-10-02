#!/usr/bin/env python
"""Build the ComfyUI UI workflows (frontend 1.52.x format) for MiniMax H3 on the H3-Turbo nodes.

    python scripts/build_h3_ui_workflows.py --object-info object_info.json --out-dir docs \
        [--copy-to "D:/ComfyUIBig/ComfyUI/ComfyUI/user/default/workflows"] \
        [--make-examples "C:/Users/<you>/AppData/Local/Comfy-Desktop/ComfyUI-Shared/input"]

--make-examples must point at the server's real --input-directory (Comfy Desktop: ComfyUI-Shared/input, not
ComfyUI/input) and needs example.png there; it writes two small CPU-made reference files used by the omni workflow.

Writes:
  MiniMax H3 FAST 40s - Text or Image to Video.json
  MiniMax H3 FAST - Omni Reference (images, video, audio).json
  MiniMax H3 FULL 5s 832x480 - Text or Image to Video.json

Inputs/outputs/widget lists are derived from /object_info with the same rules as scripts/validate_comfy_workflow.py,
so the files load in the frontend exactly as if they had been built by hand. No prompt is queued.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_comfy_workflow as V  # noqa: E402

FRONTEND_VERSION = "1.52.7"
H3T = "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_w4a8_convrot.h3t"
TE = "qwen3vl-32B-MiniMax-H3-Q2_K.gguf"
VAE_V = "minimax_h3_video_vae_fp16.safetensors"
VAE_A = "minimax_h3_audio_vae_fp32.safetensors"
EX_VIDEO = "h3_example_ref_video.mp4"
EX_AUDIO = "h3_example_ref_audio.wav"


class Builder:
    def __init__(self, oi):
        self.oi = oi
        self.nodes, self.links, self.groups = [], [], []
        self.nid, self.lid = 0, 0

    # ------------------------------------------------------------------ nodes
    def node(self, type_, pos, size, values=None, mode=0, title=None, color=None, bgcolor=None, autogrow=None):
        """values: {widget_name: value} (frontend-only companions such as control_after_generate included)."""
        self.nid += 1
        n = {"id": self.nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": 0, "mode": mode}
        if type_ in V.FRONTEND_ONLY:
            text = values["text"]
            if type_ == "MarkdownNote":  # show <Picture 1> etc. literally (inline code), never as an HTML tag
                text = re.sub(r"(?<!`)<((?:Picture|Video|Audio|Subject) \d+)>(?!`)", r"`<\1>`", text)
            n.update(inputs=[], outputs=[], properties={})
            n["widgets_values"] = [text]
            n["widgets_values_named"] = {"text": text}
        else:
            info = self.oi[type_]
            inputs = []
            for name, spec, optional in V.node_spec(self.oi, type_):
                t = spec[0]
                if t == "COMFY_AUTOGROW_V3":
                    tpl = V._spec_opts(spec)["template"]
                    inner = next(v for v in tpl["input"].values() if v)
                    ttype = next(iter(inner.values()))[0]
                    count = (autogrow or {}).get(name, 1)  # connected slots + the empty one the frontend keeps
                    for i in range(min(count, tpl["max"])):
                        base = f"{tpl['prefix']}{i}"
                        inputs.append({"label": base, "name": f"{name}.{base}", "shape": 7, "type": ttype, "link": None})
                elif not V._is_widget(spec):
                    d = {"name": name, "type": t, "link": None}
                    if optional:
                        d["shape"] = 7
                    inputs.append(d)
            n["inputs"] = inputs
            n["outputs"] = [{"name": nm, "type": t, "links": None} for nm, t in zip(info["output_name"], info["output"])]
            n["properties"] = {"Node name for S&R": type_}
            if info.get("python_module", "").startswith(("comfy_extras", "nodes", "comfy_api_nodes")):
                n["properties"]["cnr_id"] = "comfy-core"
            values = dict(values or {})
            widgets = V.widget_list(self.oi, type_, values, None)
            if widgets:
                wv, named = [], {}
                for w in widgets:
                    if w.kind == "preview":
                        wv.append(None)
                        continue
                    if w.name in values:
                        v = values[w.name]
                    elif w.kind == "control":
                        v = "randomize"
                    elif w.kind == "upload":
                        # the frontend's upload button holds "image" for image/video uploads, null for audio uploads
                        audio = any(x.spec is not None and V._spec_opts(x.spec).get("audio_upload") for x in widgets)
                        v = None if audio else "image"
                    else:
                        o = V._spec_opts(w.spec)
                        if "default" in o:
                            v = o["default"]
                        elif isinstance(w.spec[0], list) and w.spec[0]:
                            v = w.spec[0][0]
                        else:
                            raise ValueError(f"{type_}.{w.name}: no value and no default")
                    wv.append(v)
                    named[w.name] = v
                unknown = set(values) - {w.name for w in widgets}
                if unknown:
                    raise ValueError(f"{type_}: unknown widget values {unknown}")
                n["widgets_values"] = wv
                n["widgets_values_named"] = named
        if title:
            n["title"] = title
        if color:
            n["color"] = color
        if bgcolor:
            n["bgcolor"] = bgcolor
        self.nodes.append(n)
        return n

    # ------------------------------------------------------------------ links
    def link(self, src, out, dst, inp):
        so = next(i for i, o in enumerate(src["outputs"]) if o["name"] == out) if isinstance(out, str) else out
        ti = next(i for i, x in enumerate(dst["inputs"]) if x["name"] == inp)
        if dst["inputs"][ti]["link"] is not None:
            raise ValueError(f"input {inp} of #{dst['id']} already linked")
        self.lid += 1
        t = src["outputs"][so]["type"]
        if not V.type_ok(t, dst["inputs"][ti]["type"]):
            raise TypeError(f"{src['type']}.{out} ({t}) -> {dst['type']}.{inp} ({dst['inputs'][ti]['type']})")
        self.links.append([self.lid, src["id"], so, dst["id"], ti, t])
        dst["inputs"][ti]["link"] = self.lid
        src["outputs"][so]["links"] = (src["outputs"][so]["links"] or []) + [self.lid]

    def group(self, title, bounding, color="#3f789e"):
        self.groups.append({"id": len(self.groups) + 1, "title": title, "bounding": list(bounding), "color": color,
                            "font_size": 24, "flags": {}})

    # ------------------------------------------------------------------ output
    def _execution_order(self):
        deps = {n["id"]: set() for n in self.nodes}
        for l in self.links:
            deps[l[3]].add(l[1])
        order, done = [], set()
        while len(done) < len(deps):
            ready = sorted(i for i, d in deps.items() if i not in done and d <= done)
            if not ready:
                raise RuntimeError("cycle")
            for i in ready:
                done.add(i)
                order.append(i)
        return {i: k for k, i in enumerate(order)}

    def to_json(self, scale=0.62, offset=(560, 140)):
        order = self._execution_order()
        for n in self.nodes:
            n["order"] = order[n["id"]]
        return {
            "id": str(uuid.uuid4()),
            "revision": 0,
            "last_node_id": self.nid,
            "last_link_id": self.lid,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": scale, "offset": list(offset)}, "frontendVersion": FRONTEND_VERSION},
            "version": 0.4,
        }


# ======================================================================================================== shared parts
def models(b, x, y):
    unet = b.node("H3TurboFastUNetLoader", (x, y), (420, 200),
                  {"h3t_name": H3T, "precision": "a8", "resident_blocks": 0, "mlp_chunk": 0, "attention": "exact"},
                  title="H3-Turbo Fast UNET Loader (your turbo8 W4A8 model)")
    te = b.node("H3TurboCachedTextEncoder", (x, y + 240), (420, 130),
                {"clip_name": TE, "release": "auto", "cache": True},
                title="H3-Turbo Cached Text Encoder (Qwen3-VL Q2_K)")
    vv = b.node("VAELoader", (x, y + 410), (420, 60), {"vae_name": VAE_V}, title="Load VAE (video, fp16)")
    va = b.node("VAELoader", (x, y + 510), (420, 60), {"vae_name": VAE_A}, title="Load VAE (audio, fp32)")
    b.group("1. Models (do not change: this is what makes it fast with no quality loss)", (x - 20, y - 60, 460, 660))
    return unet, te, vv, va


def sampling_and_output(b, unet, cond_node, vv, va, x, y, prefix):
    noise = b.node("RandomNoise", (x, y), (320, 110), {"noise_seed": 42, "control_after_generate": "randomize"})
    ks = b.node("KSamplerSelect", (x, y + 150), (320, 60), {"sampler_name": "res_multistep"})
    sch = b.node("BasicScheduler", (x, y + 250), (320, 110), {"scheduler": "simple", "steps": 8, "denoise": 1.0})
    guider = b.node("BasicGuider", (x, y + 400), (320, 50))
    sca = b.node("SamplerCustomAdvanced", (x + 360, y), (300, 330))
    b.link(unet, "MODEL", sch, "model")
    b.link(unet, "MODEL", guider, "model")
    b.link(cond_node, "positive", guider, "conditioning")
    b.link(noise, "NOISE", sca, "noise")
    b.link(guider, "GUIDER", sca, "guider")
    b.link(ks, "SAMPLER", sca, "sampler")
    b.link(sch, "SIGMAS", sca, "sigmas")
    b.link(cond_node, "LATENT", sca, "latent_image")
    b.group("3. Sampling (8 turbo steps, cfg 1)", (x - 20, y - 60, 720, 560))

    ox = x + 760
    dec = b.node("VAEDecode", (ox, y), (260, 50), title="VAE Decode (video)")
    deca = b.node("VAEDecodeAudio", (ox, y + 100), (260, 50), title="VAE Decode Audio")
    cv = b.node("CreateVideo", (ox, y + 200), (300, 160), {"fps": 24.0, "bit_depth": "auto", "color_space": "sRGB", "codec": "none"})
    sv = b.node("SaveVideo", (ox + 340, y), (520, 560),
                {"filename_prefix": f"video/{prefix}", "format": "auto", "format.codec": "auto", "codec": "auto"})
    b.link(sca, "output", dec, "samples")
    b.link(vv, "VAE", dec, "vae")
    b.link(sca, "output", deca, "samples")
    b.link(va, "VAE", deca, "vae")
    b.link(dec, "IMAGE", cv, "images")
    b.link(deca, "AUDIO", cv, "audio")
    b.link(cv, "VIDEO", sv, "video")
    b.group("4. Decode + save (mp4 with sound, 24 fps)", (ox - 20, y - 60, 900, 660))


# ======================================================================================================== workflow A / C
NOTE_FAST = """# MiniMax H3 FAST - text or image to video (RTX 3050 6 GB)

**What you get:** 640x384, 22 frames (about 0.9 s at 24 fps) **with sound**. Same model, same 8 turbo steps, same sampler and exact attention as the full-size workflow: nothing is approximated, the clip is just shorter and smaller.

## Speed on this PC (RTX 3050 6 GB)
- **About 40 s per video when the prompt is cached** (you already ran this exact prompt once). Each run gets a new seed, so you still get a new video every time.
- **A new prompt costs more:** the 32B text encoder has to run once. About 8-9 s extra when its file is still in RAM (Windows file cache), about 70 s extra when it has to be read from the hard disk.
- The first run after starting ComfyUI is slower too: the models are read from disk.
- Changing even one character of the prompt, or the first/last frame picture, counts as a new prompt. The cache lives in ComfyUI/user/h3turbo_cond_cache (delete the folder to clear it).

## Text to video
Type the prompt in **MiniMax H3 Image to Video** and press Run. Describe the picture, the motion and the sound.

## Image to video (first and/or last frame)
The two **Load Image** nodes are purple = **bypassed** (switched off).
1. Click **Load Image - FIRST frame**, press **Ctrl+B** (it stops being purple).
2. Click *choose file to upload* and pick your picture.
3. Same for **Load Image - LAST frame** if you want an end frame. You can use either one or both.
4. Optional: mention it in the prompt as **<Picture 1>** (first frame) and **<Picture 2>** (last frame, when both are used).
- The first frame is stretched to 640x384, so use a picture with about the same shape (5:3 landscape). The last frame is center-cropped.
- To go back to text to video: select the Load Image node and press Ctrl+B again.

## Keep it fast
- **Close Chrome, games and other big apps.** The 12.5 GB video model streams from RAM; if Windows runs out of RAM it re-reads it from the hard disk and every run gets more than a minute slower.
- Keep: precision **a8**, attention **exact**, **8** steps, **res_multistep / simple**, BasicGuider (cfg 1), text encoder release **auto**. The ~40 s figure was measured at 640x384, 22 frames, 8 steps, exact attention.
- Want longer or bigger videos? Cost grows faster than the number of frames. Use the FULL 5 s workflow (minutes on this card) or the Colab notebook on a G4 GPU.
"""

NOTE_FULL = """# MiniMax H3 FULL - 832x480, 124 frames (about 5.2 s), with sound

Same model and settings as the FAST workflow (8 turbo steps, res_multistep / simple, cfg 1, exact attention); only the size and length are larger.

## Speed on an RTX 3050 6 GB
- **Expect roughly 6-7 minutes per video** (estimate for this card). This size is compute bound on a 6 GB card: there are many times more video tokens than in the 640x384 / 22-frame FAST workflow, and attention cost grows faster than the token count.
- **~40 s is only possible locally for short clips** (use *MiniMax H3 FAST 40s*). For full-length clips at high speed use the Colab notebook on a **G4 GPU** (RTX PRO 6000, 96 GB: the whole model stays on the GPU).
- A new prompt adds the text encoder (about 8-9 s from RAM, about 70 s from the hard disk); a repeated prompt is read from the cache.

## Text / image to video
- Text to video: type the prompt and press Run.
- Image to video: select **Load Image - FIRST frame** (and/or LAST frame), press **Ctrl+B** to switch it on (it stops being purple), upload your picture. Refer to it as **<Picture 1>** / **<Picture 2>** in the prompt if you like. The first frame is stretched to 832x480, so use a 16:9-ish picture.

## Keep it working
- Close Chrome, games and other big apps: the model streams from RAM, and running out of RAM makes every step re-read the hard disk.
- Length must be 5, 22, 39, 56, 73, 90, 107, 124, ... (17k + 5 frames at 24 fps).
"""


def build_t2v(oi, fast=True):
    b = Builder(oi)
    unet, te, vv, va = models(b, 0, 0)
    w, h, length = (640, 384, 22) if fast else (832, 480, 124)
    prompt = ("A golden retriever puppy runs across a sunny meadow toward the camera, ears flapping, tall grass swaying, "
              "warm late-afternoon light, shallow depth of field, smooth tracking shot, cinematic. "
              "Audio: happy panting, paws on grass, birdsong, a light breeze.")
    first = b.node("LoadImage", (500, 0), (320, 360), {"image": "example.png", "upload": "image"}, mode=V.MODE_BYPASS,
                   title="Load Image - FIRST frame (Ctrl+B to use)")
    last = b.node("LoadImage", (500, 400), (320, 360), {"image": "example.png", "upload": "image"}, mode=V.MODE_BYPASS,
                  title="Load Image - LAST frame (Ctrl+B to use)")
    i2v = b.node("MiniMaxH3ImageToVideo", (860, 0), (460, 520),
                 {"prompt": prompt, "width": w, "height": h, "length": length})
    b.group("2. Prompt + optional first / last frame (purple = off, Ctrl+B = on)", (480, -60, 860, 880), color="#8a6d3b")
    b.link(te, "CLIP", i2v, "clip")
    b.link(vv, "VAE", i2v, "vae")
    b.link(first, "IMAGE", i2v, "first_frame")
    b.link(last, "IMAGE", i2v, "last_frame")
    sampling_and_output(b, unet, i2v, vv, va, 1400, 0, "MiniMaxH3_fast" if fast else "MiniMaxH3_full")
    b.node("MarkdownNote", (-620, -60), (560, 980), {"text": NOTE_FAST if fast else NOTE_FULL},
           title="READ ME - how to use (speed, image to video)")
    return b.to_json()


# ======================================================================================================== workflow B
NOTE_OMNI = """# MiniMax H3 FAST - Omni Reference (images, video, audio)

One prompt + any mix of **reference pictures, a reference video and reference audio** -> 640x384, 22 frames (about 0.9 s) with sound. Same model, 8 turbo steps, exact attention.

## What is switched on
- **Reference image 1** is ON (example.png, the cartoon girl: upload your own picture there).
- Reference images 2 and 3, the reference video and the reference audio are purple = **bypassed** (off).

## Switch a reference on
- **Image 2 / Image 3:** click the Load Image node, press **Ctrl+B**, upload a picture.
- **Video:** select **Load Video**, **Video Slice** and **Get Video Components** together (hold Ctrl and drag a box around them, or Shift+click each one) and press **Ctrl+B**. Then upload your clip in Load Video. All three must be on (or all off) together.
- **Audio:** click **Load Audio**, press **Ctrl+B**, upload a sound file (voice, music or effects).

## How to name them in the prompt
Tags count only the references that are ON, top to bottom:
- Pictures: **<Picture 1>**, **<Picture 2>**, **<Picture 3>**
- Video: **<Video 1>**
- Audio: **<Audio 1>** is the reference audio. If your reference video has its own sound, that soundtrack becomes **<Audio 1>** and the Load Audio file becomes **<Audio 2>**.
You can also name people: *<Subject 1> is the woman in <Picture 1>.*

Example prompt with everything on:
*<Subject 1> is the girl in <Picture 1>. She stands in the garden from <Picture 2>, painted in the style of <Picture 3>. The camera moves like in <Video 1>. She sings along to <Audio 1>.*
(That is with a silent reference video, like the example clip. If your video has sound, the Load Audio file is <Audio 2>. With image 2 off, image 3 becomes <Picture 2>.)

## Speed (not measured yet for this workflow)
- Each reference adds tokens that go through all 8 steps, so expect somewhat more than the ~40 s of the FAST text/image workflow.
- **ref_image_size = match** (default) scales each picture down to the 640x384 area: fast. *max* keeps up to 2048 px and can be several times slower.
- A reference video is encoded at its own size up to a 768-pixel short edge, so **use a small clip (for example 640x360) to stay fast**. Only the first 22 frames (= the video length setting) are used; Video Slice keeps the first 2 s so a long HD clip does not fill your RAM. H3 reads the frames as 24 fps.
- The prompt cache also covers the references: same prompt + same pictures/video = no text encoder run.
- Close Chrome, games and other big apps to keep RAM free.

## Note on the model
The official ComfyUI reference template uses a separate *ref2va* model. This workflow uses your fused turbo model (refdelta r1024 merged in), loaded by the H3-Turbo Fast UNET Loader.
"""


def build_omni(oi):
    b = Builder(oi)
    unet, te, vv, va = models(b, 0, 0)
    imgs = []
    for k in range(3):
        imgs.append(b.node("LoadImage", (500, k * 390), (320, 350), {"image": "example.png", "upload": "image"},
                           mode=V.MODE_ALWAYS if k == 0 else V.MODE_BYPASS,
                           title="Reference image 1 = <Picture 1>" if k == 0 else
                           f"Reference image {k + 1} (Ctrl+B to use; tag = its place among the ON pictures)"))
    b.group("2a. Reference pictures (purple = off, Ctrl+B = on)", (480, -60, 360, 1230), color="#8a6d3b")

    lv = b.node("LoadVideo", (880, 0), (320, 420), {"file": EX_VIDEO, "upload": "image"}, mode=V.MODE_BYPASS,
                title="Load Video = <Video 1> (Ctrl+B to use)")
    sl = b.node("Video Slice", (880, 460), (320, 110), {"start_time": 0.0, "duration": 2.0, "strict_duration": False},
                mode=V.MODE_BYPASS, title="Video Slice (first 2 s)")
    gvc = b.node("GetVideoComponents", (880, 610), (320, 130), mode=V.MODE_BYPASS, title="Get Video Components")
    b.link(lv, "VIDEO", sl, "video")
    b.link(sl, "VIDEO", gvc, "video")
    b.group("2b. Reference video (switch all 3 nodes on together)", (860, -60, 360, 830), color="#8a6d3b")

    la = b.node("LoadAudio", (880, 860), (320, 140), {"audio": EX_AUDIO, "upload": None}, mode=V.MODE_BYPASS,
                title="Load Audio (Ctrl+B to use) = <Audio 1>, or <Audio 2> if the video has sound")
    b.group("2c. Reference audio", (860, 800, 360, 230), color="#8a6d3b")

    prompt = ("<Subject 1> is the cartoon girl in <Picture 1>: big yellow hair, round blue eyes, pink dress, drawn in the same "
              "simple crayon style as <Picture 1>. <Subject 1> waves at the camera, smiles and spins around once on the green "
              "hill under the blue sky with white clouds, her hair bouncing. Gentle camera push-in. "
              "Audio: <Subject 1> laughs and says \"Hello!\" in a cheerful child's voice, soft wind, birds chirping.")
    r2v = b.node("MiniMaxH3ReferenceToVideo", (1260, 0), (480, 620),
                 {"prompt": prompt, "width": 640, "height": 384, "length": 22, "ref_image_size": "match"},
                 autogrow={"ref_images": 4, "ref_videos": 2, "ref_video_audios": 2, "ref_audios": 2})
    b.group("2d. Prompt (name references as <Picture 1>, <Video 1>, <Audio 1>)", (1240, -60, 520, 720), color="#8a6d3b")
    b.link(te, "CLIP", r2v, "clip")
    b.link(vv, "VAE", r2v, "vae")
    b.link(va, "VAE", r2v, "audio_vae")
    for k, im in enumerate(imgs):
        b.link(im, "IMAGE", r2v, f"ref_images.ref_image_{k}")
    b.link(gvc, "images", r2v, "ref_videos.ref_video_0")
    b.link(gvc, "audio", r2v, "ref_video_audios.ref_video_audio_0")
    b.link(la, "AUDIO", r2v, "ref_audios.ref_audio_0")
    sampling_and_output(b, unet, r2v, vv, va, 1820, 0, "MiniMaxH3_omni")
    b.node("MarkdownNote", (-640, -60), (580, 1180), {"text": NOTE_OMNI}, title="READ ME - omni references (how to use)")
    return b.to_json(scale=0.55, offset=(620, 140))


# ======================================================================================================== example media
def make_examples(input_dir):
    """Small CPU-made reference media so the omni workflow runs end to end before the user uploads their own:
    a 2 s, 24 fps, 384x384 slow zoom over example.png (no sound), and a 3 s two-note melody (wav)."""
    import math
    import wave

    import av
    import numpy as np
    from PIL import Image

    src = Image.open(os.path.join(input_dir, "example.png")).convert("RGB")
    vpath = os.path.join(input_dir, EX_VIDEO)
    if not os.path.exists(vpath):
        out = av.open(vpath, "w")
        st = out.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = 384, 384, "yuv420p"
        W = src.width
        for i in range(48):
            z = 1.0 + 0.25 * i / 47
            c = W / z
            x0 = (W - c) / 2
            fr = src.crop((int(x0), int(x0 * 0.6), int(x0 + c), int(x0 * 0.6 + c))).resize((384, 384), Image.LANCZOS)
            for p in st.encode(av.VideoFrame.from_ndarray(np.asarray(fr), format="rgb24")):
                out.mux(p)
        for p in st.encode():
            out.mux(p)
        out.close()
    apath = os.path.join(input_dir, EX_AUDIO)
    if not os.path.exists(apath):
        sr, secs = 32000, 3.0
        t = np.arange(int(sr * secs)) / sr
        notes = [523.25, 659.25, 783.99, 659.25, 523.25, 659.25]
        y = np.zeros_like(t)
        seg = len(t) // len(notes)
        for k, f in enumerate(notes):
            s = slice(k * seg, (k + 1) * seg)
            tt = t[s] - t[s][0]
            y[s] = 0.3 * np.sin(2 * math.pi * f * tt) * np.minimum(1, tt * 40) * np.exp(-2.5 * tt)
        pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
        with wave.open(apath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
    return vpath, apath


FILES = {
    "MiniMax H3 FAST 40s - Text or Image to Video.json": lambda oi: build_t2v(oi, fast=True),
    "MiniMax H3 FAST - Omni Reference (images, video, audio).json": build_omni,
    "MiniMax H3 FULL 5s 832x480 - Text or Image to Video.json": lambda oi: build_t2v(oi, fast=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-info", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--copy-to", default=None)
    ap.add_argument("--make-examples", default=None, metavar="COMFY_INPUT_DIR")
    a = ap.parse_args()
    oi = V.load_object_info(a.object_info, a.server)
    if a.make_examples:
        for p in make_examples(a.make_examples):
            print("example media:", p, os.path.getsize(p), "bytes")
    os.makedirs(a.out_dir, exist_ok=True)
    for name, fn in FILES.items():
        wf = fn(oi)
        path = os.path.join(a.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=1, ensure_ascii=False)
        print("wrote", path)
        if a.copy_to:
            shutil.copy2(path, os.path.join(a.copy_to, name))
            print("copied to", os.path.join(a.copy_to, name))


if __name__ == "__main__":
    main()
