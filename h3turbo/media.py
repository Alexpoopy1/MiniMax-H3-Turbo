"""Media I/O with no hard dependencies: PNG and WAV are written with the stdlib.
mp4 needs PyAV (`pip install av`); reading images needs Pillow."""
from __future__ import annotations

import os
import struct
import wave
import zlib
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import AUDIO_SAMPLE_RATE


# --------------------------------------------------------------------------- audio
def write_wav(path: str, samples: torch.Tensor, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
    pcm = (samples.detach().cpu().float().clamp(-1, 1) * 32767).round().to(torch.int16).numpy()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: str, target_rate: int = AUDIO_SAMPLE_RATE) -> torch.Tensor:
    """-> mono float [S] in [-1,1] resampled (linear) to `target_rate`. 16-bit PCM only."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM wav is supported")
        ch, rate = w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    x = torch.from_numpy(data.reshape(-1, ch).mean(1))
    if rate != target_rate:
        n = round(len(x) * target_rate / rate)
        x = F.interpolate(x[None, None], size=n, mode="linear", align_corners=False)[0, 0]
    return x


# --------------------------------------------------------------------------- images
def write_png(path: str, frame: torch.Tensor) -> None:
    """uint8 [H,W,3] -> PNG."""
    a = frame.cpu().numpy().astype(np.uint8)
    H, W, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(H))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def load_image(path: str, size: Optional[tuple] = None) -> torch.Tensor:
    """-> [1,3,H,W] float in [-1,1]. `size` = (H, W) resizes (bicubic)."""
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading images needs Pillow: pip install pillow") from e
    im = np.asarray(Image.open(path).convert("RGB"))
    x = torch.from_numpy(im.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1.0
    if size is not None:
        x = F.interpolate(x, size=size, mode="bicubic", align_corners=False).clamp(-1, 1)
    return x


# --------------------------------------------------------------------------- video
def save_video(path: str, video: torch.Tensor, fps: int, audio: Optional[torch.Tensor] = None, sample_rate: int = AUDIO_SAMPLE_RATE) -> str:
    """Write frames (+ audio). `.mp4` uses PyAV (h264 + aac); anything else, or a missing
    PyAV, writes a folder of PNG frames and a .wav next to it. Returns what was written."""
    stem, ext = os.path.splitext(path)
    if ext.lower() == ".mp4":
        try:
            import av  # noqa: F401
        except ImportError:
            print("PyAV not installed (pip install av); writing PNG frames + wav instead")
        else:
            _save_mp4(path, video, fps, audio, sample_rate)
            return path
    os.makedirs(stem, exist_ok=True)
    for i, f in enumerate(video):
        write_png(os.path.join(stem, f"{i:05d}.png"), f)
    if audio is not None:
        write_wav(stem + ".wav", audio, sample_rate)
    return stem


def _save_mp4(path, video, fps, audio, sample_rate):
    import av

    T, H, W, _ = video.shape
    with av.open(path, "w") as out:
        vs = out.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
        as_ = None
        if audio is not None:
            as_ = out.add_stream("aac", rate=sample_rate)
            as_.layout = "mono"
        for f in video.cpu().numpy():
            for p in vs.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(p)
        for p in vs.encode():
            out.mux(p)
        if as_ is not None:
            pcm = (audio.cpu().float().clamp(-1, 1) * 32767).to(torch.int16).numpy()[None]
            fr = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
            fr.sample_rate = sample_rate
            for p in as_.encode(fr):
                out.mux(p)
            for p in as_.encode():
                out.mux(p)
