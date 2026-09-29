import json
import os

import numpy as np
import pytest
import torch

from h3turbo import media
from h3turbo.cli import main


def test_png_writer_is_valid_and_lossless(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    frame = torch.randint(0, 256, (17, 23, 3), dtype=torch.uint8)
    p = str(tmp_path / "f.png")
    media.write_png(p, frame)
    assert np.array_equal(np.asarray(Image.open(p).convert("RGB")), frame.numpy())
    x = media.load_image(p, size=(34, 46))
    assert x.shape == (1, 3, 34, 46) and x.min() >= -1 and x.max() <= 1


def test_wav_roundtrip_and_resample(tmp_path):
    t = torch.arange(32000) / 32000
    w = 0.5 * torch.sin(2 * np.pi * 440 * t)
    p = str(tmp_path / "a.wav")
    media.write_wav(p, w)
    r = media.read_wav(p)
    assert r.shape == w.shape and (r - w).abs().max() < 1e-3
    media.write_wav(p, w, sample_rate=16000)  # same samples, declared as 16 kHz -> twice as long at 32 kHz
    assert media.read_wav(p).shape[0] == 64000


def test_save_video_mp4_has_synced_streams(tmp_path):
    av = pytest.importorskip("av")
    video = torch.randint(0, 256, (9, 64, 64, 3), dtype=torch.uint8)
    audio = 0.3 * torch.sin(torch.arange(36000) * 0.05)
    p = str(tmp_path / "v.mp4")
    assert media.save_video(p, video, 8, audio) == p
    c = av.open(p)
    v = next(s for s in c.streams if s.type == "video")
    a = next(s for s in c.streams if s.type == "audio")
    assert v.frames == 9 and float(v.average_rate) == 8.0 and (v.width, v.height) == (64, 64)
    assert a.rate == 32000 and abs(c.duration / 1e6 - 9 / 8) < 0.05


def test_save_video_png_fallback(tmp_path):
    video = torch.randint(0, 256, (3, 32, 32, 3), dtype=torch.uint8)
    out = media.save_video(str(tmp_path / "clip.gif"), video, 8, torch.zeros(800))
    assert sorted(os.listdir(out)) == ["00000.png", "00001.png", "00002.png"]
    assert os.path.exists(out + ".wav")


def test_cli_init_info_bench_quantize_generate(tmp_path, capsys):
    ck = str(tmp_path / "n.safetensors")
    main(["init", "--tier", "nano", "--out", ck])
    assert "UNTRAINED" in capsys.readouterr().out
    main(["info", "--ckpt", ck])
    info = json.loads(capsys.readouterr().out)
    assert info["quant"] == "none" and info["trained"].startswith("no")
    assert info["h3turbo_config"]["name"] == "h3turbo-nano"

    main(["bench", "--ckpt", ck, "--width", "64", "--height", "64", "--seconds", "1", "--tflops", "5"])
    out = capsys.readouterr().out
    assert "TFLOP" in out and "unmeasured" in out

    q = str(tmp_path / "q.safetensors")
    main(["quantize", "--ckpt", ck, "--mode", "int8", "--out", q])
    capsys.readouterr()
    main(["info", "--ckpt", q])
    assert json.loads(capsys.readouterr().out)["quant"] == "int8"

    dst = str(tmp_path / "out.gif")
    main(["generate", "a red square moving left", "--ckpt", q, "--device", "cpu", "--dtype", "fp32", "--width", "64", "--height", "64",
          "--frames", "9", "--steps", "2", "--seed", "1", "--out", dst])
    assert len(os.listdir(dst[:-4])) == 9 and os.path.exists(dst[:-4] + ".wav")
