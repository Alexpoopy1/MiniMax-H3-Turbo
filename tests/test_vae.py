import torch

from h3turbo.audio_vae import AudioVAE
from h3turbo.config import make_config
from h3turbo.text import ByteTextEncoder, ByteTokenizer
from h3turbo.video_vae import VideoVAE


def _vae():
    torch.manual_seed(0)
    return VideoVAE(make_config("nano").video_vae).eval()


def test_video_vae_shapes_f16t4d24():
    v = _vae()
    x = torch.randn(1, 3, 9, 64, 96)
    z = v.encode(x)
    assert z.shape == (1, 24, 3, 4, 6)
    assert v.decode(z).shape == x.shape
    assert v.encode(x[:, :, :1]).shape == (1, 24, 1, 4, 6)  # an image is a 1-frame video


def test_video_vae_rejects_bad_sizes():
    v = _vae()
    for shape in [(1, 3, 8, 64, 64), (1, 3, 9, 60, 64)]:
        try:
            v.encode(torch.randn(*shape))
        except ValueError:
            continue
        raise AssertionError(shape)


def test_video_vae_is_causal():
    v = _vae()
    x = torch.randn(1, 3, 9, 64, 64)
    x2 = x.clone()
    x2[:, :, 5:] = torch.randn_like(x2[:, :, 5:])
    a, b = v.encode(x), v.encode(x2)
    assert torch.allclose(a[:, :, :2], b[:, :, :2], atol=1e-5)
    assert not torch.allclose(a[:, :, 2], b[:, :, 2])


def test_temporal_streaming_decode_is_exact():
    v = _vae()
    z = torch.randn(1, 24, 9, 4, 4)
    full = v.decode(z)
    for chunk in (1, 2, 4):
        t = v.decode_tiled(z, tile=99, overlap=0, chunk=chunk)
        assert t.shape == full.shape
        assert (t - full).abs().max() < 1e-4, chunk
    assert not v.decoder[0].streaming  # streaming state is always reset


def test_spatial_tiled_decode_converges_with_overlap():
    v = _vae()
    z = torch.randn(1, 24, 3, 24, 28)
    full = v.decode(z)
    errs = []
    for tile, ov in [(12, 4), (16, 8), (20, 12)]:
        t = v.decode_tiled(z, tile=tile, overlap=ov, chunk=99)
        assert t.shape == full.shape
        errs.append(((t - full).abs().mean() / full.abs().mean()).item())
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 0.01


def test_audio_vae_40hz_and_causal():
    a = AudioVAE(make_config("nano").audio_vae).eval()
    w = torch.randn(2, 1, 8000)  # 0.25 s at 32 kHz
    z = a.encode(w)
    assert z.shape == (2, 32, 10)  # 40 tokens/s
    assert a.decode(z).shape == w.shape
    w2 = w.clone()
    w2[..., 4000:] = torch.randn(2, 1, 4000)
    assert torch.allclose(z[..., :5], a.encode(w2)[..., :5], atol=1e-5)


def test_byte_text_encoder_padding_invariance():
    torch.manual_seed(0)
    enc = ByteTextEncoder(make_config("nano").text).eval()
    e_single, m1 = enc.encode_text(["hi"])
    e_batch, m2 = enc.encode_text(["hi", "a much longer prompt than that one"])
    n = int(m1.sum())
    assert torch.allclose(e_single[0, :n], e_batch[0, :n], atol=1e-5)


def test_tokenizer_utf8_roundtrip_length():
    tok = ByteTokenizer(16)
    ids = tok.encode("héllo ✓✓✓✓✓✓✓✓✓✓")
    assert ids[0] == 257 and ids[-1] == 258 and len(ids) <= 16


def test_streaming_decode_handles_image_and_ragged_chunks():
    v = _vae()
    z1 = torch.randn(1, 24, 1, 4, 4)
    assert torch.equal(v.decode_tiled(z1, tile=99, overlap=0, chunk=2), v.decode(z1))  # 1-frame video
    z = torch.randn(1, 24, 7, 4, 4)
    full = v.decode(z)
    for chunk in (3, 5, 6):  # 7 latents never divide evenly
        assert (v.decode_tiled(z, tile=99, overlap=0, chunk=chunk) - full).abs().max() < 1e-4


def test_stft_loss_is_zero_for_identical_and_grows_with_mismatch():
    from h3turbo.training import stft_loss

    g = torch.Generator().manual_seed(0)
    t = torch.arange(16000) / 32000
    x = torch.sin(2 * torch.pi * 440 * t)[None].repeat(2, 1)
    assert stft_loss(x, x).item() < 1e-6
    shifted = torch.sin(2 * torch.pi * 660 * t)[None].repeat(2, 1)
    assert stft_loss(x, shifted).item() > 0.5
    assert stft_loss(x, torch.zeros_like(x)).item() > stft_loss(x, x * 0.9).item()
