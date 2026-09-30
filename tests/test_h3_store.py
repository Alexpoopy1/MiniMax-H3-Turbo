"""h3t container + converter: lossless, aligned, contiguous, validated. CPU only, tiny synthetic models."""
import dataclasses
import json
import os
import struct

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from h3turbo.h3 import convert as cv
from h3turbo.h3.config import H3Config
from h3turbo.h3.store import (
    BLOCK_ALIGN, PAD_PREFIX, TENSOR_ALIGN, H3TError, H3TFile, LazySource, TensorSource, TorchSource, read_header, write_h3t,
)
from h3turbo.h3.types import W4A8Weight


def tiny_cfg(layers: int = 3) -> H3Config:
    return H3Config(hidden=64, layers=layers, refiner_layers=1, heads=2, head_dim=32, ffn=48, video_channels=4, audio_channels=8,
                    text_dim=32, t_dim=8, curve_grid=17, rope_inv_freq_len=16, quant_group=16, quant_convrot=16)


def raw(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def convert_sd(tmp_path, sd, cfg=None, name="m.h3t", **kw):
    dst = str(tmp_path / name)
    rep = cv.convert(sd, dst, cfg=cfg, **kw)
    return dst, rep


@pytest.fixture(scope="module")
def quant_sd():
    return cv.synthetic_state_dict(tiny_cfg(), quant=True, seed=1)


@pytest.fixture(scope="module")
def dense_sd():
    return cv.synthetic_state_dict(tiny_cfg(), quant=False, seed=2)


@pytest.mark.parametrize("quant", [True, False])
def test_every_tensor_is_bit_identical(tmp_path, quant, quant_sd, dense_sd):
    sd = quant_sd if quant else dense_sd
    dst, rep = convert_sd(tmp_path, sd)
    assert rep.verified and rep.n_tensors == len(sd)
    with H3TFile(dst) as st, safe_open(dst, "pt") as f:
        assert set(st.keys()) == set(sd)
        assert {k for k in f.keys() if not k.startswith(PAD_PREFIX)} == set(sd)
        for name, ref in sd.items():
            t = st.tensor(name)
            assert (t.dtype, tuple(t.shape)) == (ref.dtype, tuple(ref.shape)), name
            assert raw(t) == raw(ref), name  # our zero-copy view
            assert raw(f.get_tensor(name)) == raw(ref), name  # safetensors' own reader


@pytest.mark.parametrize("quant", [True, False])
def test_alignment_contiguity_and_identical_block_layout(tmp_path, quant, quant_sd, dense_sd):
    sd = quant_sd if quant else dense_sd
    dst, _ = convert_sd(tmp_path, sd)
    h = read_header(dst)
    assert h.data_start % BLOCK_ALIGN == 0
    cur = 0
    for name, e in sorted(h.tensors.items(), key=lambda kv: kv[1]["data_offsets"][0]):  # no holes, no overlap
        a, b = e["data_offsets"]
        assert a == cur, name
        if not name.startswith(PAD_PREFIX):
            assert (h.data_start + a) % TENSOR_ALIGN == 0, name
        cur = b
    assert h.data_start + cur == h.file_size
    with H3TFile(dst) as st:
        assert st.block_nbytes % BLOCK_ALIGN == 0 and st.n_blocks == 3
        for i in range(st.n_blocks):
            a, b = st.block_range(i)
            assert a % BLOCK_ALIGN == 0 and b - a == st.block_nbytes
            assert st.block_bytes(i).shape == (st.block_nbytes,)
            for e in st.block_layout:  # same relative offsets in every block, and inside the block range
                info = st.info(f"blocks.{i}.{e.name}")
                assert info.start == a + e.offset and info.nbytes == e.nbytes and e.offset + e.nbytes <= st.block_nbytes
        assert st.block_range(1)[0] == st.block_range(0)[1]


def test_block_weights_views(tmp_path, quant_sd):
    dst, _ = convert_sd(tmp_path, quant_sd)
    with H3TFile(dst) as st:
        for i in range(st.n_blocks):
            w = st.block_weights_cpu(i)
            for lin, p in ((w.qkv, "attn.qkv_proj"), (w.out, "attn.out_proj"), (w.fc1, "mlp.fc1"), (w.fc2, "mlp.fc2")):
                assert isinstance(lin, W4A8Weight) and (lin.group_size, lin.convrot) == (16, 16)
                pre = f"blocks.{i}.{p}."
                assert lin.q.dtype == torch.int8 and lin.s_rel.dtype == torch.float8_e4m3fn and lin.s_ch.dtype == torch.float32 and lin.codebook.shape == (16,)
                assert raw(lin.q) == raw(quant_sd[pre + "weight"]) and raw(lin.s_rel) == raw(quant_sd[pre + "weight_s_rel"])
                assert raw(lin.s_ch) == raw(quant_sd[pre + "weight_s_channel"]) and raw(lin.codebook) == raw(quant_sd[pre + "weight_codebook"])
                assert lin.in_features == 2 * lin.q.shape[1]
            assert raw(w.norm1) == raw(quant_sd[f"blocks.{i}.norm1.weight"]) and raw(w.norm2) == raw(quant_sd[f"blocks.{i}.norm2.weight"])
            assert raw(w.q_norm) == raw(quant_sd[f"blocks.{i}.attn.q_norm.weight"]) and raw(w.k_norm) == raw(quant_sd[f"blocks.{i}.attn.k_norm.weight"])
            assert raw(w.adaln_w) == raw(quant_sd[f"blocks.{i}.adaln_proj.linear.weight"]) and raw(w.adaln_b) == raw(quant_sd[f"blocks.{i}.adaln_proj.linear.bias"])
            # the same views over an unrelated copy of the bytes (this is what a device slot is)
            buf = torch.empty(st.block_nbytes, dtype=torch.uint8)
            buf.copy_(st.block_tensor(i))
            w2 = st.block_weights_from(buf)
            assert raw(w2.qkv.q) == raw(w.qkv.q) and w2.qkv.q.data_ptr() != w.qkv.q.data_ptr()
        with pytest.raises(ValueError):
            st.block_weights_from(torch.empty(10, dtype=torch.uint8))


def test_dense_block_weights_are_plain_tensors(tmp_path, dense_sd):
    dst, _ = convert_sd(tmp_path, dense_sd)
    with H3TFile(dst) as st:
        w = st.block_weights_cpu(2)
        assert all(isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16 for x in (w.qkv, w.out, w.fc1, w.fc2))
        assert w.qkv.shape == (192, 64) and w.fc2.shape == (64, 48)
        assert st.quant == {}


def test_load_globals(tmp_path, quant_sd):
    dst, _ = convert_sd(tmp_path, quant_sd)
    with H3TFile(dst) as st:
        g = st.load_globals("cpu")
        for field, name in (("video_patch_w", "video_patch_proj.weight"), ("audio_patch_b", "audio_patch_proj.bias"), ("condition_w", "condition_proj.weight"),
                            ("adaln_t_table", "adaln_t_table"), ("rope_inv_freq", "rope.inv_freq"), ("final_norm", "final_layer.norm.weight"),
                            ("final_adaln_w", "final_layer.adaln_proj.linear.weight"), ("final_adaln_b", "final_layer.adaln_proj.linear.bias"),
                            ("video_out_w", "final_layer.video_out.weight"), ("audio_out_b", "final_layer.audio_out.bias"),
                            ("refiner_final_norm", "token_refiner.final_norm.weight")):
            assert raw(getattr(g, field)) == raw(quant_sd[name]), field
        assert len(g.refiner) == 1 and g.refiner[0].adaln_w is None
        assert isinstance(g.refiner[0].qkv, torch.Tensor) and raw(g.refiner[0].qkv) == raw(quant_sd["token_refiner.blocks.0.attn.qkv_proj.weight"])
        assert st.globals_nbytes(True) > st.globals_nbytes(False) > 0
        g2 = st.load_globals("cpu", refiner_device="cpu")
        assert len(g2.refiner) == 1


def test_metadata_round_trip(tmp_path, quant_sd):
    dst, rep = convert_sd(tmp_path, quant_sd)
    with H3TFile(dst) as st:
        assert st.cfg == rep.cfg and st.cfg.layers == 3 and st.cfg.quant_convrot == 16
        assert st.quant == {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 16}
        assert st.meta["format"] == "h3turbo-h3" and st.meta["version"] == "1"
        assert st.source_name == "<state_dict>"
        ranges = json.loads(st.meta["block_ranges"])
        assert [b - a for a, b in ranges] == [st.block_nbytes] * 3 and ranges[0][0] == st.blocks_start
        assert st.file_size == os.path.getsize(dst) == rep.dst_bytes


def test_file_source_records_provenance_and_matches_dict_convert(tmp_path, quant_sd):
    src = str(tmp_path / "src.safetensors")
    save_file({k: v.contiguous() for k, v in quant_sd.items()}, src)
    a, ra = convert_sd(tmp_path, src, name="from_file.h3t")
    b, _ = convert_sd(tmp_path, quant_sd, name="from_dict.h3t")
    with H3TFile(a) as sa, H3TFile(b) as sb:
        assert sa.source_name == "src.safetensors" and sa.source_size == os.path.getsize(src) and len(sa.meta["source_header_sha256"]) == 64
        assert sa.keys() == sb.keys()
        for i in range(sa.n_blocks):
            assert bytes(sa.block_bytes(i)) == bytes(sb.block_bytes(i))
    assert ra.verified and ra.src_bytes == sum(v.numel() * v.element_size() for v in quant_sd.values())


def test_output_is_deterministic_and_chunk_size_independent(tmp_path, quant_sd):
    a, _ = convert_sd(tmp_path, quant_sd, name="a.h3t")
    b, _ = convert_sd(tmp_path, quant_sd, name="b.h3t", chunk_bytes=4096)
    assert open(a, "rb").read() == open(b, "rb").read()


def test_writer_memory_is_bounded_by_the_chunk(tmp_path, quant_sd):
    seen = []

    class Spy(TensorSource):
        def __init__(self, t):
            self._s = TorchSource(t)
            self.dtype, self.shape, self.nbytes = self._s.dtype, self._s.shape, self._s.nbytes

        def chunks(self, n):
            for c in self._s.chunks(n):
                seen.append(len(c))
                yield c

    dst = str(tmp_path / "spy.h3t")
    write_h3t(dst, tiny_cfg(), {k: Spy(v) for k, v in quant_sd.items()}, chunk_bytes=4096)
    assert max(seen) <= 4096 and sum(seen) == sum(v.numel() * v.element_size() for v in quant_sd.values())
    assert max(v.numel() * v.element_size() for v in quant_sd.values()) > 4096  # the bound was actually exercised


def test_extras_are_kept_and_reported(tmp_path, quant_sd):
    dst, rep = convert_sd(tmp_path, quant_sd)
    assert set(rep.extras) == {"adaln_basis", "adaln_mean"}
    with H3TFile(dst) as st:
        assert raw(st.tensor("adaln_basis")) == raw(quant_sd["adaln_basis"])


def test_report_statistics(tmp_path, quant_sd):
    _, rep = convert_sd(tmp_path, quant_sd)
    assert rep.n_quant_layers == 12 and rep.n_dense_linears == 0 and rep.codebook_sorted == 12
    assert rep.s_ch_min > 0 and rep.s_rel_range == (0.25, 2.25)
    assert "12 W4A8" in rep.summary()


def test_overwrite_protection_and_atomic_write(tmp_path, quant_sd):
    dst, _ = convert_sd(tmp_path, quant_sd)
    with pytest.raises(FileExistsError):
        cv.convert(quant_sd, dst)
    cv.convert(quant_sd, dst, overwrite=True, verify=False)

    def boom():
        raise RuntimeError("source died")

    bad = dict(quant_sd)
    bad["blocks.2.norm2.weight"] = LazySource("BF16", (64,), boom)
    fresh = str(tmp_path / "fresh.h3t")
    with pytest.raises(RuntimeError, match="source died"):
        cv.convert(bad, fresh, verify=False)
    assert not os.path.exists(fresh) and not os.path.exists(fresh + ".part")


# ------------------------------------------------------------------ validation
def _mut(sd, **repl):
    out = dict(sd)
    for k, v in repl.items():
        k = k.replace("__", ".")
        if v is None:
            out.pop(k)
        else:
            out[k] = v
    return out


def _cq(**kw):
    d = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 16}
    d.update(kw)
    return torch.tensor(list(json.dumps(d, separators=(",", ":")).encode()), dtype=torch.uint8)


CASES = {
    "missing tensor": (lambda sd: {k: v for k, v in sd.items() if k != "blocks.1.norm1.weight"}, "blocks.1.norm1.weight"),
    "missing quant part": (lambda sd: {k: v for k, v in sd.items() if k != "blocks.1.mlp.fc1.weight_s_rel"}, "quantisation tensors without weight_codebook|missing tensor|weight_s_rel"),
    "dtype": (lambda sd: {**sd, "blocks.0.norm1.weight": sd["blocks.0.norm1.weight"].to(torch.int8)}, "dtype"),
    "shape": (lambda sd: {**sd, "blocks.2.attn.q_norm.weight": torch.zeros(31, dtype=torch.bfloat16)}, "shape"),
    "quant format": (lambda sd: {**sd, "blocks.0.mlp.fc2.comfy_quant": _cq(format="other")}, "format"),
    "group mismatch": (lambda sd: {**sd, "blocks.1.attn.out_proj.comfy_quant": _cq(group_size=8)}, "differs from the first"),
    "bad json": (lambda sd: {**sd, "blocks.0.attn.qkv_proj.comfy_quant": torch.tensor([1, 2, 3], dtype=torch.uint8)}, "not valid JSON"),
    "nan scale": (lambda sd: {**sd, "blocks.1.mlp.fc1.weight_s_rel": torch.full(sd["blocks.1.mlp.fc1.weight_s_rel"].shape, float("nan")).to(torch.float8_e4m3fn)}, "NaN"),
    "inf codebook": (lambda sd: {**sd, "blocks.1.mlp.fc1.weight_codebook": torch.full((16,), float("inf"))}, "non-finite"),
    "negative s_channel": (lambda sd: {**sd, "blocks.1.mlp.fc1.weight_s_channel": -torch.ones(96)}, "negative"),
    "block gap": (lambda sd: {k.replace("blocks.2.", "blocks.5."): v for k, v in sd.items()}, "do not form"),
    "bad rope": (lambda sd: {**sd, "rope.inv_freq": torch.zeros(7)}, "shape"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_converter_rejects_invalid_checkpoints(tmp_path, quant_sd, case):
    mutate, pattern = CASES[case]
    with pytest.raises(cv.ConvertError, match=pattern):
        cv.convert(mutate(quant_sd), str(tmp_path / "x.h3t"), cfg=tiny_cfg(), verify=False)
    assert not os.path.exists(tmp_path / "x.h3t")


def test_mixed_dense_and_quant_blocks_rejected(tmp_path, quant_sd, dense_sd):
    mixed = dict(quant_sd)
    for k, v in dense_sd.items():
        if k.startswith("blocks.2.") and "mlp.fc1" in k:
            mixed[k] = v
    for k in [k for k in mixed if k.startswith("blocks.2.mlp.fc1.") and k != "blocks.2.mlp.fc1.weight"]:
        del mixed[k]
    with pytest.raises(H3TError, match="block 2"):
        cv.convert(mixed, str(tmp_path / "x.h3t"), cfg=tiny_cfg(), verify=False)


def test_unrecognised_checkpoint_gives_clear_error(tmp_path):
    with pytest.raises(cv.ConvertError, match="cannot infer"):
        cv.convert({"foo": torch.zeros(2)}, str(tmp_path / "x.h3t"))


# ------------------------------------------------------------------ reader robustness
def test_verify_detects_a_single_flipped_bit(tmp_path, quant_sd):
    dst, _ = convert_sd(tmp_path, quant_sd)
    srcs = cv._open_source(quant_sd)[0]
    cv.verify_h3t(srcs, dst, tiny_cfg(), chunk_bytes=4096)
    with H3TFile(dst) as st:
        pos = st.info("blocks.1.mlp.fc2.weight").start + 5
    with open(dst, "r+b") as f:
        f.seek(pos)
        b = f.read(1)
        f.seek(pos)
        f.write(bytes([b[0] ^ 1]))
    with pytest.raises(cv.ConvertError, match="blocks.1.mlp.fc2.weight"):
        cv.verify_h3t(srcs, dst, tiny_cfg(), chunk_bytes=4096)


def test_reader_rejects_foreign_truncated_and_corrupt_files(tmp_path, quant_sd):
    plain = str(tmp_path / "plain.safetensors")
    save_file({"a": torch.zeros(4)}, plain)
    with pytest.raises(H3TError, match="not an h3t"):
        H3TFile(plain)
    dst, _ = convert_sd(tmp_path, quant_sd)
    data = open(dst, "rb").read()
    for cut in (len(data) - 1, len(data) // 2, 10):
        p = str(tmp_path / f"cut{cut}.h3t")
        open(p, "wb").write(data[:cut])
        with pytest.raises(H3TError):
            H3TFile(p)
    p = str(tmp_path / "junk.h3t")
    open(p, "wb").write(b"\x05\x00\x00\x00\x00\x00\x00\x00hello world")
    with pytest.raises(H3TError):
        H3TFile(p)
    p = str(tmp_path / "trailing.h3t")
    open(p, "wb").write(data + b"\0" * 8)
    with pytest.raises(H3TError, match="range|exactly|trailing"):
        H3TFile(p)
    n = struct.unpack("<Q", data[:8])[0]
    hdr = json.loads(data[8 : 8 + n])
    hdr["__metadata__"]["version"] = "99"
    new = json.dumps(hdr).encode()
    new += b" " * (n - len(new))  # the writer's alignment padding leaves room for a longer version string
    assert len(new) == n
    p = str(tmp_path / "v99.h3t")
    open(p, "wb").write(struct.pack("<Q", n) + new + data[8 + n :])
    with pytest.raises(H3TError, match="version"):
        H3TFile(p)


def test_closed_file_refuses_access_and_close_is_idempotent(tmp_path, quant_sd):
    dst, _ = convert_sd(tmp_path, quant_sd)
    st = H3TFile(dst)
    view = st.tensor("rope.inv_freq")
    st.close()
    st.close()
    with pytest.raises(ValueError, match="closed"):
        st.block_bytes(0)
    assert view.shape == (16,)  # a live view keeps working after close (the mapping outlives it)


def test_writer_rejects_inconsistent_inputs(tmp_path, quant_sd):
    cfg = tiny_cfg()
    with pytest.raises(H3TError, match="reserved padding"):
        write_h3t(str(tmp_path / "a"), cfg, {**quant_sd, "__pad.1": torch.zeros(1)})
    bad = dict(quant_sd)
    bad["blocks.1.attn.k_norm.weight"] = torch.zeros(32, dtype=torch.float32)
    with pytest.raises(H3TError, match="block 0 has"):
        write_h3t(str(tmp_path / "b"), cfg, bad)
    bad = {k: v for k, v in quant_sd.items() if k != "blocks.1.attn.k_norm.weight"}
    with pytest.raises(H3TError, match="differs from block 0"):
        write_h3t(str(tmp_path / "c"), cfg, bad)
    with pytest.raises(H3TError, match="blocks"):
        write_h3t(str(tmp_path / "d"), dataclasses.replace(cfg, layers=4), quant_sd)


def test_cli(tmp_path, quant_sd, capsys):
    src = str(tmp_path / "s.safetensors")
    save_file({k: v.contiguous() for k, v in quant_sd.items()}, src)
    dst = str(tmp_path / "cli.h3t")
    assert cv.main([src, dst]) == 0
    out = capsys.readouterr().out
    assert "verified    True" in out and "12 W4A8" in out
    with pytest.raises(FileExistsError):
        cv.main([src, dst])
    assert cv.main([src, dst, "--force", "--no-verify"]) == 0
    assert cv.info_main([dst]) == 0
    info = capsys.readouterr().out
    assert "3 blocks" in info and "attn.qkv_proj.weight_codebook" in info


def test_cli_reports_invalid_checkpoints_without_a_traceback(tmp_path, capsys):
    bad = str(tmp_path / "bad.safetensors")
    save_file({"foo": torch.zeros(2)}, bad)
    assert cv.main([bad, str(tmp_path / "bad.h3t")]) == 2
    assert "cannot infer the model config" in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "bad.h3t")


def _patch_meta(path, **changes):
    """Rewrite __metadata__ entries in place (None deletes one); the JSON header keeps its length so offsets stay valid."""
    with open(path, "r+b") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        for k, v in changes.items():
            h["__metadata__"].pop(k) if v is None else h["__metadata__"].__setitem__(k, v)
        blob = json.dumps(h, separators=(",", ":")).encode()
        assert len(blob) <= n
        f.seek(8)
        f.write(blob.ljust(n, b" "))


@pytest.mark.parametrize("change", [{"block_ranges": None}, {"version": "x"}, {"block_ranges": "5"}, {"block_ranges": "[[1, 2]]"}])
def test_corrupt_metadata_raises_h3terror_not_a_raw_exception(tmp_path, quant_sd, change):
    dst, _ = convert_sd(tmp_path, quant_sd)
    _patch_meta(dst, **change)
    with pytest.raises(H3TError):
        H3TFile(dst)


def test_cpu_globals_are_private_copies_except_the_refiner(tmp_path, quant_sd):
    """A write into the read-only file mapping is an access violation, so only the (large) refiner may stay a mapped view."""
    dst, _ = convert_sd(tmp_path, quant_sd)
    with H3TFile(dst) as st:
        g = st.load_globals("cpu")
        g.rope_inv_freq[0] = 5.0  # would crash the interpreter if this were a mapping view
        g.video_out_w.mul_(1.0)
        assert g.refiner[0].qkv.data_ptr() == st.tensor("token_refiner.blocks.0.attn.qkv_proj.weight").data_ptr()
