"""Packed-sequence bookkeeping for the official H3 DiT.

The network sees ONE token sequence `[text | keyframe cond rows | refs | target audio | target video]`.
Everything that decides which token gets which position, which AdaLN modulation row and which
timestep lives here (pure torch/python, no weights), so it can be unit-tested against the reference
and reused by the pipeline: layout + 3-axis position ids, patchify/unpatchify, audio pack/unpack,
denoise-mask row helpers and the per-segment timestep/modulation plan.

Numerics note: timestep bookkeeping deliberately runs on fp32 scalars (`sigma` is an fp32 0-d
tensor) because the reference derives its AdaLN-table coordinates that way; float64 shortcuts move
the interpolation weight by ~1e-8, which is harmless but breaks bit-level comparisons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)  # pixel frames covered by latent frame k (period 5: causal VAE, first frame alone)
FRAME_RESCALE = 5.0 / 3.0  # position units per pixel frame
VISUAL_COND_TIMESTEP = 0.999  # timestep label of conditioning image rows
AUDIO_COND_TIMESTEP = 1.0
GRID_EXTENT = 32.0  # spatial position range of a frame, centred on 0
SPATIAL_PATCH = 2  # rows are 2x2 latent patches

# AdaLN carries three parameter sets per timestep row: index = t_row * 3 + tag
TAG_VIDEO, TAG_TEXT, TAG_AUDIO = 0, 1, 2
KIND_TAG = {"text": TAG_TEXT, "video": TAG_VIDEO, "cond": TAG_VIDEO, "ref_img": TAG_VIDEO,
            "audio": TAG_AUDIO, "cond_audio": TAG_AUDIO, "ref_audio": TAG_AUDIO}
_VIDEO_KINDS = ("cond", "ref_img", "video")
_AUDIO_KINDS = ("cond_audio", "ref_audio", "audio")

ModRow = Union[int, torch.Tensor]  # one mod-row index, or a per-token LongTensor of them


def time_shift_sigma(sigma, from_shift: float, to_shift: float):
    """Re-express a flow-matching sigma of a `from_shift` schedule on a `to_shift` schedule.

    Works on floats and on tensors; the operation order is fixed on purpose (fp32 parity).
    """
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


# --------------------------------------------------------------------------- patchify / audio pack


def pad_to_patch(x: torch.Tensor, patch: Sequence[int] = (1, 2, 2)) -> torch.Tensor:
    """Circular right-pad the trailing (T, H, W) dims of [B, C, T, H, W] to multiples of `patch`."""
    for i, p in enumerate(patch):
        extra = (-x.shape[2 + i]) % p
        if extra:
            x = torch.cat([x, x.narrow(2 + i, 0, extra)], dim=2 + i)
    return x


def patchify_video(latent: torch.Tensor, patch: Sequence[int] = (1, 2, 2)) -> torch.Tensor:
    """[B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]; rows in (b, t, h, w) order, features in (c, pt, ph, pw) order."""
    b, c, tf, hf, wf = latent.shape
    pt, ph, pw = patch
    if tf % pt or hf % ph or wf % pw:
        raise ValueError(f"latent {tuple(latent.shape)} is not a multiple of patch {tuple(patch)}; pad_to_patch first")
    x = latent.reshape(b, c, tf // pt, pt, hf // ph, ph, wf // pw, pw).permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b * (tf // pt) * (hf // ph) * (wf // pw), c * pt * ph * pw)


def unpatchify_video(rows: torch.Tensor, t: int, h: int, w: int, c: int, patch: Sequence[int] = (1, 2, 2)) -> torch.Tensor:
    """Inverse of `patchify_video`; (t, h, w) are patch-grid sizes. -> [B, C, t*pt, h*ph, w*pw]."""
    pt, ph, pw = patch
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw).permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    """[1, C, ch, T] -> [ch*T, C], channel-major (all rows of channel 0, then channel 1)."""
    if latent.shape[0] != 1:
        raise ValueError("audio latent must have batch size 1")
    c, ch, t = latent.shape[1:]
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows: torch.Tensor, ch: int = 2) -> torch.Tensor:
    """Inverse of `pack_audio`: [ch*T, C] -> [1, C, ch, T]."""
    return rows.reshape(ch, rows.shape[0] // ch, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


# --------------------------------------------------------------------------- positions


def _axis(dim: int, sqrt_area: float) -> torch.Tensor:
    # patch-centre coordinates of one spatial axis, area-normalised so every resolution spans ~[-16, 16]
    ratio = dim / sqrt_area
    n = dim // SPATIAL_PATCH
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * GRID_EXTENT


def _frame_coords(h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(h, w) coordinates [(h/2)*(w/2), 2] of one latent frame's patch rows, and the w axis."""
    if h < SPATIAL_PATCH or w < SPATIAL_PATCH or h % SPATIAL_PATCH or w % SPATIAL_PATCH:
        raise ValueError(f"latent h/w must be positive multiples of {SPATIAL_PATCH}, got {h}x{w}")
    area = math.sqrt(h * w)
    ys, xs = _axis(h, area), _axis(w, area)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1), xs


def _frame_spans(n: int) -> List[float]:
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(n)]


def _video_positions(frames: int, coords: torch.Tensor, origin: float) -> torch.Tensor:
    # every row of latent frame k sits at t = origin + (pixel frames covered by frames < k) * rescale
    spans = torch.tensor(_frame_spans(frames), dtype=torch.float64)
    starts = origin + torch.cat([spans.new_zeros(1), spans[:-1].cumsum(0)])
    pos = torch.empty(frames, coords.shape[0], 3, dtype=torch.float64)
    pos[:, :, 0] = starts[:, None]
    pos[:, :, 1:] = coords[None]
    return pos.reshape(-1, 3)


def _audio_positions(origin: float, frames: int, w_low: float, w_high: float) -> torch.Tensor:
    # stereo rows, channel-major: t advances per latent frame, w pins each channel to an end of the grid, h stays 0
    pos = torch.zeros(2 * frames, 3, dtype=torch.float64)
    pos[:, 0] = (origin + torch.arange(frames, dtype=torch.float64)).repeat(2)
    pos[:frames, 2] = w_low
    pos[frames:, 2] = w_high
    return pos


def _ref_span(blk: Mapping[str, Any]) -> float:
    """Time-axis extent a reference block occupies ahead of the target streams."""
    kind = blk["kind"]
    if kind == "image":
        return 1.0
    if kind == "audio":
        return float(blk["ref_audio_t"])
    if kind in ("video", "video_audio"):
        return max(float(blk["ref_audio_t"]), sum(_frame_spans(blk["latent_t"])))
    raise ValueError(f"unknown reference kind {kind!r} (expected image/audio/video/video_audio)")


class Segment(NamedTuple):
    start: int
    stop: int
    kind: str  # text / cond / cond_audio / ref_img / ref_audio / audio / video


class _Builder:
    def __init__(self) -> None:
        self.row = 0
        self.segments: List[Segment] = []
        self.pos: List[torch.Tensor] = []
        self.img: List[torch.Tensor] = []
        self.img_upd: List[torch.Tensor] = []
        self.aud: List[torch.Tensor] = []
        self.aud_upd: List[torch.Tensor] = []

    def add(self, kind: str, positions: torch.Tensor, update: bool = False) -> None:
        n = positions.shape[0]
        rows = torch.arange(self.row, self.row + n)
        flag = torch.full((n,), update, dtype=torch.bool)
        if kind in _VIDEO_KINDS:
            self.img.append(rows)
            self.img_upd.append(flag)
        elif kind in _AUDIO_KINDS:
            self.aud.append(rows)
            self.aud_upd.append(flag)
        self.segments.append(Segment(self.row, self.row + n, kind))
        self.pos.append(positions)
        self.row += n


@dataclass(eq=False)
class PackedLayout:
    """Static structure of one (shape, conditioning) signature; contains no per-step values."""

    seq_len: int
    position_ids: torch.Tensor  # [S, 3] float64 (t, h, w)
    img_pos: torch.Tensor  # rows of every video-type token (cond, ref image, target)
    img_update: torch.Tensor  # bool over img_pos: True for target video rows
    audio_pos: torch.Tensor
    audio_update: torch.Tensor
    signature: Tuple[int, int, int, int, int]  # (text_len, latent_t, latent_h, latent_w, audio_t)
    segments: List[Segment]

    @classmethod
    def build(cls, text_len: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int,
              keyframes: Optional[Sequence[Mapping[str, Any]]] = None,
              refs: Optional[Sequence[Mapping[str, Any]]] = None) -> "PackedLayout":
        if text_len < 0 or latent_t < 1 or audio_t < 1:
            raise ValueError(f"need text_len>=0, latent_t>=1, audio_t>=1; got {text_len}, {latent_t}, {audio_t}")
        frame, w_axis = _frame_coords(latent_h, latent_w)
        target_w = (float(w_axis[0]), float(w_axis[-1]))
        b = _Builder()

        text = torch.zeros(text_len, 3, dtype=torch.float64)
        text[:, 0] = torch.arange(text_len, dtype=torch.float64)
        b.add("text", text)

        # target streams start after every reference block's time extent
        origin = float(text_len)
        for blk in refs or ():
            origin += _ref_span(blk)

        for kf in keyframes or ():
            # keyframes share the target spatial grid; anchor = origin + rescale * pixel frame index
            anchor = origin + FRAME_RESCALE * kf["resolved_frame_index"]
            if kf.get("latent") is not None:
                b.add("cond", _video_positions(kf["latent"].shape[2], frame, anchor))
            if kf.get("audio_latent") is not None:
                b.add("cond_audio", _audio_positions(anchor, kf["audio_latent"].shape[-1], *target_w))

        cursor = float(text_len)
        for blk in refs or ():
            kind = blk["kind"]
            span = _ref_span(blk)
            if kind == "image":
                coords, _ = _frame_coords(blk["latent_h"], blk["latent_w"])
                pos = torch.empty(coords.shape[0], 3, dtype=torch.float64)
                pos[:, 0] = cursor
                pos[:, 1:] = coords
                b.add("ref_img", pos)
            elif kind == "audio":
                if blk["ref_audio_t"] > 0:
                    b.add("ref_audio", _audio_positions(cursor, blk["ref_audio_t"], *target_w))
            else:  # video / video_audio: the block's audio rows pack right before its video rows, same origin
                coords, r_axis = _frame_coords(blk["latent_h"], blk["latent_w"])
                if blk["ref_audio_t"] > 0:
                    b.add("ref_audio", _audio_positions(cursor, blk["ref_audio_t"], float(r_axis[0]), float(r_axis[-1])))
                b.add("ref_img", _video_positions(blk["latent_t"], coords, cursor))
            cursor += span

        b.add("audio", _audio_positions(origin, audio_t, *target_w), update=True)
        b.add("video", _video_positions(latent_t, frame, origin), update=True)
        return cls(
            seq_len=b.row, position_ids=torch.cat(b.pos), img_pos=torch.cat(b.img), img_update=torch.cat(b.img_upd),
            audio_pos=torch.cat(b.aud), audio_update=torch.cat(b.aud_upd),
            signature=(text_len, latent_t, latent_h, latent_w, audio_t), segments=b.segments)

    def span(self, kind: str) -> Segment:
        return find_segment(self, kind)


def find_segment(layout, kind: str) -> Segment:
    """First segment of `kind`; works on any layout whose segments unpack as (start, stop, kind)."""
    for a, b, k in layout.segments:
        if k == kind:
            return Segment(a, b, k)
    raise KeyError(kind)


def layout_key(text_len: int, shape: Tuple[int, int, int, int], keyframes, refs) -> tuple:
    """Hashable identity of everything `PackedLayout.build` depends on (for caching)."""
    kf = tuple((k["resolved_frame_index"],
                None if k.get("latent") is None else int(k["latent"].shape[2]),
                None if k.get("audio_latent") is None else int(k["audio_latent"].shape[-1])) for k in keyframes or ())
    rf = tuple((r["kind"], r.get("latent_t"), r.get("latent_h"), r.get("latent_w"), r.get("ref_audio_t")) for r in refs or ())
    return (text_len, shape, kf, rf)


def rope_angles(position_ids: torch.Tensor, inv_freq: torch.Tensor, device) -> torch.Tensor:
    """[S, 3] positions -> [S, 3*F] rotation angles (all t frequencies, then h, then w), fp32.

    The rotated part of a head is 2*3*F wide and split-half: angle j pairs channel j with channel j + 3*F.
    """
    pos = position_ids.to(torch.float32).to(device)
    ang = pos.unsqueeze(-1) * inv_freq.to(device=device, dtype=torch.float32).view(1, 1, -1)
    return ang.reshape(pos.shape[0], -1)


def _kitchen():
    try:
        import comfy_kitchen as ck
    except Exception:
        return None
    return ck if hasattr(ck, "rms_rope_split_half_") else None


def _kitchen_cuda_enabled() -> bool:
    """comfy_kitchen dispatches to slow eager code when its CUDA backend is unavailable or disabled (ComfyUI disables it for torch CUDA < 13)."""
    try:
        import comfy_kitchen as ck

        b = ck.list_backends()["cuda"]
        return bool(b["available"]) and not b["disabled"] and "rms_rope_split_half_" in b["capabilities"]
    except Exception:
        return False


def resolve_rope_impl(impl: str, device, dtype: torch.dtype, table_dtype: torch.dtype) -> str:
    """'auto' -> comfy_kitchen's fused op when it can run (CUDA, bf16, tables in the compute dtype), else 'torch'."""
    if impl not in ("auto", "ck", "torch", "eager"):
        raise ValueError(f"rope_impl must be auto/ck/torch/eager, got {impl!r}")
    usable = (_kitchen() is not None and torch.device(device).type == "cuda" and dtype == torch.bfloat16 and table_dtype == dtype
              and _kitchen_cuda_enabled())
    if impl == "ck" and _kitchen() is None:
        raise RuntimeError("rope_impl='ck' needs comfy_kitchen")
    return ("ck" if usable else "torch") if impl == "auto" else impl


def rope_ck_table(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """[S, half] cos/sin -> the kitchen layout [1, S, 1, half, 2, 2] of 2x2 rotation matrices."""
    return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(1, cos.shape[0], 1, cos.shape[1], 2, 2)


def rms_norm_fused_order(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim with the summation order of comfy_kitchen's fused CUDA kernel.

    Measured on sm_86: 32 lanes, lane l accumulates the squares of elements l, l+32, ... in fp32, then a halving
    tree over the lanes; the result is x * rsqrt(sum/d + eps) * w rounded ONCE to x.dtype. Bit-identical to the
    kernel on 14.7M random bf16 elements, where F.rms_norm differs on ~4 per million (1-ulp flips of the normed value).
    """
    d = x.shape[-1]
    if x.dtype == torch.float32 or d % 32:
        return F.rms_norm(x, (d,), weight.to(x.dtype), eps)
    xf = x.to(torch.float32)
    sq = xf.square().reshape(*x.shape[:-1], d // 32, 32)
    acc = sq[..., 0, :]
    for j in range(1, d // 32):
        acc = acc + sq[..., j, :]
    for off in (16, 8, 4, 2, 1):
        acc = acc[..., :off] + acc[..., off:2 * off]
    return (xf * torch.rsqrt(acc / d + eps) * weight.to(torch.float32)).to(x.dtype)


def norm_rope_(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, eps: float, cos: torch.Tensor,
               sin: torch.Tensor, impl: str = "torch", table: Optional[torch.Tensor] = None, chunk: int = 1024) -> None:
    """In place on [S, heads, D] views: per-head RMSNorm, then split-half rotation of the leading 2*half dims.

    impl 'ck'   : comfy_kitchen's fused op (what ComfyUI runs; needs `table` = rope_ck_table).
        'torch' : reproduces the fused CUDA kernel exactly (norm rounded to dtype, rotation in fp32, one rounding).
        'eager' : reproduces ComfyUI's portable path (rotation in the compute dtype, rounded twice); also used for fp32.
    """
    half = cos.shape[1]
    if impl == "ck":
        _kitchen().rms_rope_split_half_(q.unsqueeze(0), k.unsqueeze(0), table, wq, wk, epsilon=eps, rot_dim=2 * half)
        return
    for x, w in ((q, wq), (k, wk)):
        for a in range(0, x.shape[0], chunk):
            sl = slice(a, a + chunk)
            c, s = cos[sl].unsqueeze(1), sin[sl].unsqueeze(1)
            if impl == "eager" or x.dtype == torch.float32:  # kitchen has no fp32 kernel: fp32 always takes the portable path
                xn = F.rms_norm(x[sl], (x.shape[-1],), w.to(x.dtype), eps)
                t1, t2 = xn[..., :half], xn[..., half:2 * half]
                o1, o2 = (t1 * c).addcmul_(t2, -s), (t2 * c).addcmul_(t1, s)
            else:
                xn = rms_norm_fused_order(x[sl], w, eps)
                c, s = c.to(torch.float32), s.to(torch.float32)
                x1, x2 = xn[..., :half].to(torch.float32), xn[..., half:2 * half].to(torch.float32)
                o1, o2 = x1 * c - x2 * s, x1 * s + x2 * c  # both computed before either store: x1/x2 may alias xn
            xn[..., :half], xn[..., half:2 * half] = o1, o2
            x[sl] = xn


def head_bank_range(sample_sigmas, sigma: torch.Tensor, shift_video: float, n: int) -> Tuple[int, int]:
    """Heads [start, stop) of an n-head output bank spanned by the step from `sigma` to the next sampler sigma."""
    ss = torch.as_tensor(sample_sigmas, dtype=torch.float32).cpu()
    nxt = ss[min(int((ss - sigma).abs().argmin()) + 1, ss.shape[0] - 1)]
    start, stop = (round(float(1.0 - time_shift_sigma(v, shift_video, 1.0)) * n) for v in (sigma, nxt))
    start = min(start, n - 1)
    return start, max(stop, start + 1)


def bank_head_weights(weight: torch.Tensor, bias: torch.Tensor, n: int, start: int, stop: int, flow_shift: float):
    """Effective (weight, bias) of an output head bank: block 0 is a full head, blocks 1.. are offsets from it, and a step
    uses the dt-weighted mean of the heads it spans."""
    grid = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
    dt = (1.0 - flow_shift * grid / (1.0 + (flow_shift - 1.0) * grid)).diff()[start:stop]
    w = (dt / dt.sum()).to(weight)
    rows, brows = weight.reshape(n, -1, weight.shape[1]), bias.reshape(n, -1)
    first = max(start, 1)
    return (rows[0] + torch.einsum("n,noi->oi", w[first - start:], rows[first:stop]),
            brows[0] + torch.einsum("n,no->o", w[first - start:], brows[first:stop]))


def curve_lerp(table: torch.Tensor, t_values: Sequence[float]) -> torch.Tensor:
    """AdaLN input rows: linear interpolation of `table` [grid, k] at timesteps t in [0, 1] (clamped)."""
    t = torch.tensor(list(t_values), dtype=torch.float32, device=table.device)
    pos = t.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))


# --------------------------------------------------------------------------- denoise masks


def mask_row_values(mask: torch.Tensor, latent_t: int, lat_h: int, lat_w: int) -> Optional[torch.Tensor]:
    """[T, H, W] denoise mask (1 = generate) -> one value per 2x2 patch row, None if every row fully generates."""
    m = F.pad(mask, (0, lat_w - mask.shape[-1], 0, lat_h - mask.shape[-2]), mode="replicate")
    values = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4)).reshape(-1)
    return None if bool((values >= 1.0 - 1e-3).all()) else values


def token_grid_masks(video_mask: torch.Tensor, audio_mask: Optional[torch.Tensor] = None,
                     patch: Sequence[int] = (1, 2, 2)) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pool per-pixel denoise masks to the token grid the network labels timesteps on.

    Video: amax over each 2x2 patch (broadcast back to pixels). Audio: amax over channels (broadcast back).
    Both are rounded UP to a 1/256 grid so a partly masked row never reads as fully generated.
    """
    ph, pw = patch[1:]
    h, w = video_mask.shape[-2:]
    lead = video_mask.shape[:-2]
    v = F.pad(video_mask.reshape((-1,) + video_mask.shape[-3:]), (0, -w % pw, 0, -h % ph), mode="replicate")
    v = v.reshape(lead + v.shape[-2:])
    v = v.reshape(v.shape[:-2] + (v.shape[-2] // ph, ph, v.shape[-1] // pw, pw)).amax(dim=(-3, -1))
    v = v.repeat_interleave(ph, dim=-2).repeat_interleave(pw, dim=-1)[..., :h, :w]
    a = None if audio_mask is None else audio_mask.amax(dim=1, keepdim=True).expand_as(audio_mask).contiguous()
    ceil = lambda m: torch.ceil(m * 256.0) / 256.0  # noqa: E731
    return ceil(v), (None if a is None else ceil(a))


# --------------------------------------------------------------------------- timestep / modulation plan


@dataclass
class TimePlan:
    t_values: List[float]  # ascending distinct timesteps = rows of the AdaLN input
    mod_segments: List[Tuple[int, int, ModRow]]  # (start, stop, mod row) tiling the whole sequence
    video_seg: Tuple[int, int, ModRow]  # target video rows; row indexes t_values (final layer has no tag axis)
    audio_seg: Tuple[int, int, ModRow]
    t_video: float
    t_audio: float


def plan_time(layout: PackedLayout, sigma: torch.Tensor, shifts: Tuple[float, float], *,
              visual_aug: float = VISUAL_COND_TIMESTEP, audio_aug: float = AUDIO_COND_TIMESTEP,
              text_tags: Optional[torch.Tensor] = None, denoise_mask: Optional[torch.Tensor] = None,
              audio_denoise_mask: Optional[torch.Tensor] = None) -> TimePlan:
    """Assign every token its timestep and AdaLN modulation row.

    `sigma` is the video sigma as an fp32 0-d CPU tensor. Streams share one clock (t = 1 - sigma) except
    audio, which runs on its own shifted schedule; conditioning rows are pinned near t = 1. A denoise mask
    value m puts a row at sigma = m * sigma, i.e. label 1 - m*sigma, clamped at the conditioning label.
    """
    text_len, lat_t, lat_h, lat_w, _ = layout.signature
    t_v = float(1.0 - sigma)
    t_a = float(1.0 - time_shift_sigma(sigma, shifts[0], shifts[1]))
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, visual_aug), "ref_img": max(t_v, visual_aug),
             "cond_audio": max(t_a, audio_aug), "ref_audio": max(t_a, audio_aug)}
    pin_v = max(t_v, VISUAL_COND_TIMESTEP)
    pin_a = max(t_a, AUDIO_COND_TIMESTEP)

    def per_row_or_uniform(kind: str, rows_t: torch.Tensor) -> Optional[torch.Tensor]:
        if rows_t.unique().numel() == 1:
            seg_t[kind] = float(rows_t[0])
            return None
        return rows_t

    video_rows_t = audio_rows_t = None
    if denoise_mask is not None:
        m = mask_row_values(denoise_mask[0, 0].to(torch.float32).cpu(), lat_t, lat_h, lat_w)
        if m is not None:
            video_rows_t = per_row_or_uniform("video", (1.0 - m * sigma).clamp(max=pin_v))
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].to(torch.float32).cpu().reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            audio_rows_t = per_row_or_uniform("audio", (1.0 - m * (1.0 - t_a)).clamp(max=pin_a))

    values = {t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
    for rows_t in (video_rows_t, audio_rows_t):
        if rows_t is not None:
            values |= set(rows_t.unique().tolist())
    t_values = sorted(values)
    t_row = {t: i for i, t in enumerate(t_values)}

    def row_indices(rows_t: torch.Tensor, tag: int) -> torch.Tensor:
        levels = rows_t.unique()
        base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()], dtype=torch.long)
        return base[torch.searchsorted(levels, rows_t)]

    tags = None
    if text_tags is not None:
        tags = [int(v) for v in text_tags.reshape(-1).tolist()]
        if len(tags) != text_len or any(v not in (TAG_VIDEO, TAG_TEXT, TAG_AUDIO) for v in tags):
            raise ValueError(f"text_token_tags must hold {text_len} values in {{0,1,2}}")

    mod: List[Tuple[int, int, ModRow]] = []
    for a, b, kind in layout.segments:
        base = t_row[seg_t[kind]] * 3
        if kind == "text" and tags is not None:
            start = 0  # runs of equal tag (vision-pad tokens carry the video modality)
            for i in range(1, len(tags) + 1):
                if i == len(tags) or tags[i] != tags[start]:
                    mod.append((a + start, a + i, base + tags[start]))
                    start = i
        elif kind == "video" and video_rows_t is not None:
            mod.append((a, b, row_indices(video_rows_t, TAG_VIDEO)))
        elif kind == "audio" and audio_rows_t is not None:
            mod.append((a, b, row_indices(audio_rows_t, TAG_AUDIO)))
        else:
            mod.append((a, b, base + KIND_TAG[kind]))

    def final_row(kind: str, rows_t: Optional[torch.Tensor]) -> Tuple[int, int, ModRow]:
        s = find_segment(layout, kind)
        return (s.start, s.stop, row_indices(rows_t, 0) // 3 if rows_t is not None else t_row[seg_t[kind]])

    return TimePlan(t_values, mod, final_row("video", video_rows_t), final_row("audio", audio_rows_t), t_v, t_a)
