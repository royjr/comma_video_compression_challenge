#!/usr/bin/env python
"""Neural compression pipeline, MPS port of quantizr's training-time encoder.

Differences from upstream quantizr/compress.py:
  - Loads frame pairs via PyAV (CPU) instead of NVIDIA DALI (CUDA-only).
  - Encodes the SegNet mask track with libsvtav1 instead of libaom-av1
    (Homebrew ffmpeg ships svtav1; produces the same standard AV1 bitstream
    that PyAV in inflate.py can decode).
  - Defaults --device to mps and removes CUDA-only sync calls.
  - Honors PIPELINE_EPOCHS env var (scales every stage by the same factor) so
    we can run shorter trial runs and longer overnight runs from the same
    code.
The architecture, FP4 quantization, training schedule, and inflate-side state
dict format are unchanged so quantizr/inflate.py works as-is.
"""
import os
import sys
import math
import argparse
import shutil
import subprocess
import tempfile
import logging
import io
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import av
import brotli
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from tqdm import tqdm
from safetensors.torch import load_file

# Repo root: this file lives in <root>/attempts/recipes/neural_mps_v1/ when
# invoked from the harness, OR in <root>/submissions/<name>/ when invoked
# directly. Walk up until we find frame_utils.py.
_here = Path(__file__).resolve().parent
ROOT_DIR = _here
for _ in range(6):
    if (ROOT_DIR / "frame_utils.py").exists():
        break
    ROOT_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from frame_utils import seq_len as SEQ_LEN, camera_size  # noqa: E402
from modules import SegNet, PoseNet, DistortionNet, segnet_sd_path, posenet_sd_path  # noqa: E402

SEGNET_MODEL_INPUT_SIZE = (512, 384)


class Stage(Enum):
    ANCHOR = "anchor"
    FINETUNE = "finetune"
    JOINT = "joint"


@dataclass
class PipelineRun:
    name: str
    stage: Stage
    epochs: int
    lr: float
    qat_start_epoch: int
    frame1_fade_epochs: int = 0
    error_boost: float = 4.0
    ce_weight: float = 1.0
    pose_weight: float = 1.0
    warmup_epochs: int = 2
    ema_decay: float = 0.99
    grad_clip: float = 1.0


# -----------------------------
# Helpers
# -----------------------------
def diff_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1.0 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply_shadow(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data
                p.data = self.shadow[n]

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
        self.backup = {}


def diff_rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    h, w = rgb_chw.shape[-2:]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., : 2 * h2, : 2 * w2]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0)
    u = ((b - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((r - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    y00, y10, y01, y11 = y[:, 0::2, 0::2], y[:, 1::2, 0::2], y[:, 0::2, 1::2], y[:, 1::2, 1::2]
    u_sub = (u[:, 0::2, 0::2] + u[:, 1::2, 0::2] + u[:, 0::2, 1::2] + u[:, 1::2, 1::2]) * 0.25
    v_sub = (v[:, 0::2, 0::2] + v[:, 1::2, 0::2] + v[:, 0::2, 1::2] + v[:, 1::2, 1::2]) * 0.25
    return torch.stack([y00, y10, y01, y11, u_sub, v_sub], dim=1)


def pack_pair_yuv6(f1, f2):
    return torch.cat([diff_rgb_to_yuv6(f1), diff_rgb_to_yuv6(f2)], dim=1)


def get_pose_tensor(out):
    return out["pose"] if isinstance(out, dict) else getattr(out, "pose", out["pose"])


def make_coord_grid(b, h, w, device, dtype):
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0).unsqueeze(0).expand(b, -1, -1, -1)


def kl_on_logits(s, t, T=2.0):
    return F.kl_div(F.log_softmax(s / T, dim=1), F.softmax(t / T, dim=1), reduction="batchmean") * (T * T)


# -----------------------------
# Data loading (PyAV, CPU-side)
# -----------------------------
def preload_video_pair_cache_av(file_names, data_dir, device):
    """Decode every frame from every video into a (N_pairs, 2, H, W, 3) uint8 tensor on CPU."""
    from frame_utils import yuv420_to_rgb, frame_count

    logging.info("Preloading raw video RGB pairs into memory via PyAV...")
    all_pairs = []
    for fn in file_names:
        path = str(data_dir / fn)
        nframes = frame_count(path)
        n_pairs = nframes // SEQ_LEN
        logging.info(f"  {fn}: {nframes} frames -> {n_pairs} pairs")

        container = av.open(path)
        stream = container.streams.video[0]
        seq = []
        pairs = []
        for frame in container.decode(stream):
            seq.append(yuv420_to_rgb(frame))  # (H, W, 3) uint8
            if len(seq) == SEQ_LEN:
                pairs.append(torch.stack(seq))  # (2, H, W, 3)
                seq = []
            if len(pairs) >= n_pairs:
                break
        container.close()
        all_pairs.append(torch.stack(pairs))  # (n_pairs, 2, H, W, 3)
    return torch.cat(all_pairs, dim=0).contiguous()


# -----------------------------
# Mask track (libsvtav1 instead of libaom-av1)
# -----------------------------
def extract_and_compress_masks(rgb_pairs_all, segnet, device, crf, archive_dir, batch_size=8):
    expected_frames = rgb_pairs_all.shape[0]
    raw_path = archive_dir / f"raw_masks_crf{crf}.yuv"
    obu_path = archive_dir / f"mask_crf{crf}.obu"
    obu_br_path = archive_dir / f"mask_crf{crf}.obu.br"
    stable_obu_br_path = archive_dir / "mask.obu.br"

    if obu_br_path.exists():
        try:
            with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
                tmp.write(brotli.decompress(obu_br_path.read_bytes()))
                tmp_path = tmp.name
            container = av.open(tmp_path)
            frames = []
            for fr in container.decode(video=0):
                img = fr.to_ndarray(format="gray")
                frames.append(np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4))
            container.close()
            os.remove(tmp_path)
            if len(frames) == expected_frames:
                logging.info(f"Cached mask is complete ({len(frames)} frames). Reusing.")
                shutil.copyfile(obu_br_path, stable_obu_br_path)
                return torch.from_numpy(np.stack(frames)).contiguous()
            logging.warning(f"Cached mask incomplete ({len(frames)}/{expected_frames}). Regenerating.")
        except Exception as e:
            logging.warning(f"Cache reload failed ({e}); regenerating.")

    logging.info("Generating odd-frame raw masks...")
    with open(raw_path, "wb") as f_out, torch.inference_mode():
        for start in tqdm(range(0, expected_frames, batch_size), desc="Extracting masks"):
            batch = rgb_pairs_all[start:start + batch_size].to(device).float()
            batch = einops.rearrange(batch, "b t h w c -> b t c h w")
            odd = batch[:, 1]
            resized = F.interpolate(odd, size=(SEGNET_MODEL_INPUT_SIZE[1], SEGNET_MODEL_INPUT_SIZE[0]), mode="bilinear")
            mask = segnet(resized).argmax(dim=1).to(torch.uint8) * 63
            f_out.write(mask.cpu().numpy().tobytes())

    # libsvtav1 needs yuv420p; expand single-channel mask into yuv420p with U,V=128.
    raw420_path = archive_dir / f"raw_masks_crf{crf}.yuv420p"
    width, height = 512, 384
    n = expected_frames
    raw_y = np.fromfile(raw_path, dtype=np.uint8, count=n * width * height).reshape(n, height, width)
    uv_plane = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    with open(raw420_path, "wb") as f_out:
        for i in range(n):
            f_out.write(raw_y[i].tobytes())
            f_out.write(uv_plane.tobytes())
            f_out.write(uv_plane.tobytes())

    logging.info(f"Encoding masks with libsvtav1 (CRF {crf})...")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-r", "10",
        "-i", str(raw420_path),
        "-c:v", "libsvtav1", "-preset", "0", "-crf", str(crf),
        "-svtav1-params", "keyint=-1:scd=0:tune=0:enable-tf=0",
        "-f", "obu", str(obu_path),
    ]
    subprocess.run(cmd, check=True)

    obu_br_path.write_bytes(brotli.compress(obu_path.read_bytes(), quality=11, lgwin=24))
    shutil.copyfile(obu_br_path, stable_obu_br_path)

    logging.info("Decoding back to RAM for training cache...")
    container = av.open(str(obu_path))
    frames = []
    for fr in container.decode(video=0):
        img = fr.to_ndarray(format="gray")
        frames.append(np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4))
    container.close()

    if len(frames) != expected_frames:
        raise RuntimeError(f"libsvtav1 produced {len(frames)} frames; expected {expected_frames}")

    obu_path.unlink()
    raw_path.unlink()
    raw420_path.unlink()
    return torch.from_numpy(np.stack(frames)).contiguous()


def extract_and_compress_poses(rgb_pairs_all, posenet, device, archive_dir, batch_size=8):
    br_path = archive_dir / "pose.npy.br"
    all_pose6 = []
    logging.info("Generating poses...")
    with torch.inference_mode():
        for start in tqdm(range(0, rgb_pairs_all.shape[0], batch_size), desc="Extracting poses"):
            batch = rgb_pairs_all[start:start + batch_size].to(device).float()
            batch = einops.rearrange(batch, "b t h w c -> b t c h w")
            posenet_in = posenet.preprocess_input(batch)
            out = posenet(posenet_in)
            pose6 = out["pose"][..., :6].to(torch.float32)
            all_pose6.append(pose6.cpu().numpy())
    pose_arr = np.concatenate(all_pose6, axis=0)
    buf = io.BytesIO()
    np.save(buf, pose_arr)
    br_path.write_bytes(brotli.compress(buf.getvalue(), quality=11, lgwin=24))
    return torch.from_numpy(pose_arr).float().contiguous()


class CachedPairLoader:
    def __init__(self, rgb_pairs, mask2, pose6, batch_size, device, seed=123, shuffle=True):
        self.rgb_pairs = rgb_pairs.contiguous()
        self.mask2 = mask2.contiguous()
        self.pose6 = pose6.contiguous()
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.n = rgb_pairs.shape[0]

    def set_epoch(self, e): self.epoch = int(e)
    def __len__(self): return math.ceil(self.n / self.batch_size)

    def __iter__(self):
        g = torch.Generator(device="cpu"); g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g) if self.shuffle else torch.arange(self.n)
        for s in range(0, self.n, self.batch_size):
            idx = perm[s:s + self.batch_size]
            yield (self.rgb_pairs.index_select(0, idx).to(self.device, non_blocking=True),
                   self.mask2.index_select(0, idx).to(self.device, non_blocking=True),
                   self.pose6.index_select(0, idx).to(self.device, non_blocking=True))


# -----------------------------
# FP4 codebook (unchanged from quantizr)
# -----------------------------
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def quantize_blockwise(x, block_size=32):
        orig_shape = x.shape
        flat = x.reshape(-1)
        pad = (block_size - (flat.numel() % block_size)) % block_size
        if pad: flat = F.pad(flat, (0, pad))
        blocks = flat.view(-1, block_size)
        max_abs = blocks.abs().amax(dim=1, keepdim=True)
        scales = torch.where(max_abs > 0, max_abs / 6.0, torch.ones_like(max_abs))
        norm = blocks / scales
        signs = (norm < 0).to(torch.int16)
        levels = FP4Codebook.pos_levels.to(x.device, x.dtype).view(1, 1, -1)
        mag_idx = (norm.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.int16)
        q = torch.where(signs.bool(), -levels[0, 0, mag_idx.long()], levels[0, 0, mag_idx.long()])
        return (q * scales).view(-1)[:x.numel()].view(orig_shape), ((signs << 3) | mag_idx).to(torch.uint8), scales.squeeze(1)

    @staticmethod
    def dequantize_from_nibbles(nibbles, scales, orig_shape):
        flat_n = int(torch.tensor(orig_shape).prod().item())
        nibbles = nibbles.view(-1, nibbles.numel() // scales.numel())
        signs, mag_idx = (nibbles >> 3).to(torch.int64), (nibbles & 0x7).to(torch.int64)
        levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q = torch.where(signs.bool(), -levels[mag_idx], levels[mag_idx])
        return (q * scales[:, None].to(torch.float32)).view(-1)[:flat_n].reshape(orig_shape)


def fake_quant_fp4_ste(x, block_size=32):
    dq, _, _ = FP4Codebook.quantize_blockwise(x, block_size=block_size)
    return x + (dq - x).detach()


def pack_nibbles(nib):
    flat = nib.reshape(-1)
    if flat.numel() % 2 == 1: flat = F.pad(flat, (0, 1))
    return ((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)


def unpack_nibbles(packed, count):
    flat = packed.reshape(-1)
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2], out[1::2] = (flat >> 4) & 0x0F, flat & 0x0F
    return out[:count]


def load_fp4_state_dict(model, fp4_path, device):
    data = torch.load(fp4_path, map_location=device)
    sd, source = {}, data.get("tensors", data.get("quantized", {}))
    for name, rec in source.items():
        if rec["weight_kind"] == "fp4_packed":
            nibbles = unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            w = FP4Codebook.dequantize_from_nibbles(nibbles, rec["scales_fp16"].to(device), rec["weight_shape"])
        else:
            w = rec["weight_fp16"].to(device).float()
        sd[f"{name}.weight"] = w
        if rec.get("bias_fp16") is not None:
            sd[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for k, v in data.get("dense_fp16", {}).items():
        sd[k] = v.to(device).float()
    model.load_state_dict(sd, strict=False)


def export_fp4_state_dict(model, out_path, block_size=32):
    export = {"__format__": "fp4_standalone", "__block_size__": block_size,
              "__codebook__": FP4Codebook.pos_levels.clone(),
              "quantized": {}, "dense_fp16": {}}
    covered = set()
    for name, m in model.named_modules():
        if isinstance(m, (QConv2d, QEmbedding)):
            rec = {"type": "conv2d" if isinstance(m, QConv2d) else "embedding"}
            w = m.weight.detach().float().cpu()
            rec["weight_shape"] = list(w.shape)
            covered.add(f"{name}.weight")
            if isinstance(m, QConv2d):
                rec["stride"] = list(m.stride) if isinstance(m.stride, tuple) else [m.stride] * 2
                rec["padding"] = list(m.padding) if isinstance(m.padding, tuple) else [m.padding] * 2
                rec["dilation"] = list(m.dilation) if isinstance(m.dilation, tuple) else [m.dilation] * 2
                rec["groups"] = int(m.groups)
                rec["bias_fp16"] = m.bias.detach().half().cpu() if m.bias is not None else None
                if m.bias is not None: covered.add(f"{name}.bias")
            if getattr(m, "quantize_weight", False):
                _, nib, scales = FP4Codebook.quantize_blockwise(w, block_size=block_size)
                rec.update({"weight_kind": "fp4_packed", "weight_numel": int(w.numel()),
                            "packed_weight": pack_nibbles(nib.cpu()),
                            "scales_fp16": scales.half().cpu()})
            else:
                rec.update({"weight_kind": "fp16", "weight_fp16": w.half().cpu()})
            export["quantized"][name] = rec
    for k, v in model.state_dict().items():
        if k not in covered:
            export["dense_fp16"][k] = (v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu())
    torch.save(export, out_path, _use_new_zipfile_serialization=False)


# -----------------------------
# Quantizable modules + architecture (unchanged)
# -----------------------------
class QMixin:
    def set_qat(self, enabled, act_enabled=False):
        self.qat_enabled = enabled
        self.qat_act_enabled = act_enabled


class QConv2d(nn.Conv2d, QMixin):
    def __init__(self, *a, block_size=32, quantize_weight=True, **kw):
        super().__init__(*a, **kw); self.block_size, self.quantize_weight, self.qat_enabled = block_size, quantize_weight, False
    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QEmbedding(nn.Embedding, QMixin):
    def __init__(self, *a, block_size=32, quantize_weight=True, **kw):
        super().__init__(*a, **kw); self.block_size, self.quantize_weight, self.qat_enabled = block_size, quantize_weight, False
    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.embedding(x, w, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)


class SepConvGNAct(nn.Module):
    def __init__(self, ic, oc, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid = ic * depth_mult
        self.dw = QConv2d(ic, mid, k, stride=stride, padding=k // 2, groups=ic, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid, oc, 1, padding=0, bias=True, quantize_weight=quantize_weight)
        self.norm = nn.GroupNorm(2, oc); self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.norm(self.pw(self.dw(x))))


class SepConv(nn.Module):
    def __init__(self, ic, oc, k=3, stride=1, depth_mult=4, quantize_weight=True):
        super().__init__()
        mid = ic * depth_mult
        self.dw = QConv2d(ic, mid, k, stride=stride, padding=k // 2, groups=ic, bias=False, quantize_weight=quantize_weight)
        self.pw = QConv2d(mid, oc, 1, padding=0, bias=True, quantize_weight=quantize_weight)
    def forward(self, x): return self.pw(self.dw(x))


class SepResBlock(nn.Module):
    def __init__(self, ch, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch); self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(x + self.norm2(self.conv2(self.conv1(x))))


class FiLMSepResBlock(nn.Module):
    def __init__(self, ch, cond_dim, depth_mult=4, quantize_weight=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.conv2 = SepConv(ch, ch, 3, 1, depth_mult=depth_mult, quantize_weight=quantize_weight)
        self.norm2 = nn.GroupNorm(2, ch); self.film_proj = nn.Linear(cond_dim, ch * 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x, cond):
        b = self.norm2(self.conv2(self.conv1(x)))
        gamma, beta = self.film_proj(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return self.act(x + (b * (1.0 + gamma) + beta))


class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=56, c2=64, depth_mult=1):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)
        self.stem_conv = SepConvGNAct(emb_dim + 2, c1, depth_mult=depth_mult)
        self.stem_block = SepResBlock(c1, depth_mult=depth_mult)
        self.down_conv = SepConvGNAct(c1, c2, stride=2, depth_mult=depth_mult)
        self.down_block = SepResBlock(c2, depth_mult=depth_mult)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                SepConvGNAct(c2, c1, depth_mult=depth_mult))
        self.fuse = SepConvGNAct(c1 + c1, c1, depth_mult=depth_mult)
        self.fuse_block = SepResBlock(c1, depth_mult=depth_mult)
    def forward(self, mask2, coords):
        e2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        e2_up = F.interpolate(e2, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        s = self.stem_block(self.stem_conv(torch.cat([e2_up, coords], dim=1)))
        z = self.up(self.down_block(self.down_conv(s)))
        return self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))


class Frame2StaticHead(nn.Module):
    def __init__(self, in_ch, hidden=52, depth_mult=1):
        super().__init__()
        self.block1 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)
    def forward(self, f): return torch.sigmoid(self.head(self.pre(self.block2(self.block1(f))))) * 255.0


class FrameHead(nn.Module):
    def __init__(self, in_ch, cond_dim=48, hidden=52, depth_mult=1):
        super().__init__()
        self.block1 = FiLMSepResBlock(in_ch, cond_dim, depth_mult=depth_mult)
        self.block2 = SepResBlock(in_ch, depth_mult=depth_mult)
        self.pre = SepConvGNAct(in_ch, hidden, depth_mult=depth_mult)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)
    def forward(self, f, cond): return torch.sigmoid(self.head(self.pre(self.block2(self.block1(f, cond))))) * 255.0


class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, depth_mult=1):
        super().__init__()
        self.shared_trunk = SharedMaskDecoder(num_classes=num_classes, emb_dim=6, c1=56, c2=64, depth_mult=depth_mult)
        self.pose_mlp = nn.Sequential(nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.frame1_head = FrameHead(in_ch=56, cond_dim=cond_dim, hidden=52, depth_mult=depth_mult)
        self.frame2_head = Frame2StaticHead(in_ch=56, hidden=52, depth_mult=depth_mult)

    def set_qat(self, enabled):
        for m in self.modules():
            if isinstance(m, (QConv2d, QEmbedding)): m.set_qat(enabled=enabled)

    def forward(self, mask2, pose6):
        coords = make_coord_grid(mask2.shape[0], 384, 512, mask2.device, torch.float32)
        feat = self.shared_trunk(mask2, coords)
        return self.frame1_head(feat, self.pose_mlp(pose6)), self.frame2_head(feat)


# -----------------------------
# Training engine (unchanged structure, MPS-safe)
# -----------------------------
def apply_freeze(model, stage):
    for p in model.parameters(): p.requires_grad = True
    if stage == Stage.ANCHOR:
        for p in model.frame1_head.parameters(): p.requires_grad = False
        for p in model.pose_mlp.parameters(): p.requires_grad = False
    elif stage == Stage.FINETUNE:
        for p in model.shared_trunk.parameters(): p.requires_grad = False
        for p in model.frame2_head.parameters(): p.requires_grad = False
    if stage == Stage.FINETUNE:
        model.shared_trunk.eval(); model.frame2_head.eval()


def train_run(run, model, loader, device, archive_dir, aux, prev_sd=None):
    segnet, posenet, dist_net = aux
    apply_freeze(model, run.stage)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=run.lr, betas=(0.9, 0.99))
    start_epoch, best = 0, float("inf")

    latest = archive_dir / f"{run.name}_latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer_state"])
        start_epoch, best = ck["epoch"] + 1, ck["best_metric"]
    elif prev_sd is not None:
        model.load_state_dict(prev_sd)

    ema = EMA(model, decay=run.ema_decay) if run.ema_decay > 0 else None
    if ema and latest.exists() and ck.get("ema_state"):
        ema.shadow = {k: v.to(device) for k, v in ck["ema_state"].items()}

    qw = min(run.warmup_epochs, max(1, (run.epochs - run.qat_start_epoch) // 2)) if run.qat_start_epoch == 0 else run.warmup_epochs
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=qw)
    main = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, run.epochs - qw))
    sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, main], milestones=[qw])
    for _ in range(start_epoch): sch.step()

    for epoch in range(start_epoch, run.epochs):
        model.train()
        if run.stage == Stage.FINETUNE:
            model.shared_trunk.eval(); model.frame2_head.eval()
        loader.set_epoch(epoch)
        qat_on = epoch >= run.qat_start_epoch
        model.set_qat(qat_on)

        if epoch == run.qat_start_epoch and run.qat_start_epoch > 0:
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=run.lr, betas=(0.9, 0.99))
            warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=qw)
            main = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, (run.epochs - epoch) - qw))
            sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, main], milestones=[qw])

        kl_alpha = min(1.0, epoch / max(1, run.qat_start_epoch // 2)) if run.qat_start_epoch > 0 else 1.0
        seg2_kl_w = 0.9 - (0.9 * kl_alpha)
        seg2_ce_w = 0.1 + (0.9 * kl_alpha)
        f1_w = max(0.0, 1.0 - (epoch / run.frame1_fade_epochs)) if run.frame1_fade_epochs > 0 else 0.0

        sums = [0.0] * 4; nb = 0
        pbar = tqdm(loader, desc=f"{run.name} ep{epoch+1}/{run.epochs}", leave=False)
        for batch_rgb, in_mask2, in_pose6 in pbar:
            batch = einops.rearrange(batch_rgb, "b t h w c -> b t c h w").float().to(device)
            in_mask2 = in_mask2.to(device).long(); in_pose6 = in_pose6.to(device).float()

            with torch.no_grad():
                real1 = F.interpolate(batch[:, 0], size=(384, 512), mode="bilinear", align_corners=False)
                real2 = F.interpolate(batch[:, 1], size=(384, 512), mode="bilinear", align_corners=False)
                gt_l1, gt_l2 = segnet(real1).float(), segnet(real2).float()
                gt_m1, gt_m2 = gt_l1.argmax(dim=1), gt_l2.argmax(dim=1)
                gt_pose = get_pose_tensor(posenet(posenet.preprocess_input(batch))).float()[..., :6]

            opt.zero_grad(set_to_none=True)
            p1, p2 = model(in_mask2, in_pose6)
            f1u = F.interpolate(p1, size=(874, 1164), mode="bilinear", align_corners=False)
            f2u = F.interpolate(p2, size=(874, 1164), mode="bilinear", align_corners=False)
            f1d = F.interpolate(diff_round(f1u.clamp(0, 255)), size=(384, 512), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(f2u.clamp(0, 255)), size=(384, 512), mode="bilinear", align_corners=False)

            l = torch.tensor(0.0, device=device); l_pose = l_seg2 = l_seg1 = l
            l_seg2_ce = l_seg1_ce = torch.tensor(0.0, device=device)

            if run.stage in (Stage.FINETUNE, Stage.JOINT):
                fp = get_pose_tensor(posenet(pack_pair_yuv6(f1d, f2d).float())).float()[..., :6]
                l_pose = F.mse_loss(fp, gt_pose)

            if run.stage in (Stage.ANCHOR, Stage.JOINT):
                fl2 = segnet(f2d).float()
                ce2 = F.cross_entropy(fl2, gt_m2, reduction="none")
                with torch.no_grad():
                    boost = 1.0 + ((fl2.argmax(dim=1) != gt_m2).float() * run.error_boost)
                l_seg2_ce = (ce2 * boost).mean()
                l_seg2_kl = kl_on_logits(fl2, gt_l2, 2.0) / (384 * 512)
                l_seg2 = 100.0 * (seg2_kl_w * l_seg2_kl + seg2_ce_w * 0.5 * run.ce_weight * l_seg2_ce)

            if f1_w > 0:
                fl1 = segnet(f1d).float()
                ce1 = F.cross_entropy(fl1, gt_m1, reduction="none")
                with torch.no_grad():
                    boost1 = 1.0 + ((fl1.argmax(dim=1) != gt_m1).float() * run.error_boost)
                l_seg1_ce = (ce1 * boost1).mean()
                if run.stage == Stage.JOINT:
                    l_seg1_kl = kl_on_logits(fl1, gt_l1, 2.0) / (384 * 512)
                    l_seg1 = 100.0 * f1_w * (seg2_kl_w * l_seg1_kl + seg2_ce_w * 0.5 * run.ce_weight * l_seg1_ce)
                else:
                    l_seg1 = 100.0 * f1_w * (run.ce_weight * l_seg1_ce)

            if run.stage == Stage.ANCHOR: l = l_seg2
            elif run.stage == Stage.FINETUNE: l = l_seg1 + (run.pose_weight * l_pose * 10.0)
            else: l = l_seg2 + l_seg1 + (30.0 * run.pose_weight * l_pose)

            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=run.grad_clip)
            opt.step()
            if ema and epoch >= run.warmup_epochs: ema.update(model)
            sums[0] += l.item(); sums[1] += l_seg2_ce.item(); sums[2] += l_seg1_ce.item(); sums[3] += l_pose.item(); nb += 1
            pbar.set_postfix({"L": f"{l.item():.2f}", "S": f"{l_seg2_ce.item():.2f}", "P": f"{l_pose.item():.4f}"})

        sch.step()
        logging.info(f"ep{epoch+1}/{run.epochs} {'[QAT]' if qat_on else ''} L={sums[0]/max(1,nb):.3f} "
                     f"S2={sums[1]/max(1,nb):.3f} S1={sums[2]/max(1,nb):.3f} P={sums[3]/max(1,nb):.5f}")

        is_eval = qat_on and (((epoch - run.qat_start_epoch) % 5 == 0) or (run.epochs - epoch <= 10))
        if is_eval:
            if ema: ema.apply_shadow(model)

            # HONEST EVAL: round-trip the model through FP4 export+reload so the
            # score we compute matches what inflate.py will produce at deploy time.
            # Without this, the live FP32 EMA model evaluates ~5x better than the
            # FP4-quantized version that actually ships (PoseNet is hyper-sensitive
            # to small weight drift). Cost: one extra export+load per eval.
            tmp_fp4 = archive_dir / "_eval_tmp_fp4.pt"
            export_fp4_state_dict(model.cpu(), tmp_fp4); model.to(device)
            eval_model = JointFrameGenerator().to(device).eval()
            load_fp4_state_dict(eval_model, tmp_fp4, device)
            tmp_fp4.unlink(missing_ok=True)

            ts, tp, ns = 0.0, 0.0, 0
            with torch.inference_mode():
                for batch_rgb, in_mask2, in_pose6 in tqdm(loader, desc=f"eval {run.name} ep{epoch+1}", leave=False):
                    batch_gt = batch_rgb.to(device)
                    p1, p2 = eval_model(in_mask2.to(device).long(), in_pose6.to(device).float())
                    bc = torch.stack([F.interpolate(p1, size=(874, 1164), mode="bilinear", align_corners=False),
                                      F.interpolate(p2, size=(874, 1164), mode="bilinear", align_corners=False)], dim=1)
                    bc = einops.rearrange(bc, "b t c h w -> b t h w c").clamp(0, 255).round().to(torch.uint8)
                    pd, sd = dist_net.compute_distortion(batch_gt, bc)
                    ts += sd.sum().item(); tp += pd.sum().item(); ns += batch_gt.shape[0]
            del eval_model

            avg_seg, avg_pose = ts / max(1, ns), tp / max(1, ns)
            mf = archive_dir / "model.pt.br"; ms = mf.stat().st_size if mf.exists() else 70000
            kf = archive_dir / "mask.obu.br"; ks = kf.stat().st_size if kf.exists() else 0
            pf = archive_dir / "pose.npy.br"; ps = pf.stat().st_size if pf.exists() else 0
            total_bytes = ms + ks + ps
            uncomp = 37545489
            rate = total_bytes / uncomp
            score = 100.0 * avg_seg + math.sqrt(max(0, 10.0 * avg_pose)) + 25.0 * rate
            logging.info(f"  [Eval] est score={score:.4f} seg={100*avg_seg:.4f} pose={math.sqrt(10*avg_pose):.4f} rate={25*rate:.4f}")

            if score < best:
                best = score
                torch.save({k: (v.half() if torch.is_floating_point(v) else v) for k, v in model.state_dict().items()},
                           archive_dir / f"{run.name}_best_fp16.pt")
                fp4 = archive_dir / f"{run.name}_best_fp4.pt"
                export_fp4_state_dict(model.cpu(), fp4); model.to(device)
                comp = brotli.compress(fp4.read_bytes(), quality=11, lgwin=24)
                (archive_dir / f"{run.name}_best_fp4.pt.br").write_bytes(comp)
                shutil.copyfile(archive_dir / f"{run.name}_best_fp4.pt.br", archive_dir / "model.pt.br")
                logging.info(f"  *** new best score={best:.4f} (saved model.pt.br) ***")
            if ema: ema.restore(model)

        torch.save({"epoch": epoch, "best_metric": best,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "ema_state": {k: v.cpu() for k, v in ema.shadow.items()} if ema else None}, latest)

    if latest.exists(): latest.unlink()
    return load_best_fp4(model, archive_dir / f"{run.name}_best_fp4.pt", device)


def load_best_fp4(model, path, device):
    load_fp4_state_dict(model, path, device)
    model.float()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", type=Path, default=ROOT_DIR / "videos")
    p.add_argument("--video-names", type=Path, default=ROOT_DIR / "public_test_video_names.txt")
    p.add_argument("--crf", type=int, default=50, help="mask track CRF (libsvtav1)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--archive-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)

    archive_dir = args.archive_dir if args.archive_dir is not None else (Path(__file__).parent / "archive")
    archive_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(archive_dir / "pipeline.log")])
    logging.info(f"Device={device}  archive_dir={archive_dir}")

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for m in (segnet, posenet, dist_net):
        for p in m.parameters(): p.requires_grad = False

    files = [ln.strip() for ln in args.video_names.read_text().splitlines() if ln.strip()]
    rgb_pairs = preload_video_pair_cache_av(files, args.video_dir, device)
    logging.info(f"rgb_pairs.shape={tuple(rgb_pairs.shape)}")

    mask_frames = extract_and_compress_masks(rgb_pairs, segnet, device, args.crf, archive_dir)
    pose6 = extract_and_compress_poses(rgb_pairs, posenet, device, archive_dir)

    loader = CachedPairLoader(rgb_pairs, mask_frames, pose6, args.batch_size, device)
    gen = JointFrameGenerator().to(device)

    epoch_scale = float(os.environ.get("PIPELINE_EPOCHS_SCALE", "1.0"))
    pipeline = [
        PipelineRun("run1_anchor", Stage.ANCHOR, int(400 * epoch_scale), 5e-4, int(200 * epoch_scale), int(50 * epoch_scale), error_boost=9.0),
        PipelineRun("run2_anchor_boost", Stage.ANCHOR, int(80 * epoch_scale), 1e-5, 0, 0, error_boost=49.0),
        PipelineRun("run3_finetune", Stage.FINETUNE, int(320 * epoch_scale), 5e-5, int(120 * epoch_scale), int(60 * epoch_scale), pose_weight=1.0),
        PipelineRun("run4_finish", Stage.JOINT, int(160 * epoch_scale), 1e-5, 0, int(40 * epoch_scale), pose_weight=1.0),
        PipelineRun("run5_micro", Stage.FINETUNE, int(120 * epoch_scale), 5e-6, 0, 0, pose_weight=1.0),
    ]

    cur_sd = None
    for run in pipeline:
        if run.epochs <= 0: continue
        best = archive_dir / f"{run.name}_best_fp4.pt"
        latest = archive_dir / f"{run.name}_latest.pt"
        if best.exists() and not latest.exists():
            logging.info(f"[skip] {run.name} done; loading prior best")
            cur_sd = load_best_fp4(gen, best, device); continue
        logging.info(f"\n=== {run.name} ({run.stage.value}, epochs={run.epochs}, lr={run.lr}) ===")
        cur_sd = train_run(run, gen, loader, device, archive_dir, (segnet, posenet, dist_net), cur_sd)

    logging.info("\nDone. Final model saved to archive/model.pt.br")


if __name__ == "__main__":
    main()
