#!/usr/bin/env python3
"""Patch-level cascade oracle kill-check for mainline alpha (SG-alpha).

Question answered (training-free): on large images (~2K), if a PERFECT
router sent each LR patch either to a cheap path or to dense RCAN, what is
the best achievable quality-matched latency?

Cheap paths screened:
  * bicubic     — cv2.resize INTER_CUBIC upscaling of the LR patch (sub-ms CPU)
  * carn_m      — official pretrained CARN-M (nmhkahn), ~412K params, GPU
  (head-tail RCAN d=0 was screened earlier and removed: 0% usable patches)

Per patch we compute PSNR-Y of each path against GT and against dense RCAN
output. The oracle keeps the cheap path when its quality loss vs dense RCAN
output is within eps; otherwise the patch escalates to dense RCAN.

Latency model: measured per-patch median latency (CUDA events) of each path
at the fixed patch size, multiplied by oracle counts, plus measured batched
escalation. Whole-image dense RCAN latency is measured directly as baseline.
This is an upper bound screening (no router cost, no boundary handling).

Outputs one timestamped directory + tar.gz:
  cascade_oracle_report.{json,md}, per_patch.csv

Pre-registered decision rule (SG-alpha):
  For eps = 0.1 dB, oracle speedup (dense whole-image latency / oracle
  cascade latency) must be >= 1.3x on the 2K set; otherwise mainline alpha
  is closed.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import statistics
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

import utils.util as util  # noqa: E402
from data.util import bgr2ycbcr, modcrop  # noqa: E402
from models.archs.RCAN_arch import RCAN  # noqa: E402


def mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path_arg):
    temporary = None
    candidates = []
    if path_arg:
        explicit = Path(path_arg).expanduser()
        if not explicit.exists():
            raise FileNotFoundError("指定的 checkpoint/export 不存在: {}".format(explicit))
        candidates.append(explicit)
    candidates.extend([
        ROOT / "experiments/remote_exports/B2RSR_RCAN_X4_120000_export/checkpoint/120000_G.pth",
        Path("/home/featurize/data/B2RSR_RCAN_X4_120000_export.tar"),
        Path("/home/featurize/work/B2RSR_RCAN_X4_120000_export.tar"),
    ])
    source = next((p.resolve() for p in candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError("未找到 checkpoint；用 --checkpoint 指定。")
    if source.suffix == ".pth":
        return source, temporary
    temporary = tempfile.TemporaryDirectory(prefix="b2rsr-cascade-")
    with tarfile.open(source, "r") as archive:
        archive.extractall(temporary.name)
    matches = list(Path(temporary.name).rglob("*_G.pth"))
    if len(matches) != 1:
        temporary.cleanup()
        raise RuntimeError("导出包中应恰好包含一个 *_G.pth")
    return matches[0], temporary


def build_backbone(checkpoint, device):
    rcan = RCAN(n_resgroups=10, n_resblocks=20, n_feats=64, res_scale=1,
                n_colors=3, rgb_range=255, scale=4, reduction=16)
    state = torch.load(str(checkpoint), map_location="cpu")
    backbone_state = {}
    for key, value in state.items():
        clean = key[7:] if key.startswith("module.") else key
        if clean.startswith("backbone."):
            backbone_state[clean[len("backbone."):]] = value
    if not backbone_state:
        backbone_state = dict(state)
    rcan.load_state_dict(backbone_state, strict=True)
    rcan.eval()
    for p in rcan.parameters():
        p.requires_grad = False
    return rcan.to(device)


class _OfficialMeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super().__init__()
        sign = -1 if sub else 1
        r, g, b = mean_rgb
        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r * sign, g * sign, b * sign])
        for p in self.shifter.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.shifter(x)


class _OfficialBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class _OfficialEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=group),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return F.relu(self.body(x) + x)


class _OfficialUpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, group=1):
        super().__init__()
        modules = []
        if scale in (2, 4, 8):
            for _ in range(int(np.log2(scale))):
                modules += [nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group),
                            nn.PixelShuffle(2), nn.ReLU(inplace=True)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group),
                        nn.PixelShuffle(3), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        return self.body(x)


class _OfficialMultiUpsample(nn.Module):
    def __init__(self, n_channels, group=1):
        super().__init__()
        self.up2 = _OfficialUpsampleBlock(n_channels, 2, group)
        self.up3 = _OfficialUpsampleBlock(n_channels, 3, group)
        self.up4 = _OfficialUpsampleBlock(n_channels, 4, group)

    def forward(self, x, scale):
        return getattr(self, "up{}".format(scale))(x)


class _OfficialCarnBlock(nn.Module):
    def __init__(self, nf, group=1):
        super().__init__()
        self.b1 = _OfficialEResidualBlock(nf, nf, group=group)
        self.c1 = _OfficialBasicBlock(nf * 2, nf, 1, 1, 0)
        self.c2 = _OfficialBasicBlock(nf * 3, nf, 1, 1, 0)
        self.c3 = _OfficialBasicBlock(nf * 4, nf, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        b2 = self.b1(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        b3 = self.b1(o2)
        c3 = torch.cat([c2, b3], dim=1)
        return self.c3(c3)


class OfficialCARNM(nn.Module):
    """Bit-exact module tree for nmhkahn/CARN-pytorch carn_m.pth."""

    def __init__(self, group=4):
        super().__init__()
        nf = 64
        self.sub_mean = _OfficialMeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = _OfficialMeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry = nn.Conv2d(3, nf, 3, 1, 1)
        self.b1 = _OfficialCarnBlock(nf, group)
        self.b2 = _OfficialCarnBlock(nf, group)
        self.b3 = _OfficialCarnBlock(nf, group)
        self.c1 = _OfficialBasicBlock(nf * 2, nf, 1, 1, 0)
        self.c2 = _OfficialBasicBlock(nf * 3, nf, 1, 1, 0)
        self.c3 = _OfficialBasicBlock(nf * 4, nf, 1, 1, 0)
        self.upsample = _OfficialMultiUpsample(nf, group)
        self.exit = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, x, scale=4):
        # official CARN expects RGB in [0, 1]
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x
        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        out = self.upsample(o3, scale)
        out = self.exit(out)
        return self.add_mean(out)


def load_carn_m(device):
    path = ROOT / "experiments/pre_trained_models/carn_m.pth"
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://raw.githubusercontent.com/nmhkahn/CARN-pytorch/"
               "master/checkpoint/carn_m.pth")
        print("下载官方 CARN-M 权重: {}".format(url))
        urllib.request.urlretrieve(url, str(path))
    net = OfficialCARNM(group=4)
    state = torch.load(str(path), map_location="cpu")
    net.load_state_dict(state, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net.to(device)


def headtail_forward(rcan, x):
    """RCAN d=0: head -> body-tail conv -> global residual -> tail."""
    x = rcan.sub_mean(x)
    x = rcan.head(x)
    res = rcan.body[-1](x)
    res = res + x
    x = rcan.tail(res)
    return rcan.add_mean(x)


def img_to_tensor(img_bgr, device):
    # matches LQGT_rcan pipeline: BGR->RGB, HWC->CHW, float [0,255]
    img_rgb = img_bgr[:, :, [2, 1, 0]]
    tensor = torch.from_numpy(
        np.ascontiguousarray(np.transpose(img_rgb, (2, 0, 1)))
    ).float().unsqueeze(0)
    return tensor.to(device)


def tensor_to_img(tensor):
    return util.tensor2img(tensor.detach()[0], out_type=np.uint8, min_max=(0, 255))


def psnr_y(a_img, b_img, shave=4):
    a = a_img[shave:-shave, shave:-shave]
    b = b_img[shave:-shave, shave:-shave]
    a_y = bgr2ycbcr(a / 255.0, only_y=True) * 255.0
    b_y = bgr2ycbcr(b / 255.0, only_y=True) * 255.0
    return float(util.calculate_psnr(a_y, b_y))


def timed(fn, warmup, runs, device):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        timings = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    return timings


def load_pairs(data_root, dataset_dir, scale, max_images):
    base = Path(data_root) / "SRBenchmarks" / dataset_dir
    hr_dir = base / "HR"
    lr_dir = base / "LR_bicubic" / "X{}".format(scale)
    if not hr_dir.is_dir() or not lr_dir.is_dir():
        raise FileNotFoundError(
            "未找到 {} 或 {}；请先运行 scripts/data/prepare_large_benchmarks.py".format(
                hr_dir, lr_dir))
    pairs = []
    for hr_path in sorted(hr_dir.glob("*.png")):
        lr_path = lr_dir / "{}x{}.png".format(hr_path.stem, scale)
        if lr_path.exists():
            pairs.append((hr_path, lr_path))
        if max_images and len(pairs) >= max_images:
            break
    if not pairs:
        raise RuntimeError("没有配对图像")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", default="/home/featurize/data")
    parser.add_argument("--dataset-dir", default="DIV2K_valid_2K")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0=全部")
    parser.add_argument("--patch", type=int, default=64,
                        help="LR patch 大小（ClassSR 用 32 的 LR patch；64 平衡调度开销）")
    parser.add_argument("--eps", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--latency-images", type=int, default=3,
                        help="整图 dense 延迟测量的图数")
    parser.add_argument("--output-dir")
    parser.add_argument("--archive")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("需要 CUDA GPU。")
    device = torch.device("cuda:0")
    scale = args.scale
    ps = args.patch
    eps_list = [float(v) for v in args.eps.split(",")]

    checkpoint, temporary = resolve_checkpoint(args.checkpoint)
    try:
        rcan = build_backbone(checkpoint, device)
        pairs = load_pairs(args.data_root, args.dataset_dir, scale, args.max_images)
        print("数据集 {}: {} 张图，LR patch={}px".format(
            args.dataset_dir, len(pairs), ps))

        # ------------------------------------------------------------------
        # 1) per-patch-size latency of each path (fixed shapes, batch=1)
        # ------------------------------------------------------------------
        carn = load_carn_m(device)
        probe = torch.rand(1, 3, ps, ps, device=device) * 255.0
        probe01 = torch.rand(1, 3, ps, ps, device=device)
        lat_dense_patch = statistics.median(
            timed(lambda: rcan(probe), args.warmup, args.runs, device))
        lat_carn_patch = statistics.median(
            timed(lambda: carn(probe01), args.warmup, args.runs, device))
        # carn batched (cheap path can batch too)
        probe01_b16 = torch.rand(16, 3, ps, ps, device=device)
        lat_carn_b16 = statistics.median(
            timed(lambda: carn(probe01_b16), max(5, args.warmup // 2),
                  max(20, args.runs // 2), device)) / 16.0
        # bicubic cheap path: cv2.resize INTER_CUBIC (deployment-realistic, C++ CPU)
        probe_u8 = (np.random.rand(ps, ps, 3) * 255).astype(np.uint8)
        t0 = time.perf_counter()
        for _ in range(200):
            cv2.resize(probe_u8, (ps * scale, ps * scale), interpolation=cv2.INTER_CUBIC)
        lat_bicubic_patch = (time.perf_counter() - t0) / 200 * 1000.0
        print("patch 级延迟: dense={:.3f} ms, carn_m(b1)={:.3f} ms, carn_m(b16)={:.3f} ms, bicubic(cv2)={:.4f} ms".format(
            lat_dense_patch, lat_carn_patch, lat_carn_b16, lat_bicubic_patch))

        # batched escalation latency per patch (batch=16), for honest upper bound
        batch16 = torch.rand(16, 3, ps, ps, device=device) * 255.0
        lat_dense_b16 = statistics.median(
            timed(lambda: rcan(batch16), max(5, args.warmup // 2),
                  max(20, args.runs // 2), device)) / 16.0
        print("dense patch (batch=16, 每 patch): {:.3f} ms".format(lat_dense_b16))

        # ------------------------------------------------------------------
        # 2) whole-image dense latency baseline (first N images)
        # ------------------------------------------------------------------
        whole_lat = []
        for hr_path, lr_path in pairs[:args.latency_images]:
            lr_img = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
            lq = img_to_tensor(lr_img.astype(np.float32), device)
            times = timed(lambda: rcan(lq), max(5, args.warmup // 2),
                          max(20, args.runs // 2), device)
            whole_lat.append(statistics.median(times))
            del lq
            torch.cuda.empty_cache()
        lat_dense_whole = mean(whole_lat)
        print("整图 dense median 平均（{} 张）: {:.1f} ms".format(
            len(whole_lat), lat_dense_whole))

        # ------------------------------------------------------------------
        # 3) per-patch quality: bicubic / headtail vs dense RCAN and GT
        # ------------------------------------------------------------------
        csv_rows = []
        patch_records = []
        t_start = time.time()
        for idx, (hr_path, lr_path) in enumerate(pairs):
            hr_img = modcrop(cv2.imread(str(hr_path), cv2.IMREAD_COLOR), scale)
            lr_img = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
            lq_full = img_to_tensor(lr_img.astype(np.float32), device)
            with torch.no_grad():
                dense_full = rcan(lq_full)
                carn_full = carn(lq_full / 255.0) * 255.0
            dense_img = tensor_to_img(dense_full)
            carn_img = tensor_to_img(carn_full)
            bicubic_img = cv2.resize(
                lr_img, (lr_img.shape[1] * scale, lr_img.shape[0] * scale),
                interpolation=cv2.INTER_CUBIC)
            del lq_full, dense_full, carn_full
            torch.cuda.empty_cache()

            h_lr, w_lr = lr_img.shape[:2]
            for py in range(0, h_lr - ps + 1, ps):
                for px in range(0, w_lr - ps + 1, ps):
                    hy, hx, hps = py * scale, px * scale, ps * scale
                    gt_p = hr_img[hy:hy + hps, hx:hx + hps]
                    dense_p = dense_img[hy:hy + hps, hx:hx + hps]
                    bi_p = bicubic_img[hy:hy + hps, hx:hx + hps]
                    ca_p = carn_img[hy:hy + hps, hx:hx + hps]
                    rec = {
                        "image": hr_path.stem, "py": py, "px": px,
                        "dense_gt": psnr_y(dense_p, gt_p),
                        "bicubic_gt": psnr_y(bi_p, gt_p),
                        "carn_gt": psnr_y(ca_p, gt_p),
                    }
                    rec["bicubic_drop"] = rec["dense_gt"] - rec["bicubic_gt"]
                    rec["carn_drop"] = rec["dense_gt"] - rec["carn_gt"]
                    patch_records.append(rec)
                    csv_rows.append(rec)
            if (idx + 1) % 10 == 0:
                print("  {}/{} 张图（{:.0f}s）".format(
                    idx + 1, len(pairs), time.time() - t_start))

        n_patches = len(patch_records)
        patches_per_image = n_patches / len(pairs)
        print("共 {} 个 patch（平均每图 {:.0f} 个）".format(n_patches, patches_per_image))

        # ------------------------------------------------------------------
        # 4) oracle ladders
        # ------------------------------------------------------------------
        def oracle(cheap_key, eps):
            keep = sum(1 for r in patch_records if r[cheap_key + "_drop"] <= eps)
            frac_cheap = keep / n_patches
            cheap_ms = {"bicubic": lat_bicubic_patch,
                        "carn_m": lat_carn_b16}[cheap_key]
            per_image_ms = patches_per_image * (
                frac_cheap * cheap_ms + (1 - frac_cheap) * lat_dense_b16)
            return {
                "eps_db": eps,
                "cheap_fraction": frac_cheap,
                "oracle_ms_per_image": per_image_ms,
                "speedup_vs_dense_whole": lat_dense_whole / per_image_ms
                if per_image_ms > 0 else float("inf"),
            }

        ladders = {}
        for cheap_key in ("bicubic", "carn_m"):
            ladders[cheap_key] = [oracle(cheap_key, e) for e in eps_list]

        # combined best-of-both cheap path
        combined = []
        for eps in eps_list:
            keep = sum(1 for r in patch_records
                       if min(r["bicubic_drop"], r["carn_drop"]) <= eps)
            frac = keep / n_patches
            per_image_ms = patches_per_image * (
                frac * max(lat_bicubic_patch, lat_carn_b16)
                + (1 - frac) * lat_dense_b16)
            combined.append({
                "eps_db": eps, "cheap_fraction": frac,
                "oracle_ms_per_image": per_image_ms,
                "speedup_vs_dense_whole": lat_dense_whole / per_image_ms,
            })
        ladders["best_of_both"] = combined

        # SG-alpha verdict at eps=0.1
        verdicts = {}
        for key, rows in ladders.items():
            row = next((r for r in rows if abs(r["eps_db"] - 0.1) < 1e-9), None)
            if row:
                verdicts[key] = {
                    "eps": 0.1,
                    "speedup": row["speedup_vs_dense_whole"],
                    "pass_1.3x": row["speedup_vs_dense_whole"] >= 1.3,
                }

        # ------------------------------------------------------------------
        # 5) export
        # ------------------------------------------------------------------
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path(args.output_dir) if args.output_dir else (
            ROOT / "results/cascade_oracle" / stamp)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "generated": stamp,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "dataset_dir": args.dataset_dir,
            "images": len(pairs),
            "scale": scale,
            "patch_lr": ps,
            "patches_total": n_patches,
            "patches_per_image": patches_per_image,
            "latency": {
                "dense_whole_image_ms": lat_dense_whole,
                "dense_patch_b1_ms": lat_dense_patch,
                "dense_patch_b16_per_patch_ms": lat_dense_b16,
                "carn_m_patch_b1_ms": lat_carn_patch,
                "carn_m_patch_b16_per_patch_ms": lat_carn_b16,
                "bicubic_patch_cv2_ms": lat_bicubic_patch,
                "warmup": args.warmup, "runs": args.runs,
            },
            "quality_summary": {
                "mean_dense_gt": mean([r["dense_gt"] for r in patch_records]),
                "mean_bicubic_drop": mean([r["bicubic_drop"] for r in patch_records]),
                "mean_carn_drop": mean([r["carn_drop"] for r in patch_records]),
                "frac_bicubic_drop_le_0.1": sum(
                    1 for r in patch_records if r["bicubic_drop"] <= 0.1) / n_patches,
                "frac_carn_drop_le_0.1": sum(
                    1 for r in patch_records if r["carn_drop"] <= 0.1) / n_patches,
            },
            "oracle_ladders": ladders,
            "sg_alpha_verdicts_eps0.1": verdicts,
            "notes": [
                "oracle 为零成本完美路由的上界；未计入 router 开销与 patch 边界处理。",
                "escalation 按 batch=16 dense patch 摊销；carn_m 同按 batch=16 摊销；bicubic 为 cv2.resize INTER_CUBIC。",
                "carn_m 为官方预训练权重（nmhkahn/CARN-pytorch），输入 RGB [0,1]。",
                "SG-alpha 预注册门槛：eps=0.1 dB 时 oracle speedup >= 1.3x。",
            ],
        }
        (output_dir / "cascade_oracle_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2))

        # markdown
        lines = [
            "# Cascade Oracle Kill-Check (SG-alpha)", "",
            "- dataset: {} ({} images, {} patches, LR patch {}px)".format(
                args.dataset_dir, len(pairs), n_patches, ps),
            "- device: {}".format(report["device"]),
            "- dense whole-image: {:.1f} ms".format(lat_dense_whole),
            "- patch latencies: dense b1 {:.2f} / dense b16 {:.2f} / carn_m b16 {:.3f} / bicubic(cv2) {:.4f} ms".format(
                lat_dense_patch, lat_dense_b16, lat_carn_b16, lat_bicubic_patch),
            "",
            "## Oracle ladders", "",
        ]
        for key, rows in ladders.items():
            lines.extend(["### cheap path: {}".format(key), "",
                          "| eps (dB) | cheap % | oracle ms/img | speedup |",
                          "|---:|---:|---:|---:|"])
            for r in rows:
                lines.append("| {:.2f} | {:.1%} | {:.1f} | {:.3f}x |".format(
                    r["eps_db"], r["cheap_fraction"],
                    r["oracle_ms_per_image"], r["speedup_vs_dense_whole"]))
            lines.append("")
        lines.extend(["## SG-alpha verdict (eps=0.1, threshold 1.3x)", ""])
        for key, v in verdicts.items():
            lines.append("- {}: {:.3f}x → {}".format(
                key, v["speedup"], "PASS" if v["pass_1.3x"] else "FAIL"))
        lines.extend(["", "## Notes", ""] + ["- " + n for n in report["notes"]])
        (output_dir / "cascade_oracle_report.md").write_text("\n".join(lines))

        with open(output_dir / "per_patch.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        archive = (Path(args.archive).expanduser() if args.archive else
                   output_dir.parent / "B2RSR_CASCADE_ORACLE_{}.tar.gz".format(stamp))
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)

        print("\n完成。请下载这一个文件供分析：\n{}".format(archive))
        print("\n摘要：")
        print((output_dir / "cascade_oracle_report.md").read_text())
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
