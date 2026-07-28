#!/usr/bin/env python3
"""Stage-2a: generate patch-level router labels for the cascade framework.

For every LR patch (configurable stride) we record:
  * the LR patch pixels (uint8)          -> router input
  * bicubic_drop = PSNR_Y(dense) - PSNR_Y(bicubic)  -> supervision target
  * dense_gt PSNR                        -> difficulty context

Dense RCAN is run once per image (full LR image), bicubic once per image, so
the cost is ~1 dense forward per image regardless of stride.

Default source is the already-prepared DIV2K_valid_2K (100 images); use
stride < patch to densify. Pass --dataset-dir to use a larger set (e.g. a
prepared DIV2K_train_2K) once downloaded.

Output: labels npz (patches + drops + image ids) + summary json, in
results/router_labels/<stamp>/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

import utils.util as util  # noqa: E402
from data.util import bgr2ycbcr, modcrop  # noqa: E402
from models.archs.RCAN_arch import RCAN  # noqa: E402


def resolve_checkpoint(path_arg):
    candidates = []
    if path_arg:
        p = Path(path_arg).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        candidates.append(p)
    candidates.append(
        ROOT / "experiments/remote_exports/B2RSR_RCAN_X4_120000_export/checkpoint/120000_G.pth")
    src = next((c.resolve() for c in candidates if c.exists()), None)
    if src is None:
        raise FileNotFoundError("未找到 checkpoint，用 --checkpoint 指定")
    return src


def build_backbone(checkpoint, device):
    rcan = RCAN(n_resgroups=10, n_resblocks=20, n_feats=64, res_scale=1,
                n_colors=3, rgb_range=255, scale=4, reduction=16)
    state = torch.load(str(checkpoint), map_location="cpu")
    bs = {k[len("backbone."):]: v for k, v in
          ((kk[7:] if kk.startswith("module.") else kk, vv) for kk, vv in state.items())
          if k.startswith("backbone.")}
    rcan.load_state_dict(bs if bs else dict(state), strict=True)
    rcan.eval()
    for p in rcan.parameters():
        p.requires_grad = False
    return rcan.to(device)


def psnr_y(a_img, b_img, shave=4):
    a = a_img[shave:-shave, shave:-shave]
    b = b_img[shave:-shave, shave:-shave]
    a_y = bgr2ycbcr(a / 255.0, only_y=True) * 255.0
    b_y = bgr2ycbcr(b / 255.0, only_y=True) * 255.0
    return float(util.calculate_psnr(a_y, b_y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", default="/home/featurize/data")
    parser.add_argument("--dataset-dir", default="DIV2K_valid_2K")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32,
                        help="LR patch stride；<patch 时重叠采样以扩充标签量")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("需要 CUDA GPU。")
    device = torch.device("cuda:0")
    scale, ps, stride = args.scale, args.patch, args.stride

    base = Path(args.data_root) / "SRBenchmarks" / args.dataset_dir
    hr_dir, lr_dir = base / "HR", base / "LR_bicubic" / "X{}".format(scale)
    if not hr_dir.is_dir():
        raise FileNotFoundError("未找到 {}；先运行 prepare_large_benchmarks.py".format(hr_dir))
    pairs = []
    for hr_path in sorted(hr_dir.glob("*.png")):
        lr_path = lr_dir / "{}x{}.png".format(hr_path.stem, scale)
        if lr_path.exists():
            pairs.append((hr_path, lr_path))
        if args.max_images and len(pairs) >= args.max_images:
            break
    print("{}: {} 张图，patch={} stride={}".format(
        args.dataset_dir, len(pairs), ps, stride))

    rcan = build_backbone(resolve_checkpoint(args.checkpoint), device)

    patches, drops, dense_psnrs, img_ids, coords = [], [], [], [], []
    t0 = time.time()
    for idx, (hr_path, lr_path) in enumerate(pairs):
        hr_img = modcrop(cv2.imread(str(hr_path), cv2.IMREAD_COLOR), scale)
        lr_img = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
        lq = torch.from_numpy(np.ascontiguousarray(
            np.transpose(lr_img[:, :, [2, 1, 0]].astype(np.float32), (2, 0, 1)))
        ).unsqueeze(0).to(device)  # BGR->RGB 对齐训练管道
        with torch.no_grad():
            dense = rcan(lq)
        dense_img = util.tensor2img(dense.detach()[0], out_type=np.uint8, min_max=(0, 255))
        bicubic_img = cv2.resize(lr_img, (lr_img.shape[1] * scale, lr_img.shape[0] * scale),
                                 interpolation=cv2.INTER_CUBIC)
        del lq, dense
        torch.cuda.empty_cache()

        h_lr, w_lr = lr_img.shape[:2]
        for py in range(0, h_lr - ps + 1, stride):
            for px in range(0, w_lr - ps + 1, stride):
                hy, hx, hps = py * scale, px * scale, ps * scale
                gt_p = hr_img[hy:hy + hps, hx:hx + hps]
                d_ps = psnr_y(dense_img[hy:hy + hps, hx:hx + hps], gt_p)
                b_ps = psnr_y(bicubic_img[hy:hy + hps, hx:hx + hps], gt_p)
                patches.append(lr_img[py:py + ps, px:px + ps].copy())
                drops.append(d_ps - b_ps)
                dense_psnrs.append(d_ps)
                img_ids.append(hr_path.stem)
                coords.append((py, px))
        if (idx + 1) % 10 == 0:
            print("  {}/{} 张（{} patch，{:.0f}s）".format(
                idx + 1, len(pairs), len(patches), time.time() - t0))

    patches = np.stack(patches).astype(np.uint8)   # [N, ps, ps, 3] BGR
    drops = np.asarray(drops, dtype=np.float32)
    dense_psnrs = np.asarray(dense_psnrs, dtype=np.float32)
    img_ids = np.asarray(img_ids)
    coords = np.asarray(coords, dtype=np.int32)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "results/router_labels" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "labels.npz", patches=patches, drops=drops,
                        dense_psnr=dense_psnrs, image_ids=img_ids, coords=coords)
    summary = {
        "generated": stamp,
        "dataset_dir": args.dataset_dir,
        "images": len(pairs),
        "patches": int(len(drops)),
        "patch": ps, "stride": stride, "scale": scale,
        "drop_stats": {
            "mean": float(drops.mean()), "median": float(np.median(drops)),
            "p10": float(np.percentile(drops, 10)), "p90": float(np.percentile(drops, 90)),
            "frac_le_0.05": float((drops <= 0.05).mean()),
            "frac_le_0.1": float((drops <= 0.1).mean()),
            "frac_le_0.2": float((drops <= 0.2).mean()),
        },
        "npz_mb": round((out_dir / "labels.npz").stat().st_size / 1e6, 1),
    }
    (out_dir / "labels_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n完成：{}".format(out_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
