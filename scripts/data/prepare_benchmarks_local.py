#!/usr/bin/env python3
"""Prepare SR benchmark datasets without relying on cv.snu.ac.kr.

Downloads HR images from HuggingFace (eugenesiow/* datasets, mirror-friendly)
and generates LR_bicubic/X2 X3 X4 locally with the repository's MATLAB-style
`imresize_np` + modcrop — the same recipe used for the existing local Set5
(see docs/B2RSR_v1_Gate_Diagnostic_Milestone_zh.md §2.3).

Usage (on a Featurize instance):
    python scripts/data/prepare_benchmarks_local.py --data-root /home/featurize/data
    # 国内网络推荐加镜像：
    HF_ENDPOINT=https://hf-mirror.com python scripts/data/prepare_benchmarks_local.py

Notes:
  * Set5/Set14/BSD100/Urban100 are covered. Manga109 requires a manual
    license application and is NOT downloaded here.
  * LR generated this way is sufficient for mechanism screening; absolute
    PSNR for the final paper should be re-checked against the official
    benchmark distribution (same caveat as the existing Set5).
"""

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from data.util import imresize_np, modcrop  # noqa: E402

DATASETS = {
    "Set5": {"repo": "eugenesiow/Set5", "file": "Set5_HR.tar.gz", "count": 5},
    "Set14": {"repo": "eugenesiow/Set14", "file": "Set14_HR.tar.gz", "count": 14},
    "BSD100": {"repo": "eugenesiow/BSD100", "file": "BSD100_HR.tar.gz", "count": 100},
    "Urban100": {"repo": "eugenesiow/Urban100", "file": "Urban100_HR.tar.gz", "count": 100},
}
SCALES = (2, 3, 4)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def hf_url(repo, filename, endpoint):
    return "{}/datasets/{}/resolve/main/data/{}".format(endpoint.rstrip("/"), repo, filename)


def download(url, dest):
    print("  下载 {}".format(url))
    req = urllib.request.Request(url, headers={"User-Agent": "b2rsr-prepare/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print("\r  {:.1f}/{:.1f} MB".format(done / 1e6, total / 1e6), end="")
        print()


def list_images(directory):
    return sorted(p for p in Path(directory).rglob("*")
                  if p.suffix.lower() in IMG_EXTS and p.is_file())


def generate_lr(hr_dir, lr_root, scales):
    hr_paths = list_images(hr_dir)
    for scale in scales:
        out_dir = lr_root / "X{}".format(scale)
        out_dir.mkdir(parents=True, exist_ok=True)
        for hr_path in hr_paths:
            out_path = out_dir / "{}x{}.png".format(hr_path.stem, scale)
            if out_path.exists():
                continue
            img = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("无法读取 {}".format(hr_path))
            img = modcrop(img, scale)
            lr = imresize_np(img.astype(np.float32) / 255.0, 1.0 / scale, True)
            lr = np.clip(lr * 255.0, 0, 255).round().astype(np.uint8)
            cv2.imwrite(str(out_path), lr)
        print("  X{}: {} 张 LR 已生成".format(scale, len(hr_paths)))


def prepare_dataset(name, spec, target_root, endpoint, keep_archives):
    ds_dir = target_root / name
    hr_dir = ds_dir / "HR"
    lr_root = ds_dir / "LR_bicubic"

    existing = len(list_images(hr_dir)) if hr_dir.exists() else 0
    if existing != spec["count"]:
        print("{}: HR 缺失（{}/{}），开始下载".format(name, existing, spec["count"]))
        with tempfile.TemporaryDirectory(dir=str(target_root)) as tmp:
            archive = Path(tmp) / spec["file"]
            url = hf_url(spec["repo"], spec["file"], endpoint)
            download(url, archive)
            extract_dir = Path(tmp) / "extract"
            extract_dir.mkdir()
            with tarfile.open(archive, "r:*") as tar:
                tar.extractall(extract_dir)
            images = list_images(extract_dir)
            if len(images) != spec["count"]:
                raise RuntimeError("{} 解压后应有 {} 张，实际 {} 张".format(
                    name, spec["count"], len(images)))
            hr_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                dest = hr_dir / (img.stem + ".png")
                if img.suffix.lower() == ".png":
                    img.replace(dest)
                else:
                    data = cv2.imread(str(img), cv2.IMREAD_COLOR)
                    cv2.imwrite(str(dest), data)
            if keep_archives:
                archive.replace(target_root / spec["file"])
    else:
        print("{}: HR 已就位（{} 张）".format(name, existing))

    print("{}: 生成 LR（imresize_np + modcrop）".format(name))
    generate_lr(hr_dir, lr_root, SCALES)

    # validate
    for scale in SCALES:
        n = len(list_images(lr_root / "X{}".format(scale)))
        assert n == spec["count"], "{} X{} 应有 {} 张，实际 {}".format(name, scale, spec["count"], n)
    print("{}: 校验通过 ✔".format(name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/featurize/data")
    parser.add_argument("--datasets", default="Set14,BSD100,Urban100",
                        help="逗号分隔；可选 {}".format(",".join(DATASETS)))
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                        help="HuggingFace endpoint；国内可用 https://hf-mirror.com")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    target_root = Path(args.data_root).expanduser() / "SRBenchmarks"
    target_root.mkdir(parents=True, exist_ok=True)

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error("未知数据集: {}；可选 {}".format(unknown, list(DATASETS)))

    for name in names:
        prepare_dataset(name, DATASETS[name], target_root, args.endpoint, args.keep_archives)

    print("\n全部完成。注意：Manga109 需自行申请许可，未包含在本脚本中。")
    print("LR 为本地 imresize_np 生成，可用于机制筛查；论文正式 PSNR 需用官方数据复核。")


if __name__ == "__main__":
    main()
