#!/usr/bin/env python3
"""Prepare the 2K large-image benchmark (DIV2K_valid_HR) for cascade screening.

Downloads the official DIV2K validation HR set (100 images, ~2K resolution)
from ETH Zurich, then generates LR_bicubic/X4 locally with the repository's
imresize_np + modcrop (same recipe as prepare_benchmarks_local.py).

Usage:
    python scripts/data/prepare_large_benchmarks.py --data-root /home/featurize/data
    # 若下载失败，手动上传 DIV2K_valid_HR.zip 后：
    python scripts/data/prepare_large_benchmarks.py --archive /home/featurize/data/DIV2K_valid_HR.zip

Note: 正式论文的 Test2K/Test4K（ClassSR 协议，源自 DIV8K）需另行获取；
本集合用于 patch 级级联的机制筛查（2K 分辨率、100 张自然图像）。
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

from data.util import imresize_np, modcrop  # noqa: E402

URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
EXPECTED = 100


def download(url, dest):
    print("下载 {}\n  → {}".format(url, dest))
    req = urllib.request.Request(url, headers={"User-Agent": "b2rsr-prepare/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print("\r  {:.0f}/{:.0f} MB".format(done / 1e6, total / 1e6), end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/featurize/data")
    parser.add_argument("--archive", help="本地 DIV2K_valid_HR.zip 路径（跳过下载）")
    parser.add_argument("--scales", default="4", help="逗号分隔的 LR 尺度，默认仅 X4")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    target = data_root / "SRBenchmarks" / "DIV2K_valid_2K"
    hr_dir = target / "HR"
    scales = [int(s) for s in args.scales.split(",")]

    existing = sorted(hr_dir.glob("*.png")) if hr_dir.exists() else []
    if len(existing) != EXPECTED:
        archive = Path(args.archive).expanduser() if args.archive else (
            data_root / "DIV2K_valid_HR.zip")
        if not archive.exists():
            download(URL, archive)
        print("解压 {}".format(archive))
        hr_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".png")]
            if len(names) != EXPECTED:
                raise RuntimeError("压缩包应含 {} 张 png，实际 {}".format(EXPECTED, len(names)))
            for name in names:
                dest = hr_dir / Path(name).name
                if not dest.exists():
                    with zf.open(name) as src, open(dest, "wb") as out:
                        out.write(src.read())
        print("HR 就位：{} 张".format(len(list(hr_dir.glob('*.png')))))
    else:
        print("HR 已就位（{} 张），跳过下载".format(len(existing)))

    for scale in scales:
        lr_dir = target / "LR_bicubic" / "X{}".format(scale)
        lr_dir.mkdir(parents=True, exist_ok=True)
        hr_paths = sorted(hr_dir.glob("*.png"))
        made = 0
        for hr_path in hr_paths:
            out_path = lr_dir / "{}x{}.png".format(hr_path.stem, scale)
            if out_path.exists():
                continue
            img = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
            img = modcrop(img, scale)
            lr = imresize_np(img.astype(np.float32) / 255.0, 1.0 / scale, True)
            lr = np.clip(lr * 255.0, 0, 255).round().astype(np.uint8)
            cv2.imwrite(str(out_path), lr)
            made += 1
            if made % 20 == 0:
                print("  X{}: {}/{}".format(scale, made, len(hr_paths)))
        n = len(list(lr_dir.glob("*.png")))
        assert n == EXPECTED, "X{} 应有 {} 张，实际 {}".format(scale, EXPECTED, n)
        print("X{} LR 校验通过（{} 张）".format(scale, n))

    print("\n完成：{}".format(target))
    print("提示：正式 Test2K/4K（ClassSR/DIV8K 协议）需另行准备；本集合用于机制筛查。")


if __name__ == "__main__":
    main()
