#!/usr/bin/env python3
"""Environment self-check for B2R-SR experiments (no data required).

Verifies: python/torch/cuda/cv2/yaml versions, GPU visibility and compute,
RCAN checkpoint loading + forward, CARN-M weights, latency timing sanity.

Usage:
    python scripts/eval/check_env.py [--checkpoint /path/to/120000_G.pth]
Exit code 0 = all green.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail))
        print("  [OK]   {} — {}".format(name, detail))
    except Exception as exc:
        RESULTS.append((name, False, str(exc)))
        print("  [FAIL] {} — {}".format(name, exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    print("== B2R-SR environment self-check ==\n")

    # --- basics ---
    check("python", lambda: sys.version.split()[0])

    def _torch():
        import torch
        assert torch.cuda.is_available(), "cuda not available"
        return "torch {} | cuda build {} | device {}".format(
            torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    check("torch+cuda", _torch)

    def _cv2():
        import cv2
        import numpy as np
        img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        up = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.dct(np.float32(img[:, :, 0]))
        cv2.Sobel(img[:, :, 0].astype(np.float32), cv2.CV_32F, 1, 0)
        return "cv2 {} (resize/dct/sobel OK)".format(cv2.__version__)
    check("opencv", _cv2)

    check("pyyaml", lambda: __import__("yaml").__version__)

    def _repo():
        import utils.util as util  # noqa
        from data.util import bgr2ycbcr, imresize_np, modcrop  # noqa
        from models.archs.RCAN_arch import RCAN  # noqa
        return "codes/ imports OK"
    check("repo imports", _repo)

    # --- GPU compute sanity ---
    def _gpu_compute():
        import torch
        x = torch.rand(8, 3, 64, 64, device="cuda")
        w = torch.rand(16, 3, 3, 3, device="cuda")
        y = torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        assert y.shape == (8, 16, 64, 64)
        free, total = torch.cuda.mem_get_info()
        return "conv OK | vram {:.1f}/{:.1f} GB free".format(free / 1e9, total / 1e9)
    check("gpu compute", _gpu_compute)

    # --- checkpoint load + forward ---
    def _rcan():
        import torch
        from models.archs.RCAN_arch import RCAN
        candidates = []
        if args.checkpoint:
            candidates.append(Path(args.checkpoint).expanduser())
        candidates += [
            ROOT / "experiments/remote_exports/B2RSR_RCAN_X4_120000_export/checkpoint/120000_G.pth",
            Path.home() / "120000_G.pth",
        ]
        ckpt = next((c for c in candidates if c.exists()), None)
        assert ckpt is not None, "checkpoint not found in {}".format(
            [str(c) for c in candidates])
        rcan = RCAN(n_resgroups=10, n_resblocks=20, n_feats=64, res_scale=1,
                    n_colors=3, rgb_range=255, scale=4, reduction=16)
        state = torch.load(str(ckpt), map_location="cpu")
        bs = {}
        for key, value in state.items():
            clean = key[7:] if key.startswith("module.") else key
            if clean.startswith("backbone."):
                bs[clean[len("backbone."):]] = value
        rcan.load_state_dict(bs if bs else dict(state), strict=True)
        rcan.eval().cuda()
        x = torch.rand(1, 3, 48, 48, device="cuda") * 255
        with torch.no_grad():
            y = rcan(x)
        torch.cuda.synchronize()
        assert y.shape == (1, 3, 192, 192)
        return "strict load + x4 forward OK ({})".format(ckpt.name)
    check("rcan checkpoint", _rcan)

    # --- CARN-M (auto-download if missing) ---
    def _carn():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "co", str(ROOT / "scripts/eval/run_cascade_oracle.py"))
        co = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(co)
        import torch
        net = co.load_carn_m(torch.device("cuda"))
        x = torch.rand(1, 3, 64, 64, device="cuda")
        with torch.no_grad():
            y = net(x)
        torch.cuda.synchronize()
        assert y.shape == (1, 3, 256, 256)
        return "official CARN-M load + forward OK"
    check("carn-m weights", _carn)

    # --- latency timing sanity ---
    def _timing():
        import torch
        x = torch.rand(1, 3, 64, 64, device="cuda")
        w = torch.rand(64, 3, 3, 3, device="cuda")
        for _ in range(10):
            torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.nn.functional.conv2d(x, w, padding=1)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        assert 0 < ms < 1000
        return "cuda events OK ({:.3f} ms)".format(ms)
    check("cuda timing", _timing)

    print()
    fails = [r for r in RESULTS if not r[1]]
    if fails:
        print("RESULT: {} FAILED — {}".format(
            len(fails), ", ".join(r[0] for r in fails)))
        sys.exit(1)
    print("RESULT: ALL {} CHECKS PASSED — environment ready.".format(len(RESULTS)))


if __name__ == "__main__":
    main()
