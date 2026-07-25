#!/usr/bin/env python3
"""Check B2R-SR runtime dependencies and optional CUDA availability."""

import argparse
import importlib
import sys


PACKAGES = {
    "torch": "PyTorch",
    "numpy": "NumPy",
    "cv2": "OpenCV",
    "lmdb": "LMDB",
    "yaml": "PyYAML",
    "tensorboard": "TensorBoard",
    "PIL": "Pillow",
    "scipy": "SciPy",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="fail unless CUDA is usable")
    args = parser.parse_args()

    loaded = {}
    missing = []
    for module, name in PACKAGES.items():
        try:
            loaded[module] = importlib.import_module(module)
            version = getattr(loaded[module], "__version__", "installed")
            print("OK      {:12s} {}".format(name, version))
        except ImportError:
            missing.append(name)
            print("MISSING {}".format(name))

    if missing:
        print("\nInstall missing packages: python -m pip install -r requirements.txt")
        return 1

    torch = loaded["torch"]
    print("\nPyTorch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.randn(256, 256, device="cuda")
        print("CUDA tensor:", (x @ x).device)
    elif args.cuda:
        print("ERROR: --cuda requested but CUDA is unavailable.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
