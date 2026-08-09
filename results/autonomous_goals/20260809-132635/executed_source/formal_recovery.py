#!/usr/bin/env python3
"""Formal EDSR-L static-depth recovery for Autonomous Goal 20260809-132635.

The EDSR module layout follows EDSR-PyTorch revision
8dba5581a7502b92de9641eb431130d6c8ca5d7f (MIT, Sanghyun Son, 2018).
Official checkpoints are strict-loaded; deployment checkpoints contain only the
physical 24-block Student state_dict.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn


GOAL_ID = "20260809-132635"
BENCHMARK_COUNTS = {"Set5": 5, "Set14": 14, "BSD100": 100, "Urban100": 100, "Manga109": 109}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
NATIVE_MODULE_TYPES = {"EDSR", "MeanShift", "Conv2d", "Sequential", "ResBlock", "ReLU", "Upsampler", "PixelShuffle"}
FORBIDDEN_DEPLOYMENT_TERMS = ("teacher", "router", "mask", "keep_map", "keepmap", "source_index", "source_indices", "mapping", "scheduler", "dynamic")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def torch_load_weights(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def torch_load_resume(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def torch_save_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    protocol = json_read(path)
    if protocol.get("goal_id") != GOAL_ID or not protocol.get("frozen"):
        raise RuntimeError("Formal commands require the frozen Goal 20260809-132635 protocol")
    goal_root = path.parent
    manifest_path = goal_root / str(protocol["source_manifest"])
    manifest = json_read(manifest_path)
    if manifest.get("goal_id") != GOAL_ID:
        raise RuntimeError("Source manifest goal ID mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("Source manifest has no files mapping")
    required = {
        "protocol.json",
        "executed_source/formal_recovery.py",
        "executed_source/run_featurize.sh",
        "executed_source/test_edsr_checkpoint.py",
    }
    if not required.issubset(files):
        raise RuntimeError(f"Source manifest omits {sorted(required - set(files))}")
    observed: dict[str, str] = {}
    for relative, expected in files.items():
        candidate = (goal_root / str(relative)).resolve()
        if goal_root not in candidate.parents or not candidate.is_file():
            raise RuntimeError(f"Invalid frozen source path: {relative}")
        observed[str(relative)] = sha256_file(candidate)
        if observed[str(relative)] != expected:
            raise RuntimeError(f"Frozen source mismatch: {relative}")
    return protocol, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "files": observed,
    }


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range: float,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
        rgb_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sign: int = -1,
    ) -> None:
        super().__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean) / std
        for parameter in self.parameters():
            parameter.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, res_scale: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            default_conv(n_feats, n_feats, 3),
            nn.ReLU(True),
            default_conv(n_feats, n_feats, 3),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x).mul(self.res_scale)
        residual += x
        return residual


class Upsampler(nn.Sequential):
    def __init__(self, scale: int, n_feats: int) -> None:
        modules: list[nn.Module] = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.extend((default_conv(n_feats, 4 * n_feats, 3), nn.PixelShuffle(2)))
        elif scale == 3:
            modules.extend((default_conv(n_feats, 9 * n_feats, 3), nn.PixelShuffle(3)))
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        super().__init__(*modules)


class EDSR(nn.Module):
    def __init__(
        self,
        scale: int,
        n_resblocks: int = 32,
        n_feats: int = 256,
        rgb_range: float = 255.0,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.head = nn.Sequential(default_conv(3, n_feats, 3))
        body: list[nn.Module] = [ResBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        body.append(default_conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(Upsampler(scale, n_feats), default_conv(n_feats, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(self.sub_mean(x))
        residual = self.body(x)
        residual += x
        return self.add_mean(self.tail(residual))


def extract_tensor_state(payload: Any) -> OrderedDict[str, torch.Tensor]:
    candidate = payload
    if isinstance(payload, Mapping) and "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        candidate = payload["state_dict"]
    if not isinstance(candidate, Mapping) or not candidate:
        raise TypeError("Checkpoint is not a non-empty tensor mapping")
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, (torch.Tensor, nn.Parameter)):
            raise TypeError("Checkpoint state must contain only string-to-tensor entries")
        state[key] = value.detach().cpu()
    return state


def strict_load(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    result = nn.Module.load_state_dict(model, state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"Strict load failed: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    loaded = model.state_dict()
    mismatch = [key for key, value in state.items() if key not in loaded or not torch.equal(value.cpu(), loaded[key].cpu())]
    if mismatch:
        raise AssertionError(f"Strict-load tensor mismatch: {mismatch[:5]}")


def transplant_state(
    teacher_state: Mapping[str, torch.Tensor],
    scale: int,
    teacher_depth: int,
    student_depth: int,
    source_indices: Sequence[int],
    n_feats: int = 256,
) -> tuple[EDSR, OrderedDict[str, torch.Tensor]]:
    if len(source_indices) != student_depth or list(source_indices) != sorted(set(source_indices)):
        raise ValueError("Student source indices must be sorted, unique, and match physical depth")
    if source_indices[0] < 0 or source_indices[-1] >= teacher_depth:
        raise ValueError("Student source index is outside Teacher depth")
    student = EDSR(scale=scale, n_resblocks=student_depth, n_feats=n_feats)
    transplanted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for target_key, target_value in student.state_dict().items():
        source_key = target_key
        if target_key.startswith("body."):
            parts = target_key.split(".")
            target_index = int(parts[1])
            if target_index < student_depth:
                source_key = ".".join(["body", str(source_indices[target_index]), *parts[2:]])
            elif target_index == student_depth:
                source_key = ".".join(["body", str(teacher_depth), *parts[2:]])
        source_value = teacher_state.get(source_key)
        if source_value is None or source_value.shape != target_value.shape or source_value.dtype != target_value.dtype:
            raise ValueError(f"Invalid transplant {source_key} -> {target_key}")
        transplanted[target_key] = source_value.detach().clone()
    strict_load(student, transplanted)
    return student, transplanted


def static_audit(model: EDSR) -> dict[str, Any]:
    module_types = sorted({module.__class__.__name__ for module in model.modules()})
    non_native = sorted(set(module_types) - NATIVE_MODULE_TYPES)
    names = [name.lower() for name, _ in model.named_modules()] + [key.lower() for key in model.state_dict()]
    forbidden = sorted(term for term in FORBIDDEN_DEPLOYMENT_TERMS if any(term in name for name in names))
    hooks = sum(
        len(module._forward_pre_hooks) + len(module._forward_hooks) + len(module._backward_hooks)
        for module in model.modules()
    )
    passed = not non_native and not forbidden and hooks == 0
    return {
        "pass": passed,
        "module_types": module_types,
        "non_native_module_types": non_native,
        "forbidden_name_hits": forbidden,
        "hook_count": hooks,
        "deployment_attachments": [],
    }


def checkpoint_spec(protocol: Mapping[str, Any], scale: int) -> Mapping[str, Any]:
    return protocol["checkpoints"]["scales"][str(scale)]


def checkpoint_path(protocol: Mapping[str, Any], checkpoint_dir: Path, scale: int) -> Path:
    return checkpoint_dir / str(checkpoint_spec(protocol, scale)["filename"])


def validate_checkpoint(protocol: Mapping[str, Any], checkpoint_dir: Path, scale: int) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    specification = checkpoint_spec(protocol, scale)
    path = checkpoint_path(protocol, checkpoint_dir, scale)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if observed["bytes"] != int(specification["bytes"]) or observed["sha256"] != specification["sha256"]:
        raise RuntimeError(f"Official x{scale} checkpoint provenance mismatch: {observed}")
    state = extract_tensor_state(torch_load_weights(path))
    teacher = EDSR(scale=scale, n_resblocks=int(protocol["model"]["teacher_depth"]))
    strict_load(teacher, state)
    with torch.inference_mode():
        output = teacher(torch.linspace(0, 255, 1 * 3 * 8 * 11).reshape(1, 3, 8, 11))
    if list(output.shape) != [1, 3, 8 * scale, 11 * scale] or not bool(torch.isfinite(output).all()):
        raise AssertionError(f"Invalid strict Teacher forward for x{scale}")
    return state, {
        "path": str(path),
        "url": specification["url"],
        "bytes": observed["bytes"],
        "sha256": observed["sha256"],
        "tensor_count": len(state),
        "strict_load": True,
        "finite_forward": True,
    }


def source_indices(protocol: Mapping[str, Any]) -> list[int]:
    indices = [int(value) for value in protocol["model"]["student_source_indices"]]
    teacher_depth = int(protocol["model"]["teacher_depth"])
    student_depth = int(protocol["model"]["student_depth"])
    expected = [(index * (teacher_depth - 1) + (student_depth - 1) // 2) // (student_depth - 1) for index in range(student_depth)]
    if indices != expected or indices[0] != 0 or indices[-1] != teacher_depth - 1:
        raise AssertionError("Frozen endpoint-inclusive Student mapping is inconsistent")
    return indices


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No images in {directory}")
    return files


def canonical_pair_key(path: Path, scale: int) -> str:
    stem = re.sub(rf"x{scale}(?=_s\d+$)", "", path.stem, flags=re.IGNORECASE)
    return re.sub(rf"x{scale}$", "", stem, flags=re.IGNORECASE)


def original_div2k_id(key: str) -> str | None:
    match = re.fullmatch(r"(\d{4})(?:_s\d+)?", key)
    return match.group(1) if match else None


def pair_directories(
    hr_directory: Path,
    lr_directory: Path,
    scale: int,
    allowed_ids: set[str] | None = None,
) -> list[tuple[str, Path, Path]]:
    def indexed(directory: Path) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for path in image_files(directory):
            key = canonical_pair_key(path, scale)
            if allowed_ids is not None and original_div2k_id(key) not in allowed_ids:
                continue
            if key in result:
                raise ValueError(f"Duplicate normalized pair key {key} in {directory}")
            result[key] = path
        return result

    hr = indexed(hr_directory)
    lr = indexed(lr_directory)
    if set(hr) != set(lr):
        raise ValueError(
            f"Unpaired x{scale} data: HR-only={sorted(set(hr)-set(lr))[:5]}, "
            f"LR-only={sorted(set(lr)-set(hr))[:5]}"
        )
    if not hr:
        raise ValueError(f"No paired x{scale} images after split filtering")
    return [(key, hr[key], lr[key]) for key in sorted(hr)]


def pair_manifest(pairs: Sequence[tuple[str, Path, Path]], data_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    for key, hr, lr in pairs:
        for value in (key, str(hr.relative_to(data_root)), str(lr.relative_to(data_root)), str(hr.stat().st_size), str(lr.stat().st_size)):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return {"pair_count": len(pairs), "ordered_path_size_sha256": digest.hexdigest()}


def read_pair(hr_path: Path, lr_path: Path, scale: int, allow_modcrop: bool) -> tuple[np.ndarray, np.ndarray]:
    hr_bgr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
    lr_bgr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
    if hr_bgr is None or lr_bgr is None:
        raise ValueError(f"Unreadable pair: {hr_path}, {lr_path}")
    target_height, target_width = lr_bgr.shape[0] * scale, lr_bgr.shape[1] * scale
    if allow_modcrop:
        excess_height = hr_bgr.shape[0] - target_height
        excess_width = hr_bgr.shape[1] - target_width
        if excess_height < 0 or excess_width < 0 or excess_height >= scale or excess_width >= scale:
            raise ValueError(
                f"Invalid standard modcrop geometry: HR={hr_bgr.shape}, LR={lr_bgr.shape}, scale={scale}"
            )
        hr_bgr = hr_bgr[:target_height, :target_width]
    if hr_bgr.shape[:2] != (target_height, target_width):
        raise ValueError(f"Geometry mismatch: HR={hr_bgr.shape}, LR={lr_bgr.shape}, scale={scale}")
    return hr_bgr[:, :, ::-1].copy(), lr_bgr[:, :, ::-1].copy()


def validate_pair_samples(
    pairs: Sequence[tuple[str, Path, Path]],
    scale: int,
    minimum_lr_patch: int,
    allow_modcrop: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(f"{GOAL_ID}:x{scale}:{len(pairs)}")
    indices = {0, len(pairs) // 2, len(pairs) - 1}
    indices.update(rng.randrange(len(pairs)) for _ in range(min(29, len(pairs))))
    records = []
    for index in sorted(indices):
        key, hr_path, lr_path = pairs[index]
        hr, lr = read_pair(hr_path, lr_path, scale, allow_modcrop)
        if min(lr.shape[:2]) < minimum_lr_patch:
            raise ValueError(f"LR sample {lr_path} is smaller than frozen patch {minimum_lr_patch}")
        records.append({"key": key, "hr_shape": list(hr.shape), "lr_shape": list(lr.shape)})
    return records


def validate_all_pairs(
    pairs: Sequence[tuple[str, Path, Path]],
    scale: int,
    minimum_lr_patch: int,
    allow_modcrop: bool,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for key, hr_path, lr_path in pairs:
        hr, lr = read_pair(hr_path, lr_path, scale, allow_modcrop=allow_modcrop)
        if min(lr.shape[:2]) < minimum_lr_patch:
            raise ValueError(f"LR sample {lr_path} is smaller than frozen patch {minimum_lr_patch}")
        digest.update(key.encode("utf-8"))
        digest.update(json.dumps([list(hr.shape), list(lr.shape)], separators=(",", ":")).encode("ascii"))
    return {
        "validated_pair_count": len(pairs),
        "all_readable_rgb_valid_geometry": True,
        "standard_modcrop_allowed": allow_modcrop,
        "ordered_key_geometry_sha256": digest.hexdigest(),
    }


def training_pairs(protocol: Mapping[str, Any], data_root: Path, scale: int) -> tuple[list[tuple[str, Path, Path]], dict[str, Any]]:
    data = protocol["data"]
    first, last = (int(value) for value in data["recovery_div2k_id_range"])
    allowed = {f"{value:04d}" for value in range(first, last + 1)}
    pairs = pair_directories(
        data_root / data["train_hr"],
        data_root / str(data["train_lr_pattern"]).format(scale=scale),
        scale,
        allowed,
    )
    observed_ids = {original_div2k_id(key) for key, _, _ in pairs}
    missing_ids = sorted(allowed - observed_ids)
    if missing_ids:
        raise RuntimeError(f"Recovery split is missing DIV2K IDs: {missing_ids[:10]}")
    minimum = int(data["minimum_recovery_pairs"])
    if len(pairs) < minimum:
        raise RuntimeError(f"Only {len(pairs)} recovery pairs for x{scale}; protocol requires at least {minimum}")
    report = pair_manifest(pairs, data_root)
    report["sample_geometry"] = validate_pair_samples(
        pairs, scale, int(protocol["training"]["lr_patch"]), allow_modcrop=False
    )
    report["div2k_id_range"] = [f"{first:04d}", f"{last:04d}"]
    return pairs, report


def evaluation_pairs(protocol: Mapping[str, Any], data_root: Path, scale: int) -> dict[str, list[tuple[str, Path, Path]]]:
    data = protocol["data"]
    result = {
        "DIV2K_valid": pair_directories(
            data_root / data["validation_hr"],
            data_root / str(data["validation_lr_pattern"]).format(scale=scale),
            scale,
        )
    }
    if len(result["DIV2K_valid"]) != 100:
        raise RuntimeError(f"DIV2K validation count is {len(result['DIV2K_valid'])}, expected 100")
    benchmark_root = data_root / data["benchmark_root"]
    for name, expected in BENCHMARK_COUNTS.items():
        pairs = pair_directories(benchmark_root / name / "HR", benchmark_root / name / "LR_bicubic" / f"X{scale}", scale)
        if len(pairs) != expected:
            raise RuntimeError(f"{name} x{scale} count is {len(pairs)}, expected {expected}")
        result[name] = pairs
    return result


def git_environment(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(arguments, cwd=repo_root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()

    status = run("git", "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise RuntimeError(f"Cloud repository has tracked modifications; refusing formal run:\n{status}")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "tracked_status": status,
    }


def require_cloud_gpu(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one CUDA GPU, available={torch.cuda.is_available()}, count={torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    expected = str(protocol["cloud"]["required_gpu_substring"])
    if expected not in name:
        raise RuntimeError(f"Formal recovery requires {expected!r}, found {name!r}")
    free, total = torch.cuda.mem_get_info(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": name,
        "device_count": torch.cuda.device_count(),
        "free_memory_bytes": free,
        "total_memory_bytes": total,
    }


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sample_batch(
    pairs: Sequence[tuple[str, Path, Path]],
    scale: int,
    batch_size: int,
    patch: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    low_resolution: list[np.ndarray] = []
    high_resolution: list[np.ndarray] = []
    keys: list[str] = []
    for _ in range(batch_size):
        key, hr_path, lr_path = pairs[rng.randrange(len(pairs))]
        hr, lr = read_pair(hr_path, lr_path, scale, allow_modcrop=False)
        if lr.shape[0] < patch or lr.shape[1] < patch:
            raise ValueError(f"Training image {key} is smaller than {patch}x{patch}")
        top = rng.randrange(lr.shape[0] - patch + 1)
        left = rng.randrange(lr.shape[1] - patch + 1)
        lr = lr[top : top + patch, left : left + patch]
        hr = hr[top * scale : (top + patch) * scale, left * scale : (left + patch) * scale]
        if rng.random() < 0.5:
            lr, hr = np.flip(lr, axis=1), np.flip(hr, axis=1)
        if rng.random() < 0.5:
            lr, hr = np.flip(lr, axis=0), np.flip(hr, axis=0)
        if rng.random() < 0.5:
            lr, hr = np.transpose(lr, (1, 0, 2)), np.transpose(hr, (1, 0, 2))
        low_resolution.append(np.ascontiguousarray(lr))
        high_resolution.append(np.ascontiguousarray(hr))
        keys.append(key)

    def tensor(images: Sequence[np.ndarray]) -> torch.Tensor:
        array = np.stack(images).transpose(0, 3, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(array)).float()

    return tensor(low_resolution), tensor(high_resolution), keys


def truncate_trace(path: Path, completed_step: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if int(record["step"]) <= completed_step:
                        records.append(record)
    if [int(record["step"]) for record in records] != list(range(1, completed_step + 1)):
        raise RuntimeError("Training trace is not contiguous with the resumable state")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    return records


def completed_run_report(
    report_path: Path,
    checkpoint_path_value: Path,
    protocol_hash: str,
    source_manifest_hash: str,
    expected_steps: int,
    expected_scale: int,
    expected_seed: int,
) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    report = json_read(report_path)
    run_dir = report_path.parent
    config_path = run_dir / "run_config.json"
    trace_path = run_dir / "train_trace.jsonl"
    if not config_path.is_file() or not trace_path.is_file() or not checkpoint_path_value.is_file():
        raise RuntimeError(f"Completed run is missing config, trace, or checkpoint: {run_dir}")
    config = json_read(config_path)
    trace_steps: list[int] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                trace_steps.append(int(json.loads(line)["step"]))
    expected_trace = list(range(1, expected_steps + 1))
    consistent = (
        report.get("status") == "complete"
        and report.get("goal_id") == GOAL_ID
        and report.get("protocol_sha256") == protocol_hash
        and report.get("source_manifest_sha256") == source_manifest_hash
        and int(report.get("scale", -1)) == expected_scale
        and int(report.get("seed", -1)) == expected_seed
        and int(report.get("completed_steps", -1)) == expected_steps
        and config.get("goal_id") == GOAL_ID
        and config.get("protocol_sha256") == protocol_hash
        and config.get("source_manifest_sha256") == source_manifest_hash
        and int(config.get("scale", -1)) == expected_scale
        and int(config.get("seed", -1)) == expected_seed
        and report.get("run_config_sha256") == sha256_json(config)
        and report.get("initial_student_tensor_state_sha256") == config.get("initial_student_tensor_state_sha256")
        and report.get("training", {}).get("trace_sha256") == sha256_file(trace_path)
        and trace_steps == expected_trace
        and report.get("final_checkpoint", {}).get("sha256") == sha256_file(checkpoint_path_value)
        and report.get("final_checkpoint", {}).get("pure_state_dict") is True
        and report.get("final_checkpoint", {}).get("strict_round_trip") is True
        and report.get("final_checkpoint", {}).get("contains_forbidden_deployment_key") is False
    )
    if not consistent:
        raise RuntimeError(f"Existing completed-run artifacts are inconsistent: {run_dir}")
    return report


def command_check(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, source_freeze = load_protocol(protocol_path)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    repo_root = Path(args.repo_root).resolve()
    report: dict[str, Any] = {
        "goal_id": GOAL_ID,
        "phase": "cloud_preflight",
        "started_at_utc": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "source_freeze": source_freeze,
        "pass": False,
    }
    try:
        report["git"] = git_environment(repo_root)
        report["environment"] = require_cloud_gpu(protocol)
        usage = shutil.disk_usage(Path(args.run_root).resolve().parent)
        report["persistent_disk"] = {"free_bytes": usage.free, "total_bytes": usage.total}
        minimum_free = int(protocol["cloud"]["minimum_persistent_free_bytes"])
        if usage.free < minimum_free:
            raise RuntimeError(f"Persistent disk free bytes {usage.free} < required {minimum_free}")
        report["checkpoints"] = {}
        report["data"] = {}
        for scale in protocol["training"]["scales"]:
            _, checkpoint = validate_checkpoint(protocol, checkpoint_dir, int(scale))
            recovery_pairs, train = training_pairs(protocol, data_root, int(scale))
            train["exhaustive_readability_geometry"] = validate_all_pairs(
                recovery_pairs,
                int(scale),
                int(protocol["training"]["lr_patch"]),
                allow_modcrop=False,
            )
            evaluation = evaluation_pairs(protocol, data_root, int(scale))
            report["checkpoints"][str(scale)] = checkpoint
            report["data"][str(scale)] = {
                "recovery": train,
                "evaluation_counts": {name: len(pairs) for name, pairs in evaluation.items()},
                "evaluation_exhaustive_readability_geometry": {
                    name: validate_all_pairs(pairs, int(scale), 1, allow_modcrop=True)
                    for name, pairs in evaluation.items()
                },
            }
        report["pass"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        report["completed_at_utc"] = utc_now()
        json_write(output, report)
        raise
    report["completed_at_utc"] = utc_now()
    json_write(output, report)
    print(json.dumps({"pass": True, "output": str(output)}, sort_keys=True))


def command_train(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, source_freeze = load_protocol(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    scale, seed = int(args.scale), int(args.seed)
    if scale not in [int(value) for value in protocol["training"]["scales"]]:
        raise ValueError(f"Scale x{scale} is outside the frozen protocol")
    if seed not in [int(value) for value in protocol["training"]["seeds"]]:
        raise ValueError(f"Seed {seed} is outside the frozen protocol")
    environment = require_cloud_gpu(protocol)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    data_root = Path(args.data_root).resolve()
    run_root = Path(args.run_root).resolve()
    run_dir = run_root / "training" / f"x{scale}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "train_report.json"
    final_checkpoint = run_dir / "student_final.pt"
    settings = protocol["training"]
    steps = int(settings["steps"])
    complete = completed_run_report(
        report_path,
        final_checkpoint,
        protocol_hash,
        source_freeze["manifest_sha256"],
        steps,
        scale,
        seed,
    )
    if complete is not None:
        print(json.dumps({"status": "already_complete", "scale": scale, "seed": seed, "checkpoint": str(final_checkpoint)}, sort_keys=True))
        return

    checkpoint_state, checkpoint_report = validate_checkpoint(protocol, checkpoint_dir, scale)
    pairs, data_report = training_pairs(protocol, data_root, scale)
    initial_student, initial_state = transplant_state(
        checkpoint_state,
        scale,
        int(protocol["model"]["teacher_depth"]),
        int(protocol["model"]["student_depth"]),
        source_indices(protocol),
    )
    initial_state_sha256 = tensor_state_sha256(initial_state)
    del initial_state
    run_config = {
        "goal_id": GOAL_ID,
        "protocol_sha256": protocol_hash,
        "source_manifest_sha256": source_freeze["manifest_sha256"],
        "scale": scale,
        "seed": seed,
        "teacher_checkpoint": checkpoint_report,
        "initial_student_tensor_state_sha256": initial_state_sha256,
        "data": data_report,
        "settings": settings,
        "model": protocol["model"],
    }
    config_path = run_dir / "run_config.json"
    if config_path.is_file():
        if json_read(config_path) != run_config:
            raise RuntimeError(f"Existing run config differs from the frozen run: {config_path}")
    else:
        json_write(config_path, run_config)

    trace_path = run_dir / "train_trace.jsonl"
    resume_path = run_dir / "resume.pt"
    failure_path = run_dir / "failure.json"
    started = utc_now()
    wall_start = time.time()
    completed_step = 0
    trace_records: list[dict[str, Any]] = []
    try:
        configure_reproducibility(seed)
        device = torch.device("cuda:0")
        teacher = EDSR(scale=scale, n_resblocks=int(protocol["model"]["teacher_depth"]))
        strict_load(teacher, checkpoint_state)
        student = initial_student
        audit = static_audit(student)
        if not audit["pass"]:
            raise AssertionError(f"Static Student audit failed: {audit}")
        teacher = teacher.eval().float().to(device)
        student = student.train().float().to(device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(
            trainable,
            lr=float(settings["learning_rate"]),
            betas=tuple(float(value) for value in settings["betas"]),
            eps=float(settings["epsilon"]),
            weight_decay=float(settings["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.0)
        rng = random.Random(seed)
        if resume_path.is_file():
            resume = torch_load_resume(resume_path)
            if resume.get("run_config_sha256") != sha256_json(run_config):
                raise RuntimeError("Resume state does not match frozen run config")
            completed_step = int(resume["step"])
            strict_load(student, resume["student"])
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            rng.setstate(resume["python_rng_state"])
            np.random.set_state(resume["numpy_rng_state"])
            torch.set_rng_state(resume["torch_rng_state"].cpu())
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        trace_records = truncate_trace(trace_path, completed_step)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        batch_size = int(settings["batch"])
        patch = int(settings["lr_patch"])
        save_every = int(settings["resume_every_steps"])
        with trace_path.open("a", encoding="utf-8", buffering=1) as trace_handle:
            for step in range(completed_step + 1, steps + 1):
                lr_cpu, hr_cpu, keys = sample_batch(pairs, scale, batch_size, patch, rng)
                lr = lr_cpu.to(device)
                hr = hr_cpu.to(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                used_lr = float(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_output = teacher(lr)
                student_output = student(lr)
                teacher_l1 = torch.nn.functional.l1_loss(student_output, teacher_output)
                gt_l1 = torch.nn.functional.l1_loss(student_output, hr)
                total_loss = float(settings["teacher_l1_weight"]) * teacher_l1 + float(settings["gt_l1_weight"]) * gt_l1
                if not bool(torch.isfinite(total_loss)):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                end.record()
                end.synchronize()
                record = {
                    "step": step,
                    "sample_keys": keys,
                    "total_loss": float(total_loss.detach().item()),
                    "teacher_l1": float(teacher_l1.detach().item()),
                    "gt_l1": float(gt_l1.detach().item()),
                    "learning_rate": used_lr,
                    "gpu_step_ms": float(start.elapsed_time(end)),
                }
                if not all(math.isfinite(float(record[key])) for key in ("total_loss", "teacher_l1", "gt_l1", "gpu_step_ms")):
                    raise FloatingPointError(f"Non-finite trace at step {step}")
                trace_handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                trace_records.append(record)
                completed_step = step
                if step % save_every == 0 or step == steps:
                    torch_save_atomic(
                        {
                            "step": step,
                            "run_config_sha256": sha256_json(run_config),
                            "student": OrderedDict((key, value.detach().cpu()) for key, value in student.state_dict().items()),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "python_rng_state": rng.getstate(),
                            "numpy_rng_state": np.random.get_state(),
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state_all(),
                        },
                        resume_path,
                    )
                del lr_cpu, hr_cpu, lr, hr, teacher_output, student_output, teacher_l1, gt_l1, total_loss

        torch.cuda.synchronize()
        final_state = OrderedDict((key, value.detach().cpu()) for key, value in student.state_dict().items())
        if not all(bool(torch.isfinite(value).all()) for value in final_state.values()):
            raise FloatingPointError("Final Student state contains non-finite tensors")
        torch_save_atomic(final_state, final_checkpoint)
        loaded = extract_tensor_state(torch_load_weights(final_checkpoint))
        clone = EDSR(scale=scale, n_resblocks=int(protocol["model"]["student_depth"])).eval().float()
        strict_load(clone, loaded)
        with torch.inference_mode():
            output = clone(torch.linspace(0, 255, 1 * 3 * 8 * 11).reshape(1, 3, 8, 11))
        if list(output.shape) != [1, 3, 8 * scale, 11 * scale] or not bool(torch.isfinite(output).all()):
            raise AssertionError("Final strict round-trip forward failed")
        times = [float(record["gpu_step_ms"]) for record in trace_records[int(settings["timing_warmup_steps"]) :]]
        report = {
            "goal_id": GOAL_ID,
            "status": "complete",
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "protocol_sha256": protocol_hash,
            "source_manifest_sha256": source_freeze["manifest_sha256"],
            "scale": scale,
            "seed": seed,
            "completed_steps": completed_step,
            "environment": environment,
            "run_config_sha256": sha256_json(run_config),
            "initial_student_tensor_state_sha256": initial_state_sha256,
            "static_student_audit": audit,
            "parameters": {
                "teacher": sum(parameter.numel() for parameter in teacher.parameters()),
                "student": sum(parameter.numel() for parameter in student.parameters()),
            },
            "training": {
                "precision": "FP32",
                "amp_used": False,
                "nan_or_oom": False,
                "gpu_step_p50_ms": float(np.percentile(times, 50)),
                "gpu_step_p95_ms": float(np.percentile(times, 95)),
                "gpu_examples_per_second": int(settings["batch"]) * 1000.0 / statistics.mean(times),
                "wall_seconds": time.time() - wall_start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "trace_sha256": sha256_file(trace_path),
                "loss_window_note": "Random samples/crops make early and late loss windows descriptive only; they are not a continuation gate.",
            },
            "final_checkpoint": {
                "path": str(final_checkpoint),
                "bytes": final_checkpoint.stat().st_size,
                "sha256": sha256_file(final_checkpoint),
                "tensor_count": len(loaded),
                "pure_state_dict": True,
                "strict_round_trip": True,
                "finite_forward": True,
                "contains_forbidden_deployment_key": any(
                    any(term in key.lower() for term in FORBIDDEN_DEPLOYMENT_TERMS) for key in loaded
                ),
            },
        }
        if report["final_checkpoint"]["contains_forbidden_deployment_key"]:
            raise AssertionError("Final checkpoint contains a forbidden deployment key")
        json_write(report_path, report)
        if failure_path.exists():
            failure_path.rename(run_dir / f"superseded_failure_{int(time.time())}.json")
        print(json.dumps({"status": "complete", "scale": scale, "seed": seed, "checkpoint_sha256": report["final_checkpoint"]["sha256"]}, sort_keys=True))
    except Exception as error:
        failure = {
            "goal_id": GOAL_ID,
            "status": "failed",
            "failed_at_utc": utc_now(),
            "protocol_sha256": protocol_hash,
            "scale": scale,
            "seed": seed,
            "completed_steps": completed_step,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "oom": isinstance(error, torch.cuda.OutOfMemoryError),
        }
        json_write(failure_path, failure)
        raise


def rgb_to_y(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    return (65.738 * image[..., 0] + 129.057 * image[..., 1] + 25.064 * image[..., 2]) / 256.0 + 16.0


def psnr_y(output: np.ndarray, target: np.ndarray, shave: int) -> float:
    difference = rgb_to_y(output) - rgb_to_y(target)
    valid = difference[shave:-shave, shave:-shave]
    mse = float(np.mean(valid * valid))
    return float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_y(output: np.ndarray, target: np.ndarray, shave: int) -> float:
    first = rgb_to_y(output)[shave:-shave, shave:-shave]
    second = rgb_to_y(target)[shave:-shave, shave:-shave]
    if min(first.shape) < 11:
        raise ValueError("Image is too small for 11x11 SSIM")
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mu1 = cv2.GaussianBlur(first, (11, 11), 1.5)[5:-5, 5:-5]
    mu2 = cv2.GaussianBlur(second, (11, 11), 1.5)[5:-5, 5:-5]
    sigma1 = cv2.GaussianBlur(first * first, (11, 11), 1.5)[5:-5, 5:-5] - mu1 * mu1
    sigma2 = cv2.GaussianBlur(second * second, (11, 11), 1.5)[5:-5, 5:-5] - mu2 * mu2
    sigma12 = cv2.GaussianBlur(first * second, (11, 11), 1.5)[5:-5, 5:-5] - mu1 * mu2
    value = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2))
    return float(value.mean())


def evaluate_model(
    model: EDSR,
    datasets: Mapping[str, Sequence[tuple[str, Path, Path]]],
    scale: int,
    device: torch.device,
    output_dir: Path,
    label: str,
    save_images: bool = False,
) -> dict[str, Any]:
    model = model.eval().float().to(device)
    summaries: dict[str, Any] = {}
    with torch.inference_mode():
        for dataset, pairs in datasets.items():
            records: list[dict[str, Any]] = []
            temporary = output_dir / f"x{scale}_{dataset}_{label}.jsonl.tmp"
            final = output_dir / f"x{scale}_{dataset}_{label}.jsonl"
            image_dir = output_dir / "sr_images" if save_images else None
            if image_dir is not None:
                image_dir.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8", buffering=1) as handle:
                for key, hr_path, lr_path in pairs:
                    hr, lr = read_pair(hr_path, lr_path, scale, allow_modcrop=True)
                    tensor = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
                    raw_output = model(tensor)
                    if not bool(torch.isfinite(raw_output).all()):
                        raise FloatingPointError(f"Non-finite evaluation output for {dataset}/{key}/{label}")
                    output = raw_output.clamp(0, 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
                    if output.shape != hr.shape:
                        raise AssertionError(f"Evaluation output mismatch for {dataset}/{key}/{label}")
                    record = {
                        "image_id": key,
                        "psnr_y_db": psnr_y(output, hr, scale),
                        "ssim_y": ssim_y(output, hr, scale),
                    }
                    records.append(record)
                    handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    if image_dir is not None and not cv2.imwrite(str(image_dir / f"{key}.png"), cv2.cvtColor(output, cv2.COLOR_RGB2BGR)):
                        raise OSError(f"Failed to save SR image: {image_dir / f'{key}.png'}")
                    del tensor, raw_output, output
            os.replace(temporary, final)
            summaries[dataset] = {
                "count": len(records),
                "mean_psnr_y_db": statistics.mean(float(record["psnr_y_db"]) for record in records),
                "mean_ssim_y": statistics.mean(float(record["ssim_y"]) for record in records),
                "per_image_path": str(final),
                "per_image_sha256": sha256_file(final),
                "per_image": records,
            }
    model = model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summaries


def seed_statistics(values: Sequence[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    if len(values) == 1:
        return {"mean": mean, "sample_std": 0.0, "ci95_half_width_t": 0.0}
    sample_std = statistics.stdev(values)
    critical = 4.302652729911275 if len(values) == 3 else 1.96
    return {"mean": mean, "sample_std": sample_std, "ci95_half_width_t": critical * sample_std / math.sqrt(len(values))}


def validate_completed_evaluation(
    path: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    source_manifest_hash: str,
    final_hashes: Mapping[str, str],
) -> dict[str, Any]:
    report = json_read(path)
    expected_scales = {str(value) for value in protocol["training"]["scales"]}
    expected_datasets = {"DIV2K_valid", *BENCHMARK_COUNTS}
    expected_labels = {"teacher", "student_init", *{f"student_seed{seed}" for seed in protocol["training"]["seeds"]}}
    if (
        report.get("goal_id") != GOAL_ID
        or report.get("status") != "complete"
        or report.get("protocol_sha256") != protocol_hash
        or report.get("source_manifest_sha256") != source_manifest_hash
        or report.get("final_checkpoint_sha256") != dict(final_hashes)
        or set(report.get("scales", {})) != expected_scales
    ):
        raise RuntimeError(f"Completed evaluation header is inconsistent: {path}")
    output_dir = path.parent
    for scale in sorted(expected_scales):
        scale_report = report["scales"][scale]
        if set(scale_report.get("models", {})) != expected_datasets or set(scale_report.get("across_seeds", {})) != expected_datasets:
            raise RuntimeError(f"Completed x{scale} evaluation dataset matrix is incomplete")
        for dataset in expected_datasets:
            models = scale_report["models"][dataset]
            if set(models) != expected_labels:
                raise RuntimeError(f"Completed x{scale}/{dataset} model matrix is incomplete")
            expected_count = 100 if dataset == "DIV2K_valid" else BENCHMARK_COUNTS[dataset]
            for label, summary in models.items():
                per_image = output_dir / f"x{scale}_{dataset}_{label}.jsonl"
                if not per_image.is_file() or summary.get("per_image_sha256") != sha256_file(per_image):
                    raise RuntimeError(f"Missing or changed per-image evidence: {per_image}")
                records = []
                with per_image.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            records.append(json.loads(line))
                if len(records) != expected_count or len({record["image_id"] for record in records}) != expected_count:
                    raise RuntimeError(f"Invalid per-image count or IDs: {per_image}")
                mean_psnr = statistics.mean(float(record["psnr_y_db"]) for record in records)
                mean_ssim = statistics.mean(float(record["ssim_y"]) for record in records)
                if (
                    int(summary.get("count", -1)) != expected_count
                    or not math.isclose(float(summary["mean_psnr_y_db"]), mean_psnr, rel_tol=0.0, abs_tol=1e-12)
                    or not math.isclose(float(summary["mean_ssim_y"]), mean_ssim, rel_tol=0.0, abs_tol=1e-12)
                ):
                    raise RuntimeError(f"Per-image aggregates differ from summary: {per_image}")
            teacher_psnr = float(models["teacher"]["mean_psnr_y_db"])
            initial_psnr = float(models["student_init"]["mean_psnr_y_db"])
            seed_psnr = [float(models[f"student_seed{seed}"]["mean_psnr_y_db"]) for seed in protocol["training"]["seeds"]]
            seed_ssim = [float(models[f"student_seed{seed}"]["mean_ssim_y"]) for seed in protocol["training"]["seeds"]]
            expected_across = {
                "student_psnr_y_db": seed_statistics(seed_psnr),
                "student_ssim_y": seed_statistics(seed_ssim),
                "psnr_gap_to_teacher_db": seed_statistics([teacher_psnr - value for value in seed_psnr]),
                "psnr_recovery_gain_over_init_db": seed_statistics([value - initial_psnr for value in seed_psnr]),
            }
            if scale_report["across_seeds"][dataset] != expected_across:
                raise RuntimeError(f"Across-seed statistics differ from frozen recomputation: x{scale}/{dataset}")
    return report


def command_evaluate(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, source_freeze = load_protocol(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    data_root = Path(args.data_root).resolve()
    run_root = Path(args.run_root).resolve()
    output_dir = run_root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "evaluation_report.json"
    environment = require_cloud_gpu(protocol)
    final_hashes: dict[str, str] = {}
    for scale in protocol["training"]["scales"]:
        for seed in protocol["training"]["seeds"]:
            run_dir = run_root / "training" / f"x{scale}" / f"seed{seed}"
            checkpoint = run_dir / "student_final.pt"
            report = completed_run_report(
                run_dir / "train_report.json",
                checkpoint,
                protocol_hash,
                source_freeze["manifest_sha256"],
                int(protocol["training"]["steps"]),
                int(scale),
                int(seed),
            )
            if report is None:
                raise RuntimeError(f"Missing completed x{scale} seed{seed} run")
            final_hashes[f"x{scale}_seed{seed}"] = sha256_file(checkpoint)
    if output.is_file():
        validate_completed_evaluation(
            output,
            protocol,
            protocol_hash,
            source_freeze["manifest_sha256"],
            final_hashes,
        )
        print(json.dumps({"status": "already_complete", "output": str(output)}, sort_keys=True))
        return

    report: dict[str, Any] = {
        "goal_id": GOAL_ID,
        "status": "running",
        "started_at_utc": utc_now(),
        "protocol_sha256": protocol_hash,
        "source_manifest_sha256": source_freeze["manifest_sha256"],
        "environment": environment,
        "final_checkpoint_sha256": final_hashes,
        "selection_policy": "No final benchmark selects a checkpoint; every frozen final-step seed is reported.",
        "inference": protocol["evaluation"],
        "scales": {},
    }
    json_write(output_dir / "evaluation_in_progress.json", report)
    try:
        device = torch.device("cuda:0")
        for scale_value in protocol["training"]["scales"]:
            scale = int(scale_value)
            teacher_state, teacher_checkpoint = validate_checkpoint(protocol, checkpoint_dir, scale)
            datasets = evaluation_pairs(protocol, data_root, scale)
            models: list[tuple[str, EDSR]] = []
            teacher = EDSR(scale=scale, n_resblocks=int(protocol["model"]["teacher_depth"]))
            strict_load(teacher, teacher_state)
            models.append(("teacher", teacher))
            initial, _ = transplant_state(
                teacher_state,
                scale,
                int(protocol["model"]["teacher_depth"]),
                int(protocol["model"]["student_depth"]),
                source_indices(protocol),
            )
            models.append(("student_init", initial))
            for seed_value in protocol["training"]["seeds"]:
                seed = int(seed_value)
                state = extract_tensor_state(torch_load_weights(run_root / "training" / f"x{scale}" / f"seed{seed}" / "student_final.pt"))
                student = EDSR(scale=scale, n_resblocks=int(protocol["model"]["student_depth"]))
                strict_load(student, state)
                models.append((f"student_seed{seed}", student))
            model_results: dict[str, Any] = {}
            for label, model in models:
                model_results[label] = evaluate_model(model, datasets, scale, device, output_dir, label)
            scale_summary: dict[str, Any] = {
                "teacher_checkpoint": teacher_checkpoint,
                "models": {},
                "across_seeds": {},
            }
            for dataset in datasets:
                teacher_result = model_results["teacher"][dataset]
                init_result = model_results["student_init"][dataset]
                scale_summary["models"].setdefault(dataset, {})
                for label in model_results:
                    result = model_results[label][dataset]
                    compact = {key: value for key, value in result.items() if key != "per_image"}
                    if label != "teacher":
                        compact["psnr_gap_to_teacher_db"] = teacher_result["mean_psnr_y_db"] - result["mean_psnr_y_db"]
                        compact["ssim_gap_to_teacher"] = teacher_result["mean_ssim_y"] - result["mean_ssim_y"]
                    if label.startswith("student_seed"):
                        compact["psnr_recovery_gain_over_init_db"] = result["mean_psnr_y_db"] - init_result["mean_psnr_y_db"]
                    scale_summary["models"][dataset][label] = compact
                seed_psnr = [model_results[f"student_seed{seed}"][dataset]["mean_psnr_y_db"] for seed in protocol["training"]["seeds"]]
                seed_ssim = [model_results[f"student_seed{seed}"][dataset]["mean_ssim_y"] for seed in protocol["training"]["seeds"]]
                seed_gap = [teacher_result["mean_psnr_y_db"] - value for value in seed_psnr]
                scale_summary["across_seeds"][dataset] = {
                    "student_psnr_y_db": seed_statistics(seed_psnr),
                    "student_ssim_y": seed_statistics(seed_ssim),
                    "psnr_gap_to_teacher_db": seed_statistics(seed_gap),
                    "psnr_recovery_gain_over_init_db": seed_statistics([value - init_result["mean_psnr_y_db"] for value in seed_psnr]),
                }
            report["scales"][str(scale)] = scale_summary
            report["last_completed_scale"] = scale
            json_write(output_dir / "evaluation_in_progress.json", report)
            models.clear()
            del model_results, teacher_state
            gc.collect()
        report["status"] = "complete"
        report["completed_at_utc"] = utc_now()
        json_write(output, report)
        validate_completed_evaluation(
            output,
            protocol,
            protocol_hash,
            source_freeze["manifest_sha256"],
            final_hashes,
        )
        (output_dir / "evaluation_in_progress.json").unlink(missing_ok=True)
        print(json.dumps({"status": "complete", "output": str(output)}, sort_keys=True))
    except Exception as error:
        report["status"] = "failed"
        report["failed_at_utc"] = utc_now()
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        json_write(output_dir / "evaluation_failure.json", report)
        raise


def infer_edsr_architecture(state: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    head = state.get("head.0.weight")
    if head is None or head.ndim != 4 or int(head.shape[1]) != 3:
        raise RuntimeError("Checkpoint is not a canonical EDSR tensor state")
    block_indices = sorted(
        int(match.group(1))
        for key in state
        if (match := re.fullmatch(r"body\.(\d+)\.body\.0\.weight", key))
    )
    if not block_indices or block_indices != list(range(block_indices[-1] + 1)):
        raise RuntimeError("Cannot infer a contiguous physical EDSR depth")
    return len(block_indices), int(head.shape[0])


def standalone_test_pairs(args: argparse.Namespace, scale: int) -> tuple[list[tuple[str, Path, Path]], bool]:
    if bool(args.hr_dir) != bool(args.lr_dir):
        raise ValueError("--hr-dir and --lr-dir must be supplied together")
    if args.hr_dir:
        return pair_directories(Path(args.hr_dir).resolve(), Path(args.lr_dir).resolve(), scale), False
    data_root = Path(args.data_root).expanduser().resolve()
    if args.dataset == "DIV2K_valid":
        hr_dir = data_root / "DIV2K_valid_HR"
        lr_dir = data_root / "DIV2K_valid_LR_bicubic" / f"X{scale}"
    elif args.dataset in BENCHMARK_COUNTS:
        hr_dir = data_root / "SRBenchmarks" / args.dataset / "HR"
        lr_dir = data_root / "SRBenchmarks" / args.dataset / "LR_bicubic" / f"X{scale}"
    else:
        raise ValueError("A custom dataset requires explicit --hr-dir and --lr-dir")
    pairs = pair_directories(hr_dir, lr_dir, scale)
    expected = 100 if args.dataset == "DIV2K_valid" else BENCHMARK_COUNTS[args.dataset]
    if len(pairs) != expected:
        raise RuntimeError(f"{args.dataset} x{scale} count is {len(pairs)}, expected {expected}")
    return pairs, True


def command_test_checkpoint(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.dataset):
        raise ValueError("Dataset label may contain only letters, digits, dot, underscore, and hyphen")
    if int(args.max_images) < 0:
        raise ValueError("--max-images must be non-negative")
    scale = int(args.scale)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    repository_root = Path(__file__).resolve().parents[4]
    experiments_root = (repository_root / "experiments").resolve()
    if experiments_root != repository_root / "experiments":
        raise RuntimeError("Repository experiments/ must not be a symlink")
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if experiments_root not in experiment_dir.parents:
        raise ValueError(f"--experiment-dir must be a child of {experiments_root}")
    output_dir = experiment_dir / "test" / f"X{scale}" / args.dataset
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Test output already exists; choose a new --experiment-dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pairs, canonical_paths = standalone_test_pairs(args, scale)
        if args.max_images:
            pairs = pairs[: int(args.max_images)]
        if not pairs:
            raise RuntimeError("No image pairs selected")
        state = extract_tensor_state(torch_load_weights(checkpoint))
        if not all(bool(value.isfinite().all()) for value in state.values()):
            raise FloatingPointError("Checkpoint contains a non-finite tensor")
        depth, n_feats = infer_edsr_architecture(state)
        model = EDSR(scale=scale, n_resblocks=depth, n_feats=n_feats)
        strict_load(model, state)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(args.device)
        result = evaluate_model(
            model,
            {args.dataset: pairs},
            scale,
            device,
            output_dir,
            "checkpoint",
            save_images=bool(args.save_images),
        )[args.dataset]
        records = result.pop("per_image")
        csv_path = output_dir / "per_image_metrics.csv"
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_id", "psnr_y_db", "ssim_y"))
            writer.writeheader()
            writer.writerows(records)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_csv, csv_path)
        log_lines = [
            f"checkpoint: {checkpoint}",
            f"checkpoint_sha256: {sha256_file(checkpoint)}",
            f"model: canonical EDSR, depth={depth}, n_feats={n_feats}, scale=x{scale}",
            f"dataset: {args.dataset}, images={len(records)}",
            "",
        ]
        log_lines.extend(
            f"{record['image_id']}: PSNR-Y={float(record['psnr_y_db']):.6f} dB, SSIM-Y={float(record['ssim_y']):.6f}"
            for record in records
        )
        log_lines.extend(
            (
                "",
                f"AVERAGE PSNR-Y: {float(result['mean_psnr_y_db']):.6f} dB",
                f"AVERAGE SSIM-Y: {float(result['mean_ssim_y']):.6f}",
            )
        )
        log_path = output_dir / "test.log"
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        pair_digest = hashlib.sha256()
        for key, hr_path, lr_path in pairs:
            for value in (key, str(hr_path), str(lr_path), str(hr_path.stat().st_size), str(lr_path.stat().st_size)):
                pair_digest.update(value.encode("utf-8"))
                pair_digest.update(b"\0")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "completed_at_utc": utc_now(),
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "tensor_state_sha256": tensor_state_sha256(state),
            },
            "model": {"name": "canonical EDSR", "depth": depth, "n_feats": n_feats, "scale": scale},
            "dataset": {
                "name": args.dataset,
                "count": len(pairs),
                "canonical_repository_paths": canonical_paths,
                "max_images": int(args.max_images),
                "ordered_pair_path_size_sha256": pair_digest.hexdigest(),
            },
            "metrics": {
                "protocol": {
                    "output_quantization": "clamp to [0,255], round, then uint8",
                    "y_conversion": "Y=(65.738R+129.057G+25.064B)/256+16",
                    "border_shave_pixels": scale,
                    "ssim": "11x11 Gaussian window, sigma=1.5, K1=0.01, K2=0.03",
                },
                "mean_psnr_y_db": result["mean_psnr_y_db"],
                "mean_ssim_y": result["mean_ssim_y"],
            },
            "runtime": {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "torch": torch.__version__,
            },
            "outputs": {
                "directory": str(output_dir),
                "test_log": str(log_path),
                "per_image_csv": str(csv_path),
                "per_image_jsonl": result["per_image_path"],
                "per_image_jsonl_sha256": result["per_image_sha256"],
                "sr_images_saved": bool(args.save_images),
            },
        }
        json_write(output_dir / "summary.json", summary)
        print("\n".join(log_lines), flush=True)
        print(json.dumps({"status": "complete", "summary": str(output_dir / "summary.json")}, sort_keys=True), flush=True)
    except Exception as error:
        json_write(
            output_dir / "test_failure.json",
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "checkpoint": str(checkpoint),
                "scale": scale,
                "dataset": args.dataset,
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def bundle_files(
    run_root: Path,
    include_resume: bool,
    protocol: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> list[Path]:
    files: set[Path] = set()

    def include(relative: str) -> None:
        path = run_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required bundle file is missing: {path}")
        files.add(path)

    if not include_resume:
        for relative in (
            "launcher.log",
            "launcher_status.txt",
            "checkpoint_acquisition_history.tsv",
            "cloud_preflight.json",
            "source_snapshot/source_manifest.json",
        ):
            include(relative)
        optional_cli = run_root / "featurize_cli_help.txt"
        if optional_cli.is_file():
            files.add(optional_cli)
        snapshot_manifest = json_read(run_root / "source_snapshot/source_manifest.json")
        if snapshot_manifest != source_manifest:
            raise RuntimeError("Bundled source-manifest snapshot differs from the frozen manifest")
        for relative in source_manifest["files"]:
            include(str(Path("source_snapshot") / relative))
        for scale in protocol["training"]["scales"]:
            for seed in protocol["training"]["seeds"]:
                prefix = Path("training") / f"x{scale}" / f"seed{seed}"
                for name in ("run_config.json", "train_trace.jsonl", "train_report.json", "student_final.pt"):
                    include(str(prefix / name))
        include("evaluation/evaluation_report.json")
        labels = ["teacher", "student_init", *[f"student_seed{seed}" for seed in protocol["training"]["seeds"]]]
        for scale in protocol["training"]["scales"]:
            for dataset in ("DIV2K_valid", *BENCHMARK_COUNTS):
                for label in labels:
                    include(f"evaluation/x{scale}_{dataset}_{label}.jsonl")
        return sorted(files, key=lambda item: str(item.relative_to(run_root)))

    for name in (
        "launcher.log",
        "launcher_status.txt",
        "checkpoint_acquisition_history.tsv",
        "cloud_preflight.json",
        "featurize_cli_help.txt",
    ):
        path = run_root / name
        if path.is_file():
            files.add(path)
    source_snapshot = run_root / "source_snapshot"
    if source_snapshot.is_dir():
        files.update(path for path in source_snapshot.rglob("*") if path.is_file())
    training = run_root / "training"
    if training.is_dir():
        allowed_training = {"run_config.json", "train_trace.jsonl", "train_report.json", "student_final.pt", "failure.json", "resume.pt"}
        for path in training.rglob("*"):
            if path.is_file() and (path.name in allowed_training or path.name.startswith("superseded_failure_")):
                files.add(path)
    evaluation = run_root / "evaluation"
    if evaluation.is_dir():
        for path in evaluation.iterdir():
            if path.is_file() and (
                path.name in {"evaluation_report.json", "evaluation_failure.json", "evaluation_in_progress.json"}
                or path.suffix == ".jsonl"
            ):
                files.add(path)
    return sorted(files, key=lambda item: str(item.relative_to(run_root)))


def command_bundle(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, source_freeze = load_protocol(protocol_path)
    run_root = Path(args.run_root).resolve()
    archive = Path(args.archive).resolve()
    allow_partial = bool(args.allow_partial)
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    if not allow_partial:
        protocol_hash = sha256_file(protocol_path)
        final_hashes: dict[str, str] = {}
        for scale in protocol["training"]["scales"]:
            for seed in protocol["training"]["seeds"]:
                run_dir = run_root / "training" / f"x{scale}" / f"seed{seed}"
                checkpoint = run_dir / "student_final.pt"
                completed_run_report(
                    run_dir / "train_report.json",
                    checkpoint,
                    protocol_hash,
                    source_freeze["manifest_sha256"],
                    int(protocol["training"]["steps"]),
                    int(scale),
                    int(seed),
                )
                final_hashes[f"x{scale}_seed{seed}"] = sha256_file(checkpoint)
        evaluation = run_root / "evaluation" / "evaluation_report.json"
        if not evaluation.is_file():
            raise RuntimeError("A complete bundle requires the complete evaluation report")
        validate_completed_evaluation(
            evaluation,
            protocol,
            protocol_hash,
            source_freeze["manifest_sha256"],
            final_hashes,
        )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() or archive.with_suffix(archive.suffix + ".part").exists():
        raise FileExistsError(archive)
    files = bundle_files(
        run_root,
        include_resume=allow_partial,
        protocol=protocol,
        source_manifest=json_read(Path(source_freeze["manifest_path"])),
    )
    manifest = {
        "goal_id": GOAL_ID,
        "created_at_utc": utc_now(),
        "partial": allow_partial,
        "protocol_sha256": sha256_file(protocol_path),
        "source_manifest_sha256": source_freeze["manifest_sha256"],
        "files": {
            str(path.relative_to(run_root)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        },
    }
    manifest_path = run_root / "bundle_manifest.json"
    json_write(manifest_path, manifest)
    temporary = archive.with_suffix(archive.suffix + ".part")
    archive_files = [*files, manifest_path]
    with tarfile.open(temporary, "w") as handle:
        for path in archive_files:
            handle.add(path, arcname=str(Path(GOAL_ID) / path.relative_to(run_root)), recursive=False)
    with tarfile.open(temporary, "r") as handle:
        members = [member for member in handle.getmembers() if member.isfile()]
        if len(members) != len(archive_files):
            raise RuntimeError(f"Bundle verification count mismatch: archive={len(members)}, source={len(archive_files)}")
        observed_files: set[str] = set()
        for member in members:
            relative = str(Path(member.name).relative_to(GOAL_ID))
            extracted = handle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Cannot read archive member {member.name}")
            digest = hashlib.sha256()
            observed_bytes = 0
            while chunk := extracted.read(1024 * 1024):
                digest.update(chunk)
                observed_bytes += len(chunk)
            if relative == "bundle_manifest.json":
                if digest.hexdigest() != sha256_file(manifest_path):
                    raise RuntimeError("Bundled manifest hash mismatch")
                continue
            expected = manifest["files"].get(relative)
            if expected is None or observed_bytes != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise RuntimeError(f"Bundled member verification failed: {relative}")
            observed_files.add(relative)
        if observed_files != set(manifest["files"]):
            raise RuntimeError("Bundled file set differs from internal manifest")
    os.replace(temporary, archive)
    receipt = {
        "goal_id": GOAL_ID,
        "partial": allow_partial,
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "member_count": len(archive_files),
        "manifest_sha256": sha256_file(manifest_path),
        "verified_tar_read": True,
        "created_at_utc": utc_now(),
    }
    json_write(run_root / "bundle_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


def command_smoke(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, source_freeze = load_protocol(protocol_path)
    scale = int(args.scale)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the smoke test")
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    state, checkpoint = validate_checkpoint(protocol, checkpoint_dir, scale)
    pairs = pair_directories(Path(args.hr_dir).resolve(), Path(args.lr_dir).resolve(), scale)
    sample_geometry = validate_pair_samples(pairs, scale, int(args.patch), allow_modcrop=False)
    configure_reproducibility(4242)
    device = torch.device("cuda:0")
    teacher = EDSR(scale=scale, n_resblocks=int(protocol["model"]["teacher_depth"]))
    strict_load(teacher, state)
    student, _ = transplant_state(
        state,
        scale,
        int(protocol["model"]["teacher_depth"]),
        int(protocol["model"]["student_depth"]),
        source_indices(protocol),
    )
    audit = static_audit(student)
    if not audit["pass"]:
        raise AssertionError(f"Smoke Student static audit failed: {audit}")
    teacher = teacher.eval().float().to(device)
    student = student.train().float().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=float(protocol["training"]["learning_rate"]),
    )
    rng = random.Random(4242)
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(int(args.steps)):
        lr_cpu, hr_cpu, _ = sample_batch(pairs, scale, int(args.batch), int(args.patch), rng)
        lr, hr = lr_cpu.to(device), hr_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_output = teacher(lr)
        student_output = student(lr)
        loss = 0.5 * torch.nn.functional.l1_loss(student_output, teacher_output) + 0.5 * torch.nn.functional.l1_loss(student_output, hr)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Smoke loss is non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    final_state = OrderedDict((key, value.detach().cpu()) for key, value in student.state_dict().items())
    with tempfile.TemporaryDirectory(prefix="edsr_formal_smoke_") as directory:
        path = Path(directory) / "student.pt"
        torch_save_atomic(final_state, path)
        clone = EDSR(scale=scale, n_resblocks=int(protocol["model"]["student_depth"]))
        strict_load(clone, extract_tensor_state(torch_load_weights(path)))
        with torch.inference_mode():
            output = clone(torch.rand(1, 3, 8, 11).mul(255))
    report = {
        "goal_id": GOAL_ID,
        "status": "pass",
        "created_at_utc": utc_now(),
        "protocol_sha256": sha256_file(protocol_path),
        "source_manifest_sha256": source_freeze["manifest_sha256"],
        "device": torch.cuda.get_device_name(0),
        "scale": scale,
        "steps": int(args.steps),
        "batch": int(args.batch),
        "patch": int(args.patch),
        "losses": losses,
        "checkpoint": checkpoint,
        "static_student_audit": audit,
        "pair_count": len(pairs),
        "sample_geometry": sample_geometry,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "final_tensor_state_sha256": tensor_state_sha256(final_state),
        "strict_temporary_round_trip": True,
        "finite_output_shape": list(output.shape),
    }
    json_write(Path(args.output).resolve(), report)
    print(json.dumps({"pass": True, "output": str(Path(args.output).resolve()), "losses": losses}, sort_keys=True))


def command_self_test(_: argparse.Namespace) -> None:
    torch.manual_seed(7)
    teacher = EDSR(scale=2, n_resblocks=4, n_feats=8)
    student, state = transplant_state(teacher.state_dict(), 2, 4, 3, [0, 2, 3], n_feats=8)
    if not static_audit(student)["pass"]:
        raise AssertionError("Tiny Student static audit failed")
    sample = torch.rand(1, 3, 8, 9).mul(255)
    target = torch.rand(1, 3, 16, 18).mul(255)
    optimizer = torch.optim.Adam([parameter for parameter in student.parameters() if parameter.requires_grad], lr=1e-5)
    output = student(sample)
    loss = torch.nn.functional.l1_loss(output, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if list(output.shape) != [1, 3, 16, 18] or not bool(torch.isfinite(loss)):
        raise AssertionError("Tiny training step failed")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "student.pt"
        torch_save_atomic(OrderedDict((key, value.detach().cpu()) for key, value in student.state_dict().items()), checkpoint)
        clone = EDSR(scale=2, n_resblocks=3, n_feats=8)
        strict_load(clone, extract_tensor_state(torch_load_weights(checkpoint)))
    if canonical_pair_key(Path("0001x4_s001.png"), 4) != "0001_s001":
        raise AssertionError("Pair normalization failed")
    image = np.full((24, 24, 3), 127, dtype=np.uint8)
    if not math.isinf(psnr_y(image, image, 2)) or not math.isclose(ssim_y(image, image, 2), 1.0, abs_tol=1e-12):
        raise AssertionError("Metric identity test failed")
    print(json.dumps({"pass": True, "tiny_state_tensors": len(state), "loss": float(loss.detach())}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    commands.add_parser("self-test").set_defaults(function=command_self_test)

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--protocol", required=True)
    smoke.add_argument("--checkpoint-dir", required=True)
    smoke.add_argument("--hr-dir", required=True)
    smoke.add_argument("--lr-dir", required=True)
    smoke.add_argument("--scale", type=int, default=4)
    smoke.add_argument("--steps", type=int, default=2)
    smoke.add_argument("--batch", type=int, default=1)
    smoke.add_argument("--patch", type=int, default=16)
    smoke.add_argument("--output", required=True)
    smoke.set_defaults(function=command_smoke)

    check = commands.add_parser("check")
    check.add_argument("--protocol", required=True)
    check.add_argument("--repo-root", required=True)
    check.add_argument("--checkpoint-dir", required=True)
    check.add_argument("--data-root", required=True)
    check.add_argument("--run-root", required=True)
    check.add_argument("--output", required=True)
    check.set_defaults(function=command_check)

    train = commands.add_parser("train")
    train.add_argument("--protocol", required=True)
    train.add_argument("--checkpoint-dir", required=True)
    train.add_argument("--data-root", required=True)
    train.add_argument("--run-root", required=True)
    train.add_argument("--scale", required=True, type=int)
    train.add_argument("--seed", required=True, type=int)
    train.set_defaults(function=command_train)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--protocol", required=True)
    evaluate.add_argument("--checkpoint-dir", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--run-root", required=True)
    evaluate.set_defaults(function=command_evaluate)

    test_checkpoint = commands.add_parser("test-checkpoint")
    test_checkpoint.add_argument("--checkpoint", required=True)
    test_checkpoint.add_argument("--scale", required=True, type=int, choices=(2, 3, 4))
    test_checkpoint.add_argument("--dataset", required=True)
    test_checkpoint.add_argument("--data-root", default="/home/featurize/data")
    test_checkpoint.add_argument("--hr-dir")
    test_checkpoint.add_argument("--lr-dir")
    test_checkpoint.add_argument("--experiment-dir", required=True)
    test_checkpoint.add_argument("--device", default="cuda:0")
    test_checkpoint.add_argument("--max-images", type=int, default=0)
    test_checkpoint.add_argument("--save-images", action="store_true")
    test_checkpoint.set_defaults(function=command_test_checkpoint)

    bundle = commands.add_parser("bundle")
    bundle.add_argument("--protocol", required=True)
    bundle.add_argument("--run-root", required=True)
    bundle.add_argument("--archive", required=True)
    bundle.add_argument("--allow-partial", action="store_true")
    bundle.set_defaults(function=command_bundle)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
