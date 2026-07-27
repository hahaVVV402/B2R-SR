#!/usr/bin/env python3
"""B2R-SR v2 depth-sweep diagnostic (SG-0/1/2/3/5 free-of-training probes).

One instrumented pass per image produces:
  * prefix depth sweep d=0..G  (nested residual-group truncation, exact d=G)
  * leave-one-out group skipping (block-influence estimate, SG-2)
  * CA gate statistics for every CALayer (SG-5 kill-check for plan C)
  * per-depth wall-clock latency with CUDA events (SG-1 denominator data)
  * dense fp16 / channels_last probes (SG-0 honest-baseline screening)
  * offline oracle ladder: per-image minimal depth under eps constraints (SG-3 prep)

Everything is exported to one timestamped directory + tar.gz for download.
No training, no controller, backbone weights untouched.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

import options.options as option  # noqa: E402
import utils.util as util  # noqa: E402
from data import create_dataloader, create_dataset  # noqa: E402
from data.util import bgr2ycbcr  # noqa: E402
from models.archs.RCAN_arch import RCAN, CALayer  # noqa: E402


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
        raise FileNotFoundError("未找到 checkpoint；用 --checkpoint 指定 120000_G.pth 或 export tar。")
    if source.suffix == ".pth":
        return source, temporary
    if source.suffix != ".tar":
        raise ValueError("--checkpoint 仅支持 .pth 或 export .tar")
    temporary = tempfile.TemporaryDirectory(prefix="b2rsr-depthsweep-")
    with tarfile.open(source, "r") as archive:
        archive.extractall(temporary.name)
    matches = list(Path(temporary.name).rglob("*_G.pth"))
    if len(matches) != 1:
        temporary.cleanup()
        raise RuntimeError("导出包中应恰好包含一个 *_G.pth")
    return matches[0], temporary


def build_backbone(parsed_opt, checkpoint, device):
    net_opt = parsed_opt["network_G"]
    rcan = RCAN(
        n_resgroups=int(net_opt["n_resgroups"]),
        n_resblocks=int(net_opt["n_resblocks"]),
        n_feats=int(net_opt["n_feats"]),
        res_scale=int(net_opt["res_scale"]),
        n_colors=int(net_opt["n_colors"]),
        rgb_range=int(net_opt["rgb_range"]),
        scale=int(net_opt["scale"]),
        reduction=int(net_opt["reduction"]),
    )
    state = torch.load(str(checkpoint), map_location="cpu")
    backbone_state = {}
    for key, value in state.items():
        clean = key[7:] if key.startswith("module.") else key
        if clean.startswith("backbone."):
            backbone_state[clean[len("backbone."):]] = value
    if not backbone_state:
        backbone_state = {k: v for k, v in state.items()}  # 纯 RCAN checkpoint 兼容
    rcan.load_state_dict(backbone_state, strict=True)
    rcan.eval()
    for p in rcan.parameters():
        p.requires_grad = False
    return rcan.to(device)


def load_samples(parsed_opt, dataset_name, max_images, device):
    selected = None
    for dataset_opt in parsed_opt["datasets"].values():
        if dataset_opt["name"].lower() == dataset_name.lower():
            selected = dataset_opt
            break
    if selected is None:
        available = ", ".join(v["name"] for v in parsed_opt["datasets"].values())
        raise ValueError("未知数据集 {}；可选 {}".format(dataset_name, available))
    dataset = create_dataset(selected)
    loader = create_dataloader(dataset, selected)
    samples = []
    for data in loader:
        samples.append({
            "name": Path(data["GT_path"][0]).stem,
            "lq": data["LQ"].to(device),
            "gt": data["GT"].to(device),
        })
        if max_images and len(samples) >= max_images:
            break
    return samples


def psnr_pair(tensor, gt, scale, with_ssim=False):
    sr_img = util.tensor2img(tensor.detach()[0], out_type=np.uint8, min_max=(0, 255))
    gt_img = util.tensor2img(gt.detach()[0], out_type=np.uint8, min_max=(0, 255))
    sr_img, gt_img = util.crop_border([sr_img, gt_img], scale)
    psnr_rgb = float(util.calculate_psnr(sr_img, gt_img))
    sr_y = bgr2ycbcr(sr_img / 255.0, only_y=True) * 255.0
    gt_y = bgr2ycbcr(gt_img / 255.0, only_y=True) * 255.0
    psnr_y = float(util.calculate_psnr(sr_y, gt_y))
    ssim_y = float(util.calculate_ssim(sr_y, gt_y)) if with_ssim else None
    return psnr_rgb, psnr_y, ssim_y


# ---------------------------------------------------------------------------
# Core executions
# ---------------------------------------------------------------------------

def groups_and_tail(rcan):
    groups = list(rcan.body[:-1])
    final_conv = rcan.body[-1]
    return groups, final_conv


def instrumented_depth_outputs(rcan, lq):
    """One pass through all groups; reconstruct every prefix depth d=0..G.

    d=G reuses the exact dense call order (same modules, same sequence), so the
    last entry is bitwise identical to rcan(lq).
    """
    groups, final_conv = groups_and_tail(rcan)
    x = rcan.sub_mean(lq)
    x = rcan.head(x)
    feats = [x]
    f = x
    for g in groups:
        f = g(f)
        feats.append(f)
    outputs = []
    for fd in feats:
        res = final_conv(fd)
        res = res + x
        out = rcan.tail(res)
        outputs.append(rcan.add_mean(out))
    return outputs  # outputs[d] == depth-d result; outputs[G] == dense


def truncated_forward(rcan, lq, depth):
    groups, final_conv = groups_and_tail(rcan)
    x = rcan.sub_mean(lq)
    x = rcan.head(x)
    f = x
    for g in groups[:depth]:
        f = g(f)
    res = final_conv(f)
    res = res + x
    return rcan.add_mean(rcan.tail(res))


def subset_forward(rcan, lq, skip_set):
    groups, final_conv = groups_and_tail(rcan)
    x = rcan.sub_mean(lq)
    x = rcan.head(x)
    f = x
    for idx, g in enumerate(groups):
        if idx not in skip_set:
            f = g(f)
    res = final_conv(f)
    res = res + x
    return rcan.add_mean(rcan.tail(res))


@contextlib.contextmanager
def ca_gate_collector(rcan, storage):
    """Capture the sigmoid channel gates of every CALayer during one pass."""
    handles = []
    layer_names = []

    def make_hook(name):
        def hook(module, inputs, output):
            storage.setdefault(name, []).append(
                output.detach().flatten().float().cpu().numpy())
        return hook

    for name, module in rcan.named_modules():
        if isinstance(module, CALayer):
            layer_names.append(name)
            handles.append(module.conv_du.register_forward_hook(make_hook(name)))
    storage["__layer_names__"] = layer_names
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def timed(callable_, warmup, runs, device):
    with torch.no_grad():
        for _ in range(warmup):
            callable_()
        torch.cuda.synchronize(device)
        timings = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            callable_()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    return timings


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def oracle_ladder(per_image_rows, depth_latency_ms, num_groups, eps_list):
    """Offline oracle: per image, minimal depth whose PSNR-Y >= dense - eps."""
    ladders = []
    for eps in eps_list:
        chosen, satisfied = [], 0
        for row in per_image_rows:
            dense_y = row["psnr_y_by_depth"][num_groups]
            pick = num_groups
            for d in range(num_groups + 1):
                if row["psnr_y_by_depth"][d] >= dense_y - eps:
                    pick = d
                    break
            chosen.append(pick)
            if pick < num_groups:
                satisfied += 1
        latencies = [depth_latency_ms.get(str(d), depth_latency_ms.get(d)) for d in chosen]
        latencies = [l for l in latencies if l is not None]
        ladders.append({
            "eps_db": eps,
            "mean_depth": mean([float(c) for c in chosen]),
            "depth_histogram": {str(d): chosen.count(d) for d in sorted(set(chosen))},
            "images_below_dense_depth": satisfied,
            "oracle_mean_latency_ms": mean(latencies) if latencies else None,
        })
    return ladders


def summarize_gates(storage, thresholds=(0.05, 0.1, 0.2, 0.3)):
    layer_names = storage.pop("__layer_names__", [])
    if not layer_names:
        return {"note": "no CALayer captured"}, None
    per_layer = []
    for name in layer_names:
        stacked = np.stack(storage[name], axis=0)  # [images, C]
        per_layer.append(stacked)
    all_gates = np.stack([g.mean(axis=0) for g in per_layer], axis=0)  # [L, C] image-mean
    all_std = np.stack([g.std(axis=0) for g in per_layer], axis=0)
    total = all_gates.size
    summary = {
        "num_ca_layers": int(all_gates.shape[0]),
        "channels_per_layer": int(all_gates.shape[1]),
        "gate_mean_overall": float(all_gates.mean()),
        "gate_min": float(all_gates.min()),
        "gate_max": float(all_gates.max()),
        "cross_image_std_mean": float(all_std.mean()),
        "fraction_channels_below": {
            str(t): float((all_gates < t).sum() / total) for t in thresholds
        },
        "layers_with_any_channel_below_0.2": int(((all_gates < 0.2).any(axis=1)).sum()),
    }
    npz_payload = {
        "layer_names": np.array(layer_names),
        "gate_mean_per_layer_channel": all_gates,
        "gate_std_per_layer_channel": all_std,
    }
    return summary, npz_payload


def render_markdown(report):
    lines = [
        "# B2R-SR Depth-Sweep Diagnostic Report", "",
        "- checkpoint: `{}`".format(report["checkpoint"]),
        "- sha256: `{}`".format(report["checkpoint_sha256"]),
        "- device: {}".format(report["device"]),
        "- torch: {}".format(report["torch_version"]),
        "- datasets: {}".format(", ".join(
            "{} ({} images)".format(d["dataset"], d["images"]) for d in report["datasets"])),
        "",
        "## V2-G0 exactness (d=G vs dense)",
        "",
        "max |diff| = {:.3e}  → {}".format(
            report["equivalence"]["max_abs_diff"],
            report["equivalence"]["verdict"]),
        "",
    ]
    for ds in report["datasets"]:
        lines.extend([
            "## {} — depth sweep (mean over {} images)".format(ds["dataset"], ds["images"]), "",
            "| d | PSNR-Y | ΔY vs dense | PSNR-RGB | worst-image ΔY |",
            "|---:|---:|---:|---:|---:|",
        ])
        dense_y = ds["depth_summary"][-1]["psnr_y"]
        for row in ds["depth_summary"]:
            lines.append("| {} | {:.4f} | {:+.4f} | {:.4f} | {:+.4f} |".format(
                row["depth"], row["psnr_y"], row["psnr_y"] - dense_y,
                row["psnr_rgb"], row["worst_delta_y"]))
        lines.append("")
        if ds.get("loo_summary"):
            lines.extend([
                "### {} — leave-one-out group influence (skip exactly one RG)".format(ds["dataset"]), "",
                "| skipped RG | mean ΔPSNR-Y vs dense |", "|---:|---:|",
            ])
            for row in ds["loo_summary"]:
                lines.append("| {} | {:+.4f} |".format(row["skip_group"], row["delta_y"]))
            lines.append("")
        if ds.get("oracle"):
            lines.extend([
                "### {} — offline oracle ladder".format(ds["dataset"]), "",
                "| eps (dB) | mean depth | oracle mean latency (ms) | imgs < dense depth |",
                "|---:|---:|---:|---:|",
            ])
            for row in ds["oracle"]:
                lines.append("| {:.2f} | {:.2f} | {} | {}/{} |".format(
                    row["eps_db"], row["mean_depth"],
                    "{:.2f}".format(row["oracle_mean_latency_ms"])
                    if row["oracle_mean_latency_ms"] is not None else "n/a",
                    row["images_below_dense_depth"], ds["images"]))
            lines.append("")
    lat = report.get("latency", {})
    if lat:
        lines.extend([
            "## Latency per depth (median over {} images × {} runs, CUDA events)".format(
                lat["images"], lat["runs"]), "",
            "| d | median ms | p90 ms | vs dense |", "|---:|---:|---:|---:|",
        ])
        dense_ms = lat["per_depth"][str(lat["num_groups"])]["median_ms"]
        for d in range(lat["num_groups"] + 1):
            row = lat["per_depth"][str(d)]
            lines.append("| {} | {:.3f} | {:.3f} | {:.3f}× |".format(
                d, row["median_ms"], row["p90_ms"], dense_ms / row["median_ms"]))
        lines.append("")
    probes = report.get("dense_probes")
    if probes:
        lines.extend(["## Honest dense baseline probes (dense d=G only)", "",
                      "| config | median ms | PSNR-Y delta vs fp32 |", "|---|---:|---:|"])
        for row in probes:
            delta = ("{:+.4f}".format(row["psnr_y_delta"])
                     if row.get("psnr_y_delta") is not None else "n/a")
            lines.append("| {} | {:.3f} | {} |".format(row["config"], row["median_ms"], delta))
        lines.append("")
    ca = report.get("ca_gate_summary")
    if ca and "fraction_channels_below" in ca:
        lines.extend([
            "## CA gate statistics (plan C kill-check, SG-5)", "",
            "- CA layers: {} × {} channels".format(ca["num_ca_layers"], ca["channels_per_layer"]),
            "- overall gate mean: {:.4f} (min {:.4f}, max {:.4f})".format(
                ca["gate_mean_overall"], ca["gate_min"], ca["gate_max"]),
            "- cross-image std (input adaptivity): {:.4f}".format(ca["cross_image_std_mean"]),
            "- fraction of (layer,channel) with mean gate below threshold:",
        ])
        for t, frac in ca["fraction_channels_below"].items():
            lines.append("  - < {}: {:.2%}".format(t, frac))
        lines.append("")
    lines.extend([
        "## Notes", "",
        "- 本报告为冻结 checkpoint 的免训练诊断；oracle 仅为离线上界，不是可部署路由器。",
        "- latency 为固定深度截断路径的重复前向筛查，未含 controller 开销。",
        "- SG 判定（悬崖 vs 平滑、SG-1 门槛）请结合预注册标准离线判读。", "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="120000_G.pth 或 export tar；省略时自动查找")
    parser.add_argument("--config", default=str(ROOT / "codes/options/test/test_B2RSR_RCAN_X4.yml"))
    parser.add_argument("--data-root", default="/home/featurize/data")
    parser.add_argument("--datasets", default="Set5",
                        help="逗号分隔，如 Set5 或 BSD100,Urban100")
    parser.add_argument("--max-images", type=int, default=0, help="每数据集上限；0=全部")
    parser.add_argument("--ssim", action="store_true", help="额外计算 SSIM（较慢）")
    parser.add_argument("--loo", action="store_true", default=True,
                        help="leave-one-out 组影响分析（默认开）")
    parser.add_argument("--no-loo", dest="loo", action="store_false")
    parser.add_argument("--loo-images", type=int, default=20,
                        help="每数据集参与 LOO 的图数上限；0=全部（默认 20，省时）")
    parser.add_argument("--latency-images", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--eps", default="0.05,0.1,0.2,0.3",
                        help="oracle 质量约束阶梯（dB）")
    parser.add_argument("--dense-probes", action="store_true", default=True,
                        help="dense fp16/channels_last 探针（默认开）")
    parser.add_argument("--no-dense-probes", dest="dense_probes", action="store_false")
    parser.add_argument("--skip-data-prepare", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--archive")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("需要 CUDA GPU。")

    data_root = Path(args.data_root).expanduser().resolve()
    if not args.skip_data_prepare:
        subprocess.run([
            str(ROOT / "scripts/data/prepare_sr_benchmarks.sh"), str(data_root)
        ], check=True)

    checkpoint, temporary = resolve_checkpoint(args.checkpoint)
    try:
        parsed_opt = option.dict_to_nonedict(option.parse(args.config, is_train=False))
        default_root = "/home/featurize/data"
        for dataset_opt in parsed_opt["datasets"].values():
            for key in ("dataroot_GT", "dataroot_LQ"):
                dataset_opt[key] = dataset_opt[key].replace(default_root, str(data_root), 1)

        device = torch.device("cuda:0")
        scale = int(parsed_opt["scale"])
        rcan = build_backbone(parsed_opt, checkpoint, device)
        groups, _ = groups_and_tail(rcan)
        num_groups = len(groups)
        eps_list = [float(v) for v in args.eps.split(",")]
        dataset_names = [v.strip() for v in args.datasets.split(",") if v.strip()]

        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path(args.output_dir) if args.output_dir else (
            ROOT / "results/depth_sweep/B2RSR_RCAN_X4_120000" / stamp)
        output_dir.mkdir(parents=True, exist_ok=True)

        # -- V2-G0 exactness on the first image of the first dataset ---------
        first_samples = load_samples(parsed_opt, dataset_names[0], 1, device)
        with torch.no_grad():
            dense_ref = rcan(first_samples[0]["lq"])
            outs = instrumented_depth_outputs(rcan, first_samples[0]["lq"])
        max_diff = float((outs[num_groups] - dense_ref).abs().max().item())
        equivalence = {
            "max_abs_diff": max_diff,
            "verdict": "PASS" if max_diff == 0.0 else (
                "PASS(tol)" if max_diff < 1e-5 else "FAIL"),
        }
        print("V2-G0 exactness: max|diff|={:.3e} -> {}".format(
            max_diff, equivalence["verdict"]))
        if equivalence["verdict"] == "FAIL":
            print("d=G 与 dense 不一致，停止后续实验。")

        gate_storage = {}
        dataset_reports = []
        csv_rows = []

        # latency measured once, on first dataset's first images -------------
        latency_report = {}
        lat_samples = first_samples + load_samples(
            parsed_opt, dataset_names[0], args.latency_images, device)[1:]
        lat_samples = lat_samples[:args.latency_images]
        print("Latency sweep: {} depths x {} images x {} runs".format(
            num_groups + 1, len(lat_samples), args.runs))
        per_depth_lat = {}
        for d in range(num_groups + 1):
            times = []
            for s in lat_samples:
                lq = s["lq"]
                times.extend(timed(lambda: truncated_forward(rcan, lq, d),
                                   args.warmup, args.runs, device))
            per_depth_lat[str(d)] = {
                "median_ms": statistics.median(times),
                "p90_ms": percentile(times, 90),
            }
            print("  d={:2d}: median {:.3f} ms".format(d, per_depth_lat[str(d)]["median_ms"]))
        latency_report = {
            "num_groups": num_groups,
            "images": len(lat_samples),
            "warmup": args.warmup,
            "runs": args.runs,
            "per_depth": per_depth_lat,
        }
        depth_latency_medians = {d: per_depth_lat[str(d)]["median_ms"]
                                 for d in range(num_groups + 1)}

        # quality sweep per dataset ------------------------------------------
        for ds_name in dataset_names:
            samples = load_samples(parsed_opt, ds_name, args.max_images, device)
            print("{}: {} images, depth sweep...".format(ds_name, len(samples)))
            per_image_rows = []
            loo_deltas = {g: [] for g in range(num_groups)} if args.loo else None
            collect_gates = ds_name == dataset_names[0]

            for idx, sample in enumerate(samples):
                with torch.no_grad():
                    ctx = (ca_gate_collector(rcan, gate_storage)
                           if collect_gates else contextlib.nullcontext())
                    with ctx:
                        outputs = instrumented_depth_outputs(rcan, sample["lq"])
                dense_out = outputs[num_groups]
                psnr_y_by_depth, psnr_rgb_by_depth, ssim_by_depth, mae_by_depth = [], [], [], []
                for d, out in enumerate(outputs):
                    rgb, y, ssim = psnr_pair(out, sample["gt"], scale, with_ssim=args.ssim)
                    psnr_rgb_by_depth.append(rgb)
                    psnr_y_by_depth.append(y)
                    ssim_by_depth.append(ssim)
                    mae_by_depth.append(float((out - dense_out).abs().mean().item()))
                    csv_rows.append({
                        "dataset": ds_name, "image": sample["name"], "depth": d,
                        "psnr_y": y, "psnr_rgb": rgb,
                        "ssim_y": ssim if ssim is not None else "",
                        "mae_vs_dense": mae_by_depth[-1],
                    })
                per_image_rows.append({
                    "name": sample["name"],
                    "psnr_y_by_depth": psnr_y_by_depth,
                    "psnr_rgb_by_depth": psnr_rgb_by_depth,
                    "ssim_y_by_depth": ssim_by_depth if args.ssim else None,
                })
                if args.loo and (args.loo_images == 0 or idx < args.loo_images):
                    dense_y = psnr_y_by_depth[num_groups]
                    with torch.no_grad():
                        for g in range(num_groups):
                            out = subset_forward(rcan, sample["lq"], {g})
                            _, y, _ = psnr_pair(out, sample["gt"], scale)
                            loo_deltas[g].append(y - dense_y)
                del outputs
                if (idx + 1) % 10 == 0:
                    print("  {}/{} images".format(idx + 1, len(samples)))

            depth_summary = []
            for d in range(num_groups + 1):
                ys = [r["psnr_y_by_depth"][d] for r in per_image_rows]
                deltas = [r["psnr_y_by_depth"][d] - r["psnr_y_by_depth"][num_groups]
                          for r in per_image_rows]
                depth_summary.append({
                    "depth": d,
                    "psnr_y": mean(ys),
                    "psnr_rgb": mean([r["psnr_rgb_by_depth"][d] for r in per_image_rows]),
                    "ssim_y": mean([r["ssim_y_by_depth"][d] for r in per_image_rows])
                    if args.ssim else None,
                    "worst_delta_y": min(deltas),
                })
            loo_summary = None
            if args.loo:
                loo_summary = [{"skip_group": g, "delta_y": mean(loo_deltas[g]),
                                "images_used": len(loo_deltas[g])}
                               for g in range(num_groups)]
            dataset_reports.append({
                "dataset": ds_name,
                "images": len(per_image_rows),
                "depth_summary": depth_summary,
                "loo_summary": loo_summary,
                "oracle": oracle_ladder(per_image_rows, depth_latency_medians,
                                        num_groups, eps_list),
                "per_image": per_image_rows,
            })

        # dense probes ---------------------------------------------------------
        dense_probes = None
        if args.dense_probes:
            print("Dense probes (fp16 / channels_last)...")
            probe_sample = lat_samples[0]
            lq = probe_sample["lq"]
            dense_probes = []
            base_times = timed(lambda: rcan(lq), args.warmup, args.runs, device)
            with torch.no_grad():
                base_out = rcan(lq)
            _, base_y, _ = psnr_pair(base_out, probe_sample["gt"], scale)
            dense_probes.append({"config": "eager fp32", "median_ms": statistics.median(base_times),
                                 "psnr_y_delta": 0.0})

            def fp16_forward():
                with torch.autocast("cuda", dtype=torch.float16):
                    return rcan(lq)
            fp16_times = timed(fp16_forward, args.warmup, args.runs, device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                fp16_out = rcan(lq).float()
            _, fp16_y, _ = psnr_pair(fp16_out, probe_sample["gt"], scale)
            dense_probes.append({"config": "autocast fp16",
                                 "median_ms": statistics.median(fp16_times),
                                 "psnr_y_delta": fp16_y - base_y})
            try:
                lq_cl = lq.to(memory_format=torch.channels_last)
                rcan_cl = rcan.to(memory_format=torch.channels_last)
                cl_times = timed(lambda: rcan_cl(lq_cl), args.warmup, args.runs, device)
                dense_probes.append({"config": "channels_last fp32",
                                     "median_ms": statistics.median(cl_times),
                                     "psnr_y_delta": None})
            except Exception as exc:  # pragma: no cover
                dense_probes.append({"config": "channels_last fp32 (failed: {})".format(exc),
                                     "median_ms": float("nan"), "psnr_y_delta": None})

        # CA gate summary --------------------------------------------------------
        ca_summary, npz_payload = summarize_gates(gate_storage)

        # export -----------------------------------------------------------------
        report = {
            "generated": stamp,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "scale": scale,
            "num_groups": num_groups,
            "equivalence": equivalence,
            "latency": latency_report,
            "dense_probes": dense_probes,
            "ca_gate_summary": ca_summary,
            "datasets": dataset_reports,
            "args": {k: str(v) for k, v in vars(args).items()},
        }
        (output_dir / "depth_sweep_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2))
        (output_dir / "depth_sweep_report.md").write_text(render_markdown(report))
        with open(output_dir / "per_image_depth.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "dataset", "image", "depth", "psnr_y", "psnr_rgb", "ssim_y", "mae_vs_dense"])
            writer.writeheader()
            writer.writerows(csv_rows)
        if npz_payload is not None:
            np.savez_compressed(output_dir / "ca_gates.npz", **npz_payload)

        archive = (Path(args.archive).expanduser() if args.archive else
                   output_dir.parent / "B2RSR_DEPTH_SWEEP_{}.tar.gz".format(stamp))
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)

        print("\n完成。请下载这一个文件供分析：\n{}".format(archive))
        print("\n摘要：")
        print((output_dir / "depth_sweep_report.md").read_text())
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
