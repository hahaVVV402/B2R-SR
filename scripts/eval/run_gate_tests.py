#!/usr/bin/env python3
"""Run the four B2R-SR decision gates with one trained checkpoint."""

import argparse
import contextlib
import hashlib
import json
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

import models.networks as networks  # noqa: E402
import options.options as option  # noqa: E402
import utils.util as util  # noqa: E402
from data import create_dataloader, create_dataset  # noqa: E402
from data.util import bgr2ycbcr  # noqa: E402


def mean(values):
    return float(sum(values) / len(values))


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
        raise FileNotFoundError(
            "未找到 checkpoint/export tar；用 --checkpoint 指定 120000_G.pth 或导出 tar。")
    if source.suffix == ".pth":
        return source, temporary
    if source.suffix != ".tar":
        raise ValueError("--checkpoint 仅支持 .pth 或 export .tar")

    temporary = tempfile.TemporaryDirectory(prefix="b2rsr-gates-")
    with tarfile.open(source, "r") as archive:
        archive.extractall(temporary.name)
    matches = list(Path(temporary.name).rglob("120000_G.pth"))
    if len(matches) != 1:
        temporary.cleanup()
        raise RuntimeError("导出包中应恰好包含一个 120000_G.pth")
    return matches[0], temporary


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


def to_images(tensor, gt, scale):
    sr_img = util.tensor2img(tensor.detach()[0], out_type=np.uint8, min_max=(0, 255))
    gt_img = util.tensor2img(gt.detach()[0], out_type=np.uint8, min_max=(0, 255))
    sr_img, gt_img = util.crop_border([sr_img, gt_img], scale)
    psnr_rgb = util.calculate_psnr(sr_img, gt_img)
    sr_y = bgr2ycbcr(sr_img / 255.0, only_y=True) * 255.0
    gt_y = bgr2ycbcr(gt_img / 255.0, only_y=True) * 255.0
    psnr_y = util.calculate_psnr(sr_y, gt_y)
    return sr_img, gt_img, float(psnr_rgb), float(psnr_y)


def equivalence_psnr(a, b, scale):
    a_img = util.tensor2img(a.detach()[0], out_type=np.uint8, min_max=(0, 255))
    b_img = util.tensor2img(b.detach()[0], out_type=np.uint8, min_max=(0, 255))
    a_img, b_img = util.crop_border([a_img, b_img], scale)
    return float(util.calculate_psnr(a_img, b_img))


def boundary_errors(a, b, period, radius=2):
    diff = (a - b).abs().mean(dim=1)[0]
    h, w = diff.shape
    mask = torch.zeros((h, w), dtype=torch.bool, device=diff.device)
    for pos in range(period, h, period):
        mask[max(0, pos - radius):min(h, pos + radius + 1), :] = True
    for pos in range(period, w, period):
        mask[:, max(0, pos - radius):min(w, pos + radius + 1)] = True
    boundary = float(diff[mask].mean().item()) if mask.any() else 0.0
    interior = float(diff[~mask].mean().item()) if (~mask).any() else 0.0
    return boundary, interior


@contextlib.contextmanager
def route_override(plugin, mode):
    if mode == "current":
        yield
        return

    had_instance_attr = "_route_mask" in plugin.__dict__
    previous = plugin.__dict__.get("_route_mask")

    def diagnostic_route(self, feat, stage_idx, degradation_score, training_hard,
                         target_keep_stage=None):
        windows, meta = self._window_partition(feat)
        b, num_wins = windows.size(0), windows.size(1)
        h, w, hp, wp, _, _, _, _ = meta
        feat_pad = feat
        if hp != h or wp != w:
            feat_pad = F.pad(feat, (0, wp - w, 0, hp - h), mode="reflect")
        router_map = self.router_heads[stage_idx](feat_pad)
        learned = F.avg_pool2d(
            router_map, kernel_size=self.route_window, stride=self.route_window
        ).view(b, num_wins)
        variance = windows.var(dim=(2, 3, 4), unbiased=False)
        variance = (variance - variance.mean(dim=1, keepdim=True)) / (
            variance.std(dim=1, keepdim=True, unbiased=False) + 1e-6)

        if mode == "learned":
            score = learned
        elif mode == "variance":
            score = variance
        elif mode == "frequency":
            dx = (windows[..., 1:] - windows[..., :-1]).abs().mean(dim=(2, 3, 4))
            dy = (windows[..., 1:, :] - windows[..., :-1, :]).abs().mean(dim=(2, 3, 4))
            score = dx + dy
        elif mode == "feature_delta_teacher":
            dense_stage = self.stage_modules[stage_idx](feat)
            score = self._benefit_target_from_delta(dense_stage - feat)
        elif mode == "random":
            score = torch.rand_like(learned)
        else:
            raise ValueError("unknown routing mode: {}".format(mode))

        prob = torch.sigmoid(score)
        if target_keep_stage is None:
            hard = (prob >= 0.5).float()
        else:
            hard = self._topk_mask(score, target_keep_stage)
        return hard, prob, meta, score

    plugin._route_mask = types.MethodType(diagnostic_route, plugin)
    try:
        yield
    finally:
        if had_instance_attr:
            plugin._route_mask = previous
        else:
            del plugin._route_mask


@contextlib.contextmanager
def force_all_keep(plugin):
    had_instance_attr = "_target_keep_per_stage" in plugin.__dict__
    previous = plugin.__dict__.get("_target_keep_per_stage")

    def all_keep(self, degradation_score, complexity_score):
        return torch.ones(
            degradation_score.size(0), len(self.stage_modules),
            dtype=degradation_score.dtype, device=degradation_score.device)

    plugin._target_keep_per_stage = types.MethodType(all_keep, plugin)
    try:
        yield
    finally:
        if had_instance_attr:
            plugin._target_keep_per_stage = previous
        else:
            del plugin._target_keep_per_stage


def forward_plugin(plugin, lq):
    output, info = plugin(lq)
    return output, info


def gate0(plugin, samples, scale):
    rows = []
    period = plugin.route_window * scale
    with torch.no_grad(), force_all_keep(plugin):
        for sample in samples:
            dense = plugin.backbone(sample["lq"])
            routed, info = forward_plugin(plugin, sample["lq"])
            _, _, dense_rgb, dense_y = to_images(dense, sample["gt"], scale)
            _, _, routed_rgb, routed_y = to_images(routed, sample["gt"], scale)
            boundary, interior = boundary_errors(routed, dense, period)
            rows.append({
                "name": sample["name"],
                "dense_psnr_rgb": dense_rgb,
                "dense_psnr_y": dense_y,
                "all_keep_psnr_rgb": routed_rgb,
                "all_keep_psnr_y": routed_y,
                "gt_psnr_y_delta": routed_y - dense_y,
                "equivalence_psnr": equivalence_psnr(routed, dense, scale),
                "tensor_mae": float((routed - dense).abs().mean().item()),
                "boundary_mae": boundary,
                "interior_mae": interior,
                "keep": float(info["keep_ratio_total"].mean().item()),
            })
    delta = mean([r["gt_psnr_y_delta"] for r in rows])
    eq = mean([r["equivalence_psnr"] for r in rows])
    if abs(delta) <= 0.02 and eq >= 50.0:
        verdict = "PASS"
    elif abs(delta) <= 0.10:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "mean_gt_psnr_y_delta": delta,
            "mean_equivalence_psnr": eq,
            "mean_boundary_mae": mean([r["boundary_mae"] for r in rows]),
            "mean_interior_mae": mean([r["interior_mae"] for r in rows]),
            "rows": rows}


def evaluate_routing(plugin, samples, scale, mode, budget, seed):
    plugin.user_budget = float(budget)
    torch.manual_seed(seed)
    psnr_rgb, psnr_y, keeps, targets, per_stage, target_per_stage = [], [], [], [], [], []
    with torch.no_grad(), route_override(plugin, mode):
        for sample in samples:
            output, info = forward_plugin(plugin, sample["lq"])
            _, _, rgb, y = to_images(output, sample["gt"], scale)
            psnr_rgb.append(rgb)
            psnr_y.append(y)
            keeps.append(float(info["keep_ratio_total"].mean().item()))
            targets.append(float(info["target_keep_per_stage"].mean().item()))
            per_stage.append(info["keep_ratio_per_stage"].mean(dim=0).cpu().tolist())
            target_per_stage.append(info["target_keep_per_stage"].mean(dim=0).cpu().tolist())
    stage_mean = np.asarray(per_stage, dtype=np.float64).mean(axis=0)
    target_stage_mean = np.asarray(target_per_stage, dtype=np.float64).mean(axis=0)
    return {"psnr_rgb": mean(psnr_rgb), "psnr_y": mean(psnr_y),
            "keep": mean(keeps), "target": mean(targets),
            "per_stage_keep": stage_mean.tolist(),
            "per_stage_std": float(stage_mean.std()),
            "per_stage_target": target_stage_mean.tolist(),
            "per_stage_target_std": float(target_stage_mean.std())}


def gate1(plugin, samples, scale, budget, seed, random_repeats):
    results = {}
    for mode in ("current", "learned", "variance", "frequency", "feature_delta_teacher"):
        results[mode] = evaluate_routing(plugin, samples, scale, mode, budget, seed)
    random_runs = [
        evaluate_routing(plugin, samples, scale, "random", budget, seed + i)
        for i in range(random_repeats)
    ]
    results["random"] = {
        key: mean([run[key] for run in random_runs])
        for key in ("psnr_rgb", "psnr_y", "keep", "target", "per_stage_std",
                    "per_stage_target_std")
    }
    simple_best = max(results[m]["psnr_y"] for m in ("variance", "frequency", "random"))
    gain = results["current"]["psnr_y"] - simple_best
    screening = "PROMISING" if gain > 0.02 else ("TIED" if gain >= -0.02 else "WEAK")
    return {"verdict": "INCONCLUSIVE", "screening": screening,
            "current_gain_over_best_simple_y": gain,
            "results": results,
            "note": "仅比较推理期最终 PSNR；feature_delta_teacher 是训练代理上界，不是真实 GT counterfactual oracle。正式 Gate 1 仍需相关性、top-K overlap/regret 和 stage 分析。"}


def gate2(plugin, samples, scale, budgets, seed):
    rows = []
    for budget in budgets:
        result = evaluate_routing(plugin, samples, scale, "current", budget, seed)
        result["budget"] = budget
        rows.append(result)
    keeps = [r["keep"] for r in rows]
    psnrs = [r["psnr_y"] for r in rows]
    keep_monotonic = all(a <= b + 1e-9 for a, b in zip(keeps, keeps[1:]))
    psnr_monotonic = all(a <= b + 0.01 for a, b in zip(psnrs, psnrs[1:]))
    unique_keeps = len(set(round(v, 6) for v in keeps))
    verdict = "PASS" if keep_monotonic and psnr_monotonic and unique_keeps >= 4 else "FAIL"
    return {"verdict": verdict, "keep_monotonic": keep_monotonic,
            "psnr_monotonic_with_0.01db_tolerance": psnr_monotonic,
            "unique_keep_points": unique_keeps, "rows": rows}


def timed_forward(callable_, warmup, runs, device):
    with torch.no_grad():
        for _ in range(warmup):
            callable_()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(runs):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                callable_()
                end.record()
                end.synchronize()
                timings.append(float(start.elapsed_time(end)))
            else:
                start = time.perf_counter()
                callable_()
                timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def gate3(plugin, samples, budget, warmup, runs, latency_images, device):
    plugin.user_budget = float(budget)
    previous_sync = plugin.sync_latency
    plugin.sync_latency = False
    dense_times, routed_times = [], []
    try:
        for sample in samples[:latency_images]:
            lq = sample["lq"]
            dense_times.extend(timed_forward(lambda: plugin.backbone(lq), warmup, runs, device))
            routed_times.extend(timed_forward(lambda: plugin(lq), warmup, runs, device))
    finally:
        plugin.sync_latency = previous_sync
    dense_median = statistics.median(dense_times)
    routed_median = statistics.median(routed_times)
    speedup = dense_median / routed_median
    screening = "FASTER" if speedup > 1.05 else ("TIED" if speedup >= 0.95 else "SLOWER")
    verdict = "FAIL" if speedup < 0.95 else "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "screening": screening,
        "dense_median_ms": dense_median,
        "dense_p90_ms": percentile(dense_times, 90),
        "b2rsr_median_ms": routed_median,
        "b2rsr_p90_ms": percentile(routed_times, 90),
        "speedup": speedup,
        "images": min(latency_images, len(samples)),
        "warmup": warmup,
        "runs_per_image": runs,
        "note": "重复前向延迟筛查，尚未进行质量匹配和多分辨率 break-even 分析，因此不会自动判定 PASS。",
    }


def render_markdown(report):
    g0, g1, g2, g3 = (report[key] for key in ("gate0", "gate1", "gate2", "gate3"))
    lines = [
        "# B2R-SR Gate Test Report", "",
        "- checkpoint: `{}`".format(report["checkpoint"]),
        "- dataset/images: {} / {}".format(report["dataset"], report["images"]),
        "- device: {}".format(report["device"]), "",
        "## Verdicts", "",
        "| Gate | Verdict | Key evidence |", "|---|---|---|",
        "| 0 Dense equivalence | {} | ΔPSNR-Y={:+.4f} dB; equivalence={:.2f} dB |".format(
            g0["verdict"], g0["mean_gt_psnr_y_delta"], g0["mean_equivalence_psnr"]),
        "| 1 Routing utility | {} ({}) | current − best simple={:+.4f} dB |".format(
            g1["verdict"], g1["screening"], g1["current_gain_over_best_simple_y"]),
        "| 2 Budget control | {} | unique keep points={}; monotonic keep={} |".format(
            g2["verdict"], g2["unique_keep_points"], g2["keep_monotonic"]),
        "| 3 Real latency | {} ({}) | dense={:.3f} ms; B2R={:.3f} ms; speedup={:.3f}× |".format(
            g3["verdict"], g3["screening"], g3["dense_median_ms"],
            g3["b2rsr_median_ms"], g3["speedup"]),
        "", "## Gate 1 routing policies", "",
        "| Policy | PSNR-Y | Keep |", "|---|---:|---:|",
    ]
    for mode, row in g1["results"].items():
        lines.append("| {} | {:.4f} | {:.4f} |".format(mode, row["psnr_y"], row["keep"]))
    lines.extend(["", "## Gate 2 budget sweep", "",
                  "| Budget | Target | Actual keep | PSNR-Y | Stage keep std | Stage target std |",
                  "|---:|---:|---:|---:|---:|---:|"])
    for row in g2["rows"]:
        lines.append("| {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.6f} | {:.6f} |".format(
            row["budget"], row["target"], row["keep"], row["psnr_y"],
            row["per_stage_std"], row["per_stage_target_std"]))
    lines.extend(["", "## Interpretation", "",
                  "这些是现有 checkpoint 的推理期诊断，不等同于完整重训练消融。",
                  "Gate 0 或 Gate 1 失败时，不应按原 v1 配置直接重训练。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="120000_G.pth 或 export tar；省略时自动查找")
    parser.add_argument("--config", default=str(ROOT / "codes/options/test/test_B2RSR_RCAN_X4.yml"))
    parser.add_argument("--data-root", default="/home/featurize/data")
    parser.add_argument("--dataset", default="Set5")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--budget", type=float, default=0.70)
    parser.add_argument("--budgets", default="0.45,0.55,0.65,0.70,0.75,0.85,0.95")
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--latency-images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--skip-data-prepare", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--archive", help="结果 tar.gz 路径；默认保存在 output-dir 的上一级")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("Gate 测试需要 CUDA GPU。")
    if args.max_images < 1 or args.random_repeats < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("max-images/random-repeats/runs 必须为正，warmup 不能为负。")

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
        plugin = networks.define_G(parsed_opt).to(device)
        state = torch.load(str(checkpoint), map_location="cpu")
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        plugin.load_state_dict(cleaned, strict=True)
        plugin.eval()
        plugin.sync_latency = False

        samples = load_samples(parsed_opt, args.dataset, args.max_images, device)
        budgets = [float(value) for value in args.budgets.split(",")]

        print("Gate 0/4: dense vs all-keep equivalence")
        result0 = gate0(plugin, samples, int(parsed_opt["scale"]))
        print("Gate 1/4: routing policy comparison")
        result1 = gate1(plugin, samples, int(parsed_opt["scale"]), args.budget,
                        args.seed, args.random_repeats)
        print("Gate 2/4: budget sweep")
        result2 = gate2(plugin, samples, int(parsed_opt["scale"]), budgets, args.seed)
        print("Gate 3/4: repeated latency")
        result3 = gate3(plugin, samples, args.budget, args.warmup, args.runs,
                        args.latency_images, device)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path(args.output_dir) if args.output_dir else (
            ROOT / "results/gates/B2RSR_RCAN_X4_120000" / stamp)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "dataset": args.dataset,
            "images": len(samples),
            "device": torch.cuda.get_device_name(0),
            "gate0": result0,
            "gate1": result1,
            "gate2": result2,
            "gate3": result3,
        }
        (output_dir / "gate_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2))
        (output_dir / "gate_report.md").write_text(render_markdown(report))

        archive = (Path(args.archive).expanduser() if args.archive else
                   output_dir.parent / "B2RSR_GATE_RESULTS_{}.tar.gz".format(stamp))
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)

        print("\nGate 测试完成：")
        print((output_dir / "gate_report.md").read_text())
        print("\n请下载这一个结果文件：\n{}".format(archive))
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
