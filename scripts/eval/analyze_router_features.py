#!/usr/bin/env python3
"""Stage-1 router feature analysis for mainline alpha (training-free, CPU-only).

Reads per_patch.csv from run_cascade_oracle.py plus the LR images, computes
cheap hand-crafted difficulty features per patch, and evaluates how well each
feature (and a tiny logistic combination) separates "bicubic is enough"
(bicubic_drop <= eps) from "must escalate" patches.

Outputs:
  * per-feature AUC (rank-based, no sklearn needed)
  * conservative operating points: recall-of-escalation >= 99% / 99.9%
    -> achievable cheap fraction and implied cascade latency/speedup
  * logistic-combo AUC via simple gradient descent (numpy only)
  * router_feature_report.{json,md} + enriched per_patch_features.csv

Usage:
  python scripts/eval/analyze_router_features.py \
      --oracle-dir results/cascade_oracle/<stamp> \
      --data-root /home/featurize/data
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def patch_features(patch_bgr):
    """Cheap difficulty features on an LR patch (uint8 BGR)."""
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # high-frequency energy via DCT: fraction outside top-left 8x8 block
    h, w = gray.shape
    dct = cv2.dct(gray[: h // 2 * 2, : w // 2 * 2])
    total = float((dct ** 2).sum()) + 1e-8
    low = float((dct[:8, :8] ** 2).sum())
    return {
        "var": float(gray.var()),
        "grad_mean": float(grad_mag.mean()),
        "grad_p90": float(np.percentile(grad_mag, 90)),
        "lap_var": float(lap.var()),
        "hf_ratio": float(1.0 - low / total),
        "edge_density": float((grad_mag > 32).mean()),
    }


FEATURE_KEYS = ["var", "grad_mean", "grad_p90", "lap_var", "hf_ratio", "edge_density"]


# ---------------------------------------------------------------------------
# metrics (numpy only)
# ---------------------------------------------------------------------------

def auc_rank(scores, labels):
    """AUC via Mann-Whitney U. labels: 1 = escalate (positive)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def conservative_operating_point(scores, labels, recall_target):
    """Threshold so that recall of escalation (label=1) >= target.
    Returns achieved cheap fraction (patches routed cheap) and miss rate."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos_scores = np.sort(scores[labels == 1])
    if len(pos_scores) == 0:
        return {"cheap_fraction": 1.0, "escalation_recall": 1.0, "missed_pos": 0}
    # escalate if score >= t ; choose t as low quantile of positive scores
    k = int(np.floor((1.0 - recall_target) * len(pos_scores)))
    t = pos_scores[k] if k < len(pos_scores) else pos_scores[-1] + 1e-9
    escalate = scores >= t
    cheap = ~escalate
    recall = float((escalate & (labels == 1)).sum() / max(1, labels.sum()))
    return {
        "threshold": float(t),
        "cheap_fraction": float(cheap.mean()),
        "escalation_recall": recall,
        "missed_pos": int(((~escalate) & (labels == 1)).sum()),
    }


def logistic_fit(features, labels, iters=3000, lr=0.5):
    """Tiny logistic regression on standardized features, numpy only."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-8
    xs = (x - mu) / sd
    xs = np.hstack([xs, np.ones((len(xs), 1))])
    w = np.zeros(xs.shape[1])
    n = len(y)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-xs @ w))
        grad = xs.T @ (p - y) / n
        w -= lr * grad
    return xs @ w, w, mu, sd


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", required=False,
                        help="run_cascade_oracle 的输出目录（含 per_patch.csv）；"
                             "缺省自动取 results/cascade_oracle 下最新一个")
    parser.add_argument("--data-root", default="/home/featurize/data")
    parser.add_argument("--dataset-dir", default="DIV2K_valid_2K")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.1,
                        help="标签阈值：bicubic_drop <= eps 记为 cheap-able")
    parser.add_argument("--dense-whole-ms", type=float, default=959.6)
    parser.add_argument("--dense-b16-ms", type=float, default=22.86)
    parser.add_argument("--bicubic-ms", type=float, default=0.505)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    # locate oracle output
    if args.oracle_dir:
        oracle_dir = Path(args.oracle_dir)
    else:
        candidates = sorted((ROOT / "results/cascade_oracle").glob("*/per_patch.csv"))
        if not candidates:
            raise FileNotFoundError("未找到 per_patch.csv；先运行 run_cascade_oracle.py")
        oracle_dir = candidates[-1].parent
    csv_path = oracle_dir / "per_patch.csv"
    print("读取 {}".format(csv_path))
    rows = list(csv.DictReader(open(csv_path)))
    print("{} 个 patch 记录".format(len(rows)))

    # load LR images and compute features
    lr_dir = Path(args.data_root) / "SRBenchmarks" / args.dataset_dir / \
        "LR_bicubic" / "X{}".format(args.scale)
    ps = args.patch
    cache = {}
    feats, labels, drops = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        name = r["image"]
        if name not in cache:
            path = lr_dir / "{}x{}.png".format(name, args.scale)
            cache[name] = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if cache[name] is None:
                raise FileNotFoundError(path)
        img = cache[name]
        py, px = int(r["py"]), int(r["px"])
        patch = img[py:py + ps, px:px + ps]
        feats.append(patch_features(patch))
        drop = float(r["bicubic_drop"])
        drops.append(drop)
        labels.append(1 if drop > args.eps else 0)  # 1 = must escalate
        if (i + 1) % 1000 == 0:
            print("  {}/{} ({:.0f}s)".format(i + 1, len(rows), time.time() - t0))

    labels = np.array(labels)
    n = len(labels)
    esc_frac = labels.mean()
    print("必须升级的 patch 比例（eps={}）: {:.1%}".format(args.eps, esc_frac))

    # per-feature AUC
    feature_aucs = {}
    for key in FEATURE_KEYS:
        scores = [f[key] for f in feats]
        feature_aucs[key] = auc_rank(scores, labels)
    best_single = max(feature_aucs, key=lambda k: feature_aucs[k])

    # logistic combo
    x = np.array([[f[k] for k in FEATURE_KEYS] for f in feats])
    combo_scores, w, mu, sd = logistic_fit(x, labels)
    combo_auc = auc_rank(combo_scores, labels)

    # conservative operating points + implied latency
    patches_per_image = n / len(cache)

    def implied(cheap_fraction):
        ms = patches_per_image * (
            cheap_fraction * args.bicubic_ms +
            (1 - cheap_fraction) * args.dense_b16_ms)
        return {"ms_per_image": ms, "speedup": args.dense_whole_ms / ms}

    operating = {}
    for recall in (0.99, 0.999, 1.0):
        op = conservative_operating_point(combo_scores, labels, recall)
        op.update(implied(op["cheap_fraction"]))
        operating["recall_{}".format(recall)] = op
    oracle_op = implied(1 - esc_frac)

    # export
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "results/router_features" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated": stamp,
        "source": str(csv_path),
        "patches": n,
        "images": len(cache),
        "eps_db": args.eps,
        "escalate_fraction": float(esc_frac),
        "feature_aucs": feature_aucs,
        "best_single_feature": best_single,
        "logistic_combo_auc": combo_auc,
        "logistic_weights": {k: float(v) for k, v in zip(FEATURE_KEYS + ["bias"], w)},
        "latency_model": {
            "dense_whole_ms": args.dense_whole_ms,
            "dense_b16_ms": args.dense_b16_ms,
            "bicubic_ms": args.bicubic_ms,
            "patches_per_image": patches_per_image,
        },
        "oracle_reference": oracle_op,
        "conservative_operating_points": operating,
        "notes": [
            "标签: bicubic_drop > eps 记为必须升级 (positive)。",
            "AUC 为秩和法；combo 为 6 特征 logistic（全量拟合，无 train/val 切分，"
            "属于乐观筛查上界；正式判别器需按图划分训练/验证）。",
            "conservative operating points 给出漏检率约束下可实现的 cheap 比例。",
        ],
    }
    (out_dir / "router_feature_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))

    lines = [
        "# Router Feature Analysis (Stage 1)", "",
        "- source: {}".format(csv_path),
        "- patches: {} / images: {} / eps: {} dB".format(n, len(cache), args.eps),
        "- must-escalate fraction: {:.1%}".format(esc_frac),
        "", "## Per-feature AUC (higher = better separates escalation)", "",
        "| feature | AUC |", "|---|---:|",
    ]
    for k in sorted(feature_aucs, key=lambda k: -feature_aucs[k]):
        lines.append("| {} | {:.4f} |".format(k, feature_aucs[k]))
    lines.extend([
        "", "**logistic combo AUC: {:.4f}**".format(combo_auc), "",
        "## Conservative operating points (combo score)", "",
        "| escalation recall | cheap % | missed | ms/img | speedup |",
        "|---:|---:|---:|---:|---:|",
    ])
    for key, op in operating.items():
        lines.append("| {} | {:.1%} | {} | {:.1f} | {:.3f}x |".format(
            key.split("_")[1], op["cheap_fraction"], op["missed_pos"],
            op["ms_per_image"], op["speedup"]))
    lines.extend([
        "| oracle (perfect) | {:.1%} | 0 | {:.1f} | {:.3f}x |".format(
            1 - esc_frac, oracle_op["ms_per_image"], oracle_op["speedup"]),
        "", "## Notes", ""] + ["- " + s for s in report["notes"]])
    (out_dir / "router_feature_report.md").write_text("\n".join(lines))

    # enriched csv
    with open(out_dir / "per_patch_features.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "py", "px", "bicubic_drop", "label_escalate"]
                        + FEATURE_KEYS + ["combo_score"])
        for r, f, s in zip(rows, feats, combo_scores):
            writer.writerow([r["image"], r["py"], r["px"], r["bicubic_drop"],
                             int(float(r["bicubic_drop"]) > args.eps)]
                            + [f[k] for k in FEATURE_KEYS] + [float(s)])

    print("\n完成，输出目录：{}".format(out_dir))
    print((out_dir / "router_feature_report.md").read_text())


if __name__ == "__main__":
    main()
