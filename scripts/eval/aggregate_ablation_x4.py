#!/usr/bin/env python3
"""Aggregate the x4 component-ablation runs into one table-ready record.

Reads only validation histories written by codes/train.py; it never opens a
final benchmark directory, so the validation-only rule of the study is kept.

For each arm it reports the three-seed mean and sample standard deviation of the
best validation PSNR-Y / SSIM-Y, plus the step at which the best value occurred.

Usage (repo root):
    python scripts/eval/aggregate_ablation_x4.py
    python scripts/eval/aggregate_ablation_x4.py --json out/ablation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ARMS = (
    ("A", "armA_random", "Random", "0.5 T-L1 + 0.5 GT-L1"),
    ("B", "armB_gtonly", "Ordered transplant", "GT-L1 only"),
    ("C", "armC_full", "Ordered transplant", "0.5 T-L1 + 0.5 GT-L1"),
)
SEEDS = (0, 1, 2)
REPO = Path(__file__).resolve().parents[2]


def load_best(arm_stem: str, seed: int) -> dict | None:
    """Return the record with maximum validation PSNR-Y for one run.

    Layout produced by codes/run.py:
        experiments/ABL_<arm>/x4_seed<N>/validation/history.jsonl
    """
    history = (REPO / "experiments" / f"ABL_{arm_stem}"
               / f"x4_seed{seed}" / "validation" / "history.jsonl")
    if not history.is_file():
        return None
    best = None
    count = 0
    for line in history.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        psnr = row.get("psnr_y")
        if psnr is None:
            continue
        count += 1
        if best is None or psnr > best["psnr_y"]:
            best = row
    if best is None:
        return None
    best = dict(best)
    best["_records"] = count
    best["_history"] = str(history.relative_to(REPO))
    best["_history_sha256"] = hashlib.sha256(history.read_bytes()).hexdigest()
    return best


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def fmt(mean, sd, digits=4) -> str:
    if mean is None:
        return "TBD"
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the aggregate as JSON to this path")
    args = ap.parse_args()

    out = {"arms": {}, "seeds": list(SEEDS), "complete": True}
    rows = []

    for arm_id, run_stem, init_label, loss_label in ARMS:
        psnrs, ssims, steps, per_seed = [], [], [], {}
        for seed in SEEDS:
            rec = load_best(run_stem, seed)
            if rec is None:
                out["complete"] = False
                per_seed[seed] = None
                continue
            per_seed[seed] = {
                "psnr_y": rec["psnr_y"],
                "ssim_y": rec.get("ssim_y"),
                "step": rec.get("step"),
                "records": rec["_records"],
                "history": rec["_history"],
                "history_sha256": rec["_history_sha256"],
            }
            psnrs.append(float(rec["psnr_y"]))
            if rec.get("ssim_y") is not None:
                ssims.append(float(rec["ssim_y"]))
            if rec.get("step") is not None:
                steps.append(int(rec["step"]))

        p_mean, p_sd = summarize(psnrs)
        s_mean, s_sd = summarize(ssims)
        out["arms"][arm_id] = {
            "run_stem": run_stem,
            "initialization": init_label,
            "recovery_loss": loss_label,
            "n_seeds_found": len(psnrs),
            "psnr_y_mean": p_mean, "psnr_y_sd": p_sd,
            "ssim_y_mean": s_mean, "ssim_y_sd": s_sd,
            "best_steps": steps,
            "per_seed": per_seed,
        }
        rows.append((arm_id, init_label, loss_label,
                     fmt(p_mean, p_sd), fmt(s_mean, s_sd, 6),
                     "/".join(str(s) for s in steps) if steps else "TBD",
                     len(psnrs)))

    # console summary
    print(f"{'Arm':<4}{'Init':<22}{'Loss':<24}{'PSNR-Y':<22}{'SSIM-Y':<24}{'Steps':<18}seeds")
    for r in rows:
        print(f"{r[0]:<4}{r[1]:<22}{r[2]:<24}{r[3]:<22}{r[4]:<24}{r[5]:<18}{r[6]}/3")

    a = out["arms"]["A"]["psnr_y_mean"]
    b = out["arms"]["B"]["psnr_y_mean"]
    c = out["arms"]["C"]["psnr_y_mean"]
    print()
    if None not in (a, c):
        print(f"C - A (transplant contribution)      = {c - a:+.4f} dB")
    if None not in (b, c):
        print(f"C - B (Teacher-supervision contrib.) = {c - b:+.4f} dB")
    if not out["complete"]:
        print("\nWARNING: some runs are missing; values above are partial.")

    # LaTeX body rows for tab:ablation
    print("\n--- LaTeX rows (paste into tab:ablation) ---")
    for r in rows:
        print(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} \\\\")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
