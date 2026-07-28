#!/usr/bin/env python3
"""Stage-2b: train the patch difficulty router and run the REAL cascade.

Router: small 5-conv CNN on 64x64 LR patches -> P(escalate).
Split: BY IMAGE (80/20) to avoid leakage from overlapping patches.
Metrics: val AUC, image-budget operating points on val, and — the honest
part — a real composited cascade on the val images: route non-overlapping
patches with the trained router, paste bicubic or batched dense-RCAN patch
outputs, re-measure whole-image PSNR-Y and end-to-end latency (router cost
included).

Pre-registered gate G-router:
  val AUC >= 0.88, and real cascade satisfies
  ">=95% of val images lose <=0.1 dB vs dense" with measured speedup >=1.5x.

Usage:
  python scripts/eval/train_patch_router.py \
      --labels results/router_labels/<stamp>/labels.npz
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

import utils.util as util  # noqa: E402
from data.util import bgr2ycbcr, modcrop  # noqa: E402
from models.archs.RCAN_arch import RCAN  # noqa: E402


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class PatchRouter(nn.Module):
    """Tiny CNN: 64x64x3 LR patch -> escalation logit."""

    def __init__(self, width=24):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1), nn.ReLU(inplace=True),      # 32
            nn.Conv2d(w, w * 2, 3, 2, 1), nn.ReLU(inplace=True),  # 16
            nn.Conv2d(w * 2, w * 2, 3, 2, 1), nn.ReLU(inplace=True),  # 8
            nn.Conv2d(w * 2, w * 4, 3, 2, 1), nn.ReLU(inplace=True),  # 4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(w * 4, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def auc_rank(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def build_backbone(checkpoint, device):
    rcan = RCAN(n_resgroups=10, n_resblocks=20, n_feats=64, res_scale=1,
                n_colors=3, rgb_range=255, scale=4, reduction=16)
    state = torch.load(str(checkpoint), map_location="cpu")
    bs = {}
    for key, value in state.items():
        clean = key[7:] if key.startswith("module.") else key
        if clean.startswith("backbone."):
            bs[clean[len("backbone."):]] = value
    rcan.load_state_dict(bs if bs else dict(state), strict=True)
    rcan.eval()
    for p in rcan.parameters():
        p.requires_grad = False
    return rcan.to(device)


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
        raise FileNotFoundError("未找到 checkpoint")
    return src


def psnr_y_full(a_img, b_img, shave=4):
    a = a_img[shave:-shave, shave:-shave]
    b = b_img[shave:-shave, shave:-shave]
    a_y = bgr2ycbcr(a / 255.0, only_y=True) * 255.0
    b_y = bgr2ycbcr(b / 255.0, only_y=True) * 255.0
    return float(util.calculate_psnr(a_y, b_y))


def to_batch(patches_u8, device, to_rgb=False):
    arr = patches_u8[:, :, :, [2, 1, 0]] if to_rgb else patches_u8
    x = torch.from_numpy(arr.astype(np.float32)).permute(0, 3, 1, 2)
    return x.contiguous().to(device)


# ---------------------------------------------------------------------------
# real cascade on one image
# ---------------------------------------------------------------------------

def run_cascade_image(rcan, router, lr_img, threshold, ps, scale, device,
                      escalate_batch=16):
    """Route non-overlapping patches; returns (sr_image, timing dict)."""
    h_lr, w_lr = lr_img.shape[:2]
    ys = list(range(0, h_lr - ps + 1, ps))
    xs = list(range(0, w_lr - ps + 1, ps))
    coords = [(py, px) for py in ys for px in xs]

    torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    # 1) router pass (single batch over all patches)
    patch_arr = np.stack([lr_img[py:py + ps, px:px + ps] for py, px in coords])
    with torch.no_grad():
        logits = router(to_batch(patch_arr, device))
        probs = torch.sigmoid(logits).cpu().numpy()
    escalate_mask = probs >= threshold
    torch.cuda.synchronize(device)
    t_router = time.perf_counter()

    # 2) cheap path: bicubic on CPU
    out = np.zeros((h_lr * scale, w_lr * scale, 3), dtype=np.uint8)
    for k, (py, px) in enumerate(coords):
        if not escalate_mask[k]:
            patch = lr_img[py:py + ps, px:px + ps]
            out[py * scale:(py + ps) * scale, px * scale:(px + ps) * scale] = \
                cv2.resize(patch, (ps * scale, ps * scale), interpolation=cv2.INTER_CUBIC)
    t_cheap = time.perf_counter()

    # 3) escalation: batched dense RCAN patches (RCAN 吃 RGB，回贴时转回 BGR)
    esc_idx = np.where(escalate_mask)[0]
    with torch.no_grad():
        for s in range(0, len(esc_idx), escalate_batch):
            sel = esc_idx[s:s + escalate_batch]
            batch = to_batch(patch_arr[sel], device, to_rgb=True)
            sr = rcan(batch)
            sr_imgs = sr.clamp(0, 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            for bi, k in enumerate(sel):
                py, px = coords[k]
                out[py * scale:(py + ps) * scale, px * scale:(px + ps) * scale] = \
                    sr_imgs[bi][:, :, [2, 1, 0]]
    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    # leftover borders (image size not divisible by ps): bicubic fill
    if ys and ys[-1] + ps < h_lr:
        strip = lr_img[ys[-1] + ps:, :]
        out[(ys[-1] + ps) * scale:, :] = cv2.resize(
            strip, (w_lr * scale, (h_lr - ys[-1] - ps) * scale),
            interpolation=cv2.INTER_CUBIC)
    if xs and xs[-1] + ps < w_lr:
        strip = lr_img[:, xs[-1] + ps:]
        out[:, (xs[-1] + ps) * scale:] = cv2.resize(
            strip, ((w_lr - xs[-1] - ps) * scale, h_lr * scale),
            interpolation=cv2.INTER_CUBIC)

    timing = {
        "total_ms": (t_end - t_start) * 1000.0,
        "router_ms": (t_router - t_start) * 1000.0,
        "cheap_ms": (t_cheap - t_router) * 1000.0,
        "escalate_ms": (t_end - t_cheap) * 1000.0,
        "cheap_fraction": float(1.0 - escalate_mask.mean()),
        "patches": len(coords),
    }
    return out, timing


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=False,
                        help="labels.npz 路径；缺省取 results/router_labels 最新")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root", default="/home/featurize/data")
    parser.add_argument("--dataset-dir", default="DIV2K_valid_2K")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("需要 CUDA GPU。")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # load labels
    # ------------------------------------------------------------------
    if args.labels:
        labels_path = Path(args.labels)
    else:
        candidates = sorted((ROOT / "results/router_labels").glob("*/labels.npz"))
        if not candidates:
            raise FileNotFoundError("未找到 labels.npz；先运行 generate_router_labels.py")
        labels_path = candidates[-1]
    print("载入 {}".format(labels_path))
    data = np.load(labels_path, allow_pickle=False)
    patches, drops = data["patches"], data["drops"]
    image_ids = data["image_ids"]
    labels = (drops > args.eps).astype(np.float32)
    print("{} patches, escalate 比例 {:.1%}".format(len(labels), labels.mean()))

    # split by image
    unique_imgs = np.unique(image_ids)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(unique_imgs)
    n_val = max(1, int(len(unique_imgs) * args.val_frac))
    val_imgs = set(unique_imgs[:n_val].tolist())
    val_mask = np.isin(image_ids, list(val_imgs))
    tr_idx, va_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    print("train {} patches / val {} patches ({} val images)".format(
        len(tr_idx), len(va_idx), n_val))

    # ------------------------------------------------------------------
    # train router
    # ------------------------------------------------------------------
    router = PatchRouter(args.width).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=args.lr)
    pos_weight = torch.tensor([(1 - labels[tr_idx].mean()) /
                               max(labels[tr_idx].mean(), 1e-6)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state = 0.0, None
    t0 = time.time()
    for epoch in range(args.epochs):
        router.train()
        perm = np.random.permutation(tr_idx)
        losses = []
        for s in range(0, len(perm), args.batch):
            sel = perm[s:s + args.batch]
            x = to_batch(patches[sel], device)
            y = torch.from_numpy(labels[sel]).to(device)
            opt.zero_grad()
            loss = criterion(router(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        # val AUC
        router.eval()
        val_scores = []
        with torch.no_grad():
            for s in range(0, len(va_idx), 512):
                sel = va_idx[s:s + 512]
                val_scores.append(torch.sigmoid(
                    router(to_batch(patches[sel], device))).cpu().numpy())
        val_scores = np.concatenate(val_scores)
        auc = auc_rank(val_scores, labels[va_idx].astype(int))
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in router.state_dict().items()}
        print("epoch {:2d}: loss {:.4f} | val AUC {:.4f} | best {:.4f} ({:.0f}s)".format(
            epoch + 1, float(np.mean(losses)), auc, best_auc, time.time() - t0))

    router.load_state_dict(best_state)
    router.eval()
    print("best val AUC: {:.4f}".format(best_auc))

    # ------------------------------------------------------------------
    # real composited cascade on val images
    # ------------------------------------------------------------------
    rcan = build_backbone(resolve_checkpoint(args.checkpoint), device)
    base = Path(args.data_root) / "SRBenchmarks" / args.dataset_dir
    hr_dir = base / "HR"
    lr_dir = base / "LR_bicubic" / "X{}".format(args.scale)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    val_img_list = sorted(val_imgs)
    print("\n真实级联合成：{} 张 val 图 × {} 个阈值".format(
        len(val_img_list), len(thresholds)))

    # dense baseline per val image (quality + latency)
    dense_records = {}
    for name in val_img_list:
        hr_img = modcrop(cv2.imread(str(hr_dir / (name + ".png")), cv2.IMREAD_COLOR),
                         args.scale)
        lr_img = cv2.imread(str(lr_dir / "{}x{}.png".format(name, args.scale)),
                            cv2.IMREAD_COLOR)
        lq = torch.from_numpy(np.ascontiguousarray(
            np.transpose(lr_img[:, :, [2, 1, 0]].astype(np.float32), (2, 0, 1)))
        ).unsqueeze(0).to(device)  # BGR->RGB
        with torch.no_grad():
            for _ in range(2):
                rcan(lq)  # warmup
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            dense = rcan(lq)
            torch.cuda.synchronize(device)
            dense_ms = (time.perf_counter() - t0) * 1000.0
        dense_img = util.tensor2img(dense.detach()[0], out_type=np.uint8,
                                    min_max=(0, 255))
        dense_records[name] = {
            "hr": hr_img, "lr": lr_img, "dense_img": dense_img,
            "dense_psnr": psnr_y_full(dense_img, hr_img),
            "dense_ms": dense_ms,
        }
        del lq, dense
        torch.cuda.empty_cache()

    cascade_results = []
    for th in thresholds:
        rows = []
        for name in val_img_list:
            rec = dense_records[name]
            sr_img, timing = run_cascade_image(
                rcan, router, rec["lr"], th, args.patch, args.scale, device)
            casc_psnr = psnr_y_full(sr_img, rec["hr"])
            rows.append({
                "image": name,
                "dense_psnr": rec["dense_psnr"],
                "cascade_psnr": casc_psnr,
                "loss_db": rec["dense_psnr"] - casc_psnr,
                "dense_ms": rec["dense_ms"],
                **timing,
            })
        losses = [r["loss_db"] for r in rows]
        speedups = [r["dense_ms"] / r["total_ms"] for r in rows]
        cascade_results.append({
            "threshold": th,
            "mean_loss_db": float(np.mean(losses)),
            "worst_loss_db": float(np.max(losses)),
            "frac_images_loss_le_0.1": float(np.mean([l <= 0.1 for l in losses])),
            "mean_cheap_fraction": float(np.mean([r["cheap_fraction"] for r in rows])),
            "mean_total_ms": float(np.mean([r["total_ms"] for r in rows])),
            "mean_dense_ms": float(np.mean([r["dense_ms"] for r in rows])),
            "mean_speedup": float(np.mean(speedups)),
            "median_speedup": float(np.median(speedups)),
            "mean_router_ms": float(np.mean([r["router_ms"] for r in rows])),
            "rows": rows,
        })
        print("th={:.2f}: loss mean {:.4f} / worst {:.4f} dB | ≤0.1dB {:.0%} | "
              "cheap {:.1%} | speedup {:.3f}x (router {:.1f} ms)".format(
                  th, cascade_results[-1]["mean_loss_db"],
                  cascade_results[-1]["worst_loss_db"],
                  cascade_results[-1]["frac_images_loss_le_0.1"],
                  cascade_results[-1]["mean_cheap_fraction"],
                  cascade_results[-1]["mean_speedup"],
                  cascade_results[-1]["mean_router_ms"]))

    # G-router verdict: any threshold with >=95% images <=0.1dB and speedup>=1.5
    passing = [c for c in cascade_results
               if c["frac_images_loss_le_0.1"] >= 0.95 and c["mean_speedup"] >= 1.5]
    verdict = {
        "val_auc": best_auc,
        "auc_pass_0.88": best_auc >= 0.88,
        "cascade_pass": bool(passing),
        "passing_thresholds": [c["threshold"] for c in passing],
        "g_router": "PASS" if (best_auc >= 0.88 and passing) else "FAIL",
    }

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "results/router_train" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "width": args.width,
                "val_auc": best_auc, "eps": args.eps}, out_dir / "router.pth")

    report = {
        "generated": stamp,
        "labels": str(labels_path),
        "patches_train": int(len(tr_idx)), "patches_val": int(len(va_idx)),
        "val_images": len(val_img_list),
        "eps_db": args.eps, "epochs": args.epochs, "width": args.width,
        "best_val_auc": best_auc,
        "cascade": [{k: v for k, v in c.items() if k != "rows"}
                    for c in cascade_results],
        "verdict": verdict,
        "notes": [
            "按图划分 train/val，防重叠 patch 泄漏。",
            "级联延迟为端到端实测（含 router 前向与 CPU bicubic），CUDA 同步计时。",
            "边界余量区域以 bicubic 填充；未做 patch 接缝处理（后续消融项）。",
            "G-router 预注册: val AUC>=0.88 且存在阈值满足 95% 图损失<=0.1dB 且 speedup>=1.5x。",
        ],
    }
    (out_dir / "router_train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))

    lines = [
        "# Patch Router Training + Real Cascade (Stage 2)", "",
        "- labels: {} ({} train / {} val patches, {} val images)".format(
            labels_path.parent.name, len(tr_idx), len(va_idx), len(val_img_list)),
        "- best val AUC: **{:.4f}** (gate: >=0.88 → {})".format(
            best_auc, "PASS" if verdict["auc_pass_0.88"] else "FAIL"),
        "", "## Real composited cascade on val images", "",
        "| th | mean loss | worst loss | ≤0.1dB imgs | cheap % | ms/img | dense ms | speedup | router ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cascade_results:
        lines.append(
            "| {:.2f} | {:.4f} | {:.4f} | {:.0%} | {:.1%} | {:.1f} | {:.1f} | {:.3f}x | {:.1f} |".format(
                c["threshold"], c["mean_loss_db"], c["worst_loss_db"],
                c["frac_images_loss_le_0.1"], c["mean_cheap_fraction"],
                c["mean_total_ms"], c["mean_dense_ms"], c["mean_speedup"],
                c["mean_router_ms"]))
    lines.extend([
        "", "## G-router verdict: **{}**".format(verdict["g_router"]),
        "- passing thresholds: {}".format(verdict["passing_thresholds"] or "无"),
        "", "## Notes", ""] + ["- " + s for s in report["notes"]])
    (out_dir / "router_train_report.md").write_text("\n".join(lines))

    import tarfile
    archive = out_dir.parent / "B2RSR_ROUTER_TRAIN_{}.tar.gz".format(stamp)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    print("\n完成。请下载这一个文件供分析：\n{}".format(archive))
    print((out_dir / "router_train_report.md").read_text())


if __name__ == "__main__":
    main()
