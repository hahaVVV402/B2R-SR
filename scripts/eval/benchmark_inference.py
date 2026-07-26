#!/usr/bin/env python3
"""Repeatable forward-pass latency benchmark for dense RCAN or B2R-SR configs."""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codes"))

import options.options as option  # noqa: E402
from data import create_dataloader, create_dataset  # noqa: E402
from models import create_model  # noqa: E402


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark_forward(net, x, warmup, runs):
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

        timings = []
        for _ in range(runs):
            if x.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                net(x)
                end.record()
                end.synchronize()
                timings.append(float(start.elapsed_time(end)))
            else:
                start = time.perf_counter()
                net(x)
                timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", required=True, help="Test YAML path")
    parser.add_argument("--checkpoint", help="Override path.pretrain_model_G")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=5,
                        help="Images per dataset; 0 uses the complete dataset")
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1 or args.max_images < 0:
        parser.error("warmup >= 0, runs >= 1, max-images >= 0")

    parsed_opt = option.parse(args.opt, is_train=False)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            parser.error("checkpoint does not exist: {}".format(checkpoint))
        parsed_opt["path"]["pretrain_model_G"] = str(checkpoint)
    opt = option.dict_to_nonedict(parsed_opt)
    loaders = []
    for _, dataset_opt in sorted(opt["datasets"].items()):
        dataset = create_dataset(dataset_opt)
        loaders.append(create_dataloader(dataset, dataset_opt))

    model = create_model(opt)
    net = model.netG
    net.eval()
    module = net.module if hasattr(net, "module") else net
    if hasattr(module, "sync_latency"):
        module.sync_latency = False

    print("config={}".format(Path(args.opt).resolve()))
    print("device={} warmup={} runs={} max_images={}".format(
        model.device, args.warmup, args.runs, args.max_images))

    for loader in loaders:
        all_timings = []
        tested = 0
        for data in loader:
            model.feed_data(data, need_GT=False)
            all_timings.extend(benchmark_forward(
                net, model.var_L, args.warmup, args.runs))
            tested += 1
            if args.max_images and tested >= args.max_images:
                break

        print(
            "{:<12s} images={:<3d} median={:8.3f} ms mean={:8.3f} ms "
            "p90={:8.3f} ms std={:8.3f} ms".format(
                loader.dataset.opt["name"], tested,
                statistics.median(all_timings), statistics.mean(all_timings),
                percentile(all_timings, 90), statistics.pstdev(all_timings)))


if __name__ == "__main__":
    main()
