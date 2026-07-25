#!/usr/bin/env python3
"""Benchmark B2R-SR training geometry on one CUDA GPU without validation or saving."""

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

import torch

import options.options as option
from models import create_model


def parse_cases(value):
    cases = []
    for item in value.split(','):
        gt, batch = item.lower().split('x', 1)
        gt, batch = int(gt), int(batch)
        if gt <= 0 or batch < 2:
            raise argparse.ArgumentTypeError(
                'cases must use positive GT sizes and batch >= 2, e.g. 192x8')
        cases.append((gt, batch))
    return cases


def mib(value):
    return value / 1024 ** 2


def benchmark_case(model, scale, gt_size, batch_size, phase, hard_train_after,
                   warmup, steps, chunk_size):
    model.optimizer_G.zero_grad(set_to_none=True)
    for attr in ('var_L', 'real_H', 'fake_H', 'plugin_info'):
        if hasattr(model, attr):
            setattr(model, attr, None)
    gc.collect()
    torch.cuda.empty_cache()

    lq_size = gt_size // scale
    lq = torch.rand(batch_size, 3, lq_size, lq_size)
    gt = torch.rand(batch_size, 3, gt_size, gt_size)
    step_base = 1 if phase == 'soft' else hard_train_after + 1

    for i in range(warmup):
        model.feed_data({'LQ': lq, 'GT': gt})
        model.optimize_parameters(step_base + i)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    chunk_times = []
    completed = 0
    wall_start = time.perf_counter()
    while completed < steps:
        count = min(chunk_size, steps - completed)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for i in range(count):
            model.feed_data({'LQ': lq, 'GT': gt})
            model.optimize_parameters(step_base + warmup + completed + i)
        torch.cuda.synchronize()
        chunk_times.append((time.perf_counter() - started) / count)
        completed += count
    wall_seconds = time.perf_counter() - wall_start

    step_ms = statistics.median(chunk_times) * 1000
    p90_ms = sorted(chunk_times)[min(len(chunk_times) - 1, int(len(chunk_times) * 0.9))] * 1000
    images_s = batch_size * steps / wall_seconds
    lr_megapixels_s = images_s * lq_size * lq_size / 1e6
    return {
        'step_ms': step_ms,
        'p90_ms': p90_ms,
        'images_s': images_s,
        'lr_mpix_s': lr_megapixels_s,
        'allocated_mib': mib(torch.cuda.max_memory_allocated()),
        'reserved_mib': mib(torch.cuda.max_memory_reserved()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt', default='options/train/train_B2RSR_RCAN_X4.yml',
        help='training YAML, relative to codes/')
    parser.add_argument(
        '--cases', type=parse_cases,
        help='comma-separated GT-size x batch-size cases; defaults target LR 24/32/48')
    parser.add_argument('--phase', choices=('soft', 'hard'), default='soft')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--chunk-size', type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this benchmark.')

    os.chdir(Path(__file__).resolve().parent)
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = False
    scale = int(opt['scale'])
    route_window = int(opt['network_G']['plugin']['route_window'])
    hard_train_after = int(opt['network_G']['plugin']['hard_train_after'])
    cases = args.cases or [
        (24 * scale, 16), (24 * scale, 32), (32 * scale, 16),
        (48 * scale, 8), (48 * scale, 12), (48 * scale, 16)]
    for gt_size, _ in cases:
        if gt_size % scale:
            raise SystemExit('GT size {} is not divisible by scale {}.'.format(gt_size, scale))
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    print('\nscale=X{} phase={} warmup={} measured_steps={}'.format(
        scale, args.phase, args.warmup, args.steps))
    print('GT   LR   batch  windows  step_ms  p90_ms  images/s  LR-Mpix/s  alloc_MiB  reserve_MiB')
    print('-' * 94)
    for gt_size, batch_size in cases:
        lq_size = gt_size // scale
        windows = ((lq_size + route_window - 1) // route_window) ** 2
        try:
            result = benchmark_case(
                model, scale, gt_size, batch_size, args.phase, hard_train_after,
                args.warmup, args.steps, args.chunk_size)
            print(
                '{:<4d} {:<4d} {:<6d} {:<8d} {:>7.1f} {:>7.1f} {:>9.2f} {:>10.2f} '
                '{:>10.0f} {:>11.0f}'.format(
                    gt_size, lq_size, batch_size, windows,
                    result['step_ms'], result['p90_ms'], result['images_s'],
                    result['lr_mpix_s'], result['allocated_mib'], result['reserved_mib']))
        except torch.cuda.OutOfMemoryError:
            print('{:<4d} {:<4d} {:<6d} {:<8d} OOM'.format(
                gt_size, lq_size, batch_size, windows))
            model.optimizer_G.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

    print('\nSelect a case within 95% of the best throughput, with reserved memory <= 22 GiB.')
    print('Then benchmark that case again with --phase hard.')


if __name__ == '__main__':
    main()
