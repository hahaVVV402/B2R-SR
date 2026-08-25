#!/usr/bin/env python3
"""Execute a YAML experiment plan through generic train/test entrypoints."""

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import options.options as option
from data.strict_paired import pair_directories, validate_pairs

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require_experiment_path(path):
    root = EXPERIMENTS.resolve()
    path = Path(path).resolve()
    if root != EXPERIMENTS or root not in path.parents:
        raise ValueError('Plan output must be below the non-symlink repository experiments/')
    return path


def same_yaml(path, value):
    if path.is_file():
        existing = option.load(str(path))
        if json.dumps(existing, sort_keys=True) != json.dumps(value, sort_keys=True):
            raise RuntimeError('Existing resolved option differs: {}'.format(path))
    else:
        option.dump(value, str(path))


def run_command(arguments, dry_run=False):
    print('+ ' + ' '.join(str(value) for value in arguments), flush=True)
    if not dry_run:
        subprocess.run([str(value) for value in arguments], cwd=str(ROOT), check=True)


def seed_statistics(values):
    mean = statistics.mean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = 4.302652729911275 if len(values) == 3 else 1.96
    half_width = critical * sample_std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {'mean': mean, 'sample_std': sample_std, 'ci95_half_width_t': half_width}


def aggregate(output_root, completed):
    rows = []
    validation_rows = []
    for item in completed:
        train_report = json.load((item['experiment'] / 'train_report.json').open(
            'r', encoding='utf-8'))
        validation_rows.append({
            'scale': item['scale'], 'seed': item['seed'],
            'best_step': train_report['best_validation']['step'],
            'psnr_y_db': train_report['best_validation']['mean_psnr_y_db'],
            'ssim_y': train_report['best_validation']['mean_ssim_y'],
            'checkpoint_sha256': train_report['checkpoints']['best_val']['sha256'],
        })
        summary = json.load((item['experiment'] / 'test' / 'summary.json').open(
            'r', encoding='utf-8'))
        for dataset, metrics in summary['datasets'].items():
            rows.append({'scale': item['scale'], 'seed': item['seed'],
                         'dataset': dataset,
                         'psnr_y_db': metrics['mean_psnr_y_db'],
                         'ssim_y': metrics['mean_ssim_y'],
                         'checkpoint_sha256': summary['checkpoint']['sha256']})
    aggregate_dir = output_root / 'aggregate'
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    with (aggregate_dir / 'validation_summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            'scale', 'seed', 'best_step', 'psnr_y_db', 'ssim_y', 'checkpoint_sha256'),
            lineterminator='\n')
        writer.writeheader()
        writer.writerows(validation_rows)
    with (aggregate_dir / 'test_summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            'scale', 'seed', 'dataset', 'psnr_y_db', 'ssim_y', 'checkpoint_sha256'),
            lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    across = {}
    for scale in sorted({row['scale'] for row in rows}):
        across[str(scale)] = {}
        for dataset in sorted({row['dataset'] for row in rows if row['scale'] == scale}):
            selected = [row for row in rows
                        if row['scale'] == scale and row['dataset'] == dataset]
            across[str(scale)][dataset] = {
                'psnr_y_db': seed_statistics([float(row['psnr_y_db']) for row in selected]),
                'ssim_y': seed_statistics([float(row['ssim_y']) for row in selected]),
            }
    validation_across = {}
    for scale in sorted({row['scale'] for row in validation_rows}):
        selected = [row for row in validation_rows if row['scale'] == scale]
        validation_across[str(scale)] = {
            'psnr_y_db': seed_statistics([float(row['psnr_y_db']) for row in selected]),
            'ssim_y': seed_statistics([float(row['ssim_y']) for row in selected]),
        }
    payload = {'status': 'complete', 'runs': len(completed), 'rows': rows,
               'across_seeds': across, 'validation_rows': validation_rows,
               'validation_across_seeds': validation_across}
    target = aggregate_dir / 'test_summary.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n',
                         encoding='utf-8')
    os.replace(str(temporary), str(target))
    return payload


def preflight(plan_path, phase='training'):
    if phase not in ('training', 'test'):
        raise ValueError('Preflight phase must be training or test')
    plan = option.load(plan_path)
    reports = []
    seen = set()
    selected_configs = (('train', 'train_opt'),) if phase == 'training' else (('test', 'test_opt'),)
    for configured in plan['runs']:
        scale = int(configured['scale'])
        for kind, key in selected_configs:
            config = option.load(str(resolve(configured[key])))
            for dataset_phase, dataset in config['datasets'].items():
                identity = (scale, dataset['dataroot_GT'], dataset['dataroot_LQ'],
                            tuple(dataset.get('id_range') or ()), dataset.get('max_images'))
                if identity in seen:
                    continue
                seen.add(identity)
                training = kind == 'train' and dataset_phase == 'train'
                pairs = pair_directories(
                    resolve(dataset['dataroot_GT']), resolve(dataset['dataroot_LQ']),
                    scale, dataset.get('id_range') if training else None)
                expected = dataset.get('expected_count')
                if expected is not None and not dataset.get('max_images') and len(pairs) != int(expected):
                    raise RuntimeError('{} count is {}, expected {}'.format(
                        dataset['name'], len(pairs), expected))
                minimum = int(dataset.get('minimum_pairs') or 0)
                if len(pairs) < minimum:
                    raise RuntimeError('{} has {} pairs, requires {}'.format(
                        dataset['name'], len(pairs), minimum))
                max_images = int(dataset.get('max_images') or 0)
                selected = pairs[:max_images] if max_images else pairs
                geometry = validate_pairs(
                    selected, scale,
                    int(dataset.get('lr_patch_size') or 0) if training else 0,
                    allow_modcrop=not training,
                    max_pairs=int(dataset.get('preflight_max_pairs') or 0))
                reports.append({'scale': scale, 'kind': kind, 'phase': dataset_phase,
                                'dataset': dataset['name'], 'pairs': len(selected),
                                'geometry': geometry})
    print(json.dumps({'status': 'preflight_complete', 'phase': phase,
                      'datasets': reports}, indent=2, sort_keys=True))
    return {'status': 'preflight_complete', 'phase': phase, 'runs': len(plan['runs'])}


def execute(plan_path, dry_run=False):
    plan = option.load(plan_path)
    execution = plan.get('execution') or {}
    skip_test = bool(execution.get('skip_test', False))
    if not skip_test and not bool(execution.get('train_all_before_test', True)):
        raise ValueError('Formal plan must train every run before testing')
    output_root = require_experiment_path(resolve(plan['output_root']))
    output_root.mkdir(parents=True, exist_ok=True)
    same_yaml(output_root / 'run_plan.resolved.yml', plan)
    completed = []
    total_runs = sum(len(configured['seeds']) for configured in plan['runs'])
    run_index = 0

    for configured in plan['runs']:
        scale = int(configured['scale'])
        train_template = option.load(str(resolve(configured['train_opt'])))
        for seed_value in configured['seeds']:
            run_index += 1
            seed = int(seed_value)
            print('==== [train {}/{}] X{} seed{} ===='.format(
                run_index, total_runs, scale, seed), flush=True)
            run_name = 'x{}_seed{}'.format(scale, seed)
            experiment = require_experiment_path(output_root / run_name)
            experiment.mkdir(parents=True, exist_ok=True)
            train_opt = json.loads(json.dumps(train_template))
            if int(train_opt['scale']) != scale:
                raise ValueError('{} scale differs from plan'.format(configured['train_opt']))
            train_opt['name'] = '{}_{}'.format(plan['name'], run_name)
            train_opt['train']['manual_seed'] = seed
            train_opt.setdefault('path', {})['experiment_dir'] = str(experiment)
            resolved_train = experiment / 'train_config.resolved.yml'
            same_yaml(resolved_train, train_opt)
            run_command((sys.executable, ROOT / 'codes/train.py', '-opt', resolved_train),
                        dry_run=dry_run)
            completed.append({'scale': scale, 'seed': seed,
                              'experiment': experiment,
                              'test_opt': configured.get('test_opt')})

    if dry_run:
        return {'status': 'dry_run', 'runs': len(completed)}

    for item in completed:
        report_path = item['experiment'] / 'train_report.json'
        if not report_path.is_file():
            raise RuntimeError('Missing completed train report: {}'.format(report_path))
        report = json.load(report_path.open('r', encoding='utf-8'))
        if report.get('status') != 'complete':
            raise RuntimeError('Training is not complete: {}'.format(report_path))
        checkpoint = item['experiment'] / 'models' / 'best_val.pt'
        if not checkpoint.is_file():
            raise RuntimeError('Missing best-validation checkpoint: {}'.format(checkpoint))
        recorded = report['checkpoints']['best_val']['sha256']
        observed = sha256_file(checkpoint)
        if observed != recorded:
            raise RuntimeError('Best-validation checkpoint hash changed: {} != {}'.format(
                observed, recorded))
        item['checkpoint_sha256'] = observed

    if skip_test:
        # Validation-only study: no final benchmark directory is ever opened.
        report = {'status': 'complete', 'name': plan['name'],
                  'run_count': len(completed), 'skip_test': True,
                  'runs': [{'scale': item['scale'], 'seed': item['seed'],
                            'experiment': str(item['experiment']),
                            'best_val_sha256': item['checkpoint_sha256']}
                           for item in completed]}
        (output_root / 'run_report.json').write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        return report

    # Benchmark files are first opened only after every selected checkpoint is frozen.
    preflight(plan_path, phase='test')

    for test_index, item in enumerate(completed, 1):
        print('==== [test {}/{}] X{} seed{} ===='.format(
            test_index, total_runs, item['scale'], item['seed']), flush=True)
        test_opt = option.load(str(resolve(item['test_opt'])))
        if int(test_opt['scale']) != item['scale']:
            raise ValueError('{} scale differs from plan'.format(item['test_opt']))
        test_opt['name'] = '{}_x{}_seed{}_test'.format(
            plan['name'], item['scale'], item['seed'])
        test_opt.setdefault('path', {})['experiment_dir'] = str(item['experiment'])
        test_opt['path']['pretrain_model_G'] = str(
            item['experiment'] / 'models' / 'best_val.pt')
        resolved_test = item['experiment'] / 'test_config.resolved.yml'
        same_yaml(resolved_test, test_opt)
        run_command((sys.executable, ROOT / 'codes/test.py', '-opt', resolved_test))
        test_summary = json.load((item['experiment'] / 'test' / 'summary.json').open(
            'r', encoding='utf-8'))
        after_test_hash = sha256_file(item['experiment'] / 'models' / 'best_val.pt')
        if (test_summary['checkpoint']['sha256'] != item['checkpoint_sha256']
                or after_test_hash != item['checkpoint_sha256']):
            raise RuntimeError('Selected checkpoint changed before, during, or after testing')

    payload = aggregate(output_root, completed)
    report = {'status': 'complete', 'name': plan['name'],
              'run_count': len(completed),
              'aggregate': str(output_root / 'aggregate' / 'test_summary.json')}
    (output_root / 'run_report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-opt', required=True, help='Run-plan YAML')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.dry_run and args.preflight:
        parser.error('--dry-run and --preflight are mutually exclusive')
    result = preflight(args.opt, phase='training') if args.preflight else execute(args.opt, dry_run=args.dry_run)
    print(json.dumps({'status': result['status'],
                      'runs': result.get('runs', result.get('run_count'))}, sort_keys=True))


if __name__ == '__main__':
    main()
