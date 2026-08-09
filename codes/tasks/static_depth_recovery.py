"""Robust YAML-driven static-depth recovery and SR evaluation backend."""

import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from data.strict_paired import (DeterministicBatchPrefetcher, pair_directories,
                                read_pair, sample_batch, validate_pairs)
from models import networks
from models.archs.EDSR_arch import (EDSR, strict_load, tensor_state,
                                    transplant_edsr, uniform_endpoint_indices)
from options import options as option
from utils.sr_metrics import psnr_y, quantize_rgb, ssim_y

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / 'experiments'
FORBIDDEN_DEPLOYMENT_TERMS = ('teacher', 'router', 'mask', 'keep_map', 'keepmap',
                              'source_index', 'mapping', 'scheduler', 'dynamic')


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)


def percentile(values, value):
    return float(np.percentile(values, value)) if values else None


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(json.dumps(list(tensor.shape), separators=(',', ':')).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(str(Path(path)), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, str(temporary))
    with temporary.open('r+b') as handle:
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)


def torch_load_weights(path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def torch_load_resume(path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def resolve_repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def experiment_directory(opt):
    root = EXPERIMENTS_ROOT.resolve()
    if root != EXPERIMENTS_ROOT:
        raise RuntimeError('Repository experiments/ must not be a symlink')
    configured = (opt.get('path') or {}).get('experiment_dir')
    path = resolve_repo_path(configured) if configured else (root / str(opt['name'])).resolve()
    if root not in path.parents:
        raise ValueError('Experiment directory must be below {}'.format(root))
    return path


def resolved_config(opt, path):
    path = Path(path)
    digest = config_sha256(opt)
    if path.is_file():
        observed = option.load(str(path))
        if config_sha256(observed) != digest:
            raise RuntimeError('Existing resolved config differs: {}'.format(path))
    else:
        option.dump(opt, str(path))
    return digest


def configure_reproducibility(seed, reproducibility):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(reproducibility.get('cudnn_benchmark', False))
    torch.backends.cudnn.deterministic = bool(reproducibility.get('cudnn_deterministic', True))
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = bool(reproducibility.get('tf32', False))
    torch.backends.cudnn.allow_tf32 = bool(reproducibility.get('tf32', False))


def environment_report(device):
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'torch': torch.__version__, 'device': str(device)}
    if device.type == 'cuda':
        report.update({'device_name': torch.cuda.get_device_name(device),
                       'torch_cuda': torch.version.cuda,
                       'cudnn': torch.backends.cudnn.version()})
    return report


def validate_checkpoint(path, spec):
    path = resolve_repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_bytes = path.stat().st_size
    observed_sha = sha256_file(path)
    expected_bytes = int(spec['bytes'])
    expected_sha = str(spec['sha256'])
    if observed_bytes != expected_bytes or observed_sha != expected_sha:
        raise RuntimeError('Checkpoint provenance mismatch for {}: bytes {}/{}, sha {}/{}'.format(
            path, observed_bytes, expected_bytes, observed_sha, expected_sha))
    state = tensor_state(torch_load_weights(path))
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise FloatingPointError('Checkpoint contains a non-finite tensor')
    return state, {'path': str(path), 'bytes': observed_bytes, 'sha256': observed_sha}


def _dataset_pairs(dataset, scale, training=False):
    pairs = pair_directories(
        resolve_repo_path(dataset['dataroot_GT']),
        resolve_repo_path(dataset['dataroot_LQ']),
        scale,
        dataset.get('id_range') if training else None,
    )
    expected = dataset.get('expected_count')
    if expected is not None and not dataset.get('max_images') and len(pairs) != int(expected):
        raise RuntimeError('{} count is {}, expected {}'.format(
            dataset['name'], len(pairs), expected))
    minimum = int(dataset.get('minimum_pairs') or 0)
    if len(pairs) < minimum:
        raise RuntimeError('{} has {} pairs, requires at least {}'.format(
            dataset['name'], len(pairs), minimum))
    max_images = int(dataset.get('max_images') or 0)
    return pairs[:max_images] if max_images else pairs


def evaluate_model(model, pairs, scale, device, save_images=None):
    was_training = model.training
    model.eval()
    records = []
    if save_images is not None:
        Path(save_images).mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for key, hr_path, lr_path in pairs:
            hr, lr = read_pair(hr_path, lr_path, scale, allow_modcrop=True)
            tensor = torch.from_numpy(np.ascontiguousarray(
                lr.transpose(2, 0, 1))).float().unsqueeze(0).to(device)
            raw = model(tensor)
            if not bool(torch.isfinite(raw).all()):
                raise FloatingPointError('Non-finite evaluation output for {}'.format(key))
            output = quantize_rgb(raw)
            if output.shape != hr.shape:
                raise AssertionError('Output shape mismatch for {}: {} != {}'.format(
                    key, output.shape, hr.shape))
            record = {'image_id': key,
                      'psnr_y_db': psnr_y(output, hr, scale),
                      'ssim_y': ssim_y(output, hr, scale)}
            records.append(record)
            if save_images is not None:
                target = Path(save_images) / '{}.png'.format(key)
                if not cv2.imwrite(str(target), cv2.cvtColor(output, cv2.COLOR_RGB2BGR)):
                    raise OSError('Failed to save {}'.format(target))
            del tensor, raw, output
    model.train(was_training)
    return {'count': len(records),
            'mean_psnr_y_db': statistics.mean(row['psnr_y_db'] for row in records),
            'mean_ssim_y': statistics.mean(row['ssim_y'] for row in records),
            'per_image': records}


def _write_records(path, records):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)


def _write_csv(path, fieldnames, records):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)


def _log(experiment, message):
    line = '{} {}'.format(utc_now(), message)
    print(line, flush=True)
    with (Path(experiment) / 'train.log').open('a', encoding='utf-8', buffering=1) as handle:
        handle.write(line + '\n')
        handle.flush()


def _state_cpu(model):
    return OrderedDict((key, value.detach().cpu().clone())
                       for key, value in model.state_dict().items())


def _static_audit(model):
    module_types = {module.__class__.__name__ for module in model.modules()}
    state = model.state_dict()
    forbidden = [key for key in state
                 if any(term in key.lower() for term in FORBIDDEN_DEPLOYMENT_TERMS)]
    return {'module_types': sorted(module_types), 'tensor_count': len(state),
            'forbidden_state_keys': forbidden, 'pass': not forbidden}


def _truncate_trace(path, completed_step):
    path = Path(path)
    records = []
    if path.is_file():
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if int(row['step']) <= completed_step:
                        records.append(row)
    if [int(row['step']) for row in records] != list(range(1, completed_step + 1)):
        raise RuntimeError('Training trace is inconsistent with resume step {}'.format(completed_step))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    _fsync_directory(path.parent)
    return records


def _validation(model, pairs, scale, device, val_dir, step):
    result = evaluate_model(model, pairs, scale, device)
    evidence = val_dir / 'step_{:06d}.jsonl'.format(step)
    _write_records(evidence, result.pop('per_image'))
    result.update({'step': step, 'per_image_path': str(evidence),
                   'per_image_sha256': sha256_file(evidence), 'created_at_utc': utc_now()})
    with (val_dir / 'history.jsonl').open('a', encoding='utf-8', buffering=1) as handle:
        handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return result


def _truncate_validation(val_dir, completed_step):
    history = val_dir / 'history.jsonl'
    rows = []
    if history.is_file():
        with history.open('r', encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows = [row for row in rows if int(row['step']) <= completed_step]
        temporary = history.with_suffix('.jsonl.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(history))
        _fsync_directory(val_dir)
    for evidence in val_dir.glob('step_*.jsonl'):
        match = evidence.stem.split('_')[-1]
        if match.isdigit() and int(match) > completed_step:
            evidence.unlink()
    return rows


def _restore_resume_best(resume, best_path):
    if resume.get('best_student') is None:
        raise RuntimeError('Resume state does not contain the selected best checkpoint')
    best_state = tensor_state(resume['best_student'])
    atomic_torch_save(best_state, best_path)
    return best_state


def _save_resume(path, step, config_hash, student, optimizer, scheduler,
                 sampling_rng_state, best_score, best_step, best_state):
    atomic_torch_save({
        'step': step,
        'config_sha256': config_hash,
        'student': _state_cpu(student),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'sampling_rng_state': sampling_rng_state,
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        'best_score': best_score,
        'best_step': best_step,
        'best_student': best_state,
    }, path)


def train_from_options(option_path):
    opt = option.load(option_path)
    if opt.get('model') != 'static_depth_recovery':
        raise ValueError('Static backend received model={}'.format(opt.get('model')))
    if not torch.cuda.is_available():
        raise RuntimeError('Static-depth recovery requires CUDA')
    scale = int(opt['scale'])
    seed = int(opt['train']['manual_seed'])
    experiment = experiment_directory(opt)
    models_dir = experiment / 'models'
    state_dir = experiment / 'training_state'
    val_dir = experiment / 'val'
    for directory in (experiment, models_dir, state_dir, val_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config_hash = resolved_config(opt, experiment / 'train_config.resolved.yml')
    report_path = experiment / 'train_report.json'
    best_path, last_path = models_dir / 'best_val.pt', models_dir / 'last.pt'
    if report_path.is_file():
        report = json.load(report_path.open('r', encoding='utf-8'))
        if (report.get('status') == 'complete' and report.get('config_sha256') == config_hash
                and best_path.is_file() and last_path.is_file()):
            print(json.dumps({'status': 'already_complete', 'experiment': str(experiment)}))
            return report
        raise RuntimeError('Existing training report is incomplete or inconsistent: {}'.format(report_path))

    reproducibility = opt.get('reproducibility') or {}
    if str(opt['train'].get('precision', 'fp32')).lower() != 'fp32' or reproducibility.get('amp', False):
        raise ValueError('Formal static-depth recovery requires FP32 without AMP')
    if not bool(opt['validation'].get('enabled', True)):
        raise ValueError('Validation must be enabled for best-val selection')
    if opt['validation'].get('monitor') != 'psnr_y' or opt['validation'].get('mode') != 'max':
        raise ValueError('Checkpoint selection must use maximum validation PSNR-Y')
    configure_reproducibility(seed, reproducibility)
    device = torch.device('cuda:0')
    student_opt, teacher_opt = opt['network_G'], opt['teacher']
    if student_opt['which_model_G'] != 'EDSR' or teacher_opt['which_model_G'] != 'EDSR':
        raise ValueError('Static-depth recovery currently requires canonical EDSR')
    teacher_state, teacher_checkpoint = validate_checkpoint(
        (opt.get('path') or {})['teacher_checkpoint'], teacher_opt['checkpoint'])
    teacher = EDSR(teacher_opt['n_resblocks'], teacher_opt['n_feats'],
                   teacher_opt['res_scale'], teacher_opt.get('n_colors', 3),
                   teacher_opt.get('rgb_range', 255), scale)
    strict_load(teacher, teacher_state)
    student, initial_state, source_indices = transplant_edsr(
        teacher_state, scale, int(teacher_opt['n_resblocks']),
        int(student_opt['n_resblocks']), int(student_opt['n_feats']),
        float(student_opt['res_scale']), int(student_opt.get('rgb_range', 255)),
        (opt.get('initialization') or {}).get('block_mapping', 'uniform_endpoints'))
    audit = _static_audit(student)
    if not audit['pass']:
        raise AssertionError('Static Student audit failed: {}'.format(audit))

    train_data = opt['datasets']['train']
    val_data = opt['datasets']['val']
    train_pairs = _dataset_pairs(train_data, scale, training=True)
    val_pairs = _dataset_pairs(val_data, scale)
    validate_pairs(train_pairs, scale, int(train_data['lr_patch_size']), False,
                   int(train_data.get('preflight_max_pairs') or 0))
    validate_pairs(val_pairs, scale, 0, True,
                   int(val_data.get('preflight_max_pairs') or 0))

    teacher = teacher.eval().float().to(device)
    student = student.train().float().to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer_opt = opt['train']['optimizer']
    if optimizer_opt['type'].lower() != 'adam':
        raise ValueError('Only Adam is supported for formal recovery')
    if str((opt['train'].get('scheduler') or {}).get('type', 'cosine')).lower() != 'cosine':
        raise ValueError('Only niter-bound cosine decay is supported')
    optimizer = torch.optim.Adam(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=float(optimizer_opt['lr']),
        betas=tuple(float(value) for value in optimizer_opt.get('betas', [0.9, 0.999])),
        eps=float(optimizer_opt.get('eps', 1e-8)),
        weight_decay=float(optimizer_opt.get('weight_decay', 0)),
    )
    niter = int(opt['train']['niter'])
    invocation_stop = min(niter, int(os.environ.get('B2RSR_STOP_AFTER_STEP', niter)))
    if invocation_stop < 1:
        raise ValueError('B2RSR_STOP_AFTER_STEP must be positive')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=niter,
        eta_min=float((opt['train'].get('scheduler') or {}).get('eta_min', 0)))
    rng = random.Random(seed)
    resume_path = state_dir / 'resume.pt'
    trace_path = experiment / 'train_trace.jsonl'
    completed_step, best_score, best_step, best_state = 0, -math.inf, None, None
    if resume_path.is_file():
        resume = torch_load_resume(resume_path)
        if resume.get('config_sha256') != config_hash:
            raise RuntimeError('Resume state does not match resolved config')
        completed_step = int(resume['step'])
        strict_load(student, tensor_state(resume['student']))
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        rng.setstate(resume['sampling_rng_state'])
        random.setstate(resume['python_rng_state'])
        np.random.set_state(resume['numpy_rng_state'])
        torch.set_rng_state(resume['torch_rng_state'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume['cuda_rng_state'])
        best_score = float(resume['best_score'])
        best_step = resume['best_step']
        best_state = _restore_resume_best(resume, best_path)
    trace_records = _truncate_trace(trace_path, completed_step)
    validation_rows = _truncate_validation(val_dir, completed_step)
    if completed_step == 0 and best_score == -math.inf and best_path.is_file():
        initial_rows = [row for row in validation_rows if int(row['step']) == 0]
        if len(initial_rows) == 1:
            best_score = float(initial_rows[0]['mean_psnr_y_db'])
            best_step = 0
            best_state = tensor_state(torch_load_weights(best_path))

    validation_opt = opt['validation']
    validation_interval = int(validation_opt['interval'])
    resume_interval = int(opt['checkpoint']['rolling_resume_interval'])
    teacher_weight = float(opt['train']['loss']['teacher_l1']['weight'])
    ground_truth_weight = float(opt['train']['loss']['ground_truth_l1']['weight'])
    batch_size = int(train_data['batch_size'])
    patch = int(train_data['lr_patch_size'])
    prefetch_batches = int(opt['train'].get('prefetch_batches', 0))
    if prefetch_batches not in (0, 1):
        raise ValueError('train.prefetch_batches must be 0 or 1')
    started_at = utc_now()
    wall_start = time.time()
    _log(experiment, 'start/resume x{} seed{} at step {}/{} prefetch={}'.format(
        scale, seed, completed_step, niter, prefetch_batches))
    timings = [float(row.get('train_step_wall_ms',
                             row.get('step_wall_ms', row.get('gpu_step_ms'))))
               for row in trace_records]
    input_timings = [float(row['input_wait_ms']) for row in trace_records
                     if 'input_wait_ms' in row]
    h2d_timings = [float(row['h2d_ms']) for row in trace_records if 'h2d_ms' in row]
    gpu_timings = [float(row['gpu_step_ms']) for row in trace_records
                   if 'gpu_step_ms' in row]
    committed_sampling_rng_state = rng.getstate()
    failure_path = experiment / 'failure.json'
    try:
        torch.cuda.reset_peak_memory_stats()
        if (completed_step == 0 and not any(int(row['step']) == 0 for row in validation_rows)
                and bool(validation_opt.get('evaluate_initial', True))):
            initial_validation = _validation(student, val_pairs, scale, device, val_dir, 0)
            best_score = float(initial_validation['mean_psnr_y_db'])
            best_step = 0
            best_state = _state_cpu(student)
            atomic_torch_save(best_state, best_path)
            _log(experiment, 'initial validation PSNR-Y={:.6f} SSIM-Y={:.6f}'.format(
                best_score, float(initial_validation['mean_ssim_y'])))
            _save_resume(resume_path, 0, config_hash, student, optimizer, scheduler,
                         committed_sampling_rng_state, best_score, best_step, best_state)
        def load_batch(batch_rng):
            return sample_batch(train_pairs, scale, batch_size, patch, batch_rng)

        batches = DeterministicBatchPrefetcher(load_batch, rng) if prefetch_batches else None
        if batches is not None:
            batches.__enter__()
        try:
            with trace_path.open('a', encoding='utf-8', buffering=1024 * 1024) as trace:
                for step in range(completed_step + 1, invocation_stop + 1):
                    step_started = time.perf_counter()
                    if batches is None:
                        batch = load_batch(rng)
                        committed_sampling_rng_state = rng.getstate()
                    else:
                        batch, committed_sampling_rng_state = batches.get()
                    lr_cpu, hr_cpu, sample_keys = batch
                    input_wait_ms = (time.perf_counter() - step_started) * 1000.0
                    h2d_started = time.perf_counter()
                    lr, hr = lr_cpu.to(device), hr_cpu.to(device)
                    h2d_ms = (time.perf_counter() - h2d_started) * 1000.0
                    gpu_started = time.perf_counter()
                    used_lr = float(optimizer.param_groups[0]['lr'])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        teacher_output = teacher(lr)
                    student_output = student(lr)
                    teacher_l1 = torch.nn.functional.l1_loss(student_output, teacher_output)
                    gt_l1 = torch.nn.functional.l1_loss(student_output, hr)
                    total_loss = teacher_weight * teacher_l1 + ground_truth_weight * gt_l1
                    if not bool(torch.isfinite(total_loss)):
                        raise FloatingPointError('Non-finite loss at step {}'.format(step))
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    gpu_step_ms = (time.perf_counter() - gpu_started) * 1000.0
                    train_step_ms = (time.perf_counter() - step_started) * 1000.0
                    timings.append(train_step_ms)
                    input_timings.append(input_wait_ms)
                    h2d_timings.append(h2d_ms)
                    gpu_timings.append(gpu_step_ms)
                    completed_step = step
                    row = {'step': step, 'sample_keys': sample_keys,
                           'total_loss': float(total_loss.detach()),
                           'teacher_l1': float(teacher_l1.detach()),
                           'gt_l1': float(gt_l1.detach()),
                           'learning_rate': used_lr,
                           'input_wait_ms': input_wait_ms,
                           'h2d_ms': h2d_ms,
                           'gpu_step_ms': gpu_step_ms,
                           'train_step_wall_ms': train_step_ms,
                           'step_wall_ms': train_step_ms}
                    trace.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                    if (step % int((opt.get('logger') or {}).get('print_freq', 100)) == 0
                            or step == invocation_stop):
                        recent_steps = timings[-100:]
                        median_step = statistics.median(recent_steps)
                        images_per_second = batch_size * 1000.0 / median_step
                        eta_seconds = (niter - step) * statistics.mean(recent_steps) / 1000.0
                        _log(experiment, ('step {}/{} ({:.2f}%) loss={:.6f} teacher_l1={:.6f} '
                                          'gt_l1={:.6f} lr={:.3e} input={:.1f}ms gpu={:.1f}ms '
                                          'step={:.1f}ms throughput={:.2f}img/s train_eta={}').format(
                                              step, niter, 100.0 * step / niter,
                                              row['total_loss'], row['teacher_l1'],
                                              row['gt_l1'], used_lr,
                                              statistics.median(input_timings[-100:]),
                                              statistics.median(gpu_timings[-100:]),
                                              median_step, images_per_second,
                                              format_duration(eta_seconds)))
                    validation_due = step % validation_interval == 0 or step == niter
                    if validation_due:
                        result = _validation(student, val_pairs, scale, device, val_dir, step)
                        score = float(result['mean_psnr_y_db'])
                        if score > best_score:
                            best_score, best_step = score, step
                            best_state = _state_cpu(student)
                            atomic_torch_save(best_state, best_path)
                        _log(experiment, 'validation step {} PSNR-Y={:.6f} SSIM-Y={:.6f} best_step={}'.format(
                            step, score, float(result['mean_ssim_y']), best_step))
                    if step % resume_interval == 0 or validation_due or step == invocation_stop:
                        trace.flush()
                        os.fsync(trace.fileno())
                        if best_state is None:
                            raise RuntimeError('Cannot save resume without a selected best checkpoint')
                        _save_resume(resume_path, step, config_hash, student, optimizer,
                                     scheduler, committed_sampling_rng_state,
                                     best_score, best_step, best_state)
                    del lr_cpu, hr_cpu, lr, hr, teacher_output, student_output
                    del teacher_l1, gt_l1, total_loss
        finally:
            if batches is not None:
                batches.close()

        if invocation_stop < niter:
            print(json.dumps({'status': 'paused', 'experiment': str(experiment),
                              'completed_step': completed_step, 'target_step': niter},
                             sort_keys=True))
            return {'status': 'paused', 'completed_step': completed_step,
                    'config_sha256': config_hash}

        last_state = _state_cpu(student)
        atomic_torch_save(last_state, last_path)
        best_state = tensor_state(torch_load_weights(best_path))
        clone = EDSR(student_opt['n_resblocks'], student_opt['n_feats'],
                     student_opt['res_scale'], student_opt.get('n_colors', 3),
                     student_opt.get('rgb_range', 255), scale)
        strict_load(clone, best_state)
        with torch.inference_mode():
            output = clone(torch.rand(1, 3, 8, 11).mul(255))
        if list(output.shape) != [1, 3, 8 * scale, 11 * scale] or not bool(torch.isfinite(output).all()):
            raise AssertionError('Best checkpoint strict round-trip failed')
        best_rows = [row for row in _truncate_validation(val_dir, niter)
                     if int(row['step']) == int(best_step)]
        if len(best_rows) != 1:
            raise RuntimeError('Best validation evidence is missing or duplicated')
        report = {
            'status': 'complete', 'name': opt['name'], 'scale': scale, 'seed': seed,
            'started_at_utc': started_at, 'completed_at_utc': utc_now(),
            'config_sha256': config_hash, 'niter': niter,
            'environment': environment_report(device),
            'teacher_checkpoint': teacher_checkpoint,
            'initial_student_tensor_sha256': tensor_state_sha256(initial_state),
            'source_indices': source_indices, 'static_student_audit': audit,
            'best_validation': {'step': best_step, 'mean_psnr_y_db': best_score,
                                'mean_ssim_y': float(best_rows[0]['mean_ssim_y'])},
            'checkpoints': {
                'best_val': {'path': str(best_path), 'sha256': sha256_file(best_path),
                             'bytes': best_path.stat().st_size},
                'last': {'path': str(last_path), 'sha256': sha256_file(last_path),
                         'bytes': last_path.stat().st_size},
                'resume': {'path': str(resume_path), 'sha256': sha256_file(resume_path),
                           'bytes': resume_path.stat().st_size},
            },
            'training': {'wall_seconds': time.time() - wall_start,
                         'prefetch_batches': prefetch_batches,
                         'step_p50_ms': percentile(timings, 50),
                         'step_p95_ms': percentile(timings, 95),
                         'input_wait_p50_ms': percentile(input_timings, 50),
                         'input_wait_p95_ms': percentile(input_timings, 95),
                         'h2d_p50_ms': percentile(h2d_timings, 50),
                         'h2d_p95_ms': percentile(h2d_timings, 95),
                         'gpu_step_p50_ms': percentile(gpu_timings, 50),
                         'gpu_step_p95_ms': percentile(gpu_timings, 95),
                         'images_per_second_at_step_p50': (
                             batch_size * 1000.0 / percentile(timings, 50)),
                         'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                         'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                         'trace_sha256': sha256_file(trace_path)},
        }
        atomic_json(report_path, report)
        if failure_path.exists():
            failure_path.rename(experiment / 'superseded_failure_{}.json'.format(int(time.time())))
        print(json.dumps({'status': 'complete', 'experiment': str(experiment),
                          'best_step': best_step, 'best_psnr_y': best_score}, sort_keys=True))
        return report
    except Exception as error:
        atomic_json(failure_path, {'status': 'failed', 'failed_at_utc': utc_now(),
                                   'completed_step': completed_step,
                                   'type': type(error).__name__, 'message': str(error),
                                   'traceback': traceback.format_exc(),
                                   'oom': isinstance(error, torch.cuda.OutOfMemoryError)})
        raise


def test_from_options(option_path):
    opt = option.load(option_path)
    scale = int(opt['scale'])
    if not torch.cuda.is_available():
        raise RuntimeError('SR testing requires CUDA')
    experiment = experiment_directory(opt)
    output_root = experiment / 'test'
    output_root.mkdir(parents=True, exist_ok=True)
    config_hash = resolved_config(opt, experiment / 'test_config.resolved.yml')
    checkpoint = resolve_repo_path(opt['path']['pretrain_model_G'])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256_file(checkpoint)
    summary_path = output_root / 'summary.json'
    if summary_path.is_file():
        summary = json.load(summary_path.open('r', encoding='utf-8'))
        if (summary.get('status') == 'complete' and summary.get('config_sha256') == config_hash
                and summary.get('checkpoint', {}).get('sha256') == checkpoint_hash):
            print_test_table(summary)
            return summary
        raise RuntimeError('Existing test output differs: {}'.format(output_root))
    if any(output_root.iterdir()):
        raise RuntimeError('Non-empty incomplete test output: {}'.format(output_root))

    configure_reproducibility(
        int((opt.get('test') or {}).get('seed', 0)),
        opt.get('reproducibility') or {
            'cudnn_benchmark': False,
            'cudnn_deterministic': True,
            'tf32': False,
        })
    device = torch.device('cuda:0')
    model = networks.define_G(opt).eval().float()
    strict_load(model, tensor_state(torch_load_weights(checkpoint)))
    model = model.to(device)
    save_images = bool((opt.get('test') or {}).get('save_images', False))
    datasets = {}
    rows = []
    for _, dataset in opt['datasets'].items():
        name = str(dataset['name'])
        pairs = _dataset_pairs(dataset, scale)
        dataset_dir = output_root / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result = evaluate_model(model, pairs, scale, device,
                                dataset_dir / 'sr_images' if save_images else None)
        records = result.pop('per_image')
        csv_path = dataset_dir / 'per_image.csv'
        jsonl_path = dataset_dir / 'per_image.jsonl'
        _write_csv(csv_path, ['image_id', 'psnr_y_db', 'ssim_y'], records)
        _write_records(jsonl_path, records)
        result.update({'per_image_csv': str(csv_path),
                       'per_image_jsonl': str(jsonl_path),
                       'per_image_jsonl_sha256': sha256_file(jsonl_path)})
        datasets[name] = result
        rows.append({'dataset': name, 'images': result['count'],
                     'psnr_y_db': result['mean_psnr_y_db'],
                     'ssim_y': result['mean_ssim_y']})
    summary = {'status': 'complete', 'completed_at_utc': utc_now(),
               'name': opt['name'], 'scale': scale, 'config_sha256': config_hash,
               'model': dict(opt['network_G']),
               'checkpoint': {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                              'sha256': checkpoint_hash},
               'environment': environment_report(device), 'datasets': datasets}
    _write_csv(output_root / 'summary.csv',
               ['dataset', 'images', 'psnr_y_db', 'ssim_y'], rows)
    atomic_json(summary_path, summary)
    lines = test_table_lines(summary)
    (output_root / 'test.log').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return summary


def test_table_lines(summary):
    lines = ['Checkpoint: {}'.format(summary['checkpoint']['path']),
             'SHA256: {}'.format(summary['checkpoint']['sha256']),
             'Model: {}-d{}, X{}'.format(summary['model']['which_model_G'],
                                         summary['model'].get('n_resblocks', '?'),
                                         summary['scale']), '',
             '{:<12} {:>7} {:>14} {:>12}'.format('Dataset', 'Images', 'PSNR-Y', 'SSIM-Y')]
    for name, result in summary['datasets'].items():
        lines.append('{:<12} {:>7d} {:>11.6f} dB {:>12.6f}'.format(
            name, int(result['count']), float(result['mean_psnr_y_db']),
            float(result['mean_ssim_y'])))
    return lines


def print_test_table(summary):
    print('\n'.join(test_table_lines(summary)), flush=True)
