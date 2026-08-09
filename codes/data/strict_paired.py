"""Strict path-paired RGB data helpers for static-depth SR experiments."""

import hashlib
import json
import random
import re
from pathlib import Path
from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Event, Thread

import cv2
import numpy as np
import torch

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}


class DeterministicBatchPrefetcher:
    """Prepare one future batch while committing RNG state only on consumption."""

    def __init__(self, load_batch, rng):
        self._load_batch = load_batch
        self._rng = rng
        self._queue = Queue(maxsize=1)
        self._slot = BoundedSemaphore(1)
        self._stop = Event()
        self._thread = Thread(target=self._produce, name='sr-batch-prefetch', daemon=True)
        self._started = False

    def __enter__(self):
        self._thread.start()
        self._started = True
        return self

    def __exit__(self, *_):
        self.close()

    def _produce(self):
        while not self._stop.is_set():
            if not self._slot.acquire(timeout=0.1):
                continue
            if self._stop.is_set():
                self._slot.release()
                break
            try:
                item = (self._load_batch(self._rng), self._rng.getstate(), None)
            except BaseException as error:
                item = (None, None, (error, error.__traceback__))
            placed = False
            while not self._stop.is_set():
                try:
                    self._queue.put(item, timeout=0.1)
                    placed = True
                    break
                except Full:
                    pass
            if not placed:
                self._slot.release()
            if item[2] is not None:
                break

    def get(self):
        if not self._started:
            raise RuntimeError('Batch prefetcher is not started')
        while True:
            try:
                value, rng_state, failure = self._queue.get(timeout=0.1)
            except Empty:
                if not self._thread.is_alive():
                    raise RuntimeError('Batch prefetcher stopped without a result')
                continue
            self._slot.release()
            if failure is not None:
                error, traceback = failure
                raise error.with_traceback(traceback)
            return value, rng_state

    def close(self):
        self._stop.set()
        if self._started:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError('Batch prefetcher did not stop within 5 seconds')


def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError('No images in {}'.format(directory))
    return files


def canonical_pair_key(path, scale):
    stem = re.sub(r'x{}(?=_s\d+$)'.format(scale), '', Path(path).stem,
                  flags=re.IGNORECASE)
    return re.sub(r'x{}$'.format(scale), '', stem, flags=re.IGNORECASE)


def original_div2k_id(key):
    match = re.fullmatch(r'(\d{4})(?:_s\d+)?', key)
    return match.group(1) if match else None


def pair_directories(hr_directory, lr_directory, scale, id_range=None):
    allowed = None
    if id_range is not None:
        first, last = [int(value) for value in id_range]
        allowed = {'{:04d}'.format(value) for value in range(first, last + 1)}

    def indexed(directory):
        result = {}
        for path in image_files(directory):
            key = canonical_pair_key(path, scale)
            if allowed is not None and original_div2k_id(key) not in allowed:
                continue
            if key in result:
                raise ValueError('Duplicate normalized key {} in {}'.format(key, directory))
            result[key] = path
        return result

    hr, lr = indexed(Path(hr_directory)), indexed(Path(lr_directory))
    if set(hr) != set(lr):
        raise ValueError('Unpaired x{} data: HR-only={}, LR-only={}'.format(
            scale, sorted(set(hr) - set(lr))[:5], sorted(set(lr) - set(hr))[:5]))
    if not hr:
        raise ValueError('No paired x{} images'.format(scale))
    if allowed is not None:
        observed = {original_div2k_id(key) for key in hr}
        missing = sorted(allowed - observed)
        if missing:
            raise ValueError('Missing DIV2K IDs: {}'.format(missing[:10]))
    return [(key, hr[key], lr[key]) for key in sorted(hr)]


def read_pair(hr_path, lr_path, scale, allow_modcrop=False):
    hr_bgr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
    lr_bgr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
    if hr_bgr is None or lr_bgr is None:
        raise ValueError('Unreadable pair: {}, {}'.format(hr_path, lr_path))
    target_height, target_width = lr_bgr.shape[0] * scale, lr_bgr.shape[1] * scale
    if allow_modcrop:
        excess_height = hr_bgr.shape[0] - target_height
        excess_width = hr_bgr.shape[1] - target_width
        if excess_height < 0 or excess_width < 0 or excess_height >= scale or excess_width >= scale:
            raise ValueError('Invalid modcrop geometry: HR={}, LR={}, scale={}'.format(
                hr_bgr.shape, lr_bgr.shape, scale))
        hr_bgr = hr_bgr[:target_height, :target_width]
    if hr_bgr.shape[:2] != (target_height, target_width):
        raise ValueError('Geometry mismatch: HR={}, LR={}, scale={}'.format(
            hr_bgr.shape, lr_bgr.shape, scale))
    return hr_bgr[:, :, ::-1].copy(), lr_bgr[:, :, ::-1].copy()


def validate_pairs(pairs, scale, minimum_lr_size=0, allow_modcrop=False,
                   max_pairs=0):
    selected = pairs[:max_pairs] if max_pairs else pairs
    digest = hashlib.sha256()
    for key, hr_path, lr_path in selected:
        hr, lr = read_pair(hr_path, lr_path, scale, allow_modcrop)
        if min(lr.shape[:2]) < minimum_lr_size:
            raise ValueError('{} is smaller than LR patch {}'.format(lr_path, minimum_lr_size))
        digest.update(key.encode('utf-8'))
        digest.update(json.dumps([list(hr.shape), list(lr.shape)],
                                 separators=(',', ':')).encode('ascii'))
    return {'count': len(pairs), 'validated_count': len(selected),
            'ordered_key_geometry_sha256': digest.hexdigest()}


def sample_batch(pairs, scale, batch_size, patch, rng):
    low_resolution, high_resolution, keys = [], [], []
    for _ in range(batch_size):
        key, hr_path, lr_path = pairs[rng.randrange(len(pairs))]
        hr, lr = read_pair(hr_path, lr_path, scale, allow_modcrop=False)
        if lr.shape[0] < patch or lr.shape[1] < patch:
            raise ValueError('{} is smaller than {}x{}'.format(key, patch, patch))
        top = rng.randrange(lr.shape[0] - patch + 1)
        left = rng.randrange(lr.shape[1] - patch + 1)
        lr = lr[top:top + patch, left:left + patch]
        hr = hr[top * scale:(top + patch) * scale,
                left * scale:(left + patch) * scale]
        if rng.random() < 0.5:
            lr, hr = np.flip(lr, axis=1), np.flip(hr, axis=1)
        if rng.random() < 0.5:
            lr, hr = np.flip(lr, axis=0), np.flip(hr, axis=0)
        if rng.random() < 0.5:
            lr, hr = np.transpose(lr, (1, 0, 2)), np.transpose(hr, (1, 0, 2))
        low_resolution.append(np.ascontiguousarray(lr))
        high_resolution.append(np.ascontiguousarray(hr))
        keys.append(key)

    def tensor(images):
        array = np.stack(images).transpose(0, 3, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(array)).float()

    return tensor(low_resolution), tensor(high_resolution), keys
