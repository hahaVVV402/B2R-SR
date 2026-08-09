"""Canonical EDSR architecture and physical depth-transfer helpers.

Module layout matches EDSR-PyTorch revision
8dba5581a7502b92de9641eb431130d6c8ca5d7f (MIT, Sanghyun Son, 2018).
"""

import math
import re
from collections import OrderedDict

import torch
from torch import nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=kernel_size // 2, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean) / std
        for parameter in self.parameters():
            parameter.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale):
        super().__init__()
        self.body = nn.Sequential(
            default_conv(n_feats, n_feats, 3),
            nn.ReLU(True),
            default_conv(n_feats, n_feats, 3),
        )
        self.res_scale = res_scale

    def forward(self, x):
        residual = self.body(x).mul(self.res_scale)
        residual += x
        return residual


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats):
        modules = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.extend((default_conv(n_feats, 4 * n_feats, 3), nn.PixelShuffle(2)))
        elif scale == 3:
            modules.extend((default_conv(n_feats, 9 * n_feats, 3), nn.PixelShuffle(3)))
        else:
            raise ValueError('Unsupported EDSR scale: {}'.format(scale))
        super().__init__(*modules)


class EDSR(nn.Module):
    def __init__(self, n_resblocks=32, n_feats=256, res_scale=0.1,
                 n_colors=3, rgb_range=255, scale=4):
        super().__init__()
        if n_colors != 3:
            raise ValueError('Canonical EDSR requires three RGB channels')
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.head = nn.Sequential(default_conv(3, n_feats, 3))
        body = [ResBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        body.append(default_conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(Upsampler(scale, n_feats), default_conv(n_feats, 3, 3))

    def forward(self, x):
        x = self.head(self.sub_mean(x))
        residual = self.body(x)
        residual += x
        return self.add_mean(self.tail(residual))


def uniform_endpoint_indices(teacher_depth, student_depth):
    """Return an ordered, endpoint-inclusive uniform Teacher subset."""
    if not 1 < student_depth <= teacher_depth:
        raise ValueError('Require 1 < student_depth <= teacher_depth')
    indices = [int(round(i * (teacher_depth - 1) / float(student_depth - 1)))
               for i in range(student_depth)]
    if indices != sorted(set(indices)) or indices[0] != 0 or indices[-1] != teacher_depth - 1:
        raise AssertionError('Invalid uniform endpoint mapping: {}'.format(indices))
    return indices


def tensor_state(payload):
    candidate = payload
    if isinstance(payload, dict) and isinstance(payload.get('state_dict'), dict):
        candidate = payload['state_dict']
    if not isinstance(candidate, dict) or not candidate:
        raise TypeError('Checkpoint is not a non-empty tensor mapping')
    state = OrderedDict()
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, (torch.Tensor, nn.Parameter)):
            raise TypeError('Checkpoint state must contain only string-to-tensor entries')
        state[key[7:] if key.startswith('module.') else key] = value.detach().cpu()
    return state


def strict_load(model, state):
    result = nn.Module.load_state_dict(model, state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError('Strict load failed: missing={}, unexpected={}'.format(
            result.missing_keys, result.unexpected_keys))
    loaded = model.state_dict()
    mismatched = [key for key, value in state.items()
                  if key not in loaded or not torch.equal(value.cpu(), loaded[key].cpu())]
    if mismatched:
        raise AssertionError('Strict-load tensor mismatch: {}'.format(mismatched[:5]))


def transplant_edsr(teacher_state, scale, teacher_depth, student_depth,
                    n_feats=256, res_scale=0.1, rgb_range=255,
                    strategy='uniform_endpoints'):
    if strategy != 'uniform_endpoints':
        raise ValueError('Unsupported formal block mapping: {}'.format(strategy))
    source_indices = uniform_endpoint_indices(teacher_depth, student_depth)
    student = EDSR(student_depth, n_feats, res_scale, 3, rgb_range, scale)
    transplanted = OrderedDict()
    for target_key, target_value in student.state_dict().items():
        match = re.match(r'^body\.(\d+)\.(.+)$', target_key)
        if match:
            target_index = int(match.group(1))
            suffix = match.group(2)
            if target_index < student_depth:
                source_key = 'body.{}.{}'.format(source_indices[target_index], suffix)
            else:
                source_key = 'body.{}.{}'.format(teacher_depth, suffix)
        else:
            source_key = target_key
        if source_key not in teacher_state:
            raise KeyError('Teacher checkpoint has no tensor {}'.format(source_key))
        source = teacher_state[source_key]
        if source.shape != target_value.shape:
            raise ValueError('{} -> {} shape mismatch: {} != {}'.format(
                source_key, target_key, tuple(source.shape), tuple(target_value.shape)))
        transplanted[target_key] = source.detach().clone()
    strict_load(student, transplanted)
    return student, transplanted, source_indices
