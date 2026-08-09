"""Single PSNR-Y/SSIM-Y implementation for validation and testing."""

import math

import cv2
import numpy as np


def rgb_to_y(image):
    image = image.astype(np.float64)
    return (65.738 * image[..., 0] + 129.057 * image[..., 1] +
            25.064 * image[..., 2]) / 256.0 + 16.0


def psnr_y(output, target, shave):
    difference = rgb_to_y(output) - rgb_to_y(target)
    valid = difference[shave:-shave, shave:-shave]
    if valid.size == 0:
        raise ValueError('Image is too small for a {}-pixel border shave'.format(shave))
    mse = float(np.mean(valid * valid))
    return float('inf') if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_y(output, target, shave):
    first = rgb_to_y(output)[shave:-shave, shave:-shave]
    second = rgb_to_y(target)[shave:-shave, shave:-shave]
    if min(first.shape) < 11:
        raise ValueError('Image is too small for 11x11 SSIM after border shave')
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mu1 = cv2.GaussianBlur(first, (11, 11), 1.5)[5:-5, 5:-5]
    mu2 = cv2.GaussianBlur(second, (11, 11), 1.5)[5:-5, 5:-5]
    sigma1 = cv2.GaussianBlur(first * first, (11, 11), 1.5)[5:-5, 5:-5] - mu1 * mu1
    sigma2 = cv2.GaussianBlur(second * second, (11, 11), 1.5)[5:-5, 5:-5] - mu2 * mu2
    sigma12 = cv2.GaussianBlur(first * second, (11, 11), 1.5)[5:-5, 5:-5] - mu1 * mu2
    value = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) /
             ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2)))
    return float(value.mean())


def quantize_rgb(tensor):
    """Convert a 1x3xHxW RGB tensor in [0,255] convention to uint8 HWC."""
    return tensor.clamp(0, 255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
