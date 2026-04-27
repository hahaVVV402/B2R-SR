#!/usr/bin/env python3
"""
Pipeline smoke test - 完全不依赖真实数据的单元测试
测试 DART-SR 训练和推理的完整流程
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# 添加 codes 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dartsr_plugin_wrapper():
    """测试 DART-SR plugin wrapper 的基本功能"""
    print("\n" + "="*60)
    print("Test 1: DART-SR Plugin Wrapper")
    print("="*60)

    from models.archs import dart_sr_plugin_arch

    # 创建一个简单的 backbone（模拟 RCAN/CARN）
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
            self.upscale = nn.Sequential(
                nn.Conv2d(64, 64*4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(64, 64*4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(64, 3, 3, 1, 1)
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.upscale(x)
            return x

    backbone = DummyBackbone()

    # 包装成 DART-SR plugin
    plugin_config = {
        'enable': True,
        'route_window': 8,
        'tau0': 0.5,
        'alpha': 0.35,
        'var_weight': 0.2,
        'hard_train_after': 100,
        'hard_infer': True,
        'use_ste': True,
        'target_keep_min': 0.45,
        'target_keep_max': 0.95,
        'static_flops_ratio': 0.25,
        'base_flops': 1.0,
        'freeze_backbone': False,
        'deg_estimator': {
            'enable': True,
            'hidden_dim': 32
        }
    }

    model = dart_sr_plugin_arch.DARTSRPluginWrapper(backbone, plugin_config)
    model.train()

    # 测试输入
    batch_size = 2
    lr_img = torch.randn(batch_size, 3, 48, 48)

    print(f"Input shape: {lr_img.shape}")

    # 前向传播
    output = model(lr_img)

    if isinstance(output, tuple):
        sr_img, plugin_info = output
        print(f"✓ Output is tuple (sr, plugin_info)")
        print(f"  SR shape: {sr_img.shape}")
        print(f"  Plugin info keys: {list(plugin_info.keys())}")

        # 检查必要的 plugin 信息
        required_keys = ['keep_ratio', 'loss_budget', 'loss_sparse', 'loss_tv']
        for key in required_keys:
            if key in plugin_info:
                print(f"  ✓ {key}: {plugin_info[key].item():.4f}")
            else:
                print(f"  ✗ Missing key: {key}")
                return False
    else:
        sr_img = output
        print(f"✓ Output shape: {sr_img.shape}")

    # 检查输出尺寸
    expected_shape = (batch_size, 3, 192, 192)  # 4x upscale
    if sr_img.shape == expected_shape:
        print(f"✓ Output shape correct: {sr_img.shape}")
    else:
        print(f"✗ Output shape mismatch: expected {expected_shape}, got {sr_img.shape}")
        return False

    # 测试反向传播
    loss = sr_img.mean()
    if isinstance(output, tuple):
        loss = loss + plugin_info['loss_budget'] + plugin_info['loss_sparse']

    loss.backward()
    print("✓ Backward pass successful")

    print("\n✓ Test 1 PASSED\n")
    return True


def test_sr_model_integration():
    """测试 SRModel 与 DART-SR plugin 的集成"""
    print("\n" + "="*60)
    print("Test 2: SRModel Integration")
    print("="*60)

    from models import SR_model
    from utils import util

    # 创建最小化配置
    opt = {
        'name': 'test_dartsr',
        'model': 'sr',
        'scale': 4,
        'gpu_ids': [0] if torch.cuda.is_available() else [],
        'is_train': True,
        'dist': False,

        'network_G': {
            'which_model_G': 'RCAN',
            'n_resblocks': 2,
            'n_feats': 16,
            'n_resgroups': 1,
            'res_scale': 1,
            'n_colors': 3,
            'rgb_range': 255,
            'scale': 4,
            'reduction': 16,
            'plugin': {
                'enable': True,
                'route_window': 8,
                'tau0': 0.5,
                'alpha': 0.35,
                'var_weight': 0.2,
                'hard_train_after': 100,
                'hard_infer': True,
                'use_ste': True,
                'target_keep_min': 0.45,
                'target_keep_max': 0.95,
                'static_flops_ratio': 0.25,
                'base_flops': 1.0,
                'freeze_backbone': False,
                'deg_estimator': {
                    'enable': True,
                    'hidden_dim': 16
                }
            }
        },

        'path': {
            'pretrain_model_G': None,
            'strict_load': True,
            'root': '/tmp/test_dartsr',
            'models': '/tmp/test_dartsr/models',
            'training_state': '/tmp/test_dartsr/training_state',
            'log': '/tmp/test_dartsr',
            'val_images': '/tmp/test_dartsr/val_images'
        },

        'train': {
            'lr_G': 1e-4,
            'lr_scheme': 'MultiStepLR',
            'beta1': 0.9,
            'beta2': 0.99,
            'niter': 100,
            'warmup_iter': -1,
            'lr_steps': [50],
            'lr_gamma': 0.5,
            'pixel_criterion': 'l1',
            'pixel_weight': 1.0,
            'plugin_loss': {
                'budget_weight': 0.5,
                'sparse_weight': 0.1,
                'tv_weight': 0.05,
                'deg_weight': 0.05
            },
            'manual_seed': 10,
            'val_freq': 50
        },

        'logger': {
            'print_freq': 10,
            'save_checkpoint_freq': 50
        }
    }

    # 转换为 namespace 对象
    class DictToObj:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, DictToObj(v))
                else:
                    setattr(self, k, v)

    opt_obj = DictToObj(opt)

    try:
        # 创建模型
        model = SR_model.SRModel(opt_obj)
        print("✓ SRModel created successfully")

        # 准备测试数据
        batch_size = 2
        lr_img = torch.randn(batch_size, 3, 48, 48)
        hr_img = torch.randn(batch_size, 3, 192, 192)

        if torch.cuda.is_available():
            lr_img = lr_img.cuda()
            hr_img = hr_img.cuda()

        # 模拟训练步骤
        model.feed_data({'LQ': lr_img, 'GT': hr_img})
        print("✓ Data fed successfully")

        model.optimize_parameters(1)
        print("✓ Optimization step successful")

        # 获取当前损失
        losses = model.get_current_log()
        print(f"✓ Current losses: {list(losses.keys())}")

        # 检查必要的损失项
        required_losses = ['l_pix']
        for loss_name in required_losses:
            if loss_name in losses:
                print(f"  ✓ {loss_name}: {losses[loss_name]:.4f}")
            else:
                print(f"  ✗ Missing loss: {loss_name}")
                return False

        # 测试推理
        model.test()
        visuals = model.get_current_visuals()
        print(f"✓ Inference successful, output keys: {list(visuals.keys())}")

        if 'rlt' in visuals:
            print(f"  Output shape: {visuals['rlt'].shape}")

        print("\n✓ Test 2 PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """测试完整的训练循环（10 步）"""
    print("\n" + "="*60)
    print("Test 3: Training Loop (10 iterations)")
    print("="*60)

    from models import SR_model

    # 创建配置
    opt = {
        'name': 'test_training_loop',
        'model': 'sr',
        'scale': 4,
        'gpu_ids': [0] if torch.cuda.is_available() else [],
        'is_train': True,
        'dist': False,

        'network_G': {
            'which_model_G': 'RCAN',
            'n_resblocks': 2,
            'n_feats': 16,
            'n_resgroups': 1,
            'res_scale': 1,
            'n_colors': 3,
            'rgb_range': 255,
            'scale': 4,
            'reduction': 16,
            'plugin': {
                'enable': True,
                'route_window': 8,
                'tau0': 0.5,
                'alpha': 0.35,
                'var_weight': 0.2,
                'hard_train_after': 5,
                'hard_infer': True,
                'use_ste': True,
                'target_keep_min': 0.45,
                'target_keep_max': 0.95,
                'static_flops_ratio': 0.25,
                'base_flops': 1.0,
                'freeze_backbone': False,
                'deg_estimator': {
                    'enable': True,
                    'hidden_dim': 16
                }
            }
        },

        'path': {
            'pretrain_model_G': None,
            'strict_load': True,
            'root': '/tmp/test_training_loop',
            'models': '/tmp/test_training_loop/models',
            'training_state': '/tmp/test_training_loop/training_state',
            'log': '/tmp/test_training_loop',
            'val_images': '/tmp/test_training_loop/val_images'
        },

        'train': {
            'lr_G': 1e-4,
            'lr_scheme': 'MultiStepLR',
            'beta1': 0.9,
            'beta2': 0.99,
            'niter': 10,
            'warmup_iter': -1,
            'lr_steps': [5],
            'lr_gamma': 0.5,
            'pixel_criterion': 'l1',
            'pixel_weight': 1.0,
            'plugin_loss': {
                'budget_weight': 0.5,
                'sparse_weight': 0.1,
                'tv_weight': 0.05,
                'deg_weight': 0.05
            },
            'manual_seed': 10,
            'val_freq': 5
        },

        'logger': {
            'print_freq': 2,
            'save_checkpoint_freq': 10
        }
    }

    class DictToObj:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, DictToObj(v))
                else:
                    setattr(self, k, v)

    opt_obj = DictToObj(opt)

    try:
        model = SR_model.SRModel(opt_obj)
        print("✓ Model initialized")

        # 训练 10 步
        for step in range(1, 11):
            # 生成随机数据
            lr_img = torch.randn(2, 3, 48, 48)
            hr_img = torch.randn(2, 3, 192, 192)

            if torch.cuda.is_available():
                lr_img = lr_img.cuda()
                hr_img = hr_img.cuda()

            model.feed_data({'LQ': lr_img, 'GT': hr_img})
            model.optimize_parameters(step)

            if step % 2 == 0:
                losses = model.get_current_log()
                loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in losses.items()])
                print(f"  Step {step:2d}: {loss_str}")

        print("✓ Training loop completed successfully")

        # 测试推理
        model.test()
        visuals = model.get_current_visuals()
        print(f"✓ Final inference successful")

        print("\n✓ Test 3 PASSED\n")
        return True

    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("DART-SR Pipeline Smoke Test")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("="*60)

    results = []

    # Test 1: Plugin wrapper
    try:
        results.append(("Plugin Wrapper", test_dartsr_plugin_wrapper()))
    except Exception as e:
        print(f"\n✗ Test 1 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Plugin Wrapper", False))

    # Test 2: SRModel integration
    try:
        results.append(("SRModel Integration", test_sr_model_integration()))
    except Exception as e:
        print(f"\n✗ Test 2 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("SRModel Integration", False))

    # Test 3: Training loop
    try:
        results.append(("Training Loop", test_training_loop()))
    except Exception as e:
        print(f"\n✗ Test 3 FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Training Loop", False))

    # 总结
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30s}: {status}")

    all_passed = all(r[1] for r in results)
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
