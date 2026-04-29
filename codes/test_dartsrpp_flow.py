#!/usr/bin/env python3
"""Smoke test for the original DART-SR path and the DART++ improvement path.

This script does not require datasets or pretrained checkpoints. It uses tiny
random tensors to verify:

1. direct plugin forward/backward for RCAN, CARN_M, and MSRResNet;
2. both routing modes: threshold and benefit_topk;
3. SRModel training and inference integration.

Run from the repository root:

    .venv/bin/python codes/test_dartsrpp_flow.py
"""

import argparse
import os
import sys

import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODES_DIR = os.path.join(ROOT_DIR, "codes")
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)


def assert_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise AssertionError("{} contains NaN or Inf".format(name))


def plugin_opt(mode, route_window=4):
    opt = {
        "enable": True,
        "routing_mode": mode,
        "route_window": route_window,
        "tau0": 0.5,
        "alpha": 0.35,
        "var_weight": 0.2,
        "hard_train_after": 2,
        "hard_infer": True,
        "use_ste": True,
        "benefit_teacher_mode": "warmup",
        "sync_latency": False,
        "target_keep_min": 0.35,
        "target_keep_max": 0.90,
        "static_flops_ratio": 0.20,
        "base_flops": 1.0,
        "freeze_backbone": False,
        "deg_estimator": {
            "enable": True,
            "hidden_dim": 8,
        },
    }
    if mode == "benefit_topk":
        opt.update({
            "budget_allocator": {
                "enable": True,
                "user_budget": 0.60,
                "hidden_dim": 8,
                "delta_scale": 0.10,
            },
            "cheap_path": {
                "enable": True,
                "hidden_scale": 0.25,
            },
        })
    return opt


def network_opt(which_model, mode):
    common_plugin = plugin_opt(mode)
    if which_model == "RCAN":
        return {
            "which_model_G": "RCAN",
            "n_resblocks": 1,
            "n_feats": 8,
            "n_resgroups": 2,
            "res_scale": 1,
            "n_colors": 3,
            "rgb_range": 255,
            "scale": 2,
            "reduction": 4,
            "plugin": common_plugin,
        }
    if which_model == "CARN_M":
        return {
            "which_model_G": "CARN_M",
            "in_nc": 3,
            "out_nc": 3,
            "nf": 8,
            "scale": 2,
            "group": 1,
            "plugin": common_plugin,
        }
    if which_model == "MSRResNet":
        return {
            "which_model_G": "MSRResNet",
            "in_nc": 3,
            "out_nc": 3,
            "nf": 8,
            "nb": 2,
            "scale": 2,
            "upscale": 2,
            "plugin": common_plugin,
        }
    raise ValueError("Unsupported test backbone: {}".format(which_model))


def expected_scale(which_model):
    return 2


def metric_float(info, key):
    return float(info[key].detach().float().mean().cpu())


def check_plugin_info(info, mode):
    required = [
        "keep_ratio_total",
        "keep_ratio_per_stage",
        "target_keep_per_stage",
        "flops_ratio",
        "flops_estimated",
        "degradation_score",
        "complexity_score",
        "latency_ms",
        "loss_budget",
        "loss_sparse",
        "loss_benefit",
        "loss_tv",
        "loss_deg",
    ]
    for key in required:
        if key not in info:
            raise AssertionError("Missing plugin_info key [{}] for mode [{}]".format(key, mode))
        assert_finite(key, info[key])


def run_direct_plugin_smoke(device):
    from models.archs.dart_sr_plugin_arch import build_dartsr_backbone

    print("\n[1/2] Direct DART plugin smoke")
    backbones = ["RCAN", "CARN_M", "MSRResNet"]
    modes = ["threshold", "benefit_topk"]

    for backbone in backbones:
        for mode in modes:
            torch.manual_seed(123)
            model = build_dartsr_backbone(network_opt(backbone, mode)).to(device)
            model.train()

            scale = expected_scale(backbone)
            x = torch.randn(2, 3, 16, 16, device=device)

            for step in [1, 3]:
                model.zero_grad(set_to_none=True)
                model.set_train_iteration(step)
                y, info = model(x)
                expected_shape = (2, 3, 16 * scale, 16 * scale)
                if tuple(y.shape) != expected_shape:
                    raise AssertionError(
                        "{} {} step {} output shape {}, expected {}".format(
                            backbone, mode, step, tuple(y.shape), expected_shape))
                check_plugin_info(info, mode)
                loss = y.abs().mean()
                for key in ["loss_budget", "loss_sparse", "loss_benefit", "loss_tv", "loss_deg"]:
                    loss = loss + info[key].mean()
                loss.backward()

            model.eval()
            with torch.no_grad():
                y, info = model(x)
            check_plugin_info(info, mode)
            print(
                "  OK {:9s} {:12s} output={} keep={:.4f} flops={:.4f} target={}".format(
                    backbone,
                    mode,
                    tuple(y.shape),
                    metric_float(info, "keep_ratio_total"),
                    metric_float(info, "flops_ratio"),
                    [round(v, 4) for v in info["target_keep_per_stage"].mean(dim=0).cpu().tolist()],
                )
            )


def make_sr_model_opt(mode, device):
    use_cuda = device.type == "cuda"
    return {
        "name": "dartsrpp_flow_{}".format(mode),
        "model": "sr",
        "scale": 2,
        "gpu_ids": [0] if use_cuda else None,
        "is_train": True,
        "dist": False,
        "network_G": network_opt("RCAN", mode),
        "path": {
            "pretrain_model_G": None,
            "strict_load": True,
            "root": "/tmp/dartsrpp_flow",
            "models": "/tmp/dartsrpp_flow/models",
            "training_state": "/tmp/dartsrpp_flow/training_state",
            "log": "/tmp/dartsrpp_flow",
            "val_images": "/tmp/dartsrpp_flow/val_images",
        },
        "train": {
            "lr_G": 1e-4,
            "lr_scheme": "MultiStepLR",
            "beta1": 0.9,
            "beta2": 0.99,
            "weight_decay_G": 0,
            "T_period": [50],
            "restarts": [50],
            "restart_weights": [1],
            "lr_gamma": 0.5,
            "clear_state": False,
            "pixel_criterion": "l1",
            "pixel_weight": 1.0,
            "plugin_loss": {
                "budget_weight": 0.5,
                "sparse_weight": 0.1,
                "benefit_weight": 0.2,
                "tv_weight": 0.05,
                "deg_weight": 0.05,
            },
        },
        "logger": {
            "print_freq": 1,
            "save_checkpoint_freq": 100,
        },
    }


def run_sr_model_smoke(device):
    from models.SR_model import SRModel

    print("\n[2/2] SRModel integration smoke")
    for mode in ["threshold", "benefit_topk"]:
        torch.manual_seed(456)
        opt = make_sr_model_opt(mode, device)
        model = SRModel(opt)

        for step in [1, 3]:
            lr = torch.randn(2, 3, 16, 16, device=device)
            hr = torch.randn(2, 3, 32, 32, device=device)
            model.feed_data({"LQ": lr, "GT": hr})
            model.optimize_parameters(step)
            logs = model.get_current_log()
            for key in ["l_pix", "l_total", "l_budget", "l_sparse", "l_benefit", "l_tv", "l_deg"]:
                if key not in logs:
                    raise AssertionError("Missing training log [{}] for mode [{}]".format(key, mode))
            print(
                "  train {:12s} step={} l_total={:.6f} keep={:.4f} flops={:.4f}".format(
                    mode, step, logs["l_total"], logs["keep_ratio"], logs["flops_ratio"])
            )

        model.test()
        visuals = model.get_current_visuals()
        required_visuals = [
            "rlt",
            "metrics.keep_ratio_total",
            "metrics.keep_ratio_per_stage",
            "metrics.target_keep_per_stage",
            "metrics.flops_ratio",
            "metrics.degradation_score",
            "metrics.complexity_score",
        ]
        for key in required_visuals:
            if key not in visuals:
                raise AssertionError("Missing visual [{}] for mode [{}]".format(key, mode))
        if tuple(visuals["rlt"].shape) != (3, 32, 32):
            raise AssertionError("Unexpected SRModel visual shape: {}".format(tuple(visuals["rlt"].shape)))
        print(
            "  eval  {:12s} output={} keep={:.4f} target={}".format(
                mode,
                tuple(visuals["rlt"].shape),
                visuals["metrics.keep_ratio_total"],
                [round(v, 4) for v in visuals["metrics.target_keep_per_stage"]],
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Run on CUDA if available.")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__)
    print("Device:", device)

    run_direct_plugin_smoke(device)
    run_sr_model_smoke(device)

    print("\nALL DART-SR / DART++ FLOW TESTS PASSED")


if __name__ == "__main__":
    main()
