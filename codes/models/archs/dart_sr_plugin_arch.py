import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.archs.RCAN_arch import RCAN
from models.archs.CARN_arch import CARN_M
from models.archs.SRResNet_arch import MSRResNet


def _get_opt(opt, key, default):
    if opt is None:
        return default
    value = opt.get(key, default)
    if value is None:
        return default
    return value


class DegradationEstimator(nn.Module):
    def __init__(self, in_nc=3, hidden_dim=32):
        super(DegradationEstimator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_nc, hidden_dim, 3, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, *args, **kwargs):
        feat = self.encoder(x).flatten(1)
        return torch.sigmoid(self.head(feat))


class BudgetAllocator(nn.Module):
    """Stage-wise keep-ratio allocator conditioned on image-level context."""

    def __init__(self, num_stages, hidden_dim=32, delta_scale=0.15):
        super(BudgetAllocator, self).__init__()
        self.delta_scale = float(delta_scale)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_stages)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, degradation_score, complexity_score, user_budget, min_keep, max_keep):
        budget = torch.full_like(degradation_score, float(user_budget))
        ctx = torch.stack([degradation_score, complexity_score, budget], dim=1)
        learned_delta = self.delta_scale * torch.tanh(self.mlp(ctx))
        adaptive_base = min_keep + (max_keep - min_keep) * (0.5 * degradation_score + 0.5 * complexity_score)
        base = 0.5 * budget.unsqueeze(1) + 0.5 * adaptive_base.unsqueeze(1)
        return torch.clamp(base + learned_delta, min_keep, max_keep)


class CheapAdapter(nn.Module):
    """Very small residual path for low-benefit windows."""

    def __init__(self, channels, hidden_scale=0.25):
        super(CheapAdapter, self).__init__()
        hidden = max(4, int(channels * hidden_scale))
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, hidden, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, 1, 0, bias=True)
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)


class DARTSRPlugin(nn.Module):
    """Backbone-agnostic token routing plugin for block-based SR models."""

    def __init__(self, backbone, backbone_type, plugin_opt=None):
        super(DARTSRPlugin, self).__init__()
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.plugin_opt = plugin_opt or {}

        self.route_window = int(_get_opt(plugin_opt, 'route_window', 8))
        self.tau0 = float(_get_opt(plugin_opt, 'tau0', 0.5))
        self.alpha = float(_get_opt(plugin_opt, 'alpha', 0.35))
        self.var_weight = float(_get_opt(plugin_opt, 'var_weight', 0.2))
        self.hard_train_after = int(_get_opt(plugin_opt, 'hard_train_after', 20000))
        self.hard_infer = bool(_get_opt(plugin_opt, 'hard_infer', True))
        self.use_ste = bool(_get_opt(plugin_opt, 'use_ste', True))
        self.routing_mode = str(_get_opt(plugin_opt, 'routing_mode', 'threshold'))
        self.sync_latency = bool(_get_opt(plugin_opt, 'sync_latency', False))
        self.benefit_teacher_mode = str(_get_opt(plugin_opt, 'benefit_teacher_mode', 'warmup'))

        self.target_keep_min = float(_get_opt(plugin_opt, 'target_keep_min', 0.45))
        self.target_keep_max = float(_get_opt(plugin_opt, 'target_keep_max', 0.95))
        self.static_flops_ratio = float(_get_opt(plugin_opt, 'static_flops_ratio', 0.25))
        self.base_flops = float(_get_opt(plugin_opt, 'base_flops', 0.0))

        deg_opt = _get_opt(plugin_opt, 'deg_estimator', {})
        self.use_deg_estimator = bool(_get_opt(deg_opt, 'enable', True))
        hidden_dim = int(_get_opt(deg_opt, 'hidden_dim', 32))
        self.deg_estimator = DegradationEstimator(in_nc=3, hidden_dim=hidden_dim)
        if not self.use_deg_estimator:
            for p in self.deg_estimator.parameters():
                p.requires_grad = False

        self.current_iter = 0
        self.freeze_backbone = bool(_get_opt(plugin_opt, 'freeze_backbone', True))
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.stage_modules, stage_channels = self._collect_stages()
        self.router_heads = nn.ModuleList([nn.Conv2d(ch, 1, 1, 1, 0, bias=True) for ch in stage_channels])

        budget_opt = _get_opt(plugin_opt, 'budget_allocator', {})
        self.use_budget_allocator = bool(_get_opt(budget_opt, 'enable', False))
        self.user_budget = float(_get_opt(budget_opt, 'user_budget', 0.7))
        self.budget_allocator = BudgetAllocator(
            num_stages=len(self.stage_modules),
            hidden_dim=int(_get_opt(budget_opt, 'hidden_dim', 32)),
            delta_scale=float(_get_opt(budget_opt, 'delta_scale', 0.15))
        )
        if not self.use_budget_allocator:
            for p in self.budget_allocator.parameters():
                p.requires_grad = False

        cheap_opt = _get_opt(plugin_opt, 'cheap_path', {})
        self.use_cheap_path = bool(_get_opt(cheap_opt, 'enable', False))
        adapter_scale = float(_get_opt(cheap_opt, 'hidden_scale', 0.25))
        self.cheap_adapters = nn.ModuleList([CheapAdapter(ch, adapter_scale) for ch in stage_channels])
        if not self.use_cheap_path:
            for p in self.cheap_adapters.parameters():
                p.requires_grad = False

        stage_weights = _get_opt(plugin_opt, 'stage_weights', None)
        if stage_weights is None or len(stage_weights) != len(self.stage_modules):
            stage_weights = [1.0 for _ in self.stage_modules]
        weights = torch.tensor(stage_weights, dtype=torch.float32)
        weights = weights / max(weights.sum().item(), 1e-6)
        self.register_buffer('stage_weights', weights)

    def set_train_iteration(self, current_iter):
        self.current_iter = int(current_iter)

    def _collect_stages(self):
        if self.backbone_type == 'RCAN':
            modules = nn.ModuleList(list(self.backbone.body[:-1]))
            channels = [self.backbone.head[0].out_channels for _ in range(len(modules))]
            return modules, channels
        if self.backbone_type == 'CARN_M':
            modules = nn.ModuleList([self.backbone.b1, self.backbone.b2, self.backbone.b3])
            channels = [self.backbone.entry.out_channels, self.backbone.entry.out_channels, self.backbone.entry.out_channels]
            return modules, channels
        if self.backbone_type == 'MSRResNet':
            modules = nn.ModuleList(list(self.backbone.recon_trunk))
            channels = [self.backbone.conv_first.out_channels for _ in range(len(modules))]
            return modules, channels
        raise NotImplementedError('Unsupported backbone type: {}'.format(self.backbone_type))

    def _proxy_degradation_target(self, x):
        # low high-frequency energy usually indicates stronger blur-like degradation
        gray = x.mean(dim=1, keepdim=True)
        lap = F.conv2d(gray, self._lap_kernel(x.device, x.dtype), padding=1)
        hf = lap.abs().flatten(1).mean(1)
        min_v = hf.min()
        max_v = hf.max()
        norm = (hf - min_v) / (max_v - min_v + 1e-6)
        return (1.0 - norm).detach()

    def _lap_kernel(self, device, dtype):
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=dtype, device=device)
        return kernel.view(1, 1, 3, 3)

    def _normalize_per_image(self, value):
        v_min = value.min(dim=1, keepdim=True)[0]
        v_max = value.max(dim=1, keepdim=True)[0]
        return (value - v_min) / (v_max - v_min + 1e-6)

    def _image_complexity_score(self, x):
        gray = x.mean(dim=1, keepdim=True)
        lap = F.conv2d(gray, self._lap_kernel(x.device, x.dtype), padding=1)
        hf = lap.abs().flatten(1).mean(1)
        base = gray.abs().flatten(1).mean(1) + 1e-6
        ratio = hf / base
        if ratio.numel() > 1:
            min_v = ratio.min()
            max_v = ratio.max()
            return (ratio - min_v) / (max_v - min_v + 1e-6)
        return ratio / (ratio + 1.0)

    def _target_keep_per_stage(self, degradation_score, complexity_score):
        if self.routing_mode == 'benefit_topk' and self.use_budget_allocator:
            return self.budget_allocator(
                degradation_score,
                complexity_score,
                self.user_budget,
                self.target_keep_min,
                self.target_keep_max
            )
        target = self.target_keep_min + (self.target_keep_max - self.target_keep_min) * degradation_score
        return target.unsqueeze(1).expand(-1, len(self.stage_modules))

    def _window_partition(self, x):
        b, c, h, w = x.size()
        ws = self.route_window
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        hp, wp = x.size(2), x.size(3)
        nh, nw = hp // ws, wp // ws
        windows = x.view(b, c, nh, ws, nw, ws).permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = windows.view(b, nh * nw, c, ws, ws)
        meta = (h, w, hp, wp, nh, nw, pad_h, pad_w)
        return windows, meta

    def _window_merge(self, windows, meta):
        h, w, hp, wp, nh, nw, _, _ = meta
        b, _, c, ws, _ = windows.size()
        x = windows.view(b, nh, nw, c, ws, ws).permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, hp, wp)
        return x[:, :, :h, :w]

    def _upsample_mask(self, mask, meta):
        h, w, hp, wp, nh, nw, _, _ = meta
        ws = self.route_window
        mask = mask.view(mask.size(0), nh, nw)
        mask_px = mask.repeat_interleave(ws, dim=1).repeat_interleave(ws, dim=2)
        return mask_px[:, :h, :w].unsqueeze(1)

    def _topk_mask(self, score, keep_ratio):
        b, num_wins = score.size()
        mask = torch.zeros_like(score)
        keep_ratio = torch.clamp(keep_ratio, 0.0, 1.0)
        for i in range(b):
            k = int(torch.ceil(keep_ratio[i] * num_wins).item())
            k = max(1, min(num_wins, k))
            top_idx = torch.topk(score[i], k, dim=0)[1]
            mask[i, top_idx] = 1.0
        return mask

    def _route_mask(self, feat, stage_idx, degradation_score, training_hard, target_keep_stage=None):
        windows, meta = self._window_partition(feat)
        b, num_wins = windows.size(0), windows.size(1)
        h, w, hp, wp, nh, nw, _, _ = meta

        feat_pad = feat
        if hp != h or wp != w:
            feat_pad = F.pad(feat, (0, wp - w, 0, hp - h), mode='reflect')
        router_map = self.router_heads[stage_idx](feat_pad)
        router_logits = F.avg_pool2d(router_map, kernel_size=self.route_window, stride=self.route_window)
        router_logits = router_logits.view(b, num_wins)

        var = windows.var(dim=(2, 3, 4), unbiased=False)
        var = (var - var.mean(dim=1, keepdim=True)) / (var.std(dim=1, keepdim=True) + 1e-6)

        if self.routing_mode == 'benefit_topk':
            logits = router_logits + self.var_weight * var
            prob = torch.sigmoid(logits)
        else:
            tau = self.tau0 + self.alpha * degradation_score
            logits = router_logits + self.var_weight * var - tau.unsqueeze(1)
            prob = torch.sigmoid(logits)

        if self.routing_mode == 'benefit_topk' and target_keep_stage is not None:
            if self.training:
                if not training_hard:
                    mask = prob
                elif self.use_ste:
                    hard = self._topk_mask(logits, target_keep_stage)
                    mask = hard + prob - prob.detach()
                else:
                    hard = self._topk_mask(logits, target_keep_stage)
                    mask = hard
            else:
                hard = self._topk_mask(logits, target_keep_stage)
                mask = hard if self.hard_infer else prob
        else:
            if self.training:
                if training_hard and self.use_ste:
                    hard = (prob >= 0.5).float()
                    mask = hard + prob - prob.detach()
                elif training_hard:
                    mask = (prob >= 0.5).float()
                else:
                    mask = prob
            else:
                if self.hard_infer:
                    mask = (prob >= 0.5).float()
                else:
                    mask = prob

        return mask, prob, meta, logits

    def _benefit_target_from_delta(self, delta):
        windows, _ = self._window_partition(delta.abs())
        benefit = windows.mean(dim=(2, 3, 4))
        return self._normalize_per_image(benefit)

    def _should_compute_benefit_target(self, training_hard):
        if not self.training or self.routing_mode != 'benefit_topk':
            return False
        if self.benefit_teacher_mode == 'always':
            return True
        if self.benefit_teacher_mode == 'warmup':
            return not training_hard
        return False

    def _gated_forward(self, x, stage_module, mask, meta, training_hard, stage_idx):
        adapter = self.cheap_adapters[stage_idx] if self.use_cheap_path else None
        compute_benefit_target = self._should_compute_benefit_target(training_hard)

        if self.training and (not training_hard or compute_benefit_target):
            # warm-up with dense compute, mask only controls residual blending
            stage_out = stage_module(x)
            benefit_target = self._benefit_target_from_delta(stage_out - x) if compute_benefit_target else None
            if not training_hard:
                skip_out = adapter(x) if adapter is not None else x
                mask_px = self._upsample_mask(mask, meta)
                return skip_out + mask_px * (stage_out - skip_out), benefit_target
        else:
            benefit_target = None

        windows, _ = self._window_partition(x)
        b, num_wins, c, ws, _ = windows.size()
        windows_flat = windows.view(b * num_wins, c, ws, ws)
        out_windows = windows_flat.clone()

        active = (mask >= 0.5).view(-1)
        if adapter is not None:
            inactive = ~active
            if inactive.any():
                out_windows[inactive] = adapter(windows_flat[inactive])
        if active.any():
            out_windows[active] = stage_module(windows_flat[active])
        out_windows = out_windows.view(b, num_wins, c, ws, ws)
        return self._window_merge(out_windows, meta), benefit_target

    def _estimate_flops(self, keep_ratio_per_stage):
        # keep_ratio_per_stage: [B, G]
        stage_keep = keep_ratio_per_stage * self.stage_weights.unsqueeze(0)
        weighted_keep = stage_keep.sum(dim=1)
        flops_ratio = self.static_flops_ratio + (1.0 - self.static_flops_ratio) * weighted_keep
        if self.base_flops > 0:
            flops_est = flops_ratio * self.base_flops
        else:
            flops_est = flops_ratio
        return flops_ratio, flops_est

    def _forward_rcan(self, x, degradation_score, training_hard, target_keep_per_stage):
        x = self.backbone.sub_mean(x)
        x = self.backbone.head(x)

        res = x
        soft_keep = []
        hard_keep = []
        mask_maps = []
        benefit_probs = []
        benefit_targets = []
        for idx, stage_module in enumerate(self.stage_modules):
            feat = res
            mask, prob, meta, _ = self._route_mask(
                feat, idx, degradation_score, training_hard, target_keep_per_stage[:, idx])
            res, benefit_target = self._gated_forward(res, stage_module, mask, meta, training_hard, idx)
            soft_keep.append(prob.mean(dim=1))
            hard_keep.append((mask >= 0.5).float().mean(dim=1))
            h, w, _, _, nh, nw, _, _ = meta
            mask_maps.append(prob.view(prob.size(0), nh, nw))
            benefit_probs.append(prob)
            if benefit_target is not None:
                benefit_targets.append((prob, benefit_target))

        res = self.backbone.body[-1](res)
        res += x
        out = self.backbone.tail(res)
        out = self.backbone.add_mean(out)
        return out, soft_keep, hard_keep, mask_maps, benefit_probs, benefit_targets

    def _forward_carn(self, x, degradation_score, training_hard, target_keep_per_stage):
        x = self.backbone.sub_mean(x)
        x = self.backbone.entry(x)
        c0 = o0 = x

        soft_keep = []
        hard_keep = []
        mask_maps = []
        benefit_probs = []
        benefit_targets = []

        mask1, prob1, meta1, _ = self._route_mask(
            o0, 0, degradation_score, training_hard, target_keep_per_stage[:, 0])
        b1, benefit_target = self._gated_forward(o0, self.backbone.b1, mask1, meta1, training_hard, 0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.backbone.c1(c1)
        soft_keep.append(prob1.mean(dim=1))
        hard_keep.append((mask1 >= 0.5).float().mean(dim=1))
        mask_maps.append(prob1.view(prob1.size(0), meta1[4], meta1[5]))
        benefit_probs.append(prob1)
        if benefit_target is not None:
            benefit_targets.append((prob1, benefit_target))

        mask2, prob2, meta2, _ = self._route_mask(
            o1, 1, degradation_score, training_hard, target_keep_per_stage[:, 1])
        b2, benefit_target = self._gated_forward(o1, self.backbone.b2, mask2, meta2, training_hard, 1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.backbone.c2(c2)
        soft_keep.append(prob2.mean(dim=1))
        hard_keep.append((mask2 >= 0.5).float().mean(dim=1))
        mask_maps.append(prob2.view(prob2.size(0), meta2[4], meta2[5]))
        benefit_probs.append(prob2)
        if benefit_target is not None:
            benefit_targets.append((prob2, benefit_target))

        mask3, prob3, meta3, _ = self._route_mask(
            o2, 2, degradation_score, training_hard, target_keep_per_stage[:, 2])
        b3, benefit_target = self._gated_forward(o2, self.backbone.b3, mask3, meta3, training_hard, 2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.backbone.c3(c3)
        soft_keep.append(prob3.mean(dim=1))
        hard_keep.append((mask3 >= 0.5).float().mean(dim=1))
        mask_maps.append(prob3.view(prob3.size(0), meta3[4], meta3[5]))
        benefit_probs.append(prob3)
        if benefit_target is not None:
            benefit_targets.append((prob3, benefit_target))

        out = self.backbone.upsample(o3, scale=self.backbone.scale)
        out = self.backbone.exit(out)
        out = self.backbone.add_mean(out)
        return out, soft_keep, hard_keep, mask_maps, benefit_probs, benefit_targets

    def _forward_srresnet(self, x, degradation_score, training_hard, target_keep_per_stage):
        fea = self.backbone.lrelu(self.backbone.conv_first(x))
        out = fea

        soft_keep = []
        hard_keep = []
        mask_maps = []
        benefit_probs = []
        benefit_targets = []
        for idx, stage_module in enumerate(self.stage_modules):
            mask, prob, meta, _ = self._route_mask(
                out, idx, degradation_score, training_hard, target_keep_per_stage[:, idx])
            out, benefit_target = self._gated_forward(out, stage_module, mask, meta, training_hard, idx)
            soft_keep.append(prob.mean(dim=1))
            hard_keep.append((mask >= 0.5).float().mean(dim=1))
            mask_maps.append(prob.view(prob.size(0), meta[4], meta[5]))
            benefit_probs.append(prob)
            if benefit_target is not None:
                benefit_targets.append((prob, benefit_target))

        if self.backbone.upscale == 4:
            out = self.backbone.lrelu(self.backbone.pixel_shuffle(self.backbone.upconv1(out)))
            out = self.backbone.lrelu(self.backbone.pixel_shuffle(self.backbone.upconv2(out)))
        elif self.backbone.upscale == 3 or self.backbone.upscale == 2:
            out = self.backbone.lrelu(self.backbone.pixel_shuffle(self.backbone.upconv1(out)))

        out = self.backbone.conv_last(self.backbone.lrelu(self.backbone.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.backbone.upscale, mode='bilinear', align_corners=False)
        out += base
        return out, soft_keep, hard_keep, mask_maps, benefit_probs, benefit_targets

    def forward(self, x):
        if self.sync_latency and x.is_cuda:
            torch.cuda.synchronize(x.device)
        t0 = time.time()
        training_hard = self.current_iter >= self.hard_train_after if self.training else self.hard_infer

        if self.use_deg_estimator:
            degradation_score = self.deg_estimator(x).squeeze(1)
        else:
            degradation_score = torch.full((x.size(0),), 0.5, device=x.device, dtype=x.dtype)

        complexity_score = self._image_complexity_score(x)
        target_keep_per_stage = self._target_keep_per_stage(degradation_score, complexity_score)

        if self.backbone_type == 'RCAN':
            out, soft_keep, hard_keep, mask_maps, benefit_probs, benefit_targets = self._forward_rcan(
                x, degradation_score, training_hard, target_keep_per_stage)
        elif self.backbone_type == 'CARN_M':
            out, soft_keep, hard_keep, mask_maps, benefit_probs, benefit_targets = self._forward_carn(
                x, degradation_score, training_hard, target_keep_per_stage)
        elif self.backbone_type == 'MSRResNet':
            out, soft_keep, hard_keep, mask_maps, benefit_probs, benefit_targets = self._forward_srresnet(
                x, degradation_score, training_hard, target_keep_per_stage)
        else:
            raise NotImplementedError('Unsupported backbone type: {}'.format(self.backbone_type))

        soft_keep_tensor = torch.stack(soft_keep, dim=1)  # [B, G]
        hard_keep_tensor = torch.stack(hard_keep, dim=1)  # [B, G]
        keep_for_metrics = hard_keep_tensor if (not self.training or training_hard) else soft_keep_tensor
        keep_ratio_total = keep_for_metrics.mean(dim=1)
        flops_ratio, flops_estimated = self._estimate_flops(keep_for_metrics)

        loss_budget = (soft_keep_tensor - target_keep_per_stage.detach()).pow(2).mean()
        if self.routing_mode == 'benefit_topk' and self.use_budget_allocator:
            budget_ref = torch.full_like(degradation_score, self.user_budget)
            loss_budget = loss_budget + 0.25 * (target_keep_per_stage.mean(dim=1) - budget_ref).pow(2).mean()
        loss_sparse = soft_keep_tensor.mean()

        if len(benefit_targets) > 0:
            benefit_losses = []
            for benefit_prob, benefit_target in benefit_targets:
                benefit_losses.append(F.mse_loss(benefit_prob, benefit_target.detach()))
            loss_benefit = torch.stack(benefit_losses).mean()
        else:
            loss_benefit = out.new_tensor(0.0)

        tv_losses = []
        for p in mask_maps:
            tv_h = (p[:, :, 1:] - p[:, :, :-1]).abs().mean()
            tv_w = (p[:, 1:, :] - p[:, :-1, :]).abs().mean()
            tv_losses.append(tv_h + tv_w)
        if len(tv_losses) > 0:
            loss_tv = torch.stack(tv_losses).mean()
        else:
            loss_tv = out.new_tensor(0.0)

        if self.use_deg_estimator:
            deg_target = self._proxy_degradation_target(x)
            loss_deg = (degradation_score - deg_target).abs().mean()
        else:
            loss_deg = out.new_tensor(0.0)

        if self.sync_latency and x.is_cuda:
            torch.cuda.synchronize(x.device)
        latency_ms = out.new_tensor([(time.time() - t0) * 1000.0])
        plugin_info = {
            'keep_ratio_total': keep_ratio_total,
            'keep_ratio_per_stage': keep_for_metrics,
            'flops_ratio': flops_ratio,
            'flops_estimated': flops_estimated,
            'degradation_score': degradation_score,
            'complexity_score': complexity_score,
            'target_keep_per_stage': target_keep_per_stage,
            'latency_ms': latency_ms,
            'loss_budget': loss_budget.unsqueeze(0),
            'loss_sparse': loss_sparse.unsqueeze(0),
            'loss_benefit': loss_benefit.unsqueeze(0),
            'loss_tv': loss_tv.unsqueeze(0),
            'loss_deg': loss_deg.unsqueeze(0)
        }
        return out, plugin_info

    def load_backbone_state_dict(self, state_dict, strict=True):
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[7:]
            cleaned[key] = value

        direct_result = self.backbone.load_state_dict(cleaned, strict=False)
        has_backbone_prefix = any(k.startswith('backbone.') for k in cleaned.keys())
        if has_backbone_prefix:
            prefixed = {}
            for k, v in cleaned.items():
                if k.startswith('backbone.'):
                    prefixed[k[9:]] = v
            if len(prefixed) > 0:
                direct_result = self.backbone.load_state_dict(prefixed, strict=strict)
        elif strict:
            self.backbone.load_state_dict(cleaned, strict=strict)
        return direct_result


def build_dartsr_backbone(opt_net):
    which_model = opt_net['which_model_G']
    if which_model == 'RCAN':
        backbone = RCAN(
            n_resblocks=opt_net['n_resblocks'],
            n_feats=opt_net['n_feats'],
            res_scale=opt_net['res_scale'],
            n_colors=opt_net['n_colors'],
            rgb_range=opt_net['rgb_range'],
            scale=opt_net['scale'],
            reduction=opt_net['reduction'],
            n_resgroups=opt_net['n_resgroups']
        )
        return DARTSRPlugin(backbone=backbone, backbone_type='RCAN', plugin_opt=opt_net.get('plugin'))

    if which_model == 'CARN_M':
        backbone = CARN_M(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            nf=opt_net['nf'],
            scale=opt_net['scale'],
            group=opt_net['group']
        )
        return DARTSRPlugin(backbone=backbone, backbone_type='CARN_M', plugin_opt=opt_net.get('plugin'))

    if which_model == 'MSRResNet':
        upscale = opt_net.get('upscale', opt_net.get('scale', 4))
        backbone = MSRResNet(
            in_nc=opt_net['in_nc'],
            out_nc=opt_net['out_nc'],
            nf=opt_net['nf'],
            nb=opt_net['nb'],
            upscale=upscale
        )
        return DARTSRPlugin(backbone=backbone, backbone_type='MSRResNet', plugin_opt=opt_net.get('plugin'))

    raise NotImplementedError('DART-SR plugin does not support model [{:s}]'.format(which_model))
