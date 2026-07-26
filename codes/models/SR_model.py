import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss


logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        self.plugin_info = None
        self.train_opt = opt.get('train') or {}
        self.plugin_loss_weights = self.train_opt.get('plugin_loss', {})

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = self.train_opt

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)


        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['T_period'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self._set_train_iter(step)
        output = self.netG(self.var_L)
        self.fake_H, self.plugin_info = self._split_network_output(output)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        loss_total = l_pix

        if self.plugin_info is not None:
            l_budget = self._plugin_loss_term('loss_budget', 'budget_weight')
            l_sparse = self._plugin_loss_term('loss_sparse', 'sparse_weight')
            l_benefit = self._plugin_loss_term('loss_benefit', 'benefit_weight')
            l_tv = self._plugin_loss_term('loss_tv', 'tv_weight')
            l_deg = self._plugin_loss_term('loss_deg', 'deg_weight')
            loss_total = loss_total + l_budget + l_sparse + l_benefit + l_tv + l_deg
            self.log_dict['l_budget'] = l_budget.item()
            self.log_dict['l_sparse'] = l_sparse.item()
            self.log_dict['l_benefit'] = l_benefit.item()
            self.log_dict['l_tv'] = l_tv.item()
            self.log_dict['l_deg'] = l_deg.item()

        loss_total.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_total'] = loss_total.item()
        if self.plugin_info is not None:
            self.log_dict['keep_ratio'] = self._plugin_metric_mean('keep_ratio_total')
            self.log_dict['flops_ratio'] = self._plugin_metric_mean('flops_ratio')
            self.log_dict['deg_score'] = self._plugin_metric_mean('degradation_score')
            self.log_dict['complexity_score'] = self._plugin_metric_mean('complexity_score')

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            output = self.netG(self.var_L)
            self.fake_H, self.plugin_info = self._split_network_output(output)
        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if self.plugin_info is not None:
            out_dict['metrics.keep_ratio_total'] = self._plugin_metric_mean('keep_ratio_total')
            out_dict['metrics.keep_ratio_per_stage'] = self._plugin_metric_vector('keep_ratio_per_stage')
            out_dict['metrics.target_keep_per_stage'] = self._plugin_metric_vector('target_keep_per_stage')
            out_dict['metrics.flops_ratio'] = self._plugin_metric_mean('flops_ratio')
            out_dict['metrics.flops_estimated'] = self._plugin_metric_mean('flops_estimated')
            out_dict['metrics.degradation_score'] = self._plugin_metric_mean('degradation_score')
            out_dict['metrics.complexity_score'] = self._plugin_metric_mean('complexity_score')
            out_dict['metrics.latency_ms'] = self._plugin_metric_mean('latency_ms')
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self._load_network_compatible(load_path_G)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def _set_train_iter(self, step):
        net = self.netG.module if isinstance(self.netG, (nn.DataParallel, DistributedDataParallel)) else self.netG
        if hasattr(net, 'set_train_iteration'):
            net.set_train_iteration(step)

    def _split_network_output(self, output):
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
            return output[0], output[1]
        return output, None

    def _plugin_loss_term(self, info_key, weight_key):
        if self.plugin_info is None or info_key not in self.plugin_info:
            return self.fake_H.new_tensor(0.0)
        weight = float(self.plugin_loss_weights.get(weight_key, 0.0))
        return weight * self.plugin_info[info_key].mean()

    def _plugin_metric_mean(self, key):
        if self.plugin_info is None or key not in self.plugin_info:
            return 0.0
        return self.plugin_info[key].mean().item()

    def _plugin_metric_vector(self, key):
        if self.plugin_info is None or key not in self.plugin_info:
            return []
        value = self.plugin_info[key]
        if value.dim() == 1:
            return value.detach().float().cpu().tolist()
        return value.mean(dim=0).detach().float().cpu().tolist()

    def _load_network_compatible(self, load_path):
        strict = self.opt['path']['strict_load']
        net = self.netG.module if isinstance(self.netG, (nn.DataParallel, DistributedDataParallel)) else self.netG
        if hasattr(net, 'load_backbone_state_dict'):
            load_net = torch.load(load_path)
            keys = []
            for k in load_net.keys():
                if k.startswith('module.'):
                    keys.append(k[7:])
                else:
                    keys.append(k)
            has_plugin_state = any(
                k.startswith('router_heads.') or
                k.startswith('deg_estimator.') or
                k.startswith('budget_allocator.') or
                k.startswith('cheap_adapters.') or
                k.startswith('backbone.') or
                k.startswith('stage_weights')
                for k in keys
            )
            if has_plugin_state:
                self.load_network(load_path, self.netG, strict)
                logger.info('Loaded full DART-SR checkpoint from [{:s}].'.format(load_path))
                return
            net.load_backbone_state_dict(load_net, strict=strict)
            logger.info('Loaded pretrained weights into DART-SR backbone from [{:s}].'.format(load_path))
            return
        self.load_network(load_path, self.netG, strict)
