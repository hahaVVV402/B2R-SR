import argparse
import logging
import os.path as osp
import time
from collections import OrderedDict

import numpy as np

import options.options as option
import utils.util as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
from models import create_model


def strict_backend_requested(opt):
    return opt.get('test_backend') == 'strict_sr'


def legacy_test(opt_path):
    opt = option.dict_to_nonedict(option.parse(opt_path, is_train=False))
    util.mkdirs((path for key, path in opt['path'].items()
                 if key != 'experiments_root' and 'pretrain_model' not in key
                 and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'],
                      level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    which_model = opt['network_G']['which_model_G']

    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(
            dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        test_start_time = time.time()
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)
        test_results = OrderedDict((key, []) for key in
                                   ('psnr', 'ssim', 'psnr_y', 'ssim_y',
                                    'keep_ratio_total', 'flops_estimated', 'latency_ms'))
        for data in test_loader:
            model.feed_data(data, need_GT=True)
            img_path = data['GT_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]
            model.test()
            visuals = model.get_current_visuals(need_GT=True)
            if which_model == 'RCAN':
                sr_img = util.tensor2img(visuals['rlt'], out_type=np.uint8,
                                         min_max=(0, 255))
                gt_img = util.tensor2img(visuals['GT'], out_type=np.uint8,
                                         min_max=(0, 255))
            else:
                sr_img = util.tensor2img(visuals['rlt'])
                gt_img = util.tensor2img(visuals['GT'])
            suffix = opt['suffix']
            save_img_path = osp.join(dataset_dir,
                                     img_name + (suffix or '') + '.png')
            util.save_img(sr_img, save_img_path)
            if 'metrics.keep_ratio_total' in visuals:
                test_results['keep_ratio_total'].append(
                    visuals['metrics.keep_ratio_total'])
                test_results['flops_estimated'].append(
                    visuals.get('metrics.flops_estimated', 0.0))
                test_results['latency_ms'].append(
                    visuals.get('metrics.latency_ms', 0.0))
                logger.info('{:20s} - keep_ratio: {:.4f}; flops_est: {:.4f}; '
                            'latency: {:.3f} ms.'.format(
                                img_name, test_results['keep_ratio_total'][-1],
                                test_results['flops_estimated'][-1],
                                test_results['latency_ms'][-1]))
            sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
            psnr = util.calculate_psnr(sr_img, gt_img)
            ssim = util.calculate_ssim(sr_img, gt_img)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if gt_img.shape[2] == 3:
                sr_y = bgr2ycbcr(sr_img / 255., only_y=True) * 255
                gt_y = bgr2ycbcr(gt_img / 255., only_y=True) * 255
                psnr_y_value = util.calculate_psnr(sr_y, gt_y)
                ssim_y_value = util.calculate_ssim(sr_y, gt_y)
                test_results['psnr_y'].append(psnr_y_value)
                test_results['ssim_y'].append(ssim_y_value)
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; '
                            'PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.format(
                                img_name, psnr, ssim, psnr_y_value, ssim_y_value))
            else:
                logger.info('{:20s} - PSNR: {:.6f} dB;'.format(img_name, psnr))
        logger.info('----Average results for {} ({:.2f}s)----\n\tPSNR: {:.6f} dB; '
                    'SSIM: {:.6f}'.format(
                        test_set_name, time.time() - test_start_time,
                        sum(test_results['psnr']) / len(test_results['psnr']),
                        sum(test_results['ssim']) / len(test_results['ssim'])))
        if test_results['psnr_y']:
            logger.info('----Y channel average----\n\tPSNR_Y: {:.6f} dB; '
                        'SSIM_Y: {:.6f}'.format(
                            sum(test_results['psnr_y']) / len(test_results['psnr_y']),
                            sum(test_results['ssim_y']) / len(test_results['ssim_y'])))
        if test_results['keep_ratio_total']:
            logger.info('----Routing averages----\n\tkeep_ratio: {:.6f}; '
                        'flops_est: {:.6f}; latency: {:.6f} ms'.format(
                            sum(test_results['keep_ratio_total']) / len(test_results['keep_ratio_total']),
                            sum(test_results['flops_estimated']) / len(test_results['flops_estimated']),
                            sum(test_results['latency_ms']) / len(test_results['latency_ms'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True,
                        help='Path to options YAML file.')
    args = parser.parse_args()
    raw_opt = option.load(args.opt)
    if strict_backend_requested(raw_opt):
        from tasks.static_depth_recovery import test_from_options
        test_from_options(args.opt)
    else:
        legacy_test(args.opt)


if __name__ == '__main__':
    main()
