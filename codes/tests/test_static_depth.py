import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

CODES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODES))

from data.strict_paired import pair_directories, read_pair  # noqa: E402
from models import networks  # noqa: E402
from options import options  # noqa: E402
from models.archs.EDSR_arch import EDSR, transplant_edsr, uniform_endpoint_indices  # noqa: E402
from tasks.static_depth_recovery import _restore_resume_best, torch_load_weights  # noqa: E402
from utils.sr_metrics import psnr_y, ssim_y  # noqa: E402


class StaticDepthTest(unittest.TestCase):
    def test_uniform_endpoint_mapping_and_transplant(self):
        self.assertEqual(
            uniform_endpoint_indices(32, 24),
            [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15,
             16, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31])
        teacher = EDSR(n_resblocks=4, n_feats=8, scale=2)
        state = teacher.state_dict()
        for index in range(4):
            state['body.{}.body.0.weight'.format(index)].fill_(index + 1)
        student, transplanted, indices = transplant_edsr(
            state, scale=2, teacher_depth=4, student_depth=3, n_feats=8)
        self.assertEqual(indices, [0, 2, 3])
        for target, source in enumerate(indices):
            value = student.state_dict()['body.{}.body.0.weight'.format(target)]
            self.assertTrue(torch.equal(value, torch.full_like(value, source + 1)))
        self.assertIsNot(transplanted['head.0.weight'], state['head.0.weight'])

    def test_generic_network_factory(self):
        model = networks.define_G({'network_G': {
            'which_model_G': 'EDSR', 'n_resblocks': 3, 'n_feats': 8,
            'res_scale': 0.1, 'n_colors': 3, 'rgb_range': 255, 'scale': 2}})
        output = model(torch.rand(1, 3, 7, 9).mul(255))
        self.assertEqual(list(output.shape), [1, 3, 14, 18])

    def test_frozen_yaml_contract(self):
        os.environ.setdefault('SR_DATA_ROOT', '/tmp/sr-data')
        for scale in (2, 3, 4):
            train = options.load(str(CODES / 'options/train/train_EDSR_d24_X{}.yml'.format(scale)))
            test = options.load(str(CODES / 'options/test/test_EDSR_d24_X{}.yml'.format(scale)))
            self.assertEqual(train['train']['niter'], 200000)
            self.assertEqual(train['validation']['interval'], 5000)
            self.assertEqual(train['checkpoint']['rolling_resume_interval'], 2000)
            self.assertEqual(test['test_backend'], 'strict_sr')
            self.assertEqual(train['scale'], scale)
            self.assertEqual(test['scale'], scale)
        plan = options.load(str(CODES / 'options/run/run_EDSR_d24_formal.yml'))
        self.assertTrue(plan['execution']['train_all_before_test'])
        self.assertEqual(sum(len(run['seeds']) for run in plan['runs']), 9)

    def test_resume_restores_transactional_best_checkpoint(self):
        model = EDSR(n_resblocks=2, n_feats=8, scale=2)
        selected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        future = {key: value.detach().clone().add_(1) for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'best_val.pt'
            torch.save(future, path)
            restored = _restore_resume_best({'best_student': selected}, path)
            observed = torch_load_weights(path)
            self.assertTrue(all(torch.equal(restored[key], selected[key]) for key in selected))
            self.assertTrue(all(torch.equal(observed[key], selected[key]) for key in selected))

    def test_metrics_and_strict_pairing(self):
        image = np.full((24, 24, 3), 127, dtype=np.uint8)
        self.assertTrue(np.isinf(psnr_y(image, image, 2)))
        self.assertAlmostEqual(ssim_y(image, image, 2), 1.0, places=12)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'hr').mkdir()
            (root / 'lr').mkdir()
            hr = np.full((32, 36, 3), 100, dtype=np.uint8)
            lr = np.full((16, 18, 3), 100, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / 'hr/0001.png'), hr))
            self.assertTrue(cv2.imwrite(str(root / 'lr/0001x2.png'), lr))
            pairs = pair_directories(root / 'hr', root / 'lr', 2)
            self.assertEqual([row[0] for row in pairs], ['0001'])
            observed_hr, observed_lr = read_pair(pairs[0][1], pairs[0][2], 2)
            self.assertEqual(observed_hr.shape, (32, 36, 3))
            self.assertEqual(observed_lr.shape, (16, 18, 3))


if __name__ == '__main__':
    unittest.main()
