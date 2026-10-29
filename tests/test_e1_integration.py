import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lib.helpers.config_helper import load_config
from lib.helpers.optimizer_helper import build_optimizer
from lib.models.monodetr.matcher import HungarianMatcher
from lib.models.monodetr.monodetr import SetCriterion


ROOT = Path(__file__).resolve().parents[1]


class NamedToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.geometry_e1 = nn.Module()
        self.geometry_e1.horizon_head = nn.Linear(2, 2)
        self.geometry_e1.touchpoint_head = nn.Linear(2, 2)
        self.geometry_e1.feature_adapter = nn.Linear(2, 2)
        self.geometry_e1.fusion_alpha_logit = nn.Parameter(torch.tensor(-2.0))


class E1IntegrationTest(unittest.TestCase):
    def test_config_extends_and_environment_defaults(self):
        config = load_config(ROOT / 'configs' / 'e1_feature_level.yaml', ROOT / 'missing.env')
        self.assertEqual(config['model']['num_queries'], 50)
        self.assertEqual(config['model']['group_num'], 11)
        self.assertEqual(config['geometry_e1']['fusion_mode'], 'feature_level')
        self.assertEqual(config['dataset']['depth_scale'], 'none')
        self.assertIsInstance(config['geometry_e1']['plane']['camera_height_m'], float)

    def test_touchpoint_v2_config_uses_stronger_relative_supervision(self):
        config = load_config(
            ROOT / 'configs' / 'e1_feature_level_touch_v2.yaml',
            ROOT / 'missing.env')
        touch = config['geometry_e1']['touchpoint']
        self.assertEqual(touch['loss_coordinate_mode'], 'box_relative')
        self.assertAlmostEqual(touch['smooth_l1_beta'], 0.05)
        self.assertTrue(touch['require_inside_image'])
        self.assertTrue(touch['box_detach'])

    def test_joint_epoch0_config_uses_pretrained_always_two_pass(self):
        config = load_config(
            ROOT / 'configs' / 'e1_feature_joint_epoch0.yaml',
            ROOT / 'missing.env')
        geometry = config['geometry_e1']
        self.assertEqual(geometry['training']['fusion_start_epoch'], 0)
        self.assertEqual(geometry['training']['phase1_decoder_passes'], 2)
        self.assertAlmostEqual(geometry['fusion']['initial_alpha'], 0.001)
        self.assertAlmostEqual(geometry['fusion']['alpha_max'], 0.1)
        self.assertFalse(
            geometry['horizon']['vertical_edge_refinement']['enabled'])
        self.assertTrue(config['trainer']['pretrain_model'])
        self.assertEqual(config['trainer']['max_epoch'], 30)

    def test_scratch_config_removes_detector_checkpoint_and_uses_long_schedule(self):
        config = load_config(
            ROOT / 'configs' / 'e1_feature_scratch_imagenet.yaml',
            ROOT / 'missing.env')
        self.assertIsNone(config['trainer']['pretrain_model'])
        self.assertIsNone(config['geometry_e1']['pretrained_checkpoint'])
        self.assertEqual(config['trainer']['max_epoch'], 195)
        self.assertEqual(config['trainer']['grad_accum_steps'], 1)
        self.assertAlmostEqual(
            config['optimizer']['param_groups']['original_monodetr_lr'], 2e-4)
        self.assertEqual(config['lr_scheduler']['decay_list'], [125, 165])
        self.assertTrue(config['lr_scheduler']['warmup'])

    def test_distribution_epoch0_configs_match_feature_training_protocol(self):
        joint = load_config(
            ROOT / 'configs' / 'e1_distribution_joint_epoch0.yaml',
            ROOT / 'missing.env')
        scratch = load_config(
            ROOT / 'configs' / 'e1_distribution_scratch_imagenet.yaml',
            ROOT / 'missing.env')
        self.assertEqual(
            joint['geometry_e1']['fusion_mode'], 'distribution_level')
        self.assertEqual(
            joint['geometry_e1']['training']['fusion_start_epoch'], 0)
        self.assertAlmostEqual(
            joint['geometry_e1']['fusion']['initial_alpha'], 0.001)
        self.assertTrue(joint['trainer']['pretrain_model'])
        self.assertEqual(
            scratch['geometry_e1']['fusion_mode'], 'distribution_level')
        self.assertIsNone(scratch['trainer']['pretrain_model'])
        self.assertEqual(scratch['trainer']['max_epoch'], 195)
        self.assertEqual(
            scratch['geometry_e1']['training']['fusion_start_epoch'], 0)
        self.assertAlmostEqual(
            scratch['geometry_e1']['losses'][
                'pass1_aux_detection_loss_coef_phase2'], 1.0)
        self.assertAlmostEqual(
            scratch['geometry_e1']['fusion']['initial_alpha'], 0.01)
        self.assertAlmostEqual(
            scratch['optimizer']['param_groups']['fusion_alpha_lr'], 1e-5)

    def test_optimizer_groups_every_parameter_once(self):
        model = NamedToyModel()
        optimizer = build_optimizer({
            'type': 'adamw', 'lr': 1e-5, 'weight_decay': 1e-4,
            'param_groups': {
                'original_monodetr_lr': 1e-5,
                'horizon_touchpoint_lr': 1e-4,
                'geometry_fusion_lr': 2e-4,
            },
        }, model)
        parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertEqual(len(parameters), len(list(model.parameters())))
        rates = {group['group_name']: group['lr'] for group in optimizer.param_groups}
        self.assertEqual(rates['base_weight'], 1e-5)
        self.assertEqual(rates['head_weight'], 1e-4)
        self.assertEqual(rates['fusion_weight'], 2e-4)
        self.assertEqual(rates['fusion_alpha'], 2e-4)
        alpha_group = next(
            group for group in optimizer.param_groups
            if group['group_name'] == 'fusion_alpha')
        self.assertEqual(alpha_group['weight_decay'], 0)

    def test_empty_ground_truth_object_losses_are_finite(self):
        matcher = HungarianMatcher(1, 1, 1, 1)
        criterion = SetCriterion(
            3, matcher, {}, 0.25, [], group_num=1, geometry_cfg={})
        outputs = {
            'pred_logits': torch.randn(2, 5, 3, requires_grad=True),
            'pred_boxes': torch.rand(2, 5, 6, requires_grad=True),
            'pred_depth': torch.rand(2, 5, 2, requires_grad=True),
            'pred_3d_dim': torch.rand(2, 5, 3, requires_grad=True),
            'pred_angle': torch.rand(2, 5, 24, requires_grad=True),
        }
        targets = [{
            'labels': torch.empty(0, dtype=torch.long),
            'boxes': torch.empty(0, 4),
            'boxes_3d': torch.empty(0, 6),
            'depth': torch.empty(0, 1),
            'size_3d': torch.empty(0, 3),
            'heading_bin': torch.empty(0, 1, dtype=torch.long),
            'heading_res': torch.empty(0, 1),
        } for _ in range(2)]
        indices = matcher(outputs, targets, group_num=1)
        losses = {}
        for name in ('center', 'boxes', 'depths', 'dims', 'angles'):
            losses.update(criterion.get_loss(name, outputs, targets, indices, 1.0))
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()


if __name__ == '__main__':
    unittest.main()
