import unittest

import torch

from lib.models.monodetr.geometry_e1 import (
    GeometryE1,
    decode_horizon_line,
    horizon_line_to_ground_plane,
    make_horizon_heatmap_target,
    raycast_touchpoints_to_plane,
)


def geometry_cfg(mode="distribution_level"):
    return {
        "fusion_mode": mode,
        "horizon": {"fpn_channels": 32, "softargmax_temperature": 0.25},
        "touchpoint": {"max_touchpoints": 4, "class_index": 1},
        "plane": {
            "camera_height_m": 1.65,
            "min_abs_ray_plane_dot": 0.015,
            "valid_min_depth_m": 2.0,
            "valid_max_depth_m": 65.0,
        },
        "spatial_projection": {"box_edge_softness_grid_px": 1.25},
        "fusion": {
            "initial_alpha": 0.1,
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "mode_settings": {
                "gaussian_sigma_m": 2.5,
                "log_prior_clamp": [-12.0, 0.0],
            },
        },
    }


class GeometryE1Test(unittest.TestCase):
    def setUp(self):
        self.p2 = torch.tensor(
            [[[700.0, 0.0, 600.0, 0.0], [0.0, 700.0, 180.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
        )
        self.bin_values = torch.linspace(0.001, 60.0, 81)

    def test_horizon_target_and_softargmax_use_pixel_centres(self):
        line = torch.tensor([[0.0, 180.0]])
        target, mask = make_horizon_heatmap_target(
            line, torch.tensor([True]), (96, 320), (384, 1280), sigma=1.5
        )
        logits = target.clamp_min(1e-5).log() * 8.0
        decoded, _, _ = decode_horizon_line(
            logits, (384, 1280), temperature=1.0, entropy_confidence=False
        )
        self.assertTrue(mask.any())
        self.assertAlmostEqual(decoded[0, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(decoded[0, 1].item(), 180.0, delta=0.6)

    def test_flat_horizon_plane_and_road_raycast(self):
        plane, plane_valid = horizon_line_to_ground_plane(
            torch.tensor([[0.0, 180.0]]), self.p2, 1.65
        )
        self.assertTrue(plane_valid[0])
        torch.testing.assert_close(plane[0], torch.tensor([0.0, 1.0, 0.0, -1.65]))
        touch = torch.tensor([[[[600.0 / 1280.0, 300.0 / 384.0]]]])
        result = raycast_touchpoints_to_plane(touch, self.p2, plane, (384, 1280))
        self.assertTrue(result["point_valid"][0, 0, 0])
        self.assertAlmostEqual(result["point_depth"][0, 0, 0].item(), 9.625, places=3)

    def test_near_horizon_ray_is_invalid(self):
        plane, _ = horizon_line_to_ground_plane(torch.tensor([[0.0, 180.0]]), self.p2, 1.65)
        touch = torch.tensor([[[[600.0 / 1280.0, 180.0 / 384.0]]]])
        result = raycast_touchpoints_to_plane(touch, self.p2, plane, (384, 1280))
        self.assertFalse(result["point_valid"][0, 0, 0])

    def test_distribution_fusion_preserves_background_and_empty_identity(self):
        module = GeometryE1(geometry_cfg(), 256, 50, self.bin_values)
        logits = torch.randn(2, 81, 4, 5)
        geometry = {
            "object_depth": torch.zeros(2, 3),
            "object_spatial_weights": torch.zeros(2, 3, 4, 5),
            "valid_coverage": torch.zeros(2, 1, 4, 5),
        }
        fused = module.fuse_distribution(logits, geometry)
        torch.testing.assert_close(fused, logits, rtol=0, atol=0)
        torch.testing.assert_close(fused[:, -1:], logits[:, -1:], rtol=0, atol=0)

    def test_feature_alpha_zero_is_exact_identity(self):
        module = GeometryE1(geometry_cfg("feature_level"), 16, 50, self.bin_values)
        module.fusion_alpha.data.zero_()
        source = torch.randn(2, 16, 4, 5)
        geometry = {
            "spatial_prior": torch.rand(2, 1, 4, 5),
            "valid_coverage": torch.rand(2, 1, 4, 5),
            "confidence_map": torch.rand(2, 1, 4, 5),
        }
        torch.testing.assert_close(module.fuse_feature(source, geometry), source, rtol=0, atol=0)

    def test_distribution_fusion_backpropagates_to_geometry(self):
        module = GeometryE1(geometry_cfg(), 256, 50, self.bin_values)
        logits = torch.randn(1, 81, 2, 3, requires_grad=True)
        depths = torch.tensor([[15.0, 30.0]], requires_grad=True)
        weights = torch.rand(1, 2, 2, 3, requires_grad=True)
        geometry = {
            "object_depth": depths,
            "object_spatial_weights": weights,
            "valid_coverage": 1.0 - torch.exp(-weights.sum(dim=1, keepdim=True)),
        }
        fused = module.fuse_distribution(logits, geometry)
        fused[:, :-1].square().mean().backward()
        self.assertIsNotNone(depths.grad)
        self.assertGreater(depths.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(module.fusion_alpha.grad)


if __name__ == "__main__":
    unittest.main()
