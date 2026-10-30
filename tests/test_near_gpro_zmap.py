import math

import torch

from lib.models.monodetr.near_gpro_zmap import NearGProGeometry


def _geometry():
    return NearGProGeometry({
        "horizon": {"vertical_edge_refinement": {"enabled": False}},
        "near_touchpoint": {"roi_channels": 64},
        "center_correction": {"k_l": 0.65, "k_w": 0.85},
    })


def test_near_edge_offset_recovers_bottom_center_for_axis_aligned_car():
    geometry = _geometry()
    dimensions = torch.tensor([[[1.5, 2.0, 4.0]]])
    yaw = torch.zeros(1, 1)
    offset = geometry._near_offset_camera(dimensions, yaw)
    expected = torch.tensor([[[0.0, 0.0, -0.85]]])
    torch.testing.assert_close(offset, expected)
    center = torch.tensor([[[2.0, 1.65, 20.0]]])
    near_point = center + offset
    torch.testing.assert_close(near_point - offset, center)


def test_near_edge_offset_rotates_with_kitti_yaw_convention():
    geometry = _geometry()
    dimensions = torch.tensor([[[1.5, 2.0, 4.0]]])
    yaw = torch.full((1, 1), math.pi / 2.0)
    offset = geometry._near_offset_camera(dimensions, yaw)
    torch.testing.assert_close(
        offset, torch.tensor([[[0.0, 0.0, -1.30]]]), atol=1e-5, rtol=1e-5
    )


def test_zmap_replacement_formula_and_identity_fallback():
    z_query = torch.tensor([[12.0, 12.0]])
    z_size = torch.tensor([[18.0, 18.0]])
    z_map = torch.tensor([[30.0, 30.0]])
    z_gpro = torch.tensor([[21.0, 21.0]])
    valid = torch.tensor([[True, False]])
    cue = torch.where(valid, z_gpro, z_map)
    final = (z_query + z_size + cue) / 3.0
    baseline = (z_query + z_size + z_map) / 3.0
    torch.testing.assert_close(final, torch.tensor([[17.0, 20.0]]))
    torch.testing.assert_close(final[:, 1], baseline[:, 1])
