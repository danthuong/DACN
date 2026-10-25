import math
import unittest

import numpy as np

from pseudo_label_generation.geometry import (
    backproject_pixels_to_plane,
    clip_line_to_image,
    derive_horizon_line,
    fit_ground_plane,
    project_points,
    transform_touch_points,
)


class GeometryTest(unittest.TestCase):
    def test_kitti_yaw_conversion_matches_expected_front(self):
        _, camera = transform_touch_points(4.0, 2.0, [0, 1.5, 10], math.pi / 2, 1.0, 1.0)
        # KITTI +pi/2 turns local +X toward camera -Z.
        np.testing.assert_allclose(camera[0], [1.0, 1.5, 8.0], atol=1e-8)

    def test_full_p2_projection_uses_translation_column(self):
        p2 = np.asarray([[100, 0, 50, 20], [0, 100, 40, -10], [0, 0, 1, 0]], dtype=float)
        pixels, valid = project_points(np.asarray([[1, 2, 10]], dtype=float), p2)
        self.assertTrue(valid[0])
        np.testing.assert_allclose(pixels[0], [62, 59], atol=1e-8)

    def test_backprojection_uses_full_p2_and_intersects_plane(self):
        p2 = np.asarray([[100, 0, 50, 20], [0, 100, 40, -10], [0, 0, 1, 0]], dtype=float)
        original = np.asarray([[1.0, 2.0, 10.0]])
        pixels, projection_valid = project_points(original, p2)
        self.assertTrue(projection_valid[0])

        points, valid = backproject_pixels_to_plane(pixels, p2, [0.0, -1.0, 0.0, 2.0])
        self.assertTrue(valid[0])
        np.testing.assert_allclose(points[0], original[0], atol=1e-8)

    def test_backprojection_rejects_ray_parallel_to_plane(self):
        p2 = np.asarray([[100, 0, 50, 0], [0, 100, 40, 0], [0, 0, 1, 0]], dtype=float)
        points, valid = backproject_pixels_to_plane([[50.0, 40.0]], p2, [0.0, 1.0, 0.0, -2.0])
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(points[0]).all())

    def test_plane_and_horizon(self):
        # Y = 0.1 X + 0.02 Z + 1.5
        points = [[-2, 1.5, 10], [2, 1.9, 10], [-3, 1.6, 20], [4, 2.3, 20]]
        plane = fit_ground_plane(points)
        self.assertTrue(plane["valid"])
        p2 = np.asarray([[700, 0, 600, 0], [0, 700, 180, 0], [0, 0, 1, 0]], dtype=float)
        horizon = derive_horizon_line(plane["coefficients"], p2, [1242, 375])
        self.assertTrue(horizon["valid"])
        self.assertAlmostEqual(horizon["slope"], 0.1, places=7)
        self.assertAlmostEqual(horizon["intercept"], 134.0, places=7)

    def test_collinear_plane_is_invalid(self):
        plane = fit_ground_plane([[0, 1, 10], [1, 1, 11], [2, 1, 12]])
        self.assertFalse(plane["valid"])
        self.assertEqual(plane["reason"], "collinear_samples")

    def test_bottom_y_consensus_rejects_height_outlier(self):
        plane = fit_ground_plane(
            [[-2, 1.5, 10], [2, 1.6, 11], [0, 1.55, 20], [4, 3.2, 18]],
            max_y_gap=0.5,
        )
        self.assertTrue(plane["valid"])
        self.assertEqual(plane["sample_indices"], [0, 1, 2])
        self.assertEqual(plane["rejected_indices"], [3])
        self.assertLessEqual(plane["bottom_y_gap"], 0.5)

    def test_nearly_collinear_xz_samples_are_invalid(self):
        plane = fit_ground_plane(
            [[-3.72, 1.78, 7.78], [-3.76, 1.78, 13.67], [-3.41, 1.86, 30.77]],
            max_y_gap=1.0,
            min_xz_spread_ratio=0.05,
        )
        self.assertFalse(plane["valid"])
        self.assertEqual(plane["reason"], "insufficient_xz_spread")

    def test_one_box_bottom_corners_define_horizontal_plane(self):
        _, corners = transform_touch_points(4.0, 2.0, [3.0, 1.7, 20.0], 0.8, 1.0, 1.0)
        plane = fit_ground_plane(corners, min_xz_spread_ratio=0.05)
        self.assertTrue(plane["valid"])
        a, b, c, d = plane["coefficients"]
        np.testing.assert_allclose([a, b, c, d], [0.0, -1.0, 0.0, 1.7], atol=1e-8)

    def test_line_clipping(self):
        segment = clip_line_to_image(1.0, -5.0, 10, 10)
        self.assertEqual(
            {tuple(np.round(point, 8)) for point in segment},
            {(5.0, 0.0), (9.0, 4.0)},
        )


if __name__ == "__main__":
    unittest.main()
