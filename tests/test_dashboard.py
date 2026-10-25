import math
import unittest

import numpy as np

from dashboard.server import add_ground_reprojections, calculate_reprojection_metrics


class DashboardReprojectionTest(unittest.TestCase):
    def setUp(self):
        self.p2 = np.asarray(
            [[100, 0, 50, 0], [0, 100, 40, 0], [0, 0, 1, 0]], dtype=float
        )
        self.record = {
            "calibration": {"P2": self.p2.tolist()},
            "ground_plane": {"valid": True, "coefficients": [0, -1, 0, 2]},
            "objects": [
                {
                    "object_id": 0,
                    "touch_points": [
                        {
                            "name": "LF",
                            "projection_valid": True,
                            "pixel_2d": [60, 60],
                            "camera_3d": [0.5, 1.0, 5.0],
                        }
                    ],
                }
            ],
        }

    def test_frame_payload_contains_ground_reprojection(self):
        result = add_ground_reprojections(self.record, self.p2)
        point = result["objects"][0]["touch_points"][0]
        self.assertTrue(point["ground_reprojection_valid"])
        np.testing.assert_allclose(point["reprojected_ground_3d"], [1, 2, 10], atol=1e-8)
        np.testing.assert_allclose(point["ground_shift_3d"], [0.5, 1, 5], atol=1e-8)

    def test_split_metrics_report_axis_and_euclidean_errors(self):
        metrics = calculate_reprojection_metrics({"000001": self.record})
        self.assertEqual(metrics["point_count"], 1)
        self.assertEqual(metrics["valid_reprojection_frame_count"], 1)
        self.assertAlmostEqual(metrics["mae_x"], 0.5)
        self.assertAlmostEqual(metrics["mae_y"], 1.0)
        self.assertAlmostEqual(metrics["mae_z"], 5.0)
        self.assertAlmostEqual(metrics["mean_coordinate_mae"], 6.5 / 3)
        self.assertAlmostEqual(metrics["mean_euclidean_error"], math.sqrt(26.25))


if __name__ == "__main__":
    unittest.main()
