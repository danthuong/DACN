import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_val_depth_mae", ROOT / "tools" / "evaluate_val_depth_mae.py")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def kitti_line(depth, bbox):
    fields = [
        "Car", "0.0", "0", "0.0",
        *[str(value) for value in bbox],
        "1.5", "1.6", "3.8", "0.0", "1.5", str(depth), "0.0",
    ]
    return " ".join(fields) + "\n"


def artifact(final_depth, pass1_depth, predictor_depth, geometry_depth,
             geometry_valid):
    return {
        "detections": [{
            "query_id": 4,
            "class_id": 1,
            "class_name": "Car",
            "class_score": 0.9,
            "bbox_xyxy_raw": [[0.0, 0.0], [50.0, 50.0]],
            "selected_depth_m": final_depth,
            "pass1_depth_m": pass1_depth,
            "depth_predictor_depth_m": predictor_depth,
            "selected_depth_predictor_depth_m": predictor_depth,
            "geometric_depth_m": geometry_depth,
            "geometry_valid": geometry_valid,
        }]
    }


class ValDepthMaeTest(unittest.TestCase):
    def test_reports_branch_mae_and_geometry_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels"
            artifacts = root / "artifacts"
            output = root / "report"
            labels.mkdir()
            artifacts.mkdir()
            (root / "val.txt").write_text("000001\n000002\n")
            (labels / "000001.txt").write_text(kitti_line(10.0, [0, 0, 50, 50]))
            (labels / "000002.txt").write_text(kitti_line(20.0, [0, 0, 50, 50]))
            (artifacts / "000001.json").write_text(json.dumps(
                artifact(12.0, 11.0, 14.0, 9.0, True)))
            (artifacts / "000002.json").write_text(json.dumps(
                artifact(17.0, 18.0, 22.0, 0.0, False)))

            argv = [
                "evaluate_val_depth_mae.py",
                "--artifact-dir", str(artifacts),
                "--ground-truth-dir", str(labels),
                "--split-file", str(root / "val.txt"),
                "--output-dir", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                report = EVALUATOR.main()

            self.assertAlmostEqual(
                report["branches"]["monodetr_final"]["mae"], 2.5)
            self.assertAlmostEqual(
                report["branches"]["monodetr_pass1"]["mae"], 1.5)
            self.assertAlmostEqual(
                report["branches"]["depth_predictor_original"]["mae"], 3.0)
            self.assertAlmostEqual(report["branches"]["gpro3d"]["mae"], 1.0)
            self.assertAlmostEqual(
                report["branches"]["gpro3d"]["coverage_of_matched"], 0.5)
            paired = report["gpro3d_valid_paired_subset"]
            self.assertEqual(paired["count"], 1)
            self.assertAlmostEqual(
                paired["branches"]["depth_predictor_original"]["mae"], 4.0)
            self.assertTrue((output / "val_depth_mae.csv").is_file())
            self.assertTrue((output / "per_object_depth_mae.csv").is_file())


if __name__ == "__main__":
    unittest.main()
