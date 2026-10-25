import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'evaluate_e1_depth', ROOT / 'tools' / 'evaluate_e1_depth.py')
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def kitti_line(depth, bbox, score=None):
    values = [
        'Car', '0.0', '0', '0.0',
        *[str(value) for value in bbox],
        '1.5', '1.6', '3.8', '0.0', '1.5', str(depth), '0.0',
    ]
    if score is not None:
        values.append(str(score))
    return ' '.join(values) + '\n'


class E1DepthEvaluatorTest(unittest.TestCase):
    def test_one_to_one_metrics_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_dir = root / 'gt'
            baseline_dir = root / 'baseline'
            e1_dir = root / 'e1'
            output_dir = root / 'report'
            for path in (gt_dir, baseline_dir, e1_dir):
                path.mkdir()
            (root / 'val.txt').write_text('000001\n')
            (gt_dir / '000001.txt').write_text(
                kitti_line(10.0, [0, 0, 50, 50])
                + kitti_line(30.0, [100, 0, 150, 50]))
            (baseline_dir / '000001.txt').write_text(
                kitti_line(13.0, [0, 0, 50, 50], 0.9))
            (e1_dir / '000001.txt').write_text(
                kitti_line(11.0, [0, 0, 50, 50], 0.9)
                + kitti_line(28.0, [100, 0, 150, 50], 0.8))

            argv = [
                'evaluate_e1_depth.py',
                '--ground-truth-dir', str(gt_dir),
                '--split-file', str(root / 'val.txt'),
                '--model', 'baseline=' + str(baseline_dir),
                '--model', 'pass2=' + str(e1_dir),
                '--output-dir', str(output_dir),
                '--skip-official-ap',
            ]
            with mock.patch.object(sys, 'argv', argv):
                EVALUATOR.main()

            report = json.loads((output_dir / 'val_metrics.json').read_text())
            self.assertEqual(report['models']['baseline']['matched_count'], 1)
            self.assertEqual(report['models']['pass2']['matched_count'], 2)
            self.assertAlmostEqual(report['models']['pass2']['mae'], 1.5)
            paired = report['paired_intersection']['pass2_vs_baseline']
            self.assertAlmostEqual(paired['delta_model_minus_baseline_mae'], -2.0)
            self.assertEqual(len(report['distance_buckets']), 12)
            self.assertTrue((output_dir / 'per_object_depth_errors.csv').is_file())
            self.assertTrue((output_dir / 'val_comparison.md').is_file())


if __name__ == '__main__':
    unittest.main()
