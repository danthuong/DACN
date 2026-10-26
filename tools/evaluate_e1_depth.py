#!/usr/bin/env python3
"""Fair class-constrained, one-to-one center-Z analysis for KITTI outputs."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


DISTANCE_BINS = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, math.inf))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth-dir', default='data/KITTIDataset/training/label_2')
    parser.add_argument('--split-file', default='data/KITTIDataset/ImageSets/val.txt')
    parser.add_argument('--model', action='append', required=True, metavar='NAME=DIR')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--class-name', default='Car')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument(
        '--skip-official-ap', action='store_true',
        help='skip the local KITTI AP evaluator (depth metrics are still produced)')
    return parser.parse_args()


def read_kitti(path, class_name):
    records = []
    if not path.is_file():
        return records
    for annotation_id, line in enumerate(path.read_text().splitlines()):
        fields = line.split()
        if len(fields) < 15 or fields[0] != class_name:
            continue
        bbox = np.asarray([float(value) for value in fields[4:8]], dtype=np.float64)
        truncation = float(fields[1])
        occlusion = int(float(fields[2]))
        height = bbox[3] - bbox[1]
        difficulties = []
        if height >= 40 and occlusion <= 0 and truncation <= 0.15:
            difficulties.append('easy')
        if height >= 25 and occlusion <= 1 and truncation <= 0.30:
            difficulties.append('moderate')
        if height >= 25 and occlusion <= 2 and truncation <= 0.50:
            difficulties.append('hard')
        records.append({
            'annotation_id': annotation_id,
            'class': fields[0],
            'truncation': truncation,
            'occlusion': occlusion,
            'bbox': bbox,
            'depth': float(fields[13]),
            'score': float(fields[15]) if len(fields) > 15 else 1.0,
            'difficulties': difficulties,
        })
    return records


def iou_matrix(left, right):
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=np.float64)
    a = np.stack([record['bbox'] for record in left])
    b = np.stack([record['bbox'] for record in right])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.maximum(a[:, 2:] - a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(b[:, 2:] - b[:, :2], 0), axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


def match_records(ground_truth, predictions, threshold):
    overlaps = iou_matrix(ground_truth, predictions)
    if overlaps.size == 0:
        return []
    gt_indices, pred_indices = linear_sum_assignment(1.0 - overlaps)
    return [
        (int(gt), int(pred), float(overlaps[gt, pred]))
        for gt, pred in zip(gt_indices, pred_indices)
        if overlaps[gt, pred] >= threshold
    ]


def metrics(errors, gt_depths, total_gt):
    errors = np.asarray(errors, dtype=np.float64)
    gt_depths = np.asarray(gt_depths, dtype=np.float64)
    if not len(errors):
        return {
            'matched_count': 0, 'mae': None, 'rmse': None, 'median_ae': None,
            'p90_ae': None, 'signed_bias': None, 'abs_rel': None,
            'recall': 0.0 if total_gt else None, 'missed_gt': total_gt,
        }
    absolute = np.abs(errors)
    return {
        'matched_count': int(len(errors)),
        'mae': float(absolute.mean()),
        'rmse': float(np.sqrt(np.square(errors).mean())),
        'median_ae': float(np.median(absolute)),
        'p90_ae': float(np.percentile(absolute, 90)),
        'signed_bias': float(errors.mean()),
        'abs_rel': float((absolute / np.maximum(gt_depths, 1e-6)).mean()),
        'recall': float(len(errors) / total_gt) if total_gt else None,
        'missed_gt': int(total_gt - len(errors)),
    }


def official_kitti_metrics(gt_dir, prediction_dir, frame_ids, class_name):
    """Run this repository's official evaluator on the exact same frame IDs."""
    from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
    import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti

    class_ids = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
    if class_name not in class_ids:
        raise ValueError('official KITTI AP is unavailable for class ' + class_name)
    numeric_ids = [int(value) for value in frame_ids]
    gt_annos = kitti.get_label_annos(str(gt_dir), numeric_ids)
    dt_annos = kitti.get_label_annos(str(prediction_dir), numeric_ids)
    official_result = get_official_eval_result(
        gt_annos, dt_annos, class_ids[class_name]
    )
    _, values = official_result[:2]


def evaluate(args):
    models = {}
    for item in args.model:
        if '=' not in item:
            raise ValueError('--model must be NAME=DIR')
        name, directory = item.split('=', 1)
        models[name] = Path(directory)
    frame_ids = [line.strip() for line in Path(args.split_file).read_text().splitlines() if line.strip()]
    gt_dir = Path(args.ground_truth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_gt = 0
    matches_by_model = {name: {} for name in models}
    for frame_id in frame_ids:
        gt = read_kitti(gt_dir / (frame_id + '.txt'), args.class_name)
        total_gt += len(gt)
        for name, directory in models.items():
            predictions = read_kitti(directory / (frame_id + '.txt'), args.class_name)
            matched = {
                gt_index: (pred_index, overlap)
                for gt_index, pred_index, overlap in match_records(gt, predictions, args.iou_threshold)
            }
            for gt_index, target in enumerate(gt):
                key = (frame_id, target['annotation_id'])
                if gt_index not in matched:
                    rows.append({
                        'model': name, 'image_id': frame_id,
                        'gt_id': target['annotation_id'], 'matched': False,
                        'gt_depth': target['depth'], 'pred_depth': '',
                        'signed_error': '', 'absolute_error': '',
                        'iou_2d': '', 'score': '',
                        'difficulty': '|'.join(target['difficulties']),
                    })
                    continue
                pred_index, overlap = matched[gt_index]
                prediction = predictions[pred_index]
                error = prediction['depth'] - target['depth']
                matches_by_model[name][key] = (
                    error, target['depth'], tuple(target['difficulties']))
                rows.append({
                    'model': name, 'image_id': frame_id,
                    'gt_id': target['annotation_id'], 'matched': True,
                    'gt_depth': target['depth'], 'pred_depth': prediction['depth'],
                    'signed_error': error, 'absolute_error': abs(error),
                    'iou_2d': overlap, 'score': prediction['score'],
                    'difficulty': '|'.join(target['difficulties']),
                })

    summary = {
        'protocol': {
            'split_file': str(Path(args.split_file)), 'class': args.class_name,
            'matching': 'class-constrained one-to-one Hungarian on 2D IoU',
            'iou_threshold': args.iou_threshold,
            'depth_definition': 'KITTI camera-axis Z of object bottom-center/center (same Z)',
            'total_gt': total_gt,
        },
        'models': {},
        'distance_buckets': [],
        'difficulty': {},
        'paired_intersection': {},
    }
    bucket_rows = []
    for name, matched in matches_by_model.items():
        errors = [value[0] for value in matched.values()]
        depths = [value[1] for value in matched.values()]
        summary['models'][name] = metrics(errors, depths, total_gt)
        performance_path = models[name].parent / 'performance.json'
        if performance_path.is_file():
            try:
                summary['models'][name]['performance'] = json.loads(
                    performance_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                summary['models'][name]['performance_error'] = str(error)
        if not args.skip_official_ap:
            try:
                official = official_kitti_metrics(
                    gt_dir, models[name], frame_ids, args.class_name)
                summary['models'][name]['official_kitti'] = official
                summary['models'][name]['ap3d_r40_moderate'] = official.get(
                    '{}_3d_moderate_R40'.format(args.class_name))
            except Exception as error:
                summary['models'][name]['official_kitti_error'] = str(error)
                summary['models'][name]['ap3d_r40_moderate'] = None
        for lower, upper in DISTANCE_BINS:
            selected = [(value[0], value[1]) for value in matched.values()
                        if lower <= value[1] < upper]
            label = '{}-{}'.format(lower, upper) if math.isfinite(upper) else '{}+'.format(lower)
            values = metrics([x[0] for x in selected], [x[1] for x in selected],
                             sum(lower <= row['gt_depth'] < upper for row in rows if row['model'] == name))
            bucket_rows.append({'model': name, 'distance_bucket_m': label, **values})
        summary['difficulty'][name] = {}
        for difficulty in ('easy', 'moderate', 'hard'):
            selected = [value for value in matched.values() if difficulty in value[2]]
            difficulty_total = sum(
                difficulty in row['difficulty'].split('|')
                for row in rows if row['model'] == name)
            summary['difficulty'][name][difficulty] = metrics(
                [value[0] for value in selected],
                [value[1] for value in selected], difficulty_total)
    summary['distance_buckets'] = bucket_rows

    names = list(models)
    if len(names) >= 2:
        baseline = names[0]
        for name in names[1:]:
            common = sorted(set(matches_by_model[baseline]) & set(matches_by_model[name]))
            baseline_ae = np.asarray([abs(matches_by_model[baseline][key][0]) for key in common])
            model_ae = np.asarray([abs(matches_by_model[name][key][0]) for key in common])
            summary['paired_intersection'][name + '_vs_' + baseline] = {
                'matched_count': len(common),
                'baseline_mae': float(baseline_ae.mean()) if len(common) else None,
                'model_mae': float(model_ae.mean()) if len(common) else None,
                'delta_model_minus_baseline_mae': float((model_ae - baseline_ae).mean()) if len(common) else None,
            }

    (output_dir / 'val_metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output_dir / 'per_object_depth_errors.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]) if rows else [
            'model', 'image_id', 'gt_id', 'matched', 'gt_depth', 'pred_depth',
            'signed_error', 'absolute_error', 'iou_2d', 'score', 'difficulty'])
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / 'depth_errors_by_distance.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(bucket_rows[0]) if bucket_rows else [])
        if bucket_rows:
            writer.writeheader(); writer.writerows(bucket_rows)
    with (output_dir / 'val_metrics.csv').open('w', newline='') as target:
        scalar_fields = (
            'matched_count', 'mae', 'rmse', 'median_ae', 'p90_ae',
            'signed_bias', 'abs_rel', 'recall', 'missed_gt',
            'ap3d_r40_moderate')
        flat = [
            {'model': name, **{key: value.get(key) for key in scalar_fields}}
            for name, value in summary['models'].items()
        ]
        writer = csv.DictWriter(target, fieldnames=list(flat[0]) if flat else [])
        if flat:
            writer.writeheader(); writer.writerows(flat)

    lines = [
        '# E1 validation depth comparison', '',
        'Protocol: one-to-one class-constrained 2D IoU >= {:.2f}; camera-Z; {} GT objects.'.format(
            args.iou_threshold, total_gt), '',
        '| Model | matched N | MAE | RMSE | MedAE | P90 | AbsRel | Recall | AP3D R40 Mod |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    def show(value):
        return '—' if value is None else '{:.4f}'.format(value)
    for name, value in summary['models'].items():
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            name, value['matched_count'], show(value['mae']), show(value['rmse']),
            show(value['median_ae']), show(value['p90_ae']), show(value['abs_rel']),
            show(value['recall']), show(value.get('ap3d_r40_moderate'))))
    lines.extend(['', 'Paired deltas are in `val_metrics.json`; negative model-minus-baseline MAE is improvement.', ''])
    (output_dir / 'val_comparison.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    return evaluate(parse_args())


if __name__ == '__main__':
    main()
