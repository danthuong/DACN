#!/usr/bin/env python3
"""Measure MonoDETR, Depth Predictor, and GPro3D depth MAE on KITTI val.

This is an offline metric tool: it reads inference artifacts and KITTI labels;
ground truth is never passed into the model. Predictions are matched to GT by
class-constrained one-to-one Hungarian assignment on 2D IoU. MAE is reported
only for matched detections with a valid branch value, together with coverage
so an invalid/missing GPro3D estimate is never silently treated as zero depth.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


BRANCH_FIELDS = (
    ("monodetr_final", "selected_depth_m", None),
    ("monodetr_pass1", "pass1_depth_m", None),
    ("depth_predictor_original", "depth_predictor_depth_m", None),
    ("depth_predictor_selected", "selected_depth_predictor_depth_m", None),
    ("gpro3d", "geometric_depth_m", "geometry_valid"),
)
KITTI_CLASS_IDS = {"Pedestrian": 0, "Car": 1, "Cyclist": 2}
GPRO3D_PAIRED_BRANCHES = (
    "monodetr_final",
    "monodetr_pass1",
    "depth_predictor_original",
    "depth_predictor_selected",
    "gpro3d",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", required=True,
        help="directory containing one inference artifact JSON per val frame")
    parser.add_argument(
        "--ground-truth-dir", default="data/KITTIDataset/training/label_2")
    parser.add_argument(
        "--split-file", default="data/KITTIDataset/ImageSets/val.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-name", default="Car")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow-missing-artifacts", action="store_true",
        help="evaluate the available subset instead of requiring the full split")
    return parser.parse_args()


def read_ground_truth(path, class_name):
    records = []
    if not path.is_file():
        return records
    for annotation_id, line in enumerate(path.read_text().splitlines()):
        fields = line.split()
        if len(fields) < 15 or fields[0] != class_name:
            continue
        records.append({
            "annotation_id": annotation_id,
            "bbox": np.asarray(
                [float(value) for value in fields[4:8]], dtype=np.float64),
            "depth": float(fields[13]),
        })
    return records


def read_artifact(path, class_name):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    records = []
    expected_class_id = KITTI_CLASS_IDS.get(class_name)
    for detection in payload.get("detections", []):
        detected_name = detection.get("class_name")
        if detected_name is not None:
            if detected_name != class_name:
                continue
        elif expected_class_id is not None:
            if int(detection.get("class_id", -1)) != expected_class_id:
                continue
        bbox = np.asarray(
            detection.get("bbox_xyxy_raw", []), dtype=np.float64).reshape(-1)
        if bbox.size != 4 or not np.isfinite(bbox).all():
            continue
        records.append({**detection, "bbox": bbox})
    return records


def iou_matrix(ground_truth, predictions):
    if not ground_truth or not predictions:
        return np.zeros((len(ground_truth), len(predictions)), dtype=np.float64)
    left = np.stack([record["bbox"] for record in ground_truth])
    right = np.stack([record["bbox"] for record in predictions])
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    size = np.maximum(bottom_right - top_left, 0.0)
    intersection = size[..., 0] * size[..., 1]
    left_area = np.prod(np.maximum(left[:, 2:] - left[:, :2], 0.0), axis=1)
    right_area = np.prod(np.maximum(right[:, 2:] - right[:, :2], 0.0), axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def match_records(ground_truth, predictions, iou_threshold):
    overlaps = iou_matrix(ground_truth, predictions)
    if overlaps.size == 0:
        return {}
    gt_indices, prediction_indices = linear_sum_assignment(1.0 - overlaps)
    return {
        int(gt_index): (int(prediction_index), float(overlaps[gt_index, prediction_index]))
        for gt_index, prediction_index in zip(gt_indices, prediction_indices)
        if overlaps[gt_index, prediction_index] >= iou_threshold
    }


def branch_value(detection, field, validity_field):
    if validity_field is not None and not bool(detection.get(validity_field, False)):
        return None
    value = detection.get(field)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0.0 else None


def summarize(values, gt_depths, detector_matched, total_gt):
    values = np.asarray(values, dtype=np.float64)
    gt_depths = np.asarray(gt_depths, dtype=np.float64)
    if not len(values):
        return {
            "valid_count": 0,
            "mae": None,
            "rmse": None,
            "median_ae": None,
            "p90_ae": None,
            "signed_bias": None,
            "abs_rel": None,
            "coverage_of_matched": 0.0 if detector_matched else None,
            "coverage_of_all_gt": 0.0 if total_gt else None,
        }
    errors = values - gt_depths
    absolute = np.abs(errors)
    return {
        "valid_count": int(len(values)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "median_ae": float(np.median(absolute)),
        "p90_ae": float(np.percentile(absolute, 90)),
        "signed_bias": float(errors.mean()),
        "abs_rel": float((absolute / np.maximum(gt_depths, 1e-6)).mean()),
        "coverage_of_matched": float(len(values) / detector_matched),
        "coverage_of_all_gt": float(len(values) / total_gt) if total_gt else None,
    }


def evaluate(args):
    artifact_dir = Path(args.artifact_dir)
    gt_dir = Path(args.ground_truth_dir)
    frame_ids = [
        line.strip() for line in Path(args.split_file).read_text().splitlines()
        if line.strip()
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_artifacts = []
    total_gt = 0
    detector_matched = 0
    branch_predictions = {name: [] for name, _, _ in BRANCH_FIELDS}
    branch_targets = {name: [] for name, _, _ in BRANCH_FIELDS}
    paired_predictions = {name: [] for name in GPRO3D_PAIRED_BRANCHES}
    paired_targets = []
    rows = []

    for frame_id in frame_ids:
        ground_truth = read_ground_truth(
            gt_dir / (frame_id + ".txt"), args.class_name)
        total_gt += len(ground_truth)
        detections = read_artifact(
            artifact_dir / (frame_id + ".json"), args.class_name)
        if detections is None:
            missing_artifacts.append(frame_id)
            detections = []
        matched = match_records(ground_truth, detections, args.iou_threshold)
        detector_matched += len(matched)

        for gt_index, target in enumerate(ground_truth):
            row = {
                "image_id": frame_id,
                "gt_id": target["annotation_id"],
                "gt_depth_m": target["depth"],
                "matched": gt_index in matched,
                "query_id": "",
                "class_score": "",
                "iou_2d": "",
            }
            detection = None
            if gt_index in matched:
                prediction_index, overlap = matched[gt_index]
                detection = detections[prediction_index]
                row.update({
                    "query_id": detection.get("query_id", ""),
                    "class_score": detection.get("class_score", ""),
                    "iou_2d": overlap,
                })
            values_by_branch = {}
            for branch_name, field, validity_field in BRANCH_FIELDS:
                value = (
                    branch_value(detection, field, validity_field)
                    if detection is not None else None
                )
                values_by_branch[branch_name] = value
                row[branch_name + "_depth_m"] = "" if value is None else value
                row[branch_name + "_absolute_error_m"] = (
                    "" if value is None else abs(value - target["depth"])
                )
                if value is not None:
                    branch_predictions[branch_name].append(value)
                    branch_targets[branch_name].append(target["depth"])
            if all(
                    values_by_branch.get(name) is not None
                    for name in GPRO3D_PAIRED_BRANCHES):
                paired_targets.append(target["depth"])
                for name in GPRO3D_PAIRED_BRANCHES:
                    paired_predictions[name].append(values_by_branch[name])
            rows.append(row)

    if missing_artifacts and not args.allow_missing_artifacts:
        preview = ", ".join(missing_artifacts[:10])
        raise FileNotFoundError(
            "missing {} of {} val artifacts (first: {}). Re-run full val export "
            "or pass --allow-missing-artifacts.".format(
                len(missing_artifacts), len(frame_ids), preview))

    branch_summary = {
        name: summarize(
            branch_predictions[name], branch_targets[name],
            detector_matched, total_gt)
        for name, _, _ in BRANCH_FIELDS
    }
    paired_count = len(paired_targets)
    paired_summary = {
        name: summarize(
            paired_predictions[name], paired_targets, paired_count, total_gt)
        for name in GPRO3D_PAIRED_BRANCHES
    }
    report = {
        "protocol": {
            "split_file": str(Path(args.split_file)),
            "artifact_dir": str(artifact_dir),
            "class": args.class_name,
            "matching": "class-constrained one-to-one Hungarian on 2D IoU",
            "iou_threshold": args.iou_threshold,
            "depth_definition": "KITTI camera-axis object center Z",
            "frame_count": len(frame_ids),
            "missing_artifact_count": len(missing_artifacts),
            "total_gt": total_gt,
            "detector_matched_count": detector_matched,
            "detector_recall": (
                float(detector_matched / total_gt) if total_gt else None),
            "note": (
                "MAE uses matched detections only; invalid GPro3D estimates "
                "are excluded and disclosed through coverage."),
        },
        "branches": branch_summary,
        "gpro3d_valid_paired_subset": {
            "count": paired_count,
            "note": (
                "All branches are evaluated on the exact objects where "
                "GPro3D and every compared branch have valid depth."),
            "branches": paired_summary,
        },
    }

    (output_dir / "val_depth_mae.json").write_text(
        json.dumps(report, indent=2) + "\n")
    with (output_dir / "per_object_depth_mae.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary_rows = [
        {"branch": name, **metrics} for name, metrics in branch_summary.items()
    ]
    with (output_dir / "val_depth_mae.csv").open("w", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=list(summary_rows[0]) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    def show(value):
        return "—" if value is None else "{:.4f}".format(value)

    lines = [
        "# KITTI val depth MAE", "",
        "{} frames, {} GT {}, {} matched detections (recall {}).".format(
            len(frame_ids), total_gt, args.class_name, detector_matched,
            show(report["protocol"]["detector_recall"])), "",
        "| Branch | valid N | MAE | RMSE | AbsRel | coverage/matched | coverage/all GT |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in branch_summary.items():
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            name, metrics["valid_count"], show(metrics["mae"]),
            show(metrics["rmse"]), show(metrics["abs_rel"]),
            show(metrics["coverage_of_matched"]),
            show(metrics["coverage_of_all_gt"])))
    lines.extend([
        "", "## Paired comparison on GPro3D-valid objects", "",
        "| Branch | paired N | MAE | RMSE | AbsRel |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, metrics in paired_summary.items():
        lines.append("| {} | {} | {} | {} | {} |".format(
            name, metrics["valid_count"], show(metrics["mae"]),
            show(metrics["rmse"]), show(metrics["abs_rel"])))
    lines.extend([
        "", "GPro3D invalid estimates are omitted, never converted to zero.", ""])
    (output_dir / "val_depth_mae.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))
    return report


def main():
    return evaluate(parse_args())


if __name__ == "__main__":
    main()
