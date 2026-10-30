#!/usr/bin/env python3
"""Audit every scalar depth term in the near-contact Z-map ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_val_depth_mae import (
    KITTI_CLASS_IDS,
    match_records,
    read_ground_truth,
    summarize,
)
from tools.evaluate_e1_depth import official_kitti_metrics


BRANCHES = (
    ("final", "selected_depth_m", None),
    ("baseline_original_formula", "baseline_depth_m", None),
    ("query_reg", "query_reg_depth_m", None),
    ("size_geometry", "size_depth_m", None),
    ("zmap_original", "selected_depth_predictor_depth_m", None),
    ("gpro_center", "geometric_depth_m", "geometry_valid"),
    ("selected_cue", "gpro_cue_depth_m", None),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--pseudo-labels", default="outputs/near_gpro_pseudo_labels/val.jsonl")
    parser.add_argument("--ground-truth-dir", default="data/KITTIDataset/training/label_2")
    parser.add_argument("--split-file", default="data/KITTIDataset/ImageSets/val.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prediction-dir", default=None,
        help="KITTI data directory (default: ARTIFACT_DIR/../data)",
    )
    parser.add_argument("--class-name", default="Car")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument("--skip-official-ap", action="store_true")
    return parser.parse_args()


def load_pseudo(path):
    records = {}
    source = Path(path)
    if not source.is_file():
        return records
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["frame_id"]).zfill(6)] = record
    return records


def read_artifact(path, class_name):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    class_id = KITTI_CLASS_IDS.get(class_name)
    result = []
    for item in payload.get("detections", []):
        if item.get("class_name") not in (None, class_name):
            continue
        if item.get("class_name") is None and int(item.get("class_id", -1)) != class_id:
            continue
        bbox = np.asarray(item.get("bbox_xyxy_raw", []), dtype=np.float64).reshape(-1)
        if bbox.size == 4 and np.isfinite(bbox).all():
            result.append({**item, "bbox": bbox})
    return result


def finite_depth(record, field, validity=None):
    if record is None or (validity and not bool(record.get(validity, False))):
        return None
    value = record.get(field)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def near_label(pseudo_records, frame_id, annotation_id):
    objects = (pseudo_records.get(frame_id) or {}).get("objects", [])
    for item in objects:
        if int(item.get("object_id", -1)) == int(annotation_id):
            near = item.get("near_contact") or {}
            if near.get("projection_valid") and near.get("pixel_2d") is not None:
                return np.asarray(near["pixel_2d"], dtype=np.float64)
    return None


def evaluate(args):
    artifact_dir = Path(args.artifact_dir)
    gt_dir = Path(args.ground_truth_dir)
    frame_ids = [
        line.strip() for line in Path(args.split_file).read_text().splitlines()
        if line.strip()
    ]
    pseudo = load_pseudo(args.pseudo_labels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {name: [] for name, _, _ in BRANCHES}
    targets = {name: [] for name, _, _ in BRANCHES}
    rows, missing = [], []
    detector_matched = total_gt = 0
    baseline_residuals, final_residuals, near_errors = [], [], []
    paired = []

    for frame_id in frame_ids:
        gt = read_ground_truth(gt_dir / (frame_id + ".txt"), args.class_name)
        total_gt += len(gt)
        detections = read_artifact(artifact_dir / (frame_id + ".json"), args.class_name)
        if detections is None:
            missing.append(frame_id)
            detections = []
        matched = match_records(gt, detections, args.iou_threshold)
        detector_matched += len(matched)
        for gt_index, target in enumerate(gt):
            detection = detections[matched[gt_index][0]] if gt_index in matched else None
            row = {
                "image_id": frame_id,
                "gt_id": target["annotation_id"],
                "gt_depth_m": target["depth"],
                "matched": detection is not None,
                "iou_2d": matched[gt_index][1] if gt_index in matched else "",
                "query_id": detection.get("query_id", "") if detection else "",
            }
            values = {}
            for name, field, validity in BRANCHES:
                value = finite_depth(detection, field, validity)
                values[name] = value
                row[name + "_depth_m"] = "" if value is None else value
                row[name + "_ae_m"] = "" if value is None else abs(value - target["depth"])
                if value is not None:
                    predictions[name].append(value)
                    targets[name].append(target["depth"])

            required = ("final", "baseline_original_formula", "query_reg",
                        "size_geometry", "zmap_original", "selected_cue")
            if all(values[name] is not None for name in required):
                baseline_residuals.append(values["baseline_original_formula"] - (
                    values["query_reg"] + values["size_geometry"] + values["zmap_original"]
                ) / 3.0)
                final_residuals.append(values["final"] - (
                    values["query_reg"] + values["size_geometry"] + values["selected_cue"]
                ) / 3.0)
            if values["gpro_center"] is not None and values["zmap_original"] is not None:
                paired.append({
                    "gpro_ae": abs(values["gpro_center"] - target["depth"]),
                    "zmap_ae": abs(values["zmap_original"] - target["depth"]),
                    "final_ae": abs(values["final"] - target["depth"])
                        if values["final"] is not None else None,
                    "baseline_ae": abs(values["baseline_original_formula"] - target["depth"])
                        if values["baseline_original_formula"] is not None else None,
                })

            label = near_label(pseudo, frame_id, target["annotation_id"])
            predicted_raw = None
            if detection is not None:
                raw = np.asarray(detection.get("touchpoints_raw", []), dtype=np.float64).reshape(-1, 2)
                if len(raw) and np.isfinite(raw[0]).all():
                    predicted_raw = raw[0]
            if label is not None and predicted_raw is not None:
                error = float(np.linalg.norm(predicted_raw - label))
                near_errors.append(error)
                row["near_touchpoint_l2_px"] = error
            else:
                row["near_touchpoint_l2_px"] = ""
            rows.append(row)

    if missing and not args.allow_missing_artifacts:
        raise FileNotFoundError(
            "missing {} artifacts (first: {})".format(len(missing), ", ".join(missing[:10]))
        )
    branch_summary = {
        name: summarize(predictions[name], targets[name], detector_matched, total_gt)
        for name, _, _ in BRANCHES
    }
    paired_valid = [item for item in paired if item["final_ae"] is not None
                    and item["baseline_ae"] is not None]
    report = {
        "protocol": {
            "class": args.class_name,
            "matching": "class-constrained one-to-one Hungarian on raw-image 2D IoU",
            "iou_threshold": args.iou_threshold,
            "depth_definition": "KITTI rect-camera bottom-centre Z",
            "total_gt": total_gt,
            "detector_matched_count": detector_matched,
            "detector_recall": detector_matched / total_gt if total_gt else None,
            "missing_artifact_count": len(missing),
        },
        "branches": branch_summary,
        "paired_gpro_valid": {
            "count": len(paired_valid),
            "gpro_better_than_zmap_fraction": float(np.mean([
                item["gpro_ae"] < item["zmap_ae"] for item in paired_valid
            ])) if paired_valid else None,
            "mean_gpro_minus_zmap_ae_m": float(np.mean([
                item["gpro_ae"] - item["zmap_ae"] for item in paired_valid
            ])) if paired_valid else None,
            "final_better_than_baseline_fraction": float(np.mean([
                item["final_ae"] < item["baseline_ae"] for item in paired_valid
            ])) if paired_valid else None,
            "mean_final_minus_baseline_ae_m": float(np.mean([
                item["final_ae"] - item["baseline_ae"] for item in paired_valid
            ])) if paired_valid else None,
        },
        "near_touchpoint": {
            "valid_count": len(near_errors),
            "mean_l2_px": float(np.mean(near_errors)) if near_errors else None,
            "median_l2_px": float(np.median(near_errors)) if near_errors else None,
            "p90_l2_px": float(np.percentile(near_errors, 90)) if near_errors else None,
        },
        "formula_invariants": {
            "baseline_max_abs_residual_m": float(np.max(np.abs(baseline_residuals)))
                if baseline_residuals else None,
            "final_max_abs_residual_m": float(np.max(np.abs(final_residuals)))
                if final_residuals else None,
            "expected": "both residuals should be near floating-point zero",
        },
    }
    if not args.skip_official_ap:
        prediction_dir = Path(args.prediction_dir) if args.prediction_dir else (
            artifact_dir.parent / "data"
        )
        try:
            official = official_kitti_metrics(
                gt_dir, prediction_dir, frame_ids, args.class_name
            )
            report["official_kitti"] = official
            report["ap3d_r40_moderate"] = official.get(
                "{}_3d_moderate_R40".format(args.class_name)
            )
        except Exception as error:
            report["official_kitti_error"] = str(error)
            report["ap3d_r40_moderate"] = None
    (output_dir / "near_gpro_depth_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if rows:
        with (output_dir / "near_gpro_per_object.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "near_gpro_branch_summary.csv").open("w", newline="") as target:
        summary_rows = [{"branch": name, **value} for name, value in branch_summary.items()]
        writer = csv.DictWriter(target, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    def display(value):
        return "—" if value is None else "{:.4f}".format(value)
    lines = [
        "# Near-GPro Z-map replacement report", "",
        "Matched {} / {} GT {} objects (recall {}).".format(
            detector_matched, total_gt, args.class_name,
            display(report["protocol"]["detector_recall"]),
        ), "",
        "| Branch | valid N | MAE | RMSE | median AE | P90 AE | bias | coverage/matched |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in branch_summary.items():
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, value["valid_count"], display(value["mae"]),
            display(value["rmse"]), display(value["median_ae"]),
            display(value["p90_ae"]), display(value["signed_bias"]),
            display(value["coverage_of_matched"]),
        ))
    lines.extend([
        "", "## Paired decision", "",
        "- GPro − Zmap mean AE: {} m (negative is better).".format(
            display(report["paired_gpro_valid"]["mean_gpro_minus_zmap_ae_m"])),
        "- Final − baseline mean AE: {} m (negative is better).".format(
            display(report["paired_gpro_valid"]["mean_final_minus_baseline_ae_m"])),
        "- Near-touchpoint mean L2: {} px.".format(
            display(report["near_touchpoint"]["mean_l2_px"])),
        "- KITTI AP3D R40 Moderate: {}.".format(
            display(report.get("ap3d_r40_moderate"))),
        "- Formula residuals are recorded in the JSON and must be ~0.", "",
    ])
    (output_dir / "near_gpro_depth_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
