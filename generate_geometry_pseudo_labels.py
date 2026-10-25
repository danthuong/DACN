#!/usr/bin/env python3
"""Generate GPro3D-style touch-point and horizon pseudo-labels from KITTI GT."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from lib.datasets.kitti.kitti_utils import Calibration, get_objects_from_label
from pseudo_label_generation.geometry import (
    TOUCH_POINT_NAMES,
    derive_horizon_line,
    fit_ground_plane,
    project_points,
    transform_touch_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/KITTIDataset", help="KITTI root containing ImageSets")
    parser.add_argument("--split", default="val", choices=("train", "val", "trainval"))
    parser.add_argument("--output", default=None, help="Output JSONL (default: outputs/geometry_pseudo_labels/<split>.jsonl)")
    parser.add_argument(
        "--touch-classes", "--classes", dest="touch_classes", default="Car,Van,Truck",
        help="Comma-separated classes that receive four touch points",
    )
    parser.add_argument(
        "--plane-classes",
        default="Car,Van,Truck,Pedestrian,Cyclist,Tram,Person_sitting,Misc",
        help="Comma-separated ground-contact classes used to fit the plane",
    )
    parser.add_argument(
        "--plane-point-mode",
        choices=("bottom_corners", "bottom_centers"),
        default="bottom_corners",
        help="Experimental bottom corners allow one object; centers follow GPro3D",
    )
    parser.add_argument("--k-l", type=float, default=0.65, help="Wheelbase / box-length ratio")
    parser.add_argument("--k-w", type=float, default=0.85, help="Track-width / box-width ratio")
    parser.add_argument(
        "--max-bottom-y-gap", type=float, default=10.0,
        help="Maximum camera-Y range (metres) in the selected bottom-centre consensus",
    )
    parser.add_argument(
        "--min-xz-spread-ratio", type=float, default=0.05,
        help="Minimum minor/major singular-value ratio of selected X-Z samples",
    )
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N frames (debugging)")
    parser.add_argument("--frame-id", action="append", default=[], help="Generate only this frame ID; repeatable")
    return parser.parse_args()


def validate_ratio(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0 or value > 1.5:
        raise ValueError(f"{name} must be finite and in (0, 1.5], got {value}")


def build_frame_record(
    frame_id: str,
    dataset_root: Path,
    touch_classes: List[str],
    plane_classes: List[str],
    k_l: float,
    k_w: float,
    max_bottom_y_gap: float,
    min_xz_spread_ratio: float,
    plane_point_mode: str,
) -> Dict:
    data_dir = dataset_root / "training"
    image_path = data_dir / "image_2" / f"{frame_id}.png"
    label_path = data_dir / "label_2" / f"{frame_id}.txt"
    calib_path = data_dir / "calib" / f"{frame_id}.txt"
    for required in (image_path, label_path, calib_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    with Image.open(str(image_path)) as image:
        image_size = [int(image.width), int(image.height)]
    calibration = Calibration(str(calib_path))
    objects = get_objects_from_label(str(label_path))
    vehicle_records = []
    plane_centers = []
    plane_bottom_corners = []
    plane_candidate_object_ids = []

    for object_id, obj in enumerate(objects):
        dimensions = np.asarray([obj.h, obj.w, obj.l], dtype=np.float64)
        location = np.asarray(obj.pos, dtype=np.float64)
        if not np.isfinite(np.r_[dimensions, location, obj.ry]).all():
            continue
        if np.any(dimensions <= 0.0) or location[2] <= 1e-6:
            continue

        if obj.cls_type in plane_classes:
            _, bottom_corners = transform_touch_points(
                obj.l, obj.w, location, obj.ry, 1.0, 1.0
            )
            plane_centers.append(location.tolist())
            plane_bottom_corners.append(bottom_corners)
            plane_candidate_object_ids.append(object_id)
        if obj.cls_type not in touch_classes:
            continue

        local_points, camera_points = transform_touch_points(
            obj.l, obj.w, location, obj.ry, k_l, k_w
        )
        pixels, projection_valid = project_points(camera_points, calibration.P2)
        touch_points = []
        for point_index, point_name in enumerate(TOUCH_POINT_NAMES):
            valid = bool(projection_valid[point_index])
            pixel = pixels[point_index]
            inside = bool(
                valid
                and 0.0 <= pixel[0] < image_size[0]
                and 0.0 <= pixel[1] < image_size[1]
            )
            touch_points.append(
                {
                    "name": point_name,
                    "local_3d": local_points[point_index].tolist(),
                    "camera_3d": camera_points[point_index].tolist(),
                    "pixel_2d": pixel.tolist() if valid else None,
                    "projection_valid": valid,
                    "inside_image": inside,
                    "visibility": "unknown" if valid else "invalid",
                }
            )

        vehicle_records.append(
            {
                "object_id": object_id,
                "class": obj.cls_type,
                "bbox_2d": obj.box2d.astype(float).tolist(),
                "dimensions_3d": dimensions.tolist(),
                "location_3d": location.tolist(),
                "rotation_y": float(obj.ry),
                "truncation": float(obj.trucation),
                "occlusion": int(obj.occlusion),
                "touch_points": touch_points,
            }
        )
    # Height consensus is selected per object centre so a box contributes all
    # four corners or none of them.
    consensus = fit_ground_plane(
        plane_centers,
        max_y_gap=max_bottom_y_gap,
        min_xz_spread_ratio=0.0,
    )
    selected_indices = consensus["sample_indices"]
    rejected_indices = consensus["rejected_indices"]
    if plane_point_mode == "bottom_corners":
        fit_points = (
            np.concatenate([plane_bottom_corners[index] for index in selected_indices], axis=0)
            if selected_indices
            else np.empty((0, 3), dtype=np.float64)
        )
    else:
        fit_points = np.asarray(
            [plane_centers[index] for index in selected_indices], dtype=np.float64
        ).reshape(-1, 3)

    ground_plane = fit_ground_plane(
        fit_points,
        min_xz_spread_ratio=min_xz_spread_ratio,
    )
    ground_plane.pop("sample_indices")
    ground_plane.pop("rejected_indices")
    selected_centers = np.asarray(
        [plane_centers[index] for index in selected_indices], dtype=np.float64
    ).reshape(-1, 3)
    ground_plane["candidate_count"] = len(plane_centers)
    ground_plane["sample_count"] = len(selected_indices)
    ground_plane["plane_point_count"] = len(fit_points)
    ground_plane["bottom_y_gap"] = (
        float(np.ptp(selected_centers[:, 1])) if len(selected_centers) else None
    )
    ground_plane["sample_object_ids"] = [plane_candidate_object_ids[index] for index in selected_indices]
    ground_plane["rejected_object_ids"] = [plane_candidate_object_ids[index] for index in rejected_indices]
    if ground_plane["valid"]:
        horizon_line = derive_horizon_line(ground_plane["coefficients"], calibration.P2, image_size)
    else:
        horizon_line = {
            "valid": False,
            "slope": None,
            "intercept": None,
            "visible_segment": None,
            "reason": "invalid_ground_plane",
        }

    return {
        "schema_version": "geometry-pseudo-label-v1",
        "frame_id": frame_id,
        "image_path": str(Path("training") / "image_2" / f"{frame_id}.png"),
        "image_size": image_size,
        "calibration": {
            "P2": calibration.P2.astype(float).tolist(),
            "K": calibration.P2[:, :3].astype(float).tolist(),
        },
        "parameters": {
            "k_l": k_l,
            "k_w": k_w,
            "touch_classes": touch_classes,
            "plane_classes": plane_classes,
            "plane_point_mode": plane_point_mode,
            "max_bottom_y_gap": max_bottom_y_gap,
            "min_xz_spread_ratio": min_xz_spread_ratio,
            "coordinate_system": "KITTI rect camera: X right, Y down, Z forward",
            "yaw_conversion": "theta_paper = -rotation_y; equivalent KITTI R_y applied",
            "projection": "full KITTI P2 (3x4)",
        },
        "objects": vehicle_records,
        "ground_plane": ground_plane,
        "horizon_line": horizon_line,
    }


def main() -> None:
    args = parse_args()
    validate_ratio("k_l", args.k_l)
    validate_ratio("k_w", args.k_w)
    if not np.isfinite(args.max_bottom_y_gap) or args.max_bottom_y_gap <= 0:
        raise ValueError("--max-bottom-y-gap must be positive and finite")
    if not np.isfinite(args.min_xz_spread_ratio) or not 0 <= args.min_xz_spread_ratio < 1:
        raise ValueError("--min-xz-spread-ratio must be finite and in [0, 1)")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_path = dataset_root / "ImageSets" / f"{args.split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    frame_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if args.frame_id:
        requested = {str(value).zfill(6) for value in args.frame_id}
        frame_ids = [frame_id for frame_id in frame_ids if frame_id in requested]
        missing = requested.difference(frame_ids)
        if missing:
            raise ValueError(f"Frame(s) not present in {args.split}: {sorted(missing)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        frame_ids = frame_ids[: args.limit]
    touch_classes = [value.strip() for value in args.touch_classes.split(",") if value.strip()]
    plane_classes = [value.strip() for value in args.plane_classes.split(",") if value.strip()]
    if not touch_classes:
        raise ValueError("--touch-classes must contain at least one class")
    if not plane_classes:
        raise ValueError("--plane-classes must contain at least one class")

    output_path = Path(args.output or f"outputs/geometry_pseudo_labels/{args.split}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    valid_planes = 0
    object_count = 0
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for index, frame_id in enumerate(frame_ids, start=1):
            record = build_frame_record(
                frame_id,
                dataset_root,
                touch_classes,
                plane_classes,
                args.k_l,
                args.k_w,
                args.max_bottom_y_gap,
                args.min_xz_spread_ratio,
                args.plane_point_mode,
            )
            output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            valid_planes += int(record["ground_plane"]["valid"])
            object_count += len(record["objects"])
            if index % 250 == 0 or index == len(frame_ids):
                print(f"[{index}/{len(frame_ids)}] generated {frame_id}")
    os.replace(str(temporary_path), str(output_path))
    print(
        f"Saved {len(frame_ids)} frames, {object_count} vehicles, "
        f"{valid_planes} valid planes to {output_path}"
    )


if __name__ == "__main__":
    main()

