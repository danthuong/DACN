#!/usr/bin/env python3
"""Generate near-edge midpoint labels for the Z-map replacement ablation.

This is intentionally a new generator.  It preserves the original four
GPro3D-style touch points and adds one ``near_contact`` record per vehicle:
the 3-D midpoint of the two footprint candidates with the smallest camera-Z.
The midpoint is formed in 3-D and only then projected into the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from generate_geometry_pseudo_labels import build_frame_record, validate_ratio
from pseudo_label_generation.geometry import project_points


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/KITTIDataset")
    parser.add_argument("--split", default="train", choices=("train", "val", "trainval"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--touch-classes", default="Car,Van,Truck")
    parser.add_argument(
        "--plane-classes",
        default="Car,Van,Truck,Pedestrian,Cyclist,Tram,Person_sitting,Misc",
    )
    parser.add_argument(
        "--plane-point-mode",
        choices=("bottom_corners", "bottom_centers"),
        default="bottom_corners",
    )
    parser.add_argument("--k-l", type=float, default=0.65)
    parser.add_argument("--k-w", type=float, default=0.85)
    parser.add_argument("--max-bottom-y-gap", type=float, default=10.0)
    parser.add_argument("--min-xz-spread-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--frame-id", action="append", default=[])
    return parser.parse_args()


def add_near_contact(record):
    """Attach a geometrically consistent near-edge midpoint to every object."""
    p2 = np.asarray(record["calibration"]["P2"], dtype=np.float64)
    image_width, image_height = record["image_size"]
    valid_count = 0
    for obj in record.get("objects", []):
        candidates = []
        for point in obj.get("touch_points", []):
            camera = point.get("camera_3d")
            local = point.get("local_3d")
            if camera is None or local is None:
                continue
            camera = np.asarray(camera, dtype=np.float64)
            local = np.asarray(local, dtype=np.float64)
            if np.isfinite(camera).all() and np.isfinite(local).all():
                candidates.append((float(camera[2]), point["name"], local, camera))

        near = {
            "definition": "3d_midpoint_of_two_smallest_camera_z_footprint_candidates",
            "selected_names": [],
            "local_offset_3d": None,
            "camera_3d": None,
            "pixel_2d": None,
            "projection_valid": False,
            "inside_image": False,
        }
        if len(candidates) >= 2:
            selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:2]
            local_midpoint = np.mean([item[2] for item in selected], axis=0)
            camera_midpoint = np.mean([item[3] for item in selected], axis=0)
            pixels, projection_valid = project_points(camera_midpoint[None], p2)
            valid = bool(projection_valid[0])
            pixel = pixels[0]
            inside = bool(
                valid
                and 0.0 <= pixel[0] < image_width
                and 0.0 <= pixel[1] < image_height
            )
            near.update({
                "selected_names": [item[1] for item in selected],
                "local_offset_3d": local_midpoint.astype(float).tolist(),
                "camera_3d": camera_midpoint.astype(float).tolist(),
                "pixel_2d": pixel.astype(float).tolist() if valid else None,
                "projection_valid": valid,
                "inside_image": inside,
            })
            valid_count += int(valid)
        obj["near_contact"] = near

    record["schema_version"] = "near-gpro-pseudo-label-v1"
    record["near_contact_definition"] = {
        "selection_axis": "KITTI rect-camera Z (smaller is nearer)",
        "midpoint_space": "3-D camera space before projection",
        "depth_target": "KITTI bottom-centre camera-Z after yaw/dimension correction",
    }
    return valid_count


def main():
    args = parse_args()
    validate_ratio("k_l", args.k_l)
    validate_ratio("k_w", args.k_w)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_path = dataset_root / "ImageSets" / (args.split + ".txt")
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    frame_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if args.frame_id:
        selected = {str(value).zfill(6) for value in args.frame_id}
        frame_ids = [frame_id for frame_id in frame_ids if frame_id in selected]
    if args.limit is not None:
        frame_ids = frame_ids[: args.limit]

    output = Path(
        args.output
        or "outputs/near_gpro_pseudo_labels/{}.jsonl".format(args.split)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    touch_classes = [value.strip() for value in args.touch_classes.split(",") if value.strip()]
    plane_classes = [value.strip() for value in args.plane_classes.split(",") if value.strip()]
    total_objects = 0
    valid_objects = 0
    with output.open("w", encoding="utf-8") as target:
        for frame_id in frame_ids:
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
            total_objects += len(record.get("objects", []))
            valid_objects += add_near_contact(record)
            target.write(json.dumps(record, separators=(",", ":")) + "\n")
    print(
        "wrote {} frames to {} (near-contact valid: {}/{})".format(
            len(frame_ids), output, valid_objects, total_objects
        )
    )


if __name__ == "__main__":
    main()
