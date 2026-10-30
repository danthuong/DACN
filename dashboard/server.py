#!/usr/bin/env python3
"""Dependency-light localhost server for the geometry debug dashboard."""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import mimetypes
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.datasets.kitti.kitti_utils import Calibration, get_objects_from_label
from pseudo_label_generation.geometry import backproject_pixels_to_plane, project_points


STATIC_ROOT = Path(__file__).resolve().parent / "static"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/KITTIDataset")
    parser.add_argument("--split", default="val", choices=("train", "val", "trainval", "test"))
    parser.add_argument("--pseudo-labels", default=None)
    parser.add_argument("--models", default="dashboard/model_registry.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", help="live inference device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--topk", type=int, default=50)
    return parser.parse_args()


def box_corners_camera(dimensions, location, rotation_y) -> np.ndarray:
    height, width, length = [float(value) for value in dimensions]
    x = np.asarray([length / 2, length / 2, -length / 2, -length / 2] * 2)
    y = np.asarray([0.0] * 4 + [-height] * 4)
    z = np.asarray([width / 2, -width / 2, -width / 2, width / 2] * 2)
    cosine, sine = np.cos(rotation_y), np.sin(rotation_y)
    rotation = np.asarray([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    return np.column_stack((x, y, z)) @ rotation.T + np.asarray(location, dtype=float)


def serialise_box(record: Dict, p2: np.ndarray) -> Dict:
    corners_camera = box_corners_camera(record["dimensions_3d"], record["location_3d"], record["rotation_y"])
    pixels, valid = project_points(corners_camera, p2)
    record["corners_camera"] = corners_camera.tolist()
    record["corners_2d"] = [pixels[i].tolist() if valid[i] else None for i in range(8)]
    record["projection_valid"] = bool(valid.all())
    return record


def object_to_record(obj, object_id: int, p2: np.ndarray, source: str) -> Dict:
    score = float(obj.score) if obj.score >= 0 else 1.0
    return serialise_box(
        {
            "id": object_id,
            "source": source,
            "class": obj.cls_type,
            "confidence": score,
            "bbox_2d": obj.box2d.astype(float).tolist(),
            "dimensions_3d": [float(obj.h), float(obj.w), float(obj.l)],
            "location_3d": obj.pos.astype(float).tolist(),
            "rotation_y": float(obj.ry),
        },
        p2,
    )


def ground_reprojections(pseudo_labels: Optional[Dict], p2: np.ndarray):
    """Yield ``(object, touch_point, intersection)`` for valid ground rays."""
    if pseudo_labels is None:
        return
    plane = pseudo_labels.get("ground_plane") or {}
    coefficients = plane.get("coefficients") if plane.get("valid") else None
    if coefficients is None:
        return

    for obj in pseudo_labels.get("objects", []):
        touch_points = obj.get("touch_points", [])
        usable_indices = [
            index
            for index, point in enumerate(touch_points)
            if point.get("projection_valid") and point.get("pixel_2d") is not None
        ]
        if not usable_indices:
            continue
        pixels = np.asarray([touch_points[index]["pixel_2d"] for index in usable_indices], dtype=float)
        intersections, valid = backproject_pixels_to_plane(pixels, p2, coefficients)
        for local_index, point_index in enumerate(usable_indices):
            if valid[local_index]:
                yield obj, touch_points[point_index], intersections[local_index]


def add_ground_reprojections(pseudo_labels: Optional[Dict], p2: np.ndarray) -> Optional[Dict]:
    """Attach ray/ground-plane intersections to a pseudo-label record."""
    if pseudo_labels is None:
        return None
    result = copy.deepcopy(pseudo_labels)
    for obj in result.get("objects", []):
        for point in obj.get("touch_points", []):
            point["ground_reprojection_valid"] = False
            point["reprojected_ground_3d"] = None
            point["ground_shift_3d"] = None

    for _, point, reprojected in ground_reprojections(result, p2):
        point["ground_reprojection_valid"] = True
        point["reprojected_ground_3d"] = reprojected.astype(float).tolist()
        original = np.asarray(point.get("camera_3d"), dtype=float)
        if original.shape == (3,) and np.isfinite(original).all():
            point["ground_shift_3d"] = (reprojected - original).astype(float).tolist()
    return result


def calculate_reprojection_metrics(pseudo_records: Dict[str, Dict]) -> Dict:
    """Calculate split-wide 3D touch-point error after ground reprojection."""
    absolute_sum = np.zeros(3, dtype=np.float64)
    euclidean_sum = 0.0
    point_count = 0
    valid_frame_count = 0
    valid_plane_frames = 0

    for record in pseudo_records.values():
        if (record.get("ground_plane") or {}).get("valid"):
            valid_plane_frames += 1
        p2 = (record.get("calibration") or {}).get("P2")
        if p2 is None:
            continue
        frame_has_point = False
        for _, point, reprojected in ground_reprojections(record, np.asarray(p2, dtype=float)):
            original = np.asarray(point.get("camera_3d"), dtype=float)
            if original.shape != (3,) or not np.isfinite(original).all():
                continue
            delta = reprojected - original
            absolute_sum += np.abs(delta)
            euclidean_sum += float(np.linalg.norm(delta))
            point_count += 1
            frame_has_point = True
        valid_frame_count += int(frame_has_point)

    mae_xyz = absolute_sum / point_count if point_count else np.full(3, np.nan)
    return {
        "split_frame_count": len(pseudo_records),
        "valid_plane_frame_count": valid_plane_frames,
        "valid_reprojection_frame_count": valid_frame_count,
        "point_count": point_count,
        "mae_x": float(mae_xyz[0]) if point_count else None,
        "mae_y": float(mae_xyz[1]) if point_count else None,
        "mae_z": float(mae_xyz[2]) if point_count else None,
        "mean_coordinate_mae": float(mae_xyz.mean()) if point_count else None,
        "mean_euclidean_error": euclidean_sum / point_count if point_count else None,
    }


class DashboardData:
    def __init__(self, args: argparse.Namespace):
        self.dataset_root = Path(args.dataset_root).expanduser()
        if not self.dataset_root.is_absolute():
            self.dataset_root = (REPO_ROOT / self.dataset_root).resolve()
        self.split = args.split
        split_file = self.dataset_root / "ImageSets" / f"{self.split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(split_file)
        self.frame_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.frame_set = set(self.frame_ids)
        self.data_dir = self.dataset_root / ("testing" if self.split == "test" else "training")

        if args.pseudo_labels:
            pseudo_path = Path(args.pseudo_labels)
        else:
            # Prefer the extended records because they contain the near-edge
            # midpoint GT requested by the near-GPro experiment. They retain
            # the original four wheel contacts and horizon/plane labels too.
            near_path = REPO_ROOT / f"outputs/near_gpro_pseudo_labels/{self.split}.jsonl"
            pseudo_path = near_path if near_path.is_file() else Path(
                f"outputs/geometry_pseudo_labels/{self.split}.jsonl"
            )
        if not pseudo_path.is_absolute():
            pseudo_path = REPO_ROOT / pseudo_path
        self.pseudo_path = pseudo_path
        self.pseudo_records = self._load_pseudo_records(pseudo_path)
        self.reprojection_metrics = calculate_reprojection_metrics(self.pseudo_records)

        registry_path = Path(args.models)
        if not registry_path.is_absolute():
            registry_path = REPO_ROOT / registry_path
        self.models = self._load_models(registry_path)

    @staticmethod
    def _load_pseudo_records(path: Path) -> Dict[str, Dict]:
        records = {}
        if not path.is_file():
            return records
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    records[str(record["frame_id"]).zfill(6)] = record
                except (json.JSONDecodeError, KeyError) as error:
                    raise ValueError(f"Invalid pseudo-label JSONL at {path}:{line_number}: {error}")
        return records

    @staticmethod
    def _load_models(path: Path) -> List[Dict]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = []
        for model in payload.get("models", []):
            if model.get("format") != "kitti":
                raise ValueError(f"Unsupported model format for {model.get('id')}: {model.get('format')}")
            model = dict(model)
            prediction_dir = model.get("prediction_dir")
            if prediction_dir:
                prediction_dir = Path(prediction_dir)
                if not prediction_dir.is_absolute():
                    prediction_dir = REPO_ROOT / prediction_dir
                model["prediction_dir"] = prediction_dir.resolve()
            else:
                model["prediction_dir"] = None
            for path_key in ("artifact_dir", "metrics_path"):
                if model.get(path_key):
                    value = Path(model[path_key])
                    if not value.is_absolute():
                        value = REPO_ROOT / value
                    model[path_key] = value.resolve()
            inference = dict(model.get("inference") or {})
            config_path = inference.get("config")
            if config_path:
                config_path = Path(config_path)
                if not config_path.is_absolute():
                    config_path = REPO_ROOT / config_path
                inference["config_path"] = config_path.resolve()
            weights = []
            seen_paths = set()
            declared = list(inference.get("weights") or [])
            for pattern in inference.get("checkpoint_globs") or []:
                pattern_path = Path(pattern)
                absolute_pattern = pattern if pattern_path.is_absolute() else str(REPO_ROOT / pattern)
                declared.extend({"path": value} for value in sorted(glob.glob(absolute_pattern, recursive=True)))
            for index, item in enumerate(declared):
                item = {"path": item} if isinstance(item, str) else dict(item)
                checkpoint = Path(item["path"])
                if not checkpoint.is_absolute():
                    checkpoint = REPO_ROOT / checkpoint
                checkpoint = checkpoint.resolve()
                if checkpoint in seen_paths or not checkpoint.is_file():
                    continue
                seen_paths.add(checkpoint)
                identifier = item.get("id") or "w_" + hashlib.sha1(
                    str(checkpoint).encode("utf-8")
                ).hexdigest()[:10]
                weights.append({
                    "id": str(identifier),
                    "label": str(item.get("label") or checkpoint.stem),
                    "path": checkpoint,
                    "relative_path": str(checkpoint.relative_to(REPO_ROOT))
                        if REPO_ROOT in checkpoint.parents else str(checkpoint),
                })
            inference["weights"] = weights
            model["inference"] = inference
            has_predictions = bool(
                prediction_dir and prediction_dir.is_dir()
                and any(prediction_dir.glob("*.txt"))
            )
            model["precomputed_available"] = has_predictions
            model["live_available"] = bool(weights)
            model["available"] = has_predictions or bool(weights)
            models.append(model)
        return models

    def model_by_id(self, model_id: str) -> Dict:
        for model in self.models:
            if model["id"] == model_id:
                return model
        raise KeyError("unknown model: " + model_id)

    @staticmethod
    def weight_by_id(model: Dict, weight_id: str) -> Dict:
        for weight in (model.get("inference") or {}).get("weights", []):
            if weight["id"] == weight_id:
                return weight
        raise KeyError("unknown weight {} for {}".format(weight_id, model["id"]))

    def _validation_reports(self) -> List[Dict]:
        reports = []
        seen = set()
        for model in self.models:
            path = model.get("metrics_path")
            if path is None or path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                reports.append({
                    "path": str(path.relative_to(REPO_ROOT))
                        if REPO_ROOT in path.parents else str(path),
                    "report": json.loads(path.read_text(encoding="utf-8")),
                })
            except (OSError, json.JSONDecodeError) as error:
                reports.append({"path": str(path), "error": str(error)})
        return reports

    def public_config(self) -> Dict:
        public_models = []
        for model in self.models:
            inference = model.get("inference") or {}
            public_models.append({
                "id": model["id"],
                "label": model.get("label", model["id"]),
                "color": model.get("color", "#ffffff"),
                "available": model["available"],
                "precomputed_available": model["precomputed_available"],
                "live_available": model["live_available"],
                "weights": [
                    {key: weight[key] for key in ("id", "label", "relative_path")}
                    for weight in inference.get("weights", [])
                ],
                "default_weight": inference.get("default_weight"),
            })
        return {
            "split": self.split,
            "frames": self.frame_ids,
            "pseudo_labels_available": len(self.pseudo_records),
            "reprojection_metrics": self.reprojection_metrics,
            "validation_reports": self._validation_reports(),
            "models": public_models,
        }

    def image_path(self, frame_id: str) -> Path:
        return self.data_dir / "image_2" / f"{frame_id}.png"

    def frame(self, frame_id: str) -> Dict:
        if frame_id not in self.frame_set:
            raise KeyError(frame_id)
        image_path = self.image_path(frame_id)
        calib_path = self.data_dir / "calib" / f"{frame_id}.txt"
        if not image_path.is_file() or not calib_path.is_file():
            raise FileNotFoundError(frame_id)
        with Image.open(str(image_path)) as image:
            image_size = [int(image.width), int(image.height)]
        calibration = Calibration(str(calib_path))

        ground_truth = []
        label_path = self.data_dir / "label_2" / f"{frame_id}.txt"
        if label_path.is_file():
            ground_truth = [
                object_to_record(obj, object_id, calibration.P2, "ground_truth")
                for object_id, obj in enumerate(get_objects_from_label(str(label_path)))
                if obj.cls_type != "DontCare"
            ]

        predictions = {}
        inference_artifacts = {}
        for model in self.models:
            prediction_dir = model.get("prediction_dir")
            prediction_path = prediction_dir / f"{frame_id}.txt" if prediction_dir else None
            if prediction_path is not None and prediction_path.is_file():
                predictions[model["id"]] = [
                    object_to_record(obj, object_id, calibration.P2, model["id"])
                    for object_id, obj in enumerate(get_objects_from_label(str(prediction_path)))
                    if obj.cls_type != "DontCare"
                ]
            else:
                predictions[model["id"]] = []

            artifact = None
            artifact_dir = model.get("artifact_dir")
            if artifact_dir is not None:
                artifact_path = artifact_dir / f"{frame_id}.json"
                if artifact_path.is_file():
                    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                    detections = artifact.get("detections", [])
                    for prediction in predictions[model["id"]]:
                        if prediction["id"] < len(detections):
                            prediction["e1"] = detections[prediction["id"]]
            inference_artifacts[model["id"]] = artifact

        return {
            "frame_id": frame_id,
            "image_url": f"/api/frame/{frame_id}/image",
            "image_size": image_size,
            "calibration": {"P2": calibration.P2.astype(float).tolist()},
            "ground_truth": ground_truth,
            "predictions": predictions,
            "inference_artifacts": inference_artifacts,
            "pseudo_labels": add_ground_reprojections(
                self.pseudo_records.get(frame_id), calibration.P2
            ),
        }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "MonoDETRDashboard/1.0"

    def do_HEAD(self):  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/":
            path = "/index.html"
        image_match = re.fullmatch(r"/api/frame/(\d{6})/image", path)
        if image_match and image_match.group(1) in self.server.data.frame_set:
            self.send_file(self.server.data.image_path(image_match.group(1)), "image/png", head_only=True)
            return
        static_path = (STATIC_ROOT / path.lstrip("/")).resolve()
        if STATIC_ROOT in static_path.parents and static_path.is_file():
            self.send_file(static_path, head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/config":
                self.send_json(self.server.data.public_config())
                return
            match = re.fullmatch(r"/api/frame/(\d{6})", path)
            if match:
                self.send_json(self.server.data.frame(match.group(1)))
                return
            match = re.fullmatch(r"/api/frame/(\d{6})/image", path)
            if match:
                frame_id = match.group(1)
                if frame_id not in self.server.data.frame_set:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_file(self.server.data.image_path(frame_id), "image/png")
                return
            if path == "/":
                path = "/index.html"
            static_path = (STATIC_ROOT / path.lstrip("/")).resolve()
            if STATIC_ROOT not in static_path.parents or not static_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_file(static_path)
        except KeyError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except FileNotFoundError as error:
            self.send_json({"error": f"Missing file: {error}"}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # Keep local debugging errors visible to the UI.
            self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path != "/api/infer":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("invalid request body size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.server.inference.infer(
                str(payload.get("frame_id", "")),
                payload.get("selections") or [],
                payload.get("threshold", 0.2),
            )
            self.send_json(result)
        except (KeyError, ValueError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_json(self, payload: Dict, status=HTTPStatus.OK):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: Optional[str] = None, head_only: bool = False):
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main() -> None:
    args = parse_args()
    data = DashboardData(args)
    from dashboard.inference import DashboardInference
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    server.data = data
    server.inference = DashboardInference(data, device=args.device, topk=args.topk)
    print(f"MonoDETR geometry dashboard: http://{args.host}:{args.port}")
    print(f"Split: {args.split} ({len(data.frame_ids)} frames)")
    print(f"Pseudo-labels: {len(data.pseudo_records)} loaded from {data.pseudo_path}")
    live_count = sum(len((model.get("inference") or {}).get("weights", [])) for model in data.models)
    print(f"Live checkpoints: {live_count} · device: {args.device}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
