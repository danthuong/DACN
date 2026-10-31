"""Lazy, checkpoint-driven inference used only by the local dashboard.

Models are kept on CPU after their first load and moved to the selected device
one at a time.  This makes side-by-side comparisons possible without keeping
several MonoDETR variants resident in GPU memory.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Dict, Iterable

import numpy as np


class DashboardInference:
    def __init__(self, dashboard_data, device="auto", topk=50):
        self.data = dashboard_data
        self.device_spec = str(device)
        self.topk = int(topk)
        self._loaded = {}
        self._lock = threading.Lock()

    @staticmethod
    def _checkpoint_state(checkpoint):
        state = checkpoint.get("model_state", checkpoint)
        if state and all(key.startswith("module.") for key in state):
            state = {key[len("module."):]: value for key, value in state.items()}
        return state

    def _device(self, torch):
        if self.device_spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(self.device_spec)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("dashboard inference requested CUDA, but CUDA is unavailable")
        return device

    def _load(self, model_entry, weight_entry):
        cache_key = (model_entry["id"], weight_entry["id"])
        cached = self._loaded.get(cache_key)
        checkpoint_path = weight_entry["path"]
        mtime = checkpoint_path.stat().st_mtime_ns
        if cached is not None and cached["mtime"] == mtime:
            return cached

        import torch
        from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
        from lib.helpers.config_helper import load_config

        try:
            # These are trusted local training checkpoints and include the
            # resolved config plus RNG metadata, not tensors alone. PyTorch
            # 2.6 changed the torch.load default to weights_only=True.
            checkpoint = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch versions before the weights_only keyword.
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        cfg = copy.deepcopy(checkpoint.get("resolved_config") or {})
        inference_cfg = model_entry.get("inference") or {}
        if not cfg:
            config_path = inference_cfg.get("config_path")
            if config_path is None:
                raise ValueError(
                    "checkpoint has no resolved_config and model registry has no config"
                )
            cfg = load_config(str(config_path), str(inference_cfg.get("env_file", ".env")))

        model_cfg = dict(cfg["model"])
        model_cfg["device"] = "cpu"
        builder = str(inference_cfg.get("builder", "auto"))
        if builder == "auto":
            builder = "near_gpro" if cfg.get("near_gpro") else "monodetr"

        # A complete checkpoint will overwrite every backbone tensor. Avoid a
        # network download just to construct the ResNet skeleton.
        import lib.models.monodetr.backbone as backbone_module
        backbone_module.is_main_process = lambda: False

        if builder == "near_gpro":
            from lib.models.monodetr.near_gpro_zmap import build_near_gpro_zmap
            model_cfg["near_gpro"] = cfg.get("near_gpro") or {}
            model, _ = build_near_gpro_zmap(model_cfg)
        elif builder in ("monodetr", "e1"):
            from lib.helpers.model_helper import build_model
            model_cfg["geometry_e1"] = cfg.get("geometry_e1")
            model, _ = build_model(model_cfg)
        else:
            raise ValueError("unsupported dashboard model builder: " + builder)

        state = self._checkpoint_state(checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint/model mismatch; missing={} unexpected={}".format(
                    incompatible.missing_keys[:12], incompatible.unexpected_keys[:12]
                )
            )
        model.eval().cpu()

        dataset_cfg = dict(cfg["dataset"])
        dataset_cfg["root_dir"] = str(self.data.dataset_root)
        # Inference needs input_calib but not training pseudo-label targets.
        dataset_cfg["geometry_e1"] = None
        dataset = KITTI_Dataset(split=self.data.split, cfg=dataset_cfg)
        frame_positions = {
            str(frame_id).zfill(6): index
            for index, frame_id in enumerate(dataset.idx_list)
        }
        cached = {
            "model": model,
            "cfg": cfg,
            "dataset": dataset,
            "frame_positions": frame_positions,
            "mtime": mtime,
            "checkpoint_epoch": checkpoint.get("epoch"),
        }
        self._loaded[cache_key] = cached
        return cached

    @staticmethod
    def _raw_points(points_normalized, affine, input_hw):
        points = np.asarray(points_normalized, dtype=np.float64)
        pixels = points * np.asarray([input_hw[1], input_hw[0]], dtype=np.float64)
        homogeneous = np.concatenate(
            [pixels, np.ones((*pixels.shape[:-1], 1), dtype=np.float64)], axis=-1
        )
        inverse = np.linalg.inv(np.asarray(affine, dtype=np.float64))
        raw = homogeneous @ inverse.T
        return raw[..., :2] / raw[..., 2:3]

    @staticmethod
    def _raw_line_segment(line, affine, input_hw):
        width = float(input_hw[1] - 1)
        network = np.asarray(
            [[0.0, float(line[1])], [width, float(line[0]) * width + float(line[1])]],
            dtype=np.float64,
        )
        normalized = network / np.asarray([input_hw[1], input_hw[0]], dtype=np.float64)
        return DashboardInference._raw_points(normalized, affine, input_hw).tolist()

    @staticmethod
    def _prediction_record(values, prediction_id, calibration, source, serialise_box):
        class_names = ("Pedestrian", "Car", "Cyclist")
        class_id = int(values[0])
        return serialise_box({
            "id": prediction_id,
            "source": source,
            "class": class_names[class_id],
            "confidence": float(values[13]),
            "bbox_2d": np.asarray(values[2:6], dtype=float).tolist(),
            "dimensions_3d": np.asarray(values[6:9], dtype=float).tolist(),
            "location_3d": np.asarray(values[9:12], dtype=float).tolist(),
            "rotation_y": float(values[12]),
        }, calibration.P2)

    @staticmethod
    def _ranked_queries(outputs, topk, threshold):
        import torch
        probabilities = outputs["pred_logits"].sigmoid()
        scores, flat = torch.topk(
            probabilities.flatten(1), min(topk, probabilities.shape[1] * probabilities.shape[2]), dim=1
        )
        queries = flat // probabilities.shape[2]
        classes = flat % probabilities.shape[2]
        return [
            (int(queries[0, rank]), int(classes[0, rank]), float(scores[0, rank]))
            for rank in range(scores.shape[1])
            if float(scores[0, rank]) >= threshold
        ]

    def _artifact(self, outputs, info, inputs, ranked_queries, weight_entry, loaded):
        affine = np.asarray(info["affine"], dtype=np.float64)
        input_hw = (int(inputs.shape[-2]), int(inputs.shape[-1]))
        artifact = {
            "schema_version": "dashboard-live-inference-v1",
            "checkpoint": weight_entry["label"],
            "checkpoint_epoch": loaded.get("checkpoint_epoch"),
            "detections": [],
            "horizon": None,
        }
        raw_line = outputs.get("pred_horizon_line_raw")
        refined_line = outputs.get("pred_horizon_line_refined", outputs.get("pred_horizon_line"))
        if raw_line is not None:
            raw = raw_line[0].detach().float().cpu().numpy()
            refined = refined_line[0].detach().float().cpu().numpy()
            edge_valid = bool(outputs.get("vertical_edge_valid")[0]) \
                if outputs.get("vertical_edge_valid") is not None else False
            artifact["horizon"] = {
                "raw_line_network": raw.tolist(),
                "refined_line_network": refined.tolist(),
                "raw_visible_segment_raw": self._raw_line_segment(raw, affine, input_hw),
                "visible_segment_raw": self._raw_line_segment(refined, affine, input_hw),
                "vertical_edge_refinement": {
                    "valid": edge_valid,
                    "edge_count": int(outputs["vertical_edge_count"][0])
                        if outputs.get("vertical_edge_count") is not None else 0,
                },
            }

        geometry = outputs.get("geometry") or {}
        if geometry and geometry.get("plane") is not None:
            artifact["geometry_plane_camera"] = (
                geometry["plane"][0].detach().float().cpu().tolist()
            )
        for detection_id, (query, class_id, class_score) in enumerate(ranked_queries):
            record = {
                "detection_id": detection_id,
                "query_id": query,
                "class_id": class_id,
                "class_score": class_score,
                "selected_depth_m": float(outputs["pred_depth"][0, query, 0]),
            }
            tensor_fields = {
                "baseline_depth_m": "pred_depth_baseline",
                "query_reg_depth_m": "pred_depth_query_reg",
                "size_depth_m": "pred_depth_size",
                "depth_predictor_depth_m": "pred_depth_predictor_sample",
                "gpro_cue_depth_m": "pred_depth_cue",
            }
            for target_key, output_key in tensor_fields.items():
                tensor = outputs.get(output_key)
                if tensor is not None:
                    record[target_key] = float(tensor[0, query])
            if geometry and query < geometry["object_depth"].shape[1]:
                points = geometry["touchpoints_abs"][0, query].detach().float().cpu().numpy()
                record.update({
                    "touchpoints_raw": self._raw_points(points, affine, input_hw).tolist(),
                    "touchpoint_ray_valid": geometry["touchpoint_ray_valid"][0, query]
                        .detach().cpu().tolist(),
                    "geometric_depth_m": float(geometry["object_depth"][0, query]),
                    "geometry_valid": bool(geometry["object_valid"][0, query]),
                })
                intersections = geometry.get("touchpoint_intersections")
                if intersections is not None:
                    record["touchpoints_3d_camera"] = (
                        intersections[0, query].detach().float().cpu().tolist()
                    )
                near_point = geometry.get("near_point_camera")
                if near_point is not None:
                    record["near_touchpoint_3d_camera"] = (
                        near_point[0, query].detach().float().cpu().tolist()
                    )
                bottom_center = geometry.get("bottom_center_camera")
                if bottom_center is not None:
                    record["bottom_center_3d_camera"] = (
                        bottom_center[0, query].detach().float().cpu().tolist()
                    )
            artifact["detections"].append(record)
        return artifact

    def _infer_one(self, model_entry, weight_entry, frame_id, threshold):
        import torch
        from lib.helpers.decode_helper import decode_detections, extract_dets_from_outputs
        from dashboard.server import serialise_box

        loaded = self._load(model_entry, weight_entry)
        dataset = loaded["dataset"]
        position = loaded["frame_positions"].get(frame_id)
        if position is None:
            raise KeyError("frame {} is absent from model dataset split".format(frame_id))
        inputs_np, calib_np, _, info = dataset[position]
        inputs = torch.as_tensor(inputs_np).unsqueeze(0).float()
        calibs = torch.as_tensor(calib_np).unsqueeze(0).float()
        img_sizes = torch.as_tensor(info["img_size"]).unsqueeze(0).float()
        geometry_calibs = torch.as_tensor(info["input_calib"]).unsqueeze(0).float()
        device = self._device(torch)
        model = loaded["model"].to(device)
        try:
            inputs = inputs.to(device)
            calibs = calibs.to(device)
            img_sizes = img_sizes.to(device)
            geometry_calibs = geometry_calibs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                outputs = model(
                    inputs, calibs, {}, img_sizes, dn_args=0,
                    geometry_calibs=geometry_calibs,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_ms = 1000.0 * (time.perf_counter() - started)

            ranked = self._ranked_queries(outputs, self.topk, threshold)
            dets = extract_dets_from_outputs(outputs, K=dataset.max_objs, topk=self.topk)
            numpy_info = {
                "img_id": np.asarray([int(frame_id)]),
                "img_size": np.asarray([info["img_size"]]),
            }
            calibration = dataset.get_calib(int(frame_id))
            decoded = decode_detections(
                dets.detach().cpu().numpy(), numpy_info, [calibration],
                dataset.cls_mean_size, threshold,
            )[int(frame_id)]
            # decode_detections applies the same class-score threshold and ordering.
            ranked = ranked[:len(decoded)]
            artifact = self._artifact(outputs, info, inputs, ranked, weight_entry, loaded)
            predictions = []
            for prediction_id, values in enumerate(decoded):
                record = self._prediction_record(
                    values, prediction_id, calibration, model_entry["id"], serialise_box
                )
                if prediction_id < len(artifact["detections"]):
                    record["e1"] = artifact["detections"][prediction_id]
                predictions.append(record)
        finally:
            model.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return {
            "model_id": model_entry["id"],
            "weight_id": weight_entry["id"],
            "weight_label": weight_entry["label"],
            "checkpoint_epoch": loaded.get("checkpoint_epoch"),
            "latency_ms": latency_ms,
            "predictions": predictions,
            "artifact": artifact,
        }

    def infer(self, frame_id: str, selections: Iterable[Dict], threshold=0.2):
        if frame_id not in self.data.frame_set:
            raise KeyError(frame_id)
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        selections = list(selections)
        if not selections:
            raise ValueError("select at least one checkpoint")
        if len(selections) > 8:
            raise ValueError("at most 8 checkpoints may be compared at once")

        results = {}
        errors = {}
        with self._lock:
            for selection in selections:
                model_id = str(selection.get("model_id"))
                try:
                    model = self.data.model_by_id(model_id)
                    weight = self.data.weight_by_id(
                        model, str(selection.get("weight_id"))
                    )
                    results[model["id"]] = self._infer_one(
                        model, weight, frame_id, threshold
                    )
                except Exception as error:
                    errors[model_id] = str(error)
        return {"frame_id": frame_id, "models": results, "errors": errors}
