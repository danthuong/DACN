"""Runtime adapters for the isolated near-contact GPro ablation."""

from __future__ import annotations

import json
from pathlib import Path

from lib.helpers.tester_helper import Tester
from lib.helpers.trainer_helper import Trainer


class NearGProTrainer(Trainer):
    """Pass the three new per-object labels into the criterion."""

    def prepare_targets(self, targets, batch_size):
        prepared = super().prepare_targets(targets, batch_size)
        mask = targets["mask_2d"]
        keys = (
            "near_touchpoint",
            "near_touchpoint_box_offset",
            "near_touchpoint_valid",
        )
        for batch_index, item in enumerate(prepared):
            for key in keys:
                item[key] = targets[key][batch_index][mask[batch_index]]
        return prepared


class NearGProTester(Tester):
    """Extend the normal artifact JSON with every depth decomposition term."""

    def _save_e1_artifacts(self, outputs, selected_outputs, info, inputs,
                           output_name, topk):
        super()._save_e1_artifacts(
            outputs, selected_outputs, info, inputs, output_name, topk
        )
        root = Path(self._result_root(output_name)) / "artifacts"
        geometry = outputs.get("geometry") or {}
        tensor_fields = {
            "query_reg_depth_m": selected_outputs.get("pred_depth_query_reg"),
            "size_depth_m": selected_outputs.get("pred_depth_size"),
            "baseline_depth_m": selected_outputs.get("pred_depth_baseline"),
            "gpro_cue_depth_m": selected_outputs.get("pred_depth_cue"),
        }
        for batch_index, image_id_value in enumerate(info["img_id"]):
            path = root / "{:06d}.json".format(int(image_id_value))
            payload = json.loads(path.read_text(encoding="utf-8"))
            for record in payload.get("detections", []):
                query_index = int(record["query_id"])
                for json_key, tensor in tensor_fields.items():
                    if tensor is not None:
                        record[json_key] = float(
                            tensor[batch_index, query_index].detach().float()
                        )
                if "replacement_valid" in outputs:
                    record["replacement_valid"] = bool(
                        outputs["replacement_valid"][batch_index, query_index]
                    )
                if geometry:
                    record.update({
                        "near_touchpoint_box_offset": geometry[
                            "touchpoints_rel"
                        ][batch_index, query_index, 0].detach().float().cpu().tolist(),
                        "near_touchpoint_confidence": float(geometry[
                            "near_touchpoint_confidence"
                        ][batch_index, query_index].detach().float()),
                        "near_point_camera_m": geometry[
                            "near_point_camera"
                        ][batch_index, query_index].detach().float().cpu().tolist(),
                        "near_to_center_offset_camera_m": geometry[
                            "near_offset_camera"
                        ][batch_index, query_index].detach().float().cpu().tolist(),
                        "bottom_center_camera_m": geometry[
                            "bottom_center_camera"
                        ][batch_index, query_index].detach().float().cpu().tolist(),
                        "rotation_y_rad": float(geometry[
                            "rotation_y"
                        ][batch_index, query_index].detach().float()),
                    })
            payload["schema_version"] = "near-gpro-zmap-inference-v1"
            payload["depth_formula"] = (
                "Zfinal=(Zquery+Zsize+where(valid,Zgpro_center,Zmap))/3"
            )
            path.write_text(
                json.dumps(payload, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
