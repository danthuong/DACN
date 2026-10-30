"""Single-decoder MonoDETR with near-contact GPro depth replacing Z-map.

This module is deliberately separate from ``monodetr.py`` and ``geometry_e1``
so the earlier feature/distribution fusion experiments remain reproducible.
The original dense depth predictor still supplies decoder memory.  Only the
sampled scalar Z-map term in MonoDETR's final three-way depth average is
replaced, with an identity fallback whenever geometry is invalid.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from utils import box_ops
from utils.misc import (
    NestedTensor,
    get_world_size,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
)
from .backbone import build_backbone
from .depth_predictor import DepthPredictor
from .depthaware_transformer import build_depthaware_transformer
from .matcher import build_matcher
from .monodetr import MonoDETR, SetCriterion
from .geometry_e1 import (
    HorizonHead,
    VerticalEdgeHorizonRefiner,
    decode_horizon_line,
    horizon_line_to_ground_plane,
    raycast_touchpoints_to_plane,
    refit_horizon_intercept,
)


class NearContactROIHead(nn.Module):
    """Predict one box-relative near-contact point from visual ROI features."""

    def __init__(self, cfg, input_channels=256):
        super().__init__()
        self.roi_size = int(cfg.get("roi_size", 9))
        self.chunk_size = int(cfg.get("query_chunk_size", 50))
        self.temperature = float(cfg.get("softargmax_temperature", 0.25))
        offset_range = cfg.get("offset_range", [-0.2, 1.2])
        if (
            self.roi_size < 2
            or self.chunk_size < 1
            or self.temperature <= 0
            or len(offset_range) != 2
            or float(offset_range[0]) >= float(offset_range[1])
        ):
            raise ValueError("invalid near-touchpoint ROI configuration")
        self.offset_min = float(offset_range[0])
        self.offset_max = float(offset_range[1])
        channels = int(cfg.get("roi_channels", 64))
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.heatmap = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def _sample_chunk(self, feature, boxes_xyxy):
        batch, queries = boxes_xyxy.shape[:2]
        size = self.roi_size
        coordinates = torch.linspace(
            self.offset_min,
            self.offset_max,
            size,
            device=feature.device,
            dtype=feature.dtype,
        )
        # PyTorch 1.9 uses ``ij`` ordering by default and does not yet accept
        # the explicit ``indexing`` keyword introduced in later releases.
        rel_y, rel_x = torch.meshgrid(coordinates, coordinates)
        rel_x = rel_x.view(1, 1, size, size)
        rel_y = rel_y.view(1, 1, size, size)
        xmin, ymin, xmax, ymax = boxes_xyxy.unbind(dim=-1)
        width = (xmax - xmin).clamp_min(1e-4)
        height = (ymax - ymin).clamp_min(1e-4)
        sample_x = xmin[..., None, None] + rel_x * width[..., None, None]
        sample_y = ymin[..., None, None] + rel_y * height[..., None, None]
        grid = torch.stack((2.0 * sample_x - 1.0, 2.0 * sample_y - 1.0), dim=-1)
        # Stack Q ROI grids vertically, sample once per image, then unstack.
        sampled = F.grid_sample(
            feature,
            grid.reshape(batch, queries * size, size, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch, sampled.shape[1], queries, size, size
        ).permute(0, 2, 1, 3, 4)
        logits = self.heatmap(sampled.reshape(-1, sampled.shape[2], size, size))
        return logits.reshape(batch, queries, size, size)

    def forward(self, visual_feature, boxes):
        feature = self.feature_adapter(visual_feature)
        xyxy = box_ops.box_cxcylrtb_to_xyxy(boxes)
        logits = []
        for start in range(0, xyxy.shape[1], self.chunk_size):
            logits.append(self._sample_chunk(feature, xyxy[:, start:start + self.chunk_size]))
        logits = torch.cat(logits, dim=1)
        probabilities = F.softmax(
            logits.flatten(-2) / self.temperature, dim=-1
        ).reshape_as(logits)
        coordinates = torch.linspace(
            self.offset_min,
            self.offset_max,
            self.roi_size,
            device=logits.device,
            dtype=logits.dtype,
        )
        rel_x = (probabilities * coordinates.view(1, 1, 1, -1)).sum((-2, -1))
        rel_y = (probabilities * coordinates.view(1, 1, -1, 1)).sum((-2, -1))
        relative = torch.stack((rel_x, rel_y), dim=-1)
        top_left = xyxy[..., :2]
        size = (xyxy[..., 2:] - xyxy[..., :2]).clamp_min(1e-6)
        absolute = top_left + relative * size
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum((-2, -1))
        confidence = 1.0 - entropy / math.log(float(self.roi_size * self.roi_size))
        return {
            "logits": logits,
            "relative": relative,
            "absolute": absolute,
            "confidence": confidence.clamp(0.0, 1.0),
        }


class NearGProGeometry(nn.Module):
    def __init__(self, cfg, hidden_dim=256):
        super().__init__()
        self.cfg = cfg
        horizon_cfg = cfg.get("horizon", {})
        touch_cfg = cfg.get("near_touchpoint", {})
        plane_cfg = cfg.get("plane", {})
        correction_cfg = cfg.get("center_correction", {})
        self.horizon_head = HorizonHead(
            hidden_dim, int(horizon_cfg.get("fpn_channels", 128))
        )
        self.touchpoint_head = NearContactROIHead(touch_cfg, hidden_dim)
        self.horizon_temperature = float(
            horizon_cfg.get("softargmax_temperature", 0.25)
        )
        self.horizon_entropy_confidence = bool(
            horizon_cfg.get("entropy_confidence", True)
        )
        self.vertical_edge_refiner = VerticalEdgeHorizonRefiner(
            horizon_cfg.get("vertical_edge_refinement", {})
        )
        self.camera_height_m = float(plane_cfg.get("camera_height_m", 1.65))
        self.min_abs_ray_plane_dot = float(
            plane_cfg.get("min_abs_ray_plane_dot", 0.015)
        )
        self.min_depth = float(plane_cfg.get("valid_min_depth_m", 2.0))
        self.max_depth = float(plane_cfg.get("valid_max_depth_m", 65.0))
        self.k_l = float(correction_cfg.get("k_l", 0.65))
        self.k_w = float(correction_cfg.get("k_w", 0.85))
        self.detach_boxes = bool(touch_cfg.get("box_detach", True))
        self.detach_visual = bool(touch_cfg.get("visual_feature_detach", False))
        self.detach_pose = bool(correction_cfg.get("detach_dimensions_yaw", True))
        self.class_index = int(touch_cfg.get("class_index", 1))

    def predict_horizon(self, srcs, image_hw, images=None):
        logits = self.horizon_head(srcs)
        raw_line, expected_row, confidence = decode_horizon_line(
            logits,
            image_hw,
            temperature=self.horizon_temperature,
            entropy_confidence=self.horizon_entropy_confidence,
        )
        batch = logits.shape[0]
        device = logits.device
        line = raw_line
        edge_valid = torch.zeros(batch, dtype=torch.bool, device=device)
        edge_count = torch.zeros(batch, dtype=torch.long, device=device)
        edge_angle = torch.zeros(batch, device=device)
        edge_angle_std = torch.zeros(batch, device=device)
        # The OpenCV edge estimator is deliberately inference-only.  Its slope
        # is combined with the learned heatmap intercept, exactly as in E1.
        if not self.training and self.vertical_edge_refiner.enabled:
            if images is None:
                raise ValueError("images are required for vertical-edge refinement")
            (line, edge_valid, edge_count,
             edge_angle, edge_angle_std) = self.vertical_edge_refiner(
                images, raw_line
            )
            line = torch.stack((
                line[:, 0],
                refit_horizon_intercept(
                    line[:, 0], expected_row, confidence, image_hw,
                    logits.shape[-2:],
                ),
            ), dim=-1)
        return {
            "logits": logits,
            "line": line,
            "raw_line": raw_line,
            "expected_row": expected_row,
            "confidence": confidence,
            "vertical_edge_valid": edge_valid,
            "vertical_edge_count": edge_count,
            "vertical_edge_angle_deg": edge_angle,
            "vertical_edge_angle_std_deg": edge_angle_std,
        }

    @staticmethod
    def _decode_alpha(angle):
        bins, residuals = angle[..., :12], angle[..., 12:24]
        index = bins.argmax(dim=-1)
        residual = residuals.gather(-1, index[..., None]).squeeze(-1)
        return index.to(residual.dtype) * (2.0 * math.pi / 12.0) + residual

    def _near_offset_camera(self, dimensions, rotation_y):
        # Dimensions follow KITTI [height, width, length].  Local candidates
        # intentionally match pseudo_label_generation.geometry.
        width = dimensions[..., 1].clamp_min(1e-3)
        length = dimensions[..., 2].clamp_min(1e-3)
        half_l = 0.5 * self.k_l * length
        half_w = 0.5 * self.k_w * width
        local_x = torch.stack((half_l, half_l, -half_l, -half_l), dim=-1)
        local_z = torch.stack((half_w, -half_w, -half_w, half_w), dim=-1)
        cosine = rotation_y.cos()[..., None]
        sine = rotation_y.sin()[..., None]
        camera_x = cosine * local_x + sine * local_z
        camera_z = -sine * local_x + cosine * local_z
        camera_y = torch.zeros_like(camera_x)
        offsets = torch.stack((camera_x, camera_y, camera_z), dim=-1)
        near_indices = camera_z.topk(k=2, dim=-1, largest=False).indices
        gather_index = near_indices[..., None].expand(*near_indices.shape, 3)
        return offsets.gather(-2, gather_index).mean(dim=-2)

    def forward(
        self,
        visual_feature,
        boxes,
        class_logits,
        dimensions,
        angles,
        horizon,
        p2_input,
        image_hw,
    ):
        touch_boxes = boxes.detach() if self.detach_boxes else boxes
        touch_feature = visual_feature.detach() if self.detach_visual else visual_feature
        contact = self.touchpoint_head(touch_feature, touch_boxes)
        point = contact["absolute"]
        plane, plane_valid = horizon_line_to_ground_plane(
            horizon["line"], p2_input, self.camera_height_m
        )
        rays = raycast_touchpoints_to_plane(
            point[:, :, None, :],
            p2_input,
            plane,
            image_hw,
            self.min_abs_ray_plane_dot,
            self.min_depth,
            self.max_depth,
        )
        in_image = (
            (point[..., 0] >= 0.0)
            & (point[..., 0] < 1.0)
            & (point[..., 1] >= 0.0)
            & (point[..., 1] < 1.0)
        )
        ray_valid = rays["point_valid"][..., 0] & in_image
        pose_dimensions = dimensions.detach() if self.detach_pose else dimensions
        pose_angles = angles.detach() if self.detach_pose else angles
        pose_boxes = boxes.detach() if self.detach_pose else boxes
        alpha = self._decode_alpha(pose_angles)
        input_width = float(image_hw[1])
        # MonoDETR trains alpha and decodes rotation_y against the 2-D box
        # centre, not against the projected 3-D centre stored in boxes[...,0].
        pose_xyxy = box_ops.box_cxcylrtb_to_xyxy(pose_boxes)
        projected_u = 0.5 * (pose_xyxy[..., 0] + pose_xyxy[..., 2]) * input_width
        fx = p2_input[:, 0, 0][:, None]
        cx = p2_input[:, 0, 2][:, None]
        rotation_y = alpha + torch.atan2(projected_u - cx, fx)
        near_offset = self._near_offset_camera(pose_dimensions, rotation_y)
        near_point = rays["intersections"][..., 0, :]
        bottom_center = near_point - near_offset
        object_depth = bottom_center[..., 2]
        object_valid = (
            ray_valid
            & plane_valid[:, None]
            & torch.isfinite(bottom_center).all(dim=-1)
            & (object_depth >= self.min_depth)
            & (object_depth <= self.max_depth)
        )
        object_depth = torch.where(
            object_valid, object_depth, torch.zeros_like(object_depth)
        )
        horizon_reliability = horizon["confidence"].float().mean(dim=-1).clamp(0.0, 1.0)
        ray_reliability = rays["point_stability"][..., 0]
        class_confidence = class_logits.sigmoid()[..., self.class_index]
        reliability = (
            contact["confidence"]
            * horizon_reliability[:, None]
            * ray_reliability
            * class_confidence
            * object_valid.to(class_confidence.dtype)
        )
        return {
            "touchpoints_abs": point[:, :, None, :],
            "touchpoints_rel": contact["relative"][:, :, None, :],
            "near_touchpoint_heatmap_logits": contact["logits"],
            "near_touchpoint_confidence": contact["confidence"],
            "plane": plane,
            "plane_valid": plane_valid,
            "object_depth": object_depth,
            "object_valid": object_valid,
            "object_reliability": reliability,
            "horizon_reliability": horizon_reliability,
            "object_depth_std": torch.zeros_like(object_depth),
            "touchpoint_intersections": rays["intersections"],
            "touchpoint_ray_valid": object_valid[..., None],
            "near_point_camera": near_point,
            "near_offset_camera": near_offset,
            "bottom_center_camera": bottom_center,
            "rotation_y": rotation_y,
        }


class NearGProMonoDETR(MonoDETR):
    """MonoDETR whose final scalar Z-map cue is replaced by valid GPro depth."""

    def __init__(self, *args, near_cfg=None, **kwargs):
        # Do not instantiate the old E1 pipeline.
        kwargs["geometry_cfg"] = None
        super().__init__(*args, **kwargs)
        self.near_cfg = near_cfg or {}
        self.geometry_e1 = NearGProGeometry(self.near_cfg, self.hidden_dim)
        self.near_gpro_enabled = True
        # Keep the legacy trainer on its normal single-output validation path.
        self.geometry_enabled = False
        replacement = self.near_cfg.get("replacement", {})
        self.min_replacement_confidence = float(
            replacement.get("min_geometry_confidence", 0.0)
        )

    def _decode_detection_heads(self, transformer_outputs, weighted_depth,
                                calibs, img_sizes, depth_map_logits):
        (hs, init_reference, inter_references, inter_references_dim,
         enc_outputs_class, enc_outputs_coord_unact) = transformer_outputs
        outputs_coords, outputs_classes = [], []
        outputs_3d_dims, outputs_depths, outputs_angles = [], [], []
        outputs_depth_predictor_samples = []
        outputs_query_depths, outputs_size_depths = [], []
        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
            outputs_classes.append(self.class_embed[lvl](hs[lvl]))
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)
            box_height = torch.clamp(
                (outputs_coord[..., 4] + outputs_coord[..., 5])
                * img_sizes[:, 1:2],
                min=1.0,
            )
            size_depth = size3d[..., 0] / box_height * calibs[:, 0, 0].unsqueeze(1)
            depth_reg = self.depth_embed[lvl](hs[lvl])
            query_depth = 1.0 / (depth_reg[..., 0].sigmoid() + 1e-6) - 1.0
            sample_grid = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            map_depth = F.grid_sample(
                weighted_depth.unsqueeze(1), sample_grid,
                mode="bilinear", align_corners=True,
            ).squeeze(1).squeeze(-1)
            final_depth = (query_depth + size_depth + map_depth) / 3.0
            outputs_depths.append(torch.stack((final_depth, depth_reg[..., 1]), dim=-1))
            outputs_query_depths.append(query_depth)
            outputs_size_depths.append(size_depth)
            outputs_depth_predictor_samples.append(map_depth)
            outputs_angles.append(self.angle_embed[lvl](hs[lvl]))

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_3d_dim": outputs_3d_dim[-1],
            "pred_depth": outputs_depth[-1],
            "pred_depth_query_reg": torch.stack(outputs_query_depths)[-1],
            "pred_depth_size": torch.stack(outputs_size_depths)[-1],
            "pred_depth_predictor_sample": torch.stack(
                outputs_depth_predictor_samples
            )[-1],
            "pred_angle": outputs_angle[-1],
            "pred_depth_map_logits": depth_map_logits,
            "_query_features": hs[-1],
        }
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim,
                outputs_angle, outputs_depth,
            )
        if self.two_stage:
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord_unact.sigmoid(),
            }
        return out

    def forward(self, images, calibs, targets, img_sizes, dn_args=None,
                geometry_calibs=None, force_phase=None):
        features, pos = self.backbone(images)
        srcs, masks = [], []
        for level, feature in enumerate(features):
            source, mask = feature.decompose()
            srcs.append(self.input_proj[level](source))
            masks.append(mask)
        if self.num_feature_levels > len(srcs):
            original_levels = len(srcs)
            for level in range(original_levels, self.num_feature_levels):
                source = self.input_proj[level](
                    features[-1].tensors if level == original_levels else srcs[-1]
                )
                mask = F.interpolate(
                    torch.zeros(
                        source.shape[0], source.shape[2], source.shape[3],
                        device=source.device,
                    )[None],
                    size=source.shape[-2:],
                ).to(torch.bool)[0]
                srcs.append(source)
                masks.append(mask)
                pos.append(self.backbone[1](
                    NestedTensor(source, mask)
                ).to(source.dtype))

        depth_source = self.depth_predictor.extract_depth_feature(srcs)
        depth_logits = self.depth_predictor.depth_classifier(depth_source)
        depth_base = self.depth_predictor.encode_depth_feature(
            depth_source, masks[1], pos[1]
        )
        weighted_depth, depth_position = self.depth_predictor.depth_from_logits(
            depth_logits
        )
        depth_memory = depth_base + depth_position
        transformer_outputs = self.depthaware_transformer(
            srcs, masks, pos, self._query_embeddings(),
            depth_memory, depth_position,
        )
        out = self._decode_detection_heads(
            transformer_outputs, weighted_depth, calibs, img_sizes, depth_logits
        )
        image_hw = (images.shape[-2], images.shape[-1])
        horizon = self.geometry_e1.predict_horizon(srcs, image_hw, images)
        p2_input = self._input_frame_calibration(
            images, calibs, img_sizes, geometry_calibs
        )
        geometry = self.geometry_e1(
            srcs[0],
            out["pred_boxes"],
            out["pred_logits"],
            out["pred_3d_dim"],
            out["pred_angle"],
            horizon,
            p2_input,
            image_hw,
        )
        valid = geometry["object_valid"] & (
            geometry["object_reliability"] >= self.min_replacement_confidence
        )
        z_map = out["pred_depth_predictor_sample"]
        z_cue = torch.where(valid, geometry["object_depth"], z_map)
        z_baseline = out["pred_depth"][..., 0]
        z_final = (
            out["pred_depth_query_reg"] + out["pred_depth_size"] + z_cue
        ) / 3.0
        out["pred_depth_baseline"] = z_baseline
        out["pred_depth_cue"] = z_cue
        out["pred_depth"] = torch.stack(
            (z_final, out["pred_depth"][..., 1]), dim=-1
        )
        out.update({
            "pred_horizon_logits": horizon["logits"],
            "pred_horizon_line": horizon["line"],
            "pred_horizon_line_raw": horizon["raw_line"],
            "pred_horizon_line_refined": horizon["line"],
            "pred_horizon_rows": horizon["expected_row"],
            "pred_horizon_confidence": horizon["confidence"],
            "vertical_edge_valid": horizon["vertical_edge_valid"],
            "vertical_edge_count": horizon["vertical_edge_count"],
            "vertical_edge_angle_deg": horizon["vertical_edge_angle_deg"],
            "vertical_edge_angle_std_deg": horizon["vertical_edge_angle_std_deg"],
            "pred_near_touchpoint": geometry["touchpoints_abs"][:, :, 0],
            "pred_near_touchpoint_rel": geometry["touchpoints_rel"][:, :, 0],
            "geometry": geometry,
            "replacement_valid": valid,
            "e1_phase": 1,
        })
        out.pop("_query_features")
        return out


class NearGProCriterion(SetCriterion):
    def __init__(self, *args, near_cfg=None, **kwargs):
        self.near_cfg = near_cfg or {}
        geometry_cfg = {
            "enabled": True,
            "horizon": self.near_cfg.get("horizon", {}),
            "touchpoint": {"class_index": self.near_cfg.get(
                "near_touchpoint", {}).get("class_index", 1)},
            "losses": {},
        }
        kwargs["geometry_cfg"] = geometry_cfg
        super().__init__(*args, **kwargs)
        touch_cfg = self.near_cfg.get("near_touchpoint", {})
        loss_cfg = self.near_cfg.get("losses", {})
        self.near_class_index = int(touch_cfg.get("class_index", 1))
        self.near_beta = float(loss_cfg.get("near_touchpoint_smooth_l1_beta", 0.05))
        self.gpro_beta = float(loss_cfg.get("gpro_log_depth_smooth_l1_beta", 0.05))

    def loss_near_touchpoint(self, outputs, targets, indices, num_boxes):
        batch_index, query_index = self._get_src_permutation_idx(indices)
        predicted = outputs["pred_near_touchpoint_rel"][batch_index, query_index]
        target = torch.cat([
            item["near_touchpoint_box_offset"][target_index]
            for item, (_, target_index) in zip(targets, indices)
        ], dim=0)
        valid = torch.cat([
            item["near_touchpoint_valid"][target_index]
            for item, (_, target_index) in zip(targets, indices)
        ], dim=0).bool()
        labels = torch.cat([
            item["labels"][target_index]
            for item, (_, target_index) in zip(targets, indices)
        ], dim=0)
        valid &= labels == self.near_class_index
        count = valid.sum()
        if int(count.item()) == 0:
            zero = predicted.sum() * 0.0
            return {
                "loss_near_touchpoint": zero,
                "near_touchpoint_mae_px": zero.detach(),
                "near_touchpoint_valid_count": count.detach().float(),
            }
        loss = F.smooth_l1_loss(
            predicted[valid], target[valid], reduction="sum", beta=self.near_beta
        ) / count
        absolute_pred = outputs["pred_near_touchpoint"][batch_index, query_index]
        absolute_target = torch.cat([
            item["near_touchpoint"][target_index]
            for item, (_, target_index) in zip(targets, indices)
        ], dim=0)
        input_h = outputs["pred_horizon_logits"].shape[-2] * self.horizon_stride
        input_w = outputs["pred_horizon_logits"].shape[-1] * self.horizon_stride
        pixel_error = (
            (absolute_pred - absolute_target)
            * absolute_pred.new_tensor([input_w, input_h])
        ).abs()
        return {
            "loss_near_touchpoint": loss,
            "near_touchpoint_mae_px": pixel_error[valid].mean().detach(),
            "near_touchpoint_valid_count": count.detach().float(),
        }

    def loss_gpro_center(self, outputs, targets, indices, num_boxes):
        batch_index, query_index = self._get_src_permutation_idx(indices)
        geometry = outputs["geometry"]
        predicted = geometry["object_depth"][batch_index, query_index]
        valid = geometry["object_valid"][batch_index, query_index]
        target = torch.cat([
            item["depth"][target_index]
            for item, (_, target_index) in zip(targets, indices)
        ], dim=0).reshape(-1)
        label_valid = torch.cat([
            item["near_touchpoint_valid"][target_index]
            for item, (_, target_index) in zip(targets, indices)
        ], dim=0).bool()
        labels = torch.cat([
            item["labels"][target_index]
            for item, (_, target_index) in zip(targets, indices)
        ], dim=0)
        valid &= label_valid & (labels == self.near_class_index)
        count = valid.sum()
        total = max(int(valid.numel()), 1)
        if int(count.item()) == 0:
            zero = predicted.sum() * 0.0
            return {
                "loss_gpro_center_depth": zero,
                "gpro_center_mae_m": zero.detach(),
                "gpro_valid_matched_ratio": zero.detach(),
            }
        loss = F.smooth_l1_loss(
            predicted[valid].clamp_min(1e-3).log(),
            target[valid].clamp_min(1e-3).log(),
            reduction="sum",
            beta=self.gpro_beta,
        ) / count
        mae = (predicted[valid] - target[valid]).abs().mean()
        return {
            "loss_gpro_center_depth": loss,
            "gpro_center_mae_m": mae.detach(),
            "gpro_valid_matched_ratio": count.detach().float() / float(total),
        }

    def forward(self, outputs, targets, mask_dict=None):
        losses = super().forward(outputs, targets, mask_dict)
        group_num = self.group_num if self.training else 1
        output_for_match = {key: value for key, value in outputs.items()
                            if key != "aux_outputs"}
        indices = self.matcher(output_for_match, targets, group_num=group_num)
        num_boxes = sum(len(item["labels"]) for item in targets) * group_num
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=outputs["pred_logits"].device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses.update(self.loss_horizon(outputs, targets, indices, num_boxes))
        losses.update(self.loss_near_touchpoint(outputs, targets, indices, num_boxes))
        losses.update(self.loss_gpro_center(outputs, targets, indices, num_boxes))
        return losses


def build_near_gpro_zmap(cfg):
    near_cfg = cfg.get("near_gpro") or {}
    backbone = build_backbone(cfg)
    transformer = build_depthaware_transformer(cfg)
    depth_predictor = DepthPredictor(cfg)
    model = NearGProMonoDETR(
        backbone,
        transformer,
        depth_predictor,
        num_classes=cfg["num_classes"],
        num_queries=cfg["num_queries"],
        aux_loss=cfg["aux_loss"],
        num_feature_levels=cfg["num_feature_levels"],
        with_box_refine=cfg["with_box_refine"],
        two_stage=cfg["two_stage"],
        init_box=cfg["init_box"],
        use_dab=cfg["use_dab"],
        group_num=cfg.get("group_num", 11),
        two_stage_dino=cfg["two_stage_dino"],
        near_cfg=near_cfg,
    )
    matcher = build_matcher(cfg)
    weight_dict = {
        "loss_ce": cfg["cls_loss_coef"],
        "loss_bbox": cfg["bbox_loss_coef"],
        "loss_giou": cfg["giou_loss_coef"],
        "loss_dim": cfg["dim_loss_coef"],
        "loss_angle": cfg["angle_loss_coef"],
        "loss_depth": cfg["depth_loss_coef"],
        "loss_center": cfg["3dcenter_loss_coef"],
        "loss_depth_map": cfg["depth_map_loss_coef"],
    }
    geometry_losses = near_cfg.get("losses", {})
    weight_dict.update({
        "loss_horizon_heatmap": geometry_losses.get("horizon_heatmap_loss_coef", 1.0),
        "loss_horizon_line": geometry_losses.get("horizon_line_loss_coef", 0.5),
        "loss_near_touchpoint": geometry_losses.get("near_touchpoint_loss_coef", 1.0),
        "loss_gpro_center_depth": geometry_losses.get("gpro_center_depth_loss_coef", 1.0),
    })
    if cfg["aux_loss"]:
        auxiliary = {
            key: value for key, value in weight_dict.items()
            if key in (
                "loss_ce", "loss_bbox", "loss_giou", "loss_dim",
                "loss_angle", "loss_depth", "loss_center",
            )
        }
        for level in range(cfg["dec_layers"] - 1):
            weight_dict.update({key + "_{}".format(level): value
                                for key, value in auxiliary.items()})
        weight_dict.update({key + "_enc": value for key, value in auxiliary.items()})
    criterion = NearGProCriterion(
        cfg["num_classes"],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg["focal_alpha"],
        losses=[
            "labels", "boxes", "cardinality", "depths", "dims",
            "angles", "center", "depth_map",
        ],
        group_num=cfg.get("group_num", 11),
        near_cfg=near_cfg,
    )
    criterion.to(torch.device(cfg["device"]))
    return model, criterion
