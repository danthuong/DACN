"""Geometry modules for the MonoDETR x GPro3D E1 experiment.

All image coordinates accepted by this module are in the resized network-input
frame. KITTI axes are X right, Y down, Z forward. Geometric calculations are
kept in float32 even when the detector runs under autocast.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils import box_ops


TOUCH_POINT_NAMES = ("LF", "RF", "RR", "LR")


def _vertical_segments_to_horizon_slope(
    segments,
    angle_min_deg: float = 70.0,
    angle_max_deg: float = 110.0,
    cluster_tolerance_deg: float = 3.0,
    min_edge_count: int = 5,
    max_angle_std_deg: float = 3.0,
):
    """Estimate a horizon slope from a coherent cluster of vertical segments.

    Segment orientation is represented in degrees modulo 180, avoiding the
    infinite numerical slope of a perfectly vertical line. The returned
    horizon is perpendicular to the length-weighted mean vertical direction.
    This is the clustering/reliability part of GPro3D Algorithm 1 without a
    dependency on scikit-learn's BIRCH implementation.
    """
    candidates = []
    for x1, y1, x2, y2 in segments:
        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        if angle_min_deg <= angle <= angle_max_deg:
            candidates.append((angle, length))

    empty = {
        "valid": False,
        "horizon_slope": 0.0,
        "edge_count": 0,
        "angle_mean_deg": 0.0,
        "angle_std_deg": 0.0,
    }
    if not candidates:
        return empty

    # Choose the densest local angle neighbourhood. Length weighting prevents
    # many tiny texture fragments from defeating a few strong structural edges.
    best_cluster = []
    best_score = -1.0
    for center, _ in candidates:
        cluster = [
            item for item in candidates
            if abs(item[0] - center) <= cluster_tolerance_deg
        ]
        score = sum(item[1] for item in cluster)
        if score > best_score:
            best_cluster = cluster
            best_score = score

    weight_sum = sum(item[1] for item in best_cluster)
    angle_mean = sum(angle * length for angle, length in best_cluster) / weight_sum
    angle_variance = sum(
        length * (angle - angle_mean) ** 2 for angle, length in best_cluster
    ) / weight_sum
    angle_std = math.sqrt(max(angle_variance, 0.0))
    edge_count = len(best_cluster)
    angle_rad = math.radians(angle_mean)
    sine = math.sin(angle_rad)
    valid = (
        edge_count >= int(min_edge_count)
        and angle_std <= float(max_angle_std_deg)
        and abs(sine) > 1e-6
    )
    return {
        "valid": valid,
        "horizon_slope": -math.cos(angle_rad) / sine if valid else 0.0,
        "edge_count": edge_count,
        "angle_mean_deg": angle_mean,
        "angle_std_deg": angle_std,
    }


class VerticalEdgeHorizonRefiner(nn.Module):
    """GPro3D-style OpenCV refinement for evaluation/inference only.

    The operation is intentionally outside the training geometry path because
    GaussianBlur/Canny/Hough and hard clustering are non-differentiable. It
    supplies the horizon slope when reliable vertical structure exists. The
    intercept is subsequently re-fitted from the Horizon Head's ridge.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.enabled = bool(cfg.get("enabled", False))
        kernel_size = int(cfg.get("gaussian_kernel_size", 13))
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size must be a positive odd integer")
        self.gaussian_kernel_size = kernel_size
        self.gaussian_sigma = float(cfg.get("gaussian_sigma", 4.0))
        canny = cfg.get("canny_thresholds", [50, 100])
        if len(canny) != 2 or float(canny[0]) >= float(canny[1]):
            raise ValueError("canny_thresholds must be [low, high]")
        self.canny_low, self.canny_high = int(canny[0]), int(canny[1])
        self.hough_threshold = int(cfg.get("hough_threshold", 5))
        self.min_line_length = float(cfg.get("min_line_length_px", 40.0))
        self.max_line_gap = float(cfg.get("max_line_gap_px", 10.0))
        angle_range = cfg.get("vertical_angle_range_deg", [70.0, 110.0])
        if len(angle_range) != 2 or float(angle_range[0]) >= float(angle_range[1]):
            raise ValueError("vertical_angle_range_deg must be [minimum, maximum]")
        self.angle_min, self.angle_max = float(angle_range[0]), float(angle_range[1])
        self.cluster_tolerance = float(cfg.get("cluster_tolerance_deg", 3.0))
        self.min_edge_count = int(cfg.get("min_edge_count", 5))
        self.max_angle_std = float(cfg.get("max_angle_std_deg", 3.0))
        self.register_buffer(
            "image_mean",
            torch.tensor(cfg.get("image_mean", [0.485, 0.456, 0.406])).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(cfg.get("image_std", [0.229, 0.224, 0.225])).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor, raw_line: torch.Tensor):
        batch = raw_line.shape[0]
        refined = raw_line.detach().float().clone()
        valid = torch.zeros(batch, dtype=torch.bool, device=raw_line.device)
        edge_count = torch.zeros(batch, dtype=torch.long, device=raw_line.device)
        angle_mean = torch.zeros(batch, dtype=torch.float32, device=raw_line.device)
        angle_std = torch.zeros(batch, dtype=torch.float32, device=raw_line.device)
        if not self.enabled:
            return refined.to(raw_line.dtype), valid, edge_count, angle_mean, angle_std

        try:
            import cv2
        except Exception as error:
            raise RuntimeError(
                "vertical-edge refinement is enabled but OpenCV is unavailable"
            ) from error

        rgb_batch = (
            images.detach().float() * self.image_std + self.image_mean
        ).clamp(0.0, 1.0)
        rgb_batch = (
            rgb_batch.mul(255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        )
        for batch_index, rgb in enumerate(rgb_batch):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(
                gray,
                (self.gaussian_kernel_size, self.gaussian_kernel_size),
                sigmaX=self.gaussian_sigma,
                sigmaY=self.gaussian_sigma,
            )
            edges = cv2.Canny(
                blurred, self.canny_low, self.canny_high, apertureSize=3
            )
            detected = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=math.pi / 180.0,
                threshold=self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap,
            )
            segments = [] if detected is None else detected.reshape(-1, 4).tolist()
            estimate = _vertical_segments_to_horizon_slope(
                segments,
                self.angle_min,
                self.angle_max,
                self.cluster_tolerance,
                self.min_edge_count,
                self.max_angle_std,
            )
            edge_count[batch_index] = int(estimate["edge_count"])
            angle_mean[batch_index] = float(estimate["angle_mean_deg"])
            angle_std[batch_index] = float(estimate["angle_std_deg"])
            if estimate["valid"]:
                valid[batch_index] = True
                refined[batch_index, 0] = float(estimate["horizon_slope"])

        return refined.to(raw_line.dtype), valid, edge_count, angle_mean, angle_std


class HorizonHead(nn.Module):
    """Small FPN over projected stride-8/16/32 MonoDETR features."""

    def __init__(self, input_channels: int = 256, fpn_channels: int = 128):
        super().__init__()
        self.laterals = nn.ModuleList(
            [nn.Conv2d(input_channels, fpn_channels, kernel_size=1) for _ in range(3)]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, fpn_channels),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, fpn_channels),
            nn.SiLU(),
            nn.Conv2d(fpn_channels, 1, kernel_size=1),
        )

    def forward(self, srcs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(srcs) < 3:
            raise ValueError("HorizonHead requires stride-8/16/32 feature maps")
        target_size = srcs[0].shape[-2:]
        fused = self.laterals[0](srcs[0])
        for lateral, source in zip(self.laterals[1:], srcs[1:3]):
            fused = fused + F.interpolate(
                lateral(source), size=target_size, mode="bilinear", align_corners=False
            )
        fused = self.fuse(fused / 3.0)
        fused = F.interpolate(fused, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.output(fused)


def decode_horizon_line(
    logits: torch.Tensor,
    input_hw: Tuple[int, int],
    temperature: float = 0.25,
    entropy_confidence: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft-argmax a horizon ridge and fit ``v = k*u + b`` with WLS.

    Returns the line in network-input pixels, per-column expected heatmap row,
    and per-column confidence. The align_corners=False pixel-centre mapping is
    used when converting from heatmap to input pixels.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("horizon logits must have shape [B,1,H,W]")
    if temperature <= 0:
        raise ValueError("softargmax temperature must be positive")

    scores = logits[:, 0].float() / float(temperature)
    batch, heatmap_h, heatmap_w = scores.shape
    probabilities = F.softmax(scores, dim=1)
    rows = torch.arange(heatmap_h, device=scores.device, dtype=torch.float32).view(1, -1, 1)
    expected_row = (probabilities * rows).sum(dim=1)

    if entropy_confidence and heatmap_h > 1:
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=1)
        confidence = (1.0 - entropy / math.log(float(heatmap_h))).clamp_min(1e-4)
    else:
        confidence = torch.ones_like(expected_row)

    columns = torch.arange(heatmap_w, device=scores.device, dtype=torch.float32).view(1, -1)
    weights = confidence
    weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    mean_x = (weights * columns).sum(dim=1, keepdim=True) / weight_sum
    mean_y = (weights * expected_row).sum(dim=1, keepdim=True) / weight_sum
    centered_x = columns - mean_x
    slope_hm = (weights * centered_x * (expected_row - mean_y)).sum(dim=1) / (
        (weights * centered_x.square()).sum(dim=1).clamp_min(1e-6)
    )
    intercept_hm = mean_y[:, 0] - slope_hm * mean_x[:, 0]

    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    scale_x = float(input_w) / float(heatmap_w)
    scale_y = float(input_h) / float(heatmap_h)
    slope = (scale_y / scale_x) * slope_hm
    intercept = (
        scale_y * intercept_hm
        + (scale_y - 1.0) / 2.0
        - slope * (scale_x - 1.0) / 2.0
    )
    return torch.stack((slope, intercept), dim=-1), expected_row, confidence


def refit_horizon_intercept(
    slope: torch.Tensor,
    expected_row: torch.Tensor,
    confidence: torch.Tensor,
    input_hw: Tuple[int, int],
    heatmap_hw: Tuple[int, int],
) -> torch.Tensor:
    """Fit only ``b`` in ``v = slope*u + b`` from the learned heatmap ridge."""
    heatmap_h, heatmap_w = int(heatmap_hw[0]), int(heatmap_hw[1])
    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    scale_x = float(input_w) / float(heatmap_w)
    scale_y = float(input_h) / float(heatmap_h)
    columns = torch.arange(
        heatmap_w, device=expected_row.device, dtype=torch.float32
    ).view(1, -1)
    input_columns = scale_x * columns + (scale_x - 1.0) / 2.0
    input_rows = scale_y * expected_row.float() + (scale_y - 1.0) / 2.0
    weights = confidence.float()
    return (
        weights * (input_rows - slope.float()[:, None] * input_columns)
    ).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)


def make_horizon_heatmap_target(
    lines: torch.Tensor,
    valid: torch.Tensor,
    output_hw: Tuple[int, int],
    input_hw: Tuple[int, int],
    sigma: float = 1.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize pixel-space lines as Gaussian ridges at horizon-head scale."""
    if sigma <= 0:
        raise ValueError("horizon target sigma must be positive")
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    device = lines.device
    dtype = lines.dtype
    scale_x = float(input_w) / float(output_w)
    scale_y = float(input_h) / float(output_h)
    columns = torch.arange(output_w, device=device, dtype=dtype).view(1, 1, output_w)
    rows = torch.arange(output_h, device=device, dtype=dtype).view(1, output_h, 1)
    input_columns = scale_x * columns + (scale_x - 1.0) / 2.0
    input_rows = lines[:, 0:1, None] * input_columns + lines[:, 1:2, None]
    target_rows = (input_rows - (scale_y - 1.0) / 2.0) / scale_y
    heatmap = torch.exp(-0.5 * ((rows - target_rows) / float(sigma)).square()).unsqueeze(1)
    column_valid = (target_rows[:, 0] >= -3.0 * sigma) & (
        target_rows[:, 0] <= output_h - 1 + 3.0 * sigma
    )
    mask = valid.bool().view(-1, 1, 1, 1) & column_valid[:, None, None, :]
    return heatmap * mask.to(dtype), mask


class TouchpointHead(nn.Module):
    def __init__(self, hidden_dim: int = 256, max_touchpoints: int = 4,
                 offset_range: Sequence[float] = (-0.25, 1.25)):
        super().__init__()
        self.max_touchpoints = max_touchpoints
        if len(offset_range) != 2 or float(offset_range[0]) >= float(offset_range[1]):
            raise ValueError('touchpoint.offset_range must be [minimum, maximum]')
        self.offset_min = float(offset_range[0])
        self.offset_max = float(offset_range[1])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * max_touchpoints),
        )

    def forward(self, query_features: torch.Tensor, boxes: torch.Tensor):
        raw = self.mlp(query_features).view(
            *query_features.shape[:-1], self.max_touchpoints, 2
        )
        relative = (
            (self.offset_max - self.offset_min) * raw.sigmoid()
            + self.offset_min)
        xyxy = box_ops.box_cxcylrtb_to_xyxy(boxes)
        top_left = xyxy[..., None, :2]
        size = (xyxy[..., 2:] - xyxy[..., :2]).clamp_min(1e-6)[..., None, :]
        absolute = top_left + relative * size
        return absolute, relative, raw


def horizon_line_to_ground_plane(
    line: torch.Tensor, p2: torch.Tensor, camera_height_m: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a pixel horizon to a Y-down ground plane ``n.X + d = 0``."""
    with torch.cuda.amp.autocast(enabled=False):
        line = line.float()
        p2 = p2.float()
        homogeneous_line = torch.stack(
            (-line[:, 0], torch.ones_like(line[:, 0]), -line[:, 1]), dim=-1
        )
        normal = torch.bmm(p2[:, :, :3].transpose(1, 2), homogeneous_line.unsqueeze(-1)).squeeze(-1)
        norm = normal.norm(dim=-1, keepdim=True)
        valid = torch.isfinite(normal).all(dim=-1) & (norm[:, 0] > 1e-8)
        normal = normal / norm.clamp_min(1e-8)
        sign = torch.where(normal[:, 1:2] < 0, -torch.ones_like(normal[:, 1:2]), torch.ones_like(normal[:, 1:2]))
        normal = normal * sign
        offset = torch.full_like(normal[:, :1], -float(camera_height_m))
        plane = torch.cat((normal, offset), dim=-1)
        valid &= normal[:, 1] > 1e-6
    return plane, valid


def raycast_touchpoints_to_plane(
    touchpoints_normalized: torch.Tensor,
    p2: torch.Tensor,
    plane: torch.Tensor,
    image_hw: Tuple[int, int],
    min_abs_denominator: float = 0.015,
    min_depth: float = 2.0,
    max_depth: float = 65.0,
) -> Dict[str, torch.Tensor]:
    """Backproject normalized touchpoints and intersect them with ground."""
    with torch.cuda.amp.autocast(enabled=False):
        points = touchpoints_normalized.float()
        projection = p2.float()
        plane = plane.float()
        batch, queries, point_count, _ = points.shape
        image_h, image_w = int(image_hw[0]), int(image_hw[1])
        pixels = points * points.new_tensor([float(image_w), float(image_h)])
        homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)

        matrix = projection[:, :, :3]
        inverse = torch.inverse(matrix)
        camera_center = -torch.bmm(inverse, projection[:, :, 3:4]).squeeze(-1)
        directions = torch.einsum("bij,bqpj->bqpi", inverse, homogeneous)
        normal, offset = plane[:, :3], plane[:, 3]
        denominator = torch.einsum("bi,bqpi->bqp", normal, directions)
        numerator = -(torch.einsum("bi,bi->b", normal, camera_center) + offset)
        # Protect both denominator signs without changing valid ray geometry.
        safe_denominator = torch.where(
            denominator >= 0,
            denominator.clamp_min(min_abs_denominator),
            denominator.clamp_max(-min_abs_denominator),
        )
        distance = numerator[:, None, None] / safe_denominator
        intersections = camera_center[:, None, None, :] + distance[..., None] * directions
        depths = intersections[..., 2]
        valid = (
            torch.isfinite(intersections).all(dim=-1)
            & torch.isfinite(distance)
            & (denominator.abs() >= min_abs_denominator)
            & (distance > 0)
            & (depths >= min_depth)
            & (depths <= max_depth)
        )
        valid_float = valid.to(depths.dtype)
        valid_count = valid_float.sum(dim=-1)
        object_depth = (depths * valid_float).sum(dim=-1) / valid_count.clamp_min(1.0)
        object_valid = valid_count > 0
        stability = (denominator.abs() / (4.0 * min_abs_denominator)).clamp(max=1.0)
        reliability = (stability * valid_float).sum(dim=-1) / float(point_count)
        object_depth = torch.where(object_valid, object_depth, torch.zeros_like(object_depth))
    return {
        "intersections": intersections,
        "point_depth": depths,
        "point_valid": valid,
        "point_stability": stability,
        "object_depth": object_depth,
        "object_valid": object_valid,
        "reliability": reliability,
    }


class GeometryE1(nn.Module):
    """Shared horizon/touchpoint geometry and the two E1 fusion variants."""

    def __init__(
        self,
        cfg: Dict,
        hidden_dim: int,
        num_queries: int,
        depth_bin_values: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg["fusion_mode"]
        self.num_queries = int(num_queries)
        horizon_cfg = cfg.get("horizon", {})
        point_cfg = cfg.get("touchpoint", {})
        plane_cfg = cfg.get("plane", {})
        spatial_cfg = cfg.get("spatial_projection", {})
        fusion_cfg = cfg.get("fusion", {})
        mode_cfg = fusion_cfg.get("mode_settings", {})

        self.horizon_head = HorizonHead(hidden_dim, int(horizon_cfg.get("fpn_channels", 128)))
        self.vertical_edge_refiner = VerticalEdgeHorizonRefiner(
            horizon_cfg.get("vertical_edge_refinement", {}))
        self.touchpoint_head = TouchpointHead(
            hidden_dim,
            int(point_cfg.get("max_touchpoints", 4)),
            point_cfg.get("offset_range", [-0.25, 1.25]),
        )
        self.touchpoint_box_detach = bool(point_cfg.get("box_detach", False))
        self.softargmax_temperature = float(horizon_cfg.get("softargmax_temperature", 0.25))
        self.entropy_confidence = bool(horizon_cfg.get("entropy_confidence", True))
        self.camera_height_m = float(plane_cfg.get("camera_height_m", 1.65))
        self.min_abs_ray_plane_dot = float(plane_cfg.get("min_abs_ray_plane_dot", 0.015))
        self.min_depth = float(plane_cfg.get("valid_min_depth_m", 2.0))
        self.max_depth = float(plane_cfg.get("valid_max_depth_m", 65.0))
        self.consistency_scale_m = float(
            plane_cfg.get("depth_consistency_scale_m", 5.0))
        self.box_softness = float(spatial_cfg.get("box_edge_softness_grid_px", 1.25))
        self.class_index = int(point_cfg.get("class_index", 1))
        self.gaussian_sigma_m = float(mode_cfg.get("gaussian_sigma_m", 2.5))
        self.log_prior_min = float(mode_cfg.get("log_prior_clamp", [-12.0, 0.0])[0])
        self.log_prior_max = float(mode_cfg.get("log_prior_clamp", [-12.0, 0.0])[1])
        self.register_buffer("depth_bin_values", depth_bin_values.detach().clone(), persistent=False)

        self.alpha_min = float(fusion_cfg.get("alpha_min", 0.0))
        self.alpha_max = float(fusion_cfg.get("alpha_max", 1.0))
        initial_alpha = float(fusion_cfg.get("initial_alpha", 0.01))
        if not self.alpha_min < self.alpha_max:
            raise ValueError("fusion.alpha_min must be smaller than fusion.alpha_max")
        if not self.alpha_min < initial_alpha < self.alpha_max:
            raise ValueError(
                "fusion.initial_alpha must lie strictly between alpha_min and alpha_max")
        initial_ratio = (
            (initial_alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
        )
        initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
        # Optimize an unconstrained logit rather than a directly clamped gate.
        # A hard clamp has zero gradient after the raw parameter crosses a
        # boundary, which previously made an alpha=0 fusion gate irreversible.
        self.fusion_alpha_logit = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32))
        if self.mode == "feature_level":
            self.feature_adapter = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            )
        elif self.mode != "distribution_level":
            raise ValueError("fusion_mode must be feature_level or distribution_level")

    @property
    def alpha(self):
        scale = self.alpha_max - self.alpha_min
        return self.alpha_min + scale * self.fusion_alpha_logit.sigmoid()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Migrate the old directly-clamped alpha stored by E1 checkpoints."""
        old_key = prefix + "fusion_alpha"
        new_key = prefix + "fusion_alpha_logit"
        if old_key in state_dict and new_key not in state_dict:
            old_alpha = state_dict.pop(old_key).float().clamp(
                self.alpha_min, self.alpha_max)
            ratio = (
                (old_alpha - self.alpha_min)
                / (self.alpha_max - self.alpha_min)
            ).clamp(1e-6, 1.0 - 1e-6)
            state_dict[new_key] = torch.logit(ratio)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def predict_horizon(self, srcs, image_hw, images=None):
        logits = self.horizon_head(srcs)
        raw_line, expected_row, confidence = decode_horizon_line(
            logits,
            image_hw,
            temperature=self.softargmax_temperature,
            entropy_confidence=self.entropy_confidence,
        )
        batch = raw_line.shape[0]
        refined_line = raw_line
        edge_valid = torch.zeros(batch, dtype=torch.bool, device=raw_line.device)
        edge_count = torch.zeros(batch, dtype=torch.long, device=raw_line.device)
        edge_angle = torch.zeros(batch, dtype=torch.float32, device=raw_line.device)
        edge_angle_std = torch.zeros(batch, dtype=torch.float32, device=raw_line.device)
        # OpenCV refinement is a hard, non-differentiable post-process. Keep it
        # out of training so pass-2 losses can still optimize the Horizon Head.
        if not self.training and self.vertical_edge_refiner.enabled:
            if images is None:
                raise ValueError("images are required for vertical-edge refinement")
            (refined_line, edge_valid, edge_count,
             edge_angle, edge_angle_std) = self.vertical_edge_refiner(
                images, raw_line)
            refined_intercept = refit_horizon_intercept(
                refined_line[:, 0], expected_row, confidence,
                image_hw, logits.shape[-2:])
            refined_line[:, 1] = torch.where(
                edge_valid, refined_intercept.to(refined_line.dtype),
                raw_line[:, 1])
        return {
            "logits": logits,
            "raw_line": raw_line,
            "line": refined_line,
            "expected_row": expected_row,
            "confidence": confidence,
            "vertical_edge_valid": edge_valid,
            "vertical_edge_count": edge_count,
            "vertical_edge_angle_deg": edge_angle,
            "vertical_edge_angle_std_deg": edge_angle_std,
        }

    def predict_geometry(
        self,
        query_features: torch.Tensor,
        boxes: torch.Tensor,
        class_logits: torch.Tensor,
        horizon_line: torch.Tensor,
        horizon_confidence: torch.Tensor,
        p2_input: torch.Tensor,
        image_hw: Tuple[int, int],
        depth_grid_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        touch_boxes = boxes.detach() if self.touchpoint_box_detach else boxes
        touch_abs, touch_rel, touch_raw = self.touchpoint_head(
            query_features, touch_boxes)
        canonical = min(self.num_queries, touch_abs.shape[1])
        touch_canonical = touch_abs[:, :canonical]
        boxes_canonical = boxes[:, :canonical]
        logits_canonical = class_logits[:, :canonical]

        plane, plane_valid = horizon_line_to_ground_plane(
            horizon_line, p2_input, self.camera_height_m
        )
        rays = raycast_touchpoints_to_plane(
            touch_canonical,
            p2_input,
            plane,
            image_hw,
            self.min_abs_ray_plane_dot,
            self.min_depth,
            self.max_depth,
        )
        # The head deliberately permits slight out-of-box offsets, but a ray
        # outside the actual input image is not an observed ground contact.
        in_image = (
            (touch_canonical[..., 0] >= 0.0)
            & (touch_canonical[..., 0] < 1.0)
            & (touch_canonical[..., 1] >= 0.0)
            & (touch_canonical[..., 1] < 1.0)
        )
        rays["point_valid"] = rays["point_valid"] & in_image
        valid_float = rays["point_valid"].to(rays["point_depth"].dtype)
        valid_count = valid_float.sum(dim=-1)
        rays["object_depth"] = (
            (rays["point_depth"] * valid_float).sum(dim=-1)
            / valid_count.clamp_min(1.0)
        )
        rays["object_valid"] = valid_count > 0
        rays["reliability"] = (
            rays["point_stability"] * valid_float
        ).sum(dim=-1) / float(touch_canonical.shape[-2])
        rays["object_valid"] = rays["object_valid"] & plane_valid[:, None]
        rays["object_depth"] = torch.where(
            rays["object_valid"], rays["object_depth"],
            torch.zeros_like(rays["object_depth"]))
        class_confidence = logits_canonical.sigmoid()[..., self.class_index]
        horizon_reliability = horizon_confidence.float().mean(dim=-1).clamp(0.0, 1.0)
        deviations = (
            rays["point_depth"] - rays["object_depth"][..., None]
        ).square() * valid_float
        depth_std = torch.sqrt(
            deviations.sum(dim=-1) / valid_count.clamp_min(1.0) + 1e-8)
        consistency = torch.exp(-depth_std / self.consistency_scale_m)
        object_confidence = (
            class_confidence * rays["reliability"] * consistency
            * horizon_reliability[:, None]
            * rays["object_valid"].to(class_confidence.dtype)
        )
        spatial = self._spatial_projection(
            rays["object_depth"], object_confidence, boxes_canonical, depth_grid_hw
        )
        return {
            "touchpoints_abs": touch_abs,
            "touchpoints_rel": touch_rel,
            "touchpoints_raw": touch_raw,
            "plane": plane,
            "plane_valid": plane_valid,
            "object_depth": rays["object_depth"],
            "object_valid": rays["object_valid"],
            "object_reliability": object_confidence,
            "horizon_reliability": horizon_reliability,
            "object_depth_std": depth_std,
            "touchpoint_intersections": rays["intersections"],
            "touchpoint_ray_valid": rays["point_valid"],
            **spatial,
        }

    def _spatial_projection(self, depths, confidence, boxes, grid_hw):
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        dtype, device = boxes.dtype, boxes.device
        x = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) / float(grid_w)
        y = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) / float(grid_h)
        x = x.view(1, 1, 1, grid_w)
        y = y.view(1, 1, grid_h, 1)
        xyxy = box_ops.box_cxcylrtb_to_xyxy(boxes)
        xmin, ymin, xmax, ymax = [xyxy[..., index, None, None] for index in range(4)]
        tau_x = self.box_softness / float(grid_w)
        tau_y = self.box_softness / float(grid_h)
        horizontal = torch.sigmoid((x - xmin) / tau_x) * torch.sigmoid((xmax - x) / tau_x)
        vertical = torch.sigmoid((y - ymin) / tau_y) * torch.sigmoid((ymax - y) / tau_y)
        masks = horizontal * vertical
        weights = masks * confidence[..., None, None]
        weight_sum = weights.sum(dim=1, keepdim=True)
        coverage = 1.0 - torch.exp(-weight_sum)
        depth_map = (weights * depths[..., None, None]).sum(dim=1, keepdim=True) / weight_sum.clamp_min(1e-6)
        confidence_map = (weights * confidence[..., None, None]).sum(dim=1, keepdim=True) / weight_sum.clamp_min(1e-6)
        depth_map = torch.where(weight_sum > 0, depth_map, torch.zeros_like(depth_map))
        confidence_map = torch.where(weight_sum > 0, confidence_map, torch.zeros_like(confidence_map))
        return {
            "spatial_prior": depth_map,
            "valid_coverage": coverage,
            "confidence_map": confidence_map,
            "object_spatial_weights": weights,
        }

    def fuse_feature(self, source: torch.Tensor, geometry: Dict[str, torch.Tensor]):
        depth_normalized = geometry["spatial_prior"] / self.max_depth
        adapter_input = torch.cat(
            (depth_normalized, geometry["valid_coverage"], geometry["confidence_map"]), dim=1
        ).to(source.dtype)
        residual = self.feature_adapter(adapter_input)
        return source + self.alpha.to(source.dtype) * geometry["valid_coverage"].to(source.dtype) * residual

    def fuse_distribution(self, logits: torch.Tensor, geometry: Dict[str, torch.Tensor]):
        foreground_bins = self.depth_bin_values[:-1].to(device=logits.device, dtype=torch.float32)
        depths = geometry["object_depth"].float()
        object_weights = geometry["object_spatial_weights"].float()
        gaussian_logits = -0.5 * (
            (foreground_bins.view(1, 1, -1) - depths[..., None]) / self.gaussian_sigma_m
        ).square()
        object_probability = F.softmax(gaussian_logits, dim=-1)
        mixture = torch.einsum("bqd,bqhw->bdhw", object_probability, object_weights)
        mass = object_weights.sum(dim=1, keepdim=True)
        mixture = mixture / mass.clamp_min(1e-8)
        log_prior = mixture.clamp_min(1e-8).log()
        log_prior = log_prior - log_prior.max(dim=1, keepdim=True)[0]
        log_prior = log_prior.clamp(self.log_prior_min, self.log_prior_max)
        adjustment = self.alpha.float() * geometry["valid_coverage"].float() * log_prior
        fused_foreground = logits[:, :-1].float() + adjustment
        fused = torch.cat((fused_foreground.to(logits.dtype), logits[:, -1:]), dim=1)
        return fused


def horizon_focal_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Numerically stable focal-like BCE for a soft Gaussian ridge."""
    mask = mask.to(logits.dtype)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    probability = logits.sigmoid()
    positive = -target * (1.0 - probability).square() * F.logsigmoid(logits)
    negative = -(1.0 - target).pow(4) * probability.square() * F.logsigmoid(-logits)
    return ((positive + negative) * mask).sum() / mask.sum().clamp_min(1.0)
