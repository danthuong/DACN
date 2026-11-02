"""KITTI adapter for the near-contact Z-map replacement ablation.

The upstream KITTI dataset is left untouched.  This subclass reads the new
``near_contact`` pseudo-label, applies the exact input-image homography, and
exports both its absolute normalized coordinate and its GT-box-relative
offset.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.helpers.dataloader_helper import my_worker_init_fn


class NearGProKITTIDataset(KITTI_Dataset):
    def __init__(self, split, cfg):
        self.near_cfg = cfg.get("near_gpro") or {}
        adapted = dict(cfg)
        adapted["geometry_e1"] = {
            "enabled": True,
            "pseudo_labels": self.near_cfg.get("pseudo_labels", {}),
            "touchpoint": {
                "require_inside_image": self.near_cfg.get(
                    "near_touchpoint", {}
                ).get("require_inside_image", True)
            },
        }
        super().__init__(split=split, cfg=adapted)
        # With random crop and ``depth_scale: normal`` the depth label becomes
        # Z * crop_scale, while ray/plane on the true input P2 returns Z.
        # Opt-in so the earlier no-crop ablations stay bit-for-bit identical.
        self.depth_consistent_geometry_calib = bool(
            self.near_cfg.get("augmentation", {}).get(
                "depth_consistent_geometry_calib", False
            )
        )

    @staticmethod
    def _input_homography(input_p2, original_p2):
        # P_input = H P_original.  P P^+ = I for a rank-3 projection matrix.
        return np.asarray(input_p2, dtype=np.float64) @ np.linalg.pinv(
            np.asarray(original_p2, dtype=np.float64)
        )

    @staticmethod
    def _depth_consistent_calib(input_p2, depth_factor):
        """Virtual camera whose ground-plane depth equals ``Z * depth_factor``.

        Scaling the focal length about the principal point divides every ray
        slope by ``depth_factor``, so a ray hits the plane ``Y = h`` at the
        same X and ``depth_factor`` times the depth.  Pixel coordinates (and
        therefore horizon/touchpoint labels) are unchanged.
        """
        p2 = np.asarray(input_p2, dtype=np.float64)
        s = float(depth_factor)
        cx, cy = p2[0, 2], p2[1, 2]
        scale = np.array(
            [[s, 0.0, cx * (1.0 - s)], [0.0, s, cy * (1.0 - s)], [0.0, 0.0, 1.0]]
        )
        return (scale @ p2).astype(np.float32)

    def _label_depth_factor(self, targets, info):
        # affine[0, 0] = resolution / (img_size * crop_scale); flip lives in a
        # separate matrix, so this entry is always positive.
        crop_scale = float(self.resolution[0]) / (
            float(info["img_size"][0]) * float(targets["affine"][0, 0])
        )
        if self.depth_scale == "normal":
            return crop_scale
        if self.depth_scale == "inverse":
            return 1.0 / crop_scale
        return 1.0

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if self.split == "test":
            return result
        inputs, calib, targets, info = result
        near_point = np.zeros((self.max_objs, 2), dtype=np.float32)
        near_offset = np.zeros((self.max_objs, 2), dtype=np.float32)
        near_valid = np.zeros((self.max_objs,), dtype=bool)

        frame_id = "{:06d}".format(int(info["img_id"]))
        pseudo_record = self.pseudo_records.get(frame_id) or {}
        pseudo_objects = {
            int(obj["object_id"]): obj for obj in pseudo_record.get("objects", [])
        }
        original_p2 = self.get_calib(int(info["img_id"])).P2
        homography = self._input_homography(targets["input_calib"], original_p2)
        require_inside = bool(
            self.near_cfg.get("near_touchpoint", {}).get(
                "require_inside_image", True
            )
        )

        for slot, object_id in enumerate(targets["object_ids"]):
            if int(object_id) < 0:
                continue
            near = (pseudo_objects.get(int(object_id)) or {}).get("near_contact") or {}
            if not near.get("projection_valid") or near.get("pixel_2d") is None:
                continue
            raw = np.asarray([*near["pixel_2d"], 1.0], dtype=np.float64)
            transformed = homography @ raw
            if not np.isfinite(transformed).all() or abs(transformed[2]) <= 1e-8:
                continue
            pixel = transformed[:2] / transformed[2]
            inside = bool(
                0.0 <= pixel[0] < float(self.resolution[0])
                and 0.0 <= pixel[1] < float(self.resolution[1])
            )
            if require_inside and not inside:
                continue
            normalized = pixel / self.resolution
            cx, cy, left, right, top, bottom = targets["boxes_3d"][slot]
            width = max(float(left + right), 1e-6)
            height = max(float(top + bottom), 1e-6)
            relative = np.asarray(
                [
                    (normalized[0] - (cx - left)) / width,
                    (normalized[1] - (cy - top)) / height,
                ],
                dtype=np.float32,
            )
            if np.isfinite(relative).all():
                near_point[slot] = normalized.astype(np.float32)
                near_offset[slot] = relative
                near_valid[slot] = True

        targets["near_touchpoint"] = near_point
        targets["near_touchpoint_box_offset"] = near_offset
        targets["near_touchpoint_valid"] = near_valid
        # Replace only after all pixel labels were mapped with the true P2.
        if self.depth_consistent_geometry_calib:
            factor = self._label_depth_factor(targets, info)
            if abs(factor - 1.0) > 1e-6:
                geometry_calib = self._depth_consistent_calib(
                    targets["input_calib"], factor
                )
                targets["input_calib"] = geometry_calib
                info["input_calib"] = geometry_calib
        return inputs, calib, targets, info


def build_near_gpro_dataloader(cfg, workers=4):
    train_set = NearGProKITTIDataset(cfg["train_split"], cfg)
    test_set = NearGProKITTIDataset(cfg["test_split"], cfg)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        num_workers=workers,
        worker_init_fn=my_worker_init_fn,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["batch_size"],
        num_workers=workers,
        worker_init_fn=my_worker_init_fn,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, test_loader
