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

    @staticmethod
    def _input_homography(input_p2, original_p2):
        # P_input = H P_original.  P P^+ = I for a rank-3 projection matrix.
        return np.asarray(input_p2, dtype=np.float64) @ np.linalg.pinv(
            np.asarray(original_p2, dtype=np.float64)
        )

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
