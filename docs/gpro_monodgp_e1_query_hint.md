# E1 — GPro near-contact depth trên MonoDGP: query hint + anchor depth

**Workspace làm việc: `mixup3d/monodgp`** — bản clone repo MonoDGP của Đại
(fork `caoTayTang/monodgp`). Mọi đường dẫn code trong file này là tương đối so với
workspace đó. Repo DACN (`/home/ml4u/BKTeam/QuangDuong/DACN`) chỉ dùng làm **nguồn**:
copy code GPro sang và sinh pseudo label; không sửa code trong DACN. File này gồm:

- **Phần A (nền tảng, dùng chung cho E1 và E2):** dữ liệu, horizon, ROI
  near-contact, ray–plane → Z_near, near→center → Z_center, depth
  `Z_center + Z_err`, loss, tích hợp DN và MixUp3D.
- **Phần B (riêng E1):** đưa Z_near vào query của 3D decoder (GeoEmbed).

E2 (`gpro_monodgp_e2_geo_depth_memory.md`) dùng lại toàn bộ Phần A, chỉ thay
Phần B bằng geometric depth memory.

> **Lưu ý tên:** "MixUp3D" trong file này là **data augmentation** của MonoDGP
> (`dataset.random_mixup3d`, `kitti_dataset.py`: trộn 50/50 với một ảnh khác cùng
> intrinsics và giữ nhãn của cả hai ảnh). Nó **không liên quan** tới tên folder
> workspace `mixup3d/`.

---

## 0. Bối cảnh và quy tắc repo

### 0.1 Quy tắc bắt buộc (từ `docs/ONBOARDING.md` của MonoDGP)

- Làm trong `mixup3d/monodgp`, trên branch riêng, ví dụ `feat/gpro-e1`, tách từ `main`.
  Kiểm tra `git remote -v` trước khi push: không push lên repo MonoDGP gốc, và không
  push lên fork của Đại khi chưa thống nhất với Đại.
- Đường dẫn trong ONBOARDING (`/home/cdai/MonoDGP`) là workspace của Đại, không phải
  của mình; đừng sửa hay chạy gì trong đó.
- **Mọi code mới mặc định tắt** (bật qua block `gpro:` trong config). Config cũ
  (`b0*`, `b3_v4`, …) phải cho kết quả bit-identical. Thêm test khẳng định điều này.
- Có test trong `tests/`; chạy CPU được với các env flag trong ONBOARDING.
- GPU: chỉ GPU vật lý 1, **hỏi trước khi chạy**, một training mỗi lúc. Job CPU nặng
  chạy `nice -n 19`, `OMP_NUM_THREADS=2`.
- Kết quả 1 seed chỉ để sàng lọc (seed noise ~1.4 AP). Claim cần 3 seed (444, 2027, 3407).

### 0.2 Những gì MonoDGP đang làm (đã đọc code, commit `e41a19e`; nếu workspace ở commit khác thì kiểm tra lại bảng này)

| Thành phần | File / dòng | Ghi chú cho GPro |
|---|---|---|
| `srcs` sau `input_proj` (+ camera embedding nếu bật) | `monodgp.py` forward, trước `self.region_head(srcs)` | Horizon và ROI lấy `srcs` **ở đây**, trước RSH |
| RSH | `enhanced_srcs, region_probs, seg_embed = self.region_head(srcs)` | RSH trả về list mới, không sửa `srcs` tại chỗ → chỉ cần giữ tham chiếu `srcs_pre = srcs` trước dòng `srcs = enhanced_srcs` |
| Depth predictor | `self.depth_predictor(srcs, masks[1], seg_embed[1] + pos[1])` | Memory = `depth_encoder(depth_head(fuse(f8,f16,f32)))`. **Không có** depth position embedding (`#+ depth_pos_embed_ip` đã bị comment) |
| 2D decoder | `det2d_transformer(...)`, output `hs_2d`, `inter_coord`, `inter_class` | Box 2D cho ROI = `inter_coord[-1]` |
| 3D decoder | `det3d_transformer(intermediate_output, hs_2d[-1], depth_pos_embed, dn_meta)` | `tgt = hs_2d[-1]`, `query_pos = None`, ref = box cuối của 2D decoder, `mask_depth = None` |
| Heads 3D | vòng `for lvl in range(hs.shape[0])` | class + box **cuối cùng** đến từ 3D decoder; `bbox_embed`/`class_embed` dùng chung index với 2D decoder |
| Depth | `depth_geo = H·|f| / h_px` (`calibs[:,0,0].abs()`, `img_sizes`), `depth = depth_geo + err`, `logσ` | GPro thay **anchor** `depth_geo` khi hợp lệ |
| DN | `dn_meta`, DN slot ở đầu: `[0:pad_size]`, detector `[pad_size:]` | Mọi tensor per-query có shape `[B, pad + Qdet, …]` |
| Group DETR | train `Qdet = 50 × 11 = 550`, eval 50 | Group 0 = `[pad : pad+50]` |
| Matching | `SetCriterion.forward`: indices trên output cuối (3D decoder) | GPro loss dùng **cùng indices** này |
| Dataset | `kitti_dataset.py`: `image_transform` (affine ∘ flip), `crop_scale`, `depth_scale: normal`, MixUp3D p=0.5, `object_keys = img_idx*100 + line_idx` | `object_keys` khớp trực tiếp `object_id` của pseudo label DACN |

### 0.3 Vì sao làm, và rủi ro đã biết (từ `docs/WORKLOG.md` của MonoDGP)

- **3.9:** 74% lỗi depth có hệ thống được giải thích bởi kích thước thật; detector
  co chiều cao về trung bình; depth MonoDGP phụ thuộc `f·H/h`. Z_near từ mặt
  phẳng đất **không phụ thuộc chiều cao xe** → đánh trực tiếp vào nguồn lỗi chính.
- **3.13 (rủi ro):** cue mặt đất thô (đáy box 2D + chiều cao camera cố định, không
  pitch, không near→center) sai trung vị **9.7%**, detector **2.4%**, trọng số
  trộn tối ưu = 0. GPro khác ở 3 điểm (horizon học được theo từng ảnh, điểm
  near-contact học được thay vì đáy box, near→center), nhưng **phải qua Gate 0
  (§A10) trước khi tốn GPU**.
- **3.14/3.16:** 93–99% gradient vào head chiều cao đến từ depth loss; chặn hoàn
  toàn (b0_detach) làm tụt 2.7 AP. Với anchor GPro, object hợp lệ **không còn**
  gradient depth → chiều cao. Phải log R² chiều cao và so với B0 (§A9).

---

# PHẦN A — Nền tảng (dùng chung E1/E2)

## A1. Port code từ DACN

Trong `mixup3d/monodgp`, tạo `lib/models/monodgp/gpro/` và copy từ DACN (giữ nguyên
logic, chỉ sửa import):

| Từ DACN | Sang MonoDGP | Lấy gì |
|---|---|---|
| `lib/models/monodetr/geometry_e1.py` | `gpro/geometry.py` | `HorizonHead`, `decode_horizon_line`, `refit_horizon_intercept`, `VerticalEdgeHorizonRefiner` (+ `_vertical_segments_to_horizon_slope`), `make_horizon_heatmap_target`, `horizon_focal_loss`, `horizon_line_to_ground_plane`, `raycast_touchpoints_to_plane`. **Không** lấy `GeometryE1`, `TouchpointHead` |
| `lib/models/monodetr/near_gpro_zmap.py` | `gpro/near_contact.py` | `NearContactROIHead`; từ `NearGProGeometry`: `predict_horizon`, `_decode_alpha`, `_near_offset_camera` |
| `utils/torch_compat.py` | `utils/torch_compat.py` | `cuda_autocast`, `meshgrid_ij` |
| `lib/models/monodetr/monodetr.py` `SetCriterion.loss_horizon` | criterion MonoDGP | logic loss horizon (heatmap focal + line smooth-L1 dạng "stable") |
| `near_gpro_zmap.py` `NearGProCriterion.loss_near_touchpoint` | criterion MonoDGP | logic loss touchpoint box-relative |
| `lib/datasets/kitti/near_gpro_dataset.py` | `kitti_dataset.py` | logic map nhãn near-contact; `_depth_consistent_calib` |
| `generate_near_gpro_pseudo_labels.py` + `pseudo_label_generation/` + `generate_geometry_pseudo_labels.py` | (chạy ở DACN, không port) | sinh jsonl |

`VerticalEdgeHorizonRefiner` cần OpenCV (`cv2`). Kiểm tra `python -c "import cv2"`
trong env `monoia`; thiếu thì cài `opencv-python-headless`.

`utils/box_ops.box_cxcylrtb_to_xyxy` đã có sẵn trong MonoDGP.

## A2. Pseudo label

Sinh ở repo DACN (không cần sửa code), dùng đúng `k_l`, `k_w` như config:

```bash
cd /home/ml4u/BKTeam/QuangDuong/DACN
for s in train val; do
  nice -n 19 python3 generate_near_gpro_pseudo_labels.py \
    --dataset-root /media/ml4u/Samsung_990_2TB1/datasets/Kitti3D --split $s \
    --output outputs/near_gpro_pseudo_labels/$s.jsonl --k-l 0.65 --k-w 0.85
done
```

Nếu đã có sẵn từ các run DACN thì dùng lại. Nên copy 2 file jsonl vào
`mixup3d/monodgp/data/near_gpro_pseudo_labels/` (thư mục bị gitignore) để workspace
không phụ thuộc DACN lúc train. Mỗi dòng jsonl là một frame:

```
{"frame_id": "000123",
 "horizon_line": {"valid": bool, "slope": k, "intercept": b},     # pixel ảnh gốc, v = k·u + b
 "objects": [{"object_id": <số dòng trong label_2>,
              "near_contact": {"projection_valid": bool, "pixel_2d": [u, v] | null,
                               "camera_3d": [X, Y, Z], ...}, ...}]}
```

## A3. Dataset (`lib/datasets/kitti/kitti_dataset.py`)

Bật bằng `dataset.gpro_pseudo_label_root` (None → tắt, không thêm key nào).

### A3.1 Load

Trong `__init__`: đọc `<root>/<split>.jsonl` vào dict `frame_id → record`
(giống `KITTI_Dataset._load_pseudo_records` của DACN). `trainval` → đọc cả hai file.
Assert `camera_focal_augmentation == False` khi GPro bật (chế độ đó thay đổi
calib/transform; chưa hỗ trợ).

### A3.2 Calib cho geometry (per-image)

```python
input_p2      = image_transform @ calib.P2                    # [3,4], frame ảnh input (gồm flip + affine)
geometry_calib = depth_consistent_calib(input_p2, factor)     # factor = crop_scale nếu depth_scale=='normal',
                                                              # 1/crop_scale nếu 'inverse', 1 nếu 'none'
```

`depth_consistent_calib` = `NearGProKITTIDataset._depth_consistent_calib` của DACN:
nhân focal với `factor` quanh principal point, để ray ∩ plane cho đúng
`Z·crop_scale` — khớp nhãn `depth`. Lúc val (không crop) `factor = 1`.

Đưa vào cả `targets['geometry_calib']` **và** `info['geometry_calib']`. Với split
`test` (return sớm), phải gán `info['geometry_calib']` trước dòng `return`.

### A3.3 Horizon (per-image)

Lấy 2 điểm của đường horizon trong ảnh gốc `(0, b)`, `(W−1, k(W−1)+b)`, biến
đổi bằng `image_transform` (đã gồm flip), fit lại `(k', b')` trong pixel input.

```
targets['horizon_line']  [2]
targets['horizon_valid'] bool = record.valid AND (not random_mix_flag)
```

**MixUp3D:** ảnh trộn có hai mặt đất chồng nhau → `horizon_valid = False` (không
có horizon loss trên ảnh trộn).

### A3.4 Near-contact (per-object slot, cả object chính và object MixUp)

Với mỗi slot `j` có `object_keys[j] = img_idx*100 + line_idx`:

```python
rec  = pseudo[frame_of(img_idx)]; obj = rec.objects[line_idx]['near_contact']
pix  = image_transform @ [u, v, 1]   → chia z → pixel input           # object MixUp dùng record của random_index,
near_touchpoint[j]            = pix / resolution                         # cùng image_transform (ảnh phụ đã flip cùng)
near_touchpoint_box_offset[j] = ((x − (cx3d − l)) / (l + r), (y − (cy3d − t)) / (t + b))   # theo boxes_3d GT
near_touchpoint_valid[j]      = projection_valid & inside_image & finite
near_depth[j]                 = camera_3d[2] * factor                     # cùng factor như depth
from_mixup[j]                 = (img_idx != index)
```

Thêm các key per-object vào `Trainer.prepare_targets.key_list`:
`near_touchpoint, near_touchpoint_box_offset, near_touchpoint_valid, near_depth, from_mixup`.
Key per-image (`horizon_line, horizon_valid, geometry_calib`) giữ dạng batch
`[B, …]`, đọc trước `prepare_targets`.

### A3.5 Trainer / Tester

- Trainer: `model_kwargs['geometry_calibs'] = targets['geometry_calib']` (lấy trước
  `prepare_targets`); truyền thêm `horizon_line, horizon_valid` vào criterion (gắn
  vào mỗi dict target hoặc truyền riêng).
- Gọi `model.set_epoch(epoch)` đầu mỗi epoch (cho warm-up §A7).
- Tester: `geometry_calibs = info['geometry_calib']`.

## A4. Horizon branch (song song RSH)

```python
srcs_pre = srcs                                   # trước region_head
horizon  = self.gpro.predict_horizon(srcs_pre[:3], image_hw, images)
# logits [B,1,H/4,W/4]; raw_line [B,2]; line [B,2] (eval: + vertical-edge refine); confidence [B,W/4]
plane, plane_valid = horizon_line_to_ground_plane(horizon['line'], geometry_calibs, camera_height_m)
# plane [B,4] = (n, d=−h); plane_valid [B]
```

`HorizonHead(256, fpn_channels=128)`: gộp f/8, f/16, f/32 ở f/8 rồi upsample ×2 → heatmap f/4.
Cờ `gpro.horizon.visual_feature_detach` (mặc định `false`, giống DACN).

## A5. Near-contact bridge → Z_near (sau 2D decoder)

```python
box2d   = inter_coord[-1].detach()              # [B, Qt, 6] (u3d, v3d, l, r, t, b), Qt = pad + Qdet
feat    = srcs_pre[0] if cfg.roi_feature == 'pre_rsh' else enhanced_srcs[0]
contact = self.gpro.touchpoint_head(feat, box2d)
# relative [B,Qt,2], absolute [B,Qt,2] (chuẩn hoá 0–1), confidence [B,Qt], logits [B,Qt,9,9]
rays    = raycast_touchpoints_to_plane(contact['absolute'][:, :, None], geometry_calibs, plane,
                                       image_hw, min_abs_ray_plane_dot, 2.0, 65.0)
P_near     = rays['intersections'][:, :, 0]     # [B,Qt,3]
Z_near     = P_near[..., 2]                     # [B,Qt]
valid_near = rays['point_valid'][..., 0] & in_image(contact['absolute']) & plane_valid[:, None]
conf_near  = contact['confidence'] * horizon_conf.mean(-1)[:, None] * rays['point_stability'][..., 0]
```

ROI head chạy theo chunk 50 query (`query_chunk_size`) như DACN; với Qt ≈ 600 khi train
vẫn rẻ. Box **luôn detach** ở đây.

## A6. Near→center và depth `Z_center + Z_err` (trong vòng `for lvl` của 3D heads)

```python
coord_l  = outputs_coord                                       # box của 3D decoder layer lvl
alpha    = decode_alpha(outputs_angle)                         # bin 12 + residual, như DACN _decode_alpha
u2d      = 0.5 * (x_min + x_max)(coord_l.detach()) * W_input
ry       = alpha + atan2(u2d − geometry_calibs[:,0,2], geometry_calibs[:,0,0])
offset   = near_offset_camera(size3d, ry; k_l, k_w)             # [B,Qt,3]; dims/ry detach nếu detach_pose
P_center = P_near_used − offset                                 # P_near_used: xem §A7 (detach theo epoch)
Z_center = P_center[..., 2]
valid_c  = valid_near & isfinite(P_center).all(-1) & (2 ≤ Z_center ≤ 65)

err_g    = self.depth_embed_gpro[lvl](geometry_features)       # MLP(256,256,2,2) mới, clone theo layer
depth    = where(valid_c, Z_center + err_g[...,0], depth_geo + depth_geo_err[...,0])
logσ     = where(valid_c, err_g[...,1],           depth_log_scale)
depth_ave = stack([depth, logσ], -1)                            # thay depth_ave cũ
```

- `size3d` = (h, w, l) **tuyệt đối** khi `dataset.meanshape: False` (mọi config hiện có):
  `cls_mean_size` bị đặt về 0 (`kitti_dataset.py` dòng ~114) và decode_helper cộng 0.
  Nếu ai bật `meanshape: True` thì `near_offset_camera` phải nhận
  `size3d + cls_mean_size[Car]` → thêm assert `meanshape == False` khi GPro bật.
- Hai head err tách riêng vì phân bố sai số của hai anchor khác nhau.
- `cfg.gpro.depth_anchor`:
  - `gpro` (mặc định khi GPro bật): như trên.
  - `monodgp`: giữ nguyên depth MonoDGP; GPro chỉ là hint (E1-hint, §B3 ablation).
- Áp dụng cho **mọi layer** (aux loss), **mọi slot** (cả DN slot).

## A7. Gradient và warm-up

| Đường | Mặc định |
|---|---|
| box 2D → ROI | **detach** luôn |
| dims / ry → near_offset | **detach** (`gpro.detach_pose_in_center: true`) |
| Z_near → GeoEmbed (E1) / Z_geo (E2) | **detach** |
| Z_final → Z_center → Z_near → ROI head, horizon head | **mở** sau warm-up |
| epoch < `gpro.backprop_start_epoch` (mặc định 20) | `P_near_used = P_near.detach()` |

Z_near luôn học từ loss trực tiếp (§A8) kể cả trong warm-up.

## A8. Loss (thêm vào `SetCriterion`)

Dùng `indices` của output cuối (đã tính trong `forward`). Chỉ class Car
(`cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}` → `class_index: 1`, giống DACN).

| Key | Định nghĩa | Mask | Hệ số |
|---|---|---|---|
| `loss_horizon_heatmap` | focal trên heatmap f/4 (σ = 1.5 px) | `horizon_valid` | 1.0 |
| `loss_horizon_line` | smooth-L1 trên (slope·W/H, tâm/H) của `raw_line` | `horizon_valid` | 0.5 |
| `loss_near_touchpoint` | smooth-L1 (β 0.05) relative box offset | matched Car & `near_touchpoint_valid` | 1.0 |
| `loss_near_depth` | smooth-L1 (β 0.05) `log Z_near` vs `log near_depth` | như trên & `valid_near` & **not `from_mixup`** | 1.0 |
| `loss_center_depth` | smooth-L1 `log Z_center` (layer cuối) vs `log depth` | như trên & `valid_c` | 0.5 |
| (giữ) `loss_depth` | Laplacian trên depth mới | MonoDGP | 1 |

Không tính horizon/near cho aux layer (chỉ `loss_depth` của aux tự đổi theo công thức mới).

**DN slot** (trong `compute_reconstruction_losses`, chỉ output cuối):

- `dn_loss_near_touchpoint`: ROI của DN slot trỏ vào **box nhiễu**; target lấy
  `near_touchpoint` (tuyệt đối) của GT object `dn_meta['target_indices']`, đổi sang
  relative theo **box nhiễu** đầu vào của slot; bỏ slot có relative ngoài
  `offset_range`.
- `dn_loss_near_depth`: `log Z_near` vs `near_depth` của GT object.
- `dn_loss_depth` đã có tự dùng depth mới.

Lý do: B3 sinh box nhiễu nhất quán vật lý, còn điểm chạm thật trong ảnh không đổi →
ROI học tìm đúng điểm chạm khi box lệch. Đây là chỗ cộng hưởng giữa DN và GPro.

## A9. Logging (W&B)

`horizon_center_mae_px`, `horizon_slope_mae`, `near_touchpoint_mae_px`,
`near_depth_mae_m`, `gpro_center_rel_err` (median |ΔZ|/Z), `gpro_valid_ratio`
(trên matched Car), `anchor_gpro_ratio` (tỉ lệ query dùng anchor GPro),
`mean_abs_err_g`, `mean_abs_err_monodgp`, **R² và std của chiều cao dự đoán** trên
matched Car (so với B0: R² 0.42–0.44, std 0.082–0.086).

Tester: export thêm `Z_near, Z_center, valid_c, depth_geo` cho mỗi detection để
phân tích theo bin khoảng cách (dùng lại `tools/analysis/depth_bins.py`).

## A10. Gate 0 — chạy TRƯỚC khi train (không cần GPU)

Mục tiêu: chứng minh GPro không lặp lại kết quả âm 3.13 (cue mặt đất 9.7% vs
MonoDGP 2.4%).

**Giới hạn của phép thử:** GPro trong DACN chạy trên MonoDETR. Z_center phụ thuộc
(a) horizon — không phụ thuộc detector; (b) điểm near-contact — ROI cắt theo box 2D
của MonoDETR; (c) near→center — dims/yaw của MonoDETR (ảnh hưởng nhỏ). Vì vậy Gate 0
là **proxy**: FAIL là tín hiệu mạnh để không dùng anchor; PASS chỉ là điều kiện cần.
Run DACN cho final depth loss `(Zq+Zs+Zgpro)/3` kéo Z_gpro nên Z_gpro có thể lệch
có hệ thống → luôn báo cả lỗi có dấu.

**0a — box của detector (proxy thực tế):**

1. Lấy artifact val của run DACN `near_gpro_zmap_replace_aug` (checkpoint best,
   `--evaluate-only`): mỗi detection có box, `gpro_cue_depth_m` (= Z_center khi hợp lệ),
   `replacement_valid`.
2. Lấy prediction val của MonoDGP B0 seed 444 (hoặc B3-v4).
3. Ghép cả hai với GT Car qua IoU 2D ≥ 0.5 (giống `tools/analysis/depth_error_explained.py`).
   Chỉ dùng GT được **cả hai** bắt (so cặp trên cùng object).
4. Trên các GT đó: median |ΔZ|/Z và lỗi có dấu trung bình của GPro vs MonoDGP; tương
   quan lỗi có dấu; trọng số trộn tối ưu `a` (fit trên ảnh chẵn, đo trên ảnh lẻ, như
   `ground_depth.py`); hit@0.7 khi thay depth của MonoDGP bằng depth trộn (như
   `height_oracle.py`). Báo theo bin 0–10/10–20/20–30/30–40/40+ m, kèm số object mỗi bin
   và coverage `replacement_valid`.

**0b — box GT (tách lỗi GPro khỏi lỗi detector):**

Viết script nhỏ trong DACN: load checkpoint trên, chạy backbone + horizon head, đưa
**box 2D GT** (dạng `cxcylrtb` chuẩn hoá) vào `NearContactROIHead`, dùng dims/yaw GT cho
near→center. Đo median |ΔZ|/Z theo bin. Đây là sai số của riêng horizon + touchpoint.
Nếu 0b tốt mà 0a kém → nút thắt là box 2D, MonoDGP (box tốt hơn) có thể khá hơn; nếu
0b đã kém → GPro tự nó chưa đủ tốt.

**Ghi tiêu chí trước khi xem số** (pre-register trong WORKLOG):

- PASS (0a): `a ≥ 0.2` **hoặc** GPro tốt hơn MonoDGP ở ít nhất các bin < 20 m, và
  hit@0.7 với depth trộn > hit@0.7 MonoDGP.
- FAIL (0a) nhưng 0b PASS theo cùng tiêu chí: vẫn chạy E1 nhưng ưu tiên E1-hint trước.
- FAIL cả 0a và 0b: dừng anchor; chỉ chạy E1-hint (`depth_anchor: monodgp`).
- Nên dùng checkpoint DACN đã train xong (run aug đang ở ~epoch 80/195); số trước
  decay lr sẽ đánh giá thấp GPro.

## A11. Config (thêm vào `b3_v4.yaml` → `configs/gpro_e1_b3v4.yaml`)

```yaml
dataset:
  gpro_pseudo_label_root: 'data/near_gpro_pseudo_labels'   # tương đối so với mixup3d/monodgp (§A2)

gpro:
  enabled: True
  camera_height_m: 1.65
  class_index: 1                       # Car (cls2id)
  k_l: 0.65
  k_w: 0.85
  horizon:
    fpn_channels: 128
    heatmap_stride: 4
    gaussian_sigma_heatmap_px: 1.5
    softargmax_temperature: 0.25
    entropy_confidence: True
    visual_feature_detach: False
    vertical_edge_refinement: {enabled: True, gaussian_kernel_size: 13, gaussian_sigma: 4.0,
      canny_thresholds: [50, 100], hough_threshold: 5, min_line_length_px: 40.0,
      max_line_gap_px: 10.0, vertical_angle_range_deg: [70.0, 110.0],
      cluster_tolerance_deg: 3.0, min_edge_count: 5, max_angle_std_deg: 3.0}
  near_touchpoint:
    roi_feature: pre_rsh               # pre_rsh | post_rsh
    roi_size: 9
    roi_channels: 64
    query_chunk_size: 50
    softargmax_temperature: 0.25
    offset_range: [-0.20, 1.20]
  plane: {min_abs_ray_plane_dot: 0.015, valid_min_depth_m: 2.0, valid_max_depth_m: 65.0}
  depth_anchor: gpro                   # gpro | monodgp
  detach_pose_in_center: True
  backprop_start_epoch: 20
  query_hint:                          # Phần B (E1)
    enabled: True
    inject: content                    # content | content_and_pos
  losses:
    horizon_heatmap: 1.0
    horizon_line: 0.5
    near_touchpoint: 1.0
    near_depth: 1.0
    center_depth: 0.5
    dn_near_touchpoint: 1.0
    dn_near_depth: 1.0
```

`model_name`, `wandb.name`, tags: `gpro-e1`. Thêm `configs/gpro_e1_b0.yaml` (cùng
block, `use_dn: False`) để tách đóng góp với DN.

## A12. Test (thêm vào `tests/`)

1. Config cũ không có `gpro:` → output forward bit-identical với trước (seed cố định, CPU).
2. `depth_consistent_calib`: ray ∩ plane qua calib ảo cho `Z·s`, X giữ nguyên, có/không flip
   (port test từ DACN `tests/test_near_gpro_labels.py`).
3. Nhãn near-contact sau flip + crop: chiếu lại `camera_3d` bằng `geometry_calib` gốc
   (không nhân factor) ra đúng `near_touchpoint` trong 1 px.
4. Object MixUp: `object_keys // 100 == random_index` → lấy đúng record; `horizon_valid == False`.
5. Shape: `Z_near [B,Qt]`, depth `[B,Qt,2]` với DN bật/tắt, train (Qdet 550) / eval (50).
6. `valid_c == False` → depth đúng bằng công thức MonoDGP (fallback chính xác).
7. Gradient: epoch < `backprop_start_epoch` → grad của `touchpoint_head` từ `loss_depth` = 0;
   sau đó ≠ 0. Box 2D không nhận grad từ `loss_depth` qua ROI.

---

# PHẦN B — E1: Z_near làm hint cho query 3D

## B1. Thay đổi

Giữa 2D decoder và 3D decoder trong `MonoDGP.forward`:

```python
query_features_2d = hs_2d[-1]                                         # [B, Qt, 256]
if gpro.query_hint.enabled:
    g = self.gpro.geo_embed(Z_near.detach(), valid_near, conf_near.detach(),
                            contact['relative'].detach())             # [B, Qt, 256]
    query_features_2d = query_features_2d + g
hs, init_reference, inter_references = self.det3d_transformer(
    intermediate_output, query_features_2d, depth_pos_embed, dn_meta=dn_meta)
```

Depth Predictor, Depth CA, RSH: **giữ nguyên** MonoDGP.

## B2. `GeoEmbed`

```python
class GeoEmbed(nn.Module):
    def __init__(self, d=256, depth_max=60):
        self.depth_table = nn.Embedding(depth_max + 1, d)        # bảng riêng, 61 x 256
        self.aux = MLP(4, d, d, 2)                               # [conf, rel_u, rel_v, log Z]
        self.null_token = nn.Parameter(torch.zeros(d))
        self.out = nn.Linear(d, d); zero-init weight & bias      # bắt đầu = MonoDGP gốc
    def forward(self, z, valid, conf, rel):
        e = interpolate_1d(z.clamp(0, 60), self.depth_table)     # [B,Qt,256], nội suy 2 hàng kề
        e = e + self.aux(stack([conf, rel_u, rel_v, log(z.clamp_min(1))]))
        e = where(valid[..., None], e, self.null_token)
        return self.out(e)
```

Shape: `Z_near [B,Qt] → e [B,Qt,256] → g [B,Qt,256]`, cộng thẳng vào `hs_2d[-1] [B,Qt,256]`.

Zero-init `out` để lúc khởi tạo E1 trùng MonoDGP; mạng tự học dùng hint.

## B3. Ablation của E1 (1 seed để sàng lọc)

| ID | Thay đổi | Câu hỏi |
|---|---|---|
| E1 | full (hint + anchor), trên B3-v4 | thiết kế chính |
| E1-hint | `depth_anchor: monodgp` | chỉ hint có giúp không? (an toàn nếu Gate 0 FAIL) |
| E1-anchor | `query_hint.enabled: False` | chỉ anchor |
| E1-pos | `inject: content_and_pos` (g cũng làm `query_pos` cho self-attn và visual CA) | vị trí đưa hint |
| E1-b0 | E1 trên B0 (không DN) | tách tương tác với DN |
| E1-postRSH | `roi_feature: post_rsh` | ROI trước/sau RSH |
| E1-nobp | `backprop_start_epoch: 9999` | backprop về Z_near có giúp không |

## B4. Thứ tự làm và GPU

1. Gate 0 (CPU).
2. Implement Phần A + B, test CPU.
3. Smoke test 2 epoch trên GPU1 (hỏi trước), kiểm tra loss không NaN, `gpro_valid_ratio` > 0.
4. E1 seed 444 (≈ 14–15 h). So với B3-v4 s444 = 23.11 (best) / 22.47 (mean ep 200–250).
5. Tuỳ kết quả: E1-hint hoặc E1-anchor; sau đó seed 2027, 3407.

Hạn CVPR 2027: đăng ký 10/11, nộp 16/11/2026. GPU1 có queue của team (b0_dfl,
B2+flip seed, σz 0.20) — thống nhất thứ tự với team trước khi chen run.
