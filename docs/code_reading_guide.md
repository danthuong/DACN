# Lộ trình đọc code MonoDETR × E1 Geometry

Tài liệu này là bản đồ đọc code của repository hiện tại, không phải bản mô tả
MonoDETR upstream nói chung. Mục tiêu là giúp lần theo được một ảnh KITTI từ lúc
được đọc khỏi đĩa, đi qua MonoDETR, sinh touchpoint/horizon, fusion hai pass,
tính loss, lưu checkpoint, rồi xuất metric và dashboard.

## 1. Bức tranh tổng thể

Luồng train:

```text
YAML + .env
  -> tools/train_val.py
  -> KITTI_Dataset -> image, P2, targets, input_calib
  -> MonoDETR.forward
       -> backbone + projected multi-scale features
       -> DepthPredictor: original depth logits/memory
       -> visual encoder (một lần)
       -> decoder pass 1
       -> detection heads + horizon/touchpoint heads
       -> ground plane + ray/plane intersection -> geometric depth
       -> feature-level hoặc distribution-level fusion
       -> decoder pass 2
       -> final detection heads
  -> SetCriterion + HungarianMatcher
  -> backward -> optimizer -> scheduler
  -> KITTI validation + matched depth report + W&B + checkpoint
```

Luồng inference E1 luôn dùng **một ảnh** và hai decoder pass. Pass 1 không phải
frame trước và không dùng pseudo label/GT. Pseudo label chỉ là target trong lúc
train.

## 2. Các khái niệm phải giữ cố định khi đọc

- KITTI camera coordinates: `X` sang phải, `Y` xuống dưới, `Z` hướng về trước.
- `P2` là ma trận chiếu đầy đủ `3x4`; cột thứ tư không được bỏ qua khi tạo ray.
- `pred_boxes` không phải `cxcywh`. Nó là
  `[center3d_x, center3d_y, left, right, top, bottom]`, đã normalize.
- `targets['boxes']` mới là 2D `cxcywh`; `targets['boxes_3d']` là biểu diễn sáu
  giá trị nói trên.
- Query depth cuối của MonoDETR là trung bình của depth regression, depth từ
  kích thước/chiều cao bbox và depth lấy từ dense depth map.
- Dense depth dùng 80 LID foreground bins cộng một background bin.
- Train dùng 11 nhóm, mỗi nhóm 50 queries; inference chỉ dùng 50 queries đầu.
- Geometry spatial prior chỉ dùng canonical group đầu, còn Hungarian loss vẫn
  giữ cơ chế grouped matching của MonoDETR.
- `input_calib = A @ P2` thuộc hệ tọa độ ảnh sau affine resize. Raw `P2` vẫn
  được dùng cho decode KITTI gốc.

Tensor map mặc định với input `1280x384`:

| Tensor | Shape | Ý nghĩa |
| --- | --- | --- |
| `images` | `[B,3,384,1280]` | ảnh đã normalize |
| `srcs[0..3]` | `[B,256,48,160]`, `[B,256,24,80]`, `[B,256,12,40]`, `[B,256,6,20]` | feature pyramid |
| `original_depth_logits` | `[B,81,24,80]` | 80 LID bins + background |
| `original_weighted_depth` | `[B,24,80]` | kỳ vọng depth theo bin |
| `original_depth_memory` | `[B,256,24,80]` | latent depth + metric position |
| query table lúc train | `[550,512]` | 11 nhóm × 50, content + reference |
| query table lúc eval | `[50,512]` | canonical group |
| touchpoints | `[B,Q,4,2]` | `LF, RF, RR, LR`, normalized input image |
| geometric object depth | `[B,Q]` | depth từ ray/ground-plane |

## 3. Thứ tự đọc đề xuất

### Chặng 0 — Đọc hợp đồng của experiment

1. `docs/e1_repo_audit.md`
   - Bản kiểm kê code/checkpoint gốc, tensor shape và các điểm dễ hiểu sai.
   - Đọc trước để biết phần nào là MonoDETR gốc và phần nào được tách/refactor
     để hỗ trợ hai pass.

2. `docs/e1_implementation.md`
   - Hợp đồng implementation hiện tại: coordinate frame, hai phase, checkpoint,
     evaluation và dashboard.

3. `configs/monodetr.yaml`
   - Baseline: dataset, backbone, số query/layer/depth bin, loss coefficients,
     optimizer và tester.
   - Khi đọc code, luôn quay lại file này để biết một nhánh có thực sự được bật
     hay không.

4. `configs/e1_common.yaml`
   - Phần dùng chung cho hai mode: pseudo label, horizon/touchpoint, plane,
     alpha, loss, mốc phase 2, optimizer groups và W&B.

5. `configs/e1_feature_level.yaml` và `configs/e1_distribution_level.yaml`
   - Chỉ nên khác nhau ở mode fusion và output directory.
   - Distribution mode định nghĩa Gaussian sigma, clamp log-prior và yêu cầu
     dựng lại weighted depth/positional embedding.

6. `configs/e1_feature_level_touch_v2.yaml` và
   `configs/e1_distribution_level_touch_v2.yaml`
   - Các override cho touchpoint v2: box-relative loss, SmoothL1 beta nhỏ,
     lọc điểm ngoài ảnh và detach detector box.

Sau chặng này cần trả lời được: model đang chạy baseline/feature/distribution,
phase 2 bắt đầu epoch nào, checkpoint/output nằm ở đâu và loss nào đang bật.

### Chặng 1 — Entry point và config resolution

7. `lib/helpers/config_helper.py`
   - Đọc theo thứ tự `_deep_merge` -> `_load_with_extends` ->
     `_expand_environment` -> `load_config`.
   - Đây là lý do YAML con chỉ cần ghi override; các dictionary cha được
     deep-merge thay vì thay toàn bộ.

8. `tools/train_val.py`
   - Entry point duy nhất cho train/eval/export.
   - Đọc `main()` từ trên xuống và ghi lại thứ tự build dataloader, model,
     optimizer, scheduler, trainer và tester.
   - Nhánh `-e`, `--checkpoint-path` và `--export pass1=... --export pass2=...`
     là đường inference/evaluation.

9. `lib/helpers/dataloader_helper.py`
   - Tạo train/val `DataLoader`, shuffle policy và worker seed.

10. `lib/helpers/model_helper.py` và `lib/models/monodetr/__init__.py`
    - Hai cầu nối rất ngắn từ entry point vào `monodetr.py::build`.

Sau chặng này cần tự vẽ được call graph từ command line tới constructor model.

### Chặng 2 — KITTI, calibration và pseudo label

11. `lib/datasets/kitti/kitti_utils.py`
    - Đọc `Object3d`, `Calibration`, `get_affine_transform` và
      `affine_transform`.
    - Đây là nguồn định nghĩa KITTI label, projection, rect-camera coordinates
      và phép biến đổi ảnh.

12. `pseudo_label_generation/geometry.py`
    - Đọc theo thứ tự:
      `touch_points_local` -> `transform_touch_points` -> `project_points` ->
      `fit_ground_plane` -> `derive_horizon_line`.
    - `backproject_pixels_to_plane` là bản NumPy dùng cho kiểm tra/dashboard;
      chú ý camera center `-M^-1 p4`.
    - `_largest_y_consensus` thực hiện điều kiện bottom-height gap trước khi fit
      plane.

13. `generate_geometry_pseudo_labels.py`
    - `build_frame_record()` nối KITTI labels/calibration với các hàm geometry.
    - Theo dõi `object_id`: nó là số dòng gốc trong KITTI label file, không phải
      index sau filter.
    - `plane_classes` có thể rộng hơn `touch_classes`; object không phải vehicle
      vẫn có thể góp điểm fit ground.
    - Output là `outputs/geometry_pseudo_labels/<split>.jsonl`.

14. `lib/datasets/kitti/kitti_dataset.py`
    - Đây là file quan trọng nhất của data path. Đọc theo thứ tự:
      `__init__` -> `_load_pseudo_records` -> `__getitem__` -> `eval`.
    - Trong `__getitem__`, lần theo:
      1. photometric/flip/crop;
      2. affine transform ảnh;
      3. tạo `input_calib`;
      4. transform horizon và touchpoint sang input frame;
      5. tạo các target MonoDETR;
      6. trả `(inputs, raw_P2, targets, info)`.
    - So sánh `require_inside_image=false/true` để hiểu v1/v2.

15. `lib/datasets/kitti/pd.py`
    - Chỉ cần đọc sâu nếu muốn hiểu photometric augmentation. E1 mặc định tắt
      crop/flip nhưng vẫn giữ photometric distortion.

Sau chặng này cần giải thích được vì sao raycast phải dùng `input_calib`, còn
KITTI decode vẫn nhận calibration gốc.

### Chặng 3 — MonoDETR baseline

16. `lib/models/monodetr/monodetr.py::build`
    - Nên bắt đầu từ cuối file thay vì đầu file.
    - Hàm này build backbone, depth predictor, depth-aware transformer,
      `MonoDETR`, matcher, `SetCriterion` và `weight_dict`.

17. `lib/models/monodetr/backbone.py`
    - ResNet, frozen batch norm, multi-scale feature extraction và positional
      encoding joiner.

18. `lib/models/monodetr/position_encoding.py`
    - Positional encoding 2D cho visual features. Không nhầm với
      depth positional embedding trong `DepthPredictor`.

19. `lib/models/monodetr/depth_predictor/depth_predictor.py`
    - File lõi để hiểu MonoDETR khác DETR thường ở đâu.
    - Đọc `extract_depth_feature` -> `depth_classifier` ->
      `depth_from_logits` -> `encode_depth_feature`.
    - Nắm rõ ba đại lượng khác nhau:
      latent `src`, categorical `depth_logits`, và scalar `weighted_depth`.

20. `lib/models/monodetr/depth_predictor/transformer.py`
    - Transformer encoder một layer dành riêng cho latent depth feature.

21. `lib/models/monodetr/depthaware_transformer.py`
    - Đọc theo thứ tự:
      `encode_visual` -> `decode_from_memory` -> `forward` ->
      `DepthAwareDecoderLayer` -> `DepthAwareDecoder`.
    - Trong decoder layer, tách ba loại attention: query self-attention,
      visual deformable cross-attention và depth cross-attention.
    - `visual_cache` cho phép pass 2 dùng lại visual encoder output; backbone và
      visual encoder không chạy hai lần.

22. `lib/models/monodetr/ops/modules/ms_deform_attn.py` và
    `ops/functions/ms_deform_attn_func.py`
    - Python wrapper của custom multi-scale deformable attention CUDA op.
    - Chỉ đọc sau khi đã hiểu transformer call site. Không cần đọc thư mục
      `ops/build`, `ops/dist` hoặc `*.egg-info`; đó là generated artifacts.

23. `lib/models/monodetr/monodetr.py::MonoDETR`
    - Quay lại đầu file và đọc:
      `__init__` -> `_query_embeddings` -> `_decode_detection_heads` ->
      `forward`.
    - Để hiểu baseline, tạm coi `geometry_enabled=false` và dừng ở Pass 1.
    - Trong `_decode_detection_heads`, đọc kỹ công thức query depth ở các dòng
      tạo `depth_geo`, `depth_reg`, `depth_map`, rồi lấy trung bình.

24. `lib/models/monodetr/matcher.py`
    - Hungarian assignment và chi phí class/bbox/GIoU/3D center.
    - Chú ý grouped matching khi train.

25. `lib/models/monodetr/monodetr.py::SetCriterion`
    - Đọc các loss riêng trước, sau đó đọc `forward()` để biết matcher nào được
      dùng cho final pass, pass 1 và touchpoint.
    - Phân biệt query-depth loss với dense DDN depth-map loss.

26. `lib/models/monodetr/depth_predictor/ddn_loss/`
    - `ddn_loss.py`, `focalloss.py`, `balancer.py`: dense depth bin supervision
      bên trong GT 2D boxes.

27. `lib/losses/uncertainty_loss.py`, `dim_aware_loss.py`, `focal_loss.py`
    - Các loss helper cho depth uncertainty, dimension và classification.

28. `lib/models/monodetr/dn_components.py`
    - Denoising-query preparation/loss. Đọc sau cùng vì nó không phải trọng tâm
      của geometry fusion nhưng ảnh hưởng train graph khi `use_dn=true`.

Sau chặng này cần mô tả được baseline từ feature pyramid tới một KITTI 3D box,
và phân biệt dense depth map với per-query object depth.

### Chặng 4 — E1 geometry và hai kiểu fusion

29. `lib/models/monodetr/geometry_e1.py`
    - Đọc đúng thứ tự sau:
      1. `HorizonHead`;
      2. `decode_horizon_line` và `make_horizon_heatmap_target`;
      3. `TouchpointHead`;
      4. `horizon_line_to_ground_plane`;
      5. `raycast_touchpoints_to_plane`;
      6. `GeometryE1.predict_geometry`;
      7. `_spatial_projection`;
      8. `fuse_feature` và `fuse_distribution`;
      9. `horizon_focal_loss`.
    - Theo dõi riêng validity, reliability và accuracy. Một ray có thể valid về
      toán học nhưng geometric depth vẫn sai nhiều mét.

30. `lib/models/monodetr/monodetr.py::forward` lần thứ hai
    - Lần này đọc với `geometry_enabled=true`.
    - Phase 1 trả về sau khi bổ sung horizon/touchpoint vào Pass 1.
    - Phase 2 tạo geometry, fusion, rồi gọi shared decoder lần hai.

Distribution-level cụ thể:

```text
object_depth [B,Q]
  -> Gaussian trên 80 LID foreground bins
  -> spatial mixture theo soft object masks/reliability
  -> normalized log_prior, clamp [-12, 0]
  -> original foreground logits + alpha * coverage * log_prior
  -> giữ nguyên background logit
  -> tính lại weighted_depth và depth_position
  -> original_depth_base + fused_depth_position
  -> decoder pass 2
```

Feature-level cụ thể:

```text
spatial_prior + coverage + confidence
  -> convolutional feature_adapter
  -> original depth source + alpha * coverage * residual
  -> chạy lại depth classifier và depth encoder
  -> fused depth memory/position
  -> decoder pass 2
```

Điểm cần chú ý khi so sánh: cùng một giá trị alpha không có cùng thang tác động
ở hai mode. Distribution alpha nhân log-prior; feature alpha nhân một residual
không bị chuẩn hóa cố định.

31. `lib/models/monodetr/monodetr.py::loss_horizon`, `loss_touchpoints`,
    `loss_geometry_metrics`, `loss_fused_depth_map`
    - Touchpoint dùng Pass-1 Hungarian assignment.
    - Geometry MAE chỉ là metric detach, không phải loss trực tiếp khi
      `geometric_center_depth_loss_coef=0`.
    - Trong Phase 2, final detection/fused-depth losses vẫn backprop qua
      geometry path; direct horizon/touchpoint losses chống drift.

Sau chặng này cần chỉ ra được gradient từ Pass-2 detection quay về alpha,
touchpoint và horizon bằng đường nào.

### Chặng 5 — Optimizer, train loop và checkpoint

32. `lib/helpers/optimizer_helper.py`
    - Chia parameter thành base, horizon/touchpoint và geometry/fusion groups.
    - Kiểm tra LR từng group trước khi giải thích một curve W&B.

33. `lib/helpers/scheduler_helper.py`
    - Cosine/step schedule được cập nhật theo epoch.

34. `lib/helpers/trainer_helper.py`
    - Đọc `__init__` -> `train` -> `train_one_epoch`.
    - `set_epoch()` điều khiển phase; phase 2 bắt đầu từ
      `fusion_start_epoch`.
    - `train()` chạy validation mỗi epoch, lưu best AP và best depth riêng.
    - `train_one_epoch()` chuẩn bị target, forward, weighted loss, gradient
      accumulation, clipping, optimizer step và W&B logging.

35. `lib/helpers/save_helper.py`
    - Phân biệt `load_pretrained_checkpoint` với `load_checkpoint`:
      pretrained chỉ nạp compatible model weights; resume nạp cả optimizer,
      scheduler, scaler, epoch và RNG.

36. `lib/helpers/wandb_helper.py`
    - Lazy W&B initialization, run metadata, scalar/image/artifact logging.

Sau chặng này cần giải thích được vì sao “pretrained” không phải “resume”, và
checkpoint nào tối ưu AP khác checkpoint nào tối ưu depth MAE.

### Chặng 6 — Inference, KITTI decode và evaluation

37. `lib/helpers/tester_helper.py`
    - Đọc `_variant_outputs` -> `inference` -> `_save_e1_artifacts` ->
      `save_results` -> `evaluate`.
    - Một forward có thể đồng thời export Pass 1 và Pass 2.
    - KITTI `.txt` là kết quả detector; `artifacts/*.json` chứa horizon,
      touchpoint, geometric depth, reliability và alpha cho debug.

38. `lib/helpers/decode_helper.py`
    - Chọn top-k query/class và chuyển normalized network outputs thành KITTI
      dimensions, location, rotation và score.

39. `lib/datasets/kitti/kitti_eval_python/`
    - `eval.py` là official-style AP implementation; `kitti_common.py` đọc
      KITTI label/result files; `rotate_iou.py` hỗ trợ rotated overlap.
    - Không nên đọc sâu ở lượt đầu. Chỉ cần biết
      `get_official_eval_result()` trả report, metric dictionary và trong fork
      này có thể thêm giá trị moderate thứ ba.

40. `tools/evaluate_e1_depth.py`
    - Evaluation phụ cho câu hỏi nghiên cứu E1.
    - Đọc `read_kitti` -> `iou_matrix` -> `match_records` -> `metrics` ->
      `evaluate`.
    - Dùng class-constrained one-to-one 2D IoU matching; báo TP-only MAE và
      paired intersection để tránh so hai tập object khác nhau.
    - Negative `delta_model_minus_baseline_mae` mới là cải thiện.

41. `lib/helpers/e1_visualization.py`
    - Render các validation example đưa lên W&B.

Sau chặng này cần phân biệt AP chính thức, TP-only depth MAE và paired depth
MAE; không dùng một metric để thay thế metric kia.

### Chặng 7 — Dashboard

42. `dashboard/server.py`
    - `DashboardData` đọc ảnh, calibration, pseudo label, KITTI predictions và
      E1 artifacts.
    - Các hàm `ground_reprojections` và `calculate_reprojection_metrics` phục vụ
      debug ray/plane ngoài model.

43. `dashboard/model_registry.json`
    - Ánh xạ model ID tới output/artifact directory.

44. `dashboard/static/app.js`
    - Vẽ camera canvas, BEV depth-shift, toggle touchpoint/horizon/reprojection
      và model selector.

45. `dashboard/static/index.html`, `styles.css`
    - Chỉ là layout/style; đọc sau khi đã hiểu payload API.

46. `docs/dashboard.md`
    - Contract và cách chạy dashboard.

## 4. Call graph cần thuộc

### Một training batch Phase 2

```text
Trainer.train_one_epoch
  -> prepare_targets
  -> MonoDETR.forward
       -> backbone
       -> input_proj
       -> DepthPredictor.extract_depth_feature
       -> original depth classifier/encoder/position
       -> DepthAwareTransformer(pass 1, return visual_cache)
       -> _decode_detection_heads(pass 1)
       -> HorizonHead + TouchpointHead
       -> horizon_line_to_ground_plane
       -> raycast_touchpoints_to_plane
       -> _spatial_projection
       -> fuse_distribution hoặc fuse_feature
       -> decode_from_memory(pass 2)
       -> _decode_detection_heads(pass 2)
  -> SetCriterion.forward
       -> matcher(final pass)
       -> matcher(pass 1)
       -> MonoDETR losses
       -> E1 losses/metrics
  -> weighted sum
  -> backward
  -> gradient clipping
  -> optimizer.step
```

### Một validation epoch

```text
Trainer.train
  -> Tester.inference(pass1, pass2)
       -> model.forward một lần mỗi batch
       -> decode_detections
       -> KITTI txt + E1 artifact JSON
  -> Tester.evaluate(pass2) -> official AP
  -> evaluate_e1_depth -> baseline/pass1/pass2 depth report
  -> W&B metrics/artifacts/images
  -> checkpoint_last
  -> checkpoint_best nếu AP tăng
  -> checkpoint_best_depth nếu Pass-2 MAE giảm
```

## 5. File test nên đọc song song

- `tests/test_geometry.py`: projection, plane fitting, horizon và backprojection
  NumPy.
- `tests/test_geometry_e1.py`: differentiable PyTorch geometry, identity khi
  invalid/no coverage, background preservation và gradient fusion.
- `tests/test_e1_integration.py`: config inheritance, optimizer groups,
  pseudo-label loading và touchpoint v2.
- `tests/test_evaluate_e1_depth.py`: one-to-one matching, paired metrics và
  distance/difficulty report.
- `tests/test_dashboard.py`: reprojection/dashboard payload.

Đọc test ngay sau implementation tương ứng thường nhanh hơn đọc toàn bộ comment,
vì test cho biết invariant nào tác giả thực sự muốn giữ.

## 6. Lộ trình debug vấn đề hiện tại

Với hiện tượng touchpoint/geometric-depth xấu đi và alpha giảm ở Phase 2, đọc và
đặt breakpoint theo thứ tự:

1. `monodetr.py::loss_touchpoints`: số valid, coordinate mode, beta và Pass-1
   matching có đúng không.
2. `geometry_e1.py::TouchpointHead.forward`: relative point và detector box có
   bị drift không.
3. `geometry_e1.py::predict_geometry`: `point_valid`, `object_depth_std`,
   `object_reliability`; đừng chỉ nhìn `object_valid`.
4. `geometry_e1.py::fuse_distribution`: độ lớn `log_prior`, `coverage` và
   `alpha * log_prior` so với original logits.
5. `trainer_helper.py::_gradient_norms`: gradient direct head và fusion alpha.
6. So Pass 1/Pass 2 từ cùng checkpoint bằng `tester_helper.py`, không so hai
   epoch hoặc hai tập matched object khác nhau.

Alpha giảm thường là model học cách giảm ảnh hưởng của một prior có hại, không
phải alpha trực tiếp làm touchpoint xấu đi: touchpoint được tạo trước fusion.

## 7. Thứ tự đọc rút gọn nếu chỉ có một ngày

```text
docs/e1_repo_audit.md
configs/e1_common.yaml
configs/e1_distribution_level.yaml
tools/train_val.py
lib/datasets/kitti/kitti_dataset.py
lib/models/monodetr/depth_predictor/depth_predictor.py
lib/models/monodetr/depthaware_transformer.py
lib/models/monodetr/monodetr.py::build + MonoDETR.forward
lib/models/monodetr/geometry_e1.py
lib/models/monodetr/monodetr.py::SetCriterion
lib/helpers/trainer_helper.py
lib/helpers/tester_helper.py
tools/evaluate_e1_depth.py
```

Đừng bắt đầu bằng CUDA kernel, official KITTI evaluator hoặc frontend dashboard;
chúng làm mất mạch chính trước khi hiểu model/data contract.

## 8. Bài tập tự kiểm tra sau khi đọc

1. Với một query, chỉ ra ba thành phần tạo nên `pred_depth[...,0]`.
2. Giải thích tại sao sửa dense depth logits có thể đổi final 3D detection dù
   query depth có một regression head riêng.
3. Chỉ ra chính xác gradient từ Pass-2 `loss_depth` về Touchpoint MLP.
4. Giải thích vì sao `geometry_valid_matched_ratio` gần 1 không đảm bảo
   `geometry_depth_mae_m` thấp.
5. Chứng minh distribution mode giữ nguyên background logit.
6. Chỉ ra vì sao `alpha=0.01` không thể so trực tiếp giữa feature và
   distribution mode.
7. Giải thích vì sao muốn chứng minh fusion có ích phải so Pass 2 với Pass 1 từ
   cùng checkpoint, sau đó mới so với baseline pretrained.

Khi trả lời được bảy câu này mà không cần nhìn tài liệu, bạn đã nắm được phần
lõi của toàn bộ workflow.

## 9. Phụ lục: các file hỗ trợ sẽ gặp khi lần import

- `utils/box_ops.py`: chuyển đổi hai loại box, IoU và generalized IoU. Đặc biệt
  phải dùng `box_cxcylrtb_to_xyxy` cho `pred_boxes`, không dùng hàm cxcywh.
- `utils/misc.py`: `NestedTensor`, distributed reduction, metric logger,
  `inverse_sigmoid` và các tiện ích DETR dùng chung.
- `lib/datasets/utils.py`: angle-bin conversion và các hàm Gaussian/drawing
  còn lại từ data pipeline MonoDETR.
- `lib/helpers/utils_helper.py`: logger và global random seed.
- `requirements.txt`: dependency Python, gồm W&B và stack vision/scientific.
- `train.sh`, `test.sh`: command mẫu cũ; kiểm tra config/checkpoint path trước
  khi dùng, không xem chúng là nguồn sự thật thay cho `tools/train_val.py`.
- `.env.example`: tên biến môi trường cần cho dataset, pretrained checkpoint,
  pseudo-label root và W&B; không commit `.env` thật.
- `README.md`: hướng dẫn upstream/baseline và benchmark context, nhưng có thể
  chậm hơn implementation hiện tại ở `docs/e1_implementation.md`.
- `main_fig.png` và ảnh trong specification bundle: sơ đồ tham khảo, không tham
  gia runtime.
- `lib/models/monodetr/ops/src/`: C++/CUDA source của deformable attention.
  Chỉ cần đọc khi debug build/kernel/dtype; bình thường Python call graph là đủ.
- `lib/models/monodetr/ops/build/`, `ops/dist/`, `*.egg-info`: output build,
  không chỉnh tay và không dùng để hiểu thuật toán.
- `logs/`, `outputs/`, W&B run directories: runtime artifacts, không phải source
  code và không nên dùng thay config/checkpoint metadata.
