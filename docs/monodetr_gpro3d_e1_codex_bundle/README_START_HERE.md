# Bộ đặc tả E1 — MonoDETR × GPro3D

Đây là **bộ file giao cho Codex để triển khai trên repository thực tế**, chưa phải code đã được tích hợp hoặc mô hình đã train.

## Các file

- `CODEX_IMPLEMENTATION_PROMPT.md`: prompt tiếng Anh chi tiết về toàn bộ nhiệm vụ, hình học, kiến trúc, gradient flow, tests, W&B và dashboard.
- `e1_feature_level.yaml`: cấu hình feature-level geometry fusion.
- `e1_distribution_level.yaml`: cấu hình depth-bin/distribution-level fusion.
- `.env.example`: các đường dẫn, camera-height assumption và W&B/dashboard credentials. Tạo `.env` riêng trên máy của bạn, không commit.

## Dùng thế nào với Codex

1. Copy các file vào root repository MonoDETR (hai YAML vào `configs/`, prompt vào `docs/` hoặc root).
2. Trong Codex, gửi: `Read CODEX_IMPLEMENTATION_PROMPT.md and implement it end-to-end in THIS repository. Use configs/e1_feature_level.yaml and configs/e1_distribution_level.yaml, adapting new config keys to actual local code. First audit and show actual shapes/augmentation/checkpoint mapping, then implement/tests, training, evaluation, dashboard integration, and W&B. Tell me precisely if a required dataset path, pseudo label file or W&B credential is missing.`
3. Cập nhật `.env` với KITTI, pretrained checkpoint, pseudo-label root và optional W&B account info. Phải kiểm tra schema pseudo labels hiện tại, không tự sinh giả.
4. Yêu cầu Codex chạy smoke tests và Phase 1, kiểm tra `predicted geo depth MAE` trước khi bật Phase 2. Sau đó thực nghiệm 2 YAML với cùng baseline & KITTI val.

## Những gì Codex phải kiểm tra và khớp trước khi chạy

- Official source dùng ảnh mạng 1280×384, horizon map thiết kế 320×96, depth map 80×24, 256 channels, D=80+background và 50 queries inference. **Nhưng phải tin số thực tế trong local fork, không hardcode từ các giá trị này.**
- Official training may run 11 query groups, effectively 550 training queries; use one canonical group for object-to-spatial geometry map and keep full original matching/loss semantics.
- `pred_boxes` trong implementation gốc là `[3D-projected cx,cy,left,right,top,bottom]` (6 giá trị), không phải `(cx,cy,w,h)`.
- Geometric depth là camera-axis Z của object center; wheel-contact Z cần aggregation/correction. Camera height 1.65 m chỉ là assumption, không phải dữ liệu GT chính xác của mỗi ảnh KITTI.
- KITTI random crop/flip mặc định có thể làm sai hệ hình học. Hai config đề xuất tắt augmentation hình học và giữ photometric augment cho E1 khởi đầu; không bật crop/flip đến khi transform `K`, horizon, touchpoint và GT nhất quán.
- Mã chính thức `DepthPredictor` trả depth logits, depth_embed, weighted_depth và depth_pos_embed_ip, có cả depth encoder bên trong. Đặt fusion đúng vào tensor và rebuild embedding liên quan, không chỉ sửa biến không được decoder dùng.
- Hai phase liên tục: Phase 1 một decoder pass + direct supervision, fusion OFF; Phase 2 hai decoder passes cùng weights, fusion ON, detection loss có gradient tới touchpoint/horizon nếu đường geometry differentiable.
- W&B token **không** đưa vào prompt hoặc YAML công khai; dùng `.env` private.

## Biến `.env` cần bạn cung cấp

| Biến | Cần thiết? | Ý nghĩa |
|---|---|---|
| `KITTI_ROOT` | Có | thư mục dataset với `training/image_2`, `training/label_2`, `training/calib`, `ImageSets` |
| `MONODETR_CHECKPOINT` | Có | checkpoint baseline inference hoạt động |
| `PSEUDOLABEL_ROOT` | Có | dữ liệu `(k,b)` horizon và các per-object touchpoint pseudo-labels đã tạo |
| `CAMERA_HEIGHT_M` | Có hoặc config khác | giả định nguồn camera-height/plane-offset; cần kiểm chứng; mặc định thử 1.65 m |
| `WANDB_PROJECT` | Tùy chọn | tên W&B project |
| `WANDB_ENTITY` | Tùy chọn | W&B user/team |
| `WANDB_MODE` | Có giá trị mặc định | `offline`, `online` hoặc `disabled` |
| `WANDB_API_KEY` | Chỉ khi online | bí mật, lưu private `.env` |
| `DASHBOARD_API_URL` | Nếu tích hợp trực tiếp | backend đã tồn tại của dashboard 2-panel |
| `DASHBOARD_AUTH_TOKEN` | Nếu dashboard có auth | token riêng, không commit |

## Outputs kỳ vọng

Sau mỗi validation và sau khi kết thúc, xuất báo cáo metric baseline/ours-pass1/ours-pass2; MAE RMSE median/P90/AbsRel; 0–10,10–20,20–30,30–40,40–60,>=60 m buckets; paired-object comparison; KITTI AP3D R40 Car Easy/Moderate/Hard; latency/VRAM; file CSV/JSON mỗi object và summary report. Dashboard vẫn hai panels (ảnh + BEV). W&B có runs riêng `feature_level` và `distribution_level`.

## Chú ý thực tế

Hai YAML là **target configurations**. MonoDETR gốc chưa hiểu phần `geometry_e1`, `runtime`, `logging`, `validation`; Codex phải implement config parser/feature wiring hoặc adapt sang cách load config hiện có. Chúng không thể dùng trực tiếp với unmodified MonoDETR rồi kỳ vọng fusion hoạt động.
