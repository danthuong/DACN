# E2 — GPro trên MonoDGP: geometric depth memory thay Depth Predictor

Spec để implement trong workspace **`mixup3d/monodgp`** (bản clone repo MonoDGP của
Đại). Đường dẫn code là tương đối so với workspace đó; repo DACN chỉ là nguồn code
GPro và pseudo label.

**Điều kiện tiên quyết:** Phần A của `gpro_monodgp_e1_query_hint.md` (dataset, pseudo
label, horizon, ROI → Z_near, near→center, depth `Z_center + Z_err`, loss, DN,
MixUp3D, Gate 0, test) phải được implement trước. E2 **không** dùng Phần B của
E1 (GeoEmbed vào query); bản kết hợp E1+E2 là một ablation (§7).

---

## 0. Ý tưởng và khác biệt so với MonoDGP

MonoDGP (code thực tế, `depth_predictor.py`):

```python
src          = depth_head((downsample(f8) + proj(f16) + upsample(f32)) / 3)   # [B,256,H16,W16]
depth_logits = depth_classifier(src)                                          # [B,81,H16,W16] → DDN loss
depth_memory = depth_encoder(src, mask=masks[1], pos=seg_embed[1] + pos[1])   # [B,256,H16,W16]
# depth_embed = depth_embed  #+ depth_pos_embed_ip   ← position embedding của depth đã bị TẮT
```

→ Memory cho Depth Cross-Attention của 3D decoder **không chứa con số depth nào**,
chỉ là latent feature được DDN loss giám sát gián tiếp.

E2 thay hoàn toàn khối đó (biến thể "B2c-vis" trong `DACN/docs/plane.md`):

```
Z_geo  = dựng từ ground plane (mọi pixel dưới horizon) + Z_near của object (trong box)
memory = DepthEncoder( Embed(Z_geo) + DepthFeatureProj(F/16 sau RSH) )
```

Depth CA giờ attend vào memory **có depth mét thật**, lấy từ hình học, không phụ
thuộc kích thước xe (WORKLOG 3.9).

| | Block |
|---|---|
| **Bỏ** | `DepthPredictor` (depth_head, depth_classifier, weighted depth), `loss_depth_map` (DDN) |
| **Thêm** | `GeometricDepthMap` (không tham số, trừ null token) |
| **Thêm** | `depth_table = nn.Embedding(61, 256)` (Depth Position Embedding, dense) |
| **Thêm** | `DepthFeatureProj = Conv1×1(256→256) + GroupNorm(32)` trên `enhanced_srcs[1]` |
| **Thêm** | `DepthEncoder` = `TransformerEncoder(TransformerEncoderLayer(256, 8, ffn 256, drop 0.1), 1)` (lớp có sẵn trong `depth_predictor/transformer.py`) |
| **Sửa** | Depth memory tính **sau 2D decoder** (cần box + Z_near), trước 3D decoder |
| **Sửa** | Depth CA của 3D decoder dùng `key_padding_mask` cho vị trí không có depth |
| **Giữ** | Toàn bộ Phần A (Z_final = Z_center + Z_err), RSH, 2D/3D decoder, DN |

---

## 1. Thứ tự forward mới (`MonoDGP.forward` khi `gpro.geo_depth_memory.enabled`)

```
backbone → srcs → (srcs_pre) → horizon, plane                      # §A4
RSH → enhanced_srcs, seg_embed
2D decoder → hs_2d, inter_coord, inter_class
ROI (box 2D detach) → Z_near, valid_near                            # §A5
GeometricDepthMap(plane, box 2D, score, Z_near) → Z_geo, valid_geo  # §2
DepthMemory(Z_geo, valid_geo, enhanced_srcs[1]) → depth_memory      # §3
3D decoder(hs_2d[-1], depth_memory, mask) → hs                      # §4
3D heads → near→center → Z_center + Z_err                           # §A6
```

Không gọi `self.depth_predictor` nữa (không khởi tạo module khi
`replace_depth_predictor: True`, để optimizer không có tham số chết).

## 2. `GeometricDepthMap` — shape từng bước

Ví dụ input 384×1280 → `H16 × W16 = 24 × 80`, `N = 1920`. Lấy `(H16, W16)` từ
`masks[1].shape`, không hard-code.

**Input**

| Tensor | Shape | Nguồn |
|---|---|---|
| `plane`, `plane_valid` | `[B,4]`, `[B]` | §A4 |
| `horizon_line` (k, b) | `[B,2]` | §A4 (eval: đã refine) |
| `geometry_calibs` | `[B,3,4]` | dataset §A3.2 |
| `box2d` (xyxy, 0–1) | `[B,Qs,4]` | `inter_coord[-1]`, **detach**, chỉ slot splat |
| `score` | `[B,Qs]` | `sigmoid(inter_class[-1]).max(-1)`, detach |
| `Z_near`, `valid_near` | `[B,Qs]` | §A5, detach |

`Qs` = slot dùng để splat = **chỉ detector group 0**: train `[pad : pad+50]`, eval
`[0:50]`. **Không bao giờ dùng DN slot** (box DN sinh từ GT → rò GT vào memory
dùng chung cho mọi query).

**Bước 1 — lưới tâm pixel stride 16**

```python
u = (arange(W16) + 0.5) * (W_in / W16);  v = (arange(H16) + 0.5) * (H_in / H16)
uv1 = stack(meshgrid(v, u)) → [H16, W16, 3] → flatten [N, 3]
```

**Bước 2–3 — ground depth**

Dùng lại `raycast_touchpoints_to_plane` với "touchpoint" là toàn lưới
(`[B, 1, N, 2]` toạ độ chuẩn hoá) hoặc viết bản vector hoá tương đương:

```python
X        = ray ∩ plane                                   # [B, N, 3]
Z_gnd    = X[..., 2].reshape(B, H16, W16)                # [B,24,80]
v_hor    = k * u + b                                     # [B, W16]
ground   = point_valid & (v > v_hor[:, None, :]) & (Z_gnd ≥ 2)        # [B,24,80]
Z_gnd    = Z_gnd.clamp(max=60)
```

Pixel dưới horizon nhưng thuộc vật thể (thân xe, cột, tường) cho `Z_gnd` **quá xa**
(tia đi xuyên vật, cắt đất phía sau). Bước 4 sửa cho object được detect; phần
còn lại để `DepthFeatureProj` (ngoại hình) giúp Depth Encoder nhận ra.

**Bước 4 — object splat**

```python
sel    = valid_near & (score > τ)                                   # τ = 0.2
in_box = (u ∈ [x1,x2]) & (v ∈ [y1,y2])                              # [B,Qs,24,80], so với tâm pixel
in_box &= sel[..., None, None]
Z_obj  = where(in_box, Z_near[..., None, None], +inf).amin(1)       # [B,24,80]; chồng nhau → gần hơn che
in_obj = in_box.any(1)                                              # [B,24,80]
```

Dùng Z_near (mặt trước xe) chứ không Z_center: bề mặt nhìn thấy là mặt trước.

**Bước 5 — ghép**

```python
Z_geo     = where(in_obj, Z_obj, Z_gnd)                             # [B,24,80]
valid_geo = (in_obj | ground) & plane_valid[:, None, None]
Z_geo     = where(valid_geo, Z_geo, 0).clamp(0, 60)
```

Thứ tự ưu tiên mỗi pixel: trong box → Z_obj; dưới horizon → Z_gnd; còn lại → không có depth.

## 3. Depth memory

```python
E   = interpolate_1d(Z_geo, depth_table).permute(0,3,1,2)           # [B,256,24,80]
E   = where(valid_geo[:, None], E, null_depth_token[None,:,None,None])
V   = DepthFeatureProj(enhanced_srcs[1])                            # [B,256,24,80]
x   = (E + V).flatten(2).permute(2,0,1)                             # [1920,B,256]
pos = (seg_embed[1] + pos[1]).flatten(2).permute(2,0,1)             # như MonoDGP
kpm = masks[1].flatten(1) | ~valid_geo.flatten(1)                   # [B,1920]
kpm[~plane_valid] = masks[1].flatten(1)[~plane_valid]               # tránh mask toàn bộ → softmax NaN
memory = DepthEncoder(x, kpm, pos)                                  # [1920,B,256]
depth_pos_embed = memory.permute(1,2,0).reshape(B,256,24,80)        # đúng format Det3DTransformer nhận
```

`null_depth_token`: `nn.Parameter(zeros(256))`.

## 4. 3D decoder

`Det3DTransformer.forward` hiện đặt `mask_depth = None`. Thêm tham số
`depth_key_padding_mask=None` (mặc định None → hành vi cũ) và truyền `kpm` vào
`cross_attn_depth(..., key_padding_mask=mask_depth)`.

Query: `hs_2d[-1]` như MonoDGP (không GeoEmbed, trừ ablation E1+E2).

## 5. Warm-up (không còn Depth Predictor để dự phòng)

| Epoch | Ground part | Object splat | Z_near → fusion |
|---|---|---|---|
| `< E_w` (mặc định 20) | dựng từ **horizon GT** với xác suất `p = 1 − e/E_w` mỗi ảnh (ảnh `horizon_valid=False` → luôn dùng dự đoán) | **tắt** | detach (§A7) |
| `≥ E_w` | horizon dự đoán | bật | mở |

Không teacher-force object splat bằng box GT: sẽ lộ vị trí object thật vào memory
mà class/box của 3D decoder đọc → exposure bias khi bỏ.

Eval luôn dùng giá trị dự đoán.

## 6. Loss và config

- Bỏ `'depth_map'` khỏi `losses` và `weight_dict` khi `replace_depth_predictor`.
- Giữ toàn bộ loss của Phần A.
- Không có loss trực tiếp cho `Z_geo`; ground map chính xác bằng horizon loss, object
  part chính xác bằng `loss_near_depth`.

```yaml
# configs/gpro_e2_b3v4.yaml = b3_v4.yaml + block gpro của E1 với:
gpro:
  query_hint:
    enabled: False
  geo_depth_memory:
    enabled: True
    replace_depth_predictor: True     # False → giữ DepthPredictor (ablation keepDP)
    visual_feature: True              # False → E2-pure
    object_splat: True                # False → E2-ground
    mask_invalid: True                # False → E2-nomask (gán Z = 60, không mask)
    splat_score_threshold: 0.2
    detach_map: True
    warmup_epochs: 20
```

## 7. Ablation (1 seed sàng lọc trước)

| ID | Thay đổi | Câu hỏi |
|---|---|---|
| E2 | full, trên B3-v4 | thiết kế chính |
| E2-pure | `visual_feature: False` | memory có cần ngoại hình không |
| E2-ground | `object_splat: False` | splat Z_near có giúp không |
| E2-nomask | `mask_invalid: False` | mask vs giá trị cứng 60 m |
| E2-keepDP | `replace_depth_predictor: False`: giữ DepthPredictor + DDN; memory = `depth_encoder(src) + Embed(Z_geo)` ở pixel `valid_geo` | có cần bỏ hẳn Depth Predictor |
| E1+E2 | bật cả `query_hint` | kết hợp |
| E2-unlock | `detach_map: False` sau `E_w`, grad vào horizon nhân 0.1 | map có nên học từ loss 3D |

Theo dõi riêng cho E2: `horizon_center_mae_px` (map phụ thuộc trực tiếp vào horizon),
tỉ lệ `valid_geo`, tỉ lệ pixel `in_obj`; lưu 4 ảnh `Z_geo` mẫu mỗi lần validate
(W&B image) để kiểm tra bằng mắt.

## 8. Test thêm (ngoài test của Phần A)

1. Plane phẳng, camera cao 1.65, horizon ngang tại `v = c_y`: `Z_gnd(v) = f·h / (v − c_y)`
   khớp công thức trong 1e-3; hàng trên horizon có `ground == False`.
2. Hai box chồng nhau → pixel chung lấy Z nhỏ hơn.
3. Slot DN không bao giờ ảnh hưởng `Z_geo` (thay box DN bằng giá trị bất kỳ → `Z_geo` không đổi).
4. `plane_valid == False` → không có hàng nào của `kpm` bị mask toàn bộ; output không NaN.
5. `geo_depth_memory.enabled: False` → bit-identical với E1/MonoDGP.
6. Shape `depth_pos_embed` `[B,256,H16,W16]` khớp input cũ của `Det3DTransformer`.

## 9. Thứ tự làm

1. Phần A + test (dùng chung với E1).
2. §2–§4 + test §8.
3. Smoke test 2 epoch (hỏi trước GPU1). Kiểm tra ảnh `Z_geo` hợp lý.
4. E2 seed 444; so với B3-v4 s444 (23.11 best / 22.47 mean ep 200–250) và E1.
