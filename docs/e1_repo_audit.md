# E1 repository audit

This audit records the local MonoDETR fork as inspected before the E1
architecture was modified. It is intentionally about the checked-out code and
checkpoint, not assumptions from upstream.

## Baseline and checkpoint

- Entry point: `tools/train_val.py`; model builder:
  `lib/models/monodetr/monodetr.py::build`.
- Working baseline checkpoint:
  `outputs/monodetr/checkpoint_best_1.pth`.
- SHA-256:
  `b618f6da895c1aabafb6a155ab22c312de374133044cc80dfe751280de441ea6`.
- Checkpoint epoch 159, best epoch 159, recorded Car moderate AP3D R40
  `20.834125363823905`.
- The baseline model and checkpoint each contain 582 state-dict entries. All
  names and shapes match exactly: zero missing, unexpected, or mismatched
  entries.
- Existing loader is strict and does not support adding new modules. E1 needs
  a shape-aware loader that reports every intentionally new key while requiring
  every baseline key to load.

## Runtime tensor map

Measured on real KITTI frame `000001`, resized by the local dataset convention
to 1280 x 384, using the baseline checkpoint on CPU:

| Tensor | Shape |
| --- | --- |
| input | `[1, 3, 384, 1280]` |
| ResNet layer2/layer3/layer4 | `[1,512,48,160]`, `[1,1024,24,80]`, `[1,2048,12,40]` |
| projected `srcs[0..3]` | `[1,256,48,160]`, `[1,256,24,80]`, `[1,256,12,40]`, `[1,256,6,20]` |
| depth logits | `[1,81,24,80]` |
| depth encoded memory | `[1,256,24,80]` |
| weighted depth | `[1,24,80]` |
| interpolated depth position | `[1,256,24,80]` |

The visual encoder concatenates 10,200 spatial tokens. The decoder has three
layers. The learned query table is `[550,512]`: training uses 11 groups of 50,
while evaluation slices the first 50. `group_num` was implicit through default
arguments in the local source rather than present in the baseline YAML.

## Box, depth, and loss semantics

- `pred_boxes` is `[center3d_x, center3d_y, left, right, top, bottom]`, all
  normalized. It is converted with `box_cxcylrtb_to_xyxy`; it is not cxcywh.
- Dataset `targets['boxes']` is normalized 2D cxcywh, while
  `targets['boxes_3d']` uses the six-value representation above.
- Query depth is the average of inverse-regression depth, geometric
  dimension/box-height depth, and depth-map sampling. Its second output is log
  uncertainty.
- The foreground depth-map objective is a separate 81-class DDNLoss: 80 LID
  bins plus background. The final bin value is 60 m.
- Original criterion includes classification, six-value box/GIoU, projected
  3D center, dimensions, angle, query depth, and one foreground depth-map loss.
  Intermediate decoder layers repeat every loss except depth-map loss.
- The Hungarian matcher independently matches each of the 11 query groups and
  concatenates assignments. E1 spatial projection must use only the first
  canonical 50-query group while keeping full grouped matching for losses.

## Depth predictor and decoder dependency

`DepthPredictor` fuses strides 8/16/32 into a stride-16 latent `src`, applies
`depth_head`, then:

1. `depth_classifier(src)` produces 81 logits and weighted depth;
2. `depth_encoder(src)` produces latent depth memory;
3. interpolated positional embedding from weighted depth is added to that
   memory;
4. the decoder depth cross-attention reads the resulting depth memory.

Therefore E1 must retain latent `src` and the pre-position depth-encoder output.
Distribution fusion must rebuild and add the interpolated position; feature
fusion must rerun both the shared classifier and depth encoder on fused `src`.
Changing only a logged depth tensor would not affect detections.

The current transformer runs visual encoding and decoding in one method. It
must be split into cacheable visual-encode and shared-weight decode operations
so pass 2 does not rerun the backbone or visual encoder.

## Dataset geometry and augmentation

- Raw KITTI images are affinely resized to `[1280,384]`; raw sizes vary.
- The baseline returns raw `P2`, while labels are transformed to network image
  coordinates. E1 therefore needs `P_input = A * P2` for horizon/touchpoint
  ray geometry, without changing raw `P2` used by baseline depth decoding.
- Baseline random flip is 0.5 and random crop is 0.5. With crop,
  `targets['depth']` is multiplied by `crop_scale` in `depth_scale=normal`.
  This is not physical camera-Z and cannot be mixed with metric raycasting.
- Initial E1 configs disable crop and flip, retain photometric distortion, and
  use physical center-Z. Geometry transforms are still carried explicitly for
  the deterministic raw-to-network resize.
- KITTI calibration files contain four calibration groups in the validation
  split. `P2` has a nonzero fourth column and ray construction must use the
  camera center `-M^-1 p4`, not silently assume an origin-only `K[I|0]`.

## Pseudo-label schema and alignment

Current records are JSONL at
`outputs/geometry_pseudo_labels/<split>.jsonl`, schema
`geometry-pseudo-label-v1`:

- horizon: `horizon_line.{valid,slope,intercept,visible_segment}`;
- plane: `ground_plane.{valid,coefficients,...}`;
- touchpoints: `objects[].object_id` aligned to the original KITTI label-file
  row, with four ordered slots `LF, RF, RR, LR` containing raw-image
  `pixel_2d`, `camera_3d`, and validity fields.

The network must consume `slope/intercept`, transform it and touchpoints into
the network-input frame, and preserve original annotation IDs through dataset
filtering. At audit time `val.jsonl` exists; `train.jsonl` is missing and must
be generated before real training.

## Trainer, evaluator, dashboard, and W&B

- Trainer has no AMP, accumulation, gradient clipping, phase state, or W&B.
  Resume saves only model/optimizer/epoch/best fields; scheduler/scaler/config
  and phase must be added without breaking old checkpoints.
- The evaluator writes KITTI text files and returns only Car moderate AP3D R40.
  Detailed matched depth reports require a separate common analysis path.
- Dashboard is the existing two-canvas server under `dashboard/`; its frame API
  already serves pseudo horizons, touchpoints, GT/predictions, and BEV data.
  E1 should extend its registry/artifact reading, not create another UI.
- `wandb` is not installed in the inspected `monodetr` environment. Integration
  must be lazy and support `disabled`/`offline`; online mode should fail with an
  actionable install/credential message and never print a token.

## Environment limitations during audit

No CUDA device is exposed to the current coding sandbox, so the custom CUDA
deformable-attention forward/backward and GPU smoke training cannot be run
here. Backbone and complete depth-predictor runtime shapes were nevertheless
measured on a real image on CPU. GPU acceptance commands are documented and
must be run when the device is available.
