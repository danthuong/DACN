# CODEX IMPLEMENTATION TASK — MonoDETR × GPro3D, E1, two-phase end-to-end, two fusion variants

You are a senior PyTorch/computer-vision engineer working **inside my existing project/repository on the remote GPU machine**. Implement and RUN a real, testable experiment (not a conceptual report). Keep my working baseline and my existing 2-panel localhost visualization dashboard intact. Communicate progress/results in Vietnamese.

## 0. Goal, constraints, and deliverables

**Scientific question E1:** Can query-predicted ground-contact (wheel touchpoint) geometry and a learned horizon/ground plane improve MonoDETR's depth estimation and KITTI 3D detections? Evaluate TWO separate, reproducible variants using identical pretrained weights, data, training schedule and evaluation:

- `feature_level`: object geometric depth -> differentiable spatial geometry representation -> adapter into `f_D` **before** depth encoding.
- `distribution_level`: object geometric depth -> differentiable categorical prior in exactly MonoDETR's LID bins -> gated logit/probability fusion with MonoDETR's original foreground depth logits -> recomputed weighted depth / depth positional embeddings feeding decoder. The fused distribution **must actually affect decoder features**, not merely be logged.

Training is in **two phases of one continuously trainable model**: Phase 1 fusion OFF; Phase 2 two decoder passes (Pass 1 baseline depth -> Q3 touchpoints; Pass 2 fused depth -> final detections), trainable end-to-end through pass 2, fusion, differentiable raycasting, and pass-1 Touchpoint/Horizon heads. At inference run two decoder passes on the SAME image (not adjacent video frames and not cached previous epochs); reuse unchanged backbone/visual-encoder results.

**Files to create/use:** `configs/e1_feature_level.yaml`, `configs/e1_distribution_level.yaml`, `.env.example`, implementation source/tests, train/evaluate entry points integrated with project CLI, concise README `docs/e1_implementation.md`, W&B integrations, dashboard backend integration. The two YAML templates and the comprehensive implementation spec accompany this prompt. **First audit my actual repo and adapt these files to its real configuration and imports**. Do not implement a dummy parallel model, hard-coded fake tensors or placeholder metrics. Do not overwrite baseline training runs/checkpoints.

Deliverables: real train scripts (including resume), checkpoints `last`/`best`, numeric validation outputs, baseline-vs-E1 matched-depth-error report, per-image/per-object artifacts for dashboard, W&B run links if configured, tests and commands for reproducing both modes. If dataset/checkpoint/pseudo labels/credentials missing, diagnose exact missing items and describe what user must put in `.env` or config; do everything else possible first.

## 1. Mandatory repository audit BEFORE modifying architecture

Inspect exact local source, not assumptions from examples. Map modules, train/eval datasets, loss criterion, matcher, augmentation, checkpoint keys and local dashboard API contracts; produce `docs/e1_repo_audit.md` and briefly show this mapping:

1. Existing MonoDETR checkpoint format, model initialization and loaded/missing/unexpected parameters. Preserve old model weights; initialize only new modules. `strict=False` does not ignore matching-name/mismatched-shape weights; handle correctly and never silently fail to load backbone/decoder/depth predictor.
2. Inspect `MonoDETR.forward`, `DepthPredictor.forward`, visual encoder, depth encoder, depth positional encodings, `DepthAwareTransformer`/decoder, depth cross-attention, `pred_depth_map_logits`, `weighted_depth`, object `pred_depth`, `pred_boxes`, Hungarian matcher, `DDNLoss`. Preserve all existing original losses.
3. Typical official implementation reference has `srcs[0..2]` at strides 8/16/32, input `1280x384`, depth `24x80` (stride 16), hidden dimension 256, 80 foreground bins + 1 background bin (81 channels), 3 decoder layers, inference 50 queries; **verify LOCAL repo**. Note query-group training: some MonoDETR implementations use `group_num=11`, meaning ~550 train queries but 50 inference queries; honor actual grouping/matching; don't treat all 550 as unique real objects during spatial geometry projection.
4. Very important: official `pred_boxes` has 6 values `[projected_center_x, projected_center_y, left, right, top, bottom]`, NOT a conventional `cxcywh` box; use exact local `box_cxcylrtb_to_xyxy` or equivalent, and convert to true 2D xyxy when placing touchpoints/masks. GT `targets['boxes']` may instead be cxcywh. Respect normalization and frame transforms.
5. Official baseline has **two distinct depth objectives**: foreground depth map DDN/focal supervision and query-level object-depth regression with uncertainty. Keep both, do not inadvertently count either twice. `weighted_depth` and depth positional encodings can be computed from the foreground logits; do not confuse them with depth encoder's latent feature tensor.
6. Inspect KITTI loader's affine resize/crop/flip, calibration `P2`, per-object `depth`, and files generating horizon `(k,b)` and wheel touchpoint pseudo labels; inspect schema and alignment by image/object ID. Some variants multiply target depth by crop scale: **do not raycast in meters and compare to a crop-scaled GT**. For first geometry experiment use geometry-consistent preprocessing (disable geometric flip/crop by default or implement proven transforms of all images, touchpoints, horizon line, projection matrix, and GT together). Keep photometric augmentation allowed. Check camera height is supplied/calibrated, not silently available from KITTI P2.
7. Inspect current dashboard two views: image overlay and BEV with distance rings, GT/pred 3D boxes, horizon/touchpoints; extend its existing model selector/API rather than inventing another dashboard. Preserve existing routes and baseline display.
8. Show exact runtime tensor shapes for one real mini-batch with assertions before implementing fusion.

Relevant upstream behavior to check rather than blindly copy:
- `https://github.com/ZrrSkywalker/MonoDETR`, in particular `lib/models/monodetr/monodetr.py`, `lib/models/monodetr/depth_predictor/depth_predictor.py`, `lib/models/monodetr/depthaware_transformer.py`, and `configs/monodetr.yaml`.
- Never replace my local fork's code just because upstream differs; make smallest compatible modification.

## 2. Inputs, outputs and coordinate conventions

At training/inference, inputs are: normalized RGB `[B,3,H,W]`; per-image camera calibration (full P2 `[B,3,4]` where available and the corresponding effective K); image metadata; **camera height / plane offset source** (default research assumption 1.65 m configurable, never pretend it is exact GT); GT 3D detection labels on train/validation only; training-only pseudo labels for horizon (`k,b` + validity) and per-object ground-contact points plus validity masks.

Output tensors / dictionaries (exact keys may adapt to repo):
```
pass1.pred_boxes            [B,Q,6]   (repo-specific meaning!)
pass1.pred_logits           [B,Q,C]
pass1.pred_depth            [B,Q,2]   (scalar depth and uncertainty in original repo)
pass1.touchpoints_abs       [B,Q,P,2] (pixel in network-input coordinate frame)
pass1.touchpoints_rel       [B,Q,P,2] (normalized box-relative)
pass1.touchpoint_validity   [B,Q,P]   (optional predicted confidence; v1 can use mask/confidence from detection)
horizon.logits              [B,1,Hh,Wh]
horizon.line_net            [B,2]     (k,b in network-input image coords)
geometry.plane              [B,4]     normalized n and offset d
geometry.object_depth       [B,Q]     camera-axis Z, meters, where valid
geometry.object_valid       [B,Q]
geometry.spatial_prior      [B,*,Hd,Wd]
original_depth_map_logits  [B,D+1,Hd,Wd]
fused_depth_map_logits     [B,D+1,Hd,Wd] (both modes, via original/shared or fused auxiliary classifier)
pass2.pred_depth            [B,Q,2]
pass2.final_detections      (baseline-compatible structure)
```
Here P is the number of *meaningfully labelled* touchpoints per object, `P_max=4` with masks; don't fabricate invisible wheels if existing GPro3D pseudo labels only provide 1/2 points or uncertain assignment. Keep semantic slot ordering consistent within each object; if unreliable, implement permutation-invariant matching within object or adapt to true pseudo-label schema, and report it. Depth errors must use the correct GT 3D center camera-Z definition, not ground-contact-Z unless compensated.

**Input size:** use local KITTI transform (upstream default 1280x384), do not force raw KITTI images to a fixed size without carrying affine transforms to pseudo labels and calibration. Depth output stride ~16 => `[B,256,24,80]` features and `[B,81,24,80]` logits; horizon heatmap target at stride 4 => `[B,1,96,320]` for 1280x384. These are checked examples, not unconditional tensor hardcoding.

## 3. Horizon head, ground plane geometry, touchpoint head

### 3.1 Horizon Head (shared across both modes)
- Tap shared backbone/projected multi-scale tensors (actual local stride 8,16,32). Use lightweight FPN: lateral conv 1x1 to 128 channels each, bilinear resize lower-resolution features to stride-8, sum/concat + conv 3x3, bilinear upsample to stride-4, conv 3x3 + conv 1x1 to **one horizon logit channel**. Do not assume backbone exposes stride-4 C2 unless verified; use stride-8 upsample if not.
- Predicted heatmap `[B,1,H/4,W/4]`, GT Gaussian ridge target generated from pseudo label `v = k u + b` in correct augmented input frame (sigma ~1.5 heatmap px, configurable). Supervise BCE-with-logits/focal-like heatmap loss plus line SmoothL1 in center-height/slope normalized units. Respect invalid/out-of-image columns.
- Recover differentiable line: for each column softmax over vertical logits with temperature (`tau=0.25` initial), expected y via vertical soft-argmax, then differentiable, numerically stable weighted least-squares fit of `v'=k' u' + b'` (WLS confidence from entropy if configured). Use FP32 line solve under AMP. Convert line to network-input coords with explicit pixel-center convention. For `align_corners=False`, with scales `sx=W/Wh`, `sy=H/Hh`: `k=(sy/sx)k'`, `b=sy*b' + (sy-1)/2 - k*(sx-1)/2`. If target line/augmentation convention differs, **use proven exact transform and tests**, not these formulas blindly. Prefer use `(k,v_center)` for numerically stable loss.
- Optional OpenCV vertical-edge refinement for visuals/eval ONLY, default OFF for differentiable training/inference evaluation; no Hough/Canny inside gradient path. Expose debug raw and refined horizons separately. Avoid training/inference mismatch.
- Ground plane in homogeneous form: horizon line `l=[-k,1,-b]`, normal `n ∝ K^T l`, normalize/sign so normal points in camera-positive-Y direction when appropriate, plane `n·X + d = 0` with `d=-H_normal`. If `camera_height_m` is defined along gravity/normal, this is valid; verify normalization and KITTI Y-down convention. If height is vertical axis instead of normal distance, compute conversion. For KITTI P2 with nonzero projection translation, verify ray origin/backprojection correctly rather than blindly discard translation.

### 3.2 Query-based Touchpoint Head (only Q3 of decoder **pass 1**)
- Shared MLP across queries: Linear(256,256) + ReLU + Linear(256,128) + ReLU + Linear(128,2*P_max). Default `P_max=4` -> 8 coords/query. No external object detector. Add per-point confidence ONLY if needed/available from labels; first run use detection probabilities and GT valid masks for supervision.
- Use same pass-1 Q3 per-query features as existing detection heads. `q_i` predicts box,class,touchpoints of the SAME object hypothesis; no global keypoint clustering. The query **index** is the association key within one pass (not identity across images). At training use original Hungarian matcher mapping each query to GT object, use its pseudo labels, apply SmoothL1 in normalized image coordinates (or pixel units normalized by image dims); ignore invalid points/unmatched/background queries. Do not accidentally supervise all Q queries.
- Predict box-relative positions via `r=(1.5*sigmoid(raw)-0.25)` (supports slight out-of-box predictions; configurable). Convert using **actual** query 2D `xyxy` box derived from local `pred_boxes[...,0:6]`, then to original model-input pixel coordinates. Do not detach box by default; log possible side effects and support config detaching if needed. Tests must verify target coordinate consistency.
- Keep existing detection heads and original detection loss in BOTH phases. Pass-1 detection/class predictions also provide soft object reliability and spatial box masks at inference; do not use GT boxes or GT matching for geometry at inference.
- `touchpoint_loss`: SmoothL1 only valid matched Car objects, normalized by number of valid labelled points, handle zero positives safely. Ensure pseudo labels are aligned by actual KITTI image ID and object/annotation ID, and not lost through target filtering/augmentation.

### 3.3 Differentiable ray-plane geometric depth
- For point p=[u,v,1], ray r=K^-1 p in the consistent coordinate frame. Intersection: `lambda = -d / (n^T r)`, `P = lambda*r`; require positive intersection and stable denominator; invalidate near-horizon rays, out-of-range depths, missing calibration/height, implausible contacts. Use FP32 geometric calculations with autocast disabled if necessary; never propagate NaNs/Infs.
- Aggregate valid wheel-contact depths to object center Z: use existing GPro3D-defined assumption/aggregation verified against oracle GT-derived touchpoints, with explicit correction/masks when fewer contact points are valid. Don't assume each wheel depth equals center depth; verify KITTI coordinate frame, projected contact semantics, train and val expected center Z. Default confidence-weighted robust mean only if justified by pseudo label ordering; report single/partial contacts separately.
- Produce reliability per object (valid ray fraction, visibility, detection score, geometric consistency, near-horizon stability). Invalid priors must leave MonoDETR unchanged (identity fusion), not produce fabricated zero depth.

## 4. Spatial projection shared by both modes

Geometry currently is object-level `[B,Q]`; map to depth-grid size `[B,*,Hd,Wd]` using each query's **predicted 2D box from the same pass** and confidence. Use differentiable sigmoid-edge soft rectangle masks (e.g. `(sigmoid((x-xmin)/tau)*sigmoid((xmax-x)/tau)*sigmoid((y-ymin)/tau)*sigmoid((ymax-y)/tau))`) and soft object weights from class confidence * geometric validity; allow independent tau. Avoid hard GT boxes, hard top-k threshold and NMS in the learnable fusion path. Near-zero or invalid total coverage -> exact identity to original features/logits. Occlusion/overlap: use confidence/soft depth-aware weights, normalize across objects; log conflicts and test overlapping boxes.

If official training has multiple query groups (e.g. 11x50), use **one consistent canonical query group** for spatial geometry prior, not 550 duplicated proposals, while preserving the repo's full matching and detection loss semantics. Test this against inference 50 queries. The module should support a small number of objects per image with valid priors without O(B*Q*D*H*W) materialization; prefer separable masks and `einsum`/`matmul`.

For geometry representations, define `Z_geo_map [B,1,Hd,Wd]`, `valid_coverage [B,1,Hd,Wd]`, optionally confidence/dispersion channels and per-depth-bin prior `[B,D,Hd,Wd]`.

## 5. Model variant A — `feature_level`

Working point in original DepthPredictor:
- `src` after multiscale pooling/fusion + `depth_head` has shape `[B,256,Hd,Wd]` (or actual). Original baseline path: `src -> depth_classifier -> orig_logits -> orig_weighted_depth`; `src -> depth_encoder -> orig_embedding`, with positional embedding from orig weighted depth. Preserve it for pass 1 and original depth supervision.
- Pass 2 geometry adapter maps `[Z_geo_map, valid_coverage, confidence/dispersion(optional)] -> 256-channel residual delta`: e.g. Conv(3,64,3,pad1)+GroupNorm(8)+SiLU -> Conv(64,256,3,pad1). Gate: `src_fused = src + alpha * valid_coverage * adapter(geo)`. Start `alpha=0` or 0.01, learnable and/or ramped, no sudden overwrite; `alpha=0` must produce original output exactly in eval mode.
- Run the **same original** `depth_classifier(src_fused)` to produce `fused_logits` `[B,81,Hd,Wd]`, not a brand new unrelated head unless absolutely necessary. This adds a GT-supervised *fused depth map* without breaking baseline compatibility.
- Compute `weighted_depth_fused` from `softmax(fused_logits)` and the SAME official LID depth_bin_values including bg handling; run depth encoder **on `src_fused`** and recompute depth positional embedding from weighted depth. Pass both resulting depth memory/position features and weighted depth to decoder pass 2. Pass 1 uses untouched originals.
- Losses: original `depth_map_loss` on original logits (keep), `fused_depth_map_loss` via the SAME dataset labels/DDNLoss on fused logits (new), original detection loss including query depth loss. A separate physical-geometry SmoothL1 loss is optional and disabled by default.

## 6. Model variant B — `distribution_level`

- Keep `src`, original `depth_classifier(src)`, `orig_logits [B,D+1,Hd,Wd]`, original `depth_encoder(src)` unchanged.
- From each object geometric depth `z_i` build a Gaussian in **actual LID bin centers** `c_j` (j=0..D-1): `logp_geo_i,j = -0.5*((c_j-z_i)/sigma_i)^2`; normalize along D if needed. Constant sigma initially 2.5 m (configurable; later uncertainty head is not required). Softly project to grid with object soft masks/reliability; do not mix the background category into geometric foreground likelihood.
- Gated product/logit fusion on foreground categories only:
  `fused_fg_logits = orig_logits[:, :D] + alpha * coverage * clamp(log(P_geo_fg + eps) - mean_or_reference_per_pixel, min, max)`;
  `fused_bg_logit = orig_logits[:, D:D+1]` exactly unchanged. Apply spatial coverage such that where invalid `fused_logits == orig_logits` exactly. Take care of normalized priors / zero mass / out-of-range depths and numerical stability under AMP. `alpha` near zero at start, trainable; maybe confidence-modulated.
- Recompute `weighted_depth_fused = sum softmax(fused_logits)*orig_depth_bin_values` and `depth_pos_embed_ip_fused = original_depth_pos_embedding(weighted_depth_fused)` using existing differentiable interpolation. **Critical:** the original implementation's decoder reads `depth_pos_embed = depth_encoder(src) + depth_pos_embed_ip`. Therefore set `depth_memory_fused = original_depth_encoder(src) + depth_pos_embed_ip_fused` (not the old original `depth_memory`!), and also pass `depth_pos_embed_ip_fused` to decoder; Pass 2 uses fused weighted_depth for any sampled depth in the detection head. Confirm local implementation. Optionally add a `distribution_to_depth_feature_adapter` only if tests show fused logits have no effect on decoder; log exactly what changed.
- Losses: same original `depth_map_loss`, same original detection depth regression loss; **new fused depth map DDNLoss** on `fused_logits` and same GT object-wise depth-bin supervision. Preserve bg class, near-camera overlap rule and target bin code from repo. For α=0 Pass 2 must match Pass 1 numerically in eval (within FP tolerance).

NOTE: feature-level vs distribution-level is the **only intended ablation variable** beyond mode-specific adapter costs. All other settings and evaluation remain identical.

## 7. Two-phase, end-to-end training (not independent models)

Load user-provided working pretrained MonoDETR checkpoint. Existing parameter names/shapes load; new Horizon, Touchpoint and Fusion parts initialized explicitly. DO NOT freeze existing backbone/encoders/decoder by default; fine-tune original params with smaller LR, new heads with larger LR. Initialization from pretrained is not freezing.

**Phase 1 (epoch 0..N-1, default N=8):** ONE decoder pass, baseline foreground depth / depth encoder / depth cross attention **ON**. Detection/class/2D/3D/original depth losses remain ON. Horizon heatmap+line loss ON. Q3 touchpoint loss ON using GT-derived pseudo labels. Geometry ground-plane/raycast can run for metrics and optional direct-depth supervision; **fusion OFF**, no pass 2. Geometry/Fusion weights not trained from final detection yet. Inspect `val/pred_geo_depth_mae` and valid prior rate at Phase 1 end.

**Phase 2 (epoch N..E-1, default E=30):** same model and weights, fusion ON:
1. Compute backbone/projected multiscale, visual encoder once, depth predictor `src`, orig logits/weighted depth and original depth encoder once, horizon head once.
2. Pass 1 full original 3-layer decoder with **depth cross-attention ON** reading original depth. Same query learned embeddings as baseline. `Q3_pass1` -> ORIGINAL detection heads + Touchpoint MLP. Keep Pass-1 touchpoint auxiliary loss, optionally low-weight Pass-1 detection loss for stability.
3. Using predicted (never GT) touchpoints, predicted horizon, P2/K and height: differentiable plane + raycast + object geometric depth + spatial geometry prior.
4. Variant A/B fusion -> rebuild relevant depth memory/position and `weighted_depth` with original depth predictor internals reused, **don't rerun backbone/visual encoder**. Decoder **Pass 2** runs all 3 layers with SAME decoder weights and initial learned queries, uses fused depth. Final original detection heads produce final 3D boxes.
5. Backprop from final pass-2 detection and fused-depth losses through pass 2 -> geometry fusion -> GPro3D prior -> touchpoint/head and horizon. Maintain direct supervised losses on pass1 touchpoints and horizon to prevent drift. **No `.detach()`, `no_grad` or frozen teacher** for pass1/fusion during Phase 2; no hard gating that cuts required gradients. Non-differentiable OpenCV horizon refinement is OFF along training gradient path.
6. Use shared weights between decoder passes; optimizer includes each parameter once, do not create optimizer per pass. Keep original loss names intact, avoid double-counting. LR groups: pretrained 1e-5, new head 1e-4, fusion 1e-4 as initial config; actual scheduler/resume via same optimizer state. Prefer bf16 autocast when supported, FP32 geometry, grad clip 0.1, batch=2 and grad_accum=4 (verify RTX4090 memory; if OOM reduce batch, add gradient checkpointing before modifying architecture/query count).
7. New fusion gate starts at exact 0 or small positive value, preserve identity, ensure it is a trainable parameter or ramp; α=0 gives zero gradient to geometry through fusion initially but directly supervised touchpoint/horizon losses continue; gate should receive a real gradient. Log gate value/gradient norm and separately validate gradient flows once α>0.

Total conceptual loss (don't duplicate weights already inside original MonoDETR criterion):
`L_P1 = L_MonoDETR_original + λ_t L_touch + λ_hmap L_hmap + λ_hline L_hline`.
`L_P2 = L_MonoDETR_final_pass2 + λ_aux1 L_det_pass1(optional) + λ_t L_touch_pass1 + λ_hmap L_hmap + λ_hline L_hline + λ_fused L_fused_depth_map + λ_geo L_geometry_center_depth(optional)`.
`L_MonoDETR_original` already includes query depth, foreground depth-map, class/box/3D dim/angle and auxiliary decoder losses. In Phase 2 keep the original foreground depth-map loss on `orig_logits` ON; include it exactly once even if final detection criterion uses Pass 2 outputs. Prefer explicitly build separate `losses_base_detector`, `original_depth_map_loss`, `fused_depth_map_loss` dictionaries with audited double-count check. Use existing DDN targets for fused supervision, not an invented dense depth map. Default λ_t=1, λ_hmap=1, λ_hline=0.5, λ_fused=1.0; these are tuning starting points.

Phase switch must be resumable: checkpoint saves completed epoch, phase, optimizer/scheduler/scaler states, all heads and fusion parameters, random seeds, resolved config; exact `resume` shouldn't restart warm-up. `best` uses chosen validation metric (default final KITTI Car AP3D R40 Moderate, with depth MAE as secondary and NaN safety). Support saving best-depth as separate checkpoint.

**Inference/test always:** One image -> shared backbone/encoders -> original Decoder Pass 1 -> Q3 touchpoint, horizon -> geometry -> variant fusion -> Decoder Pass 2 -> 3D boxes. No GT horizon/touchpoints/boxes passed into model. If no valid geometric priors, fusion identity and Pass 2 returns baseline-equivalent boxes in eval. No optimizer updates at inference. Report measured latency ms/image and peak VRAM; two decoder passes ≠ two full backbone runs.

## 8. Mandatory validation statistics and FAIR baseline comparison

After each validation epoch and as a final explicit command, evaluate:
- `baseline_pretrained`: original MonoDETR checkpoint on EXACT SAME KITTI `val` images, preprocessing, class set, calibration, detection thresholds and matcher.
- `ours_pass1`: original-depth output in trained model (no geometry feedback).
- `ours_pass2`: final geometry-fused output.
- Debug geometric depth: `D_geo` predicted and optionally oracle-derived upper-reference, CLEARLY label oracle and never include GT-derived prior in inference metrics.

Record quantitative metrics in meters: per-object center-Z **MAE**, **RMSE**, median AE, P90 AE, signed bias, AbsRel, sample count, valid geometry rate. Use 2D IoU>=0.5 + class-constrained **one-to-one prediction–GT association for analysis only** (or official KITTI matching if already available), and report threshold/protocol explicitly; do NOT silently compare predictions with GT by query index. Include both each-model TP-only errors and **paired-intersection matched-object comparison** to control for differing detections, with missed GT counts and recall. Avoid unfairly claiming improvement from ignoring hard-to-detect objects.

Distance buckets `[0,10),[10,20),[20,30),[30,40),[40,60),>=60` m; KITTI Easy/Moderate/Hard where possible; class Car as original config (additional classes if supported). Report KITTI official 3D AP R40 Easy/Moderate/Hard, BEV AP R40 if available, and inference latency/memory. If model checkpoint is from a different KITTI split, explicitly warn about data leakage and do not publish comparisons until fixed. Validation GT can be used for *metric computation only*, never geometry input.

At end print human-readable comparison table like:
```
Model          matched N  center-Z MAE(m) RMSE(m) MedAE(m) P90(m) AbsRel  AP3D_M
Baseline        ...         ...       ...       ...      ...     ...    ...
Ours Pass1      ...         ...       ...       ...      ...     ...    ...
Ours Pass2      ...         ...       ...       ...      ...     ...    ...
Paired Δ Ours-Baseline (same objects): ... m  (negative = improvement)
```
NEVER invent these numbers. Write `outputs/<run>/val_metrics.json`, `val_metrics.csv`, `depth_errors_by_distance.csv`, `per_object_depth_errors.csv` (or parquet), and `val_comparison.md`. Per-image/per-query data fields include model ID, dataset split, image ID, matching GT ID if metric-only, class/score, 2D bbox, 3D bbox, original-pred depth, geo depth, fused predicted depth, GT center-Z (metric-only), valid flag, touchpoints, horizon/ground plane, matched IoU, error, geometry confidence; absent prediction marked as missing not zero. Never expose GT-derived fields as inference input.

Add regression script that reevaluates baseline and E1 with SAME evaluator; verify a copy of baseline checkpoint produces identical outputs to existing evaluator before reporting gains.

## 9. Dashboard integration (retain my existing EXACT two panels)

Connect to my current localhost web dashboard:
- Left panel existing image overlay: model selector `Baseline MonoDETR`, `E1 Feature`, `E1 Distribution`; GT/pred 3D boxes, horizon raw/final, predicted query touchpoints, optional GT-derived pseudo labels only on labeled splits, per-object orig/geometric/fused depth annotations, toggles. Respect original camera image coordinate transformations and projection. Distinguish Pass1 and Pass2 predictions and don't mix cached results from runs.
- Right panel existing BEV: predicted/GT 3D boxes, concentric distance rings, depth discrepancy lines and colors, object query association, hover/select same object across left/right. No new dashboard windows or elaborate UI redesign.
- Add compact validation summary beneath/alongside existing controls (without creating a third main window): baseline vs feature/distribution depth MAE, RMSE, AP3D and paired Δ; filter by distance bucket and matched-only. Read validation report generated by evaluator, NOT recalculate inconsistent metrics in frontend.
- Reuse existing backend endpoints/schema if present; append versioned model outputs & metrics, detect missing checkpoints, avoid blocking page when model not loaded. Provide commands/ports for running dashboard with newly added variants.

## 10. W&B logging; .env/config configuration

Provide opt-in Weights & Biases integration using `wandb` Python SDK, with configurable `project`, `entity`, `run_name`, `tags`, `mode` (`online|offline|disabled`), and configurable frequency. Never hardcode credential in config/code and never print token. `.env.example` must list `WANDB_API_KEY` optional (not committed), `WANDB_ENTITY`, `WANDB_PROJECT`, dataset/checkpoint/pseudo-label paths, camera height, dashboard endpoint/token if required, PORTs. Use `.gitignore` for `.env` and W&B private credentials; if key not set gracefully support offline/disabled and tell user exactly what to add, don't block training.

Log train and val separately; log `phase`, `fusion_mode`, step, epoch, LR by param group, full loss decomposition (original query depth, original foreground depth map, fused foreground depth map, touchpoint, horizon heatmap/line, detection), horizon slope/center error, touchpoint px MAE, geometric center-depth MAE, baseline original/pass1/pass2 final depth MAE/RMSE, AP3D R40, valid ray/object ratio, fusion α, gradient norms of touchpoint/horizon/fusion (at periodic intervals), latency, peak allocated VRAM, example overlay images and BEV images using dashboard renderer if possible. Record resolved config, seed, git SHA, checkpoint identity; tag the two variants for W&B run comparison. Log comparable validation `step` and `epoch` across both runs. Upload small artifacts for val reports and best checkpoint metadata, not all raw images.

## 11. Tests/acceptance gates (do not declare success until these pass)

- Checkpoint compatibility: all expected baseline modules loaded; only intended new keys missing; baseline inference same as before at fusion OFF.
- Shapes and coordinate invariants for B=1 and B=2, empty GTs, varying pseudo-label valid count, KITTI 1280x384; assert finite outputs.
- Horizon line coordinate scaling/GT Gaussian target and calibrated ray test on flat road: K principal point at true horizon, plane Y=H, ray at road pixel yields positive finite Z as expected; near-horizon ray invalidated.
- Transformation correctness: raw image ↔ affine input ↔ pseudo labels ↔ K/P2 under resize/flip/crop; default geometry-aware config disables unresolved inconsistent transforms.
- Hungarian matching associates each query's touchpoints with SAME GT object (including groups and multiple cars); unmatched queries have no regression loss.
- Depth bins: LID bins exactly align with baseline, 80 + bg if default; background logit protected in distribution-level mode, no NaNs from log(0), exact identity when no valid geometry.
- `alpha=0` eval identity: original depth features/logits/weighted depth/decoder final predictions identical (tol for floating point) between pass1 and pass2; trainable α and nonzero geometry gradient after α becomes positive.
- `lambda_t L_touch` backprops to touchpoint MLP and pass1 decoder; in Phase2 with α>0 `L_det_pass2` or `L_fused_depth_map` backprops to Touchpoint/Horizon heads through differentiable geometry. Print finite `grad_norm` and test for accidental `detach` and hard query selection.
- Phase1: 1 decoder pass, detection and touchpoint/horizon losses nonzero, fusion off. Phase2: 2 decoder passes, gradients as designed. Both inference modes support unseen single image without GT.
- Mixed precision/FP32 geometry tests, single-GPU RTX4090 training dry-run 10 iterations without OOM/NaN. If memory issue, use gradient accumulation and checkpointing, report tradeoffs.
- Baseline and E1 val metrics computed under SAME matching, distance bins, split, preprocessing and GT depth definition, with matched counts and paired comparisons. Dashboard shows exact same report numbers as JSON/CSV. W&B logs visible or offline files generated.

## 12. Suggested implementation order and final response format

1. Audit local code/config/pseudo labels/dashboard and checkpoint; report discovered differences from upstream. Do not ask for things you can inspect.
2. Add pseudo-label alignment/augmentation checks, horizon head + ground-plane + differentiable touchpoint head/raycast + unit tests; run Phase1 single-batch and short validation.
3. Refactor exact `DepthPredictor` and Transformer to encode visual memory once and decode twice; verify alpha=0 identity and baseline parity.
4. Implement Feature Fusion mode with fused DDNLoss, gradient checks and small dry-run; implement Distribution Fusion mode with valid depth bins/bg masking, positional embedding path, gradient checks.
5. Integrate two-phase schedule/resume/checkpoints and both config files; run short train and eval for both models if GPU/dataset available.
6. Integrate baseline-versus-ours statistics, dashboard, W&B, .env.example and docs.
7. Report: files changed, actual tensor shapes, verified forward/backward gradients, implemented versus pending features, commands executed, final numeric validation results (or clear blocker if not runnable), summary and next experiments. Do not assert AP/MAE improvement without measured evidence.

**CRITICAL DO NOTS:** No GT-derived touchpoints/ground planes fed at inference. No using epoch-previous predictions as model input. No turning off original decoder depth cross-attention for pass1. No unannounced change to class mapping, KITTI split, base detector criterion, learned query grouping, depth calibration or data augmentation. No replacing my 2-panel dashboard. No silent mismatch of box formats, LID bins, depth labels or camera coordinates. No fake metric values. Keep the work scoped to E1; E2 query initialization and E3 cross-dataset robustness come later.
