# E1 implementation: MonoDETR × GPro3D

This repository now contains one continuously trainable E1 model with two
ablation configurations:

- `configs/e1_feature_level.yaml`: injects a gated geometry residual into the
  stride-16 latent depth feature, then reruns the shared depth classifier and
  depth encoder.
- `configs/e1_distribution_level.yaml`: builds a Gaussian categorical prior in
  MonoDETR's 80 LID foreground bins, preserves the background logit exactly,
  and rebuilds weighted-depth positional features used by decoder pass 2.

Both variants use the same baseline checkpoint, pseudo labels, epoch schedule,
decoder weights, losses, matcher and evaluation protocol. See
`docs/e1_repo_audit.md` for the measured local tensor shapes and checkpoint-key
audit.

## Geometry and coordinate contract

The Horizon FPN consumes the projected stride-8/16/32 features and predicts a
stride-4 one-channel ridge. Vertical soft-argmax followed by confidence-weighted
least squares yields `v = k*u + b` in 1280×384 network-input pixels. The
`align_corners=False` pixel-centre mapping is implemented and unit-tested.

During evaluation/inference, the enabled vertical-edge refiner follows GPro3D
Algorithm 1: denormalize RGB, Gaussian blur, Canny, probabilistic Hough, retain
70–110 degree segments, and select the strongest coherent angle cluster. If the
cluster passes the count and angle-standard-deviation checks, its perpendicular
slope is fixed and the intercept is re-fitted by weighted least squares from
the learned heatmap ridge. If it is unreliable, the operation is an exact
fallback to the raw horizon. This OpenCV stage is deliberately bypassed under
`model.train()` so it does not cut the Phase-2 gradient path. Inference
artifacts store raw/final lines plus edge count, mean angle, standard deviation
and validity. The dashboard and validation renderer show the raw line faintly
when refinement is accepted, with the final geometry line drawn on top.

For transformed projection matrix `P2_input = A @ P2`, `M=P2_input[:,:3]` and
image line `l=[-k,1,-b]`, the ground normal is `normalize(M.T @ l)`, signed toward
positive camera Y. The plane is `n.X - H = 0`, where `H` is the configurable
normal-distance assumption `CAMERA_HEIGHT_M` (default 1.65 m). This height is a
research assumption, not KITTI per-frame ground truth. Rays use the full 3×4 P2:
camera centre `C=-M^-1 p4`, direction `M^-1 [u,v,1]`, and intersection with the
plane. Near-parallel, behind-camera, off-image and out-of-range contacts are
invalid and produce exact identity fusion.

The touchpoint head reads the final query feature of decoder pass 1. It predicts
four named slots `(LF, RF, RR, LR)` box-relatively in range `[-0.25,1.25]` and
uses the repository's real six-value `cxcylrtb` box conversion. Supervision uses
the pass-1 Hungarian assignment, valid Car slots only, and the pseudo-label
object ID is the original KITTI label-file row. Default E1 configs disable crop
and flip; photometric augmentation remains enabled. Depth targets remain physical
camera-Z (`depth_scale: none`).

Four footprint-contact Z values have mean equal to KITTI object centre Z when
all four slots are correct. Partial/invalid rays are averaged and their valid
fraction contributes to reliability; their validity is reported so partial
contacts are not silently treated as complete observations.

## Two training phases

- Epochs 0–7: one decoder pass. Original MonoDETR detection, query-depth and DDN
  depth-map losses remain active; horizon and matched touchpoint heads train.
  Fusion is off.
- Epoch 8 onward: the backbone and visual encoder run once. Decoder pass 1 uses
  untouched MonoDETR depth. Predicted horizon/touchpoints create the differentiable
  prior. Shared decoder pass 2 uses fused depth and produces final detections.
  Original DDN loss is counted once, fused DDN once, and pass-1 final detection
  losses receive weight 0.1.

The checkpoint stores model, optimizer, scheduler, AMP scaler, completed epoch,
phase, global step, best AP, best depth MAE, RNG states and resolved config.
`checkpoint_last.pth`, `checkpoint_best.pth`, and
`checkpoint_best_depth.pth` are written without touching the baseline directory.

## Setup

The `wandb` SDK is listed in `requirements.txt` and installed in the current
`monodetr` environment. Copy the environment template and keep it private:

```bash
cd /home/ml4u/BKTeam/Duong/monodetr
cp .env.example .env
```

The checked defaults already point to the local KITTI data, baseline checkpoint
and generated pseudo labels. For W&B web tracking, authenticate without putting
the key in YAML:

```bash
conda activate monodetr
wandb login
# In .env: WANDB_MODE=online, WANDB_PROJECT=..., optional WANDB_ENTITY=...
```

Keep `WANDB_MODE=offline` when the machine cannot reach W&B; runs are retained
locally and can later be uploaded with `wandb sync <offline-run-dir>`. Use
`WANDB_MODE=disabled` to turn it off. Initialization failures never abort model
training. Logged data includes phase, all weighted losses, horizon/touchpoint and
geometry errors, per-group LR, fusion alpha, head/fusion gradient norms, AP,
matched depth statistics and validation report artifacts. The API key is never
placed in config, checkpoint or logs.

## Train and resume

```bash
conda activate monodetr
python tools/train_val.py --config configs/e1_feature_level.yaml --env-file .env
python tools/train_val.py --config configs/e1_distribution_level.yaml --env-file .env
```

To resume, add one of these under `trainer` in the selected YAML (or a private
child config):

```yaml
resume_model: true  # resolves checkpoint_last.pth in this run directory
# resume_model: outputs/e1_feature_level/monodetr/checkpoint_last.pth
```

Feature and distribution checkpoints are respectively written below
`outputs/e1_feature_level/monodetr/` and
`outputs/e1_distribution_level/monodetr/`.

### Always-two-pass training experiments

The recommended main run initializes all original detector tensors from the
MonoDETR checkpoint but starts a fresh optimizer/scheduler and executes Pass 1,
geometry fusion, and Pass 2 from epoch 0:

```bash
python tools/train_val.py --config configs/e1_feature_joint_epoch0.yaml \
  --env-file .env
```

The separate scratch-detector ablation does not load a MonoDETR checkpoint. It
retains the repository's ImageNet initialization for ResNet-50 and trains the
random transformer/depth/detection/E1 modules for the original 195-epoch-style
schedule:

```bash
python tools/train_val.py --config configs/e1_feature_scratch_imagenet.yaml \
  --env-file .env
```

Both use feature-level fusion, Touchpoint v2 supervision, two decoder passes
from the first batch, a `0.001` trainable fusion gate capped at `0.1`, and no
vertical-edge eval-only refinement. The pretrained run keeps Pass-1 detection
at weight `0.5`; scratch keeps it at `1.0`. These are intentionally separate
output/W&B runs and should not resume from one another.

## Export and fair evaluation

First export the untouched baseline once on the exact same `val` loader:

```bash
python tools/train_val.py -e --config configs/monodetr.yaml \
  --checkpoint-path outputs/monodetr/checkpoint_best_1.pth
```

Export E1 pass 1 and pass 2 in one model forward per image:

```bash
python tools/train_val.py -e --config configs/e1_feature_level.yaml \
  --env-file .env \
  --checkpoint-path outputs/e1_feature_level/monodetr/checkpoint_best.pth \
  --export pass1=pass1 --export pass2=pass2
```

Then generate the matched centre-Z report:

```bash
python tools/evaluate_e1_depth.py \
  --model baseline_pretrained=outputs/monodetr/outputs/data \
  --model pass1=outputs/e1_feature_level/monodetr/outputs/pass1/data \
  --model pass2=outputs/e1_feature_level/monodetr/outputs/pass2/data \
  --output-dir outputs/e1_feature_level/validation
```

The evaluator performs class-constrained one-to-one Hungarian association at
2D IoU ≥ 0.5. It reports TP-only MAE/RMSE/median/P90/bias/AbsRel, recall and
missed GT, paired-intersection delta, distance buckets, KITTI difficulties,
official AP3D/BEV R40, latency metadata, CSV rows and a Markdown table. Validation
GT is metric-only and never enters model inference. No missing prediction is
converted to depth zero.

### Full-val branch depth MAE

Inference artifacts are now emitted for both baseline MonoDETR and E1. Each
selected query records three distinct depth definitions: final MonoDETR head
depth, the pure weighted-depth map sampled at its projected centre, and (for
E1) valid GPro3D ray/plane depth. Export the complete val split first:

```bash
python tools/train_val.py -e --config configs/monodetr.yaml \
  --checkpoint-path outputs/monodetr/checkpoint_best_1.pth \
  --export monodetr_val=final
```

Then evaluate every frame listed in `val.txt`:

```bash
python tools/evaluate_val_depth_mae.py \
  --artifact-dir outputs/monodetr/outputs/monodetr_val/artifacts \
  --output-dir outputs/monodetr/validation_depth_mae
```

For E1, point `--artifact-dir` at the exported Pass-2 artifacts. The report
contains `monodetr_final`, `monodetr_pass1`, `depth_predictor_original`,
`depth_predictor_selected`, and `gpro3d`. Matching is class-constrained,
one-to-one 2D IoU ≥ 0.5. MAE uses matched detections only and reports both
detector recall and per-branch coverage. By default the command fails if even
one val artifact is missing, preventing an accidental partial-split result.
It additionally reports all branches on the exact GPro3D-valid object subset;
use that paired table—not the unequal-coverage headline rows—when comparing
whether geometric depth is more accurate than Depth Predictor depth.

## Dashboard

```bash
python dashboard/server.py --split val --port 8765
```

Open `http://127.0.0.1:8765`. The original image and BEV panels are preserved.
The selector contains Baseline MonoDETR, E1 Feature and E1 Distribution when
their KITTI output folders exist. E1 inference artifacts add predicted horizon,
query touchpoints, pass-1/pass-2/geometric depths, plane and confidence. The
Validation popover reads evaluator JSON directly and supports distance buckets.

## Verification status

CPU checks cover checkpoint compatibility, config inheritance, optimizer-group
uniqueness, empty GTs, pseudo-label data loading, full-P2 ray geometry, horizon
pixel centres, exact no-geometry fusion identity, protected background logits,
finite gradients through distribution fusion, evaluator matching and dashboard
payloads. Full CUDA deformable-attention forward/backward, the requested 10-step
RTX 4090 dry run, latency/VRAM and real AP/MAE numbers must be run when a GPU is
available; this environment currently cannot communicate with the NVIDIA driver.
Do not interpret the absence of those numbers as a measured gain.
