# MonoDETR Geometry Debugger

The dashboard can use existing KITTI prediction files or run selected
checkpoints live on one image. It keeps exactly two primary views: the raw
camera image and the camera-centric BEV comparison.

## Start the dashboard

From the repository root:

```bash
python3 dashboard/server.py \
  --dataset-root data/KITTIDataset \
  --split val \
  --pseudo-labels outputs/near_gpro_pseudo_labels/val.jsonl \
  --device cuda:0
```

Open <http://127.0.0.1:8765>. If CUDA is omitted, `--device auto` uses CUDA
when available and otherwise falls back to CPU. Models are loaded lazily,
cached on CPU, and moved to the selected device one at a time so comparing
several versions does not keep all of them in GPU memory.

In **Models & weights**, enable the original MonoDETR and any custom model,
choose one checkpoint for each, then press **Run**. The server uses the
checkpoint's `resolved_config` when present and its registry config as a
fallback. The confidence slider filters the returned top-k detections without
running inference again.

The camera overlay uses:

- magenta dashed line: pseudo-label horizon;
- model-coloured dotted line: raw predicted horizon;
- model-coloured solid line: vertically refined horizon, only when refinement
  is valid;
- white diamond: GT near-edge midpoint;
- model-coloured dot: predicted near touchpoint (or the four E1 touchpoints);
- each predicted box label: `Zpred` and the 2D-IoU-matched `Zgt`.

Click a matched box to show `Zpred`, `Zgt`, `delta Z`, predicted touchpoint
pixel coordinates, and `Zgeo` in the BEV tooltip. Matching is class-constrained,
one-to-one Hungarian assignment on raw-image 2D IoU >= 0.5; it does not use
depth to decide the match. Original MonoDETR has no horizon/touchpoint head, so those fields
correctly appear as unavailable while its boxes and final depth remain visible.

The server automatically prefers
`outputs/near_gpro_pseudo_labels/<split>.jsonl` when `--pseudo-labels` is not
specified, falling back to the original geometry pseudo-label file.

## Register checkpoints

Edit `dashboard/model_registry.json`. A live model entry supports explicit
weights and checkpoint globs:

```json
{
  "inference": {
    "builder": "near_gpro",
    "config": "configs/near_gpro_zmap_replace_scratch.yaml",
    "default_weight": "best",
    "weights": [
      {"id": "best", "label": "best AP", "path": "outputs/run/model/checkpoint_best.pth"}
    ],
    "checkpoint_globs": ["outputs/run/model/checkpoint_epoch_*.pth"]
  }
}
```

Only checkpoint files that currently exist are exposed in the browser. Use
`builder: monodetr`, `e1`, or `near_gpro` to select the correct constructor.
Restart the dashboard after changing the registry or adding checkpoints.

## Original implementation brief

You are working in my existing MonoDETR research repository.

Implement a minimal tool for generating GPro3D-inspired geometry pseudo-labels and visualizing MonoDETR predictions.

**Do not over-engineer. Focus only on this phase.**

Reference paper:

GPro3D: Deriving 3D BBox from ground plane in monocular 3D object detection (Neurocomputing, 2023).

Paper: https://ise.thss.tsinghua.edu.cn/mig/2023-1.pdf

## 1. Inspect Existing Code

Inspect the repository to locate:

- KITTI 3D annotations and calibration.
- MonoDETR inference and checkpoint.
- Existing 3D box projection utilities.
- Dataset paths and validation split.

Reuse existing implementations wherever possible.

Do not modify the training pipeline.

## 2. Generate Wheel-Contact Pseudo-Labels (GPro3D, Eq. 4–6)

Follow Section 3.2.2 of GPro3D.

Given a vehicle GT 3D bounding box with length l, width w, bottom-center translation T = [x, y, z] and rotation θ around camera Y:

Define:

- k_l: ratio of front-to-rear wheel distance to box length.
- k_w: ratio of left-to-right wheel distance to box width.

The four wheel-contact points in GPro3D's object-local coordinate system are:

```text
P_LF = [ +(k_l/2)*l, 0, +(k_w/2)*w ]
P_RF = [ +(k_l/2)*l, 0, -(k_w/2)*w ]
P_RR = [ -(k_l/2)*l, 0, -(k_w/2)*w ]
P_LR = [ -(k_l/2)*l, 0, +(k_w/2)*w ]
```

The paper defines the local object origin at the bottom-center of the bounding box, with the vehicle front along local +X.

Transform each point to camera coordinates:

```text
P_camera = R * P_local + T

R = [[ cosθ, 0, -sinθ ],
     [ 0,    1,  0    ],
     [ sinθ, 0,  cosθ ]]
```

Then project to image coordinates using camera intrinsics:

```text
z * [u, v, 1]^T = K * P_camera

K = [[fx, 0,  cx],
     [0,  fy, cy],
     [0,  0,  1 ]]
```

Equivalently:

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

Implementation notes:

- Inspect the actual KITTI annotation and MonoDETR yaw conventions. Convert them into the paper's local coordinate convention before applying these equations.
- Use KITTI's actual projection matrix, accounting for camera translation offsets when applicable.
- Do not assume the paper's θ has the same sign or zero-angle orientation as raw KITTI rotation_y.
- Keep k_l and k_w configurable rather than hardcoding unverified values.
- Generate four candidate points for each supported vehicle.
- Filter invalid projections and points behind the camera.
- Preserve valid but potentially occluded candidate points, marking visibility as unknown.
- Save the wheel-contact pseudo-labels with their object IDs.

## 3. Generate Ground-Plane and Horizon-Line Pseudo-Labels (GPro3D, Section 3.3.5)

Follow the pseudo-label procedure described in Section 3.3.5.

### Step A — Fit the ground plane

Extract the GT 3D bounding-box bottom centers of valid objects in the current frame:

```text
P_i = (X_i, Y_i, Z_i)
```

Use these points to fit a scene-level plane:

```text
aX + bY + cZ + d = 0
```

Use a simple least-squares/SVD plane fitting procedure.

If there are not enough non-collinear valid ground-contact samples, mark the plane as invalid rather than fabricating a label.

For numerically valid b:

```text
Y = A*X + B*Z + C

A = -a/b
B = -c/b
C = -d/b
```

Here, C represents the plane's Y-intercept in camera coordinates, not necessarily the exact Euclidean camera height.

### Step B — Derive the horizon line

GPro3D represents the horizon line as:

```text
v = k_h*u + b_h
```

From its ground-plane equation (Eq. 19):

```text
Y = (k_h*fx/fy)*X
  + ((k_h*cx + b_h - cy)/fy)*Z
  + H
```

Therefore, given the fitted plane coefficients A and B:

```text
k_h = A*fy/fx

b_h = cy + B*fy - k_h*cx
```

These equations produce the projected horizon line corresponding to the fitted ground-plane orientation.

Generate its visible segment over the image width:

```text
v_left  = b_h
v_right = k_h*(image_width - 1) + b_h
```

Clip the line to the image boundaries if necessary.

Save:

```text
ground_plane: [a, b, c, d]
horizon_line: [k_h, b_h]
```

Important:

- The GPro3D pseudo-label method fits the plane from GT 3D box bottom centers.
- Do not replace it with a fixed horizontal line at cy.
- Do not use the network's predicted horizon line to construct GT pseudo-labels.
- Treat the fitted plane and derived horizon as pseudo-labels, not sensor-measured ground truth.
- This phase only generates the labels. Do not implement a horizon detection network or the vertical-edge enhancement algorithm yet.

## 4. Pseudo-Label Storage

Create a simple script:

```bash
python generate_geometry_pseudo_labels.py
```

Support configurable dataset paths and splits.

Save each frame's data in JSON, including:

- Frame ID.
- Four wheel-contact candidates per valid vehicle.
- Their 2D pixel coordinates and validity.
- Fitted ground-plane coefficients.
- Horizon-line slope and intercept.
- Relevant generation parameters.

No database or complicated data pipeline is needed.

## 5. Minimal Local Web Dashboard

Create a lightweight local web dashboard with exactly TWO main windows side by side.

### LEFT — Camera Image

Display the original KITTI RGB image with independently toggleable overlays:

- GT 3D bounding boxes (green).
- MonoDETR predicted 3D bounding boxes (orange).
- Wheel-contact pseudo-labels (colored dots).
- GPro3D horizon-line pseudo-label (colored line).

Ensure that all overlays align precisely with the KITTI image.

Allow object selection and highlighting.

### RIGHT — BEV Depth Shift Viewer

Display a top-down camera-centric X-Z view.

Requirements:

- Dark LiDAR/radar-style background.
- Ego-camera marker at the origin.
- Concentric metric rings every 10 meters.
- GT oriented 3D bounding-box footprints.
- MonoDETR predicted oriented 3D bounding-box footprints.
- Displacement arrows from matched GT centers to predicted centers.
- Optional ray/ground-plane reprojection of wheel touch points, connected as
  an LF-RF-RR-LR footprint with the original-to-reprojected shift visualized.
- Split-wide touch-point reprojection MAE. Depth MAE is shown directly, while
  per-axis and Euclidean errors are available in the metric tooltip/API.
- Zoom and pan.

For each matched object:

```text
Delta_Z = Z_pred - Z_GT
Delta_X = X_pred - X_GT
```

Display these values when selecting or hovering over an object.

Positive Delta_Z means the prediction is farther away than GT.

Make matching one-to-one, with unmatched objects still visible.

Maintain equal metric scaling along X and Z so the distance rings remain circular.

### Dashboard Controls

Only include:

- Previous/next frame.
- Frame ID selection.
- Model multi-selector.
- Confidence threshold.
- GT box toggle.
- Prediction box toggle.
- Touch-point toggle.
- Horizon-line toggle.
- Depth-shift arrow toggle.

Selecting an object in one view should highlight its corresponding object in the other view.

Do not create additional visualization panels.

## 6. Multi-Model Support

Only MonoDETR needs to run now.

However, the dashboard must allow multiple models to be selected simultaneously in the future.

Keep model outputs in a common lightweight prediction format:

```text
model_id
class
confidence
location_3d
dimensions_3d
rotation_y
```

Use a simple model registry or configurable prediction-file loader.

Different models must have distinguishable colors.

Avoid building a complicated plugin architecture.

## 7. Constraints

- Prioritize correct geometry.
- Reuse existing MonoDETR and KITTI utilities.
- Use real inference results.
- Do not fabricate predictions.
- Do not modify training code.
- Do not implement geometry fusion or query initialization.
- Do not add separate depth-map, point-cloud or 3D scene viewers.
- Do not implement GPro3D's full detection architecture.
- Do not add unnecessary frameworks, services, or abstractions.
- Keep everything runnable locally with simple commands.

## 8. Expected Result

I should be able to:

1. Generate wheel-contact and horizon-line pseudo-labels from KITTI GT.
2. Start the local dashboard.
3. Select a KITTI frame.
4. See 3D GT boxes, MonoDETR predictions, wheel points and horizon line overlaid on the original image.
5. See GT and predicted boxes in BEV with distance rings.
6. Inspect the depth shift of a selected vehicle.
7. Select multiple models when additional prediction files become available.

Start by inspecting the repository, then implement the simplest working end-to-end solution.

Validate the coordinate conventions and pseudo-label projections on a few real KITTI frames.

**Do not expand the scope beyond this MVP.**
