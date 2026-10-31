"use strict";

const BOX_EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
const TOUCH_COLORS = { LF: "#4bd8ff", RF: "#b879ff", RR: "#ff6685", LR: "#ffe066" };
const TOUCH_POINT_ORDER = ["LF", "RF", "RR", "LR"];
const CLASS_ALIASES = { Van: "Car", Truck: "Car", Tram: "Car" };
const $ = (selector) => document.querySelector(selector);

const state = {
  config: null,
  frame: null,
  frameIndex: 0,
  image: null,
  selected: null,
  hovered: null,
  matches: {},
  selectedModels: new Set(),
  selectedWeights: new Map(),
  liveInference: {},
  threshold: 0.2,
  toggles: { gt: true, pred: true, touch: true, near2d: true, reproject: false, near3d: false, horizon: true, arrows: true },
  bev: { centerX: 0, centerZ: 35, zoom: 1, dragging: false, moved: false, lastX: 0, lastY: 0, downX: 0, downY: 0 },
};

const cameraCanvas = $("#camera-canvas");
const bevCanvas = $("#bev-canvas");
const cameraContext = cameraCanvas.getContext("2d");
const bevContext = bevCanvas.getContext("2d");
let cameraTransform = null;
let bevTransform = null;
let loadSerial = 0;

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, ratio };
}

function canonicalClass(value) { return CLASS_ALIASES[value] || value; }
function objectKey(type, modelId, objectId) { return type === "gt" ? `gt:${objectId}` : `pred:${modelId}:${objectId}`; }
function selectedModels() { return state.config.models.filter((model) => state.selectedModels.has(model.id)); }
function visiblePredictions(modelId) {
  if (!state.frame || !state.toggles.pred || !state.selectedModels.has(modelId)) return [];
  return (state.frame.predictions[modelId] || []).filter((item) => item.confidence >= state.threshold);
}

function boxIou2d(left, right) {
  const a = left.bbox_2d, b = right.bbox_2d;
  if (!a || !b) return 0;
  const width = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const height = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const intersection = width * height;
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  return intersection / Math.max(1e-9, areaA + areaB - intersection);
}

function hungarianMinimum(cost) {
  // Rectangular Hungarian assignment for rows <= columns. This variant is
  // O(rows * columns^2), easily small enough for KITTI objects/top-50 boxes.
  const rows = cost.length;
  const columns = rows ? cost[0].length : 0;
  const u = Array(rows + 1).fill(0), v = Array(columns + 1).fill(0);
  const p = Array(columns + 1).fill(0), way = Array(columns + 1).fill(0);
  for (let row = 1; row <= rows; row += 1) {
    p[0] = row;
    let column0 = 0;
    const minValue = Array(columns + 1).fill(Infinity);
    const used = Array(columns + 1).fill(false);
    do {
      used[column0] = true;
      const row0 = p[column0];
      let delta = Infinity, column1 = 0;
      for (let column = 1; column <= columns; column += 1) {
        if (used[column]) continue;
        const current = cost[row0 - 1][column - 1] - u[row0] - v[column];
        if (current < minValue[column]) {
          minValue[column] = current;
          way[column] = column0;
        }
        if (minValue[column] < delta) {
          delta = minValue[column];
          column1 = column;
        }
      }
      for (let column = 0; column <= columns; column += 1) {
        if (used[column]) {
          u[p[column]] += delta;
          v[column] -= delta;
        } else {
          minValue[column] -= delta;
        }
      }
      column0 = column1;
    } while (p[column0] !== 0);
    do {
      const column1 = way[column0];
      p[column0] = p[column1];
      column0 = column1;
    } while (column0 !== 0);
  }
  const assignment = Array(rows).fill(-1);
  for (let column = 1; column <= columns; column += 1) {
    if (p[column] > 0) assignment[p[column] - 1] = column - 1;
  }
  return assignment;
}

function computeMatches() {
  state.matches = {};
  if (!state.frame) return;
  for (const model of state.config.models) {
    const predictions = visiblePredictions(model.id);
    const groundTruth = state.frame.ground_truth;
    // One dummy column per GT lets the optimum leave an object unmatched.
    const cost = groundTruth.map((gt) => [
      ...predictions.map((pred) => {
        if (canonicalClass(gt.class) !== canonicalClass(pred.class)) return 1.0;
        const iou = boxIou2d(gt, pred);
        return iou >= 0.5 ? 1.0 - iou : 1.0;
      }),
      ...Array(groundTruth.length).fill(0.75),
    ]);
    const accepted = [];
    for (const [gtIndex, predictionIndex] of hungarianMinimum(cost).entries()) {
      if (predictionIndex < 0 || predictionIndex >= predictions.length) continue;
      const gt = groundTruth[gtIndex], pred = predictions[predictionIndex];
      if (canonicalClass(gt.class) !== canonicalClass(pred.class)) continue;
      const iou = boxIou2d(gt, pred);
      if (iou >= 0.5) accepted.push({ gt, pred, iou });
    }
    state.matches[model.id] = accepted;
  }
}

function isHighlighted(type, modelId, objectId) {
  if (!state.selected) return false;
  const key = objectKey(type, modelId, objectId);
  if (state.selected === key) return true;
  for (const [matchModel, pairs] of Object.entries(state.matches)) {
    for (const pair of pairs) {
      const gtKey = objectKey("gt", null, pair.gt.id);
      const predKey = objectKey("pred", matchModel, pair.pred.id);
      if ((state.selected === gtKey && key === predKey) || (state.selected === predKey && key === gtKey)) return true;
    }
  }
  return false;
}

function drawBox2d(ctx, corners, transform, color, width, alpha = 1) {
  if (!corners || corners.some((point) => point === null)) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  for (const [start, end] of BOX_EDGES) {
    const a = transform(corners[start]);
    const b = transform(corners[end]);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  ctx.stroke();
  ctx.restore();
}

function drawCameraLabel(ctx, text, x, y, color, ratio) {
  ctx.save();
  ctx.font = `${9 * ratio}px ui-monospace, monospace`;
  const padding = 3 * ratio;
  const width = ctx.measureText(text).width + 2 * padding;
  const height = 14 * ratio;
  ctx.fillStyle = "rgba(4,9,13,.82)";
  ctx.fillRect(x, y - height, width, height);
  ctx.fillStyle = color;
  ctx.fillText(text, x + padding, y - 4 * ratio);
  ctx.restore();
}

function drawCamera() {
  const size = resizeCanvas(cameraCanvas);
  const ctx = cameraContext;
  ctx.clearRect(0, 0, size.width, size.height);
  if (!state.frame || !state.image) return;
  const [imageWidth, imageHeight] = state.frame.image_size;
  const scale = Math.min(size.width / imageWidth, size.height / imageHeight);
  const offsetX = (size.width - imageWidth * scale) / 2;
  const offsetY = (size.height - imageHeight * scale) / 2;
  cameraTransform = { scale, offsetX, offsetY, imageWidth, imageHeight };
  const map = ([x, y]) => [offsetX + x * scale, offsetY + y * scale];

  ctx.drawImage(state.image, offsetX, offsetY, imageWidth * scale, imageHeight * scale);
  ctx.save();
  ctx.beginPath();
  ctx.rect(offsetX, offsetY, imageWidth * scale, imageHeight * scale);
  ctx.clip();

  if (state.toggles.horizon && state.frame.pseudo_labels?.horizon_line?.visible_segment) {
    const [left, right] = state.frame.pseudo_labels.horizon_line.visible_segment.map(map);
    ctx.strokeStyle = "#ff4fc3";
    ctx.lineWidth = 2.2 * size.ratio;
    ctx.setLineDash([9 * size.ratio, 5 * size.ratio]);
    ctx.beginPath(); ctx.moveTo(...left); ctx.lineTo(...right); ctx.stroke();
    ctx.setLineDash([]);
  }
  if (state.toggles.horizon) {
    for (const model of selectedModels()) {
      const horizon = state.frame.inference_artifacts?.[model.id]?.horizon;
      const rawSegment = horizon?.raw_visible_segment_raw;
      if (rawSegment) {
        const [rawLeft, rawRight] = rawSegment.map(map);
        ctx.save();
        ctx.globalAlpha = 0.72;
        ctx.strokeStyle = model.color; ctx.lineWidth = 1.2 * size.ratio;
        ctx.setLineDash([1 * size.ratio, 5 * size.ratio]);
        ctx.beginPath(); ctx.moveTo(...rawLeft); ctx.lineTo(...rawRight); ctx.stroke();
        ctx.restore();
      }
      const segment = horizon?.visible_segment_raw;
      if (segment && horizon?.vertical_edge_refinement?.valid) {
        const [left, right] = segment.map(map);
        ctx.save();
        ctx.strokeStyle = model.color; ctx.lineWidth = 2.1 * size.ratio;
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(...left); ctx.lineTo(...right); ctx.stroke();
        ctx.restore();
      }
    }
  }

  if (state.toggles.gt) {
    for (const gt of state.frame.ground_truth) {
      const highlighted = isHighlighted("gt", null, gt.id);
      drawBox2d(ctx, gt.corners_2d, map, "#53df9b", (highlighted ? 4 : 1.7) * size.ratio, highlighted ? 1 : 0.82);
    }
  }
  for (const model of selectedModels()) {
    for (const pred of visiblePredictions(model.id)) {
      const highlighted = isHighlighted("pred", model.id, pred.id);
      drawBox2d(ctx, pred.corners_2d, map, model.color, (highlighted ? 4 : 1.8) * size.ratio, highlighted ? 1 : 0.88);
      const pair = (state.matches[model.id] || []).find((item) => item.pred.id === pred.id);
      const anchor = map([pred.bbox_2d[0], pred.bbox_2d[1]]);
      const predictedZ = pred.location_3d?.[2];
      const gtText = pair ? ` · Zgt ${pair.gt.location_3d[2].toFixed(2)}` : " · Zgt —";
      drawCameraLabel(
        ctx, `${model.label} · Zpred ${predictedZ.toFixed(2)}${gtText}`,
        anchor[0], anchor[1] - 2 * size.ratio, model.color, size.ratio,
      );
    }
  }

  if ((state.toggles.touch || state.toggles.near2d) && state.frame.pseudo_labels) {
    for (const object of state.frame.pseudo_labels.objects || []) {
      if (state.toggles.touch) {
        for (const point of object.touch_points || []) {
          if (!point.projection_valid || !point.pixel_2d) continue;
          const [x, y] = map(point.pixel_2d);
          const highlighted = isHighlighted("gt", null, object.object_id);
          ctx.beginPath();
          ctx.fillStyle = TOUCH_COLORS[point.name] || "#fff";
          ctx.strokeStyle = highlighted ? "white" : "#081017";
          ctx.lineWidth = (highlighted ? 3 : 1.5) * size.ratio;
          ctx.arc(x, y, (highlighted ? 6 : 4.5) * size.ratio, 0, Math.PI * 2);
          ctx.fill(); ctx.stroke();
        }
      }
      const near = object.near_contact;
      if (state.toggles.near2d && near?.projection_valid && near.pixel_2d) {
        const [x, y] = map(near.pixel_2d);
        const highlighted = isHighlighted("gt", null, object.object_id);
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(Math.PI / 4);
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = highlighted ? "#ff4fc3" : "#071018";
        ctx.lineWidth = (highlighted ? 2.5 : 1.3) * size.ratio;
        const radius = (highlighted ? 6 : 4.5) * size.ratio;
        ctx.fillRect(-radius, -radius, 2 * radius, 2 * radius);
        ctx.strokeRect(-radius, -radius, 2 * radius, 2 * radius);
        ctx.restore();
        drawCameraLabel(ctx, "near GT", x + 5 * size.ratio, y - 5 * size.ratio, "#ffffff", size.ratio);
      }
    }
  }
  if (state.toggles.touch || state.toggles.near2d) {
    for (const model of selectedModels()) {
      for (const pred of visiblePredictions(model.id)) {
        const points = pred.e1?.touchpoints_raw || [];
        const valid = pred.e1?.touchpoint_ray_valid || [];
        const isNearContact = points.length === 1;
        if ((isNearContact && !state.toggles.near2d) || (!isNearContact && !state.toggles.touch)) continue;
        points.forEach((point, index) => {
          const [x, y] = map(point);
          ctx.beginPath();
          ctx.fillStyle = valid[index] ? model.color : "rgba(160,170,178,.55)";
          ctx.strokeStyle = "#071018"; ctx.lineWidth = 1.2 * size.ratio;
          ctx.arc(x, y, 3.5 * size.ratio, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        });
        if (points.length && pred.e1?.selected_depth_m !== undefined && isHighlighted("pred", model.id, pred.id)) {
          const anchor = map(points[0]);
          const geo = pred.e1.geometry_valid ? ` · geo ${pred.e1.geometric_depth_m.toFixed(1)}m` : " · geo invalid";
          const touchText = points.length === 1 ? "near touch" : `${points.length} touches`;
          drawCameraLabel(
            ctx, `${touchText} · Zfinal ${pred.e1.selected_depth_m.toFixed(1)}m${geo}`,
            anchor[0] + 5 * size.ratio, anchor[1] - 6 * size.ratio,
            model.color, size.ratio,
          );
        }
      }
    }
  }
  ctx.restore();
}

function formatMetric(value) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—" : Number(value).toFixed(3);
}

function lineFromSegment(segment) {
  if (!segment || segment.length !== 2) return null;
  const deltaX = segment[1][0] - segment[0][0];
  if (Math.abs(deltaX) < 1e-9) return null;
  const slope = (segment[1][1] - segment[0][1]) / deltaX;
  return [slope, segment[0][1] - slope * segment[0][0]];
}

function renderValidationSummary() {
  const container = $("#validation-table");
  const reports = state.config?.validation_reports || [];
  if (!reports.length) {
    container.textContent = "No validation report yet. Run tools/evaluate_e1_depth.py.";
    return;
  }
  const bucket = $("#validation-distance").value;
  const rows = [];
  for (const item of reports) {
    if (item.error) continue;
    const report = item.report || {};
    const values = bucket === "overall" ? report.models : null;
    if (values) {
      for (const [name, metric] of Object.entries(values)) rows.push({ name, metric });
    } else {
      for (const metric of report.distance_buckets || []) {
        if (metric.distance_bucket_m === bucket) rows.push({ name: metric.model, metric });
      }
    }
  }
  if (!rows.length) {
    container.textContent = `No metrics for ${bucket}.`;
    return;
  }
  container.innerHTML = `<table><thead><tr><th>Model</th><th>N</th><th>MAE</th><th>RMSE</th><th>Recall</th><th>AP3D-M</th></tr></thead><tbody>${rows.map(({name, metric}) =>
    `<tr><td>${name}</td><td>${metric.matched_count ?? "—"}</td><td>${formatMetric(metric.mae)}</td><td>${formatMetric(metric.rmse)}</td><td>${formatMetric(metric.recall)}</td><td>${formatMetric(metric.ap3d_r40_moderate)}</td></tr>`
  ).join("")}</tbody></table>`;
}

function setupBevTransform(size) {
  const baseScale = Math.min(size.width / 64, size.height / 82);
  const scale = baseScale * state.bev.zoom;
  const centerX = size.width / 2;
  const centerY = size.height / 2;
  bevTransform = {
    scale,
    ratio: size.ratio,
    worldToCanvas: ([x, z]) => [centerX + (x - state.bev.centerX) * scale, centerY - (z - state.bev.centerZ) * scale],
    canvasToWorld: ([x, y]) => [state.bev.centerX + (x - centerX) / scale, state.bev.centerZ - (y - centerY) / scale],
  };
}

function footprint(object) { return object.corners_camera.slice(0, 4).map((corner) => [corner[0], corner[2]]); }

function drawPolygon(ctx, points, color, width, highlighted, fill = true) {
  if (!points.length) return;
  const mapped = points.map(bevTransform.worldToCanvas);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = (highlighted ? 4 : width) * bevTransform.ratio;
  ctx.fillStyle = `${color}${highlighted ? "35" : "17"}`;
  ctx.beginPath();
  ctx.moveTo(...mapped[0]);
  mapped.slice(1).forEach((point) => ctx.lineTo(...point));
  ctx.closePath();
  if (fill) ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawArrow(ctx, from, to, color) {
  const a = bevTransform.worldToCanvas(from);
  const b = bevTransform.worldToCanvas(to);
  const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
  const head = 8 * bevTransform.ratio;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1.5 * bevTransform.ratio;
  ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(...b);
  ctx.lineTo(b[0] - head * Math.cos(angle - Math.PI / 6), b[1] - head * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(b[0] - head * Math.cos(angle + Math.PI / 6), b[1] - head * Math.sin(angle + Math.PI / 6));
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawReprojectedTouchPoints(ctx) {
  const objects = state.frame?.pseudo_labels?.objects || [];
  for (const object of objects) {
    const byName = new Map((object.touch_points || []).map((point) => [point.name, point]));
    const ordered = TOUCH_POINT_ORDER.map((name) => byName.get(name));
    const valid = ordered.filter((point) => point?.ground_reprojection_valid && point.reprojected_ground_3d);
    const highlighted = isHighlighted("gt", null, object.object_id);

    ctx.save();
    ctx.setLineDash([5 * bevTransform.ratio, 4 * bevTransform.ratio]);
    ctx.strokeStyle = highlighted ? "#ffffff" : "rgba(255, 184, 77, .55)";
    ctx.lineWidth = (highlighted ? 2.3 : 1.1) * bevTransform.ratio;
    for (const point of valid) {
      if (!point.camera_3d) continue;
      const from = bevTransform.worldToCanvas([point.camera_3d[0], point.camera_3d[2]]);
      const to = bevTransform.worldToCanvas([
        point.reprojected_ground_3d[0], point.reprojected_ground_3d[2],
      ]);
      ctx.beginPath(); ctx.moveTo(...from); ctx.lineTo(...to); ctx.stroke();
    }
    ctx.restore();

    if (valid.length === TOUCH_POINT_ORDER.length) {
      const footprint = ordered.map((point) => [
        point.reprojected_ground_3d[0], point.reprojected_ground_3d[2],
      ]);
      const mapped = footprint.map(bevTransform.worldToCanvas);
      ctx.save();
      ctx.strokeStyle = highlighted ? "#ffffff" : "#ffb84d";
      ctx.fillStyle = highlighted ? "rgba(255,255,255,.10)" : "rgba(255,184,77,.08)";
      ctx.lineWidth = (highlighted ? 3.2 : 2) * bevTransform.ratio;
      ctx.beginPath();
      ctx.moveTo(...mapped[0]);
      mapped.slice(1).forEach((point) => ctx.lineTo(...point));
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.restore();
    }

    for (const point of valid) {
      const position = bevTransform.worldToCanvas([
        point.reprojected_ground_3d[0], point.reprojected_ground_3d[2],
      ]);
      const radius = (highlighted ? 5.5 : 4.2) * bevTransform.ratio;
      ctx.save();
      ctx.fillStyle = TOUCH_COLORS[point.name] || "#ffffff";
      ctx.strokeStyle = highlighted ? "#ffffff" : "#071018";
      ctx.lineWidth = (highlighted ? 2.2 : 1.3) * bevTransform.ratio;
      ctx.beginPath(); ctx.arc(position[0], position[1], radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#dbe7ed";
      ctx.font = `${9 * bevTransform.ratio}px ui-monospace, monospace`;
      ctx.fillText(point.name, position[0] + 6 * bevTransform.ratio, position[1] - 6 * bevTransform.ratio);
      ctx.restore();
    }
  }
}

function validCameraPoint(point) {
  return Array.isArray(point) && point.length === 3 && point.every((value) => Number.isFinite(Number(value)));
}

function drawPredictedNearTouchpoints3d(ctx) {
  for (const model of selectedModels()) {
    for (const pred of visiblePredictions(model.id)) {
      const e1 = pred.e1 || {};
      const near = e1.near_touchpoint_3d_camera;
      const rayValid = e1.touchpoint_ray_valid?.[0] ?? e1.geometry_valid;
      if (!rayValid || !validCameraPoint(near)) continue;

      const position = bevTransform.worldToCanvas([Number(near[0]), Number(near[2])]);
      const highlighted = isHighlighted("pred", model.id, pred.id);

      if (validCameraPoint(pred.location_3d)) {
        const center = bevTransform.worldToCanvas([
          Number(pred.location_3d[0]), Number(pred.location_3d[2]),
        ]);
        ctx.save();
        ctx.strokeStyle = model.color;
        ctx.globalAlpha = highlighted ? 0.9 : 0.42;
        ctx.lineWidth = (highlighted ? 2 : 1) * bevTransform.ratio;
        ctx.setLineDash([4 * bevTransform.ratio, 4 * bevTransform.ratio]);
        ctx.beginPath(); ctx.moveTo(...center); ctx.lineTo(...position); ctx.stroke();
        ctx.restore();
      }

      const radius = (highlighted ? 6 : 4.5) * bevTransform.ratio;
      ctx.save();
      ctx.translate(position[0], position[1]);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = model.color;
      ctx.strokeStyle = highlighted ? "#ffffff" : "#071018";
      ctx.lineWidth = (highlighted ? 2.4 : 1.3) * bevTransform.ratio;
      ctx.fillRect(-radius, -radius, 2 * radius, 2 * radius);
      ctx.strokeRect(-radius, -radius, 2 * radius, 2 * radius);
      ctx.restore();

      if (highlighted) {
        ctx.save();
        ctx.fillStyle = "#e8f4fa";
        ctx.font = `${10 * bevTransform.ratio}px ui-monospace, monospace`;
        ctx.fillText(
          `near3D (${near.map((value) => Number(value).toFixed(2)).join(", ")})m`,
          position[0] + 8 * bevTransform.ratio,
          position[1] - 8 * bevTransform.ratio,
        );
        ctx.restore();
      }
    }
  }
}

function drawBev() {
  const size = resizeCanvas(bevCanvas);
  const ctx = bevContext;
  setupBevTransform(size);
  ctx.clearRect(0, 0, size.width, size.height);
  ctx.fillStyle = "#050b10"; ctx.fillRect(0, 0, size.width, size.height);

  const origin = bevTransform.worldToCanvas([0, 0]);
  ctx.save();
  ctx.strokeStyle = "rgba(92, 145, 169, .28)";
  ctx.fillStyle = "#718895";
  ctx.font = `${10 * size.ratio}px ui-monospace, monospace`;
  for (let radius = 10; radius <= 80; radius += 10) {
    ctx.beginPath(); ctx.arc(origin[0], origin[1], radius * bevTransform.scale, 0, Math.PI * 2); ctx.stroke();
    const label = bevTransform.worldToCanvas([0, radius]);
    ctx.fillText(`${radius}m`, label[0] + 4 * size.ratio, label[1] - 4 * size.ratio);
  }
  ctx.strokeStyle = "rgba(92, 145, 169, .18)";
  ctx.beginPath(); ctx.moveTo(0, origin[1]); ctx.lineTo(size.width, origin[1]); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(origin[0], 0); ctx.lineTo(origin[0], size.height); ctx.stroke();
  ctx.restore();

  if (state.frame) {
    if (state.toggles.arrows) {
      for (const model of selectedModels()) {
        for (const pair of state.matches[model.id] || []) {
          drawArrow(ctx, [pair.gt.location_3d[0], pair.gt.location_3d[2]], [pair.pred.location_3d[0], pair.pred.location_3d[2]], model.color);
        }
      }
    }
    if (state.toggles.gt) {
      for (const gt of state.frame.ground_truth) {
        drawPolygon(ctx, footprint(gt), "#53df9b", 1.7, isHighlighted("gt", null, gt.id));
      }
    }
    for (const model of selectedModels()) {
      for (const pred of visiblePredictions(model.id)) {
        drawPolygon(ctx, footprint(pred), model.color, 1.8, isHighlighted("pred", model.id, pred.id));
      }
    }
    if (state.toggles.reproject) drawReprojectedTouchPoints(ctx);
    if (state.toggles.near3d) drawPredictedNearTouchpoints3d(ctx);
  }

  ctx.save();
  ctx.translate(origin[0], origin[1]);
  ctx.fillStyle = "#e7f6ff";
  ctx.strokeStyle = "#4bd8ff";
  ctx.lineWidth = 1.4 * size.ratio;
  ctx.beginPath();
  ctx.moveTo(0, -10 * size.ratio); ctx.lineTo(7 * size.ratio, 8 * size.ratio); ctx.lineTo(-7 * size.ratio, 8 * size.ratio); ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.restore();

  drawBevTooltip(ctx, size);
}

function matchingPairForKey(key) {
  for (const [modelId, pairs] of Object.entries(state.matches)) {
    for (const pair of pairs) {
      if (key === objectKey("gt", null, pair.gt.id) || key === objectKey("pred", modelId, pair.pred.id)) return { ...pair, modelId };
    }
  }
  return null;
}

function drawBevTooltip(ctx, size) {
  const key = state.hovered || state.selected;
  if (!key) return;
  const pair = matchingPairForKey(key);
  if (!pair) return;
  const deltaX = pair.pred.location_3d[0] - pair.gt.location_3d[0];
  const deltaZ = pair.pred.location_3d[2] - pair.gt.location_3d[2];
  const model = state.config.models.find((item) => item.id === pair.modelId);
  const touch = pair.pred.e1?.touchpoints_raw?.[0];
  const geometryDepth = pair.pred.e1?.geometric_depth_m;
  const near3d = pair.pred.e1?.near_touchpoint_3d_camera;
  const lines = [
    model?.label || pair.modelId,
    `Zpred ${pair.pred.location_3d[2].toFixed(2)} m · Zgt ${pair.gt.location_3d[2].toFixed(2)} m`,
    `ΔX ${deltaX >= 0 ? "+" : ""}${deltaX.toFixed(2)} m · ΔZ ${deltaZ >= 0 ? "+" : ""}${deltaZ.toFixed(2)} m`,
  ];
  if (touch) lines.push(`touch (${touch[0].toFixed(1)}, ${touch[1].toFixed(1)}) px`);
  if (validCameraPoint(near3d)) {
    lines.push(`near3D (${near3d.map((value) => Number(value).toFixed(2)).join(", ")}) m`);
  }
  if (Number.isFinite(geometryDepth)) lines.push(`Zgeo ${geometryDepth.toFixed(2)} m`);
  const x = 14 * size.ratio, y = 14 * size.ratio, width = 256 * size.ratio;
  const height = (16 + lines.length * 17) * size.ratio;
  ctx.save();
  ctx.fillStyle = "rgba(8, 15, 21, .92)"; ctx.strokeStyle = model?.color || "#fff"; ctx.lineWidth = size.ratio;
  ctx.beginPath(); ctx.roundRect(x, y, width, height, 6 * size.ratio); ctx.fill(); ctx.stroke();
  ctx.font = `${11 * size.ratio}px ui-monospace, monospace`;
  lines.forEach((line, index) => { ctx.fillStyle = index ? "#dce7ed" : (model?.color || "#fff"); ctx.fillText(line, x + 10 * size.ratio, y + (18 + index * 17) * size.ratio); });
  ctx.restore();
}

function drawAll() { drawCamera(); drawBev(); updateReadouts(); updateInferenceReadout(); }

function updateDatasetReprojectionMetrics() {
  const output = $("#dataset-reprojection-mae");
  const metrics = state.config?.reprojection_metrics;
  if (!metrics || metrics.mae_z === null) {
    output.textContent = "Dataset |ΔZ| MAE —";
    output.title = "No valid ground reprojections in this split";
    return;
  }
  output.textContent = `Dataset |ΔZ| MAE ${metrics.mae_z.toFixed(3)}m`;
  output.title = [
    `Whole ${state.config.split} split`,
    `MAE X ${metrics.mae_x.toFixed(3)} m`,
    `MAE Y ${metrics.mae_y.toFixed(3)} m`,
    `MAE Z ${metrics.mae_z.toFixed(3)} m`,
    `Mean coordinate MAE ${metrics.mean_coordinate_mae.toFixed(3)} m`,
    `Mean Euclidean error ${metrics.mean_euclidean_error.toFixed(3)} m`,
    `${metrics.point_count} valid points across ${metrics.valid_reprojection_frame_count} frames`,
  ].join("\n");
}

function updateReadouts() {
  if (!state.frame) return;
  const pseudo = state.frame.pseudo_labels;
  const vehicles = pseudo?.objects?.length || 0;
  const invalidReason = pseudo?.ground_plane?.reason?.replaceAll("_", " ") || "no pseudo-label";
  const plane = pseudo?.ground_plane?.valid
    ? `plane RMS ${pseudo.ground_plane.rms_error.toFixed(3)}m`
    : `plane invalid · ${invalidReason}`;
  $("#camera-readout").textContent = `${state.frame.frame_id} · ${vehicles} vehicles · ${plane}`;
  const pair = matchingPairForKey(state.selected);
  const reprojectedBoxes = (pseudo?.objects || []).filter((object) =>
    (object.touch_points || []).filter((point) => point.ground_reprojection_valid).length === TOUCH_POINT_ORDER.length
  ).length;
  if (pair) {
    $("#bev-readout").textContent = `ΔX ${(pair.pred.location_3d[0] - pair.gt.location_3d[0]).toFixed(2)}m · ΔZ ${(pair.pred.location_3d[2] - pair.gt.location_3d[2]).toFixed(2)}m`;
  } else if (state.toggles.reproject) {
    const datasetMae = state.config.reprojection_metrics?.mae_z;
    const maeText = datasetMae === null || datasetMae === undefined
      ? ""
      : ` · dataset |ΔZ| MAE ${datasetMae.toFixed(3)}m`;
    $("#bev-readout").textContent = pseudo?.ground_plane?.valid
      ? `${reprojectedBoxes} reprojected touch box${reprojectedBoxes === 1 ? "" : "es"}${maeText}`
      : "Reprojection unavailable · invalid ground plane";
  } else {
    $("#bev-readout").textContent = "Drag to pan · Wheel to zoom";
  }
}

function updateInferenceReadout() {
  const output = $("#model-inference-readout");
  if (!state.config || !state.frame) return;
  const lines = [];
  const gtHorizon = state.frame.pseudo_labels?.horizon_line;
  if (gtHorizon?.valid) {
    lines.push(`Hgt(raw) k=${Number(gtHorizon.slope).toFixed(4)} b=${Number(gtHorizon.intercept).toFixed(1)}`);
  } else {
    lines.push("Hgt(raw) unavailable");
  }
  for (const model of selectedModels()) {
    const live = state.liveInference[model.id];
    const artifact = state.frame.inference_artifacts?.[model.id];
    const horizon = artifact?.horizon;
    const selectedWeight = state.selectedWeights.get(model.id);
    const weight = model.weights?.find((item) => item.id === selectedWeight);
    const source = live
      ? `${live.weight_label}${live.checkpoint_epoch === null || live.checkpoint_epoch === undefined ? "" : ` · ep ${live.checkpoint_epoch}`} · ${live.latency_ms.toFixed(0)} ms`
      : (artifact
        ? `precomputed output${weight ? ` · ${weight.label} not run` : ""}`
        : (weight ? `${weight.label} · not run` : "no output"));
    lines.push(`${model.label} · ${source}`);
    const raw = lineFromSegment(horizon?.raw_visible_segment_raw);
    const refined = lineFromSegment(horizon?.visible_segment_raw) || raw;
    if (raw) {
      const edge = horizon.vertical_edge_refinement || {};
      lines.push(
        `  Hraw(raw) k=${raw[0].toFixed(4)} b=${raw[1].toFixed(1)} · ` +
        `Href(raw) k=${refined[0].toFixed(4)} b=${refined[1].toFixed(1)} · ` +
        (edge.valid ? `edge ${edge.edge_count}` : "edge invalid"),
      );
    } else {
      lines.push("  horizon unavailable");
    }
  }
  output.textContent = lines.length ? lines.join("\n") : "Select a model/checkpoint, then press Run.";
}

function candidateAtCamera(event) {
  if (!cameraTransform || !state.frame) return null;
  const rect = cameraCanvas.getBoundingClientRect();
  const ratioX = cameraCanvas.width / rect.width, ratioY = cameraCanvas.height / rect.height;
  const x = (event.clientX - rect.left) * ratioX, y = (event.clientY - rect.top) * ratioY;
  const candidates = [];
  const add = (type, modelId, object) => {
    const points = (object.corners_2d || []).filter(Boolean).map(([u,v]) => [cameraTransform.offsetX + u * cameraTransform.scale, cameraTransform.offsetY + v * cameraTransform.scale]);
    if (!points.length) return;
    const center = points.reduce((sum, point) => [sum[0] + point[0] / points.length, sum[1] + point[1] / points.length], [0,0]);
    candidates.push({ key: objectKey(type, modelId, object.id), distance: Math.hypot(x - center[0], y - center[1]) });
  };
  if (state.toggles.gt) state.frame.ground_truth.forEach((item) => add("gt", null, item));
  selectedModels().forEach((model) => visiblePredictions(model.id).forEach((item) => add("pred", model.id, item)));
  candidates.sort((a,b) => a.distance - b.distance);
  return candidates[0]?.distance < 70 * (window.devicePixelRatio || 1) ? candidates[0].key : null;
}

function candidateAtBev(event) {
  if (!bevTransform || !state.frame) return null;
  const rect = bevCanvas.getBoundingClientRect();
  const point = [(event.clientX - rect.left) * bevCanvas.width / rect.width, (event.clientY - rect.top) * bevCanvas.height / rect.height];
  const candidates = [];
  const add = (type, modelId, object) => {
    const center = bevTransform.worldToCanvas([object.location_3d[0], object.location_3d[2]]);
    candidates.push({ key: objectKey(type, modelId, object.id), distance: Math.hypot(point[0] - center[0], point[1] - center[1]) });
  };
  if (state.toggles.gt) state.frame.ground_truth.forEach((item) => add("gt", null, item));
  selectedModels().forEach((model) => visiblePredictions(model.id).forEach((item) => add("pred", model.id, item)));
  candidates.sort((a,b) => a.distance - b.distance);
  return candidates[0]?.distance < 18 * bevTransform.ratio ? candidates[0].key : null;
}

async function loadFrame(index) {
  const serial = ++loadSerial;
  state.frameIndex = Math.max(0, Math.min(state.config.frames.length - 1, index));
  const frameId = state.config.frames[state.frameIndex];
  $("#frame-select").value = frameId;
  $("#status").textContent = `Loading ${frameId}…`;
  try {
    const response = await fetch(`/api/frame/${frameId}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    const image = new Image();
    image.src = `${payload.image_url}?v=${serial}`;
    await image.decode();
    if (serial !== loadSerial) return;
    state.frame = payload;
    state.image = image;
    state.liveInference = {};
    state.selected = null;
    state.hovered = null;
    computeMatches();
    const predictionCount = selectedModels().reduce((sum, model) => sum + visiblePredictions(model.id).length, 0);
    $("#status").textContent = payload.pseudo_labels
      ? `${state.config.split} · ${predictionCount} visible predictions`
      : `${state.config.split} · no geometry pseudo-label for this frame`;
    $("#previous-frame").disabled = state.frameIndex === 0;
    $("#next-frame").disabled = state.frameIndex === state.config.frames.length - 1;
    drawAll();
  } catch (error) {
    $("#status").textContent = `Error: ${error.message}`;
  }
}

async function runInference() {
  if (!state.frame) return;
  const selections = selectedModels()
    .filter((model) => model.live_available && state.selectedWeights.get(model.id))
    .map((model) => ({
      model_id: model.id,
      weight_id: state.selectedWeights.get(model.id),
    }));
  if (!selections.length) {
    $("#status").textContent = "Select at least one model with an available checkpoint.";
    return;
  }

  const button = $("#run-inference");
  button.disabled = true;
  button.textContent = "Running…";
  $("#status").textContent = `Running ${selections.length} checkpoint${selections.length === 1 ? "" : "s"} on ${state.frame.frame_id}…`;
  try {
    const response = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_id: state.frame.frame_id,
        // Fetch all top-k detections once. The confidence slider then filters
        // them locally without another forward pass.
        threshold: 0,
        selections,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    if (payload.frame_id !== state.frame.frame_id) return;
    for (const [modelId, result] of Object.entries(payload.models || {})) {
      state.frame.predictions[modelId] = result.predictions || [];
      state.frame.inference_artifacts[modelId] = result.artifact || null;
      state.liveInference[modelId] = result;
    }
    computeMatches();
    drawAll();
    const completed = Object.keys(payload.models || {}).length;
    const errors = Object.entries(payload.errors || {});
    $("#status").textContent = errors.length
      ? `Inference complete: ${completed} succeeded, ${errors.length} failed · ${errors.map(([id, error]) => `${id}: ${error}`).join(" | ")}`
      : `Inference complete · ${completed} checkpoint${completed === 1 ? "" : "s"} · ${state.frame.frame_id}`;
  } catch (error) {
    $("#status").textContent = `Inference error: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Run";
  }
}

function initialiseControls() {
  const frameSelect = $("#frame-select");
  for (const frameId of state.config.frames) frameSelect.add(new Option(frameId, frameId));
  frameSelect.addEventListener("change", () => loadFrame(state.config.frames.indexOf(frameSelect.value)));
  $("#previous-frame").addEventListener("click", () => loadFrame(state.frameIndex - 1));
  $("#next-frame").addEventListener("click", () => loadFrame(state.frameIndex + 1));

  const options = $("#model-options");
  const baselineAvailable = state.config.models.some((model) => model.id === "monodetr" && model.available);
  const firstAvailable = state.config.models.findIndex((model) => model.available);
  state.config.models.forEach((model, modelIndex) => {
    const row = document.createElement("div"); row.className = "model-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = `model-${model.id}`;
    checkbox.disabled = !model.available;
    checkbox.checked = model.available && (
      model.id === "monodetr" || (!baselineAvailable && modelIndex === firstAvailable)
    );
    if (checkbox.checked) state.selectedModels.add(model.id);
    checkbox.addEventListener("change", () => {
      checkbox.checked ? state.selectedModels.add(model.id) : state.selectedModels.delete(model.id);
      $("#model-count").textContent = state.selectedModels.size;
      computeMatches(); drawAll();
    });
    const dot = document.createElement("span"); dot.className = "model-dot"; dot.style.background = model.color;
    const text = document.createElement("label");
    text.htmlFor = checkbox.id;
    text.textContent = model.available ? model.label : `${model.label} (no output/checkpoint)`;
    if (!model.available) text.className = "unavailable";
    const weightSelect = document.createElement("select");
    weightSelect.setAttribute("aria-label", `${model.label} checkpoint`);
    if (model.weights?.length) {
      for (const weight of model.weights) weightSelect.add(new Option(weight.label, weight.id));
      const preferred = model.weights.some((weight) => weight.id === model.default_weight)
        ? model.default_weight : model.weights[0].id;
      weightSelect.value = preferred;
      state.selectedWeights.set(model.id, preferred);
      weightSelect.addEventListener("change", () => {
        state.selectedWeights.set(model.id, weightSelect.value);
        delete state.liveInference[model.id];
        updateInferenceReadout();
      });
    } else {
      weightSelect.add(new Option(model.precomputed_available ? "precomputed only" : "no checkpoint", ""));
      weightSelect.disabled = true;
    }
    row.append(checkbox, dot, text, weightSelect); options.append(row);
  });
  $("#model-count").textContent = state.selectedModels.size;
  $("#run-inference").addEventListener("click", runInference);
  $("#validation-distance").addEventListener("change", renderValidationSummary);
  renderValidationSummary();

  $("#confidence").addEventListener("input", (event) => {
    state.threshold = Number(event.target.value);
    $("#confidence-value").value = state.threshold.toFixed(2);
    computeMatches(); drawAll();
  });
  for (const [id, key] of [["#toggle-gt","gt"],["#toggle-pred","pred"],["#toggle-touch","touch"],["#toggle-near-2d","near2d"],["#toggle-reproject","reproject"],["#toggle-near-3d","near3d"],["#toggle-horizon","horizon"],["#toggle-arrows","arrows"]]) {
    $(id).addEventListener("change", (event) => { state.toggles[key] = event.target.checked; computeMatches(); drawAll(); });
  }
}

cameraCanvas.addEventListener("click", (event) => { state.selected = candidateAtCamera(event); drawAll(); });
bevCanvas.addEventListener("click", (event) => {
  if (!state.bev.moved) state.selected = candidateAtBev(event);
  drawAll();
});
bevCanvas.addEventListener("pointerdown", (event) => {
  state.bev.dragging = true; state.bev.moved = false;
  state.bev.downX = state.bev.lastX = event.clientX; state.bev.downY = state.bev.lastY = event.clientY;
  bevCanvas.setPointerCapture(event.pointerId);
});
bevCanvas.addEventListener("pointermove", (event) => {
  state.hovered = candidateAtBev(event);
  if (state.bev.dragging && bevTransform) {
    if (Math.hypot(event.clientX - state.bev.downX, event.clientY - state.bev.downY) >= 4) state.bev.moved = true;
    const dx = event.clientX - state.bev.lastX, dy = event.clientY - state.bev.lastY;
    const cssScale = bevTransform.scale / (bevCanvas.width / bevCanvas.getBoundingClientRect().width);
    state.bev.centerX -= dx / cssScale; state.bev.centerZ += dy / cssScale;
    state.bev.lastX = event.clientX; state.bev.lastY = event.clientY;
  }
  drawBev();
});
bevCanvas.addEventListener("pointerup", (event) => { state.bev.dragging = false; bevCanvas.releasePointerCapture(event.pointerId); });
bevCanvas.addEventListener("pointerleave", () => { state.hovered = null; if (!state.bev.dragging) drawBev(); });
bevCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if (!bevTransform) return;
  const rect = bevCanvas.getBoundingClientRect();
  const canvasPoint = [(event.clientX - rect.left) * bevCanvas.width / rect.width, (event.clientY - rect.top) * bevCanvas.height / rect.height];
  const before = bevTransform.canvasToWorld(canvasPoint);
  state.bev.zoom = Math.max(0.45, Math.min(8, state.bev.zoom * Math.exp(-event.deltaY * 0.001)));
  setupBevTransform({ width: bevCanvas.width, height: bevCanvas.height, ratio: bevCanvas.width / rect.width });
  const after = bevTransform.canvasToWorld(canvasPoint);
  state.bev.centerX += before[0] - after[0]; state.bev.centerZ += before[1] - after[1];
  drawBev();
}, { passive: false });

window.addEventListener("resize", drawAll);
window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") loadFrame(state.frameIndex - 1);
  if (event.key === "ArrowRight") loadFrame(state.frameIndex + 1);
});

async function start() {
  try {
    const response = await fetch("/api/config");
    state.config = await response.json();
    if (!response.ok) throw new Error(state.config.error || `HTTP ${response.status}`);
    updateDatasetReprojectionMetrics();
    initialiseControls();
    if (!state.config.frames.length) throw new Error("Split has no frames");
    await loadFrame(0);
  } catch (error) {
    $("#status").textContent = `Startup error: ${error.message}`;
  }
}

start();
