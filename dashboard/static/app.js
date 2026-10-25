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
  threshold: 0.2,
  toggles: { gt: true, pred: true, touch: true, reproject: false, horizon: true, arrows: true },
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

function distance2d(left, right) {
  const dx = left.location_3d[0] - right.location_3d[0];
  const dz = left.location_3d[2] - right.location_3d[2];
  return Math.hypot(dx, dz);
}

function computeMatches() {
  state.matches = {};
  if (!state.frame) return;
  for (const model of state.config.models) {
    const predictions = visiblePredictions(model.id);
    const pairs = [];
    state.frame.ground_truth.forEach((gt) => predictions.forEach((pred) => {
      if (canonicalClass(gt.class) !== canonicalClass(pred.class)) return;
      const distance = distance2d(gt, pred);
      if (distance <= 8) pairs.push({ gt, pred, distance });
    }));
    pairs.sort((left, right) => left.distance - right.distance);
    const usedGt = new Set();
    const usedPred = new Set();
    const accepted = [];
    for (const pair of pairs) {
      if (usedGt.has(pair.gt.id) || usedPred.has(pair.pred.id)) continue;
      usedGt.add(pair.gt.id);
      usedPred.add(pair.pred.id);
      accepted.push(pair);
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
      const segment = state.frame.inference_artifacts?.[model.id]?.horizon?.visible_segment_raw;
      if (!segment) continue;
      const [left, right] = segment.map(map);
      ctx.save();
      ctx.strokeStyle = model.color; ctx.lineWidth = 1.8 * size.ratio;
      ctx.setLineDash([3 * size.ratio, 3 * size.ratio]);
      ctx.beginPath(); ctx.moveTo(...left); ctx.lineTo(...right); ctx.stroke();
      ctx.restore();
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
    }
  }

  if (state.toggles.touch && state.frame.pseudo_labels) {
    for (const object of state.frame.pseudo_labels.objects || []) {
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
  }
  if (state.toggles.touch) {
    for (const model of selectedModels()) {
      for (const pred of visiblePredictions(model.id)) {
        const points = pred.e1?.touchpoints_raw || [];
        const valid = pred.e1?.touchpoint_ray_valid || [];
        points.forEach((point, index) => {
          const [x, y] = map(point);
          ctx.beginPath();
          ctx.fillStyle = valid[index] ? model.color : "rgba(160,170,178,.55)";
          ctx.strokeStyle = "#071018"; ctx.lineWidth = 1.2 * size.ratio;
          ctx.arc(x, y, 3.5 * size.ratio, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        });
        if (points.length && pred.e1?.selected_depth_m !== undefined) {
          const anchor = map(points[0]);
          const geo = pred.e1.geometry_valid ? ` · geo ${pred.e1.geometric_depth_m.toFixed(1)}m` : " · geo invalid";
          ctx.fillStyle = model.color;
          ctx.font = `${9 * size.ratio}px ui-monospace, monospace`;
          ctx.fillText(`P1 ${pred.e1.pass1_depth_m.toFixed(1)}m · P2 ${pred.e1.selected_depth_m.toFixed(1)}m${geo}`,
            anchor[0] + 5 * size.ratio, anchor[1] - 6 * size.ratio);
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
  const lines = [model?.label || pair.modelId, `ΔX ${deltaX >= 0 ? "+" : ""}${deltaX.toFixed(2)} m`, `ΔZ ${deltaZ >= 0 ? "+" : ""}${deltaZ.toFixed(2)} m`];
  const x = 14 * size.ratio, y = 14 * size.ratio, width = 142 * size.ratio, height = 66 * size.ratio;
  ctx.save();
  ctx.fillStyle = "rgba(8, 15, 21, .92)"; ctx.strokeStyle = model?.color || "#fff"; ctx.lineWidth = size.ratio;
  ctx.beginPath(); ctx.roundRect(x, y, width, height, 6 * size.ratio); ctx.fill(); ctx.stroke();
  ctx.font = `${11 * size.ratio}px ui-monospace, monospace`;
  lines.forEach((line, index) => { ctx.fillStyle = index ? "#dce7ed" : (model?.color || "#fff"); ctx.fillText(line, x + 10 * size.ratio, y + (18 + index * 17) * size.ratio); });
  ctx.restore();
}

function drawAll() { drawCamera(); drawBev(); updateReadouts(); }

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

function initialiseControls() {
  const frameSelect = $("#frame-select");
  for (const frameId of state.config.frames) frameSelect.add(new Option(frameId, frameId));
  frameSelect.addEventListener("change", () => loadFrame(state.config.frames.indexOf(frameSelect.value)));
  $("#previous-frame").addEventListener("click", () => loadFrame(state.frameIndex - 1));
  $("#next-frame").addEventListener("click", () => loadFrame(state.frameIndex + 1));

  const options = $("#model-options");
  for (const model of state.config.models) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox"; checkbox.checked = model.available; checkbox.disabled = !model.available;
    if (model.available) state.selectedModels.add(model.id);
    checkbox.addEventListener("change", () => {
      checkbox.checked ? state.selectedModels.add(model.id) : state.selectedModels.delete(model.id);
      $("#model-count").textContent = state.selectedModels.size;
      computeMatches(); drawAll();
    });
    const dot = document.createElement("span"); dot.className = "model-dot"; dot.style.background = model.color;
    const text = document.createElement("span"); text.textContent = model.available ? model.label : `${model.label} (no files)`;
    if (!model.available) text.className = "unavailable";
    label.append(checkbox, dot, text); options.append(label);
  }
  $("#model-count").textContent = state.selectedModels.size;
  $("#validation-distance").addEventListener("change", renderValidationSummary);
  renderValidationSummary();

  $("#confidence").addEventListener("input", (event) => {
    state.threshold = Number(event.target.value);
    $("#confidence-value").value = state.threshold.toFixed(2);
    computeMatches(); drawAll();
  });
  for (const [id, key] of [["#toggle-gt","gt"],["#toggle-pred","pred"],["#toggle-touch","touch"],["#toggle-reproject","reproject"],["#toggle-horizon","horizon"],["#toggle-arrows","arrows"]]) {
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
