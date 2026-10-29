"""Small dependency-light E1 validation renderer for W&B examples."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    'horizon': (255, 79, 195),
    'prediction': (75, 216, 255),
    'valid_touch': (83, 223, 155),
    'invalid_touch': (145, 155, 164),
}


def _load_prediction_footprints(path):
    footprints = []
    if not path.is_file():
        return footprints
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 15:
            continue
        height, width, length = [float(value) for value in fields[8:11]]
        x, _, z = [float(value) for value in fields[11:14]]
        yaw = float(fields[14])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        local = [
            (length / 2, width / 2), (length / 2, -width / 2),
            (-length / 2, -width / 2), (-length / 2, width / 2),
        ]
        footprints.append([
            (x + cosine * dx + sine * dz, z - sine * dx + cosine * dz)
            for dx, dz in local
        ])
    return footprints


def render_e1_example(image_path, artifact_path, prediction_path,
                      camera_size=(640, 360), bev_size=(640, 360)):
    """Render camera geometry and prediction footprints as one two-panel image."""
    artifact = json.loads(Path(artifact_path).read_text())
    image = Image.open(str(image_path)).convert('RGB')
    raw_width, raw_height = image.size
    camera = image.resize(camera_size)
    camera_draw = ImageDraw.Draw(camera, 'RGBA')
    sx, sy = camera_size[0] / raw_width, camera_size[1] / raw_height
    transform = lambda point: (point[0] * sx, point[1] * sy)

    horizon = artifact.get('horizon', {})
    segment = horizon.get('visible_segment_raw')
    raw_segment = horizon.get('raw_visible_segment_raw')
    if raw_segment and horizon.get('vertical_edge_refinement', {}).get('valid'):
        camera_draw.line([transform(point) for point in raw_segment],
                         fill=COLORS['horizon'] + (90,), width=1)
    if segment:
        camera_draw.line([transform(point) for point in segment],
                         fill=COLORS['horizon'] + (235,), width=2)
    for detection in artifact.get('detections', []):
        corners = detection.get('bbox_xyxy_raw')
        if corners:
            left, top = transform(corners[0])
            right, bottom = transform(corners[1])
            camera_draw.rectangle((left, top, right, bottom),
                                  outline=COLORS['prediction'] + (220,), width=2)
        for index, point in enumerate(detection.get('touchpoints_raw', [])):
            x, y = transform(point)
            valid = (detection.get('touchpoint_ray_valid') or [])[index]
            color = COLORS['valid_touch'] if valid else COLORS['invalid_touch']
            camera_draw.ellipse((x - 3, y - 3, x + 3, y + 3),
                                fill=color + (245,), outline=(5, 12, 18, 255))

    bev = Image.new('RGB', bev_size, (5, 11, 16))
    bev_draw = ImageDraw.Draw(bev, 'RGBA')
    scale = min(bev_size[0] / 64.0, bev_size[1] / 82.0)
    origin = (bev_size[0] / 2.0, bev_size[1] - 20.0)
    to_bev = lambda point: (origin[0] + point[0] * scale,
                            origin[1] - point[1] * scale)
    for radius in range(10, 81, 10):
        r = radius * scale
        bev_draw.ellipse((origin[0] - r, origin[1] - r,
                          origin[0] + r, origin[1] + r),
                         outline=(92, 145, 169, 65), width=1)
    for footprint in _load_prediction_footprints(Path(prediction_path)):
        polygon = [to_bev(point) for point in footprint]
        bev_draw.polygon(polygon, fill=COLORS['prediction'] + (30,))
        bev_draw.line(polygon + [polygon[0]], fill=COLORS['prediction'] + (220,), width=2)
    bev_draw.polygon([
        (origin[0], origin[1] - 10),
        (origin[0] + 7, origin[1] + 8),
        (origin[0] - 7, origin[1] + 8),
    ], fill=(231, 246, 255, 255), outline=COLORS['prediction'] + (255,))

    combined = Image.new('RGB', (camera_size[0] + bev_size[0],
                                 max(camera_size[1], bev_size[1])), (8, 13, 19))
    combined.paste(camera, (0, 0))
    combined.paste(bev, (camera_size[0], 0))
    return combined


def render_validation_examples(dataset, result_root, count=8):
    result_root = Path(result_root)
    frame_ids = list(dataset.idx_list)
    if not frame_ids or count <= 0:
        return []
    if len(frame_ids) <= count:
        selected = frame_ids
    else:
        selected = [frame_ids[round(i * (len(frame_ids) - 1) / (count - 1))]
                    for i in range(count)] if count > 1 else [frame_ids[len(frame_ids) // 2]]
    examples = []
    for frame_id in selected:
        artifact = result_root / 'artifacts' / (frame_id + '.json')
        prediction = result_root / 'data' / (frame_id + '.txt')
        image = Path(dataset.image_dir) / (frame_id + '.png')
        if artifact.is_file() and prediction.is_file() and image.is_file():
            examples.append((frame_id, render_e1_example(image, artifact, prediction)))
    return examples
