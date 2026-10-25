import os
import tqdm
import json
from pathlib import Path

import torch
import numpy as np
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from utils import box_ops
import time


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monodetr'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name
        self.output_variant = cfg.get('output_variant', 'final')
        self.output_name = cfg.get('output_name')
        self.last_performance = None

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.evaluate()

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    @staticmethod
    def _variant_outputs(outputs, variant):
        if variant in ('final', 'pass2'):
            return outputs
        if variant == 'pass1':
            if 'pass1_outputs' not in outputs:
                raise ValueError(
                    'pass1 was requested, but this model output is not an E1 two-pass output')
            return outputs['pass1_outputs']
        raise ValueError('output variant must be final, pass1, or pass2')

    def _result_root(self, output_name=None):
        name = output_name if output_name is not None else self.output_name
        if name:
            return os.path.join(self.output_dir, 'outputs', name)
        return os.path.join(self.output_dir, 'outputs')

    @staticmethod
    def _to_raw_pixels(points_normalized, affine, input_hw):
        points = np.asarray(points_normalized, dtype=np.float64)
        pixels = points * np.asarray([input_hw[1], input_hw[0]], dtype=np.float64)
        homogeneous = np.concatenate(
            [pixels, np.ones((*pixels.shape[:-1], 1), dtype=np.float64)], axis=-1)
        inverse = np.linalg.inv(np.asarray(affine, dtype=np.float64))
        raw = homogeneous @ inverse.T
        return (raw[..., :2] / raw[..., 2:3]).tolist()

    def _save_e1_artifacts(self, outputs, selected_outputs, info, inputs,
                           output_name, topk):
        if 'geometry' not in outputs:
            return
        root = Path(self._result_root(output_name)) / 'artifacts'
        root.mkdir(parents=True, exist_ok=True)
        geometry = outputs['geometry']
        logits = selected_outputs['pred_logits'].detach().float()
        boxes = selected_outputs['pred_boxes'].detach().float()
        depths = selected_outputs['pred_depth'][..., 0].detach().float()
        pass1 = outputs.get('pass1_outputs', outputs)
        pass1_depths = pass1['pred_depth'][..., 0].detach().float()
        probabilities = logits.sigmoid()
        scores, flat_indices = torch.topk(
            probabilities.flatten(1), min(topk, probabilities.shape[1] * probabilities.shape[2]), dim=1)
        query_indices = flat_indices // probabilities.shape[2]
        class_indices = flat_indices % probabilities.shape[2]
        input_hw = (int(inputs.shape[-2]), int(inputs.shape[-1]))

        for batch_index, image_id_value in enumerate(info['img_id']):
            image_id = int(image_id_value)
            affine = info['affine'][batch_index].detach().cpu().numpy()
            detections = []
            kept_rank = 0
            for rank in range(scores.shape[1]):
                score = float(scores[batch_index, rank])
                if score < float(self.cfg.get('threshold', 0.2)):
                    continue
                query_index = int(query_indices[batch_index, rank])
                class_index = int(class_indices[batch_index, rank])
                box_xyxy = box_ops.box_cxcylrtb_to_xyxy(
                    boxes[batch_index, query_index]).detach().cpu().numpy()
                record = {
                    'detection_id': kept_rank,
                    'query_id': query_index,
                    'class_id': class_index,
                    'class_score': score,
                    'bbox_xyxy_normalized': box_xyxy.tolist(),
                    'bbox_xyxy_raw': self._to_raw_pixels(
                        box_xyxy.reshape(2, 2), affine, input_hw),
                    'pass1_depth_m': float(pass1_depths[batch_index, query_index]),
                    'selected_depth_m': float(depths[batch_index, query_index]),
                }
                if query_index < geometry['object_depth'].shape[1]:
                    touch = geometry['touchpoints_abs'][batch_index, query_index].detach().float().cpu().numpy()
                    record.update({
                        'touchpoints_normalized': touch.tolist(),
                        'touchpoints_raw': self._to_raw_pixels(touch, affine, input_hw),
                        'geometric_depth_m': float(geometry['object_depth'][batch_index, query_index]),
                        'geometry_valid': bool(geometry['object_valid'][batch_index, query_index]),
                        'geometry_confidence': float(
                            geometry['object_reliability'][batch_index, query_index]),
                        'geometry_depth_std_m': float(
                            geometry['object_depth_std'][batch_index, query_index]),
                        'touchpoint_ray_valid': geometry['touchpoint_ray_valid'][
                            batch_index, query_index].detach().cpu().tolist(),
                    })
                detections.append(record)
                kept_rank += 1

            horizon = outputs['pred_horizon_line'][batch_index].detach().float().cpu().numpy()
            network_segment = np.asarray([
                [0.0, horizon[1]],
                [float(input_hw[1] - 1), horizon[0] * float(input_hw[1] - 1) + horizon[1]],
            ])
            payload = {
                'schema_version': 'monodetr-e1-inference-v1',
                'model_variant': output_name or self.output_variant,
                'frame_id': '{:06d}'.format(image_id),
                'network_input_size': [input_hw[1], input_hw[0]],
                'original_image_size': [
                    int(info['img_size'][batch_index, 0]),
                    int(info['img_size'][batch_index, 1]),
                ],
                'horizon': {
                    'line_network': horizon.tolist(),
                    'visible_segment_raw': self._to_raw_pixels(
                        network_segment, affine, input_hw),
                },
                'ground_plane': {
                    'coefficients': geometry['plane'][batch_index].detach().float().cpu().tolist(),
                    'valid': bool(geometry['plane_valid'][batch_index]),
                    'camera_height_assumption_m': float(
                        (self.model.module if hasattr(self.model, 'module') else self.model)
                        .geometry_e1.camera_height_m),
                },
                'horizon_reliability': float(
                    geometry['horizon_reliability'][batch_index]),
                'fusion_alpha': float(outputs['fusion_alpha'].detach())
                    if 'fusion_alpha' in outputs else None,
                'detections': detections,
            }
            (root / '{:06d}.json'.format(image_id)).write_text(
                json.dumps(payload, separators=(',', ':')) + '\n', encoding='utf-8')

    def inference(self, variants=None):
        torch.set_grad_enabled(False)
        self.model.eval()

        if variants is None:
            variants = [(self.output_variant, self.output_name)]
        elif isinstance(variants, str):
            variants = [(variants, variants)]
        variants = [
            (item, item) if isinstance(item, str) else tuple(item)
            for item in variants
        ]
        results = {output_name or variant: {} for variant, output_name in variants}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        image_count = 0
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)
            geometry_calibs = info.get('input_calib')
            if geometry_calibs is not None:
                geometry_calibs = geometry_calibs.to(self.device)

            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()
            ###dn
            outputs = self.model(
                inputs, calibs, targets, img_sizes, dn_args=0,
                geometry_calibs=geometry_calibs)
            ###
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            end_time = time.perf_counter()
            model_infer_time += end_time - start_time
            image_count += int(inputs.shape[0])

            original_info = info
            calibration_objects = [
                self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            numpy_info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            for variant, output_name in variants:
                selected = self._variant_outputs(outputs, variant)
                dets = extract_dets_from_outputs(
                    outputs=selected, K=self.max_objs, topk=self.cfg['topk'])
                decoded = decode_detections(
                    dets=dets.detach().cpu().numpy(),
                    info=numpy_info,
                    calibs=calibration_objects,
                    cls_mean_size=self.dataloader.dataset.cls_mean_size,
                    threshold=self.cfg.get('threshold', 0.2))
                results[output_name or variant].update(decoded)
                self._save_e1_artifacts(
                    outputs, selected, original_info, inputs,
                    output_name, self.cfg['topk'])
            progress_bar.update()

        latency_ms = 1000.0 * model_infer_time / max(image_count, 1)
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
            if self.device.type == 'cuda' else 0.0)
        self.last_performance = {
            'image_count': image_count,
            'total_seconds': model_infer_time,
            'latency_ms_per_image': latency_ms,
            'peak_allocated_vram_mb': peak_vram_mb,
            'note': 'End-to-end model forward; excludes data loading and KITTI decoding.',
        }
        print("inference on {} images: {:.3f} ms/image, peak VRAM {:.1f} MiB".format(
            image_count, latency_ms, peak_vram_mb))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        for variant, output_name in variants:
            key = output_name or variant
            self.save_results(results[key], output_name)
            performance_path = Path(self._result_root(output_name)) / 'performance.json'
            performance_path.write_text(
                json.dumps(self.last_performance, indent=2) + '\n', encoding='utf-8')
        return results

    def save_results(self, results, output_name=None):
        output_dir = os.path.join(self._result_root(output_name), 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            with open(output_path, 'w') as target:
                for i in range(len(results[img_id])):
                    class_name = self.class_name[int(results[img_id][i][0])]
                    target.write('{} 0.0 0'.format(class_name))
                    for j in range(1, len(results[img_id][i])):
                        target.write(' {:.2f}'.format(results[img_id][i][j]))
                    target.write('\n')

    def evaluate(self, output_name=None, return_dict=False):
        results_dir = os.path.join(self._result_root(output_name), 'data')
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(
            results_dir=results_dir, logger=self.logger, return_dict=return_dict)
        return result
