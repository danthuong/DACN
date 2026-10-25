import os
import tqdm
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import load_pretrained_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name,
                 logging_cfg=None,
                 wandb_logger=None,
                 resolved_config=None,
                 validation_cfg=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.logging_cfg = logging_cfg or {}
        self.wandb = wandb_logger
        self.global_step = 0
        self.resolved_config = resolved_config
        self.validation_cfg = validation_cfg or {}
        self.best_depth_result = float('inf')
        self.grad_accum_steps = max(1, int(cfg.get('grad_accum_steps', 1)))
        self.grad_clip_norm = float(cfg.get('grad_clip_norm', 0.0))
        precision = str(cfg.get('precision', 'fp32')).lower()
        self.amp_enabled = self.device.type == 'cuda' and precision != 'fp32'
        bf16_supported = bool(
            hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported())
        self.amp_dtype = torch.bfloat16 if 'bf16' in precision and bf16_supported else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_pretrained_checkpoint(
                model=self.model,
                filename=cfg['pretrain_model'],
                map_location=self.device,
                logger=self.logger)

        if cfg.get('resume_model', None):
            resume_value = cfg['resume_model']
            if isinstance(resume_value, str) and resume_value.lower() not in ('true', 'false'):
                resume_model_path = resume_value
            else:
                resume_model_path = os.path.join(self.output_dir, "checkpoint_last.pth")
                if not os.path.exists(resume_model_path):
                    resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger,
                scheduler=self.lr_scheduler,
                scaler=self.scaler)
            checkpoint = torch.load(resume_model_path, map_location='cpu')
            self.global_step = int(checkpoint.get('global_step', 0))
            self.best_depth_result = float(
                checkpoint.get('best_depth_result', float('inf')))
            if checkpoint.get('scheduler_state') is None:
                self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))

    def _phase(self, epoch_index):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return 2 if getattr(model, 'geometry_enabled', False) and (
            epoch_index >= getattr(model, 'fusion_start_epoch', 10 ** 9)) else 1

    def _checkpoint_state(self, epoch, best_result, best_epoch):
        return get_checkpoint_state(
            self.model, self.optimizer, epoch, best_result, best_epoch,
            self.lr_scheduler, self.scaler, self.global_step,
            self._phase(max(epoch - 1, 0)), self.resolved_config,
            self.best_depth_result)

    @staticmethod
    def _gradient_norms(model):
        sums = {'horizon_touchpoint': 0.0, 'fusion': 0.0}
        alpha_gradient = None
        for name, parameter in model.named_parameters():
            if parameter.grad is None or 'geometry_e1' not in name:
                continue
            squared = float(parameter.grad.detach().float().square().sum())
            if '.horizon_head.' in name or '.touchpoint_head.' in name:
                sums['horizon_touchpoint'] += squared
            else:
                sums['fusion'] += squared
            if name.endswith('geometry_e1.fusion_alpha'):
                alpha_gradient = float(parameter.grad.detach().float().abs())
        values = {'grad/' + name: value ** 0.5 for name, value in sums.items()}
        if alpha_gradient is not None:
            values['grad/fusion_alpha_abs'] = alpha_gradient
        return values

    def _run_depth_validation(self):
        if self.tester is None or not self.validation_cfg:
            return None
        from tools.evaluate_e1_depth import evaluate

        split_file = Path(self.train_loader.dataset.root_dir) / 'ImageSets' / (
            self.validation_cfg.get('split', 'val') + '.txt')
        label_dir = Path(self.train_loader.dataset.root_dir) / 'training' / 'label_2'
        model_dirs = []
        baseline_dir = Path(self.validation_cfg.get(
            'baseline_prediction_dir', 'outputs/monodetr/outputs/data'))
        if baseline_dir.is_dir() and any(baseline_dir.glob('*.txt')):
            model_dirs.append('baseline_pretrained=' + str(baseline_dir))
        else:
            self.logger.warning(
                'Baseline predictions are absent at %s; depth report will contain '
                'pass1/pass2 only. Export the baseline once with the documented command.',
                baseline_dir)
        for name in ('pass1', 'pass2'):
            directory = Path(self.tester._result_root(name)) / 'data'
            model_dirs.append(name + '=' + str(directory))
        args = SimpleNamespace(
            ground_truth_dir=str(label_dir), split_file=str(split_file),
            model=model_dirs,
            output_dir=self.validation_cfg.get(
                'output_dir', str(Path(self.output_dir) / 'validation')),
            class_name='Car',
            iou_threshold=float(self.validation_cfg.get('iou_threshold', 0.5)),
            skip_official_ap=False,
        )
        return evaluate(args)
        
    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
               
                save_checkpoint(
                    self._checkpoint_state(self.epoch, best_result, best_epoch),
                    ckpt_name)

                if self.cfg.get('save_last_and_best', False):
                    save_checkpoint(
                        self._checkpoint_state(self.epoch, best_result, best_epoch),
                        os.path.join(self.output_dir, 'checkpoint_last'))

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    model_core = (
                        self.model.module if isinstance(self.model, nn.DataParallel)
                        else self.model)
                    if getattr(model_core, 'geometry_enabled', False):
                        self.tester.inference([
                            ('pass1', 'pass1'), ('pass2', 'pass2')])
                        cur_result = self.tester.evaluate('pass2')
                        depth_summary = self._run_depth_validation()
                    else:
                        self.tester.inference()
                        cur_result = self.tester.evaluate()
                        depth_summary = None
                    if self.wandb is not None:
                        validation_log = {
                            'epoch': self.epoch,
                            'val/ap3d_r40_car_moderate': float(cur_result),
                        }
                        if depth_summary is not None:
                            for name, values in depth_summary['models'].items():
                                for key in ('mae', 'rmse', 'median_ae', 'p90_ae',
                                            'signed_bias', 'abs_rel', 'recall'):
                                    if values.get(key) is not None:
                                        validation_log['val/{}/{}'.format(name, key)] = values[key]
                            output_dir = self.validation_cfg.get('output_dir')
                            if output_dir:
                                self.wandb.log_artifact(
                                    output_dir, 'validation-epoch-{}'.format(self.epoch),
                                    'validation')
                            wandb_cfg = self.logging_cfg.get('wandb', {})
                            if wandb_cfg.get('log_validation_images', False):
                                from lib.helpers.e1_visualization import render_validation_examples
                                examples = render_validation_examples(
                                    self.test_loader.dataset,
                                    self.tester._result_root('pass2'),
                                    int(wandb_cfg.get('val_visualization_samples', 8)))
                                self.wandb.log_images(
                                    'val/examples_camera_bev', examples,
                                    step=self.global_step)
                        self.wandb.log(validation_log, step=self.global_step)
                    if depth_summary is not None:
                        pass2_mae = depth_summary['models'].get('pass2', {}).get('mae')
                        if pass2_mae is not None and np.isfinite(pass2_mae) and (
                                pass2_mae < self.best_depth_result):
                            self.best_depth_result = pass2_mae
                            save_checkpoint(
                                self._checkpoint_state(
                                    self.epoch, best_result, best_epoch),
                                os.path.join(self.output_dir, 'checkpoint_best_depth'))
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            self._checkpoint_state(self.epoch, best_result, best_epoch),
                            ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

                # Validation may have changed best metrics; keep resume files exact.
                if self.cfg.get('save_last_and_best', False):
                    state = self._checkpoint_state(
                        self.epoch, best_result, best_epoch)
                    save_checkpoint(state, os.path.join(self.output_dir, 'checkpoint_last'))
                    if not self.cfg['save_all']:
                        save_checkpoint(state, os.path.join(self.output_dir, 'checkpoint'))

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        model_without_parallel = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(model_without_parallel, 'set_epoch'):
            model_without_parallel.set_epoch(epoch)
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        self.optimizer.zero_grad()
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            geometry_calibs = targets.get('input_calib')
            targets = self.prepare_targets(targets, inputs.shape[0])
            ##dn
            dn_args = None
            if self.cfg["use_dn"]:
                dn_args=(targets, self.cfg['scalar'], self.cfg['label_noise_scale'], self.cfg['box_noise_scale'], self.cfg['num_patterns'])
            ###
            # train one batch
            with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                outputs = self.model(
                    inputs, calibs, targets, img_sizes, dn_args=dn_args,
                    geometry_calibs=geometry_calibs)
                mask_dict = None
                detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

                weight_dict = self.detr_loss.weight_dict
                detr_losses_dict_weighted = [
                    detr_losses_dict[k] * weight_dict[k]
                    for k in detr_losses_dict.keys() if k in weight_dict]
                detr_losses = sum(detr_losses_dict_weighted)

            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {}
            metric_values = {}
            detr_losses_log = 0
            for k in detr_losses_dict.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                    detr_losses_log += detr_losses_dict_log[k]
                elif k not in ('class_error', 'cardinality_error'):
                    metric_values[k] = detr_losses_dict[k].item()
            detr_losses_dict_log["loss_detr"] = detr_losses_log

            flags = [True] * 5
            if batch_idx % 30 == 0:
                print("----", batch_idx, "----")
                print("%s: %.2f, " %("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                        if flags[int(key[-1])]:
                            print("")
                            flags[int(key[-1])] = False
                    print("%s: %.2f, " %(key, val), end="")
                print("")
                print("")

            self.scaler.scale(detr_losses / self.grad_accum_steps).backward()
            should_step = (
                (batch_idx + 1) % self.grad_accum_steps == 0
                or batch_idx + 1 == len(self.train_loader))
            grad_norm = None
            gradient_values = {}
            if should_step:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                gradient_frequency = int(
                    self.logging_cfg.get('gradient_norms_every_n_steps', 100))
                if gradient_frequency > 0 and (
                        self.global_step + 1) % gradient_frequency == 0:
                    gradient_values = self._gradient_norms(model_without_parallel)
                if self.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.global_step += 1
            log_frequency = int(self.logging_cfg.get('log_every_n_steps', 20))
            if self.wandb is not None and (
                    self.global_step % log_frequency == 0 or gradient_values):
                model_core = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                phase = 2 if getattr(model_core, 'geometry_enabled', False) and (
                    epoch >= getattr(model_core, 'fusion_start_epoch', 10 ** 9)) else 1
                values = {
                    'epoch': epoch,
                    'train/phase': phase,
                    'train/loss_total': detr_losses.detach().item(),
                }
                for key, value in detr_losses_dict_log.items():
                    values['train/' + key] = value
                for key, value in metric_values.items():
                    values['metrics/' + key] = value
                for group in self.optimizer.param_groups:
                    values['lr/' + group.get('group_name', 'default')] = group['lr']
                if grad_norm is not None:
                    values['grad/global_clipped_norm'] = float(grad_norm)
                values.update(gradient_values)
                if getattr(model_core, 'geometry_enabled', False):
                    values['geometry/fusion_alpha'] = float(
                        model_core.geometry_e1.alpha.detach())
                    geometry = outputs.get('geometry')
                    if geometry is not None:
                        values['geometry/valid_object_ratio'] = float(
                            geometry['object_valid'].float().mean().detach())
                self.wandb.log(values, step=self.global_step)

            progress_bar.update()
        progress_bar.close()

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        object_keys = [
            'labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin',
            'heading_res', 'boxes_3d', 'object_ids', 'touch_points',
            'touchpoint_valid']
        image_keys = ['horizon_line', 'horizon_valid', 'input_calib', 'affine']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in object_keys:
                    target_dict[key] = val[bz][mask[bz]]
                elif key in image_keys:
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list

