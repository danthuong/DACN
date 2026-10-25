import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
from lib.helpers.config_helper import load_config
from lib.helpers.wandb_helper import WandbLogger
from lib.helpers.save_helper import load_checkpoint


parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument('--env-file', default='.env', help='private environment file (default: .env)')
parser.add_argument('--checkpoint-path', help='explicit checkpoint for evaluation')
parser.add_argument(
    '--export', action='append', metavar='NAME=VARIANT',
    help='evaluation export, e.g. baseline=final or pass1=pass1 (repeatable)')
args = parser.parse_args()


def main():
    assert (os.path.exists(args.config))
    cfg = load_config(args.config, args.env_file)
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader
    dataset_cfg = dict(cfg['dataset'])
    dataset_cfg['geometry_e1'] = cfg.get('geometry_e1')
    train_loader, test_loader = build_dataloader(
        dataset_cfg, workers=int(cfg.get('runtime', {}).get('num_workers', 4)))

    # build model
    model_cfg = dict(cfg['model'])
    model_cfg['geometry_e1'] = cfg.get('geometry_e1')
    model, loss = build_model(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        if args.checkpoint_path:
            load_checkpoint(
                model=model, optimizer=None, filename=args.checkpoint_path,
                map_location=device, logger=logger)
        if args.export:
            variants = []
            for specification in args.export:
                if '=' not in specification:
                    raise ValueError('--export must have the form NAME=VARIANT')
                output_name, variant = specification.split('=', 1)
                variants.append((variant, output_name))
            tester.inference(variants)
            for _, output_name in variants:
                moderate, official = tester.evaluate(output_name, return_dict=True)
                metrics_path = os.path.join(
                    tester._result_root(output_name), 'official_kitti_metrics.json')
                with open(metrics_path, 'w') as target:
                    import json
                    json.dump(official, target, indent=2)
                    target.write('\n')
                logger.info('%s Car AP3D R40 Moderate: %.4f', output_name, moderate)
        elif args.checkpoint_path:
            tester.inference()
            tester.evaluate()
        else:
            tester.test()
        return
    #ipdb.set_trace()
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)
    # build lr scheduler
    scheduler_cfg = dict(cfg['lr_scheduler'])
    scheduler_cfg['max_epoch'] = cfg['trainer']['max_epoch']
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        scheduler_cfg, optimizer, last_epoch=-1)

    trainer_cfg = dict(cfg['trainer'])
    trainer_cfg['grad_clip_norm'] = cfg['optimizer'].get('grad_clip_norm', 0.0)
    wandb_logger = WandbLogger(
        cfg.get('logging', {}).get('wandb'), cfg, output_path, logger)
    trainer = Trainer(cfg=trainer_cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name,
                      logging_cfg=cfg.get('logging', {}),
                      wandb_logger=wandb_logger,
                      resolved_config=cfg,
                      validation_cfg=cfg.get('validation'))

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    try:
        trainer.train()
    finally:
        wandb_logger.finish()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()


if __name__ == '__main__':
    main()
