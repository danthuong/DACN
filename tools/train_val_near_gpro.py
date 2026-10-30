#!/usr/bin/env python3
"""Train/evaluate the isolated near-contact GPro Z-map ablation."""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from lib.datasets.kitti.near_gpro_dataset import build_near_gpro_dataloader
from lib.helpers.config_helper import load_config
from lib.helpers.near_gpro_runtime import NearGProTester, NearGProTrainer
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.utils_helper import create_logger, set_random_seed
from lib.helpers.wandb_helper import WandbLogger
from lib.models.monodetr.near_gpro_zmap import build_near_gpro_zmap


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("-e", "--evaluate-only", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--checkpoint-path")
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    cfg = load_config(args.config, args.env_file)
    set_random_seed(cfg.get("random_seed", 444))
    model_name = cfg["model_name"]
    output_path = os.path.join("./" + cfg["trainer"]["save_path"], model_name)
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(
        output_path,
        "train.log." + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    logger = create_logger(log_file)

    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg["near_gpro"] = cfg["near_gpro"]
    train_loader, test_loader = build_near_gpro_dataloader(
        dataset_cfg,
        workers=int(cfg.get("runtime", {}).get("num_workers", 4)),
    )
    model_cfg = dict(cfg["model"])
    model_cfg["near_gpro"] = cfg["near_gpro"]
    model, criterion = build_near_gpro_zmap(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = [int(value) for value in cfg["trainer"]["gpu_ids"].split(",")]
    model = (
        model.to(device)
        if len(gpu_ids) == 1
        else torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)
    )

    if args.evaluate_only:
        tester = NearGProTester(
            cfg=cfg["tester"], model=model, dataloader=test_loader,
            logger=logger, train_cfg=cfg["trainer"], model_name=model_name,
        )
        if args.checkpoint_path:
            load_checkpoint(
                model=model, optimizer=None, filename=args.checkpoint_path,
                map_location=device, logger=logger,
            )
            tester.inference()
            tester.evaluate()
        else:
            tester.test()
        return

    optimizer = build_optimizer(cfg["optimizer"], model)
    scheduler_cfg = dict(cfg["lr_scheduler"])
    scheduler_cfg["max_epoch"] = cfg["trainer"]["max_epoch"]
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(
        scheduler_cfg, optimizer, last_epoch=-1
    )
    trainer_cfg = dict(cfg["trainer"])
    trainer_cfg["grad_clip_norm"] = cfg["optimizer"].get("grad_clip_norm", 0.0)
    wandb_logger = WandbLogger(
        cfg.get("logging", {}).get("wandb"), cfg, output_path, logger
    )
    trainer = NearGProTrainer(
        cfg=trainer_cfg, model=model, optimizer=optimizer,
        train_loader=train_loader, test_loader=test_loader,
        lr_scheduler=lr_scheduler, warmup_lr_scheduler=warmup_lr_scheduler,
        logger=logger, loss=criterion, model_name=model_name,
        logging_cfg=cfg.get("logging", {}), wandb_logger=wandb_logger,
        resolved_config=cfg, validation_cfg=None,
    )
    tester = NearGProTester(
        cfg=cfg["tester"], model=trainer.model, dataloader=test_loader,
        logger=logger, train_cfg=cfg["trainer"], model_name=model_name,
    )
    if cfg["dataset"]["test_split"] != "test":
        trainer.tester = tester
    logger.info("################### Near-GPro training ###################")
    logger.info("Batch Size: %d", cfg["dataset"]["batch_size"])
    logger.info("Learning Rate: %f", cfg["optimizer"]["lr"])
    try:
        trainer.train()
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
