"""Optional Weights & Biases integration with no credential leakage."""

import os
import copy
import hashlib
import subprocess


class WandbLogger:
    def __init__(self, cfg, resolved_config, output_dir, logger):
        self.run = None
        self.logger = logger
        cfg = cfg or {}
        mode = str(cfg.get('mode', 'disabled')).lower()
        if not cfg.get('enabled', False) or mode == 'disabled':
            logger.info('W&B logging disabled')
            return
        try:
            import wandb
        except ImportError:
            logger.warning(
                "W&B requested but package is missing. Install with "
                "`pip install wandb`, then set WANDB_MODE=online (and run "
                "`wandb login`) or keep WANDB_MODE=offline. Training continues without W&B.")
            return

        if mode not in ('online', 'offline'):
            raise ValueError('WANDB_MODE must be online, offline, or disabled')
        if mode == 'online' and not (os.environ.get('WANDB_API_KEY') or os.path.isfile(
                os.path.expanduser('~/.netrc'))):
            raise RuntimeError(
                'WANDB_MODE=online but no login was found. Run `wandb login` or put '
                'WANDB_API_KEY in private .env (never in YAML).')

        entity = cfg.get('entity') or None
        tracked_config = copy.deepcopy(resolved_config)
        metadata = {
            'fusion_mode': (resolved_config.get('geometry_e1') or {}).get('fusion_mode'),
            'seed': resolved_config.get('random_seed'),
        }
        try:
            metadata['git_sha'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                text=True).strip()
        except (OSError, subprocess.SubprocessError):
            metadata['git_sha'] = None
        checkpoint = (resolved_config.get('trainer') or {}).get('pretrain_model')
        if checkpoint and os.path.isfile(checkpoint):
            digest = hashlib.sha256()
            with open(checkpoint, 'rb') as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            metadata['pretrained_checkpoint'] = checkpoint
            metadata['pretrained_sha256'] = digest.hexdigest()
        tracked_config['run_metadata'] = metadata
        run_id = cfg.get('run_id') or os.environ.get('WANDB_RUN_ID') or None
        wandb_cache = os.path.join(output_dir, 'wandb-cache')
        os.makedirs(wandb_cache, exist_ok=True)
        os.environ.setdefault('WANDB_CACHE_DIR', wandb_cache)
        try:
            self.run = wandb.init(
                project=cfg.get('project', 'monodetr-gpro3d-e1'),
                entity=entity,
                name=cfg.get('run_name'),
                tags=cfg.get('tags'),
                mode=mode,
                dir=output_dir,
                config=tracked_config,
                id=run_id,
                resume='allow' if run_id else None,
            )
        except Exception as error:
            logger.warning(
                'W&B initialization failed (%s). Training continues with local '
                'logs/checkpoints; fix W&B connectivity and resume to re-enable it.',
                error)
            self.run = None
            return
        logger.info('W&B initialized in %s mode: %s', mode, self.run.url or self.run.dir)

    @property
    def enabled(self):
        return self.run is not None

    def log(self, values, step=None):
        if self.run is not None:
            self.run.log(values, step=step)

    def log_artifact(self, path, name, artifact_type):
        if self.run is None:
            return
        import wandb
        artifact = wandb.Artifact(name=name, type=artifact_type)
        if os.path.isdir(path):
            artifact.add_dir(path)
        elif os.path.isfile(path):
            artifact.add_file(path)
        else:
            self.logger.warning('W&B artifact path does not exist: %s', path)
            return
        self.run.log_artifact(artifact)

    def log_images(self, key, images, step=None):
        if self.run is None or not images:
            return
        import wandb
        payload = [wandb.Image(image, caption=caption) for caption, image in images]
        self.run.log({key: payload}, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None

