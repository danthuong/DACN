import os
import random
import numpy as np
import torch
import torch.nn as nn


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None, best_result=None,
                         best_epoch=None, scheduler=None, scaler=None,
                         global_step=0, phase=None, resolved_config=None,
                         best_depth_result=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optim_state,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'global_step': global_step,
        'phase': phase,
        'best_result': best_result,
        'best_epoch': best_epoch,
        'best_depth_result': best_depth_result,
        'resolved_config': resolved_config,
        'random_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def _as_cpu_byte_rng_state(state):
    """Normalize RNG tensors after loading a checkpoint onto any device.

    ``torch.load(..., map_location='cuda')`` also moves the saved CPU RNG
    tensor to CUDA.  PyTorch's CPU and CUDA RNG setters both expect their
    serialized state as a CPU ByteTensor, so move/cast it back explicitly.
    """
    if torch.is_tensor(state):
        return state.detach().to(device='cpu', dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device='cpu').contiguous()


def _restore_random_state(random_state):
    if random_state.get('python') is not None:
        random.setstate(random_state['python'])
    if random_state.get('numpy') is not None:
        np.random.set_state(random_state['numpy'])

    torch_cpu_state = random_state.get('torch_cpu')
    if torch_cpu_state is not None:
        torch.set_rng_state(_as_cpu_byte_rng_state(torch_cpu_state))

    torch_cuda_states = random_state.get('torch_cuda')
    if torch.cuda.is_available() and torch_cuda_states is not None:
        normalized_states = [
            _as_cpu_byte_rng_state(state) for state in torch_cuda_states
        ]
        # A checkpoint may have been written with more GPUs than are visible
        # for the current run. Restore the states for the available devices.
        normalized_states = normalized_states[:torch.cuda.device_count()]
        if normalized_states:
            torch.cuda.set_rng_state_all(normalized_states)


def load_checkpoint(model, optimizer, filename, map_location, logger=None,
                    scheduler=None, scaler=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        best_result = checkpoint.get('best_result', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler is not None and checkpoint.get('scheduler_state') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        if scaler is not None and checkpoint.get('scaler_state') is not None:
            scaler.load_state_dict(checkpoint['scaler_state'])
        random_state = checkpoint.get('random_state')
        if random_state:
            _restore_random_state(random_state)
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_result, best_epoch


def load_pretrained_checkpoint(model, filename, map_location, logger=None):
    """Load every compatible baseline weight and report all new/mismatched keys.

    Unlike ``strict=False``, this explicitly filters shape mismatches and raises
    if a checkpoint key maps to a model key with a different shape. Missing
    model keys are expected only for newly introduced E1 modules.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)
    checkpoint = torch.load(filename, map_location)
    source = checkpoint.get('model_state', checkpoint)
    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    target = target_model.state_dict()
    if source and all(key.startswith('module.') for key in source):
        source = {key[len('module.'):]: value for key, value in source.items()}

    unexpected = sorted(set(source) - set(target))
    missing = sorted(set(target) - set(source))
    mismatched = sorted(
        (key, tuple(source[key].shape), tuple(target[key].shape))
        for key in set(source) & set(target)
        if tuple(source[key].shape) != tuple(target[key].shape))
    if mismatched:
        details = '; '.join('{} checkpoint{} model{}'.format(*item) for item in mismatched)
        raise RuntimeError('Checkpoint has matching-name shape mismatches: ' + details)
    compatible = {key: value for key, value in source.items() if key in target}
    target_model.load_state_dict(compatible, strict=False)
    if logger is not None:
        logger.info(
            'Pretrained load: %d matched, %d new/missing, %d unexpected, 0 shape mismatches',
            len(compatible), len(missing), len(unexpected))
        if missing:
            logger.info('New model keys initialized from scratch: %s', ', '.join(missing))
        if unexpected:
            logger.warning('Unexpected checkpoint keys: %s', ', '.join(unexpected))
    return {
        'matched': sorted(compatible),
        'missing': missing,
        'unexpected': unexpected,
        'mismatched': mismatched,
        'checkpoint_epoch': checkpoint.get('epoch', -1),
    }
