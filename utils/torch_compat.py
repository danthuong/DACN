"""Small public-API compatibility helpers for supported PyTorch releases."""

import torch


def cuda_autocast(enabled=True, dtype=None):
    """Return a CUDA autocast context without using deprecated APIs."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        kwargs = {"enabled": enabled}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.amp.autocast("cuda", **kwargs)

    kwargs = {"enabled": enabled}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return torch.cuda.amp.autocast(**kwargs)


def cuda_grad_scaler(enabled=True):
    """Build a CUDA GradScaler using the modern API when it is available."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # Older public API did not take a device argument.
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def meshgrid_ij(*tensors):
    """Create an ij-indexed meshgrid without warnings on modern PyTorch."""
    try:
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError:  # PyTorch 1.9 and older do not accept ``indexing``.
        return torch.meshgrid(*tensors)
