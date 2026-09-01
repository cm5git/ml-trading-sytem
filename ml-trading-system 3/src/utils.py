"""
Small shared utilities: device selection and GPU memory management.
"""

import gc

import torch


def get_device() -> torch.device:
    """Select CUDA if available, otherwise CPU, and enable TF32 on Ampere+ GPUs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU")

    return device


def clear_memory() -> None:
    """Free unused GPU memory and run garbage collection between training stages."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
