# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Kernel for euclidean distance transform (EDT) - ARM/CPU Compatible Version
"""

import torch
import numpy as np
try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    raise ImportError("This module requires scipy. Please install it via `pip install scipy`.")

"""
COMPATIBILITY NOTE:
The original implementation used Triton (CUDA) to compute the EDT.
Since Triton is not natively supported on standard ARM architectures (like standard ARM CPUs or Apple Silicon),
this version replaces the kernel with `scipy.ndimage.distance_transform_edt`.

SciPy uses the same O(N) complexity algorithm (Meijster's algorithm) implemented in optimized C,
which is the standard high-performance solution for CPU-based execution.
"""

def edt_function(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.
    
    This function mimics the behavior of the original Triton kernel but runs on CPU
    to ensure compatibility with ARM architecture.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.
              0 represents the background (target), 1 represents the foreground.
              
    Returns:
        A tensor of the same shape as data containing the EDT.
        Equivalent to a batched version of cv2.distanceTransform(input, cv2.DIST_L2, 0)
    """
    # Ensure input is 3D
    if data.dim() != 3:
        raise ValueError(f"Expected 3D input (B, H, W), got {data.shape}")

    # 1. Move data to CPU and convert to Numpy
    # We ensure it's boolean/binary. The original kernel treated 0 as target, non-zero as infinity.
    # SciPy calculates distance FROM non-zero TO zero.
    device = data.device
    data_cpu = data.detach().cpu()
    
    # Input data: 0 is empty, 1 is occupied.
    # We want distance to nearest 0. 
    # scipy.ndimage.distance_transform_edt calculates distance from non-zero elements
    # to the nearest zero element. So we pass the data directly.
    # If the input data is float/int, we treat non-zero as "foreground".
    mask_np = data_cpu.numpy() != 0

    # 2. Compute EDT via SciPy
    # SciPy's distance_transform_edt usually takes a single image. 
    # We simply loop over the batch. This is fast enough for CPU workloads because
    # the heavy lifting is done in C inside SciPy.
    
    # Pre-allocate output array
    result_np = np.empty_like(mask_np, dtype=np.float32)

    B = mask_np.shape[0]
    for b in range(B):
        # distance_transform_edt returns the Euclidean distance directly (not squared)
        result_np[b] = distance_transform_edt(mask_np[b])

    # 3. Convert back to Torch and move to original device
    output = torch.from_numpy(result_np).to(device=device, dtype=data.dtype)

    return output

# Alias for backward compatibility if other code imports 'edt_triton'
edt_triton = edt_function