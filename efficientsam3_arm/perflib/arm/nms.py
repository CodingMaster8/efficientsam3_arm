# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# Adapted from https://github.com/stackav-oss/conch/blob/main/conch/kernels/vision/nms.py

import torch
import numpy as np

"""
COMPATIBILITY NOTE:
This is an ARM-compatible replacement for the original Triton NMS kernel.
It uses NumPy to perform NMS on the CPU. While slower than a dedicated CUDA kernel
for very large numbers of boxes, it is robust and works natively on Apple Silicon 
and standard ARM CPUs.
"""

def nms_arm(
    ious: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Perform NMS given the iou matrix, the scores and the iou threshold.
    
    This version runs on CPU via NumPy to support ARM architectures.

    Args:
        ious: Pairwise IoU tensor of shape (N, N).
        scores: Scores tensor of shape (N,).
        iou_threshold: IoU threshold for suppression.

    Returns:
        Tensor: Indices of kept boxes, sorted by decreasing score.
    """
    assert scores.dim() == 1, "Scores must be 1D"
    assert ious.dim() == 2, "IoU must be 2D"
    assert ious.shape[0] == ious.shape[1] == scores.shape[0], "Dimension mismatch between IoU and scores"

    # 1. Setup: Move data to CPU/Numpy for processing
    # We use detach() to ensure gradients aren't tracked (NMS is non-differentiable)
    device = scores.device
    ious_np = ious.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()

    # 2. Sort boxes by scores in descending order
    # argsort returns indices in ascending order, so we reverse it [::-1]
    order = scores_np.argsort()[::-1]
    
    kept_indices = []
    
    # 3. Greedy NMS Loop
    while order.size > 0:
        # The first element is the one with the highest current score
        i = order.item(0)
        kept_indices.append(i)
        
        # If this was the last element, break
        if order.size == 1:
            break
            
        # Compare the current max box (i) with all remaining boxes (order[1:])
        # ious_np[i, order[1:]] gives the IoU of box 'i' with all other remaining boxes
        # We want to keep boxes where IoU is <= threshold
        mask_keep = ious_np[i, order[1:]] <= iou_threshold
        
        # Apply mask to reduce the list of candidates
        # We perform the reduction on `order[1:]` (the remaining candidates)
        order = order[1:][mask_keep]

    # 4. Convert result back to PyTorch tensor on original device
    # We use copy() to ensure memory is contiguous and compatible
    return torch.tensor(kept_indices, dtype=torch.long, device=device)