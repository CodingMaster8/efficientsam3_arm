# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch
import torch.nn.functional as F
import warnings

def flash_attn_func(q, k, v):
    """
    ARM/CPU-compatible replacement for Flash Attention 3.
    Uses PyTorch's built-in scaled_dot_product_attention (SDPA).
    
    Args:
        q, k, v: Input tensors of shape (Batch, SeqLen, NumHeads, HeadDim)
    """
    # 1. Try importing the optimized kernel (unlikely to exist on ARM, but good practice)
    try:
        from flash_attn_interface import flash_attn_func as fa3
        # Original code used FP8, but we only use it if the hardware supports it
        # and we actually have the library.
        return fa3(q, k, v)
    except (ImportError, ModuleNotFoundError):
        pass

    # 2. Fallback: Native PyTorch SDPA
    # FlashAttention inputs are usually (Batch, SeqLen, NumHeads, HeadDim).
    # PyTorch SDPA expects (Batch, NumHeads, SeqLen, HeadDim).
    # We must transpose dimensions 1 and 2.
    
    q_in = q.transpose(1, 2)
    k_in = k.transpose(1, 2)
    v_in = v.transpose(1, 2)

    # scaled_dot_product_attention automatically selects the best kernel 
    # (FlashAttention-2, MemEfficient, or C++) available on the current device (CPU/MPS/CUDA).
    output = F.scaled_dot_product_attention(
        q_in, 
        k_in, 
        v_in,
        dropout_p=0.0,
        is_causal=False  # Set to True if you specifically need causal masking
    )
    
    # Transpose back to (Batch, SeqLen, NumHeads, HeadDim) to match FlashAttn output
    return output.transpose(1, 2)

# We define the op registration just in case other parts of the code rely on 
# 'flash::flash_attn_func' existing in the library namespace, but we wire it to our safe function.
@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return flash_attn_func(q, k, v)

@flash_attn_func_op.register_fake
def _(q, k, v, **kwargs):
    return torch.empty_like(q).contiguous()