"""
EfficientSAM3 ARM Inference Package

This package provides ARM-optimized inference for EfficientSAM3 without 
requiring CUDA or Triton dependencies.

Usage:
    from efficientsam3_arm import build_efficientsam3_image_model, Sam3ProcessorARM
    
    # Load model
    model = build_sam3_model_arm(
        checkpoint_path="weights/efficient_sam3_repvit_s.pt",
        device="mps"  # or "cpu"
    )
    
    # Create processor
    processor = Sam3ProcessorARM(model, device="mps")
    
    # Process image
    from PIL import Image
    image = Image.open("image.jpg")
    state = processor.set_image(image)
    state = processor.set_text_prompt("person. car.", state)
    
    # Get results
    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]
"""

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from .model_builder_arm import build_efficientsam3_image_model

__version__ = "0.1.0"

__all__ = ["build_efficientsam3_image_model"]
