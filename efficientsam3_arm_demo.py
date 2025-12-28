#!/usr/bin/env python3
"""
EfficientSAM3 ARM Inference Example

This script demonstrates how to use EfficientSAM3 on ARM devices (Apple Silicon or CPU).

Usage:
    # Basic usage with text prompt
    python efficientsam3_arm_demo.py --checkpoint weights/efficient_sam3_repvit_s.pt --image path/to/image.jpg --text_prompt "person. car. dog."
    
    # With visualization
    python efficientsam3_arm_demo.py --checkpoint weights/efficient_sam3_repvit_s.pt --image path/to/image.jpg --text_prompt "person. car." --visualize --output results/
    
    # Batch processing
    python efficientsam3_arm_demo.py --checkpoint weights/efficient_sam3_repvit_s.pt --image_dir path/to/images/ --text_prompt "person" --output results/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image

# Import the ARM-optimized module
from efficientsam3_arm import (
    build_sam3_model_arm,
    Sam3ProcessorARM,
    load_image,
    visualize_results,
    save_results,
    get_optimal_device,
)


def process_single_image(
    processor: Sam3ProcessorARM,
    image_path: str,
    text_prompt: str,
    output_dir: str = None,
    visualize: bool = True,
) -> dict:
    """Process a single image with text prompt."""
    
    print(f"\nProcessing: {image_path}")
    print(f"Text prompt: '{text_prompt}'")
    
    # Load image
    image = load_image(image_path)
    print(f"Image size: {image.size}")
    
    # Run inference
    state = processor.set_image(image)
    state = processor.set_text_prompt(text_prompt, state)
    
    # Get results
    num_detections = len(state["scores"]) if "scores" in state else 0
    print(f"Detections: {num_detections}")
    
    if num_detections > 0:
        scores = state["scores"].cpu().numpy()
        for i, score in enumerate(scores):
            print(f"  Detection {i+1}: confidence = {score:.3f}")
    
    # Visualize and save
    if output_dir and num_detections > 0:
        os.makedirs(output_dir, exist_ok=True)
        image_name = Path(image_path).stem
        
        if visualize:
            vis_path = os.path.join(output_dir, f"{image_name}_result.png")
            visualize_results(
                image,
                masks=state.get("masks"),
                boxes=state.get("boxes"),
                scores=state.get("scores"),
                save_path=vis_path,
            )
        
        # Save raw results
        save_results(
            state["masks"],
            state["boxes"],
            state["scores"],
            output_dir,
            prefix=image_name,
        )
    
    return state


def process_image_directory(
    processor: Sam3ProcessorARM,
    image_dir: str,
    text_prompt: str,
    output_dir: str = None,
    visualize: bool = True,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
) -> List[dict]:
    """Process all images in a directory."""
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Directory not found: {image_dir}")
    
    # Find all images
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
        image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"\nFound {len(image_paths)} images in {image_dir}")
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}]", end=" ")
        try:
            result = process_single_image(
                processor,
                str(image_path),
                text_prompt,
                output_dir,
                visualize,
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='EfficientSAM3 ARM Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--backbone_type', type=str, default='repvit',
                       help='Backbone type (default: repvit)')
    parser.add_argument('--model_name', type=str, default='s',
                       help='Model size (default: s)')
    
    # Input arguments
    parser.add_argument('--image', type=str,
                       help='Path to input image')
    parser.add_argument('--image_dir', type=str,
                       help='Path to directory of images (for batch processing)')
    parser.add_argument('--text_prompt', type=str, required=True,
                       help='Text prompt (e.g., "person. car. dog.")')
    
    # Output arguments
    parser.add_argument('--output', type=str,
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to disk')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'mps', 'cuda'],
                       help='Device to use (default: auto)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image_dir must be specified")
    
    if args.image and args.image_dir:
        parser.error("Cannot specify both --image and --image_dir")
    
    # Print configuration
    print("=" * 70)
    print("EfficientSAM3 ARM Inference")
    print("=" * 70)
    
    # Determine device
    if args.device == 'auto':
        device = get_optimal_device()
        print(f"Auto-detected device: {device}")
    else:
        device = args.device
        print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    try:
        model = build_sam3_model_arm(
            checkpoint_path=args.checkpoint,
            device=device,
            backbone_type=args.backbone_type,
            model_name=args.model_name,
            enable_inst_interactivity=False,
            verbose=True,
        )
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return 1
    
    # Create processor
    print("\nCreating processor...")
    processor = Sam3ProcessorARM(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
        use_fp16=False,  # Disable FP16 for stability on ARM
    )
    
    # Print device info
    device_info = processor.get_device_info()
    print("\nDevice Configuration:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    print("=" * 70)
    
    # Process image(s)
    output_dir = args.output if not args.no_save else None
    
    try:
        if args.image:
            # Single image
            process_single_image(
                processor,
                args.image,
                args.text_prompt,
                output_dir,
                args.visualize,
            )
        else:
            # Batch processing
            results = process_image_directory(
                processor,
                args.image_dir,
                args.text_prompt,
                output_dir,
                args.visualize,
            )
            
            # Print summary
            print("\n" + "=" * 70)
            print(f"Processed {len(results)} images")
            total_detections = sum(len(r.get("scores", [])) for r in results)
            print(f"Total detections: {total_detections}")
    
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=" * 70)
    print("✓ Done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
