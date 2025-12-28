# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Utility functions for ARM-based EfficientSAM3 inference.
"""

import os
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_optimal_device() -> str:
    """
    Automatically detect the best available device for ARM inference.
    
    Returns:
        str: Device string ("mps", "cpu")
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL.Image.Image: Loaded image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return Image.open(image_path).convert("RGB")


def visualize_results(
    image: Union[str, Image.Image, np.ndarray],
    masks: Optional[torch.Tensor] = None,
    boxes: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    labels: Optional[List[str]] = None,
    score_threshold: float = 0.0,
    save_path: Optional[str] = None,
    dpi: int = 100,
    show_scores: bool = True,
) -> plt.Figure:
    """
    Visualize detection results with masks and bounding boxes.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        masks: Binary masks tensor [N, H, W] or [N, 1, H, W]
        boxes: Bounding boxes tensor [N, 4] in xyxy format
        scores: Confidence scores tensor [N]
        labels: Optional list of text labels for each detection
        score_threshold: Minimum score to display
        save_path: Optional path to save the visualization
        dpi: DPI for saved figure
        show_scores: Whether to show scores on boxes
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Load image
    if isinstance(image, str):
        image = load_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis('off')
    
    # Process masks
    if masks is not None:
        masks = masks.cpu().numpy()
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # Remove channel dimension
        
        # Create colored mask overlay
        for i, mask in enumerate(masks):
            if scores is not None and scores[i] < score_threshold:
                continue
            
            color = np.random.random(3)
            mask_binary = mask > 0.5
            
            # Create colored overlay
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask_binary] = [*color, 0.5]
            ax.imshow(colored_mask)
    
    # Process boxes
    if boxes is not None:
        boxes = boxes.cpu().numpy()
        
        for i, box in enumerate(boxes):
            if scores is not None and scores[i] < score_threshold:
                continue
            
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Random color for each box
            color = np.random.random(3)
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = ""
            if labels is not None and i < len(labels):
                label_text = labels[i]
            if show_scores and scores is not None:
                score_text = f"{scores[i]:.2f}"
                label_text = f"{label_text} {score_text}" if label_text else score_text
            
            if label_text:
                ax.text(
                    x1, y1 - 5,
                    label_text,
                    color='white',
                    fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
                )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        print(f"Saved visualization to {save_path}")
    
    return fig


def save_results(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    output_dir: str,
    prefix: str = "result",
) -> None:
    """
    Save detection results to files.
    
    Args:
        masks: Binary masks tensor [N, H, W] or [N, 1, H, W]
        boxes: Bounding boxes tensor [N, 4] in xyxy format
        scores: Confidence scores tensor [N]
        output_dir: Directory to save results
        prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save masks as numpy array
    masks_np = masks.cpu().numpy()
    if masks_np.ndim == 4:
        masks_np = masks_np.squeeze(1)
    np.save(os.path.join(output_dir, f"{prefix}_masks.npy"), masks_np)
    
    # Save boxes
    boxes_np = boxes.cpu().numpy()
    np.save(os.path.join(output_dir, f"{prefix}_boxes.npy"), boxes_np)
    
    # Save scores
    scores_np = scores.cpu().numpy()
    np.save(os.path.join(output_dir, f"{prefix}_scores.npy"), scores_np)
    
    # Save as text file for easy reading
    with open(os.path.join(output_dir, f"{prefix}_summary.txt"), 'w') as f:
        f.write(f"Number of detections: {len(scores_np)}\n\n")
        for i in range(len(scores_np)):
            f.write(f"Detection {i+1}:\n")
            f.write(f"  Score: {scores_np[i]:.4f}\n")
            f.write(f"  Box (x1,y1,x2,y2): {boxes_np[i]}\n")
            f.write(f"  Mask shape: {masks_np[i].shape}\n")
            f.write("\n")
    
    print(f"Saved results to {output_dir}/")


def batch_process_images(
    image_paths: List[str],
    processor,
    text_prompt: str,
    output_dir: Optional[str] = None,
    visualize: bool = True,
) -> List[dict]:
    """
    Process multiple images with the same text prompt.
    
    Args:
        image_paths: List of image file paths
        processor: Sam3ProcessorARM instance
        text_prompt: Text prompt for detection
        output_dir: Optional directory to save results
        visualize: Whether to create visualizations
        
    Returns:
        List of result dictionaries with masks, boxes, and scores
    """
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {image_path}")
        
        # Load and process image
        image = load_image(image_path)
        state = processor.set_image(image)
        state = processor.set_text_prompt(text_prompt, state)
        
        result = {
            "image_path": image_path,
            "masks": state.get("masks"),
            "boxes": state.get("boxes"),
            "scores": state.get("scores"),
        }
        results.append(result)
        
        # Save results
        if output_dir:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            if visualize and result["masks"] is not None:
                vis_path = os.path.join(output_dir, f"{image_name}_vis.png")
                visualize_results(
                    image,
                    masks=result["masks"],
                    boxes=result["boxes"],
                    scores=result["scores"],
                    save_path=vis_path,
                )
            
            if result["masks"] is not None:
                save_results(
                    result["masks"],
                    result["boxes"],
                    result["scores"],
                    output_dir,
                    prefix=image_name,
                )
    
    return results


def convert_mask_to_rle(mask: np.ndarray) -> dict:
    """
    Convert binary mask to RLE (Run-Length Encoding) format.
    
    Args:
        mask: Binary mask array [H, W]
        
    Returns:
        dict: RLE encoded mask
    """
    mask = mask.flatten()
    
    # Find change points
    diff = np.diff(mask, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Create RLE
    counts = []
    for start, end in zip(starts, ends):
        counts.extend([start - sum(counts), end - start])
    
    return {
        "counts": counts,
        "size": mask.shape,
    }


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU (Intersection over Union) between two masks.
    
    Args:
        mask1: Binary mask array [H, W]
        mask2: Binary mask array [H, W]
        
    Returns:
        float: IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union
