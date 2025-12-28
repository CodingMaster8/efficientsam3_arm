# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
ARM-optimized image processor for EfficientSAM3.

This module provides a drop-in replacement for the CUDA-based Sam3Processor
that works efficiently on ARM devices (Apple Silicon with MPS or CPU).
"""

from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from torchvision.transforms import v2

# Use our standalone ARM implementations instead of sam3 modules
# This avoids importing any Triton-dependent code
from .model import box_ops_arm as box_ops
from .data_misc_arm import FindStage, interpolate, GeometricPrompt


class Sam3ProcessorARM:
    """
    ARM-optimized processor for EfficientSAM3 inference.
    
    This processor is designed to work on ARM devices (Apple Silicon with MPS or CPU)
    without requiring CUDA or Triton dependencies.
    
    Args:
        model: EfficientSAM3 model instance
        resolution: Input resolution (default: 1008)
        device: Device to use ("mps", "cpu", or "cuda")
        confidence_threshold: Minimum confidence score for detections (default: 0.5)
        use_fp16: Whether to use FP16 precision (default: False, not recommended for MPS)
        optimize_for_inference: Whether to apply inference optimizations (default: True)
    """

    def __init__(
        self,
        model,
        resolution: int = 1008,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        use_fp16: bool = False,
        optimize_for_inference: bool = True,
    ):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_fp16 = use_fp16 and device != "mps"  # FP16 not stable on MPS
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Apply inference optimizations
        if optimize_for_inference:
            self._optimize_model()
        
        # Setup image transforms
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        
        # Setup find stage for grounding
        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _optimize_model(self):
        """Apply ARM-specific optimizations to the model."""
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.model.eval()
        
        # Apply FP16 if requested and supported
        if self.use_fp16:
            self.model = self.model.half()

    def get_device_info(self) -> Dict[str, any]:
        """
        Get information about the current device configuration.
        
        Returns:
            Dict with device information
        """
        info = {
            "device": self.device,
            "resolution": self.resolution,
            "fp16_enabled": self.use_fp16,
            "confidence_threshold": self.confidence_threshold,
            "pytorch_version": torch.__version__,
        }
        
        if self.device == "mps":
            info["mps_available"] = torch.backends.mps.is_available()
        elif self.device == "cuda":
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        return info

    @torch.inference_mode()
    def set_image(self, image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], state: Optional[Dict] = None) -> Dict:
        """
        Set the image on which we want to do predictions.
        
        Args:
            image: Input image (PIL Image, torch Tensor, or numpy array)
            state: Optional state dict to update (creates new if None)
            
        Returns:
            Dict: State dictionary with image encoding
        """
        if state is None:
            state = {}

        # Get original dimensions
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image, tensor, or numpy array")

        # Convert and transform image
        image = v2.functional.to_image(image).to(self.device)
        
        # Apply FP16 if enabled
        if self.use_fp16:
            image = image.half()
        
        image = self.transform(image).unsqueeze(0)

        # Store original dimensions
        state["original_height"] = height
        state["original_width"] = width
        
        # Run backbone forward pass
        state["backbone_out"] = self.model.backbone.forward_image(image)
        
        # Handle SAM2 interactive predictor if available
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[Union[PIL.Image.Image, np.ndarray]], state: Optional[Dict] = None) -> Dict:
        """
        Set a batch of images for processing.
        
        Args:
            images: List of images (PIL Images or numpy arrays)
            state: Optional state dict to update
            
        Returns:
            Dict: State dictionary with batch encoding
        """
        if state is None:
            state = {}

        if not isinstance(images, list) or len(images) == 0:
            raise ValueError("Images must be a non-empty list")

        # Get original dimensions
        state["original_heights"] = []
        state["original_widths"] = []
        
        transformed_images = []
        for image in images:
            if isinstance(image, PIL.Image.Image):
                width, height = image.size
            elif isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                raise ValueError("Each image must be a PIL Image or numpy array")
            
            state["original_heights"].append(height)
            state["original_widths"].append(width)
            
            # Transform image
            img_tensor = v2.functional.to_image(image).to(self.device)
            if self.use_fp16:
                img_tensor = img_tensor.half()
            transformed_images.append(self.transform(img_tensor))
        
        # Stack into batch
        images_batch = torch.stack(transformed_images, dim=0)
        
        # Run backbone forward pass
        state["backbone_out"] = self.model.backbone.forward_image(images_batch)
        
        # Handle SAM2 interactive predictor if available
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict) -> Dict:
        """
        Set the text prompt and run inference.
        
        Args:
            prompt: Text prompt (e.g., "person. car. dog.")
            state: State dictionary from set_image
            
        Returns:
            Dict: Updated state with detection results
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        # Process text prompt
        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        
        # Update backbone outputs with text features
        state["backbone_out"].update(text_outputs)
        
        # Initialize geometric prompt if not present
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List[float], label: bool, state: Dict) -> Dict:
        """
        Add a box prompt and run inference.
        
        Args:
            box: Bounding box in [center_x, center_y, width, height] format, normalized to [0, 1]
            label: True for positive box, False for negative box
            state: State dictionary from set_image
            
        Returns:
            Dict: Updated state with detection results
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_geometric_prompt")

        # Set dummy text if not present
        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        # Initialize geometric prompt if not present
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Add box prompt
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict) -> None:
        """
        Remove all prompts and results from state.
        
        Args:
            state: State dictionary to reset
        """
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state: Optional[Dict] = None) -> Optional[Dict]:
        """
        Set the confidence threshold for detections.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
            state: Optional state to re-process with new threshold
            
        Returns:
            Updated state if provided, None otherwise
        """
        self.confidence_threshold = threshold
        
        if state is not None and "boxes" in state:
            # Re-run inference with new threshold
            return self._forward_grounding(state)
        
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict) -> Dict:
        """
        Run the grounding forward pass.
        
        Args:
            state: State dictionary with backbone outputs and prompts
            
        Returns:
            Dict: Updated state with detection results
        """
        # Run grounding model
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        # Extract predictions
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        
        # Compute scores
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        # Filter by confidence threshold
        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # Convert boxes to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # Scale boxes to original image size
        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=self.device)
        boxes = boxes * scale_fct[None, :]

        # Resize masks to original image size
        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        # Store results in state
        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        
        return state

    def __call__(self, image: Union[PIL.Image.Image, np.ndarray], text_prompt: str) -> Dict:
        """
        Convenience method for single-step inference.
        
        Args:
            image: Input image
            text_prompt: Text prompt for detection
            
        Returns:
            Dict: State with detection results
        """
        state = self.set_image(image)
        state = self.set_text_prompt(text_prompt, state)
        return state
