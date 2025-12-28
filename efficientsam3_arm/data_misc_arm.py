# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
ARM-compatible data utilities - standalone implementation without triton dependencies.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch


MyTensor = Union[torch.Tensor, List[Any]]


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    assert (
        input.shape[0] != 0 or input.shape[1] != 0
    ), "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(
            input.transpose(0, 1), size, scale_factor, mode, align_corners
        ).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )


@dataclass
class FindStage:
    """Data structure for find stage inputs."""
    img_ids: MyTensor
    text_ids: MyTensor
    input_boxes: MyTensor
    input_boxes_mask: MyTensor
    input_boxes_label: MyTensor
    input_points: MyTensor
    input_points_mask: MyTensor
    object_ids: Optional[List[List]] = None


@dataclass
class GeometricPrompt:
    """Container for geometric prompts (boxes and points)."""
    boxes: Optional[torch.Tensor] = None
    boxes_mask: Optional[torch.Tensor] = None
    boxes_label: Optional[torch.Tensor] = None
    points: Optional[torch.Tensor] = None
    points_mask: Optional[torch.Tensor] = None
    
    def append_boxes(self, boxes: torch.Tensor, labels: torch.Tensor):
        """Append boxes to the prompt."""
        if self.boxes is None:
            self.boxes = boxes
            self.boxes_label = labels
            # Create mask (all ones for valid boxes)
            self.boxes_mask = torch.ones(boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
        else:
            self.boxes = torch.cat([self.boxes, boxes], dim=1)
            self.boxes_label = torch.cat([self.boxes_label, labels], dim=1)
            new_mask = torch.ones(boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
            self.boxes_mask = torch.cat([self.boxes_mask, new_mask], dim=1)
    
    def append_points(self, points: torch.Tensor, labels: torch.Tensor):
        """Append points to the prompt."""
        if self.points is None:
            self.points = points
            self.points_mask = torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
        else:
            self.points = torch.cat([self.points, points], dim=1)
            new_mask = torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
            self.points_mask = torch.cat([self.points_mask, new_mask], dim=1)
