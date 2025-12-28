# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
ARM-compatible box operations - standalone implementation without any sam3 dependencies.
"""

import torch


def box_cxcywh_to_xyxy(x):
    """Convert boxes from [cx, cy, w, h] to [x0, y0, x1, y1] format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xywh(x):
    """Convert boxes from [cx, cy, w, h] to [x, y, w, h] format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    """Convert boxes from [x, y, w, h] to [x0, y0, x1, y1] format."""
    x, y, w, h = x.unbind(-1)
    b = [(x), (y), (x + w), (y + h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_cxcywh(x):
    """Convert boxes from [x, y, w, h] to [cx, cy, w, h] format."""
    x, y, w, h = x.unbind(-1)
    b = [(x + 0.5 * w), (y + 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    """Convert boxes from [x0, y0, x1, y1] to [x, y, w, h] format."""
    x, y, X, Y = x.unbind(-1)
    b = [(x), (y), (X - x), (Y - y)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from [x0, y0, x1, y1] to [cx, cy, w, h] format."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """
    Compute area of boxes in [x0, y0, x1, y1] format.
    
    Args:
        boxes: Tensor of shape [N, 4]
        
    Returns:
        areas: Tensor of shape [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4] in [x0, y0, x1, y1] format
        boxes2: Tensor of shape [M, 4] in [x0, y0, x1, y1] format
        
    Returns:
        iou: Tensor of shape [N, M]
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Compute GIoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4] in [x0, y0, x1, y1] format
        boxes2: Tensor of shape [M, 4] in [x0, y0, x1, y1] format
        
    Returns:
        giou: Tensor of shape [N, M]
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
