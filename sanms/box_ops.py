"""
Bounding box geometric operations.

All boxes are in [x1, y1, x2, y2] format (top-left, bottom-right).
"""

import numpy as np


def compute_area(boxes: np.ndarray) -> np.ndarray:
    """Compute area of each box.

    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].
    Returns:
        (N,) array of areas.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes.

    Args:
        boxes_a: (N, 4)
        boxes_b: (M, 4)
    Returns:
        (N, M) IoU matrix.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    # (N, 1) vs (1, M) -> broadcast
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)  # (N, M)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    intersection = inter_w * inter_h

    area_a = compute_area(boxes_a)  # (N,)
    area_b = compute_area(boxes_b)  # (M,)

    union = area_a[:, None] + area_b[None, :] - intersection
    iou = np.where(union > 0, intersection / np.where(union == 0, 1, union), 0.0)
    return iou


def compute_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Self IoU matrix (N, N)."""
    return compute_iou(boxes, boxes)


def is_contained(
    box_inner: np.ndarray, box_outer: np.ndarray, eps: float = 1e-3
) -> bool:
    """Check if box_inner is geometrically contained in box_outer.

    Uses direct coordinate comparison with tolerance, NOT IoU.
    This is the fundamental operation distinguishing SANMS from overlap-based methods.

    Args:
        box_inner: (4,) array [x1, y1, x2, y2]
        box_outer: (4,) array [x1, y1, x2, y2]
        eps: float tolerance in pixel units.
    Returns:
        True if box_inner is inside box_outer (within eps).
    """
    return (
        box_inner[0] >= box_outer[0] - eps
        and box_inner[1] >= box_outer[1] - eps
        and box_inner[2] <= box_outer[2] + eps
        and box_inner[3] <= box_outer[3] + eps
    )


def is_contained_batch(
    boxes_inner: np.ndarray, box_outer: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """Vectorized containment check: which of boxes_inner are inside box_outer?

    Args:
        boxes_inner: (N, 4)
        box_outer: (4,)
        eps: tolerance
    Returns:
        (N,) boolean array.
    """
    if len(boxes_inner) == 0:
        return np.array([], dtype=bool)
    return (
        (boxes_inner[:, 0] >= box_outer[0] - eps)
        & (boxes_inner[:, 1] >= box_outer[1] - eps)
        & (boxes_inner[:, 2] <= box_outer[2] + eps)
        & (boxes_inner[:, 3] <= box_outer[3] + eps)
    )


def compute_diou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Distance-IoU between two sets of boxes.

    DIoU = IoU - (rho^2(b_a, b_b)) / (c^2)
    where rho = center-point distance, c = diagonal of smallest enclosing box.

    Args:
        boxes_a: (N, 4)
        boxes_b: (M, 4)
    Returns:
        (N, M) DIoU matrix.
    """
    iou = compute_iou(boxes_a, boxes_b)

    # Centers
    center_a_x = (boxes_a[:, 0] + boxes_a[:, 2]) / 2  # (N,)
    center_a_y = (boxes_a[:, 1] + boxes_a[:, 3]) / 2
    center_b_x = (boxes_b[:, 0] + boxes_b[:, 2]) / 2  # (M,)
    center_b_y = (boxes_b[:, 1] + boxes_b[:, 3]) / 2

    rho2 = (center_a_x[:, None] - center_b_x[None, :]) ** 2 + (
        center_a_y[:, None] - center_b_y[None, :]
    ) ** 2

    # Enclosing box diagonal
    enc_x1 = np.minimum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    enc_y1 = np.minimum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    enc_x2 = np.maximum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    enc_y2 = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
    c2 = np.where(c2 > 0, c2, 1)

    diou = iou - rho2 / c2
    return diou


def box_voting(
    boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5
) -> tuple:
    """Box Voting [13] post-NMS refinement.

    Averages coordinates of overlapping boxes (IoU > threshold).
    Unlike SANMS, cannot address containment (contained boxes have ~0 IoU).

    Args:
        boxes: (N, 4) NMS-filtered boxes
        scores: (N,) confidence scores
        iou_thresh: IoU threshold for voting.
    Returns:
        (voted_boxes, voted_scores)
    """
    if len(boxes) == 0:
        return boxes, scores

    order = np.argsort(scores)[::-1]
    keep = []
    voted_boxes = []
    voted_scores = []

    suppressed = np.zeros(len(boxes), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue

        # Find overlapping boxes
        ious = compute_iou(boxes[idx:idx+1], boxes)[0]  # (N,)
        overlap_mask = ious > iou_thresh
        overlap_mask[idx] = True  # include self

        # Weighted average of coordinates
        overlap_boxes = boxes[overlap_mask]
        overlap_scores = scores[overlap_mask]
        weights = overlap_scores / (overlap_scores.sum() + 1e-8)

        voted_box = (overlap_boxes * weights[:, None]).sum(axis=0)
        voted_boxes.append(voted_box)
        voted_scores.append(scores[idx])

        suppressed[overlap_mask] = True

    return np.array(voted_boxes), np.array(voted_scores)
