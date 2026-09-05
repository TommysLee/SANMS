"""
NMS baseline implementations: Greedy NMS, Soft-NMS, DIoU-NMS, Matrix NMS.

All methods accept raw detector outputs (boxes + scores) and return
filtered boxes + scores. SANMS can be applied as a post-step to any of these.
"""

import numpy as np
from .box_ops import compute_iou, compute_diou


def greedy_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
) -> tuple:
    """Standard Greedy NMS.

    Suppresses boxes that have IoU > threshold with a higher-scoring box.

    Args:
        boxes:  (N, 4) [x1, y1, x2, y2]
        scores: (N,) confidence scores
        iou_thresh: IoU suppression threshold.

    Returns:
        kept_boxes, kept_scores (sorted by score descending)
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]
        ious = compute_iou(boxes[i:i+1], boxes[rest])[0]  # (len(rest),)

        order = rest[ious <= iou_thresh]

    keep = np.array(keep, dtype=int)
    final_order = np.argsort(scores[keep])[::-1]
    return boxes[keep][final_order], scores[keep][final_order]


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_thresh: float = 0.0,
    mode: str = "gaussian",
) -> tuple:
    """Soft-NMS [2].

    Instead of hard suppression, decays the scores of overlapping boxes
    using a Gaussian or linear function of IoU.

    Args:
        boxes: (N, 4)
        scores: (N,)
        sigma: variance parameter for Gaussian decay.
        score_thresh: minimum score to keep (default 0 = keep all).
        mode: 'gaussian' or 'linear'.

    Returns:
        kept_boxes, kept_scores (sorted by score descending)
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes = np.asarray(boxes, dtype=np.float64).copy()
    scores = np.asarray(scores, dtype=np.float64).copy()
    n = len(boxes)

    indices = list(range(n))

    keep = []
    keep_scores = []

    while indices:
        # Pick highest-scoring box
        local_scores = scores[indices]
        best_local = np.argmax(local_scores)
        best_idx = indices[best_local]

        if scores[best_idx] < score_thresh:
            break

        keep.append(best_idx)
        keep_scores.append(scores[best_idx])

        # Remove from indices
        indices.pop(best_local)

        if not indices:
            break

        # Compute IoU between best box and remaining
        rest_boxes = boxes[indices]
        ious = compute_iou(boxes[best_idx:best_idx+1], rest_boxes)[0]

        # Decay scores
        if mode == "gaussian":
            decay = np.exp(-(ious ** 2) / sigma)
        elif mode == "linear":
            decay = 1 - ious
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for k, idx in enumerate(indices):
            scores[idx] *= decay[k]

        # Filter low-scoring boxes
        new_indices = []
        for idx in indices:
            if scores[idx] >= score_thresh:
                new_indices.append(idx)
        indices = new_indices

    if len(keep) == 0:
        return np.zeros((0, 4)), np.zeros(0)

    keep = np.array(keep)
    keep_scores = np.array(keep_scores)

    # Sort by score descending
    final_order = np.argsort(keep_scores)[::-1]
    return boxes[keep][final_order], keep_scores[final_order]


def diou_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    diou_thresh: float = 0.5,
) -> tuple:
    """DIoU-NMS [5].

    Uses Distance-IoU instead of standard IoU for suppression.
    DIoU accounts for center-point distance, making it more effective
    for boxes with similar overlap but different centers.

    Args:
        boxes: (N, 4)
        scores: (N,)
        diou_thresh: DIoU suppression threshold.

    Returns:
        kept_boxes, kept_scores
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]
        dious = compute_diou(boxes[i:i+1], boxes[rest])[0]

        order = rest[dious <= diou_thresh]

    keep = np.array(keep, dtype=int)
    final_order = np.argsort(scores[keep])[::-1]
    return boxes[keep][final_order], scores[keep][final_order]


def matrix_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
    sigma: float = 0.5,
    mode: str = "gaussian",
) -> tuple:
    """Matrix NMS [7].

    Computes the full IoU matrix and decays scores based on the maximum
    IoU with any higher-scoring box. Operates in a single matrix operation
    rather than the iterative loop of Greedy/Soft-NMS.

    Args:
        boxes: (N, 4)
        scores: (N,)
        iou_thresh: threshold for considering boxes as overlapping.
        sigma: decay parameter.
        mode: 'gaussian' or 'linear'.

    Returns:
        kept_boxes, kept_scores
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).copy()
    n = len(boxes)

    order = np.argsort(scores)[::-1]

    # Full IoU matrix
    iou_matrix = compute_iou(boxes, boxes)  # (N, N)

    # For each box (in score order), find max IoU with any higher-scoring box
    decay = np.ones(n)
    for rank, i in enumerate(order):
        if rank == 0:
            continue
        higher = order[:rank]
        max_iou = iou_matrix[i, higher].max() if len(higher) > 0 else 0.0

        if max_iou > iou_thresh:
            if mode == "gaussian":
                decay[i] = np.exp(-(max_iou ** 2) / sigma)
            elif mode == "linear":
                decay[i] = 1 - max_iou
            else:
                raise ValueError(f"Unknown mode: {mode}")

    scores *= decay

    # Filter by threshold
    mask = scores > 0.01
    kept_indices = np.where(mask)[0]
    kept_indices = kept_indices[np.argsort(scores[kept_indices])[::-1]]

    return boxes[kept_indices], scores[kept_indices]
