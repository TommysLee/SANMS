"""
SANMS: Structure-Aware Non-Maximum Suppression -- Post-NMS Refinement

Resolves geometric containment relationships that IoU-based NMS cannot detect.
Designed as a refinement step applied AFTER any NMS variant, specifically for
subject detection in visual retrieval (image-to-image search) pipelines.

Algorithm (from paper Section 3.3):
  1. Compute areas, sort by area descending.
  2. For each box (area-descending order), check if it is contained in any
     already-kept box via direct coordinate comparison (with eps tolerance).
  3. If contained and area_ratio < alpha (gate), suppress it and inherit its
     score to the container (score inheritance).
  4. Return refined set sorted by final confidence.

Key properties:
  - Training-free (no learned parameters)
  - Single hyperparameter alpha (area-ratio gate, default 0.75)
  - O(N^2) worst-case, O(N) average (most small boxes suppressed early)
  - Composable with any NMS variant (Greedy, Soft, DIoU, Matrix, ...)
"""

import numpy as np
from .box_ops import compute_area


def sanms(
    boxes: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.75,
    eps: float = 1e-3,
    sort_criterion: str = "area",
) -> tuple:
    """Structure-Aware Post-NMS Refinement (SANMS).

    Args:
        boxes:  (N, 4) array of NMS-filtered boxes [x1, y1, x2, y2].
        scores: (N,) array of confidence scores.
        alpha:  area-ratio gate. Containment is suppressed only when
                area_inner / area_outer < alpha. Default 0.75.
                Set to 1.0 to disable the gate (pure containment suppression).
        eps:    float tolerance for coordinate comparison (pixels). Default 1e-3.
        sort_criterion: 'area' (default, recommended), 'diagonal',
                'perimeter', or 'confidence'. Area is the natural ordering
                for containment (see Section 3.4 of the paper).

    Returns:
        kept_boxes: (M, 4) refined boxes, sorted by confidence descending.
        kept_scores: (M,) refined scores.
    """
    n = len(boxes)
    if n <= 1:
        return boxes.copy(), scores.copy().astype(np.float64)

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).copy()

    areas = compute_area(boxes)  # (N,)

    # --- Step 1: Sort by chosen criterion (descending) ---
    if sort_criterion == "area":
        sort_values = areas
    elif sort_criterion == "diagonal":
        sort_values = (boxes[:, 2] - boxes[:, 0]) ** 2 + (boxes[:, 3] - boxes[:, 1]) ** 2
    elif sort_criterion == "perimeter":
        sort_values = 2 * ((boxes[:, 2] - boxes[:, 0]) + (boxes[:, 3] - boxes[:, 1]))
    elif sort_criterion == "confidence":
        sort_values = scores
    else:
        raise ValueError(f"Unknown sort_criterion: {sort_criterion}")

    order = np.argsort(sort_values)[::-1]  # descending

    # --- Step 2: Greedy containment resolution ---
    keep = []  # indices into original arrays

    for i in order:
        is_contained_flag = False

        for j in keep:
            # Containment check: is box[i] inside box[j]?
            # box[j] is the (larger) already-kept box.
            if (
                boxes[i, 0] >= boxes[j, 0] - eps
                and boxes[i, 1] >= boxes[j, 1] - eps
                and boxes[i, 2] <= boxes[j, 2] + eps
                and boxes[i, 3] <= boxes[j, 3] + eps
            ):
                # Area-ratio gate: suppress only if inner is significantly smaller
                if areas[j] > 0:
                    area_ratio = areas[i] / areas[j]
                else:
                    area_ratio = 0.0

                if area_ratio < alpha:
                    # Score inheritance: container gets max of its scores
                    scores[j] = max(scores[j], scores[i])
                    is_contained_flag = True
                    break  # a box can only be contained in one parent
                # If gate doesn't pass, keep both (dense-scene protection)

        if not is_contained_flag:
            keep.append(i)

    if len(keep) == 0:
        return np.zeros((0, 4)), np.zeros(0)

    keep = np.array(keep)
    kept_boxes = boxes[keep]
    kept_scores = scores[keep]

    # --- Step 3: Sort by final confidence descending ---
    final_order = np.argsort(kept_scores)[::-1]
    return kept_boxes[final_order], kept_scores[final_order]


def sanms_batch(
    boxes_list: list,
    scores_list: list,
    alpha: float = 0.75,
    eps: float = 1e-3,
) -> list:
    """Apply SANMS to a batch of images.

    Args:
        boxes_list: list of (N_i, 4) arrays.
        scores_list: list of (N_i,) arrays.
        alpha, eps: SANMS parameters.

    Returns:
        list of (kept_boxes, kept_scores) tuples.
    """
    results = []
    for boxes, scores in zip(boxes_list, scores_list):
        results.append(sanms(boxes, scores, alpha=alpha, eps=eps))
    return results
