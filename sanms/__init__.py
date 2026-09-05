"""
SANMS: Structure-Aware Non-Maximum Suppression.

A post-NMS refinement module for subject detection in visual retrieval.
"""

__version__ = "1.0.0"
__author__ = "Li Zongjin"
__email__ = "li_zongjin@alumni.hust.edu.cn"
__license__ = "Apache-2.0"

from .box_ops import (
    compute_area,
    compute_iou,
    compute_iou_matrix,
    compute_diou,
    is_contained,
    is_contained_batch,
    box_voting,
)
from .sanms import sanms
from .nms_baselines import greedy_nms, soft_nms, diou_nms, matrix_nms
from .metrics import (
    coco_map,
    recall_at_k,
    oxford_map,
    top1_accuracy,
    expected_calibration_error,
    failure_rate_by_scene,
)

__all__ = [
    # Box operations
    "compute_area",
    "compute_iou",
    "compute_iou_matrix",
    "compute_diou",
    "is_contained",
    "is_contained_batch",
    "box_voting",
    # SANMS
    "sanms",
    # NMS baselines
    "greedy_nms",
    "soft_nms",
    "diou_nms",
    "matrix_nms",
    # Metrics
    "coco_map",
    "recall_at_k",
    "oxford_map",
    "top1_accuracy",
    "expected_calibration_error",
    "failure_rate_by_scene",
]
