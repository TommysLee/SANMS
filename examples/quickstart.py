"""
SANMS Quick Start Example.

Demonstrates the core workflow:
  1. Generate synthetic detections (container + contained boxes)
  2. Run Greedy NMS (baseline)
  3. Run SANMS as post-NMS refinement
  4. Compare results

Requirements: numpy only (no torch/faiss/PIL needed).

Usage:
    python examples/quickstart.py
"""

import os, sys
import numpy as np

# Support execution from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sanms import greedy_nms, sanms
from sanms.box_ops import compute_iou, is_contained


def main():
    print("=" * 60)
    print("SANMS Quick Start")
    print("=" * 60)

    # --- 1. Simulate detector output ---
    # A "container" box (loose crop, moderate score) and a
    # "contained" box (tight crop, higher score).
    # NMS cannot suppress the contained box (IoU < threshold).
    boxes = np.array([
        [50, 50, 250, 250],    # Container: large, moderate score
        [90, 90, 160, 160],    # Contained: small, high score
        [55, 55, 245, 245],   # Overlap with container (IoU > 0.5)
    ], dtype=np.float64)

    scores = np.array([0.65, 0.92, 0.45])

    print("\nRaw detections:")
    for i in range(len(boxes)):
        print(f"  Box {i}: {boxes[i].tolist()}, score={scores[i]:.2f}")

    # Show that the contained box has low IoU with the container
    iou = compute_iou(boxes[0:1], boxes[1:2])[0, 0]
    contained = is_contained(boxes[1], boxes[0])
    print(f"\nIoU(container, contained) = {iou:.3f}  (below NMS threshold)")
    print(f"is_contained(contained, container) = {contained}")

    # --- 2. Run Greedy NMS ---
    nms_boxes, nms_scores = greedy_nms(boxes, scores, iou_thresh=0.5)
    print(f"\nAfter Greedy NMS (IoU thresh=0.5): {len(nms_boxes)} boxes kept")
    for i in range(len(nms_boxes)):
        print(f"  Box {i}: {nms_boxes[i].tolist()}, score={nms_scores[i]:.4f}")

    # The contained box survives NMS because its IoU with the container
    # is too low for suppression, even though it is geometrically inside.
    print("\nProblem: The contained box (tight crop, high score)")
    print("survives NMS and will be selected as top detection.")
    print("This hurts retrieval because tight crops produce noisy features.")

    # --- 3. Apply SANMS as post-NMS refinement ---
    final_boxes, final_scores = sanms(nms_boxes, nms_scores, alpha=0.75)
    print(f"\nAfter SANMS (alpha=0.75): {len(final_boxes)} boxes kept")
    for i in range(len(final_boxes)):
        print(f"  Box {i}: {final_boxes[i].tolist()}, score={final_scores[i]:.4f}")

    # --- 4. Result ---
    print("\n" + "=" * 60)
    print("Result: SANMS suppressed the contained box and inherited")
    print("its score to the container, which is better for retrieval.")
    print("=" * 60)


if __name__ == "__main__":
    main()
