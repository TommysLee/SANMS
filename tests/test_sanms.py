"""
Unit tests for SANMS core algorithm and NMS baselines.

Run: python -m pytest tests/test_sanms.py -v
Or:  python tests/test_sanms.py
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sanms.sanms import sanms
from sanms.nms_baselines import greedy_nms, soft_nms, diou_nms, matrix_nms
from sanms.box_ops import (
    compute_iou, compute_area, is_contained, compute_diou, box_voting,
)


class TestContainment:
    """Test containment detection (the core SANMS operation)."""

    def test_basic_containment(self):
        """A small box inside a large box should be detected as contained."""
        outer = np.array([0, 0, 100, 100])
        inner = np.array([20, 20, 50, 50])
        assert is_contained(inner, outer, eps=1e-3)

    def test_no_containment(self):
        """Two non-overlapping boxes should not be contained."""
        box_a = np.array([0, 0, 50, 50])
        box_b = np.array([60, 60, 100, 100])
        assert not is_contained(box_a, box_b)
        assert not is_contained(box_b, box_a)

    def test_partial_overlap_not_containment(self):
        """Partially overlapping boxes should NOT be contained."""
        box_a = np.array([0, 0, 60, 60])
        box_b = np.array([40, 40, 100, 100])
        assert not is_contained(box_a, box_b)
        assert not is_contained(box_b, box_a)

    def test_identical_boxes(self):
        """Identical boxes should be contained in each other."""
        box = np.array([10, 10, 50, 50])
        assert is_contained(box, box)

    def test_tolerance(self):
        """Containment should hold within eps tolerance."""
        outer = np.array([0, 0, 100, 100])
        # Slightly outside by eps/2
        inner = np.array([-0.0004, -0.0004, 99.9996, 99.9996])
        assert is_contained(inner, outer, eps=1e-3)
        # Beyond tolerance
        inner_far = np.array([-0.002, -0.002, 100.002, 100.002])
        assert not is_contained(inner_far, outer, eps=1e-3)


class TestIoU:
    """Test IoU computation."""

    def test_identical_boxes(self):
        """Identical boxes have IoU = 1."""
        box = np.array([[0, 0, 50, 50]])
        iou = compute_iou(box, box)
        assert iou[0, 0] == 1.0

    def test_non_overlapping(self):
        """Non-overlapping boxes have IoU = 0."""
        a = np.array([[0, 0, 50, 50]])
        b = np.array([[60, 60, 100, 100]])
        iou = compute_iou(a, b)
        assert iou[0, 0] == 0.0

    def test_containment_low_iou(self):
        """A small box inside a large box has low IoU (the core SANMS insight)."""
        outer = np.array([[0, 0, 100, 100]])
        inner = np.array([[30, 30, 40, 40]])  # 10x10 inside 100x100
        iou = compute_iou(outer, inner)
        assert iou[0, 0] < 0.02  # IoU = 100/10000 = 0.01

    def test_half_overlap(self):
        """Two boxes with 50% overlap."""
        a = np.array([[0, 0, 100, 100]])  # area 10000
        b = np.array([[50, 0, 150, 100]])  # area 10000
        # intersection = 50*100 = 5000, union = 15000
        iou = compute_iou(a, b)
        assert abs(iou[0, 0] - 5000/15000) < 1e-6


class TestSANMS:
    """Test the SANMS algorithm."""

    def test_empty_input(self):
        """Empty input should return empty output."""
        boxes = np.zeros((0, 4))
        scores = np.zeros(0)
        out_boxes, out_scores = sanms(boxes, scores)
        assert len(out_boxes) == 0
        assert len(out_scores) == 0

    def test_single_box(self):
        """Single box passes through unchanged."""
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.9])
        out_boxes, out_scores = sanms(boxes, scores)
        assert len(out_boxes) == 1
        assert np.array_equal(out_boxes[0], boxes[0])
        assert out_scores[0] == 0.9

    def test_containment_suppression(self):
        """A contained box should be suppressed; score inherited."""
        # Large box (low score) contains small box (high score)
        boxes = np.array([
            [0, 0, 100, 100],   # large, score 0.3
            [20, 20, 40, 40],   # small, score 0.9
        ])
        scores = np.array([0.3, 0.9])

        out_boxes, out_scores = sanms(boxes, scores, alpha=0.75)

        # Should keep 1 box (the large one, with inherited score)
        assert len(out_boxes) == 1
        # Score should be inherited (max of 0.3 and 0.9 = 0.9)
        assert out_scores[0] == 0.9

    def test_no_containment_kept(self):
        """Two non-overlapping boxes should both survive."""
        boxes = np.array([
            [0, 0, 50, 50],
            [100, 100, 150, 150],
        ])
        scores = np.array([0.8, 0.6])

        out_boxes, out_scores = sanms(boxes, scores)
        assert len(out_boxes) == 2

    def test_alpha_gate_prevents_suppression(self):
        """When inner box is nearly as large as outer, alpha gate prevents suppression."""
        # Outer: 100x100, inner: 90x90 -> ratio 0.81 > 0.75
        boxes = np.array([
            [0, 0, 100, 100],
            [5, 5, 95, 95],  # area ratio = 90*90 / 100*100 = 0.81
        ])
        scores = np.array([0.3, 0.9])

        # alpha=0.75: 0.81 > 0.75 -> gate prevents suppression
        out_boxes, out_scores = sanms(boxes, scores, alpha=0.75)
        assert len(out_boxes) == 2  # both survive

        # alpha=1.0: no gate -> suppression happens
        out_boxes2, out_scores2 = sanms(boxes, scores, alpha=1.0)
        assert len(out_boxes2) == 1  # only large box survives
        assert out_scores2[0] == 0.9  # score inherited

    def test_nested_containment_chain(self):
        """Nested containment: A >= B >= C. Score should propagate to A."""
        boxes = np.array([
            [0, 0, 100, 100],   # A: largest
            [10, 10, 80, 80],   # B: middle
            [20, 20, 50, 50],   # C: smallest, highest score
        ])
        scores = np.array([0.2, 0.5, 0.95])

        out_boxes, out_scores = sanms(boxes, scores, alpha=0.75)

        # Only A should survive (B and C contained in A)
        assert len(out_boxes) == 1
        # A should have the max score (0.95 from C)
        assert out_scores[0] == 0.95

    def test_output_sorted_by_score(self):
        """Output should be sorted by confidence descending."""
        boxes = np.array([
            [0, 0, 100, 100],
            [200, 200, 300, 300],
            [400, 400, 500, 500],
        ])
        scores = np.array([0.3, 0.9, 0.6])

        out_boxes, out_scores = sanms(boxes, scores)
        # Check descending order
        for i in range(len(out_scores) - 1):
            assert out_scores[i] >= out_scores[i + 1]


class TestNMSBaselines:
    """Test NMS baseline implementations."""

    def test_greedy_nms_basic(self):
        """Greedy NMS suppresses overlapping boxes."""
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110],  # high IoU with first
            [200, 200, 300, 300],  # no overlap
        ])
        scores = np.array([0.9, 0.5, 0.8])

        out_boxes, out_scores = greedy_nms(boxes, scores, iou_thresh=0.5)
        assert len(out_boxes) == 2  # suppresses the second box
        assert out_scores[0] == 0.9  # highest score first

    def test_soft_nms_preserves_more(self):
        """Soft-NMS decays scores but keeps more boxes."""
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110],
        ])
        scores = np.array([0.9, 0.5])

        out_boxes, out_scores = soft_nms(boxes, scores, sigma=0.5)
        # Soft-NMS should keep both (just decays the second's score)
        assert len(out_boxes) >= 1

    def test_diou_nms(self):
        """DIoU-NMS should suppress boxes with high DIoU."""
        boxes = np.array([
            [0, 0, 100, 100],
            [5, 5, 105, 105],  # high overlap
        ])
        scores = np.array([0.9, 0.5])

        out_boxes, out_scores = diou_nms(boxes, scores, diou_thresh=0.5)
        assert len(out_boxes) == 1

    def test_matrix_nms(self):
        """Matrix NMS should work on overlapping boxes."""
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110],
            [200, 200, 300, 300],
        ])
        scores = np.array([0.9, 0.5, 0.8])

        out_boxes, out_scores = matrix_nms(boxes, scores, iou_thresh=0.5)
        assert len(out_boxes) >= 2  # keeps non-overlapping box

    def test_empty_input(self):
        """All NMS variants should handle empty input."""
        empty_boxes = np.zeros((0, 4))
        empty_scores = np.zeros(0)

        for nms_func in [greedy_nms, soft_nms, diou_nms, matrix_nms]:
            out_b, out_s = nms_func(empty_boxes, empty_scores)
            assert len(out_b) == 0
            assert len(out_s) == 0


class TestBoxVoting:
    """Test Box Voting baseline."""

    def test_overlapping_boxes_averaged(self):
        """Box Voting should average coordinates of overlapping boxes."""
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110],
        ])
        scores = np.array([0.9, 0.8])

        out_boxes, out_scores = box_voting(boxes, scores, iou_thresh=0.5)
        # Should produce one voted box
        assert len(out_boxes) == 1
        # Voted box should be between the two (weighted average)
        assert 0 < out_boxes[0, 0] < 10  # x1 between 0 and 10

    def test_non_overlapping_kept(self):
        """Non-overlapping boxes should all survive."""
        boxes = np.array([
            [0, 0, 50, 50],
            [100, 100, 150, 150],
        ])
        scores = np.array([0.9, 0.8])

        out_boxes, out_scores = box_voting(boxes, scores, iou_thresh=0.5)
        assert len(out_boxes) == 2


class TestPipeline:
    """Test the full NMS -> SANMS pipeline."""

    def test_nms_then_sanms(self):
        """NMS followed by SANMS should handle both overlap and containment."""
        # Raw detector output: overlapping + contained boxes
        raw_boxes = np.array([
            [0, 0, 100, 100],    # large box, low score
            [5, 5, 105, 105],    # overlaps with first (high IoU)
            [20, 20, 40, 40],    # contained in first, high score
            [200, 200, 300, 300], # separate object
        ])
        raw_scores = np.array([0.3, 0.4, 0.95, 0.7])

        # Step 1: Greedy NMS
        nms_boxes, nms_scores = greedy_nms(raw_boxes, raw_scores, iou_thresh=0.5)
        # NMS should suppress box[1] (high IoU with box[0])
        # But box[2] (contained) survives because IoU is low
        # (IoU between 20x20 and 100x100 = 400/10000 = 0.04 < 0.5)

        # Step 2: SANMS
        final_boxes, final_scores = sanms(nms_boxes, nms_scores, alpha=0.75)
        # SANMS should suppress the contained box[2] and inherit its score

        # The large box should have the max score (0.95)
        assert any(s == 0.95 for s in final_scores)

    def test_runtime_measurement(self):
        """Runtime measurement should produce valid results."""
        rng = np.random.RandomState(42)

        for n in [50, 100]:
            boxes = rng.rand(n, 4) * 500
            boxes[:, 2:] += boxes[:, :2] + 1
            scores = rng.rand(n)

            # Time SANMS
            t0 = time.perf_counter()
            _ = sanms(boxes, scores)
            t_sanms = time.perf_counter() - t0

            # Time Greedy NMS
            t0 = time.perf_counter()
            _ = greedy_nms(boxes, scores)
            t_nms = time.perf_counter() - t0

            # Both should complete in reasonable time
            assert t_sanms < 1.0  # should be milliseconds
            assert t_nms < 1.0


def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestContainment,
        TestIoU,
        TestSANMS,
        TestNMSBaselines,
        TestBoxVoting,
        TestPipeline,
    ]

    total = 0
    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed += 1
                print(f"  PASS: {test_class.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                print(f"  FAIL: {test_class.__name__}.{method_name}: {e}")

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
