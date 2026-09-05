"""
Evaluation metrics for detection and retrieval.

- COCO-style mAP (IoU 0.50:0.95)
- Recall@K (for SOP and general retrieval)
- Oxford-style mAP (for Oxford Buildings)
- Top-1 accuracy (for PRB)
- Expected Calibration Error (ECE)
"""

import numpy as np
from .box_ops import compute_iou


def coco_map(
    pred_boxes: list,
    pred_scores: list,
    pred_labels: list,
    gt_boxes: list,
    gt_labels: list,
    iou_thresholds: np.ndarray = None,
) -> dict:
    """COCO-style mAP evaluation.

    Args:
        pred_boxes: list of (N_i, 4) arrays, one per image.
        pred_scores: list of (N_i,) arrays.
        pred_labels: list of (N_i,) arrays (category IDs).
        gt_boxes: list of (M_i, 4) arrays, ground truth per image.
        gt_labels: list of (M_i,) arrays.
        iou_thresholds: array of IoU thresholds. Default: 0.50:0.05:0.95.

    Returns:
        dict with 'mAP', 'AP50', 'AP75', and per-threshold APs.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # Collect all unique labels
    all_labels = set()
    for labels in pred_labels:
        all_labels.update(labels.tolist())
    for labels in gt_labels:
        all_labels.update(labels.tolist())
    all_labels = sorted(all_labels)

    aps = {}
    aps_50 = {}
    aps_75 = {}

    for label in all_labels:
        # Gather predictions for this label
        all_pred_scores = []
        all_pred_tp_fp = []  # (score, tp/fp, image_idx) per threshold
        n_gt_total = 0

        # Collect per-image info
        for img_idx in range(len(gt_boxes)):
            gt_mask = gt_labels[img_idx] == label
            gt_img = gt_boxes[img_idx][gt_mask]
            n_gt_total += len(gt_img)

            pred_mask = pred_labels[img_idx] == label
            if len(pred_boxes[img_idx]) == 0:
                continue
            pred_img = pred_boxes[img_idx][pred_mask]
            pred_sco = pred_scores[img_idx][pred_mask]

            if len(pred_img) == 0:
                continue

            # Sort by score descending
            order = np.argsort(pred_sco)[::-1]
            pred_img = pred_img[order]
            pred_sco = pred_sco[order]

            if len(gt_img) == 0:
                # All predictions are FP
                for s in pred_sco:
                    all_pred_scores.append((s, img_idx, 0.0, -1, 0, None))
                continue

            # Match predictions to GT
            gt_matched = np.zeros(len(gt_img), dtype=bool)

            for p_idx in range(len(pred_img)):
                ious = compute_iou(pred_img[p_idx:p_idx+1], gt_img)[0]
                best_iou = ious.max() if len(ious) > 0 else 0
                best_gt = ious.argmax() if len(ious) > 0 else -1

                all_pred_scores.append(
                    (pred_sco[p_idx], img_idx, best_iou, best_gt, len(gt_img), gt_matched)
                )

        # Compute AP per IoU threshold
        ap_per_thresh = []

        for iou_thresh in iou_thresholds:
            # Re-evaluate TP/FP for this threshold
            tp_list = []
            fp_list = []
            score_list = []

            # Reset GT matching per threshold
            gt_matched_per_img = {}
            for img_idx in range(len(gt_boxes)):
                gt_mask = gt_labels[img_idx] == label
                gt_matched_per_img[img_idx] = np.zeros(gt_mask.sum(), dtype=bool)

            # Sort all predictions by score descending
            sorted_preds = sorted(all_pred_scores, key=lambda x: -x[0])

            for entry in sorted_preds:
                score = entry[0]
                img_idx = entry[1]
                best_iou = entry[2]
                best_gt = entry[3]
                n_gt_img = entry[4]
                gt_matched = entry[5]

                tp = False
                fp = True

                if best_iou >= iou_thresh and best_gt >= 0:
                    if not gt_matched_per_img[img_idx][best_gt]:
                        gt_matched_per_img[img_idx][best_gt] = True
                        tp = True
                        fp = False

                tp_list.append(tp)
                fp_list.append(fp)
                score_list.append(score)

            tp_arr = np.array(tp_list, dtype=float)
            fp_arr = np.array(fp_list, dtype=float)

            if n_gt_total == 0 or len(tp_arr) == 0:
                ap_per_thresh.append(0.0)
                continue

            # Accumulate
            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)
            recall = tp_cum / max(n_gt_total, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

            # Prepend (0, 1) for proper interpolation
            recall = np.concatenate([[0.0], recall])
            precision = np.concatenate([[1.0], precision])

            # AP = area under PR curve (using all-points interpolation)
            # COCO uses 101-point interpolation
            recall_pts = np.linspace(0, 1, 101)
            precision_interp = np.interp(recall_pts, recall, precision)
            ap = precision_interp.mean()
            ap_per_thresh.append(ap)

        aps[label] = np.mean(ap_per_thresh)
        # AP@0.50
        idx_50 = np.argmin(np.abs(iou_thresholds - 0.5))
        aps_50[label] = ap_per_thresh[idx_50]
        # AP@0.75
        idx_75 = np.argmin(np.abs(iou_thresholds - 0.75))
        aps_75[label] = ap_per_thresh[idx_75]

    # Average across all labels
    mAP = np.mean(list(aps.values())) if aps else 0.0
    mAP50 = np.mean(list(aps_50.values())) if aps_50 else 0.0
    mAP75 = np.mean(list(aps_75.values())) if aps_75 else 0.0

    return {
        "mAP": mAP,
        "AP50": mAP50,
        "AP75": mAP75,
        "per_class": aps,
        "per_class_50": aps_50,
    }


def recall_at_k(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k_values: list = [1, 10, 100],
) -> dict:
    """Compute Recall@K for image retrieval.

    Args:
        query_embeddings: (Q, D) L2-normalized query features.
        gallery_embeddings: (G, D) L2-normalized gallery features.
        query_labels: (Q,) class labels for queries.
        gallery_labels: (G,) class labels for gallery.
        k_values: list of K values.

    Returns:
        dict mapping 'R@K' to recall percentage.
    """
    # Cosine similarity (inner product on L2-normalized vectors)
    sim = query_embeddings @ gallery_embeddings.T  # (Q, G)

    results = {}
    for k in k_values:
        # For each query, get top-K gallery items
        top_k_indices = np.argpartition(-sim, k, axis=1)[:, :k]
        top_k_labels = gallery_labels[top_k_indices]

        # Check if any of the top-K matches the query label
        correct = (top_k_labels == query_labels[:, None]).any(axis=1)
        recall = correct.mean() * 100
        results[f"R@{k}"] = recall

    return results


def oxford_map(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    relevant_masks: np.ndarray,
    junk_masks: np.ndarray = None,
) -> float:
    """Oxford Buildings style mAP.

    Args:
        query_embeddings: (Q, D) L2-normalized.
        gallery_embeddings: (G, D) L2-normalized.
        relevant_masks: (Q, G) boolean, True if gallery item is relevant for query.
        junk_masks: (Q, G) boolean, True if gallery item is junk (ignored in ranking).

    Returns:
        mAP (mean average precision) in percent.
    """
    sim = query_embeddings @ gallery_embeddings.T  # (Q, G)

    aps = []
    for q_idx in range(len(query_embeddings)):
        scores = sim[q_idx].copy()

        # Remove junk from ranking
        if junk_masks is not None:
            scores[junk_masks[q_idx]] = -np.inf

        relevant = relevant_masks[q_idx]
        n_relevant = relevant.sum()

        if n_relevant == 0:
            continue

        # Sort by similarity descending
        order = np.argsort(-scores)

        # Compute AP
        cum_relevant = 0
        ap = 0.0

        for rank, g_idx in enumerate(order):
            if scores[g_idx] == -np.inf:
                break  # rest are junk

            if relevant[g_idx]:
                cum_relevant += 1
                precision = cum_relevant / (rank + 1)
                ap += precision

        ap /= n_relevant
        aps.append(ap)

    mAP = np.mean(aps) * 100 if aps else 0.0
    return mAP


def top1_accuracy(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """Top-1 retrieval accuracy.

    Args:
        query_embeddings: (Q, D) L2-normalized.
        gallery_embeddings: (G, D) L2-normalized.
        query_labels: (Q,)
        gallery_labels: (G,)

    Returns:
        Top-1 accuracy in percent.
    """
    sim = query_embeddings @ gallery_embeddings.T
    top1_indices = sim.argmax(axis=1)
    top1_labels = gallery_labels[top1_indices]
    correct = (top1_labels == query_labels).mean()
    return correct * 100


def expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) [26].

    Args:
        confidences: (N,) max confidence per detection.
        predictions: (N,) predicted class.
        labels: (N,) true class (or -1 if no GT match).
        n_bins: number of confidence bins.

    Returns:
        ECE value (lower = better calibrated).
    """
    # Filter out predictions without GT (for detection calibration)
    valid = labels >= 0
    if valid.sum() == 0:
        return 0.0

    confidences = confidences[valid]
    correct = (predictions[valid] == labels[valid]).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:  # include upper bound for last bin
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            continue

        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return ece


def failure_rate_by_scene(
    pred_boxes: list,
    pred_scores: list,
    gt_boxes: list,
    scene_types: list,
    iou_thresh: float = 0.5,
) -> dict:
    """Compute failure rate per scene type.

    A failure = no prediction matches any GT box in the image (IoU > thresh).

    Args:
        pred_boxes: list of (N_i, 4) arrays.
        pred_scores: list of (N_i,) arrays.
        gt_boxes: list of (M_i, 4) arrays.
        scene_types: list of strings, one per image.
        iou_thresh: IoU threshold for considering a match.

    Returns:
        dict mapping scene_type -> failure_rate (%).
    """
    from collections import defaultdict

    total = defaultdict(int)
    failures = defaultdict(int)

    for img_idx in range(len(gt_boxes)):
        scene = scene_types[img_idx]
        total[scene] += 1

        if len(gt_boxes[img_idx]) == 0:
            # No GT, not a failure
            continue

        if len(pred_boxes[img_idx]) == 0:
            failures[scene] += 1
            continue

        # Check if any prediction matches any GT
        ious = compute_iou(pred_boxes[img_idx], gt_boxes[img_idx])
        max_ious = ious.max(axis=1)  # best GT match for each pred
        best_match = max_ious.max()

        if best_match < iou_thresh:
            failures[scene] += 1

    rates = {}
    for scene in total:
        rates[scene] = failures[scene] / total[scene] * 100

    return rates
