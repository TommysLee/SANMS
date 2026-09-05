"""
Unified retrieval pipeline: Detection -> NMS -> SANMS -> Feature -> Retrieval.

This module is the end-to-end pipeline described in the paper (Section 5.1
and Appendix D.1). The core functions (process_image_detection, measure_runtime)
require only numpy. The full retrieval pipeline (run_retrieval_pipeline) requires
PIL, torch, and faiss; these imports are deferred to function call time.
"""

import time
import numpy as np
from typing import Optional, List, Tuple

from sanms import (
    greedy_nms,
    soft_nms,
    diou_nms,
    matrix_nms,
    sanms,
    box_voting,
)


# --- NMS registry ---
NMS_REGISTRY = {
    "greedy": greedy_nms,
    "soft": soft_nms,
    "diou": diou_nms,
    "matrix": matrix_nms,
}


def run_nms(method: str, boxes, scores, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Run a specific NMS variant."""
    if method not in NMS_REGISTRY:
        raise ValueError(f"Unknown NMS method: {method}. Available: {list(NMS_REGISTRY.keys())}")
    return NMS_REGISTRY[method](boxes, scores, **kwargs)


def run_post_refinement(
    method: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.75,
    eps: float = 1e-3,
    iou_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run post-NMS refinement (SANMS, Box Voting, or none)."""
    if method == "none":
        return boxes, scores
    elif method == "sanms":
        return sanms(boxes, scores, alpha=alpha, eps=eps)
    elif method == "box_voting":
        return box_voting(boxes, scores, iou_thresh=iou_thresh)
    else:
        raise ValueError(f"Unknown post-refinement method: {method}")


def process_image_detection(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    nms_method: str = "greedy",
    post_refinement: str = "none",
    alpha: float = 0.75,
    eps: float = 1e-3,
    nms_kwargs: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full detection post-processing: NMS -> (optional) refinement.

    Args:
        raw_boxes: (N, 4) raw detector output.
        raw_scores: (N,) raw confidence scores.
        nms_method: 'greedy', 'soft', 'diou', 'matrix'.
        post_refinement: 'none', 'sanms', 'box_voting'.
        alpha: SANMS area-ratio gate.
        eps: SANMS float tolerance.
        nms_kwargs: extra kwargs for NMS.

    Returns:
        final_boxes, final_scores (sorted by confidence descending).
    """
    if nms_kwargs is None:
        nms_kwargs = {}

    # Step 1: NMS
    nms_boxes, nms_scores = run_nms(nms_method, raw_boxes, raw_scores, **nms_kwargs)

    # Step 2: Post-refinement
    final_boxes, final_scores = run_post_refinement(
        post_refinement, nms_boxes, nms_scores, alpha=alpha, eps=eps
    )

    return final_boxes, final_scores


def run_retrieval_pipeline(
    image_paths: List[str],
    labels: np.ndarray,
    is_query: np.ndarray,
    raw_detections: List[Tuple[np.ndarray, np.ndarray]],
    extractor,
    nms_method: str = "greedy",
    post_refinement: str = "none",
    alpha: float = 0.75,
    eps: float = 1e-3,
    nms_kwargs: dict = None,
    k_values: list = [1, 10, 100],
    batch_size: int = 32,
    verbose: bool = True,
) -> dict:
    """Full end-to-end retrieval pipeline.

    For each image:
      1. Run NMS on raw detections.
      2. Run SANMS (if enabled) on NMS output.
      3. Select top-scoring box, crop image.
      4. Extract 512-dim L2-normalized embedding.
    Then build gallery index and evaluate retrieval.

    Requires: PIL, torch, faiss (install optional dependencies).

    Args:
        image_paths: list of image file paths.
        labels: (N,) class labels.
        is_query: (N,) boolean, True for query images.
        raw_detections: list of (boxes, scores) tuples.
        extractor: FeatureExtractor instance.
        nms_method, post_refinement, alpha, eps: post-processing config.
        k_values: Recall@K values.
        batch_size: feature extraction batch size.

    Returns:
        dict with retrieval metrics and timing.
    """
    # Deferred imports for optional dependencies
    from PIL import Image
    from sanms.feature_extractor import extract_top_box_feature
    from sanms.retriever import Retriever

    if nms_kwargs is None:
        nms_kwargs = {}

    n_images = len(image_paths)
    all_embeddings = np.zeros((n_images, extractor.embedding_dim), dtype=np.float32)

    t0 = time.time()

    for i in range(n_images):
        img = Image.open(image_paths[i]).convert("RGB")
        raw_boxes, raw_scores = raw_detections[i]

        final_boxes, final_scores = process_image_detection(
            raw_boxes,
            raw_scores,
            nms_method=nms_method,
            post_refinement=post_refinement,
            alpha=alpha,
            eps=eps,
            nms_kwargs=nms_kwargs,
        )

        embedding = extract_top_box_feature(img, final_boxes, final_scores, extractor)
        all_embeddings[i] = embedding

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_images - i - 1)
            print(f"  Processed {i+1}/{n_images} ({elapsed:.1f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0

    query_mask = is_query
    gallery_mask = ~is_query

    query_embeddings = all_embeddings[query_mask]
    query_labels = labels[query_mask]
    gallery_embeddings = all_embeddings[gallery_mask]
    gallery_labels = labels[gallery_mask]

    retriever = Retriever(embedding_dim=extractor.embedding_dim, use_faiss=True)
    retriever.build_index(gallery_embeddings, gallery_labels)

    metrics = retriever.recall_at_k(query_embeddings, query_labels, k_values)
    metrics["top1"] = retriever.top1_accuracy(query_embeddings, query_labels)
    metrics["time_seconds"] = elapsed
    metrics["n_images"] = n_images

    return metrics


def measure_runtime(
    n_values: list = [50, 100, 300],
    n_trials: int = 100,
) -> dict:
    """Measure runtime of NMS/SANMS for different box counts.

    Generates random boxes and measures wall-clock time.

    Args:
        n_values: list of N (number of boxes).
        n_trials: number of trials per measurement.

    Returns:
        dict mapping method name -> {N: time_ms}.
    """
    rng = np.random.RandomState(42)
    results = {}

    methods = {
        "greedy_nms": lambda b, s: greedy_nms(b, s, 0.5),
        "soft_nms": lambda b, s: soft_nms(b, s, sigma=0.5),
        "diou_nms": lambda b, s: diou_nms(b, s, 0.5),
        "matrix_nms": lambda b, s: matrix_nms(b, s, 0.5),
        "sanms": lambda b, s: sanms(b, s, alpha=0.75),
        "box_voting": lambda b, s: box_voting(b, s, 0.5),
    }

    for n in n_values:
        boxes = rng.rand(n, 4) * 500
        boxes[:, 2:] += boxes[:, :2] + 1
        scores = rng.rand(n)

        for name, func in methods.items():
            times = []
            for _ in range(n_trials):
                t0 = time.perf_counter()
                _ = func(boxes.copy(), scores.copy())
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            if name not in results:
                results[name] = {}
            results[name][n] = np.mean(times)

        print(f"  N={n}: " + ", ".join(f"{m}={results[m][n]:.2f}ms" for m in methods))

    return results
