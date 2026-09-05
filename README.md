<h1 align="center">SANMS: Structure-Aware Post-NMS Refinement for Subject Detection in Visual Retrieval</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-yellow.svg" alt="License: Apache" />
  <a href="tests/test_sanms.py"> <img src="https://img.shields.io/badge/tests-25%2F25-green.svg" alt="License: Apache" /> </a>
</p>

English | [简体中文](./README_cn.md)

SANMS (Structure-Aware Non-Maximum Suppression) is a **post-NMS refinement module** designed for **subject detection in image retrieval** (a.k.a. image-to-image search, or "search by image"). It resolves a geometric containment pattern that standard IoU-based NMS variants cannot detect: when a detector produces both a loose crop (good for retrieval) and a tight crop (bad for retrieval, but higher confidence) for the same object, NMS keeps the wrong box because their IoU is below the suppression threshold.

**SANMS is not a replacement for NMS.** It is applied *after* any NMS variant (Greedy, Soft, DIoU, Matrix) as a lightweight refinement step, using a single hyperparameter (alpha, the area-ratio gate).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Algorithm Overview](#algorithm-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Experiments](#experiments)
- [API Reference](#api-reference)
- [Paper](#paper)
- [Citation](#citation)
- [License](#license)

---

## Quick Start

```bash
# Clone
git clone https://github.com/TommysLee/SANMS.git
cd SANMS

# Install (core only: numpy + scipy)
pip install -r requirements.txt

# Run the demo (no dataset needed)
python examples/quickstart.py

# Run unit tests
python -m pytest tests/ -v
# or: python tests/test_sanms.py
```

**Minimal code example:**

```python
import numpy as np
from sanms import greedy_nms, sanms

# Simulated detector output: container box + contained box
boxes  = np.array([[50, 50, 250, 250], [90, 90, 160, 160]])
scores = np.array([0.65, 0.92])

# Step 1: Standard NMS (cannot suppress contained box due to low IoU)
nms_boxes, nms_scores = greedy_nms(boxes, scores, iou_thresh=0.5)

# Step 2: SANMS refinement (detects geometric containment, suppresses inner box)
final_boxes, final_scores = sanms(nms_boxes, nms_scores, alpha=0.75)
```

---

## Algorithm Overview

SANMS addresses a blind spot in IoU-based NMS: **geometric containment**.

When a detector outputs both a loose bounding box (container) and a tight bounding box (contained) for the same object, their IoU can be as low as 0.3, well below typical NMS thresholds (0.45-0.5). NMS therefore keeps both boxes. The contained box often has a higher confidence score, so it gets selected as the top detection, producing an over-tight crop that degrades downstream retrieval quality.

```
Container box (loose crop, score=0.65)     Contained box (tight crop, score=0.92)
  ┌──────────────────────────────┐            ┌──────────────────────────────┐
  │                              │            │   ┌──────────┐                │
  │                              │            │   │          │                │
  │                              │            │   │  IoU≈0.3 │                │
  │                              │            │   │          │                │
  │                              │            │   └──────────┘                │
  └──────────────────────────────┘            └──────────────────────────────┘
  IoU < 0.5: NMS keeps both                    is_contained = True: SANMS fixes this
```

**SANMS algorithm (3 steps):**

1. **Sort** remaining boxes by area descending (largest first)
2. **Containment check**: For each box, check if it is geometrically contained in any already-kept box (direct coordinate comparison, not IoU)
3. **Area-ratio gate**: If contained and `area(inner) / area(outer) < alpha`, suppress the inner box and inherit its score to the outer box (score inheritance)

**Key properties:**
- **Training-free**: No learned parameters
- **Single hyperparameter**: `alpha` (area-ratio gate, default 0.75)
- **Post-NMS**: Applied after any NMS variant
- **O(N^2) worst-case, O(N) average**
- **Runtime**: 10-15% of Soft-NMS execution time

---

## Installation

### Option A: From source (recommended for experiments)

```bash
git clone https://github.com/TommysLee/SANMS.git
cd SANMS
pip install -r requirements.txt          # Core: numpy, scipy
pip install -r requirements-optional.txt # Optional: torch, faiss, PIL, pycocotools
```

### Option B: pip install (package only)

```bash
pip install git+https://github.com/TommysLee/SANMS.git
```

### Option C: Editable install (for development)

```bash
git clone https://github.com/TommysLee/SANMS.git
cd SANMS
pip install -e ".[all]"
```

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| numpy   | Yes      | Core array operations |
| scipy   | Yes      | Numerical utilities |
| torch   | No       | ResNet-50 feature extraction (full pipeline) |
| torchvision | No  | Pretrained model weights |
| faiss-cpu | No     | Fast nearest-neighbor search |
| Pillow | No       | Image loading |
| pycocotools | No | COCO format evaluation |
| tqdm   | No       | Progress bars |

---

## Project Structure

```
SANMS/
|-- sanms/                          # Core Python package
|   |-- __init__.py                 # Public API exports
|   |-- sanms.py                    # SANMS algorithm (core contribution)
|   |-- box_ops.py                  # IoU, DIoU, containment, box voting
|   |-- nms_baselines.py            # Greedy NMS, Soft-NMS, DIoU-NMS, Matrix NMS
|   |-- metrics.py                  # COCO mAP, Recall@K, Oxford mAP, Top-1, ECE
|   |-- feature_extractor.py        # ResNet-50 feature extraction (optional)
|   `-- retriever.py                # Faiss-based retrieval engine (optional)
|-- experiments/                    # Experiment scripts
|   |-- pipeline.py                 # End-to-end pipeline orchestration
|   |-- run_all.py                  # Reproduce all tables (synthetic data)
|   |-- run_runtime.py              # Runtime benchmark (Table 12)
|   `-- results/                    # Pre-computed results
|       |-- all_results_v2.json     # Tables 1-11, 13 data
|       `-- runtime_with_std.json   # Table 12 data (100 trials)
|-- tests/
|   `-- test_sanms.py               # 25 unit tests (no external deps)
|-- examples/
|   `-- quickstart.py               # Minimal demo (numpy only)
|-- paper/
|   |-- SANMS Resolving Geometric Containment in Non-Maximum Suppression for Visual Retrieval_en.pdf  # Paper (English)
|-- pyproject.toml                  # Modern packaging config
|-- setup.py                        # Backward-compatible setup
|-- requirements.txt                # Core dependencies
|-- requirements-optional.txt       # Optional dependencies
|-- LICENSE
`-- .gitignore
```

---

## Usage

### Run the quick demo

```bash
python examples/quickstart.py
```

### Run unit tests

```bash
python -m pytest tests/ -v
# or directly:
python tests/test_sanms.py
```

### Reproduce experiments

All experiments use controlled synthetic data modeling the containment scenario. No external datasets are required.

```bash
# Reproduce Tables 1-11, 13 (takes ~2-5 minutes)
python -m experiments.run_all
# or: python experiments/run_all.py

# Reproduce Table 12: Runtime benchmark (takes ~2 minutes)
python -m experiments.run_runtime
# or: python experiments/run_runtime.py
```

Results are saved to `experiments/results/`.

### Use SANMS in your own code

```python
import numpy as np
from sanms import greedy_nms, soft_nms, diou_nms, matrix_nms, sanms

# Your detector output
boxes = np.array([...], dtype=np.float64)   # (N, 4): [x1, y1, x2, y2]
scores = np.array([...], dtype=np.float64)   # (N,)

# Step 1: Run any NMS variant
nms_boxes, nms_scores = greedy_nms(boxes, scores, iou_thresh=0.5)
# or: soft_nms(boxes, scores, sigma=0.5)
# or: diou_nms(boxes, scores, diou_thresh=0.5)
# or: matrix_nms(boxes, scores, iou_thresh=0.5)

# Step 2: Apply SANMS refinement
final_boxes, final_scores = sanms(nms_boxes, nms_scores, alpha=0.75)

# final_boxes, final_scores are sorted by confidence descending
```

### Use the full retrieval pipeline (requires optional deps)

```python
from experiments.pipeline import process_image_detection

# Full post-processing: NMS -> SANMS
final_boxes, final_scores = process_image_detection(
    raw_boxes, raw_scores,
    nms_method="greedy",       # or "soft", "diou", "matrix"
    post_refinement="sanms",   # or "none", "box_voting"
    alpha=0.75,
    eps=1e-3,
)
```

---

## Experiments

### Pre-computed Results

Pre-computed experiment results are included in `experiments/results/`:

| File | Contents | Tables |
|------|----------|--------|
| `all_results_v2.json` | COCO mAP, Recall@K, Oxford mAP, Top-1, ECE, ablations | Tables 1-11, 13 |
| `runtime_with_std.json` | Runtime (mean +/- std, 100 trials) | Table 12 |

### Reproduced Tables

| Table | Description | Script |
|-------|-------------|--------|
| 1-2   | COCO-style mAP (detection quality) | `run_all.py` |
| 3     | SOP Recall@K (retrieval quality) | `run_all.py` |
| 4     | Oxford Buildings mAP | `run_all.py` |
| 5     | PRB Top-1 accuracy | `run_all.py` |
| 7     | Alpha ablation | `run_all.py` |
| 8     | Sort criterion comparison | `run_all.py` |
| 9     | Epsilon tolerance | `run_all.py` |
| 10    | Score inheritance ablation | `run_all.py` |
| 11    | Scene-type failure rate | `run_all.py` |
| 12    | Runtime comparison | `run_runtime.py` |
| 13    | Expected Calibration Error | `run_all.py` |

### Runtime Benchmark

Measured on Intel Core i7-8550U (100 trials per configuration):

| Method | N=50 | N=100 | N=300 |
|--------|------|-------|-------|
| Greedy NMS | 3.4 +/- 1.0 ms | 6.3 +/- 1.7 ms | 15.5 +/- 2.7 ms |
| Soft-NMS | 6.6 +/- 1.7 ms | 16.6 +/- 6.9 ms | 79.2 +/- 14.2 ms |
| DIoU-NMS | 6.5 +/- 1.9 ms | 12.9 +/- 2.6 ms | 30.0 +/- 4.6 ms |
| Matrix NMS | 0.5 +/- 0.2 ms | 1.6 +/- 0.5 ms | 9.7 +/- 1.9 ms |
| Box Voting | 4.3 +/- 2.1 ms | 7.8 +/- 1.7 ms | 21.9 +/- 4.2 ms |
| **SANMS** | **0.9 +/- 0.3 ms** | **2.4 +/- 0.7 ms** | **10.8 +/- 1.9 ms** |

SANMS adds only 10-15% overhead relative to Soft-NMS.

---

## API Reference

### `sanms.sanms(boxes, scores, alpha=0.75, eps=1e-3, sort_criterion="area")`

The core SANMS algorithm.

**Parameters:**
- `boxes`: (N, 4) numpy array of [x1, y1, x2, y2]
- `scores`: (N,) numpy array of confidence scores
- `alpha`: float, area-ratio gate (default 0.75). A contained box is suppressed if `area(inner)/area(outer) < alpha`.
- `eps`: float, geometric tolerance in pixels (default 1e-3)
- `sort_criterion`: str, sorting order for processing. One of `"area"`, `"diagonal"`, `"perimeter"`, `"confidence"` (default `"area"`)

**Returns:** `(kept_boxes, kept_scores)` sorted by confidence descending.

### NMS Baselines

- `greedy_nms(boxes, scores, iou_thresh=0.5)`
- `soft_nms(boxes, scores, sigma=0.5, score_thresh=0.0, mode="gaussian")`
- `diou_nms(boxes, scores, diou_thresh=0.5)`
- `matrix_nms(boxes, scores, iou_thresh=0.5, sigma=0.5, mode="gaussian")`

### Box Operations

- `compute_area(boxes)` -> (N,) areas
- `compute_iou(boxes_a, boxes_b)` -> (N, M) IoU matrix
- `compute_diou(boxes_a, boxes_b)` -> (N, M) DIoU matrix
- `is_contained(box_inner, box_outer, eps=1e-3)` -> bool
- `box_voting(boxes, scores, iou_thresh=0.5)` -> (voted_boxes, voted_scores)

### Metrics

- `coco_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)` -> dict with mAP, AP50, AP75
- `recall_at_k(query_emb, gallery_emb, query_labels, gallery_labels, k_values)` -> dict
- `oxford_map(query_emb, gallery_emb, relevant_masks)` -> float
- `top1_accuracy(query_emb, gallery_emb, query_labels, gallery_labels)` -> float
- `expected_calibration_error(confidences, predictions, labels, n_bins=15)` -> float

---

## Paper

The full paper is included in [`paper/`](paper/):

- **English**: `SANMS Resolving Geometric Containment in Non-Maximum Suppression for Visual Retrieval_en.pdf`

### Key Findings

- SANMS improves detection mAP by suppressing geometrically contained boxes that NMS misses
- The improvement transfers to retrieval: Recall@K and Top-1 accuracy both increase when SANMS is applied after NMS
- SANMS has a single hyperparameter (alpha=0.75) and is robust across a wide range of values
- The algorithm is training-free and adds minimal computational overhead (10-15% of Soft-NMS)

### Limitations

The experiments in this repository use controlled synthetic data that models the containment scenario. Validation on real benchmarks (MS COCO, Stanford Online Products, Oxford Buildings) is needed for publication. The synthetic data ensures that the containment pattern is present, which may not reflect all real-world detector outputs.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sanms2026,
    author = {Li Zongjin},
    title = {SANMS: Structure-Aware Post-NMS Refinement for Subject Detection in Visual Retrieval},
    year = {2026},
    url = {https://github.com/TommysLee/SANMS},
    orcid = {0009-0009-9121-0268}
}
```

---

## License

[Apache-2.0](LICENSE)

## Author

**Li Zongjin**
- ORCID: [0009-0009-9121-0268](https://orcid.org/0009-0009-9121-0268)
- Email: li_zongjin@alumni.hust.edu.cn
- GitHub: [TommysLee](https://github.com/TommysLee)
