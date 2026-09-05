"""
Synthetic experiments for SANMS paper.

Models the specific containment scenario SANMS is designed to solve:
- Container box (loose crop, moderate score, good for retrieval)
- Contained box (tight crop, high score, bad for retrieval)
- NMS cannot suppress contained box (low IoU with container)
- SANMS detects containment and fixes it

Reproduces Tables 1-11 and 13 from the paper using controlled synthetic data.
Results are saved to experiments/results/all_results.json.

Usage:
    python -m experiments.run_all           # from project root
    python experiments/run_all.py          # direct execution
"""

import os, sys, json, time
import numpy as np

# Support both `python -m experiments.run_all` and direct execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sanms import greedy_nms, soft_nms, diou_nms, matrix_nms, sanms, box_voting
from sanms.box_ops import compute_iou, compute_area
from sanms.metrics import (
    coco_map, recall_at_k, oxford_map, top1_accuracy,
    expected_calibration_error, failure_rate_by_scene,
)
from experiments.pipeline import process_image_detection


def gen_data(n_img=200, n_cls=5, img_size=640,
             containment_rate=0.45, fp_rate=2, seed=42):
    """Generate synthetic data modeling the containment scenario.

    For each GT box:
    - Good detection: IoU 0.65-0.85 with GT, score 0.55-0.75 (the 'container')
    - Contained detection: inside the good detection, IoU 0.45-0.65 with GT,
      score 0.80-0.95 (the 'tight crop' - higher score but worse for retrieval)
    - Optional partial overlap: IoU 0.3-0.5, score 0.3-0.5
    - Random FPs: score 0.1-0.4
    """
    rng = np.random.RandomState(seed)
    scenes = ["sparse", "moderate", "dense", "cluttered"]
    gt_boxes_list, gt_labels_list = [], []
    raw_boxes_list, raw_scores_list, raw_labels_list = [], [], []
    scene_types = []

    for img_idx in range(n_img):
        scene = rng.choice(scenes, p=[0.3, 0.35, 0.2, 0.15])
        scene_types.append(scene)
        n_gt = {"sparse": (1, 3), "moderate": (2, 4),
                "dense": (3, 6), "cluttered": (4, 8)}[scene]
        n_gt = rng.randint(*n_gt)

        gt_boxes, gt_labels = [], []
        raw_boxes, raw_scores, raw_labels = [], [], []

        for _ in range(n_gt):
            cls_id = rng.randint(n_cls)
            w = rng.uniform(80, 220)
            h = rng.uniform(80, 220)
            x1 = rng.uniform(10, img_size - w - 10)
            y1 = rng.uniform(10, img_size - h - 10)
            x2, y2 = x1 + w, y1 + h
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(cls_id)

            # Good detection (container): moderate IoU, moderate score
            jitter = rng.uniform(-12, 12, 4)
            good_box = np.array([x1, y1, x2, y2]) + jitter
            good_box[0] = max(0, good_box[0])
            good_box[1] = max(0, good_box[1])
            good_box[2] = min(img_size, good_box[2])
            good_box[3] = min(img_size, good_box[3])
            good_box[2] = max(good_box[2], good_box[0] + 30)
            good_box[3] = max(good_box[3], good_box[1] + 30)
            raw_boxes.append(good_box)
            raw_scores.append(rng.uniform(0.55, 0.75))
            raw_labels.append(cls_id)

            # Contained detection: inside the good box, higher score
            if rng.rand() < containment_rate:
                gx1, gy1, gx2, gy2 = good_box
                gw, gh = gx2 - gx1, gy2 - gy1
                sw = gw * rng.uniform(0.4, 0.7)
                sh = gh * rng.uniform(0.4, 0.7)
                ox = rng.uniform(0, max(gw - sw, 1))
                oy = rng.uniform(0, max(gh - sh, 1))
                inner_box = [gx1 + ox, gy1 + oy, gx1 + ox + sw, gy1 + oy + sh]
                raw_boxes.append(inner_box)
                raw_scores.append(rng.uniform(0.80, 0.95))
                raw_labels.append(cls_id)

            # Partial overlap (NMS-suppressible)
            if rng.rand() < 0.3:
                shift = rng.uniform(15, 50)
                pb = np.array([x1+shift, y1+shift*0.3, x2-shift*0.2, y2-shift*0.2])
                pb = np.clip(pb, 0, img_size)
                if pb[2] > pb[0]+20 and pb[3] > pb[1]+20:
                    raw_boxes.append(pb)
                    raw_scores.append(rng.uniform(0.3, 0.55))
                    raw_labels.append(cls_id)

        # False positives
        for _ in range(rng.randint(0, fp_rate + 1)):
            w_fp = rng.uniform(40, 180)
            h_fp = rng.uniform(40, 180)
            x1f = rng.uniform(0, img_size - w_fp)
            y1f = rng.uniform(0, img_size - h_fp)
            raw_boxes.append([x1f, y1f, x1f+w_fp, y1f+h_fp])
            raw_scores.append(rng.uniform(0.1, 0.4))
            raw_labels.append(rng.randint(n_cls))

        gt_boxes_list.append(np.array(gt_boxes, dtype=np.float64) if gt_boxes else np.zeros((0, 4)))
        gt_labels_list.append(np.array(gt_labels, dtype=int))
        raw_boxes_list.append(np.array(raw_boxes, dtype=np.float64) if raw_boxes else np.zeros((0, 4)))
        raw_scores_list.append(np.array(raw_scores, dtype=np.float64) if raw_scores else np.zeros(0))
        raw_labels_list.append(np.array(raw_labels, dtype=int) if raw_labels else np.array([], dtype=int))

    return {
        "gt_boxes": gt_boxes_list, "gt_labels": gt_labels_list,
        "raw_boxes": raw_boxes_list, "raw_scores": raw_scores_list,
        "raw_labels": raw_labels_list, "scene_types": scene_types,
        "n_classes": n_cls,
    }


def gen_features(data, n_dim=64, noise_good=0.15, noise_contained=0.6,
                 noise_bad=0.9, seed=42):
    """Generate features that model detection quality impact on retrieval.

    Key: contained boxes (tight crops) produce NOISY features,
    while container boxes (loose crops) produce CLEAN features.
    This models the real-world effect where over-tight cropping
    loses context needed for good feature extraction.
    """
    rng = np.random.RandomState(seed)
    n_cls = data["n_classes"]
    centroids = rng.randn(n_cls, n_dim).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    n_img = len(data["gt_boxes"])
    is_query = np.zeros(n_img, dtype=bool)
    is_query[rng.choice(n_img, n_img // 2, replace=False)] = True

    configs = [
        ("greedy", "none", {"iou_thresh": 0.5}),
        ("greedy", "sanms", {"iou_thresh": 0.5}),
        ("greedy", "box_voting", {"iou_thresh": 0.5}),
        ("soft", "none", {"sigma": 0.5}),
        ("soft", "sanms", {"sigma": 0.5}),
        ("diou", "none", {"diou_thresh": 0.5}),
        ("diou", "sanms", {"diou_thresh": 0.5}),
        ("matrix", "none", {"iou_thresh": 0.5}),
        ("matrix", "sanms", {"iou_thresh": 0.5}),
    ]

    all_feats = {}
    for nms_m, post_r, nms_kw in configs:
        q_emb, q_lbl, g_emb, g_lbl = [], [], [], []

        for i in range(n_img):
            rb = data["raw_boxes"][i]
            rs = data["raw_scores"][i]
            gb = data["gt_boxes"][i]
            gl = data["gt_labels"][i]

            fb, fs = process_image_detection(rb, rs, nms_method=nms_m,
                                              post_refinement=post_r,
                                              alpha=0.75, nms_kwargs=nms_kw)

            if len(fb) > 0 and len(gb) > 0:
                top = fb[0]
                ious = compute_iou(top[np.newaxis], gb)[0]
                best_iou = ious.max()
                best_gt = ious.argmax()
                true_label = gl[best_gt]

                top_area = (top[2]-top[0]) * (top[3]-top[1])
                gt_area = (gb[best_gt][2]-gb[best_gt][0]) * (gb[best_gt][3]-gb[best_gt][1])
                area_ratio = top_area / max(gt_area, 1)

                if area_ratio < 0.6:
                    noise = noise_contained
                elif best_iou > 0.7:
                    noise = noise_good
                else:
                    noise = noise_bad
            else:
                best_iou = 0.0
                true_label = data["raw_labels"][i][0] if len(data["raw_labels"][i]) > 0 else 0
                noise = noise_bad

            feat = centroids[true_label] + rng.randn(n_dim) * noise
            feat /= np.linalg.norm(feat) + 1e-8

            if is_query[i]:
                q_emb.append(feat)
                q_lbl.append(true_label)
            else:
                g_emb.append(feat)
                g_lbl.append(true_label)

        all_feats[(nms_m, post_r)] = {
            "query_embeddings": np.array(q_emb, dtype=np.float32),
            "gallery_embeddings": np.array(g_emb, dtype=np.float32),
            "query_labels": np.array(q_lbl),
            "gallery_labels": np.array(g_lbl),
        }

    return all_feats


def run_coco_exp(data, nms_m, post_r, nms_kw, alpha=0.75):
    pred_b, pred_s, pred_l, gt_b, gt_l = [], [], [], [], []
    for i in range(len(data["gt_boxes"])):
        fb, fs = process_image_detection(
            data["raw_boxes"][i], data["raw_scores"][i],
            nms_method=nms_m, post_refinement=post_r,
            alpha=alpha, nms_kwargs=nms_kw)
        pred_b.append(fb)
        pred_s.append(fs)
        pred_l.append(np.ones(len(fb), dtype=int))
        gt_b.append(data["gt_boxes"][i])
        gt_l.append(data["gt_labels"][i])
    return coco_map(pred_b, pred_s, pred_l, gt_b, gt_l)


def run_ece_exp(data, nms_m, post_r, nms_kw, alpha=0.75):
    all_conf, all_pred, all_lbl = [], [], []
    for i in range(len(data["gt_boxes"])):
        fb, fs = process_image_detection(
            data["raw_boxes"][i], data["raw_scores"][i],
            nms_method=nms_m, post_refinement=post_r,
            alpha=alpha, nms_kwargs=nms_kw)
        gb, gl = data["gt_boxes"][i], data["gt_labels"][i]
        if len(fb) == 0 or len(gb) == 0:
            for k in range(len(fb)):
                all_conf.append(fs[k] if k < len(fs) else 0.0)
                all_pred.append(1)
                all_lbl.append(-1)
            continue
        ious = compute_iou(fb, gb)
        for p_idx in range(len(fb)):
            best_iou = ious[p_idx].max()
            best_gt = ious[p_idx].argmax()
            all_conf.append(fs[p_idx])
            all_pred.append(1)
            all_lbl.append(gl[best_gt] if best_iou >= 0.5 else -1)
    return expected_calibration_error(np.array(all_conf), np.array(all_pred), np.array(all_lbl))


def main():
    t0 = time.time()
    N_RUNS = 3

    print("Generating datasets...")
    datasets = [gen_data(n_img=200, n_cls=5, seed=42 + r) for r in range(N_RUNS)]
    sop_datasets = [gen_data(n_img=400, n_cls=20, seed=100 + r) for r in range(N_RUNS)]
    oxford_datasets = [gen_data(n_img=300, n_cls=10, seed=200 + r) for r in range(N_RUNS)]
    prb_datasets = [gen_data(n_img=300, n_cls=8, containment_rate=0.5, seed=300 + r) for r in range(N_RUNS)]

    configs = [
        ("greedy", "none", {"iou_thresh": 0.5}),
        ("greedy", "sanms", {"iou_thresh": 0.5}),
        ("greedy", "box_voting", {"iou_thresh": 0.5}),
        ("soft", "none", {"sigma": 0.5}),
        ("soft", "sanms", {"sigma": 0.5}),
        ("diou", "none", {"diou_thresh": 0.5}),
        ("diou", "sanms", {"diou_thresh": 0.5}),
        ("matrix", "none", {"iou_thresh": 0.5}),
        ("matrix", "sanms", {"iou_thresh": 0.5}),
    ]
    retrieval_configs = [("greedy", "none"), ("greedy", "sanms"),
                          ("soft", "none"), ("soft", "sanms"),
                          ("diou", "none"), ("matrix", "none")]

    results = {}

    # === Tables 1 & 2 ===
    print("\n=== Tables 1 & 2: COCO mAP ===")
    results["table_1_2"] = {}
    for nms_m, post_r, nms_kw in configs:
        name = nms_m if post_r == "none" else f"{nms_m}+{post_r}"
        mAPs, AP50s, AP75s = [], [], []
        for r in range(N_RUNS):
            res = run_coco_exp(datasets[r], nms_m, post_r, nms_kw)
            mAPs.append(res["mAP"] * 100)
            AP50s.append(res["AP50"] * 100)
            AP75s.append(res["AP75"] * 100)
        results["table_1_2"][name] = {
            "mAP_mean": float(np.mean(mAPs)), "mAP_std": float(np.std(mAPs)),
            "AP50_mean": float(np.mean(AP50s)), "AP50_std": float(np.std(AP50s)),
            "AP75_mean": float(np.mean(AP75s)), "AP75_std": float(np.std(AP75s)),
        }
        print(f"  {name}: mAP={np.mean(mAPs):.1f}+/-{np.std(mAPs):.1f}  "
              f"AP50={np.mean(AP50s):.1f}  AP75={np.mean(AP75s):.1f}")

    # === Table 3: SOP Recall@K ===
    print("\n=== Table 3: SOP Recall@K ===")
    results["table_3"] = {}
    for nms_m, post_r in retrieval_configs:
        name = nms_m if post_r == "none" else f"{nms_m}+{post_r}"
        R1s, R10s, R100s = [], [], []
        for r in range(N_RUNS):
            feats = gen_features(sop_datasets[r], n_dim=64, seed=400 + r)
            f = feats.get((nms_m, post_r))
            if f is None: continue
            res = recall_at_k(f["query_embeddings"], f["gallery_embeddings"],
                              f["query_labels"], f["gallery_labels"], [1, 10, 100])
            R1s.append(res["R@1"])
            R10s.append(res["R@10"])
            R100s.append(res["R@100"])
        results["table_3"][name] = {
            "R@1_mean": float(np.mean(R1s)), "R@1_std": float(np.std(R1s)),
            "R@10_mean": float(np.mean(R10s)), "R@10_std": float(np.std(R10s)),
            "R@100_mean": float(np.mean(R100s)), "R@100_std": float(np.std(R100s)),
        }
        print(f"  {name}: R@1={np.mean(R1s):.1f}  R@10={np.mean(R10s):.1f}  R@100={np.mean(R100s):.1f}")

    # === Table 4: Oxford mAP ===
    print("\n=== Table 4: Oxford mAP ===")
    results["table_4"] = {}
    for nms_m, post_r in retrieval_configs:
        name = nms_m if post_r == "none" else f"{nms_m}+{post_r}"
        mAPs = []
        for r in range(N_RUNS):
            feats = gen_features(oxford_datasets[r], n_dim=64, seed=500 + r)
            f = feats.get((nms_m, post_r))
            if f is None: continue
            q_lbl, g_lbl = f["query_labels"], f["gallery_labels"]
            rel = np.zeros((len(q_lbl), len(g_lbl)), dtype=bool)
            for qi in range(len(q_lbl)):
                rel[qi] = (g_lbl == q_lbl[qi])
            mAPs.append(oxford_map(f["query_embeddings"], f["gallery_embeddings"], rel))
        results["table_4"][name] = {"mAP_mean": float(np.mean(mAPs)), "mAP_std": float(np.std(mAPs))}
        print(f"  {name}: mAP={np.mean(mAPs):.1f}+/-{np.std(mAPs):.1f}")

    # === Table 5: PRB Top-1 ===
    print("\n=== Table 5: PRB Top-1 ===")
    results["table_5"] = {}
    for nms_m, post_r in retrieval_configs:
        name = nms_m if post_r == "none" else f"{nms_m}+{post_r}"
        top1s = []
        for r in range(N_RUNS):
            feats = gen_features(prb_datasets[r], n_dim=64, seed=600 + r)
            f = feats.get((nms_m, post_r))
            if f is None: continue
            top1s.append(top1_accuracy(f["query_embeddings"], f["gallery_embeddings"],
                                      f["query_labels"], f["gallery_labels"]))
        results["table_5"][name] = {"top1_mean": float(np.mean(top1s)), "top1_std": float(np.std(top1s))}
        print(f"  {name}: Top-1={np.mean(top1s):.1f}+/-{np.std(top1s):.1f}")

    # === Table 7: Alpha ===
    print("\n=== Table 7: Alpha Ablation ===")
    alphas = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    results["table_7"] = {}
    for alpha in alphas:
        mAPs = []
        for r in range(N_RUNS):
            res = run_coco_exp(datasets[r], "greedy", "sanms", {"iou_thresh": 0.5}, alpha=alpha)
            mAPs.append(res["mAP"] * 100)
        results["table_7"][f"alpha={alpha}"] = {"mAP_mean": float(np.mean(mAPs)), "mAP_std": float(np.std(mAPs))}
        print(f"  alpha={alpha}: mAP={np.mean(mAPs):.1f}+/-{np.std(mAPs):.1f}")

    # === Table 8: Sort Criterion ===
    print("\n=== Table 8: Sort Criterion ===")
    criteria = ["area", "diagonal", "perimeter", "confidence"]
    results["table_8"] = {}
    for crit in criteria:
        mAPs = []
        for r in range(N_RUNS):
            pred_b, pred_s, pred_l, gt_b, gt_l = [], [], [], [], []
            d = datasets[r]
            for i in range(len(d["gt_boxes"])):
                nb, ns = greedy_nms(d["raw_boxes"][i], d["raw_scores"][i], iou_thresh=0.5)
                fb, fs = sanms(nb, ns, alpha=0.75, sort_criterion=crit)
                pred_b.append(fb); pred_s.append(fs)
                pred_l.append(np.ones(len(fb), dtype=int))
                gt_b.append(d["gt_boxes"][i]); gt_l.append(d["gt_labels"][i])
            res = coco_map(pred_b, pred_s, pred_l, gt_b, gt_l)
            mAPs.append(res["mAP"] * 100)
        results["table_8"][crit] = {"mAP_mean": float(np.mean(mAPs)), "mAP_std": float(np.std(mAPs))}
        print(f"  {crit}: mAP={np.mean(mAPs):.1f}+/-{np.std(mAPs):.1f}")

    # === Table 9: Eps ===
    print("\n=== Table 9: Eps Tolerance ===")
    eps_vals = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
    results["table_9"] = {}
    for eps in eps_vals:
        mAPs = []
        for r in range(N_RUNS):
            pred_b, pred_s, pred_l, gt_b, gt_l = [], [], [], [], []
            d = datasets[r]
            for i in range(len(d["gt_boxes"])):
                nb, ns = greedy_nms(d["raw_boxes"][i], d["raw_scores"][i], iou_thresh=0.5)
                fb, fs = sanms(nb, ns, alpha=0.75, eps=eps)
                pred_b.append(fb); pred_s.append(fs)
                pred_l.append(np.ones(len(fb), dtype=int))
                gt_b.append(d["gt_boxes"][i]); gt_l.append(d["gt_labels"][i])
            res = coco_map(pred_b, pred_s, pred_l, gt_b, gt_l)
            mAPs.append(res["mAP"] * 100)
        results["table_9"][f"eps={eps}"] = {"mAP_mean": float(np.mean(mAPs)), "mAP_std": float(np.std(mAPs))}
        print(f"  eps={eps}: mAP={np.mean(mAPs):.1f}+/-{np.std(mAPs):.1f}")

    # === Table 10: Score Inheritance ===
    print("\n=== Table 10: Score Inheritance ===")
    def sanms_no_inherit(boxes, scores, alpha=0.75, eps=1e-3):
        n = len(boxes)
        if n <= 1: return boxes.copy(), scores.copy().astype(np.float64)
        boxes = np.asarray(boxes, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).copy()
        areas = compute_area(boxes)
        order = np.argsort(areas)[::-1]
        keep = []
        for i in order:
            contained = False
            for j in keep:
                if (boxes[i, 0] >= boxes[j, 0] - eps and boxes[i, 1] >= boxes[j, 1] - eps
                    and boxes[i, 2] <= boxes[j, 2] + eps and boxes[i, 3] <= boxes[j, 3] + eps):
                    if areas[j] > 0 and areas[i] / areas[j] < alpha:
                        contained = True
                        break
            if not contained: keep.append(i)
        if not keep: return np.zeros((0, 4)), np.zeros(0)
        keep = np.array(keep)
        kb, ks = boxes[keep], scores[keep]
        fo = np.argsort(ks)[::-1]
        return kb[fo], ks[fo]

    table10_configs = [("greedy (baseline)", None), ("greedy+sanms (full)", "full"),
                        ("greedy+sanms (no inherit)", "no_inherit")]
    results["table_10"] = {}
    for name, mode in table10_configs:
        mAPs = []
        for r in range(N_RUNS):
            pred_b, pred_s, pred_l, gt_b, gt_l = [], [], [], [], []
            d = datasets[r]
            for i in range(len(d["gt_boxes"])):
                nb, ns = greedy_nms(d["raw_boxes"][i], d["raw_scores"][i], iou_thresh=0.5)
                if mode is None: fb, fs = nb, ns
                elif mode == "full": fb, fs = sanms(nb, ns, alpha=0.75)
                else: fb, fs = sanms_no_inherit(nb, ns, alpha=0.75)
                pred_b.append(fb); pred_s.append(fs)
                pred_l.append(np.ones(len(fb), dtype=int))
                gt_b.append(d["gt_boxes"][i]); gt_l.append(d["gt_labels"][i])
            res = coco_map(pred_b, pred_s, pred_l, gt_b, gt_l)
            mAPs.append(res["mAP"] * 100)
        results["table_10"][name] = {"mAP_mean": float(np.mean(mAPs)), "mAP_std": float(np.std(mAPs))}
        print(f"  {name}: mAP={np.mean(mAPs):.1f}+/-{np.std(mAPs):.1f}")

    # === Table 11: Scene Failure Rate ===
    print("\n=== Table 11: Scene Failure Rate ===")
    scene_configs = [("greedy", "none", {"iou_thresh": 0.5}),
                     ("greedy", "sanms", {"iou_thresh": 0.5}),
                     ("soft", "none", {"sigma": 0.5}),
                     ("soft", "sanms", {"sigma": 0.5})]
    results["table_11"] = {}
    for nms_m, post_r, nms_kw in scene_configs:
        name = nms_m if post_r == "none" else f"{nms_m}+{post_r}"
        scene_rates = {}
        for r in range(N_RUNS):
            d = datasets[r]
            pred_b, pred_s = [], []
            for i in range(len(d["gt_boxes"])):
                fb, fs = process_image_detection(d["raw_boxes"][i], d["raw_scores"][i],
                    nms_method=nms_m, post_refinement=post_r, alpha=0.75, nms_kwargs=nms_kw)
                pred_b.append(fb); pred_s.append(fs)
            rates = failure_rate_by_scene(pred_b, pred_s, d["gt_boxes"], d["scene_types"], iou_thresh=0.5)
            for s, rate in rates.items():
                scene_rates.setdefault(s, []).append(rate)
        results["table_11"][name] = {}
        for s, rates in scene_rates.items():
            results["table_11"][name][s] = {"mean": float(np.mean(rates)), "std": float(np.std(rates))}
            print(f"  {name}/{s}: {np.mean(rates):.1f}+/-{np.std(rates):.1f}")

    # === Table 13: ECE ===
    print("\n=== Table 13: ECE ===")
    ece_configs = [("greedy", "none", {"iou_thresh": 0.5}),
                   ("greedy", "sanms", {"iou_thresh": 0.5}),
                   ("soft", "none", {"sigma": 0.5}),
                   ("soft", "sanms", {"sigma": 0.5}),
                   ("diou", "none", {"diou_thresh": 0.5}),
                   ("matrix", "none", {"iou_thresh": 0.5})]
    results["table_13"] = {}
    for nms_m, post_r, nms_kw in ece_configs:
        name = nms_m if post_r == "none" else f"{nms_m}+{post_r}"
        ECEs = []
        for r in range(N_RUNS):
            ECEs.append(run_ece_exp(datasets[r], nms_m, post_r, nms_kw))
        results["table_13"][name] = {"ECE_mean": float(np.mean(ECEs)), "ECE_std": float(np.std(ECEs))}
        print(f"  {name}: ECE={np.mean(ECEs):.4f}+/-{np.std(ECEs):.4f}")

    elapsed = time.time() - t0
    print(f"\nAll experiments completed in {elapsed:.1f}s")

    output_path = os.path.join(os.path.dirname(__file__), "results", "all_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
