"""
Measure runtime with std for all NMS methods.

Outputs mean +/- std for Table 12 (Runtime).
Results are saved to experiments/results/runtime_with_std.json.

Usage:
    python -m experiments.run_runtime          # from project root
    python experiments/run_runtime.py         # direct execution
"""

import os, sys, time, json
import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sanms import greedy_nms, soft_nms, diou_nms, matrix_nms, sanms, box_voting


def measure_runtime_with_std(n_values=[50, 100, 300], n_trials=100, seed=42):
    """Measure runtime of all NMS variants with statistical std.

    Args:
        n_values: list of box counts to benchmark.
        n_trials: number of repetitions per configuration for std calculation.
        seed: random seed for reproducibility.

    Returns:
        dict mapping method name -> {N: {mean, std}} in milliseconds.
    """
    rng = np.random.RandomState(seed)
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
            results[name][n] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
            }

        print(f"  N={n}: " + ", ".join(
            f"{m}={results[m][n]['mean']:.2f}+/-{results[m][n]['std']:.2f}ms"
            for m in methods
        ))

    return results


if __name__ == "__main__":
    print("=== Runtime Measurement (with std) ===")
    print(f"CPU: Intel Core i7-8550U")
    print(f"n_trials: 100 per configuration\n")
    results = measure_runtime_with_std(n_values=[50, 100, 300], n_trials=100)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "results", "runtime_with_std.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")

    # Print as table
    print("\nTable 12: Runtime comparison (ms/image, mean +/- std)")
    print(f"{'Method':<15} {'N=50':>16} {'N=100':>16} {'N=300':>16}")
    print("-" * 65)
    for m in ["greedy_nms", "soft_nms", "diou_nms", "matrix_nms", "box_voting", "sanms"]:
        row = results[m]
        n50 = f"{row[50]['mean']:.1f} +/- {row[50]['std']:.1f}"
        n100 = f"{row[100]['mean']:.1f} +/- {row[100]['std']:.1f}"
        n300 = f"{row[300]['mean']:.1f} +/- {row[300]['std']:.1f}"
        print(f"{m:<15} {n50:>16} {n100:>16} {n300:>16}")
