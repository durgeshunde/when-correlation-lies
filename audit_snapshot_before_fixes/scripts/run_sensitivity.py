"""
scripts/run_sensitivity.py
==========================
Run sensitivity analysis across trajectory lengths: T in [50, 100, 200, 500].
Tests how sample size impacts PNI estimation, finite-sample bias, and Top-1 accuracy.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json
import csv
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiments import run_experiment, save_results

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    rounds_list = [50, 100, 200, 500]
    seeds = list(range(20))
    sensitivity_results = {}

    print(f"Running sensitivity analysis across trajectory lengths: {rounds_list}")
    
    for n_rounds in rounds_list:
        print(f"\n==========================================")
        print(f"--- Running T = {n_rounds} rounds (20 seeds) ---")
        print(f"==========================================")
        
        output = run_experiment(
            conditions=["B", "C"],
            seeds=seeds,
            n_rounds=n_rounds,
            threshold=0.0,
            run_tag=f"sensitivity_T{n_rounds}",
            overwrite=True
        )
        sensitivity_results[f"T_{n_rounds}"] = output["aggregated"]
    
    # Save combined sensitivity summary
    out_file = RESULTS_DIR / "metrics" / "sensitivity_trajectory_length.json"
    with open(out_file, "w") as f:
        json.dump(sensitivity_results, f, indent=2)
    print(f"\nSaved sensitivity metrics to: {out_file}")

    # Write sensitivity table
    csv_rows = []
    for t in rounds_list:
        key = f"T_{t}"
        agg = sensitivity_results[key]
        c_pni_top1 = agg.get("C", {}).get("pni_top1", {}).get("mean", np.nan)
        c_pni_score = agg.get("C", {}).get("pni_true_score", {}).get("mean", np.nan)
        b_mi_score = agg.get("B", {}).get("marginal_mi_top_score", {}).get("mean", np.nan)
        b_pni_score = agg.get("B", {}).get("pni_top_score", {}).get("mean", np.nan)
        b_pni_fpr = agg.get("B", {}).get("pni_fpr", {}).get("mean", np.nan)
        
        csv_rows.append({
            "trajectory_length_T": t,
            "cond_C_pni_top1": c_pni_top1,
            "cond_C_pni_true_score": c_pni_score,
            "cond_B_marginal_mi_score": b_mi_score,
            "cond_B_pni_score": b_pni_score,
            "cond_B_pni_fpr": b_pni_fpr,
        })

    csv_path = TABLES_DIR / "sensitivity_trajectory_length.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved sensitivity CSV to: {csv_path}")

    # Generate Figure 5: Trajectory Length Sensitivity
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Top-1 Accuracy on Condition C vs T
    c_top1_vals = [r["cond_C_pni_top1"] for r in csv_rows]
    ax1.plot(rounds_list, c_top1_vals, "o-", color="#2196F3", lw=2, label="PNI Top-1 (Cond C)")
    ax1.axhline(1/6, color="red", linestyle="--", label="Chance (0.167)")
    ax1.set_xlabel("Trajectory Length T (rounds)")
    ax1.set_ylabel("Top-1 Accuracy")
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_title("Detection Accuracy vs Trajectory Length (Cond C)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Right: Spurious MI vs Conditional PNI under Confounding (Cond B)
    b_mi_vals = [r["cond_B_marginal_mi_score"] for r in csv_rows]
    b_pni_vals = [r["cond_B_pni_score"] for r in csv_rows]
    ax2.plot(rounds_list, b_mi_vals, "s--", color="#FF9800", lw=2, label="Marginal MI (spurious)")
    ax2.plot(rounds_list, b_pni_vals, "o-", color="#4CAF50", lw=2, label="PNI (conditioned)")
    ax2.set_xlabel("Trajectory Length T (rounds)")
    ax2.set_ylabel("Estimated Score (bits)")
    ax2.set_title("Context Confounding vs Trajectory Length (Cond B)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig5_trajectory_length_sensitivity.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
