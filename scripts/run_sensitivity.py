"""
scripts/run_sensitivity.py
==========================
Run sensitivity analysis across trajectory lengths: T in [50, 100, 200, 500].
Tests how sample size impacts PNI estimation, finite-sample bias, and Top-1 accuracy.

Methodological rules:
1. Uses fixed decision thresholds calibrated on primary Condition A validation seeds (T=200).
   Does NOT calibrate on empty sets or silently fall back to 0.0.
2. For T=200, reuses the primary Condition B and C trajectories and evaluated results directly,
   avoiding redundant execution or overwriting.
3. Produces exactly 120 new sensitivity trajectories (40 for each of T=50, 100, 500), which
   combined with the 140 primary trajectories (T=200) gives exactly 260 unique trajectories
   and 54,000 unique rounds.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json
import csv
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_experiments import run_experiment

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    rounds_list = [50, 100, 200, 500]
    seeds = list(range(20))
    sensitivity_results = {}

    # Ensure output directories exist before writing
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    # 1. Load primary calibrated thresholds from final_metrics.json (T=200)
    primary_metrics_path = RESULTS_DIR / "metrics" / "final_metrics.json"
    if not primary_metrics_path.exists():
        raise FileNotFoundError(
            f"Primary metrics file {primary_metrics_path} not found. "
            "Please run the primary experiment (python -m src.run_experiments) first."
        )
    with open(primary_metrics_path) as f:
        primary_data = json.load(f)

    calibrated_thresholds = primary_data.get("config", {}).get("calibrated_thresholds")
    if not calibrated_thresholds or "pni" not in calibrated_thresholds or "marginal_mi" not in calibrated_thresholds:
        raise ValueError(f"Invalid or missing calibrated_thresholds in {primary_metrics_path}: {calibrated_thresholds}")

    print(f"=== Sensitivity Analysis: Trajectory Length Sweep ===")
    print(f"Fixed Decision Thresholds (calibrated from T=200 primary Cond A validation):")
    for det, th in calibrated_thresholds.items():
        print(f"  {det} threshold: {th:.6f} bits")
    print(f"Trajectory lengths: {rounds_list}")

    for n_rounds in rounds_list:
        if n_rounds == 200:
            print(f"\n==========================================")
            print(f"--- Reusing T = 200 primary runs (20 seeds) ---")
            print(f"==========================================")
            # Reuse the primary evaluated metrics for B and C directly
            sensitivity_results["T_200"] = {
                "B": primary_data["aggregated"]["B"],
                "C": primary_data["aggregated"]["C"],
            }
        else:
            print(f"\n==========================================")
            print(f"--- Running T = {n_rounds} rounds (20 seeds) ---")
            print(f"==========================================")
            output = run_experiment(
                conditions=["B", "C"],
                seeds=seeds,
                n_rounds=n_rounds,
                fixed_thresholds=calibrated_thresholds,
                run_tag=f"sensitivity_T{n_rounds}",
                overwrite=True,
            )
            sensitivity_results[f"T_{n_rounds}"] = output["aggregated"]

    # Save combined sensitivity summary
    out_file = RESULTS_DIR / "metrics" / "sensitivity_trajectory_length.json"
    with open(out_file, "w") as f:
        json.dump(sensitivity_results, f, indent=2)
    print(f"\nSaved sensitivity metrics to: {out_file}")

    # Write sensitivity table with full uncertainty intervals
    csv_rows = []
    for t in rounds_list:
        key = f"T_{t}"
        agg = sensitivity_results[key]
        
        c_top1 = agg.get("C", {}).get("pni_top1", {})
        c_score = agg.get("C", {}).get("pni_true_score", {})
        b_mi = agg.get("B", {}).get("marginal_mi_top_score", {})
        b_cmi = agg.get("B", {}).get("cmi_top_score", {})
        b_cmi_fpr = agg.get("B", {}).get("cmi_fpr", {})
        b_pni = agg.get("B", {}).get("pni_top_score", {})
        b_fpr = agg.get("B", {}).get("pni_fpr", {})

        csv_rows.append({
            "trajectory_length_T": t,
            "cond_C_pni_top1": c_top1.get("mean", np.nan),
            "cond_C_pni_top1_ci_low": c_top1.get("ci_low", np.nan),
            "cond_C_pni_top1_ci_high": c_top1.get("ci_high", np.nan),
            "cond_C_pni_true_score": c_score.get("mean", np.nan),
            "cond_C_pni_true_score_ci_low": c_score.get("ci_low", np.nan),
            "cond_C_pni_true_score_ci_high": c_score.get("ci_high", np.nan),
            "cond_B_marginal_mi_score": b_mi.get("mean", np.nan),
            "cond_B_marginal_mi_score_ci_low": b_mi.get("ci_low", np.nan),
            "cond_B_marginal_mi_score_ci_high": b_mi.get("ci_high", np.nan),
            "cond_B_cmi_score": b_cmi.get("mean", np.nan),
            "cond_B_cmi_score_ci_low": b_cmi.get("ci_low", np.nan),
            "cond_B_cmi_score_ci_high": b_cmi.get("ci_high", np.nan),
            "cond_B_cmi_fpr": b_cmi_fpr.get("mean", np.nan),
            "cond_B_pni_score": b_pni.get("mean", np.nan),
            "cond_B_pni_score_ci_low": b_pni.get("ci_low", np.nan),
            "cond_B_pni_score_ci_high": b_pni.get("ci_high", np.nan),
            "cond_B_pni_fpr": b_fpr.get("mean", np.nan),
            "cond_B_pni_fpr_ci_low": b_fpr.get("ci_low", np.nan),
            "cond_B_pni_fpr_ci_high": b_fpr.get("ci_high", np.nan),
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
    c_top1_lo = [r["cond_C_pni_top1_ci_low"] for r in csv_rows]
    c_top1_hi = [r["cond_C_pni_top1_ci_high"] for r in csv_rows]
    
    ax1.plot(rounds_list, c_top1_vals, "o-", color="#2196F3", lw=2, label="PNI Top-1 (Cond C)")
    ax1.fill_between(rounds_list, c_top1_lo, c_top1_hi, color="#2196F3", alpha=0.15, label="95% Wilson CI")
    ax1.axhline(1/6, color="red", linestyle="--", label="Chance (0.167)")
    ax1.set_xlabel("Trajectory Length T (rounds)")
    ax1.set_ylabel("Top-1 Accuracy")
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_title("Detection Accuracy vs Trajectory Length (Cond C)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")

    # Right: Triangulation: Marginal MI vs CMI vs PNI under Confounding (Cond B)
    b_mi_vals = [r["cond_B_marginal_mi_score"] for r in csv_rows]
    b_cmi_vals = [r["cond_B_cmi_score"] for r in csv_rows]
    b_pni_vals = [r["cond_B_pni_score"] for r in csv_rows]
    ax2.plot(rounds_list, b_mi_vals, "s--", color="#FF9800", lw=2, label="Marginal MI (unconditioned)")
    ax2.plot(rounds_list, b_cmi_vals, "^-.", color="#9C27B0", lw=2, label="Conditional MI (state-conditioned)")
    ax2.plot(rounds_list, b_pni_vals, "o-", color="#4CAF50", lw=2, label="PNI (conditioned + rival sub)")
    ax2.set_xlabel("Trajectory Length T (rounds)")
    ax2.set_ylabel("Estimated Score (bits)")
    ax2.set_title("Confounder Disambiguation vs Trajectory Length (Cond B)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig5_trajectory_length_sensitivity.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
