"""
src/run_experiments.py
======================
Main experiment runner for the scripted (CPU-only) study.

Usage:
    python -m src.run_experiments --conditions A B C D --seeds 20 --rounds 200 --tag scripted_main

Protocol:
1. Generates trajectories for specified conditions and seeds.
2. Runs detectors on raw trajectory data (no ground truth exposed).
3. Calibrates decision thresholds on validation seeds (seeds 0-4 of Condition A) to target 5% FPR.
4. Evaluates ranking metrics (Top-1 accuracy, MRR) and calibrated FPR.
5. Aggregates results per condition using Wilson score intervals for Top-1 accuracy
   and non-parametric bootstrap CIs for continuous scores.
6. Saves results to results/metrics/ and results/tables/.
"""

from __future__ import annotations
import argparse
import json
import csv
import datetime
import sys
from pathlib import Path

import numpy as np

# Resolve project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "metrics").mkdir(exist_ok=True)
(RESULTS_DIR / "tables").mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from src.trajectories import run_trajectory, save_trajectory, load_trajectory, load_ground_truth
from src.detectors import run_detectors, ALL_PAIRS
from src.metrics import (
    evaluate_trajectory,
    aggregate_metrics,
    chance_top1_accuracy,
    calibrate_thresholds_from_validation,
)
from src.environment import N_AGENTS

COORDINATED_PAIR = (0, 1)
CHANCE_TOP1 = chance_top1_accuracy()


def run_single_trajectory(
    condition: str,
    seed: int,
    n_rounds: int,
    alpha: float = 1.0,
    noise: float = 0.1,
    overwrite: bool = True,
) -> dict:
    """Generate trajectory and run detectors (without ground truth)."""
    traj = run_trajectory(
        condition=condition,
        seed=seed,
        n_rounds=n_rounds,
        alpha=alpha,
        coordinated_pair=COORDINATED_PAIR,
        noise=noise,
    )
    
    # Save to disk and reload (validates JSONL pipeline)
    traj_path, gt_path = save_trajectory(traj, DATA_DIR, overwrite=overwrite)
    meta, rounds = load_trajectory(traj_path)
    gt = load_ground_truth(gt_path)
    
    true_pair = tuple(gt["coordinating_pair"]) if gt["coordinating_pair"] else None
    
    # Run detectors — they do NOT receive the ground truth
    detector_results = run_detectors(rounds, threshold=0.0)
    
    return {
        "run_id": meta["run_id"],
        "condition": condition,
        "seed": seed,
        "n_rounds": n_rounds,
        "alpha": alpha if condition == "D" else None,
        "true_pair": true_pair,
        "detector_results": detector_results,
    }


def run_experiment(
    conditions: list[str],
    seeds: list[int],
    n_rounds: int,
    alphas: list[float] = None,
    noise: float = 0.1,
    run_tag: str = "default",
    overwrite: bool = True,
    val_seed_cutoff: int = 5,
    target_fpr: float = 0.05,
    fixed_thresholds: Optional[dict[str, float]] = None,
) -> dict:
    """Run full experiment across conditions and seeds with validation calibration or fixed thresholds."""
    if alphas is None:
        alphas = [0.25, 0.5, 0.75, 1.0]
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw_runs: list[dict] = []
    
    print(f"--- Running Trajectories: {len(conditions)} conditions, {len(seeds)} seeds, {n_rounds} rounds ---")
    
    for condition in conditions:
        print(f"\n=== Condition {condition} ===")
        if condition == "D":
            for alpha in alphas:
                for seed in seeds:
                    print(f"  seed={seed:04d}, alpha={alpha:.2f}", end=" ... ")
                    run_dict = run_single_trajectory(
                        condition=condition,
                        seed=seed,
                        n_rounds=n_rounds,
                        alpha=alpha,
                        noise=noise,
                        overwrite=overwrite,
                    )
                    raw_runs.append(run_dict)
                    pni_top = run_dict["detector_results"]["pni"][0]["pair"]
                    print(f"pni_top_pair={pni_top}")
        else:
            for seed in seeds:
                print(f"  seed={seed:04d}", end=" ... ")
                run_dict = run_single_trajectory(
                    condition=condition,
                    seed=seed,
                    n_rounds=n_rounds,
                    noise=noise,
                    overwrite=overwrite,
                )
                raw_runs.append(run_dict)
                pni_top = run_dict["detector_results"]["pni"][0]["pair"]
                print(f"pni_top_pair={pni_top}")
                
    # Decision Threshold Determination
    if fixed_thresholds is not None:
        if not ("pni" in fixed_thresholds and "marginal_mi" in fixed_thresholds):
            raise ValueError(f"fixed_thresholds must contain 'pni' and 'marginal_mi', got {fixed_thresholds}")
        calibrated_thresholds = {k: float(v) for k, v in fixed_thresholds.items()}
        if "cmi" not in calibrated_thresholds:
            calibrated_thresholds["cmi"] = calibrated_thresholds.get("marginal_mi", 0.0)
        print(f"\nUsing Fixed Decision Thresholds:")
        for det, th in calibrated_thresholds.items():
            print(f"  {det}: {th:.6f} bits")
    else:
        # Threshold Calibration on Validation Seeds (Condition A, seeds < val_seed_cutoff)
        val_records = [
            r for r in raw_runs
            if r["condition"] == "A" and r["seed"] < val_seed_cutoff
        ]
        if not val_records:
            raise ValueError(
                "Condition A validation records (seeds < val_seed_cutoff) were not found, "
                "and no fixed_thresholds were provided! Cannot determine decision thresholds."
            )
        calibrated_thresholds = calibrate_thresholds_from_validation(val_records, target_fpr=target_fpr)
        print(f"\nCalibrated Thresholds (target FPR={target_fpr:.2f} on Cond A val seeds 0-{val_seed_cutoff-1}):")
        for det, th in calibrated_thresholds.items():
            print(f"  {det}: {th:.6f} bits")
        
    # Evaluate Metrics for all runs using calibrated thresholds
    evaluated_results: list[dict] = []
    for r in raw_runs:
        m = evaluate_trajectory(
            detector_results=r["detector_results"],
            true_pair=r["true_pair"],
            condition=r["condition"],
            thresholds=calibrated_thresholds,
        )
        m["run_id"] = r["run_id"]
        m["seed"] = r["seed"]
        m["n_rounds"] = r["n_rounds"]
        m["alpha"] = r["alpha"]
        m["is_val_seed"] = bool(r["condition"] == "A" and r["seed"] < val_seed_cutoff)
        
        # Record top ranked pairs and scores
        for det_name, ranked in r["detector_results"].items():
            if ranked and "pair" in ranked[0]:
                m[f"{det_name}_top_ranked_pair"] = list(ranked[0]["pair"])
                m[f"{det_name}_top_score"] = float(ranked[0]["score"])
                
        evaluated_results.append(m)
        
    # Aggregate metrics per condition
    # CRITICAL: Condition A test performance MUST exclude calibration seeds (seeds < val_seed_cutoff)
    agg_by_condition = {}
    for condition in conditions:
        if condition == "D":
            for alpha in alphas:
                key = f"D_alpha{alpha:.2f}"
                valid = [r for r in evaluated_results
                         if r.get("condition") == "D"
                         and abs((r.get("alpha") or 0) - alpha) < 1e-9]
                agg_by_condition[key] = aggregate_metrics(valid)
        elif condition == "A":
            # Strictly held-out test evaluation: seeds val_seed_cutoff to n_seeds-1 (n=15)
            test_runs = [r for r in evaluated_results if r.get("condition") == "A" and not r.get("is_val_seed")]
            agg_by_condition["A"] = aggregate_metrics(test_runs)
            # Separately record calibration performance for complete transparency
            cal_runs = [r for r in evaluated_results if r.get("condition") == "A" and r.get("is_val_seed")]
            if cal_runs:
                agg_by_condition["A_calibration"] = aggregate_metrics(cal_runs)
        else:
            valid = [r for r in evaluated_results if r.get("condition") == condition]
            agg_by_condition[condition] = aggregate_metrics(valid)
            
    output = {
        "run_tag": run_tag,
        "timestamp": timestamp,
        "config": {
            "conditions": conditions,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "n_rounds": n_rounds,
            "alphas": alphas if "D" in conditions else None,
            "noise": noise,
            "coordinated_pair": list(COORDINATED_PAIR),
            "n_agents": N_AGENTS,
            "chance_top1": CHANCE_TOP1,
            "val_seed_cutoff": val_seed_cutoff,
            "calibrated_thresholds": calibrated_thresholds,
            "threshold_mode": "fixed" if fixed_thresholds is not None else "calibrated_from_cond_a",
        },
        "per_run_results": evaluated_results,
        "aggregated": agg_by_condition,
    }
    
    return output


def save_results(output: dict, results_dir: Path, run_tag: str) -> dict:
    """Save results to JSON and CSV."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    
    # Save full timestamped JSON
    json_path = results_dir / "metrics" / f"results_{run_tag}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_serializer)
    print(f"Saved: {json_path}")
    
    # Save canonical final_metrics.json
    final_json = results_dir / "metrics" / "final_metrics.json"
    with open(final_json, "w") as f:
        json.dump(output, f, indent=2, default=_json_serializer)
    print(f"Saved: {final_json}")
    
    # Save summary CSV
    SKIP_METRICS = {"seed", "n_rounds", "alpha", "has_true_pair", "is_val_seed"}
    rows = []
    for cond_key, agg in output["aggregated"].items():
        row = {"condition": cond_key}
        for metric, stats in agg.items():
            if metric in SKIP_METRICS:
                continue
            if isinstance(stats, dict):
                row[f"{metric}_mean"] = stats.get("mean", np.nan)
                row[f"{metric}_ci_low"] = stats.get("ci_low", np.nan)
                row[f"{metric}_ci_high"] = stats.get("ci_high", np.nan)
                row[f"{metric}_n"] = stats.get("n", 0)
        rows.append(row)
        
    csv_path = results_dir / "tables" / f"summary_{run_tag}_{ts}.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {csv_path}")
        
        # Save canonical summary.csv
        canon_csv = results_dir / "tables" / "summary.csv"
        with open(canon_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {canon_csv}")
        
    return {"json": str(json_path), "csv": str(csv_path)}


def _json_serializer(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="Run collusion detection experiments.")
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C", "D"], help="Conditions to run.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (0 to seeds-1).")
    parser.add_argument("--rounds", type=int, default=200, help="Number of rounds per trajectory.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0], help="Alpha values for condition D.")
    parser.add_argument("--tag", default="scripted_main", help="Run identifier tag.")
    args = parser.parse_args()
    
    output = run_experiment(
        conditions=args.conditions,
        seeds=list(range(args.seeds)),
        n_rounds=args.rounds,
        alphas=args.alphas,
        run_tag=args.tag,
        overwrite=True,
    )
    
    save_results(output, RESULTS_DIR, run_tag=args.tag)
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
