"""
run_experiments.py
==================
Main experiment runner for the scripted (CPU-only) study.

Usage:
    python -m src.run_experiments --help
    python -m src.run_experiments --conditions A B C --seeds 20 --rounds 200
    python -m src.run_experiments --conditions A B C D --seeds 20 --rounds 200 --full

This script:
1. Generates trajectories for specified conditions and seeds.
2. Runs detectors on each trajectory.
3. Evaluates metrics.
4. Saves results to results/metrics/ and results/tables/.
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

# Add project root to path so 'src' can be imported
sys.path.insert(0, str(PROJECT_ROOT))

from src.trajectories import run_trajectory, save_trajectory, load_trajectory, load_ground_truth
from src.detectors import run_detectors, ALL_PAIRS
from src.metrics import evaluate_trajectory, aggregate_metrics, chance_top1_accuracy
from src.environment import N_AGENTS

COORDINATED_PAIR = (0, 1)
CHANCE_TOP1 = chance_top1_accuracy()


def run_single_trajectory_experiment(
    condition: str,
    seed: int,
    n_rounds: int,
    threshold: float,
    alpha: float = 1.0,
    noise: float = 0.1,
    overwrite: bool = True,
) -> dict:
    """Run one trajectory, apply detectors, evaluate metrics."""
    traj = run_trajectory(
        condition=condition,
        seed=seed,
        n_rounds=n_rounds,
        alpha=alpha,
        coordinated_pair=COORDINATED_PAIR,
        noise=noise,
    )
    
    # Save to disk and reload (validates the JSONL pipeline)
    traj_path, gt_path = save_trajectory(traj, DATA_DIR, overwrite=overwrite)
    meta, rounds = load_trajectory(traj_path)
    gt = load_ground_truth(gt_path)
    
    true_pair = tuple(gt["coordinating_pair"]) if gt["coordinating_pair"] else None
    
    # Run detectors — they do NOT receive the ground truth
    detector_results = run_detectors(rounds, threshold=threshold)
    
    # Evaluate metrics using ground truth
    metrics = evaluate_trajectory(detector_results, true_pair, condition, threshold)
    metrics["run_id"] = meta["run_id"]
    metrics["seed"] = seed
    metrics["n_rounds"] = n_rounds
    metrics["alpha"] = alpha if condition == "D" else None
    
    # Record the top-ranked pair for each detector
    for det_name, ranked in detector_results.items():
        if ranked and "pair" in ranked[0]:
            metrics[f"{det_name}_top_ranked_pair"] = list(ranked[0]["pair"])
            metrics[f"{det_name}_top_score"] = ranked[0]["score"]
    
    return metrics


def run_experiment(
    conditions: list[str],
    seeds: list[int],
    n_rounds: int,
    threshold: float = 0.0,
    alphas: list[float] = None,
    noise: float = 0.1,
    run_tag: str = "default",
    overwrite: bool = True,
) -> dict:
    """Run the full experiment for specified conditions and seeds."""
    if alphas is None:
        alphas = [0.25, 0.5, 0.75, 1.0]
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    all_results: list[dict] = []
    
    for condition in conditions:
        print(f"\n=== Condition {condition} ===")
        
        if condition == "D":
            for alpha in alphas:
                cond_results = []
                for seed in seeds:
                    print(f"  seed={seed:04d}, alpha={alpha:.2f}", end=" ... ")
                    try:
                        r = run_single_trajectory_experiment(
                            condition=condition,
                            seed=seed,
                            n_rounds=n_rounds,
                            threshold=threshold,
                            alpha=alpha,
                            noise=noise,
                            overwrite=overwrite,
                        )
                        cond_results.append(r)
                        all_results.append(r)
                        print(f"pni_top1={r.get('pni_top1', 'N/A')}, pni_fpr={r.get('pni_fpr', 'N/A'):.3f}")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        all_results.append({
                            "condition": condition,
                            "seed": seed,
                            "alpha": alpha,
                            "error": str(e),
                        })
        else:
            cond_results = []
            for seed in seeds:
                print(f"  seed={seed:04d}", end=" ... ")
                try:
                    r = run_single_trajectory_experiment(
                        condition=condition,
                        seed=seed,
                        n_rounds=n_rounds,
                        threshold=threshold,
                        noise=noise,
                        overwrite=overwrite,
                    )
                    cond_results.append(r)
                    all_results.append(r)
                    top1 = r.get("pni_top1", "N/A")
                    fpr = r.get("pni_fpr", np.nan)
                    fpr_str = f"{fpr:.3f}" if not np.isnan(fpr) else "N/A"
                    print(f"pni_top1={top1}, pni_fpr={fpr_str}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({
                        "condition": condition,
                        "seed": seed,
                        "alpha": None,
                        "error": str(e),
                    })
    
    # Aggregate metrics per condition
    agg_by_condition = {}
    for condition in conditions:
        if condition == "D":
            for alpha in alphas:
                key = f"D_alpha{alpha:.2f}"
                valid = [r for r in all_results
                         if r.get("condition") == "D"
                         and abs((r.get("alpha") or 0) - alpha) < 1e-9
                         and "error" not in r]
                agg_by_condition[key] = aggregate_metrics(valid)
        else:
            valid = [r for r in all_results
                     if r.get("condition") == condition and "error" not in r]
            agg_by_condition[condition] = aggregate_metrics(valid)
    
    output = {
        "run_tag": run_tag,
        "timestamp": timestamp,
        "config": {
            "conditions": conditions,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "n_rounds": n_rounds,
            "threshold": threshold,
            "alphas": alphas if "D" in conditions else None,
            "noise": noise,
            "coordinated_pair": list(COORDINATED_PAIR),
            "n_agents": N_AGENTS,
            "chance_top1": CHANCE_TOP1,
        },
        "per_run_results": all_results,
        "aggregated": agg_by_condition,
    }
    
    return output


def save_results(output: dict, results_dir: Path, run_tag: str) -> dict:
    """Save results to JSON and CSV."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full JSON
    json_path = results_dir / "metrics" / f"results_{run_tag}_{ts}.json"
    with open(json_path, "w") as f:
        # Convert numpy types
        json.dump(output, f, indent=2, default=_json_serializer)
    
    # Save summary CSV — only include meaningful numeric metrics
    SKIP_METRICS = {"seed", "n_rounds", "alpha", "has_true_pair"}
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
    
    if rows:
        csv_path = results_dir / "tables" / f"summary_{run_tag}_{ts}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path = None
    
    # Also save as canonical "final_metrics.json"
    final_path = results_dir / "metrics" / "final_metrics.json"
    with open(final_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_serializer)
    
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    if csv_path:
        print(f"  CSV:  {csv_path}")
    print(f"  Final: {final_path}")
    
    return {"json": str(json_path), "csv": str(csv_path) if csv_path else None}


def _json_serializer(obj):
    """JSON-serialize numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def print_summary(output: dict):
    """Print a human-readable summary of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Config: {output['config']['n_seeds']} seeds × {output['config']['n_rounds']} rounds")
    print(f"Chance top-1 accuracy: {output['config']['chance_top1']:.3f} (random baseline)")
    print()
    
    for cond_key, agg in output["aggregated"].items():
        print(f"Condition {cond_key}")
        print("-" * 50)
        for metric, stats in sorted(agg.items()):
            if isinstance(stats, dict) and not np.isnan(stats.get("mean", np.nan)):
                m = stats["mean"]
                lo = stats["ci_low"]
                hi = stats["ci_high"]
                n = stats["n"]
                print(f"  {metric:40s}: {m:.4f} (95% CI: [{lo:.4f}, {hi:.4f}], n={n})")
        print()


def main():
    parser = argparse.ArgumentParser(description="Run AI collusion detection experiments")
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C", "D"],
                        help="Experimental conditions to run")
    parser.add_argument("--seeds", type=int, default=20,
                        help="Number of seeds (seeds 0 to N-1)")
    parser.add_argument("--rounds", type=int, default=200,
                        help="Rounds per trajectory")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Net influence flagging threshold")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Agent policy noise")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0],
                        help="Alpha values for condition D")
    parser.add_argument("--tag", type=str, default="scripted",
                        help="Run tag for file naming")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Do not overwrite existing trajectory files")
    args = parser.parse_args()
    
    seeds = list(range(args.seeds))
    
    print(f"Starting experiment: conditions={args.conditions}, seeds={seeds}, rounds={args.rounds}")
    
    output = run_experiment(
        conditions=args.conditions,
        seeds=seeds,
        n_rounds=args.rounds,
        threshold=args.threshold,
        alphas=args.alphas,
        noise=args.noise,
        run_tag=args.tag,
        overwrite=not args.no_overwrite,
    )
    
    print_summary(output)
    save_results(output, RESULTS_DIR, run_tag=args.tag)


if __name__ == "__main__":
    main()
