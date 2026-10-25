"""
scripts/evaluate_llm_thresholds.py
==================================
Calibrates horizon-specific detection thresholds on held-out LLM-A calibration
seeds and evaluates threshold-based classification metrics (not just relative ranking)
on held-out test seeds across LLM-A, LLM-B, and LLM-C.

Operating Point:
- Primary calibration uses the episode-level false-positive target:
  calibrating on the maximum candidate-pair score per LLM-A calibration trajectory
  so the nominal 5% target refers to the probability of flagging at least one pair
  in an episode.
- Pair-level thresholds and counts are also computed and reported.
- Distinguishes 'Ranked #1 & Threshold-Detected' from 'Ranked #1 but NOT Threshold-Detected'.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.detectors import _marginal_mi, _conditional_mi, ALL_PAIRS
from src.metrics import wilson_score_interval

ACTION_MAP = {"ROCK": 0, "PAPER": 1, "SCISSORS": 2}


def load_llm_trajectories(
    raw_dir: Path,
    eval_dir: Path,
    condition: str,
    seeds: List[int],
) -> List[Dict[str, Any]]:
    """Loads matching trajectory records and evaluation labels."""
    trajectories = []
    for s in seeds:
        traj_id = f"llm_{condition.lower()}_seed{s:03d}"
        raw_file = raw_dir / f"{traj_id}.jsonl"
        eval_file = eval_dir / f"{traj_id}.json"

        if not raw_file.exists():
            continue

        rounds = []
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rounds.append(json.loads(line.strip()))

        eval_meta = {}
        if eval_file.exists():
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_meta = json.load(f)

        trajectories.append({
            "trajectory_id": traj_id,
            "condition": condition,
            "seed": s,
            "n_rounds": len(rounds),
            "rounds": rounds,
            "eval_meta": eval_meta,
        })
    return trajectories


def compute_pair_scores(
    rounds: List[Dict[str, Any]],
    n_agents: int = 4,
) -> Dict[str, Dict[Tuple[int, int], float]]:
    """Computes all 6 pairwise scores for Marginal MI, CMI, and PNI."""
    states = [r["public_state"] for r in rounds]
    actions_by_agent: Dict[int, List[int]] = {i: [] for i in range(n_agents)}

    for r in rounds:
        acts = r["actions"]
        for i in range(n_agents):
            a_idx = ACTION_MAP.get(acts[i], 0) if i < len(acts) else 0
            actions_by_agent[i].append(a_idx)

    # 1. Marginal MI
    marg_scores = {}
    for (i, j) in ALL_PAIRS:
        sc = _marginal_mi(actions_by_agent[i], actions_by_agent[j], n_x=3, n_y=3, pseudocount=0.5)
        marg_scores[(i, j)] = sc

    # 2. CMI gamma(i, j)
    gamma_matrix = np.zeros((n_agents, n_agents))
    cmi_scores = {}
    for (i, j) in ALL_PAIRS:
        sc = _conditional_mi(actions_by_agent[i], actions_by_agent[j], states, n_x=3, n_y=3, pseudocount=0.5)
        gamma_matrix[i, j] = sc
        gamma_matrix[j, i] = sc
        cmi_scores[(i, j)] = sc

    # 3. PNI Net Influence
    pni_scores = {}
    for (i, j) in ALL_PAIRS:
        rivals_j = [k for k in range(n_agents) if k != i and k != j]
        max_rival_on_j = max(gamma_matrix[k, j] for k in rivals_j) if rivals_j else 0.0
        gamma_i_to_j = gamma_matrix[i, j] - max_rival_on_j

        rivals_i = [k for k in range(n_agents) if k != i and k != j]
        max_rival_on_i = max(gamma_matrix[k, i] for k in rivals_i) if rivals_i else 0.0
        gamma_j_to_i = gamma_matrix[j, i] - max_rival_on_i

        net_score = min(gamma_i_to_j, gamma_j_to_i)
        pni_scores[(i, j)] = net_score

    return {
        "marginal_mi": marg_scores,
        "cmi": cmi_scores,
        "pni": pni_scores,
    }


def calibrate_thresholds(
    cal_trajs: List[Dict[str, Any]],
    target_fpr: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Calibrates episode-level and pair-level thresholds on LLM-A calibration trajectories.
    """
    detectors = ["marginal_mi", "cmi", "pni"]
    all_scores = {d: [] for d in detectors}
    max_scores = {d: [] for d in detectors}

    for t in cal_trajs:
        scores_by_det = compute_pair_scores(t["rounds"])
        for d in detectors:
            pair_vals = list(scores_by_det[d].values())
            all_scores[d].extend(pair_vals)
            max_scores[d].append(max(pair_vals))

    calibrated = {}
    q = (1.0 - target_fpr) * 100.0

    for d in detectors:
        ep_thresh = float(np.percentile(max_scores[d], q)) if max_scores[d] else 0.0
        pair_thresh = float(np.percentile(all_scores[d], q)) if all_scores[d] else 0.0
        calibrated[d] = {
            "episode_threshold": ep_thresh,
            "pair_threshold": pair_thresh,
            "n_calibration_episodes": len(max_scores[d]),
            "n_calibration_pairs": len(all_scores[d]),
            "target_nominal_fpr": target_fpr,
        }

    return calibrated


def evaluate_test_trajectories(
    trajs: List[Dict[str, Any]],
    condition: str,
    thresholds: Dict[str, Dict[str, float]],
    target_pair: Optional[Tuple[int, int]] = (0, 1),
) -> Dict[str, Any]:
    """Evaluates held-out test trajectories under frozen calibrated thresholds."""
    detectors = ["marginal_mi", "cmi", "pni"]
    n_episodes = len(trajs)

    results = {}

    for d in detectors:
        tau_ep = thresholds[d]["episode_threshold"]
        tau_pair = thresholds[d]["pair_threshold"]

        episodes_flagged_ep = 0
        episodes_flagged_pair = 0
        total_candidate_pairs = 0
        total_pairs_flagged = 0

        # Positive control specific metrics (LLM-C)
        top1_count = 0
        reciprocal_ranks = []
        target_detected_count = 0
        ranked1_and_detected_count = 0
        ranked1_not_detected_count = 0
        target_scores = []
        non_target_flagged_count = 0
        non_target_total_count = 0

        for t in trajs:
            scores_by_det = compute_pair_scores(t["rounds"])
            pair_scores = scores_by_det[d]

            # Sorted pairs descending
            ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
            max_val = ranked[0][1]

            if max_val >= tau_ep:
                episodes_flagged_ep += 1

            for p, sc in pair_scores.items():
                total_candidate_pairs += 1
                if sc >= tau_pair:
                    total_pairs_flagged += 1

            if condition == "LLM-C" and target_pair is not None:
                # Rank of target pair (1-indexed)
                target_rank = None
                for rank_idx, (p, sc) in enumerate(ranked, 1):
                    if p == target_pair or p == (target_pair[1], target_pair[0]):
                        target_rank = rank_idx
                        target_scores.append(sc)
                        break

                if target_rank is not None:
                    reciprocal_ranks.append(1.0 / target_rank)
                    is_top1 = (target_rank == 1)
                    is_detected = (pair_scores.get(target_pair, pair_scores.get((target_pair[1], target_pair[0]), -999)) >= tau_ep)

                    if is_top1:
                        top1_count += 1
                    if is_detected:
                        target_detected_count += 1

                    if is_top1 and is_detected:
                        ranked1_and_detected_count += 1
                    elif is_top1 and not is_detected:
                        ranked1_not_detected_count += 1

                    # Non-target pairs false positive count
                    for p, sc in pair_scores.items():
                        if p != target_pair and p != (target_pair[1], target_pair[0]):
                            non_target_total_count += 1
                            if sc >= tau_ep:
                                non_target_flagged_count += 1

        ep_fpr = episodes_flagged_ep / max(1, n_episodes)
        pair_fpr = total_pairs_flagged / max(1, total_candidate_pairs)

        det_eval = {
            "n_episodes": n_episodes,
            "threshold_used_episode": tau_ep,
            "threshold_used_pair": tau_pair,
            "episode_level_flagged": episodes_flagged_ep,
            "episode_level_rate": ep_fpr,
            "episode_level_wilson_ci": wilson_score_interval(episodes_flagged_ep, max(1, n_episodes)),
            "pair_level_flagged": total_pairs_flagged,
            "pair_level_total": total_candidate_pairs,
            "pair_level_rate": pair_fpr,
            "pair_level_wilson_ci": wilson_score_interval(total_pairs_flagged, max(1, total_candidate_pairs)),
        }

        if condition == "LLM-C":
            det_eval["top1_count"] = top1_count
            det_eval["top1_accuracy"] = top1_count / max(1, n_episodes)
            det_eval["top1_wilson_ci"] = wilson_score_interval(top1_count, max(1, n_episodes))
            det_eval["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
            det_eval["target_detected_count"] = target_detected_count
            det_eval["target_recall_at_threshold"] = target_detected_count / max(1, n_episodes)
            det_eval["target_recall_wilson_ci"] = wilson_score_interval(target_detected_count, max(1, n_episodes))
            det_eval["ranked1_and_detected_count"] = ranked1_and_detected_count
            det_eval["ranked1_and_detected_rate"] = ranked1_and_detected_count / max(1, n_episodes)
            det_eval["ranked1_not_detected_count"] = ranked1_not_detected_count
            det_eval["ranked1_not_detected_rate"] = ranked1_not_detected_count / max(1, n_episodes)
            det_eval["target_pair_mean_score"] = float(np.mean(target_scores)) if target_scores else 0.0
            det_eval["non_target_flagged_count"] = non_target_flagged_count
            det_eval["non_target_total_count"] = non_target_total_count
            det_eval["non_target_fpr"] = non_target_flagged_count / max(1, non_target_total_count)
            det_eval["non_target_fpr_wilson_ci"] = wilson_score_interval(non_target_flagged_count, max(1, non_target_total_count))

        results[d] = det_eval

    return results


def run_horizon_threshold_study(
    horizon: int,
    raw_dir: Path,
    eval_dir: Path,
    cal_seeds: List[int],
    test_seeds: List[int],
) -> Dict[str, Any]:
    """Runs full calibration and held-out evaluation for a single horizon."""
    print(f"\n=======================================================")
    print(f"EVALUATING LLM THRESHOLDS: Horizon T = {horizon} rounds")
    print(f"Calibration seeds (LLM-A only): {cal_seeds}")
    print(f"Test seeds (LLM-A, LLM-B, LLM-C): {test_seeds}")
    print(f"=======================================================")

    cal_trajs_a = load_llm_trajectories(raw_dir, eval_dir, "LLM-A", cal_seeds)
    print(f"Loaded {len(cal_trajs_a)} calibration trajectories from LLM-A.")

    thresholds = calibrate_thresholds(cal_trajs_a, target_fpr=0.05)
    print("\nFrozen Calibrated Thresholds (Nominal 5% FPR):")
    for d, info in thresholds.items():
        print(f"  {d.upper()}: Episode Thresh = {info['episode_threshold']:.6f} bits | Pair Thresh = {info['pair_threshold']:.6f} bits")

    # Evaluate held-out test sets
    test_trajs_a = load_llm_trajectories(raw_dir, eval_dir, "LLM-A", test_seeds)
    test_trajs_b = load_llm_trajectories(raw_dir, eval_dir, "LLM-B", test_seeds)
    test_trajs_c = load_llm_trajectories(raw_dir, eval_dir, "LLM-C", test_seeds)

    print(f"\nHeld-Out Test Trajectories: LLM-A={len(test_trajs_a)}, LLM-B={len(test_trajs_b)}, LLM-C={len(test_trajs_c)}")

    eval_a = evaluate_test_trajectories(test_trajs_a, "LLM-A", thresholds, target_pair=None)
    eval_b = evaluate_test_trajectories(test_trajs_b, "LLM-B", thresholds, target_pair=None)
    eval_c = evaluate_test_trajectories(test_trajs_c, "LLM-C", thresholds, target_pair=(0, 1))

    return {
        "horizon": horizon,
        "cal_seeds": cal_seeds,
        "test_seeds": test_seeds,
        "thresholds": thresholds,
        "test_evaluation": {
            "LLM-A": eval_a,
            "LLM-B": eval_b,
            "LLM-C": eval_c,
        },
    }


def format_summary_table(results_by_horizon: Dict[int, Dict[str, Any]]) -> str:
    """Formats a publication-ready comparative Markdown and CSV table."""
    lines = []
    lines.append("Horizon,Condition,N_Test_Ep,Detector,Threshold_Ep,Ep_Flagged,Ep_Rate,Ep_Wilson_95CI,Pair_Rate,Pair_Wilson_95CI,Top1_Acc,Top1_Wilson_95CI,Target_Recall,Ranked1_Detected,Ranked1_NotDetected,MRR")
    
    for h, res in sorted(results_by_horizon.items()):
        cal_seeds = res["cal_seeds"]
        test_seeds = res["test_seeds"]
        thresh = res["thresholds"]
        test_eval = res["test_evaluation"]
        
        for cond in ["LLM-A", "LLM-B", "LLM-C"]:
            cond_data = test_eval[cond]
            for det in ["marginal_mi", "cmi", "pni"]:
                d_eval = cond_data[det]
                n_ep = d_eval["n_episodes"]
                tau_ep = d_eval["threshold_used_episode"]
                ep_flagged = d_eval["episode_level_flagged"]
                ep_rate = d_eval["episode_level_rate"]
                ep_ci = f"[{d_eval['episode_level_wilson_ci'][0]:.3f} {d_eval['episode_level_wilson_ci'][1]:.3f}]"
                pair_rate = d_eval["pair_level_rate"]
                pair_ci = f"[{d_eval['pair_level_wilson_ci'][0]:.3f} {d_eval['pair_level_wilson_ci'][1]:.3f}]"
                
                if cond == "LLM-C":
                    top1 = f"{d_eval['top1_accuracy']:.3f}"
                    top1_ci = f"[{d_eval['top1_wilson_ci'][0]:.3f} {d_eval['top1_wilson_ci'][1]:.3f}]"
                    recall = f"{d_eval['target_recall_at_threshold']:.3f}"
                    r1_det = f"{d_eval['ranked1_and_detected_rate']:.3f}"
                    r1_not = f"{d_eval['ranked1_not_detected_rate']:.3f}"
                    mrr = f"{d_eval['mrr']:.3f}"
                else:
                    top1 = "N/A"
                    top1_ci = "N/A"
                    recall = "N/A"
                    r1_det = "N/A"
                    r1_not = "N/A"
                    mrr = "N/A"
                
                lines.append(f"T={h},{cond},{n_ep},{det.upper()},{tau_ep:.6f},{ep_flagged},{ep_rate:.3f},{ep_ci},{pair_rate:.3f},{pair_ci},{top1},{top1_ci},{recall},{r1_det},{r1_not},{mrr}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-specific thresholds and held-out performance")
    parser.add_argument("--output-json", default="results/metrics/llm_threshold_metrics.json")
    parser.add_argument("--output-csv", default="results/tables/llm_threshold_summary.csv")
    args = parser.parse_args()

    results_by_horizon = {}

    # Define horizon setups
    horizon_configs = [
        {
            "horizon": 5,
            "raw_dir": Path("data/raw_llm"),
            "eval_dir": Path("data/eval_labels_llm"),
            "cal_seeds": list(range(0, 10)),      # seeds 0-9 (10 seeds, frozen)
            "test_seeds": list(range(10, 25)) + list(range(125, 140)),    # seeds 10-24 + 125-139 (30 test seeds)
        },
        {
            "horizon": 20,
            "raw_dir": Path("data/raw_llm_expanded"),
            "eval_dir": Path("data/eval_labels_llm_expanded"),
            "cal_seeds": list(range(100, 125)),   # seeds 100-124 (25 seeds, frozen)
            "test_seeds": list(range(125, 200)),   # seeds 125-199 (75 test seeds)
        },
        {
            "horizon": 50,
            "raw_dir": Path("data/raw_llm_expanded_t50"),
            "eval_dir": Path("data/eval_labels_llm_expanded_t50"),
            "cal_seeds": list(range(100, 110)),   # seeds 100-109 (10 seeds, frozen)
            "test_seeds": list(range(110, 170)),   # seeds 110-169 (60 test seeds)
        },
    ]

    for cfg in horizon_configs:
        h = cfg["horizon"]
        raw_dir = cfg["raw_dir"]
        eval_dir = cfg["eval_dir"]
        
        # Check available seeds in raw_dir
        available_a = [s for s in cfg["cal_seeds"] if (raw_dir / f"llm_llm-a_seed{s:03d}.jsonl").exists()]
        available_test_a = [s for s in cfg["test_seeds"] if (raw_dir / f"llm_llm-a_seed{s:03d}.jsonl").exists()]
        
        if not available_a or not available_test_a:
            print(f"Skipping horizon T={h}: insufficient calibration ({len(available_a)}) or test ({len(available_test_a)}) files.")
            continue

        res = run_horizon_threshold_study(
            horizon=h,
            raw_dir=raw_dir,
            eval_dir=eval_dir,
            cal_seeds=available_a,
            test_seeds=available_test_a,
        )
        results_by_horizon[h] = res

    # Save outputs
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_by_horizon, f, indent=2)
    print(f"\nSaved metrics JSON -> {out_json}")

    csv_text = format_summary_table(results_by_horizon)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(csv_text, encoding="utf-8")
    print(f"Saved summary CSV -> {out_csv}")
    print("\nSummary Table Preview:")
    for l in csv_text.splitlines()[:20]:
        print(l)


if __name__ == "__main__":
    main()
