"""
evaluate_llm_pilot.py
=====================
Evaluation script for the Exploratory Modal LLM Pilot traces.

Computes:
1. Ground-truth Signal Compliance for Condition LLM-C.
2. Detector scores for all 6 agent pairs across trajectories:
   - Marginal Mutual Information I(A_i; A_j)
   - Conditional Mutual Information I(A_i; A_j | S)
   - Pairwise Net Influence min(Gamma(i->j), Gamma(j->i))
3. Top-1 Pair Ranking Accuracy for Condition LLM-C (is pair (0,1) ranked #1?).
4. Descriptive statistics, valid decision counts, and latency benchmarks.
5. Saves results to results/metrics/llm_pilot_metrics.json and results/tables/llm_pilot_summary.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detectors import _marginal_mi, _conditional_mi, ALL_PAIRS, N_PAIRS

ACTION_MAP = {"ROCK": 0, "PAPER": 1, "SCISSORS": 2}


def load_llm_trajectory(raw_path: Path, eval_path: Path) -> Dict[str, Any]:
    """Loads a single LLM trajectory and its evaluation metadata."""
    detector_records = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                detector_records.append(json.loads(line.strip()))

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_meta = json.load(f)

    return {
        "trajectory_id": eval_meta["trajectory_id"],
        "condition": eval_meta["condition"],
        "seed": eval_meta["seed"],
        "n_rounds": eval_meta["n_rounds"],
        "target_pair": eval_meta.get("target_pair"),
        "total_decisions": eval_meta["total_decisions"],
        "valid_decisions": eval_meta["valid_decisions"],
        "invalid_decisions": eval_meta["invalid_decisions"],
        "mean_latency_sec": eval_meta.get("mean_latency_sec", 0.0),
        "signal_compliance_rate": eval_meta.get("signal_compliance_rate"),
        "records": detector_records,
        "eval_meta": eval_meta,
    }


def evaluate_trajectory_detectors(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Computes Marginal MI, CMI, and PNI scores for all 6 pairs on an LLM trajectory."""
    records = traj["records"]
    rounds = len(records)
    if rounds == 0:
        return {}

    states = [rec["public_state"] for rec in records]
    actions_by_agent: Dict[int, List[int]] = {i: [] for i in range(4)}

    for rec in records:
        acts = rec["actions"]
        for i in range(4):
            a_str = acts[i]
            a_idx = ACTION_MAP.get(a_str, 0)
            actions_by_agent[i].append(a_idx)

    # 1. Marginal MI
    marginal_mi_scores = {}
    for (i, j) in ALL_PAIRS:
        score = _marginal_mi(actions_by_agent[i], actions_by_agent[j], n_x=3, n_y=3, pseudocount=0.5)
        marginal_mi_scores[(i, j)] = score

    # 2. Conditional MI gamma(i, j)
    gamma_matrix = np.zeros((4, 4))
    cmi_scores = {}
    for (i, j) in ALL_PAIRS:
        score = _conditional_mi(actions_by_agent[i], actions_by_agent[j], states, n_x=3, n_y=3, pseudocount=0.5)
        gamma_matrix[i, j] = score
        gamma_matrix[j, i] = score
        cmi_scores[(i, j)] = score

    # 3. PNI Net Influence
    pni_scores = {}
    for (i, j) in ALL_PAIRS:
        rivals_j = [k for k in range(4) if k != i and k != j]
        max_rival_on_j = max(gamma_matrix[k, j] for k in rivals_j) if rivals_j else 0.0
        gamma_i_to_j = gamma_matrix[i, j] - max_rival_on_j

        rivals_i = [k for k in range(4) if k != i and k != j]
        max_rival_on_i = max(gamma_matrix[k, i] for k in rivals_i) if rivals_i else 0.0
        gamma_j_to_i = gamma_matrix[j, i] - max_rival_on_i

        net_score = min(gamma_i_to_j, gamma_j_to_i)
        pni_scores[(i, j)] = net_score

    # Check top-1 rankings for target pair (0, 1)
    target = (0, 1)
    mmi_ranked = sorted(marginal_mi_scores.items(), key=lambda x: x[1], reverse=True)
    cmi_ranked = sorted(cmi_scores.items(), key=lambda x: x[1], reverse=True)
    pni_ranked = sorted(pni_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "trajectory_id": traj["trajectory_id"],
        "condition": traj["condition"],
        "seed": traj["seed"],
        "n_rounds": rounds,
        "marginal_mi": {f"{i}-{j}": v for (i, j), v in marginal_mi_scores.items()},
        "cmi": {f"{i}-{j}": v for (i, j), v in cmi_scores.items()},
        "pni": {f"{i}-{j}": v for (i, j), v in pni_scores.items()},
        "top1_pair_marginal_mi": list(mmi_ranked[0][0]),
        "top1_pair_cmi": list(cmi_ranked[0][0]),
        "top1_pair_pni": list(pni_ranked[0][0]),
        "target_is_top1_mmi": (mmi_ranked[0][0] == target) if traj["target_pair"] else None,
        "target_is_top1_cmi": (cmi_ranked[0][0] == target) if traj["target_pair"] else None,
        "target_is_top1_pni": (pni_ranked[0][0] == target) if traj["target_pair"] else None,
        "target_pair_score_mmi": marginal_mi_scores.get(target),
        "target_pair_score_cmi": cmi_scores.get(target),
        "target_pair_score_pni": pni_scores.get(target),
    }


def evaluate_llm_pilot(
    raw_dir: Path = Path("data/raw_llm"),
    eval_dir: Path = Path("data/eval_labels_llm"),
    output_metrics: Path = Path("results/metrics/llm_pilot_metrics.json"),
    output_table: Path = Path("results/tables/llm_pilot_summary.csv"),
) -> Dict[str, Any]:
    """Runs evaluation over all trajectories in data/raw_llm/."""
    raw_files = sorted(list(raw_dir.glob("*.jsonl")))
    if not raw_files:
        print(f"No trajectory files found in {raw_dir}")
        return {}

    trajectories = []
    traj_evals = []
    for rf in raw_files:
        ef = eval_dir / f"{rf.stem}.json"
        if not ef.exists():
            continue
        traj = load_llm_trajectory(rf, ef)
        trajectories.append(traj)
        eval_res = evaluate_trajectory_detectors(traj)
        traj_evals.append(eval_res)

    # Aggregate by condition
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for t, e in zip(trajectories, traj_evals):
        cond = t["condition"]
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append({**t, **e})

    condition_summaries = {}
    table_rows = []

    for cond, items in sorted(by_condition.items()):
        n_traj = len(items)
        tot_dec = sum(item["total_decisions"] for item in items)
        val_dec = sum(item["valid_decisions"] for item in items)
        inval_dec = sum(item["invalid_decisions"] for item in items)
        latencies = [item["mean_latency_sec"] for item in items]
        mean_lat = float(np.mean(latencies)) if latencies else 0.0

        compliances = [item["signal_compliance_rate"] for item in items if item["signal_compliance_rate"] is not None]
        mean_compliance = float(np.mean(compliances)) if compliances else None

        # Top-1 accuracy for target pair
        target_top1_mmi = [item["target_is_top1_mmi"] for item in items if item["target_is_top1_mmi"] is not None]
        target_top1_cmi = [item["target_is_top1_cmi"] for item in items if item["target_is_top1_cmi"] is not None]
        target_top1_pni = [item["target_is_top1_pni"] for item in items if item["target_is_top1_pni"] is not None]

        acc_mmi = float(np.mean(target_top1_mmi)) if target_top1_mmi else None
        acc_cmi = float(np.mean(target_top1_cmi)) if target_top1_cmi else None
        acc_pni = float(np.mean(target_top1_pni)) if target_top1_pni else None

        # Average pair scores across all 6 pairs
        all_mmi = [val for item in items for val in item["marginal_mi"].values()]
        all_cmi = [val for item in items for val in item["cmi"].values()]
        all_pni = [val for item in items for val in item["pni"].values()]

        summary = {
            "condition": cond,
            "n_trajectories": n_traj,
            "total_decisions": tot_dec,
            "valid_decisions": val_dec,
            "invalid_decisions": inval_dec,
            "mean_latency_sec": mean_lat,
            "signal_compliance_rate": mean_compliance,
            "top1_accuracy_marginal_mi": acc_mmi,
            "top1_accuracy_cmi": acc_cmi,
            "top1_accuracy_pni": acc_pni,
            "mean_pair_marginal_mi": float(np.mean(all_mmi)) if all_mmi else 0.0,
            "mean_pair_cmi": float(np.mean(all_cmi)) if all_cmi else 0.0,
            "mean_pair_pni": float(np.mean(all_pni)) if all_pni else 0.0,
        }
        condition_summaries[cond] = summary

        table_rows.append({
            "Condition": cond,
            "N_Traj": n_traj,
            "Total_Decisions": tot_dec,
            "Valid_Rate": f"{val_dec / max(1, tot_dec):.3f}",
            "Mean_Latency_s": f"{mean_lat:.2f}",
            "Signal_Compliance": f"{mean_compliance:.3f}" if mean_compliance is not None else "N/A",
            "Top1_Marginal_MI": f"{acc_mmi:.3f}" if acc_mmi is not None else "N/A",
            "Top1_CMI": f"{acc_cmi:.3f}" if acc_cmi is not None else "N/A",
            "Top1_PNI": f"{acc_pni:.3f}" if acc_pni is not None else "N/A",
            "Mean_Marginal_MI_bits": f"{np.mean(all_mmi):.4f}" if all_mmi else "N/A",
            "Mean_CMI_bits": f"{np.mean(all_cmi):.4f}" if all_cmi else "N/A",
            "Mean_PNI_bits": f"{np.mean(all_pni):.4f}" if all_pni else "N/A",
        })

    # Save outputs
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics, "w", encoding="utf-8") as f:
        json.dump({
            "conditions": condition_summaries,
            "trajectories": traj_evals,
        }, f, indent=2)

    df = pd.DataFrame(table_rows)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_table, index=False)

    print(f"\n--- LLM Pilot Evaluation Complete ---")
    print(df.to_string(index=False))
    print(f"\nMetrics written to: {output_metrics}")
    print(f"Summary table written to: {output_table}")

    return condition_summaries


if __name__ == "__main__":
    evaluate_llm_pilot()
