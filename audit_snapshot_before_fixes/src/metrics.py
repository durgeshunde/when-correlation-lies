"""
metrics.py
==========
Evaluation metrics for collusion detection.

Each episode has 6 candidate pairs. The metrics evaluate how well each 
detector ranks the true coordinating pair.

Unit of uncertainty: independent trajectories (different seeds).
Bootstrap confidence intervals are computed over seeds, not over pairs within
a single trajectory — see design note below.

Design note
-----------
Within a single 4-agent trajectory, the 6 pair scores are NOT independent:
they share the same underlying action sequences. Using all 6 pairs as if they 
were independent observations would inflate the apparent sample size and 
underestimate variance. We therefore:
  - Report metrics per trajectory (seed-level granularity).
  - Aggregate over seeds.
  - Bootstrap confidence intervals over seeds.
  - Never treat the 6 within-episode pairs as independent replicates.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


def top_k_accuracy(ranked_pairs: list[dict], true_pair: tuple[int, int], k: int = 1) -> float:
    """Whether the true pair appears in the top-k ranked pairs.
    
    Args:
        ranked_pairs: Sorted (descending) list of pair score dicts.
        true_pair: The ground-truth coordinating pair as (i, j) with i < j.
        k: Rank cutoff.
        
    Returns:
        1.0 if true pair is in top k, 0.0 otherwise.
    """
    true_set = frozenset(true_pair)
    for rank, entry in enumerate(ranked_pairs[:k]):
        if frozenset(entry["pair"]) == true_set:
            return 1.0
    return 0.0


def reciprocal_rank(ranked_pairs: list[dict], true_pair: tuple[int, int]) -> float:
    """Mean reciprocal rank component for one trajectory.
    
    Returns:
        1/rank of the true pair (1-based). 0 if not found.
    """
    true_set = frozenset(true_pair)
    for rank, entry in enumerate(ranked_pairs, start=1):
        if frozenset(entry["pair"]) == true_set:
            return 1.0 / rank
    return 0.0


def true_pair_score(ranked_pairs: list[dict], true_pair: tuple[int, int]) -> float:
    """Return the detection score assigned to the true pair."""
    true_set = frozenset(true_pair)
    for entry in ranked_pairs:
        if frozenset(entry["pair"]) == true_set:
            return float(entry["score"]) if not np.isnan(entry["score"]) else np.nan
    return np.nan


def false_positive_rate(
    ranked_pairs: list[dict],
    threshold: float = 0.0,
    true_pair: Optional[tuple[int, int]] = None,
) -> float:
    """False positive rate: fraction of non-true pairs flagged.
    
    For conditions A and B (no coordinating pair), all 6 pairs are negative.
    For conditions C and D, 5 pairs are negative.
    
    Args:
        ranked_pairs: Sorted list of pair score dicts with 'flagged' key.
        threshold: The flagging threshold (used if 'flagged' key absent).
        true_pair: If provided, exclude the true pair from FP calculation.
    """
    true_set = frozenset(true_pair) if true_pair else frozenset()
    
    negatives = [
        e for e in ranked_pairs
        if frozenset(e["pair"]) != true_set
    ]
    
    if not negatives:
        return np.nan
    
    # Use 'flagged' key if present, else apply threshold
    flagged = sum(
        1 for e in negatives
        if e.get("flagged", e["score"] > threshold)
    )
    
    return float(flagged / len(negatives))


def aggregate_metrics(
    per_seed_results: list[dict],
    bootstrap_n: int = 2000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Aggregate per-seed metrics and compute bootstrap confidence intervals.
    
    Args:
        per_seed_results: List of per-seed metric dicts with numeric values.
        bootstrap_n: Number of bootstrap samples.
        confidence: CI coverage.
        rng: Random number generator for bootstrap.
        
    Returns:
        Dict with mean, std, CI lower, CI upper for each metric key.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    if not per_seed_results:
        return {}
    
    keys = [k for k, v in per_seed_results[0].items() if isinstance(v, (int, float))]
    aggregated = {}
    
    for key in keys:
        values = np.array([r[key] for r in per_seed_results], dtype=float)
        valid = values[~np.isnan(values)]
        
        if len(valid) == 0:
            aggregated[key] = {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
            continue
        
        mean = float(np.mean(valid))
        std = float(np.std(valid, ddof=min(1, len(valid)-1)))
        
        # Bootstrap CI
        boot_means = np.array([
            np.mean(rng.choice(valid, size=len(valid), replace=True))
            for _ in range(bootstrap_n)
        ])
        alpha = (1 - confidence) / 2
        ci_low = float(np.percentile(boot_means, 100 * alpha))
        ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))
        
        aggregated[key] = {
            "mean": mean,
            "std": std,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": len(valid),
        }
    
    return aggregated


def chance_top1_accuracy(n_pairs: int = 6) -> float:
    """Random chance baseline for top-1 accuracy: 1/n_pairs."""
    return 1.0 / n_pairs


def evaluate_trajectory(
    detector_results: dict,
    true_pair: Optional[tuple[int, int]],
    condition: str,
    threshold: float = 0.0,
) -> dict:
    """Compute all metrics for one trajectory.
    
    Args:
        detector_results: Output from detectors.run_detectors().
        true_pair: Ground-truth coordinating pair (None for conditions A, B).
        condition: Condition code.
        threshold: Flagging threshold.
        
    Returns:
        Dict of metric values for each detector.
    """
    result = {"condition": condition, "has_true_pair": true_pair is not None}
    
    for detector_name, ranked_pairs in detector_results.items():
        if any("error" in e for e in ranked_pairs):
            result[f"{detector_name}_top1"] = np.nan
            result[f"{detector_name}_mrr"] = np.nan
            result[f"{detector_name}_fpr"] = np.nan
            result[f"{detector_name}_true_score"] = np.nan
            continue
        
        if true_pair is not None:
            result[f"{detector_name}_top1"] = top_k_accuracy(ranked_pairs, true_pair, k=1)
            result[f"{detector_name}_mrr"] = reciprocal_rank(ranked_pairs, true_pair)
            result[f"{detector_name}_true_score"] = true_pair_score(ranked_pairs, true_pair)
            result[f"{detector_name}_fpr"] = false_positive_rate(ranked_pairs, threshold, true_pair)
        else:
            # No true pair: only measure FPR (all pairs are negative)
            result[f"{detector_name}_top1"] = np.nan
            result[f"{detector_name}_mrr"] = np.nan
            result[f"{detector_name}_true_score"] = np.nan
            result[f"{detector_name}_fpr"] = false_positive_rate(ranked_pairs, threshold, None)
    
    return result
