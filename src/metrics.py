"""
src/metrics.py
==============
Evaluation metrics for collusion detection.

Granularity & Statistical Independence:
    Each episode/trajectory has 6 candidate pairs.
    Within an episode, the 6 pair scores share the underlying action stream and are
    statistically dependent.
    Therefore, the independent sampling unit is the TRAJECTORY (seed).
    All aggregation and confidence intervals are computed over independent seeds,
    never treating within-episode pairs as independent replicates.

Confidence Interval Methods:
    - For continuous metrics (scores, MRR, FPR): Non-parametric bootstrap over seeds (B = 2,000).
    - For bounded proportion metrics (Top-1 accuracy): Wilson score interval for binomial
      proportions, avoiding degenerate [1, 1] intervals on small sample sizes.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Optional


def top_k_accuracy(ranked_pairs: list[dict], true_pair: tuple[int, int], k: int = 1) -> float:
    """Whether the true pair appears in the top-k ranked pairs (1.0 or 0.0)."""
    true_set = frozenset(true_pair)
    for rank, entry in enumerate(ranked_pairs[:k]):
        if frozenset(entry["pair"]) == true_set:
            return 1.0
    return 0.0


def reciprocal_rank(ranked_pairs: list[dict], true_pair: tuple[int, int]) -> float:
    """Reciprocal rank of the true coordinating pair (1 / rank; 0.0 if not found)."""
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
    """False positive rate: fraction of non-coordinating candidate pairs scoring above threshold."""
    true_set = frozenset(true_pair) if true_pair else frozenset()
    
    negatives = [
        e for e in ranked_pairs
        if frozenset(e["pair"]) != true_set
    ]
    
    if not negatives:
        return np.nan
    
    flagged = sum(1 for e in negatives if e["score"] > threshold)
    return float(flagged / len(negatives))


def wilson_score_interval(
    k: int,
    n: int,
    confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for binomial proportion with continuity-aware boundaries."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    diff = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return (float(max(0.0, centre - diff)), float(min(1.0, centre + diff)))


def chance_top1_accuracy(n_pairs: int = 6) -> float:
    """Random guessing chance baseline for Top-1 accuracy across n symmetric pairs: 1 / n."""
    return 1.0 / n_pairs


def calibrate_thresholds_from_validation(
    validation_records: list[dict],
    target_fpr: float = 0.05,
) -> dict[str, float]:
    """Calibrate decision thresholds on validation trajectories of Condition A.
    
    Args:
        validation_records: List of dicts, each containing 'detector_results'
                            from run_detectors() for Condition A validation seeds.
        target_fpr: Nominal false positive rate target on validation negatives.
        
    Returns:
        Dict mapping detector_name -> calibrated threshold value.
    """
    scores_by_det: dict[str, list[float]] = {}
    for rec in validation_records:
        for det_name, ranked_pairs in rec["detector_results"].items():
            if det_name not in scores_by_det:
                scores_by_det[det_name] = []
            for entry in ranked_pairs:
                if not np.isnan(entry["score"]):
                    scores_by_det[det_name].append(float(entry["score"]))
                    
    calibrated = {}
    for det_name, scores in scores_by_det.items():
        if scores:
            # 1 - target_fpr percentile
            pct = 100.0 * (1.0 - target_fpr)
            calibrated[det_name] = float(np.percentile(scores, pct))
        else:
            calibrated[det_name] = 0.0
    return calibrated


def evaluate_trajectory(
    detector_results: dict,
    true_pair: Optional[tuple[int, int]],
    condition: str,
    thresholds: Optional[dict[str, float] | float] = None,
) -> dict:
    """Compute all metrics for one trajectory across detectors."""
    if thresholds is None:
        th_dict = {"pni": 0.0, "cmi": 0.0, "marginal_mi": 0.0}
    elif isinstance(thresholds, (int, float)):
        th_dict = {"pni": float(thresholds), "cmi": float(thresholds), "marginal_mi": float(thresholds)}
    else:
        th_dict = thresholds
        
    result = {"condition": condition, "has_true_pair": true_pair is not None}
    true_set = frozenset(true_pair) if true_pair else frozenset()
    
    for detector_name, ranked_pairs in detector_results.items():
        th = th_dict.get(detector_name, 0.0)
        
        if any("error" in e for e in ranked_pairs):
            result[f"{detector_name}_top1"] = np.nan
            result[f"{detector_name}_mrr"] = np.nan
            result[f"{detector_name}_fpr"] = np.nan
            result[f"{detector_name}_true_score"] = np.nan
            result[f"{detector_name}_top_score"] = np.nan
            result[f"{detector_name}_flagged_count"] = np.nan
            result[f"{detector_name}_negatives_count"] = np.nan
            continue
        
        top_score = ranked_pairs[0]["score"] if ranked_pairs else np.nan
        result[f"{detector_name}_top_score"] = float(top_score) if not np.isnan(top_score) else np.nan
        
        negatives = [e for e in ranked_pairs if frozenset(e["pair"]) != true_set]
        neg_count = len(negatives)
        flagged_count = sum(1 for e in negatives if e["score"] > th)
        
        result[f"{detector_name}_flagged_count"] = flagged_count
        result[f"{detector_name}_negatives_count"] = neg_count
        result[f"{detector_name}_fpr"] = float(flagged_count / neg_count) if neg_count > 0 else np.nan
        
        if true_pair is not None:
            result[f"{detector_name}_top1"] = top_k_accuracy(ranked_pairs, true_pair, k=1)
            result[f"{detector_name}_mrr"] = reciprocal_rank(ranked_pairs, true_pair)
            result[f"{detector_name}_true_score"] = true_pair_score(ranked_pairs, true_pair)
        else:
            result[f"{detector_name}_top1"] = np.nan
            result[f"{detector_name}_mrr"] = np.nan
            result[f"{detector_name}_true_score"] = np.nan
    
    return result


def aggregate_metrics(
    per_seed_results: list[dict],
    bootstrap_n: int = 2000,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Aggregate per-seed metrics with proper confidence intervals and degeneracy flags."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    if not per_seed_results:
        return {}
    
    keys = [k for k, v in per_seed_results[0].items() if isinstance(v, (int, float))]
    aggregated = {}
    
    for key in keys:
        values = np.array([r[key] for r in per_seed_results], dtype=float)
        valid = values[~np.isnan(values)]
        n = len(valid)
        
        if n == 0:
            aggregated[key] = {
                "mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n": 0, "is_degenerate": False
            }
            continue
        
        mean = float(np.mean(valid))
        std = float(np.std(valid, ddof=min(1, n - 1)))
        
        # Bounded binomial proportion metrics: use Wilson Score interval
        if key.endswith("_top1") and np.all(np.isin(valid, [0.0, 1.0])):
            k_success = int(np.sum(valid))
            ci_low, ci_high = wilson_score_interval(k_success, n, confidence)
            is_degenerate = bool(n > 0 and (k_success == 0 or k_success == n))
        else:
            # Continuous metrics: non-parametric bootstrap over seeds
            boot_means = np.array([
                np.mean(rng.choice(valid, size=n, replace=True))
                for _ in range(bootstrap_n)
            ])
            alpha_tail = (1 - confidence) / 2
            ci_low = float(np.percentile(boot_means, 100 * alpha_tail))
            ci_high = float(np.percentile(boot_means, 100 * (1 - alpha_tail)))
            is_degenerate = bool(np.isclose(ci_low, ci_high, atol=1e-6) or np.isclose(std, 0.0, atol=1e-6))
        
        agg_entry = {
            "mean": mean,
            "std": std,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n,
            "is_degenerate": is_degenerate,
        }
        
        # Add underlying counts for FPR metrics
        if key.endswith("_fpr"):
            det_prefix = key.replace("_fpr", "")
            flagged_key = f"{det_prefix}_flagged_count"
            negatives_key = f"{det_prefix}_negatives_count"
            flagged_vals = [r[flagged_key] for r in per_seed_results if flagged_key in r and not np.isnan(r[flagged_key])]
            neg_vals = [r[negatives_key] for r in per_seed_results if negatives_key in r and not np.isnan(r[negatives_key])]
            agg_entry["flagged_total"] = int(np.sum(flagged_vals)) if flagged_vals else 0
            agg_entry["negatives_total"] = int(np.sum(neg_vals)) if neg_vals else 0
            agg_entry["candidates_per_seed"] = (agg_entry["negatives_total"] // n) if n > 0 else 0
            agg_entry["n_seeds"] = n

        aggregated[key] = agg_entry
    
    return aggregated
