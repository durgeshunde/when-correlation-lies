"""
detectors.py
============
Implementation of pairwise net influence (Bonjour et al. 2022) and baseline
detectors for collusion identification in multi-agent game trajectories.

Mathematical foundation (Bonjour et al. 2022, Section 3)
---------------------------------------------------------
In a fully observable simultaneous game with N agents:

    Individual influence of agent i on agent j (conditional mutual information):
        gamma(i, j) = I(A_i; A_j | S)
                    = H(A_j | S) - H(A_j | A_i, S)

    Directed net influence (i -> j):
        Gamma(i -> j) = gamma(i, j) - max_{k ≠ i, k ≠ j} gamma(k, j)

    Directed net influence (j -> i):
        Gamma(j -> i) = gamma(j, i) - max_{k ≠ i, k ≠ j} gamma(k, i)

    A pair (i, j) is flagged as colluding when:
        Gamma(i -> j) > threshold  AND  Gamma(j -> i) > threshold

Note on symmetry
----------------
In a simultaneous game, I(A_i; A_j | S) = I(A_j; A_i | S) so gamma(i, j) =
gamma(j, i). However, Gamma(i -> j) ≠ Gamma(j -> i) in general because the
maximizing competitor k differs.

Estimator & Regularization Rationale
------------------------------------
Standard discrete plug-in estimators handle zero-probability cells naturally using
the standard information-theoretic convention 0 * log2(0) = 0. Additive smoothing
(pseudocount lambda) is therefore NOT mathematically required to prevent division by zero.
Rather, symmetric additive smoothing (e.g., pseudocount lambda = 0.5, the Jeffreys prior
analog) is a documented estimator regularization choice that alters finite-sample bias
and variance in sparse state-action contingency tables.

Baseline detectors
------------------
Detector 1: Pairwise Net Influence (Bonjour et al. 2022) — CMI conditioned on S with competitor subtraction.
Detector 2: Conditional Mutual Information gamma(i, j) = I(A_i; A_j | S) — State-conditioned baseline without competitor subtraction.
Detector 3: Marginal Mutual Information I(A_i; A_j) — Unconditioned baseline ignoring state.

Triangulation:
- Comparing Detector 3 (Marginal MI) vs Detector 2 (Conditional MI) isolates the effect of conditioning on S.
- Comparing Detector 2 (Conditional MI) vs Detector 1 (PNI) isolates the effect of competitor subtraction.
"""

from __future__ import annotations
import numpy as np
from itertools import combinations
from typing import Optional

from .environment import N_AGENTS, N_ACTIONS

# All unordered pairs of agents
ALL_PAIRS: list[tuple[int, int]] = list(combinations(range(N_AGENTS), 2))
N_PAIRS = len(ALL_PAIRS)  # 6

# Smoothing pseudocount (Jeffreys-like)
PSEUDOCOUNT = 0.5


# ---------------------------------------------------------------------------
# Core entropy / mutual information utilities
# ---------------------------------------------------------------------------

def _empirical_dist_1d(values: list[int], n_vals: int, pseudocount: float) -> np.ndarray:
    """Smoothed empirical distribution over a 1D discrete variable."""
    counts = np.zeros(n_vals) + pseudocount
    for v in values:
        counts[v] += 1
    return counts / counts.sum()


def _entropy_from_dist(p: np.ndarray) -> float:
    """Shannon entropy H(X) = -sum p log2 p, treating 0*log(0) = 0."""
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _joint_counts_2d(
    xs: list[int],
    ys: list[int],
    n_x: int,
    n_y: int,
    pseudocount: float,
) -> np.ndarray:
    """Smoothed joint count matrix P(X, Y)."""
    counts = np.zeros((n_x, n_y)) + pseudocount
    for x, y in zip(xs, ys):
        counts[x, y] += 1
    return counts


def _marginal_mi(
    xs: list[int],
    ys: list[int],
    n_x: int = N_ACTIONS,
    n_y: int = N_ACTIONS,
    pseudocount: float = PSEUDOCOUNT,
) -> float:
    """Empirical mutual information I(X; Y) with optional additive smoothing.
    
    Uses standard discrete information-theoretic convention 0 * log2(0) = 0.
    """
    counts = _joint_counts_2d(xs, ys, n_x, n_y, pseudocount)
    total = counts.sum()
    if total == 0:
        return 0.0
    p_xy = counts / total
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    
    # I(X;Y) = sum_{x,y} p(x,y) log2(p(x,y) / (p(x) p(y))) with 0 log 0 = 0
    mask = p_xy > 0
    prod = p_x * p_y
    mi = np.sum(p_xy[mask] * np.log2(p_xy[mask] / prod[mask]))
    return float(max(mi, 0.0))  # numerical safety


def _conditional_mi(
    xs: list[int],
    ys: list[int],
    states: list,
    n_x: int = N_ACTIONS,
    n_y: int = N_ACTIONS,
    pseudocount: float = PSEUDOCOUNT,
) -> float:
    """Empirical conditional mutual information I(X; Y | S).
    
    Formula:
        I(X; Y | S) = H(X | S) - H(X | Y, S)
                    = sum_s P(s) * I(X; Y | S=s)
    
    Implementation:
        For each state value s observed, compute I(X; Y | S=s) weighted by
        the empirical P(s). Sum over all states.
        
    Note: Global smoothing (pseudocount added to all cells before splitting 
    by state) can underestimate CMI when many states are rare. We use 
    within-state smoothing: each state slice gets its own smoothed counts.
    This better preserves the conditional structure but requires more data.
    """
    if len(xs) == 0:
        return 0.0
    
    # Group by state
    state_groups: dict = {}
    for x, y, s in zip(xs, ys, states):
        s_key = s if isinstance(s, (str, int)) else tuple(s)
        if s_key not in state_groups:
            state_groups[s_key] = ([], [])
        state_groups[s_key][0].append(x)
        state_groups[s_key][1].append(y)
    
    n_total = len(xs)
    cmi = 0.0
    
    for s_key, (xs_s, ys_s) in state_groups.items():
        n_s = len(xs_s)
        p_s = n_s / n_total  # empirical P(S=s)
        mi_given_s = _marginal_mi(xs_s, ys_s, n_x, n_y, pseudocount)
        cmi += p_s * mi_given_s
    
    return float(max(cmi, 0.0))


# ---------------------------------------------------------------------------
# Detector 1: Pairwise Net Influence (Bonjour et al. 2022)
# ---------------------------------------------------------------------------

class PairwiseNetInfluenceDetector:
    """Faithful implementation of pairwise net influence from Bonjour et al. (2022).
    
    Method:
        1. For each ordered pair (i, j) and trajectory, compute
           gamma(i, j) = I(A_i; A_j | S).
        2. Compute Gamma(i->j) = gamma(i,j) - max_{k≠i,k≠j} gamma(k, j).
        3. Compute Gamma(j->i) = gamma(j,i) - max_{k≠i,k≠j} gamma(k, i).
        4. Score(i, j) = min(Gamma(i->j), Gamma(j->i)).
        5. Rank all 6 unordered pairs by this score.
        
    The pair (i, j) is flagged if Score(i, j) > threshold.
    Threshold is calibrated on held-out development data.
    
    Ground-truth label is NEVER consumed by this detector.
    """
    
    def __init__(self, threshold: float = 0.0, pseudocount: float = PSEUDOCOUNT):
        """
        Args:
            threshold: Minimum net influence to flag a pair. Default 0.0 means
                       any positive net influence counts. Calibrate on dev data.
            pseudocount: Laplace smoothing parameter.
        """
        self.threshold = threshold
        self.pseudocount = pseudocount
    
    def compute_gamma_matrix(self, rounds: list[dict]) -> np.ndarray:
        """Compute the symmetric individual influence matrix gamma(i,j).
        
        Args:
            rounds: List of round records with 'state_feature' and 'actions'.
            
        Returns:
            gamma: N_AGENTS x N_AGENTS matrix of CMI values.
                   Diagonal is 0. gamma[i,j] = I(A_i; A_j | S).
        """
        if len(rounds) < 2:
            raise ValueError("Need at least 2 rounds to compute influence.")
        
        # Extract sequences
        states = [r["state_feature"] for r in rounds]
        actions = np.array([r["actions"] for r in rounds])  # (T, N_AGENTS)
        
        gamma = np.zeros((N_AGENTS, N_AGENTS))
        
        for i in range(N_AGENTS):
            for j in range(N_AGENTS):
                if i == j:
                    continue
                gamma[i, j] = _conditional_mi(
                    xs=actions[:, i].tolist(),
                    ys=actions[:, j].tolist(),
                    states=states,
                    pseudocount=self.pseudocount,
                )
        
        return gamma
    
    def compute_net_influence(self, gamma: np.ndarray) -> np.ndarray:
        """Compute directed net influence Gamma(i->j) for all ordered pairs.
        
        Gamma(i->j) = gamma(i,j) - max_{k≠i, k≠j} gamma(k, j)
        
        Returns:
            net: N_AGENTS x N_AGENTS matrix. net[i, j] = Gamma(i->j).
                 Diagonal is NaN.
        """
        net = np.full((N_AGENTS, N_AGENTS), np.nan)
        
        for i in range(N_AGENTS):
            for j in range(N_AGENTS):
                if i == j:
                    continue
                competitors = [k for k in range(N_AGENTS) if k != i and k != j]
                if not competitors:
                    # Only 2 agents — no competitor; net influence = gamma itself
                    net[i, j] = gamma[i, j]
                else:
                    max_competitor = max(gamma[k, j] for k in competitors)
                    net[i, j] = gamma[i, j] - max_competitor
        
        return net
    
    def score_pairs(self, rounds: list[dict]) -> list[dict]:
        """Score all unordered pairs and return ranked list.
        
        Args:
            rounds: Detector-visible round records.
            
        Returns:
            List of dicts sorted descending by score, each containing:
            - 'pair': (i, j) tuple.
            - 'score': min(Gamma(i->j), Gamma(j->i)).
            - 'gamma_ij': individual influence I(A_i; A_j | S).
            - 'net_ij': Gamma(i->j).
            - 'net_ji': Gamma(j->i).
            - 'flagged': True if score > threshold.
        """
        if len(rounds) < 2:
            return [{"pair": p, "score": np.nan, "error": "insufficient_data"}
                    for p in ALL_PAIRS]
        
        try:
            gamma = self.compute_gamma_matrix(rounds)
        except Exception as e:
            return [{"pair": p, "score": np.nan, "error": str(e)}
                    for p in ALL_PAIRS]
        
        net = self.compute_net_influence(gamma)
        
        results = []
        for i, j in ALL_PAIRS:
            net_ij = float(net[i, j])
            net_ji = float(net[j, i])
            score = min(net_ij, net_ji)
            results.append({
                "pair": (i, j),
                "score": score,
                "gamma_ij": float(gamma[i, j]),
                "net_ij": net_ij,
                "net_ji": net_ji,
                "flagged": score > self.threshold,
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Detector 2: Conditional MI baseline (state-conditioned without rival subtraction)
# ---------------------------------------------------------------------------

class ConditionalMIDetector:
    """Baseline: symmetric empirical conditional mutual information gamma(i, j) = I(A_i; A_j | S).
    
    This detector conditions on the observable public state S, but does NOT perform
    directed rival subtraction (it evaluates raw CMI without computing Gamma(i -> j)).
    
    Triangulation role:
    1. Marginal MI vs Conditional MI isolates the causal effect of state conditioning.
    2. Conditional MI vs PNI isolates the causal effect of competitor subtraction.
    """
    
    def __init__(self, threshold: float = 0.0, pseudocount: float = PSEUDOCOUNT):
        self.threshold = threshold
        self.pseudocount = pseudocount
        
    def score_pairs(self, rounds: list[dict]) -> list[dict]:
        """Score all pairs by conditional MI gamma(i, j) = I(A_i; A_j | S)."""
        if len(rounds) < 2:
            return [{"pair": p, "score": np.nan, "error": "insufficient_data"}
                    for p in ALL_PAIRS]
        
        states = [r["state_feature"] for r in rounds]
        actions = np.array([r["actions"] for r in rounds])
        
        results = []
        for i, j in ALL_PAIRS:
            score = _conditional_mi(
                xs=actions[:, i].tolist(),
                ys=actions[:, j].tolist(),
                states=states,
                pseudocount=self.pseudocount,
            )
            results.append({
                "pair": (i, j),
                "score": float(score),
                "flagged": float(score) > self.threshold,
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Detector 3: Naive Marginal MI baseline
# ---------------------------------------------------------------------------

class MarginalMIDetector:
    """Baseline: unconditional empirical mutual information I(A_i; A_j).
    
    This detector IGNORES the shared state. It demonstrates unconditioned
    context confounding under Condition B.
    """
    
    def __init__(self, threshold: float = 0.0, pseudocount: float = PSEUDOCOUNT):
        self.threshold = threshold
        self.pseudocount = pseudocount
    
    def score_pairs(self, rounds: list[dict]) -> list[dict]:
        """Score all pairs by unconditional MI."""
        if len(rounds) < 2:
            return [{"pair": p, "score": np.nan, "error": "insufficient_data"}
                    for p in ALL_PAIRS]
        
        actions = np.array([r["actions"] for r in rounds])
        
        results = []
        for i, j in ALL_PAIRS:
            score = _marginal_mi(
                xs=actions[:, i].tolist(),
                ys=actions[:, j].tolist(),
                pseudocount=self.pseudocount,
            )
            results.append({
                "pair": (i, j),
                "score": float(score),
                "flagged": float(score) > self.threshold,
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ---------------------------------------------------------------------------
# Convenience: run all three detectors on a trajectory
# ---------------------------------------------------------------------------

def run_detectors(
    rounds: list[dict],
    threshold: float | dict[str, float] = 0.0,
    pseudocount: float = PSEUDOCOUNT,
) -> dict:
    """Run all three detectors on a trajectory and return combined results.
    
    Args:
        rounds: Detector-visible round records.
        threshold: Float threshold applied to all, or dict mapping detector name -> float.
        pseudocount: Smoothing parameter.
        
    Returns:
        Dict with 'pni', 'cmi', and 'marginal_mi' keys, each containing ranked pair lists.
    """
    if isinstance(threshold, (int, float)):
        th_pni = float(threshold)
        th_cmi = float(threshold)
        th_mi = float(threshold)
    elif isinstance(threshold, dict):
        th_pni = float(threshold.get("pni", 0.0))
        th_cmi = float(threshold.get("cmi", 0.0))
        th_mi = float(threshold.get("marginal_mi", 0.0))
    else:
        th_pni = th_cmi = th_mi = 0.0

    pni = PairwiseNetInfluenceDetector(threshold=th_pni, pseudocount=pseudocount)
    cmi = ConditionalMIDetector(threshold=th_cmi, pseudocount=pseudocount)
    mi = MarginalMIDetector(threshold=th_mi, pseudocount=pseudocount)
    
    return {
        "pni": pni.score_pairs(rounds),
        "cmi": cmi.score_pairs(rounds),
        "marginal_mi": mi.score_pairs(rounds),
    }
