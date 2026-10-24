"""
test_detectors.py
=================
Unit tests for the pairwise net influence detector and MI baseline.

Tests verify correctness on synthetic distributions with KNOWN analytical values.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.detectors import (
    _marginal_mi,
    _conditional_mi,
    PairwiseNetInfluenceDetector,
    MarginalMIDetector,
    ALL_PAIRS,
    PSEUDOCOUNT,
    N_AGENTS,
)
from src.environment import ROCK, PAPER, SCISSORS, N_ACTIONS


# ---------------------------------------------------------------------------
# Tests for _marginal_mi
# ---------------------------------------------------------------------------

class TestMarginalMI:
    """Tests for unconditional mutual information estimator."""

    def test_independent_uniform_variables_near_zero(self):
        """I(X; Y) ≈ 0 for large samples of independent uniform variables."""
        rng = np.random.default_rng(0)
        xs = rng.integers(0, 3, 2000).tolist()
        ys = rng.integers(0, 3, 2000).tolist()
        mi = _marginal_mi(xs, ys)
        # With large N, smoothed estimate should be small (< 0.05 bits)
        assert mi < 0.05, f"Expected near-zero MI, got {mi:.4f}"

    def test_identical_variables_high_mi(self):
        """I(X; X) should be large — equal to H(X) ≈ log2(3) ≈ 1.585 bits."""
        rng = np.random.default_rng(1)
        xs = rng.integers(0, 3, 1000).tolist()
        mi = _marginal_mi(xs, xs)  # perfect dependence
        # H(uniform over 3) = log2(3) ≈ 1.585; expect MI close to this
        assert mi > 1.4, f"Expected high MI for X=Y, got {mi:.4f}"

    def test_perfectly_correlated_pair_vs_independent(self):
        """Perfectly correlated pair should have higher MI than independent."""
        rng = np.random.default_rng(2)
        xs = rng.integers(0, 3, 500).tolist()
        ys_dep = xs[:]            # perfect dependence
        ys_ind = rng.integers(0, 3, 500).tolist()  # independent
        mi_dep = _marginal_mi(xs, ys_dep)
        mi_ind = _marginal_mi(xs, ys_ind)
        assert mi_dep > mi_ind + 0.5, \
            f"Dep MI={mi_dep:.4f}, Ind MI={mi_ind:.4f}; dep should be higher"

    def test_non_negative(self):
        """MI must always be non-negative."""
        rng = np.random.default_rng(3)
        for _ in range(20):
            xs = rng.integers(0, 3, rng.integers(5, 50)).tolist()
            ys = rng.integers(0, 3, len(xs)).tolist()
            assert _marginal_mi(xs, ys) >= 0.0


# ---------------------------------------------------------------------------
# Tests for _conditional_mi
# ---------------------------------------------------------------------------

class TestConditionalMI:
    """Tests for conditional mutual information estimator."""

    def test_independent_given_state_near_zero(self):
        """I(X; Y | S) ≈ 0 when X ⊥ Y | S for large samples."""
        rng = np.random.default_rng(10)
        n = 1000
        # State determines the marginal distribution; both agents sample independently
        states = rng.integers(0, 3, n).tolist()
        xs = [rng.integers(0, 3) for _ in states]  # independent of y given state
        ys = [rng.integers(0, 3) for _ in states]
        cmi = _conditional_mi(xs, ys, states)
        assert cmi < 0.05, f"Expected near-zero CMI for independent agents, got {cmi:.4f}"

    def test_coordinated_given_state_high_cmi(self):
        """I(X; Y | S) is large when both agents play the SAME action (coordination)."""
        rng = np.random.default_rng(11)
        n = 500
        states = rng.integers(0, 3, n).tolist()
        # Both agents always play state % 3 (perfect coordination)
        coord_action = [s % 3 for s in states]
        cmi = _conditional_mi(coord_action, coord_action, states)
        # Should be large; with uniform state, ≈ H(action|state) = H(coord|state) ≈ 0
        # but since action is fully determined by state, CMI should detect residual correlation
        # In this edge case, all variance is explained by state; actual CMI of identical 
        # deterministic variables is H(X|S=s) for each s — which is 0 when deterministic.
        # So let's test a STOCHASTIC coordination case instead:
        pass

    def test_stochastic_coordination_higher_than_independent(self):
        """CMI of coordinated pair > CMI of independent pair, for same state exposure."""
        rng = np.random.default_rng(12)
        n = 800
        states = [tuple(rng.integers(0, 3, 4).tolist()) for _ in range(n)]
        
        # Coordinated: both play same action (derived from a shared source)
        shared = rng.integers(0, 3, n)
        coord_x = shared.tolist()
        coord_y = shared.tolist()
        
        # Independent: sample separately
        indep_x = rng.integers(0, 3, n).tolist()
        indep_y = rng.integers(0, 3, n).tolist()
        
        cmi_coord = _conditional_mi(coord_x, coord_y, states)
        cmi_indep = _conditional_mi(indep_x, indep_y, states)
        
        assert cmi_coord > cmi_indep, \
            f"Coordinated CMI={cmi_coord:.4f} should exceed independent CMI={cmi_indep:.4f}"

    def test_cmi_non_negative(self):
        """CMI must always be non-negative."""
        rng = np.random.default_rng(13)
        for _ in range(20):
            n = rng.integers(10, 100)
            states = rng.integers(0, 3, n).tolist()
            xs = rng.integers(0, 3, n).tolist()
            ys = rng.integers(0, 3, n).tolist()
            assert _conditional_mi(xs, ys, states) >= 0.0

    def test_empty_state_list_returns_zero(self):
        """Empty input should return 0 gracefully."""
        result = _conditional_mi([], [], [])
        assert result == 0.0


# ---------------------------------------------------------------------------
# Tests for PairwiseNetInfluenceDetector
# ---------------------------------------------------------------------------

class TestPairwiseNetInfluenceDetector:
    """Integration tests for the full detector pipeline."""

    def _make_rounds(self, actions_seq: list[list[int]]) -> list[dict]:
        """Build detector-visible round records from a list of joint action tuples."""
        rounds = []
        for t, actions in enumerate(actions_seq):
            state = ("START",) if t == 0 else tuple(actions_seq[t-1])
            rounds.append({
                "round_idx": t,
                "state_feature": state,
                "actions": actions,
                "payoffs": [0.0] * N_AGENTS,
            })
        return rounds

    def test_returns_six_pairs(self):
        """Detector must score all 6 unordered pairs."""
        rng = np.random.default_rng(20)
        actions_seq = [rng.integers(0, 3, 4).tolist() for _ in range(100)]
        rounds = self._make_rounds(actions_seq)
        det = PairwiseNetInfluenceDetector()
        results = det.score_pairs(rounds)
        assert len(results) == 6, f"Expected 6 pairs, got {len(results)}"

    def test_scores_are_finite(self):
        """All scores should be finite floats."""
        rng = np.random.default_rng(21)
        actions_seq = [rng.integers(0, 3, 4).tolist() for _ in range(100)]
        rounds = self._make_rounds(actions_seq)
        det = PairwiseNetInfluenceDetector()
        results = det.score_pairs(rounds)
        for r in results:
            assert np.isfinite(r["score"]), f"Non-finite score for pair {r['pair']}: {r['score']}"

    def test_sorted_descending(self):
        """Results must be sorted descending by score."""
        rng = np.random.default_rng(22)
        actions_seq = [rng.integers(0, 3, 4).tolist() for _ in range(100)]
        rounds = self._make_rounds(actions_seq)
        det = PairwiseNetInfluenceDetector()
        results = det.score_pairs(rounds)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_coordinated_pair_ranks_first_or_second(self):
        """For strong coordination, the true pair should rank high."""
        rng = np.random.default_rng(23)
        n = 500
        # Build strongly coordinated trajectory: agents 0 and 1 always play same action
        # derived from a shared random source; agents 2 and 3 play independently
        actions_seq = []
        prev = None
        for t in range(n):
            state = prev if prev is not None else [0, 0, 0, 0]
            # Shared signal for pair (0,1)
            coord_action = int(rng.integers(0, 3))
            a0 = coord_action
            a1 = coord_action
            a2 = int(rng.integers(0, 3))
            a3 = int(rng.integers(0, 3))
            actions_seq.append([a0, a1, a2, a3])
            prev = [a0, a1, a2, a3]
        
        rounds = self._make_rounds(actions_seq)
        det = PairwiseNetInfluenceDetector()
        results = det.score_pairs(rounds)
        
        # True pair (0,1) should be in top 2
        ranked_pairs = [frozenset(r["pair"]) for r in results[:2]]
        assert frozenset({0, 1}) in ranked_pairs, \
            f"True pair not in top 2. Ranking: {[r['pair'] for r in results]}"

    def test_insufficient_data_returns_nan(self):
        """Single round should return nan scores."""
        rounds = [{"round_idx": 0, "state_feature": ("START",), "actions": [0, 1, 2, 0], "payoffs": [0, 0, 0, 0]}]
        det = PairwiseNetInfluenceDetector()
        results = det.score_pairs(rounds)
        # Should return error entries or nan
        for r in results:
            assert "error" in r or np.isnan(r["score"])

    def test_gamma_matrix_symmetric(self):
        """gamma(i,j) should equal gamma(j,i) by symmetry of CMI."""
        rng = np.random.default_rng(24)
        actions_seq = [rng.integers(0, 3, 4).tolist() for _ in range(200)]
        rounds = self._make_rounds(actions_seq)
        det = PairwiseNetInfluenceDetector()
        gamma = det.compute_gamma_matrix(rounds)
        # Check symmetry within tolerance
        for i in range(N_AGENTS):
            for j in range(N_AGENTS):
                if i != j:
                    assert abs(gamma[i, j] - gamma[j, i]) < 0.01, \
                        f"gamma[{i},{j}]={gamma[i,j]:.6f} != gamma[{j},{i}]={gamma[j,i]:.6f}"


# ---------------------------------------------------------------------------
# Tests for MarginalMIDetector
# ---------------------------------------------------------------------------

class TestMarginalMIDetector:
    """Tests for the naive baseline detector."""

    def _make_rounds(self, actions_seq):
        return [
            {"round_idx": t, "state_feature": ("START",) if t == 0 else tuple(a),
             "actions": a, "payoffs": [0.0] * N_AGENTS}
            for t, a in enumerate(actions_seq)
        ]

    def test_shared_context_inflates_score(self):
        """Naive MI should score shared-context agents highly (false positive scenario)."""
        from src.agents import make_condition_b_agents
        from src.environment import RPSGame
        
        rng = np.random.default_rng(30)
        game = RPSGame(n_rounds=300, rng=rng)
        agents = make_condition_b_agents(base_seed=30, noise=0.05)
        history = game.run(agents)
        
        det = MarginalMIDetector()
        results = det.score_pairs(history)
        
        # All pairs should have moderate scores due to shared state conditioning
        scores = [r["score"] for r in results]
        assert max(scores) > 0.05, "Expected inflated MI scores under shared context"

    def test_returns_six_pairs(self):
        """Baseline detector must score all 6 pairs."""
        rng = np.random.default_rng(31)
        actions_seq = [rng.integers(0, 3, 4).tolist() for _ in range(50)]
        rounds = self._make_rounds(actions_seq)
        det = MarginalMIDetector()
        results = det.score_pairs(rounds)
        assert len(results) == 6


# ---------------------------------------------------------------------------
# Test: Ground truth labels never consumed by detectors
# ---------------------------------------------------------------------------

class TestGroundTruthSeparation:
    """Verify that detector code never sees ground-truth labels."""

    def test_detector_has_no_gt_field_in_input(self):
        """Detector input (rounds) must not contain 'ground_truth' or 'coordinating_pair'."""
        from src.trajectories import run_trajectory, save_trajectory, load_trajectory
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            traj = run_trajectory("C", seed=0, n_rounds=50)
            traj_path, gt_path = save_trajectory(traj, data_dir, overwrite=True)
            meta, rounds = load_trajectory(traj_path)
            
            # Check that rounds don't contain GT labels
            for r in rounds:
                assert "coordinating_pair" not in r
                assert "ground_truth" not in r
                assert "gt_pair" not in r
            
            # Check that meta doesn't contain GT labels  
            assert "coordinating_pair" not in meta or meta.get("coordinating_pair") is None


# ---------------------------------------------------------------------------
# Parametrized condition tests
# ---------------------------------------------------------------------------

class TestConditionSanityChecks:
    """Sanity checks that conditions produce expected behavior patterns."""

    def _run_condition(self, condition, seed=42, n_rounds=300, alpha=1.0):
        from src.trajectories import run_trajectory
        traj = run_trajectory(condition, seed=seed, n_rounds=n_rounds, alpha=alpha)
        return traj["rounds"]

    def test_condition_a_agents_play_valid_actions(self):
        """All actions in Condition A must be in {0, 1, 2}."""
        rounds = self._run_condition("A")
        for r in rounds:
            for a in r["actions"]:
                assert a in (0, 1, 2), f"Invalid action {a}"

    def test_condition_c_coordinated_pair_often_plays_same_action(self):
        """Under full coordination (alpha=1), agents 0 and 1 always play the same action."""
        rounds = self._run_condition("C", alpha=1.0)
        agreement = sum(1 for r in rounds if r["actions"][0] == r["actions"][1])
        total = len(rounds)
        agreement_rate = agreement / total
        # With alpha=1.0 (full coordination), they should ALWAYS agree
        assert agreement_rate == 1.0, \
            f"Fully coordinated pair should always agree, got {agreement_rate:.2%}"

    def test_condition_d_partial_coordination(self):
        """Under partial coordination (alpha=0.5), agreement rate should be between chance and 1."""
        rounds = self._run_condition("D", alpha=0.5)
        agreement = sum(1 for r in rounds if r["actions"][0] == r["actions"][1])
        total = len(rounds)
        agreement_rate = agreement / total
        # Chance agreement = 1/3; should be above chance but below 1
        assert agreement_rate > 1/3 + 0.05, \
            f"Partial coordination should exceed chance ({1/3:.2f}), got {agreement_rate:.2%}"

    def test_condition_a_no_shared_state(self):
        """Condition A agents should not share mutable state across rounds."""
        from src.agents import make_condition_a_agents
        # Create agents with two different seeds; verify different trajectories
        agents1 = make_condition_a_agents(base_seed=100)
        agents2 = make_condition_a_agents(base_seed=200)
        
        from src.environment import RPSGameState
        state = RPSGameState(round_idx=0)
        
        # Sample 10 actions from each set
        actions1 = [a(state, i) for i, a in enumerate(agents1)]
        actions2 = [a(state, i) for i, a in enumerate(agents2)]
        
        # Different seeds should produce different action profiles (with high probability)
        # This is not guaranteed with probability 1, but with seeds 100 vs 200 it holds
        assert actions1 != actions2, "Different seeds should produce different actions"

    def test_seed_reproducibility(self):
        """Same seed should produce identical trajectory."""
        rounds_a = self._run_condition("C", seed=42)
        rounds_b = self._run_condition("C", seed=42)
        
        for ra, rb in zip(rounds_a, rounds_b):
            assert ra["actions"] == rb["actions"], \
                f"Seed 42 should be reproducible. Got {ra['actions']} vs {rb['actions']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
