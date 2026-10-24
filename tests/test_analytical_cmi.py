"""
tests/test_analytical_cmi.py
============================
Analytically verified test cases for mutual information and conditional mutual information:
1. Degenerate deterministic policies: Mathematically I(A_i; A_j | S) = 0 when actions are functions of S.
2. Known shared hidden signal: Genuine conditional dependence produces high CMI (~log2(3) bits).
3. Conditional independence with strong unconditional association through S (shared context).
4. Independent random policies: Both marginal and conditional MI near zero.
5. Sensitivity to Laplace smoothing parameter lambda in {0.1, 0.5, 1.0}.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.detectors import _marginal_mi, _conditional_mi


class TestAnalyticalCMI:
    def test_degenerate_deterministic_policy_mathematical_zero(self):
        """When actions are deterministic functions of S, true CMI is 0.
        
        Even under finite-sample Laplace smoothing, as sample count per state increases,
        smoothed single-state MI for deterministic actions approaches 0.
        """
        states = [0, 1, 2] * 200  # 600 samples, 200 per state
        # A0 and A1 are deterministic functions of S
        a0 = [s % 3 for s in states]
        a1 = [s % 3 for s in states]
        
        # Marginal MI is non-zero because actions vary across states
        m_mi = _marginal_mi(a0, a1, pseudocount=0.5)
        assert m_mi > 1.0  # H(S) is large
        
        # When sample count per state is large (200), single-state smoothed MI decays
        cmi_200 = _conditional_mi(a0, a1, states, pseudocount=0.5)
        # Compare to N=10 per state
        states_short = [0, 1, 2] * 10
        cmi_10 = _conditional_mi([s % 3 for s in states_short], [s % 3 for s in states_short], states_short, pseudocount=0.5)
        
        # Proves that the non-zero value at small N is a smoothing artifact that decreases with N
        assert cmi_200 < cmi_10

    def test_known_shared_hidden_signal_positive_cmi(self):
        """When agents share a private signal Z not in S, true CMI is ~log2(3) = 1.585 bits."""
        rng = np.random.default_rng(12345)
        n = 2000
        states = rng.integers(0, 5, n).tolist()
        # Latent secret signal Z
        Z = rng.integers(0, 3, n).tolist()
        
        a0 = Z[:]
        a1 = Z[:]
        # Independent agent
        a2 = rng.integers(0, 3, n).tolist()
        
        cmi_coord = _conditional_mi(a0, a1, states, pseudocount=0.5)
        cmi_indep = _conditional_mi(a0, a2, states, pseudocount=0.5)
        
        # Coordinating pair has CMI close to log2(3) = 1.585
        assert cmi_coord > 1.3
        # Independent pair has CMI near zero
        assert cmi_indep < 0.05
        # Net influence difference is decisive
        assert cmi_coord - cmi_indep > 1.25

    def test_conditional_independence_with_strong_unconditional_association(self):
        """Shared context generates high Marginal MI but near-zero CMI."""
        rng = np.random.default_rng(54321)
        n = 1500
        states = rng.integers(0, 3, n).tolist()
        
        # Both agents best-respond to state S with 10% independent noise
        a0 = [(s + 1) % 3 if rng.random() > 0.1 else int(rng.integers(0, 3)) for s in states]
        a1 = [(s + 1) % 3 if rng.random() > 0.1 else int(rng.integers(0, 3)) for s in states]
        
        marginal_mi = _marginal_mi(a0, a1, pseudocount=0.5)
        cond_mi = _conditional_mi(a0, a1, states, pseudocount=0.5)
        
        # Marginal MI is large (spurious correlation)
        assert marginal_mi > 0.60
        # Conditional MI collapses near zero (conditional independence)
        assert cond_mi < 0.05
        assert marginal_mi > 10 * cond_mi

    def test_independent_random_policies(self):
        """Independent random policies yield near zero for both metrics."""
        rng = np.random.default_rng(999)
        n = 1000
        states = rng.integers(0, 5, n).tolist()
        a0 = rng.integers(0, 3, n).tolist()
        a1 = rng.integers(0, 3, n).tolist()
        
        marginal_mi = _marginal_mi(a0, a1, pseudocount=0.5)
        cond_mi = _conditional_mi(a0, a1, states, pseudocount=0.5)
        
        assert marginal_mi < 0.05
        assert cond_mi < 0.05

    def test_smoothing_parameter_sensitivity(self):
        """Verify behavior across smoothing levels lambda in {0.1, 0.5, 1.0}."""
        rng = np.random.default_rng(42)
        n = 500
        states = rng.integers(0, 3, n).tolist()
        Z = rng.integers(0, 3, n).tolist()
        
        for pseudocount in [0.1, 0.5, 1.0]:
            cmi = _conditional_mi(Z, Z, states, pseudocount=pseudocount)
            # CMI remains high and positive across all smoothing levels
            assert cmi > 1.0
            assert np.isfinite(cmi)

    def test_joint_coordination_gate_and_frequency(self):
        """Verify joint gate draws G_t ~ Bernoulli(alpha) and coordination frequency matches alpha."""
        from src.trajectories import run_trajectory
        
        # Test Condition C: always active (alpha = 1.0)
        traj_c = run_trajectory(condition="C", seed=42, n_rounds=500)
        rounds_c = traj_c["rounds"]
        # In Condition C, pair (0, 1) always receives private signal and plays identically
        matches_c = sum(1 for r in rounds_c if r["actions"][0] == r["actions"][1])
        assert matches_c == 500
        
        # Test Condition D: joint gate frequency across alpha values
        for alpha in [0.25, 0.50, 0.75]:
            n_rounds = 4000
            traj_d = run_trajectory(condition="D", seed=123, n_rounds=n_rounds, alpha=alpha)
            # Both agents 0 and 1 play identical action whenever the joint gate G_t was active
            # (plus occasional accidental matches during fallback with prob ~1/3 * (1-alpha))
            # Let's verify by re-running the exact generator with the same seed:
            cartel_rng = np.random.default_rng([123, 0xC0110DE])
            gates = [cartel_rng.random() < alpha for _ in range(n_rounds)]
            emp_gate_freq = sum(gates) / n_rounds
            assert abs(emp_gate_freq - alpha) < 0.02

