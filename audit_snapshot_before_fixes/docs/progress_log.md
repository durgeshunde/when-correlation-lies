# Sprint Progress Log

**Sprint:** Apart Research AI Collusion Research Sprint (Oct 23–25, 2026)  
**Project:** When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents  

---

## Log of Activities

### 2026-10-23 20:15 IST — Phase 1: Research Preflight & Environment Audit
- Sprint page accessed and verified: sprint is LIVE, submission button active.
- Primary paper by Bonjour et al. (UAI 2022) retrieved and mathematical formulation audited:
  - $\gamma(i, j) = I(A_i; A_j \mid S)$
  - $\Gamma(i \to j) = \gamma(i, j) - \max_{k \neq i, j} \gamma(k, j)$
- Audited environment: Python 3.13.7, NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.11.2 available.
- Installed missing scientific dependencies: `pandas`, `scikit-learn`, `pytest`.
- Inspected Modal credit status: `MODAL_TOKEN_ID` not configured in environment. In strict adherence to rules, prioritized local CPU-only scientific pipeline to guarantee reproducible results without billable compute risk.

### 2026-10-23 20:25 IST — Phase 2: Implementation & Unit Testing
- Implemented `src/environment.py`: zero-sum 4-player iterated RPS game with Markovian state transitions.
- Implemented `src/agents.py`: modular agent classes for Conditions A, B, C, and D with isolated RNGs.
- Implemented `src/trajectories.py`: structured JSONL logging with separate evaluation manifest for ground truth.
- Implemented `src/detectors.py`: Pairwise Net Influence (PNI) with Jeffreys-smoothed CMI ($\lambda = 0.5$) and Marginal MI baseline.
- Implemented `src/metrics.py`: Top-1 Accuracy, MRR, FPR, true pair scores, and 2,000-sample bootstrap CIs.
- Developed comprehensive test suite in `tests/test_environment.py` and `tests/test_detectors.py`.
- **Validation Gate:** Test suite executed via pytest. **38/38 tests PASSED**.

### 2026-10-23 20:33 IST — Phase 3: Core Scripted Experiments
- Executed full 4-condition experiment:
  - 20 independent random seeds per condition.
  - $T = 200$ rounds per trajectory.
  - Conditions tested: A (independent), B (shared context), C (full coordination $\alpha=1.0$), D (coordination sweep $\alpha \in \{0.25, 0.50, 0.75, 1.00\}$).
- Resolved CSV formatting issue in `run_experiments.py`.
- Generated canonical outputs:
  - `results/metrics/final_metrics.json`
  - `results/tables/summary.csv`
  - Figures 1–4 in `results/figures/`.

### 2026-10-23 20:38 IST — Sensitivity Analysis
- Scripted and executed `scripts/run_sensitivity.py` across $T \in \{50, 100, 200, 500\}$ rounds (160 trajectories).
- Generated `results/tables/sensitivity_trajectory_length.csv` and `results/figures/fig5_trajectory_length_sensitivity.png`.
- Documented findings: PNI Top-1 accuracy reaches 100% at $T \ge 100$ rounds; spurious MI in Condition B remains elevated at $\approx 0.96-0.99$ bits regardless of $T$, while conditional PNI score declines from 0.024 to 0.006 bits as $T$ increases.
