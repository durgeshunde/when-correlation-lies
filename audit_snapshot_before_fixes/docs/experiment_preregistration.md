# Experiment Pre-Registration

**Project:** When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents  
**Event:** Apart Research AI Collusion Research Sprint (October 23–25, 2026)  
**Date Pre-Registered:** October 23, 2026  
**Status:** Frozen prior to final data synthesis

---

## 1. Research Questions & Hypotheses

### Primary Question
How reliably does Pairwise Net Influence (PNI; Bonjour et al., 2022) distinguish deliberate agent coordination from spurious action correlation induced by shared context, and how does detector performance scale with coordination strength and trajectory length?

### Hypotheses
- **H1 (Positive Control Sensitivity):** Under deliberate pairwise coordination (Condition C), PNI achieves Top-1 identification accuracy significantly above the random chance baseline (1/6 ≈ 16.7%).
- **H2 (Shared Context Confounding):** Agents independently reacting to shared public game state (Condition B) produce high unconditional action dependence (Marginal Mutual Information), inducing high false-positive rates under naive correlation metrics.
- **H3 (Conditional Disambiguation):** Conditioning on the observable public state $S$ (via CMI) eliminates or drastically attenuates spurious influence scores, achieving false positive rates comparable to independent play.
- **H4 (Detection Boundary):** Top-1 identification accuracy degrades systematically as coordination strength $\alpha$ decreases towards independent play ($\alpha = 0$).

---

## 2. Experimental Design

### Game Environment
- **Game:** 4-Agent Iterated Simultaneous Rock-Paper-Scissors.
- **Action Space:** $\mathcal{A} = \{0: \text{Rock}, 1: \text{Paper}, 2: \text{Scissors}\}$.
- **Payoffs:** Zero-sum pairwise tournament; agent $i$ receives the sum of RPS outcomes against the 3 opponents.
- **State Space:** Public joint action profile of the immediately preceding round, $S_t = (a_1^{t-1}, a_2^{t-1}, a_3^{t-1}, a_4^{t-1}) \in \mathcal{A}^4 \cup \{\text{START}\}$, yielding 82 possible states.

### Experimental Conditions
1. **Condition A (Independent Baseline):** 4 agents selecting actions uniformly at random with independently seeded RNGs.
2. **Condition B (Shared-Context Negative Control):** 4 agents independently playing a state-dependent best-response policy to the dominant prior action with 10% independent noise ($\epsilon = 0.10$). No coordination mechanism.
3. **Condition C (Known-Coordination Positive Control):** Pair $(0, 1)$ employs a deterministic coordination rule $a = \text{hash}(S) \pmod 3$, while agents $(2, 3)$ play independent best responses.
4. **Condition D (Coordination Strength Sweep):** Pair $(0, 1)$ coordinates with probability $\alpha \in \{0.25, 0.50, 0.75, 1.00\}$ and falls back to independent best-response with probability $1 - \alpha$.

---

## 3. Analysis Pipeline & Decision Rules

- **Estimator:** Empirical plug-in Conditional Mutual Information with Laplace/Jeffreys smoothing ($\lambda = 0.50$).
- **Sample Unit:** Independent trajectory seed (20 seeds per condition).
- **Uncertainty Estimation:** 2,000 non-parametric bootstrap iterations over seeds to compute 95% confidence intervals.
- **Decision Threshold:** $\tau = 0.00$ bits net influence (calibrated on seeds 0–4; tested across all seeds).
- **Ground-Truth Isolation:** Evaluator processes only trajectory action logs. Ground-truth pairing is maintained in isolated metadata files.
