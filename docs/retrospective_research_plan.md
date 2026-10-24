# Retrospective Research Protocol and Exploratory Hypotheses

**Project:** When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents  
**Event:** Apart Research AI Collusion Research Sprint (October 23–25, 2026)  
**Document Status:** Retrospective Research Protocol (Not Preregistered Prior to Initial Exploration)  
**Author:** Durgesh Unde (Independent Researcher)

---

## 1. Transparency and Provenance Statement
*This document was formalized during the sprint after initial exploratory prototyping to establish a frozen, auditable experimental protocol for the corrected benchmark runs. It must not be cited or represented as a formal pre-registration, as exploratory code and trajectory inspection occurred prior to freezing this protocol.*

---

## 2. Research Questions & Hypotheses

### Primary Question
How reliably does Pairwise Net Influence (PNI; Bonjour et al., 2022) distinguish deliberate agent coordination from spurious action correlation induced by shared context, and how do estimator choice, coordination strength, and trajectory length affect detection?

### Exploratory Hypotheses
- **H1 (Positive Control Sensitivity):** Under deliberate pairwise coordination mediated by a covert coordination device (Condition C), PNI achieves Top-1 identification accuracy significantly above the random chance baseline (1/6 ≈ 16.7%).
- **H2 (Shared-Context Confounding):** Agents independently reacting to shared public game state (Condition B) produce high unconditional action dependence (Marginal Mutual Information), inducing high false-positive rates under naive correlation metrics.
- **H3 (Conditional Disambiguation):** Conditioning on observable public state $S$ (via CMI) eliminates or drastically attenuates spurious influence scores, achieving false positive rates comparable to independent play.
- **H4 (Detection Boundary):** Top-1 identification accuracy degrades systematically as coordination strength $\alpha$ decreases towards independent play ($\alpha = 0$).

---

## 3. Experimental Design

### Game Environment
- **Game:** 4-Agent Iterated Simultaneous Rock-Paper-Scissors Tournament.
- **Action Space:** $\mathcal{A} = \{0: \text{Rock}, 1: \text{Paper}, 2: \text{Scissors}\}$.
- **Payoffs:** Zero-sum pairwise tournament; agent $i$ receives the sum of RPS outcomes against the 3 opponents.
- **Public State Space:** Joint action profile of the immediately preceding round, $S_t = (a_0^{t-1}, a_1^{t-1}, a_2^{t-1}, a_3^{t-1}) \in \mathcal{A}^4 \cup \{\text{START}\}$, yielding 82 possible states.

### Corrected Experimental Conditions
1. **Condition A (Independent Baseline):** 4 agents selecting actions uniformly at random with independently seeded RNGs. Zero state dependence; zero coordination.
2. **Condition B (Shared-Context Negative Control):** 4 agents independently playing a state-dependent best-response policy to the dominant prior action with 10% independent noise ($\epsilon = 0.10$). No private coordination mechanism exists.
3. **Condition C (Known-Coordination Positive Control):** Pair $(0, 1)$ coordinates via a private coordination device $Z_t \sim \text{Uniform}(\{0, 1, 2\})$ unobservable by the public or other agents. Agents 2 and 3 play independent best responses to public state.
4. **Condition D (Coordination Strength Sweep):** Pair $(0, 1)$ coordinates on private signal $Z_t$ with probability $\alpha \in \{0.25, 0.50, 0.75, 1.00\}$ and falls back to independent best-response with probability $1 - \alpha$.

---

## 4. Analysis Protocol

- **Estimator:** Empirical plug-in Conditional Mutual Information with Laplace smoothing ($\lambda = 0.50$) across observed states.
- **Sampling Unit:** Independent trajectory seed (20 seeds per condition: seeds 0–4 for threshold validation, seeds 5–19 for held-out testing).
- **Uncertainty Estimation:** Wilson score intervals for Top-1 proportions; non-parametric bootstrap (2,000 resamples) for continuous metrics.
- **Baseline Calibration:** Marginal MI threshold calibrated on validation seeds to target 5% FPR on Condition A, and evaluated on held-out seeds.
- **Ground-Truth Isolation:** Evaluator processes only trajectory action logs. Ground-truth pairing is maintained in isolated metadata files.
