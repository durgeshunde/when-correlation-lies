# Scientific and Methodological Audit

**Project:** When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents  
**Date of Audit:** October 23, 2026  
**Auditor:** Durgesh Unde (Independent Researcher)

---

## 1. Executive Summary of Audit Findings

A rigorous audit of the initial prototype implementation revealed three significant methodological issues and one reporting discrepancy:

1. **Positive Control Mathematical Invalidation (Critical):**  
   In the original Condition C, the coordinating pair $(0, 1)$ generated actions deterministically via $a = \text{hash}(S) \pmod 3$, where $S$ was the public game state. Because both actions were deterministic functions of $S$, the true population conditional mutual information $I(A_0; A_1 \mid S)$ was mathematically **identically zero**. The reported signal ($\approx 0.0304$ bits) and 100% Top-1 accuracy were an artifact of Laplace/Jeffreys smoothing ($\lambda = 0.5$) concentrating counts into a single empirical cell $(a, a)$ under sparse sampling, rather than genuine conditional action dependence.

2. **State-Space Sparsity in 4-Agent RPS:**  
   In a 4-agent game with action space $|\mathcal{A}| = 3$, the joint action history produces $3^4 = 81$ states plus START ($|\mathcal{S}| = 82$). In $T = 200$ rounds, visit counts are severely skewed: in Condition A, 78 states are visited with a mean of only $\approx 2.5$ visits/state; in Conditions B and C, agents concentrate on only 18–27 states. Estimating 9 transition cells per state from 1–3 visits leaves estimates heavily influenced by the prior/pseudocount.

3. **Calibration Flaw in the Marginal-MI Baseline:**  
   In the original prototype, both PNI and Marginal MI were evaluated using a fixed classification threshold $\tau = 0.0$. Because empirical marginal mutual information $\hat{I}(A_i; A_j)$ is non-negative and strictly positive under finite sampling noise, a threshold of $\tau = 0.0$ resulted in a trivial False Positive Rate of 1.0 (100%) for Marginal MI across all conditions, rendering thresholded FPR uninformative as a comparative metric.

4. **Accounting Discrepancy in Trajectory Counts:**  
   The previous session handoff claimed 46,000 sensitivity rounds and 74,000 total rounds. The actual sensitivity design was $20 \text{ seeds} \times 2 \text{ conditions} \times (50 + 100 + 200 + 500) = 34,000$ rounds. Together with the primary benchmark ($140 \times 200 = 28,000$ rounds), the true total simulated was 62,000 rounds across 300 trajectory files.

5. **Preregistration Provenance:**  
   The document titled `experiment_preregistration.md` was drafted after exploratory runs had already commenced. To adhere to strict scientific integrity standards, this document must be relabeled as an *exploratory hypotheses and retrospective research protocol*, and results must not be claimed as confirmatory preregistered findings.

---

## 2. Detailed Mathematical Diagnosis of the Positive Control

### 2.1 The Theoretical Identity
Let $S$ denote the publicly observable game state. Suppose agents $i$ and $j$ execute actions $A_i$ and $A_j$. In the original implementation:
$$A_i = f(S), \quad A_j = f(S)$$
where $f(s) = \text{hash}(s) \pmod 3$.

For any observed state $s \in \mathcal{S}$:
$$P(A_i = a \mid S = s) = \begin{cases} 1 & \text{if } a = f(s) \\ 0 & \text{otherwise} \end{cases}$$
The conditional Shannon entropy of $A_i$ given $S = s$ is:
$$H(A_i \mid S = s) = - \sum_{a \in \mathcal{A}} P(a \mid s) \log_2 P(a \mid s) = - 1 \log_2(1) = 0 \text{ bits}$$
Similarly, $H(A_j \mid S = s) = 0$ and $H(A_i, A_j \mid S = s) = 0$.
By the definition of conditional mutual information:
$$I(A_i; A_j \mid S = s) = H(A_i \mid S = s) + H(A_j \mid S = s) - H(A_i, A_j \mid S = s) = 0 + 0 - 0 = 0$$
Taking the expectation over all states:
$$I(A_i; A_j \mid S) = \sum_{s \in \mathcal{S}} P(S = s) I(A_i; A_j \mid S = s) \equiv 0 \text{ bits}$$

**Conclusion:** The true population Conditional Mutual Information of any two deterministic functions of $S$ is identically zero.

### 2.2 Why Did the Plug-in Estimator Report $\approx 0.0304$ Bits?
With Laplace smoothing parameter $\lambda = 0.50$, when state $s$ is visited $N_s$ times and both agents play identical deterministic actions $a_i = a_j = c$:
- Count matrix cell $(c, c)$ has $N_s + \lambda = N_s + 0.5$.
- The other 8 cells $(a, b) \neq (c, c)$ have count $0 + \lambda = 0.5$.
- Total smoothed state count is $N_s + 9\lambda = N_s + 4.5$.
- Marginal counts: row $c$ has $N_s + 3\lambda = N_s + 1.5$; rows $a \neq c$ have $3\lambda = 1.5$.

The smoothed plug-in mutual information for this single state is:
$$\hat{I}(A_i; A_j \mid S = s) = \sum_{a, b} \hat{P}(a, b) \log_2 \frac{\hat{P}(a, b)}{\hat{P}(a)\hat{P}(b)}$$
For $N_s = 5$ and $\lambda = 0.5$, this yields:
$$\hat{I}(A_i; A_j \mid S = s) \approx 0.186 \text{ bits}$$
For independent random agents where counts spread across multiple cells, the smoothed MI for $N_s = 5$ evaluates to $\approx 0.100$ bits.
The difference $(0.186 - 0.100) \approx 0.086$ bits, when averaged over visited states and reduced by competitor subtraction, produced the apparent net influence score of $\approx 0.0304$ bits!

Thus, the detector was not measuring true residual conditional dependence; it was detecting the difference in smoothing distortion between single-cell clustered distributions and multi-cell uniform distributions.

---

## 3. The Corrected Scientific Design

### 3.1 Game-Theoretic Formulation of True Collusion
In multi-agent game theory and anti-trust economics (Bonjour et al., 2022; Calvano et al., 2020), collusive coordination cannot be achieved purely through publicly available state if external observers and competitors also condition on that state. Collusion fundamentally requires a **shared private coordination device** (e.g., secret communication channel, correlated equilibrium signal, or pre-arranged private cryptographic key) that is visible to the coordinating cartel but hidden from non-colluding agents and the auditor.

### 3.2 Corrected Positive Control (Condition C)
- Let $Z_t \in \{0, 1, 2\}$ be a latent private coordination signal sampled independently of public state $S_t$:
  $$Z_t \sim \text{Uniform}(\{0, 1, 2\})$$
- Agents 0 and 1 (the coordinating pair) observe both public state $S_t$ and the private signal $Z_t$.
- When coordinating, they play $A_0^t = Z_t$ and $A_1^t = Z_t$ (or joint best response conditional on $Z_t$).
- Non-coordinating agents 2 and 3 observe **only** the public state $S_t$ and have no access to $Z_t$.
- The auditor/detector observes **only** public state $S_t$ and action sequences $\{A_i^t\}$.

**Mathematical Verification of True CMI:**
Given public state $S_t = s$, $Z_t$ retains full conditional entropy $H(Z_t \mid S_t = s) = \log_2(3) \approx 1.585$ bits.
Because $A_0 = A_1 = Z_t$:
$$H(A_0 \mid S = s) = \log_2(3), \quad H(A_1 \mid S = s) = \log_2(3), \quad H(A_0, A_1 \mid S = s) = \log_2(3)$$
$$I(A_0; A_1 \mid S = s) = \log_2(3) + \log_2(3) - \log_2(3) = \log_2(3) \approx 1.585 \text{ bits}$$
For any non-coordinating agent $k \in \{2, 3\}$, who has no access to $Z_t$:
$$A_k \perp Z_t \mid S_t \implies I(A_0; A_k \mid S) = 0, \quad I(A_2; A_3 \mid S) = 0$$
The true directed net influence evaluates to:
$$\Gamma(0 \to 1) = I(A_0; A_1 \mid S) - \max_{k \in \{2, 3\}} I(A_k; A_1 \mid S) = 1.585 - 0.000 = \mathbf{1.585 \text{ bits}}$$
This represents a mathematically rigorous, analytically provable positive control.

### 3.3 Condition D (Coordination Strength Sweep)
In Condition D, pair $(0, 1)$ coordinates on private signal $Z_t$ with probability $\alpha \in \{0.25, 0.50, 0.75, 1.00\}$. With probability $1 - \alpha$, they fall back to independent state-conditioned play. As $\alpha \to 0$, $I(A_0; A_1 \mid S) \to 0$.

### 3.4 Condition B (Shared-Context Negative Control)
All 4 agents independently compute a best-response to the dominant action in $S_t$ with independent noise $\epsilon = 0.10$. No private signal exists. Conditionally given $S_t$, all agents are independent: $I(A_i; A_j \mid S) \equiv 0$. Unconditionally, common context $S_t$ drives strong marginal correlation: $I(A_i; A_j) > 0$.

### 3.5 Condition A (Independent Baseline)
All 4 agents sample uniformly at random from independent RNGs. Both marginal and conditional mutual information are zero.

---

## 4. Addressing Estimator Sparsity & Baseline Calibration

1. **State Sparsity:** We report visit distributions explicitly and test sample lengths $T \in \{100, 200, 500, 1000\}$ to characterize how finite-sample bias decays with $T$.
2. **Fair Calibration:** To evaluate classification FPR fairly, thresholds $\tau$ for both PNI and Marginal MI are calibrated on validation seeds (seeds 0–4) of Condition A to target a nominal 5% FPR, and evaluated on held-out test seeds (seeds 5–19).
3. **Threshold-Free Metrics:** We emphasize Top-1 identification accuracy and Mean Reciprocal Rank (MRR), which evaluate ranking quality across all 6 candidate pairs without dependence on arbitrary threshold selection.
4. **Uncertainty Intervals:** For Top-1 accuracy proportions, we report exact Wilson score intervals or Clopper-Pearson intervals to avoid degenerate $[1, 1]$ bootstrap artifacts.
