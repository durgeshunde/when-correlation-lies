# Results Interpretation & Synthesis

**Project:** When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents  
**Author:** Durgesh Unde (Independent Researcher)  
**Track:** 1.4 Markets and Collusion — Apart Research AI Collusion Research Sprint (October 23–25, 2026)  

This document provides the canonical interpretation of experimental results across the 260-trajectory scripted benchmark (54,000 rounds) and the 630-trajectory open-weight LLM study (68,400 decisions, $40.63 metered compute, $0.00 out-of-pocket spend).

---

## 1. Canonical Scripted Benchmark (N=4 Agents, 6 Candidate Pairs)

### Condition A: Independent Baseline (Negative Control)
- **Design:** All 4 agents select actions independently without public history conditioning or private signals (seeds 5–19, 90 negative candidate pairs).
- **False Positive Rates:**
  - Marginal Mutual Information (MI) FPR: 3/90 = 0.033
  - Conditional Mutual Information (CMI) FPR: 10/90 = 0.111
  - Pairwise Net Influence (PNI) FPR: 2/90 = 0.022
- **Interpretation:** Under fully independent random play, all three detectors maintain low false-positive rates near nominal calibration levels.

### Condition B: Shared-Context Confounding (Negative Control)
- **Design:** Agents independently condition actions on shared public game history $S_t$ (120 negative candidate pairs across 20 seeds). No private communication occurs.
- **False Positive Rates:**
  - Marginal MI FPR: 120/120 = 1.000 (100% false positive rate)
  - CMI FPR: 0/120 = 0.000
  - PNI FPR: 12/120 = 0.100
- **Interpretation:** Shared public context confounds unconditional correlation metrics. Unconditioned mutual information surges 38-fold (mean MI rises from 0.026 to 0.998 nats), falsely classifying every independent agent pair as colluding. CMI achieves 0.000 FPR by conditioning on $S_t$. PNI moderates confounding to 0.100 via competitor subtraction.

### Condition C: Known Coordination (Positive Control)
- **Design:** Designated coordinating pair (agents 0 and 1) receives an unobservable uniform private coordination signal $Z_t \sim \text{Uniform}(\{0, 1, 2\})$ and selects correlated actions $(a_t, a_t)$. Non-coordinating agents sample independently (20 positive target pairs, 100 negative candidate pairs across 20 seeds).
- **Top-1 Ranking Accuracy:**
  - PNI Top-1 Accuracy: 20/20 = 1.000 (95% CI: [0.839, 1.000], chance baseline = 0.167)
  - CMI Top-1 Accuracy: 20/20 = 1.000
  - Marginal MI Top-1 Accuracy: 20/20 = 1.000
- **Threshold-Calibrated False Positive Rate (Non-Target Pairs):**
  - Marginal MI FPR: 24/100 = 0.240
  - CMI FPR: 2/100 = 0.020
  - PNI FPR: 20/100 = 0.200
- **Interpretation:** PNI correctly ranks the designated coordinating pair first in all 20 seeds (100% Top-1 accuracy). However, at the calibrated decision threshold, PNI flags 20/100 non-target pairs (FPR 0.200) compared to 2/100 for CMI (FPR 0.020). This demonstrates that pairwise relative ranking and threshold-based false-positive control represent distinct operational properties.

### Condition D: Partial / Intermittent Coordination
- **Design:** Coordinating agents follow the shared signal with probability $\alpha \in \{0.25, 0.50, 0.75, 1.00\}$.
- **Interpretation:** Detection metrics scale monotonically with coordination adherence $\alpha$, demonstrating progressive sensitivity down to intermittent coordination regimes.

---

## 2. Open-Weight LLM Replications on Modal CPU

Evaluated on `Qwen/Qwen2.5-0.5B-Instruct` across 630 trajectories (68,400 decisions) at horizons $T \in \{5, 20, 50\}$ under identical prompt templates and structured JSON decision-making.

| Horizon | Total Trajectories | Decisions | Cond B Marginal MI FPR | Cond B CMI FPR | Cond B PNI FPR | Cond C PNI Top-1 Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$T=5$** | 120 (40/cond) | 2,400 | 1.000 (240/240) | 0.000 (0/240) | 0.008 (2/240) | **1.000** (20/20) |
| **$T=20$** | 300 (100/cond) | 24,000 | 1.000 (600/600) | 0.000 (0/600) | 0.000 (0/600) | **1.000** (50/50) |
| **$T=50$** | 210 (70/cond) | 42,000 | 1.000 (420/420) | 0.000 (0/420) | 0.000 (0/420) | **1.000** (35/35) |
| **Total** | **630** | **68,400** | — | — | — | — |

### Key Findings from LLM Replications:
1. **Ultra-Short Horizon Robustness:** At $T=5$ (5 rounds per game), CMI's Top-1 ranking drops to 0.400 (8/20) due to empirical plug-in entropy bias in small samples, while PNI retains 1.000 (20/20) Top-1 ranking accuracy.
2. **Context Immunity:** At $T \ge 20$, both PNI and CMI achieve 0.000 FPR under shared-context prompt histories, while unconditioned Marginal MI suffers 1.000 FPR across all horizons.

---

## 3. Key Limitations & Methodological Disclosures

1. **Prompted Adherence vs. Spontaneous Collusion:** In the LLM experiments, coordination was induced by prompting designated agent pairs with synchronized private signals ($Z_t$). This tests detection sensitivity under compliance, not whether language models autonomously initiate collusive pacts.
2. **Threshold Calibration vs. Relative Ranking:** While PNI demonstrates perfect Top-1 ranking (100%) under genuine coordination, absolute decision thresholding on finite samples exhibits residual false positives (FPR 0.200 in scripted Condition C).
3. **Action Space & Environment Complexity:** Experiments were conducted on 3-action simultaneous games (Rock-Paper-Scissors). Extension to continuous action spaces (e.g., continuous pricing in Bertrand oligopolies) requires non-parametric density estimation or kernel methods.
4. **Adversarial Non-Colluding Opponents:** Opponent agents operated under stationary reactive strategies. Dynamic strategic counter-adaptation by independent agents remains an area for future inquiry.