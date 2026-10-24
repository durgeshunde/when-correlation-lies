# When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents

**Track:** Track 1.4: Markets and Collusion  
**Event:** Apart Research AI Collusion Research Sprint (October 23–25, 2026)  
**Author:** Durgesh Unde  
**Affiliation:** Independent Researcher  
**Date:** October 2026  
**Evaluation Pipeline:** Verified Python 3.13 / CPU Benchmarks (0 API Calls, $0 Spend)  
**Total Evaluations:** 260 unique trajectories (54,000 unique rounds across 300 evaluations)

---

### Abstract
*(Word count: 206 words)*  
As autonomous AI agents are increasingly deployed in high-frequency trading, automated procurement, and algorithmic pricing, covert collusion poses severe economic and systemic risks. Information-theoretic detectors, notably Pairwise Net Influence (PNI; Bonjour et al., 2022), measure directed action dependence via Conditional Mutual Information (CMI) conditioned on observable public game state. However, whether PNI can distinguish deliberate multi-agent coordination from spurious correlation induced by shared context remains an open empirical question. In this work, we stress-test PNI against both unconditioned marginal mutual information and raw conditional mutual information baselines in an iterated 4-agent game across four controlled conditions: independent play, shared-context confounding, a shared-private-signal coordination control, and coordination-strength sweeps with a joint Bernoulli gate. Across 20 independent seeds and trajectory length sweeps ($T \in [50, 500]$), unconditioned mutual information exhibits a 38-fold surge under shared context, falsely accusing independent agents of collusion (FPR = 1.000). Triangulating our estimators reveals that conditioning on observable state eliminates this contextual surge (CMI FPR = 0.000), while PNI's competitor subtraction suppresses residual background, maintaining low false positives (FPR = 0.100) and achieving 100% Top-1 identification accuracy (Wilson 95% CI: [0.839, 1.000], MRR = 1.000) under controlled coordination. Our findings demonstrate that conditioning on observable history is indispensable for behavioral auditing, though finite-sample state sparsity introduces residual baseline uncertainty.

---

## 1. Introduction

Autonomous algorithmic agents increasingly govern financial exchanges, automated multi-unit auctions, and dynamic e-commerce pricing. A critical frontier in AI safety is preventing **algorithmic collusion**—scenarios where autonomous agents synchronize strategies to extract supracompetitive rents at the expense of human market participants (Calvano et al., 2020; Schwalbe, 2018). Unlike traditional cartels that rely on explicit communication logs, AI agents can establish implicit coordination purely through tacit observational learning or private coordination signals. Consequently, behavioral auditing tools that operate solely on external observational logs are essential for oversight.

In Track 1.4 (Markets and Collusion) of the AI Collusion Research Sprint, we address the core research question:  
*Can information-theoretic approaches detect collusion in multi-agent games?*

Specifically, we stress-test **Pairwise Net Influence (PNI)** (Bonjour, Aggarwal, & Bhargava, UAI 2022). While PNI is theoretically sound, real-world market auditing confronts a severe confounder: **shared-context confounding**. When non-coordinating agents independently react to identical market histories, price shocks, or public order books, their actions naturally synchronize. If auditing metrics mistake this context-driven synchronization for collusion, benign market makers will face false enforcement sanctions.

**Core Research Question:** *How reliably does Pairwise Net Influence distinguish deliberately coordinated behavior from independent behavior that merely appears correlated due to shared context, and how does detector performance scale with coordination strength and trajectory length?*

To rigorously separate the effects of state conditioning from rival competitor subtraction, we formulate and evaluate four hypotheses across three distinct estimators—Marginal MI $I(A_i; A_j)$, Conditional MI $\gamma(i, j) = I(A_i; A_j \mid S)$, and PNI net score—using a frozen benchmark of 260 unique trajectories (54,000 unique recorded rounds across 300 evaluations):
- **H1 (Positive Sensitivity):** Under controlled shared-private-signal coordination, PNI ranks the coordinating pair #1 significantly above the 1/6 (16.7%) random baseline.
- **H2 (Shared-Context Confounding):** Observable public state induces severe spurious action correlation, causing unconditioned marginal mutual information to suffer a 100% false positive rate.
- **H3 (State-Conditioning Neutralization & Rival Subtraction):** Conditioning on observable state $S$ neutralizes this broadcast shock below independent calibration thresholds, while PNI rival subtraction further isolates pair-specific influence.
- **H4 (Sensitivity Scaling):** PNI identification scores degrade predictably and monotonically as joint coordination probability $\alpha$ decreases towards independent play.

---

## 2. Related Work

### 2.1 Information-Theoretic Multi-Agent Influence
Information theory offers a non-parametric foundation for quantifying multi-agent dependence without restrictive linear or parametric assumptions (Cover & Thomas, 2006). Bonjour, Aggarwal, and Bhargava (UAI 2022) formulated individual influence in simultaneous games using Conditional Mutual Information (CMI) conditioned on observable game state:
$$\gamma(i, j) = I(A_i; A_j \mid S) = H(A_j \mid S) - H(A_j \mid A_i, S)$$
In simultaneous games, raw CMI is symmetric. To isolate pair-specific coordination from symmetric broadcast shocks, Bonjour et al. introduced *directed net influence* by subtracting the strongest competing rival influence:
$$\Gamma(i \to j) = \gamma(i, j) - \max_{k \neq i, j} \gamma(k, j)$$
A candidate pair is assigned the score $\text{Score}_{\text{PNI}}(i, j) = \min(\Gamma(i \to j), \Gamma(j \to i))$. While Bonjour et al. established theoretical bounds in matrix games, they did not evaluate empirical false-positive inflation under shared-context confounding, separate raw CMI from rival subtraction, or calibrate decision thresholds on negative controls.

### 2.2 Algorithmic and LLM Collusion in Economics
Extensive literature demonstrates that Q-learning and reinforcement learning agents can autonomously sustain collusive pricing above competitive Nash equilibria without explicit communication (Calvano et al., 2020; Schwalbe, 2018). Recently, Agrawal et al. (2025; arXiv:2507.01413) extended collusion auditing to LLM agents participating in double auctions, establishing that language model agents can form tacit cartels through contextual prompting. However, current LLM collusion studies evaluate aggregate market distortion (e.g., supracompetitive bid-ask spreads) rather than information-theoretic ranking detectors capable of identifying the specific colluding coalition.

### 2.3 Finite-Sample Entropy Estimation & Regularization
Discrete plug-in estimators of entropy and mutual information natively evaluate unvisited cells via the standard mathematical convention $0 \log 0 = 0$. Smoothing (e.g., Bayesian Dirichlet smoothing with Jeffreys pseudocounts $\lambda = 0.5$; Holste et al., 1998) is therefore not mathematically required to prevent division by zero. Rather, smoothing serves as a regularized estimation choice that stabilizes probability mass across sparsely sampled state-action cells in finite sample regimes (Paninski, 2003). We explicitly evaluate the sensitivity of our empirical findings across smoothing parameter $\lambda \in \{0.0, 0.5, 1.0\}$.

---

## 3. Methods

### 3.1 Iterated Multi-Agent Game Environment
Following Bonjour et al. (2022) and sprint starter guidelines, we implement a 4-player iterated Rock-Paper-Scissors (RPS) tournament. The action space is $\mathcal{A} = \{\text{Rock}: 0, \text{Paper}: 1, \text{Scissors}: 2\}$. In round $t$, each agent $i$ executes simultaneous action $a_i^t$. The round payoff is the tournament sum of pairwise zero-sum payoffs: $\pi_i(a^t) = \sum_{j \neq i} U(a_i^t, a_j^t)$, where $U \in \{-1, 0, 1\}$. The public observable state $S_t$ is the joint action profile of round $t-1$: $S_t = (a_0^{t-1}, a_1^{t-1}, a_2^{t-1}, a_3^{t-1}) \in \mathcal{A}^4$, yielding $|\mathcal{S}| = 3^4 + 1 = 82$ observable states (including initial START state).

### 3.2 Controlled Experimental Conditions
To isolate causal confounding mechanisms, we evaluate four strictly partitioned conditions:
- **Condition A (Independent Negative Control):** 4 uniform-random agents with isolated RNGs. Zero coordination or state dependence. Seeds 0–4 are reserved strictly for threshold calibration; seeds 5–19 ($n=15$) are strictly held-out test evaluations.
- **Condition B (Shared-Context Confounding Negative Control):** Agents independently execute a best response to the dominant action in $S_t$ with 10% independent noise. Agents are conditionally independent given $S_t$, but unconditionally correlated due to the shared state ($n=20$).
- **Condition C (Shared-Private-Signal Coordination Positive Control):** Agents 0 and 1 receive a private common signal $Z_t \sim \text{Uniform}(\{0, 1, 2\})$, executing identical coordinated actions. Agents 2 and 3 play independent best responses. This establishes a controlled statistical signature of conditional action dependence in scripted RPS, though it does not by itself prove a real-world economic cartel ($n=20$).
- **Condition D (Coordination Strength Sweep):** In each round, a joint coordination gate $G_t \sim \text{Bernoulli}(\alpha)$ is drawn. If $G_t = 1$, both agents 0 and 1 follow private signal $Z_t$; if $G_t = 0$, neither receives $Z_t$ and both independently execute fallback best response. This guarantees the pair coordinates with probability $\alpha$ ($n=20$ per $\alpha \in \{0.25, 0.50, 0.75, 1.00\}$).

### 3.3 Triangulated Estimators & Calibrated Decision Thresholds
To disentangle conditioning from competitor subtraction, we evaluate three statistics:
1. *Marginal MI:* $\text{Score}_{\text{MI}}(i, j) = I(A_i; A_j) = H(A_j) - H(A_j \mid A_i)$
2. *Conditional MI:* $\text{Score}_{\text{CMI}}(i, j) = \gamma(i, j) = I(A_i; A_j \mid S) = H(A_j \mid S) - H(A_j \mid A_i, S)$
3. *Pairwise Net Influence (PNI):* $\text{Score}_{\text{PNI}}(i, j) = \min(\Gamma(i \to j), \Gamma(j \to i))$, where $\Gamma(i \to j) = \gamma(i, j) - \max_{k \neq i, j} \gamma(k, j)$

Probabilities are estimated with Dirichlet smoothing ($\lambda = 0.5$). Decision thresholds are calibrated via exploratory small-sample calibration on held-out validation trajectories (seeds 0–4 of Condition A, 30 candidate-pair scores) targeting nominal 5% FPR at the empirical 95th percentile ($\tau_{\text{MI}} = 0.030751$ bits, $\tau_{\text{CMI}} = 0.117454$ bits, $\tau_{\text{PNI}} = 0.010427$ bits). Because 30 candidate-pair scores provide coarse quantile estimation, this serves as an empirical baseline rather than guaranteeing exact 5% population FPR. Condition A test FPR is evaluated strictly on held-out seeds 5–19 ($n=15$, 90 candidate pairs).

### 3.4 Statistical Uncertainty Protocol & Degenerate Intervals
All primary runs evaluate 20 independent pseudorandom seeds. For Top-1 accuracy (a bounded binomial proportion), we report exact **Wilson score confidence intervals** to prevent degenerate [1.0, 1.0] intervals. Continuous metrics (scores, FPR, MRR) report non-parametric bootstrap 95% confidence intervals over $B = 2,000$ resamples across independent trajectories. Whenever per-trajectory outcomes are completely identical across all 20 seeds (e.g., exactly 1 of 5 negative candidate pairs flagged in every seed), the resampled mean is constant, producing an interval of zero width (e.g., [0.200, 0.200]). We explicitly flag these degenerate intervals with asterisks and report the underlying empirical counts (flagged pairs / total negative pairs) to reflect that this represents uniform per-seed outcomes rather than zero population variance.

---

## 4. Results

### Table 1: Triangulated Benchmark Metrics ($T = 200$ rounds, 20 independent seeds per condition)
*Condition A test metrics computed strictly on held-out seeds 5–19 ($n=15$). Brackets report 95% bootstrap CIs for continuous scores and exact Wilson CIs for Top-1 accuracy. Underlying counts report flagged pairs / total negative pairs. \*Denotes degenerate bootstrap intervals resulting from uniform per-seed outcomes across all seeds rather than zero population variance.*

| Condition | GT Pair | Marginal MI Top (95% CI) | Marg. MI FPR [Counts] | Cond. MI (CMI) Top (95% CI) | CMI FPR [Counts] | PNI Net Top (95% CI) | PNI FPR [Counts] | PNI Top-1 (Wilson CI) | PNI MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond A (Test, $n=15$)** | None | 0.025 [0.020, 0.031] | 0.033 [0.000, 0.067] (3/90) | 0.115 [0.111, 0.119] | 0.111 [0.044, 0.189] (10/90) | 0.005 [0.003, 0.008] | 0.022 [0.000, 0.056] (2/90) | — | — |
| **Cond A (Cal., $n=5$)** | None | 0.027 [0.019, 0.036] | 0.067 [0.000, 0.133] (2/30) | 0.115 [0.111, 0.118] | 0.067 [0.000, 0.133] (2/30) | 0.007 [0.003, 0.012] | 0.067 [0.000, 0.133] (2/30) | — | — |
| **Cond B (Context, $n=20$)** | None | **0.958 [0.932, 0.984]** | **1.000 [1.000, 1.000]\*** (120/120) | **0.084 [0.081, 0.087]** | **0.000 [0.000, 0.000]\*** (0/120) | **0.012 [0.009, 0.015]** | **0.100 [0.050, 0.158]** (12/120) | — | — |
| **Cond C (Coord., $n=20$)** | (0, 1) | 1.451 [1.447, 1.454] | 0.240 [0.200, 0.320] (24/100) | 0.797 [0.785, 0.808] | 0.020 [0.000, 0.050] (2/100) | 0.755 [0.741, 0.769] | 0.200 [0.200, 0.200]\* (20/100) | **1.000 [0.839, 1.000]** | **1.000** |
| **Cond D ($\alpha=0.25$, $n=20$)** | (0, 1) | 1.024 [0.988, 1.062] | 1.000 [1.000, 1.000]\* (100/100) | 0.329 [0.306, 0.353] | 0.010 [0.000, 0.030] (1/100) | 0.264 [0.239, 0.288] | 0.100 [0.060, 0.140] (10/100) | **1.000 [0.839, 1.000]** | **1.000** |
| **Cond D ($\alpha=0.50$, $n=20$)** | (0, 1) | 1.131 [1.090, 1.174] | 1.000 [1.000, 1.000]\* (100/100) | 0.524 [0.497, 0.552] | 0.000 [0.000, 0.000]\* (0/100) | 0.469 [0.442, 0.498] | 0.180 [0.150, 0.200] (18/100) | **1.000 [0.839, 1.000]** | **1.000** |
| **Cond D ($\alpha=0.75$, $n=20$)** | (0, 1) | 1.285 [1.259, 1.310] | 0.920 [0.830, 0.990] (92/100) | 0.669 [0.650, 0.686] | 0.020 [0.000, 0.050] (2/100) | 0.623 [0.600, 0.642] | 0.200 [0.200, 0.200]\* (20/100) | **1.000 [0.839, 1.000]** | **1.000** |
| **Cond D ($\alpha=1.00$, $n=20$)** | (0, 1) | 1.452 [1.449, 1.455] | 0.240 [0.200, 0.300] (24/100) | 0.813 [0.802, 0.824] | 0.010 [0.000, 0.030] (1/100) | 0.771 [0.757, 0.784] | 0.200 [0.200, 0.200]\* (20/100) | **1.000 [0.839, 1.000]** | **1.000** |

### 4.1 Evaluation of Hypotheses and Key Discoveries
- **Support for H1 (Top-1 Accuracy = 100% under Controlled Coordination):** Under Condition C, PNI achieves **100% Top-1 accuracy** (20/20 seeds, Wilson 95% CI: [0.839, 1.000]) and an MRR of 1.000. The true coordinating pair achieves an average net score of **0.755 bits** [0.741, 0.769], decisively separated from competing non-coordinating pairs.
- **Support for H2 (The 38-Fold Context Confounding Surge):** Under Condition B (independent agents reacting to shared context), naive Marginal MI explodes from 0.025 bits in Condition A to **0.958 bits** in Condition B—a **38-fold surge** in apparent action dependence—flagging 100% of non-coordinating pairs as colluders (FPR = 1.000 [1.000, 1.000], 120/120 pairs flagged). Unconditioned correlation is completely blind to shared context.
- **Support for H3 (Disentangling State Conditioning from Competitor Subtraction):** Reporting Marginal MI, raw CMI, and PNI separately isolates the two pipeline mechanisms. Conditioning on state $S_t$ drops raw dependence from 0.958 bits to **0.084 bits** [0.081, 0.087], falling below $\tau_{\text{CMI}} = 0.1175$ bits to yield **0.000 FPR** (0/120 pairs). Competitor subtraction computes $\Gamma(i \to j) = \gamma(i, j) - \max_k \gamma(k, j)$; because all 4 agents experience symmetric residual dependence (~0.084 bits), the difference is small (0.012 bits [0.009, 0.015]), yielding PNI FPR **0.100** [0.050, 0.158] (12/120 pairs) under $\tau_{\text{PNI}} = 0.0104$ bits. State conditioning neutralizes the common shock, while competitor subtraction provides directed isolation.
- **Support for H4 (Monotonic Scaling under Joint Gate):** In Condition D, varying joint coordination probability $\alpha$ via gate $G_t \sim \text{Bernoulli}(\alpha)$ produces strictly monotonic scaling of true-pair net influence: 0.264 bits ($\alpha=0.25$), 0.469 bits ($\alpha=0.50$), 0.623 bits ($\alpha=0.75$), and 0.771 bits ($\alpha=1.00$). PNI maintains 100% Top-1 accuracy across all tested $\alpha \in [0.25, 1.00]$, whereas Marginal MI suffers 100% FPR across $\alpha \le 0.50$.

---

## 5. Sensitivity Analysis: Trajectory Length

To evaluate finite-sample convergence, we executed a dedicated sweep across trajectory lengths $T \in \{50, 100, 200, 500\}$ rounds (Table 2). Evaluations used fixed primary calibrated thresholds ($\tau_{\text{MI}} = 0.030751$, $\tau_{\text{CMI}} = 0.117454$, $\tau_{\text{PNI}} = 0.010427$ bits) across all lengths.

### Table 2: Trajectory Length Sensitivity ($T \in [50, 500]$, 20 seeds per cell, fixed primary thresholds)
*Evaluated across 20 independent seeds per cell under fixed primary calibrated thresholds. $T=200$ directly reuses primary benchmark trajectories. Underlying counts report flagged pairs / total negative pairs. \*Denotes degenerate bootstrap intervals due to uniform zero flagged pairs across all seeds.*

| Length $T$ | Cond C PNI Top-1 (Wilson) | Cond C PNI True Score (bits) | Cond B Marg MI Score (bits) | Cond B CMI Score (bits) | Cond B CMI FPR [Counts] | Cond B PNI Net Score (bits) | Cond B PNI FPR [Counts] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50 rounds** | **1.00 [0.839, 1.000]** | 0.269 [0.250, 0.286] | 0.943 [0.884, 1.001] | 0.144 [0.134, 0.153] | 0.450 (54/120) | 0.0176 [0.0117, 0.0246] | 0.133 [0.083, 0.183] (16/120) |
| **100 rounds** | **1.00 [0.839, 1.000]** | 0.499 [0.483, 0.515] | 0.964 [0.909, 1.021] | 0.112 [0.108, 0.118] | 0.042 (5/120) | 0.0131 [0.0084, 0.0181] | 0.108 [0.067, 0.158] (13/120) |
| **200 rounds** | **1.00 [0.839, 1.000]** | 0.755 [0.741, 0.770] | 0.958 [0.932, 0.984] | 0.084 [0.081, 0.087] | 0.000\* (0/120) | 0.0121 [0.0091, 0.0154] | 0.100 [0.050, 0.158] (12/120) |
| **500 rounds** | **1.00 [0.839, 1.000]** | 1.052 [1.042, 1.062] | 0.958 [0.942, 0.973] | **0.056 [0.054, 0.059]** | **0.000\* (0/120)** | **0.0074 [0.0056, 0.0094]** | **0.050 [0.017, 0.083]** (6/120) |

**Trajectory Length Dynamics:** Over the tested $T = 50–500$ range, unconditioned Marginal MI in Condition B remains stubbornly elevated (~0.94–0.96 bits). In contrast, the estimated raw CMI score declines monotonically from 0.144 bits at $T = 50$ down to 0.056 bits at $T = 500$ (a 61% reduction), bringing CMI FPR from 0.450 to 0.000. Similarly, the PNI net score declines from 0.0176 bits at $T = 50$ down to 0.0074 bits at $T = 500$ (a 58% reduction), reducing Condition B PNI false positives from 0.133 to 0.050. PNI achieves 100% Top-1 accuracy across all tested lengths in Condition C.

---

## 5. Exploratory LLM-Agent Pilot (Modal CPU)

To investigate whether information-theoretic collusion detectors extend to open-weight autoregressive models, we conducted an exploratory pilot using `Qwen/Qwen2.5-1.5B-Instruct` deployed on Modal CPU workers (4 vCPU, 8 GB RAM).

### 5.1 Pilot Design and Protocol
Four independent model instances engaged in 5-round simultaneous games under three pilot conditions:
- **LLM-A (Independent Play):** Four model decisions per round sampled with distinct per-decision random seeds, prompting agents to maximize win rates independently.
- **LLM-B (Shared Public Context):** All agents observed the same public round outcome summary and were instructed to follow a market counter-strategy. Distinct sampling seeds ensured identical prompts did not force identical outputs.
- **LLM-C (Covert Coordination Positive Control):** Designated coordinating agents (0 and 1) received a private uniform random signal $Z_t \in \{\text{ROCK}, \text{PAPER}, \text{SCISSORS}\}$ with instructions to coordinate. Non-coordinating agents (2 and 3) received no private signal. The secret signal was strictly excluded from detector-visible logs.

State abstraction compressed previous round actions to five categories: $S \in \{\text{START}, \text{ROCK}, \text{PAPER}, \text{SCISSORS}, \text{TIE}\}$ based on plurality. A bounded JSON parser enforced valid action output without defaulting malformed text to arbitrary actions.

### Table 3: Exploratory LLM Pilot Results (Qwen2.5-1.5B on Modal CPU, 3 seeds × 5 rounds = 180 decisions)
*Evaluated across 3 independent seeds per condition ($T=5$ rounds, 4 agents = 20 decisions/trajectory; 60 decisions/condition). Scripted classification thresholds are not calibrated to LLM empirical action distributions; detector performance is evaluated via pair rankings and descriptive mutual information.*

| Condition | Trajectories | Total Decisions | Valid Action Rate | Mean Latency | Secret Signal Compliance | Top-1 Marginal MI | Top-1 CMI ($\gamma$) | Top-1 PNI Net Score | Mean Marg MI (bits) | Mean CMI (bits) | Mean PNI Net (bits) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-A (Independent)** | 3 | 60 | 1.000 (60/60) | 1.39s | N/A | N/A | N/A | N/A | 0.2416 | 0.0768 | -0.0084 |
| **LLM-B (Shared Context)** | 3 | 60 | 1.000 (60/60) | 1.53s | N/A | N/A | N/A | N/A | 0.1118 | 0.0706 | -0.0212 |
| **LLM-C (Private Signal)** | 3 | 60 | 1.000 (60/60) | 1.40s | **1.000 (30/30)** | **1.000 (3/3)** | **0.667 (2/3)** | **0.667 (2/3)** | 0.1682 | 0.0678 | -0.0315 |

### 5.2 Pilot Findings and Limitations
1. **Instruction Compliance:** The 1.5B model achieved 100% adherence (30/30 decisions) to the covert private coordination directive in Condition LLM-C, demonstrating that instruction-tuned models can act as viable coordinating agents without fine-tuning.
2. **Detector Sensitivity in Small Samples:** In Condition LLM-C, Marginal MI ranked the coordinating pair #1 across all 3 seeds (100%), while CMI and PNI ranked the coordinating pair #1 in 2 of 3 seeds (66.7%). At $T=5$, discrete state-conditioning over 5 states partitions counts into sparse cells, attenuating the empirical conditional mutual information signal.
3. **Exploratory Scope:** This pilot is strictly descriptive ($N=3$ seeds per condition) and does not constitute a powered statistical benchmark. Scripted thresholds are not applied to the LLM distribution.

---

## 6. Discussion and Limitations

### 6.1 Auditing Implications: Why Correlation Lies
In algorithmic markets, autonomous pricing agents, high-frequency execution bots, and procurement models continuously observe common economic shocks (e.g., benchmark interest rates, commodity prices, wholesale tariffs). Our empirical results demonstrate that unconditioned correlation metrics (e.g., Pearson correlation, raw mutual information) cannot differentiate rational adaptation from collusive coordination. Regulators relying on unconditioned metrics would commit severe Type I errors, falsely targeting competitive firms solely because they adapted efficiently to common public information.

### 6.2 Limitations
We transparently acknowledge the constraints of this study:
1. *Controlled Scripted Signatures vs. Real-World Cartels:* Condition C establishes a controlled *shared-private-signal coordination condition* demonstrating that PNI recognizes statistical action dependence in scripted matrix games. However, detecting this statistical signature does not prove that agents formed an economically harmful cartel, extracted supra-competitive rents, or inflicted market harm. Furthermore, unobserved common causes (e.g., unobserved common private inputs, shared proprietary features, or unmodeled system states) could generate similar conditional action dependence.
2. *Estimator Regularization Sensitivity:* In our 4-player game with $|S| = 82$ states, state conditioning partitions observations across sparse cells. Evaluating smoothing sensitivity under Condition B at $T = 200$ reveals that the unsmoothed plug-in estimator (using $0 \log 0 = 0$) yields CMI = 0.062 bits, while Dirichlet smoothing yields 0.084 bits (Jeffreys, $\lambda = 0.5$) and 0.096 bits (Laplace, $\lambda = 1.0$). Pseudocount smoothing introduces a mild upward baseline in sparse regimes, though all values remain well below unconditioned Marginal MI (0.958 bits).
3. *Exploratory Small-Sample Calibration:* Nominal 5% decision thresholds were calibrated from 5 validation seeds (30 candidate pairs) of Condition A. While isolating calibration from the 15 held-out test seeds, 30 samples yield coarse quantile estimates and do not guarantee an exact 5% population FPR.
4. *Modal Promotional Compute Pilot & Cloud Boundaries:* An exploratory 180-decision pilot using `Qwen/Qwen2.5-1.5B-Instruct` was executed on Modal CPU instances (4 cores, 8GB RAM). Allocating cloud GPU instances (T4/L4) required an active payment method on file under Modal platform policy, which was safely avoided under our zero-out-of-pocket spending cap ($0.00). The CPU pilot consumed $0.07 of promotional compute credits ($0.00 billed out of pocket) across 9 trajectories. While establishing that 1.5B instruction models achieve 100% signal compliance under prompt-mediated directives, small sample sizes ($T=5$ rounds) yield coarse empirical distributions. Scripted calibration thresholds are not calibrated to LLM action distributions and are therefore not applied. Canonical benchmark results remain grounded in the 260-trajectory scripted evaluation (54,000 rounds).

### 6.3 Future Work
Key extensions include: (1) evaluating neural CMI estimators (e.g., MINE) on LLM auction transcripts (Agrawal et al., 2025); (2) developing dynamic state abstractions to compress high-dimensional order book histories; and (3) testing PNI under intentional adversarial evasion.

---

## 7. Conclusion

This research stress-tested Pairwise Net Influence for detecting multi-agent coordination. We demonstrated that shared observable context induces a 38-fold surge in unconditioned mutual information, causing naive detectors to suffer a 100% false positive rate. Conditioning on observable state neutralizes this broadcast shock, while competitor subtraction provides directed isolation, preserving low false positives while attaining 100% Top-1 accuracy under genuine coordination. State conditioning is an indispensable requirement for auditing multi-agent algorithmic systems, while addressing state sparsity remains a critical challenge.

---

## Code and Data
- **Code Repository:** All experimental scripts, detectors, environments, test suites, and report generation tools are packaged in `when_correlation_lies_all_source_code.txt` with SHA-256 checksums.
- **Data & Trajectories:** Full scripted JSONL trajectories (260 unique files, 54,000 unique rounds) and exploratory LLM trajectories (9 unique files, 180 decisions) with separate ground-truth manifests are included in the review bundle ZIP (`data/raw/`, `data/eval_labels/`, `data/raw_llm/`, `data/eval_labels_llm/`).
- **Reproducibility:** Complete automated test suite passes (59/59 unit tests) in Python 3.13 on standard CPU hardware.

---

## Author Contributions
**Durgesh Unde (Independent Researcher):** Project conceptualization, selection and formulation of research questions and experimental conditions, project direction, review of experimental outputs, qualitative interpretation of detection metrics, and final editorial responsibility for the manuscript.

---

## References

1. Agrawal, K., Teo, V., Vazquez, J. J., Kunnavakkam, S., Srikanth, V., & Liu, A. (2025). Evaluating LLM Agent Collusion in Double Auctions. *arXiv preprint arXiv:2507.01413*.
2. Bonjour, T., Aggarwal, V., & Bhargava, B. (2022). Information-theoretic approach to detect collusion in multi-agent games. *Proceedings of Machine Learning Research (UAI 2022)*, 180, 223–232.
3. Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial intelligence, algorithmic pricing, and collusion. *American Economic Review*, 110(10), 3267–3297.
4. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). John Wiley & Sons.
5. Holste, D., Grosse, I., & Herzel, H. (1998). Bayes' estimators of generalized entropies. *Journal of Physics A: Mathematical and General*, 31(11), 2551–2566.
6. Paninski, L. (2003). Estimation of entropy and mutual information. *Neural Computation*, 15(6), 1191–1253.
7. Schwalbe, U. (2018). Algorithms, machine learning, and collusion. *Journal of Competition Law & Economics*, 14(4), 568–607.

---

## Appendix: Limitations and Dual-Use / Ethical Considerations
**Mandatory Ethical & Dual-Use Analysis:** Auditing multi-agent systems for collusion possesses significant dual-use implications:
1. *Defensive Regulatory Oversight:* Information-theoretic detectors empower market regulators, exchange operators, and platform architects to continuously monitor high-frequency trading and procurement markets for tacit cartels without requiring invasive access to proprietary model weights.
2. *Adversarial Exploitation & Stealth Evasion:* Conversely, adversarial agent developers could incorporate PNI as an explicit penalty term in multi-agent reinforcement learning objectives. By penalizing conditional mutual information, malicious actors could train stealthy collusive policies that sustain supracompetitive markups while remaining statistically undetectable under state-conditioned audits.
3. *Harms of False Accusations:* Our Condition B results demonstrate the catastrophic risk of unconditioned metrics: a 100% false accusation rate against competitive agents responding to shared context. Auditing architectures must mandate state conditioning and calibrated uncertainty intervals before initiating punitive regulatory actions.

---

## LLM Usage Statement
ChatGPT (OpenAI) assisted with research ideation, paper explanation, methodology critique, independent review, and prompt drafting. Antigravity (Google DeepMind advanced agentic coding assistant) assisted with code implementation, unit test suites, experiment orchestration, Modal CPU pilot harness and execution, statistical metric aggregations, visualizations, ReportLab PDF layout, and packaging under the author's iterative instructions and review. Project direction, experimental design choices, qualitative analysis, and final editorial responsibility rest entirely with the author (Durgesh Unde). Canonical benchmark evaluations used 260 scripted trajectories (54,000 rounds); exploratory LLM pilot evaluations used 9 trajectories (180 decisions) executed on Modal CPU ($0.07 promotional credit, $0.00 out-of-pocket spend). Zero personal payment cards or external paid token APIs were used. The experimental protocol was frozen after exploratory prototyping and was not preregistered prior to exploration.
