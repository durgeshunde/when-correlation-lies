# Pre-Sprint Work Log & Project Provenance

## 1. Purpose and Integrity Declaration
This document records the exact provenance, development phases, and separation between prior literature, exploratory prototyping, and sprint contributions for the project **When Correlation Lies**.
In strict adherence to Apart Research sprint rules, this document transparently distinguishes pre-existing literature from original sprint contributions.

---

## 2. Project Timeline & Development Evolution

### Phase 1: Literature Inspection & Initial Prototyping (October 23, 2026, Early Sprint)
- Evaluated the official sprint tracks and selected Track 1.4 (Markets and Collusion).
- Read the primary detection paper: Bonjour, Aggarwal, & Bhargava (UAI 2022).
- Scaffolded the initial repository structure and implemented a draft 4-player iterated RPS environment.
- **Initial Prototyping Limitations (Discovered during Audit):**
  - The initial prototype used a deterministic hash-based coordination policy ($A_0 = A_1 = \text{hash}(S) \pmod 3$) for Condition C.
  - Initial detector used an arbitrary decision threshold $\tau = 0.0$.
  - Initial Condition D used independent Bernoulli draws per agent (resulting in $\alpha^2$ joint coordination).
  - Initial Condition A reporting evaluated on all 20 seeds including the calibration seeds.
  - These limitations were documented in `docs/scientific_audit.md`.

### Phase 2: Scientific Audit & Pipeline Correction (October 23, 2026, Sprint Core)
- Conducted an exhaustive mathematical and empirical audit (`docs/scientific_audit.md`).
- **Mathematical Repair of Positive Control:** Replaced the flawed hash function with a private, unobservable coordination device $Z_t \sim \text{Uniform}(\{0, 1, 2\})$, restoring true population CMI $I(A_0; A_1 \mid S) = \log_2(3) \approx 1.585$ bits.
- **Revised Alpha Semantics:** Redesigned Condition D to draw one shared joint coordination gate $G_t \sim \text{Bernoulli}(\alpha)$ per round, ensuring the pair coordinates jointly with exact probability $\alpha$.
- **Held-Out Validation Design:** Isolated Condition A seeds 0–4 strictly for runtime threshold calibration (targeting 5% FPR on independent play), reporting test performance strictly on held-out seeds 5–19 ($n=15$).
- **Fixed-Threshold Sensitivity:** Modified `scripts/run_sensitivity.py` to evaluate $T \in [50, 100, 200, 500]$ using fixed calibrated thresholds from the primary $T=200$ benchmark, avoiding fallback to zero and reusing primary trajectories to yield exactly 260 unique trajectories (54,000 unique rounds).
- **Statistical Rigor:** Implemented Wilson score continuity intervals for Top-1 accuracy to eliminate degenerate $[1.0, 1.0]$ intervals.
- **Test Suite Expansion:** Expanded pytest suite to 57 automated tests verifying analytical CMI, joint coordination gates, detector symmetry, environment state transitions, metric calculations, estimator regularization sensitivity, and LLM pilot state abstraction.
- **Academic Reporting:** Restructured research manuscript to strictly adhere to the official Apart Research template into an exact 5-page PDF report.

---

## 3. Pre-Existing Work (from Literature)

### Primary Detection Framework (Not our invention):
- **Citation:** Bonjour, T., Aggarwal, V., & Bhargava, B. (2022). "Information-theoretic approach to detect collusion in multi-agent games." *Proceedings of Machine Learning Research (UAI 2022)*, 180, 223–232.
- **Role:** Defined the Pairwise Net Influence (PNI) formulation, the iterated RPS matrix tournament setting, and the concept of rival influence subtraction.
- **Our Implementation:** Independent clean-room Python implementation with Laplace smoothing ($\lambda = 0.5$).

### Related LLM Collusion Literature:
- **Citation:** Agrawal, K., Teo, V., Vazquez, J. J., Kunnavakkam, S., Srikanth, V., & Liu, A. (2025). "Evaluating LLM Agent Collusion in Double Auctions." *arXiv:2507.01413*.
- **Role:** Conceptual reference for LLM pricing coordination in auction settings.
- **Our Distinction:** Our study focuses on information-theoretic detection in multi-agent games with zero LLM/API calls, providing foundational baseline benchmarking.

---

## 4. Original Sprint Contributions

1. **Discovery and Formulation of Shared-Context Confounding:** Formally designed Condition B to prove that independent agents reacting to common public history experience a 38-fold surge in unconditioned mutual information, inducing a 100% false positive rate in naive detectors.
2. **Triangulated Estimator Disambiguation:** Demonstrated that conditioning on observable state $S$ eliminates the broadcast shock below calibrated thresholds (CMI FPR = 0.000), while rival subtraction provides directed pair isolation (PNI FPR = 0.100), retaining 100% Top-1 accuracy on true coordinating pairs.
3. **Rigorous Finite-Sample Sensitivity Sweeps:** Systematically characterized how sample size $T \in [50, 500]$ and coordination strength $\alpha \in [0.25, 1.00]$ govern detector sensitivity and finite-sample estimator bias under discrete state spaces.
4. **Reproducible Open Research Package:** Full suite of 57 unit tests, 260 unique trajectory logs with separate ground-truth manifests, automated analysis pipelines, and formal 5-page PDF report.
