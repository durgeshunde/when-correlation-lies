# Research Design

## Working Title
"When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting 
Coordination Among LLM Agents"

## Research Question
How reliably does pairwise net influence distinguish deliberately coordinated 
behavior from independent behavior that merely appears correlated, and what 
happens when this detector is applied to LLM-agent trajectories?

## Track
Track 1.4 — Markets and Collusion  
Official question: Can information-theoretic approaches detect collusion in 
multi-agent games?

---

## Hypotheses

**H1.** Under a valid controlled coordination condition (Condition C), pairwise 
net influence can rank the designated pair above alternative pairs more often 
than the 1/6 chance baseline.

**H2.** Shared observable context can produce substantial unconditional action 
dependence even when agents act independently conditional on that context. 
(Measurable as: marginal MI FPR on Condition B > PNI FPR on Condition B.)

**H3.** Conditioning the information-theoretic detector appropriately on shared 
state reduces false positives caused by common context. 
(Measurable as: PNI FPR on Condition B ≈ PNI FPR on Condition A, and lower 
than marginal MI FPR on Condition B.)

**H4.** Performance measured with scripted agents may not transfer perfectly 
to stochastic LLM agents. (Testable in Phase 4; documented as a hypothesis here.)

All hypotheses are stated before experimental execution. Results will be 
reported as supported, not supported, or inconclusive based on measured evidence.

---

## Game Design

### Game: Iterated Rock-Paper-Scissors (4 agents)

**Action space:** {Rock=0, Paper=1, Scissors=2}  
**Payoff:** Zero-sum pairwise; agent i's payoff = sum of RPS outcomes vs all others.  
**State:** The full joint action profile of the previous round (fully observable).  
**Round 0 state:** START sentinel.  
**Rounds per trajectory:** 200 (primary), 50 and 500 in sensitivity analysis.

### Why 4 agents?
- 4 agents → C(4,2) = 6 candidate pairs → nontrivial ranking problem.
- Paper recommends 4-agent RPS as a starter experiment.
- Net influence requires at least 1 competitor k ≠ i, j.

---

## Conditions

### Condition A: Independent Baseline
- All 4 agents: uniform random, separate RNG per agent seeded independently.
- No shared state, no communication.
- Ground truth: no coordinating pair.
- Expected: low FPR, scores near zero.

### Condition B: Shared-Context Negative Control
- All 4 agents: SharedContextAgent — independently sample from a state-conditioned 
  best-response policy with noise ε = 0.1.
- Ground truth: no coordinating pair.
- Expected: naive MI inflated (common context), CMI-based PNI lower FPR.
- This tests H2 and H3.

### Condition C: Known-Coordination Positive Control
- Agents 0 and 1: CoordinatedAgent (α=1.0) — both play the same deterministic 
  function of the shared public state.
- Agents 2 and 3: SharedContextAgent (independent given state).
- Ground truth: coordinating pair = (0, 1).
- Expected: PNI top-1 accuracy above chance for agents 0 and 1.
- This tests H1.

### Condition D: Coordination Strength Sweep
- Same as C but coordination probability α ∈ {0.25, 0.5, 0.75, 1.0}.
- Ground truth: coordinating pair = (0, 1).
- Expected: detection accuracy increases with α.
- Tests sensitivity and transition from noise to signal.

---

## Coordination Mechanism (Condition C / D)

The coordinating pair both compute:
```
coord_action = hash(state_feature) % 3
```

where `state_feature` is the publicly observable tuple of previous actions.

This produces:
- **Perfect agreement between agents 0 and 1** for α=1.0.
- **Non-trivial CMI signal:** Even though the state "causes" the action, both 
  agents are perfectly correlated with each other GIVEN the state — because they 
  choose the SAME action rather than independently sampling.

Crucially, naive detectors may confuse this with condition B if they ignore state.
The CMI detector should distinguish them.

**NOTE:** This is a controlled experimental coordination, not evidence of 
naturally emerging collusion. We describe it as coordination detection, not 
proven harmful collusion.

---

## Detector Design

### Detector 1: Pairwise Net Influence (Bonjour et al. 2022)
- I(A_i; A_j | S) estimated via smoothed plug-in CMI.
- Gamma(i→j) = gamma(i,j) - max_{k≠i,k≠j} gamma(k,j).
- Score(i,j) = min(Gamma(i→j), Gamma(j→i)).
- Threshold τ = 0.0 (any positive net influence counts).
- Calibration: threshold selected on dev data (first 5 seeds); fixed before test.

### Detector 2: Marginal MI Baseline
- I(A_i; A_j) unconditional empirical MI.
- No state conditioning.
- Should show higher FPR under shared-context conditions.

---

## Evaluation Protocol

- **Primary metric:** Top-1 accuracy on Condition C (does the true pair rank first?).
- **Secondary:** MRR, FPR (on A and B), sensitivity to α and n_rounds.
- **Unit of analysis:** Independent trajectories (seeds), NOT pairs within a trajectory.
- **Uncertainty:** Bootstrap CIs (N=2000 samples) over seeds.
- **Chance baseline:** 1/6 ≈ 0.167 for 6 candidate pairs.
- **Calibration:** Threshold τ fixed before test seeds; dev seeds 0-4, test seeds 5-19.

---

## What the Controls Permit and Don't Permit

**Permitted conclusions:**
- Whether PNI outperforms chance at identifying a deliberately coordinated pair 
  in 4-agent iterated RPS under the tested conditions.
- Whether shared context (Condition B) inflates unconditional MI but not conditional MI.
- How detection degrades as coordination strength decreases.

**NOT permitted conclusions:**
- Whether real LLM systems are colluding in practice.
- Whether the method generalizes to economic markets (different game structure).
- Whether PNI detects subtle coordination that does not involve perfectly 
  identical actions.

---

## Practical Notes

- n_rounds=200 provides ~2.5 observations per state for 81 possible states 
  (3^4 prev action profiles). This is sparse — noted as a limitation.
- Smoothing (pseudocount=0.5) partially addresses zero-count cells.
- For n_rounds=50: highly sparse, results likely noisy.
- For n_rounds=500: much better coverage, used in sensitivity analysis.
