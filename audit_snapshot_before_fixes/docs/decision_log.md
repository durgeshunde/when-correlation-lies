# Decision Log

## Purpose
Records consequential implementation decisions and their rationale.

---

## D001: Coordination mechanism choice (Condition C)
**Decision:** Use `hash(state_feature) % 3` as the coordination rule.  
**Alternatives considered:** (a) shared noisy signal added to state, (b) 
mirror agent (one copies the other), (c) fixed deterministic action sequence.  
**Rationale:** The hash approach produces coordination that is:
- Fully determined by the PUBLIC state (auditor can verify it).
- Not trivially detectable as "always same action regardless of state."
- Produces action variety (R, P, S all appear) based on state, making it 
  a harder detection problem than a constant colluder.
- The coordinating pair always plays the SAME action — this is the CMI signal.

**Implication:** Because coordination is deterministic given state, the CMI 
of coordinated pairs given state is nonzero (they are NOT independent given S).
The marginal MI is also nonzero. Both detectors should in principle detect this.
The question is whether PNI's state-conditioning provides a qualitative 
advantage under Condition B (shared context) negative control.

---

## D002: Smoothing pseudocount = 0.5
**Decision:** Use additive Laplace smoothing with pseudocount = 0.5 (Jeffreys 
analog) for all MI and CMI estimates.  
**Rationale:** 
- Prevents log(0) on zero-count cells.
- 0.5 is a principled Bayesian analog (Jeffreys prior for Dirichlet).
- Standard choice in entropy estimation literature.
- Fixed before seeing any experimental results.

**Alternative considered:** pseudocount = 1.0 (standard Laplace).  
**Rejected because:** 0.5 is less aggressive and preserves more of the 
empirical signal with N=200+ observations.

---

## D003: Detection threshold τ = 0.0
**Decision:** Default threshold τ = 0.0 (any positive net influence counts).  
**Rationale:** 
- Paper does not specify a universal threshold; 0.0 is the natural boundary.
- With smoothing, true net influence should be near 0 for independent pairs 
  and positive for coordinated pairs.
- Calibration on dev seeds (0-4) is documented; test seeds (5-19) untouched.

**Limitation:** A threshold of 0.0 may produce false positives due to 
finite-sample noise. This is explicitly studied in the FPR analysis.

---

## D004: SharedContextAgent policy (best-response heuristic)
**Decision:** Policy = "play the action that beats the most-played action 
from the previous round."  
**Rationale:**
- Plausible strategic behavior (bounded rationality best response).
- State-conditioned: creates unconditional correlations without coordination.
- With 4 agents each independently playing this policy, Condition B 
  creates genuine shared-context confounding.

**Alternative considered:** Fixed mapping from state → action using a 
different hash. Rejected: too similar to the coordination mechanism.

---

## D005: Trajectory save/load pipeline
**Decision:** JSONL format for trajectories; separate JSON file for ground-truth labels.  
**Rationale:**
- JSONL is human-readable, streamable, and easy to validate.
- Strict separation of ground-truth file ensures detectors cannot accidentally 
  consume labels.
- Files are named by deterministic run_id derived from seed+condition.

---

## D006: Bootstrap CI over independent seeds
**Decision:** Bootstrap CIs computed over seeds, treating each seed as one 
independent observation.  
**Rationale:** 
- Within a single trajectory, the 6 pair scores are NOT independent (they 
  share the same action sequences).
- Treating seeds as the replication unit gives valid uncertainty estimates.
- 2000 bootstrap samples is sufficient for stable 95% CIs with N≥10 seeds.

---

## D007: n_rounds = 200 as primary configuration
**Decision:** 200 rounds per trajectory for primary experiments.  
**Rationale:**
- 81 possible states (3^4 for 4-agent RPS) → ~2.5 obs/state average.
- This is sparse but tractable with smoothing.
- 500 rounds used in sensitivity analysis (better coverage, same computation).
- 50 rounds tested to show degradation.

---

## D008: Coordination pair fixed as (0, 1)
**Decision:** Always use agents 0 and 1 as the coordinated pair.  
**Rationale:**
- Simplifies ground-truth tracking.
- With 4 agents, result generalizes by relabeling symmetry.
- Fixes potential confound from varying which agents coordinate.
