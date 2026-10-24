# Results Interpretation

**Experiment:** 20 seeds × 200 rounds per trajectory
**Chance top-1 accuracy:** 0.167 (1/6 pairs)
**Coordinated pair:** agents [0, 1]

## Condition A (Independent Baseline)
- PNI false positive rate: 0.022
- Marginal MI false positive rate: 0.033
- Interpretation: Under fully independent behavior, a well-calibrated
  detector should have a low false positive rate.

## Condition B (Shared-Context Negative Control)
- PNI false positive rate: 0.100
- Marginal MI false positive rate: 1.000
- H2 test: Marginal MI FPR > PNI FPR would support H2 (shared context
  confounds unconditional detectors but not conditional ones).

## Condition C (Known-Coordination Positive Control)
- PNI top-1 accuracy: 1.000 (95% CI: [0.839, 1.000])
- Marginal MI top-1 accuracy: 1.000
- Chance baseline: 0.167
- H1 support: Supported
  (H1 requires lower CI bound above chance).

## Key Limitations
- Small coordinating pair; only one pair tested.
- RPS is simpler than real economic games.
- Results apply to scripted agents with controlled coordination mechanisms.
- The coordination mechanism (shared deterministic hash function) is artificial.
- No LLM agents were tested in this run (scripted experiment only).
- Short trajectories may underestimate detection performance.