# When Correlation Lies

**Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents**

*Apart Research AI Collusion Research Sprint — October 23–25, 2026*

---

## Research Objective

This project evaluates whether **pairwise net influence** (Bonjour et al., 2022) — 
an information-theoretic detector based on conditional mutual information — can 
reliably distinguish deliberately coordinated agent behavior from independent 
behavior that merely appears correlated due to shared context.

**Research question:** How reliably does pairwise net influence distinguish 
deliberately coordinated behavior from independent behavior that merely appears 
correlated, and how does detection quality degrade as coordination weakens?

## Prior Work vs. Our Contribution

### Prior Work (not invented here)
- **Pairwise net influence method:** Bonjour, T., Aggarwal, V. & Bhargava, B. 
  (2022). *Information-theoretic approach to detect collusion in multi-agent 
  games.* PMLR 180:223–232. 
  https://proceedings.mlr.press/v180/bonjour22a.html
- **LLM agent double-auction environment (reference only):** 
  https://github.com/andyjliu/llm-agent-collusion

### Our Contributions
- Independent Python implementation of pairwise net influence and baselines.
- Controlled experimental design with four conditions (A–D).
- Shared-context negative control isolating common-context confounding.
- Coordination-strength sweep (Condition D) characterizing the detection boundary.
- Bootstrap confidence intervals over independent replications.
- Evaluation pipeline that strictly separates ground-truth labels from detector inputs.
- Honest analysis of both positive and negative results.

---

## Project Structure

```
when-correlation-lies/
├── src/
│   ├── environment.py      # RPS game environment
│   ├── agents.py           # Four condition agent types
│   ├── trajectories.py     # Trajectory logging and loading
│   ├── detectors.py        # PNI + MI baseline detectors
│   ├── metrics.py          # Evaluation metrics + bootstrap CI
│   ├── run_experiments.py  # Main experiment runner
│   └── analyze_results.py  # Analysis and figure generation
├── tests/
│   ├── test_environment.py
│   └── test_detectors.py
├── docs/
│   ├── literature_notes.md
│   ├── research_design.md
│   ├── sprint_requirements.md
│   ├── decision_log.md
│   ├── pre_sprint_work_log.md
│   └── results_interpretation.md   # generated after experiments
├── data/
│   ├── raw/                # JSONL trajectories (detector-visible)
│   └── eval_labels/        # Ground-truth manifests (separate)
├── results/
│   ├── metrics/            # JSON results + final_metrics.json
│   ├── tables/             # summary.csv
│   └── figures/            # PNG figures
└── report/                 # Research report PDF and source
```

---

## Installation

**Requirements:** Python 3.11+

```bash
pip install numpy scipy matplotlib pandas scikit-learn pytest
```

Or install all project dependencies:

```bash
pip install -e ".[dev]"
```

## Required Environment Variables

For the optional LLM pilot (Phase 4), create a `.env` file from the template:

```bash
copy .env.example .env
# Edit .env to add API keys
```

The core scripted experiment requires NO API keys.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output: **38 tests pass**.

## Reproducing the Scripted Experiment

```bash
# Primary experiment: conditions A, B, C, D with 20 seeds, 200 rounds
python -m src.run_experiments --conditions A B C D --seeds 20 --rounds 200 --alphas 0.25 0.5 0.75 1.0 --tag scripted_main

# Sensitivity: vary trajectory length
python -m src.run_experiments --conditions C --seeds 20 --rounds 50 --tag sensitivity_50r
python -m src.run_experiments --conditions C --seeds 20 --rounds 500 --tag sensitivity_500r
```

## Running the Analysis

```bash
python -m src.analyze_results
```

Produces figures in `results/figures/` and a summary in `results/tables/summary.csv`.

---

## Experimental Conditions

| Condition | Description | True Pair |
|-----------|-------------|-----------|
| A | Independent (uniform random, separate RNGs) | None |
| B | Shared context (independent best-response given public state) | None |
| C | Known coordination (pair 0,1 both play hash(state) mod 3) | (0, 1) |
| D | Partial coordination (alpha sweep: 0.25, 0.5, 0.75, 1.0) | (0, 1) |

## Detectors

1. **Pairwise Net Influence** (Bonjour et al. 2022): CMI-based, state-conditioned.
2. **Marginal MI baseline**: Unconditional MI, no state conditioning.

---

## Key Results (filled after experiment runs)

See `results/tables/summary.csv` and `docs/results_interpretation.md`.

---

## Known Limitations

- Sparse state coverage: 200 rounds vs 81 possible states in RPS.
- Artificial coordination mechanism (deterministic hash function).
- No LLM agents in the scripted study (Phase 4 is optional).
- Results apply to the tested game and coordination mechanism.
- Finite-sample CMI estimates may be biased upward.

## Implementation Deviations from Paper

- Paper does not specify smoothing; we use pseudocount = 0.5 (documented in `detectors.py`).
- Paper uses a coordination mechanism where agents exploit a third agent; 
  our mechanism (hash-based coordination) is different but produces similar 
  statistical structure (correlated actions given state).

---

## Ethical Considerations

This project studies a detection method for AI coordination. All experiments 
use fully controlled synthetic environments. No real AI systems are studied. 
The detection method, if effective, would be used to identify coordination 
that disadvantages other participants — a safety-relevant application.

See the Limitations and Ethical Considerations appendix in the report.

---

## Citation

If using this code, please also cite the primary method:

```bibtex
@inproceedings{bonjour22a,
  title     = {Information theoretic approach to detect collusion in multi-agent games},
  author    = {Bonjour, Trevor and Aggarwal, Vaneet and Bhargava, Bharat},
  booktitle = {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
  pages     = {223--232},
  year      = {2022},
  volume    = {180},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v180/bonjour22a.html}
}
```
