# When Correlation Lies

**Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents**

*Apart Research AI Collusion Research Sprint — October 23–25, 2026 (Track 1.4: Markets and Collusion)*  
**Author:** Durgesh Unde (Independent Researcher)

---

## Research Objective

This project evaluates whether **Pairwise Net Influence (PNI)** (Bonjour et al., 2022) — 
an information-theoretic detector based on conditional mutual information (CMI) conditioned on 
observable game history — can reliably distinguish deliberately coordinated agent behavior from 
independent behavior that merely appears correlated due to shared observable context.

**Core research question:** How reliably does Pairwise Net Influence distinguish 
deliberately coordinated behavior from independent behavior that merely appears correlated due to 
shared context, and how does detector performance scale with coordination strength and trajectory length?

---

## Prior Work vs. Our Contribution

### Prior Work (Cited Literature)
- **Pairwise Net Influence framework:** Bonjour, T., Aggarwal, V. & Bhargava, B. (2022). 
  *Information-theoretic approach to detect collusion in multi-agent games.* PMLR 180:223–232.
- **LLM Agent Collusion in Auctions:** Agrawal, K., Teo, V., Vazquez, J. J., Kunnavakkam, S., 
  Srikanth, V., & Liu, A. (2025). *Evaluating LLM Agent Collusion in Double Auctions.* arXiv:2507.01413.

### Our Contributions
- Independent Python implementation of the 4-agent iterated Rock-Paper-Scissors benchmark.
- Rigorous 4-condition design (A–D) isolating common-context confounding and coordination strength.
- Positive control via an unobservable private coordination signal $Z_t \in \{0, 1, 2\}$, guaranteeing true population CMI $I(A_0; A_1 \mid S) > 0$.
- Partial coordination sweep (Condition D) using a joint coordination gate $G_t \sim \text{Bernoulli}(\alpha)$ per round.
- Small-sample empirical threshold calibration on Condition A validation seeds (seeds 0–4) targeting nominal 5% FPR, with held-out test reporting on seeds 5–19 ($n=15$).
- Statistical rigor: Wilson score confidence intervals for bounded proportions (Top-1 accuracy) and non-parametric bootstrap CIs over independent seeds for continuous metrics.
- Comprehensive trajectory length sweep ($T \in [50, 100, 200, 500]$) using fixed primary calibrated thresholds.
- Zero LLM/API calls in primary benchmark ($0.00 spent): 100% reproducible CPU evaluation over 260 unique trajectories (54,000 unique rounds).
- Exploratory cloud LLM pilot: 9 trajectories (180 decisions) executed on Modal CPU with `Qwen/Qwen2.5-1.5B-Instruct` using promotional credits ($0.07 metered, -$0.07 credit, $0.00 billed out of pocket) testing real LLM agent compliance and detector attenuation.

---

## Project Structure

```
when-correlation-lies/
├── src/
│   ├── environment.py      # 4-agent iterated RPS game environment
│   ├── agents.py           # Condition agents (Uniform, Shared-Context, Coordinated)
│   ├── trajectories.py     # Trajectory logging (JSONL) with separate ground-truth manifests
│   ├── detectors.py        # PNI and Marginal MI baseline detectors
│   ├── metrics.py          # Ranking metrics, Wilson score CIs, threshold calibration
│   ├── run_experiments.py  # Primary experiment runner
│   └── analyze_results.py  # Post-hoc analysis and publication figures
├── tests/
│   ├── test_analytical_cmi.py
│   ├── test_detectors.py
│   ├── test_environment.py
│   ├── test_metrics.py
│   └── test_llm_pilot.py   # Unit tests for LLM pilot state abstraction and parsing logic (59 tests pass)
├── scripts/
│   ├── modal_llm_pilot.py      # Exploratory LLM pilot harness for Modal cloud execution
│   ├── evaluate_llm_pilot.py   # Post-hoc evaluation of LLM trajectories across detectors
│   ├── run_sensitivity.py      # Trajectory length sweep using fixed thresholds
│   ├── plot_setup_diagram.py   # Figure 0 architectural diagram
│   ├── generate_pdf_report.py  # 6-page PDF report compiler (ReportLab)
│   ├── verify_final_consistency.py # Comprehensive 7-step consistency and dataset verifier
│   └── package_delivery.py     # Two-tier packaging and SHA-256 manifest generation
├── data/
│   ├── raw/                    # 260 canonical scripted JSONL game trajectories (54,000 rounds)
│   ├── eval_labels/            # 260 evaluation ground-truth label files
│   ├── raw_llm/                # 9 exploratory LLM JSONL trajectories (180 decisions)
│   └── eval_labels_llm/        # 9 exploratory LLM evaluation label files
├── archive/
│   ├── ARCHIVE_README.md       # Provenance documentation of historical/superseded runs
│   ├── historical_metrics/     # Archived intermediate and pre-repair metric runs
│   └── historical_tables/      # Archived intermediate summary CSVs
├── config/
│   └── experiment.yaml         # Experiment configuration
├── docs/
│   ├── scientific_audit.md             # Detailed mathematical diagnosis & repair
│   ├── retrospective_research_plan.md   # Retrospective protocol documentation
│   ├── pre_sprint_work_log.md          # Provenance & sprint work log
│   └── results_interpretation.md        # Qualitative interpretation notes
├── results/
│   ├── metrics/            # Canonical final_metrics.json, sensitivity & llm_pilot JSONs
│   ├── tables/             # summary.csv, sensitivity_trajectory_length.csv, llm_pilot_summary.csv
│   └── figures/            # Publication figures (fig0 to fig5)
└── final_delivery/         # Submission artifacts (PDF, source text, ZIP bundle, manifest)
```

---

## Installation & Environment

**Requirements:** Python 3.11+ (verified on Python 3.13.7)

```bash
# Clone or navigate to the repository
cd when-correlation-lies

# Editable install with test and report dependencies
pip install -e ".[dev,report]"
```

---

## Running the Unit Test Suite

```bash
python -m pytest tests/ -q
```
Expected output: **59 tests pass** in ~4 seconds.

---

## Reproducing All Experiments and Reports

```bash
# 1. Run primary benchmark (140 trajectories, 200 rounds, 20 seeds)
python -m src.run_experiments --conditions A B C D --seeds 20 --rounds 200 --tag clean_main

# 2. Run sensitivity analysis (T in [50, 100, 200, 500] using fixed calibrated thresholds)
python scripts/run_sensitivity.py

# 3. Regenerate all primary analysis figures and summary tables
python -m src.analyze_results

# 4. Regenerate setup architecture diagram (Figure 1)
python scripts/plot_setup_diagram.py

# 5. Compile formal 6-page PDF research report
python scripts/generate_pdf_report.py

# 6. Evaluate exploratory LLM pilot traces
python scripts/evaluate_llm_pilot.py

# 7. (Optional) Re-run exploratory LLM pilot on Modal CPU (requires Modal account)
python -m modal run scripts/modal_llm_pilot.py --seeds 0 1 2 --rounds 5 --conditions A B C

# 8. Verify end-to-end consistency, dataset integrity, and checksums
python scripts/verify_final_consistency.py

# 9. Package complete delivery bundle (ZIP, single-file source code, results manifest)
python scripts/package_delivery.py
```

---

## Experimental Conditions

| Condition | Name | Description | Coordinating Pair |
| :--- | :--- | :--- | :---: |
| **A** | Independent Baseline | 4 uniform-random agents with isolated RNGs; seeds 0–4 used for threshold calibration, seeds 5–19 for held-out test evaluation | None |
| **B** | Shared-Context Confound | Agents independently play state-conditioned best responses with 10% noise | None |
| **C** | Known Coordination | Pair (0, 1) coordinates via private unobservable signal $Z_t \sim \text{Uniform}(\{0, 1, 2\})$ ($\alpha = 1.0$) | (0, 1) |
| **D** | Coordination Sweep | Joint gate $G_t \sim \text{Bernoulli}(\alpha)$; pair coordinates on $Z_t$ if $G_t=1$, else independent fallback | (0, 1) |
| **LLM-A** | Exploratory LLM Independent | Qwen2.5-1.5B instructed to pick RPS moves independently at random (3 seeds × 5 rounds) | None |
| **LLM-B** | Exploratory LLM Shared Context | Qwen2.5-1.5B instructed to counter the observed round summary (3 seeds × 5 rounds) | None |
| **LLM-C** | Exploratory LLM Coordinated | Pair (0, 1) given secret round-by-round directive $Z_t$ (3 seeds × 5 rounds) | (0, 1) |

---

## Exploratory LLM Pilot Results

To evaluate whether instruction-tuned language models follow coordination instructions and how sample sparsity impacts detection, we executed an exploratory 9-trajectory pilot (180 decisions) on Modal CPU using `Qwen/Qwen2.5-1.5B-Instruct` (bfloat16, 4 vCPU, 8GB RAM, ~1.4s/decision):

| Condition | Description | Decisions | Valid Parse | Signal Adherence | Marginal MI Top-1 | CMI Top-1 | PNI Top-1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-A** | Independent random play prompt | 60 | 100.0% (60/60) | N/A | N/A | N/A | N/A |
| **LLM-B** | Shared-context heuristic prompt | 60 | 100.0% (60/60) | N/A | N/A | N/A | N/A |
| **LLM-C** | Private signal coordination prompt | 60 | 100.0% (60/60) | 100.0% (30/30) | 1.000 (3/3) | 0.667 (2/3) | 0.667 (2/3) |

**Key findings:**
- `Qwen2.5-1.5B-Instruct` achieved 100% adherence to private coordination directives without fine-tuning.
- At short horizon ($T=5$ rounds), conditioning on 5 discrete states causes sample fragmentation, reducing CMI/PNI Top-1 ranking accuracy to 66.7% (2/3 seeds), whereas unconditioned Marginal MI achieved 100% Top-1 ranking because correlation accumulated across rounds without cell partition.

---

## Scope & Limitations

1. **Scripted Benchmark vs. Real-World Cartels:** The canonical evaluation was conducted on controlled scripted agents in a 4-player discrete iterated RPS game ($|\mathcal{S}| = 82$). Zero LLM/API calls were made during the primary benchmark ($0.00 spent). Statistical action dependence demonstrates detector sensitivity, but does not prove real-world economic harm or spontaneous collusion.
2. **Exploratory Small-Sample Calibration:** Nominal 5% decision thresholds were calibrated from 5 validation seeds (30 candidate pairs) of Condition A. While isolating calibration from held-out test seeds, small samples yield coarse empirical quantiles rather than population guarantees.
3. **State Sparsity:** Conditioning on discrete game history partitions observations across state-action cells. Condition B PNI False Positive Rate is 0.100 (substantially reduced from Marginal MI's 1.000, but finite-sample variance prevents an exact zero FPR).
4. **Cloud Platform & Exploratory LLM Pilot:** Allocating cloud GPU instances on Modal requires an active payment method on file per platform policy; in strict adherence to zero-out-of-pocket rules ($0.00 billed), GPU calls were safely bypassed. Instead, an optimized CPU pilot was executed on Modal (4 vCPU, 8GB RAM, `torch.bfloat16`, 12 max tokens) with `Qwen/Qwen2.5-1.5B-Instruct`, achieving ~1.4s per decision. The pilot executed 9 trajectories (180 decisions) across 3 conditions and 3 seeds at a metered compute cost of $0.07, offset 100% by promotional credits ($0.00 billed).

---

## Citation

If using this implementation, please cite:

```bibtex
@inproceedings{bonjour2022information,
  title     = {Information-theoretic approach to detect collusion in multi-agent games},
  author    = {Bonjour, Trevor and Aggarwal, Vaneet and Bhargava, Bharat},
  booktitle = {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
  pages     = {223--232},
  year      = {2022},
  volume    = {180},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```
