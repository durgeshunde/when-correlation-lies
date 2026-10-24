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
shared context, and how does detector performance scale with coordination strength, trajectory length, and open-weight LLM architectures?

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
- Multi-horizon open-weight LLM agent experiment on Modal CPU: 99 trajectories (9,180 decisions) executed with `Qwen/Qwen2.5-1.5B-Instruct` across $T \in \{5, 20, 50\}$ rounds using promotional credits ($5.41 metered cost, $5.34 credits, $0.07 network allowance, $0.00 billed out of pocket), demonstrating 100% secret signal compliance and revealing the finite-sample horizon sparsity effect (66.7% Top-1 accuracy at $T=5$, restored to 100.0% at $T \ge 20$).

---

## Project Structure

```
when-correlation-lies/
├── src/
│   ├── environment.py      # 4-agent iterated RPS game environment
│   ├── agents.py           # Condition agents (Uniform, Shared-Context, Coordinated)
│   ├── trajectories.py     # Trajectory logging (JSONL) with separate ground-truth manifests
│   ├── detectors.py        # PNI, CMI, and Marginal MI baseline detectors
│   ├── metrics.py          # Ranking metrics, Wilson score CIs, threshold calibration
│   ├── run_experiments.py  # Primary experiment runner
│   └── analyze_results.py  # Post-hoc analysis and publication figures
├── tests/
│   ├── test_analytical_cmi.py
│   ├── test_detectors.py
│   ├── test_environment.py
│   ├── test_metrics.py
│   └── test_llm_pilot.py   # Unit tests for LLM state abstraction and parsing logic (59 tests pass)
├── scripts/
│   ├── modal_llm_pilot.py      # Initial exploratory LLM pilot harness for Modal cloud execution
│   ├── modal_llm_expanded.py   # High-throughput batch harness for expanded multi-horizon LLM studies
│   ├── evaluate_llm_pilot.py   # Multi-horizon evaluation of LLM trajectories across detectors
│   ├── plot_llm_results.py     # Generates publication Figure 4 (LLM horizon scaling)
│   ├── run_sensitivity.py      # Trajectory length sweep using fixed thresholds
│   ├── plot_setup_diagram.py   # Figure 1 architectural diagram
│   ├── generate_pdf_report.py  # Formal 7-page PDF research report compiler (ReportLab)
│   ├── verify_final_consistency.py # Comprehensive end-to-end consistency and dataset verifier
│   └── package_delivery.py     # Two-tier packaging and SHA-256 manifest generation
├── data/
│   ├── raw/                    # 260 canonical scripted JSONL game trajectories (54,000 rounds)
│   ├── eval_labels/            # 260 evaluation ground-truth label files
│   ├── raw_llm/                # 9 pilot LLM JSONL trajectories (180 decisions, T=5)
│   ├── eval_labels_llm/        # 9 pilot LLM evaluation label files
│   ├── raw_llm_expanded/       # 75 expanded LLM JSONL trajectories (6,000 decisions, T=20, 25 seeds)
│   ├── eval_labels_llm_expanded/# 75 expanded LLM evaluation label files
│   ├── raw_llm_expanded_t50/   # 15 extended LLM JSONL trajectories (3,000 decisions, T=50, 5 seeds)
│   └── eval_labels_llm_expanded_t50/# 15 extended LLM evaluation label files
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
│   ├── metrics/            # Canonical final_metrics.json, sensitivity & llm_pilot/expanded JSONs
│   ├── tables/             # summary.csv, sensitivity_trajectory_length.csv, llm summaries
│   ├── figures/            # Publication figures (fig0 to fig6)
│   └── llm_pilot/          # Billing audit logs (pre- and post-batch credit tracking)
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
Expected output: **59 passed** in ~4 seconds.

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

# 5. Evaluate all LLM experimental traces (T=5, T=20, T=50)
python scripts/evaluate_llm_pilot.py --raw-dir data/raw_llm --eval-dir data/eval_labels_llm --output-metrics results/metrics/llm_pilot_metrics.json --output-table results/tables/llm_pilot_summary.csv
python scripts/evaluate_llm_pilot.py --raw-dir data/raw_llm_expanded --eval-dir data/eval_labels_llm_expanded --output-metrics results/metrics/llm_expanded_metrics.json --output-table results/tables/llm_expanded_summary.csv
python scripts/evaluate_llm_pilot.py --raw-dir data/raw_llm_expanded_t50 --eval-dir data/eval_labels_llm_expanded_t50 --output-metrics results/metrics/llm_expanded_t50_metrics.json --output-table results/tables/llm_expanded_t50_summary.csv

# 6. Regenerate LLM multi-horizon scaling figure (Figure 4)
python scripts/plot_llm_results.py

# 7. Compile formal 7-page PDF research report
python scripts/generate_pdf_report.py

# 8. (Optional) Run LLM experiments on Modal CPU (requires authorized Modal account)
python scripts/modal_llm_expanded.py --rounds 20 --seeds 100-124 --conditions A B C

# 9. Verify end-to-end consistency, dataset integrity, and checksums
python scripts/verify_final_consistency.py

# 10. Package complete delivery bundle (ZIP, single-file source code, results manifest)
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
| **LLM-A** | LLM Independent | Qwen2.5-1.5B instructed to pick RPS moves independently at random | None |
| **LLM-B** | LLM Shared Context | Qwen2.5-1.5B instructed to counter the observed round plurality summary | None |
| **LLM-C** | LLM Coordinated | Pair (0, 1) given secret round-by-round directive $Z_t$ (strictly excluded from detector logs) | (0, 1) |

---

## Multi-Horizon Open-Weight LLM Empirical Results

To evaluate whether instruction-tuned language models follow coordination directives and how sample horizon impacts detection, we deployed open-weight `Qwen/Qwen2.5-1.5B-Instruct` on Modal CPU instances across 99 trajectories (9,180 total decisions):

| Horizon | Condition | Trajectories | Total Decisions | Valid Parse Rate | Signal Adherence | Top-1 Marg. MI | Top-1 CMI | Top-1 PNI | Mean CMI (bits) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=5$** | **LLM-A (Ind.)** | 3 | 60 | 100.0% (60/60) | N/A | N/A | N/A | N/A | 0.0768 |
|  | **LLM-B (Cont.)** | 3 | 60 | 100.0% (60/60) | N/A | N/A | N/A | N/A | 0.0706 |
|  | **LLM-C (Coord.)** | 3 | 60 | 100.0% (60/60) | **100.0% (30/30)** | **1.000 (3/3)** | **0.667 (2/3)** | **0.667 (2/3)** | 0.0678 |
| **$T=20$** | **LLM-A (Ind.)** | 25 | 2,000 | 100.0% (2000/2000) | N/A | N/A | N/A | N/A | 0.0979 |
|  | **LLM-B (Cont.)** | 25 | 2,000 | 100.0% (2000/2000) | N/A | N/A | N/A | N/A | 0.0983 |
|  | **LLM-C (Coord.)** | 25 | 2,000 | 100.0% (2000/2000) | **100.0% (1000/1000)** | **1.000 (25/25)** | **1.000 (25/25)** | **1.000 (25/25)** | 0.1332 |
| **$T=50$** | **LLM-A (Ind.)** | 5 | 1,000 | 100.0% (1000/1000) | N/A | N/A | N/A | N/A | 0.0838 |
|  | **LLM-B (Cont.)** | 5 | 1,000 | 100.0% (1000/1000) | N/A | N/A | N/A | N/A | 0.0797 |
|  | **LLM-C (Coord.)** | 5 | 1,000 | 100.0% (1000/1000) | **100.0% (500/500)** | **1.000 (5/5)** | **1.000 (5/5)** | **1.000 (5/5)** | 0.1723 |

**Key findings:**
- `Qwen2.5-1.5B-Instruct` achieved 100.0% adherence to private coordination directives across all 1,530 opportunities in Condition LLM-C without fine-tuning.
- **Finite-Sample Horizon Sparsity Effect:** At short horizon ($T=5$ rounds), conditioning on 5 discrete states causes sample fragmentation ($E[N_s] \approx 1$), reducing CMI and PNI Top-1 ranking accuracy to 66.7% (2/3 seeds). Expanding the horizon to $T=20$ rounds (6,000 decisions across 25 independent seeds) restores **100.0% Top-1 ranking accuracy** across all estimators (25/25 seeds, Wilson 95% CI: [0.867, 1.000]), which remains 100% at $T=50$ (5/5 seeds) with CMI climbing to 0.1723 bits.
- **Resource and Cloud Accounting:** The entire cloud suite ran on Modal CPU (64.8 min container time, $5.41 metered compute, -$5.34 promotional credit, -$0.07 network allowance = $0.00 billed out-of-pocket).

---

## Scope & Limitations

1. **Scripted Benchmark vs. Real-World Cartels:** The canonical evaluation was conducted on controlled scripted agents in a 4-player discrete iterated RPS game ($|\mathcal{S}| = 82$). Zero LLM/API calls were made during the primary benchmark ($0.00 spent). Statistical action dependence demonstrates detector sensitivity, but does not prove real-world economic harm or spontaneous collusion.
2. **Exploratory Small-Sample Calibration:** Nominal 5% decision thresholds were calibrated from 5 validation seeds (30 candidate pairs) of Condition A. While isolating calibration from held-out test seeds, small samples yield coarse empirical quantiles rather than population guarantees.
3. **State Sparsity:** Conditioning on discrete game history partitions observations across state-action cells. Condition B PNI False Positive Rate is 0.100 (substantially reduced from Marginal MI's 1.000, but finite-sample variance prevents an exact zero FPR).
4. **Cloud Platform & LLM Execution:** Allocating cloud GPU instances on Modal requires an active payment method on file per platform policy; in strict adherence to zero-out-of-pocket rules ($0.00 billed), GPU calls were safely bypassed. Instead, an optimized CPU pipeline was executed on Modal (4 vCPU, 8GB RAM, `torch.bfloat16`, 12 max tokens) with `Qwen/Qwen2.5-1.5B-Instruct`, achieving ~1.4s per decision. The suite executed 99 trajectories (9,180 decisions) across 3 conditions and 3 horizons at a metered compute cost of $5.41, offset 100% by promotional credits ($0.00 billed).

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
