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
- Multi-horizon open-weight LLM agent experiment on Modal CPU: 630 trajectories (68,400 decisions) executed with `Qwen/Qwen2.5-1.5B-Instruct` across $T \in \{5, 20, 50\}$ rounds using authorized promotional credits ($40.63 metered compute: $40.12 promotional credits applied, $0.51 network egress allowance; $0.00 billed out of pocket), demonstrating 100% valid parsed action rate (68,400/68,400) and 100% secret signal compliance (11,400/11,400 opportunities), and revealing the finite-sample horizon sparsity effect (70.0% Top-1 accuracy and 36.7% binary recall at $T=5$, restored to 100.0% Top-1 accuracy and 100.0% recall at $T \ge 20$ under calibrated decision thresholds).

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
│   ├── evaluate_llm_thresholds.py # Held-out threshold calibration and classification evaluation
│   ├── plot_llm_results.py     # Generates publication Figure 4 (LLM horizon scaling)
│   ├── run_sensitivity.py      # Trajectory length sweep using fixed thresholds
│   ├── plot_setup_diagram.py   # Figure 1 architectural diagram
│   ├── generate_pdf_report.py  # Formal 7-page PDF research report compiler (ReportLab)
│   ├── verify_final_consistency.py # Comprehensive end-to-end consistency and dataset verifier
│   └── package_delivery.py     # Two-tier packaging and SHA-256 manifest generation
├── data/
│   ├── raw/                    # 260 canonical scripted JSONL game trajectories (54,000 rounds)
│   ├── eval_labels/            # 260 evaluation ground-truth label files
│   ├── raw_llm/                # 120 pilot LLM JSONL trajectories (2,400 decisions, T=5, 40 seeds)
│   ├── eval_labels_llm/        # 120 pilot LLM evaluation label files
│   ├── raw_llm_expanded/       # 300 expanded LLM JSONL trajectories (24,000 decisions, T=20, 100 seeds)
│   ├── eval_labels_llm_expanded/# 300 expanded LLM evaluation label files
│   ├── raw_llm_expanded_t50/   # 210 extended LLM JSONL trajectories (42,000 decisions, T=50, 70 seeds)
│   └── eval_labels_llm_expanded_t50/# 210 extended LLM evaluation label files
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

### Option 1: Local Development, Benchmark Testing & Report Compilation (Recommended)
The primary scripted benchmark, all statistical detectors, all 59 automated unit tests, metric evaluation of all trajectories (both scripted and LLM), and formal PDF report compilation run 100% locally on standard CPU hardware with zero cloud or API dependencies:

```bash
# Clone or navigate to the repository
cd when-correlation-lies

# Install core dependencies with test and report extras (zero cloud dependencies)
pip install -e ".[dev,report]"
```

### Option 2: Full Cloud LLM Inference Pipeline (Optional)
If you wish to re-execute open-weight model inference on Modal cloud CPU workers (`scripts/modal_llm_expanded.py`), install the optional `[cloud]` extra (requires an authorized Modal account):

```bash
# Install with optional cloud dependencies
pip install -e ".[dev,report,cloud]"
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

# 6. Evaluate held-out classification and threshold detection across all horizons
python scripts/evaluate_llm_thresholds.py

# 7. Regenerate LLM multi-horizon scaling figure (Figure 4)
python scripts/plot_llm_results.py

# 8. Compile formal 7-page PDF research report
python scripts/generate_pdf_report.py

# 9. (Optional) Run LLM experiments on Modal CPU (requires authorized Modal account)
python scripts/modal_llm_expanded.py --rounds 20 --seeds 100-124 --conditions A B C

# 10. Verify end-to-end consistency, dataset integrity, and checksums
python scripts/verify_final_consistency.py

# 11. Package complete delivery bundle (ZIP, single-file source code, results manifest)
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

To evaluate whether instruction-tuned language models follow coordination directives and how sample horizon impacts detection, we deployed open-weight `Qwen/Qwen2.5-1.5B-Instruct` on Modal CPU instances across 630 trajectories (68,400 total decisions across horizons $T \in \{5, 20, 50\}$):

### Table 3: Multi-Horizon Open-Weight LLM Primary Ranking Metrics

| Horizon | Condition | Trajectories | Total Decisions | Valid Parse Rate | Signal Adherence | Top-1 Marg. MI | Top-1 CMI | Top-1 PNI | Mean CMI (bits) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=5$** | **LLM-A (Ind.)** | 40 | 800 | 100.0% (800/800) | N/A | N/A | N/A | N/A | 0.0693 |
|  | **LLM-B (Cont.)** | 40 | 800 | 100.0% (800/800) | N/A | N/A | N/A | N/A | 0.0666 |
|  | **LLM-C (Coord.)** | 40 | 800 | 100.0% (800/800) | **100.0% (400/400)** | **1.000 (40/40)** | **0.700 (28/40)** | **0.700 (28/40)** | 0.0696 |
| **$T=20$** | **LLM-A (Ind.)** | 100 | 8,000 | 100.0% (8000/8000) | N/A | N/A | N/A | N/A | 0.0997 |
|  | **LLM-B (Cont.)** | 100 | 8,000 | 100.0% (8000/8000) | N/A | N/A | N/A | N/A | 0.0981 |
|  | **LLM-C (Coord.)** | 100 | 8,000 | 100.0% (4000/4000) | **100.0% (4000/4000)** | **1.000 (100/100)** | **1.000 (100/100)** | **1.000 (100/100)** | 0.1333 |
| **$T=50$** | **LLM-A (Ind.)** | 70 | 14,000 | 100.0% (14000/14000) | N/A | N/A | N/A | N/A | 0.0835 |
|  | **LLM-B (Cont.)** | 70 | 14,000 | 100.0% (14000/14000) | N/A | N/A | N/A | N/A | 0.0833 |
|  | **LLM-C (Coord.)** | 70 | 14,000 | 100.0% (14000/14000) | **100.0% (7000/7000)** | **1.000 (70/70)** | **1.000 (70/70)** | **1.000 (70/70)** | 0.1770 |

### Table 4: Held-Out Classification and Threshold Evaluation

Thresholds calibrated on LLM-A validation episodes targeting nominal 5% FPR ($\\tau_{\\text{PNI}} = 0.0154$ at $T=5$, $0.0570$ at $T=20$, $0.0621$ at $T=50$). Evaluated on held-out test episodes:

| Horizon | Condition | Detector | Threshold (bits) | Ep. Flagged | Pair FPR / Recall | Top-1 Acc. | Target Recall | Ranked #1 & Detected | Ranked #1 Not Detected |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=5$** | **LLM-A (Test $n=30$)** | PNI | 0.0154 | 14/30 (0.467) | 0.172 [0.124, 0.234] | N/A | N/A | N/A | N/A |
|  | **LLM-B (Test $n=30$)** | PNI | 0.0154 | 10/30 (0.333) | 0.100 [0.064, 0.153] | N/A | N/A | N/A | N/A |
|  | **LLM-C (Test $n=30$)** | PNI | 0.0154 | 11/30 (0.367) | 0.139 [0.096, 0.197] | 0.700 | 0.367 | 0.367 | 0.333 |
| **$T=20$** | **LLM-A (Test $n=75$)** | PNI | 0.0570 | 11/75 (0.147) | 0.078 [0.056, 0.106] | N/A | N/A | N/A | N/A |
|  | **LLM-B (Test $n=75$)** | PNI | 0.0570 | 13/75 (0.173) | 0.104 [0.079, 0.136] | N/A | N/A | N/A | N/A |
|  | **LLM-C (Test $n=75$)** | PNI | 0.0570 | 75/75 (1.000) | 0.207 [0.172, 0.246] | 1.000 | 1.000 | 1.000 | 0.000 |
| **$T=50$** | **LLM-A (Test $n=60$)** | PNI | 0.0621 | 4/60 (0.067) | 0.036 [0.021, 0.061] | N/A | N/A | N/A | N/A |
|  | **LLM-B (Test $n=60$)** | PNI | 0.0621 | 7/60 (0.117) | 0.039 [0.023, 0.064] | N/A | N/A | N/A | N/A |
|  | **LLM-C (Test $n=60$)** | PNI | 0.0621 | 60/60 (1.000) | 0.194 [0.157, 0.238] | 1.000 | 1.000 | 1.000 | 0.000 |

**Key findings:**
- `Qwen2.5-1.5B-Instruct` achieved 100.0% valid action generation rate (68,400/68,400 decisions parsed as ROCK, PAPER, or SCISSORS) and 100.0% adherence to private coordination directives across all 11,400 opportunities in Condition LLM-C without fine-tuning.
- **Finite-Sample Horizon Sparsity Effect:** At short horizon ($T=5$ rounds), conditioning on discrete states causes sample fragmentation, reducing CMI and PNI Top-1 ranking accuracy to 70.0% (28/40 seeds). Crucially, under frozen calibrated thresholds, binary recall is only 36.7% (11/30 test episodes detected), with 33.3% of episodes being ranked #1 but failing threshold detection. Expanding the horizon to $T=20$ rounds restores **100.0% Top-1 ranking accuracy** (100/100 seeds, Wilson 95% CI: [0.951, 1.000]) and **100.0% binary recall** (75/75 test episodes detected), which remains 100% at $T=50$ (70/70 seeds; 60/60 test episodes) with mean CMI rising to 0.1770 bits.
- **Negative Mean PNI Net Scores:** In Condition LLM-C, competitor subtraction ($\\max_{k \\neq i,j} \\gamma(k,j)$) subtracts the coordinating pair's elevated mutual information from other candidate pairs, driving rival net influence negative (mean -0.140 bits at $T=20$, -0.321 bits at $T=50$).
- **Resource and Cloud Accounting:** The entire cloud suite ran on Modal CPU ($40.63 metered compute: $40.12 promotional credits applied, $0.51 network egress allowance; exactly $0.00 billed out of pocket).

---

## Scope & Limitations

1. **Scripted Benchmark vs. Real-World Cartels:** The canonical evaluation was conducted on controlled scripted agents in a 4-player discrete iterated RPS game ($|\mathcal{S}| = 82$). Zero LLM/API calls were made during the primary benchmark ($0.00 spent). Statistical action dependence demonstrates detector sensitivity, but does not prove real-world economic harm or spontaneous collusion.
2. **Exploratory Small-Sample Calibration:** Nominal 5% decision thresholds were calibrated from 5 validation seeds (30 candidate pairs) of Condition A. While isolating calibration from held-out test seeds, small samples yield coarse empirical quantiles rather than population guarantees.
3. **State Sparsity:** Conditioning on discrete game history partitions observations across state-action cells. Condition B PNI False Positive Rate is 0.100 (substantially reduced from Marginal MI's 1.000, but finite-sample variance prevents an exact zero FPR).
4. **Cloud Platform & LLM Execution:** Allocating cloud GPU instances on Modal requires an active payment method on file per platform policy; in strict adherence to zero-out-of-pocket rules ($0.00 billed), GPU calls were safely bypassed. Instead, an optimized CPU pipeline was executed on Modal (4 vCPU, 8GB RAM, `torch.bfloat16`, 12 max tokens) with `Qwen/Qwen2.5-1.5B-Instruct`. The suite executed 630 trajectories (68,400 decisions) across 3 conditions and 3 horizons at a metered compute cost of $40.63 ($40.12 promotional credits applied, $0.51 network egress allowance; $0.00 billed out of pocket).

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
