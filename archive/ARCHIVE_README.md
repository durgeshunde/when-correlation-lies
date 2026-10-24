# Historical Experimental Archive & Superseded Runs

This directory preserves prior exploratory and intermediate benchmark runs from the development
of the study **When Correlation Lies: Stress-Testing Pairwise Net Influence** (Apart Research AI Collusion Research Sprint 2026, Track 1.4: Markets and Collusion).

These files are preserved for auditability and provenance. They are **not** the canonical results of the manuscript. Canonical benchmark metrics reside exclusively in [`results/metrics/final_metrics.json`](../results/metrics/final_metrics.json) and [`results/tables/summary.csv`](../results/tables/summary.csv).

---

## 1. Inventory of Archived Artifacts

### Historical Metrics (`archive/historical_metrics/`)
1. **`results_scripted_main_20261023_203238.json`** [INVALID / SUPERSEDED]:
   - **Status:** Scientifically invalid pre-repair run.
   - **Defect:** In this initial exploratory implementation of Condition C, coordinating agents used an MD5 hash of the round index to determine their moves. Because the hash was deterministic and did not condition on or share private stochastic signals, it produced zero empirical mutual information between the designated agents. The positive control failed to establish true action dependence.
   - **Resolution:** Completely superseded by the stochastic private common signal architecture ($Z_t \sim \text{Uniform}(\{R, P, S\})$) and joint coordination gate ($G_t \sim \text{Bernoulli}(\alpha)$).

2. **`results_corrected_main_20261023_222944.json`** [SUPERSEDED]:
   - **Status:** Intermediate development run.
   - **Defect:** Repaired the stochastic private signal $Z_t$, but Condition D implemented per-agent independent compliance draws $C_{i,t} \sim \text{Bernoulli}(\alpha)$ rather than a joint coordination gate. As a consequence, both designated agents followed $Z_t$ with probability $\alpha^2$ rather than the reported probability $\alpha$.
   - **Resolution:** Replaced with the joint coordination gate implementation where $G_t \sim \text{Bernoulli}(\alpha)$ governs joint reception.

3. **`results_clean_main_20261023_230404.json`** [SUPERSEDED]:
   - **Status:** Intermediate development run.
   - **Defect:** Used preliminary two-detector reporting (Marginal MI vs PNI) without an explicit Conditional Mutual Information (CMI) baseline to separate state conditioning from rival subtraction.
   - **Resolution:** Extended to complete three-detector triangulation (Marginal MI, CMI, and PNI net score).

4. **`results_scripted_main_20261023_235252.json`** [SUPERSEDED]:
   - **Status:** Uncurated raw dump identical in underlying telemetry to `final_metrics.json`.
   - **Resolution:** Standardized and finalized as `final_metrics.json`.

### Historical Tables (`archive/historical_tables/`)
- `summary_scripted_main_20261023_203238.csv` (corresponds to run 203238)
- `summary_corrected_main_20261023_222944.csv` (corresponds to run 222944)
- `summary_clean_main_20261023_230404.csv` (corresponds to run 230404)
- `summary_scripted_main_20261023_235252.csv` (corresponds to run 235252)

---

## 2. Canonical Active Benchmark
The definitive, verified benchmark outputs are:
- Metrics: `results/metrics/final_metrics.json`
- Sensitivity: `results/metrics/sensitivity_trajectory_length.json`
- Summary Table: `results/tables/summary.csv`
- Trajectory Length Table: `results/tables/sensitivity_trajectory_length.csv`
