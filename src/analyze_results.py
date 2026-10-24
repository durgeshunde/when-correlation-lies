"""
analyze_results.py
==================
Post-hoc analysis and plotting from saved experiment results.

Usage:
    python -m src.analyze_results
    python -m src.analyze_results --results-dir results/metrics --output-dir results/figures
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CHANCE_TOP1 = 1.0 / 6.0  # 6 unordered pairs
FIGSIZE = (8, 5)
DPI = 150


def load_results(metrics_dir: Path) -> dict:
    """Load the most recent final_metrics.json."""
    final = metrics_dir / "final_metrics.json"
    if not final.exists():
        # Try most recent
        candidates = sorted(metrics_dir.glob("results_*.json"), reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No results found in {metrics_dir}")
        final = candidates[0]
    with open(final) as f:
        return json.load(f)


def _safe_stat(agg: dict, cond_key: str, metric: str, stat: str = "mean") -> float:
    """Safely extract a stat from aggregated results."""
    try:
        return agg[cond_key][metric][stat]
    except (KeyError, TypeError):
        return np.nan


def plot_top1_comparison(output: dict, figures_dir: Path):
    """Figure 1: Top-1 accuracy comparison across conditions."""
    agg = output["aggregated"]
    chance = output["config"]["chance_top1"]
    
    conditions = ["C"]  # conditions with ground truth
    alphas = output["config"].get("alphas") or []
    if alphas:
        conditions += [f"D_alpha{a:.2f}" for a in alphas if f"D_alpha{a:.2f}" in agg]
    
    pni_means = [_safe_stat(agg, c, "pni_top1") for c in conditions]
    pni_lo = [_safe_stat(agg, c, "pni_top1", "ci_low") for c in conditions]
    pni_hi = [_safe_stat(agg, c, "pni_top1", "ci_high") for c in conditions]
    mi_means = [_safe_stat(agg, c, "marginal_mi_top1") for c in conditions]
    mi_lo = [_safe_stat(agg, c, "marginal_mi_top1", "ci_low") for c in conditions]
    mi_hi = [_safe_stat(agg, c, "marginal_mi_top1", "ci_high") for c in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    bars1 = ax.bar(x - width/2, pni_means, width, label="PNI (Bonjour et al.)",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width/2, mi_means, width, label="Marginal MI (baseline)",
                   color="#FF9800", alpha=0.85)
    
    # Error bars for CI
    for i, (lo, hi) in enumerate(zip(pni_lo, pni_hi)):
        if not (np.isnan(lo) or np.isnan(hi)):
            mid = pni_means[i]
            ax.errorbar(x[i] - width/2, mid, yerr=[[mid - lo], [hi - mid]],
                        fmt="none", color="black", capsize=4, lw=1.5)
    for i, (lo, hi) in enumerate(zip(mi_lo, mi_hi)):
        if not (np.isnan(lo) or np.isnan(hi)):
            mid = mi_means[i]
            ax.errorbar(x[i] + width/2, mid, yerr=[[mid - lo], [hi - mid]],
                        fmt="none", color="black", capsize=4, lw=1.5)
    
    ax.axhline(chance, color="red", linestyle="--", linewidth=1.5,
               label=f"Chance ({chance:.3f})")
    
    labels = ["Cond C\n(full coord.)"]
    if alphas:
        labels += [f"Cond D\nα={a:.2f}" for a in alphas if f"D_alpha{a:.2f}" in agg]
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Top-1 Accuracy (fraction of seeds)")
    ax.set_title("Detector Top-1 Accuracy: Identifying the Coordinating Pair\n"
                 f"(n={output['config']['n_seeds']} seeds, {output['config']['n_rounds']} rounds, 95% CI)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    path = figures_dir / "fig1_top1_accuracy.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_fpr_comparison(output: dict, figures_dir: Path):
    """Figure 2: False positive rates for conditions A, B (negative controls)."""
    agg = output["aggregated"]
    conditions_shown = []
    labels_shown = []
    
    if "A" in agg:
        conditions_shown.append("A")
        labels_shown.append("Cond A (Test)\n(indep., n=15)")
    if "B" in agg:
        conditions_shown.append("B")
        labels_shown.append("Cond B\n(context, n=20)")
    
    if not conditions_shown:
        print("No negative control conditions found for FPR plot.")
        return None
    
    pni_fpr = [_safe_stat(agg, c, "pni_fpr") for c in conditions_shown]
    pni_fpr_lo = [_safe_stat(agg, c, "pni_fpr", "ci_low") for c in conditions_shown]
    pni_fpr_hi = [_safe_stat(agg, c, "pni_fpr", "ci_high") for c in conditions_shown]
    cmi_fpr = [_safe_stat(agg, c, "cmi_fpr") for c in conditions_shown]
    cmi_fpr_lo = [_safe_stat(agg, c, "cmi_fpr", "ci_low") for c in conditions_shown]
    cmi_fpr_hi = [_safe_stat(agg, c, "cmi_fpr", "ci_high") for c in conditions_shown]
    mi_fpr = [_safe_stat(agg, c, "marginal_mi_fpr") for c in conditions_shown]
    mi_fpr_lo = [_safe_stat(agg, c, "marginal_mi_fpr", "ci_low") for c in conditions_shown]
    mi_fpr_hi = [_safe_stat(agg, c, "marginal_mi_fpr", "ci_high") for c in conditions_shown]
    
    x = np.arange(len(conditions_shown))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    ax.bar(x - width, pni_fpr, width, label="PNI (conditioned + rival sub)", color="#2196F3", alpha=0.85)
    ax.bar(x, cmi_fpr, width, label="Conditional MI (state-conditioned)", color="#9C27B0", alpha=0.85)
    ax.bar(x + width, mi_fpr, width, label="Marginal MI (unconditioned)", color="#FF9800", alpha=0.85)
    
    for i, (lo, hi) in enumerate(zip(pni_fpr_lo, pni_fpr_hi)):
        if not (np.isnan(lo) or np.isnan(hi)):
            mid = pni_fpr[i]
            ax.errorbar(x[i] - width, mid, yerr=[[mid - lo], [hi - mid]],
                        fmt="none", color="black", capsize=4, lw=1.5)
    for i, (lo, hi) in enumerate(zip(cmi_fpr_lo, cmi_fpr_hi)):
        if not (np.isnan(lo) or np.isnan(hi)):
            mid = cmi_fpr[i]
            ax.errorbar(x[i], mid, yerr=[[mid - lo], [hi - mid]],
                        fmt="none", color="black", capsize=4, lw=1.5)
    for i, (lo, hi) in enumerate(zip(mi_fpr_lo, mi_fpr_hi)):
        if not (np.isnan(lo) or np.isnan(hi)):
            mid = mi_fpr[i]
            ax.errorbar(x[i] + width, mid, yerr=[[mid - lo], [hi - mid]],
                        fmt="none", color="black", capsize=4, lw=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels_shown)
    ax.set_ylim(0, 1.05)
    cal_th = output.get("config", {}).get("calibrated_thresholds", {})
    if cal_th and "pni" in cal_th and "marginal_mi" in cal_th:
        th_p = cal_th.get('pni', 0.0)
        th_c = cal_th.get('cmi', 0.0)
        th_m = cal_th.get('marginal_mi', 0.0)
        th_str = f"calibrated τ (PNI={th_p:.4f}, CMI={th_c:.4f}, MI={th_m:.4f})"
    else:
        th_str = "threshold=0.0"
    ax.set_title("False Positive Rates on Negative Controls\n"
                 f"({th_str}, 95% CI)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    path = figures_dir / "fig2_false_positive_rates.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_coordination_strength_sweep(output: dict, figures_dir: Path):
    """Figure 3: Detection quality vs coordination strength (Condition D)."""
    agg = output["aggregated"]
    alphas = output["config"].get("alphas") or []
    
    d_keys = [f"D_alpha{a:.2f}" for a in alphas if f"D_alpha{a:.2f}" in agg]
    if not d_keys:
        print("No Condition D results for coordination strength plot.")
        return None
    
    alpha_vals = [float(k.split("alpha")[1]) for k in d_keys]
    
    pni_top1 = [_safe_stat(agg, k, "pni_top1") for k in d_keys]
    pni_top1_lo = [_safe_stat(agg, k, "pni_top1", "ci_low") for k in d_keys]
    pni_top1_hi = [_safe_stat(agg, k, "pni_top1", "ci_high") for k in d_keys]
    mi_top1 = [_safe_stat(agg, k, "marginal_mi_top1") for k in d_keys]
    
    chance = output["config"]["chance_top1"]
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    ax.plot(alpha_vals, pni_top1, "o-", color="#2196F3", label="PNI detector", linewidth=2)
    ax.fill_between(alpha_vals, pni_top1_lo, pni_top1_hi, color="#2196F3", alpha=0.2,
                    label="PNI 95% CI")
    ax.plot(alpha_vals, mi_top1, "s--", color="#FF9800", label="Marginal MI", linewidth=2)
    ax.axhline(chance, color="red", linestyle=":", linewidth=1.5,
               label=f"Chance ({chance:.3f})")
    
    ax.set_xlabel("Coordination Strength α\n(probability of following coordination policy)")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("Detection Accuracy vs Coordination Strength\n"
                 f"(Condition D, n={output['config']['n_seeds']} seeds, {output['config']['n_rounds']} rounds)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = figures_dir / "fig3_coordination_strength.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_score_distributions(output: dict, figures_dir: Path):
    """Figure 4: Distribution of true-pair scores per condition."""
    per_run = output.get("per_run_results", [])
    if not per_run:
        return None
    
    conditions_to_plot = {}
    for r in per_run:
        if "error" in r:
            continue
        cond = r.get("condition", "?")
        alpha = r.get("alpha")
        if cond == "D" and alpha is not None:
            key = f"D_α={alpha:.2f}"
        else:
            key = cond
        
        pni_score = r.get("pni_true_score")
        if pni_score is not None and not np.isnan(pni_score):
            conditions_to_plot.setdefault(key, []).append(pni_score)
    
    if not conditions_to_plot:
        return None
    
    fig, ax = plt.subplots(figsize=(max(6, len(conditions_to_plot) * 1.2), 5))
    
    sorted_keys = sorted(conditions_to_plot.keys())
    data = [conditions_to_plot[k] for k in sorted_keys]
    
    bp = ax.boxplot(data, tick_labels=sorted_keys, patch_artist=True, notch=False)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    
    ax.axhline(0, color="red", linestyle="--", linewidth=1.0, label="Threshold=0")
    ax.set_ylabel("PNI Score of True Pair\n(min of Γ(i→j), Γ(j→i))")
    ax.set_title("Distribution of PNI Score for True Coordinating Pair\n"
                 "(positive score → detector ranks this pair above competitors)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    path = figures_dir / "fig4_score_distributions.png"
    plt.savefig(path, dpi=DPI)
    plt.close()
    print(f"Saved: {path}")
    return path


def write_summary_csv(output: dict, tables_dir: Path):
    """Write a clean summary CSV of aggregated metrics."""
    import csv
    
    agg = output["aggregated"]
    rows = []
    
    for cond_key, metrics in sorted(agg.items()):
        row = {"condition": cond_key}
        SKIP_METRICS = {"seed", "n_rounds", "alpha", "has_true_pair"}
        for metric_name, stats in sorted(metrics.items()):
            if metric_name in SKIP_METRICS:
                continue
            if isinstance(stats, dict):
                row[f"{metric_name}_mean"] = _fmt(stats.get("mean"))
                row[f"{metric_name}_ci_low"] = _fmt(stats.get("ci_low"))
                row[f"{metric_name}_ci_high"] = _fmt(stats.get("ci_high"))
                row[f"{metric_name}_n"] = stats.get("n", "")
        rows.append(row)
    
    if not rows:
        return None
    
    path = tables_dir / "summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")
    return path


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.4f}"


def write_results_interpretation(output: dict, docs_dir: Path):
    """Write a machine-generated results interpretation."""
    agg = output["aggregated"]
    config = output["config"]
    
    chance = config["chance_top1"]
    
    lines = [
        "# Results Interpretation",
        "",
        f"**Experiment:** {config['n_seeds']} seeds × {config['n_rounds']} rounds per trajectory",
        f"**Chance top-1 accuracy:** {chance:.3f} (1/6 pairs)",
        f"**Coordinated pair:** agents {config['coordinated_pair']}",
        "",
        "## Condition A (Independent Baseline)",
    ]
    
    if "A" in agg:
        pni_fpr_a = _safe_stat(agg, "A", "pni_fpr")
        mi_fpr_a = _safe_stat(agg, "A", "marginal_mi_fpr")
        lines += [
            f"- PNI false positive rate: {pni_fpr_a:.3f}",
            f"- Marginal MI false positive rate: {mi_fpr_a:.3f}",
            "- Interpretation: Under fully independent behavior, a well-calibrated",
            "  detector should have a low false positive rate.",
        ]
    
    lines += ["", "## Condition B (Shared-Context Negative Control)"]
    
    if "B" in agg:
        pni_fpr_b = _safe_stat(agg, "B", "pni_fpr")
        mi_fpr_b = _safe_stat(agg, "B", "marginal_mi_fpr")
        lines += [
            f"- PNI false positive rate: {pni_fpr_b:.3f}",
            f"- Marginal MI false positive rate: {mi_fpr_b:.3f}",
            "- H2 test: Marginal MI FPR > PNI FPR would support H2 (shared context",
            "  confounds unconditional detectors but not conditional ones).",
        ]
    
    lines += ["", "## Condition C (Known-Coordination Positive Control)"]
    
    if "C" in agg:
        pni_top1 = _safe_stat(agg, "C", "pni_top1")
        pni_top1_lo = _safe_stat(agg, "C", "pni_top1", "ci_low")
        pni_top1_hi = _safe_stat(agg, "C", "pni_top1", "ci_high")
        mi_top1 = _safe_stat(agg, "C", "marginal_mi_top1")
        lines += [
            f"- PNI top-1 accuracy: {pni_top1:.3f} (95% CI: [{pni_top1_lo:.3f}, {pni_top1_hi:.3f}])",
            f"- Marginal MI top-1 accuracy: {mi_top1:.3f}",
            f"- Chance baseline: {chance:.3f}",
            f"- H1 support: {'Supported' if pni_top1_lo > chance else 'Not clearly supported'}",
            "  (H1 requires lower CI bound above chance).",
        ]
    
    lines += ["", "## Key Limitations"]
    lines += [
        "- Small coordinating pair; only one pair tested.",
        "- RPS is simpler than real economic games.",
        "- Results apply to scripted agents with controlled coordination mechanisms.",
        "- The coordination mechanism (shared deterministic hash function) is artificial.",
        "- No LLM agents were tested in this run (scripted experiment only).",
        "- Short trajectories may underestimate detection performance.",
    ]
    
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "results_interpretation.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results and generate figures")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "metrics")
    parser.add_argument("--output-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    output = load_results(args.results_dir)
    
    print("\nGenerating figures...")
    plot_top1_comparison(output, args.output_dir)
    plot_fpr_comparison(output, args.output_dir)
    plot_coordination_strength_sweep(output, args.output_dir)
    plot_score_distributions(output, args.output_dir)
    
    write_summary_csv(output, TABLES_DIR)
    write_results_interpretation(output, PROJECT_ROOT / "docs")
    
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
