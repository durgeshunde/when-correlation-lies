"""
scripts/plot_llm_results.py
===========================
Generates publication-quality figure visualizing LLM horizon scaling
and detector performance across T=5, T=20, and T=50 rounds.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def plot_llm_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.2), dpi=300)

    # ----------------------------------------------------
    # Panel A: Top-1 Accuracy vs Horizon T in Condition LLM-C
    # ----------------------------------------------------
    horizons = [5, 20, 50]
    top1_marg_mi = [1.0, 1.0, 1.0]
    top1_cmi = [0.700, 1.0, 1.0]
    top1_pni = [0.700, 1.0, 1.0]

    ax1.plot(horizons, top1_marg_mi, 'o--', color='#1565C0', label='Marginal MI', linewidth=1.8, markersize=6)
    ax1.plot(horizons, top1_cmi, 's-', color='#2E7D32', label='Conditional MI (CMI)', linewidth=2.0, markersize=6)
    ax1.plot(horizons, top1_pni, '^:', color='#C62828', label='PNI Net Score', linewidth=2.0, markersize=7)

    ax1.axhline(1/6, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='Random Guess (1/6)')
    ax1.set_xlabel('Horizon Length T (Rounds per Game)', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Top-1 Identification Accuracy', fontsize=9, fontweight='bold')
    ax1.set_title('A: LLM-C Coordinating Pair Ranking vs. T', fontsize=9.5, fontweight='bold')
    ax1.set_ylim([0.0, 1.08])
    ax1.set_xticks(horizons)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=7.5, framealpha=0.9)

    # Add annotation for horizon sparsity effect
    ax1.annotate('Sample Sparsity\nHorizon Effect\n(70.0% Top-1)', xy=(5, 0.700), xytext=(8, 0.42),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=4),
                 fontsize=7.5, fontweight='semibold')
    ax1.annotate('100% Recovery\n(100/100 & 70/70)', xy=(20, 1.0), xytext=(24, 0.82),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=4),
                 fontsize=7.5, fontweight='semibold')

    # ----------------------------------------------------
    # Panel B: Mean CMI (bits) Across Conditions vs Horizon
    # ----------------------------------------------------
    cmi_llm_a = [0.0693, 0.0997, 0.0835]
    cmi_llm_b = [0.0666, 0.0981, 0.0833]
    cmi_llm_c = [0.0696, 0.1333, 0.1770]

    ax2.plot(horizons, cmi_llm_a, 'o--', color='#757575', label='LLM-A (Independent)', linewidth=1.5, markersize=5)
    ax2.plot(horizons, cmi_llm_b, 'd--', color='#E65100', label='LLM-B (Shared Context)', linewidth=1.5, markersize=5)
    ax2.plot(horizons, cmi_llm_c, 's-', color='#2E7D32', label='LLM-C (Coordinated Pair)', linewidth=2.2, markersize=6)

    ax2.set_xlabel('Horizon Length T (Rounds per Game)', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Mean CMI Score (bits)', fontsize=9, fontweight='bold')
    ax2.set_title('B: Empirical CMI Separation vs. Horizon T', fontsize=9.5, fontweight='bold')
    ax2.set_ylim([0.04, 0.20])
    ax2.set_xticks(horizons)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    output_path = FIGURES_DIR / "fig6_llm_horizon_scaling.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    plot_llm_scaling()
