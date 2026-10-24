"""
scripts/plot_setup_diagram.py
=============================
Generates Figure 0: Architectural diagram of the 4 experimental conditions
and the information visibility separating agents, state, detectors, and auditor.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def draw_diagram():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)

    # Environment State Box
    box_state = patches.FancyBboxPatch((35, 46), 30, 10, boxstyle="round,pad=0.5",
                                       fc="#E3F2FD", ec="#1976D2", lw=2)
    ax.add_patch(box_state)
    ax.text(50, 51, "Observable Game State $S_t$\n(Prior Actions: $a_1^{t-1}, a_2^{t-1}, a_3^{t-1}, a_4^{t-1}$)",
            ha="center", va="center", fontsize=10, weight="bold", color="#0D47A1")

    # Conditions Grid
    conditions = [
        ("Cond A: Independent", "Uniform random\nIsolated RNGs\nNo state dep.", 8, "#E8F5E9", "#388E3C"),
        ("Cond B: Context Confound", "Indep. best response\nGiven $S_t$ with $\\epsilon=0.1$\nSpurious MI!", 32, "#FFF3E0", "#F57C00"),
        ("Cond C: Coordination", "Agents (0,1) share\na = hash(S) mod 3\nTrue colluders", 56, "#FCE4EC", "#C2185B"),
        ("Cond D: Strength Sweep", "Pair (0,1) coord. w/ prob alpha\nalpha in [0.25, 1.00]\nBoundary test", 80, "#EDE7F6", "#512DA8"),
    ]

    for title, desc, x, bg, border in conditions:
        box = patches.FancyBboxPatch((x, 26), 18, 14, boxstyle="round,pad=0.3",
                                     fc=bg, ec=border, lw=1.5)
        ax.add_patch(box)
        ax.text(x + 9, 36.5, title, ha="center", va="center", fontsize=8.5, weight="bold", color=border)
        ax.text(x + 9, 31, desc, ha="center", va="center", fontsize=7.5, color="#212121")
        # Arrow from state to condition
        ax.annotate("", xy=(x + 9, 40), xytext=(50, 46),
                    arrowprops=dict(arrowstyle="->", color="#90CAF9", lw=1.2, ls="--"))

    # Trajectory Action Log Box
    box_traj = patches.FancyBboxPatch((20, 14), 60, 7, boxstyle="round,pad=0.3",
                                      fc="#ECEFF1", ec="#455A64", lw=1.5)
    ax.add_patch(box_traj)
    ax.text(50, 17.5, "Trajectory Stream: Round Actions and Observable Context S_t\n[Auditor-Visible Logs Only — No Ground-Truth Labels]",
            ha="center", va="center", fontsize=8.5, weight="bold", color="#263238")

    # Detectors
    box_pni = patches.FancyBboxPatch((15, 1), 32, 9, boxstyle="round,pad=0.3",
                                     fc="#E1F5FE", ec="#0288D1", lw=1.8)
    ax.add_patch(box_pni)
    ax.text(31, 5.5, "Pairwise Net Influence (PNI)\n$\\Gamma(i \\to j) = I(A_i; A_j \\mid S) - \\max_{k} I(A_k; A_j \\mid S)$\nState-Conditioned (Protects against Context)",
            ha="center", va="center", fontsize=7.8, weight="bold", color="#01579B")

    box_mi = patches.FancyBboxPatch((53, 1), 32, 9, boxstyle="round,pad=0.3",
                                    fc="#FFF8E1", ec="#FFA000", lw=1.8)
    ax.add_patch(box_mi)
    ax.text(69, 5.5, "Marginal MI Baseline\n$I(A_i; A_j) = H(A_j) - H(A_j \\mid A_i)$\nUnconditioned (Vulnerable to Confounding)",
            ha="center", va="center", fontsize=7.8, weight="bold", color="#FF6F00")

    # Connect Trajectory to Detectors
    ax.annotate("", xy=(31, 10), xytext=(40, 14),
                arrowprops=dict(arrowstyle="->", color="#0288D1", lw=1.5))
    ax.annotate("", xy=(69, 10), xytext=(60, 14),
                arrowprops=dict(arrowstyle="->", color="#FFA000", lw=1.5))

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig0_experimental_setup.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    draw_diagram()
