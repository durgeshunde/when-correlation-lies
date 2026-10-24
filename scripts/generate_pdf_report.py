"""
scripts/generate_pdf_report.py
==============================
Builds the formal academic PDF research report for the Apart Research
AI Collusion Research Sprint 2026 submission using ReportLab Platypus.

Ensures strict compliance with official template and event guidelines:
- Title: When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents
- Track 1.4: Markets and Collusion
- Author: Durgesh Unde, Affiliation: Independent Researcher
- Event: Apart Research AI Collusion Research Sprint (October 23–25, 2026)
- Abstract: exactly within 150-250 words
- Hard cap: 8 pages; Target: exactly 5 pages
- Clean page breaks: zero split tables or orphaned headers
- No "Confidential" header/footer
- Accurate citations (Agrawal et al., 2025 arXiv:2507.01413; Bonjour et al., 2022)
- Required Appendices: Limitations and Dual-Use / Ethical Considerations, LLM Usage Statement
- Embedded PDF document metadata: Author, Title, Subject
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen import canvas

REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
PDF_PATH = REPORT_DIR / "when_correlation_lies_report.pdf"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' and set PDF metadata."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        
        # Embed proper document metadata
        self.setAuthor("Durgesh Unde")
        self.setTitle("When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents")
        self.setSubject("Apart Research AI Collusion Research Sprint 2026 - Track 1.4: Markets and Collusion")
        self.setCreator("ReportLab Platypus / Python 3.13")
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Running Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(50, 11 * inch - 36, "When Correlation Lies: Stress-Testing Pairwise Net Influence")
            self.drawRightString(8.5 * inch - 50, 11 * inch - 36, "Apart Research Sprint 2026 — Track 1.4")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(50, 11 * inch - 40, 8.5 * inch - 50, 11 * inch - 40)
        
        # Running Footer
        footer_text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * inch - 50, 30, footer_text)
        self.drawString(50, 30, "Apart Research AI Collusion Research Sprint 2026 — Track 1.4: Markets and Collusion")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(50, 40, 8.5 * inch - 50, 40)
        self.restoreState()


def build_pdf():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=50,
        rightMargin=50,
        topMargin=44,
        bottomMargin=44,
    )
    doc.author = "Durgesh Unde"
    doc.title = "When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents"
    doc.subject = "Apart Research AI Collusion Research Sprint 2026 - Track 1.4: Markets and Collusion"

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14.5,
        leading=18,
        textColor=colors.HexColor("#0D47A1"),
        alignment=1,
        spaceAfter=3,
    )

    track_style = ParagraphStyle(
        "DocTrack",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.0,
        leading=12.0,
        textColor=colors.HexColor("#1976D2"),
        alignment=1,
        spaceAfter=3,
    )

    author_style = ParagraphStyle(
        "DocAuthors",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#263238"),
        alignment=1,
        spaceAfter=6,
    )

    abstract_heading = ParagraphStyle(
        "AbstractHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10.5,
        textColor=colors.HexColor("#0D47A1"),
        spaceAfter=2,
    )

    abstract_body = ParagraphStyle(
        "AbstractBody",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10.0,
        textColor=colors.HexColor("#263238"),
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.3,
        leading=11.8,
        textColor=colors.HexColor("#0D47A1"),
        spaceBefore=4,
        spaceAfter=1.5,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#1976D2"),
        spaceBefore=3,
        spaceAfter=1.2,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.3,
        leading=9.3,
        textColor=colors.HexColor("#212121"),
        spaceAfter=1.8,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.3,
        leading=9.3,
        textColor=colors.HexColor("#212121"),
        leftIndent=8,
        spaceAfter=1.0,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=5.8,
        leading=7.2,
        alignment=1,
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=5.8,
        leading=7.2,
        textColor=colors.white,
        alignment=1,
    )

    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=6.4,
        leading=8.0,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceAfter=2.5,
    )

    story = []

    # =========================================================================
    # PAGE 1: Title, Author, Abstract, Section 1 (Intro & Hypotheses), Figure 1
    # =========================================================================
    story.append(Paragraph("When Correlation Lies: Stress-Testing Pairwise Net Influence for Detecting Coordination Among Agents<sup>1</sup>", title_style))
    story.append(Paragraph("Track 1.4: Markets and Collusion &nbsp;|&nbsp; Apart Research AI Collusion Research Sprint (October 23–25, 2026)", track_style))
    story.append(Paragraph("<b>Author:</b> Durgesh Unde &nbsp;|&nbsp; <b>Affiliation:</b> Independent Researcher<br/><b>Verified Pipeline:</b> Python 3.13 / CPU Benchmarks (0 API Calls, $0 Spend) &nbsp;|&nbsp; <b>Evaluations:</b> 260 unique trajectories (54,000 rounds)", author_style))

    abstract_text = (
        "As autonomous AI agents are increasingly deployed in high-frequency trading, automated procurement, and "
        "algorithmic pricing, covert collusion poses severe economic and systemic risks. Information-theoretic detectors, "
        "notably Pairwise Net Influence (PNI; Bonjour et al., 2022), measure directed action dependence via Conditional "
        "Mutual Information (CMI) conditioned on observable public game state. However, whether PNI can distinguish deliberate "
        "multi-agent coordination from spurious correlation induced by shared context remains an open empirical question. "
        "In this work, we stress-test PNI against both unconditioned marginal mutual information and raw conditional mutual information "
        "baselines in an iterated 4-agent game across four controlled conditions: independent play, shared-context confounding, "
        "a shared-private-signal coordination control, and coordination-strength sweeps with a joint Bernoulli gate. "
        "Across 20 independent seeds and trajectory length sweeps (T in [50, 500]), unconditioned mutual information exhibits a "
        "38-fold surge under shared context, falsely accusing independent agents of collusion (FPR = 1.000). Triangulating our estimators "
        "reveals that conditioning on observable state eliminates this contextual surge (CMI FPR = 0.000), while PNI's competitor subtraction "
        "suppresses residual background, maintaining low false positives (FPR = 0.100) and achieving 100% Top-1 identification accuracy "
        "(Wilson 95% CI: [0.839, 1.000], MRR = 1.000) under controlled coordination. Our findings demonstrate that conditioning on observable "
        "history is indispensable for behavioral auditing, though finite-sample state sparsity introduces residual baseline uncertainty."
    )
    abstract_content = [
        [Paragraph("<b>ABSTRACT</b>", abstract_heading)],
        [Paragraph(abstract_text, abstract_body)]
    ]
    t_abstract = Table(abstract_content, colWidths=[7.1 * inch])
    t_abstract.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#B0BEC5")),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
    ]))
    story.append(t_abstract)
    story.append(Spacer(1, 2))

    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Autonomous algorithmic agents increasingly govern financial exchanges, automated multi-unit auctions, and dynamic e-commerce pricing. "
        "A critical frontier in AI safety is preventing <b>algorithmic collusion</b>—scenarios where autonomous agents synchronize strategies to extract "
        "supracompetitive rents at the expense of human market participants (Calvano et al., 2020; Schwalbe, 2018). Unlike traditional cartels that rely on "
        "explicit communication logs, AI agents can establish implicit coordination purely through tacit observational learning or private coordination signals. "
        "Consequently, behavioral auditing tools that operate solely on external observational logs are essential for oversight.",
        body_style
    ))
    story.append(Paragraph(
        "In Track 1.4 (Markets and Collusion) of the AI Collusion Research Sprint, we address the core research question: "
        "<i>Can information-theoretic approaches detect collusion in multi-agent games?</i> "
        "Specifically, we stress-test <b>Pairwise Net Influence (PNI)</b> (Bonjour, Aggarwal, & Bhargava, UAI 2022). While PNI is theoretically sound, "
        "real-world market auditing confronts a severe confounder: <b>shared-context confounding</b>. When non-coordinating agents independently "
        "react to identical market histories, price shocks, or public order books, their actions naturally synchronize. "
        "If auditing metrics mistake this context-driven synchronization for collusion, benign market makers will face false enforcement sanctions.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Core Research Question:</b> <i>How reliably does Pairwise Net Influence distinguish deliberately coordinated behavior from "
        "independent behavior that merely appears correlated due to shared context, and how does detector performance scale with coordination "
        "strength and trajectory length?</i>",
        body_style
    ))
    story.append(Paragraph(
        "To rigorously separate the effects of state conditioning from rival competitor subtraction, we formulate and evaluate four hypotheses "
        "across three distinct estimators—Marginal MI I(A_i; A_j), Conditional MI gamma(i, j) = I(A_i; A_j | S), and PNI net score—using a frozen "
        "benchmark of 260 unique trajectories (54,000 unique recorded rounds across 300 evaluations):",
        body_style
    ))
    story.append(Paragraph("• <b>H1 (Positive Sensitivity):</b> Under controlled shared-private-signal coordination, PNI ranks the coordinating pair #1 significantly above the 1/6 (16.7%) random baseline.", bullet_style))
    story.append(Paragraph("• <b>H2 (Shared-Context Confounding):</b> Observable public state induces severe spurious action correlation, causing unconditioned marginal mutual information to suffer a 100% false positive rate.", bullet_style))
    story.append(Paragraph("• <b>H3 (State-Conditioning Neutralization & Rival Subtraction):</b> Conditioning on observable state S neutralizes this broadcast shock below independent calibration thresholds, while PNI rival subtraction further isolates pair-specific influence.", bullet_style))
    story.append(Paragraph("• <b>H4 (Sensitivity Scaling):</b> PNI identification scores degrade predictably and monotonically as joint coordination probability alpha decreases towards independent play.", bullet_style))
    story.append(Spacer(1, 2))

    fig0_path = FIGURES_DIR / "fig0_experimental_setup.png"
    if fig0_path.exists():
        story.append(Image(str(fig0_path), width=7.0 * inch, height=1.75 * inch))
        story.append(Paragraph("<b>Figure 1: Experimental Architecture and Triangulated Estimator Pipeline.</b> Agents act under one of four controlled conditions (A–D). Public history S_t is visible to agents and the auditor; private coordination signal Z_t and joint gate G_t are strictly restricted to colluders. The auditor computes three distinct statistics: Marginal MI, Conditional MI, and PNI Net Score.", caption_style))

    # =========================================================================
    # PAGE 2: Related Work and Methods
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("2. Related Work", h1_style))
    story.append(Paragraph(
        "<b>Information-Theoretic Multi-Agent Influence:</b> Information theory offers a non-parametric foundation for quantifying multi-agent dependence "
        "without restrictive linear or parametric assumptions (Cover & Thomas, 2006). Bonjour, Aggarwal, and Bhargava (UAI 2022) formulated individual "
        "influence in simultaneous games using Conditional Mutual Information (CMI) conditioned on observable game state: "
        "gamma(i, j) = I(A_i; A_j | S) = H(A_j | S) - H(A_j | A_i, S). In simultaneous games, raw CMI is symmetric. To isolate pair-specific "
        "coordination from symmetric broadcast shocks, Bonjour et al. introduced <i>directed net influence</i> by subtracting the strongest competing "
        "rival influence: Gamma(i -&gt; j) = gamma(i, j) - max_{k != i, j} gamma(k, j). A candidate pair is assigned the score: "
        "Score_PNI(i, j) = min(Gamma(i -&gt; j), Gamma(j -&gt; i)). While Bonjour et al. established theoretical bounds in matrix games, they did not "
        "evaluate empirical false-positive inflation under shared-context confounding, separate raw CMI from rival subtraction, or calibrate decision thresholds on negative controls.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Algorithmic and LLM Collusion in Economics:</b> Extensive literature demonstrates that Q-learning and reinforcement learning agents can autonomously "
        "sustain collusive pricing above competitive Nash equilibria without explicit communication (Calvano et al., 2020; Schwalbe, 2018). Recently, "
        "Agrawal et al. (2025; arXiv:2507.01413) extended collusion auditing to LLM agents participating in double auctions, establishing that language model "
        "agents can form tacit cartels through contextual prompting. However, current LLM collusion studies evaluate aggregate market distortion (e.g., supracompetitive "
        "bid-ask spreads) rather than information-theoretic ranking detectors capable of identifying the specific colluding coalition.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Finite-Sample Entropy Estimation & Regularization:</b> Discrete plug-in estimators of entropy and mutual information natively evaluate unvisited cells "
        "via the standard mathematical convention 0 log 0 = 0. Smoothing (e.g., Bayesian Dirichlet smoothing with Jeffreys pseudocounts lambda = 0.5; Holste et al., 1998) "
        "is therefore not mathematically required to prevent division by zero. Rather, smoothing serves as a regularized estimation choice that stabilizes probability mass "
        "across sparsely sampled state-action cells in finite sample regimes (Paninski, 2003). We explicitly evaluate the sensitivity of our empirical findings across smoothing parameter lambda in {0.0, 0.5, 1.0}.",
        body_style
    ))
    story.append(Spacer(1, 2))

    story.append(Paragraph("3. Methods", h1_style))
    story.append(Paragraph(
        "<b>Iterated Multi-Agent Game Environment:</b> Following Bonjour et al. (2022) and sprint starter guidelines, we implement a 4-player iterated "
        "Rock-Paper-Scissors (RPS) tournament. The action space is A = {Rock: 0, Paper: 1, Scissors: 2}. In round t, each agent i executes simultaneous "
        "action a_i^t. The round payoff is the tournament sum of pairwise zero-sum payoffs: pi_i(a^t) = sum_{j != i} U(a_i^t, a_j^t), where U in {-1, 0, 1}. "
        "The public observable state S_t is the joint action profile of round t-1: S_t = (a_0^{t-1}, a_1^{t-1}, a_2^{t-1}, a_3^{t-1}) in A^4, yielding "
        "|S| = 3^4 + 1 = 82 observable states (including initial START state).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Controlled Experimental Conditions:</b> To isolate causal confounding mechanisms, we evaluate four strictly partitioned conditions (Figure 1):<br/>"
        "• <b>Condition A (Independent Negative Control):</b> 4 uniform-random agents with isolated RNGs. Zero coordination or state dependence. "
        "Seeds 0–4 are reserved strictly for threshold calibration; seeds 5–19 (n=15) are strictly held-out test evaluations.<br/>"
        "• <b>Condition B (Shared-Context Confounding Negative Control):</b> Agents independently execute a best response to the dominant action in S_t "
        "with 10% independent noise. Agents are conditionally independent given S_t, but unconditionally correlated due to the shared state (n=20).<br/>"
        "• <b>Condition C (Shared-Private-Signal Coordination Positive Control):</b> Agents 0 and 1 receive a private common signal Z_t ~ Uniform({0, 1, 2}), "
        "executing identical coordinated actions. Agents 2 and 3 play independent best responses. This establishes a controlled statistical signature of conditional "
        "action dependence in scripted RPS, though it does not by itself prove a real-world economic cartel (n=20).<br/>"
        "• <b>Condition D (Coordination Strength Sweep):</b> In each round, a joint coordination gate G_t ~ Bernoulli(alpha) is drawn. If G_t = 1, both agents 0 and 1 "
        "follow private signal Z_t; if G_t = 0, neither receives Z_t and both independently execute fallback best response. This guarantees the pair coordinates with probability alpha (n=20 per alpha in {0.25, 0.50, 0.75, 1.00}).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Triangulated Estimators & Calibrated Decision Thresholds:</b> To disentangle conditioning from competitor subtraction, we evaluate three statistics:<br/>"
        "1. <i>Marginal MI:</i> Score_MI(i, j) = I(A_i; A_j) = H(A_j) - H(A_j | A_i).<br/>"
        "2. <i>Conditional MI:</i> Score_CMI(i, j) = gamma(i, j) = I(A_i; A_j | S) = H(A_j | S) - H(A_j | A_i, S).<br/>"
        "3. <i>Pairwise Net Influence (PNI):</i> Score_PNI(i, j) = min(Gamma(i -&gt; j), Gamma(j -&gt; i)), where Gamma(i -&gt; j) = gamma(i, j) - max_{k != i, j} gamma(k, j).<br/>"
        "Probabilities are estimated with Dirichlet smoothing (lambda = 0.5). Decision thresholds are calibrated via exploratory small-sample calibration "
        "on held-out validation trajectories (seeds 0–4 of Condition A, 30 candidate-pair scores) targeting nominal 5% FPR at the empirical 95th percentile "
        "(tau_MI = 0.030751, tau_CMI = 0.117454, tau_PNI = 0.010427 bits). Because 30 candidate-pair scores provide coarse quantile estimation, "
        "this serves as an empirical baseline rather than guaranteeing exact 5% population FPR. Condition A test FPR is evaluated strictly on held-out seeds 5–19 (n=15, 90 candidate pairs).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Statistical Uncertainty Protocol & Degenerate Intervals:</b> All primary runs evaluate 20 independent pseudorandom seeds. For Top-1 accuracy "
        "(a bounded binomial proportion), we report exact <b>Wilson score confidence intervals</b> to prevent degenerate [1.0, 1.0] intervals. "
        "Continuous metrics (scores, FPR, MRR) report non-parametric bootstrap 95% confidence intervals over B = 2,000 resamples across independent trajectories. "
        "Whenever per-trajectory outcomes are completely identical across all 20 seeds (e.g., exactly 1 of 5 negative candidate pairs flagged in every seed), "
        "the resampled mean is constant, producing an interval of zero width (e.g., [0.200, 0.200]). We explicitly flag these degenerate intervals with asterisks "
        "and report the underlying empirical counts (flagged pairs / total negative pairs) to reflect that this represents uniform per-seed outcomes rather than zero population variance.",
        body_style
    ))

    # =========================================================================
    # PAGE 3: Results, Table 1, Figure 2, Hypothesis Confirmations
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("4. Results", h1_style))
    story.append(Paragraph(
        "Table 1 reports triangulated benchmark metrics across all experimental conditions for primary trajectories of T = 200 rounds. "
        "Condition A test metrics are computed strictly on held-out seeds 5–19 (n=15).",
        body_style
    ))

    headers1 = [
        "Condition", "GT\nPair", 
        "Marginal MI\nTop (95% CI)", "Marg. MI\nFPR [Counts]", 
        "Cond. MI (CMI)\nTop (95% CI)", "CMI FPR\n[Counts]", 
        "PNI Net\nTop (95% CI)", "PNI FPR\n[Counts]", 
        "PNI Top-1\n(Wilson CI)", "PNI\nMRR"
    ]
    data1 = [
        [Paragraph(h, table_header) for h in headers1],
        [
            Paragraph("<b>Cond A (Test, n=15)</b>", table_cell), Paragraph("None", table_cell),
            Paragraph("0.025 [0.020, 0.031]", table_cell), Paragraph("0.033 [0.000, 0.067]<br/>(3/90)", table_cell),
            Paragraph("0.115 [0.111, 0.119]", table_cell), Paragraph("0.111 [0.044, 0.189]<br/>(10/90)", table_cell),
            Paragraph("0.005 [0.003, 0.008]", table_cell), Paragraph("0.022 [0.000, 0.056]<br/>(2/90)", table_cell),
            Paragraph("—", table_cell), Paragraph("—", table_cell)
        ],
        [
            Paragraph("<b>Cond A (Cal., n=5)</b>", table_cell), Paragraph("None", table_cell),
            Paragraph("0.027 [0.019, 0.036]", table_cell), Paragraph("0.067 [0.000, 0.133]<br/>(2/30)", table_cell),
            Paragraph("0.115 [0.111, 0.118]", table_cell), Paragraph("0.067 [0.000, 0.133]<br/>(2/30)", table_cell),
            Paragraph("0.007 [0.003, 0.012]", table_cell), Paragraph("0.067 [0.000, 0.133]<br/>(2/30)", table_cell),
            Paragraph("—", table_cell), Paragraph("—", table_cell)
        ],
        [
            Paragraph("<b>Cond B (Context, n=20)</b>", table_cell), Paragraph("None", table_cell),
            Paragraph("<b>0.958 [0.932, 0.984]</b>", table_cell), Paragraph("<b>1.000 [1.000, 1.000]*</b><br/>(120/120)", table_cell),
            Paragraph("<b>0.084 [0.081, 0.087]</b>", table_cell), Paragraph("<b>0.000 [0.000, 0.000]*</b><br/>(0/120)", table_cell),
            Paragraph("<b>0.012 [0.009, 0.015]</b>", table_cell), Paragraph("<b>0.100 [0.050, 0.158]</b><br/>(12/120)", table_cell),
            Paragraph("—", table_cell), Paragraph("—", table_cell)
        ],
        [
            Paragraph("<b>Cond C (Coord., n=20)</b>", table_cell), Paragraph("(0, 1)", table_cell),
            Paragraph("1.451 [1.447, 1.454]", table_cell), Paragraph("0.240 [0.200, 0.320]<br/>(24/100)", table_cell),
            Paragraph("0.797 [0.785, 0.808]", table_cell), Paragraph("0.020 [0.000, 0.050]<br/>(2/100)", table_cell),
            Paragraph("0.755 [0.741, 0.769]", table_cell), Paragraph("0.200 [0.200, 0.200]*<br/>(20/100)", table_cell),
            Paragraph("<b>1.000 [0.839, 1.000]</b>", table_cell), Paragraph("<b>1.000</b>", table_cell)
        ],
        [
            Paragraph("<b>Cond D (a=0.25, n=20)</b>", table_cell), Paragraph("(0, 1)", table_cell),
            Paragraph("1.024 [0.988, 1.062]", table_cell), Paragraph("1.000 [1.000, 1.000]*<br/>(100/100)", table_cell),
            Paragraph("0.329 [0.306, 0.353]", table_cell), Paragraph("0.010 [0.000, 0.030]<br/>(1/100)", table_cell),
            Paragraph("0.264 [0.239, 0.288]", table_cell), Paragraph("0.100 [0.060, 0.140]<br/>(10/100)", table_cell),
            Paragraph("<b>1.000 [0.839, 1.000]</b>", table_cell), Paragraph("<b>1.000</b>", table_cell)
        ],
        [
            Paragraph("<b>Cond D (a=0.50, n=20)</b>", table_cell), Paragraph("(0, 1)", table_cell),
            Paragraph("1.131 [1.090, 1.174]", table_cell), Paragraph("1.000 [1.000, 1.000]*<br/>(100/100)", table_cell),
            Paragraph("0.524 [0.497, 0.552]", table_cell), Paragraph("0.000 [0.000, 0.000]*<br/>(0/100)", table_cell),
            Paragraph("0.469 [0.442, 0.498]", table_cell), Paragraph("0.180 [0.150, 0.200]<br/>(18/100)", table_cell),
            Paragraph("<b>1.000 [0.839, 1.000]</b>", table_cell), Paragraph("<b>1.000</b>", table_cell)
        ],
        [
            Paragraph("<b>Cond D (a=0.75, n=20)</b>", table_cell), Paragraph("(0, 1)", table_cell),
            Paragraph("1.285 [1.259, 1.310]", table_cell), Paragraph("0.920 [0.830, 0.990]<br/>(92/100)", table_cell),
            Paragraph("0.669 [0.650, 0.686]", table_cell), Paragraph("0.020 [0.000, 0.050]<br/>(2/100)", table_cell),
            Paragraph("0.623 [0.600, 0.642]", table_cell), Paragraph("0.200 [0.200, 0.200]*<br/>(20/100)", table_cell),
            Paragraph("<b>1.000 [0.839, 1.000]</b>", table_cell), Paragraph("<b>1.000</b>", table_cell)
        ],
        [
            Paragraph("<b>Cond D (a=1.00, n=20)</b>", table_cell), Paragraph("(0, 1)", table_cell),
            Paragraph("1.452 [1.449, 1.455]", table_cell), Paragraph("0.240 [0.200, 0.300]<br/>(24/100)", table_cell),
            Paragraph("0.813 [0.802, 0.824]", table_cell), Paragraph("0.010 [0.000, 0.030]<br/>(1/100)", table_cell),
            Paragraph("0.771 [0.757, 0.784]", table_cell), Paragraph("0.200 [0.200, 0.200]*<br/>(20/100)", table_cell),
            Paragraph("<b>1.000 [0.839, 1.000]</b>", table_cell), Paragraph("<b>1.000</b>", table_cell)
        ],
    ]
    t1 = Table(data1, colWidths=[0.98*inch, 0.35*inch, 0.80*inch, 0.66*inch, 0.80*inch, 0.66*inch, 0.80*inch, 0.66*inch, 0.82*inch, 0.35*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 1.0),
    ]))
    story.append(t1)
    story.append(Paragraph("<b>Table 1: Triangulated Benchmark Metrics.</b> Evaluated across 20 independent seeds per condition (T = 200 rounds; Condition A test n=15 held out). Brackets report 95% bootstrap CIs for continuous scores and exact Wilson CIs for Top-1 accuracy. Underlying counts report flagged pairs / total negative pairs. *Denotes degenerate bootstrap intervals resulting from uniform per-seed outcomes across all seeds (e.g. exactly 1/5 negative pairs flagged per seed in Cond C/D, or 0/6 in Cond B CMI) rather than zero population variance.", caption_style))
    story.append(Spacer(1, 1))

    fig1_path = FIGURES_DIR / "fig1_top1_accuracy.png"
    fig2_path = FIGURES_DIR / "fig2_false_positive_rates.png"
    if fig1_path.exists() and fig2_path.exists():
        t_figs = Table([
            [Image(str(fig1_path), width=3.45 * inch, height=1.52 * inch),
             Image(str(fig2_path), width=3.45 * inch, height=1.52 * inch)]
        ], colWidths=[3.55 * inch, 3.55 * inch])
        t_figs.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('PADDING', (0, 0), (-1, -1), 0)]))
        story.append(t_figs)
        story.append(Paragraph("<b>Figure 2: Primary Detector Benchmark.</b> Left: Top-1 Identification Accuracy across coordination conditions (dashed red line denotes chance = 0.167). Right: False Positive Rates comparing Marginal MI, Conditional MI, and PNI under calibrated decision thresholds (tau_MI = 0.0308, tau_CMI = 0.1175, tau_PNI = 0.0104 bits).", caption_style))
    story.append(Spacer(1, 1))

    story.append(Paragraph("4.1 Evaluation of Hypotheses and Key Discoveries", h2_style))
    story.append(Paragraph(
        "<b>Support for H1 (Top-1 Accuracy = 100% under Controlled Coordination):</b> Under Condition C, PNI achieves <b>100% Top-1 accuracy</b> (20/20 seeds, "
        "Wilson 95% CI: [0.839, 1.000]) and an MRR of 1.000. The true coordinating pair achieves an average net score of <b>0.755 bits</b> [0.741, 0.769], "
        "decisively separated from competing non-coordinating pairs.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Support for H2 (The 38-Fold Context Confounding Surge):</b> Under Condition B (independent agents reacting to shared context), naive Marginal MI "
        "explodes from 0.025 bits in Condition A to <b>0.958 bits</b> in Condition B—a <b>38-fold surge</b> in apparent action dependence—flagging 100% of "
        "non-coordinating pairs as colluders (FPR = 1.000 [1.000, 1.000], 120/120 pairs flagged). Unconditioned correlation is completely blind to shared context.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Support for H3 (Disentangling State Conditioning from Competitor Subtraction):</b> Reporting Marginal MI, raw CMI, and PNI separately isolates the two pipeline mechanisms. "
        "Conditioning on state S_t drops raw dependence from 0.958 bits to <b>0.084 bits</b> [0.081, 0.087], falling below tau_CMI = 0.1175 bits to yield <b>0.000 FPR</b> (0/120 pairs). "
        "Competitor subtraction computes Gamma(i -&gt; j) = gamma(i, j) - max_k gamma(k, j); because all 4 agents experience symmetric residual dependence (~0.084 bits), "
        "the difference is small (0.012 bits [0.009, 0.015]), yielding PNI FPR <b>0.100</b> [0.050, 0.158] (12/120 pairs) under tau_PNI = 0.0104 bits. "
        "State conditioning neutralizes the common shock, while competitor subtraction provides directed isolation.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Support for H4 (Monotonic Scaling under Joint Gate):</b> In Condition D, varying joint coordination probability alpha via gate G_t ~ Bernoulli(alpha) "
        "produces strictly monotonic scaling of true-pair net influence: 0.264 bits (alpha=0.25), 0.469 bits (alpha=0.50), 0.623 bits (alpha=0.75), and 0.771 bits (alpha=1.00). "
        "PNI maintains 100% Top-1 accuracy across all tested alpha in [0.25, 1.00], whereas Marginal MI suffers 100% FPR across alpha &lt;= 0.50.",
        body_style
    ))

    # =========================================================================
    # PAGE 4: Sensitivity Sweep, Table 2, Figure 3, Discussion & Limitations
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("4.2 Sensitivity Analysis: Trajectory Length", h1_style))
    story.append(Paragraph(
        "To evaluate finite-sample convergence, we executed a dedicated sweep across trajectory lengths T in {50, 100, 200, 500} rounds (Table 2). "
        "Evaluations used fixed primary calibrated thresholds (tau_MI = 0.030751, tau_CMI = 0.117454, tau_PNI = 0.010427 bits) across all lengths.",
        body_style
    ))

    headers2 = [
        "Length T", "Cond C PNI\nTop-1 (Wilson)", "Cond C PNI\nTrue Score (bits)", 
        "Cond B Marg MI\nScore (bits)", "Cond B CMI\nScore (bits)", "Cond B CMI\nFPR [Counts]", 
        "Cond B PNI\nNet Score (bits)", "Cond B PNI\nFPR [Counts]"
    ]
    data2 = [
        [
            Paragraph(h, table_header) for h in headers2],
        [
            Paragraph("<b>50 rounds</b>", table_cell), Paragraph("<b>1.00 [0.839, 1.000]</b>", table_cell),
            Paragraph("0.269 [0.250, 0.286]", table_cell), Paragraph("0.943 [0.884, 1.001]", table_cell),
            Paragraph("0.144 [0.134, 0.153]", table_cell), Paragraph("0.450 (54/120)", table_cell),
            Paragraph("0.0176 [0.0117, 0.0246]", table_cell), Paragraph("0.133 [0.083, 0.183]<br/>(16/120)", table_cell)
        ],
        [
            Paragraph("<b>100 rounds</b>", table_cell), Paragraph("<b>1.00 [0.839, 1.000]</b>", table_cell),
            Paragraph("0.499 [0.483, 0.515]", table_cell), Paragraph("0.964 [0.909, 1.021]", table_cell),
            Paragraph("0.112 [0.108, 0.118]", table_cell), Paragraph("0.042 (5/120)", table_cell),
            Paragraph("0.0131 [0.0084, 0.0181]", table_cell), Paragraph("0.108 [0.067, 0.158]<br/>(13/120)", table_cell)
        ],
        [
            Paragraph("<b>200 rounds</b>", table_cell), Paragraph("<b>1.00 [0.839, 1.000]</b>", table_cell),
            Paragraph("0.755 [0.741, 0.770]", table_cell), Paragraph("0.958 [0.932, 0.984]", table_cell),
            Paragraph("0.084 [0.081, 0.087]", table_cell), Paragraph("0.000* (0/120)", table_cell),
            Paragraph("0.0121 [0.0091, 0.0154]", table_cell), Paragraph("0.100 [0.050, 0.158]<br/>(12/120)", table_cell)
        ],
        [
            Paragraph("<b>500 rounds</b>", table_cell), Paragraph("<b>1.00 [0.839, 1.000]</b>", table_cell),
            Paragraph("1.052 [1.042, 1.062]", table_cell), Paragraph("0.958 [0.942, 0.973]", table_cell),
            Paragraph("<b>0.056 [0.054, 0.059]</b>", table_cell), Paragraph("<b>0.000* (0/120)</b>", table_cell),
            Paragraph("<b>0.0074 [0.0056, 0.0094]</b>", table_cell), Paragraph("<b>0.050 [0.017, 0.083]</b><br/>(6/120)", table_cell)
        ],
    ]
    t2 = Table(data2, colWidths=[0.75*inch, 0.98*inch, 0.95*inch, 0.90*inch, 0.85*inch, 0.80*inch, 0.95*inch, 0.90*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 1.0),
    ]))
    story.append(t2)
    story.append(Paragraph("<b>Table 2: Trajectory Length Sensitivity (T in [50, 500]).</b> Evaluated across 20 independent seeds per cell under fixed primary calibrated thresholds. T=200 directly reuses primary benchmark trajectories. Underlying counts report flagged pairs / total negative pairs. *Denotes degenerate bootstrap intervals due to uniform zero flagged pairs across all seeds.", caption_style))
    story.append(Spacer(1, 1))

    fig3_path = FIGURES_DIR / "fig3_coordination_strength.png"
    fig5_path = FIGURES_DIR / "fig5_trajectory_length_sensitivity.png"
    if fig3_path.exists() and fig5_path.exists():
        t_figs2 = Table([
            [Image(str(fig3_path), width=3.45 * inch, height=1.52 * inch),
             Image(str(fig5_path), width=3.45 * inch, height=1.52 * inch)]
        ], colWidths=[3.55 * inch, 3.55 * inch])
        t_figs2.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('PADDING', (0, 0), (-1, -1), 0)]))
        story.append(t_figs2)
        story.append(Paragraph("<b>Figure 3: Detection Boundaries and Sample Scaling.</b> Left: Top-1 Accuracy across coordination strength alpha. Right: Finite-sample behavior of spurious correlation vs. raw CMI vs. conditioned PNI net influence across trajectory length T in [50, 500].", caption_style))
    story.append(Spacer(1, 1))

    story.append(Paragraph(
        "<b>Trajectory Length Dynamics:</b> Over the tested T = 50–500 range, unconditioned Marginal MI in Condition B remains stubbornly elevated (~0.94–0.96 bits). "
        "In contrast, the estimated raw CMI score declines monotonically from 0.144 bits at T = 50 down to 0.056 bits at T = 500 (a 61% reduction), bringing CMI FPR from 0.450 to 0.000. "
        "Similarly, the PNI net score declines from 0.0176 bits at T = 50 down to 0.0074 bits at T = 500 (a 58% reduction), reducing Condition B PNI false positives from 0.133 to 0.050. "
        "PNI achieves 100% Top-1 accuracy across all tested lengths in Condition C.",
        body_style
    ))

    # =========================================================================
    # PAGE 5: Exploratory LLM Pilot, Table 3, Discussion & Limitations, Conclusion
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("5. Exploratory LLM-Agent Pilot (Modal CPU)", h1_style))
    story.append(Paragraph(
        "To explore whether information-theoretic collusion detection applies to open-weight generative agents, we executed an exploratory "
        "pilot using <b>Qwen/Qwen2.5-1.5B-Instruct</b> deployed on Modal CPU workers (4 vCPU, 8 GB RAM). Four independent model instances "
        "engaged in simultaneous games under three conditions: (1) <b>LLM-A</b> (independent action selection, per-decision sampling seeds); "
        "(2) <b>LLM-B</b> (shared public plurality state counter-strategy); and (3) <b>LLM-C</b> (private coordination signal delivered to pair (0, 1) "
        "and excluded from detector logs). Observable public state was abstracted to 5 states: S in {START, ROCK, PAPER, SCISSORS, TIE}. "
        "A bounded parser enforced valid action output without defaulting malformed text to arbitrary actions.",
        body_style
    ))

    headers3 = [
        "Condition", "Traj.", "Decisions", "Valid Rate", "Latency",
        "Signal\nCompliance", "Top-1\nMarg. MI", "Top-1\nCMI", "Top-1\nPNI Net",
        "Mean Marg.\nMI (bits)", "Mean CMI\n(bits)", "Mean PNI\nNet (bits)"
    ]
    data3 = [
        [Paragraph(h, table_header) for h in headers3],
        [
            Paragraph("<b>LLM-A (Independent)</b>", table_cell), Paragraph("3", table_cell),
            Paragraph("60", table_cell), Paragraph("1.000 (60/60)", table_cell), Paragraph("1.39s", table_cell),
            Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell),
            Paragraph("0.2416", table_cell), Paragraph("0.0768", table_cell), Paragraph("-0.0084", table_cell)
        ],
        [
            Paragraph("<b>LLM-B (Shared Context)</b>", table_cell), Paragraph("3", table_cell),
            Paragraph("60", table_cell), Paragraph("1.000 (60/60)", table_cell), Paragraph("1.53s", table_cell),
            Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell),
            Paragraph("0.1118", table_cell), Paragraph("0.0706", table_cell), Paragraph("-0.0212", table_cell)
        ],
        [
            Paragraph("<b>LLM-C (Private Signal)</b>", table_cell), Paragraph("3", table_cell),
            Paragraph("60", table_cell), Paragraph("1.000 (60/60)", table_cell), Paragraph("1.40s", table_cell),
            Paragraph("<b>1.000 (30/30)</b>", table_cell), Paragraph("<b>1.000 (3/3)</b>", table_cell),
            Paragraph("<b>0.667 (2/3)</b>", table_cell), Paragraph("<b>0.667 (2/3)</b>", table_cell),
            Paragraph("0.1682", table_cell), Paragraph("0.0678", table_cell), Paragraph("-0.0315", table_cell)
        ],
    ]
    t3 = Table(data3, colWidths=[1.15*inch, 0.35*inch, 0.50*inch, 0.65*inch, 0.45*inch, 0.65*inch, 0.55*inch, 0.50*inch, 0.55*inch, 0.60*inch, 0.55*inch, 0.60*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F9FA")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 1.2),
    ]))
    story.append(t3)
    story.append(Paragraph("<b>Table 3: Exploratory LLM Pilot Results.</b> Evaluated across 3 seeds per condition (T=5 rounds, 4 agents = 20 decisions/trajectory; 180 total decisions). Scripted calibration thresholds are not applied to LLM traces; performance is evaluated via pair rankings and descriptive mutual information.", caption_style))
    story.append(Spacer(1, 1))

    story.append(Paragraph(
        "<b>Pilot Findings & Scope:</b> (1) <i>Signal Adherence:</i> Qwen2.5-1.5B achieved 100% adherence (30/30 opportunities) to the secret coordination directive in Condition LLM-C, confirming that instruction-tuned LLMs can reliably execute covert coordination rules. "
        "(2) <i>Detector Rankings:</i> Marginal MI ranked the coordinating pair #1 in 3/3 seeds (100%), while CMI and PNI ranked the pair #1 in 2/3 seeds (66.7%). In ultra-short horizons (T=5), discrete state conditioning over 5 states fragments counts, dampening the empirical CMI signal. "
        "(3) <i>Exploratory Boundary:</i> This pilot is strictly descriptive (N=3 seeds) and is not powered for statistical significance or calibrated thresholded FPR estimation.",
        body_style
    ))
    story.append(Spacer(1, 2))

    story.append(Paragraph("6. Discussion and Limitations", h1_style))
    story.append(Paragraph(
        "<b>6.1 Auditing Implications: Why Correlation Lies:</b> In algorithmic markets, autonomous pricing agents, high-frequency execution bots, "
        "and procurement models continuously observe common economic shocks (e.g., benchmark interest rates, commodity prices, wholesale tariffs). "
        "Our empirical results demonstrate that unconditioned correlation metrics (e.g., Pearson correlation, raw mutual information) cannot differentiate "
        "rational adaptation from collusive coordination. Regulators relying on unconditioned metrics would commit severe Type I errors, "
        "falsely targeting competitive firms solely because they adapted efficiently to common public information.",
        body_style
    ))
    story.append(Paragraph(
        "<b>6.2 Limitations:</b> We transparently acknowledge the constraints of this study:<br/>"
        "1. <i>Controlled Scripted Signatures vs. Real-World Cartels:</i> Condition C establishes a controlled <i>shared-private-signal coordination condition</i> "
        "demonstrating that PNI recognizes statistical action dependence in scripted matrix games. However, detecting this statistical signature does not prove "
        "that agents formed an economically harmful cartel, extracted supra-competitive rents, or inflicted market harm. Furthermore, unobserved common causes "
        "(e.g., unobserved common private inputs, shared proprietary features, or unmodeled system states) could generate similar conditional action dependence.<br/>"
        "2. <i>Estimator Regularization Sensitivity:</i> In our 4-player game with |S| = 82 states, state conditioning partitions observations across sparse cells. "
        "Evaluating smoothing sensitivity under Condition B at T = 200 reveals that the unsmoothed plug-in estimator (using 0 log 0 = 0) yields CMI = 0.062 bits, "
        "while Dirichlet smoothing yields 0.084 bits (Jeffreys, lambda = 0.5) and 0.096 bits (Laplace, lambda = 1.0). Pseudocount smoothing introduces a mild upward baseline "
        "in sparse regimes, though all values remain well below unconditioned Marginal MI (0.958 bits).<br/>"
        "3. <i>Exploratory Small-Sample Calibration:</i> Nominal 5% decision thresholds were calibrated from 5 validation seeds (30 candidate pairs) of Condition A. "
        "While isolating calibration from the 15 held-out test seeds, 30 samples yield coarse quantile estimates and do not guarantee an exact 5% population FPR.<br/>"
        "4. <i>Modal Promotional Compute Pilot & Cloud Boundaries:</i> An exploratory 180-decision pilot using Qwen2.5-1.5B-Instruct was executed on Modal CPU instances (4 cores, 8GB RAM). "
        "Allocating cloud GPU instances (T4/L4) required an active payment method on file under Modal platform policy, which was safely avoided under our zero-out-of-pocket spending cap ($0.00). "
        "The CPU pilot consumed $0.07 of promotional compute credits ($0.00 billed out of pocket) across 9 trajectories. While establishing that 1.5B instruction models achieve 100% signal compliance, "
        "small sample sizes (T=5 rounds) yield coarse empirical distributions. Scripted calibration thresholds are not calibrated to LLM action distributions and are therefore not applied. "
        "Canonical benchmark results remain grounded in the 260-trajectory scripted evaluation (54,000 rounds).",
        body_style
    ))
    story.append(Paragraph(
        "<b>6.3 Future Work:</b> Key extensions include: (1) evaluating neural CMI estimators (e.g., MINE) on LLM auction transcripts (Agrawal et al., 2025); "
        "(2) developing dynamic state abstractions to compress high-dimensional order book histories; and (3) testing PNI under intentional adversarial evasion.",
        body_style
    ))
    story.append(Spacer(1, 2))

    story.append(Paragraph("7. Conclusion", h1_style))
    story.append(Paragraph(
        "This research stress-tested Pairwise Net Influence for detecting multi-agent coordination. We demonstrated that shared observable context "
        "induces a 38-fold surge in unconditioned mutual information, causing naive detectors to suffer a 100% false positive rate. "
        "Conditioning on observable state neutralizes this broadcast shock, while competitor subtraction provides directed isolation, preserving low false positives while attaining 100% Top-1 accuracy under genuine coordination. "
        "State conditioning is an indispensable requirement for auditing multi-agent algorithmic systems, while addressing state sparsity remains a critical challenge.",
        body_style
    ))

    # =========================================================================
    # PAGE 6: Code & Data, Contributions, References, Mandatory Appendix
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("Code and Data", h1_style))
    story.append(Paragraph(
        "• <b>Code Repository:</b> All experimental scripts, detectors, environments, test suites, and report generation tools are packaged in "
        "<font face='Courier'>when_correlation_lies_all_source_code.txt</font> with SHA-256 checksums.<br/>"
        "• <b>Data & Trajectories:</b> Full scripted JSONL trajectories (260 unique files, 54,000 rounds) and exploratory LLM trajectories (9 unique files, 180 decisions) "
        "with separate ground-truth manifests are included in the review bundle ZIP (<font face='Courier'>data/raw/</font>, <font face='Courier'>data/eval_labels/</font>, "
        "<font face='Courier'>data/raw_llm/</font>, <font face='Courier'>data/eval_labels_llm/</font>).<br/>"
        "• <b>Reproducibility:</b> Complete automated test suite passes (59/59 unit tests) in Python 3.13 on standard CPU hardware.",
        body_style
    ))
    story.append(Spacer(1, 2))

    story.append(Paragraph("Author Contributions", h1_style))
    story.append(Paragraph(
        "<b>Durgesh Unde (Independent Researcher):</b> Project conceptualization, selection and formulation of research questions and experimental conditions, "
        "project direction, review of experimental outputs, qualitative interpretation of detection metrics, and final editorial responsibility for the manuscript.",
        body_style
    ))
    story.append(Spacer(1, 2))

    story.append(Paragraph("References", h1_style))
    refs = [
        "Agrawal, K., Teo, V., Vazquez, J. J., Kunnavakkam, S., Srikanth, V., & Liu, A. (2025). Evaluating LLM Agent Collusion in Double Auctions. <i>arXiv preprint arXiv:2507.01413</i>.",
        "Bonjour, T., Aggarwal, V., & Bhargava, B. (2022). Information-theoretic approach to detect collusion in multi-agent games. <i>Proceedings of Machine Learning Research (UAI 2022)</i>, 180, 223–232.",
        "Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020). Artificial intelligence, algorithmic pricing, and collusion. <i>American Economic Review</i>, 110(10), 3267–3297.",
        "Cover, T. M., & Thomas, J. A. (2006). <i>Elements of Information Theory</i> (2nd ed.). John Wiley & Sons.",
        "Holste, D., Grosse, I., & Herzel, H. (1998). Bayes' estimators of generalized entropies. <i>Journal of Physics A: Mathematical and General</i>, 31(11), 2551–2566.",
        "Paninski, L. (2003). Estimation of entropy and mutual information. <i>Neural Computation</i>, 15(6), 1191–1253.",
        "Schwalbe, U. (2018). Algorithms, machine learning, and collusion. <i>Journal of Competition Law & Economics</i>, 14(4), 568–607.",
    ]
    for r in refs:
        story.append(Paragraph(f"• {r}", bullet_style))
    story.append(Spacer(1, 2))

    story.append(Paragraph("Appendix: Limitations and Dual-Use / Ethical Considerations", h1_style))
    story.append(Paragraph(
        "<b>Mandatory Ethical & Dual-Use Analysis:</b> Auditing multi-agent systems for collusion possesses significant dual-use implications:<br/>"
        "1. <i>Defensive Regulatory Oversight:</i> Information-theoretic detectors empower market regulators, exchange operators, and platform architects "
        "to continuously monitor high-frequency trading and procurement markets for tacit cartels without requiring invasive access to proprietary model weights.<br/>"
        "2. <i>Adversarial Exploitation & Stealth Evasion:</i> Conversely, adversarial agent developers could incorporate PNI as an explicit penalty term "
        "in multi-agent reinforcement learning objectives. By penalizing conditional mutual information, malicious actors could train stealthy collusive policies "
        "that sustain supracompetitive markups while remaining statistically undetectable under state-conditioned audits.<br/>"
        "3. <i>Harms of False Accusations:</i> Our Condition B results demonstrate the catastrophic risk of unconditioned metrics: a 100% false accusation rate "
        "against competitive agents responding to shared context. Auditing architectures must mandate state conditioning and calibrated uncertainty intervals "
        "before initiating punitive regulatory actions.",
        body_style
    ))
    story.append(Spacer(1, 2))

    story.append(Paragraph("LLM Usage Statement", h1_style))
    story.append(Paragraph(
        "ChatGPT (OpenAI) assisted with research ideation, paper explanation, methodology critique, independent review, "
        "and prompt drafting. Antigravity (Google DeepMind advanced agentic coding assistant) assisted with code implementation, "
        "unit test suites, experiment orchestration, Modal CPU pilot harness and execution, statistical metric aggregations, visualizations, ReportLab PDF layout, "
        "and packaging under the author's iterative instructions and review. Project direction, experimental design choices, "
        "qualitative analysis, and final editorial responsibility rest entirely with the author (Durgesh Unde). Canonical benchmark evaluations used 260 scripted trajectories (54,000 rounds); exploratory LLM pilot evaluations used 9 trajectories (180 decisions) executed on Modal CPU ($0.07 promotional credit, $0.00 out-of-pocket spend). Zero personal payment cards or external paid token APIs were used. The experimental protocol was frozen after exploratory prototyping and was not preregistered prior to exploration.",
        body_style
    ))
    story.append(Spacer(1, 2))
    story.append(Paragraph("________________<br/><sup>1</sup> Research conducted for the Apart Research AI Collusion Research Sprint, October 23–25, 2026.", caption_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled to: {PDF_PATH}")
    return PDF_PATH


if __name__ == "__main__":
    build_pdf()
