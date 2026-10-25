"""
scripts/package_delivery.py
===========================
Packages the complete AI Collusion Research Sprint final delivery artifacts:
1. final_delivery/when_correlation_lies_report.pdf
2. final_delivery/when_correlation_lies_all_source_code.txt
3. final_delivery/when_correlation_lies_review_bundle.zip
4. final_delivery/results_manifest.txt (external manifest matching all delivered files)

Manifest Architecture:
- Writes an internal manifest (excluding the ZIP's own checksum) inside the ZIP bundle.
- Builds the review bundle ZIP with clean repository structure (zero duplicate folders).
- Computes the final size and SHA-256 checksum of the closed ZIP.
- Writes the external manifest alongside the ZIP with the exact final checksums of all deliverables.
"""

import sys
import os
import shutil
import hashlib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DELIVERY_DIR = PROJECT_ROOT / "final_delivery"
REPORT_PDF = PROJECT_ROOT / "report" / "when_correlation_lies_report.pdf"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest_text(target_pdf: Path, concat_path: Path, zip_path: Path = None) -> str:
    """Generates the manifest text. If zip_path is None, generates internal manifest without ZIP hash."""
    lines = []
    lines.append("=" * 80)
    lines.append("APART RESEARCH AI COLLUSION RESEARCH SPRINT 2026 — SUBMISSION MANIFEST")
    lines.append("Track 1.4: Markets and Collusion")
    lines.append("Project Title: When Correlation Lies: Stress-Testing Pairwise Net Influence")
    lines.append("Author: Durgesh Unde (Independent Researcher)")
    lines.append("Event: Apart Research AI Collusion Research Sprint (October 23–25, 2026)")
    lines.append("=" * 80 + "\n")

    lines.append("AUTHOR CONTRIBUTIONS:")
    lines.append("-" * 80)
    lines.append(
        "Durgesh Unde (Independent Researcher): Project conceptualization, selection and formulation of research questions\n"
        "and experimental conditions, project direction, review of experimental outputs, qualitative interpretation\n"
        "of detection metrics, and final editorial responsibility for the manuscript.\n"
    )

    lines.append("AI USAGE STATEMENT:")
    lines.append("-" * 80)
    lines.append(
        "ChatGPT (OpenAI) assisted with research ideation, paper explanation, methodology critique, independent review,\n"
        "and prompt drafting. Antigravity (Google DeepMind advanced agentic coding assistant) assisted with code implementation,\n"
        "unit test suites, experiment orchestration, Modal CPU cloud infrastructure harness and execution, statistical metric aggregations, visualizations, document and PDF layout,\n"
        "and packaging under the author's iterative instructions and review. Project direction, experimental design choices,\n"
        "qualitative analysis, and final editorial responsibility rest entirely with the author (Durgesh Unde). Canonical benchmark\n"
        "evaluations used 260 scripted trajectories (54,000 rounds); open-weight LLM evaluations used 630 trajectories (68,400 decisions)\n"
        "executed on Modal CPU ($40.63 metered compute: $40.12 promotional credits applied, $0.51 network egress allowance; $0.00 out-of-pocket spend). Zero personal payment cards or external paid token APIs were used.\n"
        "The experimental protocol was frozen after exploratory prototyping and was not preregistered prior to exploration.\n"
    )

    lines.append("PRIMARY DELIVERABLES:")
    lines.append("-" * 80)
    pdf_hash = sha256_file(target_pdf)
    lines.append(f"File: {target_pdf.name}")
    lines.append(f"  Size: {target_pdf.stat().st_size:,} bytes")
    lines.append(f"  SHA-256: {pdf_hash}\n")

    concat_hash = sha256_file(concat_path)
    lines.append(f"File: {concat_path.name}")
    lines.append(f"  Size: {concat_path.stat().st_size:,} bytes")
    lines.append(f"  SHA-256: {concat_hash}\n")

    if zip_path is not None and zip_path.exists():
        zip_hash = sha256_file(zip_path)
        lines.append(f"File: {zip_path.name}")
        lines.append(f"  Size: {zip_path.stat().st_size:,} bytes")
        lines.append(f"  SHA-256: {zip_hash}\n")
    else:
        lines.append("Package: when_correlation_lies_review_bundle.zip (self-contained repository archive)\n")

    lines.append("\nCORE BENCHMARK FIGURES:")
    lines.append("-" * 80)
    for fig in sorted((PROJECT_ROOT / "results" / "figures").glob("*.png")):
        h = sha256_file(fig)
        lines.append(f"Figure: {fig.name} ({fig.stat().st_size:,} bytes)")
        lines.append(f"  SHA-256: {h}")

    lines.append("\nKEY BENCHMARK CSV TABLES:")
    lines.append("-" * 80)
    for tab in [
        PROJECT_ROOT / "results" / "tables" / "summary.csv",
        PROJECT_ROOT / "results" / "tables" / "sensitivity_trajectory_length.csv",
        PROJECT_ROOT / "results" / "tables" / "llm_pilot_summary.csv",
        PROJECT_ROOT / "results" / "tables" / "llm_expanded_summary.csv",
        PROJECT_ROOT / "results" / "tables" / "llm_expanded_t50_summary.csv",
        PROJECT_ROOT / "results" / "tables" / "llm_threshold_summary.csv",
    ]:
        if tab.exists():
            h = sha256_file(tab)
            lines.append(f"Table: {tab.name} ({tab.stat().st_size:,} bytes)")
            lines.append(f"  SHA-256: {h}")

    lines.append("\nEXPERIMENTAL TELEMETRY & AUDIT SUMMARY:")
    lines.append("-" * 80)
    lines.append("• Canonical Scripted Trajectories: 260 unique JSONL files (140 primary T=200 + 120 sensitivity T in {50, 100, 500})")
    lines.append("• Total Scripted Rounds: 54,000 unique rounds (140 * 200 = 28,000; 40 * 50 = 2,000; 40 * 100 = 4,000; 40 * 500 = 20,000)")
    lines.append("• Total Evaluations: 300 evaluations across 4 experimental conditions and sensitivity sweeps")
    lines.append("• Open-Weight LLM Agent Trajectories: 630 unique JSONL files (68,400 total decisions across T in {5, 20, 50} rounds)")
    lines.append("  - Pilot (T=5): 120 files (2,400 decisions across 40 seeds)")
    lines.append("  - Expanded Primary (T=20): 300 files (24,000 decisions across 100 seeds)")
    lines.append("  - Extended Horizon (T=50): 210 files (42,000 decisions across 70 seeds)")
    lines.append("• Automated Unit Test Suite: 59 / 59 tests passed (pytest 9.1.1, Python 3.13.7)")
    lines.append("• Cloud Compute & Billing: Modal CPU metered usage was $40.63 ($40.12 promotional credits applied, $0.51 network egress allowance; $0.00 billed out-of-pocket)")
    lines.append("• Decision Threshold Calibration: Small-sample exploratory calibration on Condition A validation seeds (seeds 0-4, 30 candidate pairs) targeting nominal 5% FPR")
    lines.append("  - Calibrated PNI Threshold (tau_PNI): 0.010427 bits")
    lines.append("  - Calibrated Conditional MI Threshold (tau_CMI): 0.117454 bits")
    lines.append("  - Calibrated Marginal MI Threshold (tau_MI): 0.030751 bits")
    lines.append("• Open-Weight LLM Empirical Results (Qwen2.5-1.5B on Modal CPU, 68,400 decisions):")
    lines.append("  - Valid Action Generation Rate: 100.0% (68,400/68,400 decisions valid parsed actions across all horizons)")
    lines.append("  - Condition LLM-C Signal Compliance: 100.0% (11,400/11,400 opportunities adhered to secret directive)")
    lines.append("  - Finite-Sample Horizon Sparsity Effect:")
    lines.append("      * T = 5 rounds: Top-1 Marginal MI = 1.000 (40/40), Top-1 CMI = 0.700 (28/40), Top-1 PNI = 0.700 (28/40), Mean CMI = 0.0696 bits")
    lines.append("      * T = 20 rounds: Top-1 Marginal MI = 1.000 (100/100), Top-1 CMI = 1.000 (100/100), Top-1 PNI = 1.000 (100/100), Mean CMI = 0.1333 bits")
    lines.append("      * T = 50 rounds: Top-1 Marginal MI = 1.000 (70/70), Top-1 CMI = 1.000 (70/70), Top-1 PNI = 1.000 (70/70), Mean CMI = 0.1770 bits")
    lines.append("• Canonical Scripted Benchmark Key Results (T=200 rounds, 20 seeds):")
    lines.append("  - Condition A (Independent Play, Held-Out Test n=15): PNI Top = 0.005 bits, PNI FPR = 0.022 [0.000, 0.056] (2/90 pairs flagged); CMI FPR = 0.111 (10/90); Marginal MI FPR = 0.033 (3/90)")
    lines.append("  - Condition A (Calibration n=5): PNI FPR = 0.067 (2/30), CMI FPR = 0.067 (2/30), Marginal MI FPR = 0.067 (2/30)")
    lines.append("  - Condition B (Shared Context Confounding n=20):")
    lines.append("      * Marginal MI: 0.958 bits [0.932, 0.984], FPR = 1.000 [1.000, 1.000]* (120/120 pairs flagged, 38-fold surge)")
    lines.append("      * Conditional MI (CMI): 0.084 bits [0.081, 0.087], FPR = 0.000 [0.000, 0.000]* (0/120 pairs flagged; state conditioning neutralizes broadcast shock)")
    lines.append("      * PNI Net Score: 0.012 bits [0.009, 0.015], FPR = 0.100 [0.050, 0.158] (12/120 pairs flagged; competitor subtraction provides directed isolation)")
    lines.append("  - Condition C (Shared-Private-Signal Coordination Positive Control n=20):")
    lines.append("      * PNI Top-1 Accuracy: 1.000 (Wilson 95% CI: [0.839, 1.000]), MRR = 1.000, True Pair Net Score = 0.755 bits [0.741, 0.769]")
    lines.append("      * PNI FPR: 0.200 [0.200, 0.200]* (20/100 negative pairs flagged; *degenerate bootstrap interval due to uniform outcome of exactly 1/5 negative pairs flagged per seed)")
    lines.append("      * CMI FPR: 0.020 [0.000, 0.050] (2/100); Marginal MI FPR = 0.240 (24/100)")
    lines.append("  - Condition D (Joint Gate Sweep n=20 each):")
    lines.append("      alpha = 0.25: PNI True Score = 0.264 bits, Top-1 = 1.000 [0.839, 1.000], PNI FPR = 0.100 (10/100); CMI FPR = 0.010 (1/100); Marginal MI FPR = 1.000 (100/100)")
    lines.append("      alpha = 0.50: PNI True Score = 0.469 bits, Top-1 = 1.000 [0.839, 1.000], PNI FPR = 0.180 (18/100); CMI FPR = 0.000 (0/100); Marginal MI FPR = 1.000 (100/100)")
    lines.append("      alpha = 0.75: PNI True Score = 0.623 bits, Top-1 = 1.000 [0.839, 1.000], PNI FPR = 0.200* (20/100); CMI FPR = 0.020 (2/100); Marginal MI FPR = 0.920 (92/100)")
    lines.append("      alpha = 1.00: PNI True Score = 0.771 bits, Top-1 = 1.000 [0.839, 1.000], PNI FPR = 0.200* (20/100); CMI FPR = 0.010 (1/100); Marginal MI FPR = 0.240 (24/100)")
    lines.append("  - Trajectory Length Sensitivity (T in [50, 500]):")
    lines.append("      * Cond C Top-1 = 1.000 across all T")
    lines.append("      * Cond B CMI Score declines: 0.144 (T=50) -> 0.112 (T=100) -> 0.084 (T=200) -> 0.056 (T=500); CMI FPR declines: 0.450 -> 0.042 -> 0.000 -> 0.000")
    lines.append("      * Cond B PNI Score declines: 0.0176 (T=50) -> 0.0131 (T=100) -> 0.0121 (T=200) -> 0.0074 (T=500); PNI FPR declines: 0.133 -> 0.108 -> 0.100 -> 0.050")
    lines.append("  - Estimator Regularization Sensitivity (Cond B, T=200):")
    lines.append("      * lambda = 0.0 (unsmoothed plug-in, 0 log 0 = 0): CMI = 0.062 bits")
    lines.append("      * lambda = 0.5 (Jeffreys Dirichlet): CMI = 0.084 bits")
    lines.append("      * lambda = 1.0 (Laplace Dirichlet): CMI = 0.096 bits")

    return "\n".join(lines) + "\n"


def main():
    DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Packaging delivery to: {DELIVERY_DIR}")

    # 1. Copy PDF report and DOCX source
    target_pdf = DELIVERY_DIR / "when_correlation_lies_report.pdf"
    shutil.copy2(REPORT_PDF, target_pdf)
    print(f"Copied PDF report -> {target_pdf} ({target_pdf.stat().st_size:,} bytes)")

    target_docx = PROJECT_ROOT / "report" / "when_correlation_lies_report.docx"
    if target_docx.exists():
        delivery_docx = DELIVERY_DIR / "when_correlation_lies_report.docx"
        shutil.copy2(target_docx, delivery_docx)
        print(f"Copied DOCX report -> {delivery_docx} ({delivery_docx.stat().st_size:,} bytes)")

    # 2. Collect all source code, tests, scripts, configs, documentation
    files_to_bundle = [
        ".gitattributes",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "README.md",
        "config/experiment.yaml",
        "config/model.yaml",
        "src/__init__.py",
        "src/agents.py",
        "src/environment.py",
        "src/detectors.py",
        "src/trajectories.py",
        "src/metrics.py",
        "src/run_experiments.py",
        "src/analyze_results.py",
        "src/llm_helpers.py",
        "tests/__init__.py",
        "tests/test_analytical_cmi.py",
        "tests/test_detectors.py",
        "tests/test_environment.py",
        "tests/test_metrics.py",
        "tests/test_llm_pilot.py",
        "scripts/modal_llm_pilot.py",
        "scripts/modal_llm_expanded.py",
        "scripts/evaluate_llm_pilot.py",
        "scripts/evaluate_llm_thresholds.py",
        "scripts/plot_llm_results.py",
        "scripts/run_sensitivity.py",
        "scripts/plot_setup_diagram.py",
        "scripts/generate_pdf_report.py",
        "scripts/build_official_template_report.py",
        "scripts/verify_final_consistency.py",
        "scripts/package_delivery.py",
        "docs/scientific_audit.md",
        "docs/retrospective_research_plan.md",
        "docs/results_interpretation.md",
        "docs/pre_sprint_work_log.md",
        "archive/ARCHIVE_README.md",
        "report/report_draft.md",
        "results/tables/summary.csv",
        "results/tables/sensitivity_trajectory_length.csv",
        "results/tables/llm_pilot_summary.csv",
        "results/tables/llm_expanded_summary.csv",
        "results/tables/llm_expanded_t50_summary.csv",
        "results/tables/llm_threshold_summary.csv",
        "results/llm_pilot/billing_audit.json",
    ]

    concat_path = DELIVERY_DIR / "when_correlation_lies_all_source_code.txt"
    total_files = 0
    total_lines = 0

    with open(concat_path, "w", encoding="utf-8", newline="\n") as out_f:
        out_f.write("=" * 80 + "\n")
        out_f.write("WHEN CORRELATION LIES: AI COLLUSION RESEARCH SPRINT 2026\n")
        out_f.write("CONCATENATED SOURCE CODE AND REPRODUCIBILITY REPOSITORY\n")
        out_f.write("Author: Durgesh Unde | Affiliation: Independent Researcher\n")
        out_f.write("Track 1.4: Markets and Collusion\n")
        out_f.write("Apart Research AI Collusion Research Sprint (October 23-25, 2026)\n")
        out_f.write("=" * 80 + "\n\n")

        for rel_path in files_to_bundle:
            p = PROJECT_ROOT / rel_path
            if not p.exists():
                print(f"WARNING: File missing: {p}")
                continue

            file_hash = sha256_file(p)
            file_bytes = p.stat().st_size
            try:
                content = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
            except Exception as e:
                print(f"Error reading {p}: {e}")
                continue

            lines = content.count("\n") + 1
            total_lines += lines
            total_files += 1

            out_f.write("=" * 80 + "\n")
            out_f.write(f"FILE: {rel_path}\n")
            out_f.write(f"SHA-256: {file_hash}\n")
            out_f.write(f"SIZE: {file_bytes} bytes | LINES: {lines}\n")
            out_f.write("=" * 80 + "\n\n")
            out_f.write(content)
            out_f.write("\n\n")

    print(f"Generated concatenated source text -> {concat_path}")
    print(f"  Total bundled files: {total_files}")
    print(f"  Total code/doc lines: {total_lines:,}")
    print(f"  File size: {concat_path.stat().st_size:,} bytes")

    # 3. Create INTERNAL manifest content (excludes ZIP checksum)
    internal_manifest_content = generate_manifest_text(target_pdf, concat_path, zip_path=None)

    # 4. Create Clean Review Bundle ZIP (Project Root as ZIP root)
    zip_path = DELIVERY_DIR / "when_correlation_lies_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include root documents for immediate reviewer convenience
        zf.write(target_pdf, arcname="when_correlation_lies_report.pdf")
        zf.write(concat_path, arcname="when_correlation_lies_all_source_code.txt")
        # Write internal manifest into ZIP root
        zf.writestr("results_manifest.txt", internal_manifest_content)

        # Include all source, test, script, config, doc, and table files in standard repository paths
        for rel_path in files_to_bundle:
            p = PROJECT_ROOT / rel_path
            if p.exists():
                zf.write(p, arcname=rel_path)

        # Include official template in docs/
        template_doc = PROJECT_ROOT / "docs" / "template.docx"
        if template_doc.exists():
            zf.write(template_doc, arcname="docs/template.docx")

        # Include PDF and DOCX in report/
        zf.write(target_pdf, arcname="report/when_correlation_lies_report.pdf")
        if target_docx.exists():
            zf.write(target_docx, arcname="report/when_correlation_lies_report.docx")
            zf.write(target_docx, arcname="when_correlation_lies_report.docx")

        # Include all figures in results/figures/
        figures = sorted((PROJECT_ROOT / "results" / "figures").glob("*.png"))
        for fig in figures:
            zf.write(fig, arcname=f"results/figures/{fig.name}")

        # Include metrics jsons in results/metrics/
        for mf in sorted((PROJECT_ROOT / "results" / "metrics").glob("*.json")):
            zf.write(mf, arcname=f"results/metrics/{mf.name}")

        # Include all 260 raw JSONL trajectory files in data/raw/
        raw_files = sorted((PROJECT_ROOT / "data" / "raw").glob("*.jsonl"))
        for rf in raw_files:
            zf.write(rf, arcname=f"data/raw/{rf.name}")

        # Include all 260 ground truth label files in data/eval_labels/
        label_files = sorted((PROJECT_ROOT / "data" / "eval_labels").glob("*.json"))
        for lf in label_files:
            zf.write(lf, arcname=f"data/eval_labels/{lf.name}")

        # Include all exploratory LLM pilot trajectories in data/raw_llm/
        for rf in sorted((PROJECT_ROOT / "data" / "raw_llm").glob("*.jsonl")):
            zf.write(rf, arcname=f"data/raw_llm/{rf.name}")

        for lf in sorted((PROJECT_ROOT / "data" / "eval_labels_llm").glob("*.json")):
            zf.write(lf, arcname=f"data/eval_labels_llm/{lf.name}")

        # Include all expanded LLM trajectories in data/raw_llm_expanded/
        for rf in sorted((PROJECT_ROOT / "data" / "raw_llm_expanded").glob("*.jsonl")):
            zf.write(rf, arcname=f"data/raw_llm_expanded/{rf.name}")

        for lf in sorted((PROJECT_ROOT / "data" / "eval_labels_llm_expanded").glob("*.json")):
            zf.write(lf, arcname=f"data/eval_labels_llm_expanded/{lf.name}")

        # Include all extended T50 LLM trajectories in data/raw_llm_expanded_t50/
        for rf in sorted((PROJECT_ROOT / "data" / "raw_llm_expanded_t50").glob("*.jsonl")):
            zf.write(rf, arcname=f"data/raw_llm_expanded_t50/{rf.name}")

        for lf in sorted((PROJECT_ROOT / "data" / "eval_labels_llm_expanded_t50").glob("*.json")):
            zf.write(lf, arcname=f"data/eval_labels_llm_expanded_t50/{lf.name}")

    # ZIP is completely closed and sealed here!
    zip_size = zip_path.stat().st_size
    zip_hash = sha256_file(zip_path)
    print(f"Generated review bundle ZIP -> {zip_path} ({zip_size:,} bytes)")
    print(f"  SHA-256: {zip_hash}")

    # 5. Generate EXTERNAL results_manifest.txt (includes the final sealed ZIP's checksum and size)
    external_manifest_content = generate_manifest_text(target_pdf, concat_path, zip_path=zip_path).replace("\r\n", "\n")
    manifest_path = DELIVERY_DIR / "results_manifest.txt"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(external_manifest_content)
    root_manifest_path = PROJECT_ROOT / "results_manifest.txt"
    with open(root_manifest_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(external_manifest_content)
    print(f"Generated external results manifest -> {manifest_path} ({manifest_path.stat().st_size:,} bytes)")
    print(f"Synced results manifest to root -> {root_manifest_path}")


if __name__ == "__main__":
    main()
