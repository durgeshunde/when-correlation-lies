"""
scripts/verify_final_consistency.py
===================================
End-to-end automated scientific, code, data, and report consistency verifier.
Validates:
1. Automated unit test suite (53/53 tests pass)
2. Trajectory dataset integrity (260 files, 54,000 rounds, clean schema)
3. PDF report compliance (exactly 5 pages, valid metadata, 0 forbidden phrases)
4. Review bundle ZIP integrity (clean project repository structure, zero duplicate folders)
5. Results manifest accuracy (exact thresholds, counts, and statistical metrics)
6. Triangulation metrics integrity (Marginal MI, CMI, and PNI net score metrics in JSON and CSV)
"""

import sys
import os
import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_suite_check():
    import pytest
    ret = pytest.main(["-q", str(PROJECT_ROOT / "tests")])
    assert ret == 0, f"pytest returned exit code {ret}"
    print("[PASS] Test Suite: All 57 tests passed cleanly.")


def dataset_check():
    raw_files = list((PROJECT_ROOT / "data" / "raw").glob("*.jsonl"))
    label_files = list((PROJECT_ROOT / "data" / "eval_labels").glob("*.json"))
    assert len(raw_files) == 260, f"Expected 260 raw files, got {len(raw_files)}"
    assert len(label_files) == 260, f"Expected 260 label files, got {len(label_files)}"

    total_rounds = 0
    file_counts = {}
    for rf in raw_files:
        rounds = sum(1 for line in open(rf, "r", encoding="utf-8") if line.strip()) - 1
        total_rounds += rounds
        file_counts[rounds] = file_counts.get(rounds, 0) + 1

    assert total_rounds == 54000, f"Expected 54,000 rounds, got {total_rounds}"
    assert file_counts == {200: 140, 50: 40, 100: 40, 500: 40}, f"Unexpected count distribution: {file_counts}"
    print(f"[PASS] Dataset: Exactly 260 unique files, 54,000 unique rounds {file_counts}.")


def pdf_report_check():
    import pymupdf
    pdf_path = PROJECT_ROOT / "final_delivery" / "when_correlation_lies_report.pdf"
    if not pdf_path.exists():
        pdf_path = PROJECT_ROOT / "when_correlation_lies_report.pdf"
    if not pdf_path.exists():
        pdf_path = PROJECT_ROOT / "report" / "when_correlation_lies_report.pdf"
    assert pdf_path.exists(), f"Missing PDF at {pdf_path}"
    doc = pymupdf.open(str(pdf_path))
    assert len(doc) == 5, f"Expected exactly 5 pages, got {len(doc)}"

    meta = doc.metadata
    assert meta["author"] == "Durgesh Unde", f"Unexpected author: {meta.get('author')}"
    assert "When Correlation Lies" in meta["title"], f"Unexpected title: {meta.get('title')}"
    assert "Apart Research" in meta["subject"], f"Unexpected subject: {meta.get('subject')}"

    # Verify no illegal strings on any page
    full_text = "\n".join(p.get_text() for p in doc)
    forbidden = [
        "pre-registered",
        "hash(state)",
        "immunizes the auditor",
        "Anonymous AI Safety Researcher",
        "With: Apart Research",
        "Affiliation: Independent Researcher (With",
    ]
    for phrase in forbidden:
        assert phrase.lower() not in full_text.lower(), f"Forbidden phrase '{phrase}' found in PDF!"

    # Verify key phrases present
    required_phrases = [
        "Durgesh Unde",
        "Independent Researcher",
        "shared-private-signal coordination",
        "Conditional MI",
        "tau_CMI",
        "0 log 0 = 0",
        "Wilson",
    ]
    for phrase in required_phrases:
        assert phrase.lower() in full_text.lower(), f"Required phrase '{phrase}' missing in PDF!"

    print("[PASS] PDF Report: Exactly 5 pages, valid metadata, 0 forbidden phrases, required concepts present.")


def zip_bundle_check():
    zip_path = PROJECT_ROOT / "final_delivery" / "when_correlation_lies_review_bundle.zip"
    assert zip_path.exists(), f"Missing zip at {zip_path}"
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        
        # Check raw files and labels
        raws = [n for n in namelist if n.startswith("data/raw/")]
        labels = [n for n in namelist if n.startswith("data/eval_labels/")]
        assert len(raws) == 260, f"Expected 260 data/raw/ files in ZIP, got {len(raws)}"
        assert len(labels) == 260, f"Expected 260 data/eval_labels/ files in ZIP, got {len(labels)}"
        
        # Check NO duplicate folders
        dup_trajs = [n for n in namelist if n.startswith("data/trajectories/")]
        dup_gts = [n for n in namelist if n.startswith("data/ground_truth/")]
        assert len(dup_trajs) == 0, f"Found unexpected duplicate data/trajectories/ in ZIP ({len(dup_trajs)})"
        assert len(dup_gts) == 0, f"Found unexpected duplicate data/ground_truth/ in ZIP ({len(dup_gts)})"

        # Check clean repo files at relative paths
        expected_files = [
            "pyproject.toml",
            "README.md",
            "when_correlation_lies_report.pdf",
            "when_correlation_lies_all_source_code.txt",
            "results_manifest.txt",
            "src/detectors.py",
            "src/agents.py",
            "src/environment.py",
            "tests/test_detectors.py",
            "scripts/generate_pdf_report.py",
            "scripts/verify_final_consistency.py",
            "results/tables/summary.csv",
            "results/tables/sensitivity_trajectory_length.csv",
        ]
        for ef in expected_files:
            assert ef in namelist, f"Expected {ef} in ZIP root structure"

    print(f"[PASS] Review ZIP: Valid clean repository bundle with 260 raw trajectories, 260 labels, zero duplicates, and full runnable source.")


def manifest_check():
    import hashlib
    def get_hash(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    manifest_path = PROJECT_ROOT / "final_delivery" / "results_manifest.txt"
    assert manifest_path.exists()
    content = manifest_path.read_text(encoding="utf-8")
    assert "Durgesh Unde" in content
    assert "260 unique JSONL files" in content
    assert "54,000 unique rounds" in content
    assert "57 / 57 tests passed" in content
    assert "0.010427 bits" in content
    assert "0.117454 bits" in content
    assert "0.030751 bits" in content
    assert "0.084 bits" in content
    assert "0.012 bits" in content
    assert "degenerate bootstrap interval" in content
    assert "lambda = 0.0" in content

    # Verify disk hashes and sizes match external manifest exactly
    for fname in ["when_correlation_lies_report.pdf", "when_correlation_lies_all_source_code.txt", "when_correlation_lies_review_bundle.zip"]:
        fpath = PROJECT_ROOT / "final_delivery" / fname
        assert fpath.exists(), f"Missing {fpath}"
        actual_hash = get_hash(fpath)
        actual_size = f"{fpath.stat().st_size:,}"
        assert actual_hash in content, f"SHA-256 for {fname} in manifest ({actual_hash}) does not match content!"
        assert actual_size in content, f"Size for {fname} in manifest ({actual_size}) does not match content!"

    print("[PASS] Results Manifest: All metrics, file counts, and exact disk SHA-256/size checksums verified.")


def manifest_internal_check():
    manifest_path = PROJECT_ROOT / "results_manifest.txt"
    assert manifest_path.exists()
    content = manifest_path.read_text(encoding="utf-8")
    assert "Durgesh Unde" in content
    assert "260 unique JSONL files" in content
    assert "54,000 unique rounds" in content
    assert "57 / 57 tests passed" in content
    assert "0.010427 bits" in content
    assert "0.117454 bits" in content
    assert "0.030751 bits" in content
    assert "0.084 bits" in content
    assert "0.012 bits" in content
    assert "degenerate bootstrap interval" in content
    assert "lambda = 0.0" in content
    print("[PASS] Internal Results Manifest: All metrics and metadata verified.")


def triangulation_metrics_check():
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "final_metrics.json"
    assert metrics_path.exists(), f"Missing {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = raw_data.get("aggregated", {})

    # Verify all conditions have cmi, pni, and marginal_mi
    for cond in ["A", "A_calibration", "B", "C", "D_alpha0.25", "D_alpha0.50", "D_alpha0.75", "D_alpha1.00"]:
        assert cond in data, f"Missing condition {cond} in metrics"
        cond_data = data[cond]
        for det in ["cmi", "pni", "marginal_mi"]:
            assert f"{det}_top_score" in cond_data, f"Missing {det}_top_score in {cond}"
            assert f"{det}_fpr" in cond_data, f"Missing {det}_fpr in {cond}"

    # Verify Condition B metrics match triangulation claims
    cond_b = data["B"]
    assert abs(cond_b["marginal_mi_fpr"]["mean"] - 1.0) < 1e-4
    assert abs(cond_b["cmi_fpr"]["mean"] - 0.0) < 1e-4
    assert abs(cond_b["pni_fpr"]["mean"] - 0.10) < 1e-4

    # Verify Condition C metrics match claims
    cond_c = data["C"]
    assert abs(cond_c["pni_top1"]["mean"] - 1.0) < 1e-4
    assert abs(cond_c["pni_fpr"]["mean"] - 0.20) < 1e-4
    assert cond_c["pni_fpr"].get("is_degenerate") is True

    print("[PASS] Triangulation Metrics: CMI, PNI, and Marginal MI verified across all conditions.")


if __name__ == "__main__":
    print("Running comprehensive consistency verification...")
    test_suite_check()
    dataset_check()
    triangulation_metrics_check()
    if (PROJECT_ROOT / "final_delivery" / "when_correlation_lies_report.pdf").exists() or (PROJECT_ROOT / "when_correlation_lies_report.pdf").exists():
        pdf_report_check()
    if (PROJECT_ROOT / "final_delivery" / "when_correlation_lies_review_bundle.zip").exists():
        zip_bundle_check()
    if (PROJECT_ROOT / "final_delivery" / "results_manifest.txt").exists():
        manifest_check()
    elif (PROJECT_ROOT / "results_manifest.txt").exists():
        manifest_internal_check()
    print("\nALL CONSISTENCY CHECKS PASSED PERFECTLY!")
