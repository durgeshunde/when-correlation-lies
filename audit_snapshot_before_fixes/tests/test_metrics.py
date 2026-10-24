"""
tests/test_metrics.py
=====================
Unit tests for ranking, accuracy, false positive rate, and bootstrap CI metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.metrics import (
    top_k_accuracy,
    reciprocal_rank,
    true_pair_score,
    false_positive_rate,
    chance_top1_accuracy,
    aggregate_metrics,
)


class TestMetrics:
    def test_top1_accuracy_success(self):
        ranked = [{"pair": (0, 1), "score": 0.5}, {"pair": (0, 2), "score": 0.1}]
        assert top_k_accuracy(ranked, (0, 1), k=1) == 1.0
        assert top_k_accuracy(ranked, (1, 0), k=1) == 1.0  # order agnostic

    def test_top1_accuracy_failure(self):
        ranked = [{"pair": (0, 2), "score": 0.5}, {"pair": (0, 1), "score": 0.1}]
        assert top_k_accuracy(ranked, (0, 1), k=1) == 0.0
        assert top_k_accuracy(ranked, (0, 1), k=2) == 1.0

    def test_reciprocal_rank(self):
        ranked = [
            {"pair": (0, 2), "score": 0.5},
            {"pair": (1, 2), "score": 0.4},
            {"pair": (0, 1), "score": 0.3},
        ]
        assert reciprocal_rank(ranked, (0, 1)) == pytest.approx(1.0 / 3.0)
        assert reciprocal_rank(ranked, (0, 2)) == pytest.approx(1.0)
        assert reciprocal_rank(ranked, (2, 3)) == 0.0

    def test_false_positive_rate(self):
        # 6 candidate pairs total, pair (0, 1) is true
        ranked = [
            {"pair": (0, 1), "score": 0.5, "flagged": True},
            {"pair": (0, 2), "score": 0.2, "flagged": True},
            {"pair": (0, 3), "score": -0.1, "flagged": False},
            {"pair": (1, 2), "score": -0.2, "flagged": False},
            {"pair": (1, 3), "score": -0.3, "flagged": False},
            {"pair": (2, 3), "score": -0.4, "flagged": False},
        ]
        # Negatives are the 5 pairs other than (0, 1). Only (0, 2) is flagged.
        fpr = false_positive_rate(ranked, threshold=0.0, true_pair=(0, 1))
        assert fpr == pytest.approx(1.0 / 5.0)

    def test_chance_top1(self):
        assert chance_top1_accuracy(6) == pytest.approx(1.0 / 6.0)

    def test_bootstrap_aggregation(self):
        results = [{"score": 1.0}, {"score": 2.0}, {"score": 3.0}, {"score": 4.0}, {"score": 5.0}]
        agg = aggregate_metrics(results, bootstrap_n=500, confidence=0.95)
        assert "score" in agg
        assert agg["score"]["mean"] == pytest.approx(3.0)
        assert agg["score"]["ci_low"] <= agg["score"]["mean"]
        assert agg["score"]["ci_high"] >= agg["score"]["mean"]
        assert agg["score"]["n"] == 5
