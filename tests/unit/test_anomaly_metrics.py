"""Unit tests for Anomaly Detection Evaluation Metrics."""

import numpy as np
import pytest

from ml.evaluation.metrics import compute_anomaly_metrics


def test_metrics_perfect_separation():
    # 5 normal, 5 abnormal, perfectly separated
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_scores = [0.1, 0.15, 0.2, 0.22, 0.25, 0.8, 0.85, 0.9, 0.92, 0.99]

    metrics = compute_anomaly_metrics(y_true, y_scores, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pauc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["confusion_matrix"]["tp"] == 5
    assert metrics["confusion_matrix"]["tn"] == 5
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_metrics_partial_overlap():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    # Normal with 1 high score, abnormal with 1 low score
    y_scores = [0.1, 0.2, 0.3, 0.7, 0.4, 0.8, 0.85, 0.9]

    metrics = compute_anomaly_metrics(y_true, y_scores, threshold=0.5)

    assert metrics["roc_auc"] < 1.0
    assert metrics["roc_auc"] > 0.5
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1


def test_metrics_single_class_edge_case():
    # Only normal samples
    y_true = [0, 0, 0, 0]
    y_scores = [0.1, 0.2, 0.3, 0.4]

    metrics = compute_anomaly_metrics(y_true, y_scores, threshold=0.25)
    assert metrics["roc_auc"] is None
    assert metrics["n_normal"] == 4
    assert metrics["n_abnormal"] == 0

