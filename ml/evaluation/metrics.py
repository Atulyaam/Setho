"""Anomaly Detection Evaluation Metrics for Stetho.

Computes standard acoustic anomaly detection benchmarks:
- ROC-AUC: Area under the full ROC curve
- pAUC: Partial AUC across low false-alarm region (max FPR = 0.1)
- Operating point metrics derived from development calibrated threshold:
  - Precision, Recall (True Positive Rate), F1-Score
  - False Positive Rate (FPR), False Negative Rate (FNR)
  - Confusion Matrix (TP, FP, TN, FN)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_anomaly_metrics(
    y_true: Union[List[int], np.ndarray],
    y_scores: Union[List[float], np.ndarray],
    threshold: Optional[float] = None,
    max_fpr: float = 0.1,
) -> Dict[str, Any]:
    """Compute comprehensive ranking and operating point anomaly metrics.

    Args:
        y_true: Ground truth binary labels (0 = normal, 1 = abnormal).
        y_scores: Continuous anomaly scores (higher = more anomalous).
        threshold: Calibrated operating threshold. If None, median score is used.
        max_fpr: Maximum false positive rate for partial AUC (DCASE standard: 0.1).

    Returns:
        Dictionary of computed metrics.
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_scores_arr = np.array(y_scores, dtype=float)

    n_normal = int(np.sum(y_true_arr == 0))
    n_abnormal = int(np.sum(y_true_arr == 1))

    # Guard: requires both normal and abnormal samples for ROC curves
    can_compute_roc = n_normal > 0 and n_abnormal > 0

    roc_auc = float("nan")
    pauc = float("nan")

    if can_compute_roc:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
        except Exception:
            roc_auc = float("nan")

        try:
            # Scikit-learn's max_fpr normalizes pAUC to [0, 1]
            pauc = float(roc_auc_score(y_true_arr, y_scores_arr, max_fpr=max_fpr))
        except Exception:
            pauc = float("nan")

    # Operating point metrics
    if threshold is None:
        threshold = float(np.median(y_scores_arr))

    y_pred = (y_scores_arr > threshold).astype(int)

    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "n_normal": n_normal,
        "n_abnormal": n_abnormal,
        "total_samples": len(y_true_arr),
        "threshold": round(float(threshold), 6),
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "pauc": round(pauc, 4) if not np.isnan(pauc) else None,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }

