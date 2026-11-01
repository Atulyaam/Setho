"""Phase 6 Advanced Ensemble & Model Fusion Research Runner.

Evaluates multi-model fusion methods across all 4 pilot machines:
1. ResNet-AE + Embedding Mahalanobis (Weighted Average, Rank Average, Max Fusion)
2. MultiScale CNN-AE + Embedding Mahalanobis (Lightweight Fusion)
3. Tri-Model Consensus (ResNet-AE + MultiScale-AE + Mahalanobis)
4. Cascaded Early-Exit Filter (Fast Stage 1 -> Heavy Stage 2)

Zero-Leakage Assurance:
- Model weights from train_normal.csv
- Fusion standardization & thresholds calibrated strictly on dev_normal.csv
- Test evaluation on held-out test.csv
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.experiments.experiment_manager import ExperimentManager
from ml.models.research.embedding_mahalanobis import MahalanobisDetector
from ml.models.research.ensemble_fusion import CascadedFastRejectDetector, ScoreFusionEnsemble
from ml.models.research.research_detector import ResearchNeuralDetector
from ml.training.train_pipeline import _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase6_ensemble_runner")

RESEARCH_ARTIFACTS_DIR = Path("ml/experiments/research_artifacts/phase6")
PILOT_MACHINES = ["id_00", "id_02", "id_04", "id_06"]


def load_or_train_base_detectors(machine_id: str) -> Dict[str, Any]:
    """Instantiate and train/calibrate base detectors for a given machine."""
    train_split = f"ml/data/splits/machine_specific/{machine_id}/train_normal.csv"
    dev_split = f"ml/data/splits/machine_specific/{machine_id}/dev_normal.csv"

    train_rows = _read_split_csv(train_split)
    dev_rows = _read_split_csv(dev_split)

    train_paths = [r["file_path"] for r in train_rows]
    dev_paths = [r["file_path"] for r in dev_rows]

    # 1. ResNet-AE (Optimized with Cosine Annealing)
    resnet = ResearchNeuralDetector(
        machine_id=machine_id,
        arch_type="residual_cnn_ae",
        base_channels=16,
        latent_dim=64,
        optimizer_type="adamw",
        lr_schedule="cosine",
        max_epochs=20,
        early_stopping_patience=5,
        seed=42,
    )
    resnet.fit(train_paths, dev_files=dev_paths)

    # 2. MultiScale CNN-AE
    msae = ResearchNeuralDetector(
        machine_id=machine_id,
        arch_type="multiscale_cnn_ae",
        base_channels=16,
        latent_dim=64,
        optimizer_type="adamw",
        lr_schedule="cosine",
        max_epochs=20,
        early_stopping_patience=5,
        seed=42,
    )
    msae.fit(train_paths, dev_files=dev_paths)

    # 3. Embedding Mahalanobis (Statistical)
    maha = MahalanobisDetector(machine_id=machine_id, feature_preset="compact_stat_38d")
    maha.fit(train_paths)

    return {
        "resnet": resnet,
        "msae": msae,
        "mahalanobis": maha,
        "dev_paths": dev_paths,
    }


def evaluate_detector(
    detector: Any,
    machine_id: str,
    dev_paths: List[str],
) -> Dict[str, Any]:
    """Calibrate and evaluate a detector or ensemble on final test set."""
    test_split = f"ml/data/splits/machine_specific/{machine_id}/test.csv"
    test_rows = _read_split_csv(test_split)
    test_paths = [r["file_path"] for r in test_rows]
    test_labels = [1 if str(r["label"]).lower() in ("abnormal", "anomaly", "1") else 0 for r in test_rows]

    # Calibrate on dev set strictly
    calibrator = StatisticalCalibrator(target_quantile=0.95)
    cal_prof = calibrator.calibrate(detector, dev_paths)
    detector.threshold = cal_prof.operating_threshold

    # Measure latency (5 iterations)
    for _ in range(2):
        detector.score_file(test_paths[0])
    lats = []
    for _ in range(5):
        t0 = time.perf_counter()
        detector.score_file(test_paths[0])
        lats.append((time.perf_counter() - t0) * 1000.0)

    # Evaluate test predictions
    test_scores = [detector.score_file(p) for p in test_paths]
    metrics = compute_anomaly_metrics(
        y_true=test_labels,
        y_scores=test_scores,
        threshold=detector.threshold,
        max_fpr=0.1,
    )

    return {
        "machine_id": machine_id,
        "metrics": metrics,
        "threshold": detector.threshold,
        "latency_ms": round(float(np.mean(lats)), 2),
        "test_scores": [round(float(s), 6) for s in test_scores],
    }


def run_phase6_ensembles() -> Dict[str, Any]:
    """Execute all ensemble investigations across the 4 pilot machines."""
    logger.info("Initializing Phase 6 Ensemble & Fusion Research...")
    RESEARCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ensemble_configs = [
        ("resnet_maha_weighted", "weighted_average", ["resnet", "mahalanobis"], [0.65, 0.35]),
        ("resnet_maha_rank", "rank_average", ["resnet", "mahalanobis"], None),
        ("resnet_maha_max", "max_fusion", ["resnet", "mahalanobis"], None),
        ("msae_maha_weighted", "weighted_average", ["msae", "mahalanobis"], [0.70, 0.30]),
        ("tri_model_consensus", "weighted_average", ["resnet", "msae", "mahalanobis"], [0.40, 0.40, 0.20]),
    ]

    all_results: Dict[str, Any] = {cfg[0]: {} for cfg in ensemble_configs}
    all_results["cascade_fast_reject"] = {}

    for m_id in PILOT_MACHINES:
        logger.info(f"Training base detectors for machine {m_id}...")
        base = load_or_train_base_detectors(m_id)
        dev_paths = base["dev_paths"]

        # Run Score Fusion ensembles
        for ens_id, strategy, model_keys, weights in ensemble_configs:
            sub_dets = [base[k] for k in model_keys]
            ensemble = ScoreFusionEnsemble(
                machine_id=m_id,
                detectors=sub_dets,
                strategy=strategy,
                weights=weights,
            )
            ensemble.fit_fusion(dev_paths)
            res = evaluate_detector(ensemble, m_id, dev_paths)
            all_results[ens_id][m_id] = res
            logger.info(
                f"[{ens_id} | {m_id}] ROC-AUC={res['metrics']['roc_auc']:.4f}, "
                f"pAUC={res['metrics']['pauc']:.4f}, Latency={res['latency_ms']:.1f}ms"
            )

        # Run Hierarchical Cascade (Mahalanobis -> ResNet-AE)
        cascade = CascadedFastRejectDetector(
            machine_id=m_id,
            stage1_fast_detector=base["mahalanobis"],
            stage2_heavy_detector=base["resnet"],
            early_exit_quantile=0.50,
        )
        cascade.fit_cascade(dev_paths)
        cascade_res = evaluate_detector(cascade, m_id, dev_paths)
        all_results["cascade_fast_reject"][m_id] = cascade_res
        logger.info(
            f"[cascade_fast_reject | {m_id}] ROC-AUC={cascade_res['metrics']['roc_auc']:.4f}, "
            f"pAUC={cascade_res['metrics']['pauc']:.4f}, Latency={cascade_res['latency_ms']:.1f}ms"
        )

    # Compute Macro Summaries for each ensemble
    summary_report: Dict[str, Any] = {}
    for ens_id, per_m in all_results.items():
        macro_roc = float(np.mean([m["metrics"]["roc_auc"] for m in per_m.values()]))
        macro_pauc = float(np.mean([m["metrics"]["pauc"] for m in per_m.values()]))
        macro_f1 = float(np.mean([m["metrics"]["f1"] for m in per_m.values()]))
        macro_precision = float(np.mean([m["metrics"]["precision"] for m in per_m.values()]))
        macro_recall = float(np.mean([m["metrics"]["recall"] for m in per_m.values()]))
        macro_fpr = float(np.mean([m["metrics"]["fpr"] for m in per_m.values()]))
        macro_lat = float(np.mean([m["latency_ms"] for m in per_m.values()]))

        summary_report[ens_id] = {
            "macro_roc_auc": round(macro_roc, 4),
            "macro_pauc": round(macro_pauc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_fpr": round(macro_fpr, 4),
            "macro_cpu_mean_ms": round(macro_lat, 2),
            "per_machine": per_m,
        }

        logger.info(
            f"=== {ens_id.upper()} MACRO RESULTS ===\n"
            f"  Macro ROC-AUC:   {macro_roc:.4f}\n"
            f"  Macro pAUC:      {macro_pauc:.4f}\n"
            f"  Macro F1:        {macro_f1:.4f}\n"
            f"  CPU Latency:     {macro_lat:.2f} ms"
        )

    # Save artifact
    out_file = RESEARCH_ARTIFACTS_DIR / "ensemble_comparison_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    logger.info(f"Saved ensemble results to {out_file}")

    # Register in experiment registry
    mgr = ExperimentManager()
    mgr.register_experiment(
        experiment_id="phase6_advanced_ensembles",
        name="Phase 6 Advanced Ensemble & Model Fusion",
        description="Comprehensive evaluation of Score Fusion (Weighted, Rank, Max) and Hierarchical Cascades",
        model_type="ensemble_fusion",
        config={
            "tested_ensembles": list(summary_report.keys()),
            "best_accuracy_ensemble": max(summary_report.keys(), key=lambda k: summary_report[k]["macro_roc_auc"]),
            "best_speed_ensemble": min(summary_report.keys(), key=lambda k: summary_report[k]["macro_cpu_mean_ms"]),
        },
    )
    mgr.update_status(
        experiment_id="phase6_advanced_ensembles",
        status="COMPLETED",
        metrics={
            "best_macro_roc_auc": max(summary_report[k]["macro_roc_auc"] for k in summary_report),
            "summary": {k: {"roc_auc": summary_report[k]["macro_roc_auc"], "latency_ms": summary_report[k]["macro_cpu_mean_ms"]} for k in summary_report},
        },
    )

    return summary_report


if __name__ == "__main__":
    run_phase6_ensembles()

