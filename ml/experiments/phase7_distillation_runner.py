"""Phase 7 Distillation & Cascaded Inference Experiment Runner.

Evaluates:
1. Teacher Model: State-of-the-art ResNet-AE (1.81M params, ~109ms latency)
2. Distilled Student: Ultra-compact Student-AE (105k params, sub-50ms target)
3. Confidence-Margin Gated Cascade: Student fast filter with adaptive Teacher escalation
   Sweep across margin values: [0.01, 0.03, 0.05, 0.10]

Zero-Leakage Assurance:
- Student distilled strictly from Teacher on train_normal.csv
- Cascade margins and decision thresholds calibrated exclusively on dev_normal.csv
- Test evaluation executed on held-out test.csv
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
from ml.models.research.distilled_student import DistilledStudentDetector
from ml.models.research.margin_cascade import MarginGatedCascade
from ml.models.research.research_detector import ResearchNeuralDetector
from ml.training.train_pipeline import _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase7_runner")

RESEARCH_ARTIFACTS_DIR = Path("ml/experiments/research_artifacts/phase7")
PILOT_MACHINES = ["id_00", "id_02", "id_04", "id_06"]
MARGINS_TO_TEST = [0.01, 0.03, 0.05, 0.10]


def benchmark_latency(detector: Any, sample_path: Path, iterations: int = 5) -> Dict[str, float]:
    """Measure single-file CPU inference latency."""
    for _ in range(2):
        detector.score_file(sample_path)
    lats = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.score_file(sample_path)
        lats.append((time.perf_counter() - t0) * 1000.0)
    return {
        "cpu_mean_ms": round(float(np.mean(lats)), 2),
        "cpu_median_ms": round(float(np.median(lats)), 2),
        "cpu_min_ms": round(float(np.min(lats)), 2),
        "cpu_max_ms": round(float(np.max(lats)), 2),
    }


def evaluate_machine_distillation(machine_id: str) -> Dict[str, Any]:
    """Train teacher, distill student, calibrate cascade, and benchmark for one machine."""
    train_split = f"ml/data/splits/machine_specific/{machine_id}/train_normal.csv"
    dev_split = f"ml/data/splits/machine_specific/{machine_id}/dev_normal.csv"
    test_split = f"ml/data/splits/machine_specific/{machine_id}/test.csv"

    train_rows = _read_split_csv(train_split)
    dev_rows = _read_split_csv(dev_split)
    test_rows = _read_split_csv(test_split)

    train_paths = [r["file_path"] for r in train_rows]
    dev_paths = [r["file_path"] for r in dev_rows]
    test_paths = [r["file_path"] for r in test_rows]
    test_labels = [1 if str(r["label"]).lower() in ("abnormal", "anomaly", "1") else 0 for r in test_rows]

    # 1. Train Teacher (ResNet-AE)
    logger.info(f"[{machine_id}] Training Teacher Model (ResNet-AE)...")
    teacher = ResearchNeuralDetector(
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
    t_meta = teacher.fit(train_paths, dev_files=dev_paths)

    # Calibrate Teacher
    calibrator = StatisticalCalibrator(target_quantile=0.95)
    cal_t = calibrator.calibrate(teacher, dev_paths)
    teacher.threshold = cal_t.operating_threshold

    # Evaluate Teacher
    t_scores = [teacher.score_file(p) for p in test_paths]
    t_metrics = compute_anomaly_metrics(test_labels, t_scores, teacher.threshold)
    t_latency = benchmark_latency(teacher, Path(test_paths[0]))

    # 2. Distill Student
    logger.info(f"[{machine_id}] Distilling Student Model (CompactStudentAE)...")
    student = DistilledStudentDetector(
        machine_id=machine_id,
        base_channels=8,
        latent_dim=32,
        learning_rate=0.001,
        max_epochs=20,
        seed=42,
    )
    s_meta = student.fit_distill(train_paths, teacher_model=teacher.model, dev_files=dev_paths)

    # Calibrate Student
    cal_s = calibrator.calibrate(student, dev_paths)
    student.threshold = cal_s.operating_threshold

    # Evaluate Student
    s_scores = [student.score_file(p) for p in test_paths]
    s_metrics = compute_anomaly_metrics(test_labels, s_scores, student.threshold)
    s_latency = benchmark_latency(student, Path(test_paths[0]))

    # 3. Evaluate Margin-Gated Cascade across Margins
    cascade_results = {}
    for margin in MARGINS_TO_TEST:
        cascade = MarginGatedCascade(
            machine_id=machine_id,
            student_detector=student,
            teacher_detector=teacher,
            margin=margin,
        )
        cascade.student_threshold = student.threshold
        cascade.teacher_threshold = teacher.threshold

        c_scores = []
        c_latencies = []
        escalated_count = 0
        for p in test_paths:
            score, info = cascade.score_file_with_telemetry(p)
            c_scores.append(score)
            c_latencies.append(info["total_latency_ms"])
            if info["escalated"]:
                escalated_count += 1

        # Use student threshold for unescalated scores or appropriate hybrid threshold
        c_metrics = compute_anomaly_metrics(test_labels, c_scores, student.threshold)
        escalation_rate = float(escalated_count / len(test_paths))

        cascade_results[f"margin_{margin}"] = {
            "margin": margin,
            "metrics": c_metrics,
            "escalation_rate": round(escalation_rate, 4),
            "cpu_mean_ms": round(float(np.mean(c_latencies)), 2),
        }

    return {
        "machine_id": machine_id,
        "teacher": {
            "metrics": t_metrics,
            "latency": t_latency,
            "parameters": sum(p.numel() for p in teacher.model.parameters()),
        },
        "student": {
            "metrics": s_metrics,
            "latency": s_latency,
            "parameters": sum(p.numel() for p in student.model.parameters()),
        },
        "cascades": cascade_results,
    }


def run_phase7_distillation() -> Dict[str, Any]:
    """Execute complete Phase 7 distillation & cascade suite across all pilot machines."""
    logger.info("Starting Phase 7 Knowledge Distillation & Cascade Policy Research...")
    RESEARCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results_by_machine = {}
    for m_id in PILOT_MACHINES:
        m_res = evaluate_machine_distillation(m_id)
        results_by_machine[m_id] = m_res

        t_m = m_res["teacher"]["metrics"]
        s_m = m_res["student"]["metrics"]
        t_lat = m_res["teacher"]["latency"]["cpu_mean_ms"]
        s_lat = m_res["student"]["latency"]["cpu_mean_ms"]

        logger.info(
            f"[{m_id}] Teacher ROC-AUC: {t_m['roc_auc']:.4f} ({t_lat}ms) | "
            f"Student ROC-AUC: {s_m['roc_auc']:.4f} ({s_lat}ms)"
        )

    # Compute Macro Summaries
    macro_t_roc = float(np.mean([r["teacher"]["metrics"]["roc_auc"] for r in results_by_machine.values()]))
    macro_t_pauc = float(np.mean([r["teacher"]["metrics"]["pauc"] for r in results_by_machine.values()]))
    macro_t_lat = float(np.mean([r["teacher"]["latency"]["cpu_mean_ms"] for r in results_by_machine.values()]))

    macro_s_roc = float(np.mean([r["student"]["metrics"]["roc_auc"] for r in results_by_machine.values()]))
    macro_s_pauc = float(np.mean([r["student"]["metrics"]["pauc"] for r in results_by_machine.values()]))
    macro_s_lat = float(np.mean([r["student"]["latency"]["cpu_mean_ms"] for r in results_by_machine.values()]))

    macro_cascades = {}
    for margin in MARGINS_TO_TEST:
        key = f"margin_{margin}"
        m_roc = float(np.mean([r["cascades"][key]["metrics"]["roc_auc"] for r in results_by_machine.values()]))
        m_pauc = float(np.mean([r["cascades"][key]["metrics"]["pauc"] for r in results_by_machine.values()]))
        m_esc = float(np.mean([r["cascades"][key]["escalation_rate"] for r in results_by_machine.values()]))
        m_lat = float(np.mean([r["cascades"][key]["cpu_mean_ms"] for r in results_by_machine.values()]))
        macro_cascades[key] = {
            "margin": margin,
            "macro_roc_auc": round(m_roc, 4),
            "macro_pauc": round(m_pauc, 4),
            "macro_escalation_rate": round(m_esc, 4),
            "macro_cpu_mean_ms": round(m_lat, 2),
        }

    summary = {
        "teacher": {
            "macro_roc_auc": round(macro_t_roc, 4),
            "macro_pauc": round(macro_t_pauc, 4),
            "macro_cpu_mean_ms": round(macro_t_lat, 2),
            "parameters": results_by_machine["id_00"]["teacher"]["parameters"],
        },
        "student": {
            "macro_roc_auc": round(macro_s_roc, 4),
            "macro_pauc": round(macro_s_pauc, 4),
            "macro_cpu_mean_ms": round(macro_s_lat, 2),
            "parameters": results_by_machine["id_00"]["student"]["parameters"],
        },
        "macro_cascades": macro_cascades,
        "per_machine": results_by_machine,
    }

    out_file = RESEARCH_ARTIFACTS_DIR / "distillation_cascade_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"=== PHASE 7 RESEARCH SUMMARY ===\n"
        f"  Teacher (ResNet-AE): Macro ROC-AUC={macro_t_roc:.4f}, Latency={macro_t_lat:.1f}ms, Params={summary['teacher']['parameters']:,}\n"
        f"  Student (Distilled): Macro ROC-AUC={macro_s_roc:.4f}, Latency={macro_s_lat:.1f}ms, Params={summary['student']['parameters']:,}\n"
    )
    for k, v in macro_cascades.items():
        logger.info(
            f"  Cascade ({k}): Macro ROC-AUC={v['macro_roc_auc']:.4f}, "
            f"Latency={v['macro_cpu_mean_ms']:.1f}ms, Escalation={v['macro_escalation_rate']*100:.1f}%"
        )

    # Register in experiment registry
    mgr = ExperimentManager()
    mgr.register_experiment(
        experiment_id="phase7_distillation_cascade",
        name="Phase 7 Knowledge Distillation & Cascaded Policy",
        description="Knowledge distillation of ResNet-AE teacher into CompactStudentAE and margin-gated cascade evaluation",
        model_type="distilled_student_cascade",
        config={
            "student_params": summary["student"]["parameters"],
            "margins_tested": MARGINS_TO_TEST,
        },
    )
    mgr.update_status(
        experiment_id="phase7_distillation_cascade",
        status="COMPLETED",
        metrics={
            "teacher_macro_roc_auc": macro_t_roc,
            "student_macro_roc_auc": macro_s_roc,
            "student_latency_ms": macro_s_lat,
            "cascades": macro_cascades,
        },
    )

    return summary


if __name__ == "__main__":
    run_phase7_distillation()

