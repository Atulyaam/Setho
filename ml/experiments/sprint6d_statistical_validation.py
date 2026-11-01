"""Sprint 6D Phase 3: Statistical Validation Runner.

Evaluates multi-seed stability and statistical confidence for Few-Shot Latent Calibration:
- Seeds: [42, 101, 202, 303, 404]
- Shot budgets: N in {2, 5, 10}
- Machines: [id_00, id_02, id_04, id_06] (Fan Unseen Held-out split)
- Measures: ROC-AUC, pAUC, F1, Precision, Recall, FPR
- Computes: Mean, Std Dev, 95% Confidence Intervals
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.calibration.few_shot_calibrator import FewShotMachineCalibrator
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.models.research.research_detector import ResearchNeuralDetector
from ml.training.train_pipeline import _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sprint6d_statistical_validation")

SEEDS = [42, 101, 202, 303, 404]
SHOT_BUDGETS = [2, 5, 10]
MACHINES = ["id_00", "id_02", "id_04", "id_06"]
OUTPUT_PATH = Path("ml/reports/sprint_6d_statistical_confidence.json")


def load_detector() -> ResearchNeuralDetector:
    """Load baseline ResNet-AE detector."""
    det = ResearchNeuralDetector(machine_id="id_00", arch_type="residual_cnn_ae", base_channels=16, latent_dim=64)
    ckpt = torch.load("ml/experiments/research_artifacts/phase4/residual_cnn_ae/id_00/model.pt", map_location="cpu", weights_only=False)
    det.model.load_state_dict(ckpt)
    det.train_mean = -36.0
    det.train_std = 9.0
    det.model.eval()
    return det


def evaluate_machine_seed(
    detector: ResearchNeuralDetector,
    machine_id: str,
    n_shots: int,
    seed: int,
) -> Dict[str, Any]:
    """Calibrate and evaluate detector for a single machine and seed."""
    split_dir = Path(f"ml/data/splits/unseen_machine/fan/heldout_{machine_id}")
    test_rows = _read_split_csv(split_dir / "test_unseen_machine.csv")
    test_paths = [r["file_path"] for r in test_rows]
    test_labels = [1 if str(r["label"]).lower() in ("abnormal", "anomaly", "1") else 0 for r in test_rows]

    # Available normal files for calibration from diverse_machine_specific
    dev_csv = Path(f"ml/data/splits/diverse_machine_specific/fan/{machine_id}/dev_normal.csv")
    train_csv = Path(f"ml/data/splits/diverse_machine_specific/fan/{machine_id}/train_normal.csv")
    
    # Pool all normal files for calibration sampling
    normal_pool = []
    if dev_csv.exists():
        normal_pool.extend([r["file_path"] for r in _read_split_csv(dev_csv) if str(r.get("label", "normal")).lower() == "normal"])
    if train_csv.exists():
        normal_pool.extend([r["file_path"] for r in _read_split_csv(train_csv) if str(r.get("label", "normal")).lower() == "normal"])

    # Deterministic sampling based on seed
    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(len(normal_pool), size=min(n_shots, len(normal_pool)), replace=False)
    cal_paths = [normal_pool[i] for i in sampled_indices]

    calibrator = FewShotMachineCalibrator(
        base_detector=detector,
        method="mahalanobis_latent",
        n_shots=len(cal_paths),
        target_quantile=0.95,
    )
    t0 = time.perf_counter()
    calibrator.calibrate(cal_paths)
    cal_time_s = time.perf_counter() - t0

    scores = [calibrator.score_file(p) for p in test_paths]
    thresh = calibrator.operating_threshold
    m = compute_anomaly_metrics(test_labels, scores, thresh)

    return {
        "roc_auc": round(float(m["roc_auc"]), 4),
        "pauc": round(float(m["pauc"]), 4),
        "f1": round(float(m["f1"]), 4),
        "precision": round(float(m["precision"]), 4),
        "recall": round(float(m["recall"]), 4),
        "fpr": round(float(m["fpr"]), 4),
        "cal_time_s": round(float(cal_time_s), 4),
    }


def compute_confidence_intervals(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Compute mean, std, and 95% confidence interval bounds."""
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    n = len(arr)
    # 1.96 standard normal approximation for 95% CI
    margin = 1.96 * (std_val / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ci_lower": round(mean_val - margin, 4),
        "ci_upper": round(mean_val + margin, 4),
    }


def main():
    logger.info("Executing Sprint 6D Phase 3: Multi-Seed Statistical Validation...")
    detector = load_detector()

    results_by_shot = {}

    for n_shots in SHOT_BUDGETS:
        logger.info(f"Evaluating Calibration Shot Budget N = {n_shots} across {len(SEEDS)} seeds...")
        seed_runs = []

        for seed in SEEDS:
            machine_metrics = {}
            aucs, paucs, f1s, precs, recs, fprs, times = [], [], [], [], [], [], []

            for mid in MACHINES:
                res = evaluate_machine_seed(detector, mid, n_shots, seed)
                machine_metrics[mid] = res
                aucs.append(res["roc_auc"])
                paucs.append(res["pauc"])
                f1s.append(res["f1"])
                precs.append(res["precision"])
                recs.append(res["recall"])
                fprs.append(res["fpr"])
                times.append(res["cal_time_s"])

            macro_res = {
                "macro_roc_auc": round(float(np.mean(aucs)), 4),
                "macro_pauc": round(float(np.mean(paucs)), 4),
                "macro_f1": round(float(np.mean(f1s)), 4),
                "macro_precision": round(float(np.mean(precs)), 4),
                "macro_recall": round(float(np.mean(recs)), 4),
                "macro_fpr": round(float(np.mean(fprs)), 4),
                "avg_cal_time_s": round(float(np.mean(times)), 4),
                "per_machine": machine_metrics,
            }
            seed_runs.append({"seed": seed, "macro": macro_res})
            logger.info(f"  [N={n_shots}, Seed={seed}] Macro ROC-AUC: {macro_res['macro_roc_auc']:.4f} | F1: {macro_res['macro_f1']:.4f}")

        # Compute multi-seed statistical distributions
        macro_aucs = [r["macro"]["macro_roc_auc"] for r in seed_runs]
        macro_paucs = [r["macro"]["macro_pauc"] for r in seed_runs]
        macro_f1s = [r["macro"]["macro_f1"] for r in seed_runs]
        macro_precs = [r["macro"]["macro_precision"] for r in seed_runs]
        macro_recs = [r["macro"]["macro_recall"] for r in seed_runs]
        macro_fprs = [r["macro"]["macro_fpr"] for r in seed_runs]
        cal_times = [r["macro"]["avg_cal_time_s"] for r in seed_runs]

        # Per-machine aggregations
        per_machine_stats = {}
        for mid in MACHINES:
            m_aucs = [r["macro"]["per_machine"][mid]["roc_auc"] for r in seed_runs]
            m_f1s = [r["macro"]["per_machine"][mid]["f1"] for r in seed_runs]
            per_machine_stats[mid] = {
                "roc_auc_stats": compute_confidence_intervals(m_aucs),
                "f1_stats": compute_confidence_intervals(m_f1s),
            }

        results_by_shot[f"N_{n_shots}"] = {
            "n_shots": n_shots,
            "seed_count": len(SEEDS),
            "macro_roc_auc": compute_confidence_intervals(macro_aucs),
            "macro_pauc": compute_confidence_intervals(macro_paucs),
            "macro_f1": compute_confidence_intervals(macro_f1s),
            "macro_precision": compute_confidence_intervals(macro_precs),
            "macro_recall": compute_confidence_intervals(macro_recs),
            "macro_fpr": compute_confidence_intervals(macro_fprs),
            "cal_time_s": compute_confidence_intervals(cal_times),
            "per_machine_stats": per_machine_stats,
            "individual_seed_runs": seed_runs,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results_by_shot, f, indent=2)

    logger.info(f"Statistical validation results successfully recorded to {OUTPUT_PATH}")
    print("\n=== Multi-Seed Statistical Validation Summary ===")
    for shot_k, data in results_by_shot.items():
        auc_s = data["macro_roc_auc"]
        print(f"[{shot_k}] Macro ROC-AUC: {auc_s['mean']} +/- {auc_s['std']} (95% CI: [{auc_s['ci_lower']}, {auc_s['ci_upper']}])")


if __name__ == "__main__":
    main()

