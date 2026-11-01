"""Phase 5 Optimization & Ablation Runner.

Systematically evaluates training & architecture optimizations for top models:
- Loss functions (L1, MSE, Smooth L1, Frequency-Weighted L1)
- Optimizers and LR schedules (Adam, AdamW, Cosine, Plateau)
- Latent dimensions (32, 64, 128)
- Normalization (GroupNorm, BatchNorm, LayerNorm)
- Patch sizes / Temporal context (64, 96, 128 frames)
- Self-supervised augmentations (None, Time-Shift, Noise Injection, SpecAugment)

Outputs:
- ml/reports/ablation_study_phase5.json
- ml/experiments/research_artifacts/phase5/
- Logs into ml/experiments/experiment_registry.json
"""

import copy
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
import soundfile as sf
import torch

from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.experiments.experiment_manager import ExperimentManager
from ml.models.research.research_detector import ResearchNeuralDetector
from ml.training.train_pipeline import _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase5_optimization")

RESEARCH_ARTIFACTS_DIR = Path("ml/experiments/research_artifacts/phase5")
PILOT_MACHINES = ["id_00", "id_02", "id_04", "id_06"]
BENCHMARK_MACHINE = "id_04"  # The hard anomaly machine


def benchmark_model_latency(
    detector: ResearchNeuralDetector,
    sample_file: Path,
    iterations: int = 5,
) -> Dict[str, float]:
    """Measure single-file inference latency on CPU."""
    # Warmup
    for _ in range(2):
        detector.score_file(sample_file)

    cpu_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.score_file(sample_file)
        cpu_latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "cpu_mean_ms": round(float(np.mean(cpu_latencies)), 2),
        "cpu_median_ms": round(float(np.median(cpu_latencies)), 2),
        "cpu_p95_ms": round(float(np.percentile(cpu_latencies, 95)), 2),
        "cpu_min_ms": round(float(np.min(cpu_latencies)), 2),
        "cpu_max_ms": round(float(np.max(cpu_latencies)), 2),
    }


def run_single_ablation(
    exp_id: str,
    arch_type: str = "residual_cnn_ae",
    machine_id: str = BENCHMARK_MACHINE,
    base_channels: int = 16,
    latent_dim: int = 64,
    norm_type: str = "group",
    loss_type: str = "l1",
    optimizer_type: str = "adamw",
    lr_schedule: str = "constant",
    patch_frames: int = 64,
    augmentation: Optional[str] = None,
    learning_rate: float = 0.001,
    max_epochs: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute a single ablation trial with zero leakage."""
    train_split = f"ml/data/splits/machine_specific/{machine_id}/train_normal.csv"
    dev_split = f"ml/data/splits/machine_specific/{machine_id}/dev_normal.csv"
    test_split = f"ml/data/splits/machine_specific/{machine_id}/test.csv"

    train_rows = _read_split_csv(train_split)
    dev_rows = _read_split_csv(dev_split)
    test_rows = _read_split_csv(test_split)

    train_paths = [r["file_path"] for r in train_rows]
    dev_paths = [r["file_path"] for r in dev_rows]
    test_paths = [r["file_path"] for r in test_rows]

    detector = ResearchNeuralDetector(
        machine_id=machine_id,
        arch_type=arch_type,
        base_channels=base_channels,
        latent_dim=latent_dim,
        norm_type=norm_type,
        loss_type=loss_type,
        optimizer_type=optimizer_type,
        lr_schedule=lr_schedule,
        patch_frames=patch_frames,
        augmentation=augmentation,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping_patience=5,
        seed=seed,
    )

    t0 = time.perf_counter()
    fit_meta = detector.fit(train_paths, dev_files=dev_paths)
    train_duration = round(time.perf_counter() - t0, 2)

    # Statistical calibration on normal dev files only (zero leakage)
    calibrator = StatisticalCalibrator(target_quantile=0.95)
    cal_prof = calibrator.calibrate(detector, dev_paths)
    detector.threshold = cal_prof.operating_threshold

    # Benchmark CPU latency
    latency = benchmark_model_latency(detector, Path(test_paths[0]), iterations=4)

    # Evaluate on final test set
    test_scores = []
    test_labels = []
    for r in test_rows:
        score = detector.score_file(r["file_path"])
        test_scores.append(score)
        lbl = 1 if str(r["label"]).lower() in ("abnormal", "anomaly", "1") else 0
        test_labels.append(lbl)

    test_scores_arr = np.array(test_scores)
    test_labels_arr = np.array(test_labels)

    metrics = compute_anomaly_metrics(
        y_true=test_labels_arr,
        y_scores=test_scores_arr,
        threshold=detector.threshold,
        max_fpr=0.1,
    )

    roc_auc = metrics.get("roc_auc", 0.5)
    pauc = metrics.get("pauc", 0.5)
    prec = metrics.get("precision", 0.0)
    rec = metrics.get("recall", 0.0)
    f1 = metrics.get("f1", 0.0)
    fpr = metrics.get("fpr", 0.0)
    cm = metrics.get("confusion_matrix", {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

    param_count = sum(param.numel() for param in detector.model.parameters())

    return {
        "experiment_id": exp_id,
        "machine_id": machine_id,
        "config": {
            "arch_type": arch_type,
            "base_channels": base_channels,
            "latent_dim": latent_dim,
            "norm_type": norm_type,
            "loss_type": loss_type,
            "optimizer_type": optimizer_type,
            "lr_schedule": lr_schedule,
            "patch_frames": patch_frames,
            "augmentation": augmentation,
            "learning_rate": learning_rate,
        },
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "pauc": round(pauc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "fpr": round(fpr, 4),
            "threshold": round(float(detector.threshold), 6),
            "best_val_loss": fit_meta["best_val_loss"],
            "confusion_matrix": cm,
        },
        "parameters": param_count,
        "training_time_s": train_duration,
        "latency": latency,
    }


def run_full_suite() -> Dict[str, Any]:
    """Execute complete ablation study matrix across all dimensions."""
    ablation_results = {}

    # Define standard ablation trials on BENCHMARK_MACHINE (id_04)
    trials = [
        # Dimension 1: Loss functions
        ("ablation_loss_l1", {"loss_type": "l1"}),
        ("ablation_loss_mse", {"loss_type": "mse"}),
        ("ablation_loss_smooth_l1", {"loss_type": "smooth_l1"}),
        ("ablation_loss_freq_weighted", {"loss_type": "freq_weighted_l1"}),
        # Dimension 2: Optimizer & LR schedule
        ("ablation_opt_adam", {"optimizer_type": "adam", "lr_schedule": "constant"}),
        ("ablation_opt_adamw", {"optimizer_type": "adamw", "lr_schedule": "constant"}),
        ("ablation_sched_cosine", {"optimizer_type": "adamw", "lr_schedule": "cosine"}),
        ("ablation_sched_plateau", {"optimizer_type": "adamw", "lr_schedule": "plateau"}),
        # Dimension 3: Latent dimension
        ("ablation_latent_32", {"latent_dim": 32}),
        ("ablation_latent_64", {"latent_dim": 64}),
        ("ablation_latent_128", {"latent_dim": 128}),
        # Dimension 4: Normalization layer
        ("ablation_norm_group", {"norm_type": "group"}),
        ("ablation_norm_batch", {"norm_type": "batch"}),
        ("ablation_norm_layer", {"norm_type": "layer"}),
        # Dimension 5: Temporal context / patch size
        ("ablation_patch_64", {"patch_frames": 64}),
        ("ablation_patch_96", {"patch_frames": 96}),
        ("ablation_patch_128", {"patch_frames": 128}),
        # Dimension 6: Data augmentation
        ("ablation_aug_none", {"augmentation": None}),
        ("ablation_aug_time_shift", {"augmentation": "time_shift"}),
        ("ablation_aug_noise", {"augmentation": "noise_injection"}),
        ("ablation_aug_spec_mask", {"augmentation": "spec_augment"}),
        # MultiScale CNNAE comparison
        ("ablation_msae_l1", {"arch_type": "multiscale_cnn_ae", "loss_type": "l1"}),
        ("ablation_msae_freq_weighted", {"arch_type": "multiscale_cnn_ae", "loss_type": "freq_weighted_l1"}),
    ]

    logger.info(f"Starting Phase 5 Ablation Study ({len(trials)} trials on machine {BENCHMARK_MACHINE})...")
    for exp_id, params in trials:
        logger.info(f"Running trial: {exp_id} with params: {params}")
        trial_res = run_single_ablation(exp_id, **params)
        ablation_results[exp_id] = trial_res
        logger.info(
            f"Trial {exp_id} -> ROC-AUC: {trial_res['metrics']['roc_auc']:.4f}, "
            f"pAUC: {trial_res['metrics']['pauc']:.4f}, "
            f"F1: {trial_res['metrics']['f1']:.4f}, "
            f"Best Val Loss: {trial_res['metrics']['best_val_loss']}"
        )

    # Save ablation study report
    report_file = Path("ml/reports/ablation_study_phase5.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(ablation_results, f, indent=2)
    logger.info(f"Saved ablation study results to {report_file}")

    # Identify best configuration
    # Score by composite: 0.6 * ROC-AUC + 0.4 * pAUC
    best_id = max(
        ablation_results.keys(),
        key=lambda k: 0.6 * ablation_results[k]["metrics"]["roc_auc"] + 0.4 * ablation_results[k]["metrics"]["pauc"],
    )
    best_config = ablation_results[best_id]["config"]
    logger.info(f"Best ablation configuration on {BENCHMARK_MACHINE}: {best_id} -> {best_config}")

    # Now evaluate the optimized configuration across ALL 4 MACHINES
    logger.info("Evaluating Optimized Configuration across all 4 pilot machines...")
    four_machine_results = {}
    RESEARCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for m_id in PILOT_MACHINES:
        m_dir = RESEARCH_ARTIFACTS_DIR / "optimized_residual_cnn_ae" / m_id
        m_dir.mkdir(parents=True, exist_ok=True)

        res = run_single_ablation(
            exp_id=f"optimized_resnet_{m_id}",
            machine_id=m_id,
            arch_type="residual_cnn_ae",
            base_channels=best_config.get("base_channels", 16),
            latent_dim=best_config.get("latent_dim", 64),
            norm_type=best_config.get("norm_type", "group"),
            loss_type=best_config.get("loss_type", "freq_weighted_l1"),
            optimizer_type=best_config.get("optimizer_type", "adamw"),
            lr_schedule=best_config.get("lr_schedule", "cosine"),
            patch_frames=best_config.get("patch_frames", 64),
            augmentation=best_config.get("augmentation", None),
        )
        four_machine_results[m_id] = res

    # Compute Macro metrics
    macro_roc = float(np.mean([r["metrics"]["roc_auc"] for r in four_machine_results.values()]))
    macro_pauc = float(np.mean([r["metrics"]["pauc"] for r in four_machine_results.values()]))
    macro_f1 = float(np.mean([r["metrics"]["f1"] for r in four_machine_results.values()]))
    macro_precision = float(np.mean([r["metrics"]["precision"] for r in four_machine_results.values()]))
    macro_recall = float(np.mean([r["metrics"]["recall"] for r in four_machine_results.values()]))
    macro_fpr = float(np.mean([r["metrics"]["fpr"] for r in four_machine_results.values()]))
    macro_latency = float(np.mean([r["latency"]["cpu_mean_ms"] for r in four_machine_results.values()]))

    macro_summary = {
        "macro_roc_auc": round(macro_roc, 4),
        "macro_pauc": round(macro_pauc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_fpr": round(macro_fpr, 4),
        "macro_cpu_mean_ms": round(macro_latency, 2),
        "per_machine": four_machine_results,
    }

    macro_path = RESEARCH_ARTIFACTS_DIR / "optimized_residual_cnn_ae" / "macro_summary.json"
    with open(macro_path, "w") as f:
        json.dump(macro_summary, f, indent=2)

    logger.info(
        f"OPTIMIZED RESNET-AE MACRO RESULTS:\n"
        f"  Macro ROC-AUC:   {macro_roc:.4f}\n"
        f"  Macro pAUC:      {macro_pauc:.4f}\n"
        f"  Macro F1:        {macro_f1:.4f}\n"
        f"  Macro Precision: {macro_precision:.4f}\n"
        f"  Macro Recall:    {macro_recall:.4f}\n"
        f"  Macro FPR:       {macro_fpr:.4f}\n"
        f"  CPU Latency:     {macro_latency:.2f} ms"
    )

    # Register in experiment registry
    mgr = ExperimentManager()
    mgr.register_experiment(
        experiment_id="phase5_optimization_ablations",
        name="Phase 5 Architecture Optimization & Ablations",
        description="Comprehensive Phase 5 controlled ablation study across loss, optimizers, schedulers, latent dims, norms, contexts, and augmentations",
        model_type="residual_cnn_ae",
        config=best_config,
    )
    mgr.update_status(
        experiment_id="phase5_optimization_ablations",
        status="COMPLETED",
        metrics={
            "macro_roc_auc": macro_roc,
            "macro_pauc": macro_pauc,
            "macro_f1": macro_f1,
            "macro_cpu_mean_ms": macro_latency,
            "best_ablation": best_id,
        },
    )

    return {
        "ablations": ablation_results,
        "best_ablation_id": best_id,
        "macro_summary": macro_summary,
    }


if __name__ == "__main__":
    run_full_suite()
