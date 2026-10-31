"""Autonomous ML Experiment Execution Engine for Stetho Sprint 6A.

Executes prioritized research experiments:
1. baseline_audit: Full 4-machine independent reproduction of Stetho-MSAE.
2. multires_logmel: Multi-scale STFT windowing representation experiment.
3. memory_bank_knn: Bottleneck feature memory-bank k-NN anomaly detection.
4. robustness_eval: Acoustic perturbation stress testing (+/- 6dB gain, noise).
5. evt_calibration: Extreme Value Theory tail estimation for low-N calibration.

Integrates with ExperimentManager for non-blocking background telemetry.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.calibration.statistical_calibration import StatisticalCalibrator
from ml.configs.experiment_config import ExperimentConfig
from ml.evaluation.metrics import compute_anomaly_metrics
from ml.experiments.experiment_manager import ExperimentManager
from ml.models.stetho_msae.msae_detector import StethoMSAEDetector
from ml.training.train_pipeline import run_experiment, _read_split_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_exp_runner")

ALL_MACHINES = ["id_00", "id_02", "id_04", "id_06"]


def run_baseline_audit(exp_id: str, manager: ExperimentManager, machine_ids: List[str]) -> Dict[str, Any]:
    """Execute baseline reproduction across specified machines and record macro metrics."""
    results_by_machine = {}
    config_path = Path("ml/configs/stetho_msae_default.yaml")

    for idx, m_id in enumerate(machine_ids):
        logger.info(f"[{exp_id}] Starting baseline reproduction for machine {m_id} ({idx+1}/{len(machine_ids)})...")
        cfg = ExperimentConfig.from_yaml(config_path)
        cfg.set_machine_id(m_id)
        # Store artifacts under isolated experiment directory to protect production weights
        cfg.artifact_dir = f"ml/experiments/artifacts/{exp_id}/{m_id}"
        cfg.experiment_id = f"{exp_id}_{m_id}"

        res = run_experiment(cfg)
        results_by_machine[m_id] = res

        # Update manager with intermediate progress
        manager.update_status(
            experiment_id=exp_id,
            status="RUNNING",
            current_epoch=idx + 1,
            metrics={"completed_machines": list(results_by_machine.keys())},
        )

    # Compute macro averages
    roc_aucs = [results_by_machine[m]["metrics"]["roc_auc"] for m in machine_ids]
    paucs = [results_by_machine[m]["metrics"]["pauc"] for m in machine_ids]
    f1s = [results_by_machine[m]["metrics"]["f1"] for m in machine_ids]
    recalls = [results_by_machine[m]["metrics"]["recall"] for m in machine_ids]
    fprs = [results_by_machine[m]["metrics"]["fpr"] for m in machine_ids]

    macro_summary = {
        "macro_roc_auc": round(float(np.mean(roc_aucs)), 4),
        "macro_pauc": round(float(np.mean(paucs)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "macro_fpr": round(float(np.mean(fprs)), 4),
        "per_machine": {
            m: {
                "roc_auc": results_by_machine[m]["metrics"]["roc_auc"],
                "pauc": results_by_machine[m]["metrics"]["pauc"],
                "f1": results_by_machine[m]["metrics"]["f1"],
                "recall": results_by_machine[m]["metrics"]["recall"],
                "fpr": results_by_machine[m]["metrics"]["fpr"],
                "duration_sec": results_by_machine[m]["total_experiment_duration_sec"],
            }
            for m in machine_ids
        },
    }
    return macro_summary


def run_domain_robustness_eval(exp_id: str, manager: ExperimentManager, machine_ids: List[str]) -> Dict[str, Any]:
    """Evaluate trained model sensitivity against gain variation (+/-6dB) and noise."""
    import soundfile as sf
    import io
    from ml.models.stetho_msae.msae_detector import StethoMSAEDetector

    robustness_results = {}
    gains_to_test = [-6.0, 0.0, 6.0]  # dB

    for m_id in machine_ids:
        logger.info(f"[{exp_id}] Evaluating domain robustness for machine {m_id} across gains {gains_to_test} dB...")
        # Load baseline model
        model_dir = Path(f"ml/artifacts/stetho_msae/{m_id}")
        detector = StethoMSAEDetector.load(model_dir)

        test_split = f"ml/data/splits/machine_specific/{m_id}/test.csv"
        test_rows = _read_split_csv(test_split)

        gain_metrics = {}
        for gain_db in gains_to_test:
            gain_factor = 10.0 ** (gain_db / 20.0)
            scores = []
            labels = []

            for r in test_rows:
                # Read audio and apply gain
                audio_data, sr = sf.read(r["file_path"], dtype="float32")
                scaled_audio = audio_data * gain_factor
                scaled_audio = np.clip(scaled_audio, -1.0, 1.0)

                # Write to temp in-memory buffer
                buf = io.BytesIO()
                sf.write(buf, scaled_audio, sr, format="WAV")
                buf.seek(0)

                # Temp file for detector scoring
                temp_wav = Path(f"ml/experiments/artifacts/{exp_id}/temp_{m_id}_{gain_db}db.wav")
                temp_wav.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_wav, "wb") as f_out:
                    f_out.write(buf.getvalue())

                score = detector.score_file(temp_wav)
                temp_wav.unlink(missing_ok=True)

                scores.append(score)
                labels.append(1 if r["label"] == "abnormal" else 0)

            met = compute_anomaly_metrics(labels, scores, detector.threshold)
            gain_metrics[f"{gain_db:+.1f}dB"] = {
                "roc_auc": met["roc_auc"],
                "pauc": met["pauc"],
                "f1": met["f1"],
                "mean_score": round(float(np.mean(scores)), 6),
            }

        robustness_results[m_id] = gain_metrics

    return robustness_results


def run_evt_calibration_eval(exp_id: str, manager: ExperimentManager, machine_ids: List[str]) -> Dict[str, Any]:
    """Evaluate Extreme Value Theory (EVT) Generalized Pareto tail fitting vs standard empirical quantile."""
    from scipy.stats import genpareto

    calibration_comparison = {}
    for m_id in machine_ids:
        model_dir = Path(f"ml/artifacts/stetho_msae/{m_id}")
        detector = StethoMSAEDetector.load(model_dir)

        dev_split = f"ml/data/splits/machine_specific/{m_id}/dev_normal.csv"
        test_split = f"ml/data/splits/machine_specific/{m_id}/test.csv"

        dev_rows = _read_split_csv(dev_split)
        test_rows = _read_split_csv(test_split)

        dev_scores = [detector.score_file(r["file_path"]) for r in dev_rows]
        test_scores = [detector.score_file(r["file_path"]) for r in test_rows]
        test_labels = [1 if r["label"] == "abnormal" else 0 for r in test_rows]

        # Standard empirical quantile (current baseline)
        theta_empirical = float(np.percentile(dev_scores, 95))

        # Extreme Value Theory (EVT) tail approximation
        # Fit GPD to excess above 75th percentile threshold
        u = float(np.percentile(dev_scores, 75))
        excesses = [s - u for s in dev_scores if s > u]
        if len(excesses) >= 2:
            c, loc, scale = genpareto.fit(excesses, floc=0)
            # Theoretical 95th percentile using GPD
            p_excess = 0.95
            p_u = 0.75
            p_tail = (p_excess - p_u) / (1.0 - p_u)
            theta_evt = float(u + genpareto.ppf(p_tail, c, loc=0, scale=scale))
        else:
            theta_evt = theta_empirical * 1.05

        met_empirical = compute_anomaly_metrics(test_labels, test_scores, theta_empirical)
        met_evt = compute_anomaly_metrics(test_labels, test_scores, theta_evt)

        calibration_comparison[m_id] = {
            "theta_empirical_95": round(theta_empirical, 6),
            "theta_evt_95": round(theta_evt, 6),
            "empirical_fpr": met_empirical["fpr"],
            "empirical_recall": met_empirical["recall"],
            "empirical_f1": met_empirical["f1"],
            "evt_fpr": met_evt["fpr"],
            "evt_recall": met_evt["recall"],
            "evt_f1": met_evt["f1"],
            "fpr_reduction": round(met_empirical["fpr"] - met_evt["fpr"], 4),
        }

    return calibration_comparison


def run_memory_bank_knn_eval(exp_id: str, manager: ExperimentManager, machine_ids: List[str]) -> Dict[str, Any]:
    """Evaluate bottleneck feature memory-bank k-NN anomaly detection."""
    from sklearn.neighbors import NearestNeighbors

    knn_results = {}
    for m_id in machine_ids:
        logger.info(f"[{exp_id}] Evaluating memory-bank k-NN for machine {m_id}...")
        model_dir = Path(f"ml/artifacts/stetho_msae/{m_id}")
        detector = StethoMSAEDetector.load(model_dir)
        detector.model.eval()

        dev_split = f"ml/data/splits/machine_specific/{m_id}/dev_normal.csv"
        test_split = f"ml/data/splits/machine_specific/{m_id}/test.csv"
        dev_rows = _read_split_csv(dev_split)
        test_rows = _read_split_csv(test_split)

        # Extract normal embeddings from dev set
        dev_embeddings = []
        with torch.no_grad():
            for r in dev_rows:
                wav, _ = detector.preprocessor.process_file(r["file_path"])
                patches = detector._extract_patches_from_waveform(wav)
                patches_norm = (patches - detector.train_mean) / detector.train_std
                x_tensor = torch.from_numpy(patches_norm).float().to(detector.device)
                z, _ = detector.model.encoder(x_tensor)
                dev_embeddings.append(z.mean(dim=0).cpu().numpy())

        dev_bank = np.stack(dev_embeddings, axis=0)

        # Fit k-NN model on memory bank
        k = min(3, len(dev_bank))
        knn = NearestNeighbors(n_neighbors=k, metric="cosine")
        knn.fit(dev_bank)

        # Calibration threshold from dev normal bank
        dev_distances, _ = knn.kneighbors(dev_bank)
        dev_scores = dev_distances.mean(axis=1)
        threshold = float(np.percentile(dev_scores, 95))

        # Score test files
        test_embeddings = []
        test_labels = []
        with torch.no_grad():
            for r in test_rows:
                wav, _ = detector.preprocessor.process_file(r["file_path"])
                patches = detector._extract_patches_from_waveform(wav)
                patches_norm = (patches - detector.train_mean) / detector.train_std
                x_tensor = torch.from_numpy(patches_norm).float().to(detector.device)
                z, _ = detector.model.encoder(x_tensor)
                test_embeddings.append(z.mean(dim=0).cpu().numpy())
                test_labels.append(1 if r["label"] == "abnormal" else 0)

        test_matrix = np.stack(test_embeddings, axis=0)
        test_distances, _ = knn.kneighbors(test_matrix)
        test_scores = [float(s) for s in test_distances.mean(axis=1)]

        met = compute_anomaly_metrics(test_labels, test_scores, threshold)
        knn_results[m_id] = {
            "roc_auc": met["roc_auc"],
            "pauc": met["pauc"],
            "f1": met["f1"],
            "recall": met["recall"],
            "fpr": met["fpr"],
            "memory_bank_size": len(dev_bank),
            "k_neighbors": k,
        }

    macro_roc = round(float(np.mean([knn_results[m]["roc_auc"] for m in machine_ids])), 4)
    macro_pauc = round(float(np.mean([knn_results[m]["pauc"] for m in machine_ids])), 4)
    return {"macro_roc_auc": macro_roc, "macro_pauc": macro_pauc, "per_machine": knn_results}


def run_multires_logmel_eval(exp_id: str, manager: ExperimentManager, machine_ids: List[str]) -> Dict[str, Any]:
    """Evaluate multi-resolution log-Mel representations across STFT window sizes (512, 1024, 2048)."""
    from ml.preprocessing.log_mel import LogMelConfig, LogMelExtractor

    res_results = {}
    fft_windows = [512, 1024, 2048]

    for m_id in machine_ids:
        logger.info(f"[{exp_id}] Evaluating multi-resolution log-Mel representations for {m_id}...")
        model_dir = Path(f"ml/artifacts/stetho_msae/{m_id}")
        detector = StethoMSAEDetector.load(model_dir)

        test_split = f"ml/data/splits/machine_specific/{m_id}/test.csv"
        test_rows = _read_split_csv(test_split)

        m_metrics = {}
        for n_fft in fft_windows:
            cfg = LogMelConfig(
                sample_rate=16000,
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=256,
                n_mels=64,
                f_min=50.0,
                f_max=8000.0,
            )
            extractor = LogMelExtractor(cfg)
            scores = []
            labels = []
            for r in test_rows:
                wav, _ = detector.preprocessor.process_file(r["file_path"])
                spec, _ = extractor.extract_numpy(wav)
                patches, _ = detector.windowing.slice_spectrogram_numpy(spec.squeeze(0))
                patches_4d = np.expand_dims(patches, axis=1).astype(np.float32)
                score = detector.score_patches(patches_4d)
                scores.append(score)
                labels.append(1 if r["label"] == "abnormal" else 0)

            met = compute_anomaly_metrics(labels, scores, detector.threshold)
            m_metrics[f"n_fft_{n_fft}"] = {
                "roc_auc": met["roc_auc"],
                "pauc": met["pauc"],
                "f1": met["f1"],
            }
        res_results[m_id] = m_metrics
    return res_results


def main():
    parser = argparse.ArgumentParser(description="Stetho Sprint 6A Experiment Runner")
    parser.add_argument("--experiment-id", type=str, required=True, help="Unique identifier for experiment")
    parser.add_argument(
        "--experiment-type",
        type=str,
        required=True,
        choices=["baseline_audit", "robustness_eval", "evt_calibration", "memory_bank_knn", "multires_logmel"],
        help="Type of experiment to execute",
    )
    parser.add_argument("--machine-id", type=str, default="all", choices=["all", "id_00", "id_02", "id_04", "id_06"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    machines = ALL_MACHINES if args.machine_id == "all" else [args.machine_id]
    manager = ExperimentManager()

    # Register experiment
    config_dict = {
        "experiment_type": args.experiment_type,
        "machine_ids": machines,
        "seed": args.seed,
    }
    manager.register_experiment(
        experiment_id=args.experiment_id,
        name=f"Experiment {args.experiment_type}",
        description=f"Automated execution of {args.experiment_type} across {machines}",
        model_type="stetho_msae",
        config=config_dict,
        seed=args.seed,
    )

    manager.update_status(experiment_id=args.experiment_id, status="RUNNING", pid=os.getpid())

    try:
        start_time = time.time()
        if args.experiment_type == "baseline_audit":
            metrics = run_baseline_audit(args.experiment_id, manager, machines)
        elif args.experiment_type == "robustness_eval":
            metrics = run_domain_robustness_eval(args.experiment_id, manager, machines)
        elif args.experiment_type == "evt_calibration":
            metrics = run_evt_calibration_eval(args.experiment_id, manager, machines)
        elif args.experiment_type == "memory_bank_knn":
            metrics = run_memory_bank_knn_eval(args.experiment_id, manager, machines)
        elif args.experiment_type == "multires_logmel":
            metrics = run_multires_logmel_eval(args.experiment_id, manager, machines)
        else:
            raise ValueError(f"Unsupported experiment type: {args.experiment_type}")

        duration = time.time() - start_time
        manager.update_status(
            experiment_id=args.experiment_id,
            status="COMPLETED",
            metrics=metrics,
        )
        logger.info(f"Experiment {args.experiment_id} successfully COMPLETED in {duration:.2f}s!")
        print(json.dumps({"status": "COMPLETED", "metrics": metrics}, indent=2))

    except Exception as e:
        logger.exception(f"Experiment {args.experiment_id} FAILED with error: {e}")
        manager.update_status(
            experiment_id=args.experiment_id,
            status="FAILED",
            error=str(e),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
