#!/usr/bin/env python3
"""Sprint 7.6 / E0 - Reproducible Machine Learning Experiment Runner.

Standardized, deterministic execution harness for all Stetho experiments.
Enforces:
1. Strict seed pinning across random, numpy, and torch.
2. Complete provenance capture: git commit, manifest SHA-256, environment specs.
3. Standardized artifact and metric serialization.
4. Comprehensive per-machine evaluation breakdown (not just macro averages).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import soundfile as sf
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "services/api/src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_exp_runner")


def set_seed(seed: int = 42) -> None:
    """Set global deterministic seeds across all random generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Deterministic seed locked: {seed}")


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def compute_file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_auc_score(labels: List[int], scores: List[float]) -> float:
    """Compute ROC-AUC using exact Wilcoxon-Mann-Whitney formulation."""
    normal_scores = [scores[i] for i in range(len(labels)) if labels[i] == 0]
    abnormal_scores = [scores[i] for i in range(len(labels)) if labels[i] == 1]
    n_norm = len(normal_scores)
    n_abn = len(abnormal_scores)
    if n_norm == 0 or n_abn == 0:
        return 0.0
    wins = 0.0
    for anom in abnormal_scores:
        for norm in normal_scores:
            if anom > norm:
                wins += 1.0
            elif anom == norm:
                wins += 0.5
    return float(wins / (n_norm * n_abn))


class ExperimentRunner:
    """Deterministic Experiment Runner for Stetho research cycles."""

    def __init__(
        self,
        experiment_id: str,
        model_type: str = "resnet_ae",
        seed: int = 42,
        output_dir: Optional[Path] = None,
    ):
        self.experiment_id = experiment_id
        self.model_type = model_type
        self.seed = seed
        self.output_dir = output_dir or Path(f"experiments/sprint_7_6/runs/{experiment_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_seed(self.seed)

    def capture_metadata(self) -> Dict[str, Any]:
        """Record full execution environment and dataset provenance."""
        manifest_path = Path("ml/data/manifests/diverse_dataset_manifest.json")
        return {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "dataset_manifest_hash": compute_file_hash(manifest_path),
            "seed": self.seed,
            "model_type": self.model_type,
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "os": os.name,
                "cpu_count": psutil.cpu_count(logical=True),
            },
        }

    def evaluate_checkpoint(
        self,
        checkpoint_dir: Path,
        machine_ids: List[str] = ["id_00", "id_02", "id_04", "id_06"],
    ) -> Dict[str, Any]:
        """Evaluate a trained model family across all machines."""
        import csv
        from ml.models.research.research_detector import ResearchNeuralDetector

        results = {}
        macro_aucs = []

        for m in machine_ids:
            logger.info(f"Evaluating machine {m}...")
            ckpt_path = checkpoint_dir / m / "model.pt"
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint not found: {ckpt_path}")
                continue

            det = ResearchNeuralDetector.load(ckpt_path)
            test_csv = Path(f"ml/data/splits/diverse_machine_specific/fan/{m}/test.csv")
            if not test_csv.exists():
                logger.warning(f"Test split not found: {test_csv}")
                continue

            labels = []
            scores = []
            latencies = []

            with open(test_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lbl = 1 if row["label"] == "abnormal" else 0
                    labels.append(lbl)
                    t0 = time.perf_counter()
                    s = det.score_file(row["file_path"])
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                    scores.append(s)

            auc = compute_auc_score(labels, scores)
            macro_aucs.append(auc)

            n_norm = sum(1 for l in labels if l == 0)
            n_abn = sum(1 for l in labels if l == 1)

            results[m] = {
                "roc_auc": round(auc, 4),
                "n_normal": n_norm,
                "n_abnormal": n_abn,
                "mean_latency_ms": round(float(np.mean(latencies)), 2),
                "p50_latency_ms": round(float(np.median(latencies)), 2),
            }
            logger.info(f"  {m}: ROC-AUC = {auc:.4f} (N_norm={n_norm}, N_abn={n_abn})")

        macro_auc = float(np.mean(macro_aucs)) if macro_aucs else 0.0
        summary = {
            "metadata": self.capture_metadata(),
            "macro_roc_auc": round(macro_auc, 4),
            "per_machine": results,
        }

        out_file = self.output_dir / "evaluation_summary.json"
        with open(out_file, "w") as fp:
            json.dump(summary, fp, indent=2)
        logger.info(f"Evaluation complete. Macro ROC-AUC: {macro_auc:.4f}. Saved to {out_file}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Stetho Reproducible Experiment Runner")
    parser.add_argument("--experiment-id", type=str, default="e0_baseline_check", help="Unique experiment run ID")
    parser.add_argument("--model-type", type=str, default="resnet_ae", help="Model family to run/evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation without re-training")
    args = parser.parse_args()

    runner = ExperimentRunner(
        experiment_id=args.experiment_id,
        model_type=args.model_type,
        seed=args.seed,
    )

    if args.eval_only:
        runner.evaluate_checkpoint(Path("ml/experiments/research_artifacts/phase4/residual_cnn_ae"))


if __name__ == "__main__":
    main()
