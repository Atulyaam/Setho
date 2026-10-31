"""Training and Evaluation CLI for Stetho-MSAE.

Supports training across single or all machine IDs, with full ablation experiment support:
- Standard (Stetho-MSAE default)
- Ablation B: Standard 2D Convolutions (no depthwise)
- Ablation C: Single-scale (no temporal dilation)
- Ablation: Latent dim 128
- Ablation: Top-k patch anomaly scoring

Maintains strict zero-leakage partition boundaries.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.configs.experiment_config import ExperimentConfig
from ml.training.train_pipeline import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stetho_train_msae")

ALL_MACHINES = ["id_00", "id_02", "id_04", "id_06"]
ABLATION_MAP = {
    "standard_conv": Path("ml/configs/ablations/ablation_b_standard_conv.yaml"),
    "no_dilation": Path("ml/configs/ablations/ablation_c_no_dilation.yaml"),
    "latent_128": Path("ml/configs/ablations/ablation_latent_128.yaml"),
    "top_k": Path("ml/configs/ablations/ablation_top_k.yaml"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Stetho-MSAE detector.")
    parser.add_argument(
        "--machine-id",
        type=str,
        default="all",
        choices=["id_00", "id_02", "id_04", "id_06", "all"],
        help="Machine ID to train on or 'all' for all machines.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to experiment config YAML.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        choices=list(ABLATION_MAP.keys()),
        help="Optional ablation preset to execute.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Target compute device.",
    )
    return parser.parse_args()


def train_msae_for_machines(
    machine_ids: List[str],
    config_path: Path,
    device: str = "auto",
    ablation_name: str = None,
) -> List[Dict]:
    """Train Stetho-MSAE on specified machines using provided config."""
    results = []
    for m_id in machine_ids:
        logger.info(f"\n=======================================================")
        logger.info(f"Training Stetho-MSAE for Machine: {m_id}")
        logger.info(f"=======================================================")
        cfg = ExperimentConfig.from_yaml(config_path)
        cfg.set_machine_id(m_id)
        if ablation_name:
            cfg.experiment_id = f"ablation_{ablation_name}_{m_id}"
            cfg.artifact_dir = f"ml/artifacts/ablations/{ablation_name}/{m_id}"
        if device != "auto":
            cfg.device = device

        res = run_experiment(cfg)
        results.append(res)
    return results


def main():
    args = parse_args()

    # Determine config file
    if args.config:
        config_path = Path(args.config)
    elif args.ablation:
        config_path = ABLATION_MAP[args.ablation]
    else:
        config_path = Path("ml/configs/stetho_msae_default.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    target_machines = ALL_MACHINES if args.machine_id == "all" else [args.machine_id]

    logger.info(f"Using config: {config_path}")
    logger.info(f"Target machines: {target_machines}")
    logger.info(f"Device: {args.device}")

    results = train_msae_for_machines(
        target_machines, config_path, device=args.device, ablation_name=args.ablation
    )

    # Print summary
    print("\n" + "=" * 70)
    print(f"STETHO-MSAE TRAINING & EVALUATION SUMMARY ({config_path.stem})")
    print("=" * 70)
    print(f"{'Machine':<10} | {'ROC-AUC':<8} | {'pAUC':<8} | {'F1':<8} | {'Recall':<8} | {'FPR':<8}")
    print("-" * 70)
    for r in results:
        m = r["metrics"]
        print(
            f"{r['machine_id']:<10} | "
            f"{m['roc_auc']:<8.4f} | "
            f"{m['pauc']:<8.4f} | "
            f"{m['f1']:<8.4f} | "
            f"{m['recall']:<8.4f} | "
            f"{m['fpr']:<8.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
