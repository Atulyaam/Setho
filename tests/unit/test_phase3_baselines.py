"""Automated verification tests for Phase 3 Experiment Framework & Baselines."""

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ml.experiments.experiment_manager import ExperimentManager


def test_experiment_manager_records_metadata():
    """Verify ExperimentManager tracks git_commit, manifest_hash, architecture, telemetry."""
    mgr = ExperimentManager()
    reg = mgr._load_registry()
    experiments = reg.get("experiments", {})

    assert len(experiments) > 0, "Registry must contain registered experiments"
    phase3_exps = [e for k, e in experiments.items() if k.startswith("phase3_")]
    assert len(phase3_exps) >= 2, f"Expected at least 2 phase 3 experiments in registry, found {len(phase3_exps)}"

    for exp in phase3_exps:
        exp_id = exp["experiment_id"]
        assert "git_commit" in exp, f"Missing git_commit in {exp_id}"
        assert "manifest_hash" in exp, f"Missing manifest_hash in {exp_id}"
        assert "architecture" in exp, f"Missing architecture in {exp_id}"
        assert "preprocessing" in exp, f"Missing preprocessing in {exp_id}"
        assert "telemetry" in exp, f"Missing telemetry in {exp_id}"
        assert exp["telemetry"]["ram_used_mb"] > 0, f"RAM telemetry invalid in {exp_id}"


def test_production_artifacts_untouched():
    """Verify production artifacts in ml/artifacts/ are strictly untouched."""
    prod_msae_dir = Path("ml/artifacts/stetho_msae")
    assert prod_msae_dir.exists(), "Production directory must exist"
    for m in ["id_00", "id_02", "id_04", "id_06"]:
        prod_pt = prod_msae_dir / m / "model.pt"
        assert prod_pt.exists(), f"Production checkpoint {prod_pt} missing"


def test_phase3_reproducibility_and_metrics_integrity():
    """Verify Phase 3 checkpoints exist, parameter counts are correct, and metrics recalculate identically."""
    base_dir = Path("ml/experiments/research_artifacts/phase3")
    models = ["dense_autoencoder", "stetho_msae"]

    for model_type in models:
        model_dir = base_dir / model_type
        assert model_dir.exists(), f"Phase 3 directory missing for {model_type}"
        macro_summary_file = model_dir / "macro_summary.json"
        assert macro_summary_file.exists(), f"Missing macro summary for {model_type}"

        with open(macro_summary_file, "r", encoding="utf-8") as f:
            macro_data = json.load(f)

        if model_type == "dense_autoencoder":
            assert macro_data["parameter_count"] == 4341536, f"Dense AE parameter count mismatch: {macro_data['parameter_count']}"
        elif model_type == "stetho_msae":
            assert macro_data["parameter_count"] == 559873, f"Stetho-MSAE parameter count mismatch: {macro_data['parameter_count']}"

        for m_id in ["id_00", "id_02", "id_04", "id_06"]:
            m_res_file = model_dir / m_id / "evaluation_results.json"
            assert m_res_file.exists(), f"Missing evaluation results for {model_type}/{m_id}"

            with open(m_res_file, "r", encoding="utf-8") as f:
                res = json.load(f)

            scores = np.array(res["test_scores"])
            labels = np.array(res["test_labels"])
            reported_auc = res["metrics"]["roc_auc"]

            recalc_auc = round(float(roc_auc_score(labels, scores)), 4)
            assert recalc_auc == reported_auc, f"Recalculated AUC {recalc_auc} != reported {reported_auc} for {model_type}/{m_id}"

