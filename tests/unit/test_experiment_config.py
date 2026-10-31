"""Unit tests for Experiment Configuration and Reproducibility."""

from pathlib import Path
import tempfile
import pytest

from ml.configs.experiment_config import (
    ExperimentConfig,
    FeatureConfig,
    ModelHyperparameters,
    compute_file_hash,
)


def test_config_yaml_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "test_config.yaml"
        cfg = ExperimentConfig(
            experiment_id="test_exp_01",
            machine_id="id_02",
            model=ModelHyperparameters(model_type="isolation_forest", n_estimators=100),
            features=FeatureConfig(feature_type="statistical", n_mfcc=20),
            seed=123,
        )

        cfg.save_yaml(cfg_path)
        assert cfg_path.exists()

        loaded_cfg = ExperimentConfig.from_yaml(cfg_path)
        assert loaded_cfg.experiment_id == "test_exp_01"
        assert loaded_cfg.machine_id == "id_02"
        assert loaded_cfg.model.n_estimators == 100
        assert loaded_cfg.features.n_mfcc == 20
        assert loaded_cfg.seed == 123
        assert "id_02" in loaded_cfg.train_split
        assert "isolation_forest" in loaded_cfg.artifact_dir


def test_default_yaml_files_loadable():
    if_path = Path("ml/configs/isolation_forest_default.yaml")
    dae_path = Path("ml/configs/dense_autoencoder_default.yaml")

    assert if_path.exists()
    assert dae_path.exists()

    cfg_if = ExperimentConfig.from_yaml(if_path)
    assert cfg_if.model.model_type == "isolation_forest"

    cfg_dae = ExperimentConfig.from_yaml(dae_path)
    assert cfg_dae.model.model_type == "dense_autoencoder"
    assert cfg_dae.model.latent_dim == 32


def test_manifest_hash_computation():
    manifest_path = Path("ml/data/manifests/audio_metadata.csv")
    if manifest_path.exists():
        h = compute_file_hash(manifest_path)
        assert len(h) == 16
        assert isinstance(h, str)

