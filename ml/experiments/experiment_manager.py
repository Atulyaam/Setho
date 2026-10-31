"""Autonomous ML Experiment Manager and Registry for Stetho Sprint 6A.

Provides tracking, progress monitoring, hardware telemetry (RAM, CPU, GPU),
and artifact archiving for non-blocking ML research experiments.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("stetho_exp_manager")

EXPERIMENTS_DIR = Path("ml/experiments")
LOGS_DIR = EXPERIMENTS_DIR / "logs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
ARTIFACTS_DIR = EXPERIMENTS_DIR / "artifacts"
REGISTRY_FILE = EXPERIMENTS_DIR / "experiment_registry.json"


def get_system_telemetry() -> Dict[str, Any]:
    """Capture instantaneous RAM, CPU, and GPU utilization metrics."""
    vm = psutil.virtual_memory()
    telemetry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ram_used_mb": round(vm.used / (1024 * 1024), 2),
        "ram_total_mb": round(vm.total / (1024 * 1024), 2),
        "ram_percent": vm.percent,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "gpu_available": False,
        "gpu_vram_used_mb": 0.0,
        "gpu_vram_total_mb": 0.0,
    }

    if HAS_TORCH and torch.cuda.is_available():
        telemetry["gpu_available"] = True
        telemetry["gpu_device_name"] = torch.cuda.get_device_name(0)
        telemetry["gpu_vram_used_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
        telemetry["gpu_vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2)
    return telemetry


class ExperimentManager:
    """Manages the lifecycle, registry, and execution monitoring of ML experiments."""

    def __init__(self, registry_path: Path = REGISTRY_FILE):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save_registry({"experiments": {}, "updated_at": datetime.now(timezone.utc).isoformat()})

    def _load_registry(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            return {"experiments": {}, "updated_at": datetime.now(timezone.utc).isoformat()}
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, data: Dict[str, Any]) -> None:
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str,
        model_type: str,
        config: Dict[str, Any],
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Register a new experiment in QUEUED status."""
        reg = self._load_registry()
        exp_record = {
            "experiment_id": experiment_id,
            "name": name,
            "description": description,
            "model_type": model_type,
            "status": "QUEUED",
            "seed": seed,
            "config": config,
            "pid": None,
            "start_time": None,
            "completion_time": None,
            "duration_sec": None,
            "current_epoch": 0,
            "max_epochs": config.get("model", {}).get("max_epochs", 60),
            "train_loss": None,
            "val_loss": None,
            "telemetry": get_system_telemetry(),
            "metrics": {},
            "log_path": str(LOGS_DIR / f"{experiment_id}.log"),
            "result_path": str(RESULTS_DIR / f"{experiment_id}.json"),
            "artifact_dir": str(ARTIFACTS_DIR / experiment_id),
            "error": None,
        }
        reg["experiments"][experiment_id] = exp_record
        self._save_registry(reg)
        return exp_record

    def update_status(
        self,
        experiment_id: str,
        status: str,
        pid: Optional[int] = None,
        current_epoch: Optional[int] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Atomically update experiment status and progress metrics."""
        reg = self._load_registry()
        if experiment_id not in reg["experiments"]:
            return

        rec = reg["experiments"][experiment_id]
        rec["status"] = status
        rec["telemetry"] = get_system_telemetry()

        if pid is not None:
            rec["pid"] = pid
        if status == "RUNNING" and rec["start_time"] is None:
            rec["start_time"] = datetime.now(timezone.utc).isoformat()
        if current_epoch is not None:
            rec["current_epoch"] = current_epoch
        if train_loss is not None:
            rec["train_loss"] = round(train_loss, 6)
        if val_loss is not None:
            rec["val_loss"] = round(val_loss, 6)
        if metrics is not None:
            rec["metrics"] = metrics
        if error is not None:
            rec["error"] = error

        if status in ["COMPLETED", "FAILED", "STOPPED"]:
            rec["completion_time"] = datetime.now(timezone.utc).isoformat()
            if rec["start_time"]:
                try:
                    start_dt = datetime.fromisoformat(rec["start_time"])
                    end_dt = datetime.fromisoformat(rec["completion_time"])
                    rec["duration_sec"] = round((end_dt - start_dt).total_seconds(), 2)
                except Exception:
                    pass

        self._save_registry(reg)

        # Also persist to individual results file
        result_file = Path(rec["result_path"])
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        reg = self._load_registry()
        return reg["experiments"].get(experiment_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        reg = self._load_registry()
        return list(reg["experiments"].values())
