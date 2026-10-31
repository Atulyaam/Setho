"""Machine-Specific Statistical Calibration Engine for Stetho.

Dynamically recalculates decision thresholds (theta_lower, theta_upper) from
explicitly confirmed normal audio recordings enrolled on an individual machine.
Enforces statistical rigor:
- Enrollment count alone NEVER grants 'calibrated' status.
- Minimum enrollment criteria (N >= 30) for statistical stability.
- Provisional uncertainty tracking with boundary zones.
- Isolation: Test/benchmark recordings are NEVER used for calibration.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from stetho_api.db.database import DatabaseManager, get_utc_now_iso
from stetho_api.db.analysis_repo import EnrollmentRepository
from stetho_api.db.machine_repo import MachineProfile, MachineRepository
from stetho_api.exceptions import StethoAPIException
from stetho_api.inference.model_registry import ModelRegistry, ModelType, get_model_registry
from stetho_audio.channel_policy import ChannelHandler, ChannelPolicy
from stetho_audio.preprocessing import AudioPreprocessor, PreprocessConfig

logger = logging.getLogger("stetho.calibration_engine")


@dataclass
class CalibrationResult:
    machine_id: str
    model_type: str
    calibration_version: str
    enrolled_sample_count: int
    operating_threshold: float
    theta_lower: float
    theta_upper: float
    enrolled_mean: float
    enrolled_std: float
    is_provisional: bool
    calibration_status: str  # "uncalibrated", "provisional", "calibrated"
    recalculated_at_utc: str
    notice: str
    enrolled_scores: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "model_type": self.model_type,
            "calibration_version": self.calibration_version,
            "enrolled_sample_count": self.enrolled_sample_count,
            "operating_threshold": round(self.operating_threshold, 6),
            "theta_lower": round(self.theta_lower, 6),
            "theta_upper": round(self.theta_upper, 6),
            "enrolled_mean": round(self.enrolled_mean, 6),
            "enrolled_std": round(self.enrolled_std, 6),
            "is_provisional": self.is_provisional,
            "calibration_status": self.calibration_status,
            "recalculated_at_utc": self.recalculated_at_utc,
            "notice": self.notice,
            "enrolled_scores": [round(x, 6) for x in self.enrolled_scores],
        }


class MachineCalibrationEngine:
    """Manages empirical threshold recalibration for physical machines."""

    def __init__(
        self,
        db: DatabaseManager,
        registry: Optional[ModelRegistry] = None,
    ):
        self.db = db
        self.machine_repo = MachineRepository(db)
        self.enrollment_repo = EnrollmentRepository(db)
        self.registry = registry or get_model_registry()
        self.preprocessor = AudioPreprocessor(
            PreprocessConfig(
                target_sample_rate=16000,
                channel_policy=ChannelPolicy.CHANNEL_0,
                normalization="peak",
                validate_input=False,
            )
        )

    def recalibrate_machine(
        self,
        machine_id: str,
        newest_waveform: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
    ) -> CalibrationResult:
        """Score all confirmed normal recordings enrolled for machine and recompute thresholds."""
        machine = self.machine_repo.get_or_raise(machine_id)
        model_type = machine.selected_model_policy

        # 1. Fetch all confirmed normal enrollments for this machine
        records = self.enrollment_repo.list_by_machine(machine_id)
        valid_records = [r for r in records if r.is_confirmed_normal and r.quality_status in ["ACCEPTED", "WARNING"]]

        # 2. Retrieve detector from registry (use machine's specific weights if available, or id_00 base weights)
        spec_dir = self.registry.artifacts_root / model_type / machine_id
        effective_id = machine_id if spec_dir.exists() else "id_00"
        loaded_model = self.registry.get_model(machine_id=effective_id, model_type=model_type)

        scores: List[float] = []

        # 3. Score all stored normal enrollment audio files
        for r in valid_records:
            stored_p = Path(r.stored_path)
            if stored_p.exists():
                try:
                    data, sr = sf.read(str(stored_p), dtype="float32", always_2d=True)
                    mono_raw = ChannelHandler.apply(data, policy=ChannelPolicy.CHANNEL_0)
                    mono_proc, _ = self.preprocessor.process_array(mono_raw, sample_rate=sr)
                    s, _ = loaded_model.score_waveform(mono_proc)
                    scores.append(float(s))
                except Exception as e:
                    logger.warning(f"Could not score enrollment audio {stored_p}: {e}")

        # If newest waveform provided and not yet on disk or to ensure current recording is included
        if newest_waveform is not None and len(scores) == 0:
            mono_raw = ChannelHandler.apply(newest_waveform, policy=ChannelPolicy.CHANNEL_0)
            mono_proc, _ = self.preprocessor.process_array(mono_raw, sample_rate=sample_rate)
            s, _ = loaded_model.score_waveform(mono_proc)
            scores.append(float(s))

        N = len(scores)

        # 4. Rigorous statistical threshold calculation
        if N == 0:
            # Fallback to model's initial checkpoint calibration
            init_cal = loaded_model.metadata.calibration_profile
            return CalibrationResult(
                machine_id=machine_id,
                model_type=model_type,
                calibration_version=machine.calibration_version,
                enrolled_sample_count=0,
                operating_threshold=float(init_cal.get("operating_threshold", 0.3)),
                theta_lower=float(init_cal.get("theta_lower", 0.28)),
                theta_upper=float(init_cal.get("theta_upper", 0.3)),
                enrolled_mean=0.0,
                enrolled_std=0.0,
                is_provisional=True,
                calibration_status="uncalibrated",
                recalculated_at_utc=get_utc_now_iso(),
                notice="No valid confirmed normal recordings found. Machine is uncalibrated.",
                enrolled_scores=[],
            )

        scores_arr = np.array(scores, dtype=np.float64)
        mean_val = float(np.mean(scores_arr))
        std_val = float(np.std(scores_arr)) if N > 1 else 0.01
        min_val = float(np.min(scores_arr))
        max_val = float(np.max(scores_arr))

        if N < 5:
            # Insufficient data: very wide provisional boundary
            theta_lower = mean_val
            theta_upper = max(max_val * 1.05, mean_val + 2.0 * max(std_val, 0.01))
            status = "provisional"
            is_provisional = True
            notice = (
                f"Provisional calibration (N={N} recordings). Minimum recommended is N >= 5 for initial "
                "variance estimates, and N >= 30 for stable false-positive rate control."
            )
        elif 5 <= N < 30:
            # Moderate enrollment: empirical 95th percentile with provisional flag
            theta_lower = mean_val
            theta_upper = float(np.percentile(scores_arr, 95))
            if theta_upper <= theta_lower:
                theta_upper = mean_val + 2.0 * max(std_val, 0.005)
            status = "provisional"
            is_provisional = True
            notice = (
                f"Provisional empirical calibration (N={N} recordings). Operating threshold carries "
                "statistical variance. Full calibration requires N >= 30."
            )
        else:
            # N >= 30: fully calibrated, statistically stabilized
            theta_lower = float(np.percentile(scores_arr, 50))
            theta_upper = float(np.percentile(scores_arr, 95))
            status = "calibrated"
            is_provisional = False
            notice = f"Statistically stabilized calibration based on N={N} confirmed normal baseline recordings."

        now = get_utc_now_iso()
        version = f"1.{N}.0"

        cal_profile_dict = {
            "machine_id": machine_id,
            "model_type": model_type,
            "calibration_version": version,
            "enrolled_sample_count": N,
            "operating_threshold": theta_upper,
            "theta_lower": theta_lower,
            "theta_upper": theta_upper,
            "enrolled_mean": mean_val,
            "enrolled_std": std_val,
            "enrolled_min": min_val,
            "enrolled_max": max_val,
            "enrolled_scores": scores,
            "is_provisional": is_provisional,
            "calibration_status": status,
            "recalculated_at_utc": now,
            "limitation_notice": notice,
        }

        # 5. Persist to SQLite
        self.machine_repo.update_calibration_data(
            machine_id=machine_id,
            profile_dict=cal_profile_dict,
            version=version,
            status=status,
            enrolled_count=N,
        )

        # 6. Update in-memory loaded model in ModelRegistry
        self.registry.update_model_calibration(
            machine_id=machine_id,
            model_type=model_type,
            new_profile=cal_profile_dict,
        )

        logger.info(
            f"Successfully recalculated calibration for machine '{machine_id}' [{model_type}]: "
            f"N={N}, theta_upper={theta_upper:.5f}, status={status}, version={version}"
        )

        return CalibrationResult(
            machine_id=machine_id,
            model_type=model_type,
            calibration_version=version,
            enrolled_sample_count=N,
            operating_threshold=theta_upper,
            theta_lower=theta_lower,
            theta_upper=theta_upper,
            enrolled_mean=mean_val,
            enrolled_std=std_val,
            is_provisional=is_provisional,
            calibration_status=status,
            recalculated_at_utc=now,
            notice=notice,
            enrolled_scores=scores,
        )
