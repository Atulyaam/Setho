"""Sensor Guard and Calibration Compatibility Validation for Stetho.

Tracks physical sensor configuration metadata (sensor ID, channel index, sample rate)
and enforces hardware lock: prevents silent reuse of calibration profiles across
mismatched sensor hardware, requiring recalibration upon physical sensor changes.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union


@dataclass
class SensorProfileMetadata:
    """Hardware metadata associated with a calibrated machine profile."""

    machine_id: str
    sensor_id: str
    channel_index: int
    calibrated_sample_rate: int = 16000
    calibration_timestamp: Optional[str] = None
    mic_array_geometry: str = "circular_8ch_10cm_radius"


class SensorGuard:
    """Enforces sensor compatibility and prevents profile contamination."""

    def __init__(self):
        # Store calibrated profiles: key (machine_id, sensor_id, channel_index) -> profile dict
        self._profiles: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    def register_profile(
        self,
        machine_id: str,
        sensor_id: str,
        channel_index: int,
        profile_data: Dict[str, Any],
    ) -> None:
        """Register a calibrated residual whitening profile for a specific sensor."""
        key = (machine_id, sensor_id, channel_index)
        self._profiles[key] = {
            **profile_data,
            "metadata": SensorProfileMetadata(
                machine_id=machine_id,
                sensor_id=sensor_id,
                channel_index=channel_index,
            ),
        }

    def has_profile(self, machine_id: str, sensor_id: str, channel_index: int) -> bool:
        """Check if a calibrated profile exists for the given sensor configuration."""
        return (machine_id, sensor_id, channel_index) in self._profiles

    def get_profile(
        self, machine_id: str, sensor_id: str, channel_index: int
    ) -> Optional[Dict[str, Any]]:
        """Retrieve profile if compatible."""
        return self._profiles.get((machine_id, sensor_id, channel_index))

    def validate_sensor_compatibility(
        self,
        machine_id: str,
        incoming_sensor_id: str,
        incoming_channel: int,
        active_sensor_id: str,
        active_channel: int,
    ) -> Tuple[bool, str]:
        """Validate whether incoming audio matches the active calibrated sensor configuration.

        Returns:
            Tuple of (is_compatible, status_message)
        """
        if incoming_sensor_id != active_sensor_id:
            return (
                False,
                f"SENSOR_MISMATCH: Incoming sensor_id '{incoming_sensor_id}' does not match "
                f"active calibrated sensor '{active_sensor_id}'. Hardware mismatch requires recalibration.",
            )

        if incoming_channel != active_channel:
            return (
                False,
                f"CHANNEL_MISMATCH: Incoming microphone channel {incoming_channel} does not match "
                f"calibrated channel {active_channel}. Directional acoustic mismatch requires recalibration.",
            )

        return (True, "SENSOR_COMPATIBLE")
