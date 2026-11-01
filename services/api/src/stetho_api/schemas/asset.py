"""Pydantic Schemas for Asset Management and Onboarding (Sprint 7)."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class AssetCreateRequest(BaseModel):
    """Payload for registering a physical machinery asset."""
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(
        ...,
        min_length=2,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_\-]+$",
        description="Unique identifier for the asset (alphanumeric, dashes, underscores).",
        examples=["ast_fan_01", "pump_unit_04", "valve_station_9"],
    )
    name: str = Field(
        ...,
        min_length=2,
        max_length=128,
        description="Human-readable asset label.",
        examples=["Primary Cooling Fan 01"],
    )
    category: Literal["fan", "pump", "valve"] = Field(
        ...,
        description="Physical equipment category.",
        examples=["fan"],
    )
    description: Optional[str] = Field(
        default="",
        max_length=512,
        description="Asset description, physical location, or serial information.",
    )
    operating_condition: Optional[str] = Field(
        default="",
        max_length=256,
        description="Baseline operating parameters (e.g. 1800 RPM, 60 Hz).",
    )
    assigned_model: Optional[str] = Field(
        default="cascade_ensemble",
        description="Assigned anomaly detection model policy or profile.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Arbitrary industrial telemetry or asset metadata.",
    )


class AssetResponse(BaseModel):
    """Detailed response model for a registered industrial asset."""
    model_config = ConfigDict(from_attributes=True)

    asset_id: str
    name: str
    category: str
    description: str = ""
    operating_condition: str = ""
    assigned_model: str = "cascade_ensemble"
    assigned_model_version: str = "1.0.0"
    calibration_status: str = "uncalibrated"
    calibration_version: str = "1.0.0"
    enrolled_recording_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    owner_id: str = "usr_admin"
    created_at: str
    updated_at: str


class AssetListResponse(BaseModel):
    """List response encapsulating multiple registered assets."""
    assets: List[AssetResponse]
    total_count: int


class AssetCalibrationResponse(BaseModel):
    """Result of a machine few-shot calibration run."""
    asset_id: str
    model_type: str
    calibration_version: str
    calibration_status: str
    operating_threshold: float
    theta_lower: float
    theta_upper: float
    enrolled_sample_count: int
    enrolled_mean: float
    enrolled_std: float
    is_provisional: bool
    notice: str
    recalculated_at_utc: str
    enrolled_scores: List[float] = Field(default_factory=list)
