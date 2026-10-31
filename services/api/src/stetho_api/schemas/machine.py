"""Pydantic schemas for Machine profiles and enrollments."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MachineCreateRequest(BaseModel):
    machine_id: str = Field(..., min_length=2, max_length=64, description="Unique machine identifier (e.g. 'fan_01')")
    name: str = Field(..., min_length=2, max_length=128, description="Human-readable machine name")
    category: str = Field(default="fan", max_length=64, description="Machine category (e.g. 'fan', 'pump', 'motor')")
    description: Optional[str] = Field(default="", max_length=512, description="Physical location or operating role")
    operating_condition: Optional[str] = Field(default="", max_length=256, description="Baseline operating condition notes")
    selected_model_policy: Optional[str] = Field(
        default="stetho_msae",
        description="Selected model architecture ('stetho_msae' [edge champion] or 'dense_autoencoder' [server baseline])",
    )


class MachineResponse(BaseModel):
    machine_id: str
    name: str
    category: str
    description: str
    operating_condition: str
    selected_model_policy: str
    enrollment_status: str
    enrolled_recording_count: int
    calibration_status: str
    calibration_version: Optional[str] = "1.0.0"
    owner_id: Optional[str] = "usr_admin"
    created_at: str
    updated_at: str


class MachineListResponse(BaseModel):
    machines: List[MachineResponse]
    total_count: int


class EnrollmentResponse(BaseModel):
    enrollment_id: str
    machine_id: str
    upload_id: str
    duration_sec: float
    sample_rate: int
    channels: int
    quality_status: str
    is_confirmed_normal: bool
    created_at: str
    enrolled_count: int
    enrollment_status: str
    operating_threshold: Optional[float] = None
    theta_lower: Optional[float] = None
    theta_upper: Optional[float] = None
    calibration_version: Optional[str] = None
    notice: str
