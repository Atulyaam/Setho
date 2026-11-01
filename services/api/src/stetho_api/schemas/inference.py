"""Pydantic Schemas for Inference, Diagnostic Evaluation, and Models (Sprint 7)."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class InferenceResponse(BaseModel):
    """Standardized response schema for an acoustic inference inspection."""
    model_config = ConfigDict(from_attributes=True)

    result_id: Optional[str] = Field(default=None, description="Unique inference record ID.")
    asset_id: str = Field(..., description="Target asset identifier.")
    model_version: str = Field(..., description="Version of the model that executed inference.")
    model_name: str = Field(..., description="Architecture name of the evaluating model.")
    anomaly_score: Optional[float] = Field(
        default=None,
        description="Continuous reconstruction anomaly score (not a failure probability).",
    )
    anomaly_decision: Literal["NORMAL_LIKE", "UNUSUAL", "INCONCLUSIVE"] = Field(
        ...,
        description="Three-zone operational anomaly classification.",
    )
    threshold_used: Optional[float] = Field(
        default=None,
        description="Calibrated operating decision threshold.",
    )
    theta_lower: Optional[float] = Field(
        default=None,
        description="Lower normal bound of the three-zone calibrated envelope.",
    )
    theta_upper: Optional[float] = Field(
        default=None,
        description="Upper anomaly bound of the three-zone calibrated envelope.",
    )
    processing_latency_ms: float = Field(
        ...,
        description="Total end-to-end execution latency in milliseconds.",
    )
    calibration_status: str = Field(
        default="calibrated",
        description="Calibration state: uncalibrated, provisional, or calibrated.",
    )
    quality_status: str = Field(
        default="ACCEPTED",
        description="Audio Quality Gate assessment status.",
    )
    is_provisional: bool = Field(
        default=False,
        description="Flag indicating if calibration carries statistical uncertainty.",
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Structured engineering rationales for the decision.",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sub-stage telemetry (runtime engine, frequencies, cascade routing).",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 creation timestamp.",
    )


class InferenceHistoryListResponse(BaseModel):
    """List response for historical inference records on an asset."""
    results: List[InferenceResponse]
    total_count: int


class ModelInfoResponse(BaseModel):
    """Metadata describing a registered model architecture."""
    model_id: str
    name: str
    type: str
    version: str
    profile: str
    runtime_engine: str
    latency_p50_cpu_ms: float
    disk_size_kb: float
    supported_categories: List[str]
    description: str


class ModelListResponse(BaseModel):
    """List of all available models in the registry."""
    models: List[ModelInfoResponse]
    total_count: int
