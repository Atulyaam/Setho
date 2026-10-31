"""Pydantic schemas for Acoustic Analysis, Jobs, and Inspection Reports."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from stetho_api.schemas.machine import MachineResponse


class AnalysisResponse(BaseModel):
    analysis_id: str
    machine_id: str
    model_name: str
    model_version: str
    analysis_status: str
    quality_status: str
    decision: str = Field(description="'NORMAL_LIKE', 'UNUSUAL', or 'INCONCLUSIVE'")
    anomaly_score: Optional[float] = Field(default=None, description="Raw reconstruction anomaly score")
    operating_threshold: Optional[float] = Field(default=None, description="Calibrated decision threshold (theta_upper)")
    theta_lower: Optional[float] = Field(default=None, description="Lower normal baseline threshold")
    theta_upper: Optional[float] = Field(default=None, description="Upper abnormal decision threshold")
    is_provisional: bool = Field(description="True if calibration is based on limited enrollment samples (N < 30)")
    processing_duration_ms: float
    reasons: List[str]
    metrics: Dict[str, Any]
    created_at: str


class AnalysisListResponse(BaseModel):
    machine_id: str
    analyses: List[AnalysisResponse]
    total_count: int


class JobResponse(BaseModel):
    job_id: str
    machine_id: str
    status: str = Field(description="'pending', 'processing', 'completed', or 'failed'")
    progress_pct: int = Field(ge=0, le=100)
    analysis_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class InspectionReportResponse(BaseModel):
    report_id: str
    generated_at: str
    machine: MachineResponse
    analysis: AnalysisResponse
    executive_summary: str
    recommended_action: str
    technical_findings: List[str]
    engineering_disclaimer: str

