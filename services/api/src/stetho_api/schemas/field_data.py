"""Pydantic schemas for Field Data Collection and Dataset Governance."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FieldRecordingResponse(BaseModel):
    """Field recording metadata representation."""
    recording_id: str
    asset_id: Optional[str] = None
    machine_category: str
    machine_id: str
    sensor_id: str
    channel_index: int = 0
    microphone_position: str = "primary_bearing"
    sample_rate: int
    channels: int
    duration_sec: float
    file_path: str
    original_filename: str
    sha256_hash: str
    captured_at: str
    operating_condition: str
    load_level: str
    operator_id: str
    provenance_source: str
    consent_status: str
    dataset_role: str
    current_label: str
    verification_status: str
    quality_status: str
    quality_telemetry: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class FieldRecordingListResponse(BaseModel):
    """Paginated list of field recordings."""
    items: List[FieldRecordingResponse]
    total_count: int
    limit: int
    offset: int


class FieldAnnotationCreateRequest(BaseModel):
    """Payload for submitting a ground-truth annotation."""
    label: str = Field(description="'normal', 'abnormal', 'uncertain', or 'unlabelled'")
    reason: str = Field(description="Physical diagnostic justification")
    evidence_source: str = Field(
        description="e.g. 'VIBRATION_ORDER_ANALYSIS', 'THERMAL_SCAN', 'OIL_ANALYSIS', 'TEARDOWN_INSPECTION', 'VISUAL_INSPECTION', 'OPERATIONAL_LOG'"
    )
    inspection_notes: Optional[str] = Field(default="", description="Additional technician notes")


class FieldAnnotationResponse(BaseModel):
    """Ground-truth annotation response with revision number."""
    annotation_id: str
    recording_id: str
    label: str
    annotator_id: str
    reason: str
    evidence_source: str
    verification_status: str
    inspection_notes: str
    revision_number: int
    created_at: str


class DatasetRoleUpdateRequest(BaseModel):
    """Payload to assign or reclassify a recording's dataset role."""
    dataset_role: str = Field(
        description="'training', 'development', 'calibration_enrollment', 'sealed_evaluation', or 'unassigned'"
    )


class ManifestCreateRequest(BaseModel):
    """Payload for freezing and sealing a dataset manifest."""
    name: str = Field(description="Human-readable manifest name")
    dataset_role: str = Field(
        description="'training', 'development', 'calibration_enrollment', or 'sealed_evaluation'"
    )
    machine_category: str = Field(description="Target machine category")
    recording_ids: List[str] = Field(description="List of included recording IDs")
    seal_now: bool = Field(default=True, description="Whether to seal and compute SHA-256 checksum immediately")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ManifestResponse(BaseModel):
    """Dataset manifest representation."""
    manifest_id: str
    name: str
    dataset_role: str
    machine_category: str
    is_sealed: bool
    sealed_at: Optional[str] = None
    sha256_checksum: Optional[str] = None
    recording_count: int
    recordings: List[str]
    created_by: str
    created_at: str


class IndependentEvaluationReadiness(BaseModel):
    """Evaluation readiness status and warning."""
    is_ready: bool
    sealed_normal_recordings: int
    sealed_abnormal_recordings: int
    minimum_required_per_class: int = 10
    verdict: str
    warning: Optional[str] = None


class FieldDashboardStatsResponse(BaseModel):
    """Summary dashboard statistics for field data ingestion and pilot readiness."""
    total_imported: int
    by_category: Dict[str, int]
    by_machine_id: Dict[str, int]
    verified_normal: int
    verified_abnormal: int
    uncertain_count: int
    unlabelled_count: int
    by_verification_status: Dict[str, int]
    by_dataset_role: Dict[str, int]
    by_sensor_channel: Dict[str, int]
    duplicates_detected: int
    rejections_count: int
    missing_metadata_count: int
    independent_evaluation_readiness: IndependentEvaluationReadiness


class ProtocolStep(BaseModel):
    step: int
    title: str
    action: str


class ProtocolChecklistResponse(BaseModel):
    """Downloadable field data collection protocol checklist."""
    title: str
    version: str
    specifications: Dict[str, str]
    steps: List[ProtocolStep]
    evidence_sources: List[str]
