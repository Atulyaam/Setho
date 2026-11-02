/**
 * Stetho Typed API Client
 * Connects securely to the Next.js API proxy, passing HttpOnly session cookies.
 */

export interface UserProfile {
  user_id: string;
  username: string;
  role: string;
  created_at: string;
}

export interface CalibrationThresholds {
  mean_normal_score: number;
  std_normal_score: number;
  percentiles: Record<string, number>;
  operating_threshold: number;
  theta_lower: number;
  theta_upper: number;
  enrolled_sample_count: number;
  calibration_version: string;
  calibration_status: "calibrated" | "provisional" | "uncalibrated";
  calibrated_at: string;
}

export interface MachineProfile {
  machine_id: string;
  name: string;
  category: string;
  description: string;
  operating_condition: string;
  selected_model_policy: "stetho_msae" | "dense_autoencoder";
  enrollment_status: "calibrated" | "provisional" | "uncalibrated";
  enrolled_recording_count: number;
  calibration_status: "calibrated" | "provisional" | "uncalibrated";
  calibration_profile?: CalibrationThresholds | null;
  calibration_version: string;
  owner_id?: string;
  created_at: string;
  updated_at: string;
}

export interface EnrollmentRecord {
  enrollment_id: string;
  machine_id: string;
  upload_id: string;
  duration_sec: number;
  sample_rate: number;
  channels: number;
  quality_status: string;
  is_confirmed_normal: boolean;
  created_at: string;
  enrolled_count: number;
  enrollment_status: string;
  operating_threshold?: number;
  theta_lower?: number;
  theta_upper?: number;
  calibration_version?: string;
  notice?: string;
}

export interface AnalysisRecord {
  analysis_id: string;
  machine_id: string;
  model_name: string;
  model_version: string;
  analysis_status: string;
  quality_status: string;
  decision: "NORMAL_LIKE" | "UNUSUAL" | "INCONCLUSIVE";
  anomaly_score: number | null;
  operating_threshold: number | null;
  theta_lower: number | null;
  theta_upper: number | null;
  is_provisional: boolean;
  processing_duration_ms: number;
  reasons: string[];
  metrics: Record<string, any>;
  created_at: string;
}

export interface InspectionReport {
  report_id: string;
  generated_at: string;
  machine: MachineProfile;
  analysis: AnalysisRecord;
  executive_summary: string;
  recommended_action: string;
  technical_findings: string[];
  engineering_disclaimer: string;
}

export interface JobRecord {
  job_id: string;
  machine_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress_pct: number;
  analysis_id?: string | null;
  error_message?: string | null;
  retry_count: number;
  created_at: string;
  updated_at: string;
}

// Sprint 7 Asset, Inference & Registry Models
export interface AssetCreateRequest {
  asset_id: string;
  name: string;
  category: "fan" | "pump" | "valve";
  description?: string;
  operating_condition?: string;
  assigned_model?: string;
  metadata?: Record<string, any>;
}

export interface AssetResponse {
  asset_id: string;
  name: string;
  category: "fan" | "pump" | "valve" | string;
  description: string;
  operating_condition: string;
  assigned_model: string;
  assigned_model_version: string;
  calibration_status: "uncalibrated" | "provisional" | "calibrated";
  calibration_version: string;
  enrolled_recording_count: number;
  metadata: Record<string, any>;
  owner_id: string;
  created_at: string;
  updated_at: string;
}

export interface AssetListResponse {
  assets: AssetResponse[];
  total_count: number;
}

export interface AssetCalibrationResponse {
  asset_id: string;
  model_type: string;
  calibration_version: string;
  enrolled_sample_count: number;
  operating_threshold: number;
  theta_lower: number;
  theta_upper: number;
  enrolled_mean: number;
  enrolled_std: number;
  is_provisional: boolean;
  calibration_status: "uncalibrated" | "provisional" | "calibrated";
  recalculated_at_utc: string;
  notice: string;
  enrolled_scores: number[];
  latent_mean?: number[];
}

export interface InferenceResponse {
  result_id?: string;
  asset_id: string;
  model_version: string;
  model_name: string;
  anomaly_score: number | null;
  anomaly_decision: "NORMAL_LIKE" | "UNUSUAL" | "INCONCLUSIVE";
  threshold_used: number | null;
  theta_lower: number | null;
  theta_upper: number | null;
  processing_latency_ms: number;
  calibration_status: string;
  quality_status: string;
  is_provisional: boolean;
  reasons: string[];
  metrics: Record<string, any>;
  created_at?: string;
}

export interface InferenceHistoryRecord {
  result_id: string;
  asset_id: string;
  model_name: string;
  model_version: string;
  anomaly_score: number | null;
  anomaly_decision: "NORMAL_LIKE" | "UNUSUAL" | "INCONCLUSIVE";
  threshold_used: number | null;
  quality_status: string;
  processing_latency_ms: number;
  is_provisional: boolean;
  created_at: string;
}

export interface InferenceHistoryListResponse {
  asset_id: string;
  results: InferenceHistoryRecord[];
  total_count: number;
}

export interface ModelDescriptor {
  model_name: string;
  version: string;
  supported_categories: string[];
  recommended_profile: string;
  description: string;
  runtime_backend: string;
  is_int8_quantized: boolean;
}

export interface ModelListResponse {
  models: ModelDescriptor[];
  total_count: number;
}

class APIError extends Error {
  status: number;
  code?: string;
  details?: any;

  constructor(message: string, status: number, code?: string, details?: any) {
    super(message);
    this.name = "APIError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  const contentType = res.headers.get("content-type") || "";
  let data: any = {};
  if (contentType.includes("application/json")) {
    data = await res.json();
  }

  if (!res.ok) {
    const errorMsg =
      data.detail ||
      (typeof data.error === "string" ? data.error : data.error?.message) ||
      `HTTP error ${res.status}`;
    const errorCode = data.code || data.error?.code;
    throw new APIError(errorMsg, res.status, errorCode, data.details || data.error?.details);
  }

  return data as T;
}

export const apiClient = {
  // Authentication
  async login(username: string, password: string): Promise<{ user: UserProfile }> {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    return handleResponse(res);
  },

  async register(username: string, password: string, role: string = "inspector"): Promise<{ user: UserProfile }> {
    const res = await fetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, role }),
    });
    return handleResponse(res);
  },

  async logout(): Promise<void> {
    await fetch("/api/auth/logout", { method: "POST" });
  },

  async getMe(): Promise<{ user: UserProfile }> {
    const res = await fetch("/api/auth/me", { cache: "no-store" });
    return handleResponse(res);
  },

  // Machines
  async getMachines(): Promise<{ machines: MachineProfile[]; total_count: number }> {
    const res = await fetch("/api/backend/machines", { cache: "no-store" });
    return handleResponse(res);
  },

  async getMachine(machineId: string): Promise<MachineProfile> {
    const res = await fetch(`/api/backend/machines/${machineId}`, { cache: "no-store" });
    return handleResponse(res);
  },

  async createMachine(data: {
    machine_id: string;
    name: string;
    category: string;
    description?: string;
    operating_condition?: string;
    selected_model_policy?: string;
  }): Promise<MachineProfile> {
    const res = await fetch("/api/backend/machines", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  // Enrollments
  async enrollAudio(machineId: string, formData: FormData): Promise<EnrollmentRecord> {
    const res = await fetch(`/api/backend/machines/${machineId}/enrollments`, {
      method: "POST",
      body: formData,
    });
    return handleResponse(res);
  },

  async getEnrollments(machineId: string): Promise<{ machine_id: string; enrollments: any[]; total_count: number }> {
    const res = await fetch(`/api/backend/machines/${machineId}/enrollments`, { cache: "no-store" });
    return handleResponse(res);
  },

  // Analyses
  async analyzeAudio(machineId: string, formData: FormData): Promise<AnalysisRecord> {
    const res = await fetch(`/api/backend/machines/${machineId}/analyze`, {
      method: "POST",
      body: formData,
    });
    return handleResponse(res);
  },

  async submitAnalysisJob(machineId: string, formData: FormData): Promise<JobRecord> {
    const res = await fetch(`/api/backend/machines/${machineId}/analysis-jobs`, {
      method: "POST",
      body: formData,
    });
    return handleResponse(res);
  },

  async getAnalysisJob(jobId: string): Promise<JobRecord> {
    const res = await fetch(`/api/backend/analysis-jobs/${jobId}`, { cache: "no-store" });
    return handleResponse(res);
  },

  async getAnalysis(analysisId: string): Promise<AnalysisRecord> {
    const res = await fetch(`/api/backend/analyses/${analysisId}`, { cache: "no-store" });
    return handleResponse(res);
  },

  async getInspectionReport(analysisId: string): Promise<InspectionReport> {
    const res = await fetch(`/api/backend/analyses/${analysisId}/report`, { cache: "no-store" });
    return handleResponse(res);
  },

  async getMachineAnalyses(machineId: string): Promise<{ machine_id: string; analyses: AnalysisRecord[]; total_count: number }> {
    const res = await fetch(`/api/backend/machines/${machineId}/analyses`, { cache: "no-store" });
    return handleResponse(res);
  },

  async getRecentAnalyses(limit: number = 10): Promise<{ analyses: AnalysisRecord[]; total_count: number }> {
    const res = await fetch(`/api/backend/analyses/recent?limit=${limit}`, { cache: "no-store" });
    return handleResponse(res);
  },

  // Assets & Inference (Sprint 7)
  async listAssets(category?: string): Promise<AssetListResponse> {
    const q = category ? `?category=${category}` : "";
    const res = await fetch(`/api/backend/assets${q}`, { cache: "no-store" });
    return handleResponse(res);
  },

  async getAsset(assetId: string): Promise<AssetResponse> {
    const res = await fetch(`/api/backend/assets/${assetId}`, { cache: "no-store" });
    return handleResponse(res);
  },

  async createAsset(data: AssetCreateRequest): Promise<AssetResponse> {
    const res = await fetch("/api/backend/assets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  async calibrateAsset(
    assetId: string,
    files: File[],
    notes?: string,
    allowCrossCategory: boolean = false
  ): Promise<AssetCalibrationResponse> {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("is_confirmed_normal", "true");
    if (notes) formData.append("notes", notes);
    if (allowCrossCategory) formData.append("allow_cross_category", "true");

    const res = await fetch(`/api/backend/assets/${assetId}/calibrate`, {
      method: "POST",
      body: formData,
    });
    return handleResponse(res);
  },

  async rollbackCalibration(assetId: string, targetVersion?: string): Promise<AssetCalibrationResponse> {
    const q = targetVersion ? `?target_version=${targetVersion}` : "";
    const res = await fetch(`/api/backend/assets/${assetId}/calibration/rollback${q}`, {
      method: "POST",
    });
    return handleResponse(res);
  },

  async executeInference(
    assetId: string,
    file: File,
    profile?: string,
    allowCrossCategory: boolean = false
  ): Promise<InferenceResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("asset_id", assetId);
    if (profile) formData.append("profile", profile);
    if (allowCrossCategory) formData.append("allow_cross_category", "true");

    const res = await fetch("/api/backend/inference", {
      method: "POST",
      body: formData,
    });
    return handleResponse(res);
  },

  async getAssetHistory(assetId: string, limit: number = 50): Promise<InferenceHistoryListResponse> {
    const res = await fetch(`/api/backend/inference/history/${assetId}?limit=${limit}`, {
      cache: "no-store",
    });
    return handleResponse(res);
  },

  async listModels(): Promise<ModelListResponse> {
    const res = await fetch("/api/backend/models", { cache: "no-store" });
    return handleResponse(res);
  },

  // --- Sprint 7.7 / E8 Field Data & Pilot Governance ---

  async getFieldDataDashboard(): Promise<FieldDashboardStats> {
    const res = await fetch("/api/backend/field-data/dashboard", { cache: "no-store" });
    return handleResponse(res);
  },

  async listFieldRecordings(params?: {
    machine_category?: string;
    machine_id?: string;
    dataset_role?: string;
    current_label?: string;
    verification_status?: string;
    limit?: number;
    offset?: number;
  }): Promise<FieldRecordingListResponse> {
    const sp = new URLSearchParams();
    if (params?.machine_category) sp.append("machine_category", params.machine_category);
    if (params?.machine_id) sp.append("machine_id", params.machine_id);
    if (params?.dataset_role) sp.append("dataset_role", params.dataset_role);
    if (params?.current_label) sp.append("current_label", params.current_label);
    if (params?.verification_status) sp.append("verification_status", params.verification_status);
    if (params?.limit) sp.append("limit", params.limit.toString());
    if (params?.offset) sp.append("offset", params.offset.toString());
    const query = sp.toString() ? `?${sp.toString()}` : "";
    const res = await fetch(`/api/backend/field-data/recordings${query}`, { cache: "no-store" });
    return handleResponse(res);
  },

  async getFieldRecording(recordingId: string): Promise<FieldRecording> {
    const res = await fetch(`/api/backend/field-data/recordings/${recordingId}`, { cache: "no-store" });
    return handleResponse(res);
  },

  async importFieldRecording(formData: FormData): Promise<FieldRecording> {
    const res = await fetch("/api/backend/field-data/import", {
      method: "POST",
      body: formData,
    });
    return handleResponse(res);
  },

  async annotateFieldRecording(
    recordingId: string,
    payload: {
      label: "normal" | "abnormal" | "uncertain";
      reason: string;
      evidence_source: string;
      verification_status?: string;
      inspection_notes?: string;
    }
  ): Promise<FieldAnnotation> {
    const res = await fetch(`/api/backend/field-data/recordings/${recordingId}/annotate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  async getFieldRecordingAnnotations(recordingId: string): Promise<FieldAnnotation[]> {
    const res = await fetch(`/api/backend/field-data/recordings/${recordingId}/annotations`, { cache: "no-store" });
    return handleResponse(res);
  },

  async updateFieldRecordingRole(recordingId: string, datasetRole: string): Promise<FieldRecording> {
    const res = await fetch(`/api/backend/field-data/recordings/${recordingId}/role`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_role: datasetRole }),
    });
    return handleResponse(res);
  },

  async listDatasetManifests(datasetRole?: string): Promise<DatasetManifest[]> {
    const query = datasetRole ? `?dataset_role=${datasetRole}` : "";
    const res = await fetch(`/api/backend/field-data/manifests${query}`, { cache: "no-store" });
    return handleResponse(res);
  },

  async createDatasetManifest(payload: {
    name: string;
    dataset_role: string;
    machine_category: string;
    recording_ids: string[];
    seal_now?: boolean;
    metadata?: Record<string, any>;
  }): Promise<DatasetManifest> {
    const res = await fetch("/api/backend/field-data/manifests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  async downloadProtocolChecklist(format: "json" | "markdown" = "markdown"): Promise<Blob | any> {
    const res = await fetch(`/api/backend/field-data/protocol-checklist?export_format=${format}`, { cache: "no-store" });
    if (format === "markdown") {
      return res.blob();
    }
    return handleResponse(res);
  },
};

export interface FieldRecording {
  recording_id: string;
  asset_id?: string;
  machine_category: string;
  machine_id: string;
  sensor_id: string;
  channel_index: number;
  microphone_position: string;
  sample_rate: number;
  channels: number;
  duration_sec: number;
  original_filename: string;
  sha256_hash: string;
  captured_at: string;
  operating_condition: string;
  load_level: string;
  operator_id: string;
  provenance_source: string;
  consent_status: string;
  dataset_role: "unassigned" | "training" | "development" | "calibration_enrollment" | "sealed_evaluation";
  current_label: "normal" | "abnormal" | "uncertain" | "unlabelled";
  verification_status: "VERIFIED" | "PENDING";
  quality_status: "PASSED" | "FAILED";
  quality_telemetry: {
    rms_dbfs: number;
    peak_amplitude: number;
    clipping_ratio: number;
    sample_rate: number;
    channels: number;
    duration_sec: number;
  };
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface FieldRecordingListResponse {
  items: FieldRecording[];
  total_count: number;
  limit: number;
  offset: number;
}

export interface FieldAnnotation {
  annotation_id: string;
  recording_id: string;
  label: "normal" | "abnormal" | "uncertain";
  annotator_id: string;
  reason: string;
  evidence_source: string;
  verification_status: string;
  inspection_notes?: string;
  revision_number: number;
  created_at: string;
}

export interface DatasetManifest {
  manifest_id: string;
  name: string;
  dataset_role: string;
  machine_category: string;
  is_sealed: boolean;
  sealed_at?: string;
  sha256_checksum: string;
  recording_count: number;
  recordings: string[];
  metadata?: Record<string, any>;
  created_by: string;
  created_at: string;
}

export interface FieldDashboardStats {
  total_imported: number;
  by_category: Record<string, number>;
  by_machine_id: Record<string, number>;
  verified_normal: number;
  verified_abnormal: number;
  uncertain_count: number;
  unlabelled_count: number;
  by_verification_status: Record<string, number>;
  by_dataset_role: Record<string, number>;
  by_sensor_channel: Record<string, number>;
  duplicates_detected: number;
  rejections_count: number;
  missing_metadata_count: number;
  independent_evaluation_readiness: {
    is_ready: boolean;
    sealed_normal_recordings: number;
    sealed_abnormal_recordings: number;
    minimum_required_per_class: number;
    verdict: string;
    warning?: string | null;
  };
}

