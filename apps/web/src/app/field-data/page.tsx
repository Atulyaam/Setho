"use client";

import React, { useCallback, useEffect, useState } from "react";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { useAuth } from "@/context/AuthContext";
import {
  apiClient,
  FieldDashboardStats,
  FieldRecording,
  FieldAnnotation,
  DatasetManifest,
} from "@/lib/api-client";
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  Download,
  FileCheck,
  FileText,
  Filter,
  History,
  Info,
  Layers,
  Lock,
  PlusCircle,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Upload,
  Volume2,
  XCircle,
} from "lucide-react";

export default function FieldDataPage() {
  const { user } = useAuth();

  // State
  const [stats, setStats] = useState<FieldDashboardStats | null>(null);
  const [recordings, setRecordings] = useState<FieldRecording[]>([]);
  const [manifests, setManifests] = useState<DatasetManifest[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Filters
  const [filterCategory, setFilterCategory] = useState<string>("all");
  const [filterRole, setFilterRole] = useState<string>("all");
  const [filterLabel, setFilterLabel] = useState<string>("all");
  const [activeTab, setActiveTab] = useState<"recordings" | "manifests" | "audit">("recordings");

  // Modals
  const [isImportModalOpen, setIsImportModalOpen] = useState(false);
  const [isAnnotateModalOpen, setIsAnnotateModalOpen] = useState(false);
  const [isRoleModalOpen, setIsRoleModalOpen] = useState(false);
  const [isManifestModalOpen, setIsManifestModalOpen] = useState(false);

  // Active target recording for annotate / role
  const [activeRecording, setActiveRecording] = useState<FieldRecording | null>(null);
  const [activeAnnotations, setActiveAnnotations] = useState<FieldAnnotation[]>([]);

  // Form states
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [formCategory, setFormCategory] = useState("fan");
  const [formMachineId, setFormMachineId] = useState("");
  const [formSensorId, setFormSensorId] = useState("mic_sensor_01");
  const [formChannelIndex, setFormChannelIndex] = useState(0);
  const [formPosition, setFormPosition] = useState("primary_bearing");
  const [formCondition, setFormCondition] = useState("steady_full_load");
  const [formLoad, setFormLoad] = useState("100%");
  const [formRole, setFormRole] = useState("unassigned");

  // Annotation form
  const [annotLabel, setAnnotLabel] = useState<"normal" | "abnormal" | "uncertain">("normal");
  const [annotEvidence, setAnnotEvidence] = useState("VISUAL_INSPECTION");
  const [annotReason, setAnnotReason] = useState("");
  const [annotNotes, setAnnotNotes] = useState("");

  // Role form
  const [selectedRole, setSelectedRole] = useState("unassigned");

  // Manifest form
  const [manifestName, setManifestName] = useState("");
  const [manifestRole, setManifestRole] = useState("sealed_evaluation");
  const [manifestCategory, setManifestCategory] = useState("fan");
  const [sealNow, setSealNow] = useState(true);

  // Load data
  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [dashStats, recsRes, manRes] = await Promise.all([
        apiClient.getFieldDataDashboard(),
        apiClient.listFieldRecordings({
          machine_category: filterCategory === "all" ? undefined : filterCategory,
          dataset_role: filterRole === "all" ? undefined : filterRole,
          current_label: filterLabel === "all" ? undefined : filterLabel,
          limit: 100,
        }),
        apiClient.listDatasetManifests(),
      ]);
      setStats(dashStats);
      setRecordings(recsRes.items);
      setManifests(manRes);
    } catch (err: any) {
      setError(err.message || "Failed to load field data telemetry.");
    } finally {
      setIsLoading(false);
    }
  }, [filterCategory, filterRole, filterLabel]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Handle Checklist Download
  const handleDownloadProtocol = async () => {
    try {
      const blob = await apiClient.downloadProtocolChecklist("markdown");
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "stetho_field_collection_protocol.md";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      setError("Failed to download protocol checklist: " + err.message);
    }
  };

  // Open Annotation Modal
  const handleOpenAnnotate = async (rec: FieldRecording) => {
    setActiveRecording(rec);
    setAnnotLabel(rec.current_label === "unlabelled" ? "normal" : (rec.current_label as any));
    setAnnotReason("");
    setAnnotNotes("");
    setAnnotEvidence("VISUAL_INSPECTION");
    try {
      const history = await apiClient.getFieldRecordingAnnotations(rec.recording_id);
      setActiveAnnotations(history);
    } catch {
      setActiveAnnotations([]);
    }
    setIsAnnotateModalOpen(true);
  };

  // Submit Annotation
  const handleSubmitAnnotation = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeRecording) return;
    try {
      await apiClient.annotateFieldRecording(activeRecording.recording_id, {
        label: annotLabel,
        reason: annotReason || `Field operator verified as ${annotLabel}`,
        evidence_source: annotEvidence,
        verification_status: "verified",
        inspection_notes: annotNotes,
      });
      setSuccessMsg(`Recording ${activeRecording.recording_id} ground-truth updated to ${annotLabel.toUpperCase()}.`);
      setIsAnnotateModalOpen(false);
      loadData();
    } catch (err: any) {
      setError(err.message || "Failed to submit annotation.");
    }
  };

  // Open Role Modal
  const handleOpenRole = (rec: FieldRecording) => {
    setActiveRecording(rec);
    setSelectedRole(rec.dataset_role);
    setIsRoleModalOpen(true);
  };

  // Submit Role
  const handleSubmitRole = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeRecording) return;
    try {
      await apiClient.updateFieldRecordingRole(activeRecording.recording_id, selectedRole);
      setSuccessMsg(`Dataset role for ${activeRecording.recording_id} updated to ${selectedRole}.`);
      setIsRoleModalOpen(false);
      loadData();
    } catch (err: any) {
      setError(err.message || "Failed to update dataset role.");
    }
  };

  // Submit Import
  const handleImportSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) {
      setError("Please select a valid WAV audio file.");
      return;
    }
    const formData = new FormData();
    formData.append("file", uploadFile);
    formData.append("machine_category", formCategory);
    formData.append("machine_id", formMachineId.trim() || `mach_${Date.now()}`);
    formData.append("sensor_id", formSensorId.trim() || "mic_01");
    formData.append("channel_index", formChannelIndex.toString());
    formData.append("microphone_position", formPosition);
    formData.append("operating_condition", formCondition);
    formData.append("load_level", formLoad);
    formData.append("dataset_role", formRole);

    try {
      await apiClient.importFieldRecording(formData);
      setSuccessMsg(`Field recording ${uploadFile.name} successfully imported and verified.`);
      setIsImportModalOpen(false);
      setUploadFile(null);
      loadData();
    } catch (err: any) {
      setError(err.message || "Upload failed. Quality checks or duplicate file rejected.");
    }
  };

  // Submit Manifest
  const handleCreateManifest = async (e: React.FormEvent) => {
    e.preventDefault();
    // Gather matching recordings
    const matchingIds = recordings
      .filter((r) => r.machine_category === manifestCategory && (manifestRole === "sealed_evaluation" ? r.dataset_role === "sealed_evaluation" : true))
      .map((r) => r.recording_id);

    if (matchingIds.length === 0) {
      setError(`No recordings found matching category '${manifestCategory}' and role '${manifestRole}' to manifest.`);
      return;
    }

    try {
      const created = await apiClient.createDatasetManifest({
        name: manifestName || `Manifest ${manifestCategory.toUpperCase()} ${new Date().toLocaleDateString()}`,
        dataset_role: manifestRole,
        machine_category: manifestCategory,
        recording_ids: matchingIds,
        seal_now: sealNow,
      });
      setSuccessMsg(`Dataset manifest '${created.name}' created and sealed (SHA-256: ${created.sha256_checksum.slice(0, 16)}...).`);
      setIsManifestModalOpen(false);
      loadData();
    } catch (err: any) {
      setError(err.message || "Failed to create manifest.");
    }
  };

  return (
    <ProtectedRoute>
      <div className="space-y-6">
        {/* Header & Primary Actions */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div>
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-cyan-600/20 text-cyan-400 rounded-xl border border-cyan-500/30">
                <Database className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-100 tracking-tight">
                  Field Data Collection & Pilot Readiness
                </h1>
                <p className="text-sm text-slate-400">
                  Industrial acoustic ingestion, provenance logging, ground-truth annotation, and cryptographic governance.
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={handleDownloadProtocol}
              className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium flex items-center gap-2 transition"
            >
              <Download className="w-4 h-4 text-cyan-400" />
              <span>Operator Protocol (MD)</span>
            </button>
            <button
              onClick={() => setIsManifestModalOpen(true)}
              className="px-3.5 py-2 bg-indigo-600/30 hover:bg-indigo-600/40 text-indigo-200 border border-indigo-500/30 rounded-lg text-xs font-medium flex items-center gap-2 transition"
            >
              <Lock className="w-4 h-4 text-indigo-400" />
              <span>Seal Manifest</span>
            </button>
            <button
              onClick={() => setIsImportModalOpen(true)}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-semibold flex items-center gap-2 shadow-sm transition"
            >
              <Upload className="w-4 h-4" />
              <span>Import Field Audio</span>
            </button>
          </div>
        </div>

        {/* Global Notifications */}
        {error && (
          <div className="p-4 bg-red-950/60 border border-red-800/80 rounded-xl text-sm text-red-200 flex items-start justify-between gap-3 shadow-md">
            <div className="flex items-start gap-3">
              <XCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-red-100">Notice: </span>
                {error}
              </div>
            </div>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-200 text-xs font-bold">
              Dismiss
            </button>
          </div>
        )}

        {successMsg && (
          <div className="p-4 bg-emerald-950/60 border border-emerald-800/80 rounded-xl text-sm text-emerald-200 flex items-start justify-between gap-3 shadow-md">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
              <div>{successMsg}</div>
            </div>
            <button onClick={() => setSuccessMsg(null)} className="text-emerald-400 hover:text-emerald-200 text-xs font-bold">
              Dismiss
            </button>
          </div>
        )}

        {/* MANDATORY WARNING BANNER: Insufficient Independent Evidence */}
        {stats?.independent_evaluation_readiness && !stats.independent_evaluation_readiness.is_ready && (
          <div className="p-5 bg-amber-950/40 border-2 border-amber-500/60 rounded-xl shadow-lg relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <AlertTriangle className="w-32 h-32 text-amber-500" />
            </div>
            <div className="flex items-start gap-3.5 relative z-10">
              <div className="p-2 bg-amber-500/20 text-amber-400 rounded-lg shrink-0 border border-amber-500/40 mt-0.5">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <h3 className="font-bold text-amber-200 text-base">
                    INSUFFICIENT INDEPENDENT EVIDENCE: Controlled Field Pilot in Progress
                  </h3>
                  <span className="text-[10px] font-mono px-2 py-0.5 bg-amber-500/20 text-amber-300 rounded border border-amber-500/30">
                    GOVERNANCE GATE ACTIVE
                  </span>
                </div>
                <p className="text-sm text-amber-100/90 leading-relaxed">
                  {stats.independent_evaluation_readiness.warning}
                </p>
                <div className="pt-2 flex flex-wrap items-center gap-4 text-xs font-medium text-amber-200">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                    Sealed Normal: <strong>{stats.independent_evaluation_readiness.sealed_normal_recordings}</strong> / {stats.independent_evaluation_readiness.minimum_required_per_class}
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-red-400"></span>
                    Sealed Abnormal: <strong>{stats.independent_evaluation_readiness.sealed_abnormal_recordings}</strong> / {stats.independent_evaluation_readiness.minimum_required_per_class}
                  </span>
                  <span className="text-amber-300/80">
                    Research models are prohibited from production deployment until sealed validation criteria are fully satisfied.
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Telemetry KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-xl space-y-1">
            <span className="text-xs text-slate-400">Total Imported</span>
            <div className="text-2xl font-bold text-slate-100">{stats?.total_imported ?? 0}</div>
            <span className="text-[11px] text-slate-500">Cryptographically hashed</span>
          </div>

          <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-xl space-y-1">
            <span className="text-xs text-slate-400">Verified Normal</span>
            <div className="text-2xl font-bold text-emerald-400">{stats?.verified_normal ?? 0}</div>
            <span className="text-[11px] text-emerald-500/80">Audited ground-truth</span>
          </div>

          <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-xl space-y-1">
            <span className="text-xs text-slate-400">Verified Abnormal</span>
            <div className="text-2xl font-bold text-red-400">{stats?.verified_abnormal ?? 0}</div>
            <span className="text-[11px] text-red-500/80">Triangulated defects</span>
          </div>

          <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-xl space-y-1">
            <span className="text-xs text-slate-400">Unlabelled / Pending</span>
            <div className="text-2xl font-bold text-amber-400">{stats?.unlabelled_count ?? 0}</div>
            <span className="text-[11px] text-amber-500/80">Awaiting technician</span>
          </div>

          <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-xl space-y-1">
            <span className="text-xs text-slate-400">Duplicates Filtered</span>
            <div className="text-2xl font-bold text-purple-400">{stats?.duplicates_detected ?? 0}</div>
            <span className="text-[11px] text-purple-400/80">SHA-256 rejected</span>
          </div>

          <div className="p-4 bg-slate-900/70 border border-slate-800 rounded-xl space-y-1">
            <span className="text-xs text-slate-400">Quality Gating</span>
            <div className="text-2xl font-bold text-cyan-400">{stats?.rejections_count ?? 0}</div>
            <span className="text-[11px] text-cyan-400/80">Rejections logged</span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-800 text-sm font-medium">
          <button
            onClick={() => setActiveTab("recordings")}
            className={`pb-3 px-4 border-b-2 transition flex items-center gap-2 ${
              activeTab === "recordings"
                ? "border-cyan-500 text-cyan-400 font-semibold"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <Volume2 className="w-4 h-4" />
            <span>Field Audio Recordings ({recordings.length})</span>
          </button>
          <button
            onClick={() => setActiveTab("manifests")}
            className={`pb-3 px-4 border-b-2 transition flex items-center gap-2 ${
              activeTab === "manifests"
                ? "border-cyan-500 text-cyan-400 font-semibold"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <Lock className="w-4 h-4" />
            <span>Sealed Manifests ({manifests.length})</span>
          </button>
          <button
            onClick={() => setActiveTab("audit")}
            className={`pb-3 px-4 border-b-2 transition flex items-center gap-2 ${
              activeTab === "audit"
                ? "border-cyan-500 text-cyan-400 font-semibold"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            <span>Quality Gating & Rejection Audit</span>
          </button>
        </div>

        {/* Tab 1: Field Recordings List */}
        {activeTab === "recordings" && (
          <div className="space-y-4">
            {/* Filter Bar */}
            <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-900/60 border border-slate-800 rounded-xl text-xs">
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-1.5 text-slate-400">
                  <Filter className="w-3.5 h-3.5" />
                  <span>Filters:</span>
                </div>

                <select
                  value={filterCategory}
                  onChange={(e) => setFilterCategory(e.target.value)}
                  className="bg-slate-800 border border-slate-700 text-slate-200 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500"
                >
                  <option value="all">All Categories</option>
                  <option value="fan">Fan</option>
                  <option value="pump">Pump</option>
                  <option value="valve">Valve</option>
                </select>

                <select
                  value={filterRole}
                  onChange={(e) => setFilterRole(e.target.value)}
                  className="bg-slate-800 border border-slate-700 text-slate-200 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500"
                >
                  <option value="all">All Dataset Roles</option>
                  <option value="unassigned">Unassigned</option>
                  <option value="training">Training</option>
                  <option value="development">Development</option>
                  <option value="calibration_enrollment">Calibration Enrollment</option>
                  <option value="sealed_evaluation">Sealed Evaluation</option>
                </select>

                <select
                  value={filterLabel}
                  onChange={(e) => setFilterLabel(e.target.value)}
                  className="bg-slate-800 border border-slate-700 text-slate-200 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500"
                >
                  <option value="all">All Labels</option>
                  <option value="normal">Normal</option>
                  <option value="abnormal">Abnormal</option>
                  <option value="uncertain">Uncertain</option>
                  <option value="unlabelled">Unlabelled</option>
                </select>
              </div>

              <button
                onClick={loadData}
                className="p-1.5 hover:bg-slate-800 text-slate-400 hover:text-slate-200 rounded transition"
                title="Refresh table"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>

            {/* Table */}
            <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-900/50">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-800/60 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Recording ID</th>
                    <th className="px-4 py-3">Machine & Category</th>
                    <th className="px-4 py-3">Sensor / Position</th>
                    <th className="px-4 py-3">Audio Telemetry</th>
                    <th className="px-4 py-3">Dataset Role</th>
                    <th className="px-4 py-3">Ground Truth</th>
                    <th className="px-4 py-3">Captured</th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {recordings.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                        No field recordings matching filter criteria. Ingest field audio via the button above.
                      </td>
                    </tr>
                  ) : (
                    recordings.map((rec) => (
                      <tr key={rec.recording_id} className="hover:bg-slate-800/30 transition">
                        <td className="px-4 py-3 font-mono text-[11px] text-cyan-400">
                          {rec.recording_id}
                          <div className="text-[10px] text-slate-500 truncate max-w-[140px]" title={rec.sha256_hash}>
                            {rec.sha256_hash.slice(0, 14)}...
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="font-medium text-slate-200">{rec.machine_id}</div>
                          <span className="inline-block mt-0.5 text-[10px] uppercase font-semibold px-1.5 py-0.5 bg-slate-800 text-slate-400 rounded">
                            {rec.machine_category}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-slate-300">{rec.sensor_id} (ch {rec.channel_index})</div>
                          <div className="text-[11px] text-slate-500">{rec.microphone_position}</div>
                        </td>
                        <td className="px-4 py-3 font-mono text-[11px]">
                          <div>{rec.sample_rate} Hz | {rec.duration_sec.toFixed(1)}s</div>
                          <div className="text-slate-500">
                            {rec.quality_telemetry?.rms_dbfs ? `${rec.quality_telemetry.rms_dbfs} dBFS` : "PASSED"}
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className={`px-2 py-0.5 text-[11px] font-semibold rounded-full border ${
                              rec.dataset_role === "sealed_evaluation"
                                ? "bg-indigo-950/60 text-indigo-300 border-indigo-700/60"
                                : rec.dataset_role === "calibration_enrollment"
                                ? "bg-cyan-950/60 text-cyan-300 border-cyan-700/60"
                                : rec.dataset_role === "training"
                                ? "bg-slate-800 text-slate-300 border-slate-700"
                                : "bg-slate-800/50 text-slate-500 border-slate-700/40"
                            }`}
                          >
                            {rec.dataset_role}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-1.5">
                            <span
                              className={`w-2 h-2 rounded-full ${
                                rec.current_label === "normal"
                                  ? "bg-emerald-400"
                                  : rec.current_label === "abnormal"
                                  ? "bg-red-400"
                                  : rec.current_label === "uncertain"
                                  ? "bg-amber-400"
                                  : "bg-slate-600"
                              }`}
                            />
                            <span className="font-semibold text-slate-200 capitalize">
                              {rec.current_label}
                            </span>
                          </div>
                          <span className="text-[10px] text-slate-500 uppercase">
                            {rec.verification_status}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-slate-400 text-[11px]">
                          {new Date(rec.captured_at).toLocaleDateString()}
                        </td>
                        <td className="px-4 py-3 text-right space-x-2">
                          <button
                            onClick={() => handleOpenAnnotate(rec)}
                            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-cyan-400 rounded text-[11px] font-medium transition"
                          >
                            Annotate
                          </button>
                          <button
                            onClick={() => handleOpenRole(rec)}
                            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] font-medium transition"
                          >
                            Role
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 2: Sealed Manifests */}
        {activeTab === "manifests" && (
          <div className="space-y-4">
            <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2 text-xs">
              <h3 className="font-bold text-slate-100 flex items-center gap-2">
                <Lock className="w-4 h-4 text-cyan-400" />
                <span>Dataset Manifest Sealing Policy</span>
              </h3>
              <p className="text-slate-400 leading-relaxed">
                Manifests freeze fixed evaluation or training subsets. Once sealed, assigned recordings cannot be reassigned or modified. The SHA-256 checksum is computed over sorted member hashes, providing immutable provenance verification.
              </p>
            </div>

            <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-900/50">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-800/60 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="px-4 py-3">Manifest Name & ID</th>
                    <th className="px-4 py-3">Role</th>
                    <th className="px-4 py-3">Category</th>
                    <th className="px-4 py-3">Status</th>
                    <th className="px-4 py-3">Recordings</th>
                    <th className="px-4 py-3">SHA-256 Provenance Checksum</th>
                    <th className="px-4 py-3">Sealed Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {manifests.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                        No manifests created yet. Click &quot;Seal Manifest&quot; to establish a frozen dataset split.
                      </td>
                    </tr>
                  ) : (
                    manifests.map((m) => (
                      <tr key={m.manifest_id} className="hover:bg-slate-800/30 transition">
                        <td className="px-4 py-3">
                          <div className="font-medium text-slate-100">{m.name}</div>
                          <div className="text-[10px] font-mono text-cyan-400">{m.manifest_id}</div>
                        </td>
                        <td className="px-4 py-3">
                          <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-slate-800 text-slate-300">
                            {m.dataset_role}
                          </span>
                        </td>
                        <td className="px-4 py-3 uppercase text-[10px] font-semibold text-slate-400">
                          {m.machine_category}
                        </td>
                        <td className="px-4 py-3">
                          {m.is_sealed ? (
                            <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded-full border border-emerald-800/60">
                              <Lock className="w-3 h-3" /> Sealed
                            </span>
                          ) : (
                            <span className="text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded-full text-[11px]">
                              Open
                            </span>
                          )}
                        </td>
                        <td className="px-4 py-3 font-semibold text-slate-200">
                          {m.recording_count} clips
                        </td>
                        <td className="px-4 py-3 font-mono text-[11px] text-cyan-300" title={m.sha256_checksum}>
                          {m.sha256_checksum ? `${m.sha256_checksum.slice(0, 16)}...` : "—"}
                        </td>
                        <td className="px-4 py-3 text-slate-400 text-[11px]">
                          {m.sealed_at ? new Date(m.sealed_at).toLocaleDateString() : "—"}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Tab 3: Quality Gating & Rejection Audit */}
        {activeTab === "audit" && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2">
                <div className="flex items-center gap-2 text-red-400 font-semibold text-sm">
                  <XCircle className="w-4 h-4" />
                  <span>Deduplication Gate</span>
                </div>
                <p className="text-xs text-slate-400">
                  Total Rejected Duplicates: <strong>{stats?.duplicates_detected ?? 0}</strong>. Identical SHA-256 hashes are logged and rejected immediately to protect evaluation validity.
                </p>
              </div>

              <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2">
                <div className="flex items-center gap-2 text-amber-400 font-semibold text-sm">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Audio Quality Gate</span>
                </div>
                <p className="text-xs text-slate-400">
                  Total Audio Format Failures: <strong>{stats?.rejections_count ?? 0}</strong>. Rejects clips with sample rates &lt; 16 kHz, clipping &gt; 0.1%, or silence &lt; -60 dBFS.
                </p>
              </div>

              <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl space-y-2">
                <div className="flex items-center gap-2 text-cyan-400 font-semibold text-sm">
                  <ShieldCheck className="w-4 h-4" />
                  <span>Ground-Truth Verification</span>
                </div>
                <p className="text-xs text-slate-400">
                  Model predictions cannot label data. Labels must be confirmed by authorized technicians using multi-modal evidence.
                </p>
              </div>
            </div>

            <div className="p-4 bg-slate-900/40 border border-slate-800 rounded-xl text-xs text-slate-300 space-y-2">
              <h4 className="font-semibold text-slate-100">Standard Audio Acquisition Boundaries:</h4>
              <ul className="list-disc pl-5 space-y-1 text-slate-400">
                <li>Linear PCM 16-bit or 24-bit WAV format</li>
                <li>16 kHz native sampling rate (or 48 kHz high-precision capture)</li>
                <li>Transducer stand-off distance: 10 cm ± 2 cm from mechanical bearing housing</li>
                <li>Healthy machine calibration enrollment quota: N = 10 confirmed normal recordings</li>
                <li>Defect triangulation: Vibration order analysis, thermal scanning, or teardown inspection required for abnormal ground-truth</li>
              </ul>
            </div>
          </div>
        )}

        {/* MODAL: Import Field Audio */}
        {isImportModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-lg w-full p-6 space-y-5 shadow-2xl">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                  <Upload className="w-5 h-5 text-cyan-400" />
                  <span>Import Field Acoustic Audio</span>
                </h3>
                <button onClick={() => setIsImportModalOpen(false)} className="text-slate-400 hover:text-slate-200">
                  ✕
                </button>
              </div>

              <form onSubmit={handleImportSubmit} className="space-y-4 text-xs">
                <div>
                  <label className="block text-slate-300 font-medium mb-1">Select WAV Recording *</label>
                  <input
                    type="file"
                    accept=".wav,audio/wav"
                    onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                    className="w-full bg-slate-800 border border-slate-700 text-slate-300 rounded p-2 focus:border-cyan-500"
                    required
                  />
                  <span className="text-[10px] text-slate-500">Must be Linear PCM WAV, ≥ 16,000 Hz, duration ≥ 1.0s.</span>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-300 font-medium mb-1">Category</label>
                    <select
                      value={formCategory}
                      onChange={(e) => setFormCategory(e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                    >
                      <option value="fan">Fan</option>
                      <option value="pump">Pump</option>
                      <option value="valve">Valve</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-slate-300 font-medium mb-1">Machine Instance ID *</label>
                    <input
                      type="text"
                      placeholder="e.g. fan_unit_04a"
                      value={formMachineId}
                      onChange={(e) => setFormMachineId(e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                      required
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-300 font-medium mb-1">Sensor / Transducer ID</label>
                    <input
                      type="text"
                      value={formSensorId}
                      onChange={(e) => setFormSensorId(e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                    />
                  </div>

                  <div>
                    <label className="block text-slate-300 font-medium mb-1">Mounting Position</label>
                    <input
                      type="text"
                      value={formPosition}
                      onChange={(e) => setFormPosition(e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-slate-300 font-medium mb-1">Operating Load</label>
                    <input
                      type="text"
                      value={formLoad}
                      onChange={(e) => setFormLoad(e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                    />
                  </div>

                  <div>
                    <label className="block text-slate-300 font-medium mb-1">Dataset Role</label>
                    <select
                      value={formRole}
                      onChange={(e) => setFormRole(e.target.value)}
                      className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                    >
                      <option value="unassigned">Unassigned (Default)</option>
                      <option value="training">Training</option>
                      <option value="development">Development</option>
                      <option value="calibration_enrollment">Calibration Enrollment</option>
                      <option value="sealed_evaluation">Sealed Evaluation</option>
                    </select>
                  </div>
                </div>

                <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
                  <button
                    type="button"
                    onClick={() => setIsImportModalOpen(false)}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-xs font-semibold transition"
                  >
                    Verify & Ingest
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* MODAL: Annotate Recording */}
        {isAnnotateModalOpen && activeRecording && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                  <FileCheck className="w-5 h-5 text-cyan-400" />
                  <span>Ground-Truth Annotation</span>
                </h3>
                <button onClick={() => setIsAnnotateModalOpen(false)} className="text-slate-400 hover:text-slate-200">
                  ✕
                </button>
              </div>

              <div className="p-3 bg-slate-800/40 rounded-lg text-xs space-y-1">
                <div className="font-mono text-cyan-400">{activeRecording.recording_id}</div>
                <div className="text-slate-300">{activeRecording.machine_category.toUpperCase()} | {activeRecording.machine_id}</div>
                <div className="text-slate-500 text-[11px]">Current Label: <strong className="text-slate-200 uppercase">{activeRecording.current_label}</strong></div>
              </div>

              {/* Revision History */}
              {activeAnnotations.length > 0 && (
                <div className="space-y-1.5 max-h-36 overflow-y-auto p-2 bg-slate-950/60 rounded border border-slate-800 text-[11px]">
                  <div className="font-semibold text-slate-400 flex items-center gap-1">
                    <History className="w-3 h-3" /> Audit Revision History:
                  </div>
                  {activeAnnotations.map((a) => (
                    <div key={a.annotation_id} className="p-1.5 bg-slate-900 rounded border border-slate-800 space-y-0.5">
                      <div className="flex justify-between text-slate-300">
                        <span>Rev #{a.revision_number}: <strong className="text-cyan-400 uppercase">{a.label}</strong></span>
                        <span className="text-slate-500">{new Date(a.created_at).toLocaleTimeString()}</span>
                      </div>
                      <div className="text-slate-400 text-[10px]">Evidence: {a.evidence_source} by {a.annotator_id}</div>
                      {a.reason && <div className="text-slate-500 italic text-[10px]">&quot;{a.reason}&quot;</div>}
                    </div>
                  ))}
                </div>
              )}

              <form onSubmit={handleSubmitAnnotation} className="space-y-3 text-xs">
                <div>
                  <label className="block text-slate-300 font-medium mb-1">Confirmed Ground-Truth Diagnosis *</label>
                  <select
                    value={annotLabel}
                    onChange={(e) => setAnnotLabel(e.target.value as any)}
                    className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                  >
                    <option value="normal">Normal (Confirmed healthy operation)</option>
                    <option value="abnormal">Abnormal (Defect or mechanical flaw verified)</option>
                    <option value="uncertain">Uncertain (Inconclusive inspection)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-slate-300 font-medium mb-1">Approved Multi-Modal Evidence Source *</label>
                  <select
                    value={annotEvidence}
                    onChange={(e) => setAnnotEvidence(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                  >
                    <option value="VISUAL_INSPECTION">Visual Stethoscope / Borescope Inspection</option>
                    <option value="VIBRATION_ORDER_ANALYSIS">ISO 10816 Vibration Order Analysis</option>
                    <option value="TEARDOWN_INSPECTION">Teardown / Bearing Physical Inspection</option>
                    <option value="THERMAL_SCAN">Infrared Thermography Scan</option>
                    <option value="OIL_ANALYSIS">Lubricant Ferrography / Oil Analysis</option>
                    <option value="OPERATIONAL_LOG">Supervisory Maintenance Log</option>
                    <option value="OTHER_GROUND_TRUTH">Other Independent Ground-Truth</option>
                  </select>
                </div>

                <div>
                  <label className="block text-slate-300 font-medium mb-1">Reason for Diagnosis / Revision *</label>
                  <input
                    type="text"
                    placeholder="e.g. Borescope confirmed outer race bearing spalling"
                    value={annotReason}
                    onChange={(e) => setAnnotReason(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                    required
                  />
                </div>

                <div>
                  <label className="block text-slate-300 font-medium mb-1">Detailed Technical Inspection Notes</label>
                  <textarea
                    rows={2}
                    placeholder="Optional technician observations..."
                    value={annotNotes}
                    onChange={(e) => setAnnotNotes(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                  />
                </div>

                <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
                  <button
                    type="button"
                    onClick={() => setIsAnnotateModalOpen(false)}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-xs font-semibold transition"
                  >
                    Commit Revision
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* MODAL: Role Assignment */}
        {isRoleModalOpen && activeRecording && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                  <Layers className="w-5 h-5 text-cyan-400" />
                  <span>Update Dataset Separation Role</span>
                </h3>
                <button onClick={() => setIsRoleModalOpen(false)} className="text-slate-400 hover:text-slate-200">
                  ✕
                </button>
              </div>

              <div className="p-3 bg-slate-800/40 rounded-lg text-xs space-y-1">
                <div className="font-mono text-cyan-400">{activeRecording.recording_id}</div>
                <div className="text-slate-400">Current Role: <strong className="text-slate-200">{activeRecording.dataset_role}</strong></div>
              </div>

              <form onSubmit={handleSubmitRole} className="space-y-4 text-xs">
                <div>
                  <label className="block text-slate-300 font-medium mb-1">New Dataset Role</label>
                  <select
                    value={selectedRole}
                    onChange={(e) => setSelectedRole(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                  >
                    <option value="unassigned">Unassigned</option>
                    <option value="training">Training (Normal baseline only)</option>
                    <option value="development">Development (Parameter tuning)</option>
                    <option value="calibration_enrollment">Calibration Enrollment (Strictly Normal)</option>
                    <option value="sealed_evaluation">Sealed Evaluation (Independent benchmark)</option>
                  </select>
                </div>

                <div className="p-3 bg-amber-950/30 border border-amber-800/40 rounded text-[11px] text-amber-200/90">
                  <strong>Guardrails:</strong> Abnormal recordings cannot be assigned to Calibration Enrollment. Recordings in sealed manifests cannot be modified.
                </div>

                <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
                  <button
                    type="button"
                    onClick={() => setIsRoleModalOpen(false)}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-xs font-semibold transition"
                  >
                    Update Role
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {/* MODAL: Create & Seal Manifest */}
        {isManifestModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
            <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                  <Lock className="w-5 h-5 text-indigo-400" />
                  <span>Create & Seal Manifest</span>
                </h3>
                <button onClick={() => setIsManifestModalOpen(false)} className="text-slate-400 hover:text-slate-200">
                  ✕
                </button>
              </div>

              <form onSubmit={handleCreateManifest} className="space-y-4 text-xs">
                <div>
                  <label className="block text-slate-300 font-medium mb-1">Manifest Name</label>
                  <input
                    type="text"
                    placeholder="e.g. Pilot Stage 1 Fan Evaluation Split"
                    value={manifestName}
                    onChange={(e) => setManifestName(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                  />
                </div>

                <div>
                  <label className="block text-slate-300 font-medium mb-1">Target Category</label>
                  <select
                    value={manifestCategory}
                    onChange={(e) => setManifestCategory(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                  >
                    <option value="fan">Fan</option>
                    <option value="pump">Pump</option>
                    <option value="valve">Valve</option>
                  </select>
                </div>

                <div>
                  <label className="block text-slate-300 font-medium mb-1">Dataset Role</label>
                  <select
                    value={manifestRole}
                    onChange={(e) => setManifestRole(e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 text-slate-200 rounded p-2"
                  >
                    <option value="sealed_evaluation">Sealed Evaluation</option>
                    <option value="training">Training</option>
                    <option value="development">Development</option>
                  </select>
                </div>

                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="sealNow"
                    checked={sealNow}
                    onChange={(e) => setSealNow(e.target.checked)}
                    className="rounded bg-slate-800 border-slate-700 text-indigo-500 focus:ring-0"
                  />
                  <label htmlFor="sealNow" className="text-slate-300">
                    Seal and freeze manifest immediately (generates cryptographic SHA-256)
                  </label>
                </div>

                <div className="flex justify-end gap-2 pt-3 border-t border-slate-800">
                  <button
                    type="button"
                    onClick={() => setIsManifestModalOpen(false)}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs transition"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-semibold transition"
                  >
                    Generate Manifest
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </ProtectedRoute>
  );
}
