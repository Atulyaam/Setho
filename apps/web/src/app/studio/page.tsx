"use client";

import React, { useEffect, useState } from "react";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { useUI } from "@/context/UIContext";
import { LiveMicRecorder } from "@/components/audio/LiveMicRecorder";
import { AudioPlayer } from "@/components/audio/AudioPlayer";
import { AcousticFingerprintCard } from "@/components/machine/AcousticFingerprintCard";
import { MaintenanceAdviceCard } from "@/components/machine/MaintenanceAdviceCard";
import { Mic, UploadCloud, Play, AlertTriangle, CheckCircle, RefreshCw } from "lucide-react";

interface MachineOption {
  machine_id: string;
  name: string;
  category: string;
  calibration_status: string;
}

export default function AcousticStudioPage() {
  const { t } = useUI();
  const [machines, setMachines] = useState<MachineOption[]>([]);
  const [selectedMachineId, setSelectedMachineId] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"mic" | "upload">("mic");

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadMachines() {
      try {
        const res = await fetch("/api/backend/machines");
        if (res.ok) {
          const data = await res.json();
          const list = data.machines || [];
          setMachines(list);
          if (list.length > 0) {
            setSelectedMachineId(list[0].machine_id);
          }
        }
      } catch (e) {
        console.error(e);
      }
    }
    loadMachines();
  }, []);

  const handleAudioAcquired = (file: File, url: string) => {
    setAudioFile(file);
    setAudioUrl(url);
    setAnalysisResult(null);
    setError(null);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      handleAudioAcquired(file, url);
    }
  };

  const executeAnalysis = async () => {
    if (!selectedMachineId || !audioFile) return;

    setAnalyzing(true);
    setError(null);
    setAnalysisResult(null);

    try {
      const formData = new FormData();
      formData.append("file", audioFile);
      formData.append("model_policy", "stetho_msae");

      const res = await fetch(`/api/backend/machines/${selectedMachineId}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson?.error?.message || "Analysis failed.");
      }

      const result = await res.json();
      setAnalysisResult(result);
    } catch (err: any) {
      setError(err.message || "Failed to execute acoustic analysis.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <ProtectedRoute>
      <div className="space-y-8">
        {/* Header */}
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
              <Mic className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-100">
                {t("acousticStudio")}
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                {t("acousticStudioSubtitle")}
              </p>
            </div>
          </div>
        </div>

        {/* Machine Selector Bar */}
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <span className="text-xs font-mono uppercase text-slate-400 whitespace-nowrap">
              Target Machine:
            </span>
            <select
              value={selectedMachineId}
              onChange={(e) => setSelectedMachineId(e.target.value)}
              className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500 w-full sm:w-72"
            >
              {machines.map((m) => (
                <option key={m.machine_id} value={m.machine_id}>
                  {m.name} ({m.machine_id}) — {m.calibration_status}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setActiveTab("mic")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition ${
                activeTab === "mic"
                  ? "bg-cyan-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              <Mic className="w-3.5 h-3.5" />
              <span>Live Mic</span>
            </button>
            <button
              onClick={() => setActiveTab("upload")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-1.5 transition ${
                activeTab === "upload"
                  ? "bg-cyan-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              <UploadCloud className="w-3.5 h-3.5" />
              <span>Upload File</span>
            </button>
          </div>
        </div>

        {/* Audio Capture Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            {activeTab === "mic" ? (
              <LiveMicRecorder
                machineId={selectedMachineId}
                onRecordingComplete={handleAudioAcquired}
                targetDurationSec={5}
              />
            ) : (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-8 text-center">
                <input
                  type="file"
                  id="audio-upload-input"
                  accept=".wav,audio/wav"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <label
                  htmlFor="audio-upload-input"
                  className="cursor-pointer flex flex-col items-center justify-center gap-3"
                >
                  <div className="p-4 bg-slate-950 border border-slate-800 rounded-full text-cyan-400 hover:scale-105 transition">
                    <UploadCloud className="w-8 h-8" />
                  </div>
                  <span className="text-sm font-semibold text-slate-200">
                    Click to select PCM 16-bit WAV file
                  </span>
                  <span className="text-xs text-slate-500">
                    Maximum 50MB • Recommended 5-10s duration
                  </span>
                </label>
              </div>
            )}

            {/* Run Analysis Action */}
            {audioFile && (
              <div className="mt-4 p-4 bg-slate-900 border border-slate-800 rounded-xl flex items-center justify-between">
                <div>
                  <span className="text-xs font-mono text-cyan-300 block">{audioFile.name}</span>
                  <span className="text-[11px] text-slate-500">
                    {(audioFile.size / 1024).toFixed(1)} KB • Ready for inspection
                  </span>
                </div>
                <button
                  onClick={executeAnalysis}
                  disabled={analyzing}
                  className="px-5 py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center gap-2 transition shadow-lg shadow-cyan-900/30"
                >
                  {analyzing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Analyze Recording</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {error && (
              <div className="mt-4 p-3 bg-red-950/40 border border-red-800 text-red-300 text-xs rounded-lg">
                {error}
              </div>
            )}
          </div>

          {/* Reference Fingerprint Card */}
          <div>
            {selectedMachineId ? (
              <AcousticFingerprintCard machineId={selectedMachineId} />
            ) : (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-500 text-xs text-center">
                Select a machine to view reference acoustic fingerprint.
              </div>
            )}
          </div>
        </div>

        {/* Audio Visualizers */}
        {audioUrl && (
          <div className="space-y-4">
            <h3 className="text-base font-semibold text-slate-200 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              Acoustic Telemetry & Playback
            </h3>
            <AudioPlayer audioUrl={audioUrl} title={audioFile?.name || "Studio Recording"} showSpectrogram={true} />
          </div>
        )}

        {/* Real Analysis Results */}
        {analysisResult && (
          <div className="space-y-6 pt-4 border-t border-slate-800">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
                <span className="text-xs font-mono uppercase text-slate-400">Analysis Result</span>
                <span
                  className={`px-3 py-1 rounded-full text-xs font-mono font-bold border ${
                    analysisResult.decision === "UNUSUAL"
                      ? "bg-red-500/20 text-red-300 border-red-500/40"
                      : analysisResult.decision === "INCONCLUSIVE"
                      ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                      : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                  }`}
                >
                  {analysisResult.decision}
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center font-mono text-xs">
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-slate-500 block text-[10px] uppercase">Anomaly Score</span>
                  <span className="text-sm font-semibold text-cyan-300">
                    {analysisResult.anomaly_score?.toFixed(5)}
                  </span>
                </div>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-slate-500 block text-[10px] uppercase">Threshold (θ)</span>
                  <span className="text-sm font-semibold text-slate-300">
                    {analysisResult.operating_threshold?.toFixed(5)}
                  </span>
                </div>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-slate-500 block text-[10px] uppercase">Latency</span>
                  <span className="text-sm font-semibold text-emerald-400">
                    {analysisResult.processing_duration_ms?.toFixed(1)} ms
                  </span>
                </div>
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-slate-500 block text-[10px] uppercase">Quality</span>
                  <span className="text-sm font-semibold text-cyan-400">
                    {analysisResult.quality_status}
                  </span>
                </div>
              </div>
            </div>

            {/* Evidence-Based Maintenance Guidance */}
            <MaintenanceAdviceCard analysisId={analysisResult.analysis_id} />
          </div>
        )}
      </div>
    </ProtectedRoute>
  );
}
