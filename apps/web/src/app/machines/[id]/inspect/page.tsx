"use client";

import React, { useEffect, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import { useUI } from "@/context/UIContext";
import { LiveMicRecorder } from "@/components/audio/LiveMicRecorder";
import { AudioPlayer } from "@/components/audio/AudioPlayer";
import { MaintenanceAdviceCard } from "@/components/machine/MaintenanceAdviceCard";
import { QrCode, ShieldCheck, AlertCircle, ArrowLeft, Play, RefreshCw, CheckCircle2 } from "lucide-react";

export default function MobileInspectionPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const router = useRouter();
  const machineId = params.id as string;
  const token = searchParams.get("token");

  const { t, isOnline } = useUI();
  const [machineData, setMachineData] = useState<any | null>(null);
  const [tokenValid, setTokenValid] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const [inspecting, setInspecting] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function verifyToken() {
      if (!token) {
        setTokenValid(false);
        setLoading(false);
        return;
      }
      try {
        setLoading(true);
        const res = await fetch(`/api/backend/machines/inspect-by-token?token=${encodeURIComponent(token)}`);
        if (res.ok) {
          const data = await res.json();
          setMachineData(data.machine);
          setTokenValid(true);
        } else {
          setTokenValid(false);
        }
      } catch (e) {
        setTokenValid(false);
      } finally {
        setLoading(false);
      }
    }
    verifyToken();
  }, [token]);

  const handleAudioCaptured = (file: File, url: string) => {
    setAudioFile(file);
    setAudioUrl(url);
    setAnalysisResult(null);
    setError(null);
  };

  const handleRunInspection = async () => {
    if (!audioFile) return;
    setInspecting(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", audioFile);
      formData.append("model_policy", "stetho_msae");

      const res = await fetch(`/api/backend/machines/${machineId}/analyze`, {
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
      setError(err.message || "Failed to execute field inspection.");
    } finally {
      setInspecting(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-[50vh] flex flex-col items-center justify-center space-y-3">
        <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
        <span className="text-xs text-slate-400 font-mono">Verifying QR Asset Token...</span>
      </div>
    );
  }

  if (tokenValid === false) {
    return (
      <div className="max-w-md mx-auto my-12 bg-slate-900 border border-red-800/80 rounded-2xl p-6 text-center space-y-4">
        <div className="w-12 h-12 bg-red-500/10 text-red-400 rounded-full flex items-center justify-center mx-auto">
          <AlertCircle className="w-6 h-6" />
        </div>
        <h2 className="text-base font-bold text-slate-100">Invalid or Expired QR Token</h2>
        <p className="text-xs text-slate-400 leading-relaxed">
          The scanned QR identity credential has expired or was revoked by an administrator. Please scan the current machine tag.
        </p>
        <button
          onClick={() => router.push("/machines")}
          className="px-4 py-2 bg-slate-800 text-slate-300 rounded-lg text-xs font-medium"
        >
          Return to Fleet
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Top Asset Verification Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-cyan-500/10 text-cyan-400 rounded-lg">
              <QrCode className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-400 font-bold flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Verified Asset Identity
              </span>
              <h1 className="text-lg font-bold text-slate-100">{machineData?.name || machineId}</h1>
            </div>
          </div>
          <span className="px-2.5 py-1 bg-slate-950 border border-slate-800 text-slate-400 text-xs font-mono rounded-lg">
            {machineId}
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-3 text-xs font-mono">
          <div>
            <span className="text-slate-500 block text-[10px] uppercase">Category</span>
            <span className="text-slate-300 capitalize">{machineData?.category || "Industrial Fan"}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px] uppercase">Calibration</span>
            <span className="text-cyan-400 font-semibold uppercase">{machineData?.calibration_status}</span>
          </div>
        </div>
      </div>

      {/* Guided Acoustic Recorder */}
      <LiveMicRecorder
        machineId={machineId}
        machineName={machineData?.name}
        onRecordingComplete={handleAudioCaptured}
        targetDurationSec={5}
      />

      {/* Visualizer & Inspect Button */}
      {audioUrl && (
        <div className="space-y-4">
          <AudioPlayer audioUrl={audioUrl} title="Field Acoustic Recording" />

          <button
            onClick={handleRunInspection}
            disabled={inspecting}
            className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-semibold text-sm rounded-xl transition flex items-center justify-center gap-2 shadow-lg shadow-cyan-900/30"
          >
            {inspecting ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Processing Model Inference...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Submit Field Acoustic Inspection</span>
              </>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-950/40 border border-red-800 text-red-300 text-xs rounded-lg">
          {error}
        </div>
      )}

      {/* Real Inspection Result */}
      {analysisResult && (
        <div className="space-y-6 pt-2">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 text-center">
            <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1">
              Field Inspection Outcome
            </span>
            <span
              className={`inline-block px-4 py-1.5 rounded-full text-sm font-mono font-bold border my-2 ${
                analysisResult.decision === "UNUSUAL"
                  ? "bg-red-500/20 text-red-300 border-red-500/40"
                  : analysisResult.decision === "INCONCLUSIVE"
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                  : "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
              }`}
            >
              {analysisResult.decision}
            </span>
            <div className="flex justify-center gap-6 text-xs font-mono text-slate-400 mt-2">
              <span>Score: <strong className="text-cyan-300">{analysisResult.anomaly_score?.toFixed(5)}</strong></span>
              <span>Threshold: <strong className="text-slate-300">{analysisResult.operating_threshold?.toFixed(5)}</strong></span>
              <span>Latency: <strong className="text-emerald-400">{analysisResult.processing_duration_ms?.toFixed(1)} ms</strong></span>
            </div>
          </div>

          <MaintenanceAdviceCard analysisId={analysisResult.analysis_id} />
        </div>
      )}
    </div>
  );
}
