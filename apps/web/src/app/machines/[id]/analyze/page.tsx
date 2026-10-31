"use client";

import React, { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { apiClient, MachineProfile, JobRecord } from "@/lib/api-client";
import { AudioPlayer } from "@/components/audio/AudioPlayer";
import {
  Activity,
  ArrowLeft,
  Upload,
  FileAudio,
  Loader2,
  AlertCircle,
  Play,
  RotateCw,
  Cpu,
  Layers,
  CheckCircle2,
} from "lucide-react";

export default function MachineAnalyzePage() {
  const params = useParams();
  const machineId = params.id as string;
  const router = useRouter();

  const [machine, setMachine] = useState<MachineProfile | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [modelPolicy, setModelPolicy] = useState<string>("stetho_msae");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeJob, setActiveJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingMachine, setIsLoadingMachine] = useState(true);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const pollingIntervalRef = useRef<any>(null);

  useEffect(() => {
    async function loadMachine() {
      setIsLoadingMachine(true);
      try {
        const data = await apiClient.getMachine(machineId);
        setMachine(data);
        setModelPolicy(data.selected_model_policy || "stetho_msae");
      } catch (err: any) {
        setError(err.message || "Failed to load machine profile.");
      } finally {
        setIsLoadingMachine(false);
      }
    }
    loadMachine();
  }, [machineId]);

  // Clean up polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (!file.name.toLowerCase().endsWith(".wav")) {
        setError("Invalid file format. Only WAV audio recordings are supported.");
        return;
      }
      if (file.size > 15 * 1024 * 1024) {
        setError("File size exceeds 15 MB limit.");
        return;
      }
      setSelectedFile(file);
    }
  };

  const pollJobStatus = (jobId: string) => {
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const job = await apiClient.getAnalysisJob(jobId);
        setActiveJob(job);

        if (job.status === "completed" && job.analysis_id) {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
          }
          // Redirect to analysis results
          router.push(`/analyses/${job.analysis_id}`);
        } else if (job.status === "failed") {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
          }
          setIsSubmitting(false);
          setError(job.error_message || "Acoustic analysis failed during background inference.");
        }
      } catch (err: any) {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
        }
        setIsSubmitting(false);
        setError("Unable to poll background analysis job status.");
      }
    }, 1200);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!selectedFile) {
      setError("Please select an audio recording for analysis.");
      return;
    }

    setIsSubmitting(true);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      if (modelPolicy) {
        formData.append("model_policy", modelPolicy);
      }

      // Submit asynchronous analysis job (HTTP 202 Accepted)
      const job = await apiClient.submitAnalysisJob(machineId, formData);
      setActiveJob(job);

      // Begin polling job progress
      pollJobStatus(job.job_id);
    } catch (err: any) {
      setIsSubmitting(false);
      setError(err.message || "Failed to submit acoustic analysis.");
    }
  };

  if (isLoadingMachine) {
    return (
      <ProtectedRoute>
        <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
          <Loader2 className="w-8 h-8 text-brand-600 animate-spin" />
          <p className="text-sm text-slate-500">Loading machine profile...</p>
        </div>
      </ProtectedRoute>
    );
  }

  return (
    <ProtectedRoute>
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Navigation Header */}
        <div className="flex items-center gap-3">
          <Link
            href={`/machines/${machineId}`}
            className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-semibold">
                {machine?.machine_id}
              </span>
              <span className="text-xs text-slate-500 font-medium">{machine?.name}</span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-0.5">
              Machine Acoustic Inspection
            </h1>
          </div>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold">Analysis Warning</span>
              <p className="text-xs text-rose-700 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {isSubmitting && activeJob ? (
          /* Asynchronous Job Execution Progress View */
          <div className="bg-white rounded-2xl border border-indigo-200 shadow-md p-8 text-center space-y-6">
            <div className="w-16 h-16 bg-indigo-50 text-indigo-600 rounded-2xl flex items-center justify-center mx-auto">
              <Loader2 className="w-8 h-8 animate-spin" />
            </div>

            <div className="space-y-2">
              <h2 className="text-xl font-bold text-slate-900">
                Executing Acoustic Inference Pipeline
              </h2>
              <p className="text-xs text-slate-500 max-w-md mx-auto">
                Running Audio Quality Gate, computing 64-band log-mel spectrograms, and executing {modelPolicy} autoencoder reconstruction on CPU.
              </p>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2 max-w-md mx-auto">
              <div className="flex justify-between text-xs font-mono text-slate-500">
                <span className="capitalize">Status: {activeJob.status}</span>
                <span>{activeJob.progress_pct}%</span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-indigo-600 h-2.5 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${Math.max(10, activeJob.progress_pct)}%` }}
                />
              </div>
            </div>

            <div className="text-[11px] font-mono text-slate-400">
              Job ID: {activeJob.job_id}
            </div>
          </div>
        ) : (
          /* Analysis Submission Form */
          <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-6">
            <div className="space-y-4">
              <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 border-b border-slate-100 pb-2">
                <Layers className="w-4 h-4 text-brand-600" />
                <span>1. Model Architecture Policy</span>
              </h2>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                  Inference Architecture
                </label>
                <select
                  value={modelPolicy}
                  onChange={(e) => setModelPolicy(e.target.value)}
                  className="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition bg-white"
                >
                  <option value="stetho_msae">
                    Stetho-MSAE (Custom Depthwise-Separable Edge Champion — 560k params)
                  </option>
                  <option value="dense_autoencoder">
                    Dense Autoencoder (Baseline Architecture — 4.34M params)
                  </option>
                </select>
                <p className="text-[11px] text-slate-400 mt-1">
                  Stetho-MSAE applies multi-scale convolutional feature kernels optimized for subtle periodic anomaly detection.
                </p>
              </div>
            </div>

            <div className="space-y-4 pt-2">
              <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 border-b border-slate-100 pb-2">
                <Upload className="w-4 h-4 text-brand-600" />
                <span>2. Select Operating Audio Recording</span>
              </h2>

              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-200 hover:border-brand-500 rounded-2xl p-6 text-center cursor-pointer transition bg-slate-50/50 hover:bg-brand-50/30 space-y-3"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".wav,audio/wav"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="w-10 h-10 bg-indigo-50 text-indigo-600 rounded-xl flex items-center justify-center mx-auto">
                  <FileAudio className="w-5 h-5" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-800">
                    {selectedFile ? selectedFile.name : "Click to select or drag and drop WAV inspection recording"}
                  </p>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    {selectedFile
                      ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Audio ready for inference`
                      : "Standard PCM WAV • 1.0s to 30.0s duration • Max 15 MB"}
                  </p>
                </div>
              </div>

              {selectedFile && (
                <div className="space-y-2">
                  <span className="text-xs font-semibold uppercase text-slate-700 tracking-wider">
                    Waveform &amp; Signal Preview
                  </span>
                  <AudioPlayer
                    fileOrBlob={selectedFile}
                    title={`Inspection Audio: ${selectedFile.name}`}
                    showSpectrogram={true}
                  />
                </div>
              )}
            </div>

            <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-100">
              <Link
                href={`/machines/${machineId}`}
                className="px-4 py-2.5 text-xs font-semibold text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition"
              >
                Cancel
              </Link>

              <button
                type="submit"
                disabled={isSubmitting || !selectedFile}
                className="px-6 py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white text-xs font-semibold rounded-xl shadow-sm transition flex items-center gap-2"
              >
                <Activity className="w-4 h-4" />
                <span>Submit for Acoustic Analysis</span>
              </button>
            </div>
          </form>
        )}
      </div>
    </ProtectedRoute>
  );
}
