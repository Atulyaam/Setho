"use client";

import React, { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { apiClient, MachineProfile, EnrollmentRecord } from "@/lib/api-client";
import { AudioPlayer } from "@/components/audio/AudioPlayer";
import {
  Mic,
  ArrowLeft,
  Upload,
  CheckCircle2,
  AlertTriangle,
  FileAudio,
  ShieldCheck,
  Loader2,
  ArrowRight,
  Info,
} from "lucide-react";

export default function MachineEnrollmentPage() {
  const params = useParams();
  const machineId = params.id as string;
  const router = useRouter();

  const [machine, setMachine] = useState<MachineProfile | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [confirmedNormal, setConfirmedNormal] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [enrollmentResult, setEnrollmentResult] = useState<EnrollmentRecord | null>(null);
  const [isLoadingMachine, setIsLoadingMachine] = useState(true);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    async function loadMachine() {
      setIsLoadingMachine(true);
      try {
        const data = await apiClient.getMachine(machineId);
        setMachine(data);
      } catch (err: any) {
        setError(err.message || "Failed to load machine profile.");
      } finally {
        setIsLoadingMachine(false);
      }
    }
    loadMachine();
  }, [machineId]);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!selectedFile) {
      setError("Please select a WAV recording to enroll.");
      return;
    }

    if (!confirmedNormal) {
      setError("You must explicitly verify that this recording represents normal machine operation.");
      return;
    }

    setIsSubmitting(true);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("confirmed_normal", "true");

      const result = await apiClient.enrollAudio(machineId, formData);
      setEnrollmentResult(result);
    } catch (err: any) {
      setError(err.message || "Enrollment failed. Audio may have violated quality gate requirements.");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isLoadingMachine) {
    return (
      <ProtectedRoute>
        <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
          <Loader2 className="w-8 h-8 text-brand-600 animate-spin" />
          <p className="text-sm text-slate-500">Retrieving machine profile...</p>
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
              Enroll Confirmed Normal Audio
            </h1>
          </div>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-rose-500 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold">Enrollment Notice</span>
              <p className="text-xs text-rose-700 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {enrollmentResult ? (
          /* Enrollment Success & Recalibration Outcome Card */
          <div className="bg-white rounded-2xl border border-emerald-200 shadow-md p-6 sm:p-8 space-y-6">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-emerald-50 text-emerald-600 rounded-2xl">
                <CheckCircle2 className="w-8 h-8" />
              </div>
              <div className="space-y-1">
                <h2 className="text-xl font-bold text-slate-900">
                  Normal Audio Enrolled &amp; Calibrated
                </h2>
                <p className="text-xs text-slate-600 leading-relaxed">
                  The audio sample was validated by the Audio Quality Gate, ingested, and scored against the machine&apos;s baseline model. Statistical calibration thresholds have been updated dynamically.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-50 p-4 rounded-xl border border-slate-100 text-xs">
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Enrolled Count</span>
                <span className="text-lg font-bold text-slate-900 font-mono">
                  {enrollmentResult.enrolled_count} recordings
                </span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Calibration State</span>
                <span className="font-bold capitalize text-slate-800 text-sm block mt-0.5">
                  {enrollmentResult.enrollment_status}
                </span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Threshold (θ_upper)</span>
                <span className="text-base font-bold text-amber-700 font-mono">
                  {enrollmentResult.theta_upper !== undefined
                    ? enrollmentResult.theta_upper.toFixed(4)
                    : "—"}
                </span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Cal Version</span>
                <span className="font-mono text-brand-700 font-semibold text-sm">
                  {enrollmentResult.calibration_version || "1.N.0"}
                </span>
              </div>
            </div>

            {enrollmentResult.notice && (
              <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start gap-2.5">
                <Info className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                <p className="leading-relaxed">{enrollmentResult.notice}</p>
              </div>
            )}

            <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-100">
              <button
                onClick={() => {
                  setEnrollmentResult(null);
                  setSelectedFile(null);
                  setConfirmedNormal(false);
                }}
                className="px-4 py-2 text-xs font-semibold text-slate-700 hover:text-slate-900 hover:bg-slate-100 rounded-xl transition"
              >
                Enroll Another Recording
              </button>
              <Link
                href={`/machines/${machineId}`}
                className="px-5 py-2 bg-brand-600 hover:bg-brand-700 text-white text-xs font-semibold rounded-xl shadow-sm transition"
              >
                View Machine Dashboard
              </Link>
            </div>
          </div>
        ) : (
          /* Enrollment Submission Form */
          <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-6">
            <div className="space-y-2">
              <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                <Upload className="w-4 h-4 text-brand-600" />
                <span>1. Select Baseline WAV Audio Recording</span>
              </h2>
              <p className="text-xs text-slate-500">
                Upload a 10-second reference audio clip recorded under verified normal machine operation. Max 15 MB.
              </p>
            </div>

            {/* Drag & Drop File Selector */}
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
              <div className="w-10 h-10 bg-brand-50 text-brand-600 rounded-xl flex items-center justify-center mx-auto">
                <FileAudio className="w-5 h-5" />
              </div>
              <div>
                <p className="text-xs font-semibold text-slate-800">
                  {selectedFile ? selectedFile.name : "Click to select or drag and drop WAV recording"}
                </p>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  {selectedFile
                    ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB • Audio ready for inspection`
                    : "Standard PCM WAV • 16,000 Hz or multi-channel reference"}
                </p>
              </div>
            </div>

            {/* Interactive Audio Player & Waveform Visualizer */}
            {selectedFile && (
              <div className="space-y-2">
                <span className="text-xs font-semibold uppercase text-slate-700 tracking-wider">
                  Audio Verification &amp; Visual Preview
                </span>
                <AudioPlayer
                  fileOrBlob={selectedFile}
                  title={`Selected File: ${selectedFile.name}`}
                  showSpectrogram={true}
                />
              </div>
            )}

            {/* Mandatory Normal Confirmation Checkbox */}
            <div className="p-4 rounded-xl bg-amber-50/80 border border-amber-200 space-y-2">
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  required
                  checked={confirmedNormal}
                  onChange={(e) => setConfirmedNormal(e.target.checked)}
                  className="mt-1 w-4 h-4 rounded text-brand-600 focus:ring-brand-500 border-slate-300"
                />
                <div className="text-xs text-amber-950 leading-relaxed">
                  <span className="font-bold">Mandatory Normal Confirmation: </span>
                  I explicitly verify that this recording was captured when this physical machine was functioning under healthy, known-normal operating conditions without acoustic defects or unusual load.
                </div>
              </label>
              <p className="text-[11px] text-amber-800/80 pl-7">
                * Calibration profiles are derived exclusively from confirmed normal audio. Stetho never silently enrolls unconfirmed audio.
              </p>
            </div>

            {/* Submit Button */}
            <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-100">
              <Link
                href={`/machines/${machineId}`}
                className="px-4 py-2.5 text-xs font-semibold text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition"
              >
                Cancel
              </Link>

              <button
                type="submit"
                disabled={isSubmitting || !selectedFile || !confirmedNormal}
                className="px-6 py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white text-xs font-semibold rounded-xl shadow-sm transition flex items-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Processing &amp; Calibrating...</span>
                  </>
                ) : (
                  <>
                    <ShieldCheck className="w-4 h-4" />
                    <span>Submit Normal Enrollment</span>
                  </>
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </ProtectedRoute>
  );
}
