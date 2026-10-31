"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import {
  apiClient,
  MachineProfile,
  EnrollmentRecord,
  AnalysisRecord,
} from "@/lib/api-client";
import {
  Cpu,
  Mic,
  Activity,
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  FileText,
  Clock,
  Layers,
  ShieldCheck,
  Loader2,
  ChevronRight,
} from "lucide-react";

export default function MachineDetailPage() {
  const params = useParams();
  const machineId = params.id as string;
  const router = useRouter();

  const [machine, setMachine] = useState<MachineProfile | null>(null);
  const [enrollments, setEnrollments] = useState<any[]>([]);
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      setIsLoading(true);
      setError(null);
      try {
        const [machineData, enrollmentsData, analysesData] = await Promise.all([
          apiClient.getMachine(machineId),
          apiClient.getEnrollments(machineId),
          apiClient.getMachineAnalyses(machineId),
        ]);
        setMachine(machineData);
        setEnrollments(enrollmentsData.enrollments);
        setAnalyses(analysesData.analyses);
      } catch (err: any) {
        setError(err.message || "Failed to retrieve machine profile.");
      } finally {
        setIsLoading(false);
      }
    }
    loadData();
  }, [machineId]);

  if (isLoading) {
    return (
      <ProtectedRoute>
        <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
          <Loader2 className="w-8 h-8 text-brand-600 animate-spin" />
          <p className="text-sm text-slate-500">Loading machine telemetry...</p>
        </div>
      </ProtectedRoute>
    );
  }

  if (error || !machine) {
    return (
      <ProtectedRoute>
        <div className="max-w-xl mx-auto my-12 bg-white rounded-2xl border border-rose-200 p-8 text-center space-y-4 shadow-sm">
          <div className="w-12 h-12 bg-rose-50 text-rose-600 rounded-2xl flex items-center justify-center mx-auto">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">Machine Inaccessible</h2>
            <p className="text-xs text-rose-600 mt-1">{error || "Machine profile not found."}</p>
          </div>
          <Link
            href="/machines"
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold rounded-xl transition"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Machine Fleet</span>
          </Link>
        </div>
      </ProtectedRoute>
    );
  }

  const profile = machine.calibration_profile;

  return (
    <ProtectedRoute>
      <div className="space-y-8">
        {/* Breadcrumb & Navigation */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Link
              href="/machines"
              className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-semibold">
                  {machine.machine_id}
                </span>
                <span
                  className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full ${
                    machine.calibration_status === "calibrated"
                      ? "bg-emerald-100 text-emerald-800"
                      : machine.calibration_status === "provisional"
                      ? "bg-amber-100 text-amber-800"
                      : "bg-slate-100 text-slate-600"
                  }`}
                >
                  {machine.calibration_status}
                </span>
              </div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-1">
                {machine.name}
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href={`/machines/${machine.machine_id}/enroll`}
              className="px-4 py-2 bg-teal-50 hover:bg-teal-100 text-brand-700 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 border border-teal-200/50"
            >
              <Mic className="w-4 h-4" />
              <span>Enroll Normal Audio</span>
            </Link>

            <Link
              href={`/machines/${machine.machine_id}/analyze`}
              className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-xl text-xs font-semibold shadow-sm transition flex items-center gap-1.5"
            >
              <Activity className="w-4 h-4" />
              <span>Perform Inspection</span>
            </Link>
          </div>
        </div>

        {/* Machine Metadata & Calibration Specs Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Card 1: Machine Specifications */}
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-4">
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider border-b border-slate-100 pb-2 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-brand-600" />
              <span>Machine Specifications</span>
            </h2>

            <div className="space-y-3 text-xs">
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Category</span>
                <span className="font-semibold text-slate-800 capitalize">{machine.category}</span>
              </div>

              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Operating Condition</span>
                <span className="font-medium text-slate-700">
                  {machine.operating_condition || "Standard operational baseline"}
                </span>
              </div>

              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Description</span>
                <p className="text-slate-600 leading-relaxed">
                  {machine.description || "No detailed physical description provided."}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-100 flex justify-between">
                <span className="text-slate-400">Created:</span>
                <span className="font-mono text-slate-600">
                  {new Date(machine.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>

          {/* Card 2: Statistical Calibration Profile */}
          <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-brand-600" />
                <span>Statistical Calibration Profile (Version {machine.calibration_version})</span>
              </h2>
              <span className="text-[11px] font-mono text-slate-500">
                Model: {machine.selected_model_policy}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-100">
                <span className="text-[10px] uppercase tracking-wider text-slate-400 block">
                  Enrolled Samples
                </span>
                <span className="text-xl font-bold text-slate-900 font-mono">
                  {machine.enrolled_recording_count}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  {machine.enrolled_recording_count >= 30 ? "Fully Calibrated" : "Need 30 for full"}
                </span>
              </div>

              <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-100">
                <span className="text-[10px] uppercase tracking-wider text-slate-400 block">
                  Lower Bound (θ_lower)
                </span>
                <span className="text-xl font-bold text-emerald-700 font-mono">
                  {profile?.theta_lower !== undefined ? profile.theta_lower.toFixed(4) : "0.0250"}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">90th Percentile</span>
              </div>

              <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-100">
                <span className="text-[10px] uppercase tracking-wider text-slate-400 block">
                  Upper Bound (θ_upper)
                </span>
                <span className="text-xl font-bold text-amber-700 font-mono">
                  {profile?.theta_upper !== undefined ? profile.theta_upper.toFixed(4) : "0.0450"}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">98th Percentile</span>
              </div>

              <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-100">
                <span className="text-[10px] uppercase tracking-wider text-slate-400 block">
                  Operating Status
                </span>
                <span className="text-sm font-bold capitalize text-slate-800 block mt-1">
                  {machine.calibration_status}
                </span>
                <span className="text-[10px] text-slate-500 block">3-zone threshold active</span>
              </div>
            </div>

            {machine.enrolled_recording_count < 30 && (
              <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200/80 text-amber-900 text-xs flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                <div className="leading-relaxed">
                  <span className="font-semibold">Provisional Calibration Notice: </span>
                  This machine has enrolled {machine.enrolled_recording_count} confirmed normal recordings (fewer than the required 30 reference samples). Operating thresholds carry statistical variance and all inspection decisions carry a provisional notice until 30 recordings are registered.
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Enrollment History Table */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Mic className="w-4 h-4 text-brand-600" />
              <span>Confirmed Normal Audio Enrollments ({enrollments.length})</span>
            </h2>

            <Link
              href={`/machines/${machine.machine_id}/enroll`}
              className="text-xs font-semibold text-brand-600 hover:text-brand-700 flex items-center gap-1"
            >
              <span>+ Add Enrollment</span>
            </Link>
          </div>

          {enrollments.length === 0 ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-8 text-center text-xs text-slate-500">
              No normal reference recordings enrolled yet. Upload a confirmed normal audio sample to initiate calibration.
            </div>
          ) : (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase font-semibold">
                    <tr>
                      <th className="py-3 px-4">Enrollment ID</th>
                      <th className="py-3 px-4">Timestamp</th>
                      <th className="py-3 px-4">Duration</th>
                      <th className="py-3 px-4">Sample Rate</th>
                      <th className="py-3 px-4">Audio Quality</th>
                      <th className="py-3 px-4">Normal Confirmed</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {enrollments.map((enr) => (
                      <tr key={enr.enrollment_id} className="hover:bg-slate-50/70 transition">
                        <td className="py-3 px-4 font-mono text-[11px] text-slate-700">
                          {enr.enrollment_id.slice(0, 8)}...
                        </td>
                        <td className="py-3 px-4 text-slate-600 font-mono text-[11px]">
                          {new Date(enr.created_at).toLocaleString()}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-700">
                          {enr.duration_sec.toFixed(1)}s
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-700">
                          {enr.sample_rate} Hz
                        </td>
                        <td className="py-3 px-4">
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                            {enr.quality_status}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-emerald-600 font-semibold flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          <span>Confirmed</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Machine Inspection History Table */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-600" />
              <span>Inspection History ({analyses.length})</span>
            </h2>

            <Link
              href={`/machines/${machine.machine_id}/analyze`}
              className="text-xs font-semibold text-brand-600 hover:text-brand-700 flex items-center gap-1"
            >
              <span>+ New Inspection</span>
            </Link>
          </div>

          {analyses.length === 0 ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-8 text-center text-xs text-slate-500">
              No inspections recorded for this machine. Submit an audio sample to evaluate acoustic deviation.
            </div>
          ) : (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase font-semibold">
                    <tr>
                      <th className="py-3 px-4">Timestamp</th>
                      <th className="py-3 px-4">Model</th>
                      <th className="py-3 px-4">Anomaly Score</th>
                      <th className="py-3 px-4">Threshold</th>
                      <th className="py-3 px-4">Outcome</th>
                      <th className="py-3 px-4 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {analyses.map((ana) => (
                      <tr key={ana.analysis_id} className="hover:bg-slate-50/70 transition">
                        <td className="py-3 px-4 text-slate-600 font-mono text-[11px]">
                          {new Date(ana.created_at).toLocaleString()}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-700 text-[11px]">
                          {ana.model_name}
                        </td>
                        <td className="py-3 px-4 font-mono font-semibold text-slate-800">
                          {ana.anomaly_score !== null ? ana.anomaly_score.toFixed(4) : "—"}
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-500">
                          {ana.operating_threshold !== null ? ana.operating_threshold.toFixed(4) : "—"}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wider ${
                              ana.decision === "NORMAL_LIKE"
                                ? "bg-emerald-100 text-emerald-800"
                                : ana.decision === "UNUSUAL"
                                ? "bg-rose-100 text-rose-800"
                                : "bg-slate-100 text-slate-700"
                            }`}
                          >
                            <span>{ana.decision}</span>
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right">
                          <Link
                            href={`/analyses/${ana.analysis_id}`}
                            className="inline-flex items-center gap-1 font-semibold text-brand-600 hover:text-brand-700"
                          >
                            <FileText className="w-3.5 h-3.5" />
                            <span>Report</span>
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </ProtectedRoute>
  );
}

