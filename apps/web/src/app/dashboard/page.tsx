"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { useAuth } from "@/context/AuthContext";
import { apiClient, MachineProfile, AnalysisRecord } from "@/lib/api-client";
import {
  Cpu,
  Layers,
  Activity,
  PlusCircle,
  Mic,
  Search,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  ArrowRight,
  Clock,
  FileText,
  Loader2,
  ChevronRight,
  ShieldCheck,
} from "lucide-react";

export default function DashboardPage() {
  const { user } = useAuth();
  const [machines, setMachines] = useState<MachineProfile[]>([]);
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadDashboardData() {
      setIsLoading(true);
      setError(null);
      try {
        const [machinesRes, analysesRes] = await Promise.all([
          apiClient.getMachines(),
          apiClient.getRecentAnalyses(6),
        ]);
        setMachines(machinesRes.machines);
        setRecentAnalyses(analysesRes.analyses);
      } catch (err: any) {
        setError(err.message || "Failed to load dashboard telemetry.");
      } finally {
        setIsLoading(false);
      }
    }
    loadDashboardData();
  }, []);

  const calibratedCount = machines.filter((m) => m.calibration_status === "calibrated").length;
  const provisionalCount = machines.filter((m) => m.calibration_status === "provisional").length;

  return (
    <ProtectedRoute>
      <div className="space-y-8">
        {/* Header Section */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                Acoustic Inspection Dashboard
              </h1>
              <span className="text-xs px-2.5 py-0.5 rounded-full font-medium bg-brand-50 text-brand-700 border border-brand-200">
                {user?.role === "admin" ? "Fleet Administrator" : "Inspector Workspace"}
              </span>
            </div>
            <p className="text-sm text-slate-500 mt-1">
              Live acoustic monitoring, reference calibration states, and baseline deviation detection.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/machines/new"
              className="px-4 py-2.5 bg-brand-600 hover:bg-brand-700 text-white rounded-xl shadow-sm text-sm font-semibold transition flex items-center gap-2"
            >
              <PlusCircle className="w-4 h-4" />
              <span>Register Machine</span>
            </Link>
          </div>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 text-rose-500" />
            <span>{error}</span>
          </div>
        )}

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm flex items-center gap-4">
            <div className="p-3 rounded-xl bg-teal-50 text-brand-600">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-900">{machines.length}</div>
              <div className="text-xs text-slate-500 font-medium">Registered Machines</div>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm flex items-center gap-4">
            <div className="p-3 rounded-xl bg-emerald-50 text-emerald-600">
              <CheckCircle2 className="w-6 h-6" />
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-900">{calibratedCount}</div>
              <div className="text-xs text-slate-500 font-medium">Calibrated (N ≥ 30)</div>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm flex items-center gap-4">
            <div className="p-3 rounded-xl bg-amber-50 text-amber-600">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-900">{provisionalCount}</div>
              <div className="text-xs text-slate-500 font-medium">Provisional (N &lt; 30)</div>
            </div>
          </div>

          <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-sm flex items-center gap-4">
            <div className="p-3 rounded-xl bg-indigo-50 text-indigo-600">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <div className="text-2xl font-bold text-slate-900">{recentAnalyses.length}</div>
              <div className="text-xs text-slate-500 font-medium">Recent Inspections</div>
            </div>
          </div>
        </div>

        {/* Machine Fleet Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Cpu className="w-5 h-5 text-brand-600" />
              <span>Machine Fleet Overview</span>
            </h2>
            <Link
              href="/machines"
              className="text-xs font-semibold text-brand-600 hover:text-brand-700 flex items-center gap-1"
            >
              <span>View All Machines</span>
              <ChevronRight className="w-4 h-4" />
            </Link>
          </div>

          {isLoading ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center flex flex-col items-center justify-center space-y-3">
              <Loader2 className="w-6 h-6 text-brand-600 animate-spin" />
              <p className="text-sm text-slate-500">Querying machine telemetry...</p>
            </div>
          ) : machines.length === 0 ? (
            /* Clear Empty State for New Users */
            <div className="bg-white rounded-2xl border border-dashed border-slate-300 p-12 text-center space-y-4 max-w-xl mx-auto">
              <div className="w-12 h-12 bg-brand-50 text-brand-600 rounded-2xl flex items-center justify-center mx-auto">
                <Cpu className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-slate-900">No Machines Registered Yet</h3>
                <p className="text-xs text-slate-500 max-w-sm mx-auto">
                  Register your physical industrial machine profile to begin enrolling confirmed normal audio and establishing machine-specific baseline calibration.
                </p>
              </div>
              <Link
                href="/machines/new"
                className="inline-flex items-center gap-2 px-4 py-2.5 bg-brand-600 hover:bg-brand-700 text-white rounded-xl text-xs font-semibold shadow-sm transition"
              >
                <PlusCircle className="w-4 h-4" />
                <span>Register First Machine</span>
              </Link>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {machines.map((machine) => (
                <div
                  key={machine.machine_id}
                  className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 flex flex-col justify-between hover:border-brand-500/50 hover:shadow-md transition"
                >
                  <div className="space-y-3">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                          {machine.machine_id}
                        </span>
                        <h3 className="text-base font-bold text-slate-900 mt-1 line-clamp-1">
                          {machine.name}
                        </h3>
                      </div>
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

                    <p className="text-xs text-slate-600 line-clamp-2">
                      {machine.description || "Industrial operating asset with continuous acoustic monitoring."}
                    </p>

                    <div className="pt-2 border-t border-slate-100 grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-slate-400 block text-[10px] uppercase">Enrolled Normal</span>
                        <span className="font-semibold text-slate-800">
                          {machine.enrolled_recording_count} recordings
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400 block text-[10px] uppercase">Model Policy</span>
                        <span className="font-semibold text-brand-700 font-mono text-[11px]">
                          {machine.selected_model_policy}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between gap-2">
                    <Link
                      href={`/machines/${machine.machine_id}/enroll`}
                      className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-teal-50 hover:bg-teal-100 text-brand-700 transition flex items-center gap-1.5"
                    >
                      <Mic className="w-3.5 h-3.5" />
                      <span>Enroll</span>
                    </Link>

                    <Link
                      href={`/machines/${machine.machine_id}/analyze`}
                      className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-brand-600 hover:bg-brand-700 text-white transition flex items-center gap-1.5 shadow-sm"
                    >
                      <Activity className="w-3.5 h-3.5" />
                      <span>Analyze</span>
                    </Link>

                    <Link
                      href={`/machines/${machine.machine_id}`}
                      className="px-2.5 py-1.5 text-xs font-medium text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition"
                      title="View Details"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Recent Analysis Activity Section */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Activity className="w-5 h-5 text-indigo-600" />
              <span>Recent Inspection Outcomes</span>
            </h2>
          </div>

          {isLoading ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-8 text-center text-sm text-slate-500">
              Loading recent inspection telemetry...
            </div>
          ) : recentAnalyses.length === 0 ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-8 text-center text-sm text-slate-500">
              No recent acoustic inspections recorded. Select a machine above to perform your first acoustic analysis.
            </div>
          ) : (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase font-semibold">
                    <tr>
                      <th className="py-3 px-4">Timestamp</th>
                      <th className="py-3 px-4">Machine ID</th>
                      <th className="py-3 px-4">Model</th>
                      <th className="py-3 px-4">Anomaly Score</th>
                      <th className="py-3 px-4">Operating Threshold</th>
                      <th className="py-3 px-4">Outcome</th>
                      <th className="py-3 px-4 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {recentAnalyses.map((ana) => (
                      <tr key={ana.analysis_id} className="hover:bg-slate-50/70 transition">
                        <td className="py-3.5 px-4 text-slate-600 font-mono text-[11px]">
                          {new Date(ana.created_at).toLocaleString()}
                        </td>
                        <td className="py-3.5 px-4 font-mono font-semibold text-slate-800">
                          {ana.machine_id}
                        </td>
                        <td className="py-3.5 px-4 font-mono text-slate-600 text-[11px]">
                          {ana.model_name}
                        </td>
                        <td className="py-3.5 px-4 font-mono font-medium text-slate-800">
                          {ana.anomaly_score !== null ? ana.anomaly_score.toFixed(4) : "—"}
                        </td>
                        <td className="py-3.5 px-4 font-mono text-slate-500">
                          {ana.operating_threshold !== null ? ana.operating_threshold.toFixed(4) : "—"}
                        </td>
                        <td className="py-3.5 px-4">
                          <span
                            className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold uppercase tracking-wider ${
                              ana.decision === "NORMAL_LIKE"
                                ? "bg-emerald-100 text-emerald-800"
                                : ana.decision === "UNUSUAL"
                                ? "bg-rose-100 text-rose-800"
                                : "bg-slate-100 text-slate-700"
                            }`}
                          >
                            {ana.decision === "NORMAL_LIKE" && <CheckCircle2 className="w-3 h-3 text-emerald-600" />}
                            {ana.decision === "UNUSUAL" && <AlertTriangle className="w-3 h-3 text-rose-600" />}
                            {ana.decision === "INCONCLUSIVE" && <HelpCircle className="w-3 h-3 text-slate-500" />}
                            <span>{ana.decision}</span>
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-right">
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
