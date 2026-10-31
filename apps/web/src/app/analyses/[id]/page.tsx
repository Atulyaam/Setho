"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import {
  apiClient,
  AnalysisRecord,
  InspectionReport,
} from "@/lib/api-client";
import {
  Activity,
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  FileText,
  Clock,
  Cpu,
  Layers,
  ShieldAlert,
  ShieldCheck,
  Printer,
  ChevronRight,
  Loader2,
  Gauge,
  Sliders,
} from "lucide-react";

export default function AnalysisResultPage() {
  const params = useParams();
  const analysisId = params.id as string;
  const router = useRouter();

  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [report, setReport] = useState<InspectionReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadAnalysis() {
      setIsLoading(true);
      setError(null);
      try {
        const [anaData, repData] = await Promise.all([
          apiClient.getAnalysis(analysisId),
          apiClient.getInspectionReport(analysisId),
        ]);
        setAnalysis(anaData);
        setReport(repData);
      } catch (err: any) {
        setError(err.message || "Failed to retrieve analysis record.");
      } finally {
        setIsLoading(false);
      }
    }
    loadAnalysis();
  }, [analysisId]);

  if (isLoading) {
    return (
      <ProtectedRoute>
        <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
          <Loader2 className="w-8 h-8 text-brand-600 animate-spin" />
          <p className="text-sm text-slate-500">Loading inspection findings...</p>
        </div>
      </ProtectedRoute>
    );
  }

  if (error || !analysis) {
    return (
      <ProtectedRoute>
        <div className="max-w-xl mx-auto my-12 bg-white rounded-2xl border border-rose-200 p-8 text-center space-y-4 shadow-sm">
          <div className="w-12 h-12 bg-rose-50 text-rose-600 rounded-2xl flex items-center justify-center mx-auto">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">Analysis Record Inaccessible</h2>
            <p className="text-xs text-rose-600 mt-1">{error || "Analysis not found."}</p>
          </div>
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold rounded-xl transition"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Dashboard</span>
          </Link>
        </div>
      </ProtectedRoute>
    );
  }

  const score = analysis.anomaly_score ?? 0;
  const thetaLower = analysis.theta_lower ?? 0.025;
  const thetaUpper = analysis.theta_upper ?? 0.045;
  const threshold = analysis.operating_threshold ?? thetaUpper;

  // Calculate position percentage on 3-zone scale [0 to 2 * thetaUpper]
  const maxScale = Math.max(thetaUpper * 2, score * 1.25, 0.1);
  const scorePercent = Math.min(100, Math.max(0, (score / maxScale) * 100));
  const lowerPercent = (thetaLower / maxScale) * 100;
  const upperPercent = (thetaUpper / maxScale) * 100;

  return (
    <ProtectedRoute>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Navigation & Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Link
              href={`/machines/${analysis.machine_id}`}
              className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-semibold">
                  Machine: {analysis.machine_id}
                </span>
                <span className="text-xs text-slate-500">
                  {new Date(analysis.created_at).toLocaleString()}
                </span>
              </div>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight mt-0.5">
                Inspection Analysis Result
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href={`/analyses/${analysisId}/report`}
              className="px-4 py-2 bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 rounded-xl text-xs font-semibold shadow-sm transition flex items-center gap-2"
            >
              <FileText className="w-4 h-4 text-brand-600" />
              <span>Full Inspection Report</span>
            </Link>

            <Link
              href={`/machines/${analysis.machine_id}/analyze`}
              className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white rounded-xl text-xs font-semibold shadow-sm transition flex items-center gap-2"
            >
              <Activity className="w-4 h-4" />
              <span>New Inspection</span>
            </Link>
          </div>
        </div>

        {/* Primary Outcome Decision Banner */}
        <div
          className={`rounded-2xl p-6 sm:p-8 border shadow-sm ${
            analysis.decision === "NORMAL_LIKE"
              ? "bg-emerald-50/60 border-emerald-200"
              : analysis.decision === "UNUSUAL"
              ? "bg-rose-50/60 border-rose-200"
              : "bg-slate-100/70 border-slate-300"
          }`}
        >
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-start gap-4">
              <div
                className={`p-3.5 rounded-2xl ${
                  analysis.decision === "NORMAL_LIKE"
                    ? "bg-emerald-100 text-emerald-700"
                    : analysis.decision === "UNUSUAL"
                    ? "bg-rose-100 text-rose-700"
                    : "bg-slate-200 text-slate-700"
                }`}
              >
                {analysis.decision === "NORMAL_LIKE" && <CheckCircle2 className="w-8 h-8" />}
                {analysis.decision === "UNUSUAL" && <AlertTriangle className="w-8 h-8" />}
                {analysis.decision === "INCONCLUSIVE" && <HelpCircle className="w-8 h-8" />}
              </div>

              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
                    Inspection Outcome
                  </span>
                  {analysis.is_provisional && (
                    <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-full bg-amber-100 text-amber-800">
                      Provisional Calibration
                    </span>
                  )}
                </div>

                <h2 className="text-2xl font-black tracking-tight text-slate-900 mt-0.5">
                  {analysis.decision}
                </h2>

                <p className="text-xs text-slate-700 mt-1 max-w-xl leading-relaxed">
                  {analysis.decision === "NORMAL_LIKE" && (
                    <>
                      <strong>Meaning: </strong>The recording falls within the current machine-specific reference range. Reconstruction deviation ({score.toFixed(4)}) is strictly below the operating threshold ({threshold.toFixed(4)}).
                    </>
                  )}
                  {analysis.decision === "UNUSUAL" && (
                    <>
                      <strong>Meaning: </strong>The recording shows an acoustic deviation from the current reference. Measured reconstruction error ({score.toFixed(4)}) exceeds the calibrated normal threshold ({threshold.toFixed(4)}).
                    </>
                  )}
                  {analysis.decision === "INCONCLUSIVE" && (
                    <>
                      <strong>Meaning: </strong>Audio quality, calibration, or supported operating conditions do not permit a reliable interpretation.
                    </>
                  )}
                </p>
              </div>
            </div>

            <div className="sm:text-right bg-white/80 p-4 rounded-xl border border-slate-200/60 min-w-[140px]">
              <span className="text-[10px] uppercase font-mono text-slate-400 block">
                Anomaly Score
              </span>
              <span className="text-2xl font-bold font-mono text-slate-900">
                {score.toFixed(4)}
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">
                Threshold: {threshold.toFixed(4)}
              </span>
            </div>
          </div>
        </div>

        {/* Calibrated 3-Zone Reference Scale Gauge */}
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-2">
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
              <Gauge className="w-4 h-4 text-brand-600" />
              <span>Calibrated Three-Zone Reference Scale</span>
            </h2>
            <span className="text-[11px] font-mono text-slate-500">
              Score: {score.toFixed(4)} | Lower: {thetaLower.toFixed(4)} | Upper: {thetaUpper.toFixed(4)}
            </span>
          </div>

          <div className="space-y-2 pt-2">
            {/* Visual Gauge Bar */}
            <div className="relative h-6 w-full rounded-xl overflow-hidden flex bg-slate-100 border border-slate-200">
              {/* Normal Zone (0 to theta_lower) */}
              <div
                className="bg-emerald-500/80 h-full transition-all"
                style={{ width: `${lowerPercent}%` }}
                title={`NORMAL_LIKE Range (0 to ${thetaLower.toFixed(4)})`}
              />
              {/* Inconclusive / Buffer Zone (theta_lower to theta_upper) */}
              <div
                className="bg-amber-400/80 h-full transition-all"
                style={{ width: `${upperPercent - lowerPercent}%` }}
                title={`INCONCLUSIVE / Buffer Range (${thetaLower.toFixed(4)} to ${thetaUpper.toFixed(4)})`}
              />
              {/* Unusual Zone (> theta_upper) */}
              <div
                className="bg-rose-500/80 h-full flex-1 transition-all"
                title={`UNUSUAL Range (> ${thetaUpper.toFixed(4)})`}
              />

              {/* Pin for Current Anomaly Score */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-slate-950 shadow-md transform -translate-x-1/2 z-10"
                style={{ left: `${scorePercent}%` }}
              >
                <div className="absolute -top-6 -left-3 text-[10px] font-bold font-mono bg-slate-900 text-white px-1.5 py-0.5 rounded shadow">
                  ▼ {score.toFixed(3)}
                </div>
              </div>
            </div>

            {/* Zone Labels */}
            <div className="grid grid-cols-3 text-[11px] font-medium pt-3">
              <div className="text-emerald-700">
                <span className="font-bold">NORMAL_LIKE</span>
                <span className="block text-[10px] text-slate-500">
                  ≤ {thetaLower.toFixed(4)} (90th percentile)
                </span>
              </div>
              <div className="text-amber-700 text-center">
                <span className="font-bold">INCONCLUSIVE / BUFFER</span>
                <span className="block text-[10px] text-slate-500">
                  {thetaLower.toFixed(4)} — {thetaUpper.toFixed(4)}
                </span>
              </div>
              <div className="text-rose-700 text-right">
                <span className="font-bold">UNUSUAL DEVIATION</span>
                <span className="block text-[10px] text-slate-500">
                  &gt; {thetaUpper.toFixed(4)} (98th percentile)
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Technical Findings & AQG Assessment */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Card 1: Audio Quality Gate */}
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-3">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 border-b border-slate-100 pb-2">
              <ShieldCheck className="w-4 h-4 text-brand-600" />
              <span>Audio Quality Gate Assessment</span>
            </h3>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">Quality Status:</span>
                <span className="font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  {analysis.quality_status}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Processing Latency:</span>
                <span className="font-mono text-slate-700">
                  {analysis.processing_duration_ms.toFixed(1)} ms (CPU)
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Model Architecture:</span>
                <span className="font-mono text-brand-700 font-semibold">
                  {analysis.model_name} (v{analysis.model_version})
                </span>
              </div>
            </div>
          </div>

          {/* Card 2: Physical Inspection Guidance */}
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 space-y-3">
            <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 border-b border-slate-100 pb-2">
              <Activity className="w-4 h-4 text-indigo-600" />
              <span>Recommended Physical Action</span>
            </h3>

            <p className="text-xs text-slate-700 leading-relaxed">
              {report?.recommended_action || "Continue routine scheduled monitoring."}
            </p>
          </div>
        </div>

        {/* Strict Engineering Disclaimer */}
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-slate-600 text-xs flex items-start gap-3">
          <ShieldAlert className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong>Engineering Disclaimer: </strong>
            Stetho evaluates acoustic deviations against an individual machine&apos;s learned normal baseline. It does not diagnose specific mechanical failure modes, predict exact time to failure, or claim that the machine is physically safe simply because the result is NORMAL_LIKE. Certified mechanical inspection is always required for critical machinery.
          </p>
        </div>
      </div>
    </ProtectedRoute>
  );
}
