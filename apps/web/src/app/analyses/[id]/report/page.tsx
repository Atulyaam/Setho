"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { apiClient, InspectionReport } from "@/lib/api-client";
import {
  FileText,
  Printer,
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Cpu,
  Clock,
  ShieldCheck,
  ShieldAlert,
  Loader2,
  Stethoscope,
} from "lucide-react";

export default function InspectionReportPage() {
  const params = useParams();
  const analysisId = params.id as string;
  const router = useRouter();

  const [report, setReport] = useState<InspectionReport | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadReport() {
      setIsLoading(true);
      setError(null);
      try {
        const data = await apiClient.getInspectionReport(analysisId);
        setReport(data);
      } catch (err: any) {
        setError(err.message || "Failed to load inspection report.");
      } finally {
        setIsLoading(false);
      }
    }
    loadReport();
  }, [analysisId]);

  const handlePrint = () => {
    window.print();
  };

  if (isLoading) {
    return (
      <ProtectedRoute>
        <div className="flex flex-col items-center justify-center min-h-[50vh] space-y-4">
          <Loader2 className="w-8 h-8 text-brand-600 animate-spin" />
          <p className="text-sm text-slate-500">Generating structured report...</p>
        </div>
      </ProtectedRoute>
    );
  }

  if (error || !report) {
    return (
      <ProtectedRoute>
        <div className="max-w-xl mx-auto my-12 bg-white rounded-2xl border border-rose-200 p-8 text-center space-y-4">
          <AlertTriangle className="w-8 h-8 text-rose-500 mx-auto" />
          <h2 className="text-lg font-bold text-slate-900">Report Unavailable</h2>
          <p className="text-xs text-rose-600">{error || "Report not found."}</p>
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold rounded-xl"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Dashboard</span>
          </Link>
        </div>
      </ProtectedRoute>
    );
  }

  const { machine, analysis } = report;

  return (
    <ProtectedRoute>
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Action Header (Hidden in Print) */}
        <div className="flex items-center justify-between gap-4 print:hidden">
          <Link
            href={`/analyses/${analysisId}`}
            className="inline-flex items-center gap-2 text-xs font-semibold text-slate-600 hover:text-slate-900"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Analysis Findings</span>
          </Link>

          <button
            onClick={handlePrint}
            className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-semibold shadow-sm transition flex items-center gap-2"
          >
            <Printer className="w-4 h-4" />
            <span>Print / Save as PDF</span>
          </button>
        </div>

        {/* Formal Printable Document Card */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-lg p-8 sm:p-12 space-y-8 print:border-none print:shadow-none print:p-0">
          {/* Document Header */}
          <div className="flex items-start justify-between border-b border-slate-200 pb-6">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-brand-600 text-white rounded-xl">
                <Stethoscope className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900 tracking-tight">
                  Stetho Acoustic Inspection Report
                </h1>
                <p className="text-xs text-slate-500">
                  Machine-Specific Baseline Anomaly Monitoring Telemetry
                </p>
              </div>
            </div>

            <div className="text-right text-xs space-y-0.5">
              <div className="font-mono text-slate-700 font-semibold">
                Report ID: {report.report_id.slice(0, 8)}
              </div>
              <div className="text-slate-500 text-[11px]">
                Generated: {new Date(report.generated_at).toLocaleString()}
              </div>
            </div>
          </div>

          {/* Section 1: Executive Summary */}
          <div className="space-y-3">
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider border-b border-slate-100 pb-1.5">
              1. Executive Summary
            </h2>
            <div
              className={`p-4 rounded-xl border ${
                analysis.decision === "NORMAL_LIKE"
                  ? "bg-emerald-50/70 border-emerald-200 text-emerald-950"
                  : analysis.decision === "UNUSUAL"
                  ? "bg-rose-50/70 border-rose-200 text-rose-950"
                  : "bg-slate-50 border-slate-200 text-slate-900"
              }`}
            >
              <div className="flex items-center gap-2 font-bold text-sm mb-1">
                {analysis.decision === "NORMAL_LIKE" && <CheckCircle2 className="w-4 h-4 text-emerald-600" />}
                {analysis.decision === "UNUSUAL" && <AlertTriangle className="w-4 h-4 text-rose-600" />}
                {analysis.decision === "INCONCLUSIVE" && <HelpCircle className="w-4 h-4 text-slate-600" />}
                <span>Outcome: {analysis.decision}</span>
              </div>
              <p className="text-xs leading-relaxed">{report.executive_summary}</p>
            </div>
          </div>

          {/* Section 2: Machine Specifications & Operating Context */}
          <div className="space-y-3">
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider border-b border-slate-100 pb-1.5">
              2. Target Machine Profile &amp; Calibration Baseline
            </h2>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Machine Identifier</span>
                <span className="font-mono font-semibold text-slate-800">{machine.machine_id}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Asset Name</span>
                <span className="font-semibold text-slate-800">{machine.name}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Asset Category</span>
                <span className="capitalize font-medium text-slate-700">{machine.category}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Calibration State</span>
                <span className="font-bold uppercase text-slate-800">{machine.calibration_status}</span>
              </div>
            </div>

            <div className="text-xs bg-slate-50 p-3.5 rounded-xl border border-slate-100">
              <span className="text-slate-500 font-semibold">Operating Condition: </span>
              <span className="text-slate-700">{machine.operating_condition || "Standard documented baseline"}</span>
            </div>
          </div>

          {/* Section 3: Technical Acoustic Analysis Metrics */}
          <div className="space-y-3">
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider border-b border-slate-100 pb-1.5">
              3. Measured Acoustic Telemetry &amp; Quality Assessment
            </h2>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div className="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span className="text-slate-400 block text-[10px] uppercase">Reconstruction Error</span>
                <span className="font-mono text-base font-bold text-slate-900">
                  {analysis.anomaly_score?.toFixed(4) ?? "—"}
                </span>
                <span className="text-[10px] text-slate-500 block">L1 Spectrogram MSE</span>
              </div>

              <div className="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span className="text-slate-400 block text-[10px] uppercase">Operating Threshold</span>
                <span className="font-mono text-base font-bold text-slate-900">
                  {analysis.operating_threshold?.toFixed(4) ?? "—"}
                </span>
                <span className="text-[10px] text-slate-500 block">98th % Baseline (θ_upper)</span>
              </div>

              <div className="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span className="text-slate-400 block text-[10px] uppercase">Audio Quality Gate</span>
                <span className="font-semibold text-emerald-700 text-sm block mt-0.5">
                  {analysis.quality_status}
                </span>
                <span className="text-[10px] text-slate-500 block">No Clipping / Saturation</span>
              </div>

              <div className="bg-slate-50 p-3 rounded-xl border border-slate-100">
                <span className="text-slate-400 block text-[10px] uppercase">Model Policy</span>
                <span className="font-mono font-semibold text-brand-700 text-xs block mt-0.5">
                  {analysis.model_name}
                </span>
                <span className="text-[10px] text-slate-500 block">v{analysis.model_version}</span>
              </div>
            </div>

            {report.technical_findings && report.technical_findings.length > 0 && (
              <div className="text-xs space-y-1 pt-1">
                <span className="font-semibold text-slate-700">Detailed Findings:</span>
                <ul className="list-disc list-inside space-y-0.5 text-slate-600 pl-2">
                  {report.technical_findings.map((f, i) => (
                    <li key={i}>{f}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Section 4: Recommended Physical Inspection Steps */}
          <div className="space-y-3">
            <h2 className="text-xs font-bold text-slate-900 uppercase tracking-wider border-b border-slate-100 pb-1.5">
              4. Recommended Physical Inspection Actions
            </h2>
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-800 leading-relaxed">
              {report.recommended_action}
            </div>
          </div>

          {/* Section 5: Engineering Disclaimer */}
          <div className="space-y-2 pt-4 border-t border-slate-200 text-[11px] text-slate-500 leading-relaxed">
            <span className="font-bold text-slate-700 uppercase tracking-wider text-[10px] block">
              Certified Engineering Disclaimer
            </span>
            <p>{report.engineering_disclaimer}</p>
          </div>
        </div>
      </div>
    </ProtectedRoute>
  );
}

