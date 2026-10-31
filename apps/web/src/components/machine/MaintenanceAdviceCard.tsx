"use client";

import React, { useEffect, useState } from "react";
import { useUI } from "../../context/UIContext";

interface Finding {
  type: string;
  evidence: string;
  explanation: string;
}

interface ChecklistItem {
  action: string;
  priority: "IMMEDIATE" | "HIGH" | "MEDIUM" | "ROUTINE";
  instructions: string;
}

interface MaintenanceAdvice {
  analysis_id: string;
  machine_id: string;
  decision: string;
  maintenance_urgency: string;
  anomaly_score: number;
  operating_threshold: number;
  deviation_ratio: number;
  findings: Finding[];
  inspection_checklist: ChecklistItem[];
  engineering_disclaimer: string;
}

interface MaintenanceAdviceCardProps {
  analysisId: string;
}

export function MaintenanceAdviceCard({ analysisId }: MaintenanceAdviceCardProps) {
  const { t } = useUI();
  const [advice, setAdvice] = useState<MaintenanceAdvice | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchAdvice() {
      try {
        setLoading(true);
        const res = await fetch(`/api/backend/analyses/${analysisId}/maintenance-advice`);
        if (!res.ok) throw new Error("Failed to load maintenance guidance.");
        const data = await res.json();
        setAdvice(data);
      } catch (err: any) {
        setError(err.message || "Unable to fetch maintenance guidance.");
      } finally {
        setLoading(false);
      }
    }
    fetchAdvice();
  }, [analysisId]);

  if (loading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse">
        <div className="h-5 bg-slate-800 rounded w-1/3 mb-4"></div>
        <div className="h-20 bg-slate-800/50 rounded mb-4"></div>
      </div>
    );
  }

  if (error || !advice) {
    return null;
  }

  const urgencyColors = {
    HIGH: "bg-red-500/10 text-red-400 border-red-500/30",
    MEDIUM: "bg-amber-500/10 text-amber-400 border-amber-500/30",
    LOW: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
    VERIFY_RECORDING: "bg-cyan-500/10 text-cyan-400 border-cyan-500/30",
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100 shadow-xl">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-base font-semibold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            {t("maintenanceTitle")}
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">{t("maintenanceSubtitle")}</p>
        </div>
        <span
          className={`px-3 py-1 rounded-full text-xs font-mono font-semibold border ${
            urgencyColors[advice.maintenance_urgency as keyof typeof urgencyColors] ||
            "bg-slate-800 text-slate-300"
          }`}
        >
          {advice.maintenance_urgency}
        </span>
      </div>

      {/* Signal Findings */}
      <div className="space-y-2 mb-5">
        <h4 className="text-xs font-mono uppercase text-slate-400">Empirical Signal Findings</h4>
        {advice.findings.map((f, i) => (
          <div key={i} className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-lg text-xs">
            <span className="font-semibold text-cyan-300 block mb-0.5">{f.evidence}</span>
            <p className="text-slate-400 leading-relaxed">{f.explanation}</p>
          </div>
        ))}
      </div>

      {/* Actionable Maintenance Checklist */}
      <div className="space-y-2 mb-5">
        <h4 className="text-xs font-mono uppercase text-slate-400">Actionable Inspection Checklist</h4>
        <div className="space-y-2">
          {advice.inspection_checklist.map((item, idx) => (
            <div
              key={idx}
              className="p-3 bg-slate-950/40 border border-slate-800/60 rounded-lg flex items-start justify-between gap-4"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                  <span className="text-sm font-medium text-slate-200">{item.action}</span>
                </div>
                <p className="text-xs text-slate-400 leading-normal">{item.instructions}</p>
              </div>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase ${
                  item.priority === "IMMEDIATE" || item.priority === "HIGH"
                    ? "bg-red-500/20 text-red-300 border border-red-500/30"
                    : item.priority === "MEDIUM"
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                    : "bg-slate-800 text-slate-300"
                }`}
              >
                {item.priority}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <p className="text-[11px] text-slate-500 italic pt-2 border-t border-slate-800/60">
        {advice.engineering_disclaimer}
      </p>
    </div>
  );
}
