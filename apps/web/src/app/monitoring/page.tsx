"use client";

import React from "react";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { FleetPriorityTable } from "@/components/fleet/FleetPriorityTable";
import { useUI } from "@/context/UIContext";
import { AlertCircle, Activity, Clock, ShieldCheck } from "lucide-react";

export default function FleetMonitoringPage() {
  const { t } = useUI();

  return (
    <ProtectedRoute>
      <div className="space-y-8">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-400">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-100">
                {t("monitoring")}
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                Deterministic multi-factor dispatch ranking for predictive inspection routing.
              </p>
            </div>
          </div>
        </div>

        {/* Priority Engine Pillars */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl flex items-start gap-3">
            <div className="p-2 bg-red-500/10 text-red-400 rounded-lg">
              <AlertCircle className="w-5 h-5" />
            </div>
            <div>
              <span className="text-xs font-semibold text-slate-200 block">Acoustic Deviation</span>
              <span className="text-[11px] text-slate-400 leading-tight block mt-0.5">
                Weighted points when anomaly score exceeds calibrated threshold θ.
              </span>
            </div>
          </div>

          <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl flex items-start gap-3">
            <div className="p-2 bg-amber-500/10 text-amber-400 rounded-lg">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <span className="text-xs font-semibold text-slate-200 block">Repeat Abnormalities</span>
              <span className="text-[11px] text-slate-400 leading-tight block mt-0.5">
                Elevates priority if multiple consecutive inspections detect unusual acoustics.
              </span>
            </div>
          </div>

          <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl flex items-start gap-3">
            <div className="p-2 bg-yellow-500/10 text-yellow-400 rounded-lg">
              <Clock className="w-5 h-5" />
            </div>
            <div>
              <span className="text-xs font-semibold text-slate-200 block">Inspection Staleness</span>
              <span className="text-[11px] text-slate-400 leading-tight block mt-0.5">
                Assets uninspected for &gt; 3 days gain time-based maintenance priority.
              </span>
            </div>
          </div>

          <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl flex items-start gap-3">
            <div className="p-2 bg-cyan-500/10 text-cyan-400 rounded-lg">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <span className="text-xs font-semibold text-slate-200 block">Quality Verification</span>
              <span className="text-[11px] text-slate-400 leading-tight block mt-0.5">
                Inconclusive or noisy recordings prioritized for verified re-recording.
              </span>
            </div>
          </div>
        </div>

        {/* Priority Table */}
        <FleetPriorityTable />
      </div>
    </ProtectedRoute>
  );
}
