"use client";

import React, { useEffect, useState } from "react";
import { useUI } from "../../context/UIContext";

interface DriftPoint {
  analysis_id: string;
  created_at: string;
  decision: string;
  anomaly_score: number;
  operating_threshold: number;
  quality_status: string;
  spectral_drift_pct: number;
  model_version: string;
}

interface AcousticDriftTimelineProps {
  machineId: string;
}

export function AcousticDriftTimeline({ machineId }: AcousticDriftTimelineProps) {
  const { t } = useUI();
  const [points, setPoints] = useState<DriftPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchTimeline() {
      try {
        setLoading(true);
        const res = await fetch(`/api/backend/machines/${machineId}/drift-timeline`);
        if (!res.ok) throw new Error("Failed to load timeline.");
        const json = await res.json();
        setPoints(json.timeline || []);
      } catch (e) {
        console.error("Error loading drift timeline:", e);
      } finally {
        setLoading(false);
      }
    }
    fetchTimeline();
  }, [machineId]);

  if (loading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse">
        <div className="h-4 bg-slate-800 rounded w-1/4 mb-4"></div>
        <div className="h-32 bg-slate-800/50 rounded"></div>
      </div>
    );
  }

  if (points.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-400 text-sm">
        <h3 className="text-base font-semibold text-slate-200 mb-2">{t("driftTimeline")}</h3>
        <p className="text-xs text-slate-500">
          No historical inspections recorded yet. Analyze acoustic recordings to map long-term drift.
        </p>
      </div>
    );
  }

  // Calculate drift direction
  const recentScores = points.slice(-3).map((p) => p.anomaly_score);
  const isDriftingUp =
    recentScores.length >= 2 && recentScores[recentScores.length - 1] > recentScores[0] * 1.15;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100 shadow-xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-base font-semibold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            {t("driftTimeline")}
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Chronological reconstruction error and spectral shift tracking ({points.length} events)
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isDriftingUp ? (
            <span className="px-2.5 py-1 bg-amber-500/10 text-amber-300 border border-amber-500/30 text-xs rounded-full font-medium">
              📈 Upward Acoustic Drift Detected
            </span>
          ) : (
            <span className="px-2.5 py-1 bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 text-xs rounded-full font-medium">
              ✓ Baseline Stable
            </span>
          )}
        </div>
      </div>

      {/* Timeline List */}
      <div className="space-y-3 max-h-72 overflow-y-auto pr-1">
        {points.map((pt, idx) => {
          const dateStr = new Date(pt.created_at).toLocaleString();
          const isAbove = pt.anomaly_score > pt.operating_threshold;

          return (
            <div
              key={pt.analysis_id || idx}
              className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-lg flex items-center justify-between text-xs hover:border-slate-700 transition"
            >
              <div className="flex items-center gap-3">
                <span
                  className={`w-2 h-2 rounded-full ${
                    pt.decision === "UNUSUAL"
                      ? "bg-red-400"
                      : pt.decision === "INCONCLUSIVE"
                      ? "bg-amber-400"
                      : "bg-emerald-400"
                  }`}
                ></span>
                <div>
                  <span className="font-mono text-slate-300 block">{dateStr}</span>
                  <span className="text-[11px] text-slate-500 font-mono">
                    Score: {pt.anomaly_score?.toFixed(5)} / Thresh: {pt.operating_threshold?.toFixed(5)}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono ${
                    pt.decision === "UNUSUAL"
                      ? "bg-red-500/20 text-red-300 border border-red-500/30"
                      : pt.decision === "INCONCLUSIVE"
                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                      : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                  }`}
                >
                  {pt.decision}
                </span>

                <a
                  href={`/analyses/${pt.analysis_id}`}
                  className="text-cyan-400 hover:text-cyan-300 underline font-mono text-[11px]"
                >
                  Inspect →
                </a>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
