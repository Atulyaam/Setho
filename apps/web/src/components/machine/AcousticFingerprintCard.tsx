"use client";

import React, { useEffect, useState } from "react";
import { useUI } from "../../context/UIContext";

interface FingerprintData {
  fingerprint_id: string;
  machine_id: string;
  fingerprint_version: string;
  spectral_centroid_hz: number;
  sub_bass_energy_db: number;
  low_energy_db: number;
  mid_energy_db: number;
  high_energy_db: number;
  baseline_reconstruction_error: number;
  baseline_quality_score: number;
  sample_count: number;
  updated_at: string;
}

interface AcousticFingerprintCardProps {
  machineId: string;
}

export function AcousticFingerprintCard({ machineId }: AcousticFingerprintCardProps) {
  const { t } = useUI();
  const [data, setData] = useState<FingerprintData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchFingerprint() {
      try {
        setLoading(true);
        const res = await fetch(`/api/backend/machines/${machineId}/fingerprint`);
        if (!res.ok) throw new Error("Failed to load acoustic fingerprint.");
        const json = await res.json();
        setData(json.fingerprint);
      } catch (err: any) {
        setError(err.message || "Failed to load fingerprint.");
      } finally {
        setLoading(false);
      }
    }
    fetchFingerprint();
  }, [machineId]);

  if (loading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse">
        <div className="h-5 bg-slate-800 rounded w-1/3 mb-4"></div>
        <div className="h-20 bg-slate-800/50 rounded mb-4"></div>
        <div className="h-4 bg-slate-800 rounded w-2/3"></div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-400 text-sm">
        <div className="flex items-center gap-2 text-amber-400 mb-2 font-semibold">
          <span>⚠️</span>
          <span>Acoustic Fingerprint Unestablished</span>
        </div>
        <p className="text-xs text-slate-500">
          Enroll at least 1 confirmed normal acoustic recording to establish a baseline spectral fingerprint.
        </p>
      </div>
    );
  }

  // Calculate normalized percentages for band display
  const subBass = Math.max(0, 100 + data.sub_bass_energy_db * 2);
  const low = Math.max(0, 100 + data.low_energy_db * 2);
  const mid = Math.max(0, 100 + data.mid_energy_db * 2);
  const high = Math.max(0, 100 + data.high_energy_db * 2);
  const total = subBass + low + mid + high || 1;

  const subBassPct = Math.round((subBass / total) * 100);
  const lowPct = Math.round((low / total) * 100);
  const midPct = Math.round((mid / total) * 100);
  const highPct = Math.round((high / total) * 100);

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100 shadow-xl">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-semibold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            {t("acousticFingerprint")}
          </h3>
          <span className="text-xs text-slate-400 font-mono">v{data.fingerprint_version} • N={data.sample_count} baselines</span>
        </div>
        <span className="px-2.5 py-1 bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 text-xs rounded-full font-mono font-medium">
          {data.spectral_centroid_hz.toFixed(0)} Hz Centroid
        </span>
      </div>

      {/* Multi-Band Spectral Energy Distribution */}
      <div className="space-y-3 mb-6">
        <div className="text-xs font-medium text-slate-400 flex justify-between">
          <span>Spectral Frequency Bands</span>
          <span className="font-mono text-[11px] text-slate-500">Relative Acoustic Energy</span>
        </div>

        <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden flex border border-slate-800">
          <div style={{ width: `${subBassPct}%` }} className="bg-indigo-500" title={`Sub-Bass (20-250Hz): ${subBassPct}%`}></div>
          <div style={{ width: `${lowPct}%` }} className="bg-cyan-500" title={`Low (250-1kHz): ${lowPct}%`}></div>
          <div style={{ width: `${midPct}%` }} className="bg-emerald-500" title={`Mid (1k-4kHz): ${midPct}%`}></div>
          <div style={{ width: `${highPct}%` }} className="bg-amber-500" title={`High (4k-8kHz): ${highPct}%`}></div>
        </div>

        {/* Legend */}
        <div className="grid grid-cols-4 gap-2 text-[11px] font-mono text-slate-400 pt-1">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
            <span>Sub: {subBassPct}%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
            <span>Low: {lowPct}%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span>Mid: {midPct}%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-500"></span>
            <span>High: {highPct}%</span>
          </div>
        </div>
      </div>

      {/* Key Metric Pillars */}
      <div className="grid grid-cols-3 gap-3 border-t border-slate-800/80 pt-4 text-center">
        <div className="bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/40">
          <span className="block text-[11px] text-slate-400 uppercase font-mono">Centroid</span>
          <span className="text-sm font-semibold font-mono text-cyan-300">
            {data.spectral_centroid_hz.toFixed(1)} Hz
          </span>
        </div>
        <div className="bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/40">
          <span className="block text-[11px] text-slate-400 uppercase font-mono">Reconstruction</span>
          <span className="text-sm font-semibold font-mono text-emerald-300">
            {data.baseline_reconstruction_error.toFixed(5)}
          </span>
        </div>
        <div className="bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/40">
          <span className="block text-[11px] text-slate-400 uppercase font-mono">Signal Quality</span>
          <span className="text-sm font-semibold font-mono text-cyan-400">
            {data.baseline_quality_score.toFixed(1)} / 100
          </span>
        </div>
      </div>
    </div>
  );
}
