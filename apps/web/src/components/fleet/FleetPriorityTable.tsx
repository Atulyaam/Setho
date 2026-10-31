"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useUI } from "../../context/UIContext";

interface PriorityMachine {
  machine_id: string;
  name: string;
  category: string;
  calibration_status: string;
  priority_score: number;
  priority_tier: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  reasons: string[];
  recent_inspections_count: number;
  unusual_count: number;
  normal_count: number;
  latest_inspection_at: string | null;
  latest_decision: string;
}

export function FleetPriorityTable() {
  const { t } = useUI();
  const [machines, setMachines] = useState<PriorityMachine[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPriority() {
      try {
        setLoading(true);
        const res = await fetch("/api/backend/machines/fleet-priority");
        if (!res.ok) throw new Error("Failed to load fleet priorities.");
        const json = await res.json();
        setMachines(json.priority_ranking || []);
      } catch (err: any) {
        setError(err.message || "Failed to fetch priority ranking.");
      } finally {
        setLoading(false);
      }
    }
    fetchPriority();
  }, []);

  if (loading) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 animate-pulse">
        <div className="h-5 bg-slate-800 rounded w-1/4 mb-4"></div>
        <div className="space-y-3">
          <div className="h-12 bg-slate-800/60 rounded"></div>
          <div className="h-12 bg-slate-800/40 rounded"></div>
          <div className="h-12 bg-slate-800/20 rounded"></div>
        </div>
      </div>
    );
  }

  if (error || machines.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-400 text-sm">
        <h3 className="text-base font-semibold text-slate-200 mb-2">{t("fleetPriorityTitle")}</h3>
        <p className="text-xs text-slate-500">
          No assets registered or active in priority queue. Create machine profiles to establish inspection priorities.
        </p>
      </div>
    );
  }

  const tierBadges = {
    CRITICAL: "bg-red-500/20 text-red-300 border-red-500/40",
    HIGH: "bg-amber-500/20 text-amber-300 border-amber-500/40",
    MEDIUM: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
    LOW: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100 shadow-xl">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-base font-semibold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            {t("fleetPriorityTitle")}
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Transparent multi-factor ranking based on acoustic drift, quality flags, and inspection age.
          </p>
        </div>
        <span className="text-xs font-mono text-slate-400">
          {machines.length} Assets Ranked
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 uppercase font-mono text-[10px]">
              <th className="py-2.5 px-3">Asset</th>
              <th className="py-2.5 px-3">Priority Tier</th>
              <th className="py-2.5 px-3">Urgency Score</th>
              <th className="py-2.5 px-3">Trigger Rationale</th>
              <th className="py-2.5 px-3">Last Inspected</th>
              <th className="py-2.5 px-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {machines.map((m) => (
              <tr key={m.machine_id} className="hover:bg-slate-800/40 transition">
                <td className="py-3 px-3 font-medium text-slate-200">
                  <Link href={`/machines/${m.machine_id}`} className="hover:text-cyan-400 transition font-sans">
                    {m.name}
                  </Link>
                  <span className="block text-[11px] text-slate-500 font-mono">
                    ID: {m.machine_id} ({m.category})
                  </span>
                </td>
                <td className="py-3 px-3">
                  <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${tierBadges[m.priority_tier]}`}>
                    {m.priority_tier}
                  </span>
                </td>
                <td className="py-3 px-3">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className={`h-full ${
                          m.priority_tier === "CRITICAL"
                            ? "bg-red-500"
                            : m.priority_tier === "HIGH"
                            ? "bg-amber-400"
                            : m.priority_tier === "MEDIUM"
                            ? "bg-yellow-400"
                            : "bg-emerald-400"
                        }`}
                        style={{ width: `${m.priority_score}%` }}
                      ></div>
                    </div>
                    <span className="font-semibold text-slate-300">{m.priority_score}</span>
                  </div>
                </td>
                <td className="py-3 px-3 font-sans text-slate-300 text-[11px] max-w-xs">
                  {m.reasons.slice(0, 2).map((r, i) => (
                    <span key={i} className="block">• {r}</span>
                  ))}
                </td>
                <td className="py-3 px-3 text-slate-400 text-[11px]">
                  {m.latest_inspection_at
                    ? new Date(m.latest_inspection_at).toLocaleDateString()
                    : "Never"}
                </td>
                <td className="py-3 px-3 text-right">
                  <Link
                    href={`/machines/${m.machine_id}/analyze`}
                    className="px-3 py-1 bg-cyan-600/30 hover:bg-cyan-600/50 text-cyan-300 border border-cyan-500/40 rounded text-[11px] transition"
                  >
                    Inspect →
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
