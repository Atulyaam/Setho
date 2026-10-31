"use client";

import React, { useEffect, useState } from "react";
import { API_BASE_URL } from "@/lib/config";
import { Activity, CheckCircle2, AlertCircle, RefreshCw } from "lucide-react";

interface HealthState {
  status: "connected" | "connecting" | "error";
  service?: string;
  version?: string;
  environment?: string;
  readyStatus?: string;
  latencyMs?: number;
  lastChecked?: string;
}

export function BackendStatus() {
  const [health, setHealth] = useState<HealthState>({
    status: "connecting",
  });

  const checkConnection = async () => {
    setHealth((prev) => ({ ...prev, status: "connecting" }));
    const startTime = performance.now();
    try {
      const res = await fetch(`${API_BASE_URL}/health`, {
        cache: "no-store",
      });
      const latency = Math.round(performance.now() - startTime);

      if (res.ok) {
        const data = await res.json();
        // Check readiness as well
        let readyStatus = "unknown";
        try {
          const readyRes = await fetch(`${API_BASE_URL}/ready`, {
            cache: "no-store",
          });
          if (readyRes.ok) {
            const readyData = await readyRes.json();
            readyStatus = readyData.status;
          }
        } catch {
          readyStatus = "unreachable";
        }

        setHealth({
          status: "connected",
          service: data.service,
          version: data.version,
          environment: data.environment,
          readyStatus,
          latencyMs: latency,
          lastChecked: new Date().toLocaleTimeString(),
        });
      } else {
        setHealth({
          status: "error",
          latencyMs: latency,
          lastChecked: new Date().toLocaleTimeString(),
        });
      }
    } catch {
      setHealth({
        status: "error",
        lastChecked: new Date().toLocaleTimeString(),
      });
    }
  };

  useEffect(() => {
    checkConnection();
  }, []);

  return (
    <div className="flex items-center gap-3 bg-white/90 backdrop-blur border border-slate-200/80 rounded-xl px-4 py-2.5 shadow-sm text-sm">
      <div className="flex items-center gap-2">
        {health.status === "connecting" && (
          <Activity className="w-4 h-4 text-amber-500 animate-pulse" />
        )}
        {health.status === "connected" && (
          <CheckCircle2 className="w-4 h-4 text-emerald-600" />
        )}
        {health.status === "error" && (
          <AlertCircle className="w-4 h-4 text-rose-500" />
        )}

        <div>
          <div className="font-semibold text-slate-800 flex items-center gap-2">
            Backend API:
            <span
              className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                health.status === "connected"
                  ? "bg-emerald-100 text-emerald-800"
                  : health.status === "connecting"
                  ? "bg-amber-100 text-amber-800"
                  : "bg-rose-100 text-rose-800"
              }`}
            >
              {health.status === "connected"
                ? `Online (${health.readyStatus || "ready"})`
                : health.status === "connecting"
                ? "Connecting..."
                : "Offline / Standby"}
            </span>
          </div>
          <div className="text-xs text-slate-500">
            {health.status === "connected" ? (
              <span>
                {health.service} v{health.version} &bull; {health.latencyMs}ms &bull;{" "}
                {API_BASE_URL}
              </span>
            ) : (
              <span>Target: {API_BASE_URL}</span>
            )}
          </div>
        </div>
      </div>

      <button
        onClick={checkConnection}
        title="Refresh backend status"
        className="ml-auto p-1.5 rounded-lg hover:bg-slate-100 text-slate-500 transition-colors"
      >
        <RefreshCw className={`w-3.5 h-3.5 ${health.status === "connecting" ? "animate-spin" : ""}`} />
      </button>
    </div>
  );
}

