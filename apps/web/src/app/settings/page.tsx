"use client";

import React, { useEffect, useState } from "react";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { useUI } from "@/context/UIContext";
import { OfflineQueueManager, QueuedInspection } from "@/lib/offline-queue";
import { Settings, Globe, Moon, Sun, RefreshCw, Trash2, CheckCircle2 } from "lucide-react";

export default function SettingsPage() {
  const { theme, toggleTheme, language, setLanguage, t, isOnline } = useUI();
  const [queue, setQueue] = useState<QueuedInspection[]>([]);
  const [syncing, setSyncing] = useState(false);
  const [syncMsg, setSyncMsg] = useState<string | null>(null);

  useEffect(() => {
    return OfflineQueueManager.subscribe((items) => {
      setQueue(items);
    });
  }, []);

  const handleSyncAll = async () => {
    setSyncing(true);
    setSyncMsg(null);
    try {
      const res = await OfflineQueueManager.syncAll();
      setSyncMsg(`Synced ${res.synced} inspection(s) successfully. Failed: ${res.failed}`);
    } catch (e: any) {
      setSyncMsg(e.message || "Sync encountered errors.");
    } finally {
      setSyncing(false);
    }
  };

  const handleClearQueue = () => {
    if (confirm("Are you sure you want to clear the offline inspection queue?")) {
      queue.forEach((q) => OfflineQueueManager.remove(q.id));
    }
  };

  return (
    <ProtectedRoute>
      <div className="space-y-8 max-w-4xl">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-slate-800 border border-slate-700 rounded-xl text-slate-300">
              <Settings className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-100">
                {t("settings")}
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                System localization, display preferences, and offline inspection queue management.
              </p>
            </div>
          </div>
        </div>

        {/* Theme & Display */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100">
          <h2 className="text-sm font-semibold uppercase tracking-wider font-mono text-cyan-400 mb-4">
            {t("themeToggle")}
          </h2>
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-slate-200 block">
                {theme === "dark" ? t("darkTheme") : t("lightTheme")}
              </span>
              <span className="text-xs text-slate-400">
                Optimized high-contrast visual tokens for industrial inspection environments.
              </span>
            </div>
            <button
              onClick={toggleTheme}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-medium flex items-center gap-2 transition"
            >
              {theme === "dark" ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-cyan-400" />}
              <span>Toggle Theme</span>
            </button>
          </div>
        </div>

        {/* Language & Localization */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100">
          <h2 className="text-sm font-semibold uppercase tracking-wider font-mono text-cyan-400 mb-4">
            {t("languageSelect")}
          </h2>
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-slate-200 block">
                Active Language: {language === "en" ? "English" : "हिन्दी (Hindi)"}
              </span>
              <span className="text-xs text-slate-400">
                Dual-language interface for multinational field maintenance teams.
              </span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setLanguage("en")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium font-mono transition ${
                  language === "en"
                    ? "bg-cyan-600 text-white"
                    : "bg-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                English
              </button>
              <button
                onClick={() => setLanguage("hi")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium font-mono transition ${
                  language === "hi"
                    ? "bg-cyan-600 text-white"
                    : "bg-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                हिन्दी
              </button>
            </div>
          </div>
        </div>

        {/* Offline Inspection Queue (Feature 7) */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-wider font-mono text-cyan-400">
                {t("offlineQueueTitle")}
              </h2>
              <span className="text-xs text-slate-400">
                Status: {isOnline ? "Online (Ready to sync)" : "Offline (Local queuing active)"}
              </span>
            </div>
            <span className="px-2.5 py-1 bg-slate-950 border border-slate-800 rounded-full text-xs font-mono text-slate-300">
              {queue.length} Pending
            </span>
          </div>

          {syncMsg && (
            <div className="mb-4 p-3 bg-cyan-950/40 border border-cyan-800 text-cyan-300 text-xs rounded-lg flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-cyan-400" />
              <span>{syncMsg}</span>
            </div>
          )}

          {queue.length > 0 ? (
            <div className="space-y-3">
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {queue.map((item) => (
                  <div
                    key={item.id}
                    className="p-3 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-between text-xs"
                  >
                    <div>
                      <span className="font-semibold text-slate-200 block">
                        {item.machineName || item.machineId}
                      </span>
                      <span className="text-slate-500 font-mono text-[10px]">
                        {new Date(item.recordedAt).toLocaleString()} • {item.fileName}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                          item.status === "synced"
                            ? "bg-emerald-500/20 text-emerald-400"
                            : item.status === "syncing"
                            ? "bg-cyan-500/20 text-cyan-400 animate-pulse"
                            : item.status === "failed"
                            ? "bg-red-500/20 text-red-400"
                            : "bg-amber-500/20 text-amber-400"
                        }`}
                      >
                        {item.status}
                      </span>
                      <button
                        onClick={() => OfflineQueueManager.remove(item.id)}
                        className="text-slate-500 hover:text-red-400 p-1"
                        title="Remove"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex gap-3 pt-3 border-t border-slate-800">
                <button
                  onClick={handleSyncAll}
                  disabled={syncing || !isOnline}
                  className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center gap-2 transition"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${syncing ? "animate-spin" : ""}`} />
                  <span>{t("syncNow")}</span>
                </button>
                <button
                  onClick={handleClearQueue}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium transition"
                >
                  Clear Queue
                </button>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500 italic">
              No offline inspections pending synchronization.
            </p>
          )}
        </div>
      </div>
    </ProtectedRoute>
  );
}

