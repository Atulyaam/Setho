"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { apiClient } from "@/lib/api-client";
import {
  Cpu,
  ArrowLeft,
  Check,
  AlertCircle,
  Loader2,
  Layers,
  Settings,
  HelpCircle,
} from "lucide-react";

export default function NewMachinePage() {
  const router = useRouter();
  const [machineId, setMachineId] = useState("");
  const [name, setName] = useState("");
  const [category, setCategory] = useState("fan");
  const [description, setDescription] = useState("");
  const [operatingCondition, setOperatingCondition] = useState("");
  const [modelPolicy, setModelPolicy] = useState<"stetho_msae" | "dense_autoencoder">("stetho_msae");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const cleanId = machineId.trim().toLowerCase().replace(/[^a-z0-9_-]/g, "_");
    if (!cleanId) {
      setError("Please provide a valid machine identifier (alphanumeric, underscores, hyphens).");
      return;
    }

    if (!name.trim()) {
      setError("Machine name is required.");
      return;
    }

    setIsSubmitting(true);
    try {
      const created = await apiClient.createMachine({
        machine_id: cleanId,
        name: name.trim(),
        category,
        description: description.trim(),
        operating_condition: operatingCondition.trim(),
        selected_model_policy: modelPolicy,
      });

      router.push(`/machines/${created.machine_id}`);
    } catch (err: any) {
      setError(err.message || "Failed to register machine profile.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <ProtectedRoute>
      <div className="max-w-2xl mx-auto space-y-6">
        <div className="flex items-center gap-3">
          <Link
            href="/machines"
            className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              Register Physical Machine
            </h1>
            <p className="text-sm text-slate-500">
              Create an individual machine profile to establish its acoustic baseline reference.
            </p>
          </div>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold">Creation Error</span>
              <p className="text-xs text-rose-700 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-6 sm:p-8 space-y-6">
          <div className="space-y-4">
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider border-b border-slate-100 pb-2">
              1. Physical Machine Identification
            </h2>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                Machine Identifier (Unique ID) *
              </label>
              <input
                type="text"
                required
                value={machineId}
                onChange={(e) => setMachineId(e.target.value)}
                placeholder="e.g. fan_unit_04, condenser_chiller_01"
                className="w-full px-4 py-2.5 text-sm font-mono rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition"
              />
              <p className="text-[11px] text-slate-400 mt-1">
                Unique identifier used for acoustic telemetry and file routing. Alphanumeric, underscores, hyphens only.
              </p>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                Display Name *
              </label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Industrial Intake Ventilation Fan #4"
                className="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                  Machine Category *
                </label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition bg-white"
                >
                  <option value="fan">Industrial Electric Fan</option>
                  <option value="pump">Centrifugal / Water Pump</option>
                  <option value="motor">Electric Drive Motor</option>
                  <option value="valve">Solenoid / Flow Valve</option>
                  <option value="slider">Linear Slide Guide</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                  Model Architecture Policy *
                </label>
                <select
                  value={modelPolicy}
                  onChange={(e) => setModelPolicy(e.target.value as any)}
                  className="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition bg-white"
                >
                  <option value="stetho_msae">Stetho-MSAE (Edge Champion, 560k params)</option>
                  <option value="dense_autoencoder">Dense Autoencoder (Server Baseline, 4.34M params)</option>
                </select>
              </div>
            </div>
          </div>

          <div className="space-y-4 pt-4 border-t border-slate-100">
            <h2 className="text-sm font-bold text-slate-900 uppercase tracking-wider border-b border-slate-100 pb-2">
              2. Operational Parameters & Context
            </h2>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                Baseline Operating Conditions
              </label>
              <input
                type="text"
                value={operatingCondition}
                onChange={(e) => setOperatingCondition(e.target.value)}
                placeholder="e.g. Steady 1,450 RPM, ambient 65 dB, normal load"
                className="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition"
              />
              <p className="text-[11px] text-slate-400 mt-1">
                Document operating conditions during acoustic enrollment so future inspections compare like-for-like.
              </p>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700 mb-1.5">
                Physical Description & Location
              </label>
              <textarea
                rows={3}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="e.g. Building B roof intake ventilation unit. Acoustic probe mounted 5cm from rear bearing housing."
                className="w-full px-4 py-2.5 text-sm rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition"
              />
            </div>
          </div>

          <div className="pt-4 flex items-center justify-end gap-3 border-t border-slate-100">
            <Link
              href="/machines"
              className="px-4 py-2.5 text-xs font-semibold text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition"
            >
              Cancel
            </Link>

            <button
              type="submit"
              disabled={isSubmitting}
              className="px-6 py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white text-xs font-semibold rounded-xl shadow-sm transition flex items-center gap-2"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Registering Asset...</span>
                </>
              ) : (
                <>
                  <Check className="w-4 h-4" />
                  <span>Save Machine Profile</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </ProtectedRoute>
  );
}

