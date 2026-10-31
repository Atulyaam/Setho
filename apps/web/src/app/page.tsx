"use client";

import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import { APP_CONFIG } from "@/lib/config";
import {
  Activity,
  Layers,
  Cpu,
  Mic,
  ArrowRight,
  Radio,
  ShieldCheck,
  CheckCircle2,
  Lock,
} from "lucide-react";

export default function Home() {
  const { isAuthenticated, user } = useAuth();

  return (
    <div className="space-y-8">
      {/* Hero / Identity Section */}
      <section className="bg-gradient-to-br from-slate-900 via-slate-800 to-teal-950 text-white rounded-3xl p-8 sm:p-12 shadow-xl border border-slate-700/50 relative overflow-hidden">
        <div className="absolute right-0 top-0 translate-x-8 -translate-y-8 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="max-w-3xl space-y-5 relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-teal-500/20 text-teal-300 border border-teal-500/30">
            <Radio className="w-3.5 h-3.5 animate-pulse" />
            Kiroween 2026 &bull; Frankenstein Track
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight leading-tight">
            Machine-Specific Acoustic Anomaly Monitoring
          </h1>

          <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
            Stetho learns the normal acoustic signature of a specific physical machine
            under recorded operating conditions. It compares subsequent audio recordings
            against that baseline and provides evidence-backed anomaly indicators.
          </p>

          <div className="pt-2 flex flex-wrap gap-4 text-xs font-medium text-slate-300">
            <div className="flex items-center gap-1.5 bg-white/10 px-3 py-1.5 rounded-lg backdrop-blur">
              <Layers className="w-4 h-4 text-teal-400" />
              <span>{APP_CONFIG.architecture}</span>
            </div>
            <div className="flex items-center gap-1.5 bg-white/10 px-3 py-1.5 rounded-lg backdrop-blur">
              <Mic className="w-4 h-4 text-teal-400" />
              <span>Primary Demo: {APP_CONFIG.primaryDemoMachine}</span>
            </div>
            <div className="flex items-center gap-1.5 bg-white/10 px-3 py-1.5 rounded-lg backdrop-blur">
              <Cpu className="w-4 h-4 text-teal-400" />
              <span>CPU Fallback Supported</span>
            </div>
          </div>

          <div className="pt-4 flex flex-wrap items-center gap-3">
            {isAuthenticated ? (
              <Link
                href="/dashboard"
                className="px-6 py-3 bg-brand-500 hover:bg-brand-400 text-slate-950 font-bold rounded-xl shadow-lg transition flex items-center gap-2 text-sm"
              >
                <span>Enter Operator Dashboard</span>
                <ArrowRight className="w-4 h-4" />
              </Link>
            ) : (
              <>
                <Link
                  href="/login"
                  className="px-6 py-3 bg-brand-500 hover:bg-brand-400 text-slate-950 font-bold rounded-xl shadow-lg transition flex items-center gap-2 text-sm"
                >
                  <span>Inspector Sign In</span>
                  <ArrowRight className="w-4 h-4" />
                </Link>
                <Link
                  href="/register"
                  className="px-6 py-3 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-xl border border-white/20 backdrop-blur transition text-sm"
                >
                  Register New Account
                </Link>
              </>
            )}
          </div>
        </div>
      </section>

      {/* Core Principles Grid */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-xl bg-teal-50 text-brand-600 flex items-center justify-center">
            <Mic className="w-5 h-5" />
          </div>
          <h2 className="font-bold text-slate-900 text-base">Normal-Only Calibration</h2>
          <p className="text-xs text-slate-600 leading-relaxed">
            Learns exclusively from confirmed normal audio recordings of an individual asset. Establishes empirical score distributions without needing rare anomaly training examples.
          </p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
            <Layers className="w-5 h-5" />
          </div>
          <h2 className="font-bold text-slate-900 text-base">Three-Zone Decision Scale</h2>
          <p className="text-xs text-slate-600 leading-relaxed">
            Acoustic recordings are mapped to <strong>NORMAL_LIKE</strong>, <strong>UNUSUAL</strong>, or <strong>INCONCLUSIVE</strong> zones with explicit statistical thresholds (θ_lower, θ_upper).
          </p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h2 className="font-bold text-slate-900 text-base">Evidence-Based Integrity</h2>
          <p className="text-xs text-slate-600 leading-relaxed">
            Zero hallucinations. Anomaly scores are never presented as failure probabilities, and specific mechanical fault labels are never generated without certified physical proof.
          </p>
        </div>
      </section>
    </div>
  );
}
