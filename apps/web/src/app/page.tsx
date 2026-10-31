import { APP_CONFIG } from "@/lib/config";
import {
  Activity,
  Layers,
  Cpu,
  Mic,
  FileCheck2,
  AlertTriangle,
  ArrowRight,
  Database,
  Radio,
} from "lucide-react";

export default function Home() {
  return (
    <div className="space-y-8">
      {/* Hero / Identity Section */}
      <section className="bg-gradient-to-br from-slate-900 via-slate-800 to-teal-950 text-white rounded-2xl p-8 sm:p-12 shadow-xl border border-slate-700/50 relative overflow-hidden">
        <div className="absolute right-0 top-0 translate-x-8 -translate-y-8 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="max-w-3xl space-y-4 relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-teal-500/20 text-teal-300 border border-teal-500/30">
            <Radio className="w-3.5 h-3.5 animate-pulse" />
            Kiroween 2026 &bull; Frankenstein Track
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight">
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
        </div>
      </section>

      {/* Sprint 0 Environment Foundation Cards */}
      <section className="space-y-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">
            Sprint 0 — Project Foundation & Subsystem Health
          </h2>
          <p className="text-sm text-slate-500">
            Initial service integration status and architectural baseline.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Card 1: Backend API */}
          <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center">
                <Activity className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-slate-900">Python FastAPI Backend</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                Clean application entry point with modular health (<code className="bg-slate-100 px-1 py-0.5 rounded text-teal-700">/health</code>)
                and readiness (<code className="bg-slate-100 px-1 py-0.5 rounded text-teal-700">/ready</code>) probes, structured correlation
                logging, and exception handling.
              </p>
            </div>
            <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between text-xs">
              <span className="text-slate-500">Architecture</span>
              <span className="font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded-full">
                FastAPI 0.119 / Python 3.11
              </span>
            </div>
          </div>

          {/* Card 2: ML Inference Foundation */}
          <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center">
                <Cpu className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-slate-900">Stetho-MSAE Core</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                Shared multi-scale depthwise-separable CNN autoencoder. Evaluates L1 reconstruction
                error against normal-only calibration sets, avoiding complex, memory-heavy ensembles.
              </p>
            </div>
            <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between text-xs">
              <span className="text-slate-500">Hardware Target</span>
              <span className="font-medium text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded-full">
                RTX 3050 (4 GB) / CPU
              </span>
            </div>
          </div>

          {/* Card 3: Storage & Database */}
          <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm flex flex-col justify-between">
            <div className="space-y-3">
              <div className="w-10 h-10 rounded-lg bg-amber-50 text-amber-700 flex items-center justify-center">
                <Database className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-slate-900">Persistence & Storage</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                PostgreSQL schema boundary for machines, enrollment metadata, jobs, and inspection results,
                with private object storage for raw audio and artifacts.
              </p>
            </div>
            <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between text-xs">
              <span className="text-slate-500">Configuration</span>
              <span className="font-medium text-amber-800 bg-amber-50 px-2 py-0.5 rounded-full">
                Environment-Driven
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Honest Limitations Banner */}
      <section className="bg-slate-50 border border-slate-200 rounded-xl p-5 flex items-start gap-4">
        <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-slate-600 space-y-1">
          <div className="font-semibold text-slate-800">
            Engineering Constraint & Operating Principle
          </div>
          <p>
            Stetho identifies acoustic deviations from a machine-specific normal baseline. It is <strong>not</strong> an
            untested diagnostic system for arbitrary bearing, motor, or blade failure modes. Outputs are categorized
            as <span className="font-semibold text-slate-700">Normal-like</span>, <span className="font-semibold text-slate-700">Unusual</span>, or <span className="font-semibold text-slate-700">Inconclusive</span> based on statistical calibration.
          </p>
        </div>
      </section>

      {/* Roadmap Preview */}
      <section className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-4">
        <h3 className="font-bold text-slate-900">Sprint Roadmap Overview</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="p-3 rounded-lg border border-teal-200 bg-teal-50/50">
            <div className="font-semibold text-teal-900">Sprint 0 (Current)</div>
            <p className="text-teal-700 mt-1">Foundation, FastAPI skeleton, Next.js app, environment & Docker.</p>
          </div>
          <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div className="font-semibold text-slate-900">Sprint 1 (Upcoming)</div>
            <p className="text-slate-600 mt-1">MIMII inspection, audio validation, canonical log-Mel pipeline.</p>
          </div>
          <div className="p-3 rounded-lg border border-slate-200 bg-slate-50">
            <div className="font-semibold text-slate-900">Sprint 2 – 8</div>
            <p className="text-slate-600 mt-1">Baseline detector, Stetho-MSAE autoencoder, calibration, and UI.</p>
          </div>
        </div>
      </section>
    </div>
  );
}

