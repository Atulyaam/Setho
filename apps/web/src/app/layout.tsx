import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { BackendStatus } from "@/components/BackendStatus";
import { APP_CONFIG } from "@/lib/config";
import { Stethoscope, Cpu, ShieldCheck } from "lucide-react";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: `${APP_CONFIG.appName} — Machine Acoustic Anomaly Detection`,
  description: APP_CONFIG.tagline,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen flex flex-col justify-between">
          <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-slate-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-brand-600 text-white rounded-xl shadow-sm">
                  <Stethoscope className="w-5 h-5" />
                </div>
                <div>
                  <div className="font-bold text-lg text-slate-900 tracking-tight flex items-center gap-2">
                    {APP_CONFIG.appName}
                    <span className="text-[10px] font-semibold uppercase px-2 py-0.5 bg-slate-100 text-slate-600 rounded">
                      Sprint 0
                    </span>
                  </div>
                  <div className="text-xs text-slate-500 hidden sm:block">
                    Acoustic Anomaly Inspection Assistant
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <BackendStatus />
              </div>
            </div>
          </header>

          <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </main>

          <footer className="border-t border-slate-200 bg-white/70 py-6 text-xs text-slate-500">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-slate-400" />
                <span>
                  Hardware: {APP_CONFIG.targetHardware}
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-400">
                <ShieldCheck className="w-4 h-4" />
                <span>Identifies acoustic baseline deviations — not an unverified fault classifier</span>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}

