import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/context/AuthContext";
import { Navbar } from "@/components/Navbar";
import { APP_CONFIG } from "@/lib/config";
import { Cpu, ShieldCheck } from "lucide-react";

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
        <AuthProvider>
          <div className="min-h-screen flex flex-col justify-between bg-slate-50/50">
            <Navbar />

            <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
              {children}
            </main>

            <footer className="border-t border-slate-200 bg-white/70 py-6 text-xs text-slate-500">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-slate-400" />
                  <span>Hardware Target: {APP_CONFIG.targetHardware}</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-400">
                  <ShieldCheck className="w-4 h-4" />
                  <span>Identifies acoustic baseline deviations — not an unverified fault classifier</span>
                </div>
              </div>
            </footer>
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}

