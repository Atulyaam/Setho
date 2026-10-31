"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { useUI } from "@/context/UIContext";
import { BackendStatus } from "@/components/BackendStatus";
import { APP_CONFIG } from "@/lib/config";
import {
  Stethoscope,
  LayoutDashboard,
  Cpu,
  Mic,
  Activity,
  Settings as SettingsIcon,
  PlusCircle,
  LogOut,
  Sun,
  Moon,
  LogIn,
  UserPlus,
} from "lucide-react";

export function Navbar() {
  const { user, isAuthenticated, logout } = useAuth();
  const { t, language, setLanguage, theme, toggleTheme } = useUI();
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path || pathname.startsWith(path + "/");

  return (
    <header className="sticky top-0 z-40 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 text-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href={isAuthenticated ? "/dashboard" : "/"} className="flex items-center gap-3 group">
            <div className="p-2 bg-cyan-600 group-hover:bg-cyan-500 text-white rounded-xl shadow-sm transition">
              <Stethoscope className="w-5 h-5" />
            </div>
            <div>
              <div className="font-bold text-lg text-slate-100 tracking-tight flex items-center gap-2">
                {APP_CONFIG.appName}
                <span className="text-[10px] font-semibold uppercase px-2 py-0.5 bg-cyan-500/10 text-cyan-400 rounded-full border border-cyan-500/20">
                  Sprint 6A
                </span>
              </div>
              <div className="text-[11px] text-slate-400 hidden sm:block">
                Industrial Acoustic Intelligence Platform
              </div>
            </div>
          </Link>

          {isAuthenticated && (
            <nav className="hidden lg:flex items-center gap-1 ml-4 text-xs font-medium">
              <Link
                href="/dashboard"
                className={`px-3 py-2 rounded-lg flex items-center gap-1.5 transition ${
                  isActive("/dashboard")
                    ? "bg-slate-800 text-cyan-400 font-semibold"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50"
                }`}
              >
                <LayoutDashboard className="w-4 h-4" />
                <span>{t("commandCenter")}</span>
              </Link>

              <Link
                href="/machines"
                className={`px-3 py-2 rounded-lg flex items-center gap-1.5 transition ${
                  isActive("/machines") && !isActive("/machines/new")
                    ? "bg-slate-800 text-cyan-400 font-semibold"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50"
                }`}
              >
                <Cpu className="w-4 h-4" />
                <span>{t("machines")}</span>
              </Link>

              <Link
                href="/studio"
                className={`px-3 py-2 rounded-lg flex items-center gap-1.5 transition ${
                  isActive("/studio")
                    ? "bg-slate-800 text-cyan-400 font-semibold"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50"
                }`}
              >
                <Mic className="w-4 h-4" />
                <span>{t("acousticStudio")}</span>
              </Link>

              <Link
                href="/monitoring"
                className={`px-3 py-2 rounded-lg flex items-center gap-1.5 transition ${
                  isActive("/monitoring")
                    ? "bg-slate-800 text-cyan-400 font-semibold"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50"
                }`}
              >
                <Activity className="w-4 h-4" />
                <span>{t("monitoring")}</span>
              </Link>

              <Link
                href="/settings"
                className={`px-3 py-2 rounded-lg flex items-center gap-1.5 transition ${
                  isActive("/settings")
                    ? "bg-slate-800 text-cyan-400 font-semibold"
                    : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50"
                }`}
              >
                <SettingsIcon className="w-4 h-4" />
                <span>{t("settings")}</span>
              </Link>
            </nav>
          )}
        </div>

        <div className="flex items-center gap-2 sm:gap-3">
          {/* Language Toggle */}
          <button
            onClick={() => setLanguage(language === "en" ? "hi" : "en")}
            title="Toggle Language (English / हिन्दी)"
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg text-xs font-mono font-bold transition"
          >
            {language === "en" ? "HI" : "EN"}
          </button>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            title="Toggle Dark / Light Theme"
            className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg text-xs transition"
          >
            {theme === "dark" ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-cyan-400" />}
          </button>

          <BackendStatus />

          {isAuthenticated && user ? (
            <div className="flex items-center gap-2 pl-2 border-l border-slate-800">
              <div className="hidden sm:flex flex-col text-right">
                <span className="text-xs font-semibold text-slate-200 leading-tight">
                  {user.username}
                </span>
                <span className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">
                  {user.role}
                </span>
              </div>
              <button
                onClick={() => logout()}
                title="Sign Out"
                className="p-1.5 text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Link
                href="/login"
                className="px-3 py-1.5 text-xs font-semibold text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition flex items-center gap-1.5"
              >
                <LogIn className="w-3.5 h-3.5" />
                <span>Sign In</span>
              </Link>
              <Link
                href="/register"
                className="px-3 py-1.5 text-xs font-semibold text-white bg-cyan-600 hover:bg-cyan-500 rounded-lg shadow-sm transition flex items-center gap-1.5"
              >
                <UserPlus className="w-3.5 h-3.5" />
                <span>Register</span>
              </Link>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

