"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/context/AuthContext";
import { BackendStatus } from "@/components/BackendStatus";
import { APP_CONFIG } from "@/lib/config";
import {
  Stethoscope,
  LayoutDashboard,
  Cpu,
  PlusCircle,
  LogOut,
  User,
  LogIn,
  UserPlus,
} from "lucide-react";

export function Navbar() {
  const { user, isAuthenticated, logout } = useAuth();
  const pathname = usePathname();

  const isActive = (path: string) => pathname === path || pathname.startsWith(path + "/");

  return (
    <header className="sticky top-0 z-40 bg-white/90 backdrop-blur-md border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href={isAuthenticated ? "/dashboard" : "/"} className="flex items-center gap-3 group">
            <div className="p-2 bg-brand-600 group-hover:bg-brand-700 text-white rounded-xl shadow-sm transition">
              <Stethoscope className="w-5 h-5" />
            </div>
            <div>
              <div className="font-bold text-lg text-slate-900 tracking-tight flex items-center gap-2">
                {APP_CONFIG.appName}
                <span className="text-[10px] font-semibold uppercase px-2 py-0.5 bg-brand-50 text-brand-700 rounded-full border border-brand-200">
                  Production
                </span>
              </div>
              <div className="text-xs text-slate-500 hidden sm:block">
                Acoustic Anomaly Inspection Assistant
              </div>
            </div>
          </Link>

          {isAuthenticated && (
            <nav className="hidden md:flex items-center gap-1 ml-4 text-sm font-medium">
              <Link
                href="/dashboard"
                className={`px-3 py-2 rounded-lg flex items-center gap-2 transition ${
                  isActive("/dashboard")
                    ? "bg-slate-100 text-brand-700 font-semibold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                }`}
              >
                <LayoutDashboard className="w-4 h-4" />
                <span>Dashboard</span>
              </Link>

              <Link
                href="/machines"
                className={`px-3 py-2 rounded-lg flex items-center gap-2 transition ${
                  isActive("/machines") && !isActive("/machines/new")
                    ? "bg-slate-100 text-brand-700 font-semibold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                }`}
              >
                <Cpu className="w-4 h-4" />
                <span>Machines</span>
              </Link>

              <Link
                href="/machines/new"
                className={`px-3 py-2 rounded-lg flex items-center gap-2 transition ${
                  isActive("/machines/new")
                    ? "bg-slate-100 text-brand-700 font-semibold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                }`}
              >
                <PlusCircle className="w-4 h-4" />
                <span>Register Machine</span>
              </Link>
            </nav>
          )}
        </div>

        <div className="flex items-center gap-3">
          <BackendStatus />

          {isAuthenticated && user ? (
            <div className="flex items-center gap-2 pl-2 border-l border-slate-200">
              <div className="hidden sm:flex flex-col text-right">
                <span className="text-xs font-semibold text-slate-800 leading-tight">
                  {user.username}
                </span>
                <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                  {user.role}
                </span>
              </div>
              <button
                onClick={() => logout()}
                title="Sign Out"
                className="p-2 text-slate-500 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Link
                href="/login"
                className="px-3 py-1.5 text-xs font-semibold text-slate-700 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition flex items-center gap-1.5"
              >
                <LogIn className="w-3.5 h-3.5" />
                <span>Sign In</span>
              </Link>
              <Link
                href="/register"
                className="px-3 py-1.5 text-xs font-semibold text-white bg-brand-600 hover:bg-brand-700 rounded-lg shadow-sm transition flex items-center gap-1.5"
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

