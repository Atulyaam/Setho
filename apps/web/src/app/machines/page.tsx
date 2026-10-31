"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { apiClient, MachineProfile } from "@/lib/api-client";
import {
  Cpu,
  PlusCircle,
  Search,
  Filter,
  Mic,
  Activity,
  ChevronRight,
  Loader2,
  AlertTriangle,
} from "lucide-react";

export default function MachinesListPage() {
  const [machines, setMachines] = useState<MachineProfile[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadMachines() {
      setIsLoading(true);
      setError(null);
      try {
        const res = await apiClient.getMachines();
        setMachines(res.machines);
      } catch (err: any) {
        setError(err.message || "Failed to retrieve machine fleet.");
      } finally {
        setIsLoading(false);
      }
    }
    loadMachines();
  }, []);

  const filteredMachines = machines.filter((m) => {
    const matchesSearch =
      m.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.machine_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      m.category.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus =
      statusFilter === "all" || m.calibration_status === statusFilter;

    return matchesSearch && matchesStatus;
  });

  return (
    <ProtectedRoute>
      <div className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Machine Fleet</h1>
            <p className="text-sm text-slate-500 mt-1">
              Registered physical machinery under acoustic baseline monitoring and inspection.
            </p>
          </div>

          <Link
            href="/machines/new"
            className="px-4 py-2.5 bg-brand-600 hover:bg-brand-700 text-white rounded-xl shadow-sm text-sm font-semibold transition flex items-center gap-2 self-start sm:self-auto"
          >
            <PlusCircle className="w-4 h-4" />
            <span>Register New Machine</span>
          </Link>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-500 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Filter and Search Bar */}
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search by name, ID, or category..."
              className="w-full pl-10 pr-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition"
            />
          </div>

          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-400 hidden sm:block" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-white border border-slate-200 rounded-xl px-3 py-2.5 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-600 transition"
            >
              <option value="all">All Calibration States</option>
              <option value="calibrated">Calibrated (N ≥ 30)</option>
              <option value="provisional">Provisional (N &lt; 30)</option>
              <option value="uncalibrated">Uncalibrated (N = 0)</option>
            </select>
          </div>
        </div>

        {/* Machine Cards Grid */}
        {isLoading ? (
          <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center flex flex-col items-center justify-center space-y-3">
            <Loader2 className="w-6 h-6 text-brand-600 animate-spin" />
            <p className="text-sm text-slate-500">Loading registered machinery...</p>
          </div>
        ) : filteredMachines.length === 0 ? (
          <div className="bg-white rounded-2xl border border-dashed border-slate-300 p-12 text-center space-y-4 max-w-xl mx-auto">
            <div className="w-12 h-12 bg-slate-50 text-slate-500 rounded-2xl flex items-center justify-center mx-auto">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">No Machines Matching Filter</h3>
              <p className="text-xs text-slate-500 mt-1">
                {searchTerm || statusFilter !== "all"
                  ? "Try resetting your search query or status filter."
                  : "No machines registered to your inspector account."}
              </p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {filteredMachines.map((machine) => (
              <div
                key={machine.machine_id}
                className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 flex flex-col justify-between hover:border-brand-500/50 hover:shadow-md transition"
              >
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                        {machine.machine_id}
                      </span>
                      <h3 className="text-base font-bold text-slate-900 mt-1 line-clamp-1">
                        {machine.name}
                      </h3>
                    </div>
                    <span
                      className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full ${
                        machine.calibration_status === "calibrated"
                          ? "bg-emerald-100 text-emerald-800"
                          : machine.calibration_status === "provisional"
                          ? "bg-amber-100 text-amber-800"
                          : "bg-slate-100 text-slate-600"
                      }`}
                    >
                      {machine.calibration_status}
                    </span>
                  </div>

                  <p className="text-xs text-slate-600 line-clamp-2">
                    {machine.description || "Industrial operating asset with continuous acoustic monitoring."}
                  </p>

                  <div className="pt-2 border-t border-slate-100 space-y-1.5 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Category:</span>
                      <span className="font-semibold text-slate-700 capitalize">{machine.category}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Enrolled Normal Audio:</span>
                      <span className="font-semibold text-slate-800">{machine.enrolled_recording_count} samples</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Model Policy:</span>
                      <span className="font-mono text-brand-700 font-semibold">{machine.selected_model_policy}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-5 pt-4 border-t border-slate-100 flex items-center justify-between gap-2">
                  <Link
                    href={`/machines/${machine.machine_id}/enroll`}
                    className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-teal-50 hover:bg-teal-100 text-brand-700 transition flex items-center gap-1.5"
                  >
                    <Mic className="w-3.5 h-3.5" />
                    <span>Enroll Normal</span>
                  </Link>

                  <Link
                    href={`/machines/${machine.machine_id}/analyze`}
                    className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-brand-600 hover:bg-brand-700 text-white transition flex items-center gap-1.5 shadow-sm"
                  >
                    <Activity className="w-3.5 h-3.5" />
                    <span>Analyze</span>
                  </Link>

                  <Link
                    href={`/machines/${machine.machine_id}`}
                    className="px-2.5 py-1.5 text-xs font-medium text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition"
                    title="View Details"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </ProtectedRoute>
  );
}

