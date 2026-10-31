"use client";

import React, { useEffect, useState } from "react";
import { useUI } from "../../context/UIContext";

interface QRCodeModalProps {
  machineId: string;
  machineName: string;
  isOpen: boolean;
  onClose: () => void;
}

export function QRCodeModal({ machineId, machineName, isOpen, onClose }: QRCodeModalProps) {
  const { t } = useUI();
  const [qrToken, setQrToken] = useState<string | null>(null);
  const [inspectionUrl, setInspectionUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [rotating, setRotating] = useState(false);

  useEffect(() => {
    async function loadQRCode() {
      try {
        setLoading(true);
        const res = await fetch(`/api/backend/machines/${machineId}/qr-code`);
        if (!res.ok) throw new Error("Failed to load QR token");
        const data = await res.json();
        setQrToken(data.qr_token);
        setInspectionUrl(window.location.origin + data.inspection_url);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }

    if (isOpen) {
      loadQRCode();
    }
  }, [isOpen, machineId]);

  const rotateToken = async () => {
    try {
      setRotating(true);
      const res = await fetch(`/api/backend/machines/${machineId}/qr-code/rotate`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed to rotate token");
      const data = await res.json();
      setQrToken(data.qr_token);
      setInspectionUrl(window.location.origin + data.inspection_url);
    } catch (e) {
      console.error(e);
    } finally {
      setRotating(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-md w-full text-slate-100 shadow-2xl">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div>
            <h3 className="text-lg font-semibold">{t("qrIdentity")}</h3>
            <p className="text-xs text-slate-400 font-mono">{machineName} ({machineId})</p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white text-lg p-1 rounded-lg"
          >
            ✕
          </button>
        </div>

        <div className="py-6 text-center">
          {loading ? (
            <div className="w-48 h-48 mx-auto bg-slate-950 border border-slate-800 rounded-xl flex items-center justify-center animate-pulse">
              <span className="text-xs text-slate-500 font-mono">Generating Tag...</span>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Asset Identity Badge Representation */}
              <div className="w-56 h-56 mx-auto bg-white p-4 rounded-xl shadow-inner flex flex-col items-center justify-center">
                {/* SVG QR Code Pattern */}
                <svg
                  className="w-44 h-44 text-slate-900"
                  viewBox="0 0 100 100"
                  fill="currentColor"
                >
                  {/* Outer corner finders */}
                  <rect x="5" y="5" width="25" height="25" fill="none" stroke="currentColor" strokeWidth="6" />
                  <rect x="11" y="11" width="13" height="13" />
                  <rect x="70" y="5" width="25" height="25" fill="none" stroke="currentColor" strokeWidth="6" />
                  <rect x="76" y="11" width="13" height="13" />
                  <rect x="5" y="70" width="25" height="25" fill="none" stroke="currentColor" strokeWidth="6" />
                  <rect x="11" y="76" width="13" height="13" />
                  {/* Data patterns */}
                  <rect x="35" y="10" width="8" height="8" />
                  <rect x="48" y="15" width="12" height="6" />
                  <rect x="35" y="25" width="15" height="8" />
                  <rect x="55" y="25" width="8" height="15" />
                  <rect x="10" y="35" width="15" height="8" />
                  <rect x="28" y="40" width="10" height="10" />
                  <rect x="45" y="42" width="15" height="12" />
                  <rect x="68" y="38" width="18" height="8" />
                  <rect x="10" y="48" width="8" height="15" />
                  <rect x="75" y="52" width="12" height="12" />
                  <rect x="35" y="60" width="12" height="8" />
                  <rect x="52" y="60" width="14" height="14" />
                  <rect x="35" y="75" width="18" height="8" />
                  <rect x="70" y="72" width="15" height="15" />
                </svg>
                <span className="text-[10px] font-mono font-bold text-slate-900 mt-1 uppercase tracking-wider">
                  STETHO • {machineId}
                </span>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-400 break-all text-left">
                <span className="text-slate-500 block mb-0.5">Scannable Field Inspection URL:</span>
                <span className="text-cyan-400">{inspectionUrl}</span>
              </div>
            </div>
          )}
        </div>

        <div className="flex gap-2 pt-2 border-t border-slate-800">
          <button
            onClick={rotateToken}
            disabled={rotating}
            className="flex-1 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-mono transition"
          >
            {rotating ? "Rotating..." : "🔄 Revoke & Rotate Token"}
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-medium transition"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
