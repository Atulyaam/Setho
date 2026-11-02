/**
 * Frontend environment configuration for Stetho.
 */
export const API_BASE_URL =
  (process.env.NEXT_PUBLIC_API_URL && !process.env.NEXT_PUBLIC_API_URL.includes("localhost"))
    ? process.env.NEXT_PUBLIC_API_URL
    : "http://127.0.0.1:8000";

export const APP_CONFIG = {
  appName: "Stetho",
  tagline: "Machine-Specific Acoustic Anomaly Monitoring & Inspection Assistant",
  version: "0.1.0",
  architecture: "Stetho-MSAE (Shared Multi-Scale Autoencoder)",
  targetHardware: "NVIDIA RTX 3050 Laptop GPU (4GB) with CPU Fallback",
  primaryDemoMachine: "Small Electric Fan",
};

