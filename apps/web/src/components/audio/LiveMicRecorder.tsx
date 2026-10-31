"use client";

import React, { useEffect, useRef, useState } from "react";
import { useUI } from "../../context/UIContext";
import { OfflineQueueManager } from "../../lib/offline-queue";

interface LiveMicRecorderProps {
  machineId?: string;
  machineName?: string;
  onRecordingComplete: (file: File, blobUrl: string) => void;
  targetDurationSec?: number;
}

export function LiveMicRecorder({
  machineId,
  machineName,
  onRecordingComplete,
  targetDurationSec = 5,
}: LiveMicRecorderProps) {
  const { t, isOnline } = useUI();
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState(targetDurationSec);
  const [volumeRms, setVolumeRms] = useState(0);
  const [rmsStatus, setRmsStatus] = useState<"optimal" | "low" | "clipping">("optimal");
  const [permissionError, setPermissionError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const animationFrameRef = useRef<number | null>(null);
  const timerRef = useRef<any>(null);

  useEffect(() => {
    return () => {
      stopCleanup();
    };
  }, []);

  const stopCleanup = () => {
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(() => {});
    }
  };

  const startLiveRecording = async () => {
    setPermissionError(null);
    audioChunksRef.current = [];
    setCountdown(targetDurationSec);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      mediaStreamRef.current = stream;
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Monitor RMS Volume
      const pcmData = new Float32Array(analyser.fftSize);
      const monitorVolume = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getFloatTimeDomainData(pcmData);
        let sumSquares = 0;
        let peak = 0;
        for (let i = 0; i < pcmData.length; i++) {
          const val = pcmData[i];
          sumSquares += val * val;
          if (Math.abs(val) > peak) peak = Math.abs(val);
        }
        const rms = Math.sqrt(sumSquares / pcmData.length);
        const rmsDb = rms > 0 ? 20 * Math.log10(rms) : -100;
        setVolumeRms(Math.max(0, Math.min(100, (rmsDb + 60) * 1.66)));

        if (peak >= 0.98) {
          setRmsStatus("clipping");
        } else if (rmsDb < -45) {
          setRmsStatus("low");
        } else {
          setRmsStatus("optimal");
        }

        animationFrameRef.current = requestAnimationFrame(monitorVolume);
      };
      monitorVolume();

      // MediaRecorder with audio/webm or audio/wav
      let mimeType = "audio/webm";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "";
      }
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        stopCleanup();
        setIsRecording(false);
        const recordedBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        const fileName = `live_mic_${Date.now()}.wav`;
        const file = new File([recordedBlob], fileName, { type: "audio/wav" });
        const blobUrl = URL.createObjectURL(recordedBlob);

        // If offline and machine specified, enqueue
        if (!isOnline && machineId) {
          await OfflineQueueManager.enqueue(machineId, machineName || machineId, file);
        }

        onRecordingComplete(file, blobUrl);
      };

      recorder.start(100);
      setIsRecording(true);

      // Countdown Timer
      let currentSec = targetDurationSec;
      timerRef.current = setInterval(() => {
        currentSec -= 1;
        setCountdown(currentSec);
        if (currentSec <= 0) {
          if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
            mediaRecorderRef.current.stop();
          }
        }
      }, 1000);
    } catch (err: any) {
      console.error("Microphone access error:", err);
      setPermissionError(
        "Microphone access blocked or unavailable. Please enable browser audio permissions."
      );
      stopCleanup();
      setIsRecording(false);
    }
  };

  const stopLiveRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-slate-100">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            {t("recordLiveAudio")}
          </h3>
          <p className="text-xs text-slate-400 mt-1">{t("distanceGuidance")}</p>
        </div>
        {!isOnline && (
          <span className="px-2.5 py-1 bg-amber-500/20 text-amber-300 border border-amber-500/30 text-xs rounded-full font-medium">
            {t("offlineMode")}
          </span>
        )}
      </div>

      {permissionError && (
        <div className="mb-4 p-3 bg-red-950/40 border border-red-800 text-red-300 text-xs rounded-lg">
          {permissionError}
        </div>
      )}

      {/* Visual Volume Meter and Guidance */}
      {isRecording ? (
        <div className="space-y-4 my-6">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-cyan-300">
              {t("recordingNotice")}
            </span>
            <span className="text-2xl font-bold font-mono text-cyan-400">{countdown}s</span>
          </div>

          {/* Real-time Level Bar */}
          <div>
            <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800 flex">
              <div
                className={`h-full transition-all duration-75 ${
                  rmsStatus === "clipping"
                    ? "bg-red-500"
                    : rmsStatus === "low"
                    ? "bg-amber-400"
                    : "bg-emerald-400"
                }`}
                style={{ width: `${volumeRms}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
              <span>-60 dBFS (Quiet)</span>
              <span>-24 dBFS (Optimal)</span>
              <span className={rmsStatus === "clipping" ? "text-red-400 font-bold" : ""}>
                0 dBFS (Clipping)
              </span>
            </div>
          </div>

          {rmsStatus === "clipping" && (
            <p className="text-xs text-red-400">
              ⚠️ Warning: Signal clipping. Increase distance from machine.
            </p>
          )}
          {rmsStatus === "low" && (
            <p className="text-xs text-amber-400">
              ℹ️ Signal low. Move microphone closer (15-20 cm).
            </p>
          )}

          <button
            onClick={stopLiveRecording}
            className="w-full py-2.5 bg-red-600 hover:bg-red-500 text-white rounded-lg font-medium text-sm transition"
          >
            {t("stopRecording")}
          </button>
        </div>
      ) : (
        <div className="py-4 text-center">
          <button
            onClick={startLiveRecording}
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-semibold text-sm transition shadow-lg shadow-cyan-900/30 flex items-center justify-center gap-2 mx-auto"
          >
            <svg
              className="w-5 h-5 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            </svg>
            {t("startRecording")} ({targetDurationSec}s)
          </button>
          <p className="text-xs text-slate-500 mt-3">
            Captures calibrated 16kHz mono audio directly via Web Audio API.
          </p>
        </div>
      )}
    </div>
  );
}
