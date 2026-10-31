"use client";

import React, { useEffect, useRef } from "react";

interface SpectrogramVisualizerProps {
  audioBuffer: AudioBuffer;
  height?: number;
}

export function SpectrogramVisualizer({
  audioBuffer,
  height = 140,
}: SpectrogramVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioBuffer) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const h = canvas.height;
    const duration = audioBuffer.duration;
    const sampleRate = audioBuffer.sampleRate;
    const channelData = audioBuffer.getChannelData(0);

    // Clear background
    ctx.fillStyle = "#020617"; // slate-950
    ctx.fillRect(0, 0, width, h);

    // Compute Short-Time Fourier Transform (STFT) magnitudes
    const fftSize = 256;
    const halfFft = fftSize / 2;
    const hopSize = Math.max(1, Math.floor((channelData.length - fftSize) / width));

    // Precompute Hann window
    const window = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
      window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
    }

    // Color gradient map for energy (dB)
    function getColor(magnitude: number): string {
      // Normalize magnitude to dB [approx -80dB to 0dB]
      const db = 20 * Math.log10(Math.max(1e-5, magnitude));
      const norm = Math.max(0, Math.min(1, (db + 70) / 70));

      if (norm < 0.25) {
        // Deep blue to teal
        const t = norm / 0.25;
        return `rgb(${Math.floor(10 * t)}, ${Math.floor(60 * t + 20)}, ${Math.floor(100 * t + 40)})`;
      } else if (norm < 0.6) {
        // Teal to green/yellow
        const t = (norm - 0.25) / 0.35;
        return `rgb(${Math.floor(180 * t)}, ${Math.floor(160 + 50 * t)}, ${Math.floor(140 * (1 - t))})`;
      } else {
        // Yellow to bright amber/red
        const t = (norm - 0.6) / 0.4;
        return `rgb(${Math.floor(220 + 35 * t)}, ${Math.floor(210 * (1 - t * 0.6))}, ${Math.floor(20 * (1 - t))})`;
      }
    }

    // Draw STFT columns
    for (let col = 0; col < width; col++) {
      const offset = col * hopSize;
      if (offset + fftSize > channelData.length) break;

      // Extract windowed segment
      const real = new Float32Array(halfFft);
      const imag = new Float32Array(halfFft);

      for (let k = 0; k < halfFft; k++) {
        let r = 0;
        let im = 0;
        // Evaluate DFT bin k
        const angleStep = (-2 * Math.PI * k) / fftSize;
        for (let n = 0; n < fftSize; n += 2) {
          const sample = channelData[offset + n] * window[n];
          const angle = angleStep * n;
          r += sample * Math.cos(angle);
          im += sample * Math.sin(angle);
        }
        real[k] = r;
        imag[k] = im;
      }

      // Render vertical bins
      for (let k = 0; k < halfFft; k++) {
        const mag = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]) / (fftSize / 4);
        ctx.fillStyle = getColor(mag);
        const y = Math.floor(h - (k / halfFft) * h);
        const binHeight = Math.max(1, Math.ceil(h / halfFft));
        ctx.fillRect(col, y, 1, binHeight);
      }
    }

    // Overlay frequency labels and grid
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.font = "9px monospace";
    const maxFreq = Math.round(sampleRate / 2);
    ctx.fillText(`${maxFreq} Hz`, 5, 12);
    ctx.fillText(`${Math.round(maxFreq / 2)} Hz`, 5, Math.floor(h / 2));
    ctx.fillText("0 Hz", 5, h - 5);

    // Time divisions
    const numSeconds = Math.floor(duration);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1;
    for (let s = 1; s <= numSeconds; s++) {
      const x = (s / duration) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.fillText(`${s}s`, x + 3, h - 5);
    }
  }, [audioBuffer, height]);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono">
        <span>Log-Frequency Spectrogram</span>
        <span>Frequency Range: 0 — {Math.round(audioBuffer.sampleRate / 2)} Hz</span>
      </div>
      <div className="rounded-xl overflow-hidden border border-slate-700/60 shadow-inner bg-slate-950">
        <canvas
          ref={canvasRef}
          width={800}
          height={height}
          className="w-full h-auto block"
        />
      </div>
    </div>
  );
}

