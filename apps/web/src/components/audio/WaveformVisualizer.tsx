"use client";

import React, { useEffect, useRef } from "react";

interface WaveformVisualizerProps {
  audioBuffer: AudioBuffer;
  currentTime?: number;
  height?: number;
}

export function WaveformVisualizer({
  audioBuffer,
  currentTime = 0,
  height = 140,
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioBuffer) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const h = canvas.height;
    const duration = audioBuffer.duration;
    const channelData = audioBuffer.getChannelData(0); // Primary Channel 0
    const step = Math.ceil(channelData.length / width);
    const amp = h / 2;

    // Clear background
    ctx.fillStyle = "#0f172a"; // slate-900
    ctx.fillRect(0, 0, width, h);

    // Grid lines
    ctx.strokeStyle = "#1e293b"; // slate-800
    ctx.lineWidth = 1;
    // Center zero-line
    ctx.beginPath();
    ctx.moveTo(0, amp);
    ctx.lineTo(width, amp);
    ctx.stroke();

    // Time division lines (every 1 second)
    const numSeconds = Math.floor(duration);
    ctx.fillStyle = "#64748b"; // slate-500
    ctx.font = "9px monospace";
    for (let s = 1; s <= numSeconds; s++) {
      const x = (s / duration) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.fillText(`${s}s`, x + 3, h - 5);
    }

    // Amplitude markers
    ctx.fillText("+1.0", 5, 12);
    ctx.fillText(" 0.0", 5, amp - 4);
    ctx.fillText("-1.0", 5, h - 5);

    // Draw Waveform envelope
    ctx.fillStyle = "#14b8a6"; // teal-500
    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;
      const startIdx = i * step;
      const endIdx = Math.min(startIdx + step, channelData.length);

      for (let j = startIdx; j < endIdx; j++) {
        const val = channelData[j];
        if (val < min) min = val;
        if (val > max) max = val;
      }

      const yTop = Math.max(0, amp - max * amp);
      const yBottom = Math.min(h, amp - min * amp);
      const barHeight = Math.max(1, yBottom - yTop);

      ctx.fillRect(i, yTop, 1, barHeight);
    }

    // Playhead line
    if (currentTime > 0 && currentTime <= duration) {
      const playheadX = (currentTime / duration) * width;
      ctx.strokeStyle = "#f59e0b"; // amber-500
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, h);
      ctx.stroke();
    }
  }, [audioBuffer, currentTime, height]);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono">
        <span>Time-Domain Waveform (Channel 0)</span>
        <span>Duration: {audioBuffer.duration.toFixed(2)}s | Rate: {audioBuffer.sampleRate} Hz</span>
      </div>
      <div className="rounded-xl overflow-hidden border border-slate-700/60 shadow-inner bg-slate-900">
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

