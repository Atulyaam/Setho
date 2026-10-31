"use client";

import React, { useEffect, useRef, useState } from "react";
import { WaveformVisualizer } from "./WaveformVisualizer";
import { SpectrogramVisualizer } from "./SpectrogramVisualizer";
import { Play, Pause, RotateCcw, Volume2, Loader2, AlertCircle } from "lucide-react";

interface AudioPlayerProps {
  fileOrBlob?: Blob | File | null;
  audioUrl?: string | null;
  title?: string;
  showSpectrogram?: boolean;
}

export function AudioPlayer({
  fileOrBlob,
  audioUrl,
  title = "Acoustic Sample",
  showSpectrogram = true,
}: AudioPlayerProps) {
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const startTimeRef = useRef<number>(0);
  const pausedAtRef = useRef<number>(0);
  const animationFrameRef = useRef<number | null>(null);

  // Decode audio data whenever fileOrBlob or audioUrl changes
  useEffect(() => {
    let isCancelled = false;

    async function decodeAudio() {
      if (!fileOrBlob && !audioUrl) {
        setAudioBuffer(null);
        return;
      }

      setIsLoading(true);
      setError(null);
      stopPlayback();

      try {
        let arrayBuffer: ArrayBuffer;
        if (fileOrBlob) {
          arrayBuffer = await fileOrBlob.arrayBuffer();
        } else if (audioUrl) {
          const res = await fetch(audioUrl);
          if (!res.ok) throw new Error(`Failed to load audio (${res.status})`);
          arrayBuffer = await res.arrayBuffer();
        } else {
          return;
        }

        const AudioContextClass =
          window.AudioContext || (window as any).webkitAudioContext;
        if (!audioContextRef.current) {
          audioContextRef.current = new AudioContextClass();
        }

        const decoded = await audioContextRef.current.decodeAudioData(arrayBuffer);
        if (!isCancelled) {
          setAudioBuffer(decoded);
          setDuration(decoded.duration);
          setCurrentTime(0);
          pausedAtRef.current = 0;
        }
      } catch (err: any) {
        if (!isCancelled) {
          setError(err.message || "Failed to decode WAV audio data.");
        }
      } finally {
        if (!isCancelled) {
          setIsLoading(false);
        }
      }
    }

    decodeAudio();

    return () => {
      isCancelled = true;
      stopPlayback();
    };
  }, [fileOrBlob, audioUrl]);

  const updatePlayhead = () => {
    if (!audioContextRef.current || !isPlaying) return;
    const elapsed =
      audioContextRef.current.currentTime - startTimeRef.current + pausedAtRef.current;
    if (elapsed >= duration) {
      stopPlayback();
      setCurrentTime(0);
      pausedAtRef.current = 0;
    } else {
      setCurrentTime(elapsed);
      animationFrameRef.current = requestAnimationFrame(updatePlayhead);
    }
  };

  const play = () => {
    if (!audioBuffer || !audioContextRef.current) return;

    if (audioContextRef.current.state === "suspended") {
      audioContextRef.current.resume();
    }

    const source = audioContextRef.current.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContextRef.current.destination);

    const offset = pausedAtRef.current;
    source.start(0, offset);
    sourceNodeRef.current = source;
    startTimeRef.current = audioContextRef.current.currentTime;
    setIsPlaying(true);

    animationFrameRef.current = requestAnimationFrame(updatePlayhead);
  };

  const pause = () => {
    if (!isPlaying) return;
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.stop();
      } catch {}
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    if (audioContextRef.current) {
      pausedAtRef.current += audioContextRef.current.currentTime - startTimeRef.current;
    }
    setIsPlaying(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  };

  const stopPlayback = () => {
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.stop();
      } catch {}
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    setIsPlaying(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  };

  const togglePlay = () => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  };

  const handleReset = () => {
    stopPlayback();
    setCurrentTime(0);
    pausedAtRef.current = 0;
  };

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remainder = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 10);
    return `${mins}:${remainder < 10 ? "0" : ""}${remainder}.${ms}`;
  };

  return (
    <div className="bg-slate-900 text-white rounded-2xl p-5 border border-slate-800 shadow-md space-y-4">
      {/* Title & Playback Controls Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Volume2 className="w-4 h-4 text-brand-400" />
          <span className="text-xs font-semibold text-slate-200">{title}</span>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <span>{formatTime(currentTime)}</span>
          <span>/</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {isLoading && (
        <div className="py-8 flex flex-col items-center justify-center space-y-2 text-slate-400 text-xs">
          <Loader2 className="w-5 h-5 text-brand-400 animate-spin" />
          <span>Decoding audio buffer...</span>
        </div>
      )}

      {error && (
        <div className="p-3 rounded-xl bg-rose-950/60 border border-rose-800/80 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {audioBuffer && (
        <div className="space-y-4">
          {/* Waveform Canvas */}
          <WaveformVisualizer
            audioBuffer={audioBuffer}
            currentTime={currentTime}
            height={120}
          />

          {/* Optional Spectrogram Canvas */}
          {showSpectrogram && (
            <SpectrogramVisualizer audioBuffer={audioBuffer} height={120} />
          )}

          {/* Control Buttons */}
          <div className="flex items-center justify-center gap-4 pt-2">
            <button
              onClick={handleReset}
              className="p-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
              title="Reset to Start"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            <button
              onClick={togglePlay}
              className="p-3.5 rounded-2xl bg-brand-500 hover:bg-brand-400 text-slate-950 font-bold shadow-lg transition flex items-center justify-center"
              title={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? (
                <Pause className="w-5 h-5 fill-slate-950" />
              ) : (
                <Play className="w-5 h-5 fill-slate-950 ml-0.5" />
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

