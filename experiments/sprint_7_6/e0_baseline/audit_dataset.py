"""Phase B — Dataset Inventory & Quality Audit Script for Stetho Sprint 7.6 / E0."""

import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import soundfile as sf

def build_inventory_and_audit():
    root_dir = Path("data/mimii/raw")
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    records = []
    sha_map = {}
    quarantine = []
    
    # 1. Inspect all WAV files in data/mimii/raw
    wav_files = sorted(list(root_dir.glob("*/*/*/*.wav")))
    print(f"Total audio files found in {root_dir}: {len(wav_files)}")

    for idx, wav_path in enumerate(wav_files):
        rel_path = wav_path.relative_to(Path("."))
        file_bytes = wav_path.read_bytes()
        file_size = len(file_bytes)
        sha256 = hashlib.sha256(file_bytes).hexdigest()

        # Parse directory structure: data/mimii/raw/<category>/<machine_id>/<label>/<filename>
        parts = wav_path.parts
        # parts: ('data', 'mimii', 'raw', category, machine_id, label, filename)
        category = parts[3]
        machine_id = parts[4]
        label = parts[5] # 'normal' or 'abnormal'
        filename = parts[6]

        rec_id = f"{category}_{machine_id}_{label}_{filename}"

        # Audio decoding check
        decoding_status = "OK"
        sr = 0
        channels = 0
        samples = 0
        duration_sec = 0.0
        peak_amp = 0.0
        rms_energy = 0.0
        clipping_ratio = 0.0
        is_silent = False
        is_clipped = False
        error_msg = None

        try:
            data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
            samples = data.shape[0]
            channels = data.shape[1]
            duration_sec = samples / sr if sr > 0 else 0.0
            peak_amp = float(np.max(np.abs(data)))
            rms_energy = float(np.sqrt(np.mean(data**2)))
            
            # Clipping detection: sample >= 0.999
            clipped_samples = int(np.sum(np.abs(data) >= 0.999))
            clipping_ratio = float(clipped_samples / (samples * channels)) if samples > 0 else 0.0
            if clipping_ratio > 0.001:
                is_clipped = True
            
            # Silence detection
            if rms_energy < 1e-4:
                is_silent = True
        except Exception as e:
            decoding_status = "ERROR"
            error_msg = str(e)

        # Duplicate checking
        is_duplicate = sha256 in sha_map
        if is_duplicate:
            duplicate_of = sha_map[sha256]
        else:
            sha_map[sha256] = str(rel_path)
            duplicate_of = None

        # Quality issue checks
        issues = []
        if decoding_status != "OK":
            issues.append(f"Decoding failed: {error_msg}")
        if file_size == 0:
            issues.append("Empty file (0 bytes)")
        if sr != 16000:
            issues.append(f"Non-standard sampling rate: {sr} Hz (expected 16000 Hz)")
        if channels != 8:
            issues.append(f"Non-standard channel count: {channels} (expected 8 channels for raw MIMII)")
        if duration_sec < 5.0 or duration_sec > 15.0:
            issues.append(f"Unusual duration: {duration_sec:.2f}s (expected ~10s)")
        if is_clipped:
            issues.append(f"Signal clipping detected (ratio={clipping_ratio:.4f}, peak={peak_amp:.4f})")
        if is_silent:
            issues.append(f"Silent recording detected (RMS={rms_energy:.6f})")
        if is_duplicate:
            issues.append(f"Exact SHA-256 duplicate of {duplicate_of}")

        record = {
            "recording_id": rec_id,
            "relative_path": str(rel_path).replace("\\", "/"),
            "file_size_bytes": file_size,
            "sha256": sha256,
            "category": category,
            "machine_id": machine_id,
            "label": label,
            "sampling_rate": sr,
            "channels": channels,
            "duration_sec": round(duration_sec, 3),
            "num_samples": samples,
            "peak_amplitude": round(peak_amp, 4),
            "rms_energy": round(rms_energy, 6),
            "clipping_ratio": round(clipping_ratio, 6),
            "decoding_status": decoding_status,
            "dataset_source": "MIMII",
            "is_duplicate": is_duplicate,
            "duplicate_of": duplicate_of,
            "issues": issues,
        }
        records.append(record)

        if issues:
            quarantine.append({
                "recording_id": rec_id,
                "relative_path": str(rel_path).replace("\\", "/"),
                "issues": issues,
            })

    # 2. Check near-duplicates (pairwise spectrogram / energy distance)
    # Sample downsampled envelope to quickly compute pairwise cosine similarity
    print("Computing signal similarity matrix to detect near-duplicates...")
    envelopes = []
    for r in records:
        try:
            d, s = sf.read(r["relative_path"], dtype="float32", always_2d=True)
            # Use channel 0 downsampled by factor of 32
            env = np.abs(d[::64, 0])
            # normalize
            norm = np.linalg.norm(env)
            if norm > 0:
                env = env / norm
            envelopes.append(env)
        except Exception:
            envelopes.append(np.zeros(100))

    near_duplicates = []
    min_len = min(len(e) for e in envelopes)
    env_mat = np.array([e[:min_len] for e in envelopes])
    sim_mat = np.dot(env_mat, env_mat.T)
    np.fill_diagonal(sim_mat, 0.0)

    # Flag pairs with cosine similarity > 0.9999 that are not exact duplicates
    high_sim_pairs = np.argwhere(sim_mat > 0.9999)
    for i, j in high_sim_pairs:
        if i < j:
            near_duplicates.append({
                "recording_a": records[i]["recording_id"],
                "path_a": records[i]["relative_path"],
                "recording_b": records[j]["recording_id"],
                "path_b": records[j]["relative_path"],
                "cosine_similarity": float(sim_mat[i, j]),
            })

    # 3. Class distributions and splits
    class_dist = {}
    for cat in ["fan", "pump", "valve"]:
        class_dist[cat] = {}
        for m in ["id_00", "id_02", "id_04", "id_06"]:
            sub = [r for r in records if r["category"] == cat and r["machine_id"] == m]
            n_norm = sum(1 for r in sub if r["label"] == "normal")
            n_abn = sum(1 for r in sub if r["label"] == "abnormal")
            class_dist[cat][m] = {
                "normal": n_norm,
                "abnormal": n_abn,
                "total": len(sub),
            }

    # 4. Check historical splits representation
    def count_csv_rows(filepath):
        if not filepath.exists():
            return 0
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            return sum(1 for _ in reader)

    splits_summary = {}
    split_dir = Path("ml/data/splits/diverse_machine_specific")
    if split_dir.exists():
        for cat in ["fan", "pump", "valve"]:
            splits_summary[cat] = {}
            for m in ["id_00", "id_02", "id_04", "id_06"]:
                m_dir = split_dir / cat / m
                train_file = m_dir / "train_normal.csv"
                dev_file = m_dir / "dev_normal.csv"
                test_file = m_dir / "test.csv"
                splits_summary[cat][m] = {
                    "train_normal": count_csv_rows(train_file),
                    "dev_normal": count_csv_rows(dev_file),
                    "test": count_csv_rows(test_file),
                }

    # Quality Summary
    quality_summary = {
        "audit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_recordings_evaluated": len(records),
        "total_size_bytes": sum(r["file_size_bytes"] for r in records),
        "total_size_mb": round(sum(r["file_size_bytes"] for r in records) / (1024**2), 2),
        "categories": sorted(list(set(r["category"] for r in records))),
        "machine_ids": sorted(list(set(r["machine_id"] for r in records))),
        "class_distribution": class_dist,
        "splits_summary": splits_summary,
        "sampling_rates_found": list(set(r["sampling_rate"] for r in records)),
        "channel_counts_found": list(set(r["channels"] for r in records)),
        "durations_found_sec": {
            "min": round(min(r["duration_sec"] for r in records), 2),
            "max": round(max(r["duration_sec"] for r in records), 2),
            "mean": round(float(np.mean([r["duration_sec"] for r in records])), 2),
        },
        "exact_duplicates_count": sum(1 for r in records if r["is_duplicate"]),
        "near_duplicates_count": len(near_duplicates),
        "near_duplicates": near_duplicates,
        "quarantined_recordings_count": len(quarantine),
        "quarantined_recordings": quarantine,
        "clipping_recordings_count": sum(1 for r in records if r["clipping_ratio"] > 0.001),
        "silent_recordings_count": sum(1 for r in records if r["rms_energy"] < 1e-4),
        "decoding_errors_count": sum(1 for r in records if r["decoding_status"] != "OK"),
        "dataset_size_assessment": {
            "total_files": len(records),
            "normal_files": sum(1 for r in records if r["label"] == "normal"),
            "abnormal_files": sum(1 for r in records if r["label"] == "abnormal"),
            "is_statistically_sound": False,
            "statistical_warning": (
                "CRITICAL WARNING: The physical dataset contains only 360 recordings (120 per category, "
                "30 per machine: 15 normal, 15 abnormal). In historical machine-specific splits, the test set "
                "consists of only 3 normal and 15 abnormal clips per machine. A single misclassification shifts "
                "ROC-AUC by up to 0.05-0.10. Results on this benchmark cannot claim industrial-grade reliability."
            ),
        },
    }

    # Write deliverables
    inv_path = Path("experiments/sprint_7_6/e0_baseline/dataset_inventory.json")
    with open(inv_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Dataset inventory written to: {inv_path}")

    qual_path = Path("experiments/sprint_7_6/e0_baseline/dataset_quality.json")
    with open(qual_path, "w") as f:
        json.dump(quality_summary, f, indent=2)
    print(f"Dataset quality summary written to: {qual_path}")

    print("\nQuality Summary:")
    print(f"  Total Files: {len(records)}")
    print(f"  Exact Duplicates: {quality_summary['exact_duplicates_count']}")
    print(f"  Near Duplicates: {quality_summary['near_duplicates_count']}")
    print(f"  Quarantined Files: {len(quarantine)}")
    print(f"  Decoding Errors: {quality_summary['decoding_errors_count']}")
    print(f"  Sampling Rates: {quality_summary['sampling_rates_found']}")
    print(f"  Channels: {quality_summary['channel_counts_found']}")
    print(f"  Durations: {quality_summary['durations_found_sec']}")

if __name__ == "__main__":
    build_inventory_and_audit()
