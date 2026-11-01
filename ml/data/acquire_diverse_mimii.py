"""Diverse MIMII Industrial Dataset Acquisition and Manifest Generator.

Acquires multi-category MIMII 1.0 datasets (Fan, Pump, Valve, Slider)
from Zenodo Record 3384388 (Purohit et al., Hitachi Ltd., CC BY-SA 4.0)
using HTTP Range Requests with ZIP64 CRC32 integrity verification.

Generates a comprehensive dataset manifest with SHA-256 checksums,
audio properties, and split memberships.
"""

import argparse
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import zlib

import httpx
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stetho_acquire_diverse")

ZENODO_RECORD_ID = "3384388"
ZENODO_RECORD_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DEFAULT_DATA_DIR = Path("data/mimii")

ARCHIVE_MAP = {
    "fan": "0_dB_fan.zip",
    "pump": "0_dB_pump.zip",
    "valve": "0_dB_valve.zip",
    "slider": "0_dB_slider.zip",
}

MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]


class Zip64RemoteExtractor:
    """Reads and extracts entries directly from remote ZIP64 archives using HTTP Range requests."""

    def __init__(self, file_url: str, total_size: int):
        self.file_url = file_url
        self.total_size = total_size
        self.client = httpx.Client(timeout=45.0, follow_redirects=True)
        self.entries: Dict[str, dict] = {}

    def fetch_range(self, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}"}
        resp = self.client.get(self.file_url, headers=headers)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"HTTP Range request failed: {resp.status_code} for range {start}-{end}")
        return resp.content

    def read_central_directory(self) -> Dict[str, dict]:
        """Locate and parse ZIP64 central directory."""
        tail_size = min(self.total_size, 1024 * 1024)
        tail_data = self.fetch_range(self.total_size - tail_size, self.total_size - 1)

        loc_pos = tail_data.rfind(b"PK\x06\x07")
        if loc_pos == -1:
            raise RuntimeError("ZIP64 locator (PK\\x06\\x07) not found in archive tail")

        zip64_eocd_offset = struct.unpack("<Q", tail_data[loc_pos + 8 : loc_pos + 16])[0]
        eocd64 = self.fetch_range(zip64_eocd_offset, self.total_size - 1)
        cd_size = struct.unpack("<Q", eocd64[40:48])[0]
        cd_offset = struct.unpack("<Q", eocd64[48:56])[0]
        num_entries = struct.unpack("<Q", eocd64[32:40])[0]

        logger.info(f"Central Directory: {num_entries} entries, {cd_size} bytes at offset {cd_offset}")
        cd_data = self.fetch_range(cd_offset, cd_offset + cd_size - 1)

        idx = 0
        self.entries.clear()
        while idx < len(cd_data):
            if cd_data[idx : idx + 4] != b"PK\x01\x02":
                break
            compression = struct.unpack("<H", cd_data[idx + 10 : idx + 12])[0]
            crc32 = struct.unpack("<I", cd_data[idx + 16 : idx + 20])[0]
            c_size = struct.unpack("<I", cd_data[idx + 20 : idx + 24])[0]
            u_size = struct.unpack("<I", cd_data[idx + 24 : idx + 28])[0]
            fn_len = struct.unpack("<H", cd_data[idx + 28 : idx + 30])[0]
            extra_len = struct.unpack("<H", cd_data[idx + 30 : idx + 32])[0]
            comment_len = struct.unpack("<H", cd_data[idx + 32 : idx + 34])[0]
            local_offset = struct.unpack("<I", cd_data[idx + 42 : idx + 46])[0]

            fn = cd_data[idx + 46 : idx + 46 + fn_len].decode("utf-8", errors="replace")

            extra = cd_data[idx + 46 + fn_len : idx + 46 + fn_len + extra_len]
            e_idx = 0
            while e_idx < len(extra):
                e_header = struct.unpack("<H", extra[e_idx : e_idx + 2])[0]
                e_len = struct.unpack("<H", extra[e_idx + 2 : e_idx + 4])[0]
                if e_header == 0x0001:
                    offset_cursor = e_idx + 4
                    if u_size == 0xFFFFFFFF and offset_cursor + 8 <= len(extra):
                        u_size = struct.unpack("<Q", extra[offset_cursor : offset_cursor + 8])[0]
                        offset_cursor += 8
                    if c_size == 0xFFFFFFFF and offset_cursor + 8 <= len(extra):
                        c_size = struct.unpack("<Q", extra[offset_cursor : offset_cursor + 8])[0]
                        offset_cursor += 8
                    if local_offset == 0xFFFFFFFF and offset_cursor + 8 <= len(extra):
                        local_offset = struct.unpack("<Q", extra[offset_cursor : offset_cursor + 8])[0]
                e_idx += 4 + e_len

            self.entries[fn] = {
                "filename": fn,
                "compression": compression,
                "crc32": crc32,
                "c_size": c_size,
                "u_size": u_size,
                "local_offset": local_offset,
            }
            idx += 46 + fn_len + extra_len + comment_len

        logger.info(f"Successfully indexed {len(self.entries)} entries from remote archive")
        return self.entries

    def extract_file(self, entry: dict) -> bytes:
        """Extract and decompress a single file from the remote archive."""
        local_offset = entry["local_offset"]
        c_size = entry["c_size"]
        compression = entry["compression"]

        local_hdr = self.fetch_range(local_offset, local_offset + 64)
        if local_hdr[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Corrupt local header at offset {local_offset} for {entry['filename']}")
        fn_len = struct.unpack("<H", local_hdr[26:28])[0]
        extra_len = struct.unpack("<H", local_hdr[28:30])[0]
        data_start = local_offset + 30 + fn_len + extra_len

        raw_data = self.fetch_range(data_start, data_start + c_size - 1)

        if compression == 0:
            decompressed = raw_data
        elif compression == 8:
            decompressed = zlib.decompress(raw_data, -15)
        else:
            raise NotImplementedError(f"Unsupported compression method: {compression}")

        actual_crc = zlib.crc32(decompressed) & 0xFFFFFFFF
        if actual_crc != entry["crc32"]:
            raise RuntimeError(f"CRC32 mismatch for {entry['filename']}: expected {entry['crc32']}, got {actual_crc}")

        return decompressed


def acquire_category(
    category: str,
    output_dir: Path = DEFAULT_DATA_DIR,
    normal_count: int = 15,
    abnormal_count: int = 15,
) -> int:
    """Acquire target samples for all machine IDs of a category."""
    if category not in ARCHIVE_MAP:
        raise ValueError(f"Unknown category: {category}")

    archive_key = ARCHIVE_MAP[category]
    meta_path = output_dir / "metadata" / f"zenodo_{ZENODO_RECORD_ID}_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    file_info = next(f for f in meta["files"] if f["key"] == archive_key)
    logger.info(f"Connecting to remote archive for {category}: {archive_key} ({file_info['size'] / 1e6:.1f} MB)...")
    extractor = Zip64RemoteExtractor(file_info["links"]["self"], file_info["size"])
    entries = extractor.read_central_directory()

    extracted = 0
    raw_cat_dir = output_dir / "raw" / category

    for m_id in MACHINE_IDS:
        for label, target_count in [("normal", normal_count), ("abnormal", abnormal_count)]:
            dest_dir = raw_cat_dir / m_id / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            prefix = f"{category}/{m_id}/{label}/"
            matching_entries = sorted(
                [e for k, e in entries.items() if k.startswith(prefix) and k.endswith(".wav")],
                key=lambda x: x["filename"],
            )[:target_count]

            for entry in matching_entries:
                fname = Path(entry["filename"]).name
                target_file = dest_dir / fname

                if target_file.exists() and target_file.stat().st_size == entry["u_size"]:
                    extracted += 1
                    continue

                wav_bytes = extractor.extract_file(entry)
                with open(target_file, "wb") as f_out:
                    f_out.write(wav_bytes)

                extracted += 1
                logger.info(f"Extracted [{category}/{m_id}/{label}] {fname} ({len(wav_bytes)} bytes)")

    logger.info(f"Acquisition for {category} completed: {extracted} verified files present.")
    return extracted


def generate_manifest(output_dir: Path = DEFAULT_DATA_DIR) -> Dict[str, Any]:
    """Inspect all raw audio files, verify integrity, and generate machine-readable manifest."""
    raw_dir = output_dir / "raw"
    records = []
    category_counts = {}
    label_counts = {}
    seen_hashes = {}
    duplicates = []
    corrupted = []

    logger.info("Scanning dataset files and computing SHA-256 cryptographic hashes...")

    for wav_path in sorted(raw_dir.glob("*/*/*/*.wav")):
        parts = wav_path.relative_to(raw_dir).parts
        if len(parts) != 4:
            continue
        category, m_id, label, fname = parts[0], parts[1], parts[2], parts[3]

        # Compute SHA-256
        h = hashlib.sha256()
        with open(wav_path, "rb") as f:
            h.update(f.read())
        sha256 = h.hexdigest()

        # Check duplicate
        if sha256 in seen_hashes:
            duplicates.append({"path": str(wav_path), "original": seen_hashes[sha256]})
        seen_hashes[sha256] = str(wav_path)

        # Read audio metadata via soundfile
        try:
            info = sf.info(str(wav_path))
            duration_sec = round(info.duration, 4)
            sample_rate = info.samplerate
            channels = info.channels
            is_valid = True
        except Exception as e:
            corrupted.append({"path": str(wav_path), "error": str(e)})
            duration_sec = 0.0
            sample_rate = 0
            channels = 0
            is_valid = False

        record = {
            "dataset": "MIMII 1.0",
            "category": category,
            "machine_id": m_id,
            "filename": fname,
            "label": label,
            "domain": "source",
            "snr_db": 0.0,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_sec": duration_sec,
            "file_size_bytes": wav_path.stat().st_size,
            "sha256": sha256,
            "file_path": str(wav_path.resolve()).replace("\\", "/"),
            "rel_path": str(wav_path.relative_to(output_dir.parent.parent)).replace("\\", "/"),
            "is_valid": is_valid,
        }
        records.append(record)

        category_counts[category] = category_counts.get(category, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1

    manifest = {
        "dataset_name": "Stetho Diverse Industrial Acoustic Dataset (MIMII 1.0)",
        "source": "https://zenodo.org/records/3384388",
        "license": "CC BY-SA 4.0",
        "citation": "Purohit et al., MIMII Dataset, DCASE 2019 / arXiv:1909.09347",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_recordings": len(records),
        "total_categories": len(category_counts),
        "category_counts": category_counts,
        "label_counts": label_counts,
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "corrupted_count": len(corrupted),
        "corrupted": corrupted,
        "records": records,
    }

    # Save JSON manifest
    manifest_dir = Path("ml/data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    json_path = manifest_dir / "diverse_dataset_manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save CSV manifest
    csv_path = manifest_dir / "diverse_dataset_manifest.csv"
    if records:
        fieldnames = [
            "dataset", "category", "machine_id", "filename", "label",
            "domain", "snr_db", "sample_rate", "channels", "duration_sec",
            "file_size_bytes", "sha256", "file_path", "rel_path", "is_valid",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    logger.info(f"Manifest written to {json_path} and {csv_path}")
    logger.info(f"Total verified recordings: {len(records)} across categories: {category_counts}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Diverse MIMII Acquisition & Manifest Tool")
    parser.add_argument("--categories", nargs="+", default=["pump", "valve"], help="Categories to acquire")
    parser.add_argument("--skip-acquire", action="store_true", help="Skip download, only generate manifest")
    args = parser.parse_args()

    if not args.skip_acquire:
        for cat in args.categories:
            acquire_category(cat)

    generate_manifest()


if __name__ == "__main__":
    main()

