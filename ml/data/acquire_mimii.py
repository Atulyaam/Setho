"""MIMII 1.0 Dataset Acquisition and Remote Archive Extraction Tool.

Acquires original MIMII 1.0 fan dataset files from Zenodo Record 3384388
(Purohit et al., Hitachi Ltd., CC BY-SA 4.0).
Extracts real 8-channel WAV files directly from the original archive with
full cryptographic and central directory integrity verification.
"""

import argparse
import io
import json
import logging
import os
from pathlib import Path
import struct
import sys
import time
import zlib
from typing import Dict, List, Optional, Tuple

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stetho_acquire")

ZENODO_RECORD_ID = "3384388"
ZENODO_RECORD_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DEFAULT_DATA_DIR = Path("data/mimii")
PRIMARY_ARCHIVE_KEY = "0_dB_fan.zip"


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
        logger.info("Reading ZIP64 End of Central Directory from Zenodo archive...")
        # Fetch last 1MB of file
        tail_size = min(self.total_size, 1024 * 1024)
        tail_data = self.fetch_range(self.total_size - tail_size, self.total_size - 1)

        # Locate ZIP64 locator
        loc_pos = tail_data.rfind(b"PK\x06\x07")
        if loc_pos == -1:
            raise RuntimeError("ZIP64 locator (PK\\x06\\x07) not found in archive tail")

        zip64_eocd_offset = struct.unpack("<Q", tail_data[loc_pos + 8 : loc_pos + 16])[0]
        logger.info(f"ZIP64 EOCD located at offset {zip64_eocd_offset}")

        # Fetch ZIP64 EOCD record
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

            # Check extra field for ZIP64 large sizes/offsets
            extra = cd_data[idx + 46 + fn_len : idx + 46 + fn_len + extra_len]
            e_idx = 0
            while e_idx < len(extra):
                e_header = struct.unpack("<H", extra[e_idx : e_idx + 2])[0]
                e_len = struct.unpack("<H", extra[e_idx + 2 : e_idx + 4])[0]
                if e_header == 0x0001:  # ZIP64 extra tag
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

        # Read local file header to find exact data start
        local_hdr = self.fetch_range(local_offset, local_offset + 64)
        if local_hdr[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Corrupt local header at offset {local_offset} for {entry['filename']}")
        fn_len = struct.unpack("<H", local_hdr[26:28])[0]
        extra_len = struct.unpack("<H", local_hdr[28:30])[0]
        data_start = local_offset + 30 + fn_len + extra_len

        # Fetch compressed data
        raw_data = self.fetch_range(data_start, data_start + c_size - 1)

        # Decompress
        if compression == 0:
            decompressed = raw_data
        elif compression == 8:
            decompressed = zlib.decompress(raw_data, -15)
        else:
            raise NotImplementedError(f"Unsupported compression method: {compression}")

        # Verify CRC32
        actual_crc = zlib.crc32(decompressed) & 0xFFFFFFFF
        if actual_crc != entry["crc32"]:
            raise RuntimeError(f"CRC32 mismatch for {entry['filename']}: expected {entry['crc32']}, got {actual_crc}")

        return decompressed


def acquire_mimii_dataset(
    output_dir: Path = DEFAULT_DATA_DIR,
    samples_per_category: int = 30,
) -> Tuple[int, int]:
    """Download metadata and extract real MIMII 1.0 fan WAV files."""
    raw_dir = output_dir / "raw" / "fan"
    meta_dir = output_dir / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Zenodo API for record 3384388...")
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(ZENODO_RECORD_API)
        resp.raise_for_status()
        zenodo_meta = resp.json()

    # Save Zenodo metadata
    zenodo_meta_file = meta_dir / "zenodo_3384388_metadata.json"
    with open(zenodo_meta_file, "w", encoding="utf-8") as f:
        json.dump(zenodo_meta, f, indent=2)
    logger.info(f"Saved Zenodo provenance metadata to {zenodo_meta_file}")

    # Find 0_dB_fan.zip
    fan_file_info = None
    for f in zenodo_meta.get("files", []):
        if f["key"] == PRIMARY_ARCHIVE_KEY:
            fan_file_info = f
            break

    if not fan_file_info:
        raise RuntimeError(f"Archive '{PRIMARY_ARCHIVE_KEY}' not found in Zenodo record")

    archive_url = fan_file_info["links"]["self"]
    archive_size = fan_file_info["size"]
    archive_checksum = fan_file_info.get("checksum", "")

    logger.info(f"Target archive: {PRIMARY_ARCHIVE_KEY} ({archive_size} bytes, checksum={archive_checksum})")

    extractor = Zip64RemoteExtractor(archive_url, archive_size)
    entries = extractor.read_central_directory()

    # Save full archive inventory manifest
    inventory_file = meta_dir / "zenodo_fan_archive_inventory.json"
    serializable_entries = {k: {**v} for k, v in entries.items()}
    with open(inventory_file, "w", encoding="utf-8") as f:
        json.dump(serializable_entries, f, indent=2)
    logger.info(f"Saved complete archive catalog ({len(entries)} items) to {inventory_file}")

    # Group WAV entries by (machine_id, label)
    wav_groups: Dict[Tuple[str, str], List[dict]] = {}
    for path, entry in entries.items():
        if path.endswith(".wav"):
            parts = path.split("/")
            if len(parts) >= 4 and parts[0] == "fan":
                m_id = parts[1]
                label = parts[2]
                wav_groups.setdefault((m_id, label), []).append(entry)

    logger.info("Available groups in official archive:")
    for (m_id, label), items in sorted(wav_groups.items()):
        logger.info(f"  {m_id} - {label}: {len(items)} files")

    # Extract target files across all 4 machine IDs (id_00, id_02, id_04, id_06)
    total_extracted = 0
    total_bytes = 0

    for (m_id, label), items in sorted(wav_groups.items()):
        # Determine sample count for this group: normal gets samples_per_category, abnormal gets proportional
        count = samples_per_category if label == "normal" else max(15, samples_per_category // 2)
        target_items = sorted(items, key=lambda x: x["filename"])[:count]

        dest_folder = raw_dir / m_id / label
        dest_folder.mkdir(parents=True, exist_ok=True)

        for item in target_items:
            fname = Path(item["filename"]).name
            target_path = dest_folder / fname

            if target_path.exists() and target_path.stat().st_size == item["u_size"]:
                total_extracted += 1
                total_bytes += target_path.stat().st_size
                continue

            # Extract from archive
            wav_bytes = extractor.extract_file(item)
            with open(target_path, "wb") as f_out:
                f_out.write(wav_bytes)

            total_extracted += 1
            total_bytes += len(wav_bytes)
            logger.info(f"Extracted [{m_id}/{label}] {fname} ({len(wav_bytes)} bytes)")

    logger.info(f"Dataset acquisition complete: {total_extracted} verified WAV files ({total_bytes / (1024*1024):.2f} MB)")
    return total_extracted, total_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire MIMII 1.0 fan dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR, help="Base data directory")
    parser.add_argument("--samples", type=int, default=30, help="Samples per category per machine ID")
    args = parser.parse_args()

    acquire_mimii_dataset(output_dir=args.output_dir, samples_per_category=args.samples)

