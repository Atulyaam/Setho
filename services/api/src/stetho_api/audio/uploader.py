"""Audio ingestion, validation, and storage service for Stetho API."""

from dataclasses import dataclass, field
import hashlib
import io
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import soundfile as sf

from stetho_api.exceptions import StethoAPIException

logger = logging.getLogger("stetho.audio_uploader")

# Allowed audio container formats and MIME types
ALLOWED_EXTENSIONS = {".wav", ".wave"}
ALLOWED_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",  # Often sent by generic multipart clients
}

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 30.0


class AudioValidationError(StethoAPIException):
    """Raised when uploaded audio fails ingestion or format validation."""

    def __init__(self, message: str, code: str = "AUDIO_VALIDATION_ERROR", details: Any = None):
        super().__init__(message=message, code=code, status_code=400, details=details)


@dataclass
class UploadedAudioMetadata:
    """Sanitized metadata describing an ingested audio file."""
    upload_id: str
    original_filename: str
    sanitized_filename: str
    stored_path: str
    file_size_bytes: int
    duration_sec: float
    sample_rate: int
    channels: int
    format: str
    subtype: str
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "original_filename": self.original_filename,
            "sanitized_filename": self.sanitized_filename,
            "stored_path": self.stored_path,
            "file_size_bytes": self.file_size_bytes,
            "duration_sec": round(self.duration_sec, 4),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
            "subtype": self.subtype,
            "content_hash": self.content_hash,
        }


class AudioIngestionService:
    """Manages secure streaming, validation, and storage of incoming audio files."""

    def __init__(self, storage_dir: Union[str, Path] = "./data/storage"):
        self.storage_dir = Path(storage_dir)
        self.recordings_dir = self.storage_dir / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def verify_wav_header(file_header: bytes) -> bool:
        """Inspect magic bytes to verify RIFF/WAVE header."""
        if len(file_header) < 12:
            return False
        return file_header[:4] == b"RIFF" and file_header[8:12] == b"WAVE"

    def ingest_audio(
        self,
        file_obj: BinaryIO,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        machine_id: str = "generic",
    ) -> Tuple[UploadedAudioMetadata, np.ndarray]:
        """Stream, validate, and store audio file securely.

        Returns:
            metadata: Sanitized upload metadata.
            waveform: Raw decoded numpy float32 audio array.
        """
        raw_name = filename or "audio.wav"
        ext = Path(raw_name).suffix.lower()

        # 1. Extension check
        if ext not in ALLOWED_EXTENSIONS:
            raise AudioValidationError(
                f"Unsupported file extension '{ext}'. Allowed extensions: {list(ALLOWED_EXTENSIONS)}",
                code="UNSUPPORTED_AUDIO_FORMAT",
            )

        # 2. Content-type check if provided
        if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
            logger.warning(
                f"Suspicious MIME type '{content_type}' for {raw_name}; will verify container magic bytes"
            )

        # 3. Stream content and enforce size limits
        buffer = bytearray()
        chunk_size = 64 * 1024
        while chunk := file_obj.read(chunk_size):
            buffer.extend(chunk)
            if len(buffer) > MAX_FILE_SIZE_BYTES:
                raise AudioValidationError(
                    f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES / (1024*1024):.1f} MB",
                    code="PAYLOAD_TOO_LARGE",
                )

        file_bytes = bytes(buffer)
        if len(file_bytes) == 0:
            raise AudioValidationError("Uploaded file is empty (0 bytes)", code="EMPTY_FILE")

        # 4. Verify magic bytes
        if not self.verify_wav_header(file_bytes):
            raise AudioValidationError(
                "File header does not match valid RIFF/WAVE format",
                code="INVALID_AUDIO_HEADER",
            )

        # 5. Decode audio using soundfile
        try:
            byte_stream = io.BytesIO(file_bytes)
            data, sr = sf.read(byte_stream, dtype="float32", always_2d=True)
            info = sf.info(io.BytesIO(file_bytes))
        except Exception as e:
            raise AudioValidationError(
                f"Failed to decode audio payload: {str(e)}",
                code="DECODE_ERROR",
            )

        # 6. Validate numerical integrity and duration
        if np.isnan(data).any() or np.isinf(data).any():
            raise AudioValidationError("Audio data contains NaN or Infinite values", code="NUMERICAL_ERROR")

        duration_sec = data.shape[0] / sr if sr > 0 else 0.0
        if duration_sec < MIN_DURATION_SEC:
            raise AudioValidationError(
                f"Audio duration ({duration_sec:.2f}s) is shorter than minimum required ({MIN_DURATION_SEC:.1f}s)",
                code="AUDIO_TOO_SHORT",
            )
        if duration_sec > MAX_DURATION_SEC:
            raise AudioValidationError(
                f"Audio duration ({duration_sec:.2f}s) exceeds maximum allowed ({MAX_DURATION_SEC:.1f}s)",
                code="AUDIO_TOO_LONG",
            )

        # 7. Store securely under server-generated UUID (prevents path traversal)
        upload_id = str(uuid.uuid4())
        safe_filename = f"{upload_id}.wav"
        machine_store_dir = self.recordings_dir / machine_id
        machine_store_dir.mkdir(parents=True, exist_ok=True)
        target_path = machine_store_dir / safe_filename

        with open(target_path, "wb") as f:
            f.write(file_bytes)

        content_hash = hashlib.sha256(file_bytes).hexdigest()

        metadata = UploadedAudioMetadata(
            upload_id=upload_id,
            original_filename=Path(raw_name).name,  # Sanitized basename
            sanitized_filename=safe_filename,
            stored_path=str(target_path),
            file_size_bytes=len(file_bytes),
            duration_sec=duration_sec,
            sample_rate=sr,
            channels=data.shape[1],
            format=info.format,
            subtype=info.subtype,
            content_hash=content_hash,
        )

        logger.info(
            f"Successfully ingested audio upload_id={upload_id} for machine={machine_id} "
            f"({duration_sec:.2f}s, {data.shape[1]}ch, {sr}Hz, {len(file_bytes)} bytes)"
        )
        return metadata, data

    def save_recording(
        self,
        machine_id: str,
        file_bytes: bytes,
        original_filename: str = "audio.wav",
    ) -> Path:
        """Save raw audio bytes securely under a server-generated UUID filename."""
        upload_id = str(uuid.uuid4())
        safe_filename = f"{upload_id}.wav"
        machine_store_dir = self.recordings_dir / machine_id
        machine_store_dir.mkdir(parents=True, exist_ok=True)
        target_path = machine_store_dir / safe_filename
        with open(target_path, "wb") as f:
            f.write(file_bytes)
        return target_path

