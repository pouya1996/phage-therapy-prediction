"""
File handler — upload validation and temp-file management.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from src.utils.config_loader import config
from src.utils.logger_utils import setup_logger

logger = setup_logger(__name__)

ALLOWED_EXTENSIONS = {".fasta", ".fa", ".fna", ".fsa"}
MAX_FILE_SIZE = config.api.get("max_file_size", 10 * 1024 * 1024)  # 10 MB


def get_upload_dir() -> Path:
    """Return the upload directory (create if needed)."""
    upload_dir = Path(config.api.get("upload_folder", "uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def validate_fasta_file(filename: str, content: bytes) -> Optional[str]:
    """
    Validate an uploaded FASTA file.

    Returns:
        None if valid, otherwise an error message string.
    """
    if not filename:
        return "No filename provided."

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"Invalid file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if len(content) == 0:
        return "Uploaded file is empty."

    if len(content) > MAX_FILE_SIZE:
        return (
            f"File too large ({len(content) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
        )

    # Quick sanity check: should start with '>'
    text = content.decode("utf-8", errors="ignore").strip()
    if not text.startswith(">"):
        return "File does not appear to be a valid FASTA (must start with '>')."

    return None  # valid


async def save_uploaded_file(filename: str, content: bytes) -> Path:
    """
    Save uploaded bytes to a unique path in the upload directory.

    Returns:
        Path to the saved file.
    """
    upload_dir = get_upload_dir()
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    dest = upload_dir / unique_name
    dest.write_bytes(content)
    logger.info(f"Saved upload → {dest}  ({len(content):,} bytes)")
    return dest


def clean_temp_files(file_path: Path) -> None:
    """Remove a temporary file (best-effort)."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned temp file: {file_path}")
    except OSError as e:
        logger.warning(f"Failed to clean temp file {file_path}: {e}")


def clean_upload_dir() -> None:
    """Remove ALL files in the upload directory."""
    upload_dir = get_upload_dir()
    for f in upload_dir.iterdir():
        if f.is_file():
            f.unlink()
    logger.info("Cleared upload directory")
