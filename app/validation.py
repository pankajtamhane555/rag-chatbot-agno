"""File validation module for PDF file uploads."""

import os
from pathlib import Path

from app.config import settings

MAX_FILE_SIZE = settings.get_max_file_size_bytes()
ALLOWED_EXTENSIONS = settings.get_allowed_extensions()

FILE_SIGNATURES: dict[str, bytes] = {
    ".pdf": b"%PDF",
}


class ValidationError(Exception):
    """Exception raised when file validation fails."""

    pass


def validate_file(file_path: str | Path, file_size: int | None = None) -> None:
    """Validate file before parsing."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValidationError(f"Invalid file type. Allowed types: {supported}. Got: {extension}")

    if file_size is None:
        file_size = file_path.stat().st_size

    if file_size == 0:
        raise ValidationError("File is empty. Please upload a non-empty file.")

    if file_size > MAX_FILE_SIZE:
        size_mb = file_size / (1024 * 1024)
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        raise ValidationError(f"File is too large. Maximum size is {max_mb}MB. Got: {size_mb:.2f}MB")

    if not os.access(file_path, os.R_OK):
        raise ValidationError("File is not readable. Please check file permissions.")


def validate_uploaded_file(
    filename: str, file_content: bytes | None = None, file_size: int | None = None
) -> None:
    """Validate uploaded file from request."""
    if not filename:
        raise ValidationError("No filename provided.")

    file_path = Path(filename)
    extension = file_path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValidationError(f"Invalid file type. Allowed types: {supported}. Got: {extension}")

    if file_size is not None:
        if file_size == 0:
            raise ValidationError("File is empty. Please upload a non-empty file.")

        if file_size > MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(f"File is too large. Maximum size is {max_mb}MB. Got: {size_mb:.2f}MB")

    if file_content is not None:
        if len(file_content) == 0:
            raise ValidationError("File is empty. Please upload a non-empty file.")

        if len(file_content) > MAX_FILE_SIZE:
            size_mb = len(file_content) / (1024 * 1024)
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(f"File is too large. Maximum size is {max_mb}MB. Got: {size_mb:.2f}MB")

        signature = FILE_SIGNATURES.get(extension)
        if signature and not file_content.startswith(signature):
            raise ValidationError(
                f"File does not appear to be a valid {extension[1:].upper()} file. File header is invalid."
            )


def validate_pdf_file(file_path: str | Path, file_size: int | None = None) -> None:
    """Validate PDF file (backward compatibility wrapper)."""
    validate_file(file_path, file_size)
