"""Comprehensive tests for validation module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.validation import (
    ValidationError,
    validate_file,
    validate_uploaded_file,
    validate_pdf_file,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS,
    FILE_SIGNATURES,
)


def test_validation_error():
    """Test ValidationError exception."""
    error = ValidationError("Test error")
    assert str(error) == "Test error"
    assert isinstance(error, Exception)


def test_validate_file_valid_pdf(tmp_path):
    """Test validate_file with valid PDF."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nTest content")
    
    # Should not raise
    validate_file(pdf_file)


def test_validate_file_invalid_extension(tmp_path):
    """Test validate_file rejects invalid extension."""
    # Use a truly unsupported extension
    xyz_file = tmp_path / "test.xyz"
    xyz_file.write_bytes(b"Test content")
    
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_file(xyz_file)


def test_validate_file_file_not_found():
    """Test validate_file raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        validate_file("/nonexistent/file.pdf")


def test_validate_file_empty_file(tmp_path):
    """Test validate_file rejects empty file."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    
    with pytest.raises(ValidationError, match="empty"):
        validate_file(empty_file)


def test_validate_file_oversized(tmp_path):
    """Test validate_file rejects oversized file."""
    pdf_file = tmp_path / "large.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n" + b"x" * (MAX_FILE_SIZE + 1))
    
    with pytest.raises(ValidationError, match="too large"):
        validate_file(pdf_file)


@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="File permission test not applicable on Windows"
)
@pytest.mark.skip(reason="File permission manipulation may not work reliably in all environments")
def test_validate_file_not_readable(tmp_path):
    """Test validate_file rejects non-readable file."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nTest")
    
    # Make file non-readable (Unix only)
    import os
    pdf_file.chmod(0o000)
    try:
        with pytest.raises(ValidationError, match="not readable"):
            validate_file(pdf_file)
    finally:
        pdf_file.chmod(0o644)


def test_validate_uploaded_file_valid_pdf():
    """Test validate_uploaded_file with valid PDF."""
    pdf_content = b"%PDF-1.4\nTest content"
    # Should not raise
    validate_uploaded_file("test.pdf", file_content=pdf_content)


def test_validate_uploaded_file_no_filename():
    """Test validate_uploaded_file rejects missing filename."""
    with pytest.raises(ValidationError, match="filename"):
        validate_uploaded_file("", file_content=b"%PDF-1.4")


def test_validate_uploaded_file_invalid_extension():
    """Test validate_uploaded_file rejects invalid extension."""
    # Use a truly unsupported extension
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_uploaded_file("test.xyz", file_content=b"content")


def test_validate_uploaded_file_empty_content():
    """Test validate_uploaded_file rejects empty content."""
    with pytest.raises(ValidationError, match="empty"):
        validate_uploaded_file("test.pdf", file_content=b"")


def test_validate_uploaded_file_oversized_content():
    """Test validate_uploaded_file rejects oversized content."""
    large_content = b"%PDF-1.4\n" + b"x" * (MAX_FILE_SIZE + 1)
    with pytest.raises(ValidationError, match="too large"):
        validate_uploaded_file("test.pdf", file_content=large_content)


def test_validate_uploaded_file_invalid_header():
    """Test validate_uploaded_file rejects invalid PDF header."""
    with pytest.raises(ValidationError, match="valid PDF"):
        validate_uploaded_file("test.pdf", file_content=b"Not a PDF")


def test_validate_uploaded_file_valid_json():
    """Test validate_uploaded_file rejects JSON (PDF-only support)."""
    json_content = b'{"test": "data"}'
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_uploaded_file("test.json", file_content=json_content)


def test_validate_uploaded_file_valid_csv():
    """Test validate_uploaded_file rejects CSV (PDF-only support)."""
    csv_content = b"name,age\nJohn,30"
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_uploaded_file("test.csv", file_content=csv_content)


def test_validate_uploaded_file_valid_txt():
    """Test validate_uploaded_file rejects TXT (PDF-only support)."""
    txt_content = b"Plain text content"
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_uploaded_file("test.txt", file_content=txt_content)


def test_validate_uploaded_file_valid_docx():
    """Test validate_uploaded_file rejects DOCX (PDF-only support)."""
    # DOCX files start with PK (ZIP signature)
    docx_content = b"PK\x03\x04" + b"x" * 100
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_uploaded_file("test.docx", file_content=docx_content)


def test_validate_uploaded_file_valid_pptx():
    """Test validate_uploaded_file rejects PPTX (PDF-only support)."""
    # PPTX files start with PK (ZIP signature)
    pptx_content = b"PK\x03\x04" + b"x" * 100
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_uploaded_file("test.pptx", file_content=pptx_content)


def test_validate_uploaded_file_with_file_size():
    """Test validate_uploaded_file with file_size parameter."""
    pdf_content = b"%PDF-1.4\nTest"
    # Should not raise
    validate_uploaded_file("test.pdf", file_size=len(pdf_content))


def test_validate_uploaded_file_file_size_zero():
    """Test validate_uploaded_file rejects zero file_size."""
    with pytest.raises(ValidationError, match="empty"):
        validate_uploaded_file("test.pdf", file_size=0)


def test_validate_uploaded_file_file_size_oversized():
    """Test validate_uploaded_file rejects oversized file_size."""
    with pytest.raises(ValidationError, match="too large"):
        validate_uploaded_file("test.pdf", file_size=MAX_FILE_SIZE + 1)


def test_validate_uploaded_file_no_content_no_size():
    """Test validate_uploaded_file with no content and no size."""
    # Should not raise (only checks extension)
    validate_uploaded_file("test.pdf")


def test_validate_pdf_file_valid(tmp_path):
    """Test validate_pdf_file with valid PDF."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\nTest content")
    
    # Should not raise
    validate_pdf_file(pdf_file)


def test_validate_pdf_file_invalid_extension(tmp_path):
    """Test validate_pdf_file rejects invalid extension."""
    # Use a truly unsupported extension
    xyz_file = tmp_path / "test.xyz"
    xyz_file.write_bytes(b"Not a PDF")
    
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_pdf_file(xyz_file)


def test_validate_pdf_file_empty(tmp_path):
    """Test validate_pdf_file rejects empty file."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    
    with pytest.raises(ValidationError, match="empty"):
        validate_pdf_file(empty_file)


def test_validate_pdf_file_oversized(tmp_path):
    """Test validate_pdf_file rejects oversized file."""
    pdf_file = tmp_path / "large.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n" + b"x" * 100)
    
    with pytest.raises(ValidationError, match="too large"):
        validate_pdf_file(pdf_file, file_size=MAX_FILE_SIZE + 1)


def test_file_signatures_constants():
    """Test FILE_SIGNATURES contains expected entries (PDF-only)."""
    assert ".pdf" in FILE_SIGNATURES
    assert FILE_SIGNATURES[".pdf"] == b"%PDF"
    # Only PDF is supported now
    assert len(FILE_SIGNATURES) == 1


def test_allowed_extensions_constants():
    """Test ALLOWED_EXTENSIONS contains expected extensions (PDF-only)."""
    assert ".pdf" in ALLOWED_EXTENSIONS
    # Only PDF is supported now
    assert len(ALLOWED_EXTENSIONS) == 1


def test_max_file_size_constant():
    """Test MAX_FILE_SIZE is a positive integer."""
    assert MAX_FILE_SIZE > 0
    assert isinstance(MAX_FILE_SIZE, int)


def test_validate_uploaded_file_json_starts_with_brace():
    """Test validate_uploaded_file rejects JSON (PDF-only support)."""
    json_content = b'{"key": "value"}'
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_uploaded_file("test.json", file_content=json_content)


def test_validate_uploaded_file_json_starts_with_bracket():
    """Test validate_uploaded_file rejects JSON (PDF-only support)."""
    json_content = b'[{"key": "value"}]'
    # Should fail because JSON is not an allowed file type
    with pytest.raises(ValidationError, match="Invalid file type"):
        validate_uploaded_file("test.json", file_content=json_content)


def test_validate_uploaded_file_case_insensitive_extension():
    """Test validate_uploaded_file handles case-insensitive extensions."""
    pdf_content = b"%PDF-1.4\nTest"
    # Should not raise
    validate_uploaded_file("test.PDF", file_content=pdf_content)
    validate_uploaded_file("test.Pdf", file_content=pdf_content)

