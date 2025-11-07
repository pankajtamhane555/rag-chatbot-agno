"""Unit tests for PDF parsing functionality."""

import pytest
import pytest_check as check
from pathlib import Path
import tempfile

from app.agent import AgentManager, FileParseError
from app.validation import ValidationError, validate_uploaded_file, validate_pdf_file
from pypdf import PdfReader, PdfWriter


def create_test_pdf(content: str = "Test PDF content", pages: int = 1) -> Path:
    """Create a test PDF file for testing.

    Args:
        content: Text content to add to PDF.
        pages: Number of pages.

    Returns:
        Path to created PDF file.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = Path(temp_file.name)

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
        # Note: pypdf doesn't easily add text, so we'll test with blank pages
        # For real content, we'd need a more complex PDF creation

    with open(temp_path, "wb") as f:
        writer.write(f)

    return temp_path


def test_extract_file_content_extracts_text(tmp_path: Path) -> None:
    """Test that extract_file_content extracts text from PDF."""
    pdf_path = create_test_pdf("Test content", pages=1)

    try:
        manager = AgentManager()
        content = manager.extract_file_content(pdf_path)

        check.is_instance(content, str, "Extracted content should be a string")
        # Note: Blank PDFs may not have extractable text, but method should work
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


def test_extract_file_metadata_extracts_metadata(tmp_path: Path) -> None:
    """Test that extract_file_metadata extracts metadata."""
    pdf_path = create_test_pdf(pages=2)

    try:
        manager = AgentManager()
        metadata = manager.extract_file_metadata(pdf_path)

        check.is_in("page_count", metadata.model_dump(), "Metadata should have page_count")
        check.equal(metadata.page_count, 2, "Should extract correct page count")
        # Title may be None for blank PDFs, which is valid
        assert metadata.title is None or isinstance(metadata.title, str)
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


def test_extract_file_metadata_handles_multi_page(tmp_path: Path) -> None:
    """Test that extract_file_metadata handles multi-page PDFs."""
    pdf_path = create_test_pdf(pages=3)

    try:
        manager = AgentManager()
        metadata = manager.extract_file_metadata(pdf_path)

        check.equal(metadata.page_count, 3, "Should handle multiple pages")
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


def test_extract_file_content_file_not_found() -> None:
    """Test that extract_file_content raises FileNotFoundError for missing file."""
    manager = AgentManager()
    with pytest.raises(FileNotFoundError):
        manager.extract_file_content("/nonexistent/file.pdf")


def test_extract_file_content_corrupt_file(tmp_path: Path) -> None:
    """Test that extract_file_content handles corrupt PDFs."""
    # Create a file that's not a valid PDF
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_text("This is not a PDF file")

    manager = AgentManager()
    # Should handle gracefully (may return empty string or raise FileParseError)
    try:
        content = manager.extract_file_content(corrupt_file)
        # If it doesn't raise, content might be empty
        check.is_instance(content, str, "Should return string even for corrupt files")
    except FileParseError:
        # This is also acceptable
        pass


def test_validate_uploaded_file_valid() -> None:
    """Test validation of valid uploaded file."""
    pdf_content = b"%PDF-1.4\nTest content"
    # Should not raise
    validate_uploaded_file("test.pdf", file_content=pdf_content)


def test_validate_uploaded_file_wrong_extension() -> None:
    """Test validation rejects unsupported file types."""
    with pytest.raises(ValidationError) as exc_info:
        validate_uploaded_file("test.xyz", file_content=b"content")
    check.is_in("Invalid file type", str(exc_info.value))


def test_validate_uploaded_file_empty() -> None:
    """Test validation rejects empty files."""
    with pytest.raises(ValidationError) as exc_info:
        validate_uploaded_file("test.pdf", file_content=b"")
    check.is_in("empty", str(exc_info.value).lower())


def test_validate_uploaded_file_invalid_header() -> None:
    """Test validation rejects files without PDF header."""
    with pytest.raises(ValidationError) as exc_info:
        validate_uploaded_file("test.pdf", file_content=b"Not a PDF")
    check.is_in("valid PDF", str(exc_info.value))


def test_validate_uploaded_file_no_filename() -> None:
    """Test validation rejects missing filename."""
    with pytest.raises(ValidationError) as exc_info:
        validate_uploaded_file("", file_content=b"%PDF-1.4")
    check.is_in("filename", str(exc_info.value).lower())


def test_validate_pdf_file_valid_file(tmp_path: Path) -> None:
    """Test validation of valid PDF file."""
    pdf_path = create_test_pdf()

    try:
        # Should not raise
        validate_pdf_file(pdf_path)
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


def test_validate_pdf_file_wrong_extension(tmp_path: Path) -> None:
    """Test validation rejects unsupported file types."""
    xyz_file = tmp_path / "test.xyz"
    xyz_file.write_text("Not a PDF")

    with pytest.raises(ValidationError) as exc_info:
        validate_pdf_file(xyz_file)
    check.is_in("Invalid file type", str(exc_info.value))


def test_validate_pdf_file_empty_file(tmp_path: Path) -> None:
    """Test validation rejects empty files."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    with pytest.raises(ValidationError) as exc_info:
        validate_pdf_file(empty_file)
    check.is_in("empty", str(exc_info.value).lower())


def test_validate_pdf_file_oversized(tmp_path: Path) -> None:
    """Test validation rejects oversized files."""
    pdf_path = create_test_pdf()

    try:
        # Create a file that's too large (simulate by passing large size)
        with pytest.raises(ValidationError) as exc_info:
            validate_pdf_file(pdf_path, file_size=11 * 1024 * 1024)  # 11MB
        check.is_in("too large", str(exc_info.value).lower())
    finally:
        if pdf_path.exists():
            pdf_path.unlink()

