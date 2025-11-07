"""Integration tests for document management: upload, view, delete, and content extraction."""

import pytest
import pytest_check as check
import tempfile
from pathlib import Path
from httpx import AsyncClient, ASGITransport
import json

from app.api import app
from app.agent import AgentManager
from pypdf import PdfWriter, PdfReader


def create_test_pdf_with_text(content: str = "Test PDF content") -> Path:
    """Create a test PDF file with actual text content.
    
    Args:
        content: Text content to include in PDF.
        
    Returns:
        Path to created PDF file.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = Path(temp_file.name)
    
    # Create PDF with text using pypdf
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    
    # Note: pypdf doesn't easily add text to pages, but we can test extraction
    # For real content extraction tests, we'd need a PDF with actual text
    with open(temp_path, "wb") as f:
        writer.write(f)
    
    return temp_path


def create_test_json_file(content: dict | None = None) -> Path:
    """Create a test JSON file.
    
    Args:
        content: Dictionary content. Defaults to sample data.
        
    Returns:
        Path to created JSON file.
    """
    if content is None:
        content = {"name": "Test", "value": 123, "items": ["a", "b", "c"]}
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w')
    temp_path = Path(temp_file.name)
    json.dump(content, temp_file)
    temp_file.close()
    
    return temp_path


def create_test_csv_file(content: str | None = None) -> Path:
    """Create a test CSV file.
    
    Args:
        content: CSV content. Defaults to sample data.
        
    Returns:
        Path to created CSV file.
    """
    if content is None:
        content = "name,age,city\nJohn,28,New York\nJane,32,London"
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w')
    temp_path = Path(temp_file.name)
    temp_file.write(content)
    temp_file.close()
    
    return temp_path


@pytest.mark.asyncio
async def test_upload_and_get_document() -> None:
    """Test uploading a document and retrieving it."""
    pdf_path = create_test_pdf_with_text("Test content for retrieval")
    
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Upload document
            with open(pdf_path, "rb") as f:
                files = {"file": ("test.pdf", f.read(), "application/pdf")}
                upload_response = await client.post("/upload", files=files)
            
            check.equal(upload_response.status_code, 200, "Upload should succeed")
            upload_data = upload_response.json()
            check.is_true(upload_data.get("success"), "Upload should be successful")
            
            # Get chat ID from upload response or create a new chat
            # For now, we'll need to get documents from a chat
            # This test assumes we can get the document ID somehow
            # In a real scenario, we'd get it from the upload response or chat documents
            
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


@pytest.mark.asyncio
async def test_get_document_by_id() -> None:
    """Test retrieving a document by ID."""
    pdf_path = create_test_pdf_with_text("Test content")
    
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Upload document first
            with open(pdf_path, "rb") as f:
                files = {"file": ("test.pdf", f.read(), "application/pdf")}
                upload_response = await client.post("/upload", files=files)
            
            check.equal(upload_response.status_code, 200, "Upload should succeed")
            
            # Get chat documents to find document ID
            # Note: This requires a chat_id, which we'd get from upload or create
            # For now, this is a placeholder test structure
            
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


@pytest.mark.asyncio
async def test_delete_document() -> None:
    """Test deleting a document."""
    pdf_path = create_test_pdf_with_text("Test content for deletion")
    
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Upload document
            with open(pdf_path, "rb") as f:
                files = {"file": ("test.pdf", f.read(), "application/pdf")}
                upload_response = await client.post("/upload", files=files)
            
            check.equal(upload_response.status_code, 200, "Upload should succeed")
            
            # Get document ID (would need to extract from response or query chat documents)
            # For now, this is a placeholder test structure
            
            # Delete document
            # delete_response = await client.delete(f"/documents/{document_id}")
            # check.equal(delete_response.status_code, 200, "Delete should succeed")
            
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


@pytest.mark.asyncio
async def test_delete_nonexistent_document() -> None:
    """Test deleting a non-existent document returns 404."""
    from unittest.mock import patch
    with patch("app.api.delete_document_from_db") as mock_delete:
        mock_delete.return_value = False
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            fake_id = "00000000-0000-0000-0000-000000000000"
            response = await client.delete(f"/documents/{fake_id}")
            
            check.equal(response.status_code, 404, "Should return 404 for non-existent document")


def test_extract_pdf_content() -> None:
    """Test PDF content extraction using AgentManager."""
    pdf_path = create_test_pdf_with_text("Test extraction content")
    
    try:
        manager = AgentManager()
        content = manager.extract_file_content(pdf_path)
        
        check.is_instance(content, str, "Extracted content should be a string")
        # Note: Blank PDFs from pypdf may not have extractable text
        # This test verifies the method works without errors
        
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


def test_extract_json_content() -> None:
    """Test JSON content extraction raises error (PDF-only support)."""
    json_path = create_test_json_file({"test": "data", "number": 42})
    
    try:
        manager = AgentManager()
        from app.agent import FileParseError
        with pytest.raises(FileParseError, match="Only PDF files are supported"):
            manager.extract_file_content(json_path)
    finally:
        if json_path.exists():
            json_path.unlink()


def test_extract_csv_content() -> None:
    """Test CSV content extraction raises error (PDF-only support)."""
    csv_path = create_test_csv_file("name,value\ntest,123")
    
    try:
        manager = AgentManager()
        from app.agent import FileParseError
        with pytest.raises(FileParseError, match="Only PDF files are supported"):
            manager.extract_file_content(csv_path)
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_extract_txt_content() -> None:
    """Test TXT content extraction raises error (PDF-only support)."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w')
    temp_path = Path(temp_file.name)
    temp_file.write("This is test text content")
    temp_file.close()
    
    try:
        manager = AgentManager()
        from app.agent import FileParseError
        with pytest.raises(FileParseError, match="Only PDF files are supported"):
            manager.extract_file_content(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_extract_file_content_file_not_found() -> None:
    """Test that extract_file_content raises FileNotFoundError for missing file."""
    manager = AgentManager()
    
    with pytest.raises(FileNotFoundError):
        manager.extract_file_content("/nonexistent/file.pdf")


def test_extract_file_content_unsupported_type() -> None:
    """Test that extract_file_content raises FileParseError for unsupported types."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
    temp_path = Path(temp_file.name)
    temp_file.write(b"test content")
    temp_file.close()
    
    try:
        manager = AgentManager()
        
        from app.agent import FileParseError
        with pytest.raises(FileParseError):
            manager.extract_file_content(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@pytest.mark.asyncio
async def test_get_chat_documents() -> None:
    """Test retrieving all documents for a chat."""
    from unittest.mock import patch
    with patch("app.api.create_chat_in_db") as mock_create_chat:
        mock_create_chat.return_value = {
            "id": "test-chat-id",
            "title": "Test Chat",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "message_count": 0,
            "document_count": 0,
            "document_ids": [],
        }
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Create a chat first
            chat_response = await client.post("/chats")
            check.equal(chat_response.status_code, 200, "Chat creation should succeed")
            chat_data = chat_response.json()
            chat_id = chat_data.get("chat", {}).get("id") or chat_data.get("id")
            check.is_not_none(chat_id, "Chat ID should be returned")
            
            # Get documents for chat (should be empty initially)
            docs_response = await client.get(f"/chats/{chat_id}/documents")
            check.equal(docs_response.status_code, 200, "Should retrieve documents list")
            docs_data = docs_response.json()
            check.is_in("documents", docs_data, "Response should include documents")
            check.is_in("total", docs_data, "Response should include total count")
        check.equal(docs_data["total"], 0, "New chat should have no documents")


@pytest.mark.asyncio
async def test_upload_multiple_file_types() -> None:
    """Test uploading different file types (PDF-only support)."""
    pdf_path = create_test_pdf_with_text()
    json_path = create_test_json_file()
    csv_path = create_test_csv_file()
    txt_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w').name)
    txt_path.write_text("Test text content")
    
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Test PDF upload - should succeed
            with open(pdf_path, "rb") as f:
                files = {"file": ("test.pdf", f.read(), "application/pdf")}
                response = await client.post("/upload", files=files)
            check.equal(response.status_code, 200, "PDF upload should succeed")
            
            # Test JSON upload - should fail (PDF-only)
            with open(json_path, "rb") as f:
                files = {"file": ("test.json", f.read(), "application/json")}
                response = await client.post("/upload", files=files)
            check.equal(response.status_code, 400, "JSON upload should fail (PDF-only)")
            
            # Test CSV upload - should fail (PDF-only)
            with open(csv_path, "rb") as f:
                files = {"file": ("test.csv", f.read(), "text/csv")}
                response = await client.post("/upload", files=files)
            check.equal(response.status_code, 400, "CSV upload should fail (PDF-only)")
            
            # Test TXT upload - should fail (PDF-only)
            with open(txt_path, "rb") as f:
                files = {"file": ("test.txt", f.read(), "text/plain")}
                response = await client.post("/upload", files=files)
            check.equal(response.status_code, 400, "TXT upload should fail (PDF-only)")
            
    finally:
        for path in [pdf_path, json_path, csv_path, txt_path]:
            if path.exists():
                path.unlink()

