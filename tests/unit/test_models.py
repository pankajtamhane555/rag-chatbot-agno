"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from app.models import (
    PDFMetadata,
    ChatRequest,
    StreamingChunk,
    ErrorResponse,
    UploadResponse,
    StatusUpdate,
    Chat,
    ChatListResponse,
    CreateChatResponse,
)


def test_pdf_metadata_minimal():
    """Test PDFMetadata with minimal required fields."""
    metadata = PDFMetadata(page_count=5)
    assert metadata.page_count == 5
    assert metadata.title is None
    assert metadata.author is None


def test_pdf_metadata_full():
    """Test PDFMetadata with all fields."""
    metadata = PDFMetadata(
        page_count=10,
        title="Test PDF",
        author="Test Author",
        subject="Test Subject",
        creator="Test Creator",
        producer="Test Producer",
        creation_date="2024-01-01",
        modification_date="2024-01-02",
    )
    assert metadata.page_count == 10
    assert metadata.title == "Test PDF"
    assert metadata.author == "Test Author"


def test_pdf_metadata_required_field():
    """Test PDFMetadata requires page_count."""
    with pytest.raises(ValidationError):
        PDFMetadata()


def test_chat_request_minimal():
    """Test ChatRequest with minimal fields."""
    request = ChatRequest(message="Hello")
    assert request.message == "Hello"
    assert request.chat_id is None


def test_chat_request_with_chat_id():
    """Test ChatRequest with chat_id."""
    request = ChatRequest(message="Hello", chat_id="chat-123")
    assert request.message == "Hello"
    assert request.chat_id == "chat-123"


def test_chat_request_required_field():
    """Test ChatRequest requires message."""
    with pytest.raises(ValidationError):
        ChatRequest()


def test_streaming_chunk_default():
    """Test StreamingChunk with default values."""
    chunk = StreamingChunk(content="Hello")
    assert chunk.content == "Hello"
    assert chunk.done is False


def test_streaming_chunk_done():
    """Test StreamingChunk with done=True."""
    chunk = StreamingChunk(content="Final", done=True)
    assert chunk.content == "Final"
    assert chunk.done is True


def test_error_response_minimal():
    """Test ErrorResponse with minimal fields."""
    error = ErrorResponse(error="Something went wrong")
    assert error.error == "Something went wrong"
    assert error.detail is None


def test_error_response_with_detail():
    """Test ErrorResponse with detail."""
    error = ErrorResponse(error="Error", detail="Detailed error message")
    assert error.error == "Error"
    assert error.detail == "Detailed error message"


def test_upload_response_success():
    """Test UploadResponse for successful upload."""
    metadata = PDFMetadata(page_count=5)
    response = UploadResponse(
        success=True,
        message="Upload successful",
        filename="test.pdf",
        metadata=metadata,
    )
    assert response.success is True
    assert response.message == "Upload successful"
    assert response.filename == "test.pdf"
    assert response.metadata.page_count == 5


def test_upload_response_failure():
    """Test UploadResponse for failed upload."""
    response = UploadResponse(
        success=False,
        message="Upload failed",
        filename=None,
        metadata=None,
    )
    assert response.success is False
    assert response.filename is None
    assert response.metadata is None


def test_status_update_default():
    """Test StatusUpdate with default values."""
    status = StatusUpdate(status="Processing")
    assert status.status == "Processing"
    assert status.message is None
    assert status.done is False


def test_status_update_full():
    """Test StatusUpdate with all fields."""
    status = StatusUpdate(
        status="Complete",
        message="Processing complete",
        done=True,
    )
    assert status.status == "Complete"
    assert status.message == "Processing complete"
    assert status.done is True


def test_chat_minimal():
    """Test Chat with minimal fields."""
    chat = Chat(
        id="chat-123",
        title="Test Chat",
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )
    assert chat.id == "chat-123"
    assert chat.title == "Test Chat"
    assert chat.message_count == 0
    assert chat.document_count == 0


def test_chat_full():
    """Test Chat with all fields."""
    chat = Chat(
        id="chat-123",
        title="Test Chat",
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
        message_count=10,
        document_count=3,
    )
    assert chat.id == "chat-123"
    assert chat.message_count == 10
    assert chat.document_count == 3


def test_chat_list_response():
    """Test ChatListResponse."""
    chats = [
        Chat(
            id="chat-1",
            title="Chat 1",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        ),
        Chat(
            id="chat-2",
            title="Chat 2",
            created_at="2024-01-02T00:00:00",
            updated_at="2024-01-02T00:00:00",
        ),
    ]
    response = ChatListResponse(chats=chats, total=2)
    assert len(response.chats) == 2
    assert response.total == 2


def test_create_chat_response():
    """Test CreateChatResponse."""
    chat = Chat(
        id="chat-123",
        title="New Chat",
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )
    response = CreateChatResponse(chat=chat, success=True)
    assert response.chat.id == "chat-123"
    assert response.success is True


def test_create_chat_response_default_success():
    """Test CreateChatResponse defaults success to True."""
    chat = Chat(
        id="chat-123",
        title="New Chat",
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )
    response = CreateChatResponse(chat=chat)
    assert response.success is True


def test_model_serialization():
    """Test model serialization to JSON."""
    metadata = PDFMetadata(page_count=5, title="Test")
    json_str = metadata.model_dump_json()
    assert "page_count" in json_str
    assert "title" in json_str


def test_model_validation_error():
    """Test model validation with invalid data."""
    # Pydantic allows empty strings by default, so we test with missing required field
    with pytest.raises(ValidationError):
        ChatRequest()  # Missing required 'message' field


def test_pdf_metadata_model_dump():
    """Test PDFMetadata model_dump method."""
    metadata = PDFMetadata(page_count=10, title="Test")
    data = metadata.model_dump()
    assert data["page_count"] == 10
    assert data["title"] == "Test"

