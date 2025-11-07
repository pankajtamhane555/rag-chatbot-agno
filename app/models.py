"""Pydantic models for agent requests and responses."""

from pydantic import BaseModel, Field


class PDFMetadata(BaseModel):
    """PDF metadata model."""

    page_count: int = Field(..., description="Number of pages in PDF")
    title: str | None = Field(None, description="PDF title")
    author: str | None = Field(None, description="PDF author")
    subject: str | None = Field(None, description="PDF subject")
    creator: str | None = Field(None, description="PDF creator application")
    producer: str | None = Field(None, description="PDF producer application")
    creation_date: str | None = Field(None, description="PDF creation date")
    modification_date: str | None = Field(None, description="PDF modification date")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User message to send to agent")
    chat_id: str | None = Field(None, description="Optional chat ID. If not provided, creates new chat.")


class StreamingChunk(BaseModel):
    """Single chunk in streaming response."""

    content: str = Field(..., description="Chunk content")
    done: bool = Field(False, description="Whether this is the final chunk")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional error details")


class UploadResponse(BaseModel):
    """Response model for PDF upload."""

    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Status message")
    filename: str | None = Field(None, description="Uploaded filename")
    metadata: PDFMetadata | None = Field(None, description="PDF metadata if parsed")
    chat_id: str | None = Field(None, description="Chat ID associated with the upload")


class StatusUpdate(BaseModel):
    """Status update model for async status signals."""

    status: str = Field(..., description="Current status (e.g., 'Analyzing', 'Searching', 'Generating')")
    message: str | None = Field(None, description="Optional status message")
    done: bool = Field(False, description="Whether this is the final status update")


class Chat(BaseModel):
    """Chat/conversation model."""

    id: str = Field(..., description="Unique chat ID")
    title: str = Field(..., description="Chat title (first message or auto-generated)")
    created_at: str = Field(..., description="Chat creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    message_count: int = Field(0, description="Number of messages in chat")
    document_count: int = Field(0, description="Number of documents uploaded to this chat")


class ChatListResponse(BaseModel):
    """Response model for chat list."""

    chats: list[Chat] = Field(..., description="List of chats")
    total: int = Field(..., description="Total number of chats")


class CreateChatResponse(BaseModel):
    """Response model for creating a new chat."""

    chat: Chat = Field(..., description="Created chat")
    success: bool = Field(True, description="Whether creation was successful")

