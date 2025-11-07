import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.agent import AgentManager, FileParseError
from app.config import settings
from app.models import (
    Chat,
    ChatListResponse,
    ChatRequest,
    CreateChatResponse,
    ErrorResponse,
    PDFMetadata,
    StatusUpdate,
    StreamingChunk,
    UploadResponse,
)
from app.database import (
    create_chat_in_db,
    delete_chat_from_db,
    delete_document_from_db,
    get_chat_from_db,
    get_document_from_db,
    get_documents_from_db,
    get_messages_from_db,
    init_db,
    list_chats_from_db,
    save_document_to_db,
    save_message_to_db,
    update_chat_in_db,
    update_message_in_db,
)
from app.validation import ValidationError, validate_uploaded_file

logger = logging.getLogger(__name__)

_manager = AgentManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    try:
        settings.validate_openai_config()
        logger.info("OpenAI configuration validated")
        logger.info(f"Using LLM model: {settings.openai_model}")
        logger.info(f"Using embedding model: {settings.openai_embedding_model} (dimensions: {settings.openai_embedding_dimensions})")
    except ValueError as e:
        logger.error(f"OpenAI configuration validation failed: {e}")
        raise

    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")

    yield

    # Shutdown (if needed in the future)
    logger.info("Shutting down")


app = FastAPI(
    title="RAG Chatbot API",
    description="Document QA Chatbot with streaming responses. Upload PDFs and ask questions about them.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_agent_response(message: str, chat_id: str | None = None) -> AsyncGenerator[str, None]:
    """Stream agent response with status updates."""
    assistant_message_id: str | None = None
    try:
        if chat_id:
            chat = get_chat_from_db(chat_id)
            if not chat:
                raise ValueError(f"Chat {chat_id} not found")
        else:
            chat = create_chat_in_db()
            chat_id = chat["id"]

        save_message_to_db(chat_id, "user", message, "complete")

        if chat.get("title") == "New Chat" and message:
            title = message[:50] + ("..." if len(message) > 50 else "")
            update_chat_in_db(chat_id, title=title)

        current_count = chat.get("message_count", 0)
        update_chat_in_db(chat_id, message_count=current_count + 1)

        assistant_msg = save_message_to_db(chat_id, "assistant", "", "generating")
        assistant_message_id = assistant_msg["id"]

        status_update = StatusUpdate(status="Analyzing", message="Analyzing your request...", done=False)
        yield f"status: {status_update.model_dump_json()}\n\n"
        await asyncio.sleep(0.2)

        agent = _manager.create_agent(chat_id=chat_id)
        session = chat_id

        # Check if chat has documents (for status updates)
        if _manager.has_documents_for_chat(chat_id):
            status_update = StatusUpdate(status="Searching", message="Searching uploaded documents...", done=False)
            yield f"status: {status_update.model_dump_json()}\n\n"
            await asyncio.sleep(0.3)

        status_update = StatusUpdate(status="Generating", message="Generating response...", done=False)
        yield f"status: {status_update.model_dump_json()}\n\n"
        await asyncio.sleep(0.2)

        loop = asyncio.get_event_loop()
        response_text = ""
        async for chunk in _stream_agent_run(agent, session, message, loop, chat_id):
            response_text += chunk
            chunk_data = StreamingChunk(content=chunk, done=False)
            yield f"data: {chunk_data.model_dump_json()}\n\n"
            if assistant_message_id:
                update_message_in_db(assistant_message_id, content=response_text, status="generating")

        if assistant_message_id:
            update_message_in_db(assistant_message_id, content=response_text, status="complete")

        final_chunk = StreamingChunk(content="", done=True)
        yield f"data: {final_chunk.model_dump_json()}\n\n"

        status_update = StatusUpdate(status="Complete", message="Response complete", done=True)
        yield f"status: {status_update.model_dump_json()}\n\n"

    except ValueError as e:
        logger.exception(f"Validation error in stream_agent_response: {e}")
        if assistant_message_id:
            update_message_in_db(assistant_message_id, status="error")
        status_update = StatusUpdate(status="Error", message="Validation error occurred", done=True)
        yield f"status: {status_update.model_dump_json()}\n\n"
        error_chunk = StreamingChunk(
            content=f"Validation error: Please check your input and try again.",
            done=True,
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
    except ConnectionError as e:
        logger.exception(f"Connection error in stream_agent_response: {e}")
        status_update = StatusUpdate(status="Error", message="Connection error", done=True)
        yield f"status: {status_update.model_dump_json()}\n\n"
        error_chunk = StreamingChunk(
            content=f"Connection error: Unable to connect to the agent service. Please try again later.",
            done=True,
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
    except TimeoutError as e:
        logger.exception(f"Timeout error in stream_agent_response: {e}")
        status_update = StatusUpdate(status="Error", message="Request timeout", done=True)
        yield f"status: {status_update.model_dump_json()}\n\n"
        error_chunk = StreamingChunk(
            content=f"Timeout: The request took too long. Please try again with a shorter message.",
            done=True,
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
    except Exception as e:
        logger.exception(f"Unexpected error streaming agent response: {e}")
        if assistant_message_id:
            update_message_in_db(assistant_message_id, status="error")
        status_update = StatusUpdate(status="Error", message="Unexpected error", done=True)
        yield f"status: {status_update.model_dump_json()}\n\n"
        error_chunk = StreamingChunk(
            content=f"An error occurred while processing your request. Please try again.",
            done=True,
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"


async def _stream_agent_run(
    agent: Any, session: Any, message: str, loop: asyncio.AbstractEventLoop, chat_id: str | None = None
) -> AsyncGenerator[str, None]:
    """Run agent and stream response chunks.
    
    With search_knowledge=True, the agent automatically searches the knowledge base
    when needed, so we don't need to manually enhance the message.
    """
    try:
        logger.info(f"Running agent for chat {chat_id}")
        # Agent with search_knowledge=True will automatically search knowledge base
        response = await loop.run_in_executor(
            None,
            lambda: agent.run(message, session_id=session),
        )

        response_content = ""
        if hasattr(response, 'content') and response.content is not None:
            response_content = str(response.content)
        elif hasattr(response, 'messages') and response.messages:
            for msg in reversed(response.messages):
                if hasattr(msg, 'role') and getattr(msg, 'role') == 'assistant':
                    if hasattr(msg, 'content'):
                        response_content = str(msg.content)
                        break

        if not response_content:
            response_content = "I apologize, but I couldn't generate a response. Please try again."

        words = response_content.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)

    except Exception as e:
        logger.exception(f"Error in agent run: {e}")
        raise


@app.post(
    "/chat/stream",
    response_class=StreamingResponse,
    summary="Stream chat response",
    description="Stream agent responses token-by-token with async status signals",
    responses={
        200: {
            "description": "Streaming response with Server-Sent Events (SSE)",
            "content": {"text/event-stream": {}},
        },
        400: {"description": "Bad request - invalid input", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream chat response using Server-Sent Events."""
    try:
        if not request.message:
            logger.warning("Empty message received")
            raise HTTPException(
                status_code=400,
                detail="Message is required",
            )

        if not request.message.strip():
            logger.warning("Whitespace-only message received")
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty or only whitespace",
            )

        if len(request.message) > 10000:
            logger.warning(f"Message too long: {len(request.message)} characters")
            raise HTTPException(
                status_code=400,
                detail="Message is too long. Maximum 10,000 characters allowed.",
            )

        if request.chat_id and len(request.chat_id) > 100:
            logger.warning(f"Invalid chat_id format: {len(request.chat_id)} characters")
            raise HTTPException(
                status_code=400,
                detail="Invalid chat ID format",
            )

        logger.info(f"Streaming chat request: {len(request.message)} characters, chat_id={request.chat_id}")
        return StreamingResponse(
            stream_agent_response(request.message, request.chat_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.exception(f"Validation error in chat_stream: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}",
        )
    except ConnectionError as e:
        logger.exception(f"Connection error in chat_stream: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please try again later.",
        )
    except TimeoutError as e:
        logger.exception(f"Timeout error in chat_stream: {e}")
        raise HTTPException(
            status_code=504,
            detail="Request timeout. The agent took too long to respond.",
        )
    except Exception as e:
        logger.exception(f"Unexpected error in chat_stream endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred. Please try again later.",
        )


@app.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and parse PDF",
    description="Upload a PDF file, parse it, and make it available to the agent for a specific chat",
    responses={
        200: {"description": "PDF uploaded and parsed successfully", "model": UploadResponse},
        400: {"description": "Validation error - invalid file or missing chat_id", "model": ErrorResponse},
        404: {"description": "Chat not found", "model": ErrorResponse},
        422: {"description": "PDF parsing failed - file may be corrupted", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload (max 10MB)"),
    chat_id: str | None = Form(None, description="Chat ID to associate PDF with. If not provided, creates new chat."),
) -> UploadResponse:
    """Upload and parse file."""
    logger.info(f"File upload request: {file.filename}, chat_id={chat_id}")

    if chat_id:
        chat = get_chat_from_db(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    else:
        chat = create_chat_in_db()
        chat_id = chat["id"]

    try:
        validate_uploaded_file(
            filename=file.filename or "",
            file_size=file.size if hasattr(file, "size") else None,
        )
    except ValidationError as e:
        logger.warning(f"File validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    try:
        content = await file.read()
    except Exception as e:
        logger.exception(f"Error reading file: {e}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded file")

    try:
        validate_uploaded_file(filename=file.filename or "", file_content=content)
    except ValidationError as e:
        logger.warning(f"Content validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    temp_path: Path | None = None
    try:
        file_ext = Path(file.filename or "unknown").suffix or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_path = Path(temp_file.name)
            temp_path.write_bytes(content)

        extracted_content: str | None = None
        try:
            extracted_content = _manager.extract_file_content(str(temp_path))

            if extracted_content and isinstance(extracted_content, bytes):
                try:
                    extracted_content = extracted_content.decode('utf-8', errors='replace')
                except Exception:
                    extracted_content = None

            if extracted_content:
                extracted_content = extracted_content.replace('\x00', '')
                extracted_content = ''.join(char for char in extracted_content if ord(char) >= 32 or char in '\n\r\t')

                if len(extracted_content) > 100000:
                    logger.warning(f"Content truncated from {len(extracted_content)} to 100000")
                    extracted_content = extracted_content[:100000]
        except Exception as e:
            logger.warning(f"Failed to extract content: {e}", exc_info=True)
            extracted_content = None

        try:
            await _manager.add_file_knowledge(
                chat_id=chat_id,
                file_path=str(temp_path),
                filename=file.filename or "unknown",
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
        except FileParseError as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse file: {str(e)}")
        except Exception as e:
            error_msg = str(e)
            if "expected" in error_msg.lower() and "dimensions" in error_msg.lower() and "not" in error_msg.lower():
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Dimension mismatch: {error_msg}. "
                        "Drop the 'ai.knowledge_documents' table and it will be recreated automatically."
                    ),
                )
            raise HTTPException(status_code=500, detail=f"Failed to add file to knowledge base: {str(e)}")

        try:
            metadata = _manager.extract_file_metadata(str(temp_path))
            if metadata.page_count > 0:
                message = f"PDF uploaded and parsed successfully. {metadata.page_count} pages extracted."
            else:
                message = "File uploaded and parsed successfully."
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
            from app.models import PDFMetadata
            metadata = PDFMetadata(page_count=0, title=Path(file.filename or "unknown").stem or None)
            message = "File uploaded and parsed successfully."

        file_ext = Path(file.filename or "unknown").suffix.lstrip(".") or "txt"
        try:
            save_document_to_db(
                chat_id=chat_id,
                filename=file.filename or "unknown",
                file_type=file_ext,
                content=extracted_content,
                file_size=len(content),
                metadata=metadata.model_dump() if hasattr(metadata, "model_dump") else {},
            )
        except Exception as e:
            logger.error(f"Failed to save document to database: {e}", exc_info=True)

        logger.info(f"File uploaded successfully: {file.filename}")
        return UploadResponse(
            success=True,
            message=message,
            filename=file.filename,
            metadata=metadata,
            chat_id=chat_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in upload endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the PDF. Please try again.",
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.post(
    "/chats",
    response_model=CreateChatResponse,
    summary="Create a new chat",
    description="Create a new chat conversation",
    responses={
        200: {"description": "Chat created successfully", "model": CreateChatResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def create_new_chat(title: str | None = None) -> CreateChatResponse:
    """Create a new chat."""
    try:
        chat_dict = create_chat_in_db(title=title)
        chat = Chat(**chat_dict)
        return CreateChatResponse(chat=chat, success=True)
    except Exception as e:
        logger.exception(f"Error creating chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to create chat")


@app.get(
    "/chats",
    response_model=ChatListResponse,
    summary="List all chats",
    description="Get a list of all chats",
    responses={200: {"description": "List of chats", "model": ChatListResponse}},
)
async def get_chats() -> ChatListResponse:
    """List all chats."""
    chats_list = list_chats_from_db()
    chats = [Chat(**chat_dict) for chat_dict in chats_list]
    return ChatListResponse(chats=chats, total=len(chats))


@app.get(
    "/chats/{chat_id}",
    response_model=Chat,
    summary="Get chat by ID",
    description="Get details of a specific chat",
    responses={
        200: {"description": "Chat details", "model": Chat},
        404: {"description": "Chat not found", "model": ErrorResponse},
    },
)
async def get_chat_by_id(chat_id: str) -> Chat:
    """Get chat by ID."""
    chat_dict = get_chat_from_db(chat_id)
    if not chat_dict:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return Chat(**chat_dict)


@app.get(
    "/chats/{chat_id}/messages",
    summary="Get messages for a chat",
    description="Retrieve all messages for a specific chat, ordered by creation time.",
)
async def get_chat_messages(chat_id: str) -> dict[str, Any]:
    """Get all messages for a chat."""
    chat = get_chat_from_db(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    
    messages = get_messages_from_db(chat_id)
    return {"chat_id": chat_id, "messages": messages, "total": len(messages)}


@app.get(
    "/chats/{chat_id}/documents",
    summary="Get documents for a chat",
    description="Retrieve all documents uploaded to a specific chat.",
)
async def get_chat_documents(chat_id: str) -> dict[str, Any]:
    """Get all documents for a chat."""
    chat = get_chat_from_db(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    
    documents = get_documents_from_db(chat_id)
    return {"chat_id": chat_id, "documents": documents, "total": len(documents)}


@app.get(
    "/documents/{document_id}",
    summary="Get document by ID",
    description="Retrieve a specific document with its content.",
)
async def get_document_by_id(document_id: str) -> dict[str, Any]:
    """Get a document by ID."""
    document = get_document_from_db(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return document


@app.delete(
    "/documents/{document_id}",
    summary="Delete a document",
    description="Delete a document from the database.",
)
async def delete_document_by_id(document_id: str) -> dict[str, str]:
    """Delete a document by ID."""
    deleted = delete_document_from_db(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return {"message": "Document deleted successfully"}


@app.delete(
    "/chats/{chat_id}",
    summary="Delete a chat",
    description="Delete a chat and all its associated data",
    responses={
        200: {"description": "Chat deleted successfully"},
        404: {"description": "Chat not found", "model": ErrorResponse},
    },
)
async def delete_chat_by_id(chat_id: str) -> dict[str, str]:
    """Delete a chat."""
    deleted = delete_chat_from_db(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")
    return {"message": f"Chat {chat_id} deleted successfully"}


@app.get(
    "/health",
    summary="Health check",
    description="Check if the API service is running and healthy",
    responses={200: {"description": "Service is healthy"}},
)
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-chatbot"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.exception(f"Unhandled exception: {exc}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
        ).model_dump(),
    )
