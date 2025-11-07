import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Column, Integer, String, Text, create_engine, ForeignKey
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class ChatTable(Base):
    """Chat table model for PostgreSQL."""

    __tablename__ = "chats"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False, default="New Chat")
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    document_count = Column(Integer, default=0, nullable=False)
    document_ids = Column(JSON, default=list, nullable=True)


class MessageTable(Base):
    """Message table model for storing chat messages."""

    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)
    status = Column(String, nullable=True)  # 'pending', 'generating', 'complete', 'error'


class DocumentTable(Base):
    """Document table model for storing uploaded document metadata."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # 'pdf', 'csv', 'json', 'docx', 'txt', etc.
    file_size = Column(Integer, nullable=True)  # Size in bytes
    content = Column(Text, nullable=True)  # Extracted text content
    document_metadata = Column(JSON, nullable=True)  # Additional metadata (page count, etc.)
    created_at = Column(String, nullable=False)


# Database connection setup
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine.

    Returns:
        SQLAlchemy engine instance.
    """
    global _engine
    if _engine is None:
        db_url = settings.get_postgres_url()
        _engine = create_engine(db_url, echo=False)
        logger.info(f"Database engine created for: {db_url.split('@')[1] if '@' in db_url else 'database'}")
    return _engine


def get_session():
    """Get database session factory.

    Returns:
        Session factory function.
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def init_db() -> None:
    """Initialize database and create tables.

    Creates all tables defined in Base metadata.
    """
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def create_chat_in_db(title: str | None = None) -> dict[str, Any]:
    """Create a new chat in the database.

    Args:
        title: Optional chat title. Defaults to "New Chat".

    Returns:
        Dictionary containing chat data.
    """
    chat_id = str(uuid4())
    now = datetime.now(UTC).isoformat()
    
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        chat = ChatTable(
            id=chat_id,
            title=title or "New Chat",
            created_at=now,
            updated_at=now,
            message_count=0,
            document_count=0,
            document_ids=[],
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)
        logger.info(f"Created chat in database: {chat_id}")
        return {
            "id": chat.id,
            "title": chat.title,
            "created_at": chat.created_at,
            "updated_at": chat.updated_at,
            "message_count": chat.message_count,
            "document_count": chat.document_count,
            "document_ids": chat.document_ids or [],
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating chat in database: {e}")
        raise
    finally:
        db.close()


def get_chat_from_db(chat_id: str) -> dict[str, Any] | None:
    """Get chat from database by ID.

    Args:
        chat_id: Chat ID.

    Returns:
        Dictionary containing chat data or None if not found.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        chat = db.query(ChatTable).filter(ChatTable.id == chat_id).first()
        if chat:
            return {
                "id": chat.id,
                "title": chat.title,
                "created_at": chat.created_at,
                "updated_at": chat.updated_at,
                "message_count": chat.message_count,
                "document_count": chat.document_count,
                "document_ids": chat.document_ids or [],
            }
        return None
    except Exception as e:
        logger.error(f"Error getting chat from database: {e}")
        return None
    finally:
        db.close()


def list_chats_from_db() -> list[dict[str, Any]]:
    """List all chats from database, sorted by updated_at (newest first).

    Returns:
        List of chat dictionaries.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        chats = db.query(ChatTable).order_by(ChatTable.updated_at.desc()).all()
        return [
            {
                "id": chat.id,
                "title": chat.title,
                "created_at": chat.created_at,
                "updated_at": chat.updated_at,
                "message_count": chat.message_count,
                "document_count": chat.document_count,
                "document_ids": chat.document_ids or [],
            }
            for chat in chats
        ]
    except Exception as e:
        logger.error(f"Error listing chats from database: {e}")
        return []
    finally:
        db.close()


def update_chat_in_db(chat_id: str, **kwargs: Any) -> bool:
    """Update chat in database.

    Args:
        chat_id: Chat ID.
        **kwargs: Fields to update (title, message_count, document_count, etc.).

    Returns:
        True if chat was updated, False if not found.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        chat = db.query(ChatTable).filter(ChatTable.id == chat_id).first()
        if not chat:
            return False

        # Update fields
        if "title" in kwargs:
            chat.title = kwargs["title"]
        if "message_count" in kwargs:
            chat.message_count = kwargs["message_count"]
        if "document_count" in kwargs:
            chat.document_count = kwargs["document_count"]
        if "document_ids" in kwargs:
            chat.document_ids = kwargs["document_ids"]
        
        # Always update updated_at
        chat.updated_at = datetime.now(UTC).isoformat()

        db.commit()
        logger.info(f"Updated chat in database: {chat_id}")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating chat in database: {e}")
        return False
    finally:
        db.close()


def delete_chat_from_db(chat_id: str) -> bool:
    """Delete chat from database.

    Args:
        chat_id: Chat ID.

    Returns:
        True if chat was deleted, False if not found.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        # Delete all messages first
        db.query(MessageTable).filter(MessageTable.chat_id == chat_id).delete()
        
        chat = db.query(ChatTable).filter(ChatTable.id == chat_id).first()
        if not chat:
            return False

        db.delete(chat)
        db.commit()
        logger.info(f"Deleted chat from database: {chat_id}")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting chat from database: {e}")
        return False
    finally:
        db.close()


def save_message_to_db(chat_id: str, role: str, content: str, status: str | None = None) -> dict[str, Any]:
    """Save a message to the database.

    Args:
        chat_id: Chat ID.
        role: Message role ('user' or 'assistant').
        content: Message content.
        status: Optional status ('pending', 'generating', 'complete', 'error').

    Returns:
        Dictionary containing message data.
    """
    message_id = str(uuid4())
    now = datetime.now(UTC).isoformat()
    
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        message = MessageTable(
            id=message_id,
            chat_id=chat_id,
            role=role,
            content=content,
            created_at=now,
            status=status or "complete",
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        logger.debug(f"Saved message to database: {message_id} for chat {chat_id}")
        return {
            "id": message.id,
            "chat_id": message.chat_id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at,
            "status": message.status,
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving message to database: {e}")
        raise
    finally:
        db.close()


def get_messages_from_db(chat_id: str) -> list[dict[str, Any]]:
    """Get all messages for a chat, ordered by creation time.

    Args:
        chat_id: Chat ID.

    Returns:
        List of message dictionaries.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        messages = db.query(MessageTable).filter(MessageTable.chat_id == chat_id).order_by(MessageTable.created_at.asc()).all()
        return [
            {
                "id": msg.id,
                "chat_id": msg.chat_id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
                "status": msg.status,
            }
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Error getting messages from database: {e}")
        return []
    finally:
        db.close()


def update_message_in_db(message_id: str, content: str | None = None, status: str | None = None) -> bool:
    """Update a message in the database.

    Args:
        message_id: Message ID.
        content: Optional new content.
        status: Optional new status.

    Returns:
        True if message was updated, False if not found.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        message = db.query(MessageTable).filter(MessageTable.id == message_id).first()
        if not message:
            return False

        if content is not None:
            message.content = content
        if status is not None:
            message.status = status

        db.commit()
        logger.debug(f"Updated message in database: {message_id}")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating message in database: {e}")
        return False
    finally:
        db.close()


def save_document_to_db(
    chat_id: str,
    filename: str,
    file_type: str,
    content: str | None = None,
    file_size: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a document to the database.

    Args:
        chat_id: Chat ID.
        filename: Document filename.
        file_type: File type/extension (e.g., 'pdf', 'csv', 'txt').
        content: Extracted text content.
        file_size: File size in bytes.
        metadata: Additional metadata dictionary.

    Returns:
        Dictionary containing document data.
    """
    document_id = str(uuid4())
    now = datetime.now(UTC).isoformat()
    
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        # Clean content: remove null bytes and ensure it's a valid string
        cleaned_content: str | None = None
        if content:
            if isinstance(content, bytes):
                try:
                    cleaned_content = content.decode('utf-8', errors='replace')
                except Exception as e:
                    logger.warning(f"Failed to decode content bytes: {e}")
                    cleaned_content = None
            else:
                cleaned_content = str(content)
            
            # Remove null bytes (PostgreSQL text fields cannot contain NUL bytes)
            if cleaned_content:
                cleaned_content = cleaned_content.replace('\x00', '')
                # Remove other problematic control characters (keep only printable and common whitespace)
                cleaned_content = ''.join(char for char in cleaned_content if ord(char) >= 32 or char in '\n\r\t')
                # Limit size to avoid database issues
                if len(cleaned_content) > 100000:
                    cleaned_content = cleaned_content[:100000]
                    logger.warning(f"Content truncated to 100000 characters")
        
        document = DocumentTable(
            id=document_id,
            chat_id=chat_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            content=cleaned_content,
            metadata=metadata or {},
            created_at=now,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Update chat document_ids list
        chat = db.query(ChatTable).filter(ChatTable.id == chat_id).first()
        if chat:
            doc_ids = chat.document_ids or []
            if document_id not in doc_ids:
                doc_ids.append(document_id)
                chat.document_ids = doc_ids
                db.commit()
        
        logger.info(f"Saved document to database: {document_id} for chat {chat_id}")
        return {
            "id": document.id,
            "chat_id": document.chat_id,
            "filename": document.filename,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "content": document.content,
            "metadata": document.document_metadata or {},
            "created_at": document.created_at,
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving document to database: {e}")
        raise
    finally:
        db.close()


def get_documents_from_db(chat_id: str) -> list[dict[str, Any]]:
    """Get all documents for a chat, ordered by creation time.

    Args:
        chat_id: Chat ID.

    Returns:
        List of document dictionaries.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        documents = db.query(DocumentTable).filter(DocumentTable.chat_id == chat_id).order_by(DocumentTable.created_at.desc()).all()
        return [
            {
                "id": doc.id,
                "chat_id": doc.chat_id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "file_size": doc.file_size,
                "content": doc.content,
                "metadata": doc.document_metadata or {},
                "created_at": doc.created_at,
            }
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Error getting documents from database: {e}")
        return []
    finally:
        db.close()


def get_document_from_db(document_id: str) -> dict[str, Any] | None:
    """Get a document from database by ID.

    Args:
        document_id: Document ID.

    Returns:
        Dictionary containing document data or None if not found.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        document = db.query(DocumentTable).filter(DocumentTable.id == document_id).first()
        if document:
            return {
                "id": document.id,
                "chat_id": document.chat_id,
                "filename": document.filename,
                "file_type": document.file_type,
                "file_size": document.file_size,
                "content": document.content,
                "metadata": document.document_metadata or {},
                "created_at": document.created_at,
            }
        return None
    except Exception as e:
        logger.error(f"Error getting document from database: {e}")
        return None
    finally:
        db.close()


def delete_document_from_db(document_id: str) -> bool:
    """Delete a document from database.

    Args:
        document_id: Document ID.

    Returns:
        True if document was deleted, False if not found.
    """
    SessionLocal = get_session()
    db = SessionLocal()
    try:
        document = db.query(DocumentTable).filter(DocumentTable.id == document_id).first()
        if not document:
            return False
        
        chat_id = document.chat_id
        
        db.delete(document)
        db.commit()
        
        # Update chat document_ids list
        chat = db.query(ChatTable).filter(ChatTable.id == chat_id).first()
        if chat:
            doc_ids = chat.document_ids or []
            if document_id in doc_ids:
                doc_ids.remove(document_id)
                chat.document_ids = doc_ids
                db.commit()
        
        logger.info(f"Deleted document from database: {document_id}")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting document from database: {e}")
        return False
    finally:
        db.close()

