"""Shared pytest fixtures and configuration."""

import sys
from pathlib import Path
from types import ModuleType

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Mock sqlalchemy before any app modules import it
try:
    import sqlalchemy
    import sqlalchemy.orm
except ImportError:
    # Create mock sqlalchemy module if not installed (for local testing)
    mock_sqlalchemy = ModuleType("sqlalchemy")
    mock_sqlalchemy.create_engine = lambda *args, **kwargs: None
    mock_sqlalchemy.Column = lambda *args, **kwargs: None
    mock_sqlalchemy.Integer = lambda *args, **kwargs: None
    mock_sqlalchemy.String = lambda *args, **kwargs: None
    mock_sqlalchemy.Text = lambda *args, **kwargs: None
    mock_sqlalchemy.ForeignKey = lambda *args, **kwargs: None
    mock_sqlalchemy.JSON = lambda *args, **kwargs: None
    sys.modules["sqlalchemy"] = mock_sqlalchemy
    
    # Create mock sqlalchemy.orm submodule
    mock_orm = ModuleType("sqlalchemy.orm")
    mock_orm.declarative_base = lambda: type('Base', (), {})
    mock_orm.sessionmaker = lambda *args, **kwargs: lambda: None
    sys.modules["sqlalchemy.orm"] = mock_orm

import pytest
from unittest.mock import patch, MagicMock

# Import Settings after sqlalchemy is mocked
try:
    from app.config import Settings
except ImportError:
    # If import fails, create a minimal Settings mock
    from pydantic_settings import BaseSettings
    Settings = BaseSettings  # Fallback


@pytest.fixture(autouse=True)
def mock_database(monkeypatch):
    """Automatically mock database operations for all tests to prevent real DB access."""
    # Mock database engine and session creation
    mock_engine = MagicMock()
    mock_session = MagicMock()
    
    # Setup context manager behavior for session
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=None)
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.refresh = MagicMock()
    mock_session.query = MagicMock()
    mock_session.delete = MagicMock()
    mock_session.close = MagicMock()
    mock_session.rollback = MagicMock()
    
    # Mock session factory
    mock_session_factory = MagicMock(return_value=mock_session)
    
    # Mock database functions
    def mock_get_engine():
        return mock_engine
    
    def mock_get_session():
        return mock_session_factory
    
    def mock_init_db():
        pass  # No-op for tests - prevents actual DB initialization
    
    # Mock all database CRUD functions to return safe defaults
    def mock_create_chat_in_db(title: str | None = None):
        import uuid
        chat_id = str(uuid.uuid4())
        return {
            "id": chat_id,
            "title": title or "New Chat",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "message_count": 0,
            "document_count": 0,
            "document_ids": [],
        }
    
    def mock_get_chat_from_db(chat_id: str):
        return {
            "id": chat_id,
            "title": "Test Chat",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "message_count": 0,
            "document_count": 0,
            "document_ids": [],
        }
    
    def mock_list_chats_from_db():
        return []
    
    def mock_update_chat_in_db(chat_id: str, **kwargs):
        return True
    
    def mock_delete_chat_from_db(chat_id: str):
        return True
    
    def mock_save_message_to_db(chat_id: str, role: str, content: str, status: str | None = None):
        return {
            "id": "test-message-id",
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "created_at": "2024-01-01T00:00:00",
            "status": status or "complete",
        }
    
    def mock_get_messages_from_db(chat_id: str):
        return []
    
    def mock_update_message_in_db(message_id: str, content: str | None = None, status: str | None = None):
        return True
    
    # Track documents in memory for integration tests
    _saved_documents: dict[str, list[dict]] = {}
    
    def mock_save_document_to_db(chat_id: str, filename: str, file_type: str, content: str | None = None, file_size: int | None = None, metadata: dict | None = None):
        import uuid
        doc_id = str(uuid.uuid4())
        document = {
            "id": doc_id,
            "chat_id": chat_id,
            "filename": filename,
            "file_type": file_type,
            "file_size": file_size,
            "content": content,
            "metadata": metadata or {},
            "created_at": "2024-01-01T00:00:00",
        }
        # Store document in memory for integration tests
        if chat_id not in _saved_documents:
            _saved_documents[chat_id] = []
        _saved_documents[chat_id].append(document)
        return document
    
    def mock_get_documents_from_db(chat_id: str):
        # Return saved documents for this chat, or empty list if none
        return _saved_documents.get(chat_id, [])
    
    def mock_get_document_from_db(document_id: str):
        return None
    
    def mock_delete_document_from_db(document_id: str):
        return True
    
    # Patch database module functions
    # Note: app.database may already be imported, so we patch it directly
    monkeypatch.setattr("app.database.get_engine", mock_get_engine)
    monkeypatch.setattr("app.database.get_session", mock_get_session)
    monkeypatch.setattr("app.database.init_db", mock_init_db)
    monkeypatch.setattr("app.database.create_chat_in_db", mock_create_chat_in_db)
    monkeypatch.setattr("app.database.get_chat_from_db", mock_get_chat_from_db)
    monkeypatch.setattr("app.database.list_chats_from_db", mock_list_chats_from_db)
    monkeypatch.setattr("app.database.update_chat_in_db", mock_update_chat_in_db)
    monkeypatch.setattr("app.database.delete_chat_from_db", mock_delete_chat_from_db)
    monkeypatch.setattr("app.database.save_message_to_db", mock_save_message_to_db)
    monkeypatch.setattr("app.database.get_messages_from_db", mock_get_messages_from_db)
    monkeypatch.setattr("app.database.update_message_in_db", mock_update_message_in_db)
    monkeypatch.setattr("app.database.save_document_to_db", mock_save_document_to_db)
    monkeypatch.setattr("app.database.get_documents_from_db", mock_get_documents_from_db)
    monkeypatch.setattr("app.database.get_document_from_db", mock_get_document_from_db)
    monkeypatch.setattr("app.database.delete_document_from_db", mock_delete_document_from_db)
    
    # Mock create_engine to prevent actual database connections
    with patch("app.database.create_engine") as mock_create_engine:
        mock_create_engine.return_value = mock_engine
        yield mock_engine


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with default values."""
    return Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        openai_embedding_dimensions=1536,
        openai_timeout=300,
        log_level="DEBUG",
    )


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary directory for test PDFs."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    return pdf_dir

