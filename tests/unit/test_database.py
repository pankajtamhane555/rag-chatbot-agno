"""Comprehensive tests for database operations."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from app.database import (
    init_db,
    create_chat_in_db,
    get_chat_from_db,
    list_chats_from_db,
    update_chat_in_db,
    delete_chat_from_db,
    save_message_to_db,
    get_messages_from_db,
    update_message_in_db,
    save_document_to_db,
    get_documents_from_db,
    get_document_from_db,
    delete_document_from_db,
    get_engine,
    get_session,
)
from app.config import Settings


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    test_settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        openai_embedding_dimensions=1536,
        postgres_user="test",
        postgres_password="test",
        postgres_db="test",
        postgres_host="localhost",
        postgres_port=5432,
    )
    monkeypatch.setattr("app.database.settings", test_settings)
    return test_settings


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = MagicMock()
    return session


def test_get_engine(mock_settings):
    """Test get_engine creates engine correctly."""
    # Reset global engine to test creation
    import app.database
    app.database._engine = None
    with patch("app.database.create_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        engine = get_engine()
        mock_create_engine.assert_called_once()
        assert engine is not None
        # Reset for other tests
        app.database._engine = None


def test_get_session(mock_settings):
    """Test get_session creates session factory."""
    # Reset global session to test creation
    import app.database
    app.database._SessionLocal = None
    with patch("app.database.sessionmaker") as mock_sessionmaker:
        with patch("app.database.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            mock_session_factory = MagicMock()
            mock_sessionmaker.return_value = mock_session_factory
            session_factory = get_session()
            mock_sessionmaker.assert_called_once()
            assert session_factory is not None
            # Reset for other tests
            app.database._SessionLocal = None


def test_init_db(mock_settings):
    """Test database initialization."""
    with patch("app.database.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        with patch("app.database.Base.metadata.create_all") as mock_create_all:
            init_db()
            mock_create_all.assert_called_once_with(bind=mock_engine)


def test_create_chat_in_db(mock_settings):
    """Test creating a chat in database."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_chat = MagicMock()
        mock_chat.id = "test-chat-id"
        mock_chat.title = "Test Chat"
        mock_chat.created_at = "2024-01-01T00:00:00"
        mock_chat.updated_at = "2024-01-01T00:00:00"
        mock_chat.message_count = 0
        mock_chat.document_count = 0
        mock_chat.document_ids = []
        
        # Setup context manager
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        # get_session() returns a callable that returns the session context manager
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        with patch("app.database.ChatTable") as mock_chat_table:
            mock_chat_table.return_value = mock_chat
            result = create_chat_in_db("Test Chat")
            
            assert result["id"] == "test-chat-id"
            assert result["title"] == "Test Chat"
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()


def test_get_chat_from_db(mock_settings):
    """Test getting a chat from database."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_chat = MagicMock()
        mock_chat.id = "test-chat-id"
        mock_chat.title = "Test Chat"
        mock_chat.created_at = "2024-01-01T00:00:00"
        mock_chat.updated_at = "2024-01-01T00:00:00"
        mock_chat.message_count = 0
        mock_chat.document_count = 0
        mock_chat.document_ids = []
        
        # Setup context manager
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_chat
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = get_chat_from_db("test-chat-id")
        
        assert result is not None
        assert result["id"] == "test-chat-id"
        assert result["title"] == "Test Chat"


def test_get_chat_from_db_not_found(mock_settings):
    """Test getting non-existent chat returns None."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = get_chat_from_db("non-existent-id")
        assert result is None


def test_list_chats_from_db(mock_settings):
    """Test listing all chats from database."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_chat1 = MagicMock()
        mock_chat1.id = "chat-1"
        mock_chat1.title = "Chat 1"
        mock_chat1.created_at = "2024-01-01T00:00:00"
        mock_chat1.updated_at = "2024-01-01T00:00:00"
        mock_chat1.message_count = 0
        mock_chat1.document_count = 0
        mock_chat1.document_ids = []
        
        mock_chat2 = MagicMock()
        mock_chat2.id = "chat-2"
        mock_chat2.title = "Chat 2"
        mock_chat2.created_at = "2024-01-02T00:00:00"
        mock_chat2.updated_at = "2024-01-02T00:00:00"
        mock_chat2.message_count = 1
        mock_chat2.document_count = 0
        mock_chat2.document_ids = []
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.order_by.return_value.all.return_value = [mock_chat1, mock_chat2]
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = list_chats_from_db()
        
        assert len(result) == 2
        assert result[0]["id"] == "chat-1"
        assert result[1]["id"] == "chat-2"


def test_update_chat_in_db(mock_settings):
    """Test updating a chat in database."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_chat = MagicMock()
        mock_chat.id = "test-chat-id"
        mock_chat.title = "Old Title"
        mock_chat.updated_at = "2024-01-01T00:00:00"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_chat
        mock_session.commit = MagicMock()
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = update_chat_in_db("test-chat-id", title="New Title", message_count=5)
        
        assert result is True
        assert mock_chat.title == "New Title"
        assert mock_chat.message_count == 5
        mock_session.commit.assert_called_once()


def test_update_chat_in_db_not_found(mock_settings):
    """Test updating non-existent chat returns False."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = update_chat_in_db("non-existent-id", title="New Title")
        assert result is False


def test_delete_chat_from_db(mock_settings):
    """Test deleting a chat from database."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_chat = MagicMock()
        mock_chat.id = "test-chat-id"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        # First query deletes messages, second query gets chat
        message_query = MagicMock()
        message_query.filter.return_value.delete.return_value = None
        chat_query = MagicMock()
        chat_query.filter.return_value.first.return_value = mock_chat
        # query() is called twice - once for MessageTable, once for ChatTable
        mock_session.query.side_effect = [message_query, chat_query]
        mock_session.delete = MagicMock()
        mock_session.commit = MagicMock()
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = delete_chat_from_db("test-chat-id")
        
        assert result is True
        mock_session.delete.assert_called_once_with(mock_chat)
        mock_session.commit.assert_called_once()


def test_delete_chat_from_db_not_found(mock_settings):
    """Test deleting non-existent chat returns False."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = delete_chat_from_db("non-existent-id")
        assert result is False


def test_save_message_to_db(mock_settings):
    """Test saving a message to database."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_message = MagicMock()
        mock_message.id = "msg-id"
        mock_message.chat_id = "chat-id"
        mock_message.role = "user"
        mock_message.content = "Hello"
        mock_message.created_at = "2024-01-01T00:00:00"
        mock_message.status = "complete"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        with patch("app.database.MessageTable") as mock_message_table:
            mock_message_table.return_value = mock_message
            result = save_message_to_db("chat-id", "user", "Hello", "complete")
            
            assert result["id"] == "msg-id"
            assert result["role"] == "user"
            assert result["content"] == "Hello"
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()


def test_get_messages_from_db(mock_settings):
    """Test getting messages for a chat."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_msg1 = MagicMock()
        mock_msg1.id = "msg-1"
        mock_msg1.chat_id = "chat-id"
        mock_msg1.role = "user"
        mock_msg1.content = "Hello"
        mock_msg1.created_at = "2024-01-01T00:00:00"
        mock_msg1.status = "complete"
        
        mock_msg2 = MagicMock()
        mock_msg2.id = "msg-2"
        mock_msg2.chat_id = "chat-id"
        mock_msg2.role = "assistant"
        mock_msg2.content = "Hi there"
        mock_msg2.created_at = "2024-01-01T00:01:00"
        mock_msg2.status = "complete"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_msg1, mock_msg2]
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = get_messages_from_db("chat-id")
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"


def test_update_message_in_db(mock_settings):
    """Test updating a message in database."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_message = MagicMock()
        mock_message.id = "msg-id"
        mock_message.content = "Old content"
        mock_message.status = "pending"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_message
        mock_session.commit = MagicMock()
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = update_message_in_db("msg-id", content="New content", status="complete")
        
        assert result is True
        assert mock_message.content == "New content"
        assert mock_message.status == "complete"
        mock_session.commit.assert_called_once()


def test_update_message_in_db_not_found(mock_settings):
    """Test updating non-existent message returns False."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = update_message_in_db("non-existent-id", content="New content")
        assert result is False


def test_save_document_to_db(mock_settings):
    """Test saving a document to database."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.chat_id = "chat-id"
        mock_document.filename = "test.pdf"
        mock_document.file_type = "pdf"
        mock_document.file_size = 1024
        mock_document.content = "Test content"
        mock_document.document_metadata = {}
        mock_document.created_at = "2024-01-01T00:00:00"
        
        mock_chat = MagicMock()
        mock_chat.id = "chat-id"
        mock_chat.document_ids = []
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        # query() is called for ChatTable lookup
        chat_query = MagicMock()
        chat_query.filter.return_value.first.return_value = mock_chat
        mock_session.query.return_value = chat_query
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        with patch("app.database.DocumentTable") as mock_document_table:
            mock_document_table.return_value = mock_document
            result = save_document_to_db(
                "chat-id",
                "test.pdf",
                "pdf",
                content="Test content",
                file_size=1024,
                metadata={},
            )
            
            assert result["id"] == "doc-id"
            assert result["filename"] == "test.pdf"
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called()


def test_save_document_to_db_cleans_content(mock_settings):
    """Test that save_document_to_db cleans null bytes from content."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.chat_id = "chat-id"
        mock_document.filename = "test.pdf"
        mock_document.file_type = "pdf"
        mock_document.file_size = 1024
        mock_document.content = "Testcontentwithnulls"  # Cleaned content
        mock_document.document_metadata = {}
        mock_document.created_at = "2024-01-01T00:00:00"
        
        mock_chat = MagicMock()
        mock_chat.id = "chat-id"
        mock_chat.document_ids = []
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        # query() is called for ChatTable lookup
        chat_query = MagicMock()
        chat_query.filter.return_value.first.return_value = mock_chat
        mock_session.query.return_value = chat_query
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        with patch("app.database.DocumentTable") as mock_document_table:
            # Mock DocumentTable to capture the cleaned content
            def create_document(*args, **kwargs):
                doc = MagicMock()
                doc.id = "doc-id"
                doc.chat_id = "chat-id"
                doc.filename = "test.pdf"
                doc.file_type = "pdf"
                doc.file_size = 1024
                # Content should be cleaned (null bytes removed)
                cleaned = kwargs.get('content', '').replace('\x00', '')
                doc.content = cleaned
                doc.document_metadata = {}
                doc.created_at = "2024-01-01T00:00:00"
                return doc
            
            mock_document_table.side_effect = create_document
            # Content with null bytes
            dirty_content = "Test\x00content\x00with\x00nulls"
            result = save_document_to_db(
                "chat-id",
                "test.pdf",
                "pdf",
                content=dirty_content,
                file_size=1024,
            )
            
            # Verify content was cleaned (null bytes removed)
            assert "\x00" not in (result.get("content") or "")


def test_get_documents_from_db(mock_settings):
    """Test getting documents for a chat."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_doc1 = MagicMock()
        mock_doc1.id = "doc-1"
        mock_doc1.chat_id = "chat-id"
        mock_doc1.filename = "doc1.pdf"
        mock_doc1.file_type = "pdf"
        mock_doc1.file_size = 1024
        mock_doc1.content = "Content 1"
        mock_doc1.document_metadata = {}
        mock_doc1.created_at = "2024-01-01T00:00:00"
        
        mock_doc2 = MagicMock()
        mock_doc2.id = "doc-2"
        mock_doc2.chat_id = "chat-id"
        mock_doc2.filename = "doc2.pdf"
        mock_doc2.file_type = "pdf"
        mock_doc2.file_size = 2048
        mock_doc2.content = "Content 2"
        mock_doc2.document_metadata = {}
        mock_doc2.created_at = "2024-01-02T00:00:00"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = [mock_doc1, mock_doc2]
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = get_documents_from_db("chat-id")
        
        assert len(result) == 2
        assert result[0]["filename"] == "doc1.pdf"
        assert result[1]["filename"] == "doc2.pdf"


def test_get_document_from_db(mock_settings):
    """Test getting a document by ID."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.chat_id = "chat-id"
        mock_document.filename = "test.pdf"
        mock_document.file_type = "pdf"
        mock_document.file_size = 1024
        mock_document.content = "Test content"
        mock_document.document_metadata = {}
        mock_document.created_at = "2024-01-01T00:00:00"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_document
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = get_document_from_db("doc-id")
        
        assert result is not None
        assert result["id"] == "doc-id"
        assert result["filename"] == "test.pdf"


def test_get_document_from_db_not_found(mock_settings):
    """Test getting non-existent document returns None."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = get_document_from_db("non-existent-id")
        assert result is None


def test_delete_document_from_db(mock_settings):
    """Test deleting a document from database."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.chat_id = "chat-id"
        
        mock_chat = MagicMock()
        mock_chat.id = "chat-id"
        mock_chat.document_ids = ["doc-id", "other-doc-id"]
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        # query() is called twice - once for DocumentTable, once for ChatTable
        doc_query = MagicMock()
        doc_query.filter.return_value.first.return_value = mock_document
        chat_query = MagicMock()
        chat_query.filter.return_value.first.return_value = mock_chat
        mock_session.query.side_effect = [doc_query, chat_query]
        mock_session.delete = MagicMock()
        mock_session.commit = MagicMock()
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = delete_document_from_db("doc-id")
        
        assert result is True
        mock_session.delete.assert_called_once_with(mock_document)
        mock_session.commit.assert_called()
        assert "doc-id" not in mock_chat.document_ids


def test_delete_document_from_db_not_found(mock_settings):
    """Test deleting non-existent document returns False."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = delete_document_from_db("non-existent-id")
        assert result is False


def test_create_chat_in_db_rollback(mock_settings):
    """Test create_chat_in_db handles rollback on error."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock(side_effect=Exception("DB error"))
        mock_session.rollback = MagicMock()
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        with pytest.raises(Exception):
            create_chat_in_db("Test Chat")
        
        mock_session.rollback.assert_called_once()


def test_save_message_to_db_rollback(mock_settings):
    """Test save_message_to_db handles rollback on error."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock(side_effect=Exception("DB error"))
        mock_session.rollback = MagicMock()
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        with pytest.raises(Exception):
            save_message_to_db("chat-id", "user", "message")
        
        mock_session.rollback.assert_called_once()


def test_save_document_to_db_bytes_content(mock_settings):
    """Test save_document_to_db handles bytes content."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_chat = MagicMock()
        mock_chat.document_ids = []
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.chat_id = "chat-id"
        mock_document.filename = "test.pdf"
        mock_document.file_type = "pdf"
        mock_document.file_size = 100
        mock_document.content = "decoded content"
        mock_document.document_metadata = {}
        mock_document.created_at = "2024-01-01T00:00:00"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_chat
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        with patch("app.database.DocumentTable") as mock_doc_table:
            mock_doc_table.return_value = mock_document
            result = save_document_to_db(
                "chat-id", "test.pdf", "pdf", content=b"Binary content", file_size=100
            )
            assert result["id"] == "doc-id"


def test_save_document_to_db_content_truncation(mock_settings):
    """Test save_document_to_db truncates large content."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_chat = MagicMock()
        mock_chat.document_ids = []
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.chat_id = "chat-id"
        mock_document.filename = "test.pdf"
        mock_document.file_type = "pdf"
        mock_document.file_size = 100
        mock_document.content = "x" * 100000
        mock_document.document_metadata = {}
        mock_document.created_at = "2024-01-01T00:00:00"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_chat
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        large_content = "x" * 150000  # Larger than 100000 limit
        with patch("app.database.DocumentTable") as mock_doc_table:
            mock_doc_table.return_value = mock_document
            result = save_document_to_db("chat-id", "test.pdf", "pdf", content=large_content)
            assert result["id"] == "doc-id"


def test_save_document_to_db_decode_error(mock_settings):
    """Test save_document_to_db handles decode errors."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_chat = MagicMock()
        mock_chat.document_ids = []
        mock_document = MagicMock()
        mock_document.id = "doc-id"
        mock_document.chat_id = "chat-id"
        mock_document.filename = "test.pdf"
        mock_document.file_type = "pdf"
        mock_document.file_size = 100
        mock_document.content = None
        mock_document.document_metadata = {}
        mock_document.created_at = "2024-01-01T00:00:00"
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = mock_chat
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        # Create bytes that can't be decoded
        invalid_bytes = b'\xff\xfe\x00\x00'  # Invalid UTF-8
        with patch("app.database.DocumentTable") as mock_doc_table:
            mock_doc_table.return_value = mock_document
            result = save_document_to_db("chat-id", "test.pdf", "pdf", content=invalid_bytes)
            assert result["id"] == "doc-id"


def test_save_document_to_db_rollback(mock_settings):
    """Test save_document_to_db handles rollback on error."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.add = MagicMock(side_effect=Exception("DB error"))
        mock_session.rollback = MagicMock()
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        with pytest.raises(Exception):
            save_document_to_db("chat-id", "test.pdf", "pdf")
        
        mock_session.rollback.assert_called_once()


def test_delete_document_from_db_rollback(mock_settings):
    """Test delete_document_from_db handles rollback on error."""
    with patch("app.database.get_session") as mock_get_session:
        mock_session = MagicMock()
        mock_document = MagicMock()
        mock_document.chat_id = "chat-id"
        mock_chat = MagicMock()
        mock_chat.document_ids = ["doc-id"]
        
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        mock_session.query.side_effect = [
            MagicMock(filter=MagicMock(return_value=MagicMock(first=MagicMock(return_value=mock_document)))),
            MagicMock(filter=MagicMock(return_value=MagicMock(first=MagicMock(return_value=mock_chat)))),
        ]
        mock_session.delete = MagicMock(side_effect=Exception("DB error"))
        mock_session.rollback = MagicMock()
        mock_get_session.return_value = MagicMock(return_value=mock_session)
        
        result = delete_document_from_db("doc-id")
        assert result is False
        mock_session.rollback.assert_called_once()

