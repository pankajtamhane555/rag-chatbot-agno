"""Unit tests for Agno agent configuration and AgentManager."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from app.agent import AgentManager, FileParseError
from app.config import Settings


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    test_settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        openai_embedding_dimensions=1536,
        openai_timeout=300,
    )
    monkeypatch.setattr("app.agent.settings", test_settings)
    return test_settings


def test_agent_manager_initializes_correctly(mock_settings):
    """Test that AgentManager initializes correctly."""
    manager = AgentManager()
    assert manager is not None
    assert hasattr(manager, 'create_agent')
    assert hasattr(manager, 'extract_file_content')
    assert hasattr(manager, 'extract_file_metadata')
    assert hasattr(manager, 'add_file_knowledge')


def test_create_agent_returns_agent(mock_settings):
    """Test that create_agent returns an Agent instance."""
    with patch("app.agent.OpenAIChat") as mock_openai, patch("app.agent.Agent") as mock_agent_class, patch(
        "app.agent.Knowledge"
    ) as mock_knowledge_class, patch("app.agent.PgVector") as mock_pgvector, patch(
        "app.agent.OpenAIEmbedder"
    ) as mock_embedder:
        # Setup mocks
        mock_openai_instance = Mock()
        mock_openai.return_value = mock_openai_instance
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        mock_knowledge_instance = Mock()
        mock_knowledge_class.return_value = mock_knowledge_instance
        mock_pgvector_instance = Mock()
        mock_pgvector.return_value = mock_pgvector_instance

        manager = AgentManager()
        agent = manager.create_agent()

        # Verify Agent was created
        mock_agent_class.assert_called_once()
        agent_kwargs = mock_agent_class.call_args[1]
        assert agent_kwargs["search_knowledge"] is True
        assert agent == mock_agent_instance


def test_create_agent_with_chat_id(mock_settings):
    """Test that create_agent accepts chat_id parameter."""
    with patch("app.agent.OpenAIChat"), patch("app.agent.Agent") as mock_agent_class, patch(
        "app.agent.Knowledge"
    ), patch("app.agent.PgVector"), patch("app.agent.OpenAIEmbedder"):
        manager = AgentManager()
        agent = manager.create_agent(chat_id="test-chat-123")

        # Verify agent was created with knowledge_filters
        mock_agent_class.assert_called_once()
        agent_kwargs = mock_agent_class.call_args[1]
        assert agent_kwargs["knowledge_filters"] == {"chat_id": "test-chat-123"}


def test_create_agent_without_chat_id(mock_settings):
    """Test that create_agent without chat_id has no knowledge_filters."""
    with patch("app.agent.OpenAIChat"), patch("app.agent.Agent") as mock_agent_class, patch(
        "app.agent.Knowledge"
    ), patch("app.agent.PgVector"), patch("app.agent.OpenAIEmbedder"):
        manager = AgentManager()
        manager.create_agent()

        # Verify agent was created without knowledge_filters
        mock_agent_class.assert_called_once()
        agent_kwargs = mock_agent_class.call_args[1]
        assert agent_kwargs.get("knowledge_filters") is None


def test_agent_configuration_uses_environment_variables(mock_settings):
    """Test that agent configuration uses settings from environment variables."""
    with patch("app.agent.OpenAIChat") as mock_openai, patch("app.agent.Agent"), patch(
        "app.agent.Knowledge"
    ), patch("app.agent.PgVector"), patch("app.agent.OpenAIEmbedder"):
        manager = AgentManager()
        manager.create_agent()

        # Verify OpenAIChat was created with correct settings
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["id"] == mock_settings.openai_model
        assert call_kwargs["api_key"] == mock_settings.openai_api_key




def test_extract_file_metadata_pdf(mock_settings):
    """Test extract_file_metadata for PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = Path(tmp.name)
        # Create minimal PDF
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with open(tmp_path, "wb") as f:
            writer.write(f)
    
    try:
        manager = AgentManager()
        metadata = manager.extract_file_metadata(tmp_path)
        assert metadata.page_count >= 0
        assert isinstance(metadata.page_count, int)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_extract_file_metadata_non_pdf(mock_settings):
    """Test extract_file_metadata for non-PDF file raises error."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(b"Test content")
    
    try:
        manager = AgentManager()
        with pytest.raises(FileParseError, match="Only PDF files are supported"):
            manager.extract_file_metadata(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_extract_file_metadata_file_not_found(mock_settings):
    """Test extract_file_metadata raises FileNotFoundError for missing file."""
    manager = AgentManager()
    with pytest.raises(FileNotFoundError):
        manager.extract_file_metadata("/nonexistent/file.pdf")


def test_has_documents_for_chat(mock_settings):
    """Test has_documents_for_chat method."""
    with patch("app.agent.AgentManager._get_document_count_for_chat") as mock_count:
        mock_count.return_value = 2
        manager = AgentManager()
        assert manager.has_documents_for_chat("test-chat") is True
        
        mock_count.return_value = 0
        assert manager.has_documents_for_chat("test-chat") is False


def test_get_document_count_for_chat(mock_settings):
    """Test _get_document_count_for_chat method."""
    with patch("app.agent.AgentManager._get_knowledge") as mock_get_knowledge:
        mock_knowledge = Mock()
        mock_result = Mock()
        mock_result.meta_data = {"chat_id": "test-chat"}
        mock_knowledge.search.return_value = [mock_result, mock_result]
        mock_get_knowledge.return_value = mock_knowledge
        
        manager = AgentManager()
        count = manager._get_document_count_for_chat("test-chat")
        assert count == 2


def test_get_document_count_for_chat_no_results(mock_settings):
    """Test _get_document_count_for_chat with no results."""
    with patch("app.agent.AgentManager._get_knowledge") as mock_get_knowledge:
        mock_knowledge = Mock()
        mock_knowledge.search.return_value = []
        mock_get_knowledge.return_value = mock_knowledge
        
        manager = AgentManager()
        count = manager._get_document_count_for_chat("test-chat")
        assert count == 0


def test_get_document_count_for_chat_error(mock_settings):
    """Test _get_document_count_for_chat handles errors."""
    with patch("app.agent.AgentManager._get_knowledge") as mock_get_knowledge:
        mock_knowledge = Mock()
        mock_knowledge.search.side_effect = Exception("Search error")
        mock_get_knowledge.return_value = mock_knowledge
        
        manager = AgentManager()
        count = manager._get_document_count_for_chat("test-chat")
        assert count == 0


def create_test_pdf(pages: int = 1) -> Path:
    """Helper to create a test PDF file."""
    from pypdf import PdfWriter
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = Path(temp_file.name)
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    with open(temp_path, "wb") as f:
        writer.write(f)
    return temp_path


def test_extract_file_metadata_pdf_error_handling(mock_settings):
    """Test extract_file_metadata handles PDF extraction errors."""
    manager = AgentManager()
    pdf_path = create_test_pdf(pages=1)
    
    try:
        # Patch pypdf.PdfReader since it's imported inside the function
        with patch("pypdf.PdfReader") as mock_pdf_reader:
            mock_pdf_reader.side_effect = Exception("PDF read error")
            metadata = manager.extract_file_metadata(pdf_path)
            assert metadata.page_count == 0
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


def test_extract_file_content_pdf(mock_settings):
    """Test extract_file_content for PDF."""
    manager = AgentManager()
    pdf_path = create_test_pdf(pages=1)
    
    try:
        with patch("app.agent.PDFReader") as mock_pdf_reader:
            mock_reader_instance = Mock()
            mock_reader_instance.read.return_value = "Test PDF content"
            mock_pdf_reader.return_value = mock_reader_instance
            
            content = manager.extract_file_content(pdf_path)
            assert isinstance(content, str)
            assert len(content) > 0
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


def test_extract_file_content_non_pdf(mock_settings):
    """Test extract_file_content raises error for non-PDF."""
    manager = AgentManager()
    temp_path = Path(tempfile.mktemp(suffix=".txt"))
    temp_path.write_text("Test content")
    
    try:
        with pytest.raises(FileParseError, match="Only PDF files are supported"):
            manager.extract_file_content(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@pytest.mark.asyncio
async def test_add_file_knowledge_success(mock_settings):
    """Test add_file_knowledge successfully adds PDF to knowledge base."""
    manager = AgentManager()
    pdf_path = create_test_pdf(pages=1)
    
    try:
        with patch.object(manager, "_get_knowledge") as mock_get_knowledge:
            with patch("app.agent.get_chat_from_db") as mock_get_chat:
                with patch("app.agent.update_chat_in_db") as mock_update:
                    with patch("app.agent.AgentManager._get_document_count_for_chat") as mock_count:
                        mock_kb = MagicMock()
                        mock_kb.add_content_async = AsyncMock()
                        mock_get_knowledge.return_value = mock_kb
                        mock_get_chat.return_value = {"id": "test-chat"}
                        mock_count.return_value = 1
                        
                        await manager.add_file_knowledge("test-chat", str(pdf_path), "test.pdf")
                        
                        mock_kb.add_content_async.assert_called_once()
                        mock_update.assert_called_once()
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


@pytest.mark.asyncio
async def test_add_file_knowledge_non_pdf(mock_settings):
    """Test add_file_knowledge raises error for non-PDF."""
    manager = AgentManager()
    temp_path = Path(tempfile.mktemp(suffix=".txt"))
    temp_path.write_text("Test content")
    
    try:
        with pytest.raises(FileParseError, match="Only PDF files are supported"):
            await manager.add_file_knowledge("chat-id", str(temp_path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


@pytest.mark.asyncio
async def test_add_file_knowledge_add_content_error(mock_settings):
    """Test add_file_knowledge handles add_content_async errors."""
    manager = AgentManager()
    pdf_path = create_test_pdf(pages=1)
    
    try:
        with patch.object(manager, "_get_knowledge") as mock_knowledge:
            mock_kb = MagicMock()
            mock_kb.add_content_async = AsyncMock(side_effect=Exception("Add failed"))
            mock_knowledge.return_value = mock_kb
            
            with pytest.raises(FileParseError):
                await manager.add_file_knowledge("chat-id", str(pdf_path))
    finally:
        if pdf_path.exists():
            pdf_path.unlink()

