"""Simplified agent manager using Agno's built-in features.

This module follows Agno's recommended patterns for Agentic RAG:
- Knowledge base with vector database
- Agent with search_knowledge enabled
- Simple file addition using Agno readers
"""

import logging
from pathlib import Path

from agno.agent import Agent
from agno.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.vectordb.pgvector import PgVector
from agno.models.openai import OpenAIChat

from .config import settings
from .database import get_chat_from_db, update_chat_in_db
from .models import PDFMetadata

logger = logging.getLogger(__name__)


class FileParseError(Exception):
    """Exception raised when file parsing fails."""


class AgentManager:
    """Manages RAG agents using Agno's built-in features.

    Follows Agno's recommended pattern: Knowledge + Agent with search_knowledge=True.
    """

    def __init__(self) -> None:
        """Initialize AgentManager instance."""
        self._knowledge: Knowledge | None = None

    def _get_knowledge(self) -> Knowledge:
        """Get or create Knowledge instance with OpenAI embeddings."""
        if self._knowledge is None:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required. Set it in your .env file.")

            embedder = OpenAIEmbedder(
                id=settings.openai_embedding_model,
                dimensions=settings.openai_embedding_dimensions,
            )

            # Optional local reranker
            reranker = None
            try:
                from agno.knowledge.reranker.sentence_transformer import SentenceTransformerReranker
                reranker = SentenceTransformerReranker(
                    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    top_n=5,
                )
                logger.info("Local reranker enabled")
            except ImportError:
                pass  # Reranker is optional

            self._knowledge = Knowledge(
                vector_db=PgVector(
                    table_name="knowledge_documents",
                    db_url=settings.get_postgres_url(),
                    embedder=embedder,
                    reranker=reranker,
                ),
            )
        return self._knowledge

    def create_agent(self, chat_id: str | None = None) -> Agent:
        """Create and configure Agno agent with knowledge base.

        Args:
            chat_id: Optional chat ID to filter knowledge by chat.

        Returns:
            Configured Agent instance with search_knowledge enabled.
        """
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required. Set it in your .env file.")

        return Agent(
            model=OpenAIChat(
                id=settings.openai_model,
                api_key=settings.openai_api_key,
            ),
            knowledge=self._get_knowledge(),
            search_knowledge=True,  # Agentic RAG enabled by default
            knowledge_filters={"chat_id": chat_id} if chat_id else None,
            instructions=[
                "You are a helpful and direct assistant.",
                "Answer questions directly and concisely.",
                "When answering from uploaded documents, provide the specific answer without mentioning document names unless asked.",
                "Search your knowledge base first when asked about uploaded documents.",
            ],
        )

    async def add_file_knowledge(self, chat_id: str, file_path: str, filename: str | None = None) -> None:
        """Add PDF file to knowledge base using Agno's PDFReader.

        Args:
            chat_id: Chat ID to associate the document with.
            file_path: Path to the PDF file.
            filename: Optional filename (defaults to file name).
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path_obj.suffix.lower() != ".pdf":
            raise FileParseError("Only PDF files are supported.")

        if filename is None:
            filename = file_path_obj.name

        try:
            await self._get_knowledge().add_content_async(
                path=str(file_path),
                reader=PDFReader(),
                metadata={"chat_id": chat_id, "filename": filename},
            )

            # Update chat document count
            chat = get_chat_from_db(chat_id)
            if chat:
                document_count = self._get_document_count_for_chat(chat_id)
                update_chat_in_db(chat_id, document_count=document_count)

            logger.info(f"Added PDF to knowledge base: {filename} (chat: {chat_id})")
        except Exception as e:
            logger.error(f"Error adding file to knowledge base: {e}")
            raise FileParseError(f"Failed to add file: {e}") from e

    def extract_file_metadata(self, file_path: str | Path) -> PDFMetadata:
        """Extract metadata from PDF file."""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path_obj.suffix.lower() != ".pdf":
            raise FileParseError("Only PDF files are supported.")

        try:
            from pypdf import PdfReader as PyPDFReader

            reader = PyPDFReader(str(file_path_obj))
            page_count = len(reader.pages)
            pdf_metadata = reader.metadata or {}

            return PDFMetadata(
                page_count=page_count,
                title=pdf_metadata.get("/Title", "").strip() or None,
                author=pdf_metadata.get("/Author", "").strip() or None,
                subject=pdf_metadata.get("/Subject", "").strip() or None,
                creator=pdf_metadata.get("/Creator", "").strip() or None,
                producer=pdf_metadata.get("/Producer", "").strip() or None,
                creation_date=str(pdf_metadata.get("/CreationDate", "")) or None,
                modification_date=str(pdf_metadata.get("/ModDate", "")) or None,
            )
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {e}")
            return PDFMetadata(page_count=0)


    def _get_document_count_for_chat(self, chat_id: str) -> int:
        """Get document count for a chat by searching knowledge base."""
        try:
            results = self._get_knowledge().search(query="", max_results=1000)
            if not results:
                return 0

            filtered = [
                r for r in results
                if hasattr(r, 'meta_data') and r.meta_data and r.meta_data.get("chat_id") == chat_id
            ]
            return len(filtered)
        except Exception as e:
            logger.warning(f"Error counting documents for chat {chat_id}: {e}")
            return 0

    def has_documents_for_chat(self, chat_id: str) -> bool:
        """Check if chat has any documents in knowledge base."""
        return self._get_document_count_for_chat(chat_id) > 0

    def extract_file_content(self, file_path: str | Path) -> str:
        """Extract text content from PDF using Agno's PDFReader.
        
        This is a simple wrapper for display purposes. The actual content
        is stored in the knowledge base via add_file_knowledge.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path_obj.suffix.lower() != ".pdf":
            raise FileParseError("Only PDF files are supported.")

        try:
            reader = PDFReader()
            content = reader.read(str(file_path))
            if isinstance(content, bytes):
                return content.decode('utf-8', errors='replace')
            return str(content) if content else ""
        except Exception as e:
            logger.error(f"Error extracting content with Agno PDFReader: {e}")
            return ""


