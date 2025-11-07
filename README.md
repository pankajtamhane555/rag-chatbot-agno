# RAG Chatbot with FastAPI, Agno, and NiceGUI

A minimal Document QA Chatbot that streams agent responses, supports PDF upload, and provides async status signals.

## Project Description

This project implements a RAG (Retrieval-Augmented Generation) chatbot with:
- **FastAPI** for backend API and streaming endpoints
- **Agno** for agent orchestration
- **NiceGUI** for lightweight UI
- **OpenAI** for LLM and embeddings
- **Pydantic** for data validation

The chatbot allows users to upload PDF documents, parse them, and query the agent about the document content with streaming responses.

## Prerequisites

### For Docker Setup (Recommended)
- **Docker Engine** 20.10+ 
- **Docker Compose** 2.0+

### For Local Development
- **Python** 3.13+ (required)
- **OpenAI API key** (required)
- **PostgreSQL** with pgvector extension

## Dependencies

### Core Dependencies
- `fastapi>=0.115.0` - Web framework for API
- `uvicorn[standard]>=0.32.0` - ASGI server
- `agno>=0.1.0` - Agent orchestration
- `nicegui>=1.4.0` - Lightweight UI framework
- `openai>=1.54.0` - OpenAI client library
- `pypdf>=5.1.0` - PDF parsing
- `pydantic>=2.9.0` - Data validation
- `pydantic-settings>=2.5.0` - Settings management
- `python-dotenv>=1.0.0` - Environment variable loading
- `httpx>=0.27.0` - Async HTTP client
- `sqlalchemy>=2.0.0` - Database ORM
- `psycopg[binary]>=3.0.0` - PostgreSQL adapter
- `pgvector>=0.3.0` - Vector extension for PostgreSQL

### Optional Dependencies
- `sentence-transformers>=2.7.0` - Local reranking for better search results (optional)
- `python-docx>=1.1.0` - Word document parsing (for future support)
- `python-pptx>=0.6.23` - PowerPoint parsing (for future support)
- `markdown>=3.5.0` - Markdown parsing (for future support)

### Development Dependencies
- `pytest>=8.3.0` - Testing framework
- `pytest-check>=2.2.0` - Multiple assertion support
- `pytest-asyncio>=0.24.0` - Async test support
- `pytest-cov>=7.0.0` - Coverage plugin for pytest
- `ruff>=0.6.0` - Linter and formatter
- `mypy>=1.11.0` - Type checker

All dependencies are listed in `requirements.txt`.

## Setup Instructions

### Quick Start (Docker - Recommended)

**Prerequisites**: Docker Engine 20.10+ and Docker Compose 2.0+

1. **Clone and navigate**:
   ```bash
   git clone <repository-url>
   cd DemoRagSystem
   ```

2. **Create `.env` file with your OpenAI API key**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   
   > **Get your API key**: https://platform.openai.com/api-keys

3. **Start all services**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - **UI**: http://localhost:8080
   - **API Docs**: http://localhost:8000/docs
   - **API**: http://localhost:8000

That's it! The application is now running.

## Docker Commands Reference

### Basic Service Management

```bash
# Start all services in background
docker-compose up -d

# Start services and view logs
docker-compose up

# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes all data including database)
docker-compose down -v

# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart api
docker-compose restart ui
docker-compose restart postgres

# View service status
docker-compose ps

# View logs (all services)
docker-compose logs -f

# View logs for specific service
docker-compose logs -f api
docker-compose logs -f ui
docker-compose logs -f postgres

# View last 50 lines of logs
docker-compose logs --tail=50

# View logs for specific service (last 100 lines)
docker-compose logs --tail=100 api
```

### Building and Rebuilding

```bash
# Build images
docker-compose build

# Rebuild without cache (clean build)
docker-compose build --no-cache

# Rebuild and restart
docker-compose build --no-cache && docker-compose up -d

# Rebuild specific service
docker-compose build --no-cache api
docker-compose up -d api
```

### Database Management

```bash
# Connect to PostgreSQL
docker exec -it rag-chatbot-postgres psql -U ai -d ai

# Run SQL command
docker exec rag-chatbot-postgres psql -U ai -d ai -c "SELECT COUNT(*) FROM documents;"

# Backup database
docker exec rag-chatbot-postgres pg_dump -U ai ai > backup.sql

# Restore database
docker exec -i rag-chatbot-postgres psql -U ai ai < backup.sql

# Drop and recreate knowledge_documents table (fix dimension mismatch)
docker exec -it rag-chatbot-postgres psql -U ai -d ai -c "DROP TABLE IF EXISTS ai.knowledge_documents CASCADE;"

# View database size
docker exec rag-chatbot-postgres psql -U ai -d ai -c "SELECT pg_size_pretty(pg_database_size('ai'));"
```

### Container Management

```bash
# Execute command in running container
docker exec -it rag-chatbot-api bash
docker exec -it rag-chatbot-ui bash
docker exec -it rag-chatbot-postgres bash

# View container resource usage
docker stats

# View container details
docker inspect rag-chatbot-api

# View container logs directly
docker logs rag-chatbot-api
docker logs rag-chatbot-ui
docker logs rag-chatbot-postgres

# Follow logs in real-time
docker logs -f rag-chatbot-api

# Stop specific container
docker stop rag-chatbot-api

# Start specific container
docker start rag-chatbot-api

# Remove stopped containers
docker-compose rm

# Remove containers and volumes
docker-compose down -v --remove-orphans
```

### Running Tests

```bash
# Run all tests
docker exec rag-chatbot-api pytest

# Run tests with verbose output
docker exec rag-chatbot-api pytest -v

# Run specific test file
docker exec rag-chatbot-api pytest tests/test_document_management.py
docker exec rag-chatbot-api pytest tests/test_integration_upload.py
docker exec rag-chatbot-api pytest tests/test_parsing.py
docker exec rag-chatbot-api pytest tests/test_agent.py

# Run tests with coverage
docker exec rag-chatbot-api pytest --cov=app --cov-report=term-missing

# Run tests and generate HTML coverage report
docker exec rag-chatbot-api pytest --cov=app --cov-report=html
# Coverage report will be in htmlcov/ directory inside container

# Run specific test function
docker exec rag-chatbot-api pytest tests/test_document_management.py::test_extract_pdf_content

# Run tests matching a pattern
docker exec rag-chatbot-api pytest -k "test_upload"

# Run tests and stop at first failure
docker exec rag-chatbot-api pytest -x

# Run tests with detailed output
docker exec rag-chatbot-api pytest -vv

# Run tests and show print statements
docker exec rag-chatbot-api pytest -s

# Run tests in parallel (if pytest-xdist is installed)
docker exec rag-chatbot-api pytest -n auto

# Run only smoke tests
docker exec rag-chatbot-api pytest tests/test_smoke.py

# Run only integration tests
docker exec rag-chatbot-api pytest tests/test_integration_*.py

# Run tests and capture output to file
docker exec rag-chatbot-api pytest > test_results.txt 2>&1

# Run tests with markers (if using pytest markers)
docker exec rag-chatbot-api pytest -m "not slow"
```

### Debugging and Testing

```bash
# Test API health endpoint
docker exec rag-chatbot-api curl http://localhost:8000/health

# Check Python version in container
docker exec rag-chatbot-api python --version

# Check installed packages
docker exec rag-chatbot-api pip list

# View environment variables
docker exec rag-chatbot-api env | grep OPENAI

# Enter container shell for interactive testing
docker exec -it rag-chatbot-api bash
# Then run: pytest tests/test_document_management.py
```

### Cleanup Commands

```bash
# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes (⚠️ be careful)
docker volume prune

# Remove everything (containers, images, volumes, networks)
docker system prune -a --volumes

# Remove specific volume
docker volume rm rag-chatbot_postgres-data
```

### Network Management

```bash
# List networks
docker network ls

# Inspect network
docker network inspect rag-chatbot_rag-network

# Test connectivity between containers
docker exec rag-chatbot-api ping postgres
```

### Quick Troubleshooting

```bash
# Check if services are running
docker-compose ps

# Check service health
docker-compose ps | grep -E "Up|Exit"

# View recent errors
docker-compose logs --tail=100 | grep -i error

# Restart everything from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

# Check disk usage
docker system df

# View container resource limits
docker stats --no-stream
```

### Local Development Setup

**Prerequisites**: Python 3.13+ and PostgreSQL with pgvector extension

1. **Clone and navigate**:
   ```bash
   git clone <repository-url>
   cd DemoRagSystem
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```
   
   > **Get your API key**: https://platform.openai.com/api-keys

5. **Start PostgreSQL** (if not already running):
   ```bash
   # Using Docker (easiest)
   docker run -d \
     --name postgres \
     -e POSTGRES_USER=ai \
     -e POSTGRES_PASSWORD=ai \
     -e POSTGRES_DB=ai \
     -p 5432:5432 \
     pgvector/pgvector:pg16
   ```

6. **Start the application** (two terminals):

   **Terminal 1 - API**:
   ```bash
   python main.py
   ```
   API runs at: http://localhost:8000

   **Terminal 2 - UI**:
   ```bash
   python -m app.ui
   ```
   UI opens at: http://localhost:8080

## Troubleshooting

### OpenAI API Key Issues

**Missing API Key**: If you see errors about `OPENAI_API_KEY`:
1. Ensure you have created a `.env` file: `cp .env.example .env`
2. Add your OpenAI API key to `.env`: `OPENAI_API_KEY=your_key_here`
3. Get your API key from https://platform.openai.com/api-keys
4. Restart the application after updating `.env`

### Docker Issues

**Port conflicts**: If ports 8000, 8080, or 5432 are already in use, update `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change host port
```

**OpenAI connection errors**: 
```bash
# Check API service logs
docker-compose logs api

# Verify environment variables are loaded
docker exec rag-chatbot-api env | grep OPENAI
```

**Rebuild after code changes**: 
```bash
docker-compose build --no-cache
docker-compose up -d
```

**Dimension mismatch error**: If you see dimension errors, the database table may have been created with wrong dimensions. Fix it by:

1. **Drop and recreate the knowledge_documents table**:
   ```bash
   # Connect to PostgreSQL
   docker exec -it rag-chatbot-postgres psql -U ai -d ai
   
   # Drop the table (this will delete all embeddings, but files can be re-uploaded)
   DROP TABLE IF EXISTS ai.knowledge_documents CASCADE;
   
   # Exit psql
   \q
   ```
   
   The table will be automatically recreated with correct dimensions (1536 for text-embedding-3-small) on next file upload.

2. **Or restart with fresh database** (if you don't need existing data):
   ```bash
   docker-compose down -v  # This removes all volumes including database
   docker-compose up -d
   ```

### Local Development Issues

**Import errors**: Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**OpenAI API key**: Ensure your `.env` file contains a valid `OPENAI_API_KEY`

**Port conflicts**: Change ports in `main.py` (API) and `app/ui.py` (UI) if needed.

## Test Instructions

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=app --cov-report=html
```

### Test Coverage

Current test coverage (as of latest run):

```
Name                Stmts   Miss  Cover   Missing
-------------------------------------------------
app/__init__.py         0      0   100%
app/agent.py           94      9    90%   44, 55-59, 83, 111, 205, 207-209
app/api.py            304     27    91%   51-69, 152, 352-354, 376-377, 386-388, 397, 403, 417
app/config.py          37      1    97%   66
app/database.py       268     34    87%   101-103, 175-177, 204-206, 236, 244-247, 275-278, 353-355, 386-389, 427-429, 511-513, 543-545
app/models.py          42      0   100%
app/validation.py      53      1    98%   46
-------------------------------------------------
TOTAL                 798     72    91%
```

**Test Results**: 203 passed, 1 skipped

See `tests/README.md` for more testing details.

## Architecture Overview

The application follows a clean, modular architecture that separates concerns while leveraging library features.

### Application Structure

```
app/
├── api.py       # FastAPI routes & streaming endpoints
├── agent.py     # Agno agent logic & chat management
├── database.py  # PostgreSQL database models & CRUD operations
├── validation.py # File validation utilities
├── models.py    # Pydantic data models
├── ui.py        # NiceGUI interface
└── config.py    # Configuration (Pydantic settings)

tests/
├── unit/        # Unit tests (isolated component tests)
├── integration/ # Integration tests (end-to-end workflows)
└── data/        # Test data (sample PDFs)
```

### Architecture Layers

**1. API Layer (`app/api.py`)**
- FastAPI application with streaming endpoints
- Handles HTTP requests/responses
- Manages chat sessions and document uploads
- Streams agent responses using Server-Sent Events (SSE)
- Error handling and status updates

**2. Agent Layer (`app/agent.py`)**
- `AgentManager`: Manages RAG agent lifecycle
- `OpenAIEmbedder`: Handles embeddings via OpenAI
- Knowledge base integration using Agno's `Knowledge` class
- File content extraction and metadata parsing
- Chat-scoped knowledge filtering (documents per chat)

**3. Data Layer (`app/database.py`)**
- SQLAlchemy models for chats, messages, documents
- PostgreSQL with pgvector for vector storage
- CRUD operations for all entities
- Transaction management and error handling

**4. UI Layer (`app/ui.py`)**
- NiceGUI-based web interface
- Real-time streaming display
- Async status updates (Analyzing → Searching → Generating)
- Document management (upload, view, delete)
- Chat history and navigation

**5. Validation Layer (`app/validation.py`)**
- File type validation
- File size limits
- File signature verification
- Error handling for invalid uploads

### Design Principles

**Library-First Approach**: 
We leverage Agno/FastAPI/Pydantic features before writing custom code:
- Agno's `Knowledge` class for document storage and retrieval
- Agno's `Agent` for orchestration (no custom agent logic)
- FastAPI's `StreamingResponse` for SSE streaming
- Pydantic's `BaseSettings` for configuration management

**Minimal Abstractions**:
- Direct use of library APIs where possible
- Thin wrappers only when necessary (e.g., `OllamaEmbedder` for dimension validation)
- No over-engineering or premature optimization

**Separation of Concerns**:
- API layer handles HTTP, not business logic
- Agent layer handles RAG, not database operations
- Database layer handles persistence, not validation
- Each module has a single, clear responsibility

**Modern Python Practices**:
- Type hints throughout (`list[str]`, `str | None`)
- Pydantic models for all structured data (no raw dicts)
- Async/await for I/O operations
- Context managers for resource management

### Data Flow

**Streaming Chat Flow**:
1. User sends message → `POST /chat/stream`
2. API creates/retrieves chat → Database
3. API saves user message → Database
4. API creates agent with chat_id → AgentManager
5. Agent filters knowledge by chat_id → Knowledge Base
6. Agent streams response chunks → FastAPI SSE
7. UI displays chunks incrementally → NiceGUI

**Document Upload Flow**:
1. User uploads PDF → `POST /upload`
2. API validates file → Validation layer
3. API extracts content → AgentManager
4. API adds to knowledge base → Agno Knowledge (filtered by chat_id)
5. API saves document metadata → Database
6. Document available for agent queries → Knowledge Base

### Why This Architecture Works

1. **Simplicity**: Direct use of library features reduces custom code
2. **Testability**: Clear separation allows isolated unit tests and focused integration tests
3. **Maintainability**: Each module has a single responsibility
4. **Scalability**: Async operations prevent blocking, streaming enables real-time responses
5. **Flexibility**: Easy to swap components (e.g., different LLM providers, vector databases)

follow the white rabbit

## Development Tools

### Code Quality

- **Ruff**: Linter and formatter for Python
- **MyPy**: Static type checker
- **Pytest**: Testing framework with async support

### IDE Configuration

This project includes `.cursorrules` for Cursor IDE configuration:
- **Linters/Formatters**: Ruff (linter & formatter), MyPy (type checker)
- **Cursor Rules**: `.cursorrules` file provides project context to AI assistant
- **Documentation**: Indexed docs for Agno, FastAPI, NiceGUI, Pydantic

#### Cursor Configuration Details

**MCP Servers**: While not explicitly configured in this project, Cursor's built-in documentation indexing provides access to:
- Agno documentation (agent orchestration patterns, knowledge base management)
- FastAPI documentation (streaming responses, async endpoints, middleware)
- NiceGUI documentation (UI components, event handling, async updates)
- Pydantic documentation (data validation, settings management)

**Linters/Formatters**:
- **Ruff**: Configured as the primary linter and formatter. Errors are surfaced in the IDE with inline annotations and the Problems panel.
- **MyPy**: Type checker configured for static type analysis. Type errors appear in the IDE alongside linting issues.
- **Format on Save**: Enabled to automatically format code using Ruff on file save.

**`.cursorrules` Content and Purpose**:
The `.cursorrules` file contains:
- Project-specific coding standards (PEP 8 compliance, naming conventions)
- Tech stack guidelines (Agno, FastAPI, NiceGUI, Ollama)
- Code organization principles (single responsibility, DRY, minimal abstractions)
- Testing guidelines (pytest, integration vs unit tests)
- Documentation standards (docstring format, type hints)

This file helps the AI assistant understand:
- The project's architecture and design principles
- Preferred patterns and anti-patterns
- Testing requirements and structure
- Code quality expectations

**Documentation Indexing**:
Cursor automatically indexes:
- Library documentation for imported packages (Agno, FastAPI, NiceGUI, Pydantic)
- Project codebase for context-aware suggestions
- Type definitions for better autocomplete

**IDE Features Enabled**:
- Inline error annotations (Ruff, MyPy)
- Auto-formatting on save (Ruff)
- Type hints in hover tooltips
- Code completion with context awareness
- Refactoring suggestions based on project patterns

**What Actually Helped Development**:
1. **Ruff Integration**: Immediate feedback on code style issues without running external tools
2. **Type Checking**: Catches errors early with modern typing (`list[str]`, `str | None`)
3. **Cursor Rules**: AI assistant understands project context and suggests code that follows our patterns
4. **Documentation Indexing**: Quick reference to library docs without leaving the IDE
5. **Auto-formatting**: Consistent code style without manual work
6. **Context-Aware Suggestions**: AI assistant understands the codebase structure and suggests appropriate patterns
7. **Test Structure Awareness**: Assistant knows about `tests/unit/` and `tests/integration/` separation

## Trade-offs and Limitations

### Current Limitations

**Document Types**: 
- Only PDF files are supported (as per assignment requirements)
- Single document type simplifies validation and parsing logic

**LLM Provider**:
- Uses OpenAI (as specified in assignment)
- Requires OpenAI API key and internet connection
- API costs apply based on usage

**Database**:
- PostgreSQL with pgvector for vector storage
- Chat history is persistent (stored in database)
- Requires PostgreSQL setup and configuration

**Scalability**:
- Single-instance deployment (no horizontal scaling)
- No load balancing or distributed architecture
- Vector search performance depends on database size

**Security**:
- No authentication or authorization
- No rate limiting
- File uploads validated but not scanned for malware
- No encryption at rest for sensitive documents

**Error Handling**:
- Basic error handling with user-friendly messages
- No retry logic for transient failures
- Limited error recovery mechanisms

### Trade-offs Made

**OpenAI (Required by Assignment)**:
- **Chosen**: OpenAI (as per assignment requirements)
- **Trade-off**: API costs, requires internet connection
- **Rationale**: Meets assignment requirements, high-quality responses, reliable service

**PDF Only (Required by Assignment)**:
- **Chosen**: PDF only (as per assignment requirements)
- **Trade-off**: Limited to single file type
- **Rationale**: Meets assignment requirements, simpler implementation, focused scope

**PostgreSQL vs Dedicated Vector DB**:
- **Chosen**: PostgreSQL with pgvector (simpler setup, ACID guarantees)
- **Trade-off**: May have lower performance than specialized vector databases
- **Rationale**: Single database for all data, easier deployment, sufficient for moderate scale

**FastAPI Streaming vs WebSockets**:
- **Chosen**: Server-Sent Events (SSE) via FastAPI StreamingResponse
- **Trade-off**: One-way communication (server → client), no bidirectional
- **Rationale**: Simpler implementation, sufficient for chat responses, works with HTTP/2

**NiceGUI vs React/Vue**:
- **Chosen**: NiceGUI (Python-based, simple, integrated)
- **Trade-off**: Less flexible than full frontend frameworks
- **Rationale**: Faster development, no separate frontend build, good for demos

### What We'd Add Next

**Immediate Improvements**:
1. **Authentication**: User accounts, API keys, session management
2. **Rate Limiting**: Prevent abuse, fair resource usage
3. **Better Error Recovery**: Retry logic, circuit breakers
4. **Monitoring**: Logging, metrics, health checks
5. **Document Chunking**: Better text splitting for large documents

**Medium-term Enhancements**:
1. **Cloud LLM Support**: OpenAI, Anthropic, other providers
2. **Advanced RAG**: Reranking, hybrid search, query expansion
3. **Multi-user Support**: User isolation, shared documents
4. **Document Versioning**: Track document changes over time
5. **Export Features**: Download chat history, export documents

**Long-term Vision**:
1. **Horizontal Scaling**: Multiple API instances, load balancing
2. **Dedicated Vector DB**: Pinecone, Weaviate, or Qdrant for better performance
3. **Advanced Analytics**: Usage metrics, document insights
4. **Plugin System**: Extensible document processors, custom agents
5. **Mobile App**: Native iOS/Android clients

### Known Issues

- **Dimension Mismatch**: If embedding model changes, database table needs recreation
- **Large Files**: Very large PDFs may timeout during processing
- **Concurrent Uploads**: No locking mechanism for simultaneous uploads to same chat
- **Memory Usage**: Large documents loaded into memory during processing

## Development

### Code Style

Follow PEP 8 and project conventions defined in `.cursorrules`:
- Use modern typing: `list[str]`, `str | None`
- Prefer Pydantic models over raw dicts
- Keep functions focused and under 50 lines
- Write meaningful tests that fail clearly

### Contributing

1. Follow the coding standards in `.cursorrules`
2. Write tests for new features
3. Keep commits small and meaningful
4. Update documentation as needed

## Security

### Security Audit Results

The codebase has been audited for hardcoded secrets and follows security best practices:

✅ **No Hardcoded Secrets Found**
- All API keys are loaded from environment variables via Pydantic settings
- No hardcoded database credentials or connection strings
- All sensitive values are configurable through `.env` file

✅ **Security Best Practices in Place**
- All sensitive values loaded from environment variables
- `.env` file is gitignored (should be verified in your `.gitignore`)
- `.env.example` contains only placeholder values
- Pydantic settings validate configuration
- No secrets in code comments or documentation

### Production Security Recommendations

⚠️ **Important for Production Deployment:**

1. **PostgreSQL Password**: The default PostgreSQL password (`"ai"`) is weak and should be changed in production:
   ```bash
   # Set a strong password in your .env file
   POSTGRES_PASSWORD=your_strong_password_here
   ```

2. **OpenAI API Key**: Ensure your `.env` file is never committed to version control:
   ```bash
   # Verify .env is in .gitignore
   echo ".env" >> .gitignore
   ```

3. **Environment Variables**: Always use environment variables for sensitive configuration:
   - `OPENAI_API_KEY` - Required, get from https://platform.openai.com/api-keys
   - `POSTGRES_PASSWORD` - Change from default `"ai"` in production
   - `POSTGRES_USER` - Consider changing from default `"ai"` in production

4. **Database Security**: In production, consider:
   - Using SSL/TLS for database connections
   - Restricting database access to specific IPs/networks
   - Using database connection pooling with proper limits
   - Regular security updates for PostgreSQL

5. **API Security**: For production deployments, consider adding:
   - Authentication and authorization
   - Rate limiting
   - Input validation and sanitization
   - HTTPS/TLS encryption
   - CORS configuration

### Security Notes

- Default values in `app/config.py` (e.g., `postgres_password="ai"`) are **development-only defaults**
- These can and should be overridden via environment variables in production
- See `app/config.py` for security warnings in code comments

## License

_To be determined_

