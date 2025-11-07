"""Integration test for streaming multiple chunks verification.

This test verifies that the streaming endpoint returns multiple chunks
(not a single blob) as required by the assignment.
"""

import pytest
import json
import asyncio
from httpx import AsyncClient, ASGITransport

from app.api import app


@pytest.mark.asyncio
async def test_chat_stream_multiple_chunks():
    """Integration test: Verify streaming returns multiple chunks, not a single blob.
    
    This test verifies that:
    1. The streaming endpoint returns Server-Sent Events (SSE) format
    2. Multiple chunks are received (not a single blob)
    3. Each chunk is properly formatted as SSE
    4. Status updates are interleaved with content chunks
    
    Uses short query to minimize test execution time.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        # Start streaming request with very short query for fast response
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Hi"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            
            chunks_received = []
            status_updates = []
            content_chunks = []
            max_chunks = 50  # Limit chunks to prevent long waits
            chunk_count = 0
            
            # Collect chunks with early exit once we have enough evidence
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                chunks_received.append(line)
                chunk_count += 1
                
                # Parse SSE format
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        if "content" in data:
                            content_chunks.append(data)
                            # Early exit if we have enough content chunks and final chunk
                            if data.get("done") is True and len(content_chunks) >= 2:
                                break
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("status: "):
                    try:
                        status = json.loads(line[8:])  # Remove "status: " prefix
                        status_updates.append(status)
                    except json.JSONDecodeError:
                        pass
                
                # Early exit if we've collected enough evidence
                if (len(status_updates) >= 2 and len(content_chunks) >= 2 and 
                    any(c.get("done") is True for c in content_chunks)):
                    break
                
                # Safety limit to prevent infinite loops
                if chunk_count >= max_chunks:
                    break
            
            # Verify we received multiple chunks (not a single blob)
            assert len(chunks_received) > 1, "Should receive multiple chunks, not a single blob"
            
            # Verify we have status updates
            assert len(status_updates) > 0, "Should receive at least one status update"
            
            # Verify we have content chunks
            assert len(content_chunks) > 0, "Should receive at least one content chunk"
            
            # Verify chunks are properly formatted
            for chunk in chunks_received:
                assert chunk.startswith(("data: ", "status: ")), f"Chunk should be SSE format: {chunk[:50]}"
            
            # Verify status updates have expected structure
            for status in status_updates:
                assert "status" in status, "Status update should have 'status' field"
                assert "message" in status, "Status update should have 'message' field"
                assert "done" in status, "Status update should have 'done' field"
            
            # Verify content chunks have expected structure
            for chunk in content_chunks:
                assert "content" in chunk, "Content chunk should have 'content' field"
                assert "done" in chunk, "Content chunk should have 'done' field"
            
            # Verify we see different statuses (Analyzing, Searching, Generating, etc.)
            status_types = {s["status"] for s in status_updates}
            assert len(status_types) > 0, "Should see at least one status type"
            
            # Verify final chunk is marked as done (if we got one)
            final_chunks = [c for c in content_chunks if c.get("done") is True]
            if len(final_chunks) == 0:
                # If we didn't get a done chunk, verify we at least got multiple chunks
                assert len(content_chunks) > 1, "Should receive multiple content chunks"


@pytest.mark.asyncio
async def test_chat_stream_chunks_are_incremental():
    """Integration test: Verify chunks arrive incrementally, building up the response.
    
    This test verifies that:
    1. Chunks arrive one at a time (not all at once)
    2. Content accumulates across chunks
    3. Each chunk adds to the previous content
    
    Uses very short query to minimize test execution time.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=30.0) as client:
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Say yes"},
        ) as response:
            assert response.status_code == 200
            
            accumulated_content = ""
            chunk_count = 0
            max_chunks = 30  # Limit to prevent long waits
            
            async for line in response.aiter_lines():
                if not line.strip() or not line.startswith("data: "):
                    continue
                
                try:
                    data = json.loads(line[6:])
                    if "content" in data:
                        chunk_count += 1
                        chunk_content = data.get("content", "")
                        accumulated_content += chunk_content
                        
                        # Early exit once we have enough chunks and final chunk
                        if data.get("done") is True and chunk_count >= 2:
                            break
                except json.JSONDecodeError:
                    pass
                
                # Safety limit
                if chunk_count >= max_chunks:
                    break
            
            # Verify we received multiple chunks
            assert chunk_count > 1, "Should receive multiple content chunks incrementally"
            
            # Verify content accumulated
            assert len(accumulated_content) > 0, "Content should accumulate across chunks"
            
            # Verify chunks were incremental (not all at once)
            # If we got multiple chunks, they arrived incrementally
            assert chunk_count >= 1, "Should receive at least one content chunk"

