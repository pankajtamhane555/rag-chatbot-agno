import asyncio
import os
import json
import logging

from nicegui import app, ui

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


async def create_new_chat() -> dict | None:
    """Create a new chat via API."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{API_BASE_URL}/chats")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        ui.notify(f"Failed to create chat: {e}", color="negative")
        return None


async def get_chat_list() -> list[dict] | None:
    """Get list of chats from API."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/chats")
            response.raise_for_status()
            data = response.json()
            return data.get("chats", [])
    except Exception as e:
        ui.notify(f"Failed to load chats: {e}", color="negative")
        return None


async def get_chat_messages(chat_id: str) -> list[dict] | None:
    """Get messages for a chat from API."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/chats/{chat_id}/messages")
            response.raise_for_status()
            data = response.json()
            return data.get("messages", [])
    except Exception as e:
        ui.notify(f"Failed to load messages: {e}", color="negative")
        return None


async def delete_chat(chat_id: str) -> bool:
    """Delete a chat via API."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{API_BASE_URL}/chats/{chat_id}")
            response.raise_for_status()
            return True
    except Exception as e:
        ui.notify(f"Failed to delete chat: {e}", color="negative")
        return False


async def get_chat_documents(chat_id: str) -> list[dict] | None:
    """Get documents for a chat from API."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/chats/{chat_id}/documents")
            response.raise_for_status()
            data = response.json()
            return data.get("documents", [])
    except Exception as e:
        ui.notify(f"Failed to load documents: {e}", color="negative")
        return None


async def get_document(document_id: str) -> dict | None:
    """Get a document by ID from API."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/documents/{document_id}")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        ui.notify(f"Failed to load document: {e}", color="negative")
        return None


async def delete_document(document_id: str) -> bool:
    """Delete a document via API."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{API_BASE_URL}/documents/{document_id}")
            response.raise_for_status()
            return True
    except Exception as e:
        ui.notify(f"Failed to delete document: {e}", color="negative")
        return False


async def upload_document(file_content: bytes, filename: str, chat_id: str | None = None) -> dict | None:
    """Upload a document via API.
    
    Returns:
        Response dict if successful (status 200), None otherwise.
    """
    try:
        import httpx
        logger.info(f"Starting upload: filename={filename}, size={len(file_content)} bytes, chat_id={chat_id}")
        async with httpx.AsyncClient(timeout=300.0) as client:  # Increased timeout for large files
            files = {"file": (filename, file_content, "application/octet-stream")}
            data = {}
            if chat_id:
                data["chat_id"] = chat_id
            
            logger.info(f"Sending POST request to {API_BASE_URL}/upload")
            response = await client.post(
                f"{API_BASE_URL}/upload",
                files=files,
                data=data,
            )
            
            logger.info(f"Upload response status: {response.status_code}")
            
            # Only return success if we get 200 status
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Upload successful: {result.get('message', 'No message')}")
                return result
            else:
                error_detail = "Unknown error"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", str(response.status_code))
                except Exception:
                    error_detail = f"HTTP {response.status_code}: {response.text[:200]}"
                
                logger.error(f"Upload failed with status {response.status_code}: {error_detail}")
                raise httpx.HTTPStatusError(
                    f"Upload failed: {error_detail}",
                    request=response.request,
                    response=response,
                )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during upload: {e}", exc_info=True)
        raise
    except httpx.TimeoutException as e:
        logger.error(f"Upload timeout: {e}", exc_info=True)
        raise Exception(f"Upload timed out. The file may be too large or the server is busy.")
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}", exc_info=True)
        raise


async def stream_chat_response(message: str, chat_id: str | None = None):
    """Stream chat response from API."""
    try:
        import httpx

        payload = {"message": message}
        if chat_id:
            payload["chat_id"] = chat_id

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/chat/stream",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("status:"):
                        try:
                            status_data = json.loads(line[7:].strip())
                            status = status_data.get("status", "")
                            status_msg = status_data.get("message", "")
                            yield {"type": "status", "status": status, "message": status_msg}
                        except json.JSONDecodeError:
                            continue
                    elif line.startswith("data:"):
                        try:
                            chunk_data = json.loads(line[5:].strip())
                            content = chunk_data.get("content", "")
                            done = chunk_data.get("done", False)
                            yield {"type": "content", "content": content, "done": done}
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        yield {"type": "error", "error": str(e)}


@ui.page("/")
def index():
    """Main chat interface with sidebar (3:9 ratio)."""
    ui.add_head_html("""
        <style>
            * {
                box-sizing: border-box;
            }
            body, html {
                margin: 0;
                padding: 0;
                height: 100%;
                overflow: hidden;
            }
            #app {
                height: 100vh;
                overflow: hidden;
                margin: 0;
                padding: 0;
            }
            .nicegui-content {
                margin: 0 !important;
                padding: 0 !important;
            }
            /* Style expansion components for dark theme */
            .q-expansion-item {
                background: #1F2937 !important;
                color: #FFFFFF !important;
                width: 100% !important;
            }
            .q-expansion-item__header {
                background: #1F2937 !important;
                color: #FFFFFF !important;
                padding: 12px 16px !important;
                border-bottom: 1px solid #374151 !important;
                flex-shrink: 0 !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }
            .q-expansion-item__header:hover {
                background: #374151 !important;
            }
            .q-expansion-item__header-container {
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
            }
            .q-expansion-item__header-content {
                color: #FFFFFF !important;
                font-weight: 600 !important;
                flex: 1 !important;
                text-align: left !important;
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
            }
            .q-expansion-item__toggle-icon {
                color: #D1D5DB !important;
                flex-shrink: 0 !important;
            }
            .q-expansion-item__container {
                background: #1F2937 !important;
                width: 100% !important;
            }
            .q-expansion-item__content {
                background: #1F2937 !important;
                width: 100% !important;
            }
            /* Ensure expansion stays visible when collapsed - only content hides */
            .q-expansion-item {
                min-height: auto !important;
            }
            .q-expansion-item__header {
                display: flex !important;
            }
        </style>
    """)

    current_chat_id: str = ""
    components = {"chat_list": None, "message_input": None, "messages": None, "documents_list": None}

    async def load_chat_list():
        """Load and display chat list."""
        if components["chat_list"] is None:
            return
        chats = await get_chat_list()
        if chats is None:
            return

        components["chat_list"].clear()
        if not chats:
            with components["chat_list"]:
                ui.label("No chats yet").classes("text-gray-400 text-sm").style("padding: 8px 16px; margin: 0; width: 100%; text-align: left;")
            return

        for chat in chats:
            chat_id = chat.get("id", "")
            title = chat.get("title", "Untitled Chat")
            is_active = chat_id == current_chat_id

            with components["chat_list"]:
                # Selected chat has different styling
                if is_active:
                    with ui.row().classes("w-full items-center gap-2").style("padding: 8px 16px; cursor: pointer; margin: 0; width: 100%; box-sizing: border-box; display: flex; flex-direction: row; align-items: center;"):
                        ui.icon("arrow_forward").classes("text-blue-400").style("font-size: 16px; flex-shrink: 0;")
                        ui.label(title).classes("text-white text-sm font-medium").style("flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin: 0; padding: 0; min-width: 0;")
                        ui.button(icon="delete", on_click=lambda cid=chat_id: handle_delete_chat(cid)).classes("flex-shrink-0").style(
                            "background: rgba(156, 163, 175, 0.2); color: #9CA3AF; padding: 6px; min-width: 32px; height: 32px; border: none; border-radius: 9999px;"
                        ).tooltip("Delete chat")
                else:
                    chat_row = ui.row().classes("w-full items-center gap-2").style("padding: 8px 16px; cursor: pointer; margin: 0; width: 100%; box-sizing: border-box; display: flex; flex-direction: row; align-items: center;")
                    chat_row.on("click", lambda cid=chat_id: select_chat(cid))
                    with chat_row:
                        ui.icon("chat_bubble").classes("text-gray-400").style("font-size: 16px; flex-shrink: 0;")
                        ui.label(title).classes("text-gray-300 text-sm").style("flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin: 0; padding: 0; min-width: 0;")
                        ui.button(icon="delete", on_click=lambda cid=chat_id: handle_delete_chat(cid)).classes("flex-shrink-0").style(
                            "background: rgba(156, 163, 175, 0.2); color: #9CA3AF; padding: 6px; min-width: 32px; height: 32px; border: none; border-radius: 9999px;"
                        ).tooltip("Delete chat")

    def render_message(msg: dict):
        """Render a single message in the chat."""
        role = msg.get("role", "")
        content = msg.get("content", "")
        status = msg.get("status", "complete")
        
        if role == "user":
            with components["messages"]:
                with ui.row().classes("w-full mb-4 justify-end items-end gap-2").style("width: 100%;"):
                    with ui.column().classes("max-w-2xl").style("background: #3B82F6; color: #FFFFFF; padding: 12px 16px; border-radius: 12px 12px 4px 12px; max-width: 80%;"):
                        ui.label(content).classes("text-sm text-white")
                    ui.icon("person").classes("text-gray-400").style("font-size: 20px; flex-shrink: 0;")
        elif role == "assistant":
            with components["messages"]:
                with ui.row().classes("w-full mb-4 justify-start items-start gap-2").style("width: 100%;"):
                    ui.icon("smart_toy").classes("text-gray-400").style("font-size: 20px; flex-shrink: 0; margin-top: 4px;")
                    with ui.column().classes("max-w-2xl").style("background: #FFFFFF; border: 1px solid #E5E7EB; padding: 12px 16px; border-radius: 12px 12px 12px 4px; max-width: 80%;"):
                        if status in ["generating", "pending"]:
                            ui.label("Generating...").classes("text-xs text-gray-500 italic")
                        if content:
                            ui.label(content).classes("text-sm")

    async def handle_upload_document():
        """Handle document upload."""
        nonlocal current_chat_id
        
        # Only create new chat if no chat is currently selected
        upload_chat_id = current_chat_id
        if not upload_chat_id:
            result = await create_new_chat()
            if result and result.get("success"):
                upload_chat_id = result.get("chat", {}).get("id", "")
                current_chat_id = upload_chat_id
                await load_chat_list()
        
        # Create file upload dialog
        with ui.dialog() as dialog, ui.card().classes("w-full max-w-md"):
            ui.label("Upload Document").classes("text-lg font-semibold mb-4")
            
            # Container for file upload status in modal
            with ui.row().classes("items-center gap-2 mb-2") as file_status_container:
                spinner_icon = ui.icon("refresh").classes("text-blue-400").style("animation: spin 1s linear infinite; display: none; font-size: 20px;")
                check_icon = ui.icon("check_circle").classes("text-green-500").style("display: none; font-size: 20px;")
                filename_label = ui.label("").classes("text-sm font-medium text-gray-700")
                progress_percent_label = ui.label("").classes("text-sm font-medium text-blue-500").style("margin-left: auto;")
            file_status_container.set_visibility(False)
            
            # Define upload handler before creating upload component
            async def on_upload(e):
                try:
                    # NiceGUI upload event structure: e.file contains the file object
                    file_obj = e.file
                    
                    # Get filename from file object
                    filename = None
                    if hasattr(file_obj, 'name'):
                        filename = file_obj.name
                    elif hasattr(file_obj, 'filename'):
                        filename = file_obj.filename
                    elif hasattr(e, 'name'):
                        filename = e.name
                    
                    if not filename:
                        filename = 'uploaded_file'
                    
                    logger.info(f"File upload started: {filename}")
                    
                    # Show spinner and filename in modal
                    filename_label.text = filename
                    file_status_container.set_visibility(True)
                    spinner_icon.style("display: block; animation: spin 1s linear infinite;")
                    check_icon.style("display: none;")
                    progress_percent_label.text = "0%"
                    
                    # Simulate progress during file reading
                    async def update_progress(current: float, target: float, speed: float = 0.15):
                        """Update progress with dynamic timing."""
                        while current < target:
                            current = min(current + speed, target)
                            progress_percent_label.text = f"{int(current)}%"
                            await asyncio.sleep(0.05)
                    
                    # Start progress animation
                    progress_task = asyncio.create_task(update_progress(0, 20, 0.2))
                    
                    # Read file content from file object
                    file_content = None
                    try:
                        if hasattr(file_obj, 'read'):
                            read_method = file_obj.read
                            if asyncio.iscoroutinefunction(read_method):
                                file_content = await read_method()
                            else:
                                file_content = read_method()
                        elif isinstance(file_obj, bytes):
                            file_content = file_obj
                        else:
                            if hasattr(file_obj, 'read_bytes'):
                                read_bytes_method = file_obj.read_bytes
                                if asyncio.iscoroutinefunction(read_bytes_method):
                                    file_content = await read_bytes_method()
                                else:
                                    file_content = read_bytes_method()
                            else:
                                try:
                                    file_content = bytes(file_obj) if file_obj else None
                                except (TypeError, ValueError):
                                    pass
                    except Exception as read_error:
                        progress_task.cancel()
                        spinner_icon.style("display: none;")
                        progress_percent_label.text = ""
                        logger.error(f"Error reading file content: {read_error}")
                        raise ValueError(f"Cannot read file content: {read_error}")
                    
                    if not file_content:
                        progress_task.cancel()
                        spinner_icon.style("display: none;")
                        progress_percent_label.text = ""
                        logger.error(f"File content is None. File object type: {type(file_obj)}")
                        raise ValueError("Cannot read file content from upload event")
                    
                    # Update progress to 30% after reading
                    await update_progress(20, 30, 0.25)
                    
                    logger.info(f"Uploading file to API: {filename}, size: {len(file_content)} bytes, chat_id: {upload_chat_id}")
                    
                    # Simulate progress during upload (30% to 90%)
                    upload_progress_task = asyncio.create_task(update_progress(30, 90, 0.12))
                    
                    # Wait for 200 response from backend
                    result = await upload_document(file_content, filename, upload_chat_id)
                    
                    # Cancel progress task and complete to 100%
                    upload_progress_task.cancel()
                    await update_progress(90, 100, 0.3)
                    
                    # Only show success if we got a 200 response with success=True
                    if result and result.get("success"):
                        # Replace spinner with check icon in modal
                        progress_percent_label.text = "100%"
                        spinner_icon.style("display: none;")
                        check_icon.style("display: block;")
                        progress_percent_label.classes("text-green-500")
                        
                        logger.info(f"Upload completed successfully: {filename}")
                        ui.notify("Document uploaded successfully", color="positive")
                        
                        # Wait a moment to show check icon, then refresh list and close
                        await asyncio.sleep(0.5)
                        await load_documents_list()
                        dialog.close()
                    else:
                        # Stop spinner and show error
                        spinner_icon.style("display: none;")
                        progress_percent_label.text = ""
                        error_msg = result.get("message", "Upload failed") if result else "No response from server"
                        logger.error(f"Upload failed: {error_msg}")
                        ui.notify(f"Failed to upload document: {error_msg}", color="negative")
                except Exception as ex:
                    # Stop spinner and show error
                    spinner_icon.style("display: none;")
                    progress_percent_label.text = ""
                    error_msg = str(ex)
                    logger.error(f"Upload error: {error_msg}", exc_info=True)
                    ui.notify(f"Upload error: {error_msg}", color="negative")
            
            file_upload = ui.upload(
                on_upload=on_upload,
                auto_upload=True,
                max_file_size=10 * 1024 * 1024,  # 10MB
            ).classes("w-full")
            ui.button("Close", on_click=dialog.close).classes("mt-4").style("border-radius: 9999px; padding: 10px 24px;")
            
            # Add CSS for spinner animation
            ui.add_head_html("""
                <style>
                    @keyframes spin {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }
                </style>
            """)
        dialog.open()

    async def load_documents_list():
        """Load and display documents for current chat."""
        if components["documents_list"] is None:
            return
        
        # Clear the documents list first
        components["documents_list"].clear()
        
        # If no chat is selected, show placeholder
        if not current_chat_id:
            with components["documents_list"]:
                ui.label("Select a chat to view documents").classes("text-gray-400 text-sm").style("padding: 8px 0; margin: 0; width: 100%; text-align: left;")
            return
        
        # Load documents for the selected chat
        try:
            documents = await get_chat_documents(current_chat_id)
            if documents is None:
                with components["documents_list"]:
                    ui.label("Failed to load documents").classes("text-gray-400 text-sm").style("padding: 8px 0; margin: 0; width: 100%; text-align: left;")
                return
            
            if not documents:
                with components["documents_list"]:
                    ui.label("No documents uploaded").classes("text-gray-400 text-sm").style("padding: 8px 0; margin: 0; width: 100%; text-align: left;")
                return
            
            # Display documents
            for doc in documents:
                doc_id = doc.get("id", "")
                filename = doc.get("filename", "Unknown")
                file_type = doc.get("file_type", "").upper()
                
                with components["documents_list"]:
                    with ui.row().classes("w-full items-center gap-2 mb-2").style("padding: 10px 16px; background: #374151; border-radius: 9999px; cursor: pointer; border: 1px solid #4B5563; width: 100%; box-sizing: border-box;"):
                        ui.icon("description").classes("text-gray-400").style("font-size: 20px; flex-shrink: 0;")
                        with ui.column().classes("flex-1").style("min-width: 0; flex: 1;"):
                            ui.label(filename).classes("text-white text-sm").style("font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%;")
                        ui.button(icon="visibility", on_click=lambda did=doc_id: view_document(did)).classes("flex-shrink-0").style("background: rgba(96, 165, 250, 0.2); color: #60A5FA; padding: 8px; min-width: 40px; height: 40px; border-radius: 9999px;")
                        ui.button(icon="delete", on_click=lambda did=doc_id: handle_delete_document(did)).classes("flex-shrink-0").style("background: rgba(239, 68, 68, 0.2); color: #EF4444; padding: 8px; min-width: 40px; height: 40px; border-radius: 9999px;")
        except Exception as e:
            logger.error(f"Error loading documents list: {e}")
            with components["documents_list"]:
                ui.label(f"Error loading documents: {str(e)}").classes("text-red-400 text-sm").style("padding: 8px 0; margin: 0; width: 100%; text-align: left;")

    async def handle_delete_document(document_id: str):
        """Handle document deletion with confirmation."""
        try:
            # Get document info for confirmation
            doc = await get_document(document_id)
            filename = doc.get("filename", "this document") if doc else "this document"
            
            # Show confirmation dialog
            with ui.dialog() as dialog, ui.card().classes("w-full max-w-md"):
                ui.label(f"Delete Document?").classes("text-lg font-semibold mb-2")
                ui.label(f"Are you sure you want to delete '{filename}'? This action cannot be undone.").classes("text-gray-600 mb-4")
                
                with ui.row().classes("gap-2 justify-end"):
                    ui.button("Cancel", on_click=dialog.close).classes("px-4 py-2")
                    ui.button("Delete", on_click=lambda: dialog.submit("delete")).classes("px-4 py-2 bg-red-500 text-white")
                
                result = await dialog
                
                if result == "delete":
                    # Delete the document
                    success = await delete_document(document_id)
                    if success:
                        ui.notify(f"Document '{filename}' deleted successfully", color="positive")
                        # Reload documents list
                        await load_documents_list()
                    else:
                        ui.notify(f"Failed to delete document", color="negative")
        except Exception as e:
            logger.error(f"Error deleting document: {e}", exc_info=True)
            ui.notify(f"Failed to delete document: {e}", color="negative")

    async def view_document(document_id: str):
        """View document content in a modal."""
        try:
            doc = await get_document(document_id)
            if not doc:
                ui.notify("Document not found", color="negative")
                return
            
            filename = doc.get("filename", "Unknown")
            content = doc.get("content")
            file_type = doc.get("file_type", "").lower()
            
            # Debug logging
            logger.info(f"Viewing document {document_id}: filename={filename}, content_type={type(content)}, content_length={len(str(content)) if content else 0}")
            
            with ui.dialog() as dialog, ui.card().classes("w-full max-w-4xl").style("max-height: 80vh; overflow: hidden; display: flex; flex-direction: column; padding: 0;"):
                ui.label(f"📄 {filename}").classes("text-lg font-semibold mb-4").style("flex-shrink: 0; padding: 16px 16px 0 16px;")
                
                # Use a container with explicit styling
                content_container = ui.column().classes("flex-1").style(
                    "max-height: 60vh; "
                    "overflow-y: auto; "
                    "overflow-x: hidden; "
                    "padding: 16px; "
                    "background: #F9FAFB; "
                    "margin: 0 16px; "
                    "border-radius: 4px;"
                )
                
                with content_container:
                    if content:
                        content_str = str(content).strip()
                        
                        # Check if content looks like raw binary/PDF data (starts with %PDF or contains binary-like patterns)
                        if content_str.startswith('%PDF') or (len(content_str) > 100 and '\x00' in content_str[:100]):
                            ui.label("⚠️ Raw file content detected. The document content was not properly extracted. Please re-upload the file.").classes("text-orange-500 text-sm font-semibold").style("text-align: center; padding: 16px; background: #FEF3C7; border-radius: 4px; margin-bottom: 8px;")
                            ui.label("The system is showing raw file data instead of extracted text. This usually means the extraction process failed during upload.").classes("text-gray-600 text-xs").style("text-align: center; padding: 8px 16px;")
                        elif content_str:
                            # Format content based on file type
                            if file_type == 'json':
                                # Try to format JSON nicely
                                try:
                                    import json
                                    parsed = json.loads(content_str)
                                    formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                                    escaped_content = formatted.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                                except:
                                    escaped_content = content_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            elif file_type == 'csv':
                                # CSV can be displayed as-is or formatted as table
                                escaped_content = content_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            else:
                                # For PDF, TXT, DOCX, etc. - display as formatted text
                                escaped_content = content_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            
                            # Use HTML for better text rendering
                            ui.html(
                                f"""
                                <div style="
                                    white-space: pre-wrap;
                                    word-wrap: break-word;
                                    font-family: {'monospace' if file_type in ['json', 'csv', 'txt'] else "'Courier New', monospace"};
                                    font-size: 14px;
                                    line-height: 1.6;
                                    color: #1F2937;
                                    padding: 0;
                                    margin: 0;
                                ">{escaped_content}</div>
                                """,
                                sanitize=False
                            )
                        else:
                            ui.label("Content is empty (whitespace only)").classes("text-gray-400 text-sm").style("text-align: center; padding: 16px;")
                    else:
                        ui.label("No content available for this document. The document may be empty or the content was not extracted during upload.").classes("text-gray-400 text-sm").style("text-align: center; padding: 16px;")
                
                ui.button("Close", on_click=dialog.close).classes("mt-4").style(
                    "flex-shrink: 0; "
                    "border-radius: 9999px; "
                    "padding: 10px 24px; "
                    "background: #3B82F6; "
                    "color: white; "
                    "margin: 16px;"
                )
            
            dialog.open()
        except Exception as e:
            logger.error(f"Error viewing document: {e}", exc_info=True)
            ui.notify(f"Failed to load document: {e}", color="negative")

    async def select_chat(chat_id: str):
        """Select a chat and load its messages."""
        nonlocal current_chat_id
        current_chat_id = chat_id
        
        # Clear messages
        if components["messages"]:
            components["messages"].clear()
        
        # Load messages from database
        messages = await get_chat_messages(chat_id)
        if messages:
            for msg in messages:
                render_message(msg)
        
        await load_chat_list()
        await load_documents_list()

    async def handle_delete_chat(chat_id: str):
        """Handle delete chat button click."""
        nonlocal current_chat_id
        
        # Confirm deletion
        result = await delete_chat(chat_id)
        if result:
            # If deleted chat was active, clear the view
            if chat_id == current_chat_id:
                current_chat_id = ""
                if components["messages"]:
                    components["messages"].clear()
            ui.notify("Chat deleted", color="positive")
            await load_chat_list()

    async def handle_new_chat():
        """Handle new chat button click."""
        nonlocal current_chat_id
        result = await create_new_chat()
        if result and result.get("success"):
            chat = result.get("chat", {})
            chat_id = chat.get("id")
            if chat_id:
                current_chat_id = chat_id
                # Clear messages
                if components["messages"]:
                    components["messages"].clear()
                ui.notify("New chat created", color="positive")
                await load_chat_list()

    async def handle_send_message():
        """Handle send message button click."""
        if components["message_input"] is None or components["messages"] is None:
            return
        message = components["message_input"].value.strip()
        if not message:
            return

        components["message_input"].value = ""

        # Display user message on the right
        with components["messages"]:
            with ui.row().classes("w-full mb-4 justify-end items-end gap-2").style("width: 100%;"):
                with ui.column().classes("max-w-2xl").style("background: #3B82F6; color: #FFFFFF; padding: 12px 16px; border-radius: 12px 12px 4px 12px; max-width: 80%;"):
                    ui.label(message).classes("text-sm text-white")
                ui.icon("person").classes("text-gray-400").style("font-size: 20px; flex-shrink: 0;")

        nonlocal current_chat_id
        if not current_chat_id:
            result = await create_new_chat()
            if result and result.get("success"):
                current_chat_id = result.get("chat", {}).get("id", "")
                await load_chat_list()

        # Display bot response area on the left with icon
        with components["messages"]:
            with ui.row().classes("w-full mb-4 justify-start items-start gap-2").style("width: 100%;"):
                ui.icon("smart_toy").classes("text-gray-400").style("font-size: 20px; flex-shrink: 0; margin-top: 4px;")
                with ui.column().classes("max-w-2xl").style("background: #FFFFFF; border: 1px solid #E5E7EB; padding: 12px 16px; border-radius: 12px 12px 12px 4px; max-width: 80%;"):
                    status_label = ui.label("").classes("text-xs text-gray-500 italic").style("font-weight: 500;")
                    bot_response_label = ui.label("").classes("text-sm")

        accumulated_content = ""
        content_started = False
        try:
            async for chunk in stream_chat_response(message, current_chat_id if current_chat_id else None):
                if chunk.get("type") == "status":
                    status = chunk.get("status", "")
                    status_msg = chunk.get("message", "")
                    # Only update status if content hasn't started streaming yet
                    if not content_started:
                        if status == "Analyzing":
                            status_label.text = "🔍 Analyzing your request..."
                            status_label.set_visibility(True)
                            await asyncio.sleep(0.1)  # Small delay to make status visible
                        elif status == "Searching":
                            status_label.text = "📚 Searching documents..."
                            status_label.set_visibility(True)
                            await asyncio.sleep(0.1)  # Small delay to make status visible
                        elif status == "Generating":
                            status_label.text = "✨ Generating response..."
                            status_label.set_visibility(True)
                            await asyncio.sleep(0.1)  # Small delay to make status visible
                        elif status == "Complete":
                            status_label.text = "✓ Complete"
                            status_label.set_visibility(True)
                            # Hide status after a brief moment
                            await asyncio.sleep(0.5)
                            status_label.set_visibility(False)
                        elif status == "Error":
                            status_label.text = f"❌ Error: {status_msg}" if status_msg else "❌ Error occurred"
                            status_label.set_visibility(True)
                        else:
                            status_label.text = f"{status}: {status_msg}" if status_msg else status
                            status_label.set_visibility(True)
                elif chunk.get("type") == "content":
                    chunk_content = chunk.get("content", "")
                    accumulated_content += chunk_content
                    bot_response_label.text = accumulated_content
                    # Mark that content has started and hide status
                    if accumulated_content and not content_started:
                        content_started = True
                        status_label.set_visibility(False)
                    if chunk.get("done"):
                        status_label.set_visibility(False)
                elif chunk.get("type") == "error":
                    error_msg = chunk.get("error", "An error occurred")
                    bot_response_label.text = f"Error: {error_msg}"
                    status_label.text = "❌ Error occurred"
                    status_label.set_visibility(True)
                    ui.notify(f"Error: {error_msg}", color="negative")
        except Exception as e:
            bot_response_label.text = f"Error: {str(e)}"
            status_label.text = "❌ Error occurred"
            status_label.set_visibility(True)
            ui.notify(f"Failed to send message: {e}", color="negative")

    with ui.row().classes("w-full").style("margin: 0; padding: 0; display: flex; flex-wrap: nowrap; height: 100vh; overflow: hidden; max-height: 100vh;"):
        with ui.column().classes("w-1/4").style("background: #1F2937; border-right: 1px solid #374151; display: flex; flex-direction: column; flex-shrink: 0; height: 100vh; max-height: 100vh; overflow: hidden; margin: 0; padding: 0;"):
            # New Chat Section - Collapsible (includes chat list)
            expansion = ui.expansion("New Chat", icon="add", value=True).classes("w-full").style("flex-shrink: 0; background: #1F2937; border-bottom: 1px solid #374151; width: 100%;")
            with expansion:
                with ui.column().classes("w-full").style("display: flex; flex-direction: column; padding: 0; margin: 0; background: #1F2937; width: 100%;"):
                    # New Chat Button
                    with ui.column().classes("w-full").style("flex-shrink: 0; padding: 16px; margin: 0; background: #1F2937; width: 100%; box-sizing: border-box;"):
                        ui.button("NEW CHAT", icon="add", on_click=handle_new_chat).classes("w-full").style(
                            "background: #3B82F6; color: #FFFFFF; padding: 12px 24px; border-radius: 9999px; font-weight: 600; font-size: 14px; text-transform: uppercase; margin: 0; width: 100%; box-sizing: border-box;"
                        )
                    
                    # Chats List
                    with ui.column().classes("w-full gap-1").style("max-height: 400px; overflow-y: auto; padding: 0; margin: 0; width: 100%; box-sizing: border-box;") as chat_list:
                        components["chat_list"] = chat_list
                        pass
            
            # Knowledge Base (Documents) section - Collapsible
            with ui.expansion("Knowledge Base (Documents)", icon="folder", value=True).classes("w-full").style("flex-shrink: 0; border-top: 1px solid #374151; background: #1F2937; width: 100%;"):
                with ui.column().classes("w-full").style("padding: 16px; margin: 0; background: #1F2937; width: 100%; box-sizing: border-box;"):
                    with ui.column().classes("w-full gap-2").style("max-height: 200px; overflow-y: auto; margin: 0; padding: 0; width: 100%; box-sizing: border-box;") as documents_list:
                        components["documents_list"] = documents_list
                        # Initial placeholder - will be replaced by load_documents_list()
                        ui.label("Select a chat to view documents").classes("text-gray-400 text-sm").style("padding: 8px 0; margin: 0; width: 100%; text-align: left; box-sizing: border-box;")
            
            # Footer
            ui.label("Documents available for all chats").classes("text-gray-400 text-xs").style(
                "flex-shrink: 0; padding: 12px 16px; margin: 0; border-top: 1px solid #374151; text-align: left; width: 100%; box-sizing: border-box;"
            )

        with ui.column().classes("w-3/4").style("background: #F9FAFB; display: flex; flex-direction: column; flex-shrink: 0; height: 100vh; max-height: 100vh; overflow: hidden; margin: 0; padding: 0; width: 75%;"):
            ui.label("RAG Chatbot").classes("text-gray-900 font-semibold").style(
                "background: #FFFFFF; border-bottom: 1px solid #E5E7EB; flex-shrink: 0; padding: 12px 16px; margin: 0; width: 100%;"
            )
            with ui.column().classes("flex-1 overflow-y-auto").style("min-height: 0; padding: 24px; margin: 0; width: 100%; box-sizing: border-box; overflow-x: hidden;") as messages:
                components["messages"] = messages
            with ui.column().classes("w-full").style("border-top: 1px solid #E5E7EB; flex-shrink: 0; background: #FFFFFF; padding: 20px; margin: 0; padding-bottom: 40px; width: 100%; box-sizing: border-box;"):
                with ui.row().classes("w-full items-center gap-2").style("width: 100%;"):
                    ui.button(icon="attach_file", on_click=handle_upload_document).classes("flex-shrink-0").style(
                        "background: #3B82F6; color: #FFFFFF; padding: 12px; border-radius: 9999px; min-width: 48px; height: 48px; cursor: pointer;"
                    ).tooltip("Upload document")
                    components["message_input"] = ui.input(placeholder="Message RAG Chatbot...").classes("flex-1").style("border-radius: 9999px; padding: 12px 20px;")
                    # Add Enter key handler to input field
                    components["message_input"].on("keydown.enter", handle_send_message)
                    ui.button(icon="send", on_click=handle_send_message).classes("flex-shrink-0").style(
                        "background: #3B82F6; color: #FFFFFF; padding: 12px; border-radius: 9999px; min-width: 48px; height: 48px; cursor: pointer;"
                    ).tooltip("Send message")

    async def load_on_startup():
        await load_chat_list()
        # Load documents if a chat is already selected
        if current_chat_id:
            await load_documents_list()
    ui.timer(0.1, load_on_startup, once=True)


ui.run()
