# UltimaRAG — Multi-Agent RAG System
# Copyright (C) 2026 Pankaj Varma
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
FastAPI Main Application for UltimaRAG
REST API backend for the multi-agent RAG system.
"""

import io
import json
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional, Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from starlette.requests import Request
from pydantic import BaseModel

from ..core.config import Config, PathConfig
from ..core.utils import logger, set_seed
from ..data.chunking import DocumentChunker
from ..data.embedder import get_embedder, embed_chunks
from ..data.database import DatabaseManager, get_database as get_relational_db, init_database
from ..core.database import UltimaRAGDatabase, get_database as get_vector_db
from ..core.memory import MemoryManager
from ..agents.metacognitive_brain import MetacognitiveBrain
from ..vision.manager import MultimodalManager
from ..core.file_manager import save_upload, list_uploads, get_file_path
from .utils import (
    is_identity_query, 
    IDENTITY_RESPONSE, 
    simulate_streaming
)
from ..data.nuke_manager import NukeManager


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    context: Optional[str] = None  # Optional: pasted context to use directly
    mode: str = "hybrid"  # dense, sparse, hybrid
    include_debug: bool = False
    conversation_id: Optional[str] = None  # Optional: link to conversation
    project_id: Optional[str] = "default"
    mentioned_files: Optional[List[str]] = None  # @mention targeted file names
    use_web_search: bool = False  # Web-Breakout Agent toggle (default OFF)


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    success: bool
    query: str
    should_answer: bool
    final_response: str
    # Agent routing info
    agent_type: Optional[str] = None  # Which agent handled the query
    show_quality_indicators: bool = True  # Whether to show Fact/Relevance scores
    # Quality indicators (only shown if show_quality_indicators=True)
    confidence_score: Optional[float] = None
    fact_score: Optional[float] = None
    confidence_level: Optional[str] = None  # HIGH/MEDIUM/LOW/VERY_LOW
    # Relevance indicator (NEW)
    relevance_score: Optional[float] = None  # 0.0-1.0: is question covered by docs?
    is_relevant: Optional[bool] = None  # True if relevance_score >= 0.5
    # Transparency
    quality_warnings: Optional[List[str]] = None
    # Additional info
    citations: Optional[List[dict]] = None
    suggestions: Optional[List[str]] = None
    debug_info: Optional[dict] = None
    # Dual-response (NEW: for low-relevance queries <10%)
    dual_response: Optional[dict] = None
    # Conversation tracking
    conversation_id: Optional[str] = None
    # SOTA UI Hints (Adaptive Resonance)
    ui_hints: Optional[dict] = None



class IndexRequest(BaseModel):
    """Request for indexing documents"""
    texts: List[str]
    doc_ids: Optional[List[str]] = None
    metadata: Optional[List[dict]] = None
    conversation_id: Optional[str] = None
    project_id: Optional[str] = "default"
    folder_id: Optional[str] = "root"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ready: bool
    chunks_count: int
    agents: dict
    db_connected: bool = False

class NukeRequest(BaseModel):
    """Request to factory reset the application"""
    password: str


# =============================================================================
# CONVERSATION MODELS
# =============================================================================

class ConversationCreate(BaseModel):
    """Request to create a new conversation"""
    title: Optional[str] = None
    project_id: Optional[str] = "default"
    folder_id: Optional[str] = "root"


class ConversationResponse(BaseModel):
    """Single conversation response"""
    conversation_id: str
    title: Optional[str] = None
    conversation_created_at: str
    conversation_updated_at: str
    is_archived: bool = False
    message_count: Optional[int] = None
    success: bool = True


class ConversationListResponse(BaseModel):
    """List of conversations"""
    conversations: List[ConversationResponse]
    total: int


class MessageResponse(BaseModel):
    """Single message response"""
    message_id: str
    role: str
    content: str
    metadata: Optional[dict] = None
    message_created_at: str


class ConversationWithMessages(BaseModel):
    """Conversation with all messages"""
    conversation_id: str
    title: Optional[str] = None
    conversation_created_at: str
    conversation_updated_at: str
    messages: List[MessageResponse]


# =============================================================================
# APPLICATION STATE
# =============================================================================

class AppState:
    """Application state container (UltimaRAG Edition)"""
    brain: Optional[Any] = None # MetacognitiveBrain
    db: Optional[Any] = None # UltimaRAGDatabase
    sqlite_db: Optional[Any] = None # DatabaseManager
    memory: Optional[Any] = None # MemoryManager
    ready: bool = False
    db_connected: bool = False
    startup_error: Optional[str] = None


app_state = AppState()

# STOP GENERATION: Per-conversation abort flags
# Key: conversation_id, Value: True when abort requested
abort_flags: dict = {}


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown"""
    # Startup
    logger.info("🚀 Starting UltimaRAG API...")
    set_seed(Config.deterministic.RANDOM_SEED)
    Config.validate_context_budget()
    
    # 1. Initialize SQLite Persistence Layer
    try:
        logger.info("[1/3] Initializing SQLite Persistence Layer...")
        if init_database():
            app_state.sqlite_db = get_relational_db()
            app_state.db_connected = True
            logger.info("✅ SQLite Persistence Layer ready.")
        else:
            logger.error("❌ Failed to initialize SQLite Persistence Layer.")
    except Exception as e:
        logger.error(f"❌ Database Initialization Error: {e}")
    
    # 2. Initialize UltimaRAG Metacognitive Core
    try:
        logger.info("[2/3] Initializing UltimaRAG Metacognitive Brain...")
        app_state.db = UltimaRAGDatabase()
        app_state.memory = MemoryManager(app_state.db)
        # Pass sqlite_db so the brain can query scraped_content (lives in SQLite, not LanceDB)
        app_state.brain = MetacognitiveBrain(
            app_state.db, app_state.memory,
            sqlite_db=app_state.sqlite_db if hasattr(app_state, 'sqlite_db') and app_state.sqlite_db else None
        )
        logger.info("✅ UltimaRAG Brain initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize UltimaRAG Brain: {e}")
        import traceback
        logger.error(traceback.format_exc())
        app_state.ready = False
        app_state.startup_error = f"Brain Initialization Failed: {str(e)}"
        
    # 3. Final Readiness and Warming
    try:
        logger.info("[3/3] Finalizing UltimaRAG SOTA Stack...")
        # Pre-warm Vision/OCR Models to eliminate cold-start latency
        try:
            from ..vision.manager import MultimodalManager
            mm_mgr = MultimodalManager()
            asyncio.create_task(mm_mgr.image_proc.warm_up())
        except Exception as e:
            logger.error(f"⚠️ Model warming failed (non-fatal): {e}")

        app_state.ready = True
        logger.info("🧠 UltimaRAG SOTA Stack Fully Ready.")
    except Exception as e:
        logger.error(f"❌ Initialization failure: {e}")
        # Not fatal for the Brain, but some endpoints will fail
    
    yield
    
    # Shutdown
    logger.info("Shutting down UltimaRAG API...")
    if app_state.db:
        app_state.db.disconnect()


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="UltimaRAG API",
    description="Multi-Agent RAG System with hallucination prevention",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
UI_DIR = PathConfig.BASE_DIR / "ui"
if (UI_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

if (UI_DIR / "templates").exists():
    templates = Jinja2Templates(directory=str(UI_DIR / "templates"))
else:
    templates = None


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main UI"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h1>UltimaRAG API</h1><p>UI not found. Use /docs for API documentation.</p>")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handle favicon requests to prevent 404s"""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/health", response_model=HealthResponse)
async def health_check(conversation_id: Optional[str] = None):
    """Health check endpoint"""
    if app_state.brain:
        status = app_state.brain.get_status(conversation_id=conversation_id)
        return HealthResponse(
            status="ok",
            ready=status["ready"],
            chunks_count=status["chunks_count"],
            agents=status["agents"],
            db_connected=app_state.db_connected
        )
    return HealthResponse(
        status="initializing",
        ready=False,
        chunks_count=0,
        agents={},
        db_connected=app_state.db_connected
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    UltimaRAG Metacognitive Entry Point.
    Routes all queries through the LangGraph reasoning loop.
    """
    if not app_state.brain:
        raise HTTPException(status_code=503, detail="UltimaRAG Brain not initialized")
    
    try:
        # Parse @mentions from query text
        from ..agents.intent_classifier import parse_mentions, strip_mentions
        mentioned_files = request.mentioned_files or parse_mentions(request.query)
        clean_query = strip_mentions(request.query) if mentioned_files else request.query
        
        # Consume the generator to get the final result
        generator = await app_state.brain.run(
            query=clean_query, 
            conversation_id=request.conversation_id or "default",
            project_id=request.project_id or "default",
            mentioned_files=mentioned_files,
            original_query=request.query
        )
        
        final_result = None
        async for event in generator:
            if event["type"] == "final":
                final_result = event["result"]
        
        if not final_result:
            raise HTTPException(status_code=500, detail="Brain failed to produce a final result")
        
        # Safe access with fallback
        answer = final_result.get("answer", "") or "I could not generate a response."
        intent = final_result.get("intent", "UNKNOWN")
        if hasattr(intent, "name"):
            intent = intent.name
            
        # Transform Brain State to QueryResponse
        return QueryResponse(
            success=True,
            query=request.query,
            should_answer=True,
            final_response=answer,
            agent_type=intent,
            confidence_score=final_result.get("confidence_score", 0.0),
            ui_hints=final_result.get("ui_hints"),
            conversation_id=final_result.get("conversation_id")
        )
        
    except Exception as e:
        logger.error(f"Brain Execution Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return QueryResponse(
            success=False,
            query=request.query,
            should_answer=False,
            final_response=f"UltimaRAG Brain encountered an internal error: {str(e)}",
            conversation_id=request.conversation_id
        )


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """
    Process a query with streaming progress updates.
    Sends Server-Sent Events (SSE) with pipeline stage information.
    Includes: Identity shortcircuit, Duplicate cache, Brain pipeline.
    """
    import asyncio
    
    if not app_state.brain:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # ── Main Generator ──────────────────────────────────────────────────
    
    async def generate_stream():
        try:
            # Parse @mentions
            from ..agents.intent_classifier import parse_mentions, strip_mentions
            mentioned_files = request.mentioned_files or parse_mentions(request.query)
            clean_query = strip_mentions(request.query) if mentioned_files else request.query
            # ── INITIALIZE CONVERSATION ID ──
            conv_id = request.conversation_id or str(uuid.uuid4())
            
            # ── RESET ABORT FLAG: Ensure a clean slate for this turn ──
            abort_flags.pop(conv_id, None)
            
            # ── FULL BRAIN PIPELINE ─────────────────────────────────────
            # Yield ID early so client can STOP even on new chats
            yield f"data: {json.dumps({'stage': 'initializing', 'message': 'Initializing Metacognitive Core...', 'conversation_id': conv_id}, ensure_ascii=False)}\n\n"
            
            # ── SHORTCIRCUIT 1: Identity Query ──────────────────────────
            if is_identity_query(clean_query):
                yield f"data: {json.dumps({'stage': 'processing', 'agent': 'Identity', 'message': 'Recognized identity query', 'status': 'running'}, ensure_ascii=False)}\n\n"
                
                async for chunk in simulate_streaming(IDENTITY_RESPONSE):
                    yield chunk
                
                # Save to DB as assistant message
                try:
                    db = get_relational_db()
                    db.add_message(conv_id, "user", request.query)
                    db.add_message(conv_id, "assistant", IDENTITY_RESPONSE,
                                   metadata={"intent": "IDENTITY", "confidence_score": 1.0})
                except Exception:
                    pass
                
                yield f"data: {json.dumps({'stage': 'result', 'success': True, 'final_response': IDENTITY_RESPONSE, 'confidence_score': 1.0, 'agent_type': 'IDENTITY', 'conversation_id': conv_id}, ensure_ascii=False)}\n\n"
                return
            
            # ── NOTE: Duplicate query cache has been removed (all queries run fresh) ──
            
            # ── FULL BRAIN PIPELINE ─────────────────────────────────────
            # (Persistence is handled internally by app_state.brain.run)
            # Start the brain generator with abort checking capability
            generator = await app_state.brain.run(
                query=clean_query,
                conversation_id=conv_id,
                project_id=request.project_id or "default",
                mentioned_files=mentioned_files,
                original_query=request.query,
                use_web_search=request.use_web_search,
                check_abort_fn=lambda: abort_flags.get(conv_id)
            )
            
            # Accumulate streamed tokens so we have a fallback for final_response
            accumulated_tokens = []
            
            async for event in generator:
                # ── STOP BUTTON CHECK: Force-terminate if user clicked stop ──
                if abort_flags.get(conv_id):
                    abort_flags.pop(conv_id, None)
                    terminated_msg = "User Terminated the generation. Both the query and response were not stored to save session resources. This turn will be completely removed from the history, ensuring next time you ask this, it is treated as fresh."
                    # terminated_meta = {"terminated": True, "intent": "TERMINATED", "confidence_score": 0}
                    try:
                        # SOTA: Delete the last messages upon termination (Unified Deletion)
                        if hasattr(app_state, 'db') and app_state.db:
                            # 1. Wipe assistant (child) first to avoid FK error
                            app_state.db.delete_last_message(conv_id, "assistant", sqlite_db=app_state.sqlite_db)
                            # 2. Wipe user (parent) second
                            app_state.db.delete_last_message(conv_id, "user", sqlite_db=app_state.sqlite_db)
                            logger.info(f"Abort handled: Robustly deleted query/response Turn for {conv_id}.")
                    except Exception as e:
                        logger.error(f"Error during abort cleanup: {e}")
                    
                    yield f"data: {json.dumps({'stage': 'terminated', 'message': terminated_msg}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'stage': 'result', 'success': True, 'final_response': terminated_msg, 'confidence_score': 0, 'agent_type': 'TERMINATED', 'conversation_id': conv_id}, ensure_ascii=False)}\n\n"
                    return

                if event["type"] == "status":
                    status_data = {
                        "stage": "processing",
                        "agent": event["agent"],
                        "message": event["stage"],
                        "status": event["status"]
                    }
                    yield f"data: {json.dumps(status_data, ensure_ascii=False)}\n\n"
                
                elif event["type"] == "thought":
                    # STREAMING THOUGHT-UI: Forward rich thought event to frontend
                    yield f"data: {json.dumps({'type': 'thought', 'agent': event.get('agent'), 'action': event.get('action')}, ensure_ascii=False)}\n\n"
                
                elif event["type"] == "token":
                    accumulated_tokens.append(event["token"])
                    yield f"data: {json.dumps({'stage': 'streaming', 'token': event['token']}, ensure_ascii=False)}\n\n"
                
                elif event["type"] == "final":
                    result = event["result"]
                    # Use brain answer if available, otherwise use accumulated tokens
                    final_answer = result.get('answer', '') or ''.join(accumulated_tokens)
                    
                    # (Persistence and metadata handling moved to Brain core)

                    final_data = {
                        'stage': 'result',
                        'success': True,
                        'final_response': final_answer,
                        'confidence_score': result.get('confidence_score'),
                        'agent_type': result.get('intent'),
                        'ui_hints': result.get('ui_hints'),
                        'conversation_id': result.get('conversation_id'),
                    }
                    yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming Error: {e}")
            yield f"data: {json.dumps({'stage': 'error', 'message': f'Metacognitive Error: {str(e)}'}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ── STOP GENERATION ENDPOINT ─────────────────────────────────────────────

class AbortRequest(BaseModel):
    """Request to abort generation for a conversation"""
    conversation_id: str

@app.post("/query/abort")
async def abort_query(request: AbortRequest):
    """
    Force-terminate generation for a specific conversation.
    Sets an abort flag that the streaming loop checks on each iteration.
    """
    conv_id = request.conversation_id
    if not conv_id:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    
    abort_flags[conv_id] = True
    logger.info(f"ABORT requested for conversation: {conv_id}")
    return {"success": True, "message": "Abort signal sent", "conversation_id": conv_id}


@app.post("/index")
async def index_documents(request: IndexRequest):
    """
    Index documents into the knowledge base.
    
    Provide texts and optional doc_ids/metadata.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    try:
        # Create doc_ids if not provided
        doc_ids = request.doc_ids or [f"doc_{i}" for i in range(len(request.texts))]
        metadata = request.metadata or [{} for _ in request.texts]
        
        # Prepare documents
        documents = [
            {"text": text, "doc_id": doc_id, "metadata": meta}
            for text, doc_id, meta in zip(request.texts, doc_ids, metadata)
        ]
        
        # Chunk documents
        chunker = DocumentChunker(
            strategy="semantic",
            chunk_size=Config.chunking.CHUNK_SIZE,
            overlap=Config.chunking.CHUNK_OVERLAP
        )
        chunks = chunker.chunk_documents(documents)
        
        # Generate embeddings
        embedder = get_embedder()
        chunks = embed_chunks(chunks, embedder)
        
        # Add to LanceDB
        processed_chunks = []
        for c in chunks:
            processed_chunks.append({
                "text": c["text"],
                "vector": c["embedding"].tolist(),
                "file_name": c["source"],
                "project_id": request.project_id,
                "conversation_id": request.conversation_id or "default",
                "folder_id": request.folder_id,
                "metadata": c.get("metadata", {})
            })
        
        app_state.db.add_knowledge(processed_chunks)
        app_state.ready = True
        

        
        return {
            "success": True,
            "message": f"Indexed {len(request.texts)} documents into {len(chunks)} chunks in project {request.project_id}",
            "conversation_id": request.conversation_id
        }
        
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    project_id: Optional[str] = Form("default"),
    folder_id: Optional[str] = Form("root")
):
    """
    Upload and index a single document file.
    
    Supports .txt, .md, and .pdf files.
    """
    filename = file.filename.lower()
    
    # Supported extensions
    doc_exts = ('.txt', '.md', '.pdf')
    media_exts = ('.png', '.jpg', '.jpeg', '.mp3', '.wav', '.mp4', '.mov', '.webm')
    
    if not filename.endswith(doc_exts + media_exts):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {doc_exts + media_exts}"
        )
    
    try:
        content = await file.read()
        
        # 1. Handle Multimodal files (Images, Audio, Video)
        if filename.endswith(media_exts):
            logger.info(f"Routing {filename} to Multimodal Pipeline")
            
            # Ensure we have a conversation context (or use 'default')
            chat_id = conversation_id or "default"
            
            # Ensure conversation exists in DB to prevent FK error
            if app_state.db_connected and app_state.db:
                app_state.db.ensure_conversation(chat_id)
            
            # Save file to chat-specific directory
            local_path = save_upload(
                conversation_id=chat_id,
                file_type=filename.split('.')[-1],
                file_name=file.filename,
                file_content=content
            )
            
            # Process multimodal content
            from ..vision.manager import MultimodalManager
            vision_mgr = MultimodalManager()
            result = await vision_mgr.process_file(
                conversation_id=chat_id,
                file_path=local_path,
                file_type=filename.split('.')[-1],
                file_name=file.filename
            )
            
            return {
                "success": True,
                "message": f"Successfully processed {filename} in multimodal pipeline",
                "multimodal_result": result,
                "conversation_id": chat_id
            }

        # 2. Extract text for traditional documents (.pdf, .txt, .md)
        if filename.endswith('.pdf'):
            # SOTA: Use DocumentProcessor for PDF (Text + Image extraction)
            from ..core.document_processor import DocumentProcessor
            
            # Ensure folder exists
            chat_id = conversation_id or "default"
            local_path = save_upload(chat_id, 'pdf', file.filename, content)
            
            text, image_paths = DocumentProcessor.extract_from_pdf(chat_id, local_path, file.filename)
            
            # Index text first
            idx_req = IndexRequest(
                texts=[text],
                doc_ids=[file.filename],
                metadata=[{"filename": file.filename, "file_name": file.filename, "source": file.filename}],
                conversation_id=chat_id,
                project_id=project_id,
                folder_id=folder_id
            )
            await index_documents(idx_req)
            
            # Route extracted images to Multimodal Pipeline
            from ..vision.manager import MultimodalManager
            vision_mgr = MultimodalManager()
            for img_path in image_paths:
                img_name = os.path.basename(img_path)
                await vision_mgr.process_file(
                    conversation_id=chat_id,
                    file_path=img_path,
                    file_type=img_name.split('.')[-1],
                    file_name=img_name
                )
            
            return {
                "success": True, 
                "message": f"Successfully processed {filename}. Extracted {len(image_paths)} images.",
                "conversation_id": chat_id
            }
        else:
            # Plain text files
            text = content.decode('utf-8')
        
        # 3. Index the document
        request = IndexRequest(
            texts=[text],
            doc_ids=[file.filename],
            metadata=[{"filename": file.filename, "file_name": file.filename, "source": file.filename}],
            conversation_id=conversation_id or "default",
            project_id=project_id,
            folder_id=folder_id
        )

        return await index_documents(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks")
async def list_chunks(limit: int = 10):
    """List indexed chunks (for debugging)"""
    chunks = app_state.chunks[:limit]
    return {
        "total": len(app_state.chunks),
        "showing": len(chunks),
        "chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "source": c.get("source"),
                "text_preview": c.get("text", "")[:200] + "..."
            }
            for c in chunks
        ]
    }


@app.post("/query/unified")
async def unified_query(
    query: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    project_id: Optional[str] = Form("default"),
    use_web_search: Optional[str] = Form("false"),
    files: List[UploadFile] = File(None)
):
    """
    ChatGPT-style unified entry point: handles optional file uploads + query in one request.
    Broadcasts status updates via streaming to allow UI state switches.
    """
    # Parse form string to bool
    _use_web_search = str(use_web_search).lower() == "true"
    if not app_state.brain:
        error_detail = app_state.startup_error or "UltimaRAG Brain not initialized. Please check server logs."
        raise HTTPException(status_code=503, detail=error_detail)
    
    async def unified_generator():
        try:
            import uuid
            chat_id = conversation_id or str(uuid.uuid4())
            
            # ── RESET ABORT FLAG: Ensure a clean slate for this turn ──
            abort_flags.pop(chat_id, None)
            
            # SOTA: Immediately ping the UI with the conversation ID so the STOP button can bind to it
            yield f"data: {json.dumps({'stage': 'initializing', 'conversation_id': chat_id}, ensure_ascii=False)}\n\n"
            
            # SOTA Sync: Ensure conversation exists in BOTH metadata (SQLite) and vector (LanceDB)
            if app_state.db:
                # Unified layer handles both if sqlite_db is passed
                app_state.db.ensure_conversation(chat_id, title="New Chat", sqlite_db=app_state.sqlite_db)
            elif app_state.sqlite_db:
                # Fallback for relational only
                app_state.sqlite_db.ensure_conversation(chat_id)
            
            # 1. Process files if any
            processed_files = []
            enrichment_tasks = []
            
            if files:
                for file in files:
                    if abort_flags.get(chat_id): break # Abort during upload phase
                    
                    content = await file.read()
                    local_path = save_upload(chat_id, file.filename.split('.')[-1], file.filename, content)
                    
                    # SOTA: Process multimodal or text
                    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.mp4', '.mov', '.webm', '.mkv', '.avi')):
                        from ..vision.manager import MultimodalManager
                        vision_mgr = MultimodalManager()
                        res = await vision_mgr.process_file(chat_id, local_path, file.filename.split('.')[-1], file.filename)
                        if res.get("enrichment_task"):
                            enrichment_tasks.append(res["enrichment_task"])
                    elif file.filename.lower().endswith(('.txt', '.md', '.pdf')):
                        # Standard document indexing
                        text = ""
                        if file.filename.lower().endswith('.pdf'):
                            from ..core.document_processor import DocumentProcessor
                            text, image_paths = await DocumentProcessor.extract_from_pdf(chat_id, local_path, file.filename)
                            
                            # Process extracted images
                            from ..vision.manager import MultimodalManager
                            vm = MultimodalManager()
                            for ip in image_paths:
                                iname = os.path.basename(ip)
                                res = await vm.process_file(chat_id, ip, iname.split('.')[-1], iname)
                                if res.get("enrichment_task"):
                                    enrichment_tasks.append(res["enrichment_task"])
                        else:
                            text = content.decode('utf-8', errors='ignore')
                        
                        if text.strip():
                            idx_req = IndexRequest(
                                texts=[text],
                                doc_ids=[file.filename],
                                metadata=[{"filename": file.filename, "file_name": file.filename, "source": file.filename}],
                                conversation_id=chat_id,
                                project_id=project_id
                            )
                            await index_documents(idx_req)
                    processed_files.append(file.filename)

            if abort_flags.get(chat_id):
                # Terminated early during uploads
                yield f"data: {json.dumps({'stage': 'terminated', 'message': 'Upload terminated.'}, ensure_ascii=False)}\n\n"
                return

            # 2. Await background enrichment tasks if any
            if enrichment_tasks:
                yield f"data: {json.dumps({'stage': 'status', 'status': 'enriching', 'message': 'Performing Cognitive Enrichment...'}, ensure_ascii=False)}\n\n"
                await asyncio.gather(*enrichment_tasks)
            
            # SOTA Signal: Enrichment is complete, UI can switch now
            yield f"data: {json.dumps({'stage': 'status', 'status': 'enrichment_complete', 'message': 'Cognitive Assets Secured.'}, ensure_ascii=False)}\n\n"

            # 3b. SOTA: Workspace Sync — update workspace BEFORE answering so the file
            #     immediately appears in the Workspace panel before the AI starts responding.
            if processed_files:
                try:
                    workspace_files = app_state.db.get_assets(chat_id) if app_state.db else []
                    file_list = [
                        {"name": a.get("file_name", ""), "type": a.get("file_type", "file")}
                        for a in workspace_files
                    ]
                    yield f"data: {json.dumps({'stage': 'workspace_updated', 'files': file_list, 'conversation_id': chat_id}, ensure_ascii=False)}\n\n"
                    logger.info(f"Workspace sync: emitted {len(file_list)} files for {chat_id}")
                except Exception as ws_e:
                    logger.warning(f"Workspace sync emit failed (non-fatal): {ws_e}")

            working_query = query
            if processed_files:
                tags = " ".join([f"@{f}" for f in processed_files])
                working_query = f"{tags} {query}".strip()
                logger.info(f"Auto-tagging: Appended {tags} to query due to file upload")

            # 4. Intent Detection: Check for summarize intent BEFORE running the brain
            from ..agents.intent_classifier import parse_mentions, strip_mentions
            mentioned_files = parse_mentions(working_query)
            clean_query = strip_mentions(working_query) if mentioned_files else working_query
            
            # 5. Run the Brain (Streaming) — standard RAG path
            generator = await app_state.brain.run(
                query=clean_query,
                conversation_id=chat_id,
                project_id=project_id,
                mentioned_files=mentioned_files,
                uploaded_files=processed_files,  # Pass uploaded files for auto-tagging
                use_web_search=_use_web_search,
                original_query=working_query,
                check_abort_fn=lambda: abort_flags.get(chat_id)
            )
            
            final_result = None
            accumulated_tokens = []
            
            async for event in generator:
                # ── STOP BUTTON CHECK: Force-terminate if user clicked stop ──
                if abort_flags.get(chat_id):
                    abort_flags.pop(chat_id, None)
                    terminated_msg = "User Terminated the generation. Both the query and response were not stored to save session resources. This turn will be completely removed from the history, ensuring next time you ask this, it is treated as fresh."
                    # terminated_meta = {"terminated": True, "intent": "TERMINATED", "confidence_score": 0}
                    try:
                        # SOTA: Delete the last messages upon termination (Unified Deletion)
                        if hasattr(app_state, 'db') and app_state.db:
                            # 1. Wipe assistant (child) first to avoid FK error
                            app_state.db.delete_last_message(chat_id, "assistant", sqlite_db=app_state.sqlite_db)
                            # 2. Wipe user (parent) second
                            app_state.db.delete_last_message(chat_id, "user", sqlite_db=app_state.sqlite_db)
                            logger.info(f"Abort handled Robustly deleted query/response Turn for {chat_id}.")
                    except Exception as e:
                        logger.error(f"Error during abort cleanup: {e}")
                        
                    yield f"data: {json.dumps({'stage': 'terminated', 'message': terminated_msg}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'stage': 'result', 'success': True, 'final_response': terminated_msg, 'confidence_score': 0, 'agent_type': 'TERMINATED', 'conversation_id': chat_id}, ensure_ascii=False)}\n\n"
                    return

                if event["type"] == "token":
                    token = event.get("token", "")
                    accumulated_tokens.append(token)
                    yield f"data: {json.dumps({'stage': 'streaming', 'token': token}, ensure_ascii=False)}\n\n"
                elif event["type"] == "thought":
                    # STREAMING THOUGHT-UI: Forward rich thought event to frontend
                    yield f"data: {json.dumps({'type': 'thought', 'agent': event.get('agent'), 'action': event.get('action')}, ensure_ascii=False)}\n\n"
                elif event["type"] == "status":
                    yield f"data: {json.dumps({'stage': 'processing', 'agent': event.get('agent'), 'message': event.get('stage')}, ensure_ascii=False)}\n\n"
                elif event["type"] == "final":
                    final_result = event["result"]

            if not final_result:
                answer = ''.join(accumulated_tokens)
                yield f"data: {json.dumps({'stage': 'result', 'final_response': answer, 'conversation_id': chat_id, 'processed_files': processed_files}, ensure_ascii=False)}\n\n"
            else:
                answer = final_result.get("answer", "") or ''.join(accumulated_tokens)
                intent = final_result.get("intent", "UNKNOWN")
                if hasattr(intent, "name"): intent = intent.name
                
                result_payload = {
                    'stage': 'result',
                    'final_response': answer,
                    'conversation_id': chat_id,
                    'processed_files': processed_files,
                    'agent_type': intent,
                    'confidence_score': final_result.get('confidence_score'),
                    'ui_hints': final_result.get('ui_hints')
                }
                yield f"data: {json.dumps(result_payload, ensure_ascii=False)}\n\n"
                
        except Exception as e:
            logger.error(f"Unified query failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(unified_generator(), media_type="text/event-stream")


@app.get("/status")
async def get_status():
    """Get detailed system status"""
    if app_state.brain:
        return app_state.brain.get_status()
    return {"ready": False, "error": "Brain not initialized"}


@app.post("/projects")
async def create_project(name: str = Form(...), description: str = Form(""), color_code: str = Form("#3B82F6")):
    """Create a new project."""
    try:
        project_id = app_state.db.create_project(name, description, color_code)
        return {"success": True, "project_id": project_id}
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspace/tree")
async def get_workspace_tree():
    """Returns a nested JSON of Projects -> Folders -> Chats."""
    try:
        tree = app_state.db.get_workspace_tree()
        return {"success": True, "tree": tree}
    except Exception as e:
        logger.error(f"Error getting workspace tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspace/recent")
async def get_recent_conversations(limit: int = 15):
    """Returns the most recently active conversations."""
    try:
        # Use relational DB for recent history metadata
        if app_state.sqlite_db:
            conversations = app_state.sqlite_db.list_conversations(limit=limit)
        else:
            conversations = app_state.db.list_recent_conversations(limit)
        return {"success": True, "conversations": conversations}
    except Exception as e:
        logger.error(f"Error getting recent conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workspace/files/{conversation_id}")
async def get_workspace_files(conversation_id: str):
    """Returns the list of uploaded files for a conversation."""
    try:
        files = list_uploads(conversation_id)
        return {"success": True, "files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Error listing workspace files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workspace/files/{conversation_id}/view/{file_name:path}")
async def view_workspace_file(conversation_id: str, file_name: str):
    """Serve an uploaded file for viewing in the browser."""
    import mimetypes
    file_path = get_file_path(conversation_id, file_name)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(
        path=str(file_path),
        media_type=mime_type or "application/octet-stream",
        filename=file_name
    )


@app.get("/workspace/conversations/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation_with_messages_endpoint(conversation_id: str):
    """Fetch a single conversation with all its active messages."""
    if not app_state.sqlite_db:
        raise HTTPException(status_code=503, detail="Primary database not available")
    
    try:
        # Use relational DB as Source of Truth
        conv = app_state.sqlite_db.get_conversation(conversation_id)
        if not conv:
            # Fallback to LanceDB if missing from SQLite
            if app_state.db:
                conv = app_state.db.get_conversation(conversation_id)
            
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Fetch active messages from relational DB (Source of Truth for history)
        messages = app_state.sqlite_db.get_active_messages(conversation_id)
        
        # Map to MessageResponse
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(MessageResponse(
                message_id=str(msg["id"]),
                role=msg["role"],
                content=msg["content"],
                metadata=msg["metadata"] if isinstance(msg["metadata"], dict) else json.loads(msg.get("metadata", "{}")),
                message_created_at=str(msg["created_at"])
            ))
        
        return ConversationWithMessages(
            conversation_id=str(conv["id"]),
            title=conv.get("name") or conv.get("title") or "Untitled",
            conversation_created_at=str(conv.get("created_at", "")),
            conversation_updated_at=str(conv.get("updated_at", "")),
            messages=formatted_messages
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/folders")
async def create_folder(name: str = Form(...), project_id: str = Form(...), parent_folder_id: Optional[str] = Form("root")):
    """Create a new folder in a project."""
    try:
        folder_id = app_state.db.create_folder(name, project_id, parent_folder_id)
        return {"success": True, "folder_id": folder_id}
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workspace/projects")
async def list_projects():
    """List all workspace projects."""
    try:
        projects = app_state.db.list_projects()
        return {"success": True, "projects": projects}
    except Exception as e:
        logger.error(f"Error listing projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations")
async def create_conversation_endpoint(req: ConversationCreate):
    """Create a new conversation across both Relational and Vector stores."""
    try:
        import uuid
        conv_id = str(uuid.uuid4())
        title = req.title or "New Inquiry"
        
        # 1. Create in Relational DB (Source of Truth)
        if app_state.sqlite_db:
            app_state.sqlite_db.create_conversation(title=title, user_id="default", conversation_id=conv_id)
            
        # 2. Create in Vector DB (Semantic Context)
        if app_state.db:
            app_state.db.create_conversation(name=title, project_id=req.project_id, folder_id=req.folder_id, conversation_id=conv_id)
            
        return {"success": True, "conversation_id": conv_id}
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}/documents")
async def get_conversation_documents(conversation_id: str):
    """
    Get all documents associated with a specific conversation.
    Used for the chat-scoped workspace feature.
    """
    if not app_state.db_connected or not app_state.db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Check if conversation exists
        conv = app_state.db.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get all documents for this conversation
        documents = app_state.db.get_documents_by_chat(conversation_id)
        
        # Format response
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "file_id": doc.get("file_id"),
                "file_name": doc.get("file_name"),
                "file_type": doc.get("file_type"),
                "total_pages": doc.get("total_pages"),
                "duration_sec": doc.get("duration_sec"),
                "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None
            })
        
        return {
            "conversation_id": conversation_id,
            "documents": formatted_docs,
            "total": len(formatted_docs)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages across both semantic and relational stores."""
    if not app_state.db_connected:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # 1. Delete from Semantic Store (LanceDB) - Vectors, assets, enriched content
        semantic_deleted = False
        if app_state.db:
            semantic_deleted = app_state.db.delete_conversation(conversation_id)
        
        # 2. Delete from Relational Store (SQLite) - Metadata, UI sidebar entry
        relational_deleted = False
        if app_state.sqlite_db:
            relational_deleted = app_state.sqlite_db.delete_conversation(conversation_id)
            
        # 3. Delete physical files
        from ..core.file_manager import delete_chat_dir
        delete_chat_dir(conversation_id)
            
        if not semantic_deleted and not relational_deleted:
            raise HTTPException(status_code=404, detail="Conversation not found in any database")
        
        return {"success": True, "message": "Conversation purged from all systems"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ConversationUpdate(BaseModel):
    """Request to update a conversation"""
    title: Optional[str] = None
    is_archived: Optional[bool] = None


@app.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(conversation_id: str, update: ConversationUpdate):
    """
    Update a conversation's title or archive status.
    Ensures both SQLite (Metadata) and LanceDB (Vector) are in sync.
    """
    if not app_state.sqlite_db:
        raise HTTPException(status_code=503, detail="Relational database not available")
    
    try:
        # 1. Update SQLite (Source of Truth for Sidebar/Listings)
        conv = app_state.sqlite_db.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        success_sqlite = app_state.sqlite_db.update_conversation(
            conversation_id,
            title=update.title,
            is_archived=update.is_archived
        )
        
        # 2. Synchronize to LanceDB (Semantic Store)
        if app_state.db:
            app_state.db.update_conversation(
                conversation_id,
                title=update.title,
                is_archived=update.is_archived
            )
            
        if not success_sqlite:
            raise HTTPException(status_code=500, detail="Failed to update conversation in metadata store")
        
        # 3. Fetch final state
        updated_conv = app_state.sqlite_db.get_conversation(conversation_id)
        
        return ConversationResponse(
            conversation_id=str(updated_conv["conversation_id"]),
            title=updated_conv.get("title") or "Untitled",
            conversation_created_at=str(updated_conv.get("conversation_created_at", "")),
            conversation_updated_at=str(updated_conv.get("conversation_updated_at", "")),
            is_archived=bool(updated_conv.get("is_archived", False)),
            message_count=app_state.sqlite_db.get_message_count(conversation_id),
            success=True
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SPECIALIZED TASK ENDPOINTS
# =============================================================================

@app.post("/summarize")
async def summarize_text_endpoint(text: str = Form(...), word_count: int = Form(100)):
    """Summarize provided text using the Metacognitive Brain."""
    if not app_state.brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    try:
        query = f"Summarize the following text in approximately {word_count} words: {text}"
        
        # Run through the brain
        generator = await app_state.brain.run(
            query=query,
            conversation_id="summarize_task",
            project_id="system"
        )
        
        final_result = None
        async for event in generator:
            if event["type"] == "final":
                final_result = event["result"]
        
        if not final_result:
            raise HTTPException(status_code=500, detail="Summarization failed")
            
        return {
            "success": True,
            "summary": final_result.get("answer", ""),
            "status": "SUCCESS"
        }
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
async def translate_text_endpoint(text: str = Form(...), target_language: str = Form("Hindi")):
    """Translate provided text using the Metacognitive Brain."""
    if not app_state.brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
        
    try:
        query = f"Translate the following text to {target_language}: {text}"
        
        generator = await app_state.brain.run(
            query=query,
            conversation_id="translate_task",
            project_id="system"
        )
        
        final_result = None
        async for event in generator:
            if event["type"] == "final":
                final_result = event["result"]
                
        return {
            "success": True,
            "translation": final_result.get("answer", ""),
            "target_language": target_language,
            "status": "SUCCESS"
        }
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rewrite")
async def rewrite_text_endpoint(text: str = Form(...), style: str = Form("natural")):
    """Rewrite provided text using the Metacognitive Brain."""
    if not app_state.brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
        
    try:
        query = f"Rewrite the following text in a {style} style: {text}"
        
        generator = await app_state.brain.run(
            query=query,
            conversation_id="rewrite_task",
            project_id="system"
        )
        
        final_result = None
        async for event in generator:
            if event["type"] == "final":
                final_result = event["result"]
                
        return {
            "success": True,
            "rewritten_text": final_result.get("answer", ""),
            "style": style,
            "status": "SUCCESS"
        }
    except Exception as e:
        logger.error(f"Rewriting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_endpoint(message: str = Form(...)):
    """Handle general chat using the Metacognitive Brain."""
    if not app_state.brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
        
    try:
        generator = await app_state.brain.run(
            query=message,
            conversation_id="global_chat",
            project_id="system"
        )
        
        final_result = None
        async for event in generator:
            if event["type"] == "final":
                final_result = event["result"]
                
        return {
            "success": True,
            "response": final_result.get("answer", ""),
            "status": "SUCCESS"
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/route")
async def route_query_endpoint(query: str = Form(...), has_documents: bool = Form(False), has_context: bool = Form(False)):
    """Determine which agent should handle a query using the unified classifier."""
    try:
        from ..agents.intent_classifier import IntentClassifier
        classifier = IntentClassifier()
        intent, lang = await classifier.classify(query)
        
        return {
            "agent_type": str(intent),
            "agent_name": str(intent),
            "confidence": 0.95,
            "reason": "Unified Intent Classification",
            "requires_context": intent in ["RAG", "PERCEPTION"],
            "requires_documents": intent == "RAG",
            "extracted_params": {"target_language": lang}
        }
    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WORKSPACE FILE ACCESS ENDPOINTS
# =============================================================================

@app.get("/api/workspace/files")
async def list_workspace_files(conversation_id: str):
    """List all uploaded files for a conversation.
    
    Returns a list of files with name, type, size, and download URL.
    """
    from ..core.file_manager import list_uploads
    
    try:
        files = list_uploads(conversation_id)
        
        # Add download URLs for each file
        for file in files:
            file["url"] = f"/api/workspace/file?conversation_id={conversation_id}&file_name={file['name']}"
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "files": files,
            "total": len(files)
        }
    except Exception as e:
        logger.error(f"Error listing workspace files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspace/file")
async def get_workspace_file(conversation_id: str, file_name: str):
    """Serve a specific uploaded file for viewing or download."""
    from ..core.file_manager import get_file_path
    from fastapi.responses import FileResponse
    
    try:
        file_path = get_file_path(conversation_id, file_name)
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File '{file_name}' not found")
        
        # Determine media type based on extension
        extension = file_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".webm": "video/webm"
        }
        
        media_type = media_types.get(extension, "application/octet-stream")
        
        return FileResponse(
            path=str(file_path),
            filename=file_name,
            media_type=media_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {file_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspace/files/autocomplete")
async def autocomplete_files(conversation_id: str, prefix: str = ""):
    """Return file names matching a prefix for @mention autocomplete.
    
    Used by the frontend when the user types '@' followed by characters.
    Returns files from the current conversation's uploads.
    """
    from ..core.file_manager import list_uploads
    
    try:
        files = list_uploads(conversation_id)
        prefix_lower = prefix.lower()
        
        matching = [
            {
                "name": f["name"],
                "type": f.get("type", "file"),
                "size": f.get("size", 0)
            }
            for f in files
            if f["name"].lower().startswith(prefix_lower) or not prefix
        ]
        
        return {
            "success": True,
            "conversation_id": conversation_id,
            "suggestions": matching[:10]  # Limit to 10 suggestions
        }
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return {"success": False, "suggestions": []}

# =============================================================================
# PDF EXPORT: QUERY-BASED GENERATION
# =============================================================================

# In-memory session store: {session_token: {query, response, conversation_id, mentioned_files, title}}
_pdf_sessions: Dict[str, Dict] = {}

@app.post("/api/pdf/query-generate")
async def pdf_query_generate(
    query: str = Form(...),
    conversation_id: str = Form(...),
    mentioned_files: Optional[str] = Form(None)   # JSON array string, e.g. '["doc.pdf"]'
):
    """
    Run the query through the RAG/summarize brain and store the response for PDF preview.
    Returns: {session_token, response_text, mentioned_files}
    """
    import uuid, json as _json
    if not app_state.brain:
        raise HTTPException(status_code=503, detail="UltimaRAG Brain not initialized.")

    # Parse mentioned files
    files_list: List[str] = []
    if mentioned_files:
        try:
            files_list = _json.loads(mentioned_files)
        except Exception:
            files_list = [f.strip() for f in mentioned_files.split(",") if f.strip()]

    # Build the working query (prepend @mentions)
    working_query = query
    if files_list:
        tags = " ".join([f"@{f}" for f in files_list])
        working_query = f"{tags} {query}".strip()

    # Collect full streaming response into a string
    collected = []
    try:
        from ..agents.intent_classifier import strip_mentions
        from ..agents.query_analyzer import QueryAnalyzer
        import json as _json

        clean_query = strip_mentions(working_query) if files_list else working_query
        qa = QueryAnalyzer()
        analysis = qa._analyze_fallback(clean_query)

        if analysis.get('intent') == 'summarize':
            # Run the Map-Reduce Generator (yields SSE strings)
            generator = summarize_intent_generator(clean_query, conversation_id, files_list)
            async for event in generator:
                if isinstance(event, str) and event.startswith("data: "):
                    try:
                        # Slice off "data: " and any trailing whitespace including \n\n
                        data = _json.loads(event[6:].strip())
                        stage = data.get("stage")
                        
                        if stage == "summarize_file_start":
                            fn = data.get("file_name", "Document")
                            collected.append(f"\n\n--- DOCUMENT SUMMARY: {fn} ---\n\n")
                        elif stage in ("summarize_token", "summary_bubble", "streaming"):
                            t = data.get("token")
                            if t is None: t = data.get("summary", "")
                            collected.append(t)
                        elif stage == "summarize_file_end":
                            err = data.get("error")
                            if err:
                                collected.append(f"\n[Error processing: {err}]\n")
                            collected.append("\n\n")
                        elif stage == "summarize_no_files":
                            collected.append(data.get("message", "No document context available for summarization."))
                    except Exception:
                        pass
        else:
            # Normal RAG path (yields dicts)
            generator = await app_state.brain.run(
                query=clean_query,
                conversation_id=conversation_id,
                project_id="default",
            )
            async for chunk in generator:
                if isinstance(chunk, dict):
                    stage = chunk.get("stage")
                    if stage == "streaming":
                        collected.append(chunk.get("token", ""))
                    elif stage == "result":
                        if chunk.get("final_response"):
                            collected = [chunk.get("final_response", "")]
                        break
                elif isinstance(chunk, str):
                    collected.append(chunk)

    except Exception as e:
        logger.error(f"PDF query-generate brain error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    response_text = "".join(collected).strip()
    if not response_text:
        raise HTTPException(status_code=500, detail="Brain returned empty response.")

    # Store in session
    session_token = str(uuid.uuid4())
    db = get_relational_db()
    conv = db.get_conversation(conversation_id) if db else {}
    conv_title = (conv or {}).get("title") or (conv or {}).get("name") or "UltimaRAG Export"

    _pdf_sessions[session_token] = {
        "query": query,
        "response": response_text,
        "conversation_id": conversation_id,
        "mentioned_files": files_list,
        "title": conv_title,
    }

    return {
        "success": True,
        "session_token": session_token,
        "response_text": response_text,
        "mentioned_files": files_list,
    }


@app.get("/api/pdf/download/{session_token}")
async def pdf_download(session_token: str):
    """
    Convert a previously-generated response (identified by session_token) to PDF and return it.
    """
    session = _pdf_sessions.get(session_token)
    if not session:
        raise HTTPException(status_code=404, detail="PDF session expired or not found. Please regenerate.")

    try:
        from ..core.pdf_exporter import generate_query_pdf
        pdf_bytes = generate_query_pdf(
            query=session["query"],
            response=session["response"],
            conversation_id=session["conversation_id"],
            mentioned_files=session.get("mentioned_files"),
            conversation_title=session.get("title", "UltimaRAG Export"),
        )
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in session.get("title", "Export"))[:40]
        filename = f"UltimaRAG_QueryExport_{safe_title}.pdf"

        # Clean up session after download (optional: keep for re-downloads)
        # _pdf_sessions.pop(session_token, None)

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            }
        )
    except Exception as e:
        logger.error(f"PDF download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CONVERSATION EXPORT (Full Conversation PDF)
# =============================================================================

@app.post("/api/conversations/{conversation_id}/export/pdf")
async def export_conversation_pdf(conversation_id: str, scope: Optional[str] = "full"):
    """Export a conversation as a structured PDF document with smart scoping."""
    try:
        db = get_relational_db()
        
        # Get conversation metadata
        conversation = db.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages
        messages = db.get_active_messages(conversation_id)
        if not messages:
            raise HTTPException(status_code=404, detail="No messages in this conversation")
            
        # SOTA: Smart Scope Logic
        # If user query analyzes the last turn, scope=latest is extremely useful.
        # If they want an overview, we prefer the 'full' or 'summary' scope.
        
        # Generate PDF with scope context
        from ..core.pdf_exporter import generate_conversation_pdf
        pdf_bytes = generate_conversation_pdf(conversation, messages, scope=scope)
        
        # Build filename
        title = conversation.get('title') or conversation.get('name') or 'conversation'
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)[:50]
        filename = f"UltimaRAG_{safe_title}_{scope}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.post("/api/admin/nuke")
async def admin_nuke(request: NukeRequest):
    """
    Administrative endpoint to wipe all data.
    """
    success, message = NukeManager.nuke_all_data(request.password)
    
    if not success:
        # Check if auth failure or system failure
        if "escalation failed" in message:
            raise HTTPException(status_code=401, detail=message)
        else:
            raise HTTPException(status_code=500, detail=f"System nuke failed: {message}")
    
    # Refresh app_state databases to ensure the UI gets empty results
    app_state.sqlite_db = get_relational_db()
    app_state.db = UltimaRAGDatabase()
    app_state.memory = MemoryManager(app_state.db)
    app_state.brain = MetacognitiveBrain(
        app_state.db, app_state.memory,
        sqlite_db=app_state.sqlite_db
    )
    
    return {"success": True, "message": message}

# =============================================================================
# SOURCE EXPLORER API (SOTA)
# =============================================================================

@app.get("/api/workspace/files/{filename}/details")
async def get_file_details(filename: str, conversation_id: Optional[str] = None):
    """Retrieve full metadata and chunks for the Source Explorer."""
    try:
        db = get_vector_db()
        # 1. Get asset metadata
        asset = db.get_asset_by_name(filename)
            
        # 2. Get chunks (Filtered by conversation to prevent duplicate fragments from other chats)
        chunks = db.get_knowledge_chunks_by_file(filename, conversation_id=conversation_id)
        
        return {
            "success": True,
            "filename": filename,
            "metadata": asset,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "type": asset["file_type"] if asset else "unknown"
        }
    except Exception as e:
        logger.error(f"File details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discovery/pivot")
async def discovery_pivot(request: Dict):
    """Semantic Pivot Search: Find related documents based on a source file."""
    filename = request.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")
        
    try:
        db = get_vector_db()
        related = db.semantic_search_by_file_context(filename)
        return {
            "success": True,
            "source": filename,
            "related_content": related
        }
    except Exception as e:
        logger.error(f"Pivot search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/workspace/files/{filename}/export")
async def export_file_report(filename: str):
    """Generate a PDF evidence report for a specific file."""
    try:
        db = get_vector_db()
        asset = db.get_asset_by_name(filename)
        chunks = db.get_knowledge_chunks_by_file(filename)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="No content found for this file")
            
        # SOTA: Generate PDF Evidence Report
        from ..core.pdf_exporter import generate_evidence_report
        pdf_bytes = generate_evidence_report(filename, asset, chunks)
        
        headers = {
            'Content-Disposition': f'attachment; filename="Evidence_Report_{filename}.pdf"'
        }
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "success": False,
        "error": str(exc),
        "detail": "An unexpected error occurred"
    }

