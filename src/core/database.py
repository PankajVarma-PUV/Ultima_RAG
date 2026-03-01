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

import lancedb
import pyarrow as pa
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
import numpy as np
from .config import Config
from .utils import logger
import threading

# SOTA: Centralized Schema Registry to prevent Null-type inference corruption
SCHEMA_REGISTRY = {
    "projects": pa.schema([
        pa.field("id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("description", pa.string()),
        pa.field("color_code", pa.string()),
        pa.field("global_instructions", pa.string()),
        pa.field("created_at", pa.string())
    ]),
    "folders": pa.schema([
        pa.field("id", pa.string()),
        pa.field("project_id", pa.string()),
        pa.field("parent_folder_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("created_at", pa.string())
    ]),
    "conversations": pa.schema([
        pa.field("id", pa.string()),
        pa.field("project_id", pa.string()),
        pa.field("folder_id", pa.string()),
        pa.field("user_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("active_tokens", pa.int32()),
        pa.field("is_archived", pa.bool_()),
        pa.field("model_config", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("updated_at", pa.string())
    ]),
    "messages": pa.schema([
        pa.field("id", pa.string()),
        pa.field("conversation_id", pa.string()),
        pa.field("parent_id", pa.string()),
        pa.field("branch_path", pa.string()),
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), Config.embedding.DIMENSION)),
        pa.field("state", pa.string()),
        pa.field("token_count", pa.int32()),
        pa.field("duplicate_count", pa.int32()),
        pa.field("metadata", pa.string()),
        pa.field("created_at", pa.string())
    ]),
    "user_personas": pa.schema([
        pa.field("user_id", pa.string()),
        pa.field("tone_settings", pa.string()),
        pa.field("preferred_language", pa.string()),
        pa.field("instruction_profile", pa.string()),
        pa.field("updated_at", pa.string())
    ]),
    "knowledge_base": pa.schema([
        pa.field("id", pa.string()),
        pa.field("project_id", pa.string()),
        pa.field("conversation_id", pa.string()),
        pa.field("user_id", pa.string()),
        pa.field("role_id", pa.string()),
        pa.field("folder_id", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), Config.embedding.DIMENSION)),
        pa.field("metadata", pa.string()),
        pa.field("created_at", pa.string())
    ]),
    "conversation_assets": pa.schema([
        pa.field("id", pa.string()),
        pa.field("conversation_id", pa.string()),
        pa.field("file_hash", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("file_type", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("total_pages", pa.int32()),
        pa.field("duration_sec", pa.float32()),
        pa.field("summary", pa.string(), nullable=True),
        pa.field("created_at", pa.string())
    ]),
    "scraped_content": pa.schema([
        pa.field("id", pa.string()),
        pa.field("file_id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("sub_type", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("page_number", pa.int32()),
        pa.field("timestamp", pa.string()),
        pa.field("metadata", pa.string()),
        pa.field("created_at", pa.string())
    ]),
    "rag_analytics": pa.schema([
        pa.field("message_id", pa.string()),
        pa.field("score_groundedness", pa.float32()),
        pa.field("score_relevancy", pa.float32()),
        pa.field("score_utility", pa.float32()),
        pa.field("created_at", pa.string())
    ]),
    "visual_assets": pa.schema([
        pa.field("id", pa.string()),
        pa.field("project_id", pa.string()),
        pa.field("conversation_id", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 512)),
        pa.field("metadata", pa.string()),
        pa.field("created_at", pa.string())
    ]),
    "visual_cache": pa.schema([
        pa.field("id", pa.string()),
        pa.field("video_id", pa.string()),
        pa.field("variance_score", pa.float32()),
        pa.field("frame_id", pa.string()),
        pa.field("metadata", pa.string()),
        pa.field("created_at", pa.string())
    ]),
    "enriched_content": pa.schema([
        pa.field("id", pa.string()),
        pa.field("file_id", pa.string()),
        pa.field("conversation_id", pa.string()),
        pa.field("original_content", pa.string()),
        pa.field("enriched_content", pa.string()),
        pa.field("content_type", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("metadata", pa.string()),
        pa.field("processing_status", pa.string()),
        pa.field("rewriter_model", pa.string()),
        pa.field("is_deleted", pa.bool_()),
        pa.field("created_at", pa.string())
    ]),
    "document_summaries": pa.schema([
        pa.field("id", pa.string()),
        pa.field("conversation_id", pa.string()),
        pa.field("file_id", pa.string()),
        pa.field("file_name", pa.string()),
        pa.field("summary_type", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("content", pa.string()),
        pa.field("created_at", pa.string())
    ]),
    "web_search_knowledge": pa.schema([
        pa.field("id", pa.string()),
        pa.field("conversation_id", pa.string()),
        pa.field("query", pa.string()),
        pa.field("url", pa.string()),
        pa.field("title", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), Config.embedding.DIMENSION)),
        pa.field("metadata", pa.string()),
        pa.field("created_at", pa.string())
    ])
}

class UltimaRAGDatabase:
    """
    SOTA Database Layer for UltimaRAG using LanceDB.
    Supports hierarchical organization (Projects/Folders) and Vector-native storage.
    
    INTEGRITY STANDARDS:
    1. SCOPING: All retrieval methods MUST filter by `conversation_id`.
    2. DEDUPLICATION: Storage methods (add_knowledge, add_scraped, add_enriched) MUST check for existing content.
    3. ISOLATION: Cross-chat file usage is permitted via unique ID generation per conversation.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Config.paths.Ultima_DB_DIR
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = lancedb.connect(str(self.db_path))
        self._initialize_tables()
        self._perform_migrations()

    def reset_database(self):
        """Standard Factory Reset: Drops and recreates all tables."""
        logger.warning("!!! FACTORY RESET INITIATED FOR LANCEDB !!!")
        for table_name in list(self.conn.table_names()):
            try:
                self.conn.drop_table(table_name)
            except Exception as e:
                logger.error(f"Error dropping table {table_name}: {e}")
        
        self._initialize_tables()
        logger.info("LanceDB reset complete.")

    def _perform_migrations(self):
        """SOTA Auto-Migration: Detect and add missing columns to existing tables."""
        for table_name, target_schema in SCHEMA_REGISTRY.items():
            if table_name not in self.conn.table_names():
                continue
                
            table = self.conn.open_table(table_name)
            current_schema = table.schema
            
            recreated = False
            fields_to_fix = []
            for field in target_schema:
                if field.name not in current_schema.names:
                    fields_to_fix.append(field)
                else:
                    existing_type = current_schema.field(field.name).type
                    # Critical Fix: Any column typed as 'null' is corrupted and must be repaired
                    if pa.types.is_null(existing_type):
                        logger.warning(f"⚠️ Corrupted 'null' type detected in {table_name}.{field.name}. Forcing migration.")
                        fields_to_fix.append(field)
                    # ─── SOTA: Vector Dimension Mismatch Recovery ─────────────────
                    # pa.types.is_fixed_size_list is THE correct PyArrow API for fixed-
                    # size vector columns (pa.list_(pa.float32(), N)).
                    # hasattr(type, 'list_size') is fragile — avoid it.
                    elif pa.types.is_fixed_size_list(existing_type) and pa.types.is_fixed_size_list(field.type):
                        existing_dim = existing_type.list_size
                        target_dim = field.type.list_size
                        if existing_dim != target_dim:
                            logger.error(
                                f"🚨 EMBEDDING DIMENSION MISMATCH in '{table_name}.{field.name}': "
                                f"On-disk={existing_dim}D, Target={target_dim}D (from EMBEDDING_DIMENSION env). "
                                f"Auto-dropping and recreating with correct schema."
                            )
                            self.conn.drop_table(table_name)
                            self.conn.create_table(table_name, schema=target_schema)
                            logger.info(f"✅ '{table_name}' recreated with {target_dim}D vector schema.")
                            recreated = True
                            break

            if recreated:
                continue

            if not fields_to_fix:
                continue
                
            logger.info(f"⚡ Migrating table '{table_name}': Fixing/Adding {[f.name for f in fields_to_fix]}")
            
            try:
                # Use pandas for migration
                df = table.to_pandas()
                for field in fields_to_fix:
                    # SOTA: Robust Column Correction using Series-native fillna
                    if field.name not in df.columns:
                        if pa.types.is_string(field.type):
                            df[field.name] = ""
                        elif pa.types.is_int32(field.type):
                            df[field.name] = 0
                        elif pa.types.is_float32(field.type):
                            df[field.name] = 0.0
                        elif pa.types.is_boolean(field.type):
                            df[field.name] = False
                        elif pa.types.is_fixed_size_list(field.type):
                            dim = field.type.list_size
                            df[field.name] = [ [0.0] * dim ] * len(df)
                    else:
                        # Column exists but needs repair (likely 'null' type corruption)
                        if pa.types.is_string(field.type):
                            df[field.name] = df[field.name].fillna("").astype(str)
                        elif pa.types.is_int32(field.type):
                            df[field.name] = df[field.name].fillna(0).astype(int)
                        elif pa.types.is_float32(field.type):
                            df[field.name] = df[field.name].fillna(0.0).astype(float)
                        elif pa.types.is_boolean(field.type):
                            df[field.name] = df[field.name].fillna(False).astype(bool)
                        elif pa.types.is_fixed_size_list(field.type):
                            dim = field.type.list_size
                            default_vec = [0.0] * dim
                            df[field.name] = df[field.name].apply(
                                lambda x: x if (isinstance(x, (list, np.ndarray)) and len(x) == dim) else default_vec
                            )
                
                # SOTA: Overwrite the table ensuring the target_schema is EXPLICITLY provided
                # We convert to a pyarrow Table first to bypass LanceDB's internal type alignment 
                # which can fail on FixedSizeList (vector) columns.
                try:
                    pa_table = pa.Table.from_pandas(df, schema=target_schema)
                    self.conn.create_table(table_name, data=pa_table, mode="overwrite")
                    logger.info(f"✅ Table '{table_name}' migrated/repaired successfully with explicit schema.")
                except Exception as pyerr:
                    logger.error(f"❌ PyArrow Table conversion failed for '{table_name}': {pyerr}")
                    # Fallback to direct DF if conversion fails (though it likely won't if types are aligned)
                    self.conn.create_table(table_name, data=df, schema=target_schema, mode="overwrite")
                    logger.info(f"✅ Table '{table_name}' migrated via direct DataFrame fallback.")
            except Exception as e:
                logger.error(f"❌ Failed to migrate table '{table_name}': {e}")
                import traceback
                logger.error(traceback.format_exc())


    def _initialize_tables(self):
        """Initialize LanceDB tables if they don't exist using SCHEMA_REGISTRY."""
        existing_tables = self.conn.table_names()
        for table_name, schema in SCHEMA_REGISTRY.items():
            if table_name not in existing_tables:
                try:
                    self.conn.create_table(table_name, schema=schema)
                    logger.info(f"✅ Table '{table_name}' initialized successfully.")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.warning(f"⚠️ Table '{table_name}' already exists (race condition handled).")
                    else:
                        logger.error(f"❌ Failed to initialize table '{table_name}': {e}")
            
            # SOTA Phase 24: Proactive FTS Indexing
            # Ensure text columns are searchable immediately after table creation
            if table_name in ["knowledge_base", "messages", "web_search_knowledge"]:
                try:
                    table = self.conn.open_table(table_name)
                    # Use 'text' column for RAG tables, 'content' for messages
                    col = "text" if "knowledge" in table_name else "content"
                    table.create_fts_index(col, replace=False) # replace=False to avoid overhead if exists
                    logger.info(f"✅ Proactive FTS Index ensured for {table_name}.{col}")
                except Exception as fts_err:
                    logger.debug(f"FTS priming skipped for {table_name}: {fts_err}")

    # --- Project Management ---
    
    def create_project(self, name: str, description: str = "", color_code: str = "#3B82F6") -> str:
        project_id = str(uuid.uuid4())
        table = self.conn.open_table("projects")
        table.add([{
            "id": project_id,
            "name": name,
            "description": description,
            "color_code": color_code,
            "global_instructions": "",
            "created_at": datetime.utcnow().isoformat()
        }])
        return project_id

    def list_projects(self) -> List[Dict]:
        """List all workspace projects."""
        table = self.conn.open_table("projects")
        return table.search().to_list()

    # --- Folder Management ---

    def create_folder(self, name: str, project_id: str, parent_folder_id: Optional[str] = "root") -> str:
        folder_id = str(uuid.uuid4())
        table = self.conn.open_table("folders")
        table.add([{
            "id": folder_id,
            "project_id": project_id,
            "parent_folder_id": parent_folder_id,
            "name": name,
            "created_at": datetime.utcnow().isoformat()
        }])
        return folder_id

    def list_folders(self, project_id: str) -> List[Dict]:
        """List all folders within a project."""
        table = self.conn.open_table("folders")
        return table.search().where(f"project_id = '{project_id}'").to_list()

    # --- Conversation Management ---

    def create_conversation(self, name: str, project_id: Optional[str] = None, folder_id: Optional[str] = None, conversation_id: Optional[str] = None, sqlite_db: Optional[Any] = None) -> str:
        conv_id = conversation_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        # 1. Persist to LanceDB
        table = self.conn.open_table("conversations")
        table.add([{
            "id": conv_id,
            "project_id": project_id or "default",
            "folder_id": folder_id or "root",
            "user_id": "default",
            "name": name,
            "active_tokens": 0,
            "created_at": now,
            "updated_at": now
        }])

        # 2. Persist to SQLite (Unified Sync)
        if sqlite_db:
            try:
                sqlite_db.create_conversation(
                    conversation_id=conv_id,
                    title=name,
                    user_id="default"
                )
                logger.debug(f"Unified Persist: Conversation {conv_id} synced to SQLite.")
            except Exception as e:
                logger.error(f"Unified Persistence (SQLite Conversation) Error: {e}")

        return conv_id

    def list_conversations(self, folder_id: Optional[str] = None, project_id: Optional[str] = None) -> List[Dict]:
        """List conversations, optionally filtered by folder or project."""
        table = self.conn.open_table("conversations")
        query = table.search()
        if folder_id:
            query = query.where(f"folder_id = '{folder_id}'")
        elif project_id:
            query = query.where(f"project_id = '{project_id}'")
        return query.to_list()

    def list_recent_conversations(self, limit: int = 15) -> List[Dict]:
        """Fetch the most recently updated conversations across all projects."""
        table = self.conn.open_table("conversations")
        convs = table.search().to_list()
        # Sort by updated_at (or created_at if updated_at is missing)
        convs.sort(key=lambda x: x.get('updated_at') or x.get('created_at') or '', reverse=True)
        return convs[:limit]

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Fetch metadata for a single conversation."""
        table = self.conn.open_table("conversations")
        results = table.search().where(f"id = '{conversation_id}'").to_list()
        if not results:
            return None
        
        res = results[0]
        # SOTA Alignment: Ensure conversation_id key exists
        res['conversation_id'] = res['id']
        return res

    def ensure_conversation(self, conversation_id: str, title: Optional[str] = None, sqlite_db: Optional[Any] = None) -> bool:
        """Ensure a conversation exists, creating it if necessary (prevents FK/Attribute errors)."""
        if not conversation_id:
            return False
            
        conv = self.get_conversation(conversation_id)
        if not conv:
            self.create_conversation(
                name=title or "New Chat",
                project_id="default",
                folder_id="root",
                conversation_id=conversation_id,
                sqlite_db=sqlite_db
            )
        return True

    def get_documents_by_chat(self, conversation_id: str) -> List[Dict]:
        """Fetch all documents associated with a specific chat session."""
        try:
            table = self.conn.open_table("conversation_assets")
            results = table.search().where(f"conversation_id = '{conversation_id}'").to_list()
            return results
        except Exception as e:
            logger.error(f"Error getting documents by chat: {e}")
            return []

    # --- Message Management ---
    
    def add_message_unified(self, 
                            conversation_id: str, 
                            role: str, 
                            content: str, 
                            vector: Optional[List[float]] = None,
                            metadata: Dict = {},
                            sqlite_db: Optional[Any] = None) -> str:
        """
        SOTA Unified Persistence: Syncs message to both LanceDB and SQLite.
        Ensures history consistency across semantic and relational stores.
        
        Args:
            conversation_id: Unique chat identifier
            role: 'user' or 'assistant'
            content: Message text
            vector: Semantic embedding (optional, default to zeros)
            metadata: Additional context/agent details
            sqlite_db: SQLite database instance (optional)
        """
        # 1. Persist to LanceDB (Vector Store)
        # Handle vector default
        vec = vector if vector is not None else ([0.0] * Config.embedding.DIMENSION)
        msg_id = self.add_message(conversation_id, role, content, vec, metadata=metadata)
        
        # 2. Persist to SQLite (Relational Store) if available
        if sqlite_db:
            try:
                sqlite_db.add_message(conversation_id, role, content, metadata=metadata)
                logger.debug(f"Unified Persist: SQLite sync successful for {role} message.")
            except Exception as e:
                logger.error(f"Unified Persistence (SQLite) Error: {e}")
                
        return msg_id

    def add_message(self, conversation_id: str, role: str, content: str, 
                    vector: List[float], parent_id: Optional[str] = None, 
                    metadata: Dict = {}) -> str:
        msg_id = str(uuid.uuid4())
        table = self.conn.open_table("messages")
        now = datetime.utcnow().isoformat()
        table.add([{
            "id": msg_id,
            "conversation_id": conversation_id,
            "parent_id": parent_id or "root",
            "branch_path": "main",
            "role": role,
            "content": content,
            "vector": vector,
            "state": "ACTIVE",
            "token_count": 0, # Placeholder for count
            "duplicate_count": 0,
            "metadata": json.dumps(metadata, ensure_ascii=False),
            "created_at": now
        }])
        
        # SOTA: Update conversation timestamp for "Recent History" tracking
        try:
            conv_table = self.conn.open_table("conversations")
            # LanceDB 0.3.x update syntax
            conv_table.update(where=f"id = '{conversation_id}'", values={"updated_at": now})
        except Exception as e:
            logger.warning(f"Failed to update conversation timestamp: {e}")
            
        return msg_id

    def get_active_messages(self, conversation_id: str) -> List[Dict]:
        """Fetch all messages for a chat that are currently in 'ACTIVE' LLM context."""
        table = self.conn.open_table("messages")
        results = table.search().where(f"conversation_id = '{conversation_id}' AND state = 'ACTIVE'").to_list()
        return sorted(results, key=lambda x: x['created_at'])

    def get_full_history(self, conversation_id: str) -> List[Dict]:
        """Fetch ALL messages for a chat, including PAGED data."""
        table = self.conn.open_table("messages")
        results = table.search().where(f"conversation_id = '{conversation_id}'").to_list()
        return sorted(results, key=lambda x: x['created_at'])

    def page_out_messages(self, message_ids: List[str], conversation_id: str, summary: str, token_count: int = 0):
        """
        Move messages to PAGED state to free up LLM context and store the narrative summary.
        SOTA Phase 2: Infinite Chat Achives Integration.
        """
        # 1. Update State to 'PAGED' in Messages table
        table = self.conn.open_table("messages")
        for mid in message_ids:
            table.update(where=f"id = '{mid}'", values={"state": "PAGED"})

        # 2. Store the Summary generated by the Distillation Agent
        if summary and summary != "Turn archived.":
            try:
                archive_table = self.conn.open_table("conversation_archives")
                
                # Get max chunk_index for this conversation
                existing = archive_table.search().where(f"conversation_id = '{conversation_id}'").to_list()
                next_index = max([e.get('chunk_index', -1) for e in existing] + [-1]) + 1
                
                archive_table.add([{
                    "id": str(uuid.uuid4()),
                    "conversation_id": conversation_id,
                    "chunk_index": next_index,
                    "compressed_content": summary,
                    "token_count": token_count,
                    "covered_message_ids": json.dumps(message_ids),
                    "created_at": datetime.utcnow().isoformat()
                }])
                
                # Proactive FTS Indexing for rapid Recall later
                try:
                    archive_table.create_fts_index("compressed_content", replace=True)
                except Exception as e:
                    logger.debug(f"Archive FTS priming skipped: {e}")
                    
            except Exception as archive_err:
                logger.error(f"Failed to write conversation archive: {archive_err}")

    def search_messages(self, query_vector: List[float], limit: int = 5, conversation_id: Optional[str] = None) -> List[Dict]:
        """
        Perform vector search across messages, archives, and distilled knowledge.
        Provides the MemGPT 'Recall' logic for infinite chat capabilities.
        """
        all_results = []
        
        # 1. Search Active/Paged Messages
        try:
            msg_table = self.conn.open_table("messages")
            query = msg_table.search(query_vector)
            if conversation_id is not None:
                query = query.where(f"conversation_id = '{conversation_id}'")
            all_results.extend(query.limit(limit).to_list())
        except Exception as e:
            logger.warning(f"Error searching messages: {e}")
            
        # 2. Search Conversation Archives (Phase 2)
        try:
            if "conversation_archives" in self.conn.table_names():
                arch_table = self.conn.open_table("conversation_archives")
                # Archival semantic search doesn't natively support vector without
                # passing through an embedder if we didn't store vectors.
                # Assuming FTS is enabled for hybrid or just text matching.
                # If vectors aren't in archives, we might need a separate mechanism.
                # For now, we fetch recent archives.
                query = arch_table.search()
                if conversation_id is not None:
                    query = query.where(f"conversation_id = '{conversation_id}'")
                
                arch_results = query.limit(limit).to_list()
                
                for res in arch_results:
                    # Format to match message schema expected by memory manager
                    all_results.append({
                        "role": "system",
                        "content": f"[ARCHIVED MEMORY]: {res.get('compressed_content', '')}",
                        "state": "PAGED",
                        "created_at": res.get("created_at")
                    })
        except Exception as e:
            logger.warning(f"Error searching conversation_archives: {e}")
            
        # 3. Search Tier 1 Knowledge Distillation (Phase 2)
        try:
            if "knowledge_distillation" in self.conn.table_names():
                know_table = self.conn.open_table("knowledge_distillation")
                query = know_table.search()
                if conversation_id is not None:
                    query = query.where(f"conversation_id = '{conversation_id}'")
                    
                know_results = query.limit(limit).to_list()
                
                for res in know_results:
                    all_results.append({
                        "role": "system",
                        "content": f"[EXTRACTED FACT]: {res.get('extracted_fact', '')}",
                        "state": "PAGED",
                        "created_at": res.get("created_at")
                    })
        except Exception as e:
            logger.warning(f"Error searching knowledge_distillation: {e}")
            
        # Sort combined results by relevance/distance if possible, or created_at
        # For simplicity, sorting by created_at desc
        all_results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
        return all_results[:limit]

    def get_message_count(self, conversation_id: str) -> int:
        """Fetch the total count of messages in a conversation."""
        try:
            table = self.conn.open_table("messages")
            return len(table.search().where(f"conversation_id = '{conversation_id}'").to_list())
        except Exception as e:
            logger.error(f"Error getting message count: {e}")
            return 0

    # --- Knowledge Base Management (RAG) ---
    # Note: search_knowledge() is defined below (line ~644) with full conversation_id + file_names support.

    def add_knowledge_chunk(self, project_id: str, conversation_id: str, file_name: str, 
                           text: str, vector: List[float], metadata: Dict = {}):
        """Add a single pre-vectorized chunk to the knowledge base."""
        table = self.conn.open_table("knowledge_base")
        table.add([{
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "conversation_id": conversation_id,
            "folder_id": "root",
            "file_name": file_name,
            "text": text,
            "vector": vector,
            "metadata": json.dumps(metadata, ensure_ascii=False),
            "created_at": datetime.utcnow().isoformat()
        }])

    def add_knowledge_from_text(self, text: str, file_name: str, conversation_id: str, 
                               project_id: str = "default", metadata: Dict = {}):
        """
        SOTA: Automated indexing of arbitrary text.
        Handles chunking (500/100), embedding generation, and storage.
        Used for indexing enriched multimodal narratives.
        """
        from ..data.chunking import DocumentChunker
        from ..core.config import ChunkingConfig
        from ..data.embedder import get_embedder

        # 0. SOTA Deduplication Guard: Check if this file+chat combination is already indexed
        try:
            table = self.conn.open_table("knowledge_base")
            existing = table.search().where(f"file_name = '{file_name}' AND conversation_id = '{conversation_id}'").limit(1).to_list()
            if existing:
                logger.debug(f"Knowledge fragments already exist for {file_name} in chat {conversation_id}. Skipping redundant indexing.")
                return
        except Exception as e:
            logger.warning(f"Deduplication check failed (non-fatal): {e}")

        # 1. Chunking
        chunker = DocumentChunker(
            strategy="semantic",
            chunk_size=ChunkingConfig.CHUNK_SIZE,
            overlap=ChunkingConfig.CHUNK_OVERLAP
        )
        doc_id = f"text_{uuid.uuid4().hex[:8]}"
        meta = {**metadata, "source": file_name, "file_name": file_name}
        chunks = chunker.chunk_document(text, doc_id=doc_id, metadata=meta)
        
        if not chunks:
            return
            
        # 2. Embedding
        embedder = get_embedder()
        texts = [c['text'] for c in chunks]
        embeddings = embedder.encode(texts)
        
        # 3. Storage
        table = self.conn.open_table("knowledge_base")
        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            records.append({
                "id": chunk['chunk_id'],
                "project_id": project_id,
                "conversation_id": conversation_id,
                "folder_id": "root",
                "file_name": file_name,
                "text": chunk['text'],
                "vector": vector.tolist(),
                "metadata": json.dumps(chunk['metadata'], ensure_ascii=False),
                "created_at": datetime.utcnow().isoformat()
            })
        
        table.add(records)
        logger.info(f"Indexed {len(records)} knowledge chunks for {file_name}")
        
        # SOTA HYBRID SEARCH: Build Full-Text Search Index
        try:
            table.create_fts_index("text", replace=True)
            logger.info(f"Updated LanceDB FTS (BM25) index for {file_name}")
        except Exception as e:
            logger.warning(f"Could not build FTS index (ensure tantivy is installed): {e}")

    def delete_last_message(self, conversation_id: str, role: str, sqlite_db: Optional[Any] = None) -> bool:
        """Delete the most recent message with specified role in a conversation (Unified)."""
        deleted = False
        try:
            # 1. Delete from LanceDB
            table = self.conn.open_table("messages")
            # We need to find the ID first because LanceDB delete where is tricky with ORDER BY/LIMIT
            results = table.search().where(f"conversation_id = '{conversation_id}' AND role = '{role}'").to_list()
            if results:
                # Sort by created_at desc manually
                results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                target_id = results[0]['id']
                table.delete(where=f"id = '{target_id}'")
                deleted = True
                logger.info(f"LanceDB: Deleted last {role} message {target_id} in {conversation_id}")
            
            # 2. Delete from SQLite if provided
            if sqlite_db:
                sql_deleted = sqlite_db.delete_last_message(conversation_id, role)
                deleted = deleted or sql_deleted
                if sql_deleted:
                    logger.info(f"SQLite: Deleted last {role} message in {conversation_id}")
                    
        except Exception as e:
            logger.error(f"Error deleting last message: {e}")
            
        return deleted

    def add_knowledge(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the unified knowledge base."""
        table = self.conn.open_table("knowledge_base")
        processed_chunks = []
        for chunk in chunks:
            processed_chunks.append({
                "id": str(uuid.uuid4()),
                "project_id": chunk.get("project_id", "default"),
                "conversation_id": chunk.get("conversation_id", "default"),
                "user_id": chunk.get("user_id", "default"),
                "role_id": chunk.get("role_id", "user"),
                "folder_id": chunk.get("folder_id", "root"),
                "file_name": chunk.get("file_name", "unknown"),
                "text": chunk.get("text", ""),
                "vector": chunk.get("vector"),
                "metadata": json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                "created_at": datetime.utcnow().isoformat()
            })
        table.add(processed_chunks)
        
        # SOTA HYBRID SEARCH: Build Full-Text Search Index
        try:
            table.create_fts_index("text", replace=True)
            logger.info("Updated LanceDB FTS (BM25) index via add_knowledge")
        except Exception as e:
            logger.warning(f"Could not build FTS index (ensure tantivy is installed): {e}")

    def get_knowledge_count(self, conversation_id: Optional[str] = None) -> int:
        """Fetch the total count of document chunks in the knowledge base, optionally filtered by conversation.
           Includes both text chunks and media assets (images).
        """
        try:
            total_count = 0
            
            # 1. Count text chunks from knowledge_base
            kb_table = self.conn.open_table("knowledge_base")
            if conversation_id is not None:
                total_count += len(kb_table.search().where(f"conversation_id = '{conversation_id}'").to_list())
            else:
                total_count += kb_table.count_rows()
                
            # 2. Count media items (e.g. images without text chunks) from conversation_assets
            # We assume non-text parsed assets (like plain images) contribute as 1 "chunk" conceptually.
            import pyarrow as pa
            if "conversation_assets" in self.conn.table_names():
                assets_table = self.conn.open_table("conversation_assets")
                if conversation_id is not None:
                    # Only add assets that are purely visual (images) 
                    # If they were PDFs, they'd have chunks in knowledge_base, but we can just count all assets 
                    # to be safe, or specifically filter if needed. Let's count media types.
                    assets = assets_table.search().where(f"conversation_id = '{conversation_id}'").to_list()
                    image_assets = [a for a in assets if a.get('file_type', '').startswith('image/')]
                    total_count += len(image_assets)
                else:
                    assets_df = assets_table.to_pandas()
                    total_count += len(assets_df[assets_df['file_type'].str.startswith('image/', na=False)])
                    
            return total_count
        except Exception as e:
            logger.error(f"Error getting knowledge count: {e}")
            return 0

    def search_knowledge(self, query_vector: List[float], query_text: str, project_id: Optional[str] = None, 
                         conversation_id: Optional[str] = None, file_names: Optional[List[str]] = None, 
                         user_id: str = "default", limit: int = 5) -> List[Dict]:
        """Search the knowledge base using Hybrid SOTA (Vector + BM25), enforcing vector-level RBAC isolation."""
        table = self.conn.open_table("knowledge_base")
        
        # SOTA: Execute Native Hybrid Search (Guarded against empty text)
        if not query_text or not query_text.strip():
            query = table.search(query_vector)
        else:
            try:
                # LanceDB natively fuses Vector and Text via Reciprocal Rank Fusion
                query = table.search(query_type="hybrid").vector(query_vector).text(query_text)
            except Exception as e:
                # Safe Fallback to Vector-only if FTS index doesn't exist yet
                logger.warning(f"Hybrid search failed, falling back to Vector-only: {e}")
                query = table.search(query_vector)
        
        where_clauses = []
        
        # SOTA Phase 1 (RBAC): Hard-enforce user isolation. 
        # Admins can bypass, but typical users only see their own vectors.
        where_clauses.append(f"user_id = '{user_id}'")
        
        if project_id:
            where_clauses.append(f"project_id = '{project_id}'")
            
        if conversation_id is not None:
            where_clauses.append(f"conversation_id = '{conversation_id}'")
            
        if file_names:
            # SOTA Phase 22: Fuzzy Filename Matching
            # Construct a clause that matches exact names OR base names (no extension)
            sub_clauses = []
            for f in file_names:
                f_clean = f.strip().lower()
                sub_clauses.append(f"LOWER(file_name) = '{f_clean}'")
                sub_clauses.append(f"LOWER(file_name) LIKE '{f_clean}.%'")
            
            where_clauses.append(f"({' OR '.join(sub_clauses)})")
            
        if where_clauses:
            query = query.where(" AND ".join(where_clauses))
            
        try:
            return query.limit(limit).to_list()
        except Exception as e:
            # SOTA CRITICAL FIX: Execution-time FTS/BM25 failure detection.
            # If the hybrid query fails during execution (e.g. index corruption),
            # we retry with a pure vector search.
            if "inverted index" in str(e).lower() or "fts" in str(e).lower():
                logger.warning(f"LanceDB FTS execution failed. Falling back to pure vector: {e}")
                # Re-construct pure vector query
                retry_query = table.search(query_vector)
                if where_clauses:
                    retry_query = retry_query.where(" AND ".join(where_clauses))
                return retry_query.limit(limit).to_list()
            raise e

    # --- Asset Management ---

    def register_document(self, file_name: str, file_hash: str, file_type: str, conversation_id: str, file_path: Optional[str] = None) -> str:
        """SOTA Document Registration (ID-Stable). 
        Checks if hash exists in this conversation before creating new ID.
        """
        # 1. Check for existing entry to ensure ID stability
        # SOTA: Always scope hash checks to conversation to allow same-file cross-chat isolation
        existing = self.get_document_by_hash(file_hash, conversation_id=conversation_id)
        if existing:
            logger.info(f"ID-Stable hit: Returning existing ID {existing['id']} for {file_name} in chat {conversation_id}")
            return existing['id']

        # 2. Register fresh if not found
        asset_id = str(uuid.uuid4())
        table = self.conn.open_table("conversation_assets")
        table.add([{
            "id": asset_id,
            "conversation_id": conversation_id,
            "file_hash": file_hash,
            "file_path": file_path or "",
            "file_type": file_type,
            "file_name": file_name,
            "created_at": datetime.utcnow().isoformat()
        }])
        return asset_id

    def add_asset(self, conversation_id: str, file_path: str, file_type: str, file_name: str) -> str:
        """Legacy alias with no hash support."""
        return self.register_document(file_name, "legacy_no_hash", file_type, conversation_id, file_path)

    def get_assets(self, conversation_id: str) -> List[Dict]:
        table = self.conn.open_table("conversation_assets")
        return table.search().where(f"conversation_id = '{conversation_id}'").to_list()

    def get_document_by_hash(self, file_hash: str, conversation_id: Optional[str] = None) -> Optional[Dict]:
        """Search across chat assets for a file hash, optionally scoped to a conversation."""
        table = self.conn.open_table("conversation_assets")
        
        where_clause = f"file_hash = '{file_hash}'"
        if conversation_id is not None:
            where_clause += f" AND conversation_id = '{conversation_id}'"
            
        results = table.search().where(where_clause).limit(1).to_list()
        
        if results:
            res = results[0]
            # Ensure file_id is present for compatibility
            if 'file_id' not in res:
                res['file_id'] = res['id']
            return res
        return None

    def update_asset_summary(self, asset_id: str, summary_text: str) -> bool:
        """SOTA Update: Store the master summary directly alongside the file asset for quick retrieval."""
        try:
            table = self.conn.open_table("conversation_assets")
            # LanceDB update syntax requires the target ID and the dict of updates
            table.update(where=f"id = '{asset_id}'", values={"summary": summary_text})
            logger.info(f"Successfully saved master summary to conversation_assets for asset_id={asset_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to update asset summary for {asset_id}: {e}")
            return False

    def delete_document_by_hash(self, file_hash: str) -> bool:
        """Purge document registration and associated scraped content (Cascade Simulation)."""
        try:
            # 1. Find the asset_id
            asset = self.get_document_by_hash(file_hash)
            if not asset:
                return False
            
            asset_id = asset['id']
            
            # 2. Delete from conversation_assets
            assets_table = self.conn.open_table("conversation_assets")
            assets_table.delete(f"id = '{asset_id}'")
            
            # 3. Delete from scraped_content (LanceDB doesn't have FK cascade, so we do it manually)
            scraped_table = self.conn.open_table("scraped_content")
            scraped_table.delete(f"file_id = '{asset_id}'")
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete document by hash {file_hash}: {e}")
            return False

    # --- Scraped Content (OCR/Vision Perception results) ---

    def add_scraped_content(self, file_id: str, content: str, sub_type: str = 'text', 
                           chunk_index: Optional[int] = None, page_number: Optional[int] = None, 
                           timestamp: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Persist vision/audio perception results to LanceDB."""
        # SOTA Deduplication Guard: Check if similar content already exists for this file
        try:
            table = self.conn.open_table("scraped_content")
            # We check for same file_id and sub_type to avoid redundant entries for the same perception task
            # For video, we might have multiple clips, so we check chunk_index too
            where_clause = f"file_id = '{file_id}' AND sub_type = '{sub_type}'"
            if chunk_index is not None:
                where_clause += f" AND chunk_index = {chunk_index}"
            if timestamp:
                where_clause += f" AND timestamp = '{timestamp}'"
                
            existing = table.search().where(where_clause).limit(1).to_list()
            if existing:
                logger.debug(f"Scraped content already exists for file {file_id} (type: {sub_type}). Skipping.")
                return existing[0]['id']
        except Exception as e:
            logger.warning(f"Scraped deduplication check failed (non-fatal): {e}")

        content_id = str(uuid.uuid4())
        table = self.conn.open_table("scraped_content")
        
        table.add([{
            "id": content_id,
            "file_id": file_id,
            "content": content,
            "sub_type": sub_type,
            "chunk_index": chunk_index if chunk_index is not None else 0,
            "page_number": page_number if page_number is not None else 0,
            "timestamp": timestamp or "",
            "metadata": json.dumps(metadata or {}, ensure_ascii=False),
            "created_at": datetime.utcnow().isoformat()
        }])
        return content_id

    def get_scraped_content(self, file_id: str) -> List[Dict]:
        """Fetch all perception items for a specific file."""
        table = self.conn.open_table("scraped_content")
        items = table.search().where(f"file_id = '{file_id}'").to_list()
        # Parse metadata
        for item in items:
            it_meta = item.get('metadata', '{}')
            try:
                item['metadata'] = json.loads(it_meta) if isinstance(it_meta, str) else it_meta
            except:
                item['metadata'] = {}
        return items

    def get_scraped_content_by_chat(self, conversation_id: str) -> List[Dict]:
        """
        SOTA: Join-less collection of evidence for a conversation with Name Attribution.
        Ensures every chunk has its source file_name attached.
        """
        # 1. Get all assets for this chat to build a name map
        assets = self.get_assets(conversation_id)
        asset_map = {a['id']: a for a in assets}
        
        if not asset_map:
            return []
            
        # 2. Collect all scraped items referencing those assets
        scraped_table = self.conn.open_table("scraped_content")
        all_content = []
        
        for aid, asset in asset_map.items():
            items = self.get_scraped_content(aid)
            # Inject file name and other metadata for precise RAG attribution
            for item in items:
                item['file_name'] = asset.get('file_name', 'Unknown')
                item['file_type'] = asset.get('file_type', 'file')
                all_content.append(item)
            
        return all_content

    # --- Enriched Content (Unified Source of Truth) ---

    def add_enriched_content(self, file_id: str, conversation_id: str, 
                             original_content: str, enriched_content: str, 
                             content_type: str, file_name: str, 
                             metadata: Optional[Dict] = None,
                             rewriter_model: str = Config.ollama_multi_model.HEAVY_MODEL) -> str:
        """Add unified enriched content for a file in LanceDB."""
        # SOTA: Defensive check for existing enriched content for this file_id
        # Prevents duplicate entries from concurrent background tasks
        existing = self.get_enriched_content_by_file_id(file_id)
        if existing:
            logger.info(f"Enriched content already exists for {file_id} in chat {conversation_id}. Returning existing ID.")
            return existing['id']
            
        content_id = str(uuid.uuid4())
        table = self.conn.open_table("enriched_content")
        
        table.add([{
            "id": content_id,
            "file_id": file_id,
            "conversation_id": conversation_id,
            "original_content": original_content,
            "enriched_content": enriched_content,
            "content_type": content_type,
            "file_name": file_name,
            "metadata": json.dumps(metadata or {}, ensure_ascii=False),
            "processing_status": "completed",
            "rewriter_model": rewriter_model,
            "is_deleted": False,
            "created_at": datetime.utcnow().isoformat()
        }])
        return content_id

    def get_enriched_content_by_file_id(self, file_id: str) -> Optional[Dict]:
        """Check if enriched content exists for a file in LanceDB."""
        table = self.conn.open_table("enriched_content")
        results = table.search().where(f"file_id = '{file_id}' AND is_deleted = false").to_list()
        if results:
            res = results[0]
            # Parse metadata
            meta = res.get('metadata', '{}')
            try:
                res['metadata_json'] = meta # For SQLite compatibility in some paths
                res['metadata'] = json.loads(meta) if isinstance(meta, str) else meta
            except:
                res['metadata'] = {}
            return res
        return None

    def get_enriched_content_by_hash(self, file_hash: str) -> Optional[Dict]:
        """
        SOTA: Global Enrichment Cache Search.
        Finds the first completed enrichment for a file hash across any conversation.
        """
        # 1. Find ANY asset with this hash
        asset = self.get_document_by_hash(file_hash)
        if not asset:
            return None
            
        # 2. Check Enriched table for this file_id OR this file_hash if we added it (currently use file_id)
        # Note: enrichment is linked to file_id, which is unique per conversation registration.
        # But we can find ANY enriched content that used this global file hash.
        table = self.conn.open_table("enriched_content")
        # We search by checking any enriched rows where the underlying asset record has our target hash.
        # Since LanceDB doesn't do deep joins, we use our document registry as the index.
        
        # SOTA FIX: Search enriched content table by file_name as a proxy or just iterate recent registrations of the hash.
        # Efficient approach: Find ALL file_ids for this hash, then find first enriched for any.
        assets_table = self.conn.open_table("conversation_assets")
        matching_assets = assets_table.search().where(f"file_hash = '{file_hash}'").to_list()
        
        for asset in matching_assets:
            enriched = self.get_enriched_content_by_file_id(asset['id'])
            if enriched:
                return enriched
        
        return None

    def get_enriched_content_by_chat(self, conversation_id: str) -> List[Dict]:
        """Fetch all enriched content for a specific conversation in LanceDB."""
        table = self.conn.open_table("enriched_content")
        items = table.search().where(f"conversation_id = '{conversation_id}' AND is_deleted = false").to_list()
        for item in items:
            meta = item.get('metadata', '{}')
            try:
                item['metadata_json'] = meta
                item['metadata'] = json.loads(meta) if isinstance(meta, str) else meta
            except:
                item['metadata'] = {}
        return items

    def get_enriched_content_by_filenames(self, conversation_id: str, file_names: List[str]) -> List[Dict]:
        """Fetch enriched content for specific files (for @mentions) in LanceDB."""
        if not file_names:
            return []
            
        table = self.conn.open_table("enriched_content")
        # Build IN clause manually for LanceDB
        items = table.search().where(f"conversation_id = '{conversation_id}' AND is_deleted = false").to_list()
        
        # SOTA: Loose Matching (e.g. @cat matches cat.jpg)
        def is_match(fname, targets):
            if not fname: return False
            fname_lower = fname.lower()
            for t in targets:
                t_lower = t.lower()
                # 1. Exact match
                if fname_lower == t_lower: return True
                # 2. Base name match (cat.jpg matches cat)
                if fname_lower.startswith(t_lower + '.'): return True
                # 3. Target is full name (cat.jpg matches cat.jpg) - covered by #1
            return False

        # Filter by filenames using the loose matcher
        results = [i for i in items if is_match(i.get('file_name'), file_names)]
        
        for item in results:
            meta = item.get('metadata', '{}')
            try:
                item['metadata_json'] = meta
                item['metadata'] = json.loads(meta) if isinstance(meta, str) else meta
            except:
                item['metadata'] = {}
        return results

    # --- Analytics Management ---

    def add_analytics(self, message_id: str, groundedness: float, relevancy: float, utility: float):
        table = self.conn.open_table("rag_analytics")
        table.add([{
            "message_id": message_id,
            "score_groundedness": groundedness,
            "score_relevancy": relevancy,
            "score_utility": utility,
            "created_at": datetime.utcnow().isoformat()
        }])

    def get_analytics(self, message_id: str) -> Optional[Dict]:
        table = self.conn.open_table("rag_analytics")
        results = table.search().where(f"message_id = '{message_id}'").to_list()
        return results[0] if results else None

    # --- Document Summaries (Map-Reduce) ---
    def add_document_summary(self, conversation_id: str, file_id: str, file_name: str, summary_type: str, content: str, chunk_index: int = -1):
        try:
            table = self.conn.open_table("document_summaries")
            table.add([{
                "id": str(uuid.uuid4()),
                "conversation_id": conversation_id,
                "file_id": file_id,
                "file_name": file_name,
                "summary_type": summary_type,
                "chunk_index": chunk_index,
                "content": content,
                "created_at": datetime.utcnow().isoformat()
            }])
        except Exception as e:
            logger.error(f"Error adding document summary: {e}")

    def get_document_summaries(self, conversation_id: str, summary_type: str = 'combined_summary') -> List[Dict]:
        """Fetch all summaries of a specific type for a conversation."""
        try:
            table = self.conn.open_table("document_summaries")
            return table.search().where(f"conversation_id = '{conversation_id}' AND summary_type = '{summary_type}'").to_list()
        except Exception as e:
            logger.error(f"Error getting document summaries: {e}")
            return []

    def get_unique_files_for_conversation(self, conversation_id: str) -> List[str]:
        """Return a distinct list of file_names indexed in knowledge_base for a given conversation."""
        try:
            table = self.conn.open_table("knowledge_base")
            rows = table.search().where(f"conversation_id = '{conversation_id}'").select(["file_name"]).to_list()
            seen = set()
            result = []
            for r in rows:
                fn = r.get("file_name", "")
                if fn and fn not in seen:
                    seen.add(fn)
                    result.append(fn)
            return result
        except Exception as e:
            logger.error(f"Error getting unique files for conversation: {e}")
            return []

    # --- Visual Cache Management ---


    def add_visual_cache(self, video_id: str, variance_score: float, frame_id: str, metadata: Dict = {}):
        table = self.conn.open_table("visual_cache")
        table.add([{
            "id": str(uuid.uuid4()),
            "video_id": video_id,
            "variance_score": variance_score,
            "frame_id": frame_id,
            "metadata": json.dumps(metadata, ensure_ascii=False),
            "created_at": datetime.utcnow().isoformat()
        }])

    def get_visual_cache(self, video_id: str) -> List[Dict]:
        table = self.conn.open_table("visual_cache")
        return table.search().where(f"video_id = '{video_id}'").to_list()
    # --- Conversation Meta Helpers (Consolidated) ---
    def add_web_search_result(self, conversation_id: str, query: str, results: List[Dict]):
        """
        SOTA: Persist web search results to a dedicated table.
        Chunks and embeds the scraped content before storing.
        """
        from ..data.chunking import DocumentChunker
        from ..data.embedder import get_embedder
        
        table = self.conn.open_table("web_search_knowledge")
        embedder = get_embedder()
        chunker = DocumentChunker(chunk_size=512, overlap=50) # Tighter chunks for web data
        
        records = []
        for res in results:
            url = res.get("url", "")
            title = res.get("title", "Untitled")
            raw_text = res.get("text", "")
            
            if not raw_text:
                continue
                
            # Chunk the result
            chunks = chunker.chunk_document(raw_text, doc_id=f"web_{uuid.uuid4().hex[:8]}")
            if not chunks:
                continue
                
            # Embed all chunks
            texts = [c['text'] for c in chunks]
            embeddings = embedder.encode(texts)
            
            for chunk, vec in zip(chunks, embeddings):
                records.append({
                    "id": str(uuid.uuid4()),
                    "conversation_id": conversation_id,
                    "query": query,
                    "url": url,
                    "title": title,
                    "text": chunk['text'],
                    "vector": vec.tolist(),
                    "metadata": json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                    "created_at": datetime.utcnow().isoformat()
                })
        
        if records:
            table.add(records)
            logger.info(f"Persisted {len(records)} web chunks for query: '{query}'")
            
            # SOTA Phase 25: Build FTS Index for web results to enable hybrid search
            try:
                table.create_fts_index("text", replace=True)
                logger.info("Updated LanceDB FTS (BM25) index for web_search_knowledge")
            except Exception as e:
                logger.warning(f"Could not build web FTS index: {e}")

    def search_web_knowledge(self, query_vector: List[float], conversation_id: str, limit: int = 5) -> List[Dict]:
        """Search specifically within paged web content for this chat."""
        try:
            table = self.conn.open_table("web_search_knowledge")
            return table.search(query_vector).where(f"conversation_id = '{conversation_id}'").limit(limit).to_list()
        except:
            return []

    def update_conversation(self, conversation_id: str, title: Optional[str] = None, is_archived: Optional[bool] = None) -> bool:
        """Update a conversation's metadata (Rename or Archive)."""
        try:
            table = self.conn.open_table("conversations")
            values = {"updated_at": datetime.utcnow().isoformat()}
            if title:
                values["name"] = title
            if is_archived is not None:
                values["is_archived"] = is_archived
            
            table.update(where=f"id = '{conversation_id}'", values=values)
            return True
        except Exception as e:
            logger.error(f"Error updating conversation: {e}")
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        try:
            # 1. Delete messages
            msg_table = self.conn.open_table("messages")
            msg_table.delete(where=f"conversation_id = '{conversation_id}'")
            
            # 2. Delete assets
            asset_table = self.conn.open_table("conversation_assets")
            asset_table.delete(where=f"conversation_id = '{conversation_id}'")
            
            # 3. Delete conversation record
            conv_table = self.conn.open_table("conversations")
            conv_table.delete(where=f"id = '{conversation_id}'")
            
            # 4. Optional: Delete enriched content
            try:
                enriched_table = self.conn.open_table("enriched_content")
                enriched_table.delete(where=f"conversation_id = '{conversation_id}'")
            except: pass

            # 5. SOTA: Wipe Map-Reduce Summaries & Web Knowledge Cache to prevent storage leak
            try:
                sum_table = self.conn.open_table("document_summaries")
                sum_table.delete(where=f"conversation_id = '{conversation_id}'")
            except: pass

            try:
                web_table = self.conn.open_table("web_search_knowledge")
                web_table.delete(where=f"conversation_id = '{conversation_id}'")
            except: pass

            # 6. SOTA Phase 26: Purge Knowledge Base (the missing leak)
            try:
                kb_table = self.conn.open_table("knowledge_base")
                kb_table.delete(where=f"conversation_id = '{conversation_id}'")
                logger.info(f"Purged knowledge_base chunks for conversion: {conversation_id}")
            except Exception as e:
                logger.debug(f"Knowledge purge skipped/failed: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False

    # --- Workspace Tree Logic ---
    def get_workspace_tree(self) -> Dict:
        """Returns a nested JSON of Projects -> Folders -> Chats."""
        projects = self.list_projects()
        tree = []
        
        for p in projects:
            p_id = p['id']
            folders = self.list_folders(p_id)
            p_node = {
                "id": p_id,
                "name": p['name'],
                "type": "project",
                "color": p.get('color_code', '#3B82F6'),
                "children": []
            }
            
            # Add folders and their conversations
            for f in folders:
                f_id = f['id']
                conversations = self.list_conversations(folder_id=f_id)
                f_node = {
                    "id": f_id,
                    "name": f['name'],
                    "type": "folder",
                    "children": [{
                        "id": c['id'],
                        "name": c['name'],
                        "type": "conversation"
                    } for c in conversations]
                }
                p_node["children"].append(f_node)
            
            # Add conversations directly under project (root folder)
            root_convs = self.list_conversations(project_id=p_id, folder_id="root")
            for c in root_convs:
                p_node["children"].append({
                    "id": c['id'],
                    "name": c['name'],
                    "type": "conversation"
                })
                
            tree.append(p_node)
            
        return {"projects": tree}

    def get_knowledge_chunks_by_file(
        self,
        file_name: str,
        conversation_id: Optional[str] = None,
        limit: int = 500
    ) -> List[Dict]:
        """
        CANONICAL (merged) method — replaces all previous duplicate definitions.
        Returns knowledge_base text chunks for a specific file.
        - Uses loose filename matching (exact OR with any extension) for Source Explorer.
        - Optionally scoped to a conversation to prevent cross-chat bleed.
        - Limit defaults to 500 for summarization, can be set lower for UI preview.
        """
        try:
            table = self.conn.open_table("knowledge_base")
            f_clean = file_name.lower().strip().replace("'", "''")
            where_clause = f"(LOWER(file_name) = '{f_clean}' OR LOWER(file_name) LIKE '{f_clean}.%' OR LOWER(file_name) = '{file_name.replace(chr(39), chr(39)+chr(39))}')"

            if conversation_id:
                where_clause += f" AND conversation_id = '{conversation_id}'"

            results = (
                table.search()
                .where(where_clause)
                .select(["id", "text", "file_name", "metadata"])
                .limit(limit)
                .to_list()
            )

            # Parse metadata JSON strings for callers that need structured metadata
            for res in results:
                if 'metadata' in res and isinstance(res['metadata'], str):
                    try:
                        res['metadata'] = json.loads(res['metadata'])
                    except Exception:
                        pass
            return results
        except Exception as e:
            logger.error(f"Error getting knowledge chunks for {file_name}: {e}")
            return []


    def semantic_search_by_file_context(self, file_name: str, limit: int = 10) -> List[Dict]:
        """
        Pivot Search: Find content related to a specific file.
        Uses the first chunk of the file as a query vector for global search.
        """
        try:
            table = self.conn.open_table("knowledge_base")
            f_clean = file_name.lower().strip()
            
            # 1. Get a representative vector for this file
            file_chunks = table.search().where(f"LOWER(file_name) = '{f_clean}' OR LOWER(file_name) LIKE '{f_clean}.%'").limit(1).to_list()
            if not file_chunks:
                return []
                
            query_vector = file_chunks[0]['vector']
            
            # 2. Search for related content (excluding the file itself)
            results = table.search(query_vector).where(f"LOWER(file_name) != '{f_clean}' AND LOWER(file_name) NOT LIKE '{f_clean}.%'").limit(limit).to_list()
            
            # Parse metadata
            for res in results:
                if 'metadata' in res and isinstance(res['metadata'], str):
                    try:
                        res['metadata'] = json.loads(res['metadata'])
                    except:
                        pass
            return results
        except Exception as e:
            logger.error(f"Error in pivot search: {e}")
            return []

    def get_asset_by_name(self, file_name: str, conversation_id: Optional[str] = None) -> Optional[Dict]:
        """Fetch asset metadata by its name, optionally scoped to a conversation."""
        try:
            table = self.conn.open_table("conversation_assets")
            f_clean = file_name.lower().strip()
            where_clause = f"LOWER(file_name) = '{f_clean}'"
            if conversation_id:
                where_clause += f" AND conversation_id = '{conversation_id}'"
            results = table.search().where(where_clause).to_list()
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error getting asset by name: {e}")
            return None

# --- Global Instance Management ---

_db_instance = None
_db_lock = threading.Lock()

def get_database() -> UltimaRAGDatabase:
    """Get or create singleton UltimaRAGDatabase instance with thread safety."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            # Double-check pattern
            if _db_instance is None:
                _db_instance = UltimaRAGDatabase()
    return _db_instance

