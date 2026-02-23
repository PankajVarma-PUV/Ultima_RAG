"""
Database Module for Ultima_RAG
Supports both SQLite (local, zero-config) and PostgreSQL (cloud).
Database type is selected via DB_TYPE environment variable.
"""

import os
import uuid
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

# Load environment variables from Credentials folder
from dotenv import load_dotenv
CREDENTIALS_PATH = Path(__file__).parent.parent.parent / "Credentials" / ".env"
load_dotenv(CREDENTIALS_PATH)

from ..core.utils import logger


# =============================================================================
# DATABASE ABSTRACT BASE CLASS
# =============================================================================

class BaseDatabase(ABC):
    """Abstract base class for database implementations."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish database connection."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close database connection."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if database is connected."""
        pass
    
    @abstractmethod
    def reset_database(self):
        """Drop all tables and re-initialize schema (FRESH START)."""
        pass
    
    @abstractmethod
    def initialize_schema(self):
        """Create tables if they don't exist."""
        pass
    
    @abstractmethod
    def create_conversation(self, title: Optional[str] = None, user_id: str = "default", conversation_id: Optional[str] = None) -> str:
        """Create a new conversation."""
        pass
    
    @abstractmethod
    def list_conversations(self, limit: int = 50, include_archived: bool = False, user_id: str = "default") -> List[Dict]:
        """List all conversations."""
        pass
    
    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a single conversation by ID."""
        pass
    
    @abstractmethod
    def update_conversation(self, conversation_id: str, title: Optional[str] = None, is_archived: Optional[bool] = None) -> bool:
        """Update a conversation's title or archive status."""
        pass
    
    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        pass
    
    @abstractmethod
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None, token_count: Optional[int] = None) -> str:
        """Add a message to a conversation."""
        pass
    
    @abstractmethod
    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages for a conversation."""
        pass
    
    @abstractmethod
    def get_active_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages for a conversation (alias for get_messages in relational DB)."""
        pass
    
    @abstractmethod
    def find_duplicate_query(self, conversation_id: str, query: str) -> Optional[Dict]:
        """Find an identical query in this conversation and return the assistant's previous response."""
        pass
    
    @abstractmethod
    def get_message_count(self, conversation_id: str) -> int:
        """Get the number of messages in a conversation."""
        pass

    @abstractmethod
    def ensure_conversation(self, conversation_id: str, title: Optional[str] = None) -> bool:
        """Ensure a conversation exists, creating it if necessary."""
        pass


# =============================================================================
# SQLITE DATABASE IMPLEMENTATION
# =============================================================================

class SQLiteDatabase(BaseDatabase):
    """
    SQLite database implementation.
    Zero configuration - uses a local file.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite database.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/Ultima_RAG.db
        """
        default_path = Path(__file__).parent.parent.parent / "data" / "Ultima_RAG.db"
        self.db_path = Path(db_path) if db_path else default_path
        self._connection: Optional[sqlite3.Connection] = None
        self._connected = False
    
    def connect(self) -> bool:
        """Establish connection to SQLite database."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect with row factory for dict-like access
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,  # Allow multi-thread access (FastAPI)
                timeout=30.0
            )
            self._connection.row_factory = sqlite3.Row
            
            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys = ON")
            
            self._connected = True
            logger.info(f"SQLite database connected: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Close SQLite connection."""
        if self._connection:
            self._connection.close()
            self._connected = False
            logger.info("SQLite connection closed")
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self._connection is not None
    
    @contextmanager
    def get_cursor(self):
        """Get a cursor with automatic commit/rollback."""
        if not self._connection:
            raise RuntimeError("Database not connected")
        
        cursor = self._connection.cursor()
        try:
            yield cursor
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        finally:
            cursor.close()
    
    def initialize_schema(self):
        """Create tables if they don't exist (ChatGPT-style schema)."""
        if not self.is_connected():
            logger.warning("Cannot initialize schema - not connected")
            return
        
        schema_sql = """
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            user_id              TEXT PRIMARY KEY,
            user_created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            settings_json        TEXT
        );
        
        -- Insert default user
        INSERT OR IGNORE INTO users (user_id) VALUES ('default');
        
        -- Conversations table
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id           TEXT PRIMARY KEY,
            user_id                   TEXT DEFAULT 'default',
            title                     TEXT,
            conversation_created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            conversation_updated_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            is_archived               INTEGER DEFAULT 0,
            model_config              TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_conversations_user 
            ON conversations(user_id, conversation_updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated 
            ON conversations(conversation_updated_at DESC);
        
        -- Messages table
        CREATE TABLE IF NOT EXISTS messages (
            message_id              TEXT PRIMARY KEY,
            conversation_id         TEXT NOT NULL,
            parent_message_id       TEXT,
            role                    TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content                 TEXT NOT NULL,
            metadata_json           TEXT,
            token_count             INTEGER,
            duplicate_count         INTEGER DEFAULT 0,
            message_created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            FOREIGN KEY (parent_message_id) REFERENCES messages(message_id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages(conversation_id, message_created_at);
        """
        
        try:
            with self.get_cursor() as cursor:
                cursor.executescript(schema_sql)
            logger.info("SQLite schema initialized (ChatGPT-style)")
            # Run migrations
            self._run_migrations()
        except Exception as e:
            logger.error(f"Failed to initialize SQLite schema: {e}")
            raise

    def _run_migrations(self):
        """Run idempotent schema migrations."""
        pass

    def create_conversation(self, title: Optional[str] = None, user_id: str = "default", conversation_id: Optional[str] = None) -> str:
        """Create a new conversation."""
        conversation_id = conversation_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO conversations (conversation_id, user_id, title, conversation_created_at, conversation_updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, user_id, title, now, now)
            )
        
        logger.info(f"Created conversation: {conversation_id}")
        return conversation_id
    
    def list_conversations(self, limit: int = 50, include_archived: bool = False, user_id: str = "default") -> List[Dict]:
        """List all conversations, ordered by last update."""
        with self.get_cursor() as cursor:
            if include_archived:
                cursor.execute(
                    """
                    SELECT conversation_id, user_id, title, conversation_created_at, conversation_updated_at, is_archived
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY conversation_updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT conversation_id, user_id, title, conversation_created_at, conversation_updated_at, is_archived
                    FROM conversations
                    WHERE user_id = ? AND is_archived = 0
                    ORDER BY conversation_updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit)
                )
            
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a single conversation by ID."""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT conversation_id, user_id, title, conversation_created_at, conversation_updated_at, is_archived
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,)
            )
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None
    
    def update_conversation(self, conversation_id: str, title: Optional[str] = None, is_archived: Optional[bool] = None) -> bool:
        """Update a conversation's title or archive status."""
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        
        if is_archived is not None:
            updates.append("is_archived = ?")
            params.append(1 if is_archived else 0)
        
        if not updates:
            return True
        
        updates.append("conversation_updated_at = ?")
        params.append(datetime.utcnow().isoformat())
        params.append(conversation_id)
        
        with self.get_cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE conversations
                SET {', '.join(updates)}
                WHERE conversation_id = ?
                """,
                params
            )
            return cursor.rowcount > 0
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self.get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,)
            )
            deleted = cursor.rowcount > 0
        
        if deleted:
            logger.info(f"Deleted conversation: {conversation_id}")
        
        return deleted
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None, token_count: Optional[int] = None) -> str:
        """Add a message to a conversation."""
        message_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata) if metadata else None
        now = datetime.utcnow().isoformat()
        
        with self.get_cursor() as cursor:
            # Get latest assistant message for parent linking (if role is assistant)
            parent_id = None
            if role == "assistant":
                cursor.execute(
                    "SELECT message_id FROM messages WHERE conversation_id = ? AND role = 'user' ORDER BY message_created_at DESC LIMIT 1",
                    (conversation_id,)
                )
                row = cursor.fetchone()
                if row:
                    parent_id = row['message_id']

            # Insert message
            cursor.execute(
                """
                INSERT INTO messages (message_id, conversation_id, role, content, metadata_json, token_count, message_created_at, parent_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content, metadata_json, token_count, now, parent_id)
            )
            
            # Update conversation's updated_at
            cursor.execute(
                """
                UPDATE conversations SET conversation_updated_at = ?
                WHERE conversation_id = ?
                """,
                (now, conversation_id)
            )
        
        return message_id
    
    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages for a conversation."""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT message_id, conversation_id, parent_message_id, role, content, metadata_json, token_count, message_created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY message_created_at ASC
                """,
                (conversation_id,)
            )
            
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
            
    def get_active_messages(self, conversation_id: str) -> List[Dict]:
        """Alias for get_messages to support API consistency."""
        return self.get_messages(conversation_id)
            
    def find_duplicate_query(self, conversation_id: str, query: str) -> Optional[Dict]:
        """
        Find exact match for query in this chat and return its response.
        Logic: Find last USER message with same content, then find the ASSISTANT message that followed it.
        CRITICAL: Skips terminated responses (force-stopped by user) to ensure fresh generation.
        """
        query_clean = query.strip().lower()
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT m2.content, m2.metadata_json
                FROM messages m1
                JOIN messages m2 ON m2.parent_message_id = m1.message_id
                WHERE m1.conversation_id = ? 
                  AND m1.role = 'user' 
                  AND LOWER(TRIM(m1.content)) = ?
                  AND m2.role = 'assistant'
                ORDER BY m1.message_created_at DESC
                LIMIT 1
                """,
                (conversation_id, query_clean)
            )
            row = cursor.fetchone()
            if row:
                result = self._row_to_dict(row)
                # STOP BUTTON GUARD: Nuclear string match for the "simple solution"
                content = result.get("content", "").strip().lower()
                meta = result.get("metadata", {})
                if content == "user terminated the generation" or meta.get("terminated"):
                    return None
                result["source"] = "parent_link"
                return result
            
            # FALLBACK (Positional): If no parent_link found (for legacy data)
            # Find the last USER message matching query, and the message that followed it
            cursor.execute(
                """
                SELECT m2.message_id, m2.content, m2.metadata_json
                FROM messages m1
                JOIN messages m2 ON m2.conversation_id = m1.conversation_id
                WHERE m1.conversation_id = ? 
                  AND m1.role = 'user' 
                  AND LOWER(TRIM(m1.content)) = ?
                  AND m2.role = 'assistant'
                  AND m2.message_created_at > m1.message_created_at
                ORDER BY m1.message_created_at DESC, m2.message_created_at ASC
                LIMIT 1
                """,
                (conversation_id, query_clean)
            )
            row = cursor.fetchone()
            if row:
                result = self._row_to_dict(row)
                # STOP BUTTON GUARD: Nuclear string match for the "simple solution"
                content = result.get("content", "").strip().lower()
                meta = result.get("metadata", {})
                if content == "user terminated the generation" or meta.get("terminated"):
                    return None
                result["source"] = "positional_fallback"
                return result

        return None
    
    def increment_duplicate_count(self, message_id: str) -> bool:
        """Increment the duplicate tally for a specific message."""
        with self.get_cursor() as cursor:
            cursor.execute(
                "UPDATE messages SET duplicate_count = duplicate_count + 1 WHERE message_id = ?",
                (message_id,)
            )
            return cursor.rowcount > 0
    
    def get_message_count(self, conversation_id: str) -> int:
        """Get the number of messages in a conversation."""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            return cursor.fetchone()[0]

    def delete_last_message(self, conversation_id: str, role: str) -> bool:
        """Delete the most recent message with specified role in a conversation."""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM messages 
                WHERE message_id = (
                    SELECT message_id FROM messages 
                    WHERE conversation_id = ? AND role = ?
                    ORDER BY message_created_at DESC 
                    LIMIT 1
                )
                """,
                (conversation_id, role)
            )
            return cursor.rowcount > 0

    def ensure_conversation(self, conversation_id: str, title: Optional[str] = None) -> bool:
        """Ensure a conversation exists in SQLite, creating it if necessary."""
        if not conversation_id:
            return False
            
        now = datetime.utcnow().isoformat()
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO conversations (conversation_id, user_id, title, conversation_created_at, conversation_updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, "default", title or "New Chat", now, now)
            )
            return True


        # 4. Re-init Schema
        self.initialize_schema()
        logger.info("Database reset complete (SQLite)")

    def _safe_drop_all_tables(self):
        """Emergency Fallback: Drop all tables if file deletion is blocked by locks."""
        try:
            with self.get_cursor() as cursor:
                # Find all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                    logger.info(f"Dropped table: {table}")
                # Clear migrations if any
                try: cursor.execute("DROP TABLE IF EXISTS alembic_version")
                except: pass
        except Exception as e:
            logger.error(f"Failed to drop tables during reset: {e}")

    def reset_database(self):
        """Drop all tables and re-initialize schema (FRESH START)."""
        logger.warning(f"RESETTING DATABASE: {self.db_path}")
        
        # 1. Force Disconnect to release locks
        if self.is_connected():
            self.disconnect()
            
        # 2. Delete the physical file (Nuclear Option)
        deleted = False
        try:
            if self.db_path.exists():
                os.remove(self.db_path)
                logger.info(f"Deleted database file: {self.db_path}")
                deleted = True
        except Exception as e:
            logger.warning(f"Could not delete database file (likely locked by another process): {e}")
            
        # 3. Reconnect
        self.connect()
        
        # 4. If file deletion failed, we must manually drop all tables to simulate a nuke
        if not deleted:
            logger.info("Falling back to manual TABLE DROP (Nuclear Simulation)...")
            self._safe_drop_all_tables()
        
        # 5. Re-init Schema
        self.initialize_schema()
        logger.info("Database reset complete (SQLite)")

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert a sqlite3.Row to a dictionary with datetime parsing and field standardization."""
        if row is None:
            return None
        
        result = dict(row)
        
        # Parse common timestamp fields into datetime objects
        # Parse common timestamp fields into datetime objects
        for key in ['user_created_at', 'conversation_created_at', 'conversation_updated_at', 
                   'message_created_at', 'file_created_at', 'content_created_at']:
            if key in result and result[key] and isinstance(result[key], str):
                try:
                    val = result[key]
                    if 'T' in val:
                        result[key] = datetime.fromisoformat(val.replace('Z', '+00:00'))
                    else:
                        result[key] = datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    pass
            
        # Standardize Booleans
        if 'is_archived' in result:
            result['is_archived'] = bool(result['is_archived'])
            
        # Parse metadata JSON
        if 'metadata_json' in result:
            if result['metadata_json']:
                try:
                    result['metadata'] = json.loads(result['metadata_json'])
                except (json.JSONDecodeError, TypeError):
                    result['metadata'] = {}
            else:
                result['metadata'] = {}
            del result['metadata_json']
        
        if 'message_id' in result:
            result['id'] = result['message_id']
        elif 'conversation_id' in result:
            result['id'] = result['conversation_id']
        
        # SOTA: Harmonize keys for API compatibility (Ensuring both versions exist)
        if 'id' in result and 'conversation_id' not in result and 'message_id' not in result:
            result['conversation_id'] = result['id']
            
        if 'message_created_at' in result and 'created_at' not in result:
            result['created_at'] = result['message_created_at']
        if 'conversation_created_at' in result and 'created_at' not in result:
            if 'message_created_at' not in result:
                result['created_at'] = result['conversation_created_at']
        if 'conversation_updated_at' in result and 'updated_at' not in result:
            result['updated_at'] = result['conversation_updated_at']
        
        if 'title' in result and 'name' not in result:
            result['name'] = result['title']
            
        return result


# =============================================================================
# POSTGRESQL DATABASE IMPLEMENTATION
# =============================================================================

class PostgreSQLDatabase(BaseDatabase):
    """
    PostgreSQL database implementation.
    Used for cloud deployments (NeonDB, Supabase, etc.).
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize PostgreSQL database.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self._pool = None
        self._connected = False
        
        if not self.database_url:
            logger.warning("DATABASE_URL not set for PostgreSQL")
    
    def connect(self) -> bool:
        """Establish connection pool to PostgreSQL."""
        if not self.database_url:
            return False
        
        try:
            import psycopg2
            from psycopg2 import pool
            
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=self.database_url
            )
            self._connected = True
            logger.info("PostgreSQL connection pool established")
            return True
        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Close all database connections."""
        if self._pool:
            self._pool.closeall()
            self._connected = False
            logger.info("PostgreSQL connections closed")
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self._pool is not None
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        if not self._pool:
            raise RuntimeError("Database not connected")
        
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)
    
    def initialize_schema(self):
        """Create tables if they don't exist (ChatGPT-style schema)."""
        if not self.is_connected():
            logger.warning("Cannot initialize schema - not connected")
            return
        
        from psycopg2.extras import RealDictCursor
        
        schema_sql = """
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            user_id           TEXT PRIMARY KEY,
            user_created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            settings_json     JSONB
        );
        
        -- Insert default user
        INSERT INTO users (user_id) VALUES ('default') ON CONFLICT DO NOTHING;
        
        -- Conversations table
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id           UUID PRIMARY KEY,
            user_id                   TEXT DEFAULT 'default' REFERENCES users(user_id),
            title                     TEXT,
            conversation_created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            conversation_updated_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            is_archived               BOOLEAN DEFAULT FALSE,
            model_config              JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_conversations_user 
            ON conversations(user_id, conversation_updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_conversations_updated 
            ON conversations(conversation_updated_at DESC);
        
        -- Messages table
        CREATE TABLE IF NOT EXISTS messages (
            message_id              UUID PRIMARY KEY,
            conversation_id         UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            parent_message_id       UUID REFERENCES messages(message_id),
            role                    TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content                 TEXT NOT NULL,
            metadata_json           JSONB,
            token_count             INTEGER,
            duplicate_count         INTEGER DEFAULT 0,
            message_created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages(conversation_id, message_created_at);
        """
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(schema_sql)
            logger.info("PostgreSQL schema initialized (ChatGPT-style)")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL schema: {e}")
            raise
    
    def reset_database(self):
        """Drop all tables and re-initialize PostgreSQL schema."""
        logger.warning("RESETTING POSTGRESQL DATABASE")
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS messages CASCADE")
                cur.execute("DROP TABLE IF EXISTS conversations CASCADE")
                cur.execute("DROP TABLE IF EXISTS users CASCADE")
        self.initialize_schema()
        logger.info("Database reset complete (PostgreSQL)")
    
    def create_conversation(self, title: Optional[str] = None, user_id: str = "default", conversation_id: Optional[str] = None) -> str:
        """Create a new conversation."""
        conversation_id = conversation_id or str(uuid.uuid4())
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (conversation_id, user_id, title, conversation_created_at, conversation_updated_at)
                    VALUES (%s, %s, %s, NOW(), NOW())
                    """,
                    (conversation_id, user_id, title)
                )
        
        logger.info(f"Created conversation: {conversation_id}")
        return conversation_id
    
    def list_conversations(self, limit: int = 50, include_archived: bool = False, user_id: str = "default") -> List[Dict]:
        """List all conversations, ordered by last update."""
        from psycopg2.extras import RealDictCursor
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if include_archived:
                    cur.execute(
                        """
                        SELECT conversation_id, user_id, title, conversation_created_at, conversation_updated_at, is_archived
                        FROM conversations
                        WHERE user_id = %s
                        ORDER BY conversation_updated_at DESC
                        LIMIT %s
                        """,
                        (user_id, limit)
                    )
                else:
                    cur.execute(
                        """
                        SELECT conversation_id, user_id, title, conversation_created_at, conversation_updated_at, is_archived
                        FROM conversations
                        WHERE user_id = %s AND is_archived = FALSE
                        ORDER BY conversation_updated_at DESC
                        LIMIT %s
                        """,
                        (user_id, limit)
                    )
                
                rows = cur.fetchall()
                return [dict(row) for row in rows]
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a single conversation by ID."""
        from psycopg2.extras import RealDictCursor
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT conversation_id, user_id, title, conversation_created_at, conversation_updated_at, is_archived
                    FROM conversations
                    WHERE conversation_id = %s
                    """,
                    (conversation_id,)
                )
                row = cur.fetchone()
                return dict(row) if row else None
    
    def update_conversation(self, conversation_id: str, title: Optional[str] = None, is_archived: Optional[bool] = None) -> bool:
        """Update a conversation's title or archive status."""
        updates = []
        params = []
        
        if title is not None:
            updates.append("title = %s")
            params.append(title)
        
        if is_archived is not None:
            updates.append("is_archived = %s")
            params.append(is_archived)
        
        if not updates:
            return True
        
        updates.append("conversation_updated_at = NOW()")
        params.append(conversation_id)
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE conversations
                    SET {', '.join(updates)}
                    WHERE conversation_id = %s
                    """,
                    params
                )
                return cur.rowcount > 0
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversations WHERE conversation_id = %s",
                    (conversation_id,)
                )
                deleted = cur.rowcount > 0
        
        if deleted:
            logger.info(f"Deleted conversation: {conversation_id}")
        
        return deleted
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None, token_count: Optional[int] = None) -> str:
        """Add a message to a conversation."""
        message_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata) if metadata else None
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Get latest user message for parent linking
                parent_id = None
                if role == "assistant":
                    cur.execute(
                        "SELECT message_id FROM messages WHERE conversation_id = %s AND role = 'user' ORDER BY message_created_at DESC LIMIT 1",
                        (conversation_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        parent_id = row[0]

                # Insert message
                cur.execute(
                    """
                    INSERT INTO messages (message_id, conversation_id, role, content, metadata_json, token_count, message_created_at, parent_message_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (message_id, conversation_id, role, content, metadata_json, token_count, NOW(), parent_id)
                )
                
                # Update conversation's conversation_updated_at
                cur.execute(
                    """
                    UPDATE conversations SET conversation_updated_at = NOW()
                    WHERE conversation_id = %s
                    """,
                    (conversation_id,)
                )
        
        return message_id
    
    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages for a conversation."""
        from psycopg2.extras import RealDictCursor
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT message_id, conversation_id, parent_message_id, role, content, metadata_json, token_count, message_created_at
                    FROM messages
                    WHERE conversation_id = %s
                    ORDER BY message_created_at ASC
                    """,
                    (conversation_id,)
                )
                
                rows = cur.fetchall()
                messages = []
                for row in rows:
                    msg = dict(row)
                    # Parse metadata JSON
                    if msg.get('metadata_json'):
                        msg['metadata'] = json.loads(msg['metadata_json']) if isinstance(msg['metadata_json'], str) else msg['metadata_json']
                    else:
                        msg['metadata'] = {}
                    del msg['metadata_json']
                    messages.append(msg)
                
                return messages

    def get_active_messages(self, conversation_id: str) -> List[Dict]:
        """Alias for get_messages for PostgreSQL."""
        return self.get_messages(conversation_id)
                
    def find_duplicate_query(self, conversation_id: str, query: str) -> Optional[Dict]:
        """Find exact match for query in this chat and return its response on PostgreSQL."""
        from psycopg2.extras import RealDictCursor
        query_clean = query.strip().lower()
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT m2.content, m2.metadata_json
                    FROM messages m1
                    JOIN messages m2 ON m2.parent_message_id = m1.message_id
                    WHERE m1.conversation_id = %s 
                      AND m1.role = 'user' 
                      AND LOWER(TRIM(m1.content)) = %s
                      AND m2.role = 'assistant'
                    ORDER BY m1.message_created_at DESC
                    LIMIT 1
                    """,
                    (conversation_id, query_clean)
                )
                row = cur.fetchone()
                if row:
                    result = dict(row)
                    # SOTA: Harmonize metadata
                    if result.get('metadata_json'):
                        result['metadata'] = json.loads(result['metadata_json']) if isinstance(result['metadata_json'], str) else result['metadata_json']
                    
                    # STOP BUTTON GUARD: Nuclear string match for the "simple solution"
                    content = result.get("content", "").strip().lower()
                    meta = result.get("metadata", {})
                    if content == "user terminated the generation" or meta.get("terminated"):
                        return None

                    result["source"] = "parent_link"
                    return result
                
                # FALLBACK (Positional) for PostgreSQL
                cur.execute(
                    """
                    SELECT m2.message_id, m2.content, m2.metadata_json
                    FROM messages m1
                    JOIN messages m2 ON m2.conversation_id = m1.conversation_id
                    WHERE m1.conversation_id = %s 
                      AND m1.role = 'user' 
                      AND LOWER(TRIM(m1.content)) = %s
                      AND m2.role = 'assistant'
                      AND m2.message_created_at > m1.message_created_at
                    ORDER BY m1.message_created_at DESC, m2.message_created_at ASC
                    LIMIT 1
                    """,
                    (conversation_id, query_clean)
                )
                row = cur.fetchone()
                if row:
                    result = dict(row)
                    # SOTA: Harmonize metadata
                    if result.get('metadata_json'):
                        result['metadata'] = json.loads(result['metadata_json']) if isinstance(result['metadata_json'], str) else result['metadata_json']
                    
                    # STOP BUTTON GUARD: Nuclear string match for the "simple solution"
                    content = result.get("content", "").strip().lower()
                    meta = result.get("metadata", {})
                    if content == "user terminated the generation" or meta.get("terminated"):
                        return None
                    
                    result["source"] = "positional_fallback"
                    return result

        return None
    
    def increment_duplicate_count(self, message_id: str) -> bool:
        """Increment the duplicate tally for a specific message on PostgreSQL."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE messages SET duplicate_count = duplicate_count + 1 WHERE message_id = %s",
                    (message_id,)
                )
                return cur.rowcount > 0
    def get_message_count(self, conversation_id: str) -> int:
        """Get the number of messages in a conversation."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = %s",
                    (conversation_id,)
                )
                return cur.fetchone()[0]

    def ensure_conversation(self, conversation_id: str, title: Optional[str] = None) -> bool:
        """Ensure a conversation exists in PostgreSQL, creating it if necessary."""
        if not conversation_id:
            return False
            
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (conversation_id, user_id, title, conversation_created_at, conversation_updated_at)
                    VALUES (%s, %s, %s, NOW(), NOW())
                    ON CONFLICT (conversation_id) DO NOTHING
                    """,
                    (conversation_id, "default", title or "New Chat")
                )
                return True

# =============================================================================
# DATABASE MANAGER (Factory + Singleton)
# =============================================================================

class DatabaseManager:
    """
    Database manager that selects the appropriate backend based on configuration.
    Acts as a factory and provides a consistent interface.
    """
    
    def __init__(self, db_type: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            db_type: Database type ('sqlite' or 'postgresql'). 
                     Defaults to DB_TYPE env var or 'sqlite'.
        """
        self.db_type = db_type or os.getenv("DB_TYPE", "sqlite").lower()
        self._backend: Optional[BaseDatabase] = None
        
        # Initialize appropriate backend
        if self.db_type == "postgresql":
            database_url = os.getenv("DATABASE_URL", "")
            if database_url:
                self._backend = PostgreSQLDatabase(database_url)
            else:
                logger.warning("PostgreSQL selected but DATABASE_URL not set, falling back to SQLite")
                self.db_type = "sqlite"
                self._backend = SQLiteDatabase(os.getenv("SQLITE_DB_PATH"))
        else:
            self._backend = SQLiteDatabase(os.getenv("SQLITE_DB_PATH"))
        
        logger.info(f"Database type selected: {self.db_type}")
    
    def connect(self) -> bool:
        """Connect to the database."""
        return self._backend.connect() if self._backend else False
    
    def disconnect(self):
        """Disconnect from the database."""
        if self._backend:
            self._backend.disconnect()
    
    def is_connected(self) -> bool:
        """Check if connected to the database."""
        return self._backend.is_connected() if self._backend else False
    
    def initialize_schema(self):
        """Initialize database schema."""
        if self._backend:
            self._backend.initialize_schema()
    
    # Delegate all operations to the backend
    def create_conversation(self, title: Optional[str] = None, user_id: str = "default", conversation_id: Optional[str] = None) -> str:
        return self._backend.create_conversation(title, user_id, conversation_id)
    
    def list_conversations(self, limit: int = 50, include_archived: bool = False, user_id: str = "default") -> List[Dict]:
        return self._backend.list_conversations(limit, include_archived, user_id)
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        return self._backend.get_conversation(conversation_id)
    
    def update_conversation(self, conversation_id: str, title: Optional[str] = None, is_archived: Optional[bool] = None) -> bool:
        return self._backend.update_conversation(conversation_id, title, is_archived)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        return self._backend.delete_conversation(conversation_id)
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None, token_count: Optional[int] = None) -> str:
        return self._backend.add_message(conversation_id, role, content, metadata, token_count)
    
    def get_messages(self, conversation_id: str) -> List[Dict]:
        return self._backend.get_messages(conversation_id)
    
    def get_active_messages(self, conversation_id: str) -> List[Dict]:
        return self._backend.get_active_messages(conversation_id)
    
    def find_duplicate_query(self, conversation_id: str, query: str) -> Optional[Dict]:
        return self._backend.find_duplicate_query(conversation_id, query)
    
    def increment_duplicate_count(self, message_id: str) -> bool:
        return self._backend.increment_duplicate_count(message_id)
    
    def get_message_count(self, conversation_id: str) -> int:
        return self._backend.get_message_count(conversation_id)

    def delete_last_message(self, conversation_id: str, role: str) -> bool:
        """Proxy for backend deletion."""
        return self._backend.delete_last_message(conversation_id, role)

    def ensure_conversation(self, conversation_id: str, title: Optional[str] = None) -> bool:
        return self._backend.ensure_conversation(conversation_id, title)

    @property
    def backend(self):
        """Direct access to backend for migration scripts."""
        return self._backend

    def reset_database(self):
        """Perform a factory reset of the database."""
        if self._backend:
            self._backend.reset_database()
        else:
            logger.error("No database backend initialized for reset")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_db_manager: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database() -> bool:
    """Initialize database connection and schema."""
    db = get_database()
    if db.connect():
        db.initialize_schema()
        return True
    return False

