# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""SQLite connection manager with schema initialization."""

import sqlite3
from pathlib import Path
from typing import cast

from rlmstudio.branding import state_dir

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """\
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Conversations (top-level sessions)
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'Untitled',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'compare',
    provider TEXT,
    model TEXT,
    metadata_json TEXT DEFAULT '{}'
);

-- Messages (one row per user turn, all mode responses together)
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    user_query TEXT NOT NULL,
    mode TEXT NOT NULL,
    file_context_hash TEXT,
    rlm_response_json TEXT,
    rlm_metrics_json TEXT,
    rlm_trace_json TEXT,
    direct_response_json TEXT,
    direct_metrics_json TEXT,
    rag_response_json TEXT,
    rag_metrics_json TEXT,
    rag_trace_json TEXT,
    comparison_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, sequence);

-- File contexts (deduplicated by content hash)
CREATE TABLE IF NOT EXISTS file_contexts (
    content_hash TEXT PRIMARY KEY,
    filename TEXT,
    content TEXT NOT NULL,
    size_bytes INTEGER,
    created_at TEXT NOT NULL
);

-- Vector store: document chunks with embeddings
CREATE TABLE IF NOT EXISTS vector_chunks (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding BLOB,
    embedding_dim INTEGER,
    source_id TEXT,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vectors_collection
    ON vector_chunks(collection);
CREATE INDEX IF NOT EXISTS idx_vectors_content_hash
    ON vector_chunks(content_hash);
"""


def get_default_db_path() -> Path:
    """Return the default database path: ``<state dir>/conversations.db``."""
    return Path(state_dir()) / "conversations.db"


class Database:
    """SQLite connection manager with schema initialization."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = get_default_db_path()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def connect(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist and record schema version."""
        conn = self.connect()
        conn.executescript(_SCHEMA_SQL)
        # Record version if not already present
        existing = conn.execute(
            "SELECT version FROM schema_version WHERE version = ?",
            (_SCHEMA_VERSION,),
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (_SCHEMA_VERSION,),
            )
            conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement and commit."""
        conn = self.connect()
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a SQL statement for each set of params and commit."""
        conn = self.connect()
        conn.executemany(sql, params_list)
        conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute and return a single row."""
        return cast(sqlite3.Row | None, self.connect().execute(sql, params).fetchone())

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute and return all rows."""
        return self.connect().execute(sql, params).fetchall()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
