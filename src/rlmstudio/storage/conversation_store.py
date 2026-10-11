# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Conversation and message CRUD operations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .database import Database

if TYPE_CHECKING:
    from rlmstudio.ui.services.models import ChatMessage


class ConversationStore:
    """CRUD operations for conversations and messages."""

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------
    # Conversation lifecycle
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        name: str = "Untitled",
        mode: str = "compare",
        provider: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new conversation and return its ID."""
        conv_id = str(uuid4())
        now = datetime.now().isoformat()
        self.db.execute(
            """INSERT INTO conversations
               (id, name, created_at, updated_at, mode, provider, model, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (conv_id, name, now, now, mode, provider, model, json.dumps(metadata or {})),
        )
        return conv_id

    def list_conversations(self) -> list[dict[str, Any]]:
        """List all conversations ordered by most recently updated."""
        rows = self.db.fetchall(
            """SELECT c.*, COUNT(m.id) AS message_count
               FROM conversations c
               LEFT JOIN messages m ON m.conversation_id = c.id
               GROUP BY c.id
               ORDER BY c.updated_at DESC"""
        )
        return [self._row_to_dict(r) for r in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get a single conversation by ID."""
        row = self.db.fetchone(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        return self._row_to_dict(row) if row else None

    def rename_conversation(self, conversation_id: str, name: str) -> None:
        """Rename a conversation."""
        now = datetime.now().isoformat()
        self.db.execute(
            "UPDATE conversations SET name = ?, updated_at = ? WHERE id = ?",
            (name, now, conversation_id),
        )

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all its messages (cascade)."""
        self.db.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )

    def update_conversation_timestamp(self, conversation_id: str) -> None:
        """Touch the updated_at field."""
        now = datetime.now().isoformat()
        self.db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )

    # ------------------------------------------------------------------
    # Message persistence
    # ------------------------------------------------------------------

    def save_message(self, conversation_id: str, message: ChatMessage) -> None:  # noqa: F821
        """Serialize a ChatMessage and store it."""

        seq = self.get_message_count(conversation_id)

        # Handle file context dedup
        file_hash = None
        if message.file_context:
            filename = None
            if message.file_info:
                filename = message.file_info.get("name")
            file_hash = self.save_file_context(message.file_context, filename)

        self.db.execute(
            """INSERT OR REPLACE INTO messages
               (id, conversation_id, sequence, timestamp, user_query, mode,
                file_context_hash,
                rlm_response_json, rlm_metrics_json, rlm_trace_json,
                direct_response_json, direct_metrics_json,
                rag_response_json, rag_metrics_json, rag_trace_json,
                comparison_json, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                message.id,
                conversation_id,
                seq,
                message.timestamp.isoformat(),
                message.user_query,
                message.mode,
                file_hash,
                _serialize_optional(message.rlm_response),
                _serialize_optional(message.rlm_metrics),
                json.dumps(message.rlm_trace) if message.rlm_trace else None,
                _serialize_optional(message.direct_response),
                _serialize_optional(message.direct_metrics),
                _serialize_optional(message.rag_response),
                _serialize_optional(message.rag_metrics),
                json.dumps(message.rag_trace) if message.rag_trace else None,
                _serialize_optional(message.comparison_metrics),
                message.error,
            ),
        )
        self.update_conversation_timestamp(conversation_id)

    def load_messages(self, conversation_id: str) -> list[ChatMessage]:
        """Load all messages for a conversation, ordered by sequence."""
        from rlmstudio.ui.services.models import (
            ChatMessage,
        )

        rows = self.db.fetchall(
            """SELECT * FROM messages
               WHERE conversation_id = ?
               ORDER BY sequence""",
            (conversation_id,),
        )
        messages = []
        for row in rows:
            r = dict(row)
            # Restore file context
            file_context = None
            file_info = None
            if r["file_context_hash"]:
                file_context = self.get_file_context(r["file_context_hash"])
                # Reconstruct minimal file_info from file_contexts table
                fc_row = self.db.fetchone(
                    "SELECT filename, size_bytes FROM file_contexts WHERE content_hash = ?",
                    (r["file_context_hash"],),
                )
                if fc_row:
                    file_info = {"name": fc_row["filename"], "size_bytes": fc_row["size_bytes"]}

            msg = ChatMessage(
                id=r["id"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                user_query=r["user_query"],
                file_context=file_context,
                file_info=file_info,
                mode=r["mode"],
                rlm_response=_deserialize_response(r["rlm_response_json"]),
                rlm_metrics=_deserialize_metrics(r["rlm_metrics_json"]),
                rlm_trace=json.loads(r["rlm_trace_json"]) if r["rlm_trace_json"] else None,
                direct_response=_deserialize_response(r["direct_response_json"]),
                direct_metrics=_deserialize_metrics(r["direct_metrics_json"]),
                rag_response=_deserialize_response(r["rag_response_json"]),
                rag_metrics=_deserialize_metrics(r["rag_metrics_json"]),
                rag_trace=json.loads(r["rag_trace_json"]) if r["rag_trace_json"] else None,
                comparison_metrics=_deserialize_comparison(r["comparison_json"]),
                in_progress=False,
                error=r["error"],
            )
            messages.append(msg)
        return messages

    def get_message_count(self, conversation_id: str) -> int:
        """Return the number of messages in a conversation."""
        row = self.db.fetchone(
            "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # File context dedup
    # ------------------------------------------------------------------

    def save_file_context(
        self,
        content: str,
        filename: str | None = None,
    ) -> str:
        """Save file content (deduped by hash). Returns content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = self.db.fetchone(
            "SELECT content_hash FROM file_contexts WHERE content_hash = ?",
            (content_hash,),
        )
        if not existing:
            now = datetime.now().isoformat()
            self.db.execute(
                """INSERT INTO file_contexts
                   (content_hash, filename, content, size_bytes, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (content_hash, filename, content, len(content.encode()), now),
            )
        return content_hash

    def get_file_context(self, content_hash: str) -> str | None:
        """Retrieve file content by hash."""
        row = self.db.fetchone(
            "SELECT content FROM file_contexts WHERE content_hash = ?",
            (content_hash,),
        )
        return row["content"] if row else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        d = dict(row)
        if "metadata_json" in d:
            d["metadata"] = json.loads(d.pop("metadata_json"))
        return d


# ------------------------------------------------------------------
# Serialization helpers (module-level for testability)
# ------------------------------------------------------------------


def _serialize_optional(obj: Any) -> str | None:
    """Serialize a dataclass instance to JSON string, or None."""
    if obj is None:
        return None
    from dataclasses import asdict

    data = asdict(obj)
    # Convert datetime fields to ISO strings
    for k, v in data.items():
        if isinstance(v, datetime):
            data[k] = v.isoformat()
    return json.dumps(data)


def _deserialize_response(json_str: str | None) -> Any:
    """Deserialize a Response from JSON string."""
    if not json_str:
        return None
    from rlmstudio.ui.services.models import Response

    d = json.loads(json_str)
    return Response(**d)


def _deserialize_metrics(json_str: str | None) -> Any:
    """Deserialize ExecutionMetrics from JSON string."""
    if not json_str:
        return None
    from rlmstudio.ui.services.models import ExecutionMetrics

    d = json.loads(json_str)
    return ExecutionMetrics(**d)


def _deserialize_comparison(json_str: str | None) -> Any:
    """Deserialize ComparisonMetrics from JSON string."""
    if not json_str:
        return None
    from rlmstudio.ui.services.models import ComparisonMetrics

    d = json.loads(json_str)
    return ComparisonMetrics(**d)
