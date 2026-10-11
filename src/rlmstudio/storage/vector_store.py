# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""SQLite-backed vector storage with cosine similarity search."""

import hashlib
import math
import struct
from datetime import datetime
from typing import Any
from uuid import uuid4

from .database import Database


class VectorStore:
    """SQLite-backed vector storage with cosine similarity search."""

    def __init__(self, db: Database):
        self.db = db

    def add_chunks(
        self,
        collection: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store chunks with embeddings. Deduplicates by content_hash within collection.

        Returns:
            Number of new chunks actually added.
        """
        import json

        added = 0
        now = datetime.now().isoformat()
        meta_json = json.dumps(metadata or {})

        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()

            # Check for duplicate within collection
            existing = self.db.fetchone(
                "SELECT id FROM vector_chunks WHERE collection = ? AND content_hash = ?",
                (collection, content_hash),
            )
            if existing:
                continue

            chunk_id = str(uuid4())
            emb_blob = self._serialize_embedding(embedding)

            self.db.execute(
                """INSERT INTO vector_chunks
                   (id, collection, content_hash, chunk_text, chunk_index,
                    embedding, embedding_dim, source_id, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    collection,
                    content_hash,
                    chunk_text,
                    i,
                    emb_blob,
                    len(embedding),
                    source_id,
                    meta_json,
                    now,
                ),
            )
            added += 1

        return added

    def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[float, str, str]]:
        """Search for similar chunks by cosine similarity.

        Returns:
            List of (score, chunk_id, chunk_text) tuples, highest score first.
        """
        rows = self.db.fetchall(
            "SELECT id, chunk_text, embedding, embedding_dim FROM vector_chunks WHERE collection = ?",
            (collection,),
        )
        if not rows:
            return []

        scored = []
        for row in rows:
            emb = self._deserialize_embedding(row["embedding"], row["embedding_dim"])
            score = self._cosine_similarity(query_embedding, emb)
            scored.append((score, row["id"], row["chunk_text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def get_collection_size(self, collection: str) -> int:
        """Return the number of chunks in a collection."""
        row = self.db.fetchone(
            "SELECT COUNT(*) AS cnt FROM vector_chunks WHERE collection = ?",
            (collection,),
        )
        return row["cnt"] if row else 0

    def delete_collection(self, collection: str) -> None:
        """Delete all chunks in a collection."""
        self.db.execute(
            "DELETE FROM vector_chunks WHERE collection = ?",
            (collection,),
        )

    def has_source(self, collection: str, source_id: str) -> bool:
        """Check if a source has already been indexed in this collection."""
        row = self.db.fetchone(
            "SELECT id FROM vector_chunks WHERE collection = ? AND source_id = ? LIMIT 1",
            (collection, source_id),
        )
        return row is not None

    # ------------------------------------------------------------------
    # Embedding serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_embedding(embedding: list[float]) -> bytes:
        """Pack float32 array to bytes."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _deserialize_embedding(data: bytes, dim: int) -> list[float]:
        """Unpack bytes to float32 array."""
        return list(struct.unpack(f"{dim}f", data))

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
