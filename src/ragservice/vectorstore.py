"""Persistent vector store backed by SQLite with an in-memory cosine index.

The brute-force matrix product is exact and fast for corpora up to a few hundred thousand chunks.
For larger corpora, swap this class for pgvector (HNSW) or Vertex AI Vector Search behind the same
`VectorStore` protocol; the pipeline does not change.

Every write is validated before it touches the database, and the in-memory index is replaced only
after the database commit succeeds. A rejected write therefore leaves both copies unchanged.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from ragservice.models import Chunk, ScoredChunk


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def replace_documents(
        self, sources: Iterable[str], chunks: Sequence[Chunk], vectors: np.ndarray
    ) -> None: ...

    def search(self, query_vector: np.ndarray, k: int) -> list[ScoredChunk]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def __len__(self) -> int: ...


class SQLiteVectorStore:
    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                   chunk_id TEXT PRIMARY KEY,
                   doc_id TEXT NOT NULL,
                   source TEXT NOT NULL,
                   text TEXT NOT NULL,
                   position INTEGER NOT NULL,
                   metadata TEXT NOT NULL,
                   vector BLOB NOT NULL)"""
        )
        self._conn.commit()
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        with self._lock:
            self._load_locked()

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Insert or overwrite chunks by chunk_id. Other chunks are kept."""
        self._write(chunks, vectors, delete_sources=())

    def replace_documents(
        self, sources: Iterable[str], chunks: Sequence[Chunk], vectors: np.ndarray
    ) -> None:
        """Remove every chunk from `sources`, then insert `chunks`, in one transaction."""
        self._write(chunks, vectors, delete_sources=tuple(sources))

    def search(self, query_vector: np.ndarray, k: int) -> list[ScoredChunk]:
        with self._lock:
            if self._matrix is None or not len(self._chunks):
                return []
            query = np.asarray(query_vector, dtype=np.float32)
            if query.shape[0] != self._matrix.shape[1]:
                raise ValueError(
                    f"query has dimension {query.shape[0]} but this index uses "
                    f"{self._matrix.shape[1]}; query with the same embedder that built it"
                )
            scores = self._matrix @ query
            top = np.argsort(-scores)[:k]
            return [ScoredChunk(self._chunks[i], float(scores[i]), "vector") for i in top]

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks)

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    def _write(
        self, chunks: Sequence[Chunk], vectors: np.ndarray, delete_sources: tuple[str, ...]
    ) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or len(chunks) != len(matrix):
            raise ValueError("chunks and vectors must pair up one to one")
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.source,
                c.text,
                c.position,
                json.dumps(c.metadata, default=str),
                v.tobytes(),
            )
            for c, v in zip(chunks, matrix, strict=True)
        ]
        with self._lock:
            if len(matrix):
                existing = self._matrix.shape[1] if self._matrix is not None else None
                if existing is not None and matrix.shape[1] != existing:
                    raise ValueError(
                        f"embedding dimension {matrix.shape[1]} does not match this index "
                        f"({existing}). Build a separate index file for each embedder."
                    )
            with self._conn:  # one transaction: deletes and inserts commit together or not at all
                if delete_sources:
                    self._conn.executemany(
                        "DELETE FROM chunks WHERE source = ?", [(s,) for s in delete_sources]
                    )
                self._conn.executemany(
                    "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", rows
                )
            self._load_locked()

    def _load_locked(self) -> None:
        cursor = self._conn.execute(
            "SELECT chunk_id, doc_id, source, text, position, metadata, vector FROM chunks"
        )
        chunks: list[Chunk] = []
        vectors: list[np.ndarray] = []
        for chunk_id, doc_id, source, text, position, metadata, blob in cursor:
            chunks.append(Chunk(chunk_id, doc_id, source, text, position, json.loads(metadata)))
            vectors.append(np.frombuffer(blob, dtype=np.float32))
        matrix = np.vstack(vectors) if vectors else None
        # Assign only after the full load succeeds, so a failure never leaves a half-updated index.
        self._chunks = chunks
        self._matrix = matrix

    def close(self) -> None:
        self._conn.close()
