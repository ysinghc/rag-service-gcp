"""Persistent vector store backed by SQLite with an in-memory cosine index.

The brute-force matrix product is exact and fast for corpora up to a few hundred thousand chunks.
For larger corpora, swap this class for pgvector (HNSW) or Vertex AI Vector Search behind the same
`VectorStore` protocol; the pipeline does not change.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ragservice.models import Chunk, ScoredChunk


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

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
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._load()

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.source,
                c.text,
                c.position,
                json.dumps(c.metadata, default=str),
                np.asarray(v, dtype=np.float32).tobytes(),
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", rows
            )
            self._load_locked()

    def search(self, query_vector: np.ndarray, k: int) -> list[ScoredChunk]:
        with self._lock:
            if self._matrix is None or not len(self._chunks):
                return []
            query = np.asarray(query_vector, dtype=np.float32)
            scores = self._matrix @ query
            top = np.argsort(-scores)[:k]
            return [ScoredChunk(self._chunks[i], float(scores[i]), "vector") for i in top]

    def all_chunks(self) -> list[Chunk]:
        with self._lock:
            return list(self._chunks)

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    def _load(self) -> None:
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        cursor = self._conn.execute(
            "SELECT chunk_id, doc_id, source, text, position, metadata, vector FROM chunks"
        )
        chunks, vectors = [], []
        for chunk_id, doc_id, source, text, position, metadata, blob in cursor:
            chunks.append(Chunk(chunk_id, doc_id, source, text, position, json.loads(metadata)))
            vectors.append(np.frombuffer(blob, dtype=np.float32))
        self._chunks = chunks
        self._matrix = np.vstack(vectors) if vectors else None

    def close(self) -> None:
        self._conn.close()
