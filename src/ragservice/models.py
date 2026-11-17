"""Core data types shared across ingestion, storage, retrieval and serving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A unit of source material before chunking (one file, or one row of a table)."""

    doc_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a document."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    retriever: str  # "vector", "bm25" or "hybrid"
