"""Loaders for unstructured text and structured records, plus a recursive chunker."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Sequence
from typing import Literal

from ragservice.models import Chunk, Document

SourceFormat = Literal["text", "csv", "jsonl"]

_SEPARATORS = ("\n\n", "\n", ". ", " ")


def load_documents(source: str, content: str, fmt: SourceFormat = "text") -> list[Document]:
    """Turn raw content into documents. Structured rows become one document each."""
    if fmt == "text":
        return [_text_document(source, content)]
    if fmt == "csv":
        return _csv_documents(source, content)
    if fmt == "jsonl":
        return _jsonl_documents(source, content)
    raise ValueError(f"unsupported format: {fmt}")


def _text_document(source: str, content: str) -> Document:
    return Document(doc_id=_digest(source), source=source, text=content.strip())


def _csv_documents(source: str, content: str) -> list[Document]:
    reader = csv.DictReader(io.StringIO(content))
    docs = []
    for row_number, row in enumerate(reader, start=1):
        record = {k: (v or "").strip() for k, v in row.items() if k}
        docs.append(_record_document(source, row_number, record))
    return docs


def _jsonl_documents(source: str, content: str) -> list[Document]:
    docs = []
    for row_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"line {row_number} of {source} is not a JSON object")
        docs.append(_record_document(source, row_number, record))
    return docs


def _record_document(source: str, row_number: int, record: dict) -> Document:
    # Render each field as "name: value" so the embedder and the LLM see the column names too.
    text = "\n".join(f"{key}: {value}" for key, value in record.items() if value not in ("", None))
    return Document(
        doc_id=f"{_digest(source)}-row{row_number}",
        source=source,
        text=text,
        metadata={"row": row_number, "record": record},
    )


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into <= chunk_size character windows that prefer paragraph and sentence breaks.

    Consecutive chunks share up to `overlap` characters so facts that straddle a boundary remain
    retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    pieces = _split(text.strip(), chunk_size, _SEPARATORS)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        candidate = f"{current} {piece}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        if " " in tail:
            tail = tail[tail.index(" ") + 1 :]  # start the overlap on a word boundary
        carried = f"{tail} {piece}".strip()
        current = carried if len(carried) <= chunk_size else piece
    if current:
        chunks.append(current)
    return chunks


def _split(text: str, size: int, separators: Sequence[str]) -> list[str]:
    if not text:
        return []
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    head, *rest = separators
    pieces: list[str] = []
    for part in text.split(head):
        part = part.strip()
        if part:
            pieces.extend(_split(part, size, rest))
    return pieces


def make_chunks(documents: Sequence[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        for position, text in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{position}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    text=text,
                    position=position,
                    metadata=dict(doc.metadata),
                )
            )
    return chunks


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
