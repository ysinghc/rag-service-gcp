"""Embedding backends. `HashingEmbedder` runs offline; `VertexEmbedder` calls Vertex AI."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from ragservice.retrieval import STOPWORDS, tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams, signed hashing, L2-normalised).

    It is not a neural model, but it has real lexical-similarity behaviour, is reproducible
    across processes, and needs no credentials. Used for tests, CI and offline evaluation.
    """

    name = "hashing-512"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._embed(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def _embed(self, text: str) -> np.ndarray:
        terms = [t for t in tokenize(text) if t not in STOPWORDS]
        features = terms + [f"{a}_{b}" for a, b in zip(terms, terms[1:], strict=False)]
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return _normalise(vector)


class VertexEmbedder:
    """Embeddings from a Vertex AI text-embedding model via the google-genai SDK."""

    _BATCH = 100

    def __init__(
        self,
        model: str = "text-embedding-005",
        project: str | None = None,
        location: str = "us-central1",
        client: Any | None = None,
    ) -> None:
        self.name = model
        if client is None:
            from google import genai

            client = genai.Client(vertexai=True, project=project, location=location)
        self._client = client
        self._model = model
        self.dim = 768  # text-embedding-005 default output dimensionality

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(list(texts), task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self._BATCH):
            batch = texts[start : start + self._BATCH]
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            rows.extend(np.asarray(e.values, dtype=np.float32) for e in response.embeddings)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        self.dim = rows[0].shape[0]
        return np.vstack([_normalise(r) for r in rows])


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector
