"""Runtime settings, read from environment variables with safe local defaults."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_backend: str = "extractive"  # "gemini" (Vertex AI) or "extractive" (offline)
    embedder: str = "hashing"  # "vertex" or "hashing" (offline)
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-005"
    gcp_project: str | None = None
    gcp_location: str = "us-central1"
    store_path: str = "ragservice.sqlite"
    top_k: int = 4
    min_similarity: float = 0.2  # calibrate per embedder; see README
    price_input_per_mtok: float = 0.30  # placeholder rates; set to your model's list price
    price_output_per_mtok: float = 2.50
    trace_exporter: str | None = None  # "console" or None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        return cls(
            llm_backend=e.get("LLM_BACKEND", cls.llm_backend),
            embedder=e.get("EMBEDDER", cls.embedder),
            gemini_model=e.get("GEMINI_MODEL", cls.gemini_model),
            embedding_model=e.get("EMBEDDING_MODEL", cls.embedding_model),
            gcp_project=e.get("GCP_PROJECT") or None,
            gcp_location=e.get("GCP_LOCATION", cls.gcp_location),
            store_path=e.get("STORE_PATH", cls.store_path),
            top_k=int(e.get("TOP_K", cls.top_k)),
            min_similarity=float(e.get("MIN_SIMILARITY", cls.min_similarity)),
            price_input_per_mtok=float(e.get("PRICE_INPUT_PER_MTOK", cls.price_input_per_mtok)),
            price_output_per_mtok=float(e.get("PRICE_OUTPUT_PER_MTOK", cls.price_output_per_mtok)),
            trace_exporter=e.get("TRACE_EXPORTER") or None,
        )
