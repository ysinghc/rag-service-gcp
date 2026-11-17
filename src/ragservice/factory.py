"""Wire settings to concrete backends. Kept separate so importing modules has no side effects."""

from __future__ import annotations

from ragservice.config import Settings
from ragservice.embeddings import HashingEmbedder, VertexEmbedder
from ragservice.llm import ExtractiveLLM, GeminiLLM
from ragservice.pipeline import RAGPipeline
from ragservice.telemetry import ModelPrice, build_tracer_provider, console_exporter, tracer_for
from ragservice.vectorstore import SQLiteVectorStore


def build_pipeline(settings: Settings) -> RAGPipeline:
    if settings.embedder == "vertex":
        embedder = VertexEmbedder(
            settings.embedding_model, settings.gcp_project, settings.gcp_location
        )
    else:
        embedder = HashingEmbedder()
    if settings.llm_backend == "gemini":
        llm = GeminiLLM(settings.gemini_model, settings.gcp_project, settings.gcp_location)
    else:
        llm = ExtractiveLLM()
    exporter = console_exporter() if settings.trace_exporter == "console" else None
    provider = build_tracer_provider(exporter)
    return RAGPipeline(
        embedder,
        SQLiteVectorStore(settings.store_path),
        llm,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        price=ModelPrice(settings.price_input_per_mtok, settings.price_output_per_mtok),
        tracer=tracer_for(provider),
    )
