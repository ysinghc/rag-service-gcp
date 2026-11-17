"""HTTP API (FastAPI). Run locally with: uvicorn ragservice.api:app --reload"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragservice import __version__
from ragservice.config import Settings
from ragservice.factory import build_pipeline
from ragservice.ingestion import load_documents
from ragservice.pipeline import RAGPipeline


class DocumentIn(BaseModel):
    source: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1, max_length=2_000_000)
    format: Literal["text", "csv", "jsonl"] = "text"


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


def create_app(settings: Settings | None = None, pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    pipeline = pipeline or build_pipeline(settings)
    app = FastAPI(title="RAG service", version=__version__)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "chunks": pipeline.chunk_count}

    @app.post("/documents", status_code=201)
    def ingest(body: DocumentIn) -> dict[str, Any]:
        try:
            documents = load_documents(body.source, body.content, body.format)
        except ValueError as exc:  # includes json.JSONDecodeError
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        report = pipeline.ingest(documents)
        return {
            "documents": report.documents,
            "chunks": report.chunks,
            "seconds": round(report.seconds, 4),
        }

    @app.post("/query")
    def query(body: QueryIn) -> dict[str, Any]:
        return asdict(pipeline.query(body.question, top_k=body.top_k))

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return pipeline.metrics.snapshot()

    return app


_default_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    """Build the default app on first access so importing this module has no side effects."""
    global _default_app
    if name == "app":
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(name)
