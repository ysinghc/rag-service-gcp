from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ragservice.embeddings import HashingEmbedder
from ragservice.ingestion import load_documents
from ragservice.llm import ExtractiveLLM
from ragservice.pipeline import RAGPipeline
from ragservice.telemetry import build_tracer_provider, tracer_for
from ragservice.vectorstore import SQLiteVectorStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture
def spans():
    return InMemorySpanExporter()


@pytest.fixture
def make_pipeline(spans):
    def _make(llm=None, exporter=None, **kwargs) -> RAGPipeline:
        provider = build_tracer_provider(exporter or spans)
        return RAGPipeline(
            HashingEmbedder(),
            SQLiteVectorStore(":memory:"),
            llm or ExtractiveLLM(),
            tracer=tracer_for(provider),
            **kwargs,
        )

    return _make


@pytest.fixture
def indexed_pipeline(make_pipeline):
    """A pipeline loaded with the bundled sample corpus (documents, CSV tickets, JSONL products)."""
    pipeline = make_pipeline()
    files = [
        ("docs/refund_policy.md", "text"),
        ("docs/security_overview.md", "text"),
        ("docs/onboarding_guide.md", "text"),
        ("tickets.csv", "csv"),
        ("products.jsonl", "jsonl"),
    ]
    for relative, fmt in files:
        text = (DATA / relative).read_text(encoding="utf-8")
        pipeline.ingest(load_documents(f"data/{relative}", text, fmt))
    return pipeline
