"""The RAG pipeline: ingestion, hybrid retrieval, abstention, grounded generation and telemetry."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4

from opentelemetry import trace

from ragservice.embeddings import Embedder
from ragservice.ingestion import make_chunks
from ragservice.llm import LLM
from ragservice.models import Document, ScoredChunk
from ragservice.prompting import extract_citations
from ragservice.retrieval import BM25Index, reciprocal_rank_fusion
from ragservice.telemetry import MetricsRegistry, ModelPrice, RequestMetrics
from ragservice.vectorstore import VectorStore

ABSTAIN_TEXT = "I don't know based on the provided documents."


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    source: str
    score: float
    snippet: str


@dataclass(frozen=True)
class Answer:
    request_id: str
    trace_id: str
    question: str
    answer: str
    grounded: bool
    abstained: bool
    citations: list[Citation] = field(default_factory=list)
    metrics: RequestMetrics | None = None


@dataclass(frozen=True)
class IngestReport:
    documents: int
    chunks: int
    seconds: float


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        *,
        top_k: int = 4,
        candidate_k: int = 20,
        min_similarity: float = 0.2,
        price: ModelPrice | None = None,
        tracer: trace.Tracer | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._min_similarity = min_similarity
        self._price = price or ModelPrice()
        self._tracer = tracer or trace.get_tracer("ragservice")
        self.metrics = metrics or MetricsRegistry()
        self._write_lock = threading.Lock()
        self._bm25 = BM25Index(store.all_chunks())

    @property
    def chunk_count(self) -> int:
        return len(self._store)

    @property
    def llm_name(self) -> str:
        return self._llm.name

    def ingest(
        self, documents: Iterable[Document], chunk_size: int = 800, overlap: int = 120
    ) -> IngestReport:
        started = perf_counter()
        docs = list(documents)
        chunks = make_chunks(docs, chunk_size, overlap)
        sources = {doc.source for doc in docs}
        with self._write_lock:  # serialise writers; readers see the index swap atomically
            # Re-ingesting a source replaces its previous content, so retired chunks disappear.
            vectors = self._embedder.embed_documents([c.text for c in chunks])
            self._store.replace_documents(sources, chunks, vectors)
            self._bm25 = BM25Index(self._store.all_chunks())
        return IngestReport(len(docs), len(chunks), perf_counter() - started)

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        hits, _, _ = self._retrieve(question, top_k or self._top_k)
        return hits

    def query(self, question: str, top_k: int | None = None) -> Answer:
        request_id = uuid4().hex[:16]
        started = perf_counter()
        with self._tracer.start_as_current_span("rag.query") as root:
            trace_id = format(root.get_span_context().trace_id, "032x")
            root.set_attribute("rag.request_id", request_id)
            root.set_attribute("rag.question_chars", len(question))

            with self._tracer.start_as_current_span("rag.retrieve") as span:
                t0 = perf_counter()
                hits, best_similarity, lexical_match = self._retrieve(
                    question, top_k or self._top_k
                )
                retrieval_s = perf_counter() - t0
                span.set_attribute("retrieve.best_similarity", round(best_similarity, 4))
                span.set_attribute("retrieve.lexical_match", lexical_match)
                span.set_attribute("retrieve.hits", len(hits))

            # Evidence gate: answer only if a content term appears in the corpus, or the best
            # embedding similarity clears the threshold. Otherwise abstain; the LLM is not called.
            supported = lexical_match or best_similarity >= self._min_similarity
            if not hits or not supported:
                root.set_attribute("rag.abstained", True)
                metrics = RequestMetrics(
                    request_id=request_id,
                    latency_s=perf_counter() - started,
                    retrieval_s=retrieval_s,
                    generation_s=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    tokens_per_second=0.0,
                    cost_usd=0.0,
                    abstained=True,
                )
                self.metrics.record(metrics)
                return Answer(
                    request_id, trace_id, question, ABSTAIN_TEXT, False, True, [], metrics
                )

            context = [h.chunk.text for h in hits]
            with self._tracer.start_as_current_span("rag.generate") as span:
                completion = self._llm.generate(question, context)
                tps = (
                    completion.completion_tokens / completion.seconds
                    if completion.seconds > 0
                    else 0.0
                )
                cost = self._price.cost(completion.prompt_tokens, completion.completion_tokens)
                span.set_attribute("llm.model", self._llm.name)
                span.set_attribute("llm.prompt_tokens", completion.prompt_tokens)
                span.set_attribute("llm.completion_tokens", completion.completion_tokens)
                span.set_attribute("llm.tokens_per_second", round(tps, 2))
                span.set_attribute("llm.cost_usd", round(cost, 8))

            cited = extract_citations(completion.text, len(hits))
            citations = [
                Citation(
                    index=n,
                    chunk_id=hits[n - 1].chunk.chunk_id,
                    source=hits[n - 1].chunk.source,
                    score=round(hits[n - 1].score, 6),
                    snippet=hits[n - 1].chunk.text[:240],
                )
                for n in cited
            ]
            metrics = RequestMetrics(
                request_id=request_id,
                latency_s=perf_counter() - started,
                retrieval_s=retrieval_s,
                generation_s=completion.seconds,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                tokens_per_second=tps,
                cost_usd=cost,
                abstained=False,
            )
            root.set_attribute("rag.cited_sources", len(citations))
            self.metrics.record(metrics)
            return Answer(
                request_id=request_id,
                trace_id=trace_id,
                question=question,
                answer=completion.text,
                grounded=bool(citations),
                abstained=False,
                citations=citations,
                metrics=metrics,
            )

    def _retrieve(self, question: str, top_k: int) -> tuple[list[ScoredChunk], float, bool]:
        """Hybrid retrieval. Returns fused hits, the best cosine, and whether BM25 matched."""
        query_vector = self._embedder.embed_query(question)
        vector_hits = self._store.search(query_vector, self._candidate_k)
        lexical_hits = self._bm25.search(question, self._candidate_k)
        best_similarity = vector_hits[0].score if vector_hits else 0.0
        fused = reciprocal_rank_fusion([vector_hits, lexical_hits], top_k=top_k)
        return fused, best_similarity, bool(lexical_hits)
