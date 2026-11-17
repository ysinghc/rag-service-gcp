"""Tracing and LLM-native metrics: tokens/sec, cost per request, latency percentiles."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens. Replace the defaults with your model's current list price."""

    input_per_mtok: float = 0.30
    output_per_mtok: float = 2.50

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    latency_s: float
    retrieval_s: float
    generation_s: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    cost_usd: float
    abstained: bool

    def as_dict(self) -> dict:
        return asdict(self)


class MetricsRegistry:
    """Thread-safe rolling window of per-request metrics with aggregate snapshots."""

    def __init__(self, window: int = 1000) -> None:
        self._window: deque[RequestMetrics] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._total_requests = 0

    def record(self, metrics: RequestMetrics) -> None:
        with self._lock:
            self._window.append(metrics)
            self._total_requests += 1

    def snapshot(self) -> dict:
        with self._lock:
            items = list(self._window)
            total = self._total_requests
        if not items:
            return {"requests_total": total, "requests_in_window": 0}
        latencies = np.array([m.latency_s for m in items])
        generating = [m for m in items if not m.abstained]
        tps = [m.tokens_per_second for m in generating if m.tokens_per_second > 0]
        cost = sum(m.cost_usd for m in items)
        return {
            "requests_total": total,
            "requests_in_window": len(items),
            "abstain_rate": round(sum(m.abstained for m in items) / len(items), 4),
            "latency_s": {
                "p50": round(float(np.percentile(latencies, 50)), 4),
                "p95": round(float(np.percentile(latencies, 95)), 4),
            },
            "tokens_per_second_mean": round(float(np.mean(tps)), 2) if tps else None,
            "prompt_tokens_total": sum(m.prompt_tokens for m in items),
            "completion_tokens_total": sum(m.completion_tokens for m in items),
            "cost_usd_total": round(cost, 6),
            "cost_usd_per_request": round(cost / len(items), 6),
        }


def build_tracer_provider(exporter: SpanExporter | None = None) -> TracerProvider:
    """Create an isolated tracer provider. Pass the console exporter in dev, OTLP in production."""
    provider = TracerProvider()
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def console_exporter() -> SpanExporter:
    return ConsoleSpanExporter()


def tracer_for(provider: TracerProvider) -> trace.Tracer:
    return provider.get_tracer("ragservice")
