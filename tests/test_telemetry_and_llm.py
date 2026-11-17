from types import SimpleNamespace

import pytest

from ragservice.llm import ExtractiveLLM
from ragservice.prompting import build_prompt, extract_citations
from ragservice.telemetry import MetricsRegistry, ModelPrice, RequestMetrics


def _metric(latency, tps=40.0, cost=0.001, abstained=False, prompt=100, completion=20):
    return RequestMetrics(
        request_id="r",
        latency_s=latency,
        retrieval_s=0.01,
        generation_s=0.5,
        prompt_tokens=prompt,
        completion_tokens=completion,
        tokens_per_second=tps,
        cost_usd=cost,
        abstained=abstained,
    )


def test_model_price_is_per_million_tokens():
    assert ModelPrice(0.30, 2.50).cost(1_000_000, 1_000_000) == pytest.approx(2.80)


def test_snapshot_reports_percentiles_cost_and_throughput():
    registry = MetricsRegistry()
    for latency in [0.1, 0.2, 0.3, 0.4, 5.0]:
        registry.record(_metric(latency, tps=50.0, cost=0.002))
    snap = registry.snapshot()
    assert snap["requests_total"] == 5
    assert snap["latency_s"]["p50"] == 0.3
    assert snap["latency_s"]["p95"] > 0.4
    assert snap["tokens_per_second_mean"] == 50.0
    assert snap["cost_usd_per_request"] == pytest.approx(0.002)


def test_abstained_requests_do_not_skew_throughput():
    registry = MetricsRegistry()
    registry.record(_metric(0.1, tps=60.0))
    registry.record(_metric(0.1, tps=0.0, abstained=True))
    snap = registry.snapshot()
    assert snap["tokens_per_second_mean"] == 60.0
    assert snap["abstain_rate"] == 0.5


def test_empty_registry_snapshot_is_safe():
    assert MetricsRegistry().snapshot() == {"requests_total": 0, "requests_in_window": 0}


def test_rolling_window_bounds_memory_but_counts_all_requests():
    registry = MetricsRegistry(window=3)
    for _ in range(10):
        registry.record(_metric(0.1))
    snap = registry.snapshot()
    assert snap["requests_total"] == 10
    assert snap["requests_in_window"] == 3


def test_gemini_adapter_maps_provider_usage_metadata():
    pytest.importorskip("google.genai")
    from ragservice.llm import GeminiLLM

    calls = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.update(model=model, contents=contents, system=config.system_instruction)
            return SimpleNamespace(
                text=" Refunds take 5 to 10 days [1]. ",
                usage_metadata=SimpleNamespace(prompt_token_count=321, candidates_token_count=45),
            )

    llm = GeminiLLM(model="gemini-test", client=SimpleNamespace(models=FakeModels()))
    completion = llm.generate("How long?", ["Refunds take 5 to 10 days."])
    assert completion.text == "Refunds take 5 to 10 days [1]."
    assert (completion.prompt_tokens, completion.completion_tokens) == (321, 45)
    assert completion.seconds >= 0
    assert calls["model"] == "gemini-test"
    assert "[1] Refunds take 5 to 10 days." in calls["contents"]


def test_extractive_model_cites_the_best_matching_source():
    completion = ExtractiveLLM().generate(
        "How long are audit logs retained?",
        ["Dashboards refresh often.", "Audit logs are retained for 400 days. Backups are daily."],
    )
    assert completion.text == "Audit logs are retained for 400 days. [2]"


def test_extractive_model_says_idk_when_nothing_overlaps():
    completion = ExtractiveLLM().generate("zebra migration?", ["Audit logs are retained."])
    assert "don't know" in completion.text


def test_prompt_numbers_sources_and_forbids_outside_knowledge():
    system, user = build_prompt("q?", ["first passage", "second passage"])
    assert "[1] first passage" in user and "[2] second passage" in user
    assert "Do not use outside knowledge" in system


def test_extract_citations_keeps_order_drops_duplicates_and_out_of_range():
    assert extract_citations("A [2] then B [1] and again [2] and [9].", 3) == [2, 1]
    assert extract_citations("no citations here", 3) == []
