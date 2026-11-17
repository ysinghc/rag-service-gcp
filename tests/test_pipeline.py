from ragservice.ingestion import load_documents
from ragservice.llm import Completion
from ragservice.pipeline import ABSTAIN_TEXT
from ragservice.telemetry import ModelPrice


class SpyLLM:
    name = "spy"

    def __init__(self, text: str = "Refunds are allowed within 30 days [1]."):
        self.calls = 0
        self.text = text

    def generate(self, question, context):
        self.calls += 1
        return Completion(self.text, prompt_tokens=200, completion_tokens=25, seconds=0.5)


def _index(pipeline):
    text = "Annual subscriptions have a 30-day refund window. Monthly plans can be cancelled."
    pipeline.ingest(load_documents("refund.md", text))
    return pipeline


def test_answer_is_grounded_with_a_valid_citation(make_pipeline):
    pipeline = _index(make_pipeline())
    answer = pipeline.query("How long is the refund window for annual subscriptions?")
    assert not answer.abstained
    assert answer.grounded
    assert answer.citations[0].index == 1
    assert answer.citations[0].source == "refund.md"
    assert "30-day" in answer.answer
    assert len(answer.trace_id) == 32


def test_off_topic_question_abstains_without_calling_the_model(make_pipeline):
    llm = SpyLLM()
    pipeline = _index(make_pipeline(llm=llm))
    answer = pipeline.query("What is the capital of France?")
    assert answer.abstained
    assert answer.answer == ABSTAIN_TEXT
    assert llm.calls == 0
    assert answer.metrics.cost_usd == 0.0


def test_lexical_match_passes_the_gate_even_when_embedding_similarity_is_low(make_pipeline):
    llm = SpyLLM()
    pipeline = _index(make_pipeline(llm=llm, min_similarity=0.99))  # embedding gate can never pass
    answer = pipeline.query("Monthly plans cancelled?")
    assert llm.calls == 1
    assert not answer.abstained


def test_uncited_model_output_is_reported_as_not_grounded(make_pipeline):
    pipeline = _index(make_pipeline(llm=SpyLLM("Probably, but I have no source for that.")))
    answer = pipeline.query("What is the refund window for annual subscriptions?")
    assert not answer.abstained
    assert not answer.grounded
    assert answer.citations == []


def test_tracing_emits_nested_spans_with_model_attributes(make_pipeline, spans):
    pipeline = _index(make_pipeline(llm=SpyLLM(), price=ModelPrice(1.0, 2.0)))
    spans.clear()
    answer = pipeline.query("refund window for annual subscriptions")
    by_name = {s.name: s for s in spans.get_finished_spans()}
    assert {"rag.query", "rag.retrieve", "rag.generate"} <= by_name.keys()

    root = by_name["rag.query"]
    generate = by_name["rag.generate"]
    assert generate.parent.span_id == root.context.span_id
    assert generate.attributes["llm.model"] == "spy"
    assert generate.attributes["llm.completion_tokens"] == 25
    assert generate.attributes["llm.tokens_per_second"] == 50.0
    assert answer.trace_id == format(root.context.trace_id, "032x")


def test_metrics_are_recorded_per_request(make_pipeline):
    pipeline = _index(make_pipeline(llm=SpyLLM(), price=ModelPrice(1.0, 2.0)))
    pipeline.query("refund window for annual subscriptions")
    pipeline.query("What is the capital of France?")
    snap = pipeline.metrics.snapshot()
    assert snap["requests_total"] == 2
    assert snap["abstain_rate"] == 0.5
    assert snap["tokens_per_second_mean"] == 50.0
    # 200 prompt tokens at $1/Mtok + 25 completion tokens at $2/Mtok
    assert round(snap["cost_usd_total"], 8) == round((200 + 50) / 1_000_000, 8)


def test_incremental_ingest_adds_chunks_and_is_immediately_queryable(make_pipeline):
    pipeline = _index(make_pipeline())
    before = pipeline.chunk_count
    pipeline.ingest(load_documents("other.md", "Dashboards refresh every fifteen minutes."))
    assert pipeline.chunk_count == before + 1
    answer = pipeline.query("How often do dashboards refresh?")
    assert answer.citations and answer.citations[0].source == "other.md"
