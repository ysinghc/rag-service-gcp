"""Regression tests for defects found by end-to-end testing."""

import numpy as np
import pytest

from ragservice import cli
from ragservice.ingestion import load_documents
from ragservice.models import Chunk
from ragservice.retrieval import tokenize
from ragservice.vectorstore import SQLiteVectorStore


def _chunk(cid: str, source: str = "s.md") -> Chunk:
    return Chunk(chunk_id=cid, doc_id=cid, source=source, text=f"text {cid}", position=0)


def _unit(dim: int) -> np.ndarray:
    v = np.ones(dim, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_reingesting_a_shorter_document_replaces_its_old_chunks(make_pipeline):
    pipeline = make_pipeline()
    long_text = " ".join(f"Clause {i} covers legacy returns." for i in range(200))
    pipeline.ingest(load_documents("policy.md", long_text))
    assert pipeline.chunk_count > 1

    pipeline.ingest(load_documents("policy.md", "Returns now close after 14 days."))

    assert pipeline.chunk_count == 1
    assert all("Clause" not in c.chunk.text for c in pipeline.retrieve("legacy returns clause"))


def test_rejected_mixed_dimension_write_leaves_the_index_unchanged():
    store = SQLiteVectorStore(":memory:")
    store.upsert([_chunk("a")], _unit(512).reshape(1, -1))

    with pytest.raises(ValueError, match="dimension"):
        store.upsert([_chunk("b")], _unit(768).reshape(1, -1))

    assert len(store) == 1
    assert [c.chunk_id for c in store.all_chunks()] == ["a"]
    assert store.search(_unit(512), k=5)[0].chunk.chunk_id == "a"


def test_replace_documents_removes_old_chunks_and_inserts_new_ones_atomically():
    store = SQLiteVectorStore(":memory:")
    store.upsert([_chunk("old1", "p.md"), _chunk("old2", "p.md")], np.vstack([_unit(4)] * 2))
    store.replace_documents({"p.md"}, [_chunk("new1", "p.md")], _unit(4).reshape(1, -1))
    assert [c.chunk_id for c in store.all_chunks()] == ["new1"]


def test_tokenizer_keeps_non_ascii_words_whole():
    assert tokenize("Rückerstattung 30 Tagen") == ["rückerstattung", "30", "tagen"]
    assert tokenize("Café-Gutscheine") == ["café", "gutscheine"]


def test_non_english_question_retrieves_the_matching_document(make_pipeline):
    pipeline = make_pipeline()
    pipeline.ingest(load_documents("de.md", "Die Rückerstattung erfolgt innerhalb von 30 Tagen."))
    answer = pipeline.query("Rückerstattung 30 Tagen")
    assert not answer.abstained
    assert answer.citations[0].source == "de.md"


def test_cli_reports_an_unavailable_gemini_backend_without_a_traceback(monkeypatch, tmp_path):
    def broken(_settings):
        raise RuntimeError("Your default credentials were not found")

    monkeypatch.setattr(cli, "build_pipeline", broken)
    with pytest.raises(SystemExit, match="application-default login"):
        cli.main(["--store", str(tmp_path / "x.sqlite"), "ask", "anything at all"])
