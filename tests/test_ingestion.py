import pytest

from ragservice.ingestion import chunk_text, load_documents, make_chunks
from ragservice.models import Document


def test_chunks_never_exceed_chunk_size():
    text = " ".join(f"Sentence number {i} explains a detail." for i in range(200))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_overlap_on_word_boundaries():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, chunk_size=120, overlap=30)
    for left, right in zip(chunks, chunks[1:], strict=False):
        shared = set(left.split()) & set(right.split())
        assert shared, "adjacent chunks should share words"


def test_no_words_are_lost():
    words = [f"token{i}" for i in range(300)]
    chunks = chunk_text(" ".join(words), chunk_size=90, overlap=20)
    recovered = {w for c in chunks for w in c.split()}
    assert recovered == set(words)


def test_empty_text_produces_no_chunks():
    assert chunk_text("   ") == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=size, overlap=overlap)


def test_csv_rows_become_one_document_each_with_column_names():
    csv_text = "id,plan,issue\n1,Pro,export fails\n2,Standard,colours\n"
    docs = load_documents("tickets.csv", csv_text, "csv")
    assert len(docs) == 2
    assert "plan: Pro" in docs[0].text and "issue: export fails" in docs[0].text
    assert docs[1].metadata["row"] == 2
    assert docs[0].doc_id != docs[1].doc_id


def test_jsonl_skips_blank_lines_and_rejects_non_objects():
    docs = load_documents("p.jsonl", '{"sku": "A"}\n\n{"sku": "B"}\n', "jsonl")
    assert [d.text for d in docs] == ["sku: A", "sku: B"]
    with pytest.raises(ValueError):
        load_documents("bad.jsonl", "[1, 2]\n", "jsonl")


def test_text_document_is_stripped_and_stable_id():
    first = load_documents("a.md", "  hello  ", "text")[0]
    second = load_documents("a.md", "other", "text")[0]
    assert first.text == "hello"
    assert first.doc_id == second.doc_id


def test_make_chunks_assigns_ids_and_keeps_metadata():
    doc = Document("d1", "src.md", "alpha beta gamma " * 50, {"owner": "team"})
    chunks = make_chunks([doc], chunk_size=100, overlap=10)
    assert chunks[0].chunk_id == "d1:0"
    assert all(c.metadata == {"owner": "team"} for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))
