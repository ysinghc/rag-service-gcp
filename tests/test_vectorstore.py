import numpy as np
import pytest

from ragservice.models import Chunk
from ragservice.vectorstore import SQLiteVectorStore


def _chunk(cid: str, text: str = "t") -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", source="s.md", text=text, position=0, metadata={"k": 1})


def _unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_search_returns_nearest_neighbour_first():
    store = SQLiteVectorStore(":memory:")
    store.upsert(
        [_chunk("x"), _chunk("y"), _chunk("z")],
        np.vstack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(0.7, 0.7, 0)]),
    )
    hits = store.search(_unit(1, 0.05, 0), k=2)
    assert [h.chunk.chunk_id for h in hits] == ["x", "z"]
    assert hits[0].retriever == "vector"
    assert len(store) == 3


def test_upsert_replaces_existing_chunk_with_same_id():
    store = SQLiteVectorStore(":memory:")
    store.upsert([_chunk("x", "old")], np.vstack([_unit(1, 0)]))
    store.upsert([_chunk("x", "new")], np.vstack([_unit(1, 0)]))
    assert len(store) == 1
    assert store.all_chunks()[0].text == "new"


def test_index_survives_reopening_the_database_file(tmp_path):
    path = str(tmp_path / "index.sqlite")
    first = SQLiteVectorStore(path)
    first.upsert([_chunk("x", "persisted")], np.vstack([_unit(0, 1)]))
    first.close()

    reopened = SQLiteVectorStore(path)
    hit = reopened.search(_unit(0, 1), k=1)[0]
    assert hit.chunk.text == "persisted"
    assert hit.chunk.metadata == {"k": 1}
    reopened.close()


def test_empty_store_returns_no_results():
    assert SQLiteVectorStore(":memory:").search(_unit(1, 0), k=3) == []


def test_mismatched_chunks_and_vectors_are_rejected():
    store = SQLiteVectorStore(":memory:")
    with pytest.raises(ValueError):
        store.upsert([_chunk("x")], np.vstack([_unit(1, 0), _unit(0, 1)]))
