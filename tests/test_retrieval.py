import numpy as np

from ragservice.embeddings import HashingEmbedder
from ragservice.models import Chunk, ScoredChunk
from ragservice.retrieval import BM25Index, reciprocal_rank_fusion, tokenize


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=cid, source=f"{cid}.md", text=text, position=0)


def test_bm25_ranks_the_document_with_the_rare_term_first():
    index = BM25Index(
        [
            _chunk("a", "the refund policy covers annual plans"),
            _chunk("b", "the plan includes dashboards and the plan supports sso"),
            _chunk("c", "the the the the the"),
        ]
    )
    hits = index.search("refund annual", k=3)
    assert hits[0].chunk.chunk_id == "a"
    assert all(h.retriever == "bm25" for h in hits)


def test_bm25_ignores_stopwords_and_unknown_terms():
    index = BM25Index([_chunk("a", "audit logs are retained for 400 days")])
    assert index.search("what is the", k=5) == []
    assert index.search("zebra", k=5) == []


def test_bm25_on_empty_index_returns_nothing():
    assert BM25Index([]).search("anything", k=3) == []


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("SAML-SSO, 2FA!") == ["saml", "sso", "2fa"]


def test_tokenize_folds_plurals_consistently_on_both_sides():
    assert tokenize("warehouses") == tokenize("warehouse") == ["warehouse"]
    assert tokenize("business") == ["business"]  # double-s words are not folded


def test_rrf_rewards_items_ranked_well_by_both_retrievers():
    a, b, c = _chunk("a", "x"), _chunk("b", "x"), _chunk("c", "x")
    vector = [ScoredChunk(a, 0.9, "vector"), ScoredChunk(b, 0.8, "vector")]
    lexical = [ScoredChunk(c, 9.0, "bm25"), ScoredChunk(b, 8.0, "bm25")]
    fused = reciprocal_rank_fusion([vector, lexical], top_k=3)
    assert fused[0].chunk.chunk_id == "b"  # ranked second by both beats first by one
    assert {f.chunk.chunk_id for f in fused} == {"a", "b", "c"}
    assert all(f.retriever == "hybrid" for f in fused)


def test_hashing_embedder_is_deterministic_normalised_and_semantically_ordered():
    embedder = HashingEmbedder(dim=256)
    q = embedder.embed_query("how long are audit logs retained")
    close = embedder.embed_documents(["audit logs are retained for 400 days"])[0]
    far = embedder.embed_documents(["chocolate cake needs flour and sugar"])[0]
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.allclose(q, embedder.embed_query("how long are audit logs retained"))
    assert float(q @ close) > float(q @ far) + 0.1


def test_hashing_embedder_handles_empty_batch():
    assert HashingEmbedder(dim=8).embed_documents([]).shape == (0, 8)
