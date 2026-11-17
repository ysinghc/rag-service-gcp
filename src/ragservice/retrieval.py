"""Lexical retrieval (BM25) and rank fusion (Reciprocal Rank Fusion) for hybrid search."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from ragservice.models import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from had has have how i if in into is it its
    of on or our that the their them then there these this to was we were what when where which
    who why will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [_fold_plural(t) for t in _TOKEN.findall(text.lower())]


def _fold_plural(token: str) -> str:
    """Crude plural folding (warehouses to warehouse), applied to queries and documents alike."""
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


class BM25Index:
    """Okapi BM25 over chunk text, backed by an inverted index of postings lists."""

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        for index, chunk in enumerate(self._chunks):
            counts = Counter(t for t in tokenize(chunk.text) if t not in STOPWORDS)
            self._lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings[term].append((index, tf))
        n = len(self._chunks)
        self._avgdl = (sum(self._lengths) / n) if n else 0.0
        self._idf = {
            term: math.log(1 + (n - len(posts) + 0.5) / (len(posts) + 0.5))
            for term, posts in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            if term in STOPWORDS or term not in self._postings:
                continue
            idf = self._idf[term]
            for index, tf in self._postings[term]:
                norm = self._k1 * (1 - self._b + self._b * self._lengths[index] / self._avgdl)
                scores[index] += idf * tf * (self._k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [ScoredChunk(self._chunks[i], s, "bm25") for i, s in ranked]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], top_k: int, k: int = 60
) -> list[ScoredChunk]:
    """Merge ranked lists by summing 1 / (k + rank). Robust to incomparable score scales."""
    fused: dict[str, float] = defaultdict(float)
    best: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item.chunk.chunk_id] += 1.0 / (k + rank)
            best.setdefault(item.chunk.chunk_id, item.chunk)
    ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [ScoredChunk(best[cid], score, "hybrid") for cid, score in ordered]
