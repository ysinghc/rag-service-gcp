"""Offline evaluation: retrieval recall and MRR, abstention accuracy, generation telemetry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from ragservice.pipeline import RAGPipeline


@dataclass(frozen=True)
class GoldenItem:
    question: str
    # None means the corpus does not answer the question, so the system should abstain.
    expected_source: str | None


def load_golden(path: str | Path) -> list[GoldenItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            items.append(GoldenItem(row["question"], row.get("expected_source")))
    return items


def evaluate(pipeline: RAGPipeline, items: list[GoldenItem], k: int = 4) -> dict:
    positives = [i for i in items if i.expected_source is not None]

    reciprocal_ranks: list[float] = []
    hits_at_k = 0
    for item in positives:
        ranked_sources = [h.chunk.source for h in pipeline.retrieve(item.question, top_k=k)]
        if item.expected_source in ranked_sources:
            rank = ranked_sources.index(item.expected_source) + 1
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    answers = [pipeline.query(i.question, top_k=k) for i in items]
    neg_answers = [a for a, i in zip(answers, items, strict=True) if i.expected_source is None]
    pos_answers = [a for a, i in zip(answers, items, strict=True) if i.expected_source is not None]
    generated = [a for a in answers if not a.abstained and a.metrics is not None]
    generated_metrics = [a.metrics for a in generated]

    return {
        "model": pipeline.llm_name,
        "questions": len(items),
        "retrieval": {
            f"recall_at_{k}": round(hits_at_k / len(positives), 4) if positives else None,
            "mrr": round(mean(reciprocal_ranks), 4) if reciprocal_ranks else None,
        },
        "abstention": {
            "correct_abstain_rate": round(
                sum(a.abstained for a in neg_answers) / len(neg_answers), 4
            )
            if neg_answers
            else None,
            "false_abstain_rate": round(sum(a.abstained for a in pos_answers) / len(pos_answers), 4)
            if pos_answers
            else None,
        },
        "generation": {
            "citation_rate": round(sum(a.grounded for a in generated) / len(generated), 4)
            if generated
            else None,
            "tokens_per_second_mean": round(mean(m.tokens_per_second for m in generated_metrics), 2)
            if generated
            else None,
            "cost_usd_per_request": round(mean(m.cost_usd for m in generated_metrics), 8)
            if generated
            else None,
        },
    }
