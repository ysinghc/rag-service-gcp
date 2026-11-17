"""Generation backends behind one interface: Gemini on Vertex AI, or an offline extractive model."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ragservice.prompting import build_prompt
from ragservice.retrieval import STOPWORDS, tokenize


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class LLM(Protocol):
    name: str

    def generate(self, question: str, context: Sequence[str]) -> Completion: ...


class GeminiLLM:
    """Gemini on Vertex AI. Token counts come from the provider's usage metadata."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        project: str | None = None,
        location: str = "us-central1",
        client: Any | None = None,
    ) -> None:
        self.name = model
        if client is None:
            from google import genai

            client = genai.Client(vertexai=True, project=project, location=location)
        self._client = client

    def generate(self, question: str, context: Sequence[str]) -> Completion:
        from google.genai import types

        system, user = build_prompt(question, context)
        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.name,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0.1, max_output_tokens=1024
            ),
        )
        elapsed = time.perf_counter() - started
        usage = response.usage_metadata
        return Completion(
            text=(response.text or "").strip(),
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            completion_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            seconds=elapsed,
        )


class ExtractiveLLM:
    """Offline stand-in that answers with the sentence most similar to the question, cited as [1].

    It is deliberately simple: it exists so the whole service (retrieval, citations, metrics,
    API) can be exercised in CI and locally without cloud credentials.
    """

    name = "extractive-offline"
    _SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

    def generate(self, question: str, context: Sequence[str]) -> Completion:
        started = time.perf_counter()
        query_terms = {t for t in tokenize(question) if t not in STOPWORDS}
        best_sentence, best_overlap = "", -1
        for index, passage in enumerate(context, start=1):
            for sentence in self._SENTENCE.split(passage):
                sentence = sentence.strip()
                if not sentence:
                    continue
                overlap = len(query_terms & set(tokenize(sentence)))
                if overlap > best_overlap:
                    best_sentence, best_overlap = sentence, overlap
                    best_index = index
        if best_overlap <= 0:
            text = "I don't know based on the provided documents."
        else:
            text = f"{best_sentence} [{best_index}]"
        elapsed = time.perf_counter() - started
        prompt_tokens = _estimate_tokens(question) + sum(_estimate_tokens(c) for c in context)
        return Completion(text, prompt_tokens, _estimate_tokens(text), elapsed)


def _estimate_tokens(text: str) -> int:
    # Rough word-based estimate used only by the offline model.
    return max(1, round(len(text.split()) * 1.3))
