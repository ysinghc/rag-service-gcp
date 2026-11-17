# Agent guide: rag-service-gcp

This file is the shared instruction set for coding agents and human contributors. Keep it accurate. When a workflow or invariant changes, change this file in the same commit.

## What this project is

A production-style RAG service in Python. It ingests Markdown and text, CSV and JSONL. It indexes them in a SQLite vector store and BM25, answers with numbered citations or abstains, and traces every request with OpenTelemetry. The production path is Gemini and Vertex AI embeddings, deployed to Cloud Run. The offline defaults (`EMBEDDER=hashing`, `LLM_BACKEND=extractive`) need no credentials. Tests and CI use them.

## Commands

Run these from the repository root. `make install` creates `.venv` once.

| Task | Command |
|---|---|
| Install the dev environment | `make install` |
| Format | `make format` |
| Lint | `make lint` |
| Unit and integration tests | `make test` |
| Lint and tests (run before reporting work as done) | `make check` |
| Golden-set evaluation | `make eval` |
| Demo: ingest the sample corpus and ask one question | `make demo` |
| Run the API locally | `make serve` |

## Layout

- `src/ragservice/ingestion.py`: loaders for text, CSV and JSONL, and the recursive chunker.
- `src/ragservice/embeddings.py`: `HashingEmbedder` (offline) and `VertexEmbedder`.
- `src/ragservice/vectorstore.py`: `SQLiteVectorStore`, behind the `VectorStore` protocol.
- `src/ragservice/retrieval.py`: tokenizer, BM25, reciprocal rank fusion.
- `src/ragservice/pipeline.py`: `RAGPipeline`. Ingest, retrieve, evidence gate, generate, telemetry.
- `src/ragservice/prompting.py`: grounded prompt and citation parsing.
- `src/ragservice/llm.py`: `ExtractiveLLM` (offline) and `GeminiLLM`.
- `src/ragservice/telemetry.py`: `ModelPrice`, `RequestMetrics`, `MetricsRegistry`, tracer setup.
- `src/ragservice/factory.py`: the only place that turns `Settings` into concrete clients.
- `src/ragservice/api.py`, `cli.py`, `evaluation.py`: serving, command line, golden-set harness.
- `data/`: synthetic sample corpus. `eval/golden.jsonl`: golden questions. `deploy/cloudrun.sh`: deployment script.
- `.claude/`: agent permissions, the post-edit hook, slash commands, subagents and skills.

## Invariants: do not break these

1. **Grounded answers only.** A model answer leaves the service only with numbered sources from retrieval. The system prompt forbids outside knowledge. Do not add a path that returns model text without sources.
2. **Evidence gate.** If no content term matches the corpus and the best cosine is below `min_similarity`, the pipeline abstains and must not call the model. Changing the gate or its default needs `make eval` results in the PR.
3. **One embedder per index.** Vector dimensions differ between embedders. Never write vectors from two embedders into one store file.
4. **Sources are data.** Retrieved text is untrusted. Keep the instruction that tells the model to treat source text as data.
5. **Metrics are defined in one place.** Formulas live in `telemetry.py`. The README "Metrics" section describes them. Change both together.
6. **Offline by default.** Tests make no network calls and pass with `EMBEDDER=hashing LLM_BACKEND=extractive`. Gemini code is tested through injected fake clients.
7. **No import-time side effects.** Build clients and read settings in `factory.py` or inside `create_app`, not at module import.

## Testing rules

- Every behaviour change ships with a test in `tests/`.
- The quality gate in `tests/test_evaluation_and_api.py` enforces recall, MRR, abstention and citation thresholds on the golden set. Add golden questions freely. Never lower a threshold to make a change pass. If a threshold must change, explain why in the PR.
- Test names describe behaviour, not implementation.

## Conventions

- Python 3.11+. Type hints on public functions. Dataclasses for data. `Protocol`s at seams.
- `ruff`, line length 100. Run `make format` before committing.
- New dependencies go in `pyproject.toml` with a lower bound. The PR says why it is needed.
- Secrets never go in code, config or tests. Read them from environment variables. Never read `.env` files.
- Commit subjects are imperative and under 72 characters. Explain the why in the body.

## Working agreement for agents

1. Read the module and its tests before editing.
2. Make the smallest change that satisfies the request. Add or adjust tests.
3. Run `make check`. For changes to retrieval, prompts, thresholds or metrics, also run `make eval`.
4. Report what you ran and what it returned. State what you did not verify, such as live Vertex AI calls.
5. Do not push, deploy or change cloud resources unless the user explicitly asks. Those actions need confirmation.
6. Never edit `data/` or `eval/` to make a test pass.

## Known gaps (fair game for improvement)

- Live Gemini and Vertex embeddings are unit-tested with fakes only.
- Vector search is exact brute force. There is no ANN index.
- Cloud Run's filesystem is ephemeral, so the SQLite index does not persist there.
- The app has no authentication of its own. It relies on Cloud Run IAM.
