# Rules for `src/ragservice`

These rules apply on top of `AGENTS.md` when working in this package.

- `factory.py` is the only module that reads `Settings` and constructs clients. Everything else receives its dependencies.
- Backends implement a protocol from `embeddings.py`, `vectorstore.py` or `llm.py`. A new backend needs three things: the class, a branch in `factory.py`, and a unit test that uses a fake client. Tests make no network calls.
- `prompting.py` is the grounding contract. Any change there needs updated citation tests and a `make eval` run.
- `RAGPipeline.query` must record a `RequestMetrics` on every path, including abstention, so `/metrics` stays complete.
- Keep the trace shape. `rag.query` is the root, with `rag.retrieve` and `rag.generate` as children. Span and attribute names are part of the observability contract, so rename them only with a README update.
- `min_similarity` is calibrated per embedder. Document any calibration change in the PR and in the README configuration table.
