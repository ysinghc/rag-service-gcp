---
name: add-document-format
description: Adds a new source format, such as HTML, PDF text or Parquet, to the RAG ingestion path, with tests and API and CLI support. Use when the user asks to index a new kind of file.
---

# Add a document format

Use this when a new file type or record format needs to be indexed.

## Steps

1. Add a loader to `src/ragservice/ingestion.py` that returns `list[Document]`. Structured formats produce one document per record, and each document keeps its field names in its text, as the CSV and JSONL loaders do. Plain text uses `_text_document`.
2. Extend the `SourceFormat` literal and the dispatch in `load_documents`.
3. Extend `DocumentIn.format` in `src/ragservice/api.py` so the API accepts the new format.
4. Extend `_format_for` in `src/ragservice/cli.py` so the file suffix maps to the format.
5. Add tests to `tests/test_ingestion.py`: a happy path, an empty input, and one malformed input that must raise `ValueError`.
6. If the format needs a new dependency, add it to `pyproject.toml` as an optional extra. Import it inside the loader, and skip the test with `pytest.importorskip` when it is missing.
7. Add a sample file under `data/`. Add it to `CORPUS` in the `Makefile` and to the `files` list in `indexed_pipeline` in `tests/conftest.py`. Add a golden question to `eval/golden.jsonl` that the new file answers.
8. Add the format to the "What it demonstrates" table in `README.md`.

## Done when

- `make check` passes.
- `make eval` passes every gate, and the new golden question retrieves the new file.
- The format appears in the README.
