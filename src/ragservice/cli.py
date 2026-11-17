"""Command-line entry points: ingest files, ask a question, run the offline evaluation."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from ragservice.config import Settings
from ragservice.evaluation import evaluate, load_golden
from ragservice.factory import build_pipeline
from ragservice.ingestion import load_documents


def _format_for(path: Path) -> str:
    return {".csv": "csv", ".jsonl": "jsonl"}.get(path.suffix.lower(), "text")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragservice")
    parser.add_argument("--store", default=None, help="SQLite index path (default: STORE_PATH)")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index files (.md, .txt, .csv, .jsonl)")
    ingest.add_argument("paths", nargs="+", type=Path)

    ask = sub.add_parser("ask", help="answer a question from the index")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)

    ev = sub.add_parser("eval", help="run the golden-set evaluation")
    ev.add_argument("--golden", type=Path, required=True)
    ev.add_argument("--k", type=int, default=4)

    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.store:
        settings = dataclasses.replace(settings, store_path=args.store)
    pipeline = build_pipeline(settings)

    if args.command == "ingest":
        total_docs = total_chunks = 0
        for path in args.paths:
            source = path.as_posix()
            documents = load_documents(source, path.read_text(encoding="utf-8"), _format_for(path))
            report = pipeline.ingest(documents)
            total_docs += report.documents
            total_chunks += report.chunks
        print(json.dumps({"documents": total_docs, "chunks": total_chunks}))
        return 0

    if args.command == "ask":
        answer = pipeline.query(args.question, top_k=args.top_k)
        print(json.dumps(dataclasses.asdict(answer), indent=2, default=str))
        return 0

    report = evaluate(pipeline, load_golden(args.golden), k=args.k)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
