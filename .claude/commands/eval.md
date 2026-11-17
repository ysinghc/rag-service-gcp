---
description: Run the golden-set evaluation and compare each metric with its quality-gate threshold
allowed-tools: Bash(make eval), Bash(make test), Read
---

Run `make eval`, then `make test` to exercise the quality gate.

Report a table with one row per metric: recall@4, MRR, correct abstention, false abstention and citation rate. Columns: value, threshold (read from `tests/test_evaluation_and_api.py`), and pass or fail.

If a metric fails, read `src/ragservice/evaluation.py` and `src/ragservice/pipeline.py`, then explain which part of the pipeline most likely caused the failure.

Do not edit thresholds, the golden set or the corpus.

Focus, if any: $ARGUMENTS
