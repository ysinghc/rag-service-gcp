---
description: Ask the demo index a question and show the grounded answer with its citations
argument-hint: <question>
allowed-tools: Bash(make demo), Bash(ragservice:*), Read
---

If `.demo/index.sqlite` does not exist, run `make demo` first.

Then run `ragservice --store .demo/index.sqlite ask "$ARGUMENTS"` and summarise:

- the answer text
- whether the service abstained, and whether the answer is grounded
- each citation as `[n] source: snippet`
- latency, tokens and cost from the metrics block
