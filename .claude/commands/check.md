---
description: Run lint and tests, then summarise any failures with file and line references
allowed-tools: Bash(make check), Bash(make lint), Bash(make test), Read, Grep
---

Run `make check` from the repository root.

- If everything passes, reply with the test count and stop.
- If anything fails, list each failure as `file:line`, the assertion or error message, and your best one-sentence diagnosis.
- Do not change code unless the user asks you to fix it.

Extra focus from the user, if any: $ARGUMENTS
