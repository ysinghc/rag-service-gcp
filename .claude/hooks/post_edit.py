#!/usr/bin/env python3
"""PostToolUse hook: format an edited Python file, then lint-fix it.

Lint errors that remain are written to stderr with exit code 2, so the agent sees them and fixes
the code. Edits to non-Python files, and machines without ruff, are ignored. The hook never blocks
ordinary work.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_S = 60


def find_ruff() -> str | None:
    local = REPO_ROOT / ".venv" / "bin" / "ruff"
    if local.exists():
        return str(local)
    return shutil.which("ruff")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    raw_path = (payload.get("tool_input") or {}).get("file_path", "")
    if not isinstance(raw_path, str) or not raw_path.endswith(".py"):
        return 0
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    ruff = find_ruff()
    if ruff is None or not path.is_file():
        return 0

    try:
        subprocess.run([ruff, "format", "--quiet", str(path)], check=False, timeout=TIMEOUT_S)
        lint = subprocess.run(
            [ruff, "check", "--fix", "--quiet", "--output-format", "concise", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return 0

    if lint.returncode != 0:
        sys.stderr.write(f"ruff still reports issues in {path.name}:\n{lint.stdout}{lint.stderr}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
