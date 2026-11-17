# Instructions for the coding agent

@AGENTS.md

## Workflow in this repository

- For changes to `pipeline.py`, `prompting.py`, `retrieval.py` or `telemetry.py`, plan before editing. Summarise the approach and the tests you will add, then wait for approval.
- Run `/check` before saying work is finished. Run `/eval` after any change that could move retrieval or abstention numbers.
- Before committing a change under `src/ragservice/`, ask the `rag-reviewer` subagent to review the diff. Fix blockers. Discuss majors with the user.
- Edited Python files are formatted and lint-fixed by the post-edit hook. If lint errors remain, the hook reports them back to you. Fix the code instead of adding `# noqa`.
- Pre-approved without asking: the `make` targets listed in `.claude/settings.json`, `pytest`, `ruff`, `ragservice`, and read-only git commands. `git commit`, `git push`, `gcloud` and `docker` need confirmation.
- Never open `.env` files. Configuration comes from the environment variables listed in the README.
