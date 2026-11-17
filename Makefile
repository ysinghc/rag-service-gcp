# Developer entry points. Humans and coding agents run the same commands.
PYTHON ?= python3
BIN ?= .venv/bin
CORPUS := data/docs/refund_policy.md data/docs/security_overview.md data/docs/onboarding_guide.md data/tickets.csv data/products.jsonl

.PHONY: install format lint test check eval demo serve clean

install:
	$(PYTHON) -m venv .venv
	$(BIN)/pip install -e ".[dev]"

format:
	$(BIN)/ruff format .

lint:
	$(BIN)/ruff check .

test:
	$(BIN)/pytest -q

check: lint test

eval:
	mkdir -p .demo
	rm -f .demo/eval.sqlite
	$(BIN)/ragservice --store .demo/eval.sqlite ingest $(CORPUS)
	$(BIN)/ragservice --store .demo/eval.sqlite eval --golden eval/golden.jsonl

demo:
	mkdir -p .demo
	rm -f .demo/index.sqlite
	$(BIN)/ragservice --store .demo/index.sqlite ingest $(CORPUS)
	$(BIN)/ragservice --store .demo/index.sqlite ask "How long are audit logs retained?"

serve:
	$(BIN)/uvicorn ragservice.api:app --reload

clean:
	rm -rf .pytest_cache .ruff_cache .demo build dist *.egg-info
