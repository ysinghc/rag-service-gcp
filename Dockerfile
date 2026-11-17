FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[gcp]"

# Cloud Run injects PORT. The default index lives in the container; see README for production storage.
ENV PORT=8080 STORE_PATH=/tmp/ragservice.sqlite
CMD ["sh", "-c", "uvicorn ragservice.api:app --host 0.0.0.0 --port ${PORT}"]
