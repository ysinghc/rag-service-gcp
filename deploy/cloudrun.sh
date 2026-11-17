#!/usr/bin/env bash
# Deploy the service to Cloud Run with Gemini on Vertex AI and Vertex AI embeddings.
#
# Prerequisites:
#   gcloud auth login && gcloud config set project "$GCP_PROJECT"
#   gcloud services enable run.googleapis.com aiplatform.googleapis.com cloudbuild.googleapis.com
#   The runtime service account needs roles/aiplatform.user.
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT to your project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-rag-service}"
SA="${SERVICE_ACCOUNT:-}"

extra=()
if [[ -n "$SA" ]]; then extra+=(--service-account "$SA"); fi

gcloud run deploy "$SERVICE" \
  --source . \
  --project "$GCP_PROJECT" \
  --region "$REGION" \
  --no-allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 120 \
  --concurrency 20 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "LLM_BACKEND=gemini,EMBEDDER=vertex,GCP_PROJECT=$GCP_PROJECT,GCP_LOCATION=$REGION,GEMINI_MODEL=${GEMINI_MODEL:-gemini-2.5-flash},EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-005},TRACE_EXPORTER=console" \
  "${extra[@]}"
