#!/usr/bin/env bash
set -euo pipefail

# One-command smoke check for frontend metrics payload quality.
# It starts the API, calls /pipeline-metrics, then runs strict checker.
#
# Usage:
#   ./run_dashboard_check.sh
#   ./run_dashboard_check.sh "Iran Israel War"
#
# Optional env overrides:
#   APP_MODULE="main12:app"   (or "mainResearch:app")
#   PORT=8000
#   OUT_JSON=/tmp/pipeline_metrics.json

TOPIC="${1:-AI agents}"
APP_MODULE="${APP_MODULE:-main12:app}"
PORT="${PORT:-8000}"
OUT_JSON="${OUT_JSON:-/tmp/pipeline_metrics.json}"

echo "[check] topic: ${TOPIC}"
echo "[check] app: ${APP_MODULE} on :${PORT}"

cleanup() {
  if [[ -n "${UVICORN_PID:-}" ]] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill "${UVICORN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python3 -m uvicorn "${APP_MODULE}" --host 127.0.0.1 --port "${PORT}" >/tmp/uvicorn_dashboard_check.log 2>&1 &
UVICORN_PID=$!

for i in {1..40}; do
  if curl -sf "http://127.0.0.1:${PORT}/" >/dev/null; then
    break
  fi
  sleep 0.5
done

if ! curl -sf "http://127.0.0.1:${PORT}/" >/dev/null; then
  echo "[check] API did not start. Tail logs:"
  tail -n 120 /tmp/uvicorn_dashboard_check.log || true
  exit 1
fi

curl -s -X POST "http://127.0.0.1:${PORT}/pipeline-metrics" \
  -H "Content-Type: application/json" \
  -d "{\"topic\":\"${TOPIC}\"}" > "${OUT_JSON}"

python3 dashboard_payload_check.py --input "${OUT_JSON}" --strict-briefs

echo "[check] PASS"
