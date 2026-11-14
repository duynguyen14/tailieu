#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example; review secrets before exposing port 8002."
fi

required_files=(
  "models/page_classifier/model.onnx"
  "models/page_classifier/labels.json"
  "models/passport_mrz/detector.onnx"
  "models/passport_mrz/recognition/inference.pdiparams"
  "output/ocr_recognition_latin_ppocrv5_mobile_phase3_v3/inference/inference.pdiparams"
  "output/student_extractor_qwen3_1_7b_augmented_v9/merged_model/model.safetensors"
)

missing=0
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "MISSING: $ROOT_DIR/$path" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Production stack was not started because model artifacts are missing." >&2
  exit 1
fi

command -v docker >/dev/null 2>&1 || {
  echo "Docker is not installed." >&2
  exit 1
}
nvidia-smi >/dev/null

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon needs elevated access; requesting sudo once."
  DOCKER=(sudo docker)
  "${DOCKER[@]}" info >/dev/null
fi
"${DOCKER[@]}" compose version >/dev/null

COMPOSE=("${DOCKER[@]}" compose -f docker-compose.production.yml --env-file .env)

# Complete slow downloads/builds while the existing manually-started services
# are still serving traffic.
"${COMPOSE[@]}" build api ocr passport-mrz
"${COMPOSE[@]}" pull vllm
"${COMPOSE[@]}" down --remove-orphans

stop_legacy_listener() {
  local port="$1"
  local expected_pattern="$2"
  local pids pid command_line
  pids="$(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  [[ -z "$pids" ]] && return 0

  for pid in $pids; do
    command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ ! "$command_line" =~ $expected_pattern ]]; then
      echo "Port $port is occupied by an unexpected process; refusing to kill it:" >&2
      echo "  PID=$pid CMD=$command_line" >&2
      exit 1
    fi
    echo "Stopping legacy service on port $port: PID=$pid"
    kill "$pid"
  done

  for _ in {1..30}; do
    lsof -t -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 || return 0
    sleep 1
  done
  echo "Legacy service on port $port did not stop within 30 seconds." >&2
  exit 1
}

command -v lsof >/dev/null 2>&1 || {
  echo "lsof is required to migrate the existing manually-started services." >&2
  exit 1
}

stop_legacy_listener 8001 'app\.production_ocr_main:app'
stop_legacy_listener 8002 'app\.production_main:app'
stop_legacy_listener 8005 'vllm.*student|student.*vllm'
stop_legacy_listener 8007 'app\.passport_mrz_main:app'

"${COMPOSE[@]}" up -d --no-build --remove-orphans

"${COMPOSE[@]}" ps

cat <<'EOF'

Stack is starting. Useful commands:
  docker compose -f docker-compose.production.yml ps
  docker compose -f docker-compose.production.yml logs -f
  curl -s http://127.0.0.1:8002/api/v1/document-extractions/readiness | python3 -m json.tool
EOF

