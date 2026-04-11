#!/usr/bin/env bash
# Run frontend (browser) tests against a locally booted ComfyUI server.
# Any extra arguments are passed through to `playwright test`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="${FRONTEND_DIR:-$SCRIPT_DIR/ComfyUI_frontend}"
PORT="${PORT:-8188}"

if [ ! -d "$FRONTEND_DIR" ]; then
  echo "Frontend checkout not found at $FRONTEND_DIR" >&2
  echo "Run ./tests-frontend/setup.sh first." >&2
  exit 1
fi

SERVER_PID=""
if [ -z "${SKIP_SERVER:-}" ]; then
  echo "==> Starting ComfyUI server (--cpu --multi-user) on port $PORT"
  cd "$COMFYUI_DIR"
  python main.py --cpu --multi-user --port "$PORT" > "$SCRIPT_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  trap '[[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true' EXIT

  echo "==> Waiting for server to become ready"
  ready=""
  for _ in $(seq 1 60); do
    if curl -fs "http://127.0.0.1:$PORT/system_stats" -o /dev/null 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server process exited; see $SCRIPT_DIR/server.log" >&2
      exit 1
    fi
    sleep 1
  done
  if [ -z "$ready" ]; then
    echo "Server did not become ready within 60s; see $SCRIPT_DIR/server.log" >&2
    exit 1
  fi
  echo "==> Server ready"
else
  echo "==> SKIP_SERVER set; assuming server already running on port $PORT"
fi

cp "$SCRIPT_DIR/playwright.config.ts" "$FRONTEND_DIR/playwright.backend.config.ts"

cd "$FRONTEND_DIR"
export PLAYWRIGHT_TEST_URL="http://localhost:$PORT"
export TEST_COMFYUI_DIR="$COMFYUI_DIR"
export BACKEND_SPECS_DIR="$SCRIPT_DIR/specs"

echo "==> Running Playwright tests ($BACKEND_SPECS_DIR)"
pnpm exec playwright test --config playwright.backend.config.ts "$@"
