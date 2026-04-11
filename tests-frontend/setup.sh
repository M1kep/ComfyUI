#!/usr/bin/env bash
# One-time setup for frontend (browser) tests.
# Clones ComfyUI_frontend, installs its deps + Playwright, installs the
# devtools custom_node, and links this repo's specs/assets into the checkout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="${FRONTEND_DIR:-$SCRIPT_DIR/ComfyUI_frontend}"
FRONTEND_REPO="${FRONTEND_REPO:-https://github.com/Comfy-Org/ComfyUI_frontend.git}"
FRONTEND_REF="${FRONTEND_REF:-main}"

echo "==> ComfyUI dir:      $COMFYUI_DIR"
echo "==> Frontend checkout: $FRONTEND_DIR"

if [ ! -d "$FRONTEND_DIR/.git" ]; then
  echo "==> Cloning $FRONTEND_REPO ($FRONTEND_REF)"
  git clone --depth 1 --branch "$FRONTEND_REF" "$FRONTEND_REPO" "$FRONTEND_DIR"
else
  echo "==> Frontend checkout already exists, skipping clone"
fi

echo "==> Installing ComfyUI_devtools custom_node"
DEVTOOLS_DST="$COMFYUI_DIR/custom_nodes/ComfyUI_devtools"
rm -rf "$DEVTOOLS_DST"
cp -r "$FRONTEND_DIR/tools/devtools" "$DEVTOOLS_DST"

echo "==> Linking backend assets into frontend checkout"
ln -sfn "$SCRIPT_DIR/assets" "$FRONTEND_DIR/browser_tests/assets/backend"

cd "$FRONTEND_DIR"

echo "==> Installing frontend dependencies (pnpm)"
corepack enable || true
pnpm install --frozen-lockfile

echo "==> Installing Playwright chromium"
pnpm exec playwright install chromium

echo
echo "Setup complete. Run tests with:"
echo "  ./tests-frontend/run-tests.sh"
