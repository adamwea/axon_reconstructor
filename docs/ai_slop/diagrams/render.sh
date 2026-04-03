#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIAG_DIR="$ROOT_DIR/docs/diagrams"
OUT_DIR="$DIAG_DIR/rendered"
CONFIG_JSON="$DIAG_DIR/mmdc-config.json"

# Slide-friendly defaults (override via env vars if desired)
# With large font sizes, a bigger canvas prevents label crowding.
SVG_WIDTH="${SVG_WIDTH:-2560}"
SVG_HEIGHT="${SVG_HEIGHT:-1440}"
PNG_SCALE="${PNG_SCALE:-2}"

if ! command -v mmdc >/dev/null 2>&1; then
  echo "ERROR: 'mmdc' not found. Install it with:" >&2
  echo "  npm install -g @mermaid-js/mermaid-cli" >&2
  echo "Then re-run:" >&2
  echo "  bash docs/diagrams/render.sh" >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: 'node' not found on PATH (Mermaid CLI requires Node.js)." >&2
  echo "" >&2
  echo "WSL note: If 'npm' points into /mnt/c (Windows), you likely installed mmdc" >&2
  echo "on the Windows side but are running this script in WSL." >&2
  echo "" >&2
  echo "Fix options:" >&2
  echo "  - Install Node.js inside WSL (recommended):" >&2
  echo "      conda install -c conda-forge nodejs" >&2
  echo "    then re-run: npm install -g @mermaid-js/mermaid-cli" >&2
  echo "  - OR run this script from a Windows shell where node is available." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[diagrams] Rendering .mmd -> SVG/PNG" 

failures=0
failed_names=()

shopt -s nullglob
for f in "$DIAG_DIR"/*.mmd; do
  base="$(basename "$f" .mmd)"
  svg="$OUT_DIR/${base}.svg"
  png="$OUT_DIR/${base}.png"

  # SVG: best for slides (vector)
  set +e
  mmdc -i "$f" -o "$svg" -b transparent -c "$CONFIG_JSON" -w "$SVG_WIDTH" -H "$SVG_HEIGHT" >/dev/null
  rc_svg=$?
  set -e
  if [[ $rc_svg -ne 0 ]]; then
    echo "[diagrams] ERROR: failed SVG for ${base}" >&2
    failures=$((failures + 1))
    failed_names+=("${base}")
    continue
  fi

  # PNG: handy for quick embeds
  # NOTE: mmdc PNG rendering depends on Chromium; if this fails, use SVG instead.
  mmdc -i "$f" -o "$png" -b transparent -c "$CONFIG_JSON" -w "$SVG_WIDTH" -H "$SVG_HEIGHT" -s "$PNG_SCALE" >/dev/null || true

  echo "[diagrams] ${base}: $(basename "$svg"), $(basename "$png")"
done

if [[ $failures -ne 0 ]]; then
  echo "[diagrams] Completed with ${failures} failure(s): ${failed_names[*]}" >&2
  echo "[diagrams] Partial outputs in: $OUT_DIR" >&2
  exit 1
fi

echo "[diagrams] Done. Outputs in: $OUT_DIR"
