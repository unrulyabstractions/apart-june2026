#!/bin/bash
# Launch the interactive SESGO geometry PCA visualizer (MULTI-MODEL).
#
# Usage:
#   sesgo/geometry/visualize_geometry.sh [SAMPLES_JSON]
#   PORT=8003 sesgo/geometry/visualize_geometry.sh out/sesgo/geometry/Qwen3-0.6B/response_samples.json
#
# SAMPLES_JSON only picks which model the page boots INTO; the server then
# discovers every model under out/sesgo/geometry/*/ (each needing both
# response_samples.json and analysis/projections.json) and exposes a Model
# selector so you can switch between them in the browser.
#
# Ensures the boot model's PCA projection exists (runs analyze_geometry.py if
# needed-on every launch so the served data is fresh), opens the browser, then
# serves the app.
set -euo pipefail

# Repo root = two levels up from this script (sesgo/geometry/<here>).
cd "$(dirname "$0")/../.."

SAMPLES="${1:-out/sesgo/geometry/Qwen3-0.6B/response_samples.json}"
PORT="${PORT:-8002}"

# 1) Produce / refresh projections.json (writes to <MODEL>/analysis/).
uv run python sesgo/geometry/analyze_geometry.py "$SAMPLES"

# 2) Open the browser shortly after the server comes up.
(sleep 2 && open "http://localhost:$PORT") &

# 3) Serve (run-by-path; no module import needed).
uv run python sesgo/geometry/geometry_viz_server.py --samples "$SAMPLES" --port "$PORT"
