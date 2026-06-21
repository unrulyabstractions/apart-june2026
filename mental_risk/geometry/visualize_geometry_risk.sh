#!/bin/bash
# Launch the interactive mental_risk geometry PCA visualizer.
#
# Usage:
#   mental_risk/geometry/visualize_geometry_risk.sh [SAMPLES_JSON]
#   PORT=8006 mental_risk/geometry/visualize_geometry_risk.sh \
#       out/mental_risk/geometry/Qwen3-0.6B/samples.json
#
# Ensures the PCA projection exists (runs analyze_geometry_risk.py on every launch
# so the served data is fresh), opens the browser, then serves the app.
set -euo pipefail

# Repo root = two levels up from this script (mental_risk/geometry/<here>).
cd "$(dirname "$0")/../.."

SAMPLES="${1:-out/mental_risk/geometry/Qwen3-0.6B/samples.json}"
PORT="${PORT:-8003}"

# 1) Produce / refresh projections.json (writes to <MODEL>/analysis/).
uv run python mental_risk/geometry/analyze_geometry_risk.py "$SAMPLES"

# 2) Open the browser shortly after the server comes up.
(sleep 2 && open "http://localhost:$PORT") &

# 3) Serve (run-by-path; no module import needed).
uv run python mental_risk/geometry/geometry_viz_server_risk.py --samples "$SAMPLES" --port "$PORT"
