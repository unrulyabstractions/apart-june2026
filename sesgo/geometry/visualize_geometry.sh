#!/bin/bash
# Launch the interactive SESGO geometry PCA visualizer (MULTI-MODEL).
#
# Usage:
#   PORT=8003 sesgo/geometry/visualize_geometry.sh
#
# No model needs to be specified. The server discovers EVERY analysed model under
# out/sesgo/geometry/*/ (each needing response_samples.json + analysis/
# projections.json) and exposes a Model selector so you switch between them live
# in the browser. Optionally pass a different geometry root as $1.
#
# (To refresh a model's projections after new data, run analyze_geometry.py on it
#  separately — this launcher no longer re-analyses on every start, so it's fast.)
set -euo pipefail

# Repo root = two levels up from this script (sesgo/geometry/<here>).
cd "$(dirname "$0")/../.."

PORT="${PORT:-8002}"
ROOT="${1:-out/sesgo/geometry}"

# Open the browser shortly after the server comes up.
(sleep 2 && open "http://localhost:$PORT") >/dev/null 2>&1 &

# Serve (run-by-path; discovers + serves all models; no model arg required).
uv run python sesgo/geometry/geometry_viz_server.py --geometry-root "$ROOT" --port "$PORT"
