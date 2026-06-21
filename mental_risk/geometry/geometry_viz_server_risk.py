"""Interactive web visualizer for the mental_risk geometry PCA.

Run-by-path FastAPI app (risk analogue of sesgo/geometry/geometry_viz_server.py).
Serves the precomputed PCA projection (analyze_geometry_risk.py ->
out/mental_risk/geometry/<MODEL>/analysis/projections.json) as an interactive
Plotly scatter, plus a per-sample DETAIL panel backed by the RiskGeometryDataset
(response_samples.json). The projection is fully precomputed, so the frontend GETs it once
and re-slices client-side; per-sample detail (prompt + the risk readouts) is too
heavy to ship in the blob, so it is fetched lazily by sample_idx.

The HTML/CSS/JS lives in geometry_page_risk.py to keep this file a thin set of
routes. Endpoints mirror SESGO's: / , /api/projections , /api/sample/{idx} ,
/api/config.

Usage (run as a script, NOT a module path):
  uv run python mental_risk/geometry/geometry_viz_server_risk.py \
      --samples out/mental_risk/geometry/Qwen3-0.6B/response_samples.json --port 8003
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves the
# same way the other run-by-path scripts do (two levels deep under the root).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mental_risk.geometry.geometry_page_risk import render_html  # noqa: E402
from src.datasets.risk_geometry import RiskGeometryDataset, RiskGeometrySample  # noqa: E402

# State loaded once at startup so the import side stays free of disk I/O.
_STATE: dict = {"projections": {}, "by_idx": {}, "model_name": ""}


def _projections_path(samples: Path) -> Path:
    """Locate projections.json for a given response_samples.json (sibling analysis/ dir)."""
    return samples.resolve().parent / "analysis" / "projections.json"


def _sample_detail(s: RiskGeometrySample) -> dict:
    """Flatten one RiskGeometrySample into the detail dict the panel renders."""
    nt = s.non_thinking
    th = s.thinking
    return {
        "sample_idx": s.sample_idx, "subject_id": s.subject_id, "framing": s.framing,
        "disorder": s.disorder, "language": s.language, "gold_risk": s.gold_risk,
        "prompt_text": s.prompt_text,
        "non_thinking": ({"predicted_risk": nt.predicted_risk} if nt is not None else None),
        "thinking": ({"mean": th.mean, "std": th.std, "entropy": th.entropy, "n": th.n}
                     if th is not None else None),
    }


def build_app(samples_path: Path) -> FastAPI:
    """Load projections + dataset for ``samples_path`` and wire up the routes."""
    proj_path = _projections_path(samples_path)
    if not proj_path.exists():
        raise FileNotFoundError(
            f"projections not found at {proj_path} — run "
            f"`uv run python mental_risk/geometry/analyze_geometry_risk.py "
            f"{samples_path}` first."
        )
    with open(proj_path) as f:
        projections = json.load(f)
    dataset = RiskGeometryDataset.from_json(samples_path)
    _STATE["projections"] = projections
    _STATE["by_idx"] = {s.sample_idx: s for s in dataset.samples}
    _STATE["model_name"] = projections.get("model_name", dataset.model_name)

    app = FastAPI(title="MentalRisk Geometry Visualizer")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        proj = _STATE["projections"]
        return render_html(_STATE["model_name"], len(_STATE["by_idx"]), proj.get("params", {}))

    @app.get("/api/projections")
    async def api_projections() -> JSONResponse:
        return JSONResponse(_STATE["projections"])

    @app.get("/api/sample/{idx}")
    async def api_sample(idx: int) -> JSONResponse:
        s = _STATE["by_idx"].get(idx)
        if s is None:
            raise HTTPException(status_code=404, detail=f"sample_idx {idx} not found")
        return JSONResponse(_sample_detail(s))

    @app.get("/api/config")
    async def api_config() -> JSONResponse:
        proj = _STATE["projections"]
        params = proj.get("params", {})
        return JSONResponse({
            "model_name": _STATE["model_name"],
            "layers": params.get("layers", list(proj.get("results", {}).keys())),
            "positions": params.get("positions", []),
            "axes": ["framing", "disorder", "language"],
            "n_samples": len(_STATE["by_idx"]),
        })

    return app


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the geometry visualizer server."""
    parser = argparse.ArgumentParser(
        description="Interactive web visualizer for the mental_risk geometry PCA"
    )
    parser.add_argument(
        "--samples", type=Path,
        default=Path("out/mental_risk/geometry/Qwen3-0.6B/response_samples.json"),
        help="response_samples.json (a RiskGeometryDataset); projections.json is read from "
        "its sibling analysis/ dir",
    )
    parser.add_argument("--port", type=int, default=8003, help="server port (default 8003)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="bind host")
    return parser.parse_args()


# Build the app only when run as a script (the supported run-by-path entrypoint).
args = parse_args() if __name__ == "__main__" else None
if args is not None:
    app = build_app(args.samples)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
