"""Interactive web visualizer for the SESGO geometry PCA (multi-model).

Run-by-path FastAPI app. Serves the precomputed PCA projection (from
analyze_geometry.py -> out/sesgo/geometry/<MODEL>/analysis/projections.json) as an
interactive Plotly scatter, plus a per-sample DETAIL panel backed by the
GeometryDataset (response_samples.json from collect_geometry_samples.py).

MODEL SWITCHING: the server discovers EVERY model under out/sesgo/geometry/*/
that has both a response_samples.json and a sibling analysis/projections.json
(see geometry_model_registry). The frontend exposes a model selector; switching
re-fetches that model's config + projections and re-renders. Each model's blobs
load lazily on first selection, so startup stays instant with one model or many.

WHY a static server (no websockets): unlike the temporal-manifolds webapp, the
data here is fully precomputed by analyze_geometry.py, so the frontend only needs
to GET a model's projection once and re-slice it client-side as the user flips
model/layer/position/color-by/2D-3D. Per-sample detail (prompt text + the
non_thinking / thinking readouts) is too heavy to ship in the projection blob, so
it is fetched lazily by (model, sample_idx) from the loaded GeometryDataset.

The HTML page is built as a Python f-string (Plotly.js from CDN); there is no
separate static bundle, mirroring how the projection JSON is self-contained.

Endpoints (every data endpoint takes a ?model= query param; default = first):
  GET /                    the single-page app (HTML).
  GET /api/models          {models: [name, ...], default: name}.
  GET /api/projections     the selected model's raw projections.json.
  GET /api/sample/{idx}    per-sample detail dict (prompt + readouts).
  GET /api/config          {model_name, layers, positions, axes, n_samples};
                           ``axes`` is the FULL colour-by registry restricted to
                           axes with real variation in this model (each
                           {key, label, continuous}).

Usage (run as a script, NOT a module path):
  uv run python sesgo/geometry/geometry_viz_server.py \
      --samples out/sesgo/geometry/Qwen3-0.6B/response_samples.json --port 8002
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves the
# same way the other sesgo run-by-path scripts do (this file lives two levels
# deep under the repo root: sesgo/geometry/<here>).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.datasets.sesgo_eval import GeometrySample  # noqa: E402

from sesgo.geometry.geometry_color_axes import COLOR_AXES, SCAFFOLD_AXIS_KEY  # noqa: E402
from sesgo.geometry.geometry_model_registry import (  # noqa: E402
    GeometryModel,
    default_model_name,
    discover_models,
)

# ── State loaded once at startup ──────────────────────────────────────────────
# Populated by build_app() so the import side stays free of disk I/O. ``models``
# maps model_name -> GeometryModel (lazy blobs); ``default`` is the initially
# selected model the frontend boots into.
_STATE: dict = {
    "models": {},  # model_name -> GeometryModel
    "default": "",  # initially selected model_name
}


def _select(model_name: str | None) -> GeometryModel:
    """Resolve the ``?model=`` param to a loaded model (404 if unknown)."""
    models: dict[str, GeometryModel] = _STATE["models"]
    name = model_name or _STATE["default"]
    m = models.get(name)
    if m is None:
        raise HTTPException(status_code=404, detail=f"unknown model {name!r}")
    return m


def _enum_value(v):
    """Stringify an enum-or-plain label (SesgoLabel.UNKNOWN -> 'unknown')."""
    return getattr(v, "value", v)


def _iter_projection_rows(projections: dict):
    """Yield every per-sample projection row across all (layer, position) cells.

    The projections blob is ``results[layer][position]["samples"] -> [row, ...]``;
    each row is the flat metadata dict (one value per COLOR_AXES key) plus coords.
    """
    for per_pos in (projections.get("results") or {}).values():
        for block in (per_pos or {}).values():
            yield from (block or {}).get("samples", [])


def _live_axes(projections: dict) -> list[dict]:
    """The COLOR_AXES that carry real variation in this model's projection rows.

    An axis is kept when its value across the projection rows is neither entirely
    null/missing nor single-valued (a dead axis would only clutter the dropdown).
    Returns a flat ``{key, label, continuous}`` dict per surviving axis, preserving
    the registry's display order. ``scaffold_id`` carries None as the no-scaffold
    baseline, which counts as a distinct value (so it is never spuriously dropped).
    """
    seen: dict[str, set] = {a.key: set() for a in COLOR_AXES}
    for row in _iter_projection_rows(projections):
        for key in seen:
            if key in row:
                seen[key].add(row[key])
    out: list[dict] = []
    for axis in COLOR_AXES:
        values = seen[axis.key]
        # Drop only when entirely absent/null or single-valued (no real variation).
        non_null = {v for v in values if v is not None}
        distinct = len(values) if axis.key == SCAFFOLD_AXIS_KEY else len(non_null)
        if distinct < 2:
            continue
        out.append({"key": axis.key, "label": axis.pretty, "continuous": axis.continuous})
    return out


def _sample_detail(s: GeometrySample) -> dict:
    """Flatten one GeometrySample into the detail dict the panel renders.

    Mirrors the projection's flat axes and adds the two readouts. ``non_thinking``
    carries the 3-way prob vector [target, other, unknown] + the greedy answer +
    Shannon entropy; ``thinking`` carries the per-role mean/std + parsed count.
    Both readouts may be None (sample never queried) -> we emit null.
    """
    nt = s.non_thinking
    th = s.thinking
    non_thinking = None
    if nt is not None:
        non_thinking = {
            "predicted": _enum_value(nt.predicted),
            "prob": list(nt.prob),  # [target, other, unknown]
            "greedy_text": nt.greedy_text,
            "entropy": nt.entropy,
        }
    thinking = None
    if th is not None:
        thinking = {
            "predicted": _enum_value(th.predicted),
            "mean": list(th.mean),  # [target, other, unknown]
            "std": list(th.std),
            "sample_size": th.sample_size,
        }
    return {
        "sample_idx": s.sample_idx,
        "question_id": s.question_id,
        "prompt_text": s.prompt_text,
        "scaffold_id": s.scaffold_id,
        "bias_category": s.bias_category,
        "question_polarity": s.question_polarity,
        "language": s.language,
        "gold_label": _enum_value(s.gold_label),
        "non_thinking": non_thinking,
        "thinking": thinking,
    }


# ── App factory ───────────────────────────────────────────────────────────────


def build_app(geometry_root: Path, preferred: str | None = None) -> FastAPI:
    """Discover every analysed model under ``geometry_root`` and serve them all.

    No model needs to be specified: every model under ``out/sesgo/geometry/*/``
    (each needing response_samples.json + analysis/projections.json) is listed and
    switchable via the Model selector. ``preferred`` only picks which one the
    frontend boots INTO (the selector changes it live); it is optional.
    """
    geometry_root = geometry_root.resolve()
    models = discover_models(geometry_root)
    if not models:
        raise FileNotFoundError(
            f"no analysed models found under {geometry_root} — run "
            f"`uv run python sesgo/geometry/analyze_geometry.py <MODEL>/response_samples.json` first."
        )
    _STATE["models"] = models
    _STATE["default"] = default_model_name(models, preferred=preferred)

    app = FastAPI(title="SESGO Geometry Visualizer")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _render_html()

    @app.get("/api/models")
    async def api_models() -> JSONResponse:
        # The selector's options + which one to boot into.
        return JSONResponse({"models": list(_STATE["models"].keys()), "default": _STATE["default"]})

    @app.get("/api/projections")
    async def api_projections(model: str | None = Query(default=None)) -> JSONResponse:
        # Raw projections.json for the selected model — sliced client-side.
        return JSONResponse(_select(model).projections)

    @app.get("/api/sample/{idx}")
    async def api_sample(idx: int, model: str | None = Query(default=None)) -> JSONResponse:
        s = _select(model).by_idx.get(idx)
        if s is None:
            raise HTTPException(status_code=404, detail=f"sample_idx {idx} not found")
        return JSONResponse(_sample_detail(s))

    @app.get("/api/config")
    async def api_config(model: str | None = Query(default=None)) -> JSONResponse:
        m = _select(model)
        proj = m.projections
        params = proj.get("params", {})
        return JSONResponse(
            {
                "model_name": proj.get("model_name", m.name),
                "layers": params.get("layers", list(proj.get("results", {}).keys())),
                "positions": params.get("positions", []),
                # EVERY colour-by axis from the shared registry that has real
                # variation in this model's rows — each {key, label, continuous}.
                # Categorical axes get a discrete legend; continuous ones a colorbar.
                "axes": _live_axes(proj),
                "n_samples": len(m.by_idx),
            }
        )

    return app


# ── HTML page builder ─────────────────────────────────────────────────────────


def _render_html() -> str:
    """Build the single-page app as one HTML string (Plotly.js from CDN).

    The shell is model-agnostic: every header chip + control is filled by JS
    after it fetches the selected model's config/projections, so switching
    models only re-runs that client-side fill (no server-rendered model facts).
    """
    title = "SESGO Geometry — PCA of activations"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{_CSS}</style>
</head>
<body>
<header class="app-header">
  <div class="header-left">
    <div class="logo-dot"></div>
    <div>
      <h1>SESGO Geometry <span class="accent">— PCA of activations</span></h1>
      <div class="subtitle">How scaffolds &amp; attributes reshape the residual stream</div>
    </div>
  </div>
  <div class="header-chips">
    <div class="chip"><span class="chip-k">model</span><span class="chip-v" id="chip-model">—</span></div>
    <div class="chip"><span class="chip-k">samples</span><span class="chip-v" id="chip-samples">—</span></div>
    <div class="chip"><span class="chip-k">layers</span><span class="chip-v" id="chip-layers">—</span></div>
    <div class="chip"><span class="chip-k">positions</span><span class="chip-v" id="chip-positions">—</span></div>
  </div>
</header>

<div class="controls">
  <div class="ctrl-group">
    <label>Model</label>
    <select id="sel-model"></select>
  </div>
  <div class="ctrl-group">
    <label>Layer</label>
    <select id="sel-layer"></select>
  </div>
  <div class="ctrl-group">
    <label>Position</label>
    <select id="sel-position"></select>
  </div>
  <div class="ctrl-group">
    <label>Color by</label>
    <select id="sel-color"></select>
  </div>
  <div class="ctrl-group">
    <label>View</label>
    <div class="pills" id="view-toggle">
      <button class="pill active" data-view="2d">2D</button>
      <button class="pill" data-view="3d">3D</button>
    </div>
  </div>
  <div class="ctrl-group grow"></div>
  <div class="ctrl-group">
    <label>&nbsp;</label>
    <button class="pill ghost" id="btn-reset">Reset view</button>
  </div>
</div>

<main class="layout">
  <section class="plot-card">
    <div id="plot"></div>
    <div id="plot-empty" class="empty-state hidden">
      <div class="empty-icon">∅</div>
      <div class="empty-title">No projection here</div>
      <div class="empty-sub">This layer/position has too few samples to embed.</div>
    </div>
    <div id="plot-loading" class="loading-state">
      <div class="spinner"></div><div>Loading projections…</div>
    </div>
  </section>

  <aside class="side">
    <div class="card stats-card">
      <div class="card-title">Statistics
        <span class="card-sub" id="stats-where"></span>
      </div>
      <div id="stats-body">
        <div class="muted">Pick a layer &amp; position.</div>
      </div>
    </div>

    <div class="card detail-card">
      <div class="card-title">Sample detail</div>
      <div id="detail-body">
        <div class="detail-hint">Click a point in the scatter to inspect a sample.</div>
      </div>
    </div>
  </aside>
</main>

<div id="toast" class="toast"></div>

<script>
const ROLE_NAMES = ["target", "other", "unknown"];
{_JS}
</script>
</body>
</html>"""


# ── Static CSS (cohesive dark-glass palette) ──────────────────────────────────

_CSS = r"""
:root {
  --bg:#0b1020; --bg2:#0f1730; --panel:rgba(22,30,55,.72); --panel-solid:#161e37;
  --border:rgba(120,140,200,.18); --border2:rgba(120,140,200,.32);
  --txt:#e8edff; --muted:#94a0c4; --faint:#6b7699;
  --accent:#7aa2ff; --accent2:#9b7dff; --good:#48d597; --warn:#ffb454; --bad:#ff6b8a;
  --shadow:0 10px 40px rgba(0,0,0,.45);
  --radius:16px;
}
* { box-sizing:border-box; }
html,body { margin:0; height:100%; }
body {
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
  color:var(--txt);
  background:
    radial-gradient(1200px 600px at 12% -10%, rgba(122,162,255,.16), transparent 60%),
    radial-gradient(1000px 500px at 100% 0%, rgba(155,125,255,.14), transparent 55%),
    linear-gradient(160deg, var(--bg), var(--bg2));
  background-attachment:fixed;
  min-height:100vh;
}
.accent { color:var(--accent); font-weight:500; -webkit-text-fill-color:initial; }

/* header */
.app-header {
  display:flex; align-items:center; justify-content:space-between; gap:18px;
  padding:18px 26px; border-bottom:1px solid var(--border);
  background:linear-gradient(180deg, rgba(20,28,52,.85), rgba(20,28,52,.45));
  backdrop-filter:blur(14px); position:sticky; top:0; z-index:20;
}
.header-left { display:flex; align-items:center; gap:14px; }
.logo-dot {
  width:36px; height:36px; border-radius:11px;
  background:conic-gradient(from 200deg, var(--accent), var(--accent2), var(--good), var(--accent));
  box-shadow:0 6px 20px rgba(122,162,255,.45); flex:none;
}
.app-header h1 { font-size:19px; margin:0; letter-spacing:.2px; font-weight:650; }
.subtitle { font-size:12.5px; color:var(--muted); margin-top:2px; }
.header-chips { display:flex; gap:10px; flex-wrap:wrap; }
.chip {
  display:flex; flex-direction:column; padding:6px 12px; border-radius:11px;
  background:var(--panel); border:1px solid var(--border); min-width:62px;
}
.chip-k { font-size:9.5px; text-transform:uppercase; letter-spacing:.8px; color:var(--faint); }
.chip-v { font-size:13px; font-weight:600; margin-top:1px; }

/* controls */
.controls {
  display:flex; align-items:flex-end; gap:16px; padding:14px 26px; flex-wrap:wrap;
  border-bottom:1px solid var(--border); background:rgba(13,19,40,.5); backdrop-filter:blur(8px);
  position:sticky; top:73px; z-index:15;
}
.ctrl-group { display:flex; flex-direction:column; gap:5px; }
.ctrl-group.grow { flex:1 1 auto; }
.ctrl-group label { font-size:10px; text-transform:uppercase; letter-spacing:.9px; color:var(--faint); }
select {
  appearance:none; -webkit-appearance:none;
  background:var(--panel-solid) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'><path d='M3 4.5L6 8l3-3.5' stroke='%2394a0c4' stroke-width='1.4' fill='none' stroke-linecap='round'/></svg>") no-repeat right 10px center;
  color:var(--txt); border:1px solid var(--border2); border-radius:10px;
  padding:9px 30px 9px 12px; font-size:13px; cursor:pointer; min-width:170px;
  transition:border-color .15s, box-shadow .15s;
}
select:hover { border-color:var(--accent); }
select:focus { outline:none; border-color:var(--accent); box-shadow:0 0 0 3px rgba(122,162,255,.18); }
.pills { display:flex; gap:6px; background:var(--panel-solid); padding:3px; border-radius:11px; border:1px solid var(--border2); }
.pill {
  border:none; background:transparent; color:var(--muted); padding:7px 16px; border-radius:8px;
  font-size:13px; font-weight:600; cursor:pointer; transition:all .15s;
}
.pill:hover { color:var(--txt); }
.pill.active { background:linear-gradient(135deg, var(--accent), var(--accent2)); color:#fff; box-shadow:0 4px 14px rgba(122,162,255,.4); }
.pill.ghost {
  background:var(--panel-solid); border:1px solid var(--border2); color:var(--muted); padding:9px 16px; border-radius:10px;
}
.pill.ghost:hover { color:var(--txt); border-color:var(--accent); }

/* layout */
.layout {
  display:grid; grid-template-columns:1fr 380px; gap:18px; padding:18px 26px 32px;
  align-items:start;
}
@media (max-width:1100px){ .layout { grid-template-columns:1fr; } }

.plot-card {
  position:relative; background:var(--panel); border:1px solid var(--border);
  border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden;
  min-height:640px;
}
#plot { width:100%; height:640px; }

.empty-state, .loading-state {
  position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:10px; background:rgba(13,19,40,.65); backdrop-filter:blur(3px); color:var(--muted);
}
.empty-state.hidden, .loading-state.hidden { display:none; }
.empty-icon { font-size:48px; opacity:.5; }
.empty-title { font-size:16px; font-weight:650; color:var(--txt); }
.empty-sub { font-size:12.5px; }
.spinner {
  width:34px; height:34px; border-radius:50%;
  border:3px solid rgba(122,162,255,.2); border-top-color:var(--accent);
  animation:spin .8s linear infinite;
}
@keyframes spin { to { transform:rotate(360deg); } }

/* side panels */
.side { display:flex; flex-direction:column; gap:18px; position:sticky; top:152px; }
.card {
  background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
  box-shadow:var(--shadow); padding:16px 18px; backdrop-filter:blur(10px);
}
.card-title {
  font-size:13px; font-weight:700; letter-spacing:.3px; margin-bottom:12px;
  display:flex; align-items:baseline; gap:8px; text-transform:uppercase; color:var(--txt);
}
.card-sub { font-size:11px; font-weight:500; color:var(--muted); text-transform:none; letter-spacing:0; }
.muted { color:var(--muted); font-size:12.5px; }

/* stats */
.stat-row { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid rgba(120,140,200,.08); }
.stat-row:last-child { border-bottom:none; }
.stat-k { font-size:12px; color:var(--muted); }
.stat-v { font-size:13px; font-weight:650; font-variant-numeric:tabular-nums; }
.bar-wrap { margin:8px 0 12px; }
.bar-label { display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-bottom:4px; }
.bar-track { height:9px; background:rgba(120,140,200,.14); border-radius:6px; overflow:hidden; }
.bar-fill { height:100%; border-radius:6px; background:linear-gradient(90deg, var(--accent), var(--accent2)); transition:width .4s ease; }
.bar-fill.cum { background:linear-gradient(90deg, var(--good), var(--accent)); }

.mini-h { font-size:10.5px; text-transform:uppercase; letter-spacing:.8px; color:var(--faint); margin:14px 0 6px; }
table.shifts { width:100%; border-collapse:collapse; font-size:11.5px; }
table.shifts td { padding:4px 0; border-bottom:1px solid rgba(120,140,200,.08); }
table.shifts td.name { color:var(--muted); max-width:170px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
table.shifts td.val { text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }
.shift-mini { display:inline-block; height:6px; border-radius:4px; background:linear-gradient(90deg,var(--accent2),var(--bad)); vertical-align:middle; margin-right:6px; }

.axis-rank { display:flex; flex-direction:column; gap:6px; margin-top:4px; }
.axis-item {
  display:flex; align-items:center; justify-content:space-between; padding:8px 11px; border-radius:10px;
  background:rgba(120,140,200,.07); border:1px solid transparent; font-size:12px;
}
.axis-item.best { border-color:var(--good); background:rgba(72,213,151,.12); box-shadow:0 0 0 1px rgba(72,213,151,.18) inset; }
.axis-item .a-name { font-weight:600; }
.axis-item.best .a-name::after { content:" ★"; color:var(--good); }
.axis-item .a-val { font-variant-numeric:tabular-nums; color:var(--muted); }

/* detail */
.detail-hint { color:var(--faint); font-size:12.5px; line-height:1.5; }
.detail-card { max-height:none; }
.kv { display:grid; grid-template-columns:auto 1fr; gap:5px 12px; font-size:12px; margin:4px 0 12px; }
.kv .k { color:var(--faint); text-transform:uppercase; font-size:10px; letter-spacing:.6px; align-self:center; }
.kv .v { font-weight:600; }
.tag {
  display:inline-block; padding:2px 9px; border-radius:999px; font-size:11px; font-weight:600;
  background:rgba(122,162,255,.16); color:var(--accent); border:1px solid rgba(122,162,255,.3);
}
.tag.muted { background:rgba(120,140,200,.1); color:var(--muted); border-color:var(--border2); }
.prompt-box {
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px; line-height:1.55;
  background:#0a0f22; border:1px solid var(--border); border-radius:10px; padding:11px 13px;
  max-height:200px; overflow:auto; white-space:pre-wrap; color:#cdd6f4; margin-bottom:12px;
}
.role-bars { margin:6px 0 12px; }
.role-row { display:flex; align-items:center; gap:9px; margin:5px 0; }
.role-name { width:54px; font-size:11px; color:var(--muted); text-transform:capitalize; }
.role-track { flex:1; height:8px; background:rgba(120,140,200,.14); border-radius:5px; overflow:hidden; }
.role-fill { height:100%; border-radius:5px; }
.role-fill.target { background:linear-gradient(90deg,var(--bad),#ff9eb3); }
.role-fill.other { background:linear-gradient(90deg,var(--warn),#ffd28a); }
.role-fill.unknown { background:linear-gradient(90deg,var(--good),#86f0c2); }
.role-val { width:48px; text-align:right; font-size:11px; font-variant-numeric:tabular-nums; color:var(--muted); }
.greedy-box {
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11px; line-height:1.5;
  background:rgba(120,140,200,.07); border-left:3px solid var(--accent2); border-radius:0 8px 8px 0;
  padding:8px 11px; color:#cdd6f4; margin:4px 0 10px; max-height:120px; overflow:auto; white-space:pre-wrap;
}
.detail-section-h { font-size:10.5px; text-transform:uppercase; letter-spacing:.8px; color:var(--faint); margin:14px 0 6px; display:flex; align-items:center; gap:8px; }
.detail-section-h::after { content:""; flex:1; height:1px; background:var(--border); }
.pred-line { font-size:12px; margin-bottom:8px; }
.pred-line b { color:var(--txt); }

.toast {
  position:fixed; bottom:24px; left:50%; transform:translateX(-50%) translateY(20px);
  background:var(--panel-solid); border:1px solid var(--border2); color:var(--txt);
  padding:11px 18px; border-radius:12px; box-shadow:var(--shadow); font-size:13px;
  opacity:0; pointer-events:none; transition:all .25s; z-index:50;
}
.toast.show { opacity:1; transform:translateX(-50%) translateY(0); }

::-webkit-scrollbar { width:9px; height:9px; }
::-webkit-scrollbar-thumb { background:rgba(120,140,200,.3); border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:rgba(120,140,200,.5); }
::-webkit-scrollbar-track { background:transparent; }
"""


# ── Static JS (client-side slicing + Plotly rendering) ────────────────────────

_JS = r"""
// ── Categorical palette (cohesive, repeated cyclically) ──────────────────────
// Okabe-Ito colorblind-safe qualitative palette, ordered so the FIRST colors are
// maximally distinct -- the common 2-class case (e.g. scaffold vs none) must NOT
// collide, so sky-blue vs orange rather than two near-identical blues.
const PALETTE = ["#56b4e9","#e69f00","#009e73","#cc79a7","#f0e442","#d55e00",
                 "#0072b2","#bcbd22","#e377c2","#17becf","#aec7e8","#98df8a"];
const BASELINE = "(baseline)";
const SCAFFOLD_KEY = "scaffold_id";  // None == no-scaffold baseline (special-cased)

let PROJ = null;        // selected model's full projections.json
let CFG = null;         // selected model's config (layers/positions/axes/...)
let MODEL = null;       // selected model name (sent as ?model= on every fetch)
// AXES: the current model's colour-by axes (each {key, label, continuous}),
// straight from /api/config; AXIS_BY_KEY maps key -> that axis for O(1) lookup.
let AXES = [];
let AXIS_BY_KEY = {};
const els = {};
let state = { layer:null, position:null, color:"scaffold_id", view:"2d" };

function $(id){ return document.getElementById(id); }
function toast(msg){
  const t = $("toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(()=>t.classList.remove("show"), 2200);
}
// Every data endpoint is scoped to the selected model via ?model=.
function api(path){ return path + "?model=" + encodeURIComponent(MODEL); }

// ── Boot ─────────────────────────────────────────────────────────────────────
// Wire the static controls ONCE, fetch the model list, then load the default
// model. Switching models only re-runs loadModel (no re-binding of controls).
async function boot(){
  els.model = $("sel-model"); els.layer = $("sel-layer");
  els.position = $("sel-position"); els.color = $("sel-color");
  let info;
  try { info = await fetch("/api/models").then(r=>r.json()); }
  catch(e){ $("plot-loading").innerHTML = "<div class='empty-title'>Failed to load models</div>"; return; }

  fillSelect(els.model, info.models);
  // The COLOR BY dropdown is filled per-model in loadModel (axes are model-specific).
  MODEL = info.default;
  els.model.value = MODEL;

  els.model.onchange = ()=> loadModel(els.model.value);
  els.layer.onchange = ()=>{ state.layer = els.layer.value; state.position = firstNonEmptyPosition() || state.position; els.position.value = state.position; render(); };
  els.position.onchange = ()=>{ state.position = els.position.value; render(); };
  els.color.onchange = ()=>{ state.color = els.color.value; render(); };
  document.querySelectorAll("#view-toggle .pill").forEach(b=>{
    b.onclick = ()=>{
      document.querySelectorAll("#view-toggle .pill").forEach(x=>x.classList.remove("active"));
      b.classList.add("active"); state.view = b.dataset.view; render();
    };
  });
  $("btn-reset").onclick = ()=>{ render(); toast("View reset"); };

  await loadModel(MODEL);
}

// Fetch + render one model's geometry: config + projections, repopulate the
// layer/position selectors (they differ per model), refresh header chips.
async function loadModel(name){
  MODEL = name;
  $("plot-loading").classList.remove("hidden");
  $("plot-loading").innerHTML = "<div class='spinner'></div><div>Loading "+name+"…</div>";
  $("detail-body").innerHTML = "<div class='detail-hint'>Click a point in the scatter to inspect a sample.</div>";
  try {
    [CFG, PROJ] = await Promise.all([
      fetch(api("/api/config")).then(r=>r.json()),
      fetch(api("/api/projections")).then(r=>r.json()),
    ]);
  } catch(e){
    $("plot-loading").innerHTML = "<div class='empty-title'>Failed to load "+name+"</div>"; return;
  }
  // The colour-by axes are model-specific; rebuild the dropdown from CFG.axes
  // (each {key, label, continuous}) using the human label, keyed by axis key.
  AXES = CFG.axes || [];
  AXIS_BY_KEY = {};
  AXES.forEach(a => { AXIS_BY_KEY[a.key] = a; });
  const prevColor = state.color;
  fillSelect(els.color, AXES.map(a=>a.key), prettyAxis);
  // Keep the prior colour-by if this model also has it, else default to scaffold.
  state.color = AXIS_BY_KEY[prevColor] ? prevColor
              : (AXIS_BY_KEY[SCAFFOLD_KEY] ? SCAFFOLD_KEY : (AXES[0] && AXES[0].key));
  els.color.value = state.color;

  fillSelect(els.layer, CFG.layers);
  fillSelect(els.position, CFG.positions);
  state.layer = CFG.layers[0];
  state.position = firstNonEmptyPosition() || CFG.positions[0];
  els.layer.value = state.layer;
  els.position.value = state.position;
  fillChips();
  $("plot-loading").classList.add("hidden");
  render();
}

// Header chips reflect the currently loaded model.
function fillChips(){
  $("chip-model").textContent = CFG.model_name;
  $("chip-samples").textContent = CFG.n_samples;
  $("chip-layers").textContent = (CFG.layers||[]).length;
  $("chip-positions").textContent = (CFG.positions||[]).length;
}

function fillSelect(sel, items, label){
  sel.innerHTML = "";
  (items||[]).forEach(it=>{
    const o = document.createElement("option");
    o.value = it; o.textContent = label ? label(it) : it; sel.appendChild(o);
  });
}
// Human label for an axis KEY, from the current model's axis registry (CFG.axes).
function prettyAxis(a){
  const ax = AXIS_BY_KEY[a];
  return ax ? ax.label : a;
}
// Is the currently-selected colour-by axis a continuous scalar (colorbar) vs
// categorical (discrete legend)? Drives the render branch.
function isContinuous(key){ return !!(AXIS_BY_KEY[key] && AXIS_BY_KEY[key].continuous); }
function firstNonEmptyPosition(){
  const r = (PROJ.results||{})[state.layer || CFG.layers[0]] || {};
  return CFG.positions.find(p => r[p] && r[p].samples && r[p].samples.length);
}

// ── Data access ──────────────────────────────────────────────────────────────
function currentBlock(){
  const lr = (PROJ.results||{})[state.layer];
  if(!lr) return null;
  return lr[state.position] || null;
}
function axisLabel(s, axis){
  if(axis === SCAFFOLD_KEY) return s.scaffold_id == null ? BASELINE : String(s.scaffold_id);
  return String(s[axis]);
}
function colorFor(i){ return PALETTE[i % PALETTE.length]; }

// ── Render ───────────────────────────────────────────────────────────────────
function render(){
  const block = currentBlock();
  const empty = $("plot-empty");
  if(!block || !block.samples || !block.samples.length){
    Plotly.purge($("plot")); empty.classList.remove("hidden"); renderStats(null); return;
  }
  empty.classList.add("hidden");
  state.view === "3d" ? render3d(block) : render2d(block);
  renderStats(block);
}

// Group samples into one trace per category so the legend toggles categories.
function groupTraces(block){
  const groups = new Map();
  block.samples.forEach(s=>{
    const lab = axisLabel(s, state.color);
    if(!groups.has(lab)) groups.set(lab, []);
    groups.get(lab).push(s);
  });
  // Stable order: baseline first (if present), then alphabetical.
  const labels = [...groups.keys()].sort((a,b)=>{
    if(a===BASELINE) return -1; if(b===BASELINE) return 1; return a.localeCompare(b);
  });
  return { groups, labels };
}

function hoverText(s){
  // Lead with the axis currently coloured by (its value for this point), then
  // the always-shown core attributes.
  const cv = state.color === SCAFFOLD_KEY
    ? (s.scaffold_id==null?BASELINE:s.scaffold_id) : s[state.color];
  return `idx ${s.sample_idx}<br>${prettyAxis(state.color)}: ${cv}`
       + `<br>scaffold: ${s.scaffold_id==null?BASELINE:s.scaffold_id}`
       + `<br>bias: ${s.bias_category}<br>polarity: ${s.question_polarity}`
       + `<br>lang: ${s.language}<br>gold: ${s.gold_label}<extra></extra>`;
}

function render2d(block){
  const traces = isContinuous(state.color)
    ? continuousTraces2d(block)
    : categoricalTraces2d(block);
  Plotly.react($("plot"), traces, layout2d(block), {responsive:true, displaylogo:false,
    modeBarButtonsToRemove:["lasso2d","select2d"]});
  bindClick();
}

// Categorical 2D: one trace per category (discrete legend toggles categories),
// plus the scaffold-centroid rings when colouring by scaffold.
function categoricalTraces2d(block){
  const { groups, labels } = groupTraces(block);
  const traces = labels.map((lab,i)=>{
    const ss = groups.get(lab);
    return {
      type:"scattergl", mode:"markers", name:lab,
      x:ss.map(s=>s.coord2d[0]), y:ss.map(s=>s.coord2d[1]),
      customdata:ss.map(s=>s.sample_idx),
      text:ss.map(hoverText), hovertemplate:"%{text}",
      marker:{ size:8, color:colorFor(i), line:{width:.8, color:"rgba(255,255,255,.35)"}, opacity:.9 },
    };
  });
  if(state.color === SCAFFOLD_KEY){
    const cents = ((block.scaffold_stats||{}).centroids)||{};
    const cl = Object.keys(cents);
    if(cl.length){
      traces.push({
        type:"scattergl", mode:"markers", name:"▣ centroids",
        x:cl.map(k=>cents[k].coord2d[0]), y:cl.map(k=>cents[k].coord2d[1]),
        text:cl.map(k=>`centroid: ${k}<br>n=${cents[k].n}<extra></extra>`), hovertemplate:"%{text}",
        marker:{ size:17, color:"rgba(0,0,0,0)", line:{width:2.4, color:"#ffffff"}, symbol:"circle" },
        hoverlabel:{bgcolor:"#161e37"},
      });
    }
  }
  return traces;
}

// Continuous 2D: a SINGLE trace whose marker.color is the scalar, a perceptual
// Viridis colorscale, and a visible colorbar (no per-category legend).
function continuousTraces2d(block){
  const ss = block.samples;
  return [{
    type:"scattergl", mode:"markers", name:prettyAxis(state.color),
    x:ss.map(s=>s.coord2d[0]), y:ss.map(s=>s.coord2d[1]),
    customdata:ss.map(s=>s.sample_idx),
    text:ss.map(hoverText), hovertemplate:"%{text}",
    marker:continuousMarker(ss, 8),
  }];
}

// Shared continuous marker: Viridis colorscale over the scalar + a labelled colorbar.
function continuousMarker(ss, size){
  return {
    size, opacity:.92, line:{width:.5, color:"rgba(255,255,255,.25)"},
    color:ss.map(s=>s[state.color]), colorscale:"Viridis", showscale:true,
    colorbar:{ title:{text:prettyAxis(state.color), font:{size:11, color:"#94a0c4"}},
               tickfont:{size:10, color:"#94a0c4"}, outlinewidth:0, thickness:14, len:.85 },
  };
}

function render3d(block){
  const traces = isContinuous(state.color)
    ? continuousTraces3d(block)
    : categoricalTraces3d(block);
  Plotly.react($("plot"), traces, layout3d(block), {responsive:true, displaylogo:false});
  bindClick();
}

// Categorical 3D: one trace per category + scaffold-centroid rings.
function categoricalTraces3d(block){
  const { groups, labels } = groupTraces(block);
  const traces = labels.map((lab,i)=>{
    const ss = groups.get(lab);
    return {
      type:"scatter3d", mode:"markers", name:lab,
      x:ss.map(s=>s.coord3d[0]), y:ss.map(s=>s.coord3d[1]), z:ss.map(s=>s.coord3d[2]),
      customdata:ss.map(s=>s.sample_idx),
      text:ss.map(hoverText), hovertemplate:"%{text}",
      marker:{ size:4.5, color:colorFor(i), opacity:.88, line:{width:.5,color:"rgba(255,255,255,.25)"} },
    };
  });
  if(state.color === SCAFFOLD_KEY){
    const cents = ((block.scaffold_stats||{}).centroids)||{};
    const cl = Object.keys(cents);
    if(cl.length){
      traces.push({
        type:"scatter3d", mode:"markers", name:"▣ centroids",
        x:cl.map(k=>cents[k].coord3d[0]), y:cl.map(k=>cents[k].coord3d[1]), z:cl.map(k=>cents[k].coord3d[2]),
        text:cl.map(k=>`centroid: ${k}<br>n=${cents[k].n}<extra></extra>`), hovertemplate:"%{text}",
        marker:{ size:9, color:"rgba(0,0,0,0)", line:{width:3,color:"#ffffff"}, symbol:"circle" },
      });
    }
  }
  return traces;
}

// Continuous 3D: a single Viridis-coloured trace with a colorbar.
function continuousTraces3d(block){
  const ss = block.samples;
  return [{
    type:"scatter3d", mode:"markers", name:prettyAxis(state.color),
    x:ss.map(s=>s.coord3d[0]), y:ss.map(s=>s.coord3d[1]), z:ss.map(s=>s.coord3d[2]),
    customdata:ss.map(s=>s.sample_idx),
    text:ss.map(hoverText), hovertemplate:"%{text}",
    marker:continuousMarker(ss, 4.5),
  }];
}

function baseLayout(block){
  const evr = block.explained_variance_ratio||[];
  return {
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:{color:"#cdd6f4", family:"Inter,-apple-system,sans-serif", size:12},
    margin:{l:50,r:16,t:14,b:44}, height:640,
    legend:{bgcolor:"rgba(13,19,40,.6)", bordercolor:"rgba(120,140,200,.25)", borderwidth:1,
            font:{size:11}, orientation:"v", x:1.01, y:1, itemsizing:"constant"},
    hovermode:"closest", _evr:evr,
  };
}
function ax(t){ return {title:{text:t,font:{size:11,color:"#94a0c4"}}, gridcolor:"rgba(120,140,200,.12)",
  zerolinecolor:"rgba(120,140,200,.25)", color:"#94a0c4"}; }
function pcLabel(evr,i){ const v = evr[i]; return `PC${i+1}` + (v!=null?` (${(v*100).toFixed(1)}%)`:""); }
function layout2d(block){
  const L = baseLayout(block);
  L.xaxis = ax(pcLabel(L._evr,0)); L.yaxis = ax(pcLabel(L._evr,1)); return L;
}
function layout3d(block){
  const L = baseLayout(block);
  L.scene = { xaxis:ax(pcLabel(L._evr,0)), yaxis:ax(pcLabel(L._evr,1)), zaxis:ax(pcLabel(L._evr,2)),
              bgcolor:"rgba(0,0,0,0)" };
  return L;
}

// ── Click -> detail ──────────────────────────────────────────────────────────
function bindClick(){
  const p = $("plot");
  p.removeAllListeners && p.removeAllListeners("plotly_click");
  p.on("plotly_click", ev=>{
    const pt = ev.points && ev.points[0];
    if(!pt || pt.customdata==null) return;
    loadDetail(pt.customdata);
  });
}

async function loadDetail(idx){
  const body = $("detail-body");
  body.innerHTML = "<div class='detail-hint'>Loading sample "+idx+"…</div>";
  let d;
  try { d = await fetch(api("/api/sample/"+idx)).then(r=>{ if(!r.ok) throw 0; return r.json(); }); }
  catch(e){ body.innerHTML = "<div class='detail-hint'>Could not load sample "+idx+".</div>"; return; }
  body.innerHTML = renderDetail(d);
}

function roleBars(vec){
  if(!vec) return "";
  return "<div class='role-bars'>" + ROLE_NAMES.map((r,i)=>{
    const v = vec[i]||0;
    return `<div class='role-row'><div class='role-name'>${r}</div>`
         + `<div class='role-track'><div class='role-fill ${r}' style='width:${(v*100).toFixed(1)}%'></div></div>`
         + `<div class='role-val'>${v.toFixed(3)}</div></div>`;
  }).join("") + "</div>";
}

function esc(s){ return String(s==null?"":s).replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function renderDetail(d){
  const scaf = d.scaffold_id==null ? `<span class='tag muted'>${BASELINE}</span>` : `<span class='tag'>${esc(d.scaffold_id)}</span>`;
  let html = "";
  html += `<div class='kv'>`
        + `<div class='k'>idx</div><div class='v'>${d.sample_idx}</div>`
        + `<div class='k'>scaffold</div><div class='v'>${scaf}</div>`
        + `<div class='k'>bias</div><div class='v'>${esc(d.bias_category)}</div>`
        + `<div class='k'>polarity</div><div class='v'>${esc(d.question_polarity)}</div>`
        + `<div class='k'>language</div><div class='v'>${esc(d.language)}</div>`
        + `<div class='k'>gold</div><div class='v'><span class='tag muted'>${esc(d.gold_label)}</span></div>`
        + `</div>`;
  html += `<div class='detail-section-h'>prompt</div>`;
  html += `<div class='prompt-box'>${esc(d.prompt_text)}</div>`;

  if(d.non_thinking){
    const nt = d.non_thinking;
    html += `<div class='detail-section-h'>non-thinking</div>`;
    html += `<div class='pred-line'>predicted <b>${esc(nt.predicted)}</b>`
          + (nt.entropy!=null?` · entropy <b>${nt.entropy.toFixed(3)}</b>`:"") + `</div>`;
    html += roleBars(nt.prob);
    if(nt.greedy_text){
      html += `<div class='mini-h'>greedy answer</div><div class='greedy-box'>${esc(nt.greedy_text)}</div>`;
    }
  }
  if(d.thinking){
    const th = d.thinking;
    html += `<div class='detail-section-h'>thinking</div>`;
    if(th.sample_size>0){
      html += `<div class='pred-line'>predicted <b>${esc(th.predicted)}</b> · ${th.sample_size} draw(s)</div>`;
      html += roleBars(th.mean);
      html += `<div class='mini-h'>per-role std</div>` + roleBars(th.std);
    } else {
      html += `<div class='detail-hint'>No parsed thinking draws for this sample.</div>`;
    }
  }
  return html;
}

// ── Stats panel ──────────────────────────────────────────────────────────────
function fmt(v,d=3){ return v==null ? "n/a" : Number(v).toFixed(d); }

function renderStats(block){
  $("stats-where").textContent = block ? `${state.layer} · ${state.position}` : "";
  const body = $("stats-body");
  if(!block){ body.innerHTML = "<div class='muted'>No data for this layer/position.</div>"; return; }
  const evr = block.explained_variance_ratio||[];
  const pc1 = evr[0]||0; const cum3 = (evr[0]||0)+(evr[1]||0)+(evr[2]||0);
  const ss = block.scaffold_stats||{};
  let html = "";

  // Explained variance bars.
  html += `<div class='bar-wrap'>`
        + `<div class='bar-label'><span>PC1 variance</span><span>${(pc1*100).toFixed(1)}%</span></div>`
        + `<div class='bar-track'><div class='bar-fill' style='width:${(pc1*100).toFixed(1)}%'></div></div></div>`;
  html += `<div class='bar-wrap'>`
        + `<div class='bar-label'><span>Cumulative PC1–3</span><span>${(cum3*100).toFixed(1)}%</span></div>`
        + `<div class='bar-track'><div class='bar-fill cum' style='width:${(cum3*100).toFixed(1)}%'></div></div></div>`;

  // Scaffold separation scalars.
  html += `<div class='stat-row'><span class='stat-k'>n samples</span><span class='stat-v'>${block.n_samples}</span></div>`;
  html += `<div class='stat-row'><span class='stat-k'>silhouette (scaffold)</span><span class='stat-v'>${fmt(ss.silhouette)}</span></div>`;
  html += `<div class='stat-row'><span class='stat-k'>between / within</span><span class='stat-v'>${fmt(ss.between_within_ratio)}</span></div>`;

  // Per-scaffold shift magnitudes (baseline-relative), bar-scaled.
  const shifts = ss.shifts||{};
  const sk = Object.keys(shifts);
  if(sk.length){
    const maxMag = Math.max(...sk.map(k=>shifts[k].shift_magnitude||0)) || 1;
    html += `<div class='mini-h'>scaffold shift magnitude (vs baseline)</div>`;
    html += `<table class='shifts'>`;
    sk.sort((a,b)=>shifts[b].shift_magnitude - shifts[a].shift_magnitude).forEach(k=>{
      const m = shifts[k].shift_magnitude||0; const w = Math.max(6, (m/maxMag)*90);
      html += `<tr><td class='name' title='${esc(k)}'><span class='shift-mini' style='width:${w}px'></span>${esc(k)}</td>`
            + `<td class='val'>${m.toFixed(2)}</td></tr>`;
    });
    html += `</table>`;
  }

  // Axis separation ranking over EVERY categorical colour-by axis (scaffold
  // silhouette lives in scaffold_stats; the rest in axis_separation) — highlight
  // the axis the geometry separates best.
  const sep = block.axis_separation||{};
  const rows = AXES.filter(a=>!a.continuous).map(a=>{
    const sil = a.key === SCAFFOLD_KEY ? ss.silhouette : (sep[a.key]||{}).silhouette;
    return [a.key, sil];
  }).filter(r=>r[1]!=null);
  if(rows.length){
    rows.sort((a,b)=>b[1]-a[1]);
    const best = rows[0][0];
    html += `<div class='mini-h'>which axis the geometry separates best (silhouette)</div>`;
    html += `<div class='axis-rank'>`;
    rows.forEach(([name,val])=>{
      html += `<div class='axis-item ${name===best?"best":""}'>`
            + `<span class='a-name'>${prettyAxis(name)}</span>`
            + `<span class='a-val'>${fmt(val)}</span></div>`;
    });
    html += `</div>`;
  }
  body.innerHTML = html;
}

window.addEventListener("DOMContentLoaded", boot);
"""


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the geometry visualizer server."""
    parser = argparse.ArgumentParser(
        description="Interactive web visualizer for the SESGO geometry PCA"
    )
    parser.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("out/sesgo/geometry"),
        help="dir whose subdirs are per-model geometry outputs (default "
        "out/sesgo/geometry); every analysed model under it is served + switchable",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="optional: a specific <MODEL>/response_samples.json to boot INTO "
        "(the Model selector changes it live; NOT required)",
    )
    parser.add_argument("--port", type=int, default=8002, help="server port (default 8002)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="bind host")
    return parser.parse_args()


# Module-level `app` so `uvicorn module:app` also works, but the script entrypoint
# below is the supported run-by-path path (no module import needed).
args = parse_args() if __name__ == "__main__" else None
if args is not None:
    # --samples (optional) just picks the boot model; otherwise serve the whole
    # geometry root and let the UI's Model selector choose. No model required.
    if args.samples is not None:
        _root, _pref = args.samples.resolve().parents[1], args.samples.resolve().parent.name
    else:
        _root, _pref = args.geometry_root, None
    app = build_app(_root, _pref)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
