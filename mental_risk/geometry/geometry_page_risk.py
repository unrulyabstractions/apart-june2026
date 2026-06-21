"""The single-page HTML/CSS/JS for the mental_risk geometry visualizer.

Split out of geometry_viz_server_risk.py so the server file stays a thin set of
routes and this file owns the (large but static) frontend blob. The page is a
Plotly scatter of the PCA projection with a risk-specific detail panel
(calibrated non-thinking risk + sampled thinking score cloud), colored by
framing / disorder / language. Mirrors the SESGO geometry page, adapted from the
3-way role distribution to the continuous risk readouts.
"""

from __future__ import annotations

import json

_CSS = r"""
:root{--bg:#0b1020;--bg2:#0f1730;--panel:rgba(22,30,55,.72);--panel-solid:#161e37;
--border:rgba(120,140,200,.18);--border2:rgba(120,140,200,.32);--txt:#e8edff;
--muted:#94a0c4;--faint:#6b7699;--accent:#7aa2ff;--accent2:#9b7dff;--good:#48d597;
--bad:#ff6b8a;--shadow:0 10px 40px rgba(0,0,0,.45);--radius:16px;}
*{box-sizing:border-box;}html,body{margin:0;height:100%;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;color:var(--txt);
background:linear-gradient(160deg,var(--bg),var(--bg2));min-height:100vh;}
.app-header{display:flex;align-items:center;justify-content:space-between;gap:18px;
padding:18px 26px;border-bottom:1px solid var(--border);}
.app-header h1{font-size:19px;margin:0;font-weight:650;}
.subtitle{font-size:12.5px;color:var(--muted);margin-top:2px;}
.header-chips{display:flex;gap:10px;flex-wrap:wrap;}
.chip{display:flex;flex-direction:column;padding:6px 12px;border-radius:11px;
background:var(--panel);border:1px solid var(--border);min-width:62px;}
.chip-k{font-size:9.5px;text-transform:uppercase;letter-spacing:.8px;color:var(--faint);}
.chip-v{font-size:13px;font-weight:600;margin-top:1px;}
.controls{display:flex;align-items:flex-end;gap:16px;padding:14px 26px;flex-wrap:wrap;
border-bottom:1px solid var(--border);}
.ctrl-group{display:flex;flex-direction:column;gap:5px;}
.ctrl-group label{font-size:10px;text-transform:uppercase;letter-spacing:.9px;color:var(--faint);}
select{background:var(--panel-solid);color:var(--txt);border:1px solid var(--border2);
border-radius:10px;padding:9px 12px;font-size:13px;cursor:pointer;min-width:170px;}
.pill{border:none;background:var(--panel-solid);color:var(--muted);padding:9px 16px;
border-radius:10px;font-size:13px;font-weight:600;cursor:pointer;border:1px solid var(--border2);}
.pill.active{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;}
.layout{display:grid;grid-template-columns:1fr 380px;gap:18px;padding:18px 26px 32px;align-items:start;}
@media(max-width:1100px){.layout{grid-template-columns:1fr;}}
.plot-card{position:relative;background:var(--panel);border:1px solid var(--border);
border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden;min-height:640px;}
#plot{width:100%;height:640px;}
.card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);
box-shadow:var(--shadow);padding:16px 18px;margin-bottom:18px;}
.card-title{font-size:13px;font-weight:700;margin-bottom:12px;text-transform:uppercase;}
.muted{color:var(--muted);font-size:12.5px;}
.stat-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(120,140,200,.08);}
.stat-k{font-size:12px;color:var(--muted);}.stat-v{font-size:13px;font-weight:650;}
.prompt-box{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;line-height:1.55;
background:#0a0f22;border:1px solid var(--border);border-radius:10px;padding:11px 13px;
max-height:200px;overflow:auto;white-space:pre-wrap;color:#cdd6f4;margin-bottom:12px;}
.risk-bar-track{height:10px;background:rgba(120,140,200,.14);border-radius:6px;overflow:hidden;margin:4px 0 10px;}
.risk-bar-fill{height:100%;border-radius:6px;background:linear-gradient(90deg,var(--good),var(--bad));}
.tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;
background:rgba(122,162,255,.16);color:var(--accent);border:1px solid rgba(122,162,255,.3);}
"""

_JS = r"""
const PALETTE=["#7aa2ff","#9b7dff","#48d597","#ffb454","#ff6b8a","#43c6e8","#f78fb3","#a0e57a"];
let PROJ=null,CFG=null;const els={};
let state={layer:null,position:null,color:"framing",view:"2d"};
function $(id){return document.getElementById(id);}
async function boot(){
  els.layer=$("sel-layer");els.position=$("sel-position");els.color=$("sel-color");
  [CFG,PROJ]=await Promise.all([fetch("/api/config").then(r=>r.json()),
    fetch("/api/projections").then(r=>r.json())]);
  fill(els.layer,CFG.layers);fill(els.position,CFG.positions);fill(els.color,CFG.axes);
  state.layer=CFG.layers[0];state.position=CFG.positions[0];state.color="framing";
  els.layer.value=state.layer;els.position.value=state.position;els.color.value=state.color;
  els.layer.onchange=()=>{state.layer=els.layer.value;render();};
  els.position.onchange=()=>{state.position=els.position.value;render();};
  els.color.onchange=()=>{state.color=els.color.value;render();};
  document.querySelectorAll("#view-toggle .pill").forEach(b=>{b.onclick=()=>{
    document.querySelectorAll("#view-toggle .pill").forEach(x=>x.classList.remove("active"));
    b.classList.add("active");state.view=b.dataset.view;render();};});
  render();
}
function fill(sel,items){sel.innerHTML="";(items||[]).forEach(it=>{
  const o=document.createElement("option");o.value=it;o.textContent=it;sel.appendChild(o);});}
function block(){return ((PROJ.results||{})[state.layer]||{})[state.position]||null;}
function groups(b){const g=new Map();b.samples.forEach(s=>{const l=String(s[state.color]);
  if(!g.has(l))g.set(l,[]);g.get(l).push(s);});return g;}
function render(){const b=block();if(!b||!b.samples||!b.samples.length){Plotly.purge($("plot"));
  renderStats(null);return;}state.view==="3d"?r3(b):r2(b);renderStats(b);}
function hov(s){return `idx ${s.sample_idx}<br>framing: ${s.framing}<br>disorder: ${s.disorder}`
  +`<br>lang: ${s.language}<br>gold: ${s.gold_risk}<extra></extra>`;}
function r2(b){const g=groups(b);const tr=[...g.keys()].sort().map((l,i)=>{const ss=g.get(l);
  return{type:"scattergl",mode:"markers",name:l,x:ss.map(s=>s.coord2d[0]),y:ss.map(s=>s.coord2d[1]),
  customdata:ss.map(s=>s.sample_idx),text:ss.map(hov),hovertemplate:"%{text}",
  marker:{size:8,color:PALETTE[i%PALETTE.length],opacity:.9}};});
  Plotly.react($("plot"),tr,lay(b,2),{responsive:true,displaylogo:false});click();}
function r3(b){const g=groups(b);const tr=[...g.keys()].sort().map((l,i)=>{const ss=g.get(l);
  return{type:"scatter3d",mode:"markers",name:l,x:ss.map(s=>s.coord3d[0]),y:ss.map(s=>s.coord3d[1]),
  z:ss.map(s=>s.coord3d[2]),customdata:ss.map(s=>s.sample_idx),text:ss.map(hov),hovertemplate:"%{text}",
  marker:{size:4.5,color:PALETTE[i%PALETTE.length],opacity:.88}};});
  Plotly.react($("plot"),tr,lay(b,3),{responsive:true,displaylogo:false});click();}
function pc(evr,i){const v=evr[i];return `PC${i+1}`+(v!=null?` (${(v*100).toFixed(1)}%)`:"");}
function lay(b,d){const e=b.explained_variance_ratio||[];const L={paper_bgcolor:"rgba(0,0,0,0)",
  plot_bgcolor:"rgba(0,0,0,0)",font:{color:"#cdd6f4",size:12},margin:{l:50,r:16,t:14,b:44},height:640,
  legend:{font:{size:11},x:1.01,y:1}};
  if(d===3){L.scene={xaxis:{title:pc(e,0)},yaxis:{title:pc(e,1)},zaxis:{title:pc(e,2)}};}
  else{L.xaxis={title:pc(e,0)};L.yaxis={title:pc(e,1)};}return L;}
function click(){const p=$("plot");p.removeAllListeners&&p.removeAllListeners("plotly_click");
  p.on("plotly_click",ev=>{const pt=ev.points&&ev.points[0];if(pt&&pt.customdata!=null)detail(pt.customdata);});}
async function detail(idx){const d=await fetch("/api/sample/"+idx).then(r=>r.json());
  $("detail-body").innerHTML=renderDetail(d);}
function esc(s){return String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function riskBar(v){if(v==null)return "<div class='muted'>n/a</div>";
  return `<div class='risk-bar-track'><div class='risk-bar-fill' style='width:${(v*100).toFixed(1)}%'></div></div>`
   +`<div class='muted'>${v.toFixed(3)}</div>`;}
function renderDetail(d){let h=`<div class='stat-row'><span class='stat-k'>framing</span>`
  +`<span class='stat-v'><span class='tag'>${esc(d.framing)}</span></span></div>`
  +`<div class='stat-row'><span class='stat-k'>disorder</span><span class='stat-v'>${esc(d.disorder)}</span></div>`
  +`<div class='stat-row'><span class='stat-k'>language</span><span class='stat-v'>${esc(d.language)}</span></div>`
  +`<div class='stat-row'><span class='stat-k'>gold risk</span><span class='stat-v'>${d.gold_risk==null?"n/a":d.gold_risk}</span></div>`;
  h+=`<div class='card-title' style='margin-top:14px'>prompt</div><div class='prompt-box'>${esc(d.prompt_text)}</div>`;
  if(d.non_thinking){h+=`<div class='card-title'>non-thinking risk</div>`+riskBar(d.non_thinking.predicted_risk);}
  if(d.thinking){const t=d.thinking;h+=`<div class='card-title'>thinking risk (n=${t.n})</div>`+riskBar(t.mean)
    +`<div class='muted'>std ${t.std==null?"n/a":t.std.toFixed(3)} · entropy ${t.entropy==null?"n/a":t.entropy.toFixed(3)}</div>`;}
  return h;}
function renderStats(b){const body=$("stats-body");if(!b){body.innerHTML="<div class='muted'>No data here.</div>";return;}
  const fs=b.framing_stats||{};let h=`<div class='stat-row'><span class='stat-k'>n samples</span><span class='stat-v'>${b.n_samples}</span></div>`
  +`<div class='stat-row'><span class='stat-k'>silhouette (framing)</span><span class='stat-v'>${fs.silhouette==null?"n/a":fs.silhouette.toFixed(3)}</span></div>`
  +`<div class='stat-row'><span class='stat-k'>anchor framing</span><span class='stat-v'>${esc(fs.anchor)}</span></div>`;
  const sh=fs.shifts||{};Object.keys(sh).sort().forEach(k=>{h+=`<div class='stat-row'><span class='stat-k'>shift ${esc(k)}</span>`
    +`<span class='stat-v'>${sh[k].shift_magnitude.toFixed(2)}</span></div>`;});body.innerHTML=h;}
window.addEventListener("DOMContentLoaded",boot);
"""


def render_html(model_name: str, n_samples: int, params: dict) -> str:
    """Build the full single-page app for the given projection metadata."""
    n_layers = len(params.get("layers", []))
    n_pos = len(params.get("positions", []))
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MentalRisk Geometry — PCA</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{_CSS}</style></head><body>
<header class="app-header"><div><h1>MentalRisk Geometry <span style="color:#7aa2ff">— PCA of activations</span></h1>
<div class="subtitle">How framings reshape the residual stream of a risk judgement</div></div>
<div class="header-chips">
<div class="chip"><span class="chip-k">model</span><span class="chip-v">{model_name}</span></div>
<div class="chip"><span class="chip-k">samples</span><span class="chip-v">{n_samples}</span></div>
<div class="chip"><span class="chip-k">layers</span><span class="chip-v">{n_layers}</span></div>
<div class="chip"><span class="chip-k">positions</span><span class="chip-v">{n_pos}</span></div></div></header>
<div class="controls">
<div class="ctrl-group"><label>Layer</label><select id="sel-layer"></select></div>
<div class="ctrl-group"><label>Position</label><select id="sel-position"></select></div>
<div class="ctrl-group"><label>Color by</label><select id="sel-color"></select></div>
<div class="ctrl-group"><label>View</label><div id="view-toggle">
<button class="pill active" data-view="2d">2D</button><button class="pill" data-view="3d">3D</button></div></div></div>
<main class="layout"><section class="plot-card"><div id="plot"></div></section>
<aside><div class="card"><div class="card-title">Statistics</div><div id="stats-body">
<div class="muted">Pick a layer &amp; position.</div></div></div>
<div class="card"><div class="card-title">Sample detail</div><div id="detail-body">
<div class="muted">Click a point to inspect a sample.</div></div></div></aside></main>
<script>const PARAMS={json.dumps(params)};{_JS}</script></body></html>"""
