"""Self-contained interactive HTML Node-Graph & Full PBR Inspector for `synccraft.texgraph`.

Exports a single offline HTML file showing:
- Full DAG node cards in topological order with inputs, parameters, timings, and pixelated thumbnails
- Full PBR Studio Tabs: PBR Lit GGX Preview, Normal (_n), LabPBR Specular (_s), Roughness, Metallic, Height, AO, Bedrock MER
- Interactive real-time Light Azimuth Slider right inside the browser (re-lights Albedo + Normal + Specular on HTML Canvas!)
- Click-to-inspect zoomed canvas with 3x3 seamless tiling check, live animated flipbook playback & UV Zone Map tab
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import EvalResult, TexGraph
from .pixel_art_ops import encode_png, rgba_float_to_u8


def _thumb_b64(arr: np.ndarray, *, max_dim: int = 160) -> str:
    u8 = arr if (isinstance(arr, np.ndarray) and arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 4) else rgba_float_to_u8(arr)
    h, w = u8.shape[:2]
    im = Image.fromarray(u8, mode="RGBA")
    if max(h, w) > max_dim:
        sc = max_dim / float(max(h, w))
        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), resample=Image.Resampling.NEAREST)
    elif max(h, w) < 96:
        sc = max(1, int(96 // max(1, max(h, w))))
        im = im.resize((w * sc, h * sc), resample=Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _png_b64_capped(arr: np.ndarray, *, max_dim: int = 512, force_full: bool = False) -> str:
    u8 = arr if (isinstance(arr, np.ndarray) and arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 4) else rgba_float_to_u8(arr)
    if not force_full and max(u8.shape[:2]) > max_dim:
        return _thumb_b64(u8, max_dim=max_dim)
    buf = io.BytesIO()
    Image.fromarray(u8, mode="RGBA").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def export_html_inspector(
    graph: TexGraph,
    out_html_path: Path | str,
    *,
    eval_result: EvalResult | None = None,
    flipbook_frames: list[np.ndarray] | None = None,
    uv_zone_map_png: bytes | None = None,
    max_preview_dim: int = 512,
) -> Path:
    """Write a self-contained dark-studio interactive HTML Node-Graph & Full PBR Inspector."""
    res = eval_result or graph.evaluate(phase=0.0, capture_intermediates=True, build_pbr=True)
    if not res.intermediates or res.pbr_maps is None:
        res = graph.evaluate(phase=0.0, capture_intermediates=True, build_pbr=True)

    node_cards: list[dict[str, Any]] = []
    for nid in graph.node_order:
        sp = graph.nodes[nid]
        arr = res.intermediates[nid]
        is_out = (nid == graph.output_node)
        is_em = (nid == graph.emissive_node)
        pbr_role = next((k for k, v in graph.pbr_nodes.items() if v == nid), "")
        node_cards.append(
            {
                "id": nid,
                "op": sp.op,
                "inputs": sp.inputs,
                "params": sp.params,
                "shape": list(arr.shape),
                "time_ms": res.timings_ms.get(nid, 0.0),
                "thumb": _thumb_b64(arr, max_dim=192),
                "full": _png_b64_capped(arr, max_dim=max_preview_dim, force_full=(is_out or is_em)),
                "is_output": is_out,
                "is_emissive": is_em,
                "pbr_role": pbr_role,
            }
        )

    pbr_views: dict[str, str] = {}
    if res.pbr_maps is not None:
        for k in ("albedo", "normal_rgb", "labpbr_n", "labpbr_s", "bedrock_mer", "height_u8", "roughness_u8", "metallic_u8", "ao_u8"):
            if k in res.pbr_maps:
                pbr_views[k] = _png_b64_capped(res.pbr_maps[k], max_dim=max_preview_dim, force_full=(k == "albedo"))
    if res.pbr_preview is not None:
        pbr_views["pbr_lit"] = _png_b64_capped(res.pbr_preview, max_dim=max_preview_dim, force_full=True)
    if res.emissive is not None:
        pbr_views["emissive"] = _png_b64_capped(res.emissive, max_dim=max_preview_dim)

    anim_b64: list[str] = []
    if flipbook_frames:
        for fr in flipbook_frames:
            anim_b64.append(_png_b64_capped(fr, max_dim=max_preview_dim))

    uv_b64 = (
        "data:image/png;base64," + base64.b64encode(uv_zone_map_png).decode("ascii")
        if uv_zone_map_png
        else ""
    )
    recipe_json = json.dumps(graph.to_recipe(), indent=2)
    payload_json = json.dumps(
        {
            "name": graph.name,
            "width": graph.width,
            "height": graph.height,
            "tileable": graph.tileable,
            "seed": graph.seed,
            "total_ms": round(sum(res.timings_ms.values()), 2),
            "nodes": node_cards,
            "pbr": pbr_views,
            "frames": anim_b64,
            "uv_map": uv_b64,
        }
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SyncCraft TexGraph PBR Inspector — {html.escape(graph.name)}</title>
<style>
  :root {{ --bg:#0b0f17; --card:#141a26; --border:#263147; --text:#e8effa; --muted:#8da0bd; --accent:#22d3ee; --accent2:#e84be0; --gold:#f59e0b; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); }}
  header {{ display:flex; align-items:center; justify-content:space-between; padding:12px 20px; background:#101521; border-bottom:1px solid var(--border); }}
  h1 {{ margin:0; font-size:15px; letter-spacing:0.35px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; background:#172436; color:var(--accent); border:1px solid #2b4560; margin-left:6px; }}
  .badge.pbr {{ color:var(--gold); border-color:#5b4218; background:#231b0e; }}
  .layout {{ display:grid; grid-template-columns: 1fr 450px; height:calc(100vh - 50px); }}
  .graph-grid {{ padding:14px; overflow:auto; display:grid; grid-template-columns:repeat(auto-fill, minmax(185px, 1fr)); gap:11px; align-content:start; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:9px; cursor:pointer; transition:border-color 0.14s, transform 0.12s; }}
  .card:hover {{ border-color:var(--accent); transform:translateY(-1px); }}
  .card.active {{ border-color:var(--accent2); box-shadow:0 0 0 1px var(--accent2); }}
  .card.is-out {{ border-color:#22c55e; }}
  .card.is-pbr {{ border-color:var(--gold); }}
  .card img {{ width:100%; aspect-ratio:1/1; object-fit:contain; image-rendering:pixelated; background:repeating-conic-gradient(#1f2433 0% 25%, #161b26 0% 50%) 50% / 16px 16px; border-radius:6px; }}
  .card-title {{ font-size:12px; font-weight:600; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .card-meta {{ font-size:11px; color:var(--muted); margin-top:2px; display:flex; justify-content:space-between; }}
  .sidebar {{ background:#101521; border-left:1px solid var(--border); padding:14px; overflow:auto; display:flex; flex-direction:column; gap:10px; }}
  .preview-box {{ width:100%; aspect-ratio:1/1; background:repeating-conic-gradient(#1f2433 0% 25%, #161b26 0% 50%) 50% / 20px 20px; border:1px solid var(--border); border-radius:8px; display:flex; align-items:center; justify-content:center; overflow:hidden; position:relative; }}
  .preview-box img, .preview-box canvas {{ max-width:100%; max-height:100%; image-rendering:pixelated; }}
  .tile-3x3 {{ display:grid; grid-template-columns:repeat(3,1fr); width:100%; height:100%; }}
  .tile-3x3 img {{ width:100%; height:100%; object-fit:fill; image-rendering:pixelated; }}
  .controls {{ display:flex; gap:6px; flex-wrap:wrap; }}
  button {{ background:#1a2234; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:5px 9px; font-size:11px; cursor:pointer; }}
  button:hover, button.sel {{ border-color:var(--accent); color:var(--accent); }}
  .pbr-bar {{ display:flex; align-items:center; gap:8px; font-size:11px; color:var(--muted); background:#141a26; border:1px solid var(--border); border-radius:6px; padding:6px 10px; }}
  input[type=range] {{ flex:1; accent-color:var(--accent); }}
  pre {{ margin:0; padding:10px; background:#090c12; border:1px solid var(--border); border-radius:8px; font-size:11px; color:#b8c7e0; overflow:auto; max-height:190px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>SyncCraft TexGraph Studio <span class="badge">{html.escape(graph.name)}</span><span class="badge pbr">LabPBR 1.3 + MER</span></h1>
  </div>
  <div id="stats" style="font-size:12px;color:var(--muted);"></div>
</header>
<div class="layout">
  <div class="graph-grid" id="grid"></div>
  <div class="sidebar">
    <div class="controls" id="btnRow">
      <button onclick="setView('pbr_lit')">PBR Lit (GGX)</button>
      <button onclick="setView('pbr_live')">Live Light PBR</button>
      <button onclick="setView('node')">Selected Node</button>
      <button onclick="setView('normal_rgb')">Normal (_n)</button>
      <button onclick="setView('labpbr_s')">LabPBR (_s)</button>
      <button onclick="setView('roughness_u8')">Roughness</button>
      <button onclick="setView('metallic_u8')">Metallic</button>
      <button onclick="setView('height_u8')">Height</button>
      <button onclick="setView('ao_u8')">Baked AO</button>
      <button onclick="setView('tile')">3x3 Tile</button>
      <button id="btnAnim" style="display:none" onclick="setView('anim')">Flipbook</button>
      <button id="btnUV" style="display:none" onclick="setView('uv')">UV Zones</button>
    </div>
    <div class="pbr-bar" id="lightBar" style="display:none;">
      <span>Light Azimuth</span>
      <input type="range" id="azSlider" min="-180" max="180" value="-38" oninput="renderLivePBR()"/>
      <span id="azVal">-38°</span>
    </div>
    <div class="preview-box" id="preview"></div>
    <div id="nodeInfo" style="font-size:12px;color:var(--muted);"></div>
    <pre id="paramsPre"></pre>
    <details>
      <summary style="cursor:pointer;font-size:12px;color:var(--accent);">Declarative .texgraph.json v2 Recipe</summary>
      <pre>{html.escape(recipe_json)}</pre>
    </details>
  </div>
</div>
<script>
const DATA = {payload_json};
let selectedIdx = DATA.nodes.findIndex(n => n.is_output);
if (selectedIdx < 0) selectedIdx = DATA.nodes.length - 1;
let viewMode = DATA.pbr && DATA.pbr.pbr_lit ? 'pbr_lit' : 'node';
let animTimer = null;
let animIdx = 0;
let liveBufs = null;

document.getElementById('stats').textContent =
  `${{DATA.width}}x${{DATA.height}} | ${{DATA.nodes.length}} nodes | Tileable: ${{DATA.tileable}} | Eval: ${{DATA.total_ms}} ms`;
if (DATA.frames.length > 1) document.getElementById('btnAnim').style.display = 'inline-block';
if (DATA.uv_map) document.getElementById('btnUV').style.display = 'inline-block';

function renderGrid() {{
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  DATA.nodes.forEach((n, i) => {{
    const card = document.createElement('div');
    card.className = 'card' + (i === selectedIdx ? ' active' : '') + (n.is_output ? ' is-out' : '') + (n.pbr_role ? ' is-pbr' : '');
    const tag = n.is_output ? 'ALBEDO OUT' : (n.is_emissive ? 'EMISSIVE' : (n.pbr_role ? 'PBR:' + n.pbr_role.toUpperCase() : n.op));
    card.innerHTML = `<img src="${{n.thumb}}"/><div class="card-title">${{n.id}}</div><div class="card-meta"><span>${{tag}}</span><span>${{n.time_ms}}ms</span></div>`;
    card.onclick = () => {{ selectedIdx = i; setView('node'); renderGrid(); }};
    grid.appendChild(card);
  }});
}}

function setView(m) {{
  viewMode = m;
  if (animTimer) {{ clearInterval(animTimer); animTimer = null; }}
  document.getElementById('lightBar').style.display = (m === 'pbr_live') ? 'flex' : 'none';
  renderSidebar();
}}

function loadImageData(src) {{
  return new Promise(res => {{
    const im = new Image();
    im.onload = () => {{
      const c = document.createElement('canvas');
      c.width = im.width; c.height = im.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(im, 0, 0);
      res({{w: im.width, h: im.height, data: ctx.getImageData(0, 0, im.width, im.height).data}});
    }};
    im.src = src;
  }});
}}

async function renderLivePBR() {{
  const box = document.getElementById('preview');
  const az = parseFloat(document.getElementById('azSlider').value);
  document.getElementById('azVal').textContent = az + '°';
  if (!liveBufs && DATA.pbr && DATA.pbr.albedo && DATA.pbr.normal_rgb && DATA.pbr.labpbr_s) {{
    const [alb, nrm, spc] = await Promise.all([
      loadImageData(DATA.pbr.albedo),
      loadImageData(DATA.pbr.normal_rgb),
      loadImageData(DATA.pbr.labpbr_s),
    ]);
    liveBufs = {{alb, nrm, spc}};
  }}
  if (!liveBufs) return;
  const {{alb, nrm, spc}} = liveBufs;
  let canvas = document.getElementById('livePbrCanvas');
  if (!canvas) {{
    box.innerHTML = `<canvas id="livePbrCanvas" width="${{alb.w}}" height="${{alb.h}}" style="width:100%;height:100%;object-fit:contain;"></canvas>`;
    canvas = document.getElementById('livePbrCanvas');
  }}
  const ctx = canvas.getContext('2d');
  const out = ctx.createImageData(alb.w, alb.h);
  const od = out.data, ad = alb.data, nd = nrm.data, sd = spc.data;
  const rad = az * Math.PI / 180.0, el = 50.0 * Math.PI / 180.0;
  const lx = Math.cos(el) * Math.sin(rad), ly = Math.cos(el) * Math.cos(rad), lz = Math.sin(el);
  const hx = lx, hy = ly, hz = lz + 1.0;
  const hl = Math.hypot(hx, hy, hz) || 1.0;
  const hxn = hx / hl, hyn = hy / hl, hzn = hz / hl;
  for (let i = 0; i < ad.length; i += 4) {{
    const a = ad[i+3];
    if (a < 8) {{ od[i+3] = 0; continue; }}
    const ar = Math.pow(ad[i]/255, 2.2), ag = Math.pow(ad[i+1]/255, 2.2), ab = Math.pow(ad[i+2]/255, 2.2);
    const nx = nd[i]/127.5 - 1.0, ny = nd[i+1]/127.5 - 1.0, nz = nd[i+2]/127.5 - 1.0;
    const rough = Math.max(0.08, 1.0 - sd[i]/255.0);
    const metal = sd[i+1] >= 230 ? 0.92 : 0.0;
    const em = sd[i+3] < 255 ? sd[i+3]/254.0 : 0.0;
    const ndl = Math.max(0.0, nx*lx + ny*ly + nz*lz);
    const ndh = Math.max(0.0, nx*hxn + ny*hyn + nz*hzn);
    const shin = Math.pow(ndh, 2.0 / (rough * rough + 0.01));
    const specW = (0.04 + 0.85 * metal) * shin * (1.2 - rough * 0.7);
    const diffW = (0.28 + 1.15 * ndl) * (1.0 - metal * 0.65);
    let r = ar * diffW + (metal ? ar : 1.0) * specW + ar * em * 1.45;
    let g = ag * diffW + (metal ? ag : 1.0) * specW + ag * em * 1.45;
    let b = ab * diffW + (metal ? ab : 1.0) * specW + ab * em * 1.45;
    od[i]   = Math.min(255, Math.round(Math.pow(r/(1+r*0.35), 1/2.2)*255));
    od[i+1] = Math.min(255, Math.round(Math.pow(g/(1+g*0.35), 1/2.2)*255));
    od[i+2] = Math.min(255, Math.round(Math.pow(b/(1+b*0.35), 1/2.2)*255));
    od[i+3] = a;
  }}
  ctx.putImageData(out, 0, 0);
}}

function renderSidebar() {{
  const box = document.getElementById('preview');
  const n = DATA.nodes[selectedIdx];
  document.getElementById('nodeInfo').textContent = `${{n.id}} (${{n.op}}) — [${{n.shape.join('x')}}] — inputs: ${{JSON.stringify(n.inputs)}}`;
  document.getElementById('paramsPre').textContent = JSON.stringify(n.params, null, 2);
  if (viewMode === 'pbr_live') {{
    renderLivePBR();
  }} else if (DATA.pbr && DATA.pbr[viewMode]) {{
    box.innerHTML = `<img src="${{DATA.pbr[viewMode]}}" style="width:100%;height:100%;object-fit:contain;"/>`;
  }} else if (viewMode === 'tile') {{
    const src = (DATA.pbr && DATA.pbr.pbr_lit) ? DATA.pbr.pbr_lit : n.full;
    let html = '<div class="tile-3x3">';
    for (let k = 0; k < 9; k++) html += `<img src="${{src}}"/>`;
    html += '</div>';
    box.innerHTML = html;
  }} else if (viewMode === 'anim' && DATA.frames.length > 1) {{
    box.innerHTML = `<img id="animImg" src="${{DATA.frames[0]}}" style="width:100%;height:100%;object-fit:contain;"/>`;
    animTimer = setInterval(() => {{
      animIdx = (animIdx + 1) % DATA.frames.length;
      const el = document.getElementById('animImg');
      if (el) el.src = DATA.frames[animIdx];
    }}, 110);
  }} else if (viewMode === 'uv' && DATA.uv_map) {{
    box.innerHTML = `<img src="${{DATA.uv_map}}"/>`;
  }} else {{
    box.innerHTML = `<img src="${{n.full}}" style="width:100%;height:100%;object-fit:contain;"/>`;
  }}
}}

renderGrid();
renderSidebar();
</script>
</body>
</html>
"""
    p = Path(out_html_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page, encoding="utf-8")
    return p
