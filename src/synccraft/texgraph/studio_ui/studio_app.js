/* ==========================================================================
   SyncCraft Studio — Main Workstation Controller (`studio_app.js`)
   Orchestrates API state, Schema v4.0 Inspector, 2D Pixel Canvas, 3D WebGL
   Viewport, Node Graph Canvas, Animation Timeline, Bidirectional .texdsl,
   Multi-Domain QA Scorecard, Dock Splitters & Multi-Engine PBR v2.0 Export.
   ========================================================================== */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const state = {
    schemaPayload: null,
    opMap: {},
    serverState: null,
    evalData: null,
    selectedNodeId: null,
    activePass: 'ALL',
    vpMode: '3d',
    channel: 'pbr_lit',
    tile3x3: false,
    zoom2d: 1.0,
    phase: 0.0,
    animFrames: 8,
    animPlaying: false,
    animTimer: null,
    lightAz: -38,
    lightEl: 52,
    lightingPreset: 'studio_hero',
    paletteFilterType: null,
    paletteConnectFrom: null,
    paletteSpawnPos: null,
    paletteItems: [],
    paletteIdx: 0,
    img2d: null,
    img2dBuf: null,
  };

  let gl3d = null;
  let graphCanvas = null;
  let evalTimer = null;

  // ------------------------------------------------------------------------
  // API Helpers & Toast Notifications
  // ------------------------------------------------------------------------
  async function apiGet(path) {
    const res = await fetch(path, { cache: 'no-store' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || data.error || `HTTP ${res.status}`);
    return data;
  }

  async function apiPost(path, body) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      throw new Error(data.message || data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  let toastTimer = null;
  function showToast(msg, kind) {
    const el = $('studioToast');
    el.textContent = msg;
    el.className = 'toast show' + (kind ? ' ' + kind : '');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.className = 'toast'; }, 2800);
  }

  // ------------------------------------------------------------------------
  // Core Evaluation & Synchronization Loop
  // ------------------------------------------------------------------------
  async function runEvaluate(opts) {
    opts = opts || {};
    try {
      const ev = await apiPost('/api/evaluate', {
        phase: state.phase,
        light_az: state.lightAz,
        light_el: state.lightEl,
        lighting_preset: state.lightingPreset,
        preview_node: state.selectedNodeId,
        include_thumbs: opts.includeThumbs !== false,
        include_qa: opts.includeQa !== false,
      });
      state.evalData = ev;
      updatePerfAndQaBadges();
      updateViewportTextures();
      graphCanvas.setData(state.serverState.recipe, state.evalData, state.selectedNodeId, state.activePass);
      renderQaPanel();
    } catch (err) {
      showToast(`Eval error: ${err.message}`, 'err');
      const alerts = $('validationAlerts');
      if (alerts) {
        alerts.innerHTML = `<div class="log-line error">EVAL ERROR: ${err.message}</div>`;
      }
      $('perfBadge').textContent = 'Eval err';
      $('qaBadge').textContent = 'QA ERR';
      $('qaBadge').className = 'pill-badge err';
    }
  }

  function scheduleEvaluate(delayMs) {
    if (evalTimer) clearTimeout(evalTimer);
    evalTimer = setTimeout(() => runEvaluate({ includeThumbs: true, includeQa: true }), delayMs || 45);
  }

  function applyServerState(st, opts) {
    opts = opts || {};
    state.serverState = st;
    const rec = st.recipe;

    // Ensure valid selected node
    const nodeIds = (rec.nodes || []).map((n) => n.id);
    if (st.created_node_id && nodeIds.includes(st.created_node_id)) {
      state.selectedNodeId = st.created_node_id;
    } else if (!state.selectedNodeId || !nodeIds.includes(state.selectedNodeId)) {
      const outBase = (rec.output_material || rec.output || rec.output_material_node || rec.output_node || '').split('.')[0];
      state.selectedNodeId = nodeIds.includes(outBase) ? outBase : (nodeIds[nodeIds.length - 1] || null);
    }

    // Update topbar controls
    $('btnUndo').disabled = !st.undo_depth;
    $('btnRedo').disabled = !st.redo_depth;
    $('btnTileable').classList.toggle('active', !!rec.tileable);
    const resStr = String(rec.width);
    const resSel = $('resSelect');
    if ([...resSel.options].some((o) => o.value === resStr)) resSel.value = resStr;
    resSel.disabled = !!st.has_mesh;

    // Sync 3D mesh geometry
    if (gl3d) {
      gl3d.setBBModelMesh(st.mesh || null);
      if (opts.autoPrim) {
        const prim = st.has_mesh ? 'bbmodel' : 'cube';
        $('primSelect').value = prim;
        gl3d.primitive = prim;
      }
    }

    // Sync TexDSL editor (unless user is actively typing in it)
    const ed = $('texdslEditor');
    if (document.activeElement !== ed) {
      ed.value = st.texdsl || '';
      $('texdslStatus').textContent = `Synced · ${nodeIds.length} nodes · ${rec.width}×${rec.height}`;
      $('texdslStatus').style.color = 'var(--text-muted)';
    }

    renderPassFilterBar();
    renderInspector();
    renderTimelineDrivers();
    renderExposedParams();
    renderConsoleLog();
    graphCanvas.setData(rec, state.evalData, state.selectedNodeId, state.activePass);
    if (opts.fitGraph) {
      setTimeout(() => graphCanvas.fitToView(), 30);
    }
  }

  async function mutate(payload, opts) {
    opts = opts || {};
    try {
      const st = await apiPost('/api/graph/mutate', payload);
      applyServerState(st, opts);
      if (opts.skipEval) return st;
      if (opts.immediateEval) {
        await runEvaluate({ includeThumbs: true, includeQa: true });
      } else {
        scheduleEvaluate(opts.delayMs || 35);
      }
      return st;
    } catch (err) {
      showToast(err.message, 'err');
      return null;
    }
  }

  // ------------------------------------------------------------------------
  // Topbar, Perf & QA Badges
  // ------------------------------------------------------------------------
  function updatePerfAndQaBadges() {
    const ev = state.evalData;
    if (!ev) return;
    const total = (ev.cache_hits || 0) + (ev.cache_misses || 0);
    const hitPct = total > 0 ? Math.round((ev.cache_hits / total) * 100) : 0;
    $('perfBadge').textContent = `${ev.eval_ms.toFixed(0)}ms · ${hitPct}%`;

    const qa = ev.qa;
    const qb = $('qaBadge');
    if (qa && typeof qa.score === 'number') {
      qb.textContent = `QA ${Math.round(qa.score)}/100`;
      qb.className = 'pill-badge ' + (qa.passed ? 'ok' : (qa.score >= 70 ? 'warn' : 'err'));
    }
  }

  // ------------------------------------------------------------------------
  // Viewport: 2D Pixel Canvas + 3D WebGL PBR Switcher
  // ------------------------------------------------------------------------
  function updateViewportTextures() {
    if (!state.evalData || !state.evalData.channels) return;
    const ch = state.evalData.channels;
    const tileable = !!(state.serverState && state.serverState.recipe && state.serverState.recipe.tileable);
    if (gl3d) {
      gl3d.channel = state.channel;
      gl3d.lightAz = state.lightAz;
      gl3d.lightEl = state.lightEl;
      gl3d.lightingPreset = state.lightingPreset;
      gl3d.updateChannels(ch, state.evalData.selected_preview, tileable);
    }

    let url2d = ch[state.channel] || ch.pbr_lit || ch.albedo;
    if (state.channel === 'selected' && state.evalData.selected_preview) {
      url2d = state.evalData.selected_preview;
    }
    if (url2d) {
      const im = new Image();
      im.onload = () => {
        state.img2d = im;
        const off = document.createElement('canvas');
        off.width = im.width; off.height = im.height;
        const octx = off.getContext('2d');
        octx.drawImage(im, 0, 0);
        state.img2dBuf = { w: im.width, h: im.height, data: octx.getImageData(0, 0, im.width, im.height).data };
        render2DCanvas();
      };
      im.src = url2d;
    }
    updateViewportHud();
  }

  function render2DCanvas() {
    const c = $('canvas2d');
    if (c.style.display === 'none' || !state.img2d) return;
    const r = c.parentElement.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    c.width = Math.max(10, Math.floor(r.width * dpr));
    c.height = Math.max(10, Math.floor(r.height * dpr));
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.imageSmoothingEnabled = false;

    const im = state.img2d;
    const tiles = state.tile3x3 ? 3 : 1;
    const pad = 16 * dpr;
    const availW = c.width - pad * 2;
    const availH = c.height - pad * 2;
    const aspect = im.width / Math.max(1, im.height);
    let drawW = availW, drawH = drawW / aspect;
    if (drawH > availH) { drawH = availH; drawW = drawH * aspect; }
    drawW *= state.zoom2d;
    drawH *= state.zoom2d;
    const ox = (c.width - drawW) * 0.5;
    const oy = (c.height - drawH) * 0.5;
    const tw = drawW / tiles;
    const th = drawH / tiles;

    state._draw2dRect = { ox, oy, drawW, drawH, tiles, dpr };

    for (let ty = 0; ty < tiles; ty++) {
      for (let tx = 0; tx < tiles; tx++) {
        ctx.drawImage(im, ox + tx * tw, oy + ty * th, tw, th);
      }
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.lineWidth = dpr;
    ctx.strokeRect(ox, oy, drawW, drawH);
  }

  function updateViewportHud(pixelInfo) {
    const hud = $('vpHud');
    const ev = state.evalData;
    const res = ev ? `${ev.width}×${ev.height}` : '—';
    const ms = state.serverState && state.serverState.mesh_stats;
    if (state.vpMode === '3d') {
      const mText = ms ? ` · ${ms.elements} boxes · ${ms.seam_pairs} 3D UV seams` : ` · ${$('primSelect').value}`;
      hud.textContent = `3D WebGL PBR · ${state.channel} · ${res}${mText}`;
    } else {
      hud.textContent = `2D Pixel · ${state.channel} · ${res}${state.tile3x3 ? ' · 3×3 Wrap' : ''}${pixelInfo ? ' · ' + pixelInfo : ''}`;
    }
  }

  function setViewportMode(mode) {
    state.vpMode = mode;
    $('tabVp3d').classList.toggle('active', mode === '3d');
    $('tabVp2d').classList.toggle('active', mode === '2d');
    $('gl3dCanvas').style.display = mode === '3d' ? 'block' : 'none';
    $('canvas2d').style.display = mode === '2d' ? 'block' : 'none';
    $('vpTools3d').style.display = mode === '3d' ? 'flex' : 'none';
    $('vpTools2d').style.display = mode === '2d' ? 'flex' : 'none';
    $('viewportStage').classList.toggle('checker', mode === '2d');
    if (mode === '2d') render2DCanvas();
    updateViewportHud();
  }

  // ------------------------------------------------------------------------
  // Schema v4.0 Auto-Generated Inspector (100% of 144 Nodes!)
  // ------------------------------------------------------------------------
  function evalVisibleIf(rule, params) {
    if (!rule || typeof rule !== 'object') return true;
    for (const [k, expected] of Object.entries(rule)) {
      const cur = params[k];
      if (Array.isArray(expected)) {
        if (!expected.includes(cur)) return false;
      } else if (cur !== expected) {
        return false;
      }
    }
    return true;
  }

  function renderInspector() {
    const root = $('rightTabInspector');
    root.innerHTML = '';
    if (!state.serverState || !state.selectedNodeId) {
      root.innerHTML = '<div style="color:var(--text-muted);padding:12px;">Select a node on the canvas to inspect its Schema v4.0 parameters.</div>';
      return;
    }
    const rec = state.serverState.recipe;
    const node = (rec.nodes || []).find((n) => n.id === state.selectedNodeId);
    if (!node) return;
    const meta = state.opMap[node.op] || { op: node.op, category: 'Custom', description: '', params: [], inputs: [], outputs: [] };

    // Node Banner
    const banner = document.createElement('div');
    banner.className = 'insp-node-banner';
    banner.innerHTML = `
      <h3><span>${node.id}</span><span class="brand-sub">${node.op}</span></h3>
      <p>${meta.description || meta.category}</p>
    `;
    root.appendChild(banner);

    // Group parameters by Schema v4.0 `group`
    const groups = new Map();
    const paramsList = meta.params || [];
    for (const p of paramsList) {
      const gName = p.group || 'Parameters';
      if (!groups.has(gName)) groups.set(gName, []);
      groups.get(gName).push(p);
    }

    // Also surface any extra runtime params not in schema list
    const knownKeys = new Set(paramsList.map((p) => p.name));
    for (const [k, v] of Object.entries(node.params || {})) {
      if (!knownKeys.has(k)) {
        if (!groups.has('Custom')) groups.set('Custom', []);
        const w = typeof v === 'boolean' ? 'checkbox' : (typeof v === 'number' ? 'slider' : 'text');
        groups.get('Custom').push({ name: k, widget: w, default: v, min: 0, max: 10, step: 0.05 });
      }
    }

    const effParams = Object.assign({}, ...paramsList.map((p) => ({ [p.name]: p.default })), node.params || {});

    for (const [gName, items] of groups.entries()) {
      const visibleItems = items.filter((p) => evalVisibleIf(p.visible_if, effParams));
      if (!visibleItems.length) continue;

      const card = document.createElement('div');
      card.className = 'insp-group';
      const head = document.createElement('div');
      head.className = 'insp-group-head';
      head.innerHTML = `<span>${gName}</span><span>▾</span>`;
      const body = document.createElement('div');
      body.className = 'insp-group-body';
      head.addEventListener('click', () => {
        const hidden = body.style.display === 'none';
        body.style.display = hidden ? 'flex' : 'none';
        head.lastChild.textContent = hidden ? '▾' : '▸';
      });

      for (const p of visibleItems) {
        const curVal = node.params && (p.name in node.params) ? node.params[p.name] : p.default;
        const row = document.createElement('div');
        row.className = 'param-row';

        const lblRow = document.createElement('div');
        lblRow.className = 'param-label-row';
        const lbl = document.createElement('span');
        lbl.textContent = p.name;
        if (p.tooltip) lbl.title = p.tooltip;
        lblRow.appendChild(lbl);

        const acts = document.createElement('div');
        acts.className = 'param-actions';
        if (p.widget === 'slider' || p.widget === 'int' || typeof curVal === 'number') {
          const btnExp = document.createElement('button');
          btnExp.className = 'tg-mini-btn';
          btnExp.textContent = '+';
          btnExp.title = 'Expose Parameter to Top-Level Graph (@expose)';
          btnExp.onclick = () => {
            mutate({
              action: 'expose_param',
              node_id: node.id,
              param: p.name,
              public_name: `${node.id}_${p.name}`,
              label: `${node.id} · ${p.name}`,
              min: p.min_val != null ? p.min_val : (p.min != null ? p.min : 0),
              max: p.max_val != null ? p.max_val : (p.max != null ? p.max : 1),
              step: p.step != null ? p.step : 0.05,
            }, { skipEval: true });
            showToast(`Exposed ${node.id}.${p.name}`, 'ok');
          };
          const btnAnim = document.createElement('button');
          btnAnim.className = 'tg-mini-btn';
          btnAnim.textContent = '~';
          btnAnim.title = 'Bind Animation Sine Driver (@animate)';
          btnAnim.onclick = () => {
            const num = Number(curVal) || 0.5;
            const pMin = p.min_val != null ? p.min_val : p.min;
            const pMax = p.max_val != null ? p.max_val : p.max;
            const lo = pMin != null ? Math.max(pMin, num * 0.6) : num * 0.6;
            const hi = pMax != null ? Math.min(pMax, Math.max(num * 1.4, lo + 0.1)) : Math.max(num * 1.4, lo + 0.1);
            mutate({
              action: 'add_driver',
              node_id: node.id,
              param: p.name,
              kind: 'sine',
              min_val: Number(lo.toFixed(3)),
              max_val: Number(hi.toFixed(3)),
              freq: 1.0,
            });
            showToast(`Bound sine driver to ${node.id}.${p.name}`, 'ok');
          };
          acts.appendChild(btnExp);
          acts.appendChild(btnAnim);
        }
        lblRow.appendChild(acts);
        row.appendChild(lblRow);

        const ctrl = document.createElement('div');
        ctrl.className = 'param-control-row';

        if (p.widget === 'select' && p.choices && p.choices.length) {
          const sel = document.createElement('select');
          sel.className = 'sel-input';
          for (const ch of p.choices) {
            const o = document.createElement('option');
            o.value = String(ch);
            o.textContent = String(ch);
            if (String(curVal) === String(ch)) o.selected = true;
            sel.appendChild(o);
          }
          sel.onchange = () => mutate({ action: 'set_param', node_id: node.id, param: p.name, value: sel.value });
          ctrl.appendChild(sel);
        } else if (p.widget === 'checkbox' || p.widget === 'bool' || typeof curVal === 'boolean') {
          const chk = document.createElement('input');
          chk.type = 'checkbox';
          chk.checked = !!curVal;
          chk.onchange = () => mutate({ action: 'set_param', node_id: node.id, param: p.name, value: chk.checked });
          const txt = document.createElement('span');
          txt.style.color = 'var(--text-secondary)';
          txt.textContent = chk.checked ? 'Enabled' : 'Disabled';
          ctrl.appendChild(chk);
          ctrl.appendChild(txt);
        } else if (p.widget === 'color' || (typeof curVal === 'string' && /^#[0-9a-fA-F]{6}$/.test(curVal))) {
          const col = document.createElement('input');
          col.type = 'color';
          col.value = (typeof curVal === 'string' && /^#[0-9a-fA-F]{6}$/.test(curVal)) ? curVal : '#888888';
          const hex = document.createElement('input');
          hex.type = 'text';
          hex.className = 'txt-input';
          hex.value = col.value;
          hex.style.fontFamily = 'var(--font-mono)';
          col.oninput = () => {
            hex.value = col.value;
            mutate({ action: 'set_param', node_id: node.id, param: p.name, value: col.value, undoable: false }, { delayMs: 65 });
          };
          col.onchange = () => mutate({ action: 'set_param', node_id: node.id, param: p.name, value: col.value, undoable: true });
          hex.onchange = () => { col.value = hex.value; mutate({ action: 'set_param', node_id: node.id, param: p.name, value: hex.value }); };
          ctrl.appendChild(col);
          ctrl.appendChild(hex);
        } else if (p.widget === 'gradient_stops' || (Array.isArray(curVal) && curVal.length && typeof curVal[0] === 'string' && curVal[0].startsWith('#'))) {
          const stops = Array.isArray(curVal) ? [...curVal] : ['#1e293b', '#64748b', '#e2e8f0'];
          const wrap = document.createElement('div');
          wrap.style.width = '100%';
          const bar = document.createElement('div');
          bar.className = 'ramp-preview';
          bar.style.background = `linear-gradient(90deg, ${stops.join(', ')})`;
          const sw = document.createElement('div');
          sw.className = 'ramp-swatches';
          stops.forEach((hx, idx) => {
            const ci = document.createElement('input');
            ci.type = 'color';
            ci.value = /^#[0-9a-fA-F]{6}$/.test(hx) ? hx : '#888888';
            ci.onchange = () => {
              stops[idx] = ci.value;
              bar.style.background = `linear-gradient(90deg, ${stops.join(', ')})`;
              mutate({ action: 'set_param', node_id: node.id, param: p.name, value: stops });
            };
            sw.appendChild(ci);
          });
          wrap.appendChild(bar);
          wrap.appendChild(sw);
          ctrl.appendChild(wrap);
        } else if (p.widget === 'slider' || p.widget === 'int' || typeof curVal === 'number') {
          const isInt = p.widget === 'int';
          const numVal = Number(curVal != null ? curVal : 0);
          const rawMin = p.min_val != null ? p.min_val : p.min;
          const rawMax = p.max_val != null ? p.max_val : p.max;
          const mn = rawMin != null ? Number(rawMin) : Math.min(0, numVal);
          const mx = rawMax != null ? Number(rawMax) : Math.max(1, numVal * 2 || 10);
          const stp = p.step != null ? Number(p.step) : (isInt ? 1 : (mx - mn) / 100);
          const rng = document.createElement('input');
          rng.type = 'range';
          rng.min = String(mn);
          rng.max = String(mx);
          rng.step = String(stp);
          rng.value = String(numVal);

          const num = document.createElement('input');
          num.type = 'number';
          num.className = 'num-input';
          num.step = String(stp);
          num.value = String(isInt ? Math.round(numVal) : Number(numVal.toFixed(3)));

          rng.oninput = () => {
            const v = isInt ? parseInt(rng.value, 10) : parseFloat(rng.value);
            num.value = String(v);
            mutate({ action: 'set_param', node_id: node.id, param: p.name, value: v, undoable: false }, { delayMs: 70 });
          };
          rng.onchange = () => {
            const v = isInt ? parseInt(rng.value, 10) : parseFloat(rng.value);
            mutate({ action: 'set_param', node_id: node.id, param: p.name, value: v, undoable: true });
          };
          num.onchange = () => {
            const v = isInt ? parseInt(num.value, 10) : parseFloat(num.value);
            if (!Number.isNaN(v)) {
              rng.value = String(v);
              mutate({ action: 'set_param', node_id: node.id, param: p.name, value: v, undoable: true });
            }
          };
          ctrl.appendChild(rng);
          ctrl.appendChild(num);
        } else {
          const inp = document.createElement('input');
          inp.type = 'text';
          inp.className = 'txt-input';
          inp.value = typeof curVal === 'object' ? JSON.stringify(curVal) : String(curVal != null ? curVal : '');
          inp.onchange = () => {
            let val = inp.value;
            try { if (val.startsWith('[') || val.startsWith('{')) val = JSON.parse(val); } catch (_) {}
            mutate({ action: 'set_param', node_id: node.id, param: p.name, value: val });
          };
          ctrl.appendChild(inp);
        }

        row.appendChild(ctrl);
        body.appendChild(row);
      }

      card.appendChild(head);
      card.appendChild(body);
      root.appendChild(card);
    }
  }

  // ------------------------------------------------------------------------
  // Pass Filter Bar, Node Library & Fuzzy Command Palette
  // ------------------------------------------------------------------------
  function renderPassFilterBar() {
    const bar = $('passFilterBar');
    bar.innerHTML = '';
    const passes = (state.serverState && state.serverState.recipe && state.serverState.recipe.passes) || [];
    if (!passes.length) return;
    const mk = (id, label) => {
      const b = document.createElement('button');
      b.className = 'btn sm' + (state.activePass === id ? ' active' : '');
      b.textContent = label;
      b.onclick = () => {
        state.activePass = id;
        renderPassFilterBar();
        graphCanvas.setData(state.serverState.recipe, state.evalData, state.selectedNodeId, state.activePass);
      };
      bar.appendChild(b);
    };
    mk('ALL', 'All');
    passes.forEach((p, idx) => {
      const clean = String(p.title || p.name).replace(/^Pass\s*\d+\s*:\s*/i, '');
      const short = clean.length > 22 ? clean.slice(0, 20) + '…' : clean;
      mk(p.name, `P${idx + 1}: ${short}`);
    });
  }

  function renderNodeLibrary() {
    const body = $('libBody');
    const q = ($('libSearch').value || '').trim().toLowerCase();
    body.innerHTML = '';
    if (!state.schemaPayload) return;
    const cats = state.schemaPayload.schema.nodes_by_category || {};
    let totalShown = 0;

    for (const [cat, nodes] of Object.entries(cats)) {
      const filtered = q
        ? nodes.filter((n) => n.op.toLowerCase().includes(q) || (n.description || '').toLowerCase().includes(q) || cat.toLowerCase().includes(q))
        : nodes;
      if (!filtered.length) continue;
      totalShown += filtered.length;

      const wrap = document.createElement('div');
      wrap.className = 'lib-cat';
      const head = document.createElement('div');
      head.className = 'lib-cat-head';
      head.innerHTML = `<span>${cat}</span><span style="color:var(--text-muted);font-family:var(--font-mono);">${filtered.length}</span>`;
      const list = document.createElement('div');
      list.className = 'lib-items';
      // Keep non-search list tidy by expanding key categories first
      const defaultOpen = !!q || cat.includes('Seamless') || cat.includes('Material') || cat.includes('Pixel Composer') || cat.includes('Noise');
      list.style.display = defaultOpen ? 'grid' : 'none';
      head.onclick = () => { list.style.display = list.style.display === 'none' ? 'grid' : 'none'; };

      for (const n of filtered) {
        const it = document.createElement('div');
        it.className = 'lib-item';
        it.title = `${n.op} — ${n.description || ''}`;
        const outType = n.output_type || (n.outputs && n.outputs[0] && (n.outputs[0].dtype || n.outputs[0].port_type)) || 'RGBA';
        const col = (state.schemaPayload.schema.port_colors || {})[outType] || '#94a3b8';
        it.innerHTML = `<span class="lib-dot" style="background:${col};"></span><span>${n.op}</span>`;
        it.onclick = () => {
          const cx = (-graphCanvas.panX + graphCanvas.stage.clientWidth * 0.5) / graphCanvas.zoom - 100;
          const cy = (-graphCanvas.panY + graphCanvas.stage.clientHeight * 0.45) / graphCanvas.zoom - 60;
          mutate({ action: 'add_node', op: n.op, pos: [Math.round(cx), Math.round(cy)] }, { immediateEval: true });
          showToast(`Added ${n.op}`, 'ok');
        };
        list.appendChild(it);
      }

      wrap.appendChild(head);
      wrap.appendChild(list);
      body.appendChild(wrap);
    }
    $('libCount').textContent = String(totalShown);
  }

  function openPalette(opts) {
    opts = opts || {};
    state.paletteFilterType = opts.filterType || null;
    state.paletteConnectFrom = opts.connectFrom || null;
    state.paletteSpawnPos = opts.spawnPos || null;
    $('paletteModal').classList.add('open');
    const inp = $('paletteInput');
    inp.value = '';
    inp.placeholder = state.paletteFilterType
      ? `Add node compatible with ${state.paletteFilterType} output...`
      : 'Search 144 TexGraph nodes by name, category or description...';
    renderPaletteList();
    setTimeout(() => inp.focus(), 10);
  }

  function closePalette() {
    $('paletteModal').classList.remove('open');
    state.paletteFilterType = null;
    state.paletteConnectFrom = null;
  }

  function renderPaletteList() {
    const q = ($('paletteInput').value || '').trim().toLowerCase();
    const list = $('paletteList');
    list.innerHTML = '';
    if (!state.schemaPayload) return;

    const compatSet = state.paletteFilterType
      ? new Set(((state.schemaPayload.schema.port_compatibility) || {})[state.paletteFilterType] || [state.paletteFilterType])
      : null;

    const items = [];
    for (const [cat, nodes] of Object.entries(state.schemaPayload.schema.nodes_by_category || {})) {
      for (const n of nodes) {
        if (compatSet) {
          const hasCompatIn = (n.inputs || []).some((inp) => compatSet.has(inp.dtype || inp.port_type));
          if (!hasCompatIn) continue;
        }
        if (q && !(n.op.toLowerCase().includes(q) || cat.toLowerCase().includes(q) || (n.description || '').toLowerCase().includes(q))) {
          continue;
        }
        items.push({ op: n.op, cat, desc: n.description || '' });
      }
    }
    state.paletteItems = items.slice(0, 45);
    state.paletteIdx = 0;

    state.paletteItems.forEach((it, idx) => {
      const row = document.createElement('div');
      row.className = 'palette-item' + (idx === 0 ? ' sel' : '');
      row.innerHTML = `
        <div class="palette-item-main">
          <span class="palette-item-name">${it.op}</span>
          <span class="palette-item-desc">${it.desc}</span>
        </div>
        <span class="palette-item-cat">${it.cat}</span>
      `;
      row.onclick = () => spawnFromPalette(it.op);
      list.appendChild(row);
    });
  }

  function spawnFromPalette(op) {
    const pos = state.paletteSpawnPos || [
      Math.round((-graphCanvas.panX + graphCanvas.stage.clientWidth * 0.5) / graphCanvas.zoom - 100),
      Math.round((-graphCanvas.panY + graphCanvas.stage.clientHeight * 0.45) / graphCanvas.zoom - 60),
    ];
    const payload = { action: 'add_node', op, pos };
    if (state.paletteConnectFrom) payload.connect_from = state.paletteConnectFrom;
    closePalette();
    mutate(payload, { immediateEval: true });
    showToast(`Spawned ${op}`, 'ok');
  }

  // ------------------------------------------------------------------------
  // Timeline Drivers, Exposed Params, QA & Diagnostics
  // ------------------------------------------------------------------------
  function renderTimelineDrivers() {
    const list = $('driversList');
    list.innerHTML = '';
    const drivers = (state.serverState && state.serverState.recipe && state.serverState.recipe.drivers) || [];
    if (!drivers.length) {
      list.innerHTML = '<div style="font-size:11px;color:var(--text-muted);padding:4px 2px;">No active @animate drivers. Click <code>~</code> next to any numeric parameter in the Inspector to animate it!</div>';
      return;
    }
    drivers.forEach((d, idx) => {
      const row = document.createElement('div');
      row.className = 'driver-row';
      row.innerHTML = `
        <span style="font-family:var(--font-mono);color:var(--text-primary);overflow:hidden;text-overflow:ellipsis;">${d.node_id}.${d.param_key}</span>
        <span style="font-family:var(--font-mono);color:var(--focus-amber);">${d.kind} [${d.min_val}..${d.max_val}]</span>
        <canvas class="driver-wave" width="220" height="20"></canvas>
        <button class="tg-mini-btn" title="Remove Driver">✕</button>
      `;
      row.querySelector('button').onclick = () => mutate({ action: 'remove_driver', index: idx });
      const cv = row.querySelector('canvas');
      const ctx = cv.getContext('2d');
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let x = 0; x < 220; x++) {
        const ph = x / 219;
        let v = 0.5 + 0.5 * Math.sin(ph * Math.PI * 2 * (d.freq || 1));
        if (d.kind === 'sawtooth') v = (ph * (d.freq || 1)) % 1;
        if (d.kind === 'wiggler') v = 0.5 + 0.35 * Math.sin(ph * Math.PI * 2) + 0.15 * Math.cos(ph * Math.PI * 6);
        const y = 17 - v * 14;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
      // Playhead line
      const px = Math.round(state.phase * 219);
      ctx.strokeStyle = '#3b82f6';
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, 20); ctx.stroke();
      list.appendChild(row);
    });
  }

  function renderExposedParams() {
    const grid = $('exposedParamsGrid');
    grid.innerHTML = '';
    const exposed = (state.serverState && state.serverState.recipe && state.serverState.recipe.exposed_params) || {};
    const keys = Object.keys(exposed);
    $('exposedCount').textContent = String(keys.length);
    if (!keys.length) {
      grid.innerHTML = '<div style="font-size:11px;color:var(--text-muted);">No exposed graph parameters yet. Click <code>+</code> beside any Inspector slider to promote it here!</div>';
      return;
    }
    const nodes = (state.serverState.recipe && state.serverState.recipe.nodes) || [];
    for (const pub of keys) {
      const info = exposed[pub];
      const node = nodes.find((n) => n.id === info.node_id);
      const curVal = node && node.params && (info.param_key in node.params) ? Number(node.params[info.param_key]) : 0.5;
      const card = document.createElement('div');
      card.className = 'qa-metric-card';
      card.innerHTML = `
        <div class="param-label-row">
          <span style="font-weight:600;color:var(--text-primary);">${info.label || pub}</span>
          <button class="tg-mini-btn" title="Unexpose">✕</button>
        </div>
        <div class="param-control-row" style="margin-top:5px;">
          <input type="range" min="${info.min != null ? info.min : 0}" max="${info.max != null ? info.max : 1}" step="${info.step != null ? info.step : 0.05}" value="${curVal}" />
          <span style="font-family:var(--font-mono);font-size:11px;min-width:36px;text-align:right;">${curVal.toFixed(2)}</span>
        </div>
      `;
      card.querySelector('button').onclick = () => mutate({ action: 'remove_exposed', public_name: pub }, { skipEval: true });
      const rng = card.querySelector('input');
      const readout = card.querySelector('span:last-child');
      rng.oninput = () => {
        const v = parseFloat(rng.value);
        readout.textContent = v.toFixed(2);
        mutate({ action: 'set_param', node_id: info.node_id, param: info.param_key, value: v, undoable: false }, { delayMs: 65 });
      };
      rng.onchange = () => mutate({ action: 'set_param', node_id: info.node_id, param: info.param_key, value: parseFloat(rng.value), undoable: true });
      grid.appendChild(card);
    }
  }

  function renderConsoleLog() {
    const alerts = $('validationAlerts');
    const val = state.serverState && state.serverState.validation;
    alerts.innerHTML = '';
    if (val) {
      if (val.errors && val.errors.length) {
        val.errors.forEach((e) => {
          const d = document.createElement('div');
          d.className = 'log-line error';
          d.textContent = `ERROR: ${e}`;
          alerts.appendChild(d);
        });
      }
      if (val.warnings && val.warnings.length) {
        val.warnings.forEach((w) => {
          const d = document.createElement('div');
          d.className = 'log-line';
          d.style.color = '#fbbf24';
          d.textContent = `WARN: ${w}`;
          alerts.appendChild(d);
        });
      }
    }
    const list = $('consoleLogList');
    list.innerHTML = '';
    const logs = (state.serverState && state.serverState.log) || [];
    for (const item of [...logs].reverse()) {
      const d = document.createElement('div');
      d.className = 'log-line' + (item.level === 'error' ? ' error' : '');
      d.textContent = `[${item.level.toUpperCase()}] ${item.message}`;
      list.appendChild(d);
    }
  }

  function renderQaPanel() {
    const root = $('qaPanelBody');
    root.innerHTML = '';
    const ev = state.evalData;
    if (!ev) return;
    const qa = ev.qa || { score: 100, passed: true, metrics: {}, issues: [] };
    const ring = document.createElement('div');
    ring.className = 'qa-score-ring';
    ring.innerHTML = `
      <div>
        <div style="font-size:11px;color:var(--text-secondary);">Automated Multi-Domain QA</div>
        <div style="font-size:10.5px;color:var(--text-muted);">${qa.passed ? 'PASSED Studio Quality Gate' : 'Needs Review'}</div>
      </div>
      <div class="qa-big" style="color:${qa.passed ? '#4ade80' : '#fbbf24'};">${Math.round(qa.score)}/100</div>
    `;
    root.appendChild(ring);

    const grid = document.createElement('div');
    grid.className = 'qa-metric-grid';
    const entries = [
      ['Resolution', `${ev.width}×${ev.height}`],
      ['Eval Time', `${ev.eval_ms.toFixed(1)} ms`],
      ['Cache Hits', `${ev.cache_hits} / ${ev.cache_hits + ev.cache_misses}`],
      ['Total Nodes', `${Object.keys(ev.timings_ms || {}).length}`],
    ];
    for (const [k, v] of Object.entries(qa.metrics || {})) {
      if (typeof v === 'number') entries.push([k, v.toFixed(2)]);
    }
    for (const [k, v] of entries.slice(0, 10)) {
      const c = document.createElement('div');
      c.className = 'qa-metric-card';
      c.innerHTML = `<div class="qa-metric-k">${k}</div><div class="qa-metric-v">${v}</div>`;
      grid.appendChild(c);
    }
    root.appendChild(grid);

    // Slowest nodes profile breakdown
    const prof = document.createElement('div');
    prof.className = 'insp-group';
    prof.innerHTML = '<div class="insp-group-head"><span>Per-Node Execution Profile</span></div><div class="insp-group-body"></div>';
    const pb = prof.querySelector('.insp-group-body');
    const ranked = Object.entries(ev.timings_ms || {}).sort((a, b) => b[1] - a[1]).slice(0, 8);
    const maxMs = Math.max(0.5, ...ranked.map((r) => r[1]));
    for (const [nid, ms] of ranked) {
      const pct = Math.min(100, Math.round((ms / maxMs) * 100));
      const line = document.createElement('div');
      line.innerHTML = `
        <div class="param-label-row"><span style="font-family:var(--font-mono);">${nid}</span><span style="font-family:var(--font-mono);color:${ms === 0 ? '#4ade80' : 'var(--text-primary)'};">${ms} ms</span></div>
        <div style="height:4px;background:var(--bg-input);border-radius:2px;overflow:hidden;margin-top:2px;"><div style="width:${pct}%;height:100%;background:var(--accent);"></div></div>
      `;
      pb.appendChild(line);
    }
    root.appendChild(prof);
  }

  // ------------------------------------------------------------------------
  // Resizable Splitters & Docking Layout Presets
  // ------------------------------------------------------------------------
  function setupSplitters() {
    const bindV = (el, target, invert) => {
      let dragging = false, sx = 0, sw = 0;
      el.addEventListener('pointerdown', (e) => {
        dragging = true; sx = e.clientX; sw = target.getBoundingClientRect().width;
        el.classList.add('dragging');
      });
      window.addEventListener('pointermove', (e) => {
        if (!dragging) return;
        const dx = invert ? (sx - e.clientX) : (e.clientX - sx);
        target.style.width = `${Math.max(240, Math.min(680, sw + dx))}px`;
        if (state.vpMode === '2d') render2DCanvas();
      });
      window.addEventListener('pointerup', () => { dragging = false; el.classList.remove('dragging'); });
    };

    const bindH = (el, target, invert) => {
      let dragging = false, sy = 0, sh = 0;
      el.addEventListener('pointerdown', (e) => {
        dragging = true; sy = e.clientY; sh = target.getBoundingClientRect().height;
        el.classList.add('dragging');
      });
      window.addEventListener('pointermove', (e) => {
        if (!dragging) return;
        const dy = invert ? (sy - e.clientY) : (e.clientY - sy);
        target.style.height = `${Math.max(110, Math.min(540, sh + dy))}px`;
        if (state.vpMode === '2d') render2DCanvas();
      });
      window.addEventListener('pointerup', () => { dragging = false; el.classList.remove('dragging'); });
    };

    bindV($('splitMainLeft'), $('colLeft'), false);
    bindV($('splitMainRight'), $('colRight'), true);
    bindH($('splitLeftH'), $('paneViewport'), false);
    bindH($('splitCenterH'), $('paneBottom'), true);
  }

  function applyLayoutPreset(name) {
    const L = $('colLeft'), R = $('colRight'), B = $('paneBottom'), V = $('paneViewport');
    if (name === 'default') {
      L.style.width = '380px'; R.style.width = '332px'; B.style.height = '176px'; V.style.height = '57%';
    } else if (name === 'lookdev') {
      L.style.width = '540px'; R.style.width = '310px'; B.style.height = '136px'; V.style.height = '72%';
    } else if (name === 'graph') {
      L.style.width = '290px'; R.style.width = '290px'; B.style.height = '126px'; V.style.height = '52%';
    } else if (name === 'code') {
      L.style.width = '320px'; R.style.width = '490px'; B.style.height = '140px'; V.style.height = '56%';
      switchRightTab('texdsl');
    }
    setTimeout(() => { graphCanvas.fitToView(); render2DCanvas(); }, 40);
  }

  function switchRightTab(tab) {
    document.querySelectorAll('[data-right-tab]').forEach((b) => b.classList.toggle('active', b.dataset.rightTab === tab));
    $('rightTabInspector').style.display = tab === 'inspector' ? 'flex' : 'none';
    $('rightTabTexdsl').style.display = tab === 'texdsl' ? 'flex' : 'none';
    $('rightTabQa').style.display = tab === 'qa' ? 'flex' : 'none';
  }

  function switchBottomTab(tab) {
    document.querySelectorAll('[data-bot-tab]').forEach((b) => b.classList.toggle('active', b.dataset.botTab === tab));
    $('botTabTimeline').style.display = tab === 'timeline' ? 'flex' : 'none';
    $('botTabExposed').style.display = tab === 'exposed' ? 'flex' : 'none';
    $('botTabConsole').style.display = tab === 'console' ? 'flex' : 'none';
  }

  // ------------------------------------------------------------------------
  // Animation Playback Controls
  // ------------------------------------------------------------------------
  function setPhase(ph) {
    state.phase = ((ph % 1) + 1) % 1;
    $('phaseSlider').value = state.phase.toFixed(2);
    $('phaseLabel').textContent = state.phase.toFixed(2);
    const fIdx = Math.floor(state.phase * state.animFrames);
    $('frameReadout').textContent = `F${fIdx}/${state.animFrames} · φ${state.phase.toFixed(2)}`;
    renderTimelineDrivers();
    scheduleEvaluate(20);
  }

  function togglePlayAnim() {
    state.animPlaying = !state.animPlaying;
    $('btnPlayAnim').textContent = state.animPlaying ? '⏸ Pause' : '▶ Play';
    $('btnPlayAnim').classList.toggle('amber', state.animPlaying);
    if (state.animTimer) { clearInterval(state.animTimer); state.animTimer = null; }
    if (state.animPlaying) {
      state.animTimer = setInterval(() => {
        setPhase(state.phase + 1.0 / Math.max(2, state.animFrames));
      }, 260);
    }
  }

  function triggerDownload(filename, textContent) {
    const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ------------------------------------------------------------------------
  // Bootstrap Studio Application
  // ------------------------------------------------------------------------
  async function boot() {
    gl3d = new StudioGL3DViewport($('gl3dCanvas'), $('vpHud'));
    graphCanvas = new StudioGraphCanvas({
      stage: $('graphStage'),
      gridBg: $('graphGridBg'),
      world: $('graphWorld'),
      passLayer: $('passFramesLayer'),
      wiresSvg: $('graphWiresSvg'),
      nodesLayer: $('nodesLayer'),
      minimapCanvas: $('minimapCanvas'),
      zoomBadge: $('graphZoomBadge'),
      callbacks: {
        onSelectNode: (nid, inspectInVp) => {
          state.selectedNodeId = nid;
          renderInspector();
          graphCanvas.setData(state.serverState.recipe, state.evalData, state.selectedNodeId, state.activePass);
          if (inspectInVp) {
            state.channel = 'selected';
            $('channelSelect').value = 'selected';
            runEvaluate({ includeThumbs: false, includeQa: false });
          }
        },
        onMoveNode: (nid, pos) => mutate({ action: 'move_nodes', positions: { [nid]: pos }, undoable: false }, { skipEval: true }),
        onConnect: (src, srcPort, dst, dstPort) => mutate({ action: 'connect', src_node: src, src_port: srcPort, dst_node: dst, dst_port: dstPort }, { immediateEval: true }),
        onDisconnect: (dst, dstPort) => mutate({ action: 'disconnect', dst_node: dst, dst_port: dstPort }, { immediateEval: true }),
        onDropWireEmpty: (srcNode, srcPort, portType, wPos) => openPalette({
          filterType: portType,
          connectFrom: { node: srcNode, port: srcPort },
          spawnPos: [Math.round(wPos.x), Math.round(wPos.y)],
        }),
        onToggleBypass: (nid) => mutate({ action: 'toggle_bypass', node_id: nid }, { immediateEval: true }),
        onToggleCollapse: (nid) => mutate({ action: 'toggle_collapse', node_id: nid }, { skipEval: true }),
        onSetOutput: (nid) => mutate({ action: 'set_output', node_id: nid }, { immediateEval: true }),
      },
    });

    // Focus outline tracking (Pixel Composer convention!)
    document.querySelectorAll('.pane').forEach((p) => {
      p.addEventListener('pointerdown', () => {
        document.querySelectorAll('.pane').forEach((q) => q.classList.remove('focused'));
        p.classList.add('focused');
      });
    });

    setupSplitters();

    // Load Schema v4.0 + Initial Workspace State
    const [schemaPayload, initState] = await Promise.all([apiGet('/api/schema'), apiGet('/api/state')]);
    state.schemaPayload = schemaPayload;
    for (const [cat, nodes] of Object.entries(schemaPayload.schema.nodes_by_category || {})) {
      for (const n of nodes) state.opMap[n.op] = Object.assign({ category: cat }, n);
    }
    graphCanvas.setSchema(schemaPayload.schema);
    $('versionBadge').textContent = `v${schemaPayload.schema.engine_version} · ${schemaPayload.schema.node_count} nodes`;

    // Populate Project / Preset Dropdown
    const pSel = $('projectSelect');
    const ogSamples = document.createElement('optgroup');
    ogSamples.label = 'Flagship Studio Projects (v10)';
    for (const s of schemaPayload.samples) {
      const o = document.createElement('option');
      o.value = `sample:${s.id}`;
      o.textContent = s.title;
      ogSamples.appendChild(o);
    }
    pSel.appendChild(ogSamples);

    const byCat = {};
    for (const p of schemaPayload.presets) {
      (byCat[p.category] = byCat[p.category] || []).push(p);
    }
    for (const [cat, list] of Object.entries(byCat)) {
      const og = document.createElement('optgroup');
      og.label = `Preset · ${cat}`;
      for (const p of list) {
        const o = document.createElement('option');
        o.value = `preset:${p.id}`;
        o.textContent = p.title;
        og.appendChild(o);
      }
      pSel.appendChild(og);
    }
    pSel.value = initState.source || 'sample:monolith_v10_3d';

    renderNodeLibrary();
    applyServerState(initState, { fitGraph: true, autoPrim: true });
    await runEvaluate({ includeThumbs: true, includeQa: true });
    setTimeout(() => graphCanvas.fitToView(), 80);

    // ----------------------------------------------------------------------
    // Event Listeners
    // ----------------------------------------------------------------------
    pSel.onchange = async () => {
      const [kind, id] = pSel.value.split(':');
      try {
        // Clear stale 2D/eval buffers immediately on project switch
        state.evalData = null;
        state.img2d = null;
        state.img2dBuf = null;
        const c2 = $('canvas2d');
        if (c2) { const ctx = c2.getContext('2d'); if (ctx) ctx.clearRect(0, 0, c2.width, c2.height); }
        const st = await apiPost('/api/load', { kind, id, size: parseInt($('resSelect').value, 10) || 64 });
        state.activePass = 'ALL';
        state.selectedNodeId = null;
        applyServerState(st, { fitGraph: true, autoPrim: true });
        await runEvaluate({ includeThumbs: true, includeQa: true });
        requestAnimationFrame(() => graphCanvas.fitToView());
        showToast(`Loaded ${st.recipe.name}`, 'ok');
      } catch (err) {
        showToast(err.message, 'err');
      }
    };

    $('resSelect').onchange = () => {
      const sz = parseInt($('resSelect').value, 10);
      mutate({ action: 'set_graph_settings', width: sz, height: sz }, { immediateEval: true });
    };
    $('btnTileable').onclick = () => {
      const cur = !!(state.serverState && state.serverState.recipe && state.serverState.recipe.tileable);
      mutate({ action: 'set_graph_settings', tileable: !cur }, { immediateEval: true });
    };
    $('btnUndo').onclick = () => mutate({ action: 'undo' }, { immediateEval: true });
    $('btnRedo').onclick = () => mutate({ action: 'redo' }, { immediateEval: true });
    $('layoutPresetSelect').onchange = (e) => applyLayoutPreset(e.target.value);

    // Viewport controls
    $('tabVp3d').onclick = () => setViewportMode('3d');
    $('tabVp2d').onclick = () => setViewportMode('2d');
    $('channelSelect').onchange = (e) => {
      state.channel = e.target.value;
      updateViewportTextures();
    };
    $('primSelect').onchange = (e) => {
      gl3d.primitive = e.target.value;
      updateViewportTextures();
    };
    $('btnSpin3d').onclick = () => {
      gl3d.spin = !gl3d.spin;
      $('btnSpin3d').classList.toggle('active', gl3d.spin);
    };
    $('btnWire3d').onclick = () => {
      gl3d.wireframe = !gl3d.wireframe;
      $('btnWire3d').classList.toggle('active', gl3d.wireframe);
    };
    $('btnResetCam3d').onclick = () => gl3d.resetCamera();
    $('btnTile3x3').onclick = () => {
      state.tile3x3 = !state.tile3x3;
      $('btnTile3x3').classList.toggle('active', state.tile3x3);
      render2DCanvas();
      updateViewportHud();
    };
    $('btnZoom2dFit').onclick = () => { state.zoom2d = 1.0; render2DCanvas(); };
    $('btnZoom2dIn').onclick = () => { state.zoom2d = Math.min(8, state.zoom2d * 1.25); render2DCanvas(); };
    $('btnZoom2dOut').onclick = () => { state.zoom2d = Math.max(0.4, state.zoom2d * 0.8); render2DCanvas(); };
    $('lightingPresetSelect').onchange = (e) => {
      state.lightingPreset = e.target.value;
      runEvaluate({ includeThumbs: false, includeQa: false });
    };
    $('lightAzSlider').oninput = (e) => {
      state.lightAz = parseFloat(e.target.value);
      $('lightAzVal').textContent = `${state.lightAz}°`;
      if (gl3d) gl3d.lightAz = state.lightAz;
      scheduleEvaluate(90);
    };
    $('lightElSlider').oninput = (e) => {
      state.lightEl = parseFloat(e.target.value);
      $('lightElVal').textContent = `${state.lightEl}°`;
      if (gl3d) gl3d.lightEl = state.lightEl;
      scheduleEvaluate(90);
    };

    // 2D Pixel Eyedropper HUD
    $('canvas2d').addEventListener('pointermove', (e) => {
      const dr = state._draw2dRect;
      const buf = state.img2dBuf;
      if (!dr || !buf) return;
      const r = $('canvas2d').getBoundingClientRect();
      const cx = (e.clientX - r.left) * dr.dpr;
      const cy = (e.clientY - r.top) * dr.dpr;
      const u = ((cx - dr.ox) / dr.drawW) * dr.tiles;
      const v = ((cy - dr.oy) / dr.drawH) * dr.tiles;
      if (u < 0 || v < 0 || u >= dr.tiles || v >= dr.tiles) return;
      const px = Math.floor((u % 1) * buf.w);
      const py = Math.floor((v % 1) * buf.h);
      const idx = (py * buf.w + px) * 4;
      const R = buf.data[idx], G = buf.data[idx + 1], B = buf.data[idx + 2];
      const hex = '#' + [R, G, B].map((c) => c.toString(16).padStart(2, '0')).join('').toUpperCase();
      updateViewportHud(`(${px},${py}) ${hex}`);
    });

    // Graph Toolbar
    $('btnAddNode').onclick = () => openPalette();
    $('btnOpenPaletteLib').onclick = () => openPalette();
    $('btnAutoLayout').onclick = () => mutate({ action: 'auto_layout' }, { skipEval: true, fitGraph: true });
    $('btnFitGraph').onclick = () => graphCanvas.fitToView();
    $('btnBypassSel').onclick = () => { if (state.selectedNodeId) mutate({ action: 'toggle_bypass', node_id: state.selectedNodeId }, { immediateEval: true }); };
    $('btnDeleteSel').onclick = () => { if (state.selectedNodeId) mutate({ action: 'delete_nodes', ids: [state.selectedNodeId] }, { immediateEval: true }); };
    $('btnSetOutputSel').onclick = () => { if (state.selectedNodeId) mutate({ action: 'set_output', node_id: state.selectedNodeId }, { immediateEval: true }); };
    $('libSearch').oninput = renderNodeLibrary;

    // Right & Bottom Tabs
    document.querySelectorAll('[data-right-tab]').forEach((b) => { b.onclick = () => switchRightTab(b.dataset.rightTab); });
    document.querySelectorAll('[data-bot-tab]').forEach((b) => { b.onclick = () => switchBottomTab(b.dataset.botTab); });

    // Live Bidirectional TexDSL Editor
    $('btnSyncFromGraph').onclick = () => {
      if (state.serverState) {
        $('texdslEditor').value = state.serverState.texdsl || '';
        $('texdslStatus').textContent = 'Refreshed from active node graph';
      }
    };
    $('btnApplyTexdsl').onclick = async () => {
      const code = $('texdslEditor').value;
      const st = await mutate({ action: 'apply_texdsl', code }, { immediateEval: true, fitGraph: true });
      if (st) {
        $('texdslStatus').textContent = `Compiled cleanly · ${st.recipe.nodes.length} nodes`;
        $('texdslStatus').style.color = '#4ade80';
        showToast('Compiled .texdsl -> live graph updated!', 'ok');
      } else {
        $('texdslStatus').textContent = 'Compile failed (see toast)';
        $('texdslStatus').style.color = '#f87171';
      }
    };

    // Timeline
    $('phaseSlider').oninput = (e) => setPhase(parseFloat(e.target.value));
    $('btnPlayAnim').onclick = togglePlayAnim;
    $('btnStepPrev').onclick = () => setPhase(state.phase - 1.0 / Math.max(2, state.animFrames));
    $('btnStepNext').onclick = () => setPhase(state.phase + 1.0 / Math.max(2, state.animFrames));
    $('animFramesInput').onchange = (e) => {
      state.animFrames = Math.max(2, Math.min(32, parseInt(e.target.value, 10) || 8));
      setPhase(state.phase);
    };

    // Command Palette
    $('paletteInput').oninput = renderPaletteList;
    $('paletteInput').onkeydown = (e) => {
      if (e.key === 'Escape') { closePalette(); return; }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        state.paletteIdx = Math.min(state.paletteItems.length - 1, state.paletteIdx + 1);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        state.paletteIdx = Math.max(0, state.paletteIdx - 1);
      } else if (e.key === 'Enter' && state.paletteItems[state.paletteIdx]) {
        e.preventDefault();
        spawnFromPalette(state.paletteItems[state.paletteIdx].op);
        return;
      } else {
        return;
      }
      document.querySelectorAll('.palette-item').forEach((el, i) => el.classList.toggle('sel', i === state.paletteIdx));
    };
    $('paletteModal').addEventListener('click', (e) => { if (e.target === $('paletteModal')) closePalette(); });

    // 3D Mesh Modal
    $('btnLoadMesh').onclick = () => $('meshModal').classList.add('open');
    $('btnCloseMesh').onclick = () => $('meshModal').classList.remove('open');
    $('meshModal').addEventListener('click', (e) => { if (e.target === $('meshModal')) $('meshModal').classList.remove('open'); });
    const loadMeshSample = async (sample) => {
      $('meshModal').classList.remove('open');
      try {
        const st = await apiPost('/api/mesh/load', { sample });
        applyServerState(st, { autoPrim: true });
        await runEvaluate({ includeThumbs: true, includeQa: true });
        showToast(sample === 'none' ? 'Detached 3D mesh' : `Baked 3D mesh (${sample})`, 'ok');
      } catch (err) {
        showToast(err.message, 'err');
      }
    };
    $('btnMeshMonolith').onclick = () => loadMeshSample('temple_monolith');
    $('btnMeshOrc').onclick = () => loadMeshSample('ironjaw_orc');
    $('btnMeshDetach').onclick = () => loadMeshSample('none');
    $('btnMeshUpload').onclick = () => $('bbmodelFileInput').click();
    $('bbmodelFileInput').onchange = async (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      $('meshModal').classList.remove('open');
      try {
        const doc = JSON.parse(await f.text());
        const st = await apiPost('/api/mesh/load', { bbmodel: doc });
        applyServerState(st, { autoPrim: true });
        await runEvaluate({ includeThumbs: true, includeQa: true });
        showToast(`Baked uploaded .bbmodel: ${f.name}`, 'ok');
      } catch (err) {
        showToast(`Invalid .bbmodel: ${err.message}`, 'err');
      }
    };

    // Multi-Engine Export Modal
    $('btnOpenExport').onclick = () => {
      $('exportStemInput').value = (state.serverState && state.serverState.recipe && state.serverState.recipe.name) || 'synccraft_asset';
      $('exportPresetSelect').value = state.lightingPreset;
      $('exportResultPre').style.display = 'none';
      $('exportModal').classList.add('open');
    };
    $('btnCloseExport').onclick = () => $('exportModal').classList.remove('open');
    $('exportModal').addEventListener('click', (e) => { if (e.target === $('exportModal')) $('exportModal').classList.remove('open'); });
    $('btnDownloadTexdsl').onclick = () => {
      const stem = $('exportStemInput').value.trim() || 'asset';
      triggerDownload(`${stem}.texdsl`, (state.serverState && state.serverState.texdsl) || '');
      showToast(`Downloaded ${stem}.texdsl`, 'ok');
    };
    $('btnDownloadRecipe').onclick = () => {
      const stem = $('exportStemInput').value.trim() || 'asset';
      triggerDownload(`${stem}.texgraph.json`, JSON.stringify(state.serverState.recipe, null, 2));
      showToast(`Downloaded ${stem}.texgraph.json`, 'ok');
    };
    $('btnRunExport').onclick = async () => {
      const stem = $('exportStemInput').value.trim() || 'asset';
      const outDir = $('exportDirInput').value.trim();
      const preset = $('exportPresetSelect').value;
      try {
        const res = await apiPost('/api/export', { stem, preset, out_dir: outDir || null, embed_manifest: true });
        const pre = $('exportResultPre');
        pre.style.display = 'block';
        if (res.mode === 'disk') {
          pre.textContent = `Exported ${Object.keys(res.files).length} files to:\n${res.out_dir}\n\n` + Object.keys(res.files).join(', ');
          showToast(`Exported PBR v2.0 bundle to disk!`, 'ok');
        } else {
          pre.textContent = `Prepared PBR v2.0 bundle (${res.file_keys.length} channels):\n${res.file_keys.join(', ')}\n\nDownloading ${stem}.synccraft.pbr.json...`;
          if (res.downloads && res.downloads[`${stem}.synccraft.pbr.json`]) {
            triggerDownload(`${stem}.synccraft.pbr.json`, res.downloads[`${stem}.synccraft.pbr.json`]);
          }
          showToast(`Exported ${stem}.synccraft.pbr.json (all PBR channels embedded)`, 'ok');
        }
      } catch (err) {
        showToast(err.message, 'err');
      }
    };

    // Global Pro Workstation Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      const tag = (document.activeElement && document.activeElement.tagName || '').toLowerCase();
      const inField = (tag === 'input' || tag === 'textarea' || tag === 'select');

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault(); openPalette(); return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        if (inField) return;
        e.preventDefault();
        mutate({ action: e.shiftKey ? 'redo' : 'undo' }, { immediateEval: true });
        return;
      }
      if (e.key === 'Escape') {
        closePalette();
        $('exportModal').classList.remove('open');
        $('meshModal').classList.remove('open');
        return;
      }
      if (inField) return;
      if (e.key === 'Tab') { e.preventDefault(); openPalette(); }
      else if (e.key.toLowerCase() === 'f') { e.preventDefault(); graphCanvas.fitToView(); }
      else if (e.key.toLowerCase() === 'l') { e.preventDefault(); mutate({ action: 'auto_layout' }, { skipEval: true, fitGraph: true }); }
      else if (e.key.toLowerCase() === 'b' && state.selectedNodeId) { e.preventDefault(); mutate({ action: 'toggle_bypass', node_id: state.selectedNodeId }, { immediateEval: true }); }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && state.selectedNodeId) {
        e.preventDefault();
        mutate({ action: 'delete_nodes', ids: [state.selectedNodeId] }, { immediateEval: true });
      }
    });
  }

  boot().catch((err) => {
    console.error('Studio boot failed:', err);
    showToast(`Boot error: ${err.message}`, 'err');
  });
})();
