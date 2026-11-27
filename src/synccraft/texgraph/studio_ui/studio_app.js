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
    animFps: 8,
    animPingPong: false,
    animDir: 1,
    animPlaying: false,
    animTimer: null,
    animBusy: false,
    lightAz: -38,
    lightEl: 52,
    lightingPreset: 'studio_hero',
    paletteFilterType: null,
    paletteConnectFrom: null,
    paletteSpawnPos: null,
    paletteItems: [],
    paletteIdx: 0,
    paletteViewMode: 'grid',
    paletteCategory: 'ALL',
    libViewMode: 'grid',
    nodePreviews: {},
    timeAwareOps: new Set(),
    animPreviews: {},
    palAnimTimer: null,
    palAnimOp: null,
    img2d: null,
    img2dBuf: null,
    lastDiagnosis: null,
    lastDna: null,
    lastPatchSummary: null,
    selectedStylePreset: 'preset:chronos_amethyst',
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
      if (state.evalData && (!ev.thumbs || !Object.keys(ev.thumbs).length)) {
        ev.thumbs = state.evalData.thumbs || {};
      }
      if (state.evalData && !ev.qa && state.evalData.qa) {
        ev.qa = state.evalData.qa;
      }
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
    const seedInp = $('graphSeedInput');
    if (seedInp && document.activeElement !== seedInp) {
      seedInp.value = String(rec.seed != null ? rec.seed : 42);
    }
    const resStr = String(rec.width);
    const resSel = $('resSelect');
    const meshResBadge = $('meshResBadge');
    if ([...resSel.options].some((o) => o.value === resStr)) resSel.value = resStr;
    if (st.has_mesh) {
      resSel.style.display = 'none';
      if (meshResBadge) {
        meshResBadge.style.display = 'inline-flex';
        meshResBadge.textContent = `${rec.width}×${rec.height} UV`;
      }
    } else {
      resSel.style.display = '';
      resSel.disabled = false;
      if (meshResBadge) meshResBadge.style.display = 'none';
    }

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
    renderAiVariantsPanel();
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
      const seams = ms ? (ms.seam_pairs >= 1000 ? `${(ms.seam_pairs / 1000).toFixed(1)}k` : `${ms.seam_pairs}`) : '';
      const mText = ms ? ` · ${ms.elements}b · ${seams} seams` : ` · ${$('primSelect').value}`;
      hud.textContent = `3D PBR · ${state.channel} · ${res}${mText}`;
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

    // Node Banner + Commercial Quick-Action Toolbar
    const banner = document.createElement('div');
    banner.className = 'insp-node-banner';
    const hasSeedParam = (meta.params || []).some((p) => p.name === 'seed' || p.widget === 'seed');
    const animTag = meta.time_aware ? '<span class="pill-badge warn" style="margin-left:4px;font-size:9px;">▶ ANIM</span>' : '';
    banner.innerHTML = `
      <h3><span>${node.id}${animTag}</span><span class="brand-sub">${node.op}</span></h3>
      <p>${meta.description || meta.category}</p>
      <div class="insp-action-bar">
        <button class="btn sm" id="inspBtnInspect" title="Preview this node directly in Viewport">👁 Inspect</button>
        <button class="btn sm" id="inspBtnDup" title="Duplicate Node (Ctrl/Cmd+D)">⧉ Dup</button>
        ${hasSeedParam ? '<button class="btn sm" id="inspBtnRandSeed" title="Randomize Node Seed">🎲 Seed</button>' : ''}
        <button class="btn sm${node.bypassed ? ' active' : ''}" id="inspBtnBypass" title="Bypass / Mute Node (B)">B</button>
        <button class="btn sm danger-hover" id="inspBtnDelete" title="Delete Node (Del)">✕ Del</button>
      </div>
    `;
    root.appendChild(banner);
    banner.querySelector('#inspBtnInspect').onclick = () => {
      state.channel = 'selected';
      $('channelSelect').value = 'selected';
      runEvaluate({ includeThumbs: false, includeQa: false });
      showToast(`Inspecting '${node.id}' in viewport`, 'ok');
    };
    banner.querySelector('#inspBtnDup').onclick = () => mutate({ action: 'duplicate_nodes', ids: [node.id] }, { immediateEval: true });
    const rsBtn = banner.querySelector('#inspBtnRandSeed');
    if (rsBtn) {
      rsBtn.onclick = () => mutate({ action: 'randomize_seed', node_id: node.id }, { immediateEval: true });
    }
    banner.querySelector('#inspBtnBypass').onclick = () => mutate({ action: 'toggle_bypass', node_id: node.id }, { immediateEval: true });
    banner.querySelector('#inspBtnDelete').onclick = () => mutate({ action: 'delete_nodes', ids: [node.id] }, { immediateEval: true });

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
        const locks = (state.serverState && state.serverState.artist_overrides) || {};
        if (`${node.id}.${p.name}` in locks) {
          const lk = document.createElement('span');
          lk.className = 'param-lock-badge';
          lk.textContent = 'LOCK';
          lk.title = 'Artist Override Locked — preserved during AI Patch & Variant Morph';
          lblRow.appendChild(lk);
        }

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
        } else if (p.widget === 'seed' || p.name === 'seed') {
          const sVal = Math.max(0, Math.round(Number(curVal != null ? curVal : (state.serverState.recipe.seed || 42))));
          const sWrap = document.createElement('div');
          sWrap.className = 'seed-control-row';
          const rng = document.createElement('input');
          rng.type = 'range';
          rng.min = '0';
          rng.max = '9999';
          rng.step = '1';
          rng.value = String(Math.min(9999, sVal));
          const num = document.createElement('input');
          num.type = 'number';
          num.className = 'num-input';
          num.min = '0';
          num.max = '99999';
          num.step = '1';
          num.value = String(sVal);
          const dice = document.createElement('button');
          dice.className = 'btn-dice';
          dice.textContent = '🎲';
          dice.title = 'Randomize Seed';
          rng.oninput = () => {
            const v = parseInt(rng.value, 10) || 0;
            num.value = String(v);
            mutate({ action: 'set_param', node_id: node.id, param: p.name, value: v, undoable: false }, { delayMs: 55 });
          };
          rng.onchange = () => {
            const v = parseInt(rng.value, 10) || 0;
            mutate({ action: 'set_param', node_id: node.id, param: p.name, value: v, undoable: true }, { immediateEval: true });
          };
          num.onchange = () => {
            const v = Math.max(0, Math.min(99999, parseInt(num.value, 10) || 0));
            rng.value = String(Math.min(9999, v));
            mutate({ action: 'set_param', node_id: node.id, param: p.name, value: v, undoable: true }, { immediateEval: true });
          };
          dice.onclick = () => {
            const rnd = Math.floor(1 + Math.random() * 99998);
            num.value = String(rnd);
            rng.value = String(Math.min(9999, rnd));
            mutate({ action: 'set_param', node_id: node.id, param: p.name, value: rnd, undoable: true }, { immediateEval: true });
            showToast(`${node.id} seed -> ${rnd}`, 'ok');
          };
          sWrap.appendChild(rng);
          sWrap.appendChild(num);
          sWrap.appendChild(dice);
          ctrl.appendChild(sWrap);
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

          // Clickable 3D Model Part / Zone Quick-Picker for `part_pattern`!
          if (p.name === 'part_pattern' && state.serverState && state.serverState.mesh_parts) {
            const mp = state.serverState.mesh_parts;
            const chips = document.createElement('div');
            chips.className = 'part-chips-wrap';
            const curStr = String(curVal || '');
            const mkChip = (label, pat, extraCls) => {
              const b = document.createElement('button');
              b.type = 'button';
              b.className = 'part-chip' + (extraCls ? ' ' + extraCls : '') + (curStr === pat ? ' active' : '');
              b.textContent = label;
              b.title = pat ? `Click: select '${pat}' · Shift+Click: append |${pat}` : 'Select entire model';
              b.onclick = (ev) => {
                let next = pat;
                if (ev.shiftKey && curStr && pat) {
                  const parts = curStr.split('|').filter(Boolean);
                  if (!parts.includes(pat)) parts.push(pat);
                  next = parts.join('|');
                }
                inp.value = next;
                mutate({ action: 'set_param', node_id: node.id, param: p.name, value: next }, { immediateEval: true });
              };
              chips.appendChild(b);
            };
            mkChip('All', '', 'zone');
            for (const z of (mp.zones || [])) mkChip(`zone:${z}`, z, 'zone');
            for (const elName of (mp.elements || []).slice(0, 26)) mkChip(elName, elName, '');
            row.appendChild(ctrl);
            row.appendChild(chips);
            body.appendChild(row);
            continue;
          }
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
    const mk = (id, label, tip) => {
      const b = document.createElement('button');
      b.className = 'btn sm' + (state.activePass === id ? ' active' : '');
      b.textContent = label;
      if (tip) b.title = tip;
      b.onclick = () => {
        state.activePass = id;
        renderPassFilterBar();
        graphCanvas.setData(state.serverState.recipe, state.evalData, state.selectedNodeId, state.activePass);
      };
      bar.appendChild(b);
    };
    mk('ALL', 'All', 'Show all passes');
    passes.forEach((p, idx) => {
      const fullTitle = String(p.title || p.name);
      mk(p.name, `P${idx + 1}`, fullTitle);
    });
  }

  const LIB_CAT_COLORS = {
    'Gradients & Paths': '#818cf8',
    'Noise': '#38bdf8',
    'Geometric Patterns': '#34d399',
    '3D Model & UV Masks': '#14b8a6',
    '3D Mesh Baker': '#06b6d4',
    'Materials — Metal (PBR)': '#eab308',
    'Materials — Dielectric & Glass (PBR)': '#38bdf8',
    'Painterly & Bio': '#2dd4bf',
    'Region & Per-UV Pixel Composer': '#c084fc',
    'Pixel Composer FX': '#fb923c',
    'Material Bus (PBR)': '#f59e0b',
    'Character & Gear (PBR)': '#a78bfa',
    'World Blocks (PBR)': '#4ade80',
    'PBR & Normal': '#60a5fa',
    'Color & Grading': '#f472b6',
    'Denoise & Polish': '#a3e635',
    'Transform & Distort': '#fbbf24',
    'Filter & SDF': '#67e8f9',
    'Simulation': '#fb7185',
    'Animation': '#f43f5e',
    'Compositing': '#94a3b8',
    'Sprites & Masks': '#e879f9',
    'Scatter & Layout': '#f97316',
  };

  function getNodePreviewSrc(op) {
    return (state.nodePreviews && state.nodePreviews[op]) || '';
  }

  function isOpAnimated(op) {
    if (state.timeAwareOps && state.timeAwareOps.has(op)) return true;
    return !!(state.opMap[op] && state.opMap[op].time_aware);
  }

  function stopPaletteAnimLoop() {
    if (state.palAnimTimer) {
      clearInterval(state.palAnimTimer);
      state.palAnimTimer = null;
    }
    state.palAnimOp = null;
  }

  function startPaletteAnimLoop(op) {
    stopPaletteAnimLoop();
    const badge = $('palPrevAnimBadge');
    const img = $('palPrevImg');
    if (!isOpAnimated(op)) {
      if (badge) badge.style.display = 'none';
      return;
    }
    state.palAnimOp = op;
    if (badge) {
      badge.style.display = 'inline-block';
      badge.textContent = '▶ LIVE ANIM';
    }
    const playFrames = (frames) => {
      if (state.palAnimOp !== op || !frames || !frames.length) return;
      let fIdx = 0;
      if (img) img.src = frames[0];
      state.palAnimTimer = setInterval(() => {
        if (state.palAnimOp !== op || !$('paletteModal').classList.contains('open')) {
          stopPaletteAnimLoop();
          return;
        }
        fIdx = (fIdx + 1) % frames.length;
        if (img) img.src = frames[fIdx];
      }, 150);
    };
    if (state.animPreviews[op]) {
      playFrames(state.animPreviews[op]);
    } else {
      apiGet(`/api/schema/preview_anim?op=${encodeURIComponent(op)}&size=48&frames=8`)
        .then((res) => {
          if (res && Array.isArray(res.images) && res.images.length) {
            state.animPreviews[op] = res.images;
            playFrames(res.images);
          }
        })
        .catch(() => {});
    }
  }

  function renderNodeLibrary() {
    const body = $('libBody');
    const q = ($('libSearch').value || '').trim().toLowerCase();
    body.innerHTML = '';
    if (!state.schemaPayload) return;
    const isGrid = state.libViewMode === 'grid';
    if ($('btnLibGrid')) $('btnLibGrid').classList.toggle('active', isGrid);
    if ($('btnLibList')) $('btnLibList').classList.toggle('active', !isGrid);

    const cats = state.schemaPayload.schema.nodes_by_category || {};
    let totalShown = 0;

    for (const [cat, nodes] of Object.entries(cats)) {
      const filtered = q
        ? nodes.filter((n) => n.op.toLowerCase().includes(q) || (n.display_name || '').toLowerCase().includes(q) || (n.description || '').toLowerCase().includes(q) || cat.toLowerCase().includes(q))
        : nodes;
      if (!filtered.length) continue;
      totalShown += filtered.length;

      const wrap = document.createElement('div');
      wrap.className = 'lib-cat';
      const head = document.createElement('div');
      head.className = 'lib-cat-head';
      const catColor = LIB_CAT_COLORS[cat] || '#64748b';
      head.innerHTML = `<span style="display:inline-flex;align-items:center;gap:5px;"><span class="cat-chip-dot" style="background:${catColor};"></span>${cat}</span><span style="color:var(--text-muted);font-family:var(--font-mono);">${filtered.length}</span>`;
      const list = document.createElement('div');
      list.className = 'lib-items' + (isGrid ? '' : ' list-mode');
      const defaultOpen = !!q || cat === 'Noise' || cat === '3D Model & UV Masks' || cat === 'Materials — Metal (PBR)' || cat === 'Materials — Dielectric & Glass (PBR)' || cat.includes('Region');
      list.style.display = defaultOpen ? (isGrid ? 'grid' : 'flex') : 'none';
      head.onclick = () => { list.style.display = list.style.display === 'none' ? (isGrid ? 'grid' : 'flex') : 'none'; };

      for (const n of filtered) {
        const it = document.createElement('div');
        it.className = 'lib-item' + (isGrid ? ' grid-card' : '');
        it.title = `${n.display_name || n.op} (${n.op}) — ${n.description || ''}`;
        it.style.setProperty('--item-accent', catColor);
        const outType = n.output_type || (n.outputs && n.outputs[0] && (n.outputs[0].dtype || n.outputs[0].port_type)) || 'RGBA';
        const col = (state.schemaPayload.schema.port_colors || {})[outType] || '#94a3b8';
        const prevSrc = getNodePreviewSrc(n.op);
        const isAnim = isOpAnimated(n.op);
        if (isGrid) {
          it.innerHTML = `
            <div class="lib-swatch">${prevSrc ? `<img src="${prevSrc}" alt="${n.op}" loading="lazy" />` : ''}${isAnim ? '<span class="anim-pill-mini">▶</span>' : ''}</div>
            <div class="lib-item-name">${n.op}</div>
          `;
        } else {
          it.innerHTML = `
            <div class="lib-swatch-sm">${prevSrc ? `<img src="${prevSrc}" alt="${n.op}" loading="lazy" />` : ''}</div>
            <span class="lib-item-name">${isAnim ? '<span class="tg-anim-dot">▶</span>' : ''}${n.op}</span>
            <span class="lib-dot" style="background:${col};" title="${outType}"></span>
          `;
        }
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

  function updatePalettePreview(item) {
    const img = $('palPrevImg');
    const title = $('palPrevTitle');
    const opEl = $('palPrevOp');
    const catEl = $('palPrevCat');
    const descEl = $('palPrevDesc');
    const portsEl = $('palPrevPorts');
    const addBtn = $('btnPalAddSelected');
    if (!item) {
      stopPaletteAnimLoop();
      if ($('palPrevAnimBadge')) $('palPrevAnimBadge').style.display = 'none';
      if (img) img.src = '';
      if (title) title.textContent = 'No matching nodes';
      if (opEl) opEl.textContent = '';
      if (catEl) catEl.textContent = '';
      if (descEl) descEl.textContent = 'Try clearing your search or switching category.';
      if (portsEl) portsEl.innerHTML = '';
      if (addBtn) addBtn.disabled = true;
      return;
    }
    const meta = state.opMap[item.op] || item;
    const pColors = (state.schemaPayload && state.schemaPayload.schema.port_colors) || {};
    if (img) img.src = getNodePreviewSrc(item.op);
    startPaletteAnimLoop(item.op);
    if (title) title.textContent = meta.display_name || item.op;
    if (opEl) opEl.textContent = item.op;
    if (catEl) catEl.textContent = item.cat;
    if (descEl) descEl.textContent = meta.description || item.desc || '';
    if (portsEl) {
      portsEl.innerHTML = '';
      const ins = meta.inputs || [];
      const outT = meta.output_type || 'RGBA';
      for (const inp of ins.slice(0, 4)) {
        const dt = inp.dtype || inp.port_type || 'ANY';
        const tag = document.createElement('span');
        tag.className = 'pal-port-tag';
        tag.innerHTML = `<span class="cat-chip-dot" style="background:${pColors[dt] || '#94a3b8'};"></span>in:${inp.name}`;
        portsEl.appendChild(tag);
      }
      const oTag = document.createElement('span');
      oTag.className = 'pal-port-tag';
      oTag.innerHTML = `<span class="cat-chip-dot" style="background:${pColors[outT] || '#94a3b8'};"></span>out:${outT}`;
      portsEl.appendChild(oTag);
    }
    if (addBtn) {
      addBtn.disabled = false;
      addBtn.onclick = () => spawnFromPalette(item.op);
    }
  }

  function renderPaletteCategories() {
    const bar = $('paletteCatBar');
    if (!bar || !state.schemaPayload) return;
    bar.innerHTML = '';
    const cats = Object.keys(state.schemaPayload.schema.nodes_by_category || {});
    const mk = (id, label, color) => {
      const b = document.createElement('button');
      b.className = 'cat-chip' + (state.paletteCategory === id ? ' active' : '');
      b.innerHTML = color ? `<span class="cat-chip-dot" style="background:${color};"></span>${label}` : label;
      b.onclick = () => {
        state.paletteCategory = id;
        renderPaletteCategories();
        renderPaletteList();
      };
      bar.appendChild(b);
    };
    mk('ALL', 'All Nodes', null);
    for (const c of cats) {
      const shortLbl = c
        .replace('Region & Per-UV Pixel Composer', 'Region / Per-UV')
        .replace('Geometric Patterns', 'Patterns')
        .replace('Material Bus (PBR)', 'Material Bus')
        .replace('Materials — Metal (PBR)', 'Metals (PBR)')
        .replace('Materials — Dielectric & Glass (PBR)', 'Glass & Plastic')
        .replace('3D Model & UV Masks', '3D & UV Masks');
      mk(c, shortLbl, LIB_CAT_COLORS[c] || '#94a3b8');
    }
  }

  function openPalette(opts) {
    opts = opts || {};
    state.paletteFilterType = opts.filterType || null;
    state.paletteConnectFrom = opts.connectFrom || null;
    state.paletteSpawnPos = opts.spawnPos || null;
    state.paletteCategory = opts.category || 'ALL';
    $('paletteModal').classList.add('open');
    const inp = $('paletteInput');
    inp.value = '';
    const nCount = (state.schemaPayload && state.schemaPayload.schema && state.schemaPayload.schema.node_count) || 210;
    inp.placeholder = state.paletteFilterType
      ? `Add node compatible with ${state.paletteFilterType} output...`
      : `Search ${nCount} TexGraph nodes by name, noise, brass, crystal, mask or anim...`;
    const cBadge = $('paletteCompatBadge');
    if (cBadge) {
      if (state.paletteFilterType) {
        cBadge.style.display = 'inline-flex';
        cBadge.textContent = `Compat: ${state.paletteFilterType}`;
      } else {
        cBadge.style.display = 'none';
      }
    }
    renderPaletteCategories();
    renderPaletteList();
    setTimeout(() => inp.focus(), 10);
  }

  function closePalette() {
    stopPaletteAnimLoop();
    $('paletteModal').classList.remove('open');
    state.paletteFilterType = null;
    state.paletteConnectFrom = null;
  }

  function renderPaletteList() {
    const q = ($('paletteInput').value || '').trim().toLowerCase();
    const list = $('paletteList');
    list.innerHTML = '';
    if (!state.schemaPayload) return;

    const isGrid = state.paletteViewMode === 'grid';
    list.className = 'palette-results ' + (isGrid ? 'grid-mode' : 'list-mode');
    if ($('btnPalGrid')) $('btnPalGrid').classList.toggle('active', isGrid);
    if ($('btnPalList')) $('btnPalList').classList.toggle('active', !isGrid);

    const compatSet = state.paletteFilterType
      ? new Set(((state.schemaPayload.schema.port_compatibility) || {})[state.paletteFilterType] || [state.paletteFilterType])
      : null;

    const items = [];
    for (const [cat, nodes] of Object.entries(state.schemaPayload.schema.nodes_by_category || {})) {
      if (state.paletteCategory !== 'ALL' && cat !== state.paletteCategory) continue;
      for (const n of nodes) {
        if (compatSet) {
          const hasCompatIn = (n.inputs || []).some((inp) => compatSet.has(inp.dtype || inp.port_type));
          if (!hasCompatIn) continue;
        }
        if (q && !(n.op.toLowerCase().includes(q) || (n.display_name || '').toLowerCase().includes(q) || cat.toLowerCase().includes(q) || (n.description || '').toLowerCase().includes(q))) {
          continue;
        }
        items.push({
          op: n.op,
          display_name: n.display_name || n.op,
          cat,
          desc: n.description || '',
          output_type: n.output_type || 'RGBA',
        });
      }
    }
    state.paletteItems = items;
    state.paletteIdx = Math.min(state.paletteIdx, Math.max(0, items.length - 1));
    if ($('paletteCountLabel')) {
      $('paletteCountLabel').textContent = `${items.length} node${items.length === 1 ? '' : 's'} shown`;
    }

    const pColors = state.schemaPayload.schema.port_colors || {};
    state.paletteItems.forEach((it, idx) => {
      const prevSrc = getNodePreviewSrc(it.op);
      const accent = LIB_CAT_COLORS[it.cat] || '#64748b';
      const isAnim = isOpAnimated(it.op);
      const el = document.createElement('div');
      if (isGrid) {
        el.className = 'palette-card' + (idx === state.paletteIdx ? ' sel' : '');
        el.style.setProperty('--card-accent', accent);
        el.innerHTML = `
          <div class="palette-card-thumb">
            ${prevSrc ? `<img src="${prevSrc}" alt="${it.op}" loading="lazy" />` : ''}
            ${isAnim ? '<span class="anim-pill-mini">▶ ANIM</span>' : ''}
            <span class="palette-port-pill" style="color:${pColors[it.output_type] || '#cbd5e1'};">${it.output_type}</span>
          </div>
          <div class="palette-card-title" title="${it.display_name}">${it.display_name}</div>
          <div class="palette-card-op">${it.op}</div>
        `;
      } else {
        el.className = 'palette-item' + (idx === state.paletteIdx ? ' sel' : '');
        el.innerHTML = `
          <div class="palette-list-thumb">${prevSrc ? `<img src="${prevSrc}" alt="${it.op}" loading="lazy" />` : ''}</div>
          <div class="palette-item-main">
            <span class="palette-item-name">${isAnim ? '<span class="tg-anim-dot">▶</span>' : ''}${it.display_name} <span style="font-family:var(--font-mono);font-size:10px;color:var(--text-muted);font-weight:400;">${it.op}</span></span>
            <span class="palette-item-desc">${it.desc}</span>
          </div>
          <span class="palette-item-cat">${it.cat}</span>
        `;
      }
      el.onmouseenter = () => {
        state.paletteIdx = idx;
        list.querySelectorAll('.palette-card, .palette-item').forEach((n, i) => n.classList.toggle('sel', i === idx));
        updatePalettePreview(it);
      };
      el.onclick = () => spawnFromPalette(it.op);
      list.appendChild(el);
    });

    updatePalettePreview(state.paletteItems[state.paletteIdx] || null);
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
  // Milestone v12: AI Co-Pilot, Style-DNA Extractor & Live 3D Variant Factory
  // ------------------------------------------------------------------------
  function renderAiVariantsPanel() {
    const root = $('aiPanelBody');
    if (!root) return;
    root.innerHTML = '';

    const st = state.serverState || {};
    const locks = st.artist_overrides || {};
    const lockKeys = Object.keys(locks);
    const variants = (state.schemaPayload && state.schemaPayload.variants) || [];
    const stylePresets = (state.schemaPayload && state.schemaPayload.style_presets) || [];
    const activeVar = st.active_variant || 'chronos_prime';

    // 1. AI Co-Pilot Diagnosis + Human-in-the-Loop Safe Surgical Patching
    const allSuggestedOps = [];
    if (state.lastDiagnosis && Array.isArray(state.lastDiagnosis.findings)) {
      for (const f of state.lastDiagnosis.findings) {
        if (Array.isArray(f.patch_ops)) allSuggestedOps.push(...f.patch_ops);
      }
    }
    const secAi = document.createElement('div');
    secAi.className = 'ai-section';
    secAi.innerHTML = `
      <div class="ai-section-head">
        <span>AI Co-Pilot (Pure Algorithmic)</span>
        <span class="pill-badge ${lockKeys.length ? 'warn' : ''}" id="artistLocksBadge">${lockKeys.length} Lock${lockKeys.length === 1 ? '' : 's'}</span>
      </div>
      <div style="font-size:10.5px;color:var(--text-secondary);">
        Audits QA, 3D ray-traced shadows &amp; UV seam hygiene, proposing surgical node patches while strictly protecting human artist overrides.
      </div>
      <div style="display:flex;gap:5px;flex-wrap:wrap;">
        <button class="btn sm primary" id="btnAiDiagnose">Run AI Diagnosis</button>
        <button class="btn sm amber" id="btnAiApplyPatch" ${!allSuggestedOps.length ? 'disabled' : ''}>Apply Safe Patch (${allSuggestedOps.length})</button>
        <button class="btn sm" id="btnClearLocks" ${!lockKeys.length ? 'disabled' : ''}>Clear Locks</button>
      </div>
    `;
    if (lockKeys.length) {
      const chipWrap = document.createElement('div');
      chipWrap.className = 'lock-chip-list';
      for (const k of lockKeys.slice(0, 8)) {
        const c = document.createElement('span');
        c.className = 'lock-chip';
        const v = locks[k];
        c.textContent = `LOCK ${k}=${typeof v === 'number' ? Number(v.toFixed(2)) : String(v).slice(0, 14)}`;
        chipWrap.appendChild(c);
      }
      secAi.appendChild(chipWrap);
    }
    if (state.lastPatchSummary) {
      const ps = document.createElement('div');
      ps.className = 'ai-finding ok';
      const presNames = (state.lastPatchSummary.preserved_overrides || []).map((o) => (typeof o === 'string' ? o : `${o.node_id}.${o.param}`));
      const lockStatus = lockKeys.length
        ? `${lockKeys.length} lock(s) safe${presNames.length ? ` (${presNames.length} conflict blocked)` : ''}`
        : '0 locks active';
      ps.innerHTML = `<div class="ai-finding-title"><span>Last Patch Applied</span><span>QA ${state.lastPatchSummary.qa_before} → ${state.lastPatchSummary.qa_after}</span></div><div>Applied ${state.lastPatchSummary.applied_count} op(s) · ${lockStatus}</div>`;
      secAi.appendChild(ps);
    }
    if (state.lastDiagnosis) {
      const d = state.lastDiagnosis;
      if (!d.findings || !d.findings.length) {
        const okBox = document.createElement('div');
        okBox.className = 'ai-finding ok';
        okBox.innerHTML = `<div class="ai-finding-title"><span>Studio Asset Healthy</span><span>QA ${Math.round(d.qa.score)}/100</span></div><div>Zero QA, shadow or UV-seam issues detected.</div>`;
        secAi.appendChild(okBox);
      } else {
        for (const f of d.findings.slice(0, 6)) {
          const sev = f.severity || 'ok';
          const box = document.createElement('div');
          box.className = 'ai-finding ' + (sev === 'error' ? 'error' : ((sev === 'warning' || sev === 'warn') ? 'warn' : 'ok'));
          box.innerHTML = `<div class="ai-finding-title"><span>${f.title || f.id || 'Finding'}</span><span>${sev}</span></div><div>${f.detail || f.message || ''}</div>`;
          secAi.appendChild(box);
        }
      }
    }
    root.appendChild(secAi);

    // 2. Pure-Algorithmic Style-DNA Extractor (Presets or Reference PNG/JPG)
    const secDna = document.createElement('div');
    secDna.className = 'ai-section';
    const presetOpts = stylePresets
      .map((p) => {
        const val = `preset:${p.id}`;
        const short = (p.title || p.id).split('(')[0].trim();
        return `<option value="${val}"${state.selectedStylePreset === val ? ' selected' : ''}>${short}</option>`;
      })
      .join('');
    secDna.innerHTML = `
      <div class="ai-section-head">
        <span>Style-DNA Palette Extractor</span>
        <span style="font-size:10px;color:var(--text-muted);font-family:var(--font-mono);">OKLCH</span>
      </div>
      <div style="display:flex;gap:5px;align-items:center;">
        <select id="stylePresetSelect" class="sel-input" style="flex:1;height:24px;">${presetOpts}</select>
        <button class="btn sm" id="btnExtractPresetDna">Extract</button>
      </div>
      <div style="display:flex;gap:5px;flex-wrap:wrap;">
        <button class="btn sm" id="btnUploadStyleRef">From Ref Image...</button>
        <button class="btn sm primary" id="btnApplyDna" ${!state.lastDna ? 'disabled' : ''}>Inject Palette (Safe)</button>
      </div>
    `;
    if (state.lastDna && state.lastDna.stops && state.lastDna.stops.length) {
      const dna = state.lastDna;
      const bar = document.createElement('div');
      bar.className = 'dna-swatch-bar';
      bar.style.background = `linear-gradient(90deg, ${dna.stops.join(', ')})`;
      const info = document.createElement('div');
      info.style.cssText = 'font-family:var(--font-mono);font-size:10px;color:var(--text-secondary);';
      const hueInfo = dna.stats && dna.stats.hue_shift_deg != null ? `ΔH ${Math.round(dna.stats.hue_shift_deg)}°` : 'OKLCH';
      info.textContent = `${dna.name} · ${hueInfo} · ${dna.stops.join(' ')}`;
      secDna.appendChild(bar);
      secDna.appendChild(info);
    }
    root.appendChild(secDna);

    // 3. Live 3D Chronal Variant Factory + Unified .syncproj Export
    const secVar = document.createElement('div');
    secVar.className = 'ai-section';
    secVar.innerHTML = `
      <div class="ai-section-head">
        <span>Live 3D Chronal Variants</span>
        <span style="font-size:10px;color:var(--text-muted);font-family:var(--font-mono);">${activeVar}</span>
      </div>
      <div class="variant-grid" id="variantGrid"></div>
      <div style="display:flex;gap:5px;margin-top:2px;">
        <button class="btn sm primary" id="btnExportSyncproj" style="flex:1;">Export Unified .syncproj</button>
      </div>
    `;
    const vGrid = secVar.querySelector('#variantGrid');
    for (const v of variants) {
      const b = document.createElement('button');
      b.className = 'variant-btn' + (activeVar === v.id ? ' active' : '');
      b.dataset.variantId = v.id;
      const shortTitle = (v.title || v.id).split('(')[0].trim();
      b.innerHTML = `
        <div class="variant-dots">
          <span class="variant-dot" style="background:${v.eye_iris || '#a855f7'};"></span>
          <span class="variant-dot" style="background:${v.rune_color || '#38f2ff'};"></span>
          <span style="font-size:11px;font-weight:600;">${shortTitle}</span>
        </div>
        <span style="font-size:9.5px;color:var(--text-muted);font-family:var(--font-mono);">${v.id}</span>
      `;
      b.onclick = async () => {
        try {
          const res = await apiPost('/api/variants/apply', { variant_id: v.id });
          if (res.state) applyServerState(res.state, { autoPrim: false });
          await runEvaluate({ includeThumbs: true, includeQa: true });
          showToast(`Morphed live 3D variant -> ${shortTitle}`, 'ok');
        } catch (err) {
          showToast(err.message, 'err');
        }
      };
      vGrid.appendChild(b);
    }
    root.appendChild(secVar);

    // Bind AI & DNA buttons
    $('btnAiDiagnose').onclick = async () => {
      try {
        const rep = await apiPost('/api/ai/diagnose', { phase: state.phase });
        state.lastDiagnosis = rep;
        renderAiVariantsPanel();
        showToast(`AI Co-Pilot: ${rep.finding_count} finding(s) · QA ${Math.round(rep.qa.score)}/100`, 'ok');
      } catch (err) {
        showToast(err.message, 'err');
      }
    };
    $('btnAiApplyPatch').onclick = async () => {
      if (!allSuggestedOps.length) return;
      try {
        const res = await apiPost('/api/ai/patch', { patch_ops: allSuggestedOps, preserve_artist_overrides: true });
        state.lastPatchSummary = res.patch_result;
        if (res.state) applyServerState(res.state, { autoPrim: false });
        await runEvaluate({ includeThumbs: true, includeQa: true });
        showToast(`Safe Patch applied (${res.patch_result.preserved_overrides.length} artist lock(s) kept!)`, 'ok');
      } catch (err) {
        showToast(err.message, 'err');
      }
    };
    $('btnClearLocks').onclick = () => {
      mutate({ action: 'clear_artist_overrides' }, { skipEval: true });
      showToast('Cleared artist override locks', 'ok');
    };
    $('stylePresetSelect').onchange = (e) => { state.selectedStylePreset = e.target.value; };
    $('btnExtractPresetDna').onclick = async () => {
      const src = $('stylePresetSelect').value;
      state.selectedStylePreset = src;
      try {
        const res = await apiPost('/api/style/extract', { source: src, name: src.replace('preset:', ''), apply: false });
        state.lastDna = res.dna;
        renderAiVariantsPanel();
        showToast(`Extracted OKLCH Style-DNA (${res.dna.stops.length} stops)`, 'ok');
      } catch (err) {
        showToast(err.message, 'err');
      }
    };
    $('btnUploadStyleRef').onclick = () => $('styleRefFileInput').click();
    $('btnApplyDna').onclick = async () => {
      if (!state.lastDna) return;
      try {
        const src = $('stylePresetSelect') ? $('stylePresetSelect').value : 'preset:chronos_amethyst';
        const res = await apiPost('/api/ai/patch', {
          patch_ops: [{ op: 'replace_palette', stops: state.lastDna.stops }],
          preserve_artist_overrides: true,
        });
        state.lastPatchSummary = res.patch_result;
        if (res.state) applyServerState(res.state, { autoPrim: false });
        await runEvaluate({ includeThumbs: true, includeQa: true });
        showToast(`Injected Style-DNA '${state.lastDna.name}' (locks preserved)`, 'ok');
      } catch (err) {
        showToast(err.message, 'err');
      }
    };
    $('btnExportSyncproj').onclick = async () => {
      try {
        const stem = (state.serverState && state.serverState.recipe && state.serverState.recipe.name) || 'kaelith_chrono_dragonborn';
        const res = await apiPost('/api/syncproj/export', { stem, preset: state.lightingPreset });
        triggerDownload(res.filename || `${stem}.syncproj`, JSON.stringify(res.manifest, null, 2));
        showToast(`Exported unified ${res.filename || stem + '.syncproj'}`, 'ok');
      } catch (err) {
        showToast(err.message, 'err');
      }
    };
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
    $('rightTabAi').style.display = tab === 'ai' ? 'flex' : 'none';
    if (tab === 'ai') renderAiVariantsPanel();
  }

  function switchBottomTab(tab) {
    document.querySelectorAll('[data-bot-tab]').forEach((b) => b.classList.toggle('active', b.dataset.botTab === tab));
    $('botTabTimeline').style.display = tab === 'timeline' ? 'flex' : 'none';
    $('botTabExposed').style.display = tab === 'exposed' ? 'flex' : 'none';
    $('botTabConsole').style.display = tab === 'console' ? 'flex' : 'none';
  }

  // ------------------------------------------------------------------------
  // Floating Right-Click Context Menus (Nodes & Graph Canvas)
  // ------------------------------------------------------------------------
  function hideContextMenu() {
    const m = $('graphContextMenu');
    if (m) m.style.display = 'none';
  }

  function showContextMenu(x, y, title, items) {
    const m = $('graphContextMenu');
    if (!m) return;
    m.innerHTML = '';
    if (title) {
      const t = document.createElement('div');
      t.className = 'ctx-menu-title';
      t.textContent = title;
      m.appendChild(t);
    }
    for (const it of items) {
      if (it.sep) {
        const s = document.createElement('div');
        s.className = 'ctx-menu-sep';
        m.appendChild(s);
        continue;
      }
      const b = document.createElement('button');
      b.className = 'ctx-menu-item' + (it.danger ? ' danger' : '');
      b.innerHTML = `<span>${it.label}</span>${it.kbd ? `<span class="ctx-kbd">${it.kbd}</span>` : ''}`;
      b.onclick = () => {
        hideContextMenu();
        it.action();
      };
      m.appendChild(b);
    }
    m.style.display = 'block';
    const mw = m.offsetWidth || 200;
    const mh = m.offsetHeight || 220;
    const left = Math.min(window.innerWidth - mw - 10, Math.max(8, x));
    const top = Math.min(window.innerHeight - mh - 10, Math.max(8, y));
    m.style.left = `${left}px`;
    m.style.top = `${top}px`;
  }

  // ------------------------------------------------------------------------
  // Animation Playback Controls (Sequential No-Stutter Live Transport)
  // ------------------------------------------------------------------------
  function setPhase(ph, immediate) {
    state.phase = ((ph % 1) + 1) % 1;
    $('phaseSlider').value = state.phase.toFixed(2);
    $('phaseLabel').textContent = state.phase.toFixed(2);
    const fIdx = Math.floor(state.phase * state.animFrames);
    $('frameReadout').textContent = `F${fIdx}/${state.animFrames} · φ${state.phase.toFixed(2)}`;
    renderTimelineDrivers();
    if (immediate) return runEvaluate({ includeThumbs: false, includeQa: false });
    scheduleEvaluate(20);
    return Promise.resolve();
  }

  function stopAnimLoop() {
    state.animPlaying = false;
    $('btnPlayAnim').textContent = '▶ Play';
    $('btnPlayAnim').classList.remove('amber');
    if (state.animTimer) { clearTimeout(state.animTimer); state.animTimer = null; }
  }

  function togglePlayAnim() {
    if (state.animPlaying) {
      stopAnimLoop();
      return;
    }
    state.animPlaying = true;
    $('btnPlayAnim').textContent = '⏸ Pause';
    $('btnPlayAnim').classList.add('amber');
    const tick = async () => {
      if (!state.animPlaying) return;
      const step = 1.0 / Math.max(2, state.animFrames);
      let nextPh = state.phase + step * state.animDir;
      if (state.animPingPong) {
        if (nextPh >= 1.0 - 1e-4) { state.animDir = -1; nextPh = Math.max(0.0, 1.0 - step); }
        else if (nextPh <= 1e-4) { state.animDir = 1; nextPh = 0.0; }
      }
      const t0 = performance.now();
      await setPhase(nextPh, true);
      if (!state.animPlaying) return;
      const targetMs = Math.max(40, Math.round(1000 / Math.max(2, state.animFps)));
      const waitMs = Math.max(16, targetMs - (performance.now() - t0));
      state.animTimer = setTimeout(tick, waitMs);
    };
    tick();
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
        onOpenAddAt: (wPos) => openPalette({ spawnPos: [Math.round(wPos.x), Math.round(wPos.y)] }),
        onDeleteNode: (nid) => {
          mutate({ action: 'delete_nodes', ids: [nid] }, { immediateEval: true });
          showToast(`Deleted node '${nid}'`, 'ok');
        },
        onDuplicateNode: (nid) => mutate({ action: 'duplicate_nodes', ids: [nid] }, { immediateEval: true }),
        onNodeContextMenu: (nid, cx, cy) => {
          const nObj = ((state.serverState && state.serverState.recipe && state.serverState.recipe.nodes) || []).find((q) => q.id === nid);
          const meta = nObj ? state.opMap[nObj.op] : null;
          const items = [
            { label: '👁 Inspect in Viewport', kbd: 'Dbl-Clk', action: () => { state.channel = 'selected'; $('channelSelect').value = 'selected'; runEvaluate({ includeThumbs: false, includeQa: false }); } },
            { label: '★ Set Final Graph Output', action: () => mutate({ action: 'set_output', node_id: nid }, { immediateEval: true }) },
            { sep: true },
            { label: '⧉ Duplicate Node', kbd: '⌘D', action: () => mutate({ action: 'duplicate_nodes', ids: [nid] }, { immediateEval: true }) },
            { label: '🎲 Randomize Seed', action: () => mutate({ action: 'randomize_seed', node_id: nid }, { immediateEval: true }) },
            { label: nObj && nObj.bypassed ? '✓ Enable Node' : 'B Bypass / Mute', kbd: 'B', action: () => mutate({ action: 'toggle_bypass', node_id: nid }, { immediateEval: true }) },
            { label: '✂ Disconnect All Wires', action: () => mutate({ action: 'disconnect_all', node_id: nid }, { immediateEval: true }) },
            { sep: true },
            { label: '✕ Delete Node', kbd: 'Del', danger: true, action: () => { mutate({ action: 'delete_nodes', ids: [nid] }, { immediateEval: true }); showToast(`Deleted '${nid}'`, 'ok'); } },
          ];
          showContextMenu(cx, cy, `${nid}${meta ? ` (${meta.op})` : ''}`, items);
        },
        onCanvasContextMenu: (cx, cy, wPos) => {
          showContextMenu(cx, cy, 'Node Graph Canvas', [
            { label: '+ Add Node Here...', kbd: 'Tab', action: () => openPalette({ spawnPos: [Math.round(wPos.x), Math.round(wPos.y)] }) },
            { sep: true },
            { label: '⊞ Auto-Layout Graph', kbd: 'L', action: () => mutate({ action: 'auto_layout' }, { skipEval: true, fitGraph: true }) },
            { label: '⌖ Fit All Nodes', kbd: 'F', action: () => graphCanvas.fitToView() },
            { label: '🎲 Randomize Master Seed', action: () => mutate({ action: 'randomize_seed' }, { immediateEval: true }) },
          ]);
        },
        onToggleBypass: (nid) => mutate({ action: 'toggle_bypass', node_id: nid }, { immediateEval: true }),
        onToggleCollapse: (nid) => mutate({ action: 'toggle_collapse', node_id: nid }, { skipEval: true }),
        onSetOutput: (nid) => mutate({ action: 'set_output', node_id: nid }, { immediateEval: true }),
      },
    });
    window.addEventListener('pointerdown', (e) => {
      if (!e.target.closest('#graphContextMenu')) hideContextMenu();
    });

    // Focus outline tracking (Pixel Composer convention!)
    document.querySelectorAll('.pane').forEach((p) => {
      p.addEventListener('pointerdown', () => {
        document.querySelectorAll('.pane').forEach((q) => q.classList.remove('focused'));
        p.classList.add('focused');
      });
    });

    setupSplitters();

    // Load Schema v4.0 + Initial Workspace State + Real Procedural Node Previews in parallel!
    const [schemaPayload, initState, prevPayload] = await Promise.all([
      apiGet('/api/schema'),
      apiGet('/api/state'),
      apiGet('/api/schema/previews?size=40').catch(() => ({ previews: {} })),
    ]);
    state.schemaPayload = schemaPayload;
    state.nodePreviews = (prevPayload && prevPayload.previews) || {};
    state.timeAwareOps = new Set((prevPayload && prevPayload.time_aware_ops) || []);
    for (const [cat, nodes] of Object.entries(schemaPayload.schema.nodes_by_category || {})) {
      for (const n of nodes) state.opMap[n.op] = Object.assign({ category: cat }, n);
    }
    graphCanvas.setSchema(schemaPayload.schema);
    $('versionBadge').textContent = `v13 · ${schemaPayload.schema.node_count} nodes`;

    // Populate Project / Preset Dropdown
    const pSel = $('projectSelect');
    const ogSamples = document.createElement('optgroup');
    ogSamples.label = 'Flagship 3D Studio Projects (v12)';
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
    if ($('graphSeedInput')) {
      $('graphSeedInput').onchange = (e) => {
        const s = Math.max(0, Math.min(99999, parseInt(e.target.value, 10) || 0));
        mutate({ action: 'set_graph_settings', seed: s }, { immediateEval: true });
      };
    }
    if ($('btnRandGraphSeed')) {
      $('btnRandGraphSeed').onclick = () => {
        const rnd = Math.floor(1 + Math.random() * 99998);
        mutate({ action: 'randomize_seed', seed: rnd }, { immediateEval: true });
        showToast(`Master Graph Seed -> ${rnd}`, 'ok');
      };
    }
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

    // Graph Toolbar & Canvas Zoom HUD
    $('btnAddNode').onclick = () => openPalette();
    $('btnOpenPaletteLib').onclick = () => openPalette();
    if ($('btnLibList')) $('btnLibList').onclick = () => { state.libViewMode = 'list'; renderNodeLibrary(); };
    if ($('btnLibGrid')) $('btnLibGrid').onclick = () => { state.libViewMode = 'grid'; renderNodeLibrary(); };
    if ($('btnGraphZoomIn')) $('btnGraphZoomIn').onclick = () => graphCanvas.zoomBy(1.22);
    if ($('btnGraphZoomOut')) $('btnGraphZoomOut').onclick = () => graphCanvas.zoomBy(0.82);
    if ($('btnGraphZoomFit')) $('btnGraphZoomFit').onclick = () => graphCanvas.fitToView();
    $('btnAutoLayout').onclick = () => mutate({ action: 'auto_layout' }, { skipEval: true, fitGraph: true });
    if ($('btnDupSel')) $('btnDupSel').onclick = () => { if (state.selectedNodeId) mutate({ action: 'duplicate_nodes', ids: [state.selectedNodeId] }, { immediateEval: true }); };
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
    if ($('btnAnimPingPong')) {
      $('btnAnimPingPong').onclick = () => {
        state.animPingPong = !state.animPingPong;
        $('btnAnimPingPong').classList.toggle('active', state.animPingPong);
        $('btnAnimPingPong').title = state.animPingPong ? 'Mode: Ping-Pong (click for Loop)' : 'Mode: Loop (click for Ping-Pong)';
      };
    }
    if ($('animFpsSelect')) {
      $('animFpsSelect').onchange = (e) => {
        state.animFps = Math.max(2, Math.min(30, parseInt(e.target.value, 10) || 8));
      };
    }
    $('animFramesInput').onchange = (e) => {
      state.animFrames = Math.max(2, Math.min(32, parseInt(e.target.value, 10) || 8));
      setPhase(state.phase);
    };

    // Visual Add-Node Browser & Command Palette
    if ($('btnPalList')) $('btnPalList').onclick = () => { state.paletteViewMode = 'list'; renderPaletteList(); };
    if ($('btnPalGrid')) $('btnPalGrid').onclick = () => { state.paletteViewMode = 'grid'; renderPaletteList(); };
    if ($('btnClosePalette')) $('btnClosePalette').onclick = closePalette;
    $('paletteInput').oninput = () => { state.paletteIdx = 0; renderPaletteList(); };
    $('paletteInput').onkeydown = (e) => {
      if (e.key === 'Escape') { closePalette(); return; }
      const isGrid = state.paletteViewMode === 'grid';
      const cols = isGrid ? Math.max(1, Math.floor(($('paletteList').clientWidth - 16) / 115)) : 1;
      if (e.key === 'ArrowRight' && isGrid) {
        e.preventDefault();
        state.paletteIdx = Math.min(state.paletteItems.length - 1, state.paletteIdx + 1);
      } else if (e.key === 'ArrowLeft' && isGrid) {
        e.preventDefault();
        state.paletteIdx = Math.max(0, state.paletteIdx - 1);
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        state.paletteIdx = Math.min(state.paletteItems.length - 1, state.paletteIdx + cols);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        state.paletteIdx = Math.max(0, state.paletteIdx - cols);
      } else if (e.key === 'Enter' && state.paletteItems[state.paletteIdx]) {
        e.preventDefault();
        spawnFromPalette(state.paletteItems[state.paletteIdx].op);
        return;
      } else {
        return;
      }
      const cards = $('paletteList').querySelectorAll('.palette-card, .palette-item');
      cards.forEach((el, i) => {
        el.classList.toggle('sel', i === state.paletteIdx);
        if (i === state.paletteIdx) el.scrollIntoView({ block: 'nearest' });
      });
      updatePalettePreview(state.paletteItems[state.paletteIdx] || null);
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
    $('btnMeshDragonborn').onclick = () => loadMeshSample('kaelith_dragonborn');
    $('btnMeshMonolith').onclick = () => loadMeshSample('temple_monolith');
    $('btnMeshDetach').onclick = () => loadMeshSample('none');
    $('btnMeshUpload').onclick = () => $('bbmodelFileInput').click();
    $('styleRefFileInput').onchange = async (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const res = await apiPost('/api/style/extract', { source: reader.result, name: f.name.replace(/\.[^.]+$/, ''), apply: false });
          state.lastDna = res.dna;
          renderAiVariantsPanel();
          showToast(`Extracted Style-DNA from ${f.name}`, 'ok');
        } catch (err) {
          showToast(`Style-DNA error: ${err.message}`, 'err');
        }
      };
      reader.readAsDataURL(f);
    };
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
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'd') {
        if (inField || !state.selectedNodeId) return;
        e.preventDefault();
        mutate({ action: 'duplicate_nodes', ids: [state.selectedNodeId] }, { immediateEval: true });
        return;
      }
      if (e.key === 'Escape') {
        hideContextMenu();
        closePalette();
        $('exportModal').classList.remove('open');
        $('meshModal').classList.remove('open');
        return;
      }
      if (inField) return;
      if (e.key === 'Tab' || (e.shiftKey && e.key.toLowerCase() === 'a')) { e.preventDefault(); openPalette(); }
      else if (e.key === ' ') { e.preventDefault(); togglePlayAnim(); }
      else if (e.key.toLowerCase() === 'f') { e.preventDefault(); graphCanvas.fitToView(); }
      else if (e.key.toLowerCase() === 'l') { e.preventDefault(); mutate({ action: 'auto_layout' }, { skipEval: true, fitGraph: true }); }
      else if ((e.key.toLowerCase() === 'b' || e.key.toLowerCase() === 'm') && state.selectedNodeId) { e.preventDefault(); mutate({ action: 'toggle_bypass', node_id: state.selectedNodeId }, { immediateEval: true }); }
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
