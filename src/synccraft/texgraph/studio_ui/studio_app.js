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
    selectedIds: new Set(),
    activePass: 'ALL',
    vpMode: '3d',
    channel: 'pbr_lit',
    pickMode: false,
    brushMode: false,
    decalMode: false,
    decal: {
      stencil: 'chrono_sigil',
      tint_color: '#F0C85A',
      scale_u: 10.0,
      roll_deg: 0,
      max_angle_deg: 72,
      opacity: 0.95,
      emissive_strength: 0.75,
      custom_data_url: '',
      custom_name: '',
    },
    decalGizmo: { nx: 0.5, ny: 0.56 },
    activeLod: 'lod0',
    lodLevels: [],
    activeTexSlot: -1,
    poseSyncTimer: null,
    showBones: false,
    skelSpeedIdx: 1,
    exportTemplateId: 'full_pbr_v2',
    batchModalMode: 'pbr',
    brushTargetMode: 'mask',
    brushRadius: 6,
    brushValue: 1.0,
    brushPending: [],
    brushFlushTimer: null,
    pbrBrush: {
      preset: 'gold_trim',
      color: '#F0C85A',
      roughness: 0.18,
      metallic: 0.95,
      height: 0.78,
      emissive_color: '#38F2FF',
      emissive_strength: 0.0,
      paint_albedo: true,
      paint_roughness: true,
      paint_metallic: true,
      paint_height: true,
      paint_emissive: false,
      erase: false,
    },
    pbrLayers: [],
    activePbrLayerId: 'layer_1',
    syncpacks: [],
    curveDriverIdx: 0,
    curveDragIdx: -1,
    curveDragKfs: null,
    batchPollTimer: null,
    lightDragMode: false,
    envRotDeg: 0,
    lightIntensity: 1.0,
    splitWipe: false,
    splitPos: 0.5,
    splitRightChannel: 'normal',
    uvWire2d: false,
    tile3x3: false,
    zoom2d: 1.0,
    phase: 0.0,
    animFrames: 12,
    animFps: 30,
    animPingPong: false,
    animDir: 1,
    animPlaying: false,
    animTimer: null,
    animRafId: null,
    animSeqKey: null,
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
    libTab: 'nodes',
    liveSwatches: true,
    nodePreviews: {},
    timeAwareOps: new Set(),
    animSheet: {},
    animSheetFrame: 0,
    liveSwatchTimer: null,
    animPreviews: {},
    palAnimTimer: null,
    palAnimOp: null,
    syncmats: [],
    img2d: null,
    img2dBuf: null,
    img2dGen: 0,
    workspaceEpoch: 0,
    evalReqSeq: 0,
    evalAppliedSeq: 0,
    mutationRev: 0,
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

  function syncSelectionUI() {
    const badge = $('selCountBadge');
    if (badge) {
      if (state.selectedIds.size > 1) {
        badge.style.display = 'inline-flex';
        badge.textContent = `${state.selectedIds.size} sel`;
      } else {
        badge.style.display = 'none';
      }
    }
    const ungrpBtn = $('btnUngroupSel');
    if (ungrpBtn && state.serverState && state.serverState.recipe) {
      const selNode = (state.serverState.recipe.nodes || []).find((n) => n.id === state.selectedNodeId);
      ungrpBtn.style.display = (selNode && selNode.op === 'subgraph_group') ? 'inline-flex' : 'none';
    }
  }

  function invalidateAnimRing() {
    state.animSeqKey = null;
    if (gl3d) gl3d.clearAnimFrames();
  }

  function cancelPendingEval() {
    if (evalTimer) {
      clearTimeout(evalTimer);
      evalTimer = null;
    }
    state.evalReqSeq += 1;
    state.evalAppliedSeq = state.evalReqSeq;
  }

  function resetWorkspaceVisuals(opts) {
    opts = opts || {};
    cancelPendingEval();
    stopAnimLoop();
    invalidateAnimRing();
    state.evalData = null;
    state.img2d = null;
    state.img2dBuf = null;
    state.img2dGen += 1;
    if (gl3d) gl3d.resetTextures();
    const c2 = $('canvas2d');
    if (c2) {
      const ctx = c2.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, c2.width, c2.height);
    }
    if (opts.fullProjectReset) {
      setViewportTool('orbit');
      closeViewportPopovers();
      state.channel = 'pbr_lit';
      if ($('channelSelect')) $('channelSelect').value = 'pbr_lit';
      state.phase = 0.0;
      if ($('phaseSlider')) $('phaseSlider').value = '0.00';
      if ($('phaseLabel')) $('phaseLabel').textContent = '0.00';
      if ($('frameReadout')) $('frameReadout').textContent = `F0/${state.animFrames} · φ0.00`;
      state.activePass = 'ALL';
      state.selectedNodeId = null;
      state.selectedIds = new Set();
    }
  }

  function updateBridgeBadge(bridge) {
    const el = $('bridgeBadge');
    if (!el || !bridge) return;
    const c = bridge.clients || 0;
    if (c > 0) {
      el.textContent = `BB5 (${c})`;
      el.className = 'pill-badge ok';
    } else {
      el.textContent = `BB5`;
      el.className = 'pill-badge';
    }
    const poseInfo = (bridge.pose && bridge.pose.clip) ? ` · Pose: ${bridge.pose.clip} @ ${Number(bridge.pose.time || 0).toFixed(2)}s` : '';
    el.title = `Blockbench 5 Live Link SSE Bridge · rev ${bridge.rev || 0} · ${c} client(s)${poseInfo}`;
  }

  function renderBreadcrumbs() {
    const bar = $('subgraphBreadcrumbs');
    if (!bar || !state.serverState) return;
    const crumbs = state.serverState.breadcrumbs || [];
    if (!state.serverState.in_subgraph || crumbs.length <= 1) {
      bar.style.display = 'none';
      bar.innerHTML = '';
      return;
    }
    bar.style.display = 'flex';
    bar.innerHTML = '';
    const backBtn = document.createElement('button');
    backBtn.className = 'crumb-btn';
    backBtn.textContent = '← Back to Parent';
    backBtn.onclick = () => mutate({ action: 'exit_group', level: crumbs.length - 2 }, { immediateEval: true, fitGraph: true });
    bar.appendChild(backBtn);
    crumbs.forEach((cr, idx) => {
      if (idx > 0) {
        const sep = document.createElement('span');
        sep.className = 'crumb-sep';
        sep.textContent = '›';
        bar.appendChild(sep);
      }
      if (idx < crumbs.length - 1) {
        const b = document.createElement('button');
        b.className = 'crumb-btn';
        b.textContent = cr.title;
        b.onclick = () => mutate({ action: 'exit_group', level: cr.level }, { immediateEval: true, fitGraph: true });
        bar.appendChild(b);
      } else {
        const cur = document.createElement('strong');
        cur.textContent = `⊞ ${cr.title}`;
        bar.appendChild(cur);
      }
    });
  }

  function getActiveSelectionIds() {
    const ids = Array.from(state.selectedIds).filter(Boolean);
    if (ids.length) return ids;
    return state.selectedNodeId ? [state.selectedNodeId] : [];
  }

  // ------------------------------------------------------------------------
  // Core Evaluation & Synchronization Loop (Strict Epoch + Sequence Guard!)
  // ------------------------------------------------------------------------
  async function runEvaluate(opts) {
    opts = opts || {};
    const thumbsMode = opts.includeThumbsMode || (opts.includeThumbs === false ? 'none' : 'all');
    const reqEpoch = state.workspaceEpoch;
    const reqSeq = ++state.evalReqSeq;
    try {
      const ev = await apiPost('/api/evaluate', {
        phase: state.phase,
        light_az: state.lightAz,
        light_el: state.lightEl,
        lighting_preset: state.lightingPreset,
        preview_node: state.selectedNodeId,
        include_thumbs: thumbsMode !== 'none',
        include_thumbs_mode: thumbsMode,
        include_qa: opts.includeQa !== false,
      });
      // CRITICAL ANTI-GLITCH GUARD: Discard any response from a prior project/epoch or superseded request!
      if (reqEpoch !== state.workspaceEpoch) return;
      if (ev.workspace_epoch != null && ev.workspace_epoch !== state.workspaceEpoch) return;
      if (reqSeq < state.evalAppliedSeq) return;
      state.evalAppliedSeq = reqSeq;

      const sameEpochPrev = (state.evalData && state.evalData.workspace_epoch === state.workspaceEpoch) ? state.evalData : null;
      if (thumbsMode === 'dynamic') {
        const mergedThumbs = Object.assign({}, (sameEpochPrev && sameEpochPrev.thumbs) || {}, ev.thumbs || {});
        ev.thumbs = mergedThumbs;
        if (sameEpochPrev && !ev.qa && sameEpochPrev.qa) ev.qa = sameEpochPrev.qa;
        state.evalData = ev;
        updatePerfAndQaBadges();
        updateViewportTextures();
        graphCanvas.updateNodeThumbs(ev.thumbs, ev.timings_ms);
        return;
      }
      if (sameEpochPrev && (!ev.thumbs || !Object.keys(ev.thumbs).length)) {
        ev.thumbs = sameEpochPrev.thumbs || {};
      }
      if (sameEpochPrev && !ev.qa && sameEpochPrev.qa) {
        ev.qa = sameEpochPrev.qa;
      }
      state.evalData = ev;
      updatePerfAndQaBadges();
      updateViewportTextures();
      graphCanvas.setData(state.serverState.recipe, state.evalData, state.selectedNodeId, state.activePass, state.selectedIds);
      renderQaPanel();
    } catch (err) {
      if (reqEpoch !== state.workspaceEpoch) return;
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
    const prevEpoch = state.workspaceEpoch;
    const nextEpoch = (st && st.workspace_epoch != null) ? st.workspace_epoch : prevEpoch;
    const epochChanged = nextEpoch !== prevEpoch;
    state.workspaceEpoch = nextEpoch;
    state.mutationRev += 1;
    if (epochChanged) {
      cancelPendingEval();
      stopAnimLoop();
      invalidateAnimRing();
      state.evalData = null;
      state.img2d = null;
      state.img2dBuf = null;
      state.img2dGen += 1;
      if (gl3d) gl3d.resetTextures();
    }
    state.serverState = st;
    if (st.log_file_path && $('logFilePathSpan')) {
      $('logFilePathSpan').textContent = `File log: ${st.log_file_path}`;
    }
    const rec = st.recipe;

    // Ensure valid selected node & multi-selection set
    const nodeIds = (rec.nodes || []).map((n) => n.id);
    const idSet = new Set(nodeIds);
    state.selectedIds = new Set(Array.from(state.selectedIds).filter((id) => idSet.has(id)));
    if (Array.isArray(st.created_node_ids) && st.created_node_ids.length) {
      const validCreated = st.created_node_ids.filter((id) => idSet.has(id));
      state.selectedIds = new Set(validCreated);
      state.selectedNodeId = (st.created_node_id && idSet.has(st.created_node_id))
        ? st.created_node_id
        : (validCreated[validCreated.length - 1] || null);
    } else if (st.created_node_id && idSet.has(st.created_node_id)) {
      state.selectedNodeId = st.created_node_id;
      state.selectedIds = new Set([st.created_node_id]);
    } else if (!state.selectedNodeId || !idSet.has(state.selectedNodeId)) {
      const outBase = (rec.output_material || rec.output || rec.output_material_node || rec.output_node || '').split('.')[0];
      state.selectedNodeId = idSet.has(outBase) ? outBase : (nodeIds[nodeIds.length - 1] || null);
      state.selectedIds = new Set(state.selectedNodeId ? [state.selectedNodeId] : []);
    } else if (!state.selectedIds.size && state.selectedNodeId) {
      state.selectedIds.add(state.selectedNodeId);
    }
    syncSelectionUI();

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
        meshResBadge.textContent = `${rec.width}×${rec.height}`;
      }
    } else {
      resSel.style.display = '';
      resSel.disabled = false;
      if (meshResBadge) meshResBadge.style.display = 'none';
    }

    // Sync 3D mesh geometry + live skeletal transport dock + Auto-LOD & multi-texture slots (`v18`)
    state.activeLod = st.active_lod || 'lod0';
    if (Array.isArray(st.lod_levels)) state.lodLevels = st.lod_levels;
    state.activeTexSlot = (st.active_texture_slot != null) ? Number(st.active_texture_slot) : -1;
    if (gl3d) {
      gl3d.setBBModelMesh(st.mesh || null);
      gl3d.activeTexSlot = state.activeTexSlot;
      if (opts.autoPrim) {
        const prim = st.has_mesh ? 'bbmodel' : 'cube';
        $('primSelect').value = prim;
        gl3d.primitive = prim;
      }
      syncSkelDockFromMesh(st.mesh || null);
    }
    syncLodAndSlotsFromState(st);

    // Sync TexDSL editor (unless user is actively typing in it)
    const ed = $('texdslEditor');
    if (document.activeElement !== ed) {
      ed.value = st.texdsl || '';
      $('texdslStatus').textContent = `Synced · ${nodeIds.length} nodes · ${rec.width}×${rec.height}`;
      $('texdslStatus').style.color = 'var(--text-muted)';
    }

    renderPassFilterBar();
    renderBreadcrumbs();
    updateBridgeBadge(st.bridge);
    syncPbrLayersFromGraph();
    syncDecalCountFromGraph();
    renderInspector();
    renderTimelineDrivers();
    renderCurveEditor();
    renderExposedParams();
    renderConsoleLog();
    renderAiVariantsPanel();
    graphCanvas.setData(rec, state.evalData, state.selectedNodeId, state.activePass, state.selectedIds);
    if (opts.fitGraph) {
      setTimeout(() => graphCanvas.fitToView(), 30);
    }
  }

  async function mutate(payload, opts) {
    opts = opts || {};
    invalidateAnimRing();
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
    $('perfBadge').textContent = `${Math.round(ev.eval_ms)}ms`;
    $('perfBadge').title = `Eval ${ev.eval_ms.toFixed(1)} ms · Cache Hit Rate ${hitPct}%`;

    const qa = ev.qa;
    const qb = $('qaBadge');
    if (qa && typeof qa.score === 'number') {
      qb.textContent = `QA ${Math.round(qa.score)}`;
      qb.title = `Automated Multi-Domain Texture QA Score: ${Math.round(qa.score)}/100`;
      qb.className = 'pill-badge ' + (qa.passed ? 'ok' : (qa.score >= 70 ? 'warn' : 'err'));
    }
    if (ev.bridge) updateBridgeBadge(ev.bridge);
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
      const my2dGen = ++state.img2dGen;
      const im = new Image();
      im.onload = () => {
        if (state.img2dGen !== my2dGen) return;
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

    // Optional 3D .bbmodel UV Island Wireframe Overlay (`UV Wire` toggle!)
    if (state.uvWire2d && state.serverState && state.serverState.mesh && Array.isArray(state.serverState.mesh.uv_rects)) {
      ctx.save();
      ctx.strokeStyle = 'rgba(56, 224, 255, 0.72)';
      ctx.lineWidth = Math.max(1, dpr * 0.85);
      for (const rect of state.serverState.mesh.uv_rects) {
        const [u0, v0, u1, v1] = rect;
        for (let ty = 0; ty < tiles; ty++) {
          for (let tx = 0; tx < tiles; tx++) {
            ctx.strokeRect(ox + (tx + u0) * tw, oy + (ty + v0) * th, (u1 - u0) * tw, (v1 - v0) * th);
          }
        }
      }
      ctx.restore();
    }
  }

  function updateViewportHud(extraInfo) {
    const hud = $('vpHud');
    const ev = state.evalData;
    const res = ev ? `${ev.width}×${ev.height}` : '—';
    const ms = state.serverState && state.serverState.mesh_stats;
    const pickTag = state.pickMode ? ' · 🎯 PICK MODE' : '';
    if (state.vpMode === '3d') {
      let mText = ` · ${$('primSelect').value}`;
      if (ms) {
        const lodInfo = (state.activeLod && state.activeLod !== 'lod0')
          ? (state.lodLevels || []).find((x) => x.level === state.activeLod)
          : null;
        if (lodInfo) {
          mText = ` · ${state.activeLod.toUpperCase()} ${lodInfo.box_count}b (-${lodInfo.reduction_pct}%)`;
        } else {
          const seams = ms.seam_pairs >= 1000 ? `${(ms.seam_pairs / 1000).toFixed(1)}k` : `${ms.seam_pairs}`;
          mText = ` · ${ms.elements}b · ${seams} seams`;
        }
      }
      hud.textContent = `3D PBR · ${state.channel} · ${res}${mText}${pickTag}${extraInfo ? ' · ' + extraInfo : ''}`;
    } else {
      hud.textContent = `2D Pixel · ${state.channel} · ${res}${state.tile3x3 ? ' · 3×3 Wrap' : ''}${pickTag}${extraInfo ? ' · ' + extraInfo : ''}`;
    }
  }

  function closeViewportPopovers() {
    if ($('lightingPopover')) $('lightingPopover').style.display = 'none';
    if ($('viewPopover')) $('viewPopover').style.display = 'none';
    if ($('btnLightingPop')) $('btnLightingPop').classList.remove('active');
    if ($('btnViewPop')) $('btnViewPop').classList.remove('active');
  }

  function toggleViewportPopover(which) {
    const isLight = which === 'lighting';
    const target = $(isLight ? 'lightingPopover' : 'viewPopover');
    const other = $(isLight ? 'viewPopover' : 'lightingPopover');
    const btn = $(isLight ? 'btnLightingPop' : 'btnViewPop');
    const otherBtn = $(isLight ? 'btnViewPop' : 'btnLightingPop');
    if (!target) return;
    const willOpen = target.style.display === 'none' || !target.style.display;
    if (other) other.style.display = 'none';
    if (otherBtn) otherBtn.classList.remove('active');
    target.style.display = willOpen ? 'flex' : 'none';
    if (btn) btn.classList.toggle('active', willOpen);
  }

  function setViewportTool(mode) {
    const m = ['orbit', 'pick', 'paint', 'decal', 'light'].includes(mode) ? mode : 'orbit';
    state.pickMode = m === 'pick';
    state.brushMode = m === 'paint';
    state.decalMode = m === 'decal';
    state.lightDragMode = m === 'light';

    if (gl3d) {
      gl3d.pickMode = state.pickMode;
      gl3d.brushMode = state.brushMode;
      gl3d.decalMode = state.decalMode;
      gl3d.lightDragMode = state.lightDragMode;
    }

    if ($('btnOrbit3d')) $('btnOrbit3d').classList.toggle('active', m === 'orbit');
    if ($('btnPickPart')) $('btnPickPart').classList.toggle('pick-on', state.pickMode);
    if ($('btnPickPart2d')) $('btnPickPart2d').classList.toggle('pick-on', state.pickMode);
    if ($('btnBrush3d')) $('btnBrush3d').classList.toggle('brush-on', state.brushMode);
    if ($('btnBrush2d')) $('btnBrush2d').classList.toggle('brush-on', state.brushMode);
    if ($('btnDecal3d')) $('btnDecal3d').classList.toggle('decal-on', state.decalMode);
    if ($('btnLightDrag')) $('btnLightDrag').classList.toggle('light-on', state.lightDragMode);

    if ($('brushToolbar')) $('brushToolbar').style.display = state.brushMode ? 'flex' : 'none';
    if ($('decalToolbar')) $('decalToolbar').style.display = (state.decalMode && state.vpMode === '3d') ? 'flex' : 'none';
    if ($('decalGizmoLayer')) $('decalGizmoLayer').style.display = (state.decalMode && state.vpMode === '3d') ? 'block' : 'none';
    if (state.decalMode && state.vpMode === '3d') updateDecalGizmoUI();
    if ($('viewportStage')) {
      $('viewportStage').classList.toggle('pick-active', state.pickMode || state.brushMode || state.decalMode);
    }
    if (m !== 'orbit') closeViewportPopovers();
    const labelMap = {
      orbit: null,
      pick: 'Pick Part Mode',
      paint: state.brushTargetMode === 'pbr' ? '3D PBR Layers' : 'Mask Paint',
      decal: '3D Decal Gizmo (Drag / Scale / Roll · Enter or Stamp)',
      light: 'Drag Light & HDRI',
    };
    updateViewportHud(labelMap[m]);
  }

  function setPickMode(on) { setViewportTool(on ? 'pick' : 'orbit'); }
  function setBrushMode(on) { setViewportTool(on ? 'paint' : 'orbit'); }
  function setDecalMode(on) { setViewportTool(on ? 'decal' : 'orbit'); }
  function setLightDragMode(on) { setViewportTool(on ? 'light' : 'orbit'); }

  function setShowBones(on) {
    state.showBones = !!on;
    if (gl3d) gl3d.showBones = state.showBones;
    if ($('btnBones3d')) $('btnBones3d').classList.toggle('active', state.showBones);
    if ($('btnSkelBonesQuick')) $('btnSkelBonesQuick').classList.toggle('active', state.showBones);
  }

  function syncSkelDockFromMesh(mesh) {
    const dock = $('skelTransportDock');
    const stage = $('viewportStage');
    const sel = $('skelClipSelect');
    const anims = (mesh && Array.isArray(mesh.animations)) ? mesh.animations : [];
    const show = !!(anims.length && state.vpMode === '3d' && (!gl3d || gl3d.primitive === 'bbmodel'));
    if (dock) dock.style.display = show ? 'flex' : 'none';
    if (stage) stage.classList.toggle('has-skel-dock', show);
    if (!sel) return;
    sel.innerHTML = '<option value="-1">Bind Pose</option>';
    anims.forEach((a, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = `${a.name} (${Number(a.length || 1).toFixed(1)}s)`;
      sel.appendChild(opt);
    });
    const curIdx = gl3d ? gl3d.skelClipIdx : -1;
    sel.value = String(curIdx >= 0 && curIdx < anims.length ? curIdx : -1);
    const activeClip = (curIdx >= 0 && curIdx < anims.length) ? anims[curIdx] : null;
    const len = activeClip ? Math.max(0.05, activeClip.length || 1.0) : 1.0;
    const curT = gl3d ? gl3d.skelTime : 0.0;
    if ($('skelTimeSlider')) {
      $('skelTimeSlider').max = len.toFixed(3);
      $('skelTimeSlider').value = curT.toFixed(3);
      $('skelTimeSlider').disabled = !activeClip;
    }
    if ($('skelTimeVal')) $('skelTimeVal').textContent = `${curT.toFixed(2)}s`;
    if ($('btnSkelPlay')) {
      const pl = !!(gl3d && gl3d.skelPlaying);
      $('btnSkelPlay').textContent = pl ? '⏸' : '▶';
      $('btnSkelPlay').classList.toggle('amber', pl);
    }
  }

  function populateDecalStencils() {
    const sel = $('decalStencilSelect');
    const stencils = (state.schemaPayload && state.schemaPayload.decal_stencils) || {};
    if (!sel) return;
    sel.innerHTML = '';
    for (const [sid, meta] of Object.entries(stencils)) {
      const o = document.createElement('option');
      o.value = sid;
      o.textContent = meta.title || sid;
      sel.appendChild(o);
    }
    sel.value = state.decal.stencil;
    const cur = stencils[state.decal.stencil];
    if (cur && $('decalTintColor')) {
      state.decal.tint_color = cur.color || '#F0C85A';
      state.decal.emissive_strength = cur.emissive_strength != null ? cur.emissive_strength : 0.75;
      $('decalTintColor').value = state.decal.tint_color;
    }
    updateDecalGizmoUI();
  }

  function updateDecalGizmoUI() {
    const layer = $('decalGizmoLayer');
    const box = $('decalGizmoBox');
    const img = $('decalGizmoImg');
    const readout = $('decalGizmoReadout');
    const strip = $('decalCustomStrip');
    const nameEl = $('decalCustomName');
    const show = !!(state.decalMode && state.vpMode === '3d');
    if (layer) layer.style.display = show ? 'block' : 'none';
    if (strip) {
      strip.style.display = state.decal.custom_data_url ? 'flex' : 'none';
      if (nameEl) nameEl.textContent = state.decal.custom_name || 'Custom PNG Stencil';
    }
    if (!box) return;
    const nx = Math.max(0.08, Math.min(0.92, state.decalGizmo.nx));
    const ny = Math.max(0.10, Math.min(0.90, state.decalGizmo.ny));
    const px = Math.max(46, Math.min(220, Math.round((state.decal.scale_u || 10) * 8.5)));
    box.style.left = `${(nx * 100).toFixed(2)}%`;
    box.style.top = `${(ny * 100).toFixed(2)}%`;
    box.style.width = `${px}px`;
    box.style.height = `${px}px`;
    box.style.transform = `translate(-50%, -50%) rotate(${Number(state.decal.roll_deg || 0).toFixed(1)}deg)`;
    if (img) {
      const previews = (state.schemaPayload && state.schemaPayload.decal_stencil_previews) || {};
      const src = state.decal.custom_data_url || previews[state.decal.stencil] || '';
      if (src && img.getAttribute('src') !== src) img.src = src;
      img.style.opacity = String(Math.max(0.28, Math.min(0.92, (state.decal.opacity || 0.95) * 0.85)));
    }
    if (readout) {
      const sTxt = Number(state.decal.scale_u || 10).toFixed(1).replace(/\.0$/, '');
      const rTxt = Math.round(Number(state.decal.roll_deg || 0));
      readout.textContent = `${sTxt}u · ${rTxt}°`;
    }
  }

  function syncLodAndSlotsFromState(st) {
    const hasMesh = !!(st && st.has_mesh);
    const lodSec = $('vpLodSection');
    const slotSec = $('vpTexSlotSection');
    if (lodSec) lodSec.style.opacity = hasMesh ? '1' : '0.5';
    if (slotSec) slotSec.style.opacity = hasMesh ? '1' : '0.5';

    ['lod0', 'lod1', 'lod2'].forEach((lvl, i) => {
      const btn = $(`btnLod${i}`);
      if (!btn) return;
      btn.classList.toggle('active', state.activeLod === lvl);
      btn.disabled = !hasMesh;
    });

    const statsEl = $('vpLodStats');
    if (statsEl) {
      if (!hasMesh) {
        statsEl.textContent = 'No mesh';
      } else {
        const info = (state.lodLevels || []).find((x) => x.level === state.activeLod);
        if (info) {
          const red = info.reduction_pct > 0 ? ` · -${info.reduction_pct}%` : ' · 100%';
          statsEl.textContent = `${state.activeLod.toUpperCase()} · ${info.box_count}b${red}`;
        } else if (st && st.mesh_stats) {
          statsEl.textContent = `${state.activeLod.toUpperCase()} · ${st.mesh_stats.elements}b · 100%`;
        } else {
          statsEl.textContent = `${state.activeLod.toUpperCase()} · 100%`;
        }
      }
    }

    const slots = (st && Array.isArray(st.texture_slots)) ? st.texture_slots : [];
    const sel = $('texSlotSelect');
    if (sel) {
      sel.innerHTML = '';
      const optAll = document.createElement('option');
      optAll.value = '-1';
      optAll.textContent = slots.length > 1 ? `All Texture Slots (${slots.length})` : 'All Texture Slots';
      sel.appendChild(optAll);
      slots.forEach((s) => {
        const o = document.createElement('option');
        o.value = String(s.index);
        o.textContent = `[${s.index}] ${s.name || 'texture'} (${s.width}×${s.height} · ${s.face_count}f)`;
        sel.appendChild(o);
      });
      sel.value = String(state.activeTexSlot >= 0 && state.activeTexSlot < slots.length ? state.activeTexSlot : -1);
      sel.disabled = !hasMesh;
    }
    if ($('vpTexSlotCount')) {
      $('vpTexSlotCount').textContent = hasMesh ? `${slots.length} slot${slots.length === 1 ? '' : 's'}` : '—';
    }
    if ($('lodStemInput') && document.activeElement !== $('lodStemInput') && st && st.recipe && st.recipe.name) {
      $('lodStemInput').value = st.recipe.name;
    }
  }

  async function switchLodLevel(lvl) {
    if (!state.serverState || !state.serverState.has_mesh) {
      showToast('Attach a 3D .bbmodel mesh first to preview Auto-LODs', 'err');
      return;
    }
    try {
      const res = await apiPost('/api/lod/preview', { level: lvl });
      state.activeLod = res.active_lod || lvl;
      if (Array.isArray(res.levels)) state.lodLevels = res.levels;
      if (gl3d && res.mesh) {
        gl3d.setBBModelMesh(res.mesh);
        gl3d.activeTexSlot = state.activeTexSlot;
      }
      syncLodAndSlotsFromState(state.serverState);
      const info = (state.lodLevels || []).find((x) => x.level === state.activeLod);
      const hudExtra = info
        ? `${state.activeLod.toUpperCase()} (${info.box_count}b, ${info.tri_count}t${info.reduction_pct > 0 ? `, -${info.reduction_pct}%` : ''})`
        : state.activeLod.toUpperCase();
      updateViewportHud();
      showToast(`Live 3D Preview -> ${hudExtra}`, 'ok');
    } catch (err) {
      showToast(`Auto-LOD error: ${err.message}`, 'err');
    }
  }

  function scheduleBridgePoseSync() {
    if (state.poseSyncTimer) clearTimeout(state.poseSyncTimer);
    state.poseSyncTimer = setTimeout(async () => {
      state.poseSyncTimer = null;
      if (!gl3d) return;
      const anims = (state.serverState && state.serverState.mesh && Array.isArray(state.serverState.mesh.animations))
        ? state.serverState.mesh.animations
        : [];
      const clipObj = (gl3d.skelClipIdx >= 0 && gl3d.skelClipIdx < anims.length) ? anims[gl3d.skelClipIdx] : null;
      try {
        const res = await apiPost('/api/bridge/pose', {
          clip: clipObj ? clipObj.name : null,
          time: Number(gl3d.skelTime || 0),
          playing: !!gl3d.skelPlaying,
          speed: Number(gl3d.skelSpeed || 1.0),
        });
        updateBridgeBadge(res);
      } catch (_) {}
    }, 110);
  }

  function findActiveDecalNode() {
    const rec = state.serverState && state.serverState.recipe;
    if (!rec || !Array.isArray(rec.nodes)) return null;
    const sel = rec.nodes.find((n) => n.id === state.selectedNodeId && n.op === 'decal_stamp');
    if (sel) return sel;
    for (let i = rec.nodes.length - 1; i >= 0; i--) {
      if (rec.nodes[i].op === 'decal_stamp') return rec.nodes[i];
    }
    return null;
  }

  function syncDecalCountFromGraph() {
    const node = findActiveDecalNode();
    let cnt = 0;
    if (node && node.params && node.params.stamps_json) {
      try {
        const parsed = typeof node.params.stamps_json === 'string'
          ? JSON.parse(node.params.stamps_json)
          : node.params.stamps_json;
        if (Array.isArray(parsed)) cnt = parsed.length;
      } catch (_) {}
    }
    if ($('decalStackCount')) $('decalStackCount').textContent = `${cnt}`;
  }

  async function applyDecalStampFromProjector(proj, actionName) {
    const act = actionName || 'add';
    const node = findActiveDecalNode();
    const payload = {
      node_id: node ? node.id : null,
      action: act,
      wire_output: true,
    };
    if (act === 'add') {
      const stencils = (state.schemaPayload && state.schemaPayload.decal_stencils) || {};
      const meta = stencils[state.decal.stencil] || {};
      payload.stamp = Object.assign({
        stencil: state.decal.stencil,
        tint_color: state.decal.tint_color,
        scale_u: state.decal.scale_u,
        scale_v: state.decal.scale_u,
        roll_deg: state.decal.roll_deg,
        max_angle_deg: state.decal.max_angle_deg,
        angle_fade_deg: 20.0,
        opacity: state.decal.opacity,
        occlusion_test: true,
        emissive_strength: meta.emissive_strength != null ? meta.emissive_strength : state.decal.emissive_strength,
      }, proj || {});
      if (state.decal.custom_data_url) {
        payload.stamp.custom_data_url = state.decal.custom_data_url;
        payload.stamp.custom_name = state.decal.custom_name || 'custom_stencil.png';
      }
    }
    try {
      const st = await apiPost('/api/decal_stamp/apply', payload);
      invalidateAnimRing();
      applyServerState(st, { autoPrim: false });
      await runEvaluate({ includeThumbsMode: 'all', includeQa: false });
      if (act === 'add') {
        const sTitle = state.decal.custom_data_url
          ? (state.decal.custom_name || 'Custom PNG')
          : (((state.schemaPayload && state.schemaPayload.decal_stencils && state.schemaPayload.decal_stencils[state.decal.stencil]) || {}).title || state.decal.stencil);
        showToast(`Stamped 3D decal '${sTitle}' (stack: ${st.stamp_count || 1})`, 'ok');
      } else if (act === 'undo_last') {
        showToast(`Removed last 3D decal (${st.stamp_count || 0} remaining)`, 'ok');
      } else if (act === 'clear') {
        showToast('Cleared 3D decal stack', 'ok');
      }
    } catch (err) {
      showToast(`Decal error: ${err.message}`, 'err');
    }
  }

  const EXTRA_EXPORT_TEMPLATES = [
    {
      id: 'glb_skinned',
      title: 'glTF 2.0 (.glb) Skinned Mesh + Animations + Embedded PBR',
      engine: 'glTF 2.0 (UE5 / Unity / Godot / Blender / Three.js)',
      normal_convention: 'OpenGL (+Y Up) · Khronos Verified',
      description: 'Single binary .glb packing meter-scaled (1/16) indexed mesh, Armature skin + IBMs, baked Molang skeletal animations, and embedded PNG PBR maps (Albedo, shared Linear ORM, OpenGL Normal, Emissive + KHR extensions).',
      channels: [
        { file: '<stem>.glb', slots: ['Skinned Rig + IBMs', 'Baked Skeletal Clips', 'sRGB Albedo', 'Linear ORM (R=AO G=R B=M)', 'OpenGL +Y Normal', 'Emissive'] },
      ],
    },
    {
      id: 'ctm47',
      title: 'Connected Block CTM 47-Tile (OptiFine / Continuity)',
      engine: 'Minecraft Connected Textures (CTM)',
      normal_convention: '2D Seamless Autotile Sheet',
      description: 'Generates a 47-tile border-connected sprite sheet + ready-to-drop OptiFine / Continuity .properties file.',
      channels: [
        { file: '<stem>_ctm47_sheet.png', slots: ['47 Connected Tiles (8×6 Sheet)'] },
        { file: '<stem>.properties', slots: ['OptiFine / Continuity CTM Rules'] },
      ],
    },
    {
      id: 'wang16',
      title: 'Wang 16-Tile 4-Bit Autotile Sheet',
      engine: '2D / Tilemap Engine (Godot / Unity / RPG)',
      normal_convention: '4-Bit Corner Wang Tileset',
      description: 'Generates a seamless 16-tile 4-bit Wang autotile sprite sheet for modular terrain & walls.',
      channels: [
        { file: '<stem>_wang16_sheet.png', slots: ['16 Wang Tiles (4×4 Sheet)'] },
        { file: '<stem>.properties', slots: ['Autotile Index Manifest'] },
      ],
    },
  ];

  function getAllExportTemplates() {
    const base = (state.schemaPayload && Array.isArray(state.schemaPayload.export_presets))
      ? state.schemaPayload.export_presets
      : [];
    return [...base, ...EXTRA_EXPORT_TEMPLATES];
  }

  function populateExportTemplates() {
    const sel = $('exportTemplateSelect');
    if (!sel) return;
    const list = getAllExportTemplates();
    sel.innerHTML = '';
    for (const tpl of list) {
      const o = document.createElement('option');
      o.value = tpl.id;
      o.textContent = tpl.title;
      sel.appendChild(o);
    }
    sel.value = state.exportTemplateId || (list[0] && list[0].id) || 'full_pbr_v2';
    renderExportTemplatePreview();
  }

  function renderExportTemplatePreview() {
    const box = $('exportTemplatePreview');
    const sel = $('exportTemplateSelect');
    if (!box || !sel) return;
    const tid = sel.value || state.exportTemplateId || 'full_pbr_v2';
    state.exportTemplateId = tid;
    const tpl = getAllExportTemplates().find((t) => t.id === tid);
    if (!tpl) return;
    const stem = ($('exportStemInput') && $('exportStemInput').value.trim()) ||
      (state.serverState && state.serverState.recipe && state.serverState.recipe.name) || 'asset';
    const nrm = tpl.normal_convention || '';
    const nrmCls = nrm.includes('DirectX') ? 'dx' : (nrm.includes('OpenGL') ? 'gl' : '');
    const rowsHtml = (tpl.channels || []).map((ch) => {
      const fname = String(ch.file || '').replace(/<stem>/g, stem);
      const chips = (ch.slots || []).map((s) => `<span class="exp-slot-chip">${s}</span>`).join('');
      return `<div class="exp-channel-row"><span class="exp-ch-file">${fname}</span><span class="exp-ch-slots">${chips}</span></div>`;
    }).join('');
    const safeDesc = String(tpl.description || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    box.innerHTML = `
      <div class="exp-tpl-top">
        <span class="exp-tpl-engine">${tpl.engine}</span>
        <span class="exp-tpl-nrm ${nrmCls}">${nrm}</span>
      </div>
      <div class="exp-tpl-desc">${safeDesc}</div>
      <div class="exp-channel-list">${rowsHtml}</div>
    `;
    if ($('btnRunExport')) {
      const shortName = String(tpl.engine || 'Template').split(/[/(·]/)[0].trim();
      $('btnRunExport').textContent = `Export ${shortName}`;
    }
  }

  function setBrushTargetMode(mode) {
    state.brushTargetMode = mode === 'pbr' ? 'pbr' : 'mask';
    const isPbr = state.brushTargetMode === 'pbr';
    if ($('btnBrushModeMask')) $('btnBrushModeMask').classList.toggle('active', !isPbr);
    if ($('btnBrushModePbr')) $('btnBrushModePbr').classList.toggle('active', isPbr);
    if ($('brushMaskControls')) $('brushMaskControls').style.display = isPbr ? 'none' : 'inline-flex';
    if ($('brushPbrControls')) $('brushPbrControls').style.display = isPbr ? 'inline-flex' : 'none';
    if ($('brushPbrChannels')) $('brushPbrChannels').style.display = isPbr ? 'grid' : 'none';
    if (isPbr) {
      syncPbrLayersFromGraph();
      renderPbrLayerPills();
    }
    updateViewportHud(state.brushMode ? (isPbr ? '🖌 3D PBR LAYERS' : '🖌 MASK BRUSH') : null);
  }

  function applyPbrBrushPreset(presetId) {
    const presets = (state.schemaPayload && state.schemaPayload.brush_presets) || {};
    const p = presets[presetId];
    state.pbrBrush.preset = presetId;
    if (p) {
      state.pbrBrush.color = p.color || '#F0C85A';
      state.pbrBrush.roughness = p.roughness != null ? p.roughness : 0.25;
      state.pbrBrush.metallic = p.metallic != null ? p.metallic : 0.9;
      state.pbrBrush.height = p.height != null ? p.height : 0.7;
      state.pbrBrush.emissive_color = p.emissive_color || p.color || '#38F2FF';
      state.pbrBrush.emissive_strength = p.emissive_strength != null ? p.emissive_strength : 0.0;
      state.pbrBrush.paint_albedo = p.paint_albedo !== false;
      state.pbrBrush.paint_roughness = p.paint_roughness !== false;
      state.pbrBrush.paint_metallic = p.paint_metallic !== false;
      state.pbrBrush.paint_height = p.paint_height !== false;
      state.pbrBrush.paint_emissive = !!p.paint_emissive || state.pbrBrush.emissive_strength > 0.01;
      state.pbrBrush.erase = !!p.erase || presetId === 'eraser';
    } else if (presetId === 'eraser') {
      state.pbrBrush.erase = true;
    } else {
      state.pbrBrush.erase = false;
    }
    if ($('pbrBrushColor')) $('pbrBrushColor').value = state.pbrBrush.color;
    if ($('pbrBrushRough')) $('pbrBrushRough').value = String(state.pbrBrush.roughness);
    if ($('pbrBrushMetal')) $('pbrBrushMetal').value = String(state.pbrBrush.metallic);
    if ($('pbrBrushHeight')) $('pbrBrushHeight').value = String(state.pbrBrush.height);
    if ($('pbrBrushEmissive')) $('pbrBrushEmissive').value = String(state.pbrBrush.emissive_strength);
  }

  function findActivePaintStackNode() {
    const rec = state.serverState && state.serverState.recipe;
    if (!rec || !Array.isArray(rec.nodes)) return null;
    const sel = rec.nodes.find((n) => n.id === state.selectedNodeId && n.op === 'paint_layer_stack');
    if (sel) return sel;
    for (let i = rec.nodes.length - 1; i >= 0; i--) {
      if (rec.nodes[i].op === 'paint_layer_stack') return rec.nodes[i];
    }
    return null;
  }

  function syncPbrLayersFromGraph() {
    const node = findActivePaintStackNode();
    if (!node) {
      state.pbrLayers = [{ id: 'layer_1', name: 'Hero Detail Paint', visible: true, opacity: 1.0, blend_mode: 'normal' }];
      state.activePbrLayerId = 'layer_1';
      renderPbrLayerPills();
      return;
    }
    try {
      const raw = node.params && node.params.layers_json;
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : (Array.isArray(raw) ? raw : []);
      if (Array.isArray(parsed) && parsed.length) {
        state.pbrLayers = parsed.map((l) => ({
          id: l.id || 'layer_1',
          name: l.name || l.id || 'Layer',
          visible: l.visible !== false,
          opacity: l.opacity != null ? l.opacity : 1.0,
          blend_mode: l.blend_mode || 'normal',
        }));
      }
      state.activePbrLayerId = (node.params && node.params.active_layer_id) || state.pbrLayers[0].id;
    } catch (_) {}
    renderPbrLayerPills();
  }

  async function runPbrLayerAction(layerAction, extra) {
    const node = findActivePaintStackNode();
    try {
      const st = await apiPost('/api/paint_stack/stroke', Object.assign({
        node_id: node ? node.id : null,
        layer_id: state.activePbrLayerId,
        layer_action: layerAction,
      }, extra || {}));
      invalidateAnimRing();
      applyServerState(st, { autoPrim: false });
      await runEvaluate({ includeThumbsMode: 'all', includeQa: false });
      return st;
    } catch (err) {
      showToast(err.message, 'err');
      return null;
    }
  }

  function renderPbrLayerPills() {
    const wrap = $('pbrLayerPills');
    if (!wrap) return;
    wrap.innerHTML = '';
    const layers = state.pbrLayers || [];
    for (const l of layers) {
      const pill = document.createElement('span');
      pill.className = 'pbr-layer-pill' + (l.id === state.activePbrLayerId ? ' active' : '') + (!l.visible ? ' hidden-layer' : '');
      pill.title = `${l.name} (${l.blend_mode} · ${Math.round((l.opacity != null ? l.opacity : 1) * 100)}%)`;
      const eyeBtn = document.createElement('button');
      eyeBtn.textContent = l.visible ? '👁' : '○';
      eyeBtn.title = l.visible ? 'Hide Paint Layer' : 'Show Paint Layer';
      eyeBtn.onclick = (ev) => {
        ev.stopPropagation();
        runPbrLayerAction('toggle_visible', { layer_id: l.id });
      };
      const lbl = document.createElement('span');
      lbl.textContent = l.name;
      pill.appendChild(eyeBtn);
      pill.appendChild(lbl);
      if (layers.length > 1) {
        const delBtn = document.createElement('button');
        delBtn.textContent = '✕';
        delBtn.title = 'Delete Paint Layer';
        delBtn.onclick = (ev) => {
          ev.stopPropagation();
          runPbrLayerAction('delete_layer', { layer_id: l.id });
        };
        pill.appendChild(delBtn);
      }
      pill.onclick = () => {
        state.activePbrLayerId = l.id;
        renderPbrLayerPills();
        if (findActivePaintStackNode()) {
          runPbrLayerAction('select_layer', { layer_id: l.id });
        }
      };
      wrap.appendChild(pill);
    }
  }

  function queueBrushStroke(u, v, isStart) {
    state.brushPending.push({ u, v, radius_px: state.brushRadius, value: state.brushValue, softness: 0.35, opacity: 0.95 });
    if (isStart) state._brushUndoNext = true;
    if (!state.brushFlushTimer) {
      state.brushFlushTimer = setTimeout(flushBrushStrokes, 60);
    }
  }

  async function flushBrushStrokes() {
    state.brushFlushTimer = null;
    if (!state.brushPending.length) return;
    const strokes = state.brushPending.splice(0, state.brushPending.length);
    const undoable = !!state._brushUndoNext;
    state._brushUndoNext = false;

    // v16 Multi-Channel 3D/2D PBR Paint Layer Stack Mode!
    if (state.brushTargetMode === 'pbr') {
      const node = findActivePaintStackNode();
      const b = Object.assign({}, state.pbrBrush, {
        emissive_color: state.pbrBrush.emissive_strength > 0.01 ? state.pbrBrush.color : '#000000',
        paint_emissive: state.pbrBrush.emissive_strength > 0.01,
      });
      try {
        const st = await apiPost('/api/paint_stack/stroke', {
          node_id: node ? node.id : null,
          layer_id: state.activePbrLayerId,
          strokes,
          brush: b,
          undoable,
          wire_output: true,
        });
        invalidateAnimRing();
        applyServerState(st, { autoPrim: false });
        if (st.created_node_id) {
          if (state.channel === 'selected') {
            state.channel = 'pbr_lit';
            $('channelSelect').value = 'pbr_lit';
          }
          showToast(`🖌 Auto-wired 3D PBR Paint Stack '${st.created_node_id}'!`, 'ok');
        }
        await runEvaluate({ includeThumbsMode: 'all', includeQa: false });
      } catch (err) {
        showToast(err.message, 'err');
      }
      return;
    }

    const rec = state.serverState && state.serverState.recipe;
    const sel = rec && (rec.nodes || []).find((n) => n.id === state.selectedNodeId);
    const targetId = (sel && sel.op === 'paint_mask') ? sel.id : null;
    try {
      const cx = Math.round((-graphCanvas.panX + graphCanvas.stage.clientWidth * 0.45) / graphCanvas.zoom - 90);
      const cy = Math.round((-graphCanvas.panY + graphCanvas.stage.clientHeight * 0.42) / graphCanvas.zoom - 50);
      const st = await apiPost('/api/paint_mask/stroke', {
        node_id: targetId,
        strokes,
        undoable,
        pos: [cx, cy],
      });
      invalidateAnimRing();
      applyServerState(st, { autoPrim: false });
      if (st.created_node_id) {
        state.channel = 'selected';
        $('channelSelect').value = 'selected';
        showToast(`🖌 Spawned 3D/2D Brush Mask '${st.created_node_id}'`, 'ok');
      }
      await runEvaluate({ includeThumbsMode: 'all', includeQa: false });
    } catch (err) {
      showToast(err.message, 'err');
    }
  }

  async function handlePickedPart(partShortName, additive) {
    if (!partShortName) return;
    const rec = state.serverState && state.serverState.recipe;
    const selNode = rec && (rec.nodes || []).find((n) => n.id === state.selectedNodeId);
    const meta = selNode ? state.opMap[selNode.op] : null;
    const hasPartPattern = !!(meta && (meta.params || []).some((p) => p.name === 'part_pattern'));

    if (selNode && hasPartPattern) {
      const curStr = String((selNode.params && selNode.params.part_pattern) || '');
      let nextPat = partShortName;
      if (additive && curStr) {
        const parts = curStr.split('|').filter(Boolean);
        if (!parts.includes(partShortName)) parts.push(partShortName);
        nextPat = parts.join('|');
      }
      await mutate({ action: 'set_param', node_id: selNode.id, param: 'part_pattern', value: nextPat }, { immediateEval: true });
      showToast(`🎯 Masked '${nextPat}' on ${selNode.id}`, 'ok');
      return;
    }

    // No part_pattern node selected -> spawn a ready-to-wire `mask_uv_part` node & preview it!
    const cx = Math.round((-graphCanvas.panX + graphCanvas.stage.clientWidth * 0.45) / graphCanvas.zoom - 90);
    const cy = Math.round((-graphCanvas.panY + graphCanvas.stage.clientHeight * 0.42) / graphCanvas.zoom - 50);
    const st = await mutate({
      action: 'add_node',
      op: 'mask_uv_part',
      params: { part_pattern: partShortName },
      pos: [cx, cy],
    }, { skipEval: true });
    if (st) {
      state.channel = 'selected';
      $('channelSelect').value = 'selected';
      await runEvaluate({ includeThumbs: true, includeQa: false });
      showToast(`🎯 Created 3D part mask '${partShortName}'`, 'ok');
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
    closeViewportPopovers();
    if ($('decalToolbar')) $('decalToolbar').style.display = (state.decalMode && mode === '3d') ? 'flex' : 'none';
    if ($('decalGizmoLayer')) $('decalGizmoLayer').style.display = (state.decalMode && mode === '3d') ? 'block' : 'none';
    if (state.decalMode && mode === '3d') updateDecalGizmoUI();
    syncSkelDockFromMesh((state.serverState && state.serverState.mesh) || null);
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
        graphCanvas.setData(state.serverState.recipe, state.evalData, state.selectedNodeId, state.activePass, state.selectedIds);
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
    'Math & Utility': '#a855f7',
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

  function stopLiveSwatchLoop() {
    if (state.liveSwatchTimer) {
      clearInterval(state.liveSwatchTimer);
      state.liveSwatchTimer = null;
    }
  }

  function startLiveSwatchLoop() {
    stopLiveSwatchLoop();
    if (!state.liveSwatches) return;
    state.liveSwatchTimer = setInterval(() => {
      if (!state.liveSwatches || !state.animSheet) return;
      state.animSheetFrame = (state.animSheetFrame + 1) % 6000;
      const updateScope = (container) => {
        if (!container) return;
        const imgs = container.querySelectorAll('img[data-anim-op]');
        for (const im of imgs) {
          const op = im.dataset.animOp;
          const frames = state.animSheet[op];
          if (frames && frames.length) {
            im.src = frames[state.animSheetFrame % frames.length];
          }
        }
      };
      if (state.libTab === 'nodes') updateScope($('libBody'));
      if ($('paletteModal') && $('paletteModal').classList.contains('open')) {
        updateScope($('paletteList'));
      }
    }, 185);
  }

  function switchLibTab(tab) {
    state.libTab = (tab === 'syncmat' || tab === 'syncpack') ? tab : 'nodes';
    if ($('tabLibNodes')) $('tabLibNodes').classList.toggle('active', state.libTab === 'nodes');
    if ($('tabLibSyncmat')) $('tabLibSyncmat').classList.toggle('active', state.libTab === 'syncmat');
    if ($('tabLibSyncpack')) $('tabLibSyncpack').classList.toggle('active', state.libTab === 'syncpack');
    if ($('libNodesSearchRow')) $('libNodesSearchRow').style.display = state.libTab === 'nodes' ? '' : 'none';
    if ($('libBody')) $('libBody').style.display = state.libTab === 'nodes' ? '' : 'none';
    if ($('syncmatBody')) $('syncmatBody').style.display = state.libTab === 'syncmat' ? 'flex' : 'none';
    if ($('syncpackBody')) $('syncpackBody').style.display = state.libTab === 'syncpack' ? 'flex' : 'none';
    if (state.libTab === 'syncmat') renderSyncmatList();
    if (state.libTab === 'syncpack') renderSyncpackList();
  }

  function renderSyncpackList() {
    const list = $('syncpackList');
    if (!list) return;
    list.innerHTML = '';
    const packs = state.syncpacks || [];
    if ($('syncpackCount')) $('syncpackCount').textContent = String(packs.length);
    if (!packs.length) {
      list.innerHTML = '<div style="font-size:11px;color:var(--text-muted);padding:6px;">No .syncpack macro packages found.</div>';
      return;
    }
    for (const pk of packs) {
      const card = document.createElement('div');
      card.className = 'syncmat-card';
      card.style.borderLeftColor = pk.color || '#7E22CE';
      const macros = Array.isArray(pk.macros) ? pk.macros : [];
      const tagsHtml = (pk.tags || []).slice(0, 4).map((t) => `<span class="syncpack-tag">${t}</span>`).join('');
      card.innerHTML = `
        <div class="syncmat-card-top">
          <span class="syncmat-card-name" title="${pk.name || pk.id}">${pk.name || pk.id}</span>
          <span class="syncmat-card-cat">${pk.builtin ? 'Built-in' : 'User'} · v${pk.version || '1.0'}</span>
        </div>
        <div class="syncmat-card-desc">${pk.description || ''}</div>
        <div style="display:flex;align-items:center;justify-content:space-between;gap:4px;">
          <div class="syncpack-tags">${tagsHtml}</div>
          <button class="btn sm" data-act="dl-pack" title="Download .syncpack.json" style="height:18px;font-size:9px;padding:0 5px;">⬇ .syncpack</button>
        </div>
        <div class="syncpack-macro-list"></div>
      `;
      card.querySelector('[data-act="dl-pack"]').onclick = () => {
        triggerDownload(`${pk.id || 'pack'}.syncpack.json`, JSON.stringify(pk, null, 2));
        showToast(`Downloaded ${pk.id}.syncpack.json`, 'ok');
      };
      const mList = card.querySelector('.syncpack-macro-list');
      for (const m of macros) {
        const mRow = document.createElement('div');
        mRow.className = 'syncpack-macro-row';
        const nLen = (m.nodes || []).length;
        mRow.innerHTML = `
          <div class="syncpack-macro-info">
            <span class="syncpack-macro-title" title="${m.name || m.id}">${m.name || m.id}</span>
            <span class="syncpack-macro-desc" title="${m.description || ''}">${m.description || `${nLen} nodes`}</span>
          </div>
          <button class="btn sm primary" data-spawn-macro="${m.id}" style="height:20px;font-size:9.5px;padding:0 6px;flex-shrink:0;">+ Spawn</button>
        `;
        mRow.querySelector('[data-spawn-macro]').onclick = async () => {
          try {
            const cx = Math.round((-graphCanvas.panX + graphCanvas.stage.clientWidth * 0.44) / graphCanvas.zoom - 110);
            const cy = Math.round((-graphCanvas.panY + graphCanvas.stage.clientHeight * 0.42) / graphCanvas.zoom - 50);
            const st = await apiPost('/api/syncpack/spawn', {
              pack_id: pk.id,
              macro_id: m.id,
              pos: [cx, cy],
              wrap_frame: true,
            });
            invalidateAnimRing();
            applyServerState(st, { autoPrim: false });
            await runEvaluate({ includeThumbs: true, includeQa: true });
            showToast(`Spawned '${m.name}' from ${pk.name} (${nLen} nodes)`, 'ok');
          } catch (err) {
            showToast(err.message, 'err');
          }
        };
        mList.appendChild(mRow);
      }
      list.appendChild(card);
    }
  }

  function renderSyncmatList() {
    const list = $('syncmatList');
    if (!list) return;
    list.innerHTML = '';
    const presets = state.syncmats || [];
    if ($('syncmatCount')) $('syncmatCount').textContent = String(presets.length);
    if (!presets.length) {
      list.innerHTML = '<div style="font-size:11px;color:var(--text-muted);padding:6px;">No .syncmat presets yet. Select nodes and click + Save .syncmat!</div>';
      return;
    }
    for (const p of presets) {
      const card = document.createElement('div');
      card.className = 'syncmat-card';
      card.style.borderLeftColor = p.frame_color || '#2563EB';
      const nCount = (p.nodes || []).length;
      const badge = p.builtin ? 'Built-in' : 'User';
      card.innerHTML = `
        <div class="syncmat-card-top">
          <span class="syncmat-card-name">${p.name || p.id}</span>
          <span class="syncmat-card-cat">${badge} · ${nCount}n</span>
        </div>
        <div class="syncmat-card-desc">${p.description || p.category || ''}</div>
        <div class="syncmat-card-acts">
          <button class="btn sm primary" data-act="insert">+ Insert into Graph</button>
          <button class="btn sm" data-act="dl" title="Download .syncmat.json">⬇ .syncmat</button>
        </div>
      `;
      card.querySelector('[data-act="insert"]').onclick = async () => {
        try {
          const cx = Math.round((-graphCanvas.panX + graphCanvas.stage.clientWidth * 0.45) / graphCanvas.zoom - 120);
          const cy = Math.round((-graphCanvas.panY + graphCanvas.stage.clientHeight * 0.42) / graphCanvas.zoom - 60);
          const st = await apiPost('/api/syncmat/insert', { preset_id: p.id, pos: [cx, cy], wrap_frame: true });
          applyServerState(st, { autoPrim: false });
          await runEvaluate({ includeThumbs: true, includeQa: true });
          showToast(`Inserted .syncmat '${p.name}' (${nCount} nodes)`, 'ok');
        } catch (err) {
          showToast(err.message, 'err');
        }
      };
      card.querySelector('[data-act="dl"]').onclick = () => {
        triggerDownload(`${p.id || 'material'}.syncmat.json`, JSON.stringify(p, null, 2));
        showToast(`Downloaded ${p.id}.syncmat.json`, 'ok');
      };
      list.appendChild(card);
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
      const defaultOpen = !!q || cat === 'Noise' || cat === 'Math & Utility' || cat === '3D Model & UV Masks' || cat === 'Materials — Metal (PBR)' || cat.includes('Region');
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
        const animAttr = isAnim ? ` data-anim-op="${n.op}"` : '';
        if (isGrid) {
          it.innerHTML = `
            <div class="lib-swatch">${prevSrc ? `<img src="${prevSrc}" alt="${n.op}" loading="lazy"${animAttr} />` : ''}${isAnim ? '<span class="anim-pill-mini">▶</span>' : ''}</div>
            <div class="lib-item-name">${n.op}</div>
          `;
        } else {
          it.innerHTML = `
            <div class="lib-swatch-sm">${prevSrc ? `<img src="${prevSrc}" alt="${n.op}" loading="lazy"${animAttr} />` : ''}</div>
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
      const animAttr = isAnim ? ` data-anim-op="${it.op}"` : '';
      const el = document.createElement('div');
      if (isGrid) {
        el.className = 'palette-card' + (idx === state.paletteIdx ? ' sel' : '');
        el.style.setProperty('--card-accent', accent);
        el.innerHTML = `
          <div class="palette-card-thumb">
            ${prevSrc ? `<img src="${prevSrc}" alt="${it.op}" loading="lazy"${animAttr} />` : ''}
            ${isAnim ? '<span class="anim-pill-mini">▶ ANIM</span>' : ''}
            <span class="palette-port-pill" style="color:${pColors[it.output_type] || '#cbd5e1'};">${it.output_type}</span>
          </div>
          <div class="palette-card-title" title="${it.display_name}">${it.display_name}</div>
          <div class="palette-card-op">${it.op}</div>
        `;
      } else {
        el.className = 'palette-item' + (idx === state.paletteIdx ? ' sel' : '');
        el.innerHTML = `
          <div class="palette-list-thumb">${prevSrc ? `<img src="${prevSrc}" alt="${it.op}" loading="lazy"${animAttr} />` : ''}</div>
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

  // ------------------------------------------------------------------------
  // Milestone v16: Interactive Curve / Spline Graph Editor (Fritsch-Carlson)
  // ------------------------------------------------------------------------
  function sampleDriverToKeyframes(d) {
    if (Array.isArray(d.keyframes) && d.keyframes.length >= 2) {
      return d.keyframes.map((pt) => [Number(pt[0]), Number(pt[1]), pt[2] || 'smooth']);
    }
    const lo = Number(d.min_val != null ? d.min_val : 0.0);
    const hi = Number(d.max_val != null ? d.max_val : 1.0);
    const freq = Number(d.freq || 1.0);
    const off = Number(d.phase_offset || 0.0);
    const kfs = [];
    const nPts = 5;
    for (let i = 0; i < nPts; i++) {
      const ph = i / (nPts - 1);
      const t = (ph * freq + off) % 1.0;
      let u = 0.5 + 0.5 * Math.sin(t * Math.PI * 2);
      if (d.kind === 'sawtooth') u = t;
      else if (d.kind === 'pulse') u = t < (d.duty || 0.5) ? 1.0 : 0.0;
      else if (d.kind === 'wiggler') u = 0.5 + 0.35 * Math.sin(t * Math.PI * 2) + 0.15 * Math.cos(t * Math.PI * 6);
      kfs.push([Number(ph.toFixed(3)), Number((lo + (hi - lo) * Math.max(0, Math.min(1, u))).toFixed(4)), 'smooth']);
    }
    return kfs;
  }

  function evalMonotoneSplineJS(kfs, ph) {
    const pts = kfs.slice().sort((a, b) => a[0] - b[0]);
    const n = pts.length;
    if (!n) return 0;
    if (n === 1 || ph <= pts[0][0]) return pts[0][1];
    if (ph >= pts[n - 1][0]) return pts[n - 1][1];
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const hs = [], ds = [];
    for (let i = 0; i < n - 1; i++) {
      const h = Math.max(1e-6, xs[i + 1] - xs[i]);
      hs.push(h);
      ds.push((ys[i + 1] - ys[i]) / h);
    }
    const ms = new Array(n).fill(0);
    ms[0] = ds[0];
    ms[n - 1] = ds[n - 2];
    for (let i = 1; i < n - 1; i++) {
      if (ds[i - 1] * ds[i] <= 0) ms[i] = 0;
      else {
        const w1 = 2 * hs[i] + hs[i - 1];
        const w2 = hs[i] + 2 * hs[i - 1];
        ms[i] = (w1 + w2) / (w1 / ds[i - 1] + w2 / ds[i]);
      }
    }
    for (let i = 0; i < n - 1; i++) {
      if (Math.abs(ds[i]) < 1e-8) { ms[i] = 0; ms[i + 1] = 0; }
      else {
        const a = ms[i] / ds[i], b = ms[i + 1] / ds[i];
        const r2 = a * a + b * b;
        if (r2 > 9) {
          const tau = 3 / Math.sqrt(r2);
          ms[i] = tau * a * ds[i];
          ms[i + 1] = tau * b * ds[i];
        }
      }
    }
    let idx = 0;
    for (let i = 0; i < n - 1; i++) {
      if (ph >= xs[i] && ph <= xs[i + 1]) { idx = i; break; }
    }
    const h = hs[idx];
    const s = Math.max(0, Math.min(1, (ph - xs[idx]) / h));
    const s2 = s * s, s3 = s2 * s;
    return (2 * s3 - 3 * s2 + 1) * ys[idx] + (s3 - 2 * s2 + s) * h * ms[idx]
      + (-2 * s3 + 3 * s2) * ys[idx + 1] + (s3 - s2) * h * ms[idx + 1];
  }

  function evalCurveAtPhaseJS(kfs, ph, mode) {
    const pts = kfs.slice().sort((a, b) => a[0] - b[0]);
    if (!pts.length) return 0;
    if (pts.length === 1 || ph <= pts[0][0]) return pts[0][1];
    if (ph >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];
    if (mode === 'spline') return evalMonotoneSplineJS(pts, ph);
    for (let i = 0; i < pts.length - 1; i++) {
      if (ph >= pts[i][0] && ph <= pts[i + 1][0]) {
        if (mode === 'step') return pts[i][1];
        const u = (ph - pts[i][0]) / Math.max(1e-6, pts[i + 1][0] - pts[i][0]);
        return pts[i][1] + (pts[i + 1][1] - pts[i][1]) * u;
      }
    }
    return pts[pts.length - 1][1];
  }

  function getCurveBounds(kfs, driver) {
    const vals = kfs.map((p) => p[1]);
    if (driver) {
      if (driver.min_val != null) vals.push(Number(driver.min_val));
      if (driver.max_val != null) vals.push(Number(driver.max_val));
    }
    let lo = Math.min(...vals, 0);
    let hi = Math.max(...vals, 1);
    if (hi - lo < 0.05) { lo -= 0.25; hi += 0.25; }
    const pad = (hi - lo) * 0.12;
    return { lo: lo - pad, hi: hi + pad };
  }

  function getCurveInterpMode(driver) {
    if (!driver) return 'spline';
    if (driver.kind === 'bezier_curve' || driver.kind === 'spline') return 'spline';
    if (Array.isArray(driver.keyframes) && driver.keyframes.length && driver.keyframes[0][2] === 'step') return 'step';
    if (driver.kind === 'keyframes') return 'linear';
    return 'spline';
  }

  function renderCurveEditor() {
    const cv = $('curveEditorCanvas');
    const sel = $('curveDriverSelect');
    const interpSel = $('curveInterpSelect');
    if (!cv || !sel) return;
    const drivers = (state.serverState && state.serverState.recipe && state.serverState.recipe.drivers) || [];
    sel.innerHTML = '';
    if (!drivers.length) {
      const o = document.createElement('option');
      o.value = '-1';
      o.textContent = 'No @animate tracks (click ~ in Inspector)';
      sel.appendChild(o);
      state.curveDriverIdx = -1;
    } else {
      if (state.curveDriverIdx < 0 || state.curveDriverIdx >= drivers.length) state.curveDriverIdx = 0;
      drivers.forEach((d, idx) => {
        const o = document.createElement('option');
        o.value = String(idx);
        o.textContent = `${d.node_id}.${d.param_key} (${d.kind})`;
        if (idx === state.curveDriverIdx) o.selected = true;
        sel.appendChild(o);
      });
    }

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = cv.getBoundingClientRect();
    const cw = Math.max(240, Math.floor((rect.width || 560) * dpr));
    const ch = Math.max(76, Math.floor((rect.height || 92) * dpr));
    if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cw, ch);

    const padL = 34 * dpr, padR = 12 * dpr, padT = 10 * dpr, padB = 14 * dpr;
    const plotW = Math.max(10, cw - padL - padR);
    const plotH = Math.max(10, ch - padT - padB);

    // Grid lines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
    ctx.lineWidth = 1;
    for (let gx = 0; gx <= 4; gx++) {
      const x = padL + (gx / 4) * plotW;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
    }
    for (let gy = 0; gy <= 2; gy++) {
      const y = padT + (gy / 2) * plotH;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
    }

    if (!drivers.length || state.curveDriverIdx < 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = `${11 * dpr}px system-ui, sans-serif`;
      ctx.fillText('Bind any Inspector slider with ~ to edit its live Spline / Keyframe curve here.', padL + 8 * dpr, padT + plotH * 0.55);
      return;
    }

    const driver = drivers[state.curveDriverIdx];
    const mode = interpSel ? (state._curveModeOverride || getCurveInterpMode(driver)) : 'spline';
    if (interpSel && !state._curveModeOverride) interpSel.value = mode;
    const kfs = state.curveDragKfs || sampleDriverToKeyframes(driver);
    const { lo, hi } = getCurveBounds(kfs, driver);
    const toX = (t) => padL + Math.max(0, Math.min(1, t)) * plotW;
    const toY = (v) => padT + (1 - (v - lo) / Math.max(1e-6, hi - lo)) * plotH;
    state._curveGeom = { padL, padR, padT, padB, plotW, plotH, lo, hi, dpr };

    // Y axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = `${9 * dpr}px monospace`;
    ctx.fillText(hi.toFixed(2), 3 * dpr, padT + 7 * dpr);
    ctx.fillText(lo.toFixed(2), 3 * dpr, padT + plotH);

    // Fill area under curve
    ctx.beginPath();
    const samples = 120;
    for (let i = 0; i <= samples; i++) {
      const ph = i / samples;
      const v = evalCurveAtPhaseJS(kfs, ph, mode);
      const x = toX(ph), y = toY(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.lineTo(padL, padT + plotH);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
    grad.addColorStop(0, 'rgba(245, 158, 11, 0.28)');
    grad.addColorStop(1, 'rgba(245, 158, 11, 0.02)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Curve stroke
    ctx.beginPath();
    for (let i = 0; i <= samples; i++) {
      const ph = i / samples;
      const v = evalCurveAtPhaseJS(kfs, ph, mode);
      const x = toX(ph), y = toY(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2 * dpr;
    ctx.stroke();

    // Playhead line + live dot
    const playX = toX(state.phase);
    const playV = evalCurveAtPhaseJS(kfs, state.phase, mode);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.4 * dpr;
    ctx.beginPath(); ctx.moveTo(playX, padT); ctx.lineTo(playX, padT + plotH); ctx.stroke();
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath(); ctx.arc(playX, toY(playV), 3.2 * dpr, 0, Math.PI * 2); ctx.fill();

    // Draggable Keyframe Dots
    kfs.forEach((pt, idx) => {
      const kx = toX(pt[0]);
      const ky = toY(pt[1]);
      const isDrag = idx === state.curveDragIdx;
      ctx.beginPath();
      ctx.arc(kx, ky, (isDrag ? 5.6 : 4.4) * dpr, 0, Math.PI * 2);
      ctx.fillStyle = isDrag ? '#38bdf8' : '#fef3c7';
      ctx.fill();
      ctx.lineWidth = 1.6 * dpr;
      ctx.strokeStyle = isDrag ? '#ffffff' : '#d97706';
      ctx.stroke();
    });

    if ($('curveInfoBadge')) {
      $('curveInfoBadge').textContent = `${kfs.length} kfs · φ=${state.phase.toFixed(2)} → ${playV.toFixed(2)}`;
    }
  }

  function commitCurveKeyframes(kfs, mode, undoable) {
    const drivers = (state.serverState && state.serverState.recipe && state.serverState.recipe.drivers) || [];
    if (state.curveDriverIdx < 0 || state.curveDriverIdx >= drivers.length) return;
    const ease = mode === 'step' ? 'step' : (mode === 'linear' ? 'linear' : 'smooth');
    const kind = mode === 'spline' ? 'bezier_curve' : 'keyframes';
    const packed = kfs.slice().sort((a, b) => a[0] - b[0]).map((p) => [Number(p[0].toFixed(4)), Number(p[1].toFixed(4)), ease]);
    state._curveModeOverride = mode;
    mutate({
      action: 'update_driver',
      index: state.curveDriverIdx,
      kind,
      keyframes: packed,
      undoable: undoable !== false,
    }, { immediateEval: undoable !== false, delayMs: 60 });
  }

  function setupCurveEditorEvents() {
    const cv = $('curveEditorCanvas');
    if (!cv) return;
    const getCanvasPos = (e) => {
      const r = cv.getBoundingClientRect();
      const dpr = (state._curveGeom && state._curveGeom.dpr) || 1;
      return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
    };
    const hitKeyframe = (kfs, x, y) => {
      const g = state._curveGeom;
      if (!g) return -1;
      let best = -1, bestD = 12 * g.dpr;
      kfs.forEach((pt, idx) => {
        const kx = g.padL + pt[0] * g.plotW;
        const ky = g.padT + (1 - (pt[1] - g.lo) / Math.max(1e-6, g.hi - g.lo)) * g.plotH;
        const d = Math.hypot(x - kx, y - ky);
        if (d < bestD) { bestD = d; best = idx; }
      });
      return best;
    };

    cv.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const drivers = (state.serverState && state.serverState.recipe && state.serverState.recipe.drivers) || [];
      if (state.curveDriverIdx < 0 || state.curveDriverIdx >= drivers.length || !state._curveGeom) return;
      const driver = drivers[state.curveDriverIdx];
      const mode = $('curveInterpSelect') ? $('curveInterpSelect').value : 'spline';
      const kfs = sampleDriverToKeyframes(driver);
      const { x, y } = getCanvasPos(e);
      const hit = hitKeyframe(kfs, x, y);
      const g = state._curveGeom;
      if (hit >= 0) {
        state.curveDragIdx = hit;
        state.curveDragKfs = kfs;
        cv.setPointerCapture(e.pointerId);
        renderCurveEditor();
      } else {
        const ph = Math.max(0, Math.min(1, (x - g.padL) / g.plotW));
        const val = g.lo + (1 - (y - g.padT) / g.plotH) * (g.hi - g.lo);
        kfs.push([Number(ph.toFixed(3)), Number(val.toFixed(3)), mode === 'step' ? 'step' : 'smooth']);
        kfs.sort((a, b) => a[0] - b[0]);
        commitCurveKeyframes(kfs, mode, true);
        showToast(`Inserted keyframe at φ=${ph.toFixed(2)}`, 'ok');
      }
    });

    cv.addEventListener('pointermove', (e) => {
      if (state.curveDragIdx < 0 || !state.curveDragKfs || !state._curveGeom) return;
      const g = state._curveGeom;
      const { x, y } = getCanvasPos(e);
      const ph = Math.max(0, Math.min(1, (x - g.padL) / g.plotW));
      const val = g.lo + (1 - (y - g.padT) / g.plotH) * (g.hi - g.lo);
      state.curveDragKfs[state.curveDragIdx][0] = Number(ph.toFixed(3));
      state.curveDragKfs[state.curveDragIdx][1] = Number(val.toFixed(3));
      renderCurveEditor();
    });

    const finishDrag = () => {
      if (state.curveDragIdx < 0 || !state.curveDragKfs) return;
      const kfs = state.curveDragKfs;
      const mode = $('curveInterpSelect') ? $('curveInterpSelect').value : 'spline';
      state.curveDragIdx = -1;
      state.curveDragKfs = null;
      commitCurveKeyframes(kfs, mode, true);
    };
    cv.addEventListener('pointerup', finishDrag);
    cv.addEventListener('pointercancel', finishDrag);

    cv.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const drivers = (state.serverState && state.serverState.recipe && state.serverState.recipe.drivers) || [];
      if (state.curveDriverIdx < 0 || state.curveDriverIdx >= drivers.length || !state._curveGeom) return;
      const driver = drivers[state.curveDriverIdx];
      const kfs = sampleDriverToKeyframes(driver);
      if (kfs.length <= 2) {
        showToast('Curve requires at least 2 keyframes', 'err');
        return;
      }
      const { x, y } = getCanvasPos(e);
      const hit = hitKeyframe(kfs, x, y);
      if (hit >= 0) {
        kfs.splice(hit, 1);
        const mode = $('curveInterpSelect') ? $('curveInterpSelect').value : 'spline';
        commitCurveKeyframes(kfs, mode, true);
        showToast('Removed keyframe', 'ok');
      }
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
    if ($('botTabCurves')) $('botTabCurves').style.display = tab === 'curves' ? 'flex' : 'none';
    $('botTabExposed').style.display = tab === 'exposed' ? 'flex' : 'none';
    $('botTabConsole').style.display = tab === 'console' ? 'flex' : 'none';
    if (tab === 'curves') requestAnimationFrame(() => renderCurveEditor());
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
    $('frameReadout').textContent = `F${fIdx}/${state.animFrames}`;
    renderTimelineDrivers();
    if ($('botTabCurves') && $('botTabCurves').style.display !== 'none') renderCurveEditor();
    if (immediate) return runEvaluate({ includeThumbsMode: 'dynamic', includeQa: false });
    if (evalTimer) clearTimeout(evalTimer);
    evalTimer = setTimeout(() => runEvaluate({ includeThumbsMode: 'dynamic', includeQa: false }), 22);
    return Promise.resolve();
  }

  function stopAnimLoop() {
    state.animPlaying = false;
    $('btnPlayAnim').textContent = '▶';
    $('btnPlayAnim').classList.remove('amber');
    if (state.animTimer) { clearTimeout(state.animTimer); state.animTimer = null; }
    if (state.animRafId) { cancelAnimationFrame(state.animRafId); state.animRafId = null; }
    if (gl3d) gl3d.setAnimFrameIdx(-1);
  }

  async function togglePlayAnim() {
    if (state.animPlaying) {
      stopAnimLoop();
      return;
    }
    state.animPlaying = true;
    const btn = $('btnPlayAnim');
    btn.classList.add('amber');

    const rec = state.serverState && state.serverState.recipe;
    const bakeEpoch = state.workspaceEpoch;
    const bakeRev = state.mutationRev;
    const seqKey = `e${bakeEpoch}:r${bakeRev}:${rec ? rec.name : ''}:${state.animFrames}:${state.channel}:${state.lightAz}:${state.lightEl}:${state.selectedNodeId || ''}`;
    const tileable = !!(rec && rec.tileable);

    if (state.animSeqKey !== seqKey || !gl3d || !gl3d.animRing.length) {
      btn.textContent = `⏳ ${state.animFrames}f...`;
      try {
        const res = await apiPost('/api/evaluate_anim_sequence', {
          frames: state.animFrames,
          channel: state.channel,
          light_az: state.lightAz,
          light_el: state.lightEl,
          preview_node: state.selectedNodeId,
          max_dim: 256,
        });
        if (!state.animPlaying || bakeEpoch !== state.workspaceEpoch || bakeRev !== state.mutationRev) {
          stopAnimLoop();
          return;
        }
        if (res.workspace_epoch != null && res.workspace_epoch !== state.workspaceEpoch) {
          stopAnimLoop();
          return;
        }
        const okUp = await gl3d.uploadAnimFrames(res.sequence || [], tileable);
        if (!okUp || !state.animPlaying || bakeEpoch !== state.workspaceEpoch || bakeRev !== state.mutationRev) {
          stopAnimLoop();
          return;
        }
        state.animSeqKey = seqKey;
        showToast(`⚡ GPU Ring Ready: ${res.frames}f baked in ${Math.round(res.bake_ms)}ms -> ${state.animFps}fps!`, 'ok');
      } catch (err) {
        stopAnimLoop();
        showToast(`Anim bake error: ${err.message}`, 'err');
        return;
      }
    }

    if (!state.animPlaying || !gl3d || !gl3d.animRing.length) {
      stopAnimLoop();
      return;
    }

    btn.textContent = '⏸';
    let lastStamp = performance.now();
    let curIdx = Math.floor(state.phase * gl3d.animRing.length) % gl3d.animRing.length;
    let dir = 1;

    const rafTick = (now) => {
      if (!state.animPlaying) return;
      const intervalMs = 1000 / Math.max(4, state.animFps);
      if (now - lastStamp >= intervalMs - 1.5) {
        lastStamp = now;
        const nF = gl3d.animRing.length;
        curIdx += dir;
        if (state.animPingPong) {
          if (curIdx >= nF) { dir = -1; curIdx = Math.max(0, nF - 2); }
          else if (curIdx < 0) { dir = 1; curIdx = Math.min(nF - 1, 1); }
        } else {
          curIdx = ((curIdx % nF) + nF) % nF;
        }
        const entry = gl3d.setAnimFrameIdx(curIdx);
        if (entry) {
          state.phase = entry.phase;
          $('phaseSlider').value = state.phase.toFixed(2);
          $('phaseLabel').textContent = state.phase.toFixed(2);
          $('frameReadout').textContent = `⚡F${curIdx}/${nF}`;
          if ($('botTabCurves') && $('botTabCurves').style.display !== 'none') renderCurveEditor();
          if (state.vpMode === '2d' && entry.displayImg) {
            state.img2d = entry.displayImg;
            render2DCanvas();
          }
          if (entry.thumbs && Object.keys(entry.thumbs).length) {
            graphCanvas.updateNodeThumbs(entry.thumbs);
          }
        }
      }
      state.animRafId = requestAnimationFrame(rafTick);
    };
    state.animRafId = requestAnimationFrame(rafTick);
  }

  function triggerDataUrlDownload(filename, dataUrl) {
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = filename;
    a.click();
  }

  function triggerDownload(filename, textContent) {
    const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  async function createFrameFromSelection() {
    const ids = getActiveSelectionIds();
    if (!ids.length) {
      showToast('Select one or more nodes first to create a Comment Frame', 'err');
      return;
    }
    const title = ids.length > 1 ? `Material Group (${ids.length})` : `${ids[0]} Group`;
    await mutate({ action: 'create_frame', node_ids: ids, title }, { skipEval: true });
    showToast(`Wrapped ${ids.length} node(s) in Comment Frame`, 'ok');
  }

  async function groupSelectionIntoSubgraph() {
    const ids = getActiveSelectionIds();
    if (!ids.length) {
      showToast('Select one or more nodes first to collapse into a Compound Subgraph', 'err');
      return;
    }
    const title = ids.length > 1 ? `Subgraph (${ids.length}n)` : `${ids[0]} Group`;
    await mutate({ action: 'group_nodes', ids, title }, { immediateEval: true });
    showToast(`Collapsed ${ids.length} node(s) into Compound Subgraph (double-click to edit inside!)`, 'ok');
  }

  // ------------------------------------------------------------------------
  // Bootstrap Studio Application
  // ------------------------------------------------------------------------
  async function boot() {
    gl3d = new StudioGL3DViewport($('gl3dCanvas'), $('vpHud'));
    gl3d.onPickPart = (hit, shift) => handlePickedPart(hit.short, shift);
    gl3d.onHoverPart = (hit) => updateViewportHud(hit ? `Pick: ${hit.short}` : null);
    gl3d.onBrush3d = (u, v, isStart) => queueBrushStroke(u, v, isStart);
    gl3d.onDecalStamp = (proj) => applyDecalStampFromProjector(proj, 'add');
    gl3d.onSkelTimeUpdate = (t, len, playing) => {
      if ($('skelTimeSlider')) {
        $('skelTimeSlider').max = Math.max(0.05, len).toFixed(3);
        $('skelTimeSlider').value = t.toFixed(3);
      }
      if ($('skelTimeVal')) $('skelTimeVal').textContent = `${t.toFixed(2)}s`;
      if ($('btnSkelPlay')) {
        $('btnSkelPlay').textContent = playing ? '⏸' : '▶';
        $('btnSkelPlay').classList.toggle('amber', !!playing);
      }
    };
    gl3d.onLightChange = (az, el, envRot) => {
      state.lightAz = az;
      state.lightEl = el;
      state.envRotDeg = envRot;
      if ($('lightAzSlider')) $('lightAzSlider').value = String(az);
      if ($('lightAzVal')) $('lightAzVal').textContent = `${az}°`;
      if ($('lightElSlider')) $('lightElSlider').value = String(el);
      if ($('lightElVal')) $('lightElVal').textContent = `${el}°`;
      if ($('envRotSlider')) $('envRotSlider').value = String(envRot);
      if ($('envRotVal')) $('envRotVal').textContent = `${envRot}°`;
    };

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
          state.selectedIds = new Set(nid ? [nid] : []);
          syncSelectionUI();
          renderInspector();
          graphCanvas.setData(state.serverState.recipe, state.evalData, state.selectedNodeId, state.activePass, state.selectedIds);
          if (inspectInVp) {
            state.channel = 'selected';
            $('channelSelect').value = 'selected';
            runEvaluate({ includeThumbs: false, includeQa: false });
          } else if (state.channel === 'selected') {
            // Keep 'Selected Node Output' strictly 1-to-1 with whichever node the user clicks!
            runEvaluate({ includeThumbsMode: 'none', includeQa: false });
          }
        },
        onSelectionChange: (ids, primary) => {
          state.selectedIds = new Set(ids || []);
          state.selectedNodeId = primary || (ids && ids[0]) || null;
          syncSelectionUI();
          renderInspector();
          graphCanvas.setData(state.serverState.recipe, state.evalData, state.selectedNodeId, state.activePass, state.selectedIds);
        },
        onMoveNode: (nid, pos) => mutate({ action: 'move_nodes', positions: { [nid]: pos }, undoable: true }, { skipEval: true }),
        onMoveNodes: (positions) => mutate({ action: 'move_nodes', positions, undoable: true }, { skipEval: true }),
        onConnect: (src, srcPort, dst, dstPort) => mutate({ action: 'connect', src_node: src, src_port: srcPort, dst_node: dst, dst_port: dstPort }, { immediateEval: true }),
        onDisconnect: (dst, dstPort) => mutate({ action: 'disconnect', dst_node: dst, dst_port: dstPort }, { immediateEval: true }),
        onInsertReroute: (dst, dstPort, pos) => {
          mutate({ action: 'insert_reroute', dst_node: dst, dst_port: dstPort, pos }, { immediateEval: true });
          showToast('Inserted Reroute Dot on wire', 'ok');
        },
        onInsertNodeOnWire: (nid, dst, dstPort) => {
          mutate({ action: 'insert_node_on_wire', node_id: nid, dst_node: dst, dst_port: dstPort }, { immediateEval: true });
          showToast(`Auto-wired '${nid}' inline on wire!`, 'ok');
        },
        onEnterGroup: (gid) => {
          mutate({ action: 'enter_group', node_id: gid }, { immediateEval: true, fitGraph: true });
        },
        onUpdateFrame: (fid, patch) => mutate(Object.assign({ action: 'update_frame', frame_id: fid }, patch), { skipEval: true }),
        onDeleteFrame: (fid) => {
          mutate({ action: 'delete_frame', frame_id: fid }, { skipEval: true });
          showToast('Removed Comment Frame', 'ok');
        },
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
        onNodeContextMenu: (nid, cx, cy, selIds) => {
          const targetIds = (Array.isArray(selIds) && selIds.length > 1) ? selIds : [nid];
          const multi = targetIds.length > 1;
          const nObj = ((state.serverState && state.serverState.recipe && state.serverState.recipe.nodes) || []).find((q) => q.id === nid);
          const meta = nObj ? state.opMap[nObj.op] : null;
          const items = [
            { label: '👁 Inspect in Viewport', kbd: 'Dbl-Clk', action: () => { state.channel = 'selected'; $('channelSelect').value = 'selected'; runEvaluate({ includeThumbs: false, includeQa: false }); } },
            ...(nObj && nObj.op === 'subgraph_group' ? [
              { label: '↗ Enter Compound Subgraph', kbd: 'Dbl-Clk', action: () => mutate({ action: 'enter_group', node_id: nid }, { immediateEval: true, fitGraph: true }) },
              { label: '⊟ Ungroup Subgraph', kbd: 'Ctrl+Alt+G', action: () => mutate({ action: 'ungroup_node', node_id: nid }, { immediateEval: true }) },
            ] : []),
            { label: '★ Set Final Graph Output', action: () => mutate({ action: 'set_output', node_id: nid }, { immediateEval: true }) },
            { sep: true },
            { label: `⊞ Group ${multi ? targetIds.length + ' Nodes' : 'Node'} into Subgraph`, kbd: 'Ctrl+G', action: () => groupSelectionIntoSubgraph() },
            { label: `▭ Wrap ${multi ? targetIds.length + ' Nodes' : 'Node'} in Frame`, kbd: 'Ctrl+J', action: () => createFrameFromSelection() },
            { label: `⧉ Duplicate ${multi ? targetIds.length + ' Nodes' : 'Node'}`, kbd: '⌘D', action: () => mutate({ action: 'duplicate_nodes', ids: targetIds }, { immediateEval: true }) },
            { label: '🎲 Randomize Seed', action: () => mutate({ action: 'randomize_seed', node_id: nid }, { immediateEval: true }) },
            { label: nObj && nObj.bypassed ? '✓ Enable Node(s)' : 'B Bypass / Mute', kbd: 'B', action: () => mutate(multi ? { action: 'toggle_bypass_multi', ids: targetIds } : { action: 'toggle_bypass', node_id: nid }, { immediateEval: true }) },
            { label: '✂ Disconnect All Wires', action: () => mutate({ action: 'disconnect_all', node_id: nid }, { immediateEval: true }) },
            { sep: true },
            { label: `✕ Delete ${multi ? targetIds.length + ' Nodes' : 'Node'}`, kbd: 'Del', danger: true, action: () => { mutate({ action: 'delete_nodes', ids: targetIds }, { immediateEval: true }); showToast(`Deleted ${targetIds.length} node(s)`, 'ok'); } },
          ];
          showContextMenu(cx, cy, multi ? `${targetIds.length} Selected Nodes` : `${nid}${meta ? ` (${meta.op})` : ''}`, items);
        },
        onCanvasContextMenu: (cx, cy, wPos) => {
          showContextMenu(cx, cy, 'Node Graph Canvas', [
            { label: '+ Add Node Here...', kbd: 'Tab', action: () => openPalette({ spawnPos: [Math.round(wPos.x), Math.round(wPos.y)] }) },
            { label: '• Add Reroute Dot Here', action: () => mutate({ action: 'add_node', op: 'reroute', pos: [Math.round(wPos.x), Math.round(wPos.y)] }, { skipEval: true }) },
            { label: '▭ Wrap Selection in Frame', kbd: 'Ctrl+J', action: () => createFrameFromSelection() },
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

    // Progressive Non-Blocking Boot: fetch Schema + State first (~10ms), evaluate immediately,
    // and stream 228 Node Previews + Live Anim Sheet + .syncmat Presets in parallel!
    const [schemaPayload, initState] = await Promise.all([
      apiGet('/api/schema'),
      apiGet('/api/state'),
    ]);
    state.schemaPayload = schemaPayload;
    for (const [cat, nodes] of Object.entries(schemaPayload.schema.nodes_by_category || {})) {
      for (const n of nodes) state.opMap[n.op] = Object.assign({ category: cat }, n);
    }
    graphCanvas.setSchema(schemaPayload.schema);
    $('versionBadge').textContent = `v19 · ${schemaPayload.schema.node_count}`;
    state.syncpacks = (schemaPayload && schemaPayload.syncpacks) || [];
    populateDecalStencils();
    populateExportTemplates();

    // Kick off non-blocking background fetch for 234 procedural swatches, anim sheet, .syncmat & .syncpack presets
    const bgAssetsPromise = Promise.all([
      apiGet('/api/schema/previews?size=40').catch(() => ({ previews: {} })),
      apiGet('/api/schema/preview_anim_sheet?size=36&frames=6').catch(() => ({ sheet: {} })),
      apiGet('/api/syncmat/list').catch(() => ({ presets: [] })),
      apiGet('/api/syncpack/list').catch(() => ({ packs: state.syncpacks })),
    ]).then(([prevPayload, animSheetPayload, syncmatPayload, syncpackPayload]) => {
      state.nodePreviews = (prevPayload && prevPayload.previews) || {};
      state.timeAwareOps = new Set((prevPayload && prevPayload.time_aware_ops) || []);
      state.animSheet = (animSheetPayload && animSheetPayload.sheet) || {};
      state.syncmats = (syncmatPayload && syncmatPayload.presets) || [];
      if (syncpackPayload && Array.isArray(syncpackPayload.packs)) state.syncpacks = syncpackPayload.packs;
      renderNodeLibrary();
      renderSyncmatList();
      renderSyncpackList();
      startLiveSwatchLoop();
    });

    // Populate Project / Preset Dropdown
    const pSel = $('projectSelect');
    const ogSamples = document.createElement('optgroup');
    ogSamples.label = 'Flagship 3D Studio Projects (v19)';
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
    renderSyncmatList();
    renderSyncpackList();
    applyServerState(initState, { fitGraph: true, autoPrim: true });
    await runEvaluate({ includeThumbs: true, includeQa: true });
    await bgAssetsPromise;
    setTimeout(() => graphCanvas.fitToView(), 80);

    // ----------------------------------------------------------------------
    // Event Listeners
    // ----------------------------------------------------------------------
    pSel.onchange = async () => {
      const [kind, id] = pSel.value.split(':');
      try {
        // Complete visual, animation ring, GPU texture & in-flight eval reset BEFORE switching!
        resetWorkspaceVisuals({ fullProjectReset: true });
        const st = await apiPost('/api/load', { kind, id, size: parseInt($('resSelect').value, 10) || 64 });
        applyServerState(st, { fitGraph: true, autoPrim: true });
        await runEvaluate({ includeThumbs: true, includeQa: true });
        requestAnimationFrame(() => graphCanvas.fitToView());
        showToast(`Loaded ${st.recipe.name} (${st.recipe.width}×${st.recipe.height})`, 'ok');
      } catch (err) {
        showToast(err.message, 'err');
      }
    };

    // Strict 1-to-1 Project File Opener (.syncproj, .texdsl, .texgraph.json, .bbmodel)
    if ($('btnOpenProject') && $('projectFileInput')) {
      $('btnOpenProject').onclick = () => $('projectFileInput').click();
      $('projectFileInput').onchange = async (e) => {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        try {
          resetWorkspaceVisuals({ fullProjectReset: true });
          const text = await f.text();
          const lower = f.name.toLowerCase();
          let payload;
          if (lower.endsWith('.texdsl')) {
            payload = { kind: 'texdsl', code: text, filename: f.name, keep_mesh: false };
          } else if (lower.endsWith('.bbmodel')) {
            payload = { kind: 'bbmodel_project', bbmodel: JSON.parse(text), filename: f.name };
          } else if (lower.endsWith('.syncproj')) {
            payload = { kind: 'syncproj', syncproj: JSON.parse(text), filename: f.name };
          } else {
            const doc = JSON.parse(text);
            if (doc.schema && String(doc.schema).startsWith('synccraft.project')) {
              payload = { kind: 'syncproj', syncproj: doc, filename: f.name };
            } else if (doc.elements && doc.resolution) {
              payload = { kind: 'bbmodel_project', bbmodel: doc, filename: f.name };
            } else {
              payload = { kind: 'recipe', recipe: doc, filename: f.name, keep_mesh: false };
            }
          }
          const st = await apiPost('/api/load', payload);
          applyServerState(st, { fitGraph: true, autoPrim: true });
          await runEvaluate({ includeThumbs: true, includeQa: true });
          requestAnimationFrame(() => graphCanvas.fitToView());
          showToast(`Opened 1-to-1 project '${f.name}' (${st.recipe.width}×${st.recipe.height})`, 'ok');
        } catch (err) {
          showToast(`Open failed: ${err.message}`, 'err');
        }
        e.target.value = '';
      };
    }

    // Persistent File Log Controls (File Tail, Download .log, Clear)
    if ($('btnRefreshFileLog')) {
      $('btnRefreshFileLog').onclick = async () => {
        try {
          const info = await apiGet('/api/logs/file?tail=80');
          if ($('logFilePathSpan')) {
            $('logFilePathSpan').textContent = `${info.path} (${(info.size_bytes / 1024).toFixed(1)} KB · ${info.line_count} lines)`;
          }
          const list = $('consoleLogList');
          if (list) {
            list.innerHTML = '';
            for (const line of info.lines.slice().reverse()) {
              const d = document.createElement('div');
              d.className = 'log-line' + (line.includes('"ERROR"') ? ' error' : '');
              d.textContent = line;
              list.appendChild(d);
            }
          }
          showToast(`Loaded ${info.line_count} lines from studio file log`, 'ok');
        } catch (err) {
          showToast(err.message, 'err');
        }
      };
    }
    if ($('btnDownloadFileLog')) {
      $('btnDownloadFileLog').onclick = () => {
        window.open('/api/logs/file?tail=2000&download=1', '_blank');
      };
    }
    if ($('btnClearFileLog')) {
      $('btnClearFileLog').onclick = async () => {
        try {
          await apiPost('/api/logs/clear', {});
          const st = await apiGet('/api/state');
          applyServerState(st, { autoPrim: false });
          showToast('Cleared persistent Studio file log', 'ok');
        } catch (err) {
          showToast(err.message, 'err');
        }
      };
    }

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
      if (state.channel === 'selected') {
        runEvaluate({ includeThumbsMode: 'none', includeQa: false });
      } else {
        updateViewportTextures();
      }
    };
    $('primSelect').onchange = (e) => {
      gl3d.primitive = e.target.value;
      syncSkelDockFromMesh((state.serverState && state.serverState.mesh) || null);
      updateViewportTextures();
    };
    if ($('btnOrbit3d')) $('btnOrbit3d').onclick = () => setViewportTool('orbit');
    if ($('btnDecal3d')) $('btnDecal3d').onclick = () => setViewportTool(state.decalMode ? 'orbit' : 'decal');
    if ($('btnLightingPop')) $('btnLightingPop').onclick = (e) => { e.stopPropagation(); toggleViewportPopover('lighting'); };
    if ($('btnViewPop')) $('btnViewPop').onclick = (e) => { e.stopPropagation(); toggleViewportPopover('view'); };
    window.addEventListener('pointerdown', (e) => {
      if (!e.target.closest('.vp-popover') && !e.target.closest('.vp-pop-triggers')) {
        closeViewportPopovers();
      }
    });
    if ($('btnBones3d')) $('btnBones3d').onclick = () => setShowBones(!state.showBones);
    if ($('btnSkelBonesQuick')) $('btnSkelBonesQuick').onclick = () => setShowBones(!state.showBones);

    // Live 3D Skeletal Animation Transport Dock + BB5 Live Pose Sync (`v18`)
    if ($('skelClipSelect')) {
      $('skelClipSelect').onchange = (e) => {
        const idx = parseInt(e.target.value, 10);
        if (gl3d) gl3d.setSkeletonClip(Number.isFinite(idx) ? idx : -1, idx >= 0);
        syncSkelDockFromMesh((state.serverState && state.serverState.mesh) || null);
        scheduleBridgePoseSync();
      };
    }
    if ($('btnSkelPlay')) {
      $('btnSkelPlay').onclick = () => {
        if (!gl3d) return;
        gl3d.setSkeletonPlaying(!gl3d.skelPlaying);
        syncSkelDockFromMesh((state.serverState && state.serverState.mesh) || null);
        scheduleBridgePoseSync();
      };
    }
    if ($('skelTimeSlider')) {
      const scrubSkel = (e) => {
        if (!gl3d) return;
        const targetT = parseFloat(e.target.value);
        gl3d.skelPlaying = false;
        gl3d.setSkeletonTime(targetT);
        if ($('btnSkelPlay')) {
          $('btnSkelPlay').textContent = '▶';
          $('btnSkelPlay').classList.remove('amber');
        }
        scheduleBridgePoseSync();
      };
      $('skelTimeSlider').addEventListener('pointerdown', () => {
        if (gl3d && gl3d.skelPlaying) {
          gl3d.setSkeletonPlaying(false);
          if ($('btnSkelPlay')) {
            $('btnSkelPlay').textContent = '▶';
            $('btnSkelPlay').classList.remove('amber');
          }
        }
      });
      $('skelTimeSlider').oninput = scrubSkel;
      $('skelTimeSlider').onchange = scrubSkel;
    }
    if ($('btnSkelSpeed')) {
      const skelSpeeds = [0.5, 1.0, 1.5, 2.0];
      $('btnSkelSpeed').onclick = () => {
        state.skelSpeedIdx = (state.skelSpeedIdx + 1) % skelSpeeds.length;
        const spd = skelSpeeds[state.skelSpeedIdx];
        if (gl3d) gl3d.skelSpeed = spd;
        $('btnSkelSpeed').textContent = `${spd.toFixed(1)}×`;
        scheduleBridgePoseSync();
      };
    }

    // View Popover: Auto-LOD Switcher & Multi-Texture `.bbmodel` Slot Selector (`v18`)
    if ($('btnLod0')) $('btnLod0').onclick = () => switchLodLevel('lod0');
    if ($('btnLod1')) $('btnLod1').onclick = () => switchLodLevel('lod1');
    if ($('btnLod2')) $('btnLod2').onclick = () => switchLodLevel('lod2');
    if ($('texSlotSelect')) {
      $('texSlotSelect').onchange = async (e) => {
        const slot = parseInt(e.target.value, 10);
        try {
          const st = await apiPost('/api/mesh/texture_slot', { slot: Number.isFinite(slot) ? slot : -1 });
          applyServerState(st, { autoPrim: false });
          showToast(slot >= 0 ? `Isolated .bbmodel texture slot [${slot}]` : 'Showing all .bbmodel texture slots', 'ok');
        } catch (err) {
          showToast(`Texture slot error: ${err.message}`, 'err');
        }
      };
    }

    // Floating 3D Decal Stamp Toolbar + Interactive 3D Transform Gizmo + Custom PNG Upload (`v18`)
    if ($('decalStencilSelect')) {
      $('decalStencilSelect').onchange = (e) => {
        const sid = e.target.value;
        state.decal.stencil = sid;
        state.decal.custom_data_url = '';
        state.decal.custom_name = '';
        const meta = (state.schemaPayload && state.schemaPayload.decal_stencils && state.schemaPayload.decal_stencils[sid]) || null;
        if (meta) {
          state.decal.tint_color = meta.color || state.decal.tint_color;
          state.decal.emissive_strength = meta.emissive_strength != null ? meta.emissive_strength : 0.5;
          if ($('decalTintColor')) $('decalTintColor').value = state.decal.tint_color;
        }
        updateDecalGizmoUI();
      };
    }
    if ($('btnDecalUploadPng') && $('decalPngFileInput')) {
      $('btnDecalUploadPng').onclick = () => $('decalPngFileInput').click();
      $('decalPngFileInput').onchange = (e) => {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        const reader = new FileReader();
        reader.onload = () => {
          state.decal.custom_data_url = String(reader.result || '');
          state.decal.custom_name = f.name || 'custom_stencil.png';
          updateDecalGizmoUI();
          showToast(`Loaded custom PNG stencil: ${state.decal.custom_name}`, 'ok');
        };
        reader.readAsDataURL(f);
        e.target.value = '';
      };
    }
    if ($('btnDecalResetGizmo')) {
      $('btnDecalResetGizmo').onclick = () => {
        state.decalGizmo.nx = 0.5;
        state.decalGizmo.ny = 0.56;
        state.decal.scale_u = 10.0;
        state.decal.roll_deg = 0;
        if ($('decalScaleSlider')) $('decalScaleSlider').value = '10';
        if ($('decalScaleVal')) $('decalScaleVal').textContent = '10';
        if ($('decalRollSlider')) $('decalRollSlider').value = '0';
        if ($('decalRollVal')) $('decalRollVal').textContent = '0°';
        updateDecalGizmoUI();
      };
    }
    if ($('btnDecalClearCustom')) {
      $('btnDecalClearCustom').onclick = () => {
        state.decal.custom_data_url = '';
        state.decal.custom_name = '';
        updateDecalGizmoUI();
        showToast('Restored built-in procedural stencil', 'ok');
      };
    }
    if ($('decalTintColor')) $('decalTintColor').oninput = (e) => { state.decal.tint_color = e.target.value; };
    if ($('decalScaleSlider')) {
      $('decalScaleSlider').oninput = (e) => {
        state.decal.scale_u = parseFloat(e.target.value) || 10;
        if ($('decalScaleVal')) $('decalScaleVal').textContent = String(state.decal.scale_u);
        updateDecalGizmoUI();
      };
    }
    if ($('decalRollSlider')) {
      $('decalRollSlider').oninput = (e) => {
        state.decal.roll_deg = parseFloat(e.target.value) || 0;
        if ($('decalRollVal')) $('decalRollVal').textContent = `${state.decal.roll_deg}°`;
        updateDecalGizmoUI();
      };
    }
    if ($('decalAngleSlider')) {
      $('decalAngleSlider').oninput = (e) => {
        state.decal.max_angle_deg = parseFloat(e.target.value) || 72;
        if ($('decalAngleVal')) $('decalAngleVal').textContent = `${state.decal.max_angle_deg}°`;
      };
    }
    if ($('decalOpacitySlider')) {
      $('decalOpacitySlider').oninput = (e) => {
        state.decal.opacity = parseFloat(e.target.value) || 0.95;
        if ($('decalOpacityVal')) $('decalOpacityVal').textContent = `${Math.round(state.decal.opacity * 100)}%`;
        updateDecalGizmoUI();
      };
    }
    // Interactive 3D Decal Transform Gizmo Pointer Manipulators (Move / Scale / Roll)
    if ($('decalGizmoBox')) {
      const gizmoBox = $('decalGizmoBox');
      const stageEl = $('viewportStage');
      let gizmoDrag = null;
      const onGizmoMove = (e) => {
        if (!gizmoDrag || !stageEl) return;
        const rect = stageEl.getBoundingClientRect();
        if (gizmoDrag.kind === 'move') {
          const nx = (e.clientX - rect.left - gizmoDrag.offX) / Math.max(1, rect.width);
          const ny = (e.clientY - rect.top - gizmoDrag.offY) / Math.max(1, rect.height);
          state.decalGizmo.nx = Math.max(0.08, Math.min(0.92, nx));
          state.decalGizmo.ny = Math.max(0.10, Math.min(0.90, ny));
        } else if (gizmoDrag.kind === 'scale') {
          const cx = rect.left + state.decalGizmo.nx * rect.width;
          const cy = rect.top + state.decalGizmo.ny * rect.height;
          const dist = Math.hypot(e.clientX - cx, e.clientY - cy);
          const nextScale = Math.max(3, Math.min(24, Math.round((dist / 6.0) * 2) / 2));
          state.decal.scale_u = nextScale;
          if ($('decalScaleSlider')) $('decalScaleSlider').value = String(nextScale);
          if ($('decalScaleVal')) $('decalScaleVal').textContent = String(nextScale);
        } else if (gizmoDrag.kind === 'roll') {
          const cx = rect.left + state.decalGizmo.nx * rect.width;
          const cy = rect.top + state.decalGizmo.ny * rect.height;
          let deg = (Math.atan2(e.clientY - cy, e.clientX - cx) * 180 / Math.PI) + 90;
          if (deg > 180) deg -= 360;
          if (deg < -180) deg += 360;
          if (e.shiftKey) deg = Math.round(deg / 15) * 15;
          else deg = Math.round(deg);
          state.decal.roll_deg = Math.max(-180, Math.min(180, deg));
          if ($('decalRollSlider')) $('decalRollSlider').value = String(state.decal.roll_deg);
          if ($('decalRollVal')) $('decalRollVal').textContent = `${state.decal.roll_deg}°`;
        }
        updateDecalGizmoUI();
      };
      const onGizmoUp = () => {
        if (!gizmoDrag) return;
        gizmoDrag = null;
        gizmoBox.classList.remove('dragging');
        window.removeEventListener('pointermove', onGizmoMove);
        window.removeEventListener('pointerup', onGizmoUp);
      };
      const startGizmoDrag = (kind, e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!stageEl) return;
        const rect = stageEl.getBoundingClientRect();
        const cx = rect.left + state.decalGizmo.nx * rect.width;
        const cy = rect.top + state.decalGizmo.ny * rect.height;
        gizmoDrag = { kind, offX: e.clientX - cx, offY: e.clientY - cy };
        gizmoBox.classList.add('dragging');
        window.addEventListener('pointermove', onGizmoMove);
        window.addEventListener('pointerup', onGizmoUp);
      };
      gizmoBox.addEventListener('pointerdown', (e) => {
        if (e.target && e.target.closest && (e.target.closest('#decalGizmoScaleHandle') || e.target.closest('#decalGizmoRotHandle'))) return;
        startGizmoDrag('move', e);
      });
      if ($('decalGizmoScaleHandle')) {
        $('decalGizmoScaleHandle').addEventListener('pointerdown', (e) => startGizmoDrag('scale', e));
      }
      if ($('decalGizmoRotHandle')) {
        $('decalGizmoRotHandle').addEventListener('pointerdown', (e) => startGizmoDrag('roll', e));
      }
    }
    if ($('btnDecalStampCenter')) {
      $('btnDecalStampCenter').onclick = () => {
        if (!gl3d) return;
        applyDecalStampFromProjector(gl3d.getCameraProjectorAtNorm(state.decalGizmo.nx, state.decalGizmo.ny), 'add');
      };
    }
    if ($('btnDecalUndo')) $('btnDecalUndo').onclick = () => applyDecalStampFromProjector(null, 'undo_last');
    if ($('btnDecalClear')) $('btnDecalClear').onclick = () => applyDecalStampFromProjector(null, 'clear');
    $('btnSpin3d').onclick = () => {
      gl3d.spin = !gl3d.spin;
      $('btnSpin3d').classList.toggle('active', gl3d.spin);
    };
    $('btnWire3d').onclick = () => {
      gl3d.wireframe = !gl3d.wireframe;
      $('btnWire3d').classList.toggle('active', gl3d.wireframe);
    };
    if ($('btnEnvBg3d')) {
      $('btnEnvBg3d').onclick = () => {
        gl3d.showEnvBg = !gl3d.showEnvBg;
        $('btnEnvBg3d').classList.toggle('active', gl3d.showEnvBg);
      };
    }
    if ($('btnPickPart')) $('btnPickPart').onclick = () => setPickMode(!state.pickMode);
    if ($('btnPickPart2d')) $('btnPickPart2d').onclick = () => setPickMode(!state.pickMode);
    if ($('btnBrush3d')) $('btnBrush3d').onclick = () => setBrushMode(!state.brushMode);
    if ($('btnBrush2d')) $('btnBrush2d').onclick = () => setBrushMode(!state.brushMode);
    if ($('btnLightDrag')) $('btnLightDrag').onclick = () => setLightDragMode(!state.lightDragMode);
    if ($('btnUvWire2d')) {
      $('btnUvWire2d').onclick = () => {
        state.uvWire2d = !state.uvWire2d;
        $('btnUvWire2d').classList.toggle('active', state.uvWire2d);
        render2DCanvas();
      };
    }
    if ($('bridgeBadge')) {
      $('bridgeBadge').onclick = () => {
        showToast('Blockbench 5 Live Link ready on http://127.0.0.1:18790/api/bridge/events — toggle in BB5 Tools menu!', 'ok');
      };
    }
    if ($('brushSizeSlider')) {
      $('brushSizeSlider').oninput = (e) => {
        state.brushRadius = parseInt(e.target.value, 10) || 6;
        if ($('brushSizeVal')) $('brushSizeVal').textContent = `${state.brushRadius}px`;
      };
    }
    if ($('btnBrushAdd')) {
      $('btnBrushAdd').onclick = () => {
        state.brushValue = 1.0;
        $('btnBrushAdd').classList.add('active');
        if ($('btnBrushErase')) $('btnBrushErase').classList.remove('active');
      };
    }
    if ($('btnBrushErase')) {
      $('btnBrushErase').onclick = () => {
        state.brushValue = 0.0;
        $('btnBrushErase').classList.add('active');
        if ($('btnBrushAdd')) $('btnBrushAdd').classList.remove('active');
      };
    }
    if ($('btnBrushClear')) {
      $('btnBrushClear').onclick = async () => {
        const rec = state.serverState && state.serverState.recipe;
        const sel = rec && (rec.nodes || []).find((n) => n.id === state.selectedNodeId);
        if (!sel || sel.op !== 'paint_mask') return;
        const st = await apiPost('/api/paint_mask/stroke', { node_id: sel.id, clear: true });
        invalidateAnimRing();
        applyServerState(st, { autoPrim: false });
        await runEvaluate({ includeThumbsMode: 'all', includeQa: false });
        showToast(`Cleared brush mask '${sel.id}'`, 'ok');
      };
    }
    // Milestone v16: 3D/2D Multi-Material PBR Paint Layer Stack Controls
    if ($('btnBrushModeMask')) $('btnBrushModeMask').onclick = () => setBrushTargetMode('mask');
    if ($('btnBrushModePbr')) $('btnBrushModePbr').onclick = () => setBrushTargetMode('pbr');
    if ($('pbrBrushPresetSelect')) {
      $('pbrBrushPresetSelect').onchange = (e) => applyPbrBrushPreset(e.target.value);
    }
    if ($('pbrBrushColor')) {
      $('pbrBrushColor').oninput = (e) => {
        state.pbrBrush.color = e.target.value;
        state.pbrBrush.erase = false;
      };
    }
    if ($('pbrBrushRough')) {
      $('pbrBrushRough').oninput = (e) => { state.pbrBrush.roughness = parseFloat(e.target.value); };
    }
    if ($('pbrBrushMetal')) {
      $('pbrBrushMetal').oninput = (e) => { state.pbrBrush.metallic = parseFloat(e.target.value); };
    }
    if ($('pbrBrushHeight')) {
      $('pbrBrushHeight').oninput = (e) => { state.pbrBrush.height = parseFloat(e.target.value); };
    }
    if ($('pbrBrushEmissive')) {
      $('pbrBrushEmissive').oninput = (e) => {
        const v = parseFloat(e.target.value);
        state.pbrBrush.emissive_strength = v;
        state.pbrBrush.paint_emissive = v > 0.01;
      };
    }
    if ($('btnPbrAddLayer')) {
      $('btnPbrAddLayer').onclick = async () => {
        const st = await runPbrLayerAction('add_layer', { name: `PBR Layer ${(state.pbrLayers || []).length + 1}` });
        if (st) showToast(`Added PBR paint layer (${state.activePbrLayerId})`, 'ok');
      };
    }
    if ($('btnPbrClearLayer')) {
      $('btnPbrClearLayer').onclick = async () => {
        const st = await runPbrLayerAction('clear_layer');
        if (st) showToast(`Cleared active PBR paint layer`, 'ok');
      };
    }
    if ($('btnSplitWipe')) {
      $('btnSplitWipe').onclick = () => {
        state.splitWipe = !state.splitWipe;
        $('btnSplitWipe').classList.toggle('active', state.splitWipe);
        if ($('splitWipeBar')) $('splitWipeBar').style.display = state.splitWipe ? 'block' : 'none';
        if ($('splitWipeHandle')) $('splitWipeHandle').style.display = state.splitWipe ? 'flex' : 'none';
        if (gl3d) {
          gl3d.splitWipe = state.splitWipe;
          gl3d.splitPos = state.splitPos;
          gl3d.splitRightChannel = state.splitRightChannel;
        }
        updateViewportTextures();
      };
    }
    if ($('splitRightSelect')) {
      $('splitRightSelect').onchange = (e) => {
        state.splitRightChannel = e.target.value;
        if (gl3d) gl3d.splitRightChannel = state.splitRightChannel;
        updateViewportTextures();
      };
    }
    if ($('splitWipeHandle')) {
      let wiping = false;
      $('splitWipeHandle').addEventListener('pointerdown', (e) => {
        e.stopPropagation();
        wiping = true;
        $('splitWipeHandle').setPointerCapture(e.pointerId);
      });
      window.addEventListener('pointermove', (e) => {
        if (!wiping) return;
        const sr = $('viewportStage').getBoundingClientRect();
        const nx = Math.max(0.05, Math.min(0.95, (e.clientX - sr.left) / Math.max(1, sr.width)));
        state.splitPos = nx;
        $('splitWipeHandle').style.left = `${(nx * 100).toFixed(1)}%`;
        if (gl3d) gl3d.splitPos = nx;
      });
      window.addEventListener('pointerup', () => { wiping = false; });
    }
    if ($('btnRecordGif')) {
      $('btnRecordGif').onclick = async () => {
        if (!gl3d) return;
        const btn = $('btnRecordGif');
        btn.disabled = true;
        btn.textContent = '⏳ GIF';
        try {
          const frames = gl3d.captureTurntableFrames(18);
          const stem = (state.serverState && state.serverState.recipe && state.serverState.recipe.name) || 'synccraft_3d';
          const res = await apiPost('/api/export/turntable_gif', { frames, fps: 14, stem });
          triggerDataUrlDownload(res.filename, res.gif_data_url);
          showToast(`🎬 Exported 360° Turntable GIF (${res.frame_count}f -> ${res.filename})`, 'ok');
        } catch (err) {
          showToast(`GIF error: ${err.message}`, 'err');
        } finally {
          btn.disabled = false;
          btn.textContent = '🎬 GIF';
        }
      };
    }
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
      if (gl3d) gl3d.lightingPreset = state.lightingPreset;
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
    if ($('envRotSlider')) {
      $('envRotSlider').oninput = (e) => {
        const v = parseFloat(e.target.value);
        state.envRotDeg = v;
        if ($('envRotVal')) $('envRotVal').textContent = `${Math.round(v)}°`;
        if (gl3d) gl3d.envRotDeg = v;
      };
    }
    if ($('lightIntSlider')) {
      $('lightIntSlider').oninput = (e) => {
        const v = parseFloat(e.target.value);
        state.lightIntensity = v;
        if ($('lightIntVal')) $('lightIntVal').textContent = v.toFixed(2);
        if (gl3d) gl3d.lightIntensity = v;
      };
    }
    if ($('coatSlider')) {
      $('coatSlider').oninput = (e) => {
        const v = parseFloat(e.target.value);
        if ($('coatVal')) $('coatVal').textContent = v.toFixed(2);
        if (gl3d) gl3d.clearcoat = v;
      };
    }
    if ($('bloomSlider')) {
      $('bloomSlider').oninput = (e) => {
        const v = parseFloat(e.target.value);
        if ($('bloomVal')) $('bloomVal').textContent = v.toFixed(2);
        if (gl3d) gl3d.bloomStrength = v;
      };
    }

    // 2D Canvas Direct Brush Painting
    let brush2dDown = false;
    const sample2dUv = (e) => {
      const dr = state._draw2dRect;
      if (!dr) return null;
      const r = $('canvas2d').getBoundingClientRect();
      const cx = (e.clientX - r.left) * dr.dpr;
      const cy = (e.clientY - r.top) * dr.dpr;
      const u = (((cx - dr.ox) / dr.drawW) * dr.tiles) % 1;
      const v = (((cy - dr.oy) / dr.drawH) * dr.tiles) % 1;
      if (u < 0 || v < 0 || u >= 1 || v >= 1) return null;
      return { u, v };
    };
    $('canvas2d').addEventListener('pointerdown', (e) => {
      if (!state.brushMode || e.button !== 0) return;
      const pt = sample2dUv(e);
      if (pt) { brush2dDown = true; queueBrushStroke(pt.u, pt.v, true); }
    });
    $('canvas2d').addEventListener('pointermove', (e) => {
      if (brush2dDown && state.brushMode) {
        const pt = sample2dUv(e);
        if (pt) queueBrushStroke(pt.u, pt.v, false);
      }
    });
    window.addEventListener('pointerup', () => { brush2dDown = false; });

    // 2D Pixel Eyedropper HUD + 2D UV Island Click-to-Mask!
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
    $('canvas2d').addEventListener('click', async (e) => {
      if (!state.pickMode && !e.altKey) return;
      const dr = state._draw2dRect;
      if (!dr) return;
      const r = $('canvas2d').getBoundingClientRect();
      const cx = (e.clientX - r.left) * dr.dpr;
      const cy = (e.clientY - r.top) * dr.dpr;
      const u = (((cx - dr.ox) / dr.drawW) * dr.tiles) % 1;
      const v = (((cy - dr.oy) / dr.drawH) * dr.tiles) % 1;
      if (u < 0 || v < 0 || u >= 1 || v >= 1) return;
      try {
        const res = await apiGet(`/api/mesh/pick_uv?u=${u.toFixed(5)}&v=${v.toFixed(5)}`);
        if (res.hit && res.short) {
          await handlePickedPart(res.short, e.shiftKey);
        } else {
          showToast(res.message || 'No UV island under cursor', 'err');
        }
      } catch (err) {
        showToast(err.message, 'err');
      }
    });

    // Node Library Tabs, Live Swatches, .syncmat & .syncpack Marketplace Drawer
    if ($('tabLibNodes')) $('tabLibNodes').onclick = () => switchLibTab('nodes');
    if ($('tabLibSyncmat')) $('tabLibSyncmat').onclick = () => switchLibTab('syncmat');
    if ($('tabLibSyncpack')) $('tabLibSyncpack').onclick = () => switchLibTab('syncpack');
    if ($('btnSaveSyncpack')) {
      $('btnSaveSyncpack').onclick = async () => {
        const ids = getActiveSelectionIds();
        const defName = (state.serverState && state.serverState.recipe && state.serverState.recipe.name) || 'Custom Pack';
        try {
          const res = await apiPost('/api/syncpack/save', {
            node_ids: ids.length ? ids : null,
            name: `${defName} Pack`,
            description: `Distributable .syncpack exported from ${defName}`,
          });
          if (res.packs) state.syncpacks = res.packs;
          switchLibTab('syncpack');
          showToast(`Saved .syncpack '${res.pack.name}'`, 'ok');
        } catch (err) {
          showToast(err.message, 'err');
        }
      };
    }
    if ($('btnUploadSyncpack')) $('btnUploadSyncpack').onclick = () => $('syncpackFileInput').click();
    if ($('syncpackFileInput')) {
      $('syncpackFileInput').onchange = async (e) => {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        try {
          const doc = JSON.parse(await f.text());
          const res = await apiPost('/api/syncpack/import', { pack: doc });
          if (res.packs) state.syncpacks = res.packs;
          switchLibTab('syncpack');
          showToast(`Imported .syncpack '${doc.name || f.name}'`, 'ok');
        } catch (err) {
          showToast(`Invalid .syncpack: ${err.message}`, 'err');
        }
        e.target.value = '';
      };
    }
    if ($('btnLiveSwatches')) {
      $('btnLiveSwatches').onclick = () => {
        state.liveSwatches = !state.liveSwatches;
        $('btnLiveSwatches').classList.toggle('active', state.liveSwatches);
        if (state.liveSwatches) startLiveSwatchLoop();
        else stopLiveSwatchLoop();
      };
    }
    if ($('btnSaveSyncmat')) {
      $('btnSaveSyncmat').onclick = async () => {
        const ids = getActiveSelectionIds();
        const defName = (state.serverState && state.serverState.recipe && state.serverState.recipe.name) || 'Custom Material';
        try {
          const res = await apiPost('/api/syncmat/save', {
            node_ids: ids.length ? ids : null,
            name: ids.length ? `${defName} (${ids.length}n)` : defName,
            category: 'User Presets',
            description: `Saved from ${defName} (${ids.length || 'all'} nodes)`,
          });
          state.syncmats = res.presets || state.syncmats;
          switchLibTab('syncmat');
          showToast(`Saved .syncmat preset '${res.preset.name}'`, 'ok');
        } catch (err) {
          showToast(err.message, 'err');
        }
      };
    }
    if ($('btnUploadSyncmat')) $('btnUploadSyncmat').onclick = () => $('syncmatFileInput').click();
    if ($('syncmatFileInput')) {
      $('syncmatFileInput').onchange = async (e) => {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        try {
          const doc = JSON.parse(await f.text());
          const cx = Math.round((-graphCanvas.panX + graphCanvas.stage.clientWidth * 0.45) / graphCanvas.zoom - 120);
          const cy = Math.round((-graphCanvas.panY + graphCanvas.stage.clientHeight * 0.42) / graphCanvas.zoom - 60);
          const st = await apiPost('/api/syncmat/insert', { preset: doc, pos: [cx, cy], wrap_frame: true });
          applyServerState(st, { autoPrim: false });
          await runEvaluate({ includeThumbs: true, includeQa: true });
          showToast(`Imported .syncmat '${doc.name || f.name}'`, 'ok');
        } catch (err) {
          showToast(`Invalid .syncmat: ${err.message}`, 'err');
        }
        e.target.value = '';
      };
    }

    // Graph Toolbar & Canvas Zoom HUD
    if ($('btnToolSelect')) {
      $('btnToolSelect').onclick = () => {
        graphCanvas.setToolMode('select');
        $('btnToolSelect').classList.add('active');
        if ($('btnToolPan')) $('btnToolPan').classList.remove('active');
      };
    }
    if ($('btnToolPan')) {
      $('btnToolPan').onclick = () => {
        graphCanvas.setToolMode('pan');
        $('btnToolPan').classList.add('active');
        if ($('btnToolSelect')) $('btnToolSelect').classList.remove('active');
      };
    }
    if ($('btnGroupSel')) $('btnGroupSel').onclick = () => groupSelectionIntoSubgraph();
    if ($('btnUngroupSel')) {
      $('btnUngroupSel').onclick = () => {
        if (state.selectedNodeId) mutate({ action: 'ungroup_node', node_id: state.selectedNodeId }, { immediateEval: true });
      };
    }
    if ($('btnFrameSel')) $('btnFrameSel').onclick = () => createFrameFromSelection();
    $('btnAddNode').onclick = () => openPalette();
    $('btnOpenPaletteLib').onclick = () => openPalette();
    if ($('btnLibList')) $('btnLibList').onclick = () => { state.libViewMode = 'list'; renderNodeLibrary(); };
    if ($('btnLibGrid')) $('btnLibGrid').onclick = () => { state.libViewMode = 'grid'; renderNodeLibrary(); };
    if ($('btnGraphZoomIn')) $('btnGraphZoomIn').onclick = () => graphCanvas.zoomBy(1.22);
    if ($('btnGraphZoomOut')) $('btnGraphZoomOut').onclick = () => graphCanvas.zoomBy(0.82);
    if ($('btnGraphZoomFit')) $('btnGraphZoomFit').onclick = () => graphCanvas.fitToView();
    $('btnAutoLayout').onclick = () => mutate({ action: 'auto_layout' }, { skipEval: true, fitGraph: true });
    if ($('btnDupSel')) $('btnDupSel').onclick = () => { const ids = getActiveSelectionIds(); if (ids.length) mutate({ action: 'duplicate_nodes', ids }, { immediateEval: true }); };
    $('btnFitGraph').onclick = () => graphCanvas.fitToView();
    $('btnBypassSel').onclick = () => {
      const ids = getActiveSelectionIds();
      if (ids.length > 1) mutate({ action: 'toggle_bypass_multi', ids }, { immediateEval: true });
      else if (ids.length === 1) mutate({ action: 'toggle_bypass', node_id: ids[0] }, { immediateEval: true });
    };
    $('btnDeleteSel').onclick = () => { const ids = getActiveSelectionIds(); if (ids.length) mutate({ action: 'delete_nodes', ids }, { immediateEval: true }); };
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
        state.animFps = Math.max(4, Math.min(60, parseInt(e.target.value, 10) || 30));
      };
    }
    $('animFramesInput').onchange = (e) => {
      state.animFrames = Math.max(2, Math.min(32, parseInt(e.target.value, 10) || 12));
      invalidateAnimRing();
      setPhase(state.phase);
    };

    // Milestone v16: Curve / Spline Graph Editor Controls
    setupCurveEditorEvents();
    if ($('curveDriverSelect')) {
      $('curveDriverSelect').onchange = (e) => {
        state.curveDriverIdx = parseInt(e.target.value, 10);
        state._curveModeOverride = null;
        renderCurveEditor();
      };
    }
    if ($('curveInterpSelect')) {
      $('curveInterpSelect').onchange = (e) => {
        const mode = e.target.value;
        state._curveModeOverride = mode;
        const drivers = (state.serverState && state.serverState.recipe && state.serverState.recipe.drivers) || [];
        if (state.curveDriverIdx >= 0 && state.curveDriverIdx < drivers.length) {
          const kfs = sampleDriverToKeyframes(drivers[state.curveDriverIdx]);
          commitCurveKeyframes(kfs, mode, true);
        }
        renderCurveEditor();
      };
    }
    if ($('btnCurveAddKf')) {
      $('btnCurveAddKf').onclick = () => {
        const drivers = (state.serverState && state.serverState.recipe && state.serverState.recipe.drivers) || [];
        if (state.curveDriverIdx < 0 || state.curveDriverIdx >= drivers.length) {
          showToast('No @animate track selected. Click ~ in Inspector first!', 'err');
          return;
        }
        const d = drivers[state.curveDriverIdx];
        const mode = $('curveInterpSelect') ? $('curveInterpSelect').value : 'spline';
        const kfs = sampleDriverToKeyframes(d);
        const ph = Number(state.phase.toFixed(3));
        const val = Number(evalCurveAtPhaseJS(kfs, ph, mode).toFixed(3));
        const existsIdx = kfs.findIndex((p) => Math.abs(p[0] - ph) < 0.02);
        if (existsIdx >= 0) {
          const midPh = Number(((ph + 0.15) % 1.0).toFixed(3));
          kfs.push([midPh, Number(evalCurveAtPhaseJS(kfs, midPh, mode).toFixed(3)), mode === 'step' ? 'step' : 'smooth']);
        } else {
          kfs.push([ph, val, mode === 'step' ? 'step' : 'smooth']);
        }
        commitCurveKeyframes(kfs, mode, true);
        showToast(`Added keyframe on ${d.node_id}.${d.param_key}`, 'ok');
      };
    }

    // Milestone v16: Multi-Asset Batch Bakery Queue Modal
    const stopBatchPoll = () => {
      if (state.batchPollTimer) { clearInterval(state.batchPollTimer); state.batchPollTimer = null; }
    };
    const renderBatchQueue = (items, summaryText) => {
      if ($('batchSummaryLine') && summaryText) $('batchSummaryLine').textContent = summaryText;
      const list = $('batchQueueList');
      if (!list) return;
      list.innerHTML = '';
      if (!items || !items.length) {
        list.innerHTML = '<div style="font-size:11px;color:var(--text-muted);padding:8px;">No assets scanned yet. Click Scan Folder or Run Batch Bakery.</div>';
        return;
      }
      for (const it of items) {
        const row = document.createElement('div');
        row.className = 'batch-item-row';
        const stName = it.status || 'pending';
        const metaParts = [];
        if (it.kind) metaParts.push(it.kind);
        else {
          if (it.bbmodel_path || it.bbmodel) metaParts.push('3D .bbmodel');
          if (it.texdsl_path || it.texdsl) metaParts.push('.texdsl');
        }
        if (it.qa_score != null) metaParts.push(`QA ${Math.round(it.qa_score)}/100`);
        const msVal = it.ms != null ? it.ms : it.elapsed_ms;
        if (msVal != null && msVal > 0) metaParts.push(`${Math.round(msVal)}ms`);
        if (it.files_count != null && it.files_count > 0) metaParts.push(`${it.files_count} files`);
        if (it.mesh_cache_hit) metaParts.push('cache hit');
        if (it.error) metaParts.push(`ERR: ${it.error}`);
        row.innerHTML = `
          <div class="batch-item-main">
            <span class="batch-item-stem">${it.stem || 'asset'}</span>
            <span class="batch-item-meta">${metaParts.join(' · ')}</span>
          </div>
          <span class="batch-status-pill ${stName}">${stName.toUpperCase()}</span>
        `;
        list.appendChild(row);
      }
    };
    const pollBatchStatus = async () => {
      try {
        const st = await apiGet('/api/batch/status');
        const tot = st.total || 0;
        const done = st.completed || 0;
        const meanQa = st.report && st.report.mean_qa != null ? ` · Mean QA ${Math.round(st.report.mean_qa)}/100` : '';
        const sum = st.running
          ? `Baking ${done}/${tot} assets -> ${st.out_dir || ''}`
          : (st.error ? `Batch error: ${st.error}` : `Done: ${st.succeeded || 0}/${tot} succeeded${meanQa} -> ${st.out_dir || ''}`);
        renderBatchQueue(st.items || [], sum);
        if (!st.running) {
          stopBatchPoll();
          if ($('btnBatchRun')) $('btnBatchRun').disabled = false;
          if (!st.error && tot > 0) showToast(`Batch Bakery complete: ${st.succeeded}/${tot} PBR v2.0 + .mtlx bundles!`, 'ok');
        }
      } catch (_) {}
    };
    if ($('btnOpenBatch')) {
      $('btnOpenBatch').onclick = () => {
        $('batchModal').classList.add('open');
        pollBatchStatus();
      };
    }
    if ($('btnCloseBatch')) {
      $('btnCloseBatch').onclick = () => { stopBatchPoll(); $('batchModal').classList.remove('open'); };
    }
    if ($('batchModal')) {
      $('batchModal').addEventListener('click', (e) => {
        if (e.target === $('batchModal')) { stopBatchPoll(); $('batchModal').classList.remove('open'); }
      });
    }
    if ($('btnBatchScan')) {
      $('btnBatchScan').onclick = async () => {
        try {
          const res = await apiPost('/api/batch/scan', {
            input_dir: ($('batchInputDir') && $('batchInputDir').value.trim()) || '',
            max_assets: 16,
          });
          if ($('batchInputDir') && !$('batchInputDir').value.trim() && res.input_dir) {
            $('batchInputDir').value = res.input_dir;
          }
          const items = (res.assets || []).map((a) => Object.assign({}, a, { status: 'pending' }));
          renderBatchQueue(items, `Scanned ${res.count} asset(s) in ${res.input_dir}`);
          showToast(`Found ${res.count} bakeable asset(s)`, 'ok');
        } catch (err) {
          showToast(err.message, 'err');
        }
      };
    }
    if ($('btnBatchRun')) {
      $('btnBatchRun').onclick = async () => {
        try {
          $('btnBatchRun').disabled = true;
          const res = await apiPost('/api/batch/start', {
            input_dir: ($('batchInputDir') && $('batchInputDir').value.trim()) || '',
            out_dir: ($('batchOutDir') && $('batchOutDir').value.trim()) || '',
            preset: ($('batchPresetSelect') && $('batchPresetSelect').value) || 'studio_hero',
            max_assets: 12,
          });
          if (res.status) {
            renderBatchQueue(res.status.items || [], `Started Batch Bakery -> ${res.status.out_dir || ''}`);
          }
          stopBatchPoll();
          state.batchPollTimer = setInterval(pollBatchStatus, 350);
          pollBatchStatus();
        } catch (err) {
          $('btnBatchRun').disabled = false;
          showToast(err.message, 'err');
        }
      };
    }

    // Milestone v18: Batch Bakery, Multi-Asset Shared Atlas & Auto-LOD Exporter Mode Switcher
    const setBatchModalMode = (mode) => {
      state.batchModalMode = (mode === 'atlas' || mode === 'lod') ? mode : 'pbr';
      const isPbr = state.batchModalMode === 'pbr';
      const isAtlas = state.batchModalMode === 'atlas';
      const isLod = state.batchModalMode === 'lod';
      if ($('btnBatchModePbr')) $('btnBatchModePbr').classList.toggle('active', isPbr);
      if ($('btnBatchModeAtlas')) $('btnBatchModeAtlas').classList.toggle('active', isAtlas);
      if ($('btnBatchModeLod')) $('btnBatchModeLod').classList.toggle('active', isLod);
      const inputDirRow = $('batchInputDir') && $('batchInputDir').closest('.param-row');
      if (inputDirRow) inputDirRow.style.display = isLod ? 'none' : '';
      if ($('batchPbrOptions')) $('batchPbrOptions').style.display = isPbr ? 'grid' : 'none';
      if ($('atlasBakeOptions')) $('atlasBakeOptions').style.display = isAtlas ? 'grid' : 'none';
      if ($('lodBakeOptions')) $('lodBakeOptions').style.display = isLod ? 'grid' : 'none';
      if ($('btnBatchScan')) $('btnBatchScan').style.display = isLod ? 'none' : '';
      if ($('btnBatchRun')) $('btnBatchRun').style.display = isPbr ? '' : 'none';
      if ($('btnAtlasRun')) $('btnAtlasRun').style.display = isAtlas ? '' : 'none';
      if ($('btnLodGlbExportRun')) $('btnLodGlbExportRun').style.display = isLod ? '' : 'none';
      if ($('btnLodExportRun')) $('btnLodExportRun').style.display = isLod ? '' : 'none';
      if ($('atlasResultCard') && !isAtlas) $('atlasResultCard').style.display = 'none';
      if ($('batchModeDesc')) {
        if (isLod) {
          $('batchModeDesc').innerHTML = 'Generate silhouette- &amp; skeleton-preserving cuboid <code>LOD0 / LOD1 / LOD2</code> tiers as <code>.bbmodel</code> (+ mip-dilated textures) OR rigged Khronos <code>glTF 2.0 (.glb)</code> ready for UE5, Unity, Godot &amp; Three.js.';
        } else if (isAtlas) {
          $('batchModeDesc').innerHTML = 'Pack multiple <code>.bbmodel</code> / <code>.texdsl</code> assets into one shared Power-of-Two or tight PBR atlas with mip-safe Voronoi edge bleed and remapped <code>*_atlas.bbmodel</code> UVs.';
        } else {
          $('batchModeDesc').innerHTML = 'Scan any folder of <code>.bbmodel</code> / <code>.texdsl</code> assets, auto-pair graphs, bake 3D ray-traced maps with SHA-256 caching, run QA audits, and export full <code>PBR v2.0 + MaterialX</code> bundles.';
        }
      }
    };
    if ($('btnBatchModePbr')) $('btnBatchModePbr').onclick = () => setBatchModalMode('pbr');
    if ($('btnBatchModeAtlas')) $('btnBatchModeAtlas').onclick = () => setBatchModalMode('atlas');
    if ($('btnBatchModeLod')) $('btnBatchModeLod').onclick = () => setBatchModalMode('lod');
    if ($('btnAtlasRun')) {
      $('btnAtlasRun').onclick = async () => {
        const btn = $('btnAtlasRun');
        btn.disabled = true;
        btn.textContent = 'Packing Atlas...';
        try {
          const res = await apiPost('/api/atlas/bake', {
            input_dir: ($('batchInputDir') && $('batchInputDir').value.trim()) || '',
            out_dir: ($('batchOutDir') && $('batchOutDir').value.trim()) || '',
            stem: ($('atlasStemInput') && $('atlasStemInput').value.trim()) || 'synccraft_atlas',
            canvas_mode: ($('atlasCanvasMode') && $('atlasCanvasMode').value) || 'pot',
            bleed_px: parseInt(($('atlasBleedSelect') && $('atlasBleedSelect').value) || '4', 10),
            max_assets: 12,
          });
          const fillPct = ((res.fill_ratio || 0) * 100).toFixed(1);
          const sum = `Packed ${res.asset_count} asset(s) into ${res.atlas_width}×${res.atlas_height} (${res.canvas_mode.toUpperCase()}, ${fillPct}% fill, ${res.bleed_px}px bleed) in ${Math.round(res.elapsed_ms || 0)}ms`;
          const rows = (res.entries || []).map((ent) => ({
            stem: ent.stem,
            kind: `Slot (${ent.x}, ${ent.y}) · ${ent.w}×${ent.h}`,
            status: 'done',
          }));
          renderBatchQueue(rows, sum);
          if ($('atlasResultCard')) {
            $('atlasResultCard').style.display = 'flex';
            if ($('atlasPreviewImg') && res.albedo_preview_url) $('atlasPreviewImg').src = res.albedo_preview_url;
            if ($('atlasResultMeta')) {
              $('atlasResultMeta').innerHTML = `
                <strong style="color:var(--text-primary);">${res.stem} · ${res.atlas_width}×${res.atlas_height} (${res.canvas_mode.toUpperCase()})</strong>
                <span>Fill efficiency: ${fillPct}% · Voronoi bleed: ${res.bleed_px}px · ${(res.remapped_bbmodels || []).length} remapped .bbmodel(s)</span>
                <span style="font-family:var(--font-mono);font-size:9.5px;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${res.manifest_path || ''}</span>
              `;
            }
          }
          showToast(`Packed Shared PBR Atlas (${res.atlas_width}×${res.atlas_height}, ${fillPct}% fill)`, 'ok');
        } catch (err) {
          showToast(`Atlas bake error: ${err.message}`, 'err');
        } finally {
          btn.disabled = false;
          btn.textContent = 'Pack Shared Atlas';
        }
      };
    }
    if ($('btnLodExportRun')) {
      $('btnLodExportRun').onclick = async () => {
        const btn = $('btnLodExportRun');
        btn.disabled = true;
        btn.textContent = 'Exporting LODs...';
        try {
          const res = await apiPost('/api/lod/export', {
            stem: ($('lodStemInput') && $('lodStemInput').value.trim()) || 'synccraft_lod',
            out_dir: ($('batchOutDir') && $('batchOutDir').value.trim()) || '',
            lod1_ratio: parseFloat(($('lod1RatioInput') && $('lod1RatioInput').value) || '0.58'),
            lod2_ratio: parseFloat(($('lod2RatioInput') && $('lod2RatioInput').value) || '0.35'),
            bleed_px: parseInt(($('lodBleedSelect') && $('lodBleedSelect').value) || '4', 10),
          });
          const lvlsRaw = Array.isArray(res.levels)
            ? res.levels
            : ['lod0', 'lod1', 'lod2'].filter((k) => res.levels && res.levels[k]).map((k) => Object.assign({ level: k }, res.levels[k]));
          const rows = lvlsRaw.map((info) => {
            const k = String(info.level || 'lod').toUpperCase();
            const texSz = info.texture_size ? ` · ${info.texture_size[0]}×${info.texture_size[1]}` : '';
            const red = info.reduction_pct > 0 ? ` (-${info.reduction_pct}%)` : ' (100%)';
            return {
              stem: `${k} · ${info.box_count} boxes · ${info.tri_count} tris${red}`,
              kind: `${info.bone_count} bones${texSz}`,
              status: 'done',
            };
          });
          const cleanDir = String(res.out_dir || '').replace(/^\/Users\/[^/]+\//, '~/').replace(/^\/home\/[^/]+\//, '~/');
          const sum = `Exported LOD0/LOD1/LOD2 bundle '${res.stem}' (bleed ${res.bleed_px}px) -> ${cleanDir}`;
          renderBatchQueue(rows, sum);
          showToast(`Exported Auto-LOD bundle '${res.stem}' (LOD0–LOD2)`, 'ok');
        } catch (err) {
          showToast(`Auto-LOD export error: ${err.message}`, 'err');
        } finally {
          btn.disabled = false;
          btn.textContent = 'Export LOD0–LOD2 .bbmodel';
        }
      };
    }
    if ($('btnLodGlbExportRun')) {
      $('btnLodGlbExportRun').onclick = async () => {
        const btn = $('btnLodGlbExportRun');
        btn.disabled = true;
        btn.textContent = 'Exporting .glb...';
        try {
          const coat = $('coatSlider') ? parseFloat($('coatSlider').value) : 0.25;
          const res = await apiPost('/api/export/glb', {
            export_lods: true,
            stem: ($('lodStemInput') && $('lodStemInput').value.trim()) || 'synccraft_lod',
            out_dir: ($('batchOutDir') && $('batchOutDir').value.trim()) || '',
            lod1_ratio: parseFloat(($('lod1RatioInput') && $('lod1RatioInput').value) || '0.58'),
            lod2_ratio: parseFloat(($('lod2RatioInput') && $('lod2RatioInput').value) || '0.35'),
            bleed_px: parseInt(($('lodBleedSelect') && $('lodBleedSelect').value) || '4', 10),
            coat_weight: coat,
          });
          const rows = (res.levels || []).map((info) => {
            const k = String(info.level || 'lod').toUpperCase();
            const red = info.reduction_pct > 0 ? ` (-${info.reduction_pct}%)` : ' (100%)';
            return {
              stem: `${k}.glb · ${info.tri_count} tris · ${info.kb} KB${red}`,
              kind: `${info.joint_count} joints · ${info.animation_count} clips · glTF 2.0 VALID`,
              status: info.valid_gltf ? 'done' : 'err',
            };
          });
          const cleanDir = String(res.out_dir || '').replace(/^\/Users\/[^/]+\//, '~/').replace(/^\/home\/[^/]+\//, '~/');
          const sum = `Exported rigged glTF 2.0 LOD0–LOD2 (.glb) bundle '${res.stem}' (${Math.round(res.elapsed_ms || 0)}ms) -> ${cleanDir}`;
          renderBatchQueue(rows, sum);
          showToast(`Exported rigged glTF 2.0 LOD0–LOD2 (.glb) bundle '${res.stem}'`, 'ok');
        } catch (err) {
          showToast(`glTF LOD export error: ${err.message}`, 'err');
        } finally {
          btn.disabled = false;
          btn.textContent = 'Export LOD0–LOD2 .glb';
        }
      };
    }

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
        resetWorkspaceVisuals({ fullProjectReset: false });
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
        resetWorkspaceVisuals({ fullProjectReset: false });
        const doc = JSON.parse(await f.text());
        const st = await apiPost('/api/mesh/load', { bbmodel: doc });
        applyServerState(st, { autoPrim: true });
        await runEvaluate({ includeThumbs: true, includeQa: true });
        showToast(`Baked uploaded .bbmodel: ${f.name}`, 'ok');
      } catch (err) {
        showToast(`Invalid .bbmodel: ${err.message}`, 'err');
      }
      e.target.value = '';
    };

    // Substance-Style Multi-Engine Output Template Export Modal (`v17`)
    $('btnOpenExport').onclick = () => {
      $('exportStemInput').value = (state.serverState && state.serverState.recipe && state.serverState.recipe.name) || 'synccraft_asset';
      if ($('exportPresetSelect') && [...$('exportPresetSelect').options].some((o) => o.value === state.lightingPreset)) {
        $('exportPresetSelect').value = state.lightingPreset;
      }
      $('exportResultPre').style.display = 'none';
      if ($('exportTemplateSelect') && !$('exportTemplateSelect').options.length) populateExportTemplates();
      renderExportTemplatePreview();
      $('exportModal').classList.add('open');
    };
    if ($('exportTemplateSelect')) $('exportTemplateSelect').onchange = () => renderExportTemplatePreview();
    if ($('exportStemInput')) $('exportStemInput').oninput = () => renderExportTemplatePreview();
    $('btnCloseExport').onclick = () => $('exportModal').classList.remove('open');
    $('exportModal').addEventListener('click', (e) => { if (e.target === $('exportModal')) $('exportModal').classList.remove('open'); });
    const runCtmExport = async (mode) => {
      const stem = $('exportStemInput').value.trim() || 'synccraft_block';
      const outDir = $('exportDirInput').value.trim();
      try {
        const res = await apiPost('/api/export/ctm', { mode, stem, out_dir: outDir || null, style: 'metal_frame' });
        const pre = $('exportResultPre');
        if (pre) {
          pre.style.display = 'block';
          pre.textContent = `Generated ${res.mode.toUpperCase()} Connected Tileset (${res.tile_count} tiles, ${res.grid_cols}×${res.grid_rows} sheet)\n\n${res.optifine_properties}`;
        }
        triggerDataUrlDownload(res.sheet_filename, res.sheet_data_url);
        triggerDownload(res.props_filename, res.optifine_properties);
        showToast(`Exported ${res.mode.toUpperCase()} (${res.tile_count} tiles + .properties)`, 'ok');
      } catch (err) {
        showToast(err.message, 'err');
      }
    };
    const runGlbExport = async () => {
      const stem = $('exportStemInput').value.trim() || 'synccraft_asset';
      const outDir = $('exportDirInput').value.trim();
      const preset = $('exportPresetSelect').value;
      const bleedPx = parseInt(($('exportBleedSelect') && $('exportBleedSelect').value) || '4', 10);
      const coat = $('coatSlider') ? parseFloat($('coatSlider').value) : 0.25;
      try {
        const res = await apiPost('/api/export/glb', {
          stem,
          out_dir: outDir || null,
          lighting_preset: preset,
          bleed_px: bleedPx,
          coat_weight: coat,
          include_animations: true,
        });
        const rep = res.report || {};
        const pre = $('exportResultPre');
        const valTag = (rep.validation && rep.validation.valid) ? 'Khronos glTF 2.0 VALID (0 errors)' : 'Validation warnings';
        const exts = (rep.extensions_used || []).length ? `\nExtensions: ${rep.extensions_used.join(', ')}` : '';
        const clips = (rep.animation_names || []).length ? `\nClips (${rep.animation_count}): ${rep.animation_names.join(', ')}` : '';
        const cleanDest = res.mode === 'disk'
          ? String(res.out_path || '').replace(/^\/Users\/[^/]+\//, '~/').replace(/^\/home\/[^/]+\//, '~/')
          : `Downloading ${res.filename}...`;
        if (pre) {
          pre.style.display = 'block';
          pre.textContent = `${valTag} · ${res.filename} (${rep.kb} KB)\nGeometry: ${rep.triangle_count} tris · ${rep.vertex_count} verts · ${rep.joint_count} joints · ${rep.image_count} embedded PNGs${clips}${exts}\n${cleanDest}`;
        }
        if (res.mode !== 'disk' && res.glb_data_url) {
          triggerDataUrlDownload(res.filename || `${stem}.glb`, res.glb_data_url);
        }
        showToast(`Exported ${res.filename} (${rep.kb} KB, ${rep.joint_count} joints, ${rep.animation_count} clips)`, 'ok');
      } catch (err) {
        showToast(`glTF export error: ${err.message}`, 'err');
      }
    };
    if ($('btnDownloadGlb')) {
      $('btnDownloadGlb').onclick = () => runGlbExport();
    }
    if ($('btnDownloadMtlx')) {
      $('btnDownloadMtlx').onclick = async () => {
        const stem = $('exportStemInput').value.trim() || 'asset';
        const shader = ($('exportMtlxShader') && $('exportMtlxShader').value) || 'open_pbr_surface';
        const coat = $('coatSlider') ? parseFloat($('coatSlider').value) : 0.25;
        const ver = shader === 'open_pbr_surface' ? '1.39' : '1.38';
        try {
          const res = await apiPost('/api/export/materialx', { stem, shader_model: shader, coat_weight: coat });
          triggerDownload(res.filename || `${stem}.mtlx`, res.xml);
          const pre = $('exportResultPre');
          if (pre) {
            pre.style.display = 'block';
            pre.textContent = `Downloaded ASWF MaterialX ${ver} (${shader}): ${res.filename}\n\n${res.xml.slice(0, 620)}...`;
          }
          showToast(`Downloaded MaterialX ${ver} (${res.filename})`, 'ok');
        } catch (err) {
          showToast(err.message, 'err');
        }
      };
    }
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
      const tplId = ($('exportTemplateSelect') && $('exportTemplateSelect').value) || 'full_pbr_v2';
      if (tplId === 'glb_skinned') {
        await runGlbExport();
        return;
      }
      if (tplId === 'ctm47' || tplId === 'wang16') {
        await runCtmExport(tplId);
        return;
      }
      const stem = $('exportStemInput').value.trim() || 'asset';
      const outDir = $('exportDirInput').value.trim();
      const preset = $('exportPresetSelect').value;
      const bleedPx = parseInt(($('exportBleedSelect') && $('exportBleedSelect').value) || '4', 10);
      const shader = ($('exportMtlxShader') && $('exportMtlxShader').value) || 'open_pbr_surface';
      const coat = $('coatSlider') ? parseFloat($('coatSlider').value) : 0.25;
      try {
        const res = await apiPost('/api/export/preset', {
          preset_id: tplId,
          stem,
          out_dir: outDir || null,
          lighting_preset: preset,
          bleed_px: bleedPx,
          shader_model: shader,
          coat_weight: coat,
        });
        const pre = $('exportResultPre');
        pre.style.display = 'block';
        const fnames = res.file_names || [];
        if (res.mode === 'disk') {
          pre.textContent = `Exported '${tplId}' (${fnames.length} files, ${res.bleed_px}px mip bleed) to:\n${res.out_dir}\n\n${fnames.join(', ')}`;
          showToast(`Exported ${tplId} (${fnames.length} files) to disk`, 'ok');
        } else {
          pre.textContent = `Prepared '${tplId}' (${fnames.length} files, ${res.bleed_px}px mip bleed):\n${fnames.join(', ')}\n\nDownloading ${res.bundle_filename}...`;
          if (res.bundle_filename && res.bundle_json) {
            triggerDownload(res.bundle_filename, res.bundle_json);
          }
          showToast(`Exported ${res.bundle_filename} (${fnames.length} maps)`, 'ok');
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
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'a') {
        if (inField) return;
        e.preventDefault();
        const vis = (graphCanvas && graphCanvas._visibleNodes) || ((state.serverState && state.serverState.recipe && state.serverState.recipe.nodes) || []);
        const allIds = vis.map((n) => n.id);
        state.selectedIds = new Set(allIds);
        if (allIds.length) state.selectedNodeId = allIds[allIds.length - 1];
        syncSelectionUI();
        renderInspector();
        graphCanvas.setData(state.serverState.recipe, state.evalData, state.selectedNodeId, state.activePass, state.selectedIds);
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'g') {
        if (inField) return;
        e.preventDefault();
        if (e.altKey && state.selectedNodeId) {
          mutate({ action: 'ungroup_node', node_id: state.selectedNodeId }, { immediateEval: true });
        } else {
          groupSelectionIntoSubgraph();
        }
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') {
        if (inField) return;
        e.preventDefault();
        createFrameFromSelection();
        return;
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'd') {
        const ids = getActiveSelectionIds();
        if (inField || !ids.length) return;
        e.preventDefault();
        mutate({ action: 'duplicate_nodes', ids }, { immediateEval: true });
        return;
      }
      if (e.key === 'Escape') {
        hideContextMenu();
        closePalette();
        closeViewportPopovers();
        if (state.pickMode || state.brushMode || state.decalMode || state.lightDragMode) setViewportTool('orbit');
        $('exportModal').classList.remove('open');
        $('meshModal').classList.remove('open');
        if ($('batchModal')) $('batchModal').classList.remove('open');
        return;
      }
      if (inField) return;
      if (e.key === 'Enter' && state.decalMode && state.vpMode === '3d' && gl3d) {
        e.preventDefault();
        applyDecalStampFromProjector(gl3d.getCameraProjectorAtNorm(state.decalGizmo.nx, state.decalGizmo.ny), 'add');
        return;
      }
      if (e.key === 'Tab' || (e.shiftKey && e.key.toLowerCase() === 'a')) { e.preventDefault(); openPalette(); }
      else if (e.key === ' ' && !e.repeat) { e.preventDefault(); togglePlayAnim(); }
      else if (e.key.toLowerCase() === 'c' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); createFrameFromSelection(); }
      else if (e.key.toLowerCase() === 'f') { e.preventDefault(); graphCanvas.fitToView(); }
      else if (e.key.toLowerCase() === 'l') { e.preventDefault(); mutate({ action: 'auto_layout' }, { skipEval: true, fitGraph: true }); }
      else if (e.key.toLowerCase() === 'b' || e.key.toLowerCase() === 'm') {
        const ids = getActiveSelectionIds();
        if (!ids.length) return;
        e.preventDefault();
        if (ids.length > 1) mutate({ action: 'toggle_bypass_multi', ids }, { immediateEval: true });
        else mutate({ action: 'toggle_bypass', node_id: ids[0] }, { immediateEval: true });
      }
      else if (e.key === 'Delete' || e.key === 'Backspace') {
        const ids = getActiveSelectionIds();
        if (!ids.length) return;
        e.preventDefault();
        mutate({ action: 'delete_nodes', ids }, { immediateEval: true });
      }
    });
  }

  boot().catch((err) => {
    console.error('Studio boot failed:', err);
    showToast(`Boot error: ${err.message}`, 'err');
  });
})();
