/* ==========================================================================
   SyncCraft Studio v14 — Commercial Infinite Node Graph Canvas & Minimap
   Features: Multi-Select, Box-Select Marquee, Group Drag, Custom Comment
   Frames (Ctrl+J / C), Compact Reroute Dots (Double-Click Wire), Select/Pan
   Tools, Space/Middle/Right-Drag Pan, Alt+Click Sever, Zero-Flicker Live
   Animated Card Thumbnail Patching, Right-Click Context Menus & Minimap.
   ========================================================================== */
(function (global) {
  'use strict';

  const CAT_COLORS = {
    'Gradients & Paths': '#818cf8',
    'Noise': '#38bdf8',
    'Noise & Procedural': '#38bdf8',
    'Geometric Patterns': '#34d399',
    'Patterns & Biology': '#34d399',
    'Painterly & Bio': '#2dd4bf',
    'Region & Per-UV Pixel Composer': '#c084fc',
    '3D Model & UV Masks': '#14b8a6',
    '3D Mesh Baker': '#06b6d4',
    'Materials — Metal (PBR)': '#eab308',
    'Materials — Dielectric & Glass (PBR)': '#38bdf8',
    'Pixel Composer FX': '#fb923c',
    'Material Bus (PBR)': '#f59e0b',
    'Character & Gear (PBR)': '#a78bfa',
    'Character & Gear PBR': '#a78bfa',
    'World Blocks (PBR)': '#4ade80',
    'Bio & Creature PBR': '#10b981',
    'Seamless & Precomputed PBR': '#22d3ee',
    'PBR & Normal': '#60a5fa',
    'Color & Grading': '#f472b6',
    'Denoise & Polish': '#a3e635',
    'Transform & Distort': '#fbbf24',
    'Filter & SDF': '#67e8f9',
    'Simulation': '#fb7185',
    'Animation': '#f43f5e',
    '3D Mesh & Masks': '#94a3b8',
    'Compositing': '#94a3b8',
    'Compositing & Math': '#94a3b8',
    'Sprites & Masks': '#e879f9',
    'Scatter & Layout': '#f97316',
    'Substance & Geometry': '#2dd4bf',
    'Math & Utility': '#a855f7',
  };

  const FRAME_PALETTE = ['#2563EB', '#0F766E', '#B45309', '#7E22CE', '#BE123C', '#475569'];

  function socketShape(portType) {
    if (portType === 'MATERIAL') return 'shape-square';
    if (portType === 'MASK' || portType === 'VECTOR3') return 'shape-diamond';
    return 'shape-circle';
  }

  function bezierPath(x1, y1, x2, y2) {
    const dx = Math.max(42, Math.abs(x2 - x1) * 0.46);
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }

  class StudioGraphCanvas {
    constructor(opts) {
      this.stage = opts.stage;
      this.gridBg = opts.gridBg;
      this.world = opts.world;
      this.passLayer = opts.passLayer;
      this.wiresSvg = opts.wiresSvg;
      this.nodesLayer = opts.nodesLayer;
      this.minimapCanvas = opts.minimapCanvas;
      this.zoomBadge = opts.zoomBadge;
      this.callbacks = opts.callbacks;

      this.panX = 48;
      this.panY = 48;
      this.zoom = 1.0;
      this.toolMode = 'select'; // 'select' (marquee left-drag) | 'pan' (hand left-drag)

      this.schema = null;
      this.opMap = {};
      this.recipe = null;
      this.evalData = null;
      this.selectedNodeId = null;
      this.selectedIds = new Set();
      this.activePass = 'ALL';

      this._dragNodes = null;
      this._panning = false;
      this._rightPanPending = null;
      this._marquee = null;
      this._spaceDown = false;
      this._wireDrag = null;
      this._socketEls = new Map();
      this._nodeEls = new Map();
      this._thumbImgs = new Map();

      // Marquee selection rectangle element
      this._marqueeEl = document.createElement('div');
      this._marqueeEl.className = 'graph-marquee';
      this._marqueeEl.style.display = 'none';
      this.stage.appendChild(this._marqueeEl);

      this._bindEvents();
    }

    setSchema(schema) {
      this.schema = schema;
      this.opMap = {};
      for (const [cat, nodes] of Object.entries(schema.nodes_by_category || {})) {
        for (const n of nodes) {
          this.opMap[n.op] = Object.assign({ category: cat }, n);
        }
      }
    }

    setToolMode(mode) {
      this.toolMode = mode === 'pan' ? 'pan' : 'select';
      this.stage.style.cursor = this.toolMode === 'pan' ? 'grab' : 'default';
    }

    setData(recipe, evalData, selectedNodeId, activePass, selectedIds) {
      this.recipe = recipe;
      this.evalData = evalData;
      this.selectedNodeId = selectedNodeId;
      this.activePass = activePass || 'ALL';
      if (selectedIds instanceof Set) {
        this.selectedIds = new Set(selectedIds);
      } else if (Array.isArray(selectedIds)) {
        this.selectedIds = new Set(selectedIds);
      } else {
        this.selectedIds = new Set(selectedNodeId ? [selectedNodeId] : []);
      }
      if (selectedNodeId && !this.selectedIds.has(selectedNodeId)) {
        this.selectedIds.add(selectedNodeId);
      }
      this.render();
    }

    /** Zero-DOM-rebuild fast live thumbnail updater for smooth animated canvas cards! */
    updateNodeThumbs(thumbsMap, timingsMap) {
      if (!thumbsMap) return;
      for (const [nid, url] of Object.entries(thumbsMap)) {
        const im = this._thumbImgs.get(nid);
        if (im && url && im.src !== url) {
          im.src = url;
        }
      }
      if (timingsMap) {
        for (const [nid, ms] of Object.entries(timingsMap)) {
          const el = this._nodeEls.get(nid);
          if (!el) continue;
          const sp = el.querySelector('.tg-node-ms');
          if (sp) {
            sp.textContent = ms === 0 ? 'cached · 0ms' : `${ms}ms`;
            sp.style.color = ms === 0 ? '#4ade80' : '';
          }
        }
      }
    }

    screenToWorld(clientX, clientY) {
      const r = this.stage.getBoundingClientRect();
      return {
        x: (clientX - r.left - this.panX) / this.zoom,
        y: (clientY - r.top - this.panY) / this.zoom,
      };
    }

    applyTransform() {
      this.world.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
      const sMinor = Math.max(8, 24 * this.zoom);
      const sMajor = sMinor * 5;
      this.gridBg.style.backgroundSize = `${sMinor}px ${sMinor}px, ${sMajor}px ${sMajor}px, ${sMajor}px ${sMajor}px`;
      this.gridBg.style.backgroundPosition = `${this.panX}px ${this.panY}px`;
      if (this.zoomBadge) this.zoomBadge.textContent = `${Math.round(this.zoom * 100)}%`;
      this.renderMinimap();
    }

    fitToView() {
      if (!this.recipe || !this.recipe.nodes || !this.recipe.nodes.length) return;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const n of this.recipe.nodes) {
        const x = (n.pos && n.pos[0]) || 0;
        const y = (n.pos && n.pos[1]) || 0;
        const isDot = n.op === 'reroute';
        minX = Math.min(minX, x - 24);
        minY = Math.min(minY, y - 34);
        maxX = Math.max(maxX, x + (isDot ? 52 : 240));
        maxY = Math.max(maxY, y + (isDot ? 44 : 165));
      }
      const r = this.stage.getBoundingClientRect();
      if (r.width < 20 || r.height < 20) return;
      const pad = 42;
      const gw = Math.max(120, maxX - minX);
      const gh = Math.max(120, maxY - minY);
      const zx = (r.width - pad * 2) / gw;
      const zy = (r.height - pad * 2) / gh;
      this.zoom = Math.max(0.14, Math.min(1.12, Math.min(zx, zy)));
      this.panX = (r.width - gw * this.zoom) * 0.5 - minX * this.zoom;
      this.panY = (r.height - gh * this.zoom) * 0.5 - minY * this.zoom;
      this.applyTransform();
    }

    zoomBy(factor) {
      const r = this.stage.getBoundingClientRect();
      const mx = r.width * 0.5;
      const my = r.height * 0.5;
      const nextZoom = Math.max(0.14, Math.min(2.5, this.zoom * factor));
      this.panX = mx - (mx - this.panX) * (nextZoom / this.zoom);
      this.panY = my - (my - this.panY) * (nextZoom / this.zoom);
      this.zoom = nextZoom;
      this.applyTransform();
    }

    _portColor(portType) {
      const colors = (this.schema && this.schema.port_colors) || {};
      return colors[portType] || '#94a3b8';
    }

    _startGroupDrag(e, ids) {
      const items = [];
      for (const id of ids) {
        const el = this._nodeEls.get(id);
        const nObj = (this.recipe.nodes || []).find((q) => q.id === id);
        if (!el || !nObj) continue;
        items.push({
          id,
          el,
          nObj,
          origX: (nObj.pos && nObj.pos[0]) || 0,
          origY: (nObj.pos && nObj.pos[1]) || 0,
        });
      }
      if (!items.length) return;
      this._dragNodes = {
        items,
        startClientX: e.clientX,
        startClientY: e.clientY,
        moved: false,
      };
    }

    _renderFrames(visibleNodes) {
      this.passLayer.innerHTML = '';

      // 1. Render @pass frames behind grouped nodes
      for (const ps of (this.recipe.passes || [])) {
        if (this.activePass !== 'ALL' && ps.name !== this.activePass) continue;
        const pNodes = ps.nodes || ps.node_ids || [];
        const members = visibleNodes.filter((n) => pNodes.includes(n.id));
        if (!members.length) continue;
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const m of members) {
          const mx = (m.pos && m.pos[0]) || 0;
          const my = (m.pos && m.pos[1]) || 0;
          x0 = Math.min(x0, mx - 20);
          y0 = Math.min(y0, my - 26);
          x1 = Math.max(x1, mx + (m.op === 'reroute' ? 48 : 234));
          y1 = Math.max(y1, my + (m.op === 'reroute' ? 42 : (m.collapsed ? 48 : 152)));
        }
        const fr = document.createElement('div');
        fr.className = 'pass-frame';
        fr.style.left = `${x0}px`;
        fr.style.top = `${y0}px`;
        fr.style.width = `${x1 - x0}px`;
        fr.style.height = `${y1 - y0}px`;
        if (ps.color) {
          fr.style.borderColor = `${ps.color}88`;
          fr.style.background = `${ps.color}20`;
        }
        const lbl = document.createElement('div');
        lbl.className = 'pass-frame-title';
        lbl.textContent = `${ps.title || ps.name} (${members.length})`;
        fr.appendChild(lbl);
        this.passLayer.appendChild(fr);
      }

      // 2. Render Interactive User Comment Frames (`recipe.frames` — Unreal/Blender Group parity!)
      for (const cf of (this.recipe.frames || [])) {
        const fNodes = cf.node_ids || [];
        const members = visibleNodes.filter((n) => fNodes.includes(n.id));
        if (!members.length) continue;
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const m of members) {
          const mx = (m.pos && m.pos[0]) || 0;
          const my = (m.pos && m.pos[1]) || 0;
          x0 = Math.min(x0, mx - 18);
          y0 = Math.min(y0, my - 34);
          x1 = Math.max(x1, mx + (m.op === 'reroute' ? 46 : 232));
          y1 = Math.max(y1, my + (m.op === 'reroute' ? 40 : (m.collapsed ? 48 : 150)));
        }
        const col = cf.color || '#2563EB';
        const fr = document.createElement('div');
        fr.className = 'comment-frame';
        fr.dataset.frameId = cf.id;
        fr.style.left = `${x0}px`;
        fr.style.top = `${y0}px`;
        fr.style.width = `${x1 - x0}px`;
        fr.style.height = `${y1 - y0}px`;
        fr.style.borderColor = `${col}bb`;
        fr.style.background = `${col}1d`;

        const head = document.createElement('div');
        head.className = 'comment-frame-head';
        head.style.background = `${col}42`;
        head.title = 'Drag header to move all grouped nodes · Double-click title to rename';

        const titleSpan = document.createElement('span');
        titleSpan.className = 'comment-frame-title';
        titleSpan.textContent = `${cf.title || 'Frame'} (${members.length})`;
        titleSpan.addEventListener('dblclick', (e) => {
          e.stopPropagation();
          const next = window.prompt('Rename Comment Frame:', cf.title || 'Comment Frame');
          if (next && next.trim() && this.callbacks.onUpdateFrame) {
            this.callbacks.onUpdateFrame(cf.id, { title: next.trim() });
          }
        });

        const acts = document.createElement('div');
        acts.className = 'comment-frame-acts';
        const btnCol = document.createElement('button');
        btnCol.className = 'tg-mini-btn';
        btnCol.textContent = '●';
        btnCol.style.color = col;
        btnCol.title = 'Cycle Frame Color';
        btnCol.addEventListener('pointerdown', (e) => e.stopPropagation());
        btnCol.onclick = (e) => {
          e.stopPropagation();
          const idx = FRAME_PALETTE.indexOf(col.toUpperCase());
          const nextCol = FRAME_PALETTE[(idx + 1) % FRAME_PALETTE.length];
          if (this.callbacks.onUpdateFrame) this.callbacks.onUpdateFrame(cf.id, { color: nextCol });
        };

        const btnDel = document.createElement('button');
        btnDel.className = 'tg-mini-btn del-corner';
        btnDel.textContent = '✕';
        btnDel.title = 'Remove Frame (keeps enclosed nodes)';
        btnDel.addEventListener('pointerdown', (e) => e.stopPropagation());
        btnDel.onclick = (e) => {
          e.stopPropagation();
          if (this.callbacks.onDeleteFrame) this.callbacks.onDeleteFrame(cf.id);
        };

        acts.appendChild(btnCol);
        acts.appendChild(btnDel);
        head.appendChild(titleSpan);
        head.appendChild(acts);

        head.addEventListener('pointerdown', (e) => {
          if (e.button !== 0) return;
          e.stopPropagation();
          const memberIds = members.map((m) => m.id);
          this.selectedIds = new Set(memberIds);
          if (this.callbacks.onSelectionChange) this.callbacks.onSelectionChange(memberIds, memberIds[0]);
          this._startGroupDrag(e, memberIds);
        });

        fr.appendChild(head);
        this.passLayer.appendChild(fr);
      }
    }

    render() {
      if (!this.recipe) return;
      this.nodesLayer.innerHTML = '';
      this._socketEls.clear();
      this._nodeEls.clear();
      this._thumbImgs.clear();

      const outBase = (this.recipe.output_material || this.recipe.output || this.recipe.output_material_node || this.recipe.output_node || '').split('.')[0];
      const passNodeSets = {};
      for (const ps of (this.recipe.passes || [])) {
        passNodeSets[ps.name] = new Set(ps.nodes || ps.node_ids || []);
      }

      const visibleNodes = (this.recipe.nodes || []).filter((n) => {
        if (this.activePass === 'ALL') return true;
        const s = passNodeSets[this.activePass];
        return s ? s.has(n.id) : true;
      });
      const visibleIds = new Set(visibleNodes.map((n) => n.id));
      this._visibleNodes = visibleNodes;

      this._renderFrames(visibleNodes);

      // Render Nodes (compact Reroute Dot vs Full Commercial Node Card)
      for (const n of visibleNodes) {
        const meta = this.opMap[n.op] || { op: n.op, category: 'Custom', inputs: [], outputs: [{ name: 'out', port_type: 'RGBA' }] };
        const accent = CAT_COLORS[meta.category] || '#64748b';
        const isOut = n.id === outBase;
        const isSel = this.selectedIds.has(n.id) || n.id === this.selectedNodeId;
        const x = (n.pos && n.pos[0]) || 0;
        const y = (n.pos && n.pos[1]) || 0;

        if (n.op === 'reroute') {
          const dot = document.createElement('div');
          dot.className = 'tg-node tg-reroute' + (isSel ? ' selected' : '');
          dot.dataset.nodeId = n.id;
          dot.style.left = `${x}px`;
          dot.style.top = `${y}px`;
          dot.title = `${n.id} (Reroute Dot) — Drag to position · Del to delete`;

          const inSock = document.createElement('div');
          inSock.className = 'tg-socket shape-circle';
          inSock.style.setProperty('--port-color', this._portColor('ANY'));
          inSock.dataset.dir = 'in';
          inSock.dataset.nodeId = n.id;
          inSock.dataset.port = 'in';
          inSock.dataset.portType = 'ANY';
          inSock.addEventListener('pointerdown', (e) => {
            if (e.altKey) { e.stopPropagation(); e.preventDefault(); this.callbacks.onDisconnect(n.id, 'in'); }
          });

          const core = document.createElement('div');
          core.className = 'tg-reroute-core';

          const outSock = document.createElement('div');
          outSock.className = 'tg-socket shape-circle';
          outSock.style.setProperty('--port-color', this._portColor('ANY'));
          outSock.dataset.dir = 'out';
          outSock.dataset.nodeId = n.id;
          outSock.dataset.port = 'out';
          outSock.dataset.portType = 'ANY';
          outSock.addEventListener('pointerdown', (e) => this._startWireDrag(e, n.id, 'out', 'ANY'));

          const delX = document.createElement('button');
          delX.className = 'tg-reroute-del del-corner';
          delX.textContent = '✕';
          delX.title = 'Delete Reroute';
          delX.addEventListener('pointerdown', (e) => e.stopPropagation());
          delX.onclick = (e) => { e.stopPropagation(); if (this.callbacks.onDeleteNode) this.callbacks.onDeleteNode(n.id); };

          dot.appendChild(inSock);
          dot.appendChild(core);
          dot.appendChild(outSock);
          dot.appendChild(delX);
          this._socketEls.set(`in:${n.id}:in`, inSock);
          this._socketEls.set(`out:${n.id}:out`, outSock);

          dot.addEventListener('pointerdown', (e) => this._onNodePointerDown(e, n.id));
          dot.addEventListener('contextmenu', (e) => {
            e.preventDefault(); e.stopPropagation();
            this.callbacks.onSelectNode(n.id, false);
            if (this.callbacks.onNodeContextMenu) this.callbacks.onNodeContextMenu(n.id, e.clientX, e.clientY);
          });

          this.nodesLayer.appendChild(dot);
          this._nodeEls.set(n.id, dot);
          continue;
        }

        const el = document.createElement('div');
        const isGroup = n.op === 'subgraph_group';
        el.className = 'tg-node' + (isSel ? ' selected' : '') + (isOut ? ' is-output' : '') + (isGroup ? ' is-group' : '') + (n.bypassed ? ' bypassed' : '') + (n.collapsed ? ' collapsed' : '');
        el.dataset.nodeId = n.id;
        el.style.left = `${x}px`;
        el.style.top = `${y}px`;
        el.style.setProperty('--node-accent', isGroup ? ((n.params && n.params.color) || '#7e22ce') : accent);

        const head = document.createElement('div');
        head.className = 'tg-node-head';
        const title = document.createElement('div');
        title.className = 'tg-node-title';
        const grpCount = isGroup && n.params && n.params.sub_recipe && Array.isArray(n.params.sub_recipe.nodes) ? n.params.sub_recipe.nodes.length : 0;
        title.textContent = isGroup ? `⊞ ${(n.params && n.params.title) || n.label || n.id} (${grpCount})` : (n.label || n.id);
        title.title = isGroup ? `Compound Subgraph (${grpCount} nodes) — Double-click to step inside!` : `${n.id} (${n.op})`;

        const acts = document.createElement('div');
        acts.className = 'tg-node-actions';

        const mkMini = (txt, tip, cls, onClick) => {
          const b = document.createElement('button');
          b.className = 'tg-mini-btn' + (cls ? ' ' + cls : '');
          b.textContent = txt;
          b.title = tip;
          b.addEventListener('pointerdown', (e) => e.stopPropagation());
          b.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
          return b;
        };

        if (isGroup) {
          acts.appendChild(mkMini('↗', 'Enter Compound Subgraph', 'out-on', () => {
            if (this.callbacks.onEnterGroup) this.callbacks.onEnterGroup(n.id);
          }));
        }
        acts.appendChild(mkMini('👁', 'Inspect Output in Viewport', 'eye-btn', () => this.callbacks.onSelectNode(n.id, true)));
        acts.appendChild(mkMini(n.collapsed ? '▸' : '▾', 'Collapse/Expand', '', () => this.callbacks.onToggleCollapse(n.id)));
        acts.appendChild(mkMini('B', 'Bypass Node (B)', n.bypassed ? 'on' : '', () => this.callbacks.onToggleBypass(n.id)));
        acts.appendChild(mkMini('★', 'Set Final Output', isOut ? 'out-on' : '', () => this.callbacks.onSetOutput(n.id)));
        acts.appendChild(mkMini('✕', 'Delete Node (Del)', 'del-corner', () => {
          if (this.callbacks.onDeleteNode) this.callbacks.onDeleteNode(n.id);
        }));

        head.appendChild(title);
        head.appendChild(acts);
        head.addEventListener('dblclick', (e) => {
          e.stopPropagation();
          if (isGroup && this.callbacks.onEnterGroup) {
            this.callbacks.onEnterGroup(n.id);
          } else {
            this.callbacks.onSelectNode(n.id, true);
          }
        });
        el.appendChild(head);

        const body = document.createElement('div');
        body.className = 'tg-node-body';

        const inCol = document.createElement('div');
        inCol.className = 'tg-ports-col in';
        const inputsMeta = (meta.inputs && meta.inputs.length) ? meta.inputs : Object.keys(n.inputs || {}).map((k) => ({ name: k, dtype: 'RGBA' }));
        for (const inp of inputsMeta.slice(0, 6)) {
          const pType = inp.dtype || inp.port_type || 'RGBA';
          const row = document.createElement('div');
          row.className = 'tg-port-row in';
          const sock = document.createElement('div');
          sock.className = `tg-socket ${socketShape(pType)}`;
          sock.style.setProperty('--port-color', this._portColor(pType));
          sock.dataset.dir = 'in';
          sock.dataset.nodeId = n.id;
          sock.dataset.port = inp.name;
          sock.dataset.portType = pType;
          sock.title = `Input ${inp.name} (${pType}) — Drag to re-plug · Alt+Click to sever`;
          sock.addEventListener('pointerdown', (e) => {
            if (e.altKey) {
              e.stopPropagation();
              e.preventDefault();
              this.callbacks.onDisconnect(n.id, inp.name);
              return;
            }
            // Blender-style drag-off connected input socket to re-plug it smoothly!
            const rawRef = n.inputs && n.inputs[inp.name];
            if (e.button === 0 && rawRef) {
              const parts = String(rawRef).split('.');
              const srcId = parts[0];
              const srcPort = parts[1] || 'out';
              this.callbacks.onDisconnect(n.id, inp.name);
              this._startWireDrag(e, srcId, srcPort, pType);
            }
          });
          row.appendChild(sock);
          const lbl = document.createElement('span');
          lbl.textContent = inp.name;
          row.appendChild(lbl);
          inCol.appendChild(row);
          this._socketEls.set(`in:${n.id}:${inp.name}`, sock);
        }

        const thumbWrap = document.createElement('div');
        thumbWrap.className = 'tg-node-thumb-wrap';
        thumbWrap.title = 'Click to inspect node output in Viewport';
        const thumbUrl = this.evalData && this.evalData.thumbs && this.evalData.thumbs[n.id];
        if (thumbUrl) {
          const im = document.createElement('img');
          im.src = thumbUrl;
          im.draggable = false;
          thumbWrap.appendChild(im);
          this._thumbImgs.set(n.id, im);
        }
        thumbWrap.addEventListener('click', (e) => {
          e.stopPropagation();
          this.callbacks.onSelectNode(n.id, true);
        });

        const outCol = document.createElement('div');
        outCol.className = 'tg-ports-col out';
        const defOutType = meta.output_type || 'RGBA';
        const rawOuts = (meta.outputs && meta.outputs.length)
          ? meta.outputs
          : [{ name: defOutType === 'MATERIAL' ? 'mat' : 'out', dtype: defOutType }];
        const outputsMeta = (defOutType === 'MATERIAL' && !rawOuts.some((o) => (o.dtype || o.port_type) === 'MATERIAL'))
          ? [{ name: 'mat', dtype: 'MATERIAL' }, ...rawOuts]
          : rawOuts;
        const shownOuts = outputsMeta.length > 5
          ? [outputsMeta[0], ...outputsMeta.slice(1, 4)]
          : outputsMeta;
        for (const outp of shownOuts) {
          const pType = outp.dtype || outp.port_type || defOutType;
          const row = document.createElement('div');
          row.className = 'tg-port-row out';
          const sock = document.createElement('div');
          sock.className = `tg-socket ${socketShape(pType)}`;
          sock.style.setProperty('--port-color', this._portColor(pType));
          sock.dataset.dir = 'out';
          sock.dataset.nodeId = n.id;
          sock.dataset.port = outp.name;
          sock.dataset.portType = pType;
          sock.title = `Output ${outp.name} (${pType}) — drag to connect`;
          sock.addEventListener('pointerdown', (e) => this._startWireDrag(e, n.id, outp.name, pType));
          row.appendChild(sock);
          const lbl = document.createElement('span');
          lbl.textContent = outp.name;
          row.appendChild(lbl);
          outCol.appendChild(row);
          this._socketEls.set(`out:${n.id}:${outp.name}`, sock);
        }

        body.appendChild(inCol);
        body.appendChild(thumbWrap);
        body.appendChild(outCol);
        el.appendChild(body);

        const foot = document.createElement('div');
        foot.className = 'tg-node-foot';
        const opSpan = document.createElement('span');
        if (meta.time_aware) {
          opSpan.innerHTML = `<span class="tg-anim-dot" title="Dynamic / Animated Node">▶</span>${n.op}`;
        } else {
          opSpan.textContent = n.op;
        }
        const ms = this.evalData && this.evalData.timings_ms ? this.evalData.timings_ms[n.id] : null;
        const msSpan = document.createElement('span');
        msSpan.className = 'tg-node-ms';
        if (ms !== null && ms !== undefined) {
          msSpan.textContent = ms === 0 ? 'cached · 0ms' : `${ms}ms`;
          if (ms === 0) msSpan.style.color = '#4ade80';
        }
        foot.appendChild(opSpan);
        foot.appendChild(msSpan);
        el.appendChild(foot);

        el.addEventListener('contextmenu', (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!this.selectedIds.has(n.id)) {
            this.selectedIds = new Set([n.id]);
            this.callbacks.onSelectNode(n.id, false);
          }
          if (this.callbacks.onNodeContextMenu) {
            this.callbacks.onNodeContextMenu(n.id, e.clientX, e.clientY, Array.from(this.selectedIds));
          }
        });

        el.addEventListener('pointerdown', (e) => this._onNodePointerDown(e, n.id));

        this.nodesLayer.appendChild(el);
        this._nodeEls.set(n.id, el);
      }

      this.renderWires(visibleIds);
      this.applyTransform();
    }

    _onNodePointerDown(e, nodeId) {
      if (e.target.classList.contains('tg-socket')) return;
      e.stopPropagation();
      if (e.button !== 0) return;

      if (e.shiftKey) {
        if (this.selectedIds.has(nodeId) && this.selectedIds.size > 1) {
          this.selectedIds.delete(nodeId);
        } else {
          this.selectedIds.add(nodeId);
        }
        const primary = this.selectedIds.has(nodeId) ? nodeId : Array.from(this.selectedIds)[0];
        if (this.callbacks.onSelectionChange) {
          this.callbacks.onSelectionChange(Array.from(this.selectedIds), primary);
        }
      } else {
        if (!this.selectedIds.has(nodeId)) {
          this.selectedIds = new Set([nodeId]);
          this.callbacks.onSelectNode(nodeId, false);
        } else {
          this.selectedNodeId = nodeId;
          if (this.callbacks.onSelectionChange) {
            this.callbacks.onSelectionChange(Array.from(this.selectedIds), nodeId);
          }
        }
      }

      this._startGroupDrag(e, Array.from(this.selectedIds));
    }

    _socketWorldCenter(sockEl) {
      if (!sockEl) return null;
      const sr = sockEl.getBoundingClientRect();
      const st = this.stage.getBoundingClientRect();
      return {
        x: (sr.left + sr.width * 0.5 - st.left - this.panX) / this.zoom,
        y: (sr.top + sr.height * 0.5 - st.top - this.panY) / this.zoom,
      };
    }

    _nodeFallbackPortPos(nodeId, dir) {
      const n = (this.recipe.nodes || []).find((x) => x.id === nodeId);
      if (!n) return { x: 0, y: 0 };
      const x = (n.pos && n.pos[0]) || 0;
      const y = (n.pos && n.pos[1]) || 0;
      if (n.op === 'reroute') return { x: dir === 'out' ? x + 30 : x + 4, y: y + 17 };
      return { x: dir === 'out' ? x + 214 : x, y: y + 40 };
    }

    renderWires(visibleIds) {
      this.wiresSvg.innerHTML = '';
      if (!this.recipe) return;
      const ns = 'http://www.w3.org/2000/svg';

      for (const dstNode of (this.recipe.nodes || [])) {
        if (visibleIds && !visibleIds.has(dstNode.id)) continue;
        for (const [inPort, rawRef] of Object.entries(dstNode.inputs || {})) {
          const parts = String(rawRef).split('.');
          const srcId = parts[0];
          const srcPort = parts[1] || null;
          if (visibleIds && !visibleIds.has(srcId)) continue;

          const srcNode = (this.recipe.nodes || []).find((x) => x.id === srcId);
          const srcMeta = srcNode ? this.opMap[srcNode.op] : null;
          let outPortName = srcPort;
          let outPortType = (srcMeta && srcMeta.output_type) || 'RGBA';
          if (srcMeta && srcMeta.outputs && srcMeta.outputs.length) {
            const match = outPortName ? srcMeta.outputs.find((o) => o.name === outPortName) : null;
            if (match) {
              outPortName = match.name;
              outPortType = match.dtype || match.port_type || outPortType;
            } else if (outPortType === 'MATERIAL') {
              outPortName = 'mat';
            } else {
              outPortName = srcMeta.outputs[0].name;
              outPortType = srcMeta.outputs[0].dtype || srcMeta.outputs[0].port_type || outPortType;
            }
          } else {
            outPortName = outPortType === 'MATERIAL' ? 'mat' : 'out';
          }

          const sEl = this._socketEls.get(`out:${srcId}:${outPortName}`) || this._socketEls.get(`out:${srcId}:out`) || this._socketEls.get(`out:${srcId}:mat`);
          const dEl = this._socketEls.get(`in:${dstNode.id}:${inPort}`);
          const p1 = this._socketWorldCenter(sEl) || this._nodeFallbackPortPos(srcId, 'out');
          const p2 = this._socketWorldCenter(dEl) || this._nodeFallbackPortPos(dstNode.id, 'in');

          const isRel = this.selectedIds.has(srcId) || this.selectedIds.has(dstNode.id);
          const g = document.createElementNS(ns, 'g');
          g.setAttribute('class', 'wire-group' + (isRel ? ' wire-active' : ''));

          const dStr = bezierPath(p1.x, p1.y, p2.x, p2.y);
          const hit = document.createElementNS(ns, 'path');
          hit.setAttribute('class', 'wire-hit');
          hit.setAttribute('d', dStr);
          hit.addEventListener('click', (e) => {
            e.stopPropagation();
            if (e.altKey) {
              this.callbacks.onDisconnect(dstNode.id, inPort);
            }
          });
          hit.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            const wp = this.screenToWorld(e.clientX, e.clientY);
            if (this.callbacks.onInsertReroute) {
              this.callbacks.onInsertReroute(dstNode.id, inPort, [Math.round(wp.x - 16), Math.round(wp.y - 16)]);
            }
          });
          hit.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.callbacks.onDisconnect(dstNode.id, inPort);
          });

          const titleEl = document.createElementNS(ns, 'title');
          titleEl.textContent = `${srcId} → ${dstNode.id}.${inPort}  |  Double-Click: Add Reroute · Alt+Click / Right-Click: Sever`;
          hit.appendChild(titleEl);

          const path = document.createElementNS(ns, 'path');
          path.setAttribute('class', 'wire-path');
          path.setAttribute('d', dStr);
          path.setAttribute('stroke', this._portColor(outPortType));

          g.appendChild(hit);
          g.appendChild(path);
          this.wiresSvg.appendChild(g);
        }
      }

      if (this._wireDrag) {
        const ghost = document.createElementNS(ns, 'path');
        ghost.setAttribute('class', 'wire-ghost');
        ghost.setAttribute('stroke', this._portColor(this._wireDrag.portType));
        ghost.setAttribute('d', bezierPath(this._wireDrag.x1, this._wireDrag.y1, this._wireDrag.x2, this._wireDrag.y2));
        this.wiresSvg.appendChild(ghost);
      }
    }

    _startWireDrag(e, srcNode, srcPort, portType) {
      e.stopPropagation();
      e.preventDefault();
      const sEl = this._socketEls.get(`out:${srcNode}:${srcPort}`);
      const p1 = this._socketWorldCenter(sEl) || this.screenToWorld(e.clientX, e.clientY);
      this._wireDrag = { srcNode, srcPort, portType, x1: p1.x, y1: p1.y, x2: p1.x, y2: p1.y };

      const compat = new Set(((this.schema && this.schema.port_compatibility) || {})[portType] || [portType]);
      for (const [key, sock] of this._socketEls.entries()) {
        if (!key.startsWith('in:')) continue;
        if (sock.dataset.nodeId === srcNode) {
          sock.classList.add('compat-dim');
        } else if (compat.has(sock.dataset.portType)) {
          sock.classList.add('compat-ok');
        } else {
          sock.classList.add('compat-dim');
        }
      }
      this.renderWires();
    }

    _findNearestCompatSocket(wx, wy) {
      if (!this._wireDrag) return null;
      const compat = new Set(((this.schema && this.schema.port_compatibility) || {})[this._wireDrag.portType] || [this._wireDrag.portType]);
      let best = null;
      let bestDist = 44 / Math.max(0.35, this.zoom);
      for (const [key, sock] of this._socketEls.entries()) {
        if (!key.startsWith('in:')) continue;
        if (sock.dataset.nodeId === this._wireDrag.srcNode) continue;
        if (!compat.has(sock.dataset.portType) && sock.dataset.portType !== 'ANY' && this._wireDrag.portType !== 'ANY') continue;
        const c = this._socketWorldCenter(sock);
        if (!c) continue;
        const d = Math.hypot(c.x - wx, c.y - wy);
        if (d < bestDist) {
          bestDist = d;
          best = { sock, x: c.x, y: c.y, nodeId: sock.dataset.nodeId, port: sock.dataset.port };
        }
      }
      return best;
    }

    _clearWireHighlights() {
      for (const sock of this._socketEls.values()) {
        sock.classList.remove('compat-ok', 'compat-dim', 'snap-target');
      }
    }

    renderMinimap() {
      if (!this.minimapCanvas || !this.recipe) return;
      const ctx = this.minimapCanvas.getContext('2d');
      const W = this.minimapCanvas.width, H = this.minimapCanvas.height;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#0f1116';
      ctx.fillRect(0, 0, W, H);

      const nodes = this.recipe.nodes || [];
      if (!nodes.length) return;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const n of nodes) {
        const x = (n.pos && n.pos[0]) || 0;
        const y = (n.pos && n.pos[1]) || 0;
        minX = Math.min(minX, x - 60);
        minY = Math.min(minY, y - 60);
        maxX = Math.max(maxX, x + 280);
        maxY = Math.max(maxY, y + 200);
      }
      const sc = Math.min((W - 12) / Math.max(1, maxX - minX), (H - 12) / Math.max(1, maxY - minY));
      const ox = (W - (maxX - minX) * sc) * 0.5 - minX * sc;
      const oy = (H - (maxY - minY) * sc) * 0.5 - minY * sc;
      this._miniBounds = { sc, ox, oy };

      ctx.strokeStyle = 'rgba(148,163,184,0.35)';
      ctx.lineWidth = 1;
      for (const n of nodes) {
        const dx = ((n.pos && n.pos[0]) || 0) * sc + ox;
        const dy = ((n.pos && n.pos[1]) || 0) * sc + oy + 5;
        for (const ref of Object.values(n.inputs || {})) {
          const sn = nodes.find((q) => q.id === String(ref).split('.')[0]);
          if (!sn) continue;
          const sx = ((sn.pos && sn.pos[0]) || 0) * sc + ox + (sn.op === 'reroute' ? 28 : 214) * sc;
          const sy = ((sn.pos && sn.pos[1]) || 0) * sc + oy + 5;
          ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(dx, dy); ctx.stroke();
        }
      }

      for (const n of nodes) {
        const meta = this.opMap[n.op] || {};
        const x = ((n.pos && n.pos[0]) || 0) * sc + ox;
        const y = ((n.pos && n.pos[1]) || 0) * sc + oy;
        ctx.fillStyle = this.selectedIds.has(n.id) ? '#3b82f6' : (CAT_COLORS[meta.category] || '#64748b');
        const bw = n.op === 'reroute' ? Math.max(4, 32 * sc) : Math.max(5, 214 * sc);
        const bh = n.op === 'reroute' ? Math.max(4, 32 * sc) : Math.max(4, 110 * sc);
        ctx.fillRect(x, y, bw, bh);
      }

      const r = this.stage.getBoundingClientRect();
      const vx0 = (-this.panX / this.zoom) * sc + ox;
      const vy0 = (-this.panY / this.zoom) * sc + oy;
      const vw = (r.width / this.zoom) * sc;
      const vh = (r.height / this.zoom) * sc;
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.25;
      ctx.strokeRect(vx0, vy0, vw, vh);
    }

    _bindEvents() {
      window.addEventListener('keydown', (e) => { if (e.code === 'Space') this._spaceDown = true; });
      window.addEventListener('keyup', (e) => { if (e.code === 'Space') this._spaceDown = false; });

      this.stage.addEventListener('pointerdown', (e) => {
        if (e.target.closest('.graph-minimap') || e.target.closest('.graph-zoom-hud')) return;
        // Middle mouse, Space+Left, or Pan tool -> Pan canvas
        if (e.button === 1 || (e.button === 0 && (this.toolMode === 'pan' || this._spaceDown))) {
          this._panning = true;
          this._panStartX = e.clientX;
          this._panStartY = e.clientY;
          this._origPanX = this.panX;
          this._origPanY = this.panY;
          this.stage.style.cursor = 'grabbing';
          return;
        }
        // Right mouse drag -> Pan canvas (if moved > 4px), else context menu opens!
        if (e.button === 2) {
          this._rightPanPending = {
            startX: e.clientX,
            startY: e.clientY,
            origPanX: this.panX,
            origPanY: this.panY,
            panned: false,
          };
          return;
        }
        // Left drag in Select mode -> Box-Select Marquee!
        if (e.button === 0) {
          const r = this.stage.getBoundingClientRect();
          const sx = e.clientX - r.left;
          const sy = e.clientY - r.top;
          this._marquee = {
            startX: sx,
            startY: sy,
            curX: sx,
            curY: sy,
            additive: e.shiftKey,
            moved: false,
          };
          this._marqueeEl.style.left = `${sx}px`;
          this._marqueeEl.style.top = `${sy}px`;
          this._marqueeEl.style.width = '0px';
          this._marqueeEl.style.height = '0px';
          this._marqueeEl.style.display = 'block';
        }
      });

      this.stage.addEventListener('dblclick', (e) => {
        if (e.target.closest('.tg-node') || e.target.closest('.graph-minimap') || e.target.closest('.graph-zoom-hud') || e.target.closest('.comment-frame-head') || e.target.classList.contains('wire-hit')) return;
        const wPos = this.screenToWorld(e.clientX, e.clientY);
        if (this.callbacks.onOpenAddAt) this.callbacks.onOpenAddAt(wPos);
      });

      this.stage.addEventListener('contextmenu', (e) => {
        if (e.target.closest('.tg-node') || e.target.closest('.graph-minimap') || e.target.classList.contains('wire-hit')) return;
        e.preventDefault();
        if (this._suppressNextCtxMenu) {
          this._suppressNextCtxMenu = false;
          return;
        }
        const wPos = this.screenToWorld(e.clientX, e.clientY);
        if (this.callbacks.onCanvasContextMenu) {
          this.callbacks.onCanvasContextMenu(e.clientX, e.clientY, wPos);
        }
      });

      window.addEventListener('pointermove', (e) => {
        if (this._wireDrag) {
          const w = this.screenToWorld(e.clientX, e.clientY);
          const snap = this._findNearestCompatSocket(w.x, w.y);
          for (const sock of this._socketEls.values()) sock.classList.remove('snap-target');
          if (snap) {
            snap.sock.classList.add('snap-target');
            this._wireDrag.x2 = snap.x;
            this._wireDrag.y2 = snap.y;
            this._wireDrag.snapTarget = snap;
          } else {
            this._wireDrag.x2 = w.x;
            this._wireDrag.y2 = w.y;
            this._wireDrag.snapTarget = null;
          }
          this.renderWires();
          return;
        }
        if (this._dragNodes) {
          const dx = (e.clientX - this._dragNodes.startClientX) / this.zoom;
          const dy = (e.clientY - this._dragNodes.startClientY) / this.zoom;
          if (Math.hypot(dx, dy) > 2) this._dragNodes.moved = true;
          for (const it of this._dragNodes.items) {
            const nx = Math.round((it.origX + dx) / 12) * 12;
            const ny = Math.round((it.origY + dy) / 12) * 12;
            it.el.style.left = `${nx}px`;
            it.el.style.top = `${ny}px`;
            it.nObj.pos = [nx, ny];
          }
          this._renderFrames(this._visibleNodes || []);
          this.renderWires();
          this.renderMinimap();
          return;
        }
        if (this._marquee) {
          const r = this.stage.getBoundingClientRect();
          const cx = e.clientX - r.left;
          const cy = e.clientY - r.top;
          if (Math.hypot(cx - this._marquee.startX, cy - this._marquee.startY) > 4) {
            this._marquee.moved = true;
          }
          this._marquee.curX = cx;
          this._marquee.curY = cy;
          const x0 = Math.min(this._marquee.startX, cx);
          const y0 = Math.min(this._marquee.startY, cy);
          const w = Math.abs(cx - this._marquee.startX);
          const h = Math.abs(cy - this._marquee.startY);
          this._marqueeEl.style.left = `${x0}px`;
          this._marqueeEl.style.top = `${y0}px`;
          this._marqueeEl.style.width = `${w}px`;
          this._marqueeEl.style.height = `${h}px`;
          return;
        }
        if (this._rightPanPending) {
          const dx = e.clientX - this._rightPanPending.startX;
          const dy = e.clientY - this._rightPanPending.startY;
          if (Math.hypot(dx, dy) > 5) {
            this._rightPanPending.panned = true;
            this.panX = this._rightPanPending.origPanX + dx;
            this.panY = this._rightPanPending.origPanY + dy;
            this.stage.style.cursor = 'grabbing';
            this.applyTransform();
          }
          return;
        }
        if (this._panning) {
          this.panX = this._origPanX + (e.clientX - this._panStartX);
          this.panY = this._origPanY + (e.clientY - this._panStartY);
          this.applyTransform();
        }
      });

      window.addEventListener('pointerup', (e) => {
        if (this._wireDrag) {
          const drag = this._wireDrag;
          this._wireDrag = null;
          this._clearWireHighlights();
          const target = document.elementFromPoint(e.clientX, e.clientY);
          if (target && target.classList && target.classList.contains('tg-socket') && target.dataset.dir === 'in') {
            this.callbacks.onConnect(drag.srcNode, drag.srcPort, target.dataset.nodeId, target.dataset.port);
          } else if (drag.snapTarget) {
            this.callbacks.onConnect(drag.srcNode, drag.srcPort, drag.snapTarget.nodeId, drag.snapTarget.port);
          } else {
            const wPos = this.screenToWorld(e.clientX, e.clientY);
            this.callbacks.onDropWireEmpty(drag.srcNode, drag.srcPort, drag.portType, wPos);
          }
          this.renderWires();
          return;
        }
        if (this._dragNodes) {
          const dn = this._dragNodes;
          this._dragNodes = null;
          if (dn.moved) {
            const positions = {};
            for (const it of dn.items) {
              if (it.nObj && it.nObj.pos) positions[it.id] = it.nObj.pos;
            }
            if (this.callbacks.onMoveNodes) {
              this.callbacks.onMoveNodes(positions);
            } else if (dn.items.length === 1 && this.callbacks.onMoveNode) {
              this.callbacks.onMoveNode(dn.items[0].id, dn.items[0].nObj.pos);
            }
            // Smart Drop-on-Wire: if a single unconnected node was dropped near a wire midpoint, auto-wire it inline!
            if (dn.items.length === 1 && this.callbacks.onInsertNodeOnWire) {
              const dropped = dn.items[0].nObj;
              if (dropped && dropped.pos && (!dropped.inputs || !Object.keys(dropped.inputs).length)) {
                const cx = dropped.pos[0] + 105;
                const cy = dropped.pos[1] + 55;
                for (const dstN of (this._visibleNodes || [])) {
                  if (dstN.id === dropped.id) continue;
                  for (const [inP, rawR] of Object.entries(dstN.inputs || {})) {
                    const srcId = String(rawR).split('.')[0];
                    if (srcId === dropped.id) continue;
                    const sEl = this._socketEls.get(`out:${srcId}:out`) || this._socketEls.get(`out:${srcId}:mat`);
                    const dEl = this._socketEls.get(`in:${dstN.id}:${inP}`);
                    const p1 = this._socketWorldCenter(sEl) || this._nodeFallbackPortPos(srcId, 'out');
                    const p2 = this._socketWorldCenter(dEl) || this._nodeFallbackPortPos(dstN.id, 'in');
                    const mx = (p1.x + p2.x) * 0.5, my = (p1.y + p2.y) * 0.5;
                    if (Math.hypot(cx - mx, cy - my) < 42) {
                      this.callbacks.onInsertNodeOnWire(dropped.id, dstN.id, inP);
                      return;
                    }
                  }
                }
              }
            }
          }
          return;
        }
        if (this._marquee) {
          const mq = this._marquee;
          this._marquee = null;
          this._marqueeEl.style.display = 'none';
          if (mq.moved) {
            const r = this.stage.getBoundingClientRect();
            const wx0 = (Math.min(mq.startX, mq.curX) - this.panX) / this.zoom;
            const wy0 = (Math.min(mq.startY, mq.curY) - this.panY) / this.zoom;
            const wx1 = (Math.max(mq.startX, mq.curX) - this.panX) / this.zoom;
            const wy1 = (Math.max(mq.startY, mq.curY) - this.panY) / this.zoom;
            const nextSel = mq.additive ? new Set(this.selectedIds) : new Set();
            for (const n of (this._visibleNodes || [])) {
              const nx = (n.pos && n.pos[0]) || 0;
              const ny = (n.pos && n.pos[1]) || 0;
              const nw = n.op === 'reroute' ? 34 : 214;
              const nh = n.op === 'reroute' ? 34 : (n.collapsed ? 34 : 132);
              if (nx + nw >= wx0 && nx <= wx1 && ny + nh >= wy0 && ny <= wy1) {
                nextSel.add(n.id);
              }
            }
            this.selectedIds = nextSel;
            const ids = Array.from(nextSel);
            if (this.callbacks.onSelectionChange) {
              this.callbacks.onSelectionChange(ids, ids[ids.length - 1] || null);
            }
          }
          return;
        }
        if (this._rightPanPending) {
          if (this._rightPanPending.panned) {
            this._suppressNextCtxMenu = true;
            this.stage.style.cursor = this.toolMode === 'pan' ? 'grab' : 'default';
          }
          this._rightPanPending = null;
          return;
        }
        if (this._panning) {
          this._panning = false;
          this.stage.style.cursor = this.toolMode === 'pan' ? 'grab' : 'default';
        }
      });

      this.stage.addEventListener('wheel', (e) => {
        e.preventDefault();
        const r = this.stage.getBoundingClientRect();
        const mx = e.clientX - r.left;
        const my = e.clientY - r.top;
        const factor = e.deltaY > 0 ? 0.91 : 1.10;
        const nextZoom = Math.max(0.14, Math.min(2.5, this.zoom * factor));
        this.panX = mx - (mx - this.panX) * (nextZoom / this.zoom);
        this.panY = my - (my - this.panY) * (nextZoom / this.zoom);
        this.zoom = nextZoom;
        this.applyTransform();
      }, { passive: false });

      if (this.minimapCanvas) {
        let miniDrag = false;
        const jump = (e) => {
          if (!this._miniBounds) return;
          const r = this.minimapCanvas.getBoundingClientRect();
          const mx = (e.clientX - r.left) * (this.minimapCanvas.width / r.width);
          const my = (e.clientY - r.top) * (this.minimapCanvas.height / r.height);
          const wx = (mx - this._miniBounds.ox) / this._miniBounds.sc;
          const wy = (my - this._miniBounds.oy) / this._miniBounds.sc;
          const sr = this.stage.getBoundingClientRect();
          this.panX = sr.width * 0.5 - wx * this.zoom;
          this.panY = sr.height * 0.5 - wy * this.zoom;
          this.applyTransform();
        };
        this.minimapCanvas.addEventListener('pointerdown', (e) => { e.stopPropagation(); miniDrag = true; jump(e); });
        window.addEventListener('pointermove', (e) => { if (miniDrag) jump(e); });
        window.addEventListener('pointerup', () => { miniDrag = false; });
      }
    }
  }

  global.StudioGraphCanvas = StudioGraphCanvas;
})(window);
