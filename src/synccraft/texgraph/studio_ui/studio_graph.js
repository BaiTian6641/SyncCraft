/* ==========================================================================
   SyncCraft Studio — Infinite Node Graph Canvas & Interactive Minimap
   Features: Pan/Zoom, @pass frames, Cubic Bezier typed wires, Color+Shape
   sockets, Live PORT_COMPATIBILITY highlighting, Drag-to-Empty Quick-Add,
   Node Thumbnails, Bypass/Collapse/Pin controls & Click-Drag Minimap.
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
  };

  function socketShape(portType) {
    if (portType === 'MATERIAL') return 'shape-square';
    if (portType === 'MASK' || portType === 'VECTOR3') return 'shape-diamond';
    return 'shape-circle';
  }

  function bezierPath(x1, y1, x2, y2) {
    const dx = Math.max(46, Math.abs(x2 - x1) * 0.48);
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

      this.schema = null;
      this.opMap = {};
      this.recipe = null;
      this.evalData = null;
      this.selectedNodeId = null;
      this.activePass = 'ALL';

      this._dragNode = null;
      this._panning = false;
      this._spaceDown = false;
      this._wireDrag = null;
      this._socketEls = new Map();
      this._nodeEls = new Map();

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

    setData(recipe, evalData, selectedNodeId, activePass) {
      this.recipe = recipe;
      this.evalData = evalData;
      this.selectedNodeId = selectedNodeId;
      this.activePass = activePass || 'ALL';
      this.render();
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
        minX = Math.min(minX, x - 24);
        minY = Math.min(minY, y - 34);
        maxX = Math.max(maxX, x + 240);
        maxY = Math.max(maxY, y + 165);
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
      const nextZoom = Math.max(0.14, Math.min(2.2, this.zoom * factor));
      this.panX = mx - (mx - this.panX) * (nextZoom / this.zoom);
      this.panY = my - (my - this.panY) * (nextZoom / this.zoom);
      this.zoom = nextZoom;
      this.applyTransform();
    }

    _portColor(portType) {
      const colors = (this.schema && this.schema.port_colors) || {};
      return colors[portType] || '#94a3b8';
    }

    render() {
      if (!this.recipe) return;
      this.nodesLayer.innerHTML = '';
      this.passLayer.innerHTML = '';
      this._socketEls.clear();
      this._nodeEls.clear();

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
          x1 = Math.max(x1, mx + 234);
          y1 = Math.max(y1, my + (m.collapsed ? 48 : 152));
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

      // 2. Render Nodes
      for (const n of visibleNodes) {
        const meta = this.opMap[n.op] || { op: n.op, category: 'Custom', inputs: [], outputs: [{ name: 'out', port_type: 'RGBA' }] };
        const accent = CAT_COLORS[meta.category] || '#64748b';
        const isOut = n.id === outBase;
        const isSel = n.id === this.selectedNodeId;

        const el = document.createElement('div');
        el.className = 'tg-node' + (isSel ? ' selected' : '') + (isOut ? ' is-output' : '') + (n.bypassed ? ' bypassed' : '') + (n.collapsed ? ' collapsed' : '');
        el.dataset.nodeId = n.id;
        const x = (n.pos && n.pos[0]) || 0;
        const y = (n.pos && n.pos[1]) || 0;
        el.style.left = `${x}px`;
        el.style.top = `${y}px`;
        el.style.setProperty('--node-accent', accent);

        // Header
        const head = document.createElement('div');
        head.className = 'tg-node-head';
        const title = document.createElement('div');
        title.className = 'tg-node-title';
        title.textContent = n.label || n.id;
        title.title = `${n.id} (${n.op})`;

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
          this.callbacks.onSelectNode(n.id, true);
        });
        el.appendChild(head);

        // Body: Inputs | Thumbnail | Outputs
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
          sock.title = `Input ${inp.name} (${pType}) — Alt+Click to sever wire`;
          sock.addEventListener('pointerdown', (e) => {
            if (e.altKey) {
              e.stopPropagation();
              e.preventDefault();
              this.callbacks.onDisconnect(n.id, inp.name);
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
        // Keep Material nodes concise (show primary MATERIAL bus port + top 3 typed channels)
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

        // Footer with op name, live-anim dot, & timing
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
        if (ms !== null && ms !== undefined) {
          msSpan.textContent = ms === 0 ? 'cached · 0ms' : `${ms}ms`;
          if (ms === 0) msSpan.style.color = '#4ade80';
        }
        foot.appendChild(opSpan);
        foot.appendChild(msSpan);
        el.appendChild(foot);

        // Right-click Node Context Menu
        el.addEventListener('contextmenu', (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.callbacks.onSelectNode(n.id, false);
          if (this.callbacks.onNodeContextMenu) {
            this.callbacks.onNodeContextMenu(n.id, e.clientX, e.clientY);
          }
        });

        // Selection + Drag handling
        el.addEventListener('pointerdown', (e) => {
          if (e.target.classList.contains('tg-socket')) return;
          e.stopPropagation();
          this.callbacks.onSelectNode(n.id, false);
          if (e.button === 0) {
            this._dragNode = {
              id: n.id,
              el,
              startClientX: e.clientX,
              startClientY: e.clientY,
              origX: x,
              origY: y,
              moved: false,
            };
          }
        });

        this.nodesLayer.appendChild(el);
        this._nodeEls.set(n.id, el);
      }

      this.renderWires(visibleIds);
      this.applyTransform();
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

          const isRel = this.selectedNodeId && (srcId === this.selectedNodeId || dstNode.id === this.selectedNodeId);
          const g = document.createElementNS(ns, 'g');
          g.setAttribute('class', 'wire-group' + (isRel ? ' wire-active' : ''));

          const dStr = bezierPath(p1.x, p1.y, p2.x, p2.y);
          const hit = document.createElementNS(ns, 'path');
          hit.setAttribute('class', 'wire-hit');
          hit.setAttribute('d', dStr);
          hit.addEventListener('click', (e) => {
            e.stopPropagation();
            this.callbacks.onDisconnect(dstNode.id, inPort);
          });

          const path = document.createElementNS(ns, 'path');
          path.setAttribute('class', 'wire-path');
          path.setAttribute('d', dStr);
          path.setAttribute('stroke', this._portColor(outPortType));

          g.appendChild(hit);
          g.appendChild(path);
          this.wiresSvg.appendChild(g);
        }
      }

      // Live ghost wire when dragging from a socket
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

      // Highlight compatible target sockets, dim incompatible ones!
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

    _clearWireHighlights() {
      for (const sock of this._socketEls.values()) {
        sock.classList.remove('compat-ok', 'compat-dim');
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

      // Draw wires
      ctx.strokeStyle = 'rgba(148,163,184,0.35)';
      ctx.lineWidth = 1;
      for (const n of nodes) {
        const dx = ((n.pos && n.pos[0]) || 0) * sc + ox;
        const dy = ((n.pos && n.pos[1]) || 0) * sc + oy + 5;
        for (const ref of Object.values(n.inputs || {})) {
          const sn = nodes.find((q) => q.id === String(ref).split('.')[0]);
          if (!sn) continue;
          const sx = ((sn.pos && sn.pos[0]) || 0) * sc + ox + 214 * sc;
          const sy = ((sn.pos && sn.pos[1]) || 0) * sc + oy + 5;
          ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(dx, dy); ctx.stroke();
        }
      }

      // Draw simplified node blocks
      for (const n of nodes) {
        const meta = this.opMap[n.op] || {};
        const x = ((n.pos && n.pos[0]) || 0) * sc + ox;
        const y = ((n.pos && n.pos[1]) || 0) * sc + oy;
        ctx.fillStyle = n.id === this.selectedNodeId ? '#3b82f6' : (CAT_COLORS[meta.category] || '#64748b');
        ctx.fillRect(x, y, Math.max(5, 214 * sc), Math.max(4, 110 * sc));
      }

      // Draw viewport frustum
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
      this.stage.addEventListener('pointerdown', (e) => {
        if (e.button === 1 || e.button === 0) {
          this._panning = true;
          this._panStartX = e.clientX;
          this._panStartY = e.clientY;
          this._origPanX = this.panX;
          this._origPanY = this.panY;
          this.stage.style.cursor = 'grabbing';
        }
      });

      this.stage.addEventListener('dblclick', (e) => {
        if (e.target.closest('.tg-node') || e.target.closest('.graph-minimap') || e.target.closest('.graph-zoom-hud')) return;
        const wPos = this.screenToWorld(e.clientX, e.clientY);
        if (this.callbacks.onOpenAddAt) this.callbacks.onOpenAddAt(wPos);
      });

      this.stage.addEventListener('contextmenu', (e) => {
        if (e.target.closest('.tg-node') || e.target.closest('.graph-minimap')) return;
        e.preventDefault();
        const wPos = this.screenToWorld(e.clientX, e.clientY);
        if (this.callbacks.onCanvasContextMenu) {
          this.callbacks.onCanvasContextMenu(e.clientX, e.clientY, wPos);
        }
      });

      window.addEventListener('pointermove', (e) => {
        if (this._wireDrag) {
          const w = this.screenToWorld(e.clientX, e.clientY);
          this._wireDrag.x2 = w.x;
          this._wireDrag.y2 = w.y;
          this.renderWires();
          return;
        }
        if (this._dragNode) {
          const dx = (e.clientX - this._dragNode.startClientX) / this.zoom;
          const dy = (e.clientY - this._dragNode.startClientY) / this.zoom;
          if (Math.hypot(dx, dy) > 2) this._dragNode.moved = true;
          const nx = Math.round((this._dragNode.origX + dx) / 12) * 12;
          const ny = Math.round((this._dragNode.origY + dy) / 12) * 12;
          this._dragNode.el.style.left = `${nx}px`;
          this._dragNode.el.style.top = `${ny}px`;
          const nObj = (this.recipe.nodes || []).find((q) => q.id === this._dragNode.id);
          if (nObj) nObj.pos = [nx, ny];
          this.renderWires();
          this.renderMinimap();
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
          } else {
            const wPos = this.screenToWorld(e.clientX, e.clientY);
            this.callbacks.onDropWireEmpty(drag.srcNode, drag.srcPort, drag.portType, wPos);
          }
          this.renderWires();
          return;
        }
        if (this._dragNode) {
          const dn = this._dragNode;
          this._dragNode = null;
          if (dn.moved) {
            const nObj = (this.recipe.nodes || []).find((q) => q.id === dn.id);
            if (nObj && nObj.pos) {
              this.callbacks.onMoveNode(dn.id, nObj.pos);
            }
          }
        }
        if (this._panning) {
          this._panning = false;
          this.stage.style.cursor = 'default';
        }
      });

      this.stage.addEventListener('wheel', (e) => {
        e.preventDefault();
        const r = this.stage.getBoundingClientRect();
        const mx = e.clientX - r.left;
        const my = e.clientY - r.top;
        const factor = e.deltaY > 0 ? 0.91 : 1.10;
        const nextZoom = Math.max(0.25, Math.min(2.2, this.zoom * factor));
        this.panX = mx - (mx - this.panX) * (nextZoom / this.zoom);
        this.panY = my - (my - this.panY) * (nextZoom / this.zoom);
        this.zoom = nextZoom;
        this.applyTransform();
      }, { passive: false });

      // Interactive click-to-pan minimap
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
