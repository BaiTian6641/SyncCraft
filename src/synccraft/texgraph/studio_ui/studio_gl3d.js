/* ==========================================================================
   SyncCraft Studio v15 — Commercial WebGL 3D Lookdev PBR Viewport (`studio_gl3d.js`)
   Features:
   - Real Blockbench 5 `.bbmodel` meshes + standard Lookdev primitives
   - Double-Sided Two-Pass Translucent Glass/Crystal/Foliage PBR (`gl_FrontFacing`
     flipped normals + back-face -> front-face ordering so BOTH sides render cleanly!)
   - Observable Multi-Tier Studio Floor Grid (Minor + Bright Major + Colored X/Z Axes)
   - Interactive 3D Environment & Light Direction Orbit (`Shift+Drag` or `☀ Light` mode)
     + Configurable `envRotDeg` HDRI Sky Rotation & `lightIntensity` + 3D Sun Ray Gizmo
   - Zero-Lag 30-60 FPS GPU WebGL Animation Frame Ring Buffer (`uploadAnimFrames`)
   - GPU Color-ID 3D Part Picking (`pickPartAt`) & GPU UV-Coord Picking (`pickUvAt`)
     for Direct 3D Brush Mask Painting directly onto the 3D model!
   - Live Shader Split Before/After Channel Wipe (`splitWipe` + `splitPos`)
   - 360° Turntable Frame Capture (`captureTurntableFrames`)
   - Dual-Lobe Clearcoat GGX + Chromatic Refraction + Thickness SSS + WebGL FBO Bloom
   ========================================================================== */
(function (global) {
  'use strict';

  function m4Identity() {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  }
  function m4Perspective(fovRad, aspect, near, far) {
    const f = 1.0 / Math.tan(fovRad * 0.5);
    const nf = 1.0 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ]);
  }
  function m4LookAt(eye, target, up) {
    let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
    let zl = Math.hypot(zx, zy, zz) || 1;
    zx /= zl; zy /= zl; zz /= zl;
    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    let xl = Math.hypot(xx, xy, xz) || 1;
    xx /= xl; xy /= xl; xz /= xl;
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    return new Float32Array([
      xx, yx, zx, 0,
      xy, yy, zy, 0,
      xz, yz, zz, 0,
      -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
      -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
      -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
      1,
    ]);
  }
  function m4Mul(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return out;
  }

  // --------------------------------------------------------------------------
  // Safe Zero-eval Pure-JS Molang Expression & Blockbench 5.0 Skeletal Evaluator
  // --------------------------------------------------------------------------
  const _NUM_RE = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i;

  function evalMolang(val, animTime) {
    if (typeof val === 'number') return Number.isFinite(val) ? val : 0;
    if (val == null) return 0;
    const s = String(val).trim();
    if (!s) return 0;
    if (_NUM_RE.test(s)) return parseFloat(s);
    // Tokenize safe subset of Molang math expressions (degrees for trig!)
    const src = s.toLowerCase();
    let idx = 0;
    const len = src.length;
    const skipWs = () => { while (idx < len && src.charCodeAt(idx) <= 32) idx++; };

    const parseExpr = () => parseTernary();
    const parseTernary = () => {
      let c = parseCompare();
      skipWs();
      if (idx < len && src[idx] === '?') {
        idx++;
        const a = parseExpr();
        skipWs();
        if (idx < len && src[idx] === ':') idx++;
        const b = parseExpr();
        return c ? a : b;
      }
      return c;
    };
    const parseCompare = () => {
      let left = parseAddSub();
      while (true) {
        skipWs();
        const two = src.slice(idx, idx + 2);
        if (two === '<=' || two === '>=' || two === '==' || two === '!=') {
          idx += 2;
          const r = parseAddSub();
          if (two === '<=') left = left <= r ? 1 : 0;
          else if (two === '>=') left = left >= r ? 1 : 0;
          else if (two === '==') left = Math.abs(left - r) < 1e-6 ? 1 : 0;
          else left = Math.abs(left - r) >= 1e-6 ? 1 : 0;
        } else if (src[idx] === '<' || src[idx] === '>') {
          const op = src[idx++];
          const r = parseAddSub();
          left = op === '<' ? (left < r ? 1 : 0) : (left > r ? 1 : 0);
        } else break;
      }
      return left;
    };
    const parseAddSub = () => {
      let left = parseMulDiv();
      while (true) {
        skipWs();
        const ch = src[idx];
        if (ch === '+' || ch === '-') {
          idx++;
          const r = parseMulDiv();
          left = ch === '+' ? left + r : left - r;
        } else break;
      }
      return left;
    };
    const parseMulDiv = () => {
      let left = parseUnary();
      while (true) {
        skipWs();
        const ch = src[idx];
        if (ch === '*' || ch === '/') {
          idx++;
          const r = parseUnary();
          left = ch === '*' ? left * r : (Math.abs(r) > 1e-8 ? left / r : 0);
        } else break;
      }
      return left;
    };
    const parseUnary = () => {
      skipWs();
      const ch = src[idx];
      if (ch === '+') { idx++; return parseUnary(); }
      if (ch === '-') { idx++; return -parseUnary(); }
      if (ch === '!') { idx++; return parseUnary() ? 0 : 1; }
      return parsePrimary();
    };
    const parsePrimary = () => {
      skipWs();
      if (idx >= len) return 0;
      if (src[idx] === '(') {
        idx++;
        const v = parseExpr();
        skipWs();
        if (src[idx] === ')') idx++;
        return v;
      }
      // Number literal
      const start = idx;
      if ((src[idx] >= '0' && src[idx] <= '9') || src[idx] === '.') {
        while (idx < len && ((src[idx] >= '0' && src[idx] <= '9') || src[idx] === '.')) idx++;
        if (idx < len && (src[idx] === 'e' || src[idx] === 'E')) {
          idx++;
          if (src[idx] === '+' || src[idx] === '-') idx++;
          while (idx < len && src[idx] >= '0' && src[idx] <= '9') idx++;
        }
        return parseFloat(src.slice(start, idx)) || 0;
      }
      // Identifier / function call (query.anim_time, math.sin(...), etc.)
      while (idx < len && /[a-z0-9_.]/.test(src[idx])) idx++;
      const ident = src.slice(start, idx);
      skipWs();
      if (idx < len && src[idx] === '(') {
        idx++;
        const args = [];
        skipWs();
        if (src[idx] !== ')') {
          while (idx < len) {
            args.push(parseExpr());
            skipWs();
            if (src[idx] === ',') { idx++; continue; }
            break;
          }
        }
        if (src[idx] === ')') idx++;
        const a0 = args[0] || 0, a1 = args[1] || 0, a2 = args[2] || 0;
        const D2R = Math.PI / 180.0;
        const R2D = 180.0 / Math.PI;
        switch (ident) {
          case 'math.sin': case 'sin': return Math.sin(a0 * D2R);
          case 'math.cos': case 'cos': return Math.cos(a0 * D2R);
          case 'math.tan': case 'tan': return Math.tan(a0 * D2R);
          case 'math.asin': return Math.asin(Math.max(-1, Math.min(1, a0))) * R2D;
          case 'math.acos': return Math.acos(Math.max(-1, Math.min(1, a0))) * R2D;
          case 'math.atan': return Math.atan(a0) * R2D;
          case 'math.atan2': return Math.atan2(a0, a1) * R2D;
          case 'math.abs': case 'abs': return Math.abs(a0);
          case 'math.floor': return Math.floor(a0);
          case 'math.ceil': return Math.ceil(a0);
          case 'math.round': return Math.round(a0);
          case 'math.sqrt': return Math.sqrt(Math.max(0, a0));
          case 'math.pow': return Math.pow(a0, a1);
          case 'math.mod': return a1 !== 0 ? a0 % a1 : 0;
          case 'math.min': return Math.min(a0, a1);
          case 'math.max': return Math.max(a0, a1);
          case 'math.clamp': return Math.max(a1, Math.min(a2, a0));
          case 'math.lerp': return a0 + (a1 - a0) * a2;
          case 'math.hermite_blend': { const u = Math.max(0, Math.min(1, a0)); return u * u * (3 - 2 * u); }
          default: return 0;
        }
      }
      if (ident === 'query.anim_time' || ident === 'q.anim_time' || ident === 'query.life_time' || ident === 'q.life_time' || ident === 'time') {
        return animTime;
      }
      if (ident === 'math.pi' || ident === 'pi') return Math.PI;
      return 0;
    };

    try {
      const res = parseExpr();
      return Number.isFinite(res) ? res : 0;
    } catch (_) {
      return 0;
    }
  }

  function _evalKfVec(kf, animTime, usePost) {
    const pts = kf.p;
    const raw = (usePost && pts.length > 1) ? pts[1] : pts[0];
    return [evalMolang(raw[0], animTime), evalMolang(raw[1], animTime), evalMolang(raw[2], animTime)];
  }

  function evalBoneChannel(kfs, t, length, loopMode, isScale) {
    const def = isScale ? [1, 1, 1] : [0, 0, 0];
    if (!kfs || !kfs.length) return def;
    const n = kfs.length;
    if (n === 1) return _evalKfVec(kfs[0], t, false);
    if (t <= kfs[0].t) return _evalKfVec(kfs[0], t, false);
    if (t >= kfs[n - 1].t) {
      const last = kfs[n - 1];
      return _evalKfVec(last, t, last.p.length > 1);
    }
    let i1 = 1;
    while (i1 < n - 1 && kfs[i1].t < t) i1++;
    const i0 = i1 - 1;
    const k0 = kfs[i0];
    const k1 = kfs[i1];
    const dt = Math.max(1e-6, k1.t - k0.t);
    const u = Math.max(0, Math.min(1, (t - k0.t) / dt));
    const v0 = _evalKfVec(k0, t, k0.p.length > 1);
    if (k0.i === 'step') return v0;
    const v1 = _evalKfVec(k1, t, false);
    if (k0.i === 'catmullrom' || k1.i === 'catmullrom') {
      const im = i0 > 0 ? i0 - 1 : (loopMode === 'loop' ? n - 1 : 0);
      const ip = i1 < n - 1 ? i1 + 1 : (loopMode === 'loop' ? 0 : n - 1);
      const vm = _evalKfVec(kfs[im], t, false);
      const vp = _evalKfVec(kfs[ip], t, false);
      const u2 = u * u, u3 = u2 * u;
      const out = [0, 0, 0];
      for (let c = 0; c < 3; c++) {
        out[c] = 0.5 * (
          (2 * v0[c]) +
          (-vm[c] + v1[c]) * u +
          (2 * vm[c] - 5 * v0[c] + 4 * v1[c] - vp[c]) * u2 +
          (-vm[c] + 3 * v0[c] - 3 * v1[c] + vp[c]) * u3
        );
      }
      return out;
    }
    return [
      v0[0] + (v1[0] - v0[0]) * u,
      v0[1] + (v1[1] - v0[1]) * u,
      v0[2] + (v1[2] - v0[2]) * u,
    ];
  }

  /** Build 4x4 column-major local bone matrix T(tx,ty,tz) * Rz(rz) * Ry(ry) * Rx(rx) * S(sx,sy,sz). */
  function m4BoneLocal(tx, ty, tz, rxDeg, ryDeg, rzDeg, sx, sy, sz) {
    const D2R = Math.PI / 180.0;
    const rx = rxDeg * D2R, ry = ryDeg * D2R, rz = rzDeg * D2R;
    const cx = Math.cos(rx), sxr = Math.sin(rx);
    const cy = Math.cos(ry), syr = Math.sin(ry);
    const cz = Math.cos(rz), szr = Math.sin(rz);
    // R = Rz * Ry * Rx (matches Python _euler_xyz_mat exactly!)
    const r00 = cz * cy;
    const r01 = cz * syr * sxr - szr * cx;
    const r02 = cz * syr * cx + szr * sxr;
    const r10 = szr * cy;
    const r11 = szr * syr * sxr + cz * cx;
    const r12 = szr * syr * cx - cz * sxr;
    const r20 = -syr;
    const r21 = cy * sxr;
    const r22 = cy * cx;
    // Column-major storage for m4Mul
    return new Float32Array([
      r00 * sx, r10 * sx, r20 * sx, 0,
      r01 * sy, r11 * sy, r21 * sy, 0,
      r02 * sz, r12 * sz, r22 * sz, 0,
      tx,       ty,       tz,       1,
    ]);
  }

  function buildBoxGeom(sx, sy, sz, uvScale) {
    const hx = sx * 0.5, hy = sy * 0.5, hz = sz * 0.5;
    const us = uvScale || 1.0;
    const faces = [
      { n:[0,0,1], t:[1,0,0], p:[[-hx,hy,hz],[hx,hy,hz],[hx,-hy,hz],[-hx,-hy,hz]] },
      { n:[0,0,-1], t:[-1,0,0], p:[[hx,hy,-hz],[-hx,hy,-hz],[-hx,-hy,-hz],[hx,-hy,-hz]] },
      { n:[1,0,0], t:[0,0,-1], p:[[hx,hy,hz],[hx,hy,-hz],[hx,-hy,-hz],[hx,-hy,hz]] },
      { n:[-1,0,0], t:[0,0,1], p:[[-hx,hy,-hz],[-hx,hy,hz],[-hx,-hy,hz],[-hx,-hy,-hz]] },
      { n:[0,1,0], t:[1,0,0], p:[[-hx,hy,-hz],[hx,hy,-hz],[hx,hy,hz],[-hx,hy,hz]] },
      { n:[0,-1,0], t:[1,0,0], p:[[-hx,-hy,hz],[hx,-hy,hz],[hx,-hy,-hz],[-hx,-hy,-hz]] },
    ];
    const pos = [], nrm = [], tan = [], uv = [], pick = [];
    const uvQuad = [[0,0],[us,0],[us,us],[0,us]];
    const idx = [0, 3, 2, 0, 2, 1];
    for (const f of faces) {
      for (const i of idx) {
        pos.push(f.p[i][0], f.p[i][1], f.p[i][2]);
        nrm.push(f.n[0], f.n[1], f.n[2]);
        tan.push(f.t[0], f.t[1], f.t[2]);
        uv.push(uvQuad[i][0], uvQuad[i][1]);
        pick.push(0);
      }
    }
    return { positions: pos, normals: nrm, tangents: tan, uvs: uv, pick_ids: pick, count: pos.length / 3 };
  }

  function buildSphereGeom(radius, segU, segV) {
    const pos = [], nrm = [], tan = [], uv = [], pick = [];
    for (let y = 0; y < segV; y++) {
      const v0 = y / segV, v1 = (y + 1) / segV;
      const th0 = v0 * Math.PI, th1 = v1 * Math.PI;
      for (let x = 0; x < segU; x++) {
        const u0 = x / segU, u1 = (x + 1) / segU;
        const ph0 = u0 * Math.PI * 2, ph1 = u1 * Math.PI * 2;
        const pt = (th, ph) => [
          Math.sin(th) * Math.cos(ph),
          Math.cos(th),
          Math.sin(th) * Math.sin(ph),
        ];
        const tn = (ph) => [-Math.sin(ph), 0, Math.cos(ph)];
        const verts = [
          [pt(th0, ph0), tn(ph0), [u0 * 2, v0]],
          [pt(th1, ph0), tn(ph0), [u0 * 2, v1]],
          [pt(th1, ph1), tn(ph1), [u1 * 2, v1]],
          [pt(th0, ph0), tn(ph0), [u0 * 2, v0]],
          [pt(th1, ph1), tn(ph1), [u1 * 2, v1]],
          [pt(th0, ph1), tn(ph1), [u1 * 2, v0]],
        ];
        for (const [p, t, u] of verts) {
          pos.push(p[0] * radius, p[1] * radius, p[2] * radius);
          nrm.push(p[0], p[1], p[2]);
          tan.push(t[0], t[1], t[2]);
          uv.push(u[0], u[1]);
          pick.push(0);
        }
      }
    }
    return { positions: pos, normals: nrm, tangents: tan, uvs: uv, pick_ids: pick, count: pos.length / 3 };
  }

  /** Observable 3-Tier Architectural Studio Grid: Minor + Bright Major + Colored X/Z Center Axes! */
  function buildLayeredGrid(halfExtent, minorStep, majorEvery, yLevel) {
    const minor = [], major = [];
    const steps = Math.round(halfExtent / minorStep);
    for (let i = -steps; i <= steps; i++) {
      if (i === 0) continue; // Center axes drawn with dedicated X/Z colors
      const v = i * minorStep;
      const target = (Math.abs(i) % majorEvery === 0) ? major : minor;
      target.push(-halfExtent, yLevel, v, halfExtent, yLevel, v);
      target.push(v, yLevel, -halfExtent, v, yLevel, halfExtent);
    }
    const axisX = [-halfExtent, yLevel + 0.002, 0, halfExtent, yLevel + 0.002, 0];
    const axisZ = [0, yLevel + 0.002, -halfExtent, 0, yLevel + 0.002, halfExtent];
    return {
      minor: new Float32Array(minor),
      major: new Float32Array(major),
      axisX: new Float32Array(axisX),
      axisZ: new Float32Array(axisZ),
    };
  }

  // Analytical Rotatable Studio HDRI Environment Function shared by Sky Backdrop & IBL PBR!
  const GLSL_HDRI_ENV = `
    vec3 rotYVec(vec3 v, float ang) {
      float c = cos(ang), s = sin(ang);
      return vec3(v.x * c - v.z * s, v.y, v.x * s + v.z * c);
    }
    vec3 sampleHdriEnv(vec3 dir, float rough, vec3 skyCol, vec3 groundCol, vec3 keyCol, vec3 accentCol, vec3 lightDir, float envRot) {
      vec3 d = normalize(rotYVec(dir, envRot));
      float h = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
      vec3 base = mix(groundCol, skyCol, smoothstep(0.32, 0.88, h));
      float horiz = exp(-abs(d.y) * mix(9.0, 2.2, rough));
      base += mix(skyCol, accentCol, 0.5) * horiz * 0.34;
      float topBox = smoothstep(mix(0.84, 0.45, rough), mix(0.95, 0.85, rough), d.y);
      vec3 l1 = normalize(rotYVec(lightDir, envRot));
      vec3 l2 = normalize(vec3(-l1.x, max(0.25, l1.y * 0.6), -l1.z));
      float keySpot = pow(max(dot(d, l1), 0.0), mix(42.0, 3.5, rough));
      float rimSpot = pow(max(dot(d, l2), 0.0), mix(18.0, 2.5, rough));
      return base + vec3(1.0) * topBox * mix(0.65, 0.20, rough)
                  + keyCol * keySpot * mix(1.85, 0.45, rough)
                  + accentCol * rimSpot * mix(0.95, 0.25, rough);
    }
  `;

  const VS_PBR = `
    attribute vec3 aPos;
    attribute vec3 aNormal;
    attribute vec3 aTangent;
    attribute vec2 aUv;
    attribute float aPickId;
    uniform mat4 uMvp;
    uniform mat4 uModel;
    varying vec3 vWorldPos;
    varying vec3 vNormal;
    varying vec3 vTangent;
    varying vec2 vUv;
    varying float vPickId;
    void main() {
      vec4 wp = uModel * vec4(aPos, 1.0);
      vWorldPos = wp.xyz;
      vNormal = normalize(mat3(uModel) * aNormal);
      vTangent = normalize(mat3(uModel) * aTangent);
      vUv = aUv;
      vPickId = aPickId;
      gl_Position = uMvp * vec4(aPos, 1.0);
    }
  `;

  const FS_PBR = `
    precision mediump float;
    varying vec3 vWorldPos;
    varying vec3 vNormal;
    varying vec3 vTangent;
    varying vec2 vUv;
    varying float vPickId;

    uniform sampler2D uAlbedo;
    uniform sampler2D uNormal;
    uniform sampler2D uOrm;
    uniform sampler2D uShadow;
    uniform sampler2D uEmissive;
    uniform sampler2D uDisplay;
    uniform sampler2D uThickness;
    uniform sampler2D uSplitRight;

    uniform vec3 uCamPos;
    uniform vec3 uLightDir;
    uniform vec3 uKeyColor;
    uniform vec3 uSkyColor;
    uniform vec3 uGroundColor;
    uniform vec3 uAccentColor;
    uniform int uDirectMode;
    uniform int uHasNormal;
    uniform float uClearcoat;
    uniform float uSssStrength;
    uniform float uHighlightPickId;
    uniform float uEnvRot;
    uniform float uLightIntensity;
    uniform int uSplitWipe;
    uniform float uSplitPos;
    uniform float uViewportW;
    uniform int uBackPass;

    const float PI = 3.14159265;

    ${GLSL_HDRI_ENV}

    float D_GGX(float ndh, float r) {
      float a = r * r;
      float a2 = a * a;
      float d = (ndh * ndh) * (a2 - 1.0) + 1.0;
      return a2 / max(0.0005, PI * d * d);
    }

    vec3 F_Schlick(float cosT, vec3 f0) {
      return f0 + (1.0 - f0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
    }

    void main() {
      vec4 baseSample = texture2D(uAlbedo, vUv);
      if (baseSample.a < 0.03) discard;
      if (uBackPass == 1 && baseSample.a >= 0.97) discard;

      float isPicked = (uHighlightPickId > 0.5 && abs(vPickId - uHighlightPickId) < 0.4) ? 1.0 : 0.0;

      // Split Before/After Channel Wipe Right Side
      if (uSplitWipe == 1) {
        float sx = gl_FragCoord.x / max(1.0, uViewportW);
        if (abs(sx - uSplitPos) < 0.003) {
          gl_FragColor = vec4(0.22, 0.88, 1.0, 1.0);
          return;
        }
        if (sx > uSplitPos) {
          vec4 sp = texture2D(uSplitRight, vUv);
          gl_FragColor = vec4(sp.rgb, clamp(baseSample.a, 0.14, 1.0));
          return;
        }
      }

      if (uDirectMode == 1) {
        vec4 disp = texture2D(uDisplay, vUv);
        vec3 dc = mix(disp.rgb, vec3(0.22, 0.88, 1.0), isPicked * 0.38);
        gl_FragColor = vec4(dc, clamp(disp.a, 0.08, 1.0));
        return;
      }

      // True Double-Sided Normal Flip so BOTH sides of transparent Glass/Crystal/Foliage light identically!
      float faceSign = gl_FrontFacing ? 1.0 : -1.0;
      vec3 N = normalize(vNormal) * faceSign;
      if (uHasNormal == 1) {
        vec3 T = normalize(vTangent - N * dot(vTangent, N));
        vec3 B = normalize(cross(N, T));
        vec3 nMap = texture2D(uNormal, vUv).xyz * 2.0 - 1.0;
        N = normalize(T * (nMap.x * faceSign) + B * nMap.y + N * max(0.15, nMap.z));
      }

      vec3 alb = pow(baseSample.rgb, vec3(2.2));
      vec3 orm = texture2D(uOrm, vUv).rgb;
      float ao = clamp(orm.r, 0.15, 1.0);
      float rough = clamp(orm.g, 0.06, 0.96);
      float metal = clamp(orm.b, 0.0, 1.0);
      float sh = clamp(texture2D(uShadow, vUv).r, 0.12, 1.0);
      float thick = clamp(texture2D(uThickness, vUv).r, 0.05, 1.0);
      vec4 em = texture2D(uEmissive, vUv);

      vec3 V = normalize(uCamPos - vWorldPos);
      vec3 L = normalize(uLightDir);
      vec3 H = normalize(V + L);
      vec3 R = reflect(-V, N);

      float ndl = max(dot(N, L), 0.0);
      float ndv = max(dot(N, V), 0.001);
      float ndh = max(dot(N, H), 0.0);
      float hdv = max(dot(H, V), 0.0);

      vec3 f0 = mix(vec3(0.04), alb, metal);
      vec3 F = F_Schlick(hdv, f0);
      float D = D_GGX(ndh, rough);
      float G = min(1.0, min(2.0 * ndh * ndv / max(hdv, 0.001), 2.0 * ndh * ndl / max(hdv, 0.001)));
      vec3 spec = (D * G * F) / max(0.04, 4.0 * ndv * ndl);
      vec3 kD = (1.0 - F) * (1.0 - metal);

      float vis = mix(0.28, 1.0, sh);
      float lInt = max(0.15, uLightIntensity);
      vec3 direct = (kD * alb / PI + spec * 0.72) * uKeyColor * ndl * vis * 2.35 * lInt;

      // Dual-Lobe Clearcoat GGX Specular
      float coatW = clamp(uClearcoat, 0.0, 1.0);
      float D_coat = D_GGX(ndh, 0.08);
      float F_coat = 0.04 + 0.96 * pow(1.0 - hdv, 5.0);
      vec3 coatDirect = vec3(D_coat * F_coat * 0.34) * uKeyColor * ndl * vis * coatW * lInt;

      // Rotatable Procedural Studio HDRI Image-Based Lighting
      vec3 envIrrad = sampleHdriEnv(N, 0.92, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L, uEnvRot) * lInt;
      vec3 envSpecCol = sampleHdriEnv(R, rough, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L, uEnvRot) * lInt;
      vec3 envCoatCol = sampleHdriEnv(R, 0.07, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L, uEnvRot) * lInt;
      vec3 envDiff = envIrrad * alb * (1.0 - metal * 0.72) * ao * 0.56;
      vec3 envSpec = envSpecCol * F * (1.0 - rough * 0.65) * ao * mix(0.45, 1.15, metal);
      vec3 envCoat = envCoatCol * (0.04 + 0.96 * pow(1.0 - ndv, 5.0)) * coatW * ao * 0.45;

      // Backlit Subsurface Scattering (SSS) via Ray-Traced Thickness
      float backLight = pow(clamp(dot(V, -L + N * 0.45), 0.0, 1.0), 2.8);
      float wrapDiff = clamp((dot(N, L) + 0.45) / 1.45, 0.0, 1.0);
      float thinness = clamp(1.0 - thick * 0.85, 0.1, 1.0);
      vec3 sssCol = alb * uKeyColor * (backLight * 0.85 + wrapDiff * 0.25) * thinness * (1.0 - metal) * uSssStrength * lInt;

      // Double-Sided Translucent Glass & Crystal Chromatic Refraction + Beer-Lambert Tint
      float fresnelRim = pow(clamp(1.0 - ndv, 0.0, 1.0), 2.8);
      float isTranslucent = step(baseSample.a, 0.97);
      vec3 refrCol = vec3(0.0);
      if (isTranslucent > 0.5) {
        vec3 rR = refract(-V, N, 1.0 / 1.45);
        vec3 rG = refract(-V, N, 1.0 / 1.49);
        vec3 rB = refract(-V, N, 1.0 / 1.54);
        float cR = sampleHdriEnv(rR, rough * 0.65, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L, uEnvRot).r;
        float cG = sampleHdriEnv(rG, rough * 0.65, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L, uEnvRot).g;
        float cB = sampleHdriEnv(rB, rough * 0.65, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L, uEnvRot).b;
        vec3 gemTint = mix(alb, pow(max(alb, vec3(0.02)), vec3(1.35)) * 1.15, 0.55);
        refrCol = vec3(cR, cG, cB) * gemTint * 0.68 * lInt + envSpecCol * mix(vec3(1.0), gemTint, 0.45) * fresnelRim * 0.52;
      }

      vec3 emissiveTerm = pow(em.rgb, vec3(2.2)) * em.a * (isTranslucent > 0.5 ? 1.15 : 1.95);
      vec3 opaqueCol = direct + coatDirect + envDiff + envSpec + envCoat + sssCol;
      vec3 glassCol = direct * 0.48 + coatDirect * 0.72 + envDiff * 0.36 + envSpec * 0.78 + sssCol * 0.55 + refrCol;
      vec3 col = mix(opaqueCol, glassCol, isTranslucent) + emissiveTerm;
      if (!gl_FrontFacing && isTranslucent > 0.5) {
        col *= 0.74;
      }
      col += vec3(0.18, 0.86, 1.0) * (0.28 + 0.55 * fresnelRim) * isPicked;

      // ACES Filmic tonemap + sRGB encode
      col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
      col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
      float outAlpha = mix(1.0, clamp(baseSample.a * (gl_FrontFacing ? 0.92 : 0.72) + fresnelRim * 0.22, 0.20, 0.94), isTranslucent);
      gl_FragColor = vec4(col, outAlpha);
    }
  `;

  // Blurred Rotatable Studio HDRI Sky Backdrop Shader
  const VS_QUAD = `
    attribute vec2 aQuad;
    varying vec2 vUv;
    void main() {
      vUv = aQuad * 0.5 + 0.5;
      gl_Position = vec4(aQuad, 0.999, 1.0);
    }
  `;
  const FS_BG = `
    precision mediump float;
    varying vec2 vUv;
    uniform float uYaw;
    uniform float uPitch;
    uniform float uEnvRot;
    uniform float uLightIntensity;
    uniform vec3 uKeyColor;
    uniform vec3 uSkyColor;
    uniform vec3 uGroundColor;
    uniform vec3 uAccentColor;
    uniform vec3 uLightDir;
    uniform int uShowBg;
    ${GLSL_HDRI_ENV}
    void main() {
      if (uShowBg == 0) {
        vec3 base = mix(vec3(0.086, 0.090, 0.100), vec3(0.126, 0.132, 0.146), vUv.y);
        gl_FragColor = vec4(base, 1.0);
        return;
      }
      vec2 ndc = vUv * 2.0 - 1.0;
      float cy = cos(uYaw), sy = sin(uYaw);
      float cp = cos(uPitch), sp = sin(uPitch);
      vec3 fwd = normalize(vec3(-cp * sy, -sp, -cp * cy));
      vec3 rgt = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
      vec3 up  = normalize(cross(rgt, fwd));
      vec3 dir = normalize(fwd * 1.65 + rgt * ndc.x * 1.1 + up * ndc.y * 0.85);
      vec3 env = sampleHdriEnv(dir, 0.72, uSkyColor, uGroundColor, uKeyColor, uAccentColor, uLightDir, uEnvRot) * 0.34 * max(0.3, uLightIntensity);
      env = pow(clamp(env, 0.0, 1.0), vec3(1.0 / 2.2));
      gl_FragColor = vec4(env, 1.0);
    }
  `;

  // Post-Process Screen-Space Bloom Shader
  const FS_BLOOM = `
    precision mediump float;
    varying vec2 vUv;
    uniform sampler2D uScene;
    uniform vec2 uTexel;
    uniform float uBloom;
    void main() {
      vec4 base = texture2D(uScene, vUv);
      if (uBloom <= 0.005) {
        gl_FragColor = vec4(base.rgb, 1.0);
        return;
      }
      vec3 glow = vec3(0.0);
      float wSum = 0.0;
      for (int dx = -2; dx <= 2; dx++) {
        for (int dy = -2; dy <= 2; dy++) {
          vec2 off = vec2(float(dx), float(dy)) * uTexel * 2.4;
          vec3 s = texture2D(uScene, clamp(vUv + off, 0.002, 0.998)).rgb;
          float lum = max(s.r, max(s.g, s.b));
          float keep = smoothstep(0.66, 0.98, lum);
          float w = 1.0 / (1.0 + float(dx * dx + dy * dy) * 0.55);
          glow += s * keep * w;
          wSum += w;
        }
      }
      glow /= max(0.001, wSum);
      vec3 col = clamp(base.rgb + glow * uBloom * 0.95, 0.0, 1.0);
      gl_FragColor = vec4(col, 1.0);
    }
  `;

  // GPU Color-ID Picking Shader
  const VS_PICK = `
    attribute vec3 aPos;
    attribute float aPickId;
    uniform mat4 uMvp;
    varying float vPickId;
    void main() {
      vPickId = aPickId;
      gl_Position = uMvp * vec4(aPos, 1.0);
    }
  `;
  const FS_PICK = `
    precision mediump float;
    varying float vPickId;
    void main() {
      float id = floor(vPickId + 0.5);
      float lo = mod(id, 256.0) / 255.0;
      float hi = floor(id / 256.0) / 255.0;
      gl_FragColor = vec4(lo, hi, 0.0, 1.0);
    }
  `;

  // GPU UV-Coordinate Picking Shader for Direct 3D Brush Painting!
  const VS_UV_PICK = `
    attribute vec3 aPos;
    attribute vec2 aUv;
    uniform mat4 uMvp;
    varying vec2 vUv;
    void main() {
      vUv = aUv;
      gl_Position = uMvp * vec4(aPos, 1.0);
    }
  `;
  const FS_UV_PICK = `
    precision mediump float;
    varying vec2 vUv;
    void main() {
      gl_FragColor = vec4(fract(vUv.x), fract(vUv.y), 1.0, 1.0);
    }
  `;

  const VS_LINE = `
    attribute vec3 aPos;
    uniform mat4 uMvp;
    void main() { gl_Position = uMvp * vec4(aPos, 1.0); }
  `;
  const FS_LINE = `
    precision mediump float;
    uniform vec4 uColor;
    void main() { gl_FragColor = uColor; }
  `;

  const LIGHTING_RIGS = {
    studio_hero:      { key: [1.0, 0.96, 0.88], sky: [0.56, 0.72, 0.96], ground: [0.18, 0.17, 0.23], accent: [0.46, 0.78, 1.0] },
    dramatic_rim:     { key: [1.0, 0.78, 0.52], sky: [0.30, 0.48, 0.92], ground: [0.14, 0.10, 0.18], accent: [1.0, 0.55, 0.25] },
    sunset_forge:     { key: [1.0, 0.68, 0.36], sky: [0.76, 0.42, 0.54], ground: [0.24, 0.13, 0.10], accent: [1.0, 0.82, 0.44] },
    cyber_neon:       { key: [0.48, 0.94, 1.0], sky: [0.38, 0.18, 0.68], ground: [0.09, 0.10, 0.20], accent: [0.95, 0.28, 0.78] },
    high_noon:        { key: [1.0, 1.0, 0.96],  sky: [0.74, 0.86, 1.0],  ground: [0.30, 0.29, 0.27], accent: [0.85, 0.93, 1.0] },
    moonlit_obsidian: { key: [0.62, 0.78, 1.0], sky: [0.22, 0.32, 0.58], ground: [0.08, 0.09, 0.16], accent: [0.68, 0.44, 1.0] },
  };

  class StudioGL3DViewport {
    constructor(canvas, hudEl) {
      this.canvas = canvas;
      this.hudEl = hudEl;
      this.yaw = -0.38;
      this.pitch = 0.24;
      this.dist = 3.55;
      this.spin = false;
      this.wireframe = false;
      this.showEnvBg = true;
      this.clearcoat = 0.25;
      this.bloomStrength = 0.45;
      this.sssStrength = 0.38;
      this.envRotDeg = 0.0;
      this.lightIntensity = 1.0;
      this.lightDragMode = false;
      this.pickMode = false;
      this.brushMode = false;
      this.decalMode = false;
      this.showBones = false;
      this.splitWipe = false;
      this.splitPos = 0.5;
      this.splitRightChannel = 'normal';
      this.hasTransparency = false;
      this.highlightPickId = 0;

      this.onPickPart = null;
      this.onHoverPart = null;
      this.onBrush3d = null;
      this.onDecalStamp = null;
      this.onLightChange = null;
      this.onSkelTimeUpdate = null;

      // Live Blockbench 5.0 Skeletal Animation State
      this.bones = [];
      this.animations = [];
      this.skelClipIdx = -1;
      this.skelTime = 0.0;
      this.skelPlaying = false;
      this.skelSpeed = 1.0;
      this._boneLineCount = 0;
      this._boneJointCount = 0;
      this._bbBounds = null;
      this._bbScale = 1.0;
      this._skinPos = null;
      this._skinNrm = null;
      this._skinTan = null;
      this._lastSkelHudPush = 0;

      this.primitive = 'bbmodel';
      this.channel = 'pbr_lit';
      this.lightAz = -38;
      this.lightEl = 52;
      this.lightingPreset = 'studio_hero';
      this.hasBBModel = false;
      this.bbMeshRaw = null;
      this.elemMeta = [];
      this.fallbackImg = null;
      this._fboW = 0;
      this._fboH = 0;
      this._lastMvp = null;
      this._lightDragging = false;
      this._brushDragging = false;

      // GPU WebGL Ring Buffer for Zero-Lag 30-60 FPS Animation Playback!
      this.animRing = [];
      this.activeAnimIdx = -1;

      this.gl = canvas.getContext('webgl', { antialias: true, alpha: false, preserveDrawingBuffer: true }) ||
                canvas.getContext('experimental-webgl', { antialias: true, alpha: false, preserveDrawingBuffer: true });
      this.isWebGL = !!this.gl;

      if (this.isWebGL) {
        this._initGL();
      }
      this._bindInteraction();
      this._resize();
      window.addEventListener('resize', () => this._resize());
      this._lastT = performance.now();
      const loop = (t) => {
        const dt = Math.min(0.05, (t - this._lastT) * 0.001);
        this._lastT = t;
        if (this.spin && !this._dragging && !this._lightDragging && !this._brushDragging) {
          this.yaw += dt * 0.36;
        }
        if (this.skelPlaying && this.hasBBModel && this.skelClipIdx >= 0 && this.skelClipIdx < this.animations.length) {
          const clip = this.animations[this.skelClipIdx];
          const len = Math.max(0.05, clip.length || 1.0);
          let nextT = this.skelTime + dt * (this.skelSpeed || 1.0);
          if (clip.loop === 'loop' || clip.loop === true) {
            nextT = ((nextT % len) + len) % len;
          } else if (nextT >= len) {
            nextT = len;
            this.skelPlaying = false;
          }
          this.skelTime = nextT;
          this._updateSkelPose(nextT);
          if (this.onSkelTimeUpdate && (t - this._lastSkelHudPush) > 45) {
            this._lastSkelHudPush = t;
            this.onSkelTimeUpdate(this.skelTime, len, this.skelPlaying);
          }
        }
        this.render();
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }

    _compile(type, src) {
      const gl = this.gl;
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        console.error('Shader error:', gl.getShaderInfoLog(sh));
      }
      return sh;
    }

    _link(vsSrc, fsSrc) {
      const gl = this.gl;
      const prog = gl.createProgram();
      gl.attachShader(prog, this._compile(gl.VERTEX_SHADER, vsSrc));
      gl.attachShader(prog, this._compile(gl.FRAGMENT_SHADER, fsSrc));
      gl.linkProgram(prog);
      return prog;
    }

    _makeSolidTex(rgba) {
      const gl = this.gl;
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(rgba));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      return tex;
    }

    _uploadGeom(g) {
      const gl = this.gl;
      const mk = (arr) => {
        const b = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW);
        return b;
      };
      const picks = g.pick_ids && g.pick_ids.length === g.count ? g.pick_ids : new Float32Array(g.count);
      return {
        pos: mk(g.positions),
        nrm: mk(g.normals),
        tan: mk(g.tangents),
        uv: mk(g.uvs),
        pick: mk(picks),
        count: g.count,
      };
    }

    _uploadGridBuffers(yLevel) {
      const gl = this.gl;
      const g = buildLayeredGrid(2.8, 0.35, 2, yLevel);
      const mk = (oldBuf, arr) => {
        const b = oldBuf || gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, b);
        gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
        return { buf: b, count: arr.length / 3 };
      };
      this.gridLayers = {
        minor: mk(this.gridLayers && this.gridLayers.minor.buf, g.minor),
        major: mk(this.gridLayers && this.gridLayers.major.buf, g.major),
        axisX: mk(this.gridLayers && this.gridLayers.axisX.buf, g.axisX),
        axisZ: mk(this.gridLayers && this.gridLayers.axisZ.buf, g.axisZ),
      };
      this.gridY = yLevel;
    }

    _ensureFBOs(w, h) {
      if (!this.isWebGL || (this._fboW === w && this._fboH === h && this.fboScene)) return;
      const gl = this.gl;
      this._fboW = w;
      this._fboH = h;
      const mkFbo = () => {
        const fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        const rb = gl.createRenderbuffer();
        gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb);
        const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return ok ? { fb, tex, rb } : null;
      };
      this.fboScene = mkFbo();
      this.fboPick = mkFbo();
    }

    _initGL() {
      const gl = this.gl;
      this.progPbr = this._link(VS_PBR, FS_PBR);
      this.progLine = this._link(VS_LINE, FS_LINE);
      this.progBg = this._link(VS_QUAD, FS_BG);
      this.progBloom = this._link(VS_QUAD, FS_BLOOM);
      this.progPick = this._link(VS_PICK, FS_PICK);
      this.progUvPick = this._link(VS_UV_PICK, FS_UV_PICK);

      this.quadBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);

      this.lightGizmoBuf = gl.createBuffer();
      this.boneLinesBuf = gl.createBuffer();
      this.boneJointsBuf = gl.createBuffer();

      this.textures = {
        albedo: this._makeSolidTex([180, 180, 180, 255]),
        normal: this._makeSolidTex([128, 128, 255, 255]),
        orm: this._makeSolidTex([255, 165, 10, 255]),
        shadow: this._makeSolidTex([255, 255, 255, 255]),
        emissive: this._makeSolidTex([0, 0, 0, 255]),
        display: this._makeSolidTex([180, 180, 180, 255]),
        thickness: this._makeSolidTex([140, 140, 140, 255]),
        splitRight: this._makeSolidTex([128, 128, 255, 255]),
      };
      this._texDefaults = {
        albedo: [180, 180, 180, 255],
        normal: [128, 128, 255, 255],
        orm: [255, 165, 10, 255],
        shadow: [255, 255, 255, 255],
        emissive: [0, 0, 0, 255],
        display: [180, 180, 180, 255],
        thickness: [140, 140, 140, 255],
        splitRight: [128, 128, 255, 255],
      };
      this._texSlotGen = {};
      this._animUploadGen = 0;
      this.hasNormalTex = false;

      this.geoms = {
        cube: this._uploadGeom(buildBoxGeom(1.35, 1.35, 1.35, 1.0)),
        wall: this._uploadGeom(buildBoxGeom(2.15, 2.15, 0.24, 3.0)),
        pillar: this._uploadGeom(buildBoxGeom(0.92, 1.95, 0.92, 1.0)),
        sphere: this._uploadGeom(buildSphereGeom(0.95, 28, 18)),
        bbmodel: null,
      };

      this._uploadGridBuffers(-0.78);
    }

    setBBModelMesh(mesh) {
      const prevClipName = (this.skelClipIdx >= 0 && this.animations[this.skelClipIdx]) ? this.animations[this.skelClipIdx].name : null;
      const prevTime = this.skelTime;
      const prevPlaying = this.skelPlaying;

      this.bbMeshRaw = mesh || null;
      this.elemMeta = (mesh && mesh.elem_meta) || [];
      this.bones = (mesh && Array.isArray(mesh.bones)) ? mesh.bones : [];
      this.animations = (mesh && Array.isArray(mesh.animations)) ? mesh.animations : [];
      this.hasBBModel = !!(mesh && mesh.positions && mesh.positions.length >= 9);
      this.highlightPickId = 0;
      this._boneLineCount = 0;
      this._boneJointCount = 0;

      if (!this.isWebGL || !this.hasBBModel) {
        this.skelClipIdx = -1;
        this.skelTime = 0;
        this.skelPlaying = false;
        return;
      }
      const b = mesh.bounds;
      const cx = b.center[0], cy = b.center[1], cz = b.center[2];
      const sc = 1.2 / Math.max(1.0, b.radius);
      this._bbBounds = b;
      this._bbScale = sc;
      const pos = new Float32Array(mesh.positions.length);
      for (let i = 0; i < mesh.positions.length; i += 3) {
        pos[i] = (mesh.positions[i] - cx) * sc;
        pos[i + 1] = (mesh.positions[i + 1] - cy) * sc;
        pos[i + 2] = (mesh.positions[i + 2] - cz) * sc;
      }
      this._restPos = pos;
      this._restNrm = new Float32Array(mesh.normals);
      this._restTan = new Float32Array(mesh.tangents);
      this._skinPos = new Float32Array(pos.length);
      this._skinNrm = new Float32Array(mesh.normals.length);
      this._skinTan = new Float32Array(mesh.tangents.length);

      const gl = this.gl;
      const mkDyn = (arr) => {
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.DYNAMIC_DRAW);
        return buf;
      };
      const mkStat = (arr) => {
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW);
        return buf;
      };
      const picks = mesh.pick_ids && mesh.pick_ids.length === mesh.vertex_count ? mesh.pick_ids : new Float32Array(mesh.vertex_count);
      this.geoms.bbmodel = {
        pos: mkDyn(pos),
        nrm: mkDyn(mesh.normals),
        tan: mkDyn(mesh.tangents),
        uv: mkStat(mesh.uvs),
        pick: mkStat(picks),
        count: mesh.vertex_count,
      };
      const minY = (b.min[1] - cy) * sc - 0.02;
      this._uploadGridBuffers(minY);

      // Preserve active skeletal animation across texture re-evaluations of the same rig!
      let restoredIdx = -1;
      if (prevClipName) {
        restoredIdx = this.animations.findIndex((a) => a.name === prevClipName);
      }
      if (restoredIdx >= 0) {
        this.skelClipIdx = restoredIdx;
        this.skelTime = Math.min(prevTime, this.animations[restoredIdx].length || 1.0);
        this.skelPlaying = prevPlaying;
        this._updateSkelPose(this.skelTime);
      } else {
        this.skelClipIdx = -1;
        this.skelTime = 0.0;
        this.skelPlaying = false;
        this._updateSkelPose(0.0);
      }
    }

    setSkeletonClip(idx, autoPlay) {
      if (!this.hasBBModel || !this.animations.length || idx < 0 || idx >= this.animations.length) {
        this.skelClipIdx = -1;
        this.skelTime = 0.0;
        this.skelPlaying = false;
        this._updateSkelPose(0.0);
        if (this.onSkelTimeUpdate) this.onSkelTimeUpdate(0, 1, false);
        return;
      }
      this.skelClipIdx = idx;
      this.skelTime = 0.0;
      this.skelPlaying = autoPlay !== false;
      this._updateSkelPose(0.0);
      const clip = this.animations[idx];
      if (this.onSkelTimeUpdate) this.onSkelTimeUpdate(0, clip.length || 1.0, this.skelPlaying);
    }

    setSkeletonTime(t) {
      if (!this.hasBBModel || this.skelClipIdx < 0 || this.skelClipIdx >= this.animations.length) return;
      const clip = this.animations[this.skelClipIdx];
      const len = Math.max(0.05, clip.length || 1.0);
      this.skelTime = Math.max(0, Math.min(len, Number(t) || 0));
      this._updateSkelPose(this.skelTime);
      if (this.onSkelTimeUpdate) this.onSkelTimeUpdate(this.skelTime, len, this.skelPlaying);
    }

    setSkeletonPlaying(on) {
      if (!this.hasBBModel || !this.animations.length) {
        this.skelPlaying = false;
        return;
      }
      if (on && this.skelClipIdx < 0) {
        this.skelClipIdx = 0;
        this.skelTime = 0.0;
      }
      this.skelPlaying = !!on;
      const clip = this.animations[this.skelClipIdx];
      this._updateSkelPose(this.skelTime);
      if (this.onSkelTimeUpdate && clip) this.onSkelTimeUpdate(this.skelTime, clip.length || 1.0, this.skelPlaying);
    }

    _updateSkelPose(t) {
      if (!this.isWebGL || !this.hasBBModel || !this.geoms.bbmodel || !this.bbMeshRaw || !this._bbBounds) return;
      const mesh = this.bbMeshRaw;
      const bones = this.bones || [];
      const b = this._bbBounds;
      const cx = b.center[0], cy = b.center[1], cz = b.center[2];
      const sc = this._bbScale;
      const clip = (this.skelClipIdx >= 0 && this.skelClipIdx < this.animations.length) ? this.animations[this.skelClipIdx] : null;
      const tracks = (clip && clip.tracks) || {};
      const clipLen = clip ? (clip.length || 1.0) : 1.0;
      const clipLoop = clip ? clip.loop : 'loop';

      // 1. Compute Forward-Kinematics M_world[bi] for every bone (parent-before-child order!)
      const worldMats = new Array(bones.length);
      for (let bi = 0; bi < bones.length; bi++) {
        const bn = bones[bi];
        const pIdx = bn.parent;
        const pOrig = (pIdx >= 0 && pIdx < bones.length) ? bones[pIdx].origin : [0, 0, 0];
        let dx = bn.origin[0] - pOrig[0];
        let dy = bn.origin[1] - pOrig[1];
        let dz = bn.origin[2] - pOrig[2];
        let rx = bn.rest_rot[0], ry = bn.rest_rot[1], rz = bn.rest_rot[2];
        let sx = 1, sy = 1, sz = 1;

        const tr = clip ? tracks[String(bi)] : null;
        if (tr) {
          if (tr.position) {
            const dp = evalBoneChannel(tr.position, t, clipLen, clipLoop, false);
            dx += dp[0]; dy += dp[1]; dz += dp[2];
          }
          if (tr.rotation) {
            const dr = evalBoneChannel(tr.rotation, t, clipLen, clipLoop, false);
            rx += dr[0]; ry += dr[1]; rz += dr[2];
          }
          if (tr.scale) {
            const ds = evalBoneChannel(tr.scale, t, clipLen, clipLoop, true);
            sx = ds[0]; sy = ds[1]; sz = ds[2];
          }
        }
        const mLoc = m4BoneLocal(dx, dy, dz, rx, ry, rz, sx, sy, sz);
        worldMats[bi] = (pIdx >= 0 && worldMats[pIdx]) ? m4Mul(worldMats[pIdx], mLoc) : mLoc;
      }

      // 2. Build 3D X-Ray Bone Hierarchy Gizmo buffers (limbs + joint markers)
      const gl = this.gl;
      const bLines = [];
      const jLines = [];
      const jRad = 0.032;
      for (let bi = 0; bi < bones.length; bi++) {
        const m = worldMats[bi];
        const jx = (m[12] - cx) * sc;
        const jy = (m[13] - cy) * sc;
        const jz = (m[14] - cz) * sc;
        // Amber joint crosshair marker
        jLines.push(jx - jRad, jy, jz, jx + jRad, jy, jz);
        jLines.push(jx, jy - jRad, jz, jx, jy + jRad, jz);
        jLines.push(jx, jy, jz - jRad, jx, jy, jz + jRad);
        const pIdx = bones[bi].parent;
        if (pIdx >= 0 && worldMats[pIdx]) {
          const pm = worldMats[pIdx];
          const px = (pm[12] - cx) * sc;
          const py = (pm[13] - cy) * sc;
          const pz = (pm[14] - cz) * sc;
          bLines.push(px, py, pz, jx, jy, jz);
        }
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.boneLinesBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(bLines), gl.DYNAMIC_DRAW);
      this._boneLineCount = bLines.length / 3;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.boneJointsBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(jLines), gl.DYNAMIC_DRAW);
      this._boneJointCount = jLines.length / 3;

      // 3. Transform bone-local vertices into posed GPU vertex buffers (or restore exact rest buffers in Bind Pose!)
      const g = this.geoms.bbmodel;
      if (!clip || !mesh.local_positions || !mesh.bone_ids || mesh.bone_ids.length !== mesh.vertex_count) {
        gl.bindBuffer(gl.ARRAY_BUFFER, g.pos);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this._restPos);
        gl.bindBuffer(gl.ARRAY_BUFFER, g.nrm);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this._restNrm);
        gl.bindBuffer(gl.ARRAY_BUFFER, g.tan);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this._restTan);
        return;
      }

      const lPos = mesh.local_positions;
      const lNrm = mesh.local_normals;
      const lTan = mesh.local_tangents;
      const bIds = mesh.bone_ids;
      const outP = this._skinPos;
      const outN = this._skinNrm;
      const outT = this._skinTan;
      const vCount = mesh.vertex_count;

      for (let v = 0; v < vCount; v++) {
        const bi = bIds[v];
        const i3 = v * 3;
        const px = lPos[i3], py = lPos[i3 + 1], pz = lPos[i3 + 2];
        const nx = lNrm[i3], ny = lNrm[i3 + 1], nz = lNrm[i3 + 2];
        const tx = lTan[i3], ty = lTan[i3 + 1], tz = lTan[i3 + 2];
        if (bi >= 0 && bi < worldMats.length) {
          const m = worldMats[bi];
          const wx = m[0] * px + m[4] * py + m[8] * pz + m[12];
          const wy = m[1] * px + m[5] * py + m[9] * pz + m[13];
          const wz = m[2] * px + m[6] * py + m[10] * pz + m[14];
          outP[i3] = (wx - cx) * sc;
          outP[i3 + 1] = (wy - cy) * sc;
          outP[i3 + 2] = (wz - cz) * sc;

          let rnx = m[0] * nx + m[4] * ny + m[8] * nz;
          let rny = m[1] * nx + m[5] * ny + m[9] * nz;
          let rnz = m[2] * nx + m[6] * ny + m[10] * nz;
          const nl = Math.hypot(rnx, rny, rnz) || 1;
          outN[i3] = rnx / nl; outN[i3 + 1] = rny / nl; outN[i3 + 2] = rnz / nl;

          let rtx = m[0] * tx + m[4] * ty + m[8] * tz;
          let rty = m[1] * tx + m[5] * ty + m[9] * tz;
          let rtz = m[2] * tx + m[6] * ty + m[10] * tz;
          const tl = Math.hypot(rtx, rty, rtz) || 1;
          outT[i3] = rtx / tl; outT[i3 + 1] = rty / tl; outT[i3 + 2] = rtz / tl;
        } else {
          outP[i3] = (px - cx) * sc;
          outP[i3 + 1] = (py - cy) * sc;
          outP[i3 + 2] = (pz - cz) * sc;
          outN[i3] = nx; outN[i3 + 1] = ny; outN[i3 + 2] = nz;
          outT[i3] = tx; outT[i3 + 1] = ty; outT[i3 + 2] = tz;
        }
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, g.pos);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, outP);
      gl.bindBuffer(gl.ARRAY_BUFFER, g.nrm);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, outN);
      gl.bindBuffer(gl.ARRAY_BUFFER, g.tan);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, outT);
    }

    /** Compute exact `.bbmodel`-space 3D decal projector basis through a viewport click (or viewport center). */
    getCameraProjectorAt(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      const nx = (clientX != null) ? (clientX - rect.left) / Math.max(1, rect.width) : 0.5;
      const ny = (clientY != null) ? (clientY - rect.top) / Math.max(1, rect.height) : 0.5;
      const ndcX = nx * 2.0 - 1.0;
      const ndcY = 1.0 - ny * 2.0;

      const eye = [
        Math.cos(this.pitch) * Math.sin(this.yaw) * this.dist,
        Math.sin(this.pitch) * this.dist,
        Math.cos(this.pitch) * Math.cos(this.yaw) * this.dist,
      ];
      const tgt = [0, -0.09, 0];
      let fx = tgt[0] - eye[0], fy = tgt[1] - eye[1], fz = tgt[2] - eye[2];
      const fl = Math.hypot(fx, fy, fz) || 1;
      fx /= fl; fy /= fl; fz /= fl;

      // Camera right = normalize(cross(fwd, worldUp))
      let rx = fy * 0 - fz * 1, ry = fz * 0 - fx * 0, rz = fx * 1 - fy * 0;
      const rl = Math.hypot(rx, ry, rz) || 1;
      rx /= rl; ry /= rl; rz /= rl;
      // Camera up = normalize(cross(right, fwd))
      const ux = ry * fz - rz * fy;
      const uy = rz * fx - rx * fz;
      const uz = rx * fy - ry * fx;

      const aspect = Math.max(0.1, rect.width / Math.max(1, rect.height));
      const tanHalf = Math.tan(20 * Math.PI / 180);
      let dx = fx + rx * ndcX * tanHalf * aspect + ux * ndcY * tanHalf;
      let dy = fy + ry * ndcX * tanHalf * aspect + uy * ndcY * tanHalf;
      let dz = fz + rz * ndcX * tanHalf * aspect + uz * ndcY * tanHalf;
      const dl = Math.hypot(dx, dy, dz) || 1;
      dx /= dl; dy /= dl; dz /= dl;

      // Orthonormalize projector right/up around the exact clicked ray direction
      let prx = dy * uz - dz * uy;
      let pry = dz * ux - dx * uz;
      let prz = dx * uy - dy * ux;
      const prl = Math.hypot(prx, pry, prz) || 1;
      prx /= prl; pry /= prl; prz /= prl;
      const pux = pry * dz - prz * dy;
      const puy = prz * dx - prx * dz;
      const puz = prx * dy - pry * dx;

      // Map camera eye into .bbmodel world space (any surface along ray `eye + t*rayDir` projects cleanly to u=0.5, v=0.5!)
      let origin = [0, 12, 24];
      if (this._bbBounds && this._bbScale > 1e-6) {
        const c = this._bbBounds.center;
        const invS = 1.0 / this._bbScale;
        origin = [eye[0] * invS + c[0], eye[1] * invS + c[1], eye[2] * invS + c[2]];
      }
      return {
        origin: origin.map((v) => Number(v.toFixed(4))),
        forward: [dx, dy, dz].map((v) => Number(v.toFixed(5))),
        right: [prx, pry, prz].map((v) => Number(v.toFixed(5))),
        up: [pux, puy, puz].map((v) => Number(v.toFixed(5))),
      };
    }

    getCenterCameraProjector() {
      return this.getCameraProjectorAt(null, null);
    }

    _uploadImgToTex(tex, img, wrapRepeat) {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      const isPot = ((img.width & (img.width - 1)) === 0) && ((img.height & (img.height - 1)) === 0);
      const wrap = (wrapRepeat && isPot) ? gl.REPEAT : gl.CLAMP_TO_EDGE;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    }

    _detectTransparency(img) {
      try {
        const c = document.createElement('canvas');
        const sw = Math.min(64, img.width), sh = Math.min(64, img.height);
        c.width = sw; c.height = sh;
        const ctx = c.getContext('2d');
        ctx.drawImage(img, 0, 0, sw, sh);
        const d = ctx.getImageData(0, 0, sw, sh).data;
        for (let i = 3; i < d.length; i += 16) {
          if (d[i] > 10 && d[i] < 248) return true;
        }
      } catch (_) {}
      return false;
    }

    _resetSlotToDefault(slot) {
      this._texSlotGen[slot] = (this._texSlotGen[slot] || 0) + 1;
      if (slot === 'normal') this.hasNormalTex = false;
      if (slot === 'albedo') this.hasTransparency = false;
      if (slot === 'display') this.fallbackImg = null;
      if (!this.isWebGL || !this.textures || !this.textures[slot]) return;
      const rgba = (this._texDefaults && this._texDefaults[slot]) || [180, 180, 180, 255];
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.textures[slot]);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(rgba));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }

    /** Immediately clear ALL GPU texture slots + animRing on project/mesh switch (prevents cross-project texture glitching!). */
    resetTextures() {
      this.clearAnimFrames();
      const slots = ['albedo', 'normal', 'orm', 'shadow', 'emissive', 'display', 'thickness', 'splitRight'];
      for (const s of slots) this._resetSlotToDefault(s);
    }

    _loadTexFromUrl(slot, url, wrapRepeat) {
      if (!url) {
        this._resetSlotToDefault(slot);
        return;
      }
      const gen = (this._texSlotGen[slot] = (this._texSlotGen[slot] || 0) + 1);
      const img = new Image();
      img.onload = () => {
        // Strict generation guard: discard stale out-of-order async image decodes!
        if (this._texSlotGen[slot] !== gen) return;
        if (slot === 'display') this.fallbackImg = img;
        if (slot === 'albedo') this.hasTransparency = this._detectTransparency(img);
        if (!this.isWebGL) return;
        this._uploadImgToTex(this.textures[slot], img, wrapRepeat);
        if (slot === 'normal') this.hasNormalTex = true;
      };
      img.src = url;
    }

    updateChannels(channels, selectedPreview, tileable) {
      if (!channels) return;
      const rep = !!tileable && this.primitive !== 'bbmodel';
      this._loadTexFromUrl('albedo', channels.albedo || channels.pbr_lit || null, rep);
      this._loadTexFromUrl('normal', channels.normal || null, rep);
      this._loadTexFromUrl('orm', channels.orm || null, rep);
      this._loadTexFromUrl('shadow', channels.shadow || null, rep);
      this._loadTexFromUrl('emissive', channels.emissive || null, rep);
      this._loadTexFromUrl('thickness', channels.thickness || null, rep);

      let dispUrl = channels[this.channel] || channels.pbr_lit || channels.albedo || null;
      if (this.channel === 'selected' && selectedPreview) dispUrl = selectedPreview;
      this._loadTexFromUrl('display', dispUrl, rep);

      const splitUrl = channels[this.splitRightChannel] || channels.normal || channels.albedo || null;
      this._loadTexFromUrl('splitRight', splitUrl, rep);
    }

    /** Pre-upload N animation frames into GPU WebGLTextures for silky-smooth locked 30-60 FPS playback! */
    clearAnimFrames() {
      this._animUploadGen = (this._animUploadGen || 0) + 1;
      if (this.isWebGL && this.animRing.length) {
        const gl = this.gl;
        for (const f of this.animRing) {
          if (f.albedoTex) gl.deleteTexture(f.albedoTex);
          if (f.emissiveTex) gl.deleteTexture(f.emissiveTex);
          if (f.displayTex) gl.deleteTexture(f.displayTex);
        }
      }
      this.animRing = [];
      this.activeAnimIdx = -1;
    }

    async uploadAnimFrames(seqFrames, tileable) {
      this.clearAnimFrames();
      const myGen = this._animUploadGen;
      if (!Array.isArray(seqFrames) || !seqFrames.length) return false;
      const rep = !!tileable && this.primitive !== 'bbmodel';
      const loadImg = (url) => new Promise((resolve) => {
        if (!url) { resolve(null); return; }
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = () => resolve(null);
        im.src = url;
      });
      const ring = [];
      for (const fr of seqFrames) {
        const [albImg, emImg, dispImg] = await Promise.all([
          loadImg(fr.albedo),
          loadImg(fr.emissive),
          loadImg(fr.display || fr.pbr_lit || fr.albedo),
        ]);
        // Abort & free GPU textures immediately if project switched or ring invalidated mid-upload!
        if (this._animUploadGen !== myGen) {
          if (this.isWebGL) {
            const gl = this.gl;
            for (const f of ring) {
              if (f.albedoTex) gl.deleteTexture(f.albedoTex);
              if (f.emissiveTex) gl.deleteTexture(f.emissiveTex);
              if (f.displayTex) gl.deleteTexture(f.displayTex);
            }
          }
          return false;
        }
        const entry = {
          phase: fr.phase,
          thumbs: fr.thumbs || {},
          displayImg: dispImg || albImg,
          albedoTex: null,
          emissiveTex: null,
          displayTex: null,
        };
        if (this.isWebGL) {
          if (albImg) {
            entry.albedoTex = this._makeSolidTex([180, 180, 180, 255]);
            this._uploadImgToTex(entry.albedoTex, albImg, rep);
          }
          if (emImg) {
            entry.emissiveTex = this._makeSolidTex([0, 0, 0, 255]);
            this._uploadImgToTex(entry.emissiveTex, emImg, rep);
          }
          if (dispImg) {
            entry.displayTex = this._makeSolidTex([180, 180, 180, 255]);
            this._uploadImgToTex(entry.displayTex, dispImg, rep);
          }
        }
        ring.push(entry);
      }
      if (this._animUploadGen !== myGen) {
        if (this.isWebGL) {
          const gl = this.gl;
          for (const f of ring) {
            if (f.albedoTex) gl.deleteTexture(f.albedoTex);
            if (f.emissiveTex) gl.deleteTexture(f.emissiveTex);
            if (f.displayTex) gl.deleteTexture(f.displayTex);
          }
        }
        return false;
      }
      this.animRing = ring;
      return true;
    }

    setAnimFrameIdx(idx) {
      if (!this.animRing.length || idx < 0) {
        this.activeAnimIdx = -1;
        return null;
      }
      this.activeAnimIdx = ((idx % this.animRing.length) + this.animRing.length) % this.animRing.length;
      return this.animRing[this.activeAnimIdx];
    }

    resetCamera() {
      this.yaw = -0.38;
      this.pitch = 0.24;
      this.dist = 3.55;
    }

    _activeGeom() {
      const activeKey = (this.primitive === 'bbmodel' && this.geoms.bbmodel) ? 'bbmodel' :
                        (this.primitive === 'bbmodel' ? 'cube' : this.primitive);
      return this.geoms[activeKey] || this.geoms.cube;
    }

    pickPartAt(clientX, clientY) {
      if (!this.isWebGL || !this.hasBBModel || !this.geoms.bbmodel || !this.fboPick || !this._lastMvp) return null;
      const rect = this.canvas.getBoundingClientRect();
      const nx = (clientX - rect.left) / Math.max(1, rect.width);
      const ny = (clientY - rect.top) / Math.max(1, rect.height);
      if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return null;

      const gl = this.gl;
      const w = this.canvas.width, h = this.canvas.height;
      const px = Math.max(0, Math.min(w - 1, Math.floor(nx * w)));
      const py = Math.max(0, Math.min(h - 1, Math.floor((1.0 - ny) * h)));

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboPick.fb);
      gl.viewport(0, 0, w, h);
      gl.disable(gl.BLEND);
      gl.disable(gl.CULL_FACE);
      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(true);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      gl.useProgram(this.progPick);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.progPick, 'uMvp'), false, this._lastMvp);
      const g = this.geoms.bbmodel;
      const locP = gl.getAttribLocation(this.progPick, 'aPos');
      gl.bindBuffer(gl.ARRAY_BUFFER, g.pos);
      gl.enableVertexAttribArray(locP);
      gl.vertexAttribPointer(locP, 3, gl.FLOAT, false, 0, 0);
      const locI = gl.getAttribLocation(this.progPick, 'aPickId');
      gl.bindBuffer(gl.ARRAY_BUFFER, g.pick);
      gl.enableVertexAttribArray(locI);
      gl.vertexAttribPointer(locI, 1, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, g.count);

      const pix = new Uint8Array(4);
      gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pix);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      const id = pix[0] + pix[1] * 256;
      if (id <= 0 || id > this.elemMeta.length) return null;
      return this.elemMeta[id - 1];
    }

    /** GPU UV-Coordinate Pick at (clientX, clientY) for Direct 3D Brush Mask Painting! */
    pickUvAt(clientX, clientY) {
      if (!this.isWebGL || !this.fboPick || !this._lastMvp) return null;
      const rect = this.canvas.getBoundingClientRect();
      const nx = (clientX - rect.left) / Math.max(1, rect.width);
      const ny = (clientY - rect.top) / Math.max(1, rect.height);
      if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return null;

      const gl = this.gl;
      const w = this.canvas.width, h = this.canvas.height;
      const px = Math.max(0, Math.min(w - 1, Math.floor(nx * w)));
      const py = Math.max(0, Math.min(h - 1, Math.floor((1.0 - ny) * h)));

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboPick.fb);
      gl.viewport(0, 0, w, h);
      gl.disable(gl.BLEND);
      gl.disable(gl.CULL_FACE);
      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(true);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const g = this._activeGeom();
      gl.useProgram(this.progUvPick);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.progUvPick, 'uMvp'), false, this._lastMvp);
      const locP = gl.getAttribLocation(this.progUvPick, 'aPos');
      gl.bindBuffer(gl.ARRAY_BUFFER, g.pos);
      gl.enableVertexAttribArray(locP);
      gl.vertexAttribPointer(locP, 3, gl.FLOAT, false, 0, 0);
      const locU = gl.getAttribLocation(this.progUvPick, 'aUv');
      gl.bindBuffer(gl.ARRAY_BUFFER, g.uv);
      gl.enableVertexAttribArray(locU);
      gl.vertexAttribPointer(locU, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, g.count);

      const pix = new Uint8Array(4);
      gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pix);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      if (pix[2] < 128) return null;
      return { u: pix[0] / 255.0, v: pix[1] / 255.0 };
    }

    /** Capture N smooth 360° Turntable PNG frames directly from live WebGL canvas! */
    captureTurntableFrames(count) {
      const n = Math.max(8, Math.min(36, count || 20));
      const origYaw = this.yaw;
      const origSpin = this.spin;
      this.spin = false;
      const frames = [];
      for (let i = 0; i < n; i++) {
        this.yaw = origYaw + (i / n) * Math.PI * 2;
        this.render();
        frames.push(this.canvas.toDataURL('image/png'));
      }
      this.yaw = origYaw;
      this.spin = origSpin;
      this.render();
      return frames;
    }

    _resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(10, Math.floor(rect.width * dpr));
      const h = Math.max(10, Math.floor(rect.height * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
    }

    _bindInteraction() {
      this._dragging = false;
      this._dragMoved = false;
      let lx = 0, ly = 0;
      this.canvas.addEventListener('contextmenu', (e) => {
        if (this.decalMode || this.brushMode) e.preventDefault();
      });
      this.canvas.addEventListener('pointerdown', (e) => {
        // 0. 3D Camera-Projected Decal Stamping (left-click stamps through exact cursor ray; right-drag still orbits!)
        if (this.decalMode && e.button === 0 && !e.shiftKey && !e.altKey) {
          if (this.onDecalStamp) {
            const proj = this.getCameraProjectorAt(e.clientX, e.clientY);
            this.onDecalStamp(proj);
          }
          return;
        }
        // 1. Direct 3D Brush Mask Painting
        if (this.brushMode && e.button === 0) {
          const hit = this.pickUvAt(e.clientX, e.clientY);
          if (hit && this.onBrush3d) {
            this._brushDragging = true;
            this.canvas.setPointerCapture(e.pointerId);
            this.onBrush3d(hit.u, hit.v, true);
            return;
          }
        }
        // 2. Direct 3D Part Picking
        if (this.pickMode || e.altKey || e.ctrlKey) {
          const hit = this.pickPartAt(e.clientX, e.clientY);
          if (hit) {
            this.highlightPickId = hit.id;
            if (this.onPickPart) this.onPickPart(hit, e.shiftKey);
            return;
          }
        }
        // 3. Interactive Environment & Light Direction Orbit (`Shift+Drag` or `☀ Light` mode!)
        if (this.lightDragMode || e.shiftKey) {
          this._lightDragging = true;
          lx = e.clientX; ly = e.clientY;
          this.canvas.setPointerCapture(e.pointerId);
          return;
        }
        // 4. Standard Camera Orbit
        this._dragging = true;
        this._dragMoved = false;
        lx = e.clientX; ly = e.clientY;
        this.canvas.setPointerCapture(e.pointerId);
      });

      this.canvas.addEventListener('pointermove', (e) => {
        if (this._brushDragging) {
          const hit = this.pickUvAt(e.clientX, e.clientY);
          if (hit && this.onBrush3d) this.onBrush3d(hit.u, hit.v, false);
          return;
        }
        if (this._lightDragging) {
          const dx = e.clientX - lx, dy = e.clientY - ly;
          lx = e.clientX; ly = e.clientY;
          this.lightAz = ((this.lightAz + dx * 0.9 + 540) % 360) - 180;
          this.lightEl = Math.max(10, Math.min(85, this.lightEl - dy * 0.45));
          this.envRotDeg = (this.envRotDeg + dx * 0.7 + 360) % 360;
          if (this.onLightChange) this.onLightChange(Math.round(this.lightAz), Math.round(this.lightEl), Math.round(this.envRotDeg));
          return;
        }
        if (this._dragging) {
          const dx = e.clientX - lx, dy = e.clientY - ly;
          if (Math.hypot(dx, dy) > 2) this._dragMoved = true;
          lx = e.clientX; ly = e.clientY;
          this.yaw -= dx * 0.011;
          this.pitch = Math.max(-1.25, Math.min(1.38, this.pitch + dy * 0.011));
          return;
        }
        if (this.pickMode && this.hasBBModel) {
          const hit = this.pickPartAt(e.clientX, e.clientY);
          this.highlightPickId = hit ? hit.id : 0;
          if (this.onHoverPart) this.onHoverPart(hit);
        }
      });
      this.canvas.addEventListener('pointerup', () => {
        this._dragging = false;
        this._lightDragging = false;
        this._brushDragging = false;
      });
      this.canvas.addEventListener('pointerleave', () => {
        if (this.pickMode) {
          this.highlightPickId = 0;
          if (this.onHoverPart) this.onHoverPart(null);
        }
      });
      this.canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        this.dist = Math.max(1.5, Math.min(7.0, this.dist * (e.deltaY > 0 ? 1.09 : 0.92)));
      }, { passive: false });
    }

    render() {
      this._resize();
      if (this.canvas.style.display === 'none') return;
      if (!this.isWebGL) {
        const ctx = this.canvas.getContext('2d');
        if (!ctx) return;
        ctx.fillStyle = '#181c23';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        const im = (this.activeAnimIdx >= 0 && this.animRing[this.activeAnimIdx] && this.animRing[this.activeAnimIdx].displayImg) || this.fallbackImg;
        if (im) {
          ctx.imageSmoothingEnabled = false;
          const s = Math.min(this.canvas.width, this.canvas.height) * 0.78;
          ctx.drawImage(im, (this.canvas.width - s) * 0.5, (this.canvas.height - s) * 0.5, s, s);
        }
        return;
      }

      const gl = this.gl;
      const w = this.canvas.width, h = this.canvas.height;
      this._ensureFBOs(w, h);
      const useBloomFbo = !!(this.fboScene && this.bloomStrength > 0.005);

      gl.bindFramebuffer(gl.FRAMEBUFFER, useBloomFbo ? this.fboScene.fb : null);
      gl.viewport(0, 0, w, h);
      gl.clearColor(0.085, 0.098, 0.125, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const rig = LIGHTING_RIGS[this.lightingPreset] || LIGHTING_RIGS.studio_hero;
      const azR = this.lightAz * Math.PI / 180;
      const elR = this.lightEl * Math.PI / 180;
      const envRotRad = (this.envRotDeg || 0) * Math.PI / 180;
      const lightDir = [Math.cos(elR) * Math.sin(azR), Math.sin(elR), Math.cos(elR) * Math.cos(azR)];

      // 0. Blurred Rotatable Studio HDRI Environment Sky Backdrop
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.disable(gl.CULL_FACE);
      gl.useProgram(this.progBg);
      gl.uniform1f(gl.getUniformLocation(this.progBg, 'uYaw'), this.yaw);
      gl.uniform1f(gl.getUniformLocation(this.progBg, 'uPitch'), this.pitch);
      gl.uniform1f(gl.getUniformLocation(this.progBg, 'uEnvRot'), envRotRad);
      gl.uniform1f(gl.getUniformLocation(this.progBg, 'uLightIntensity'), this.lightIntensity);
      gl.uniform3fv(gl.getUniformLocation(this.progBg, 'uKeyColor'), new Float32Array(rig.key));
      gl.uniform3fv(gl.getUniformLocation(this.progBg, 'uSkyColor'), new Float32Array(rig.sky));
      gl.uniform3fv(gl.getUniformLocation(this.progBg, 'uGroundColor'), new Float32Array(rig.ground));
      gl.uniform3fv(gl.getUniformLocation(this.progBg, 'uAccentColor'), new Float32Array(rig.accent));
      gl.uniform3fv(gl.getUniformLocation(this.progBg, 'uLightDir'), new Float32Array(lightDir));
      gl.uniform1i(gl.getUniformLocation(this.progBg, 'uShowBg'), this.showEnvBg ? 1 : 0);
      const locBg = gl.getAttribLocation(this.progBg, 'aQuad');
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
      gl.enableVertexAttribArray(locBg);
      gl.vertexAttribPointer(locBg, 2, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      gl.enable(gl.DEPTH_TEST);
      gl.depthMask(true);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      const eye = [
        Math.cos(this.pitch) * Math.sin(this.yaw) * this.dist,
        Math.sin(this.pitch) * this.dist,
        Math.cos(this.pitch) * Math.cos(this.yaw) * this.dist,
      ];
      const proj = m4Perspective(40 * Math.PI / 180, w / Math.max(1, h), 0.1, 40.0);
      const view = m4LookAt(eye, [0, -0.09, 0], [0, 1, 0]);
      const mvp = m4Mul(proj, view);
      this._lastMvp = mvp;
      const model = m4Identity();

      // 1. Observable Multi-Tier Studio Floor Grid (Minor + Bright Major + Colored X/Z Axes!)
      gl.useProgram(this.progLine);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.progLine, 'uMvp'), false, mvp);
      const lPos = gl.getAttribLocation(this.progLine, 'aPos');
      const drawLines = (layer, rgba) => {
        if (!layer || !layer.count) return;
        gl.uniform4f(gl.getUniformLocation(this.progLine, 'uColor'), rgba[0], rgba[1], rgba[2], rgba[3]);
        gl.bindBuffer(gl.ARRAY_BUFFER, layer.buf);
        gl.enableVertexAttribArray(lPos);
        gl.vertexAttribPointer(lPos, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, layer.count);
      };
      if (this.gridLayers) {
        drawLines(this.gridLayers.minor, [0.42, 0.50, 0.66, 0.42]);
        drawLines(this.gridLayers.major, [0.64, 0.76, 0.94, 0.80]);
        drawLines(this.gridLayers.axisX, [0.96, 0.38, 0.42, 0.90]);
        drawLines(this.gridLayers.axisZ, [0.24, 0.78, 0.98, 0.90]);
      }

      // Optional visible 3D Sun / Key Light Direction Ray Gizmo while adjusting light!
      if (this.lightDragMode || this._lightDragging) {
        const lx = lightDir[0] * 1.95, ly = lightDir[1] * 1.95, lz = lightDir[2] * 1.95;
        const rayVerts = new Float32Array([
          0, 0, 0, lx, ly, lz,
          lx - 0.12, ly, lz, lx + 0.12, ly, lz,
          lx, ly - 0.12, lz, lx, ly + 0.12, lz,
          lx, ly, lz - 0.12, lx, ly, lz + 0.12,
        ]);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.lightGizmoBuf);
        gl.bufferData(gl.ARRAY_BUFFER, rayVerts, gl.DYNAMIC_DRAW);
        gl.uniform4f(gl.getUniformLocation(this.progLine, 'uColor'), 1.0, 0.82, 0.28, 0.95);
        gl.enableVertexAttribArray(lPos);
        gl.vertexAttribPointer(lPos, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, 8);
      }

      // 2. Main 3D Geometry (Double-Sided Two-Pass when Transparent!)
      const geom = this._activeGeom();
      const animF = (this.activeAnimIdx >= 0 && this.animRing[this.activeAnimIdx]) || null;

      gl.useProgram(this.progPbr);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.progPbr, 'uMvp'), false, mvp);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.progPbr, 'uModel'), false, model);
      gl.uniform3fv(gl.getUniformLocation(this.progPbr, 'uCamPos'), new Float32Array(eye));
      gl.uniform3fv(gl.getUniformLocation(this.progPbr, 'uLightDir'), new Float32Array(lightDir));
      gl.uniform3fv(gl.getUniformLocation(this.progPbr, 'uKeyColor'), new Float32Array(rig.key));
      gl.uniform3fv(gl.getUniformLocation(this.progPbr, 'uSkyColor'), new Float32Array(rig.sky));
      gl.uniform3fv(gl.getUniformLocation(this.progPbr, 'uGroundColor'), new Float32Array(rig.ground));
      gl.uniform3fv(gl.getUniformLocation(this.progPbr, 'uAccentColor'), new Float32Array(rig.accent));
      gl.uniform1i(gl.getUniformLocation(this.progPbr, 'uDirectMode'), this.channel === 'pbr_lit' ? 0 : 1);
      gl.uniform1i(gl.getUniformLocation(this.progPbr, 'uHasNormal'), this.hasNormalTex ? 1 : 0);
      gl.uniform1f(gl.getUniformLocation(this.progPbr, 'uClearcoat'), this.clearcoat);
      gl.uniform1f(gl.getUniformLocation(this.progPbr, 'uSssStrength'), this.sssStrength);
      gl.uniform1f(gl.getUniformLocation(this.progPbr, 'uHighlightPickId'), this.highlightPickId);
      gl.uniform1f(gl.getUniformLocation(this.progPbr, 'uEnvRot'), envRotRad);
      gl.uniform1f(gl.getUniformLocation(this.progPbr, 'uLightIntensity'), this.lightIntensity);
      gl.uniform1i(gl.getUniformLocation(this.progPbr, 'uSplitWipe'), this.splitWipe ? 1 : 0);
      gl.uniform1f(gl.getUniformLocation(this.progPbr, 'uSplitPos'), this.splitPos);
      gl.uniform1f(gl.getUniformLocation(this.progPbr, 'uViewportW'), w);

      const texBindings = [
        ['uAlbedo', (animF && animF.albedoTex) || this.textures.albedo],
        ['uNormal', this.textures.normal],
        ['uOrm', this.textures.orm],
        ['uShadow', this.textures.shadow],
        ['uEmissive', (animF && animF.emissiveTex) || this.textures.emissive],
        ['uDisplay', (animF && animF.displayTex) || this.textures.display],
        ['uThickness', this.textures.thickness],
        ['uSplitRight', this.textures.splitRight],
      ];
      for (let i = 0; i < texBindings.length; i++) {
        gl.activeTexture(gl.TEXTURE0 + i);
        gl.bindTexture(gl.TEXTURE_2D, texBindings[i][1]);
        gl.uniform1i(gl.getUniformLocation(this.progPbr, texBindings[i][0]), i);
      }

      const bindAttr = (name, buf, size) => {
        const loc = gl.getAttribLocation(this.progPbr, name);
        if (loc < 0) return;
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      };
      bindAttr('aPos', geom.pos, 3);
      bindAttr('aNormal', geom.nrm, 3);
      bindAttr('aTangent', geom.tan, 3);
      bindAttr('aUv', geom.uv, 2);
      bindAttr('aPickId', geom.pick, 1);

      const locBackPass = gl.getUniformLocation(this.progPbr, 'uBackPass');
      if (this.hasTransparency) {
        // Two-Pass Double-Sided Transparency: draw translucent interior back faces first, then front faces!
        gl.enable(gl.CULL_FACE);
        gl.cullFace(gl.FRONT);
        gl.depthMask(false);
        gl.uniform1i(locBackPass, 1);
        gl.drawArrays(gl.TRIANGLES, 0, geom.count);

        gl.cullFace(gl.BACK);
        gl.depthMask(true);
        gl.uniform1i(locBackPass, 0);
        gl.drawArrays(gl.TRIANGLES, 0, geom.count);
        gl.disable(gl.CULL_FACE);
      } else {
        gl.disable(gl.CULL_FACE);
        gl.depthMask(true);
        gl.uniform1i(locBackPass, 0);
        gl.drawArrays(gl.TRIANGLES, 0, geom.count);
      }

      if (this.wireframe) {
        gl.useProgram(this.progLine);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.progLine, 'uMvp'), false, mvp);
        gl.uniform4f(gl.getUniformLocation(this.progLine, 'uColor'), 0.23, 0.51, 0.96, 0.72);
        gl.bindBuffer(gl.ARRAY_BUFFER, geom.pos);
        gl.enableVertexAttribArray(lPos);
        gl.vertexAttribPointer(lPos, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, geom.count);
      }

      // Optional X-Ray 3D Skeletal Bone Hierarchy Gizmo Overlay (cyan limbs + amber joints!)
      if (this.showBones && this.hasBBModel && this.primitive === 'bbmodel' && this._boneJointCount > 0) {
        gl.disable(gl.DEPTH_TEST);
        gl.useProgram(this.progLine);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.progLine, 'uMvp'), false, mvp);
        if (this._boneLineCount > 0) {
          gl.uniform4f(gl.getUniformLocation(this.progLine, 'uColor'), 0.22, 0.88, 1.0, 0.92);
          gl.bindBuffer(gl.ARRAY_BUFFER, this.boneLinesBuf);
          gl.enableVertexAttribArray(lPos);
          gl.vertexAttribPointer(lPos, 3, gl.FLOAT, false, 0, 0);
          gl.drawArrays(gl.LINES, 0, this._boneLineCount);
        }
        gl.uniform4f(gl.getUniformLocation(this.progLine, 'uColor'), 0.98, 0.78, 0.24, 0.96);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.boneJointsBuf);
        gl.enableVertexAttribArray(lPos);
        gl.vertexAttribPointer(lPos, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, this._boneJointCount);
        gl.enable(gl.DEPTH_TEST);
      }

      // 3. Post-Process Emissive Bloom Pass -> Screen
      if (useBloomFbo) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, w, h);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);
        gl.useProgram(this.progBloom);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.fboScene.tex);
        gl.uniform1i(gl.getUniformLocation(this.progBloom, 'uScene'), 0);
        gl.uniform2f(gl.getUniformLocation(this.progBloom, 'uTexel'), 1.0 / Math.max(1, w), 1.0 / Math.max(1, h));
        gl.uniform1f(gl.getUniformLocation(this.progBloom, 'uBloom'), this.bloomStrength);
        const locBl = gl.getAttribLocation(this.progBloom, 'aQuad');
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
        gl.enableVertexAttribArray(locBl);
        gl.vertexAttribPointer(locBl, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
    }
  }

  global.StudioGL3DViewport = StudioGL3DViewport;
})(window);
