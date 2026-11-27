/* ==========================================================================
   SyncCraft Studio v14 — Commercial WebGL 3D Lookdev PBR Viewport (`studio_gl3d.js`)
   Features:
   - Real Blockbench 5 `.bbmodel` meshes + standard Lookdev primitives
   - Procedural Studio HDRI Environments (5 rigs: Studio Softbox, Sunset Forge,
     Cyber Neon, Moonlit Cathedral, Overcast Day) + toggleable blurred Env Backdrop
   - Cook-Torrance GGX PBR + Dual-Lobe Clearcoat + Refracted Chromatic Glass/Crystal
     Dispersion + Ray-Traced 3D Shadow + Backlit SSS Thickness Glow
   - Post-Process Emissive Bloom Pass (WebGL FBO multi-tap halo)
   - Pixel-Accurate GPU Color-ID 3D Part Picking + Live Cyan Hover Highlight!
   - Automatic Canvas2D fallback if WebGL is unavailable.
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

  function buildGridGeom(halfExtent, step, yLevel) {
    const pos = [];
    for (let v = -halfExtent; v <= halfExtent + 1e-4; v += step) {
      pos.push(-halfExtent, yLevel, v, halfExtent, yLevel, v);
      pos.push(v, yLevel, -halfExtent, v, yLevel, halfExtent);
    }
    return { positions: new Float32Array(pos), count: pos.length / 3 };
  }

  // Procedural Analytical Studio HDRI Environment Function shared by Sky Backdrop & IBL PBR!
  const GLSL_HDRI_ENV = `
    vec3 sampleHdriEnv(vec3 dir, float rough, vec3 skyCol, vec3 groundCol, vec3 keyCol, vec3 accentCol, vec3 lightDir) {
      vec3 d = normalize(dir);
      float h = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
      vec3 base = mix(groundCol, skyCol, smoothstep(0.32, 0.88, h));
      // Horizon studio glow band
      float horiz = exp(-abs(d.y) * mix(9.0, 2.2, rough));
      base += mix(skyCol, accentCol, 0.5) * horiz * 0.32;
      // Overhead softbox panel + secondary rim softbox
      float topBox = smoothstep(mix(0.84, 0.45, rough), mix(0.95, 0.85, rough), d.y);
      vec3 l1 = normalize(lightDir);
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
      if (baseSample.a < 0.04) discard;

      float isPicked = (uHighlightPickId > 0.5 && abs(vPickId - uHighlightPickId) < 0.4) ? 1.0 : 0.0;

      if (uDirectMode == 1) {
        vec4 disp = texture2D(uDisplay, vUv);
        vec3 dc = mix(disp.rgb, vec3(0.22, 0.88, 1.0), isPicked * 0.38);
        gl_FragColor = vec4(dc, clamp(disp.a, 0.08, 1.0));
        return;
      }

      vec3 N = normalize(vNormal);
      if (uHasNormal == 1) {
        vec3 T = normalize(vTangent - N * dot(vTangent, N));
        vec3 B = normalize(cross(N, T));
        vec3 nMap = texture2D(uNormal, vUv).xyz * 2.0 - 1.0;
        N = normalize(T * nMap.x + B * nMap.y + N * max(0.15, nMap.z));
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
      vec3 direct = (kD * alb / PI + spec * 0.72) * uKeyColor * ndl * vis * 2.35;

      // Dual-Lobe Clearcoat GGX Specular (lacquer, enamel, wet scales, glass, crystal)
      float coatW = clamp(uClearcoat, 0.0, 1.0);
      float D_coat = D_GGX(ndh, 0.08);
      float F_coat = 0.04 + 0.96 * pow(1.0 - hdv, 5.0);
      vec3 coatDirect = vec3(D_coat * F_coat * 0.34) * uKeyColor * ndl * vis * coatW;

      // Procedural Studio HDRI Image-Based Lighting (Diffuse + Specular Reflection!)
      vec3 envIrrad = sampleHdriEnv(N, 0.92, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L);
      vec3 envSpecCol = sampleHdriEnv(R, rough, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L);
      vec3 envCoatCol = sampleHdriEnv(R, 0.07, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L);
      vec3 envDiff = envIrrad * alb * (1.0 - metal * 0.72) * ao * 0.56;
      vec3 envSpec = envSpecCol * F * (1.0 - rough * 0.65) * ao * mix(0.45, 1.15, metal);
      vec3 envCoat = envCoatCol * (0.04 + 0.96 * pow(1.0 - ndv, 5.0)) * coatW * ao * 0.45;

      // Subsurface Scattering (SSS) Backlit Glow via Ray-Traced Thickness!
      float backLight = pow(clamp(dot(V, -L + N * 0.45), 0.0, 1.0), 2.8);
      float wrapDiff = clamp((dot(N, L) + 0.45) / 1.45, 0.0, 1.0);
      float thinness = clamp(1.0 - thick * 0.85, 0.1, 1.0);
      vec3 sssCol = alb * uKeyColor * (backLight * 0.85 + wrapDiff * 0.25) * thinness * (1.0 - metal) * uSssStrength;

      // Translucent Glass & Crystal Chromatic Refraction Distortion + Fresnel Sheen!
      float fresnelRim = pow(clamp(1.0 - ndv, 0.0, 1.0), 3.0);
      float isTranslucent = step(baseSample.a, 0.97);
      vec3 refrCol = vec3(0.0);
      if (isTranslucent > 0.5) {
        vec3 rR = refract(-V, N, 1.0 / 1.45);
        vec3 rG = refract(-V, N, 1.0 / 1.49);
        vec3 rB = refract(-V, N, 1.0 / 1.54);
        float cR = sampleHdriEnv(rR, rough * 0.65, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L).r;
        float cG = sampleHdriEnv(rG, rough * 0.65, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L).g;
        float cB = sampleHdriEnv(rB, rough * 0.65, uSkyColor, uGroundColor, uKeyColor, uAccentColor, L).b;
        refrCol = vec3(cR, cG, cB) * (0.45 + 0.85 * alb) * 0.85 + envSpecCol * fresnelRim * 0.85;
      }

      vec3 emissiveTerm = pow(em.rgb, vec3(2.2)) * em.a * 1.95;
      vec3 col = direct + coatDirect + envDiff + envSpec + envCoat + sssCol + refrCol * isTranslucent + emissiveTerm;
      col += vec3(0.18, 0.86, 1.0) * (0.28 + 0.55 * fresnelRim) * isPicked;

      // ACES Filmic tonemap + sRGB encode
      col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
      col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
      float outAlpha = mix(1.0, clamp(baseSample.a + fresnelRim * 0.28, 0.14, 0.98), isTranslucent);
      gl_FragColor = vec4(col, outAlpha);
    }
  `;

  // Blurred Studio HDRI Sky Backdrop Shader
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
    uniform vec3 uKeyColor;
    uniform vec3 uSkyColor;
    uniform vec3 uGroundColor;
    uniform vec3 uAccentColor;
    uniform vec3 uLightDir;
    uniform int uShowBg;
    ${GLSL_HDRI_ENV}
    void main() {
      if (uShowBg == 0) {
        vec3 base = mix(vec3(0.082, 0.095, 0.122), vec3(0.122, 0.140, 0.176), vUv.y);
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
      vec3 env = sampleHdriEnv(dir, 0.72, uSkyColor, uGroundColor, uKeyColor, uAccentColor, uLightDir) * 0.34;
      env = pow(clamp(env, 0.0, 1.0), vec3(1.0 / 2.2));
      gl_FragColor = vec4(env, 1.0);
    }
  `;

  // Post-Process Screen-Space Bloom Shader (Multi-Tap Kawase Glow)
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

  // GPU Color-ID Picking Shader (encodes 16-bit element ID in R,G)
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
      this.pickMode = false;
      this.highlightPickId = 0;
      this.onPickPart = null;
      this.onHoverPart = null;

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
        if (this.spin && !this._dragging) {
          this.yaw += dt * 0.36;
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

      this.quadBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);

      this.textures = {
        albedo: this._makeSolidTex([180, 180, 180, 255]),
        normal: this._makeSolidTex([128, 128, 255, 255]),
        orm: this._makeSolidTex([255, 165, 10, 255]),
        shadow: this._makeSolidTex([255, 255, 255, 255]),
        emissive: this._makeSolidTex([0, 0, 0, 255]),
        display: this._makeSolidTex([180, 180, 180, 255]),
        thickness: this._makeSolidTex([140, 140, 140, 255]),
      };
      this.hasNormalTex = false;

      this.geoms = {
        cube: this._uploadGeom(buildBoxGeom(1.35, 1.35, 1.35, 1.0)),
        wall: this._uploadGeom(buildBoxGeom(2.15, 2.15, 0.24, 3.0)),
        pillar: this._uploadGeom(buildBoxGeom(0.92, 1.95, 0.92, 1.0)),
        sphere: this._uploadGeom(buildSphereGeom(0.95, 28, 18)),
        bbmodel: null,
      };

      const grid = buildGridGeom(2.4, 0.4, -0.78);
      this.gridBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuf);
      gl.bufferData(gl.ARRAY_BUFFER, grid.positions, gl.STATIC_DRAW);
      this.gridCount = grid.count;
      this.gridY = -0.78;
    }

    setBBModelMesh(mesh) {
      this.bbMeshRaw = mesh || null;
      this.elemMeta = (mesh && mesh.elem_meta) || [];
      this.hasBBModel = !!(mesh && mesh.positions && mesh.positions.length >= 9);
      this.highlightPickId = 0;
      if (!this.isWebGL || !this.hasBBModel) return;
      const b = mesh.bounds;
      const cx = b.center[0], cy = b.center[1], cz = b.center[2];
      const sc = 1.2 / Math.max(1.0, b.radius);
      const pos = new Float32Array(mesh.positions.length);
      for (let i = 0; i < mesh.positions.length; i += 3) {
        pos[i] = (mesh.positions[i] - cx) * sc;
        pos[i + 1] = (mesh.positions[i + 1] - cy) * sc;
        pos[i + 2] = (mesh.positions[i + 2] - cz) * sc;
      }
      this.geoms.bbmodel = this._uploadGeom({
        positions: pos,
        normals: mesh.normals,
        tangents: mesh.tangents,
        uvs: mesh.uvs,
        pick_ids: mesh.pick_ids,
        count: mesh.vertex_count,
      });
      const minY = (b.min[1] - cy) * sc - 0.02;
      const grid = buildGridGeom(2.4, 0.4, minY);
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuf);
      gl.bufferData(gl.ARRAY_BUFFER, grid.positions, gl.STATIC_DRAW);
      this.gridCount = grid.count;
      this.gridY = minY;
    }

    _loadTexFromUrl(slot, url, wrapRepeat) {
      if (!url) return;
      const img = new Image();
      img.onload = () => {
        if (slot === 'display') this.fallbackImg = img;
        if (!this.isWebGL) return;
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.textures[slot]);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        const isPot = ((img.width & (img.width - 1)) === 0) && ((img.height & (img.height - 1)) === 0);
        const wrap = (wrapRepeat && isPot) ? gl.REPEAT : gl.CLAMP_TO_EDGE;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        if (slot === 'normal') this.hasNormalTex = true;
      };
      img.src = url;
    }

    updateChannels(channels, selectedPreview, tileable) {
      if (!channels) return;
      const rep = !!tileable && this.primitive !== 'bbmodel';
      this._loadTexFromUrl('albedo', channels.albedo || channels.pbr_lit, rep);
      if (channels.normal) this._loadTexFromUrl('normal', channels.normal, rep);
      if (channels.orm) this._loadTexFromUrl('orm', channels.orm, rep);
      if (channels.shadow) this._loadTexFromUrl('shadow', channels.shadow, rep);
      if (channels.emissive) this._loadTexFromUrl('emissive', channels.emissive, rep);
      if (channels.thickness) this._loadTexFromUrl('thickness', channels.thickness, rep);

      let dispUrl = channels[this.channel] || channels.pbr_lit || channels.albedo;
      if (this.channel === 'selected' && selectedPreview) dispUrl = selectedPreview;
      this._loadTexFromUrl('display', dispUrl, rep);
    }

    resetCamera() {
      this.yaw = -0.38;
      this.pitch = 0.24;
      this.dist = 3.55;
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
      gl.enable(gl.DEPTH_TEST);
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
      this.canvas.addEventListener('pointerdown', (e) => {
        if (this.pickMode || e.altKey || e.ctrlKey) {
          const hit = this.pickPartAt(e.clientX, e.clientY);
          if (hit) {
            this.highlightPickId = hit.id;
            if (this.onPickPart) this.onPickPart(hit, e.shiftKey);
            return;
          }
        }
        this._dragging = true;
        this._dragMoved = false;
        lx = e.clientX; ly = e.clientY;
        this.canvas.setPointerCapture(e.pointerId);
      });
      this.canvas.addEventListener('pointermove', (e) => {
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
      this.canvas.addEventListener('pointerup', () => { this._dragging = false; });
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
        if (this.fallbackImg) {
          ctx.imageSmoothingEnabled = false;
          const s = Math.min(this.canvas.width, this.canvas.height) * 0.78;
          ctx.drawImage(this.fallbackImg, (this.canvas.width - s) * 0.5, (this.canvas.height - s) * 0.5, s, s);
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
      const lightDir = [Math.cos(elR) * Math.sin(azR), Math.sin(elR), Math.cos(elR) * Math.cos(azR)];

      // 0. Blurred Studio HDRI Environment Sky Backdrop
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.useProgram(this.progBg);
      gl.uniform1f(gl.getUniformLocation(this.progBg, 'uYaw'), this.yaw);
      gl.uniform1f(gl.getUniformLocation(this.progBg, 'uPitch'), this.pitch);
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

      // 1. Subtle Studio Floor Grid
      gl.useProgram(this.progLine);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.progLine, 'uMvp'), false, mvp);
      gl.uniform4f(gl.getUniformLocation(this.progLine, 'uColor'), 0.25, 0.29, 0.36, 0.45);
      const lPos = gl.getAttribLocation(this.progLine, 'aPos');
      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuf);
      gl.enableVertexAttribArray(lPos);
      gl.vertexAttribPointer(lPos, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, this.gridCount);

      // 2. Main 3D Geometry (Loaded .bbmodel OR Standard Lookdev Primitive)
      const activeKey = (this.primitive === 'bbmodel' && this.geoms.bbmodel) ? 'bbmodel' :
                        (this.primitive === 'bbmodel' ? 'cube' : this.primitive);
      const geom = this.geoms[activeKey] || this.geoms.cube;

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

      const slots = ['albedo', 'normal', 'orm', 'shadow', 'emissive', 'display', 'thickness'];
      const uniforms = ['uAlbedo', 'uNormal', 'uOrm', 'uShadow', 'uEmissive', 'uDisplay', 'uThickness'];
      for (let i = 0; i < slots.length; i++) {
        gl.activeTexture(gl.TEXTURE0 + i);
        gl.bindTexture(gl.TEXTURE_2D, this.textures[slots[i]]);
        gl.uniform1i(gl.getUniformLocation(this.progPbr, uniforms[i]), i);
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

      gl.drawArrays(gl.TRIANGLES, 0, geom.count);

      if (this.wireframe) {
        gl.useProgram(this.progLine);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.progLine, 'uMvp'), false, mvp);
        gl.uniform4f(gl.getUniformLocation(this.progLine, 'uColor'), 0.23, 0.51, 0.96, 0.72);
        gl.bindBuffer(gl.ARRAY_BUFFER, geom.pos);
        gl.enableVertexAttribArray(lPos);
        gl.vertexAttribPointer(lPos, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, geom.count);
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
