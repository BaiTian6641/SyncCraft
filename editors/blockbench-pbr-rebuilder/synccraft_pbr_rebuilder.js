/**
 * SyncCraft PBR Texture Rebuilder — Blockbench 5 Plugin (`v1.0.0`).
 *
 * Full-featured offline + bridge-connected PBR texture workflow for Blockbench 5:
 *   1. Import `.synccraft.pbr.json` manifests (single-file embedded OR sibling files),
 *      LabPBR 1.3 (`_n.png`, `_s.png`), Bedrock RTX (`_mer.png`, `_normal.png`), or standalone PBR maps.
 *   2. Toggle real-time Three.js `MeshStandardMaterial` PBR viewport lighting right inside Blockbench 5
 *      (unpacks LabPBR 1.3 & Bedrock MER cleanly with `NearestFilter` crisp pixels + 3-point studio lights).
 *   3. Interactive "Rebuild & Bake PBR -> Albedo Texture" studio dialog (pure-JS Cook-Torrance GGX baker,
 *      works 100% offline with zero Python server, wrapped in `Undo.initEdit` / `Undo.finishEdit`).
 *   4. Live "Re-Run SyncCraft TexGraph" via optional local `127.0.0.1:18765` bridge (`synccraft serve-blockbench`).
 *   5. Export Baked Studio PNG (`_baked.png`) & `.synccraft.pbr.json` manifest straight from Blockbench.
 */
(function () {
  "use strict";

  const DEFAULT_BRIDGE_URL = "http://127.0.0.1:18765";

  const BAKE_PRESETS = {
    studio_hero: {
      label: "Studio Hero (Warm Key / Cool Rim)",
      key_az_deg: -36.0,
      key_el_deg: 54.0,
      key_rgb: [2.25, 2.16, 2.02],
      fill_az_deg: 138.0,
      fill_el_deg: 28.0,
      fill_rgb: [0.78, 0.92, 1.18],
      sky_rgb: [0.46, 0.52, 0.64],
      ground_rgb: [0.26, 0.23, 0.30],
      normal_strength: 1.0,
      spec_strength: 1.0,
      ao_strength: 0.62,
      sss_strength: 0.85,
      emissive_strength: 1.55,
      blend_with_albedo: 0.22,
      exposure: 1.08
    },
    dramatic_rim: {
      label: "Dramatic Obsidian Rim",
      key_az_deg: -52.0,
      key_el_deg: 42.0,
      key_rgb: [2.55, 2.38, 2.15],
      fill_az_deg: 152.0,
      fill_el_deg: 24.0,
      fill_rgb: [0.65, 1.05, 1.48],
      sky_rgb: [0.34, 0.40, 0.56],
      ground_rgb: [0.18, 0.16, 0.24],
      normal_strength: 1.15,
      spec_strength: 1.25,
      ao_strength: 0.78,
      sss_strength: 1.05,
      emissive_strength: 1.85,
      blend_with_albedo: 0.14,
      exposure: 1.12
    },
    high_noon: {
      label: "High Noon Daylight",
      key_az_deg: -18.0,
      key_el_deg: 72.0,
      key_rgb: [2.35, 2.32, 2.24],
      fill_az_deg: 160.0,
      fill_el_deg: 35.0,
      fill_rgb: [0.85, 0.96, 1.12],
      sky_rgb: [0.56, 0.64, 0.78],
      ground_rgb: [0.34, 0.32, 0.28],
      normal_strength: 0.95,
      spec_strength: 0.88,
      ao_strength: 0.50,
      sss_strength: 0.75,
      emissive_strength: 1.25,
      blend_with_albedo: 0.26,
      exposure: 1.05
    },
    moonlit_obsidian: {
      label: "Moonlit Obsidian",
      key_az_deg: 40.0,
      key_el_deg: 48.0,
      key_rgb: [1.55, 1.88, 2.45],
      fill_az_deg: -135.0,
      fill_el_deg: 24.0,
      fill_rgb: [0.72, 0.48, 0.85],
      sky_rgb: [0.28, 0.36, 0.54],
      ground_rgb: [0.16, 0.14, 0.24],
      normal_strength: 1.1,
      spec_strength: 1.20,
      ao_strength: 0.72,
      sss_strength: 0.90,
      emissive_strength: 1.95,
      blend_with_albedo: 0.16,
      exposure: 1.10
    }
  };

  function srgbToLin(c) {
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }

  function linToSrgb(c) {
    const v = Math.max(0.0, Math.min(1.0, c));
    return v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1.0 / 2.4) - 0.055;
  }

  function lightDir(azDeg, elDeg) {
    const a = (azDeg * Math.PI) / 180.0;
    const e = (elDeg * Math.PI) / 180.0;
    const x = Math.cos(e) * Math.sin(a);
    const y = Math.cos(e) * Math.cos(a);
    const z = Math.sin(e);
    const l = Math.hypot(x, y, z) || 1.0;
    return [x / l, y / l, z / l];
  }

  /**
   * Unpack LabPBR 1.3 (`labpbrN`, `labpbrS`) or Bedrock RTX (`bedrockMer`, `normalData`)
   * into normalized per-pixel channel arrays for both Three.js PBR viewport & GGX baking.
   */
  function unpackPbrChannels(width, height, maps) {
    const count = width * height;
    const nx = new Float32Array(count);
    const ny = new Float32Array(count);
    const nz = new Float32Array(count);
    const ao = new Float32Array(count);
    const roughness = new Float32Array(count);
    const metallic = new Float32Array(count);
    const f0 = new Float32Array(count);
    const sss = new Float32Array(count);
    const emissive = new Float32Array(count);

    const nData = maps.labpbrN || maps.normal || null;
    const sData = maps.labpbrS || null;
    const merData = maps.bedrockMer || null;
    const isLabN = Boolean(maps.labpbrN);

    for (let i = 0; i < count; i++) {
      const p = i * 4;
      if (nData) {
        const vx = nData[p] / 127.5 - 1.0;
        const vy = nData[p + 1] / 127.5 - 1.0;
        const vz = isLabN
          ? Math.sqrt(Math.max(0.04, 1.0 - Math.min(0.96, vx * vx + vy * vy)))
          : (nData[p + 2] / 127.5 - 1.0);
        const len = Math.hypot(vx, vy, vz) || 1.0;
        nx[i] = vx / len;
        ny[i] = vy / len;
        nz[i] = vz / len;
        ao[i] = isLabN ? Math.max(0.08, nData[p + 2] / 255.0) : 1.0;
      } else {
        nx[i] = 0.0;
        ny[i] = 0.0;
        nz[i] = 1.0;
        ao[i] = 1.0;
      }

      if (sData) {
        const smooth = sData[p] / 255.0;
        roughness[i] = Math.max(0.06, Math.min(1.0, 1.0 - smooth));
        const gRaw = sData[p + 1];
        metallic[i] = gRaw >= 230 ? 1.0 : 0.0;
        f0[i] = Math.max(0.02, Math.min(0.25, gRaw / 255.0));
        const bRaw = sData[p + 2];
        sss[i] = bRaw >= 65 ? (bRaw - 64) / 191.0 : 0.0;
        const aRaw = sData[p + 3];
        emissive[i] = aRaw < 255 ? aRaw / 254.0 : 0.0;
      } else if (merData) {
        metallic[i] = merData[p] / 255.0;
        emissive[i] = merData[p + 1] / 255.0;
        roughness[i] = Math.max(0.06, Math.min(1.0, merData[p + 2] / 255.0));
        f0[i] = 0.04;
        sss[i] = 0.0;
      } else {
        roughness[i] = 0.55;
        metallic[i] = 0.0;
        f0[i] = 0.04;
        sss[i] = 0.0;
        emissive[i] = 0.0;
      }
    }
    return { nx, ny, nz, ao, roughness, metallic, f0, sss, emissive };
  }

  /**
   * Pure-JS Cook-Torrance GGX PBR -> Baked Albedo pixel engine (matches Python `bake_pbr_to_albedo`).
   * Works 100% offline inside Blockbench 5 with zero external libraries.
   */
  function bakePbrPixels(width, height, albedoData, maps, options) {
    const presetKey = (options && options.preset) || "studio_hero";
    const baseCfg = BAKE_PRESETS[presetKey] || BAKE_PRESETS.studio_hero;
    const cfg = Object.assign({}, baseCfg, options || {});
    const ch = unpackPbrChannels(width, height, maps || {});
    const emRgba = (maps && maps.emissive) || null;
    const out = new Uint8ClampedArray(width * height * 4);

    const L1 = lightDir(cfg.key_az_deg, cfg.key_el_deg);
    const L2 = lightDir(cfg.fill_az_deg, cfg.fill_el_deg);
    const H1x = L1[0], H1y = L1[1], H1z = L1[2] + 1.0;
    const H1l = Math.hypot(H1x, H1y, H1z) || 1.0;
    const H1 = [H1x / H1l, H1y / H1l, H1z / H1l];
    const h1dotv = Math.max(0.0, H1[2]);

    const H2x = L2[0], H2y = L2[1], H2z = L2[2] + 1.0;
    const H2l = Math.hypot(H2x, H2y, H2z) || 1.0;
    const H2 = [H2x / H2l, H2y / H2l, H2z / H2l];
    const h2dotv = Math.max(0.0, H2[2]);

    const nStr = Number(cfg.normal_strength !== undefined ? cfg.normal_strength : 1.0);
    const sSpec = Number(cfg.spec_strength !== undefined ? cfg.spec_strength : 1.0);
    const sAo = Number(cfg.ao_strength !== undefined ? cfg.ao_strength : 0.62);
    const sSss = Number(cfg.sss_strength !== undefined ? cfg.sss_strength : 0.85);
    const sEm = Number(cfg.emissive_strength !== undefined ? cfg.emissive_strength : 1.55);
    const bAlb = Math.max(0.0, Math.min(1.0, Number(cfg.blend_with_albedo !== undefined ? cfg.blend_with_albedo : 0.22)));
    const exp = Number(cfg.exposure !== undefined ? cfg.exposure : 1.08);
    const kRad = cfg.key_rgb, fRad = cfg.fill_rgb, skyC = cfg.sky_rgb, gndC = cfg.ground_rgb;

    const count = width * height;
    for (let i = 0; i < count; i++) {
      const p = i * 4;
      const alpha = albedoData[p + 3];
      if (alpha < 4) {
        out[p] = albedoData[p];
        out[p + 1] = albedoData[p + 1];
        out[p + 2] = albedoData[p + 2];
        out[p + 3] = alpha;
        continue;
      }

      const br = srgbToLin(albedoData[p] / 255.0);
      const bg = srgbToLin(albedoData[p + 1] / 255.0);
      const bb = srgbToLin(albedoData[p + 2] / 255.0);

      let vx = ch.nx[i] * nStr;
      let vy = ch.ny[i] * nStr;
      let vz = Math.sqrt(Math.max(0.04, 1.0 - Math.min(0.96, vx * vx + vy * vy)));
      const vl = Math.hypot(vx, vy, vz) || 1.0;
      vx /= vl; vy /= vl; vz /= vl;

      const rough = ch.roughness[i];
      const metal = ch.metallic[i];
      const aoM = Math.max(0.12, Math.min(1.0, 1.0 - (1.0 - ch.ao[i]) * sAo));
      const f0r = ch.f0[i] * (1.0 - metal) + br * metal;
      const f0g = ch.f0[i] * (1.0 - metal) + bg * metal;
      const f0b = ch.f0[i] * (1.0 - metal) + bb * metal;
      const ndotv = Math.max(0.05, vz);
      const a2 = Math.pow(rough * rough, 2.0);
      const kG = Math.pow(rough + 1.0, 2.0) / 8.0;
      const gv = ndotv / (ndotv * (1.0 - kG) + kG);

      function addLight(L, H, hdv, rad) {
        const ndlRaw = vx * L[0] + vy * L[1] + vz * L[2];
        const ndl = Math.max(0.0, ndlRaw);
        const ndh = Math.max(0.0, vx * H[0] + vy * H[1] + vz * H[2]);
        const denom = ndh * ndh * (a2 - 1.0) + 1.0;
        const D = a2 / (Math.PI * denom * denom + 1e-6);
        const G = gv * (ndl / (ndl * (1.0 - kG) + kG + 1e-6));
        const sch = Math.pow(1.0 - hdv, 5.0);
        const Fr = f0r + (1.0 - f0r) * sch;
        const Fg = f0g + (1.0 - f0g) * sch;
        const Fb = f0b + (1.0 - f0b) * sch;
        const spScale = ((D * G) / (4.0 * ndotv * ndl + 1e-4)) * sSpec;
        const wrap = Math.max(0.0, Math.min(1.0, (ndlRaw + 0.48) / 1.48));
        const sssF = ch.sss[i] * sSss * (1.0 - metal) * wrap * 0.45;
        return [
          (((1.0 - Fr) * (1.0 - metal) * br) / Math.PI + Fr * spScale) * rad[0] * ndl + br * sssF * rad[0],
          (((1.0 - Fg) * (1.0 - metal) * bg) / Math.PI + Fg * spScale) * rad[1] * ndl + bg * sssF * rad[1],
          (((1.0 - Fb) * (1.0 - metal) * bb) / Math.PI + Fb * spScale) * rad[2] * ndl + bb * sssF * rad[2]
        ];
      }

      const c1 = addLight(L1, H1, h1dotv, kRad);
      const c2 = addLight(L2, H2, h2dotv, fRad);
      let cr = c1[0] + c2[0];
      let cg = c1[1] + c2[1];
      let cb = c1[2] + c2[2];

      const skyW = Math.max(0.0, Math.min(1.0, 0.5 + 0.5 * vy));
      const er = skyW * skyC[0] + (1.0 - skyW) * gndC[0];
      const eg = skyW * skyC[1] + (1.0 - skyW) * gndC[1];
      const eb = skyW * skyC[2] + (1.0 - skyW) * gndC[2];
      const ry = Math.max(-1.0, Math.min(1.0, 2.0 * vz * vy));
      const esr = ry > 0.0 ? skyC[0] * (1.25 + 0.95 * ry) : gndC[0] * 0.95;
      const esg = ry > 0.0 ? skyC[1] * (1.25 + 0.95 * ry) : gndC[1] * 0.95;
      const esb = ry > 0.0 ? skyC[2] * (1.25 + 0.95 * ry) : gndC[2] * 0.95;
      const fAmbS = Math.pow(1.0 - ndotv, 5.0);
      const fAr = f0r + (Math.max(1.0 - rough, f0r) - f0r) * fAmbS;
      const fAg = f0g + (Math.max(1.0 - rough, f0g) - f0g) * fAmbS;
      const fAb = f0b + (Math.max(1.0 - rough, f0b) - f0b) * fAmbS;
      const specDamp = sSpec * (1.0 - 0.52 * rough);

      cr += ((1.0 - fAr) * (1.0 - metal) * br * er + fAr * esr * specDamp) * aoM;
      cg += ((1.0 - fAg) * (1.0 - metal) * bg * eg + fAg * esg * specDamp) * aoM;
      cb += ((1.0 - fAb) * (1.0 - metal) * bb * eb + fAb * esb * specDamp) * aoM;

      if (bAlb > 1e-4) {
        cr = cr * (1.0 - bAlb) + br * aoM * bAlb;
        cg = cg * (1.0 - bAlb) + bg * aoM * bAlb;
        cb = cb * (1.0 - bAlb) + bb * aoM * bAlb;
      }

      if (emRgba) {
        const ea = emRgba[p + 3] / 255.0;
        if (ea > 0.01) {
          cr += srgbToLin(emRgba[p] / 255.0) * ea * sEm;
          cg += srgbToLin(emRgba[p + 1] / 255.0) * ea * sEm;
          cb += srgbToLin(emRgba[p + 2] / 255.0) * ea * sEm;
        }
      } else if (ch.emissive[i] > 1e-4) {
        cr += br * ch.emissive[i] * sEm;
        cg += bg * ch.emissive[i] * sEm;
        cb += bb * ch.emissive[i] * sEm;
      }

      // ACES Filmic tonemap
      const xr = Math.max(0.0, cr * exp), xg = Math.max(0.0, cg * exp), xb = Math.max(0.0, cb * exp);
      const mr = (xr * (2.51 * xr + 0.03)) / (xr * (2.43 * xr + 0.59) + 0.14);
      const mg = (xg * (2.51 * xg + 0.03)) / (xg * (2.43 * xg + 0.59) + 0.14);
      const mb = (xb * (2.51 * xb + 0.03)) / (xb * (2.43 * xb + 0.59) + 0.14);

      out[p] = Math.round(linToSrgb(mr) * 255.0);
      out[p + 1] = Math.round(linToSrgb(mg) * 255.0);
      out[p + 2] = Math.round(linToSrgb(mb) * 255.0);
      out[p + 3] = alpha;
    }
    return out;
  }

  function makeCanvasFromPixels(rt, width, height, pixels) {
    const doc = rt.document;
    const canvas = doc.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(width, height);
    imgData.data.set(pixels);
    ctx.putImageData(imgData, 0, 0);
    return canvas;
  }

  function loadImagePixels(rt, source) {
    return new Promise((resolve, reject) => {
      const ImgCtor = rt.Image;
      const img = new ImgCtor();
      img.onload = () => {
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        const canvas = rt.document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        resolve({ width: w, height: h, data: ctx.getImageData(0, 0, w, h).data, img });
      };
      img.onerror = (err) => reject(new Error("Failed to decode PBR texture image: " + err));
      img.src = source;
    });
  }

  function createPbrRebuilderController(env, options) {
    const rt = env || globalThis;
    const bridgeUrl = (options && options.bridgeUrl) || DEFAULT_BRIDGE_URL;
    const state = {
      manifest: null,
      width: 0,
      height: 0,
      maps: {}, // { albedo, labpbrN, labpbrS, bedrockMer, normal, emissive } (Uint8ClampedArray)
      viewportActive: false,
      savedMaterials: new Map(),
      injectedLights: [],
      createdThreeObjects: []
    };
    const registeredActions = [];

    function getActiveTexture() {
      if (rt.Texture && rt.Texture.selected) return rt.Texture.selected;
      if (rt.Texture && Array.isArray(rt.Texture.all) && rt.Texture.all.length > 0) return rt.Texture.all[0];
      return null;
    }

    function restoreViewportPbr() {
      if (state.savedMaterials.size > 0) {
        for (const [obj, mat] of state.savedMaterials.entries()) {
          try { obj.material = mat; } catch (_) {}
        }
        state.savedMaterials.clear();
      }
      if (rt.scene && state.injectedLights.length > 0) {
        for (const l of state.injectedLights) {
          try { rt.scene.remove(l); } catch (_) {}
        }
        state.injectedLights = [];
      }
      for (const obj of state.createdThreeObjects) {
        try { if (obj && typeof obj.dispose === "function") obj.dispose(); } catch (_) {}
      }
      state.createdThreeObjects = [];
      state.viewportActive = false;
      if (rt.Canvas && typeof rt.Canvas.updateAll === "function") rt.Canvas.updateAll();
      return { viewportActive: false };
    }

    function toggleViewportPbr(enable) {
      const want = enable === undefined ? !state.viewportActive : Boolean(enable);
      if (!want) return restoreViewportPbr();
      if (!rt.THREE || !state.maps.albedo) {
        return { viewportActive: false, reason: !rt.THREE ? "three_unavailable" : "no_pbr_maps_loaded" };
      }

      restoreViewportPbr();
      const THREE = rt.THREE;
      const w = state.width, h = state.height;
      const ch = unpackPbrChannels(w, h, state.maps);

      // Build crisp pixel-art Three.js textures from unpacked LabPBR / MER channels
      const nRgba = new Uint8ClampedArray(w * h * 4);
      const rRgba = new Uint8ClampedArray(w * h * 4);
      const mRgba = new Uint8ClampedArray(w * h * 4);
      const eRgba = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i++) {
        const p = i * 4;
        nRgba[p] = Math.round((ch.nx[i] * 0.5 + 0.5) * 255);
        nRgba[p + 1] = Math.round((ch.ny[i] * 0.5 + 0.5) * 255);
        nRgba[p + 2] = Math.round((ch.nz[i] * 0.5 + 0.5) * 255);
        nRgba[p + 3] = 255;
        const rv = Math.round(ch.roughness[i] * 255);
        rRgba[p] = rv; rRgba[p + 1] = rv; rRgba[p + 2] = rv; rRgba[p + 3] = 255;
        const mv = Math.round(ch.metallic[i] * 255);
        mRgba[p] = mv; mRgba[p + 1] = mv; mRgba[p + 2] = mv; mRgba[p + 3] = 255;
        if (state.maps.emissive) {
          eRgba[p] = state.maps.emissive[p];
          eRgba[p + 1] = state.maps.emissive[p + 1];
          eRgba[p + 2] = state.maps.emissive[p + 2];
          eRgba[p + 3] = state.maps.emissive[p + 3];
        } else {
          const ev = ch.emissive[i];
          eRgba[p] = Math.round(state.maps.albedo[p] * ev);
          eRgba[p + 1] = Math.round(state.maps.albedo[p + 1] * ev);
          eRgba[p + 2] = Math.round(state.maps.albedo[p + 2] * ev);
          eRgba[p + 3] = 255;
        }
      }

      function makeTex(pix, isSrgb) {
        const cv = makeCanvasFromPixels(rt, w, h, pix);
        const t = new THREE.CanvasTexture(cv);
        t.magFilter = THREE.NearestFilter;
        t.minFilter = THREE.NearestFilter;
        t.flipY = false;
        if (isSrgb && THREE.SRGBColorSpace) t.colorSpace = THREE.SRGBColorSpace;
        t.needsUpdate = true;
        state.createdThreeObjects.push(t);
        return t;
      }

      const pbrMat = new THREE.MeshStandardMaterial({
        map: makeTex(state.maps.albedo, true),
        normalMap: makeTex(nRgba, false),
        roughnessMap: makeTex(rRgba, false),
        metalnessMap: makeTex(mRgba, false),
        emissiveMap: makeTex(eRgba, true),
        emissive: new THREE.Color(0xffffff),
        emissiveIntensity: 1.25,
        roughness: 1.0,
        metalness: 1.0,
        alphaTest: 0.5,
        side: THREE.DoubleSide
      });
      state.createdThreeObjects.push(pbrMat);

      if (rt.Canvas && rt.Canvas.materials) {
        const tex = getActiveTexture();
        if (tex && tex.uuid && rt.Canvas.materials[tex.uuid]) {
          state.savedMaterials.set(rt.Canvas.materials, rt.Canvas.materials[tex.uuid]);
        }
      }
      if (rt.Cube && Array.isArray(rt.Cube.all)) {
        for (const cube of rt.Cube.all) {
          if (cube && cube.mesh) {
            state.savedMaterials.set(cube.mesh, cube.mesh.material);
            cube.mesh.material = pbrMat;
          }
        }
      }
      if (rt.scene) {
        const keyL = new THREE.DirectionalLight(0xfff4e6, 2.2);
        keyL.position.set(-24, 42, 26);
        const rimL = new THREE.DirectionalLight(0x88ccff, 0.95);
        rimL.position.set(28, 18, -30);
        const hemi = new THREE.HemisphereLight(0xdce8ff, 0x3a3248, 0.85);
        for (const l of [keyL, rimL, hemi]) {
          rt.scene.add(l);
          state.injectedLights.push(l);
        }
      }
      state.viewportActive = true;
      if (rt.Canvas && typeof rt.Canvas.updateAll === "function") rt.Canvas.updateAll();
      return { viewportActive: true, material: pbrMat };
    }

    async function loadManifestObject(manifestObj, fileReaderFn) {
      state.manifest = manifestObj;
      state.maps = {};
      const emb = manifestObj.embedded_data_urls || {};
      const mapSpec = manifestObj.maps || {};
      const keys = [
        ["albedo", "albedo"],
        ["labpbrN", "labpbr_n"],
        ["labpbrS", "labpbr_s"],
        ["bedrockMer", "bedrock_mer"],
        ["normal", "normal"],
        ["emissive", "emissive"]
      ];
      for (const [dstKey, mKey] of keys) {
        let src = emb[mKey] || null;
        if (!src && mapSpec[mKey] && typeof fileReaderFn === "function") {
          src = await fileReaderFn(mapSpec[mKey]);
        }
        if (src) {
          const decoded = await loadImagePixels(rt, src);
          state.maps[dstKey] = decoded.data;
          state.width = decoded.width;
          state.height = decoded.height;
        }
      }
      return { loadedKeys: Object.keys(state.maps), width: state.width, height: state.height };
    }

    function applyBakedPixelsToBlockbench(bakedPixels, width, height, mode, label) {
      const targetMode = mode || "add_layer";
      const cv = makeCanvasFromPixels(rt, width, height, bakedPixels);
      const dataUrl = cv.toDataURL("image/png");
      const activeTex = getActiveTexture();

      if (targetMode === "replace_active" && activeTex) {
        if (rt.Undo && typeof rt.Undo.initEdit === "function") {
          rt.Undo.initEdit({ textures: [activeTex], bitmap: true });
        }
        if (typeof activeTex.fromDataURL === "function") {
          activeTex.fromDataURL(dataUrl);
        } else {
          activeTex.source = dataUrl;
        }
        if (rt.Undo && typeof rt.Undo.finishEdit === "function") {
          rt.Undo.finishEdit(label || "SyncCraft PBR: Rebuild & Bake Active Texture");
        }
        return { applied: true, mode: "replace_active", texture: activeTex, dataUrl };
      }

      const added = [];
      if (rt.Undo && typeof rt.Undo.initEdit === "function") {
        rt.Undo.initEdit({ textures: added });
      }
      const stem = (state.manifest && state.manifest.stem) || (activeTex && activeTex.name ? activeTex.name.replace(/\.png$/i, "") : "synccraft");
      const newTex = new rt.Texture({ name: `${stem}_baked.png`, width, height });
      if (typeof newTex.fromDataURL === "function") {
        newTex.fromDataURL(dataUrl);
      } else {
        newTex.source = dataUrl;
      }
      if (typeof newTex.add === "function") {
        newTex.add(false);
      } else if (rt.Texture && Array.isArray(rt.Texture.all)) {
        rt.Texture.all.push(newTex);
      }
      added.push(newTex);
      if (rt.Undo && typeof rt.Undo.finishEdit === "function") {
        rt.Undo.finishEdit(label || "SyncCraft PBR: Bake Studio Albedo Layer", { textures: added });
      }
      return { applied: true, mode: "add_layer", texture: newTex, dataUrl };
    }

    function rebuildAndBakeTexture(bakeOptions) {
      if (!state.maps.albedo || !state.width || !state.height) {
        throw new Error("Load a .synccraft.pbr.json bundle or LabPBR/MER maps before baking.");
      }
      const opts = bakeOptions || {};
      const baked = bakePbrPixels(state.width, state.height, state.maps.albedo, state.maps, opts);
      return applyBakedPixelsToBlockbench(baked, state.width, state.height, opts.target_mode || "add_layer");
    }

    async function rebuildViaBridge(params, fetchImpl) {
      const doFetch = fetchImpl || (typeof fetch !== "undefined" ? fetch : null);
      if (!doFetch) throw new Error("fetch is unavailable");
      const compiled = rt.Codecs && rt.Codecs.project ? rt.Codecs.project.compile({ raw: true }) : null;
      const bbmodel = typeof compiled === "string" ? JSON.parse(compiled) : compiled;
      const url = bridgeUrl.replace(/\/+$/, "") + "/pbr_rebuild";
      const resp = await doFetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(Object.assign({ bbmodel }, params || {}))
      });
      const body = await resp.json();
      if (!resp.ok || body.ok === false) {
        throw new Error((body && body.message) || ("SyncCraft bridge HTTP " + resp.status));
      }
      if (body.manifest) {
        await loadManifestObject(body.manifest);
      }
      if (body.baked_data_url) {
        const dec = await loadImagePixels(rt, body.baked_data_url);
        applyBakedPixelsToBlockbench(dec.data, dec.width, dec.height, (params && params.target_mode) || "add_layer", "SyncCraft Bridge: Live TexGraph PBR Rebuild");
      }
      return body;
    }

    function openBakeDialog() {
      if (!rt.Dialog) return rebuildAndBakeTexture({});
      const presetChoices = {};
      for (const [k, v] of Object.entries(BAKE_PRESETS)) presetChoices[k] = v.label;
      const dlg = new rt.Dialog({
        id: "synccraft_pbr_bake_dialog",
        title: "SyncCraft PBR Texture Rebuilder & Baker",
        form: {
          preset: { label: "Lighting Rig Preset", type: "select", options: presetChoices, value: "studio_hero" },
          key_az_deg: { label: "Key Light Azimuth (deg)", type: "range", min: -180, max: 180, step: 2, value: -36 },
          key_el_deg: { label: "Key Light Elevation (deg)", type: "range", min: 5, max: 85, step: 1, value: 54 },
          normal_strength: { label: "Normal Relief Strength", type: "range", min: 0, max: 3.0, step: 0.05, value: 1.0 },
          spec_strength: { label: "Specular / Metallic Sheen", type: "range", min: 0, max: 2.0, step: 0.05, value: 1.0 },
          ao_strength: { label: "Baked Contact AO Depth", type: "range", min: 0, max: 1.5, step: 0.05, value: 0.62 },
          sss_strength: { label: "Wing SSS Backlight", type: "range", min: 0, max: 1.5, step: 0.05, value: 0.85 },
          emissive_strength: { label: "Emissive Glow Strength", type: "range", min: 0, max: 2.5, step: 0.05, value: 1.55 },
          blend_with_albedo: { label: "Preserve Unlit Albedo Ratio", type: "range", min: 0, max: 0.6, step: 0.02, value: 0.22 },
          target_mode: {
            label: "Output Target",
            type: "select",
            options: { add_layer: "Add as New Baked Texture Layer (Safe)", replace_active: "Replace Active Texture (Undoable)" },
            value: "add_layer"
          }
        },
        onConfirm(formData) {
          try {
            rebuildAndBakeTexture(formData);
            if (rt.Blockbench && rt.Blockbench.showQuickMessage) {
              rt.Blockbench.showQuickMessage("SyncCraft PBR Texture Baked!", 2200);
            }
          } catch (err) {
            if (rt.Blockbench && rt.Blockbench.showMessageBox) {
              rt.Blockbench.showMessageBox({ title: "SyncCraft PBR Baker", icon: "error", message: String(err.message || err) });
            }
          }
        }
      });
      dlg.show();
      return dlg;
    }

    function openImportManifestDialog() {
      if (!rt.Blockbench || typeof rt.Blockbench.import !== "function") return;
      rt.Blockbench.import(
        {
          resource_id: "synccraft_pbr_manifest",
          extensions: ["json"],
          type: "SyncCraft PBR Manifest (.synccraft.pbr.json)",
          readtype: "text"
        },
        async (files) => {
          if (!files || !files.length) return;
          try {
            const f = files[0];
            const manifest = JSON.parse(f.content);
            const pathMod = typeof require === "function" ? require("path") : null;
            const fsMod = typeof require === "function" ? require("fs") : null;
            const baseDir = (pathMod && f.path) ? pathMod.dirname(f.path) : null;
            await loadManifestObject(manifest, async (relName) => {
              if (!baseDir || !fsMod || !pathMod) return null;
              const full = pathMod.join(baseDir, relName);
              if (!fsMod.existsSync(full)) return null;
              return "data:image/png;base64," + fsMod.readFileSync(full).toString("base64");
            });
            if (rt.Blockbench.showQuickMessage) {
              rt.Blockbench.showQuickMessage(`Loaded SyncCraft PBR (${Object.keys(state.maps).length} maps)`, 2500);
            }
          } catch (err) {
            if (rt.Blockbench.showMessageBox) {
              rt.Blockbench.showMessageBox({ title: "SyncCraft PBR Import Error", icon: "error", message: String(err.message || err) });
            }
          }
        }
      );
    }

    function onload() {
      if (!rt.Action) return;
      const defs = [
        ["synccraft_pbr_import", "SyncCraft PBR: Import Bundle (.synccraft.pbr.json)", "folder_open", openImportManifestDialog],
        ["synccraft_pbr_viewport", "SyncCraft PBR: Toggle Live Three.js PBR Viewport", "wb_sunny", () => toggleViewportPbr()],
        ["synccraft_pbr_bake", "SyncCraft PBR: Rebuild & Bake Studio Texture...", "auto_fix_high", openBakeDialog],
        ["synccraft_pbr_bridge", "SyncCraft PBR: Live Re-Run TexGraph via Bridge", "sync", () => rebuildViaBridge({})]
      ];
      for (const [id, name, icon, click] of defs) {
        const act = new rt.Action(id, { name, icon, category: "textures", click });
        registeredActions.push(act);
        if (rt.MenuBar && typeof rt.MenuBar.addAction === "function") {
          rt.MenuBar.addAction(act, "tools");
        }
      }
    }

    function onunload() {
      restoreViewportPbr();
      while (registeredActions.length > 0) {
        const act = registeredActions.pop();
        if (act && typeof act.delete === "function") act.delete();
      }
    }

    return {
      state,
      onload,
      onunload,
      registeredActions,
      loadManifestObject,
      toggleViewportPbr,
      restoreViewportPbr,
      rebuildAndBakeTexture,
      rebuildViaBridge,
      openBakeDialog,
      openImportManifestDialog
    };
  }

  const pluginRegistry = (typeof BBPlugin !== "undefined" && BBPlugin) || (typeof Plugin !== "undefined" && Plugin) || null;
  if (pluginRegistry && typeof pluginRegistry.register === "function") {
    const controller = createPbrRebuilderController(globalThis);
    pluginRegistry.register("synccraft_pbr_rebuilder", {
      title: "SyncCraft PBR Texture Rebuilder",
      author: "SyncCraft Team",
      description: "Import LabPBR 1.3 / Bedrock MER / .synccraft.pbr.json bundles, preview live Three.js PBR materials in viewport, and rebuild/bake studio lighting into Blockbench textures offline or via live SyncCraft bridge.",
      icon: "auto_fix_high",
      version: "1.0.0",
      variant: "both",
      onload: controller.onload,
      onunload: controller.onunload
    });
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      BAKE_PRESETS,
      unpackPbrChannels,
      bakePbrPixels,
      createPbrRebuilderController
    };
  }
})();
