/**
 * SyncCraft Assistant — Blockbench 5 Plugin (`ADR 0013`).
 *
 * Connects Blockbench to the local `127.0.0.1` SyncCraft Python bridge
 * (`synccraft serve-blockbench`) to provide:
 *   1. `Analyze Model` (`@morphology`, CoM/support, 6-pillar sculpted style score)
 *   2. `Check Rig & Motion` (zero-ML 15-axis OBB SAT + foot-slip & joint-step audit)
 *   3. `Generate Locomotion` (Froude-scaled verified gait -> ordinary editable keyframes)
 *   4. `Retarget Animation` (cross-species role retargeting via `@morphology`)
 *   5. `Add Secondary Motion` (`breathe`, `tail_sway`, `wing_settle` + additive layering)
 *   6. `Non-Destructive Texture Pass` (`preserve_and_layer` overlays, OKLCH variants, selective face rebake)
 *
 * Design Contracts:
 *   - All scene/animation/texture mutations execute inside a single `Undo.initEdit` / `Undo.finishEdit` step.
 *   - Generated animations are ordinary `Animation` + `BoneAnimator` keyframes that artists can edit freely.
 *   - Refuses to silently write unverified outputs (`verified === false`).
 *   - Painted textures are never overwritten in-place under `preserve_and_layer`.
 */
(function () {
  "use strict";

  const DEFAULT_BRIDGE_URL = "http://127.0.0.1:18765";

  async function callBridge(endpoint, payload, options) {
    const baseUrl = (options && options.bridgeUrl) || DEFAULT_BRIDGE_URL;
    const fetchImpl = (options && options.fetchImpl) || (typeof fetch !== "undefined" ? fetch : null);
    if (!fetchImpl) {
      throw new Error("Global fetch is unavailable in this JavaScript runtime");
    }
    const url = baseUrl.replace(/\/+$/, "") + "/" + endpoint.replace(/^\/+/, "");
    const response = await fetchImpl(url, {
      method: endpoint === "health" && !payload ? "GET" : "POST",
      headers: { "Content-Type": "application/json" },
      body: endpoint === "health" && !payload ? undefined : JSON.stringify(payload || {})
    });
    const body = await response.json();
    if (!response.ok || body.ok === false) {
      throw new Error((body && body.message) || ("SyncCraft bridge HTTP " + response.status));
    }
    return body;
  }

  function compileCurrentBBModel(env) {
    const rt = env || globalThis;
    if (rt.Codecs && rt.Codecs.project && typeof rt.Codecs.project.compile === "function") {
      const compiled = rt.Codecs.project.compile({ raw: true });
      return typeof compiled === "string" ? JSON.parse(compiled) : compiled;
    }
    throw new Error("Blockbench Codecs.project.compile is unavailable");
  }

  function applyVerifiedAnimationToBlockbench(result, env, editLabel) {
    const rt = env || globalThis;
    if (!result || result.verified !== true || !result.animation) {
      const msg = "SyncCraft refused to apply unverified animation: check failure diagnostics.";
      if (rt.Blockbench && typeof rt.Blockbench.showMessageBox === "function") {
        rt.Blockbench.showMessageBox({
          title: "SyncCraft Verification Gate",
          icon: "warning",
          message: msg
        });
      }
      return { applied: false, reason: "unverified" };
    }

    const animSpec = result.animation;
    const animationsArr = (rt.Animation && Array.isArray(rt.Animation.all)) ? rt.Animation.all : [];
    const existing = animationsArr.find((a) => a && a.name === animSpec.name) || null;
    const trackedAnimations = existing ? [existing] : [];

    if (rt.Undo && typeof rt.Undo.initEdit === "function") {
      rt.Undo.initEdit({ animations: trackedAnimations });
    }

    let animInstance = existing;
    if (!animInstance) {
      animInstance = new rt.Animation({
        name: animSpec.name,
        length: animSpec.length,
        loop: animSpec.loop || "loop",
        snapping: animSpec.snapping || 24
      });
      if (typeof animInstance.add === "function") {
        animInstance.add(false);
      } else if (Array.isArray(rt.Animation.all) && !rt.Animation.all.includes(animInstance)) {
        rt.Animation.all.push(animInstance);
      }
      trackedAnimations.push(animInstance);
    } else {
      animInstance.length = animSpec.length;
      animInstance.loop = animSpec.loop || "loop";
      animInstance.animators = {};
    }

    animInstance.animators = animInstance.animators || {};
    const animatorsObj = animSpec.animators || {};
    for (const [uid, animatorData] of Object.entries(animatorsObj)) {
      if (!animatorData || typeof animatorData !== "object") continue;
      if (uid === "effects") {
        const fxAnimator = rt.EffectAnimator
          ? new rt.EffectAnimator(animInstance)
          : { name: "Effects", type: "effect", keyframes: [] };
        fxAnimator.keyframes = [];
        for (const kf of animatorData.keyframes || []) {
          if (typeof fxAnimator.addKeyframe === "function") {
            fxAnimator.addKeyframe(kf);
          } else {
            fxAnimator.keyframes.push(JSON.parse(JSON.stringify(kf)));
          }
        }
        animInstance.animators.effects = fxAnimator;
        continue;
      }

      const boneAnimator = rt.BoneAnimator
        ? new rt.BoneAnimator(uid, animInstance, animatorData.name)
        : { uuid: uid, name: animatorData.name, type: "bone", keyframes: [] };
      boneAnimator.keyframes = [];
      for (const kf of animatorData.keyframes || []) {
        if (typeof boneAnimator.addKeyframe === "function") {
          boneAnimator.addKeyframe(kf);
        } else {
          boneAnimator.keyframes.push(JSON.parse(JSON.stringify(kf)));
        }
      }
      animInstance.animators[uid] = boneAnimator;
    }

    if (typeof animInstance.select === "function") {
      animInstance.select();
    }
    if (rt.Undo && typeof rt.Undo.finishEdit === "function") {
      rt.Undo.finishEdit(editLabel || "SyncCraft: Apply Verified Animation", { animations: [animInstance] });
    }
    return { applied: true, animation: animInstance };
  }

  function applyTextureLayersToBlockbench(result, env, editLabel) {
    const rt = env || globalThis;
    const layers = (result && Array.isArray(result.layers)) ? result.layers : [];
    if (!layers.length) {
      return { applied: false, addedTextures: [] };
    }

    const addedTextures = [];
    if (rt.Undo && typeof rt.Undo.initEdit === "function") {
      rt.Undo.initEdit({ textures: addedTextures });
    }

    for (const lyr of layers) {
      const tex = new rt.Texture({
        name: lyr.name,
        width: lyr.width,
        height: lyr.height
      });
      if (typeof tex.fromDataURL === "function") {
        tex.fromDataURL(lyr.source);
      } else {
        tex.source = lyr.source;
      }
      if (typeof tex.add === "function") {
        tex.add(false);
      } else if (rt.Texture && Array.isArray(rt.Texture.all)) {
        rt.Texture.all.push(tex);
      }
      addedTextures.push(tex);
    }

    if (rt.Undo && typeof rt.Undo.finishEdit === "function") {
      rt.Undo.finishEdit(editLabel || "SyncCraft: Non-Destructive Texture Pass", { textures: addedTextures });
    }
    return { applied: true, addedTextures };
  }

  function createSyncCraftPluginController(env, bridgeOptions) {
    const rt = env || globalThis;
    const registeredActions = [];

    const handlers = {
      async analyze(extra) {
        const bbmodel = (extra && extra.bbmodel) || compileCurrentBBModel(rt);
        const res = await callBridge("analyze", { bbmodel }, bridgeOptions);
        if (rt.Blockbench && typeof rt.Blockbench.showMessageBox === "function") {
          const limbCount = (res.morphology && res.morphology.limbs) ? res.morphology.limbs.length : 0;
          const styleScore = res.sculpted_style ? res.sculpted_style.style_anchor_score : 0;
          rt.Blockbench.showMessageBox({
            title: "SyncCraft Model Analysis",
            icon: "insights",
            message:
              `Bones: ${res.bones} | Boxes: ${res.boxes} | Limbs: ${limbCount}\n` +
              `Static Audit: ${res.static_audit.passed ? "PASSED" : "ISSUES FOUND"}\n` +
              `Sculpted Style Score: ${styleScore}`
          });
        }
        return res;
      },

      async check(extra) {
        const bbmodel = (extra && extra.bbmodel) || compileCurrentBBModel(rt);
        const clipName = (extra && extra.clip_name) ||
          (rt.Animation && rt.Animation.selected ? rt.Animation.selected.name : undefined);
        const res = await callBridge("check", { bbmodel, clip_name: clipName }, bridgeOptions);
        if (rt.Blockbench && typeof rt.Blockbench.showMessageBox === "function") {
          rt.Blockbench.showMessageBox({
            title: "SyncCraft Rig & Motion Audit",
            icon: res.passed ? "check_circle" : "error",
            message: res.passed
              ? "All audited static and motion contracts PASSED."
              : "Audit detected collisions, foot slip, or joint-step issues."
          });
        }
        return res;
      },

      async generateLocomotion(params) {
        const opts = params || {};
        const bbmodel = opts.bbmodel || compileCurrentBBModel(rt);
        const res = await callBridge(
          "generate_locomotion",
          {
            bbmodel,
            style_phrase: opts.style_phrase || "steady balanced walk",
            clip_name: opts.clip_name || "walk",
            duration_s: opts.duration_s || 1.6,
            num_frames: opts.num_frames || 16,
            footstep_sound: opts.footstep_sound,
            footstep_particle: opts.footstep_particle
          },
          bridgeOptions
        );
        const outcome = applyVerifiedAnimationToBlockbench(res, rt, "SyncCraft: Generate Locomotion");
        return { response: res, outcome };
      },

      async retarget(params) {
        const opts = params || {};
        if (!opts.source_bbmodel || !opts.clip_name) {
          throw new Error("retarget requires source_bbmodel and clip_name");
        }
        const targetBBModel = opts.target_bbmodel || compileCurrentBBModel(rt);
        const res = await callBridge(
          "retarget",
          {
            source_bbmodel: opts.source_bbmodel,
            clip_name: opts.clip_name,
            target_bbmodel: targetBBModel,
            new_clip_name: opts.new_clip_name || opts.clip_name,
            num_frames: opts.num_frames || 16
          },
          bridgeOptions
        );
        const outcome = applyVerifiedAnimationToBlockbench(res, rt, "SyncCraft: Retarget Animation");
        return { response: res, outcome };
      },

      async addSecondaryMotion(params) {
        const opts = params || {};
        const bbmodel = opts.bbmodel || compileCurrentBBModel(rt);
        const res = await callBridge(
          "secondary_motion",
          {
            bbmodel,
            kind: opts.kind || "breathe",
            clip_name: opts.clip_name,
            duration_s: opts.duration_s || 2.0,
            base_clip_name: opts.base_clip_name,
            layer_weight: opts.layer_weight !== undefined ? opts.layer_weight : 0.75,
            num_frames: opts.num_frames || 16
          },
          bridgeOptions
        );
        const outcome = applyVerifiedAnimationToBlockbench(res, rt, "SyncCraft: Add Secondary Motion");
        return { response: res, outcome };
      },

      async texturePass(params) {
        const opts = params || {};
        const bbmodel = opts.bbmodel || compileCurrentBBModel(rt);
        const res = await callBridge(
          "texture_pass",
          {
            bbmodel,
            mode: opts.mode || "preserve_and_layer",
            wear_amount: opts.wear_amount !== undefined ? opts.wear_amount : 0.35,
            hue_rotate_deg: opts.hue_rotate_deg || 0.0,
            chroma_scale: opts.chroma_scale !== undefined ? opts.chroma_scale : 1.0,
            lightness_offset: opts.lightness_offset || 0.0,
            selected_boxes: opts.selected_boxes,
            selected_faces: opts.selected_faces
          },
          bridgeOptions
        );
        const outcome = applyTextureLayersToBlockbench(res, rt, "SyncCraft: Non-Destructive Texture Pass");
        return { response: res, outcome };
      }
    };

    function onload() {
      if (!rt.Action) return;
      const defs = [
        ["synccraft_analyze", "SyncCraft: Analyze Model", "insights", () => handlers.analyze()],
        ["synccraft_check", "SyncCraft: Check Rig & Motion", "fact_check", () => handlers.check()],
        ["synccraft_generate_locomotion", "SyncCraft: Generate Locomotion", "directions_walk", () => handlers.generateLocomotion()],
        ["synccraft_retarget", "SyncCraft: Retarget Animation", "swap_horiz", () => handlers.retarget({})],
        ["synccraft_secondary_motion", "SyncCraft: Add Secondary Motion", "air", () => handlers.addSecondaryMotion()],
        ["synccraft_texture_pass", "SyncCraft: Non-Destructive Texture Pass", "layers", () => handlers.texturePass()]
      ];
      for (const [id, name, icon, click] of defs) {
        const act = new rt.Action(id, { name, icon, category: "tools", click });
        registeredActions.push(act);
        if (rt.MenuBar && typeof rt.MenuBar.addAction === "function") {
          rt.MenuBar.addAction(act, "tools");
        }
      }
    }

    function onunload() {
      while (registeredActions.length > 0) {
        const act = registeredActions.pop();
        if (act && typeof act.delete === "function") {
          act.delete();
        }
      }
    }

    return { handlers, onload, onunload, registeredActions };
  }

  const pluginRegistry = (typeof BBPlugin !== "undefined" && BBPlugin) || (typeof Plugin !== "undefined" && Plugin) || null;
  if (pluginRegistry && typeof pluginRegistry.register === "function") {
    const controller = createSyncCraftPluginController(globalThis);
    pluginRegistry.register("synccraft_assistant", {
      title: "SyncCraft Assistant",
      author: "SyncCraft Team",
      description: "Verified Froude locomotion, zero-ML rig/motion audit, cross-rig retargeting, and non-destructive texture layers via localhost SyncCraft bridge.",
      icon: "precision_manufacturing",
      version: "0.4.1",
      variant: "both",
      onload: controller.onload,
      onunload: controller.onunload
    });
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      DEFAULT_BRIDGE_URL,
      callBridge,
      compileCurrentBBModel,
      applyVerifiedAnimationToBlockbench,
      applyTextureLayersToBlockbench,
      createSyncCraftPluginController
    };
  }
})();
