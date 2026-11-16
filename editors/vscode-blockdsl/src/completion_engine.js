// @ts-check
"use strict";

const catalog = require("./catalog");
const { getCursorContext } = require("./symbol_index");

/**
 * Build portable completion items `{ label, kind, detail, documentation, insertText, isSnippet, sortText }`
 * for the given indexed Block-DSL document and cursor character offset.
 *
 * `kind` strings map cleanly to `vscode.CompletionItemKind` in `extension.js`.
 *
 * @param {ReturnType<import("./symbol_index").indexDocument>} docIndex
 * @param {number} offset
 */
function provideCompletions(docIndex, offset) {
  const ctx = getCursorContext(docIndex, offset);
  const { symbols } = docIndex;
  const items = [];

  function add(label, kind, detail, opts) {
    const o = opts || {};
    items.push({
      label,
      kind,
      detail: detail || "",
      documentation: o.doc || "",
      insertText: o.insertText !== undefined ? o.insertText : label,
      isSnippet: !!o.isSnippet,
      sortText: o.sortText || `5_${label}`
    });
  }

  // 0. `@` prefix anywhere at top level or start of line -> suggest @sections
  if (/^\s*@[A-Za-z_]*$/.test(ctx.linePrefix)) {
    for (const sec of catalog.TOP_SECTIONS) {
      add(sec.name, "Keyword", sec.detail, {
        doc: sec.doc,
        insertText: sec.insertText.slice(1), // avoid doubling `@` when user already typed `@`
        isSnippet: true,
        sortText: `0_${sec.name}`
      });
    }
    return dedupeItems(items);
  }

  // 1. Value / Target completions (`prop: <cursor>`, `arg=<cursor>`, `list: [<cursor>`, `track <cursor>`)
  if (ctx.valueTarget) {
    const vt = ctx.valueTarget;
    if (vt.kind === "drive_subject") {
      if (vt.keyword === "track" || vt.keyword === "pose_bias" || vt.keyword === "spring") {
        for (const b of symbols.expandedBones) {
          if (b.isRoot) continue;
          add(b.name, "Struct", b.expanded ? `Expanded bone (${b.origin})` : "Declared child bone", {
            sortText: b.expanded ? `1_${b.name}` : `0_${b.name}`
          });
        }
        return dedupeItems(items);
      }
      if (vt.keyword === "muscle_drive" || vt.keyword === "muscle_tone") {
        for (const m of symbols.muscles) {
          add(m.name, "Function", "Declared @muscles actuator", { sortText: `0_${m.name}` });
        }
        return dedupeItems(items);
      }
      if (vt.keyword === "ik_drive") {
        for (const ik of symbols.ikChains) {
          add(ik.name, "Interface", "Declared @kinematics ik_chain", { sortText: `0_${ik.name}` });
        }
        return dedupeItems(items);
      }
      if (vt.keyword === "material_mod") {
        for (const p of symbols.palettes) {
          add(p.name, "Color", "Declared @palette slot", { sortText: `0_${p.name}` });
        }
        return dedupeItems(items);
      }
    }

    if (vt.kind === "key_bone") {
      for (const b of symbols.expandedBones) {
        add(b.name, "Struct", b.isRoot ? "Root bone" : (b.expanded ? `Expanded bone (${b.origin})` : "Child bone"), {
          sortText: b.expanded ? `1_${b.name}` : `0_${b.name}`
        });
      }
      return dedupeItems(items);
    }

    if (vt.kind === "key_channel") {
      for (const ch of catalog.KEYFRAME_CHANNELS) {
        add(ch, "Property", `Keyframe ${ch} channel`, {
          insertText: `${ch} { 0: [0, 0, 0], \${1:1.0}: [\${2:0, 0, 0}], \${3|catmullrom,linear|} }`,
          isSnippet: true,
          sortText: `0_${ch}`
        });
      }
      return dedupeItems(items);
    }

    // Track / drive law completions right after `track <bone>: <cursor>`
    if (vt.kind === "prop" && ctx.section === "@behavior" && ctx.containerKind === "clip") {
      if (/\btrack\s+[A-Za-z_][A-Za-z0-9_]*\s*:\s*[A-Za-z_]*$/.test(ctx.linePrefix)) {
        for (const law of catalog.TRACK_LAWS) {
          const defaultAmp = law === "spin" ? "360" : "20";
          add(
            law,
            "Function",
            law === "spin"
              ? "Continuous linear rotation [-1440..1440 deg] (gears, wheels, rotors)"
              : `${law} harmonic/impulse law [-120..120 deg]`,
            {
              insertText: `${law}(axis=\${1|x,y,z|}, amp_deg=\${2:${defaultAmp}}, freq_hz=\${3:0.5})`,
              isSnippet: true,
              doc: catalog.HOVER_DOCUMENTATION[law] || "",
              sortText: `0_${law}`
            }
          );
        }
        return dedupeItems(items);
      }
      if (/\bmuscle_drive\s+[A-Za-z_][A-Za-z0-9_]*\s*:\s*[A-Za-z_]*$/.test(ctx.linePrefix)) {
        add("activation", "Function", "Zajac/Hill muscle activation drive", {
          insertText: "activation(peak=${1:0.85}, freq_hz=${2:0.5}, phase_deg=${3:0}, baseline=${4:0.05})",
          isSnippet: true,
          sortText: "0_activation"
        });
        return dedupeItems(items);
      }
      if (/\bik_drive\s+[A-Za-z_][A-Za-z0-9_]*\s*:\s*[A-Za-z_]*$/.test(ctx.linePrefix)) {
        add("foot_cycle", "Function", "World-anchored stance/swing IK drive (< 0.02u slip)", {
          insertText: "foot_cycle(stride=${1:4.5}, lift=${2:1.8}, phase_deg=${3:0})",
          isSnippet: true,
          doc: catalog.HOVER_DOCUMENTATION.foot_cycle,
          sortText: "0_foot_cycle"
        });
        return dedupeItems(items);
      }
    }

    const keyName = vt.name;
    if (keyName === "archetype") {
      catalog.ARCHETYPES.forEach((v, idx) => add(v, "EnumMember", "@meta archetype", { sortText: `0_${idx}_${v}` }));
      return dedupeItems(items);
    }
    if (keyName === "target_profile") {
      catalog.PROFILES.forEach((v, idx) =>
        add(
          v,
          "EnumMember",
          catalog.ANIMATED_PROFILES.includes(v) ? "Supports animated @behavior clips" : "Static geometry profile ([-16, 32] bounds)",
          { sortText: `0_${idx}_${v}` }
        )
      );
      return dedupeItems(items);
    }
    if (keyName === "audit_profile") {
      add("strict", "EnumMember", "Default flush P1-02 contact & zero SAT overlap (Gate G2)", { doc: catalog.HOVER_DOCUMENTATION.audit_profile, sortText: "0_0_strict" });
      add("sculpted", "EnumMember", "ADR 0007 sloped miters, >60° horns, blend_with seams (0 swallowed boxes)", { doc: catalog.HOVER_DOCUMENTATION.audit_profile, sortText: "0_1_sculpted" });
      add("authored", "EnumMember", "Reserved for decompiled human imports only", { doc: catalog.HOVER_DOCUMENTATION.audit_profile, sortText: "0_2_authored" });
      return dedupeItems(items);
    }
    if (keyName === "dsl_version") {
      catalog.DSL_VERSIONS.forEach((v, idx) => add(v, "Constant", "Block-DSL schema version", { sortText: `0_${idx}` }));
      return dedupeItems(items);
    }
    if (keyName === "size_step") {
      catalog.ALLOWED_SIZE_STEPS.forEach((v, idx) => add(v, "Constant", "Discrete size grid step", { sortText: `0_${String(idx).padStart(2, "0")}` }));
      return dedupeItems(items);
    }
    if (keyName === "material" || keyName === "mat") {
      for (const p of symbols.palettes) {
        add(p.name, "Color", "Declared @palette color ramp", { sortText: `0_${p.name}` });
      }
      return dedupeItems(items);
    }
    if (keyName === "texture") {
      for (const t of symbols.textures) {
        add(t.name, "File", "Declared @textures image sheet", { sortText: `0_${t.name}` });
      }
      return dedupeItems(items);
    }
    if (keyName === "cullface") {
      catalog.FACES.forEach((f, idx) => add(f, "EnumMember", "Minecraft face direction", { sortText: `0_${idx}_${f}` }));
      return dedupeItems(items);
    }
    if (keyName === "blend_with") {
      const curBoxName = ctx.containerKind === "box" || ctx.containerKind === "plane" ? ctx.containerName : null;
      for (const bx of symbols.expandedBoxes) {
        if (bx.name === curBoxName) continue;
        const isTwinOrRadial = bx.expanded && curBoxName && bx.name.endsWith(`__${curBoxName}`);
        add(
          bx.name,
          "Field",
          isTwinOrRadial
            ? `Mirrored/radial counterpart (${bx.origin})`
            : `Box in bone '${bx.boneName}'${bx.expanded ? ` (${bx.origin})` : ""}`,
          { sortText: isTwinOrRadial ? `0_0_${bx.name}` : (bx.expanded ? `0_2_${bx.name}` : `0_1_${bx.name}`) }
        );
      }
      return dedupeItems(items);
    }
    if (keyName === "parent_box" || keyName === "parent") {
      // Suggest elements from the enclosing or parent bone first!
      const candidateParentBone = ctx.headerBoneParent || ctx.enclosingBone;
      const prioritized = new Set();
      if (candidateParentBone) {
        for (const bx of candidateParentBone.boxes || []) {
          prioritized.add(bx.name);
          add(bx.name, "Field", `Element in parent bone '${candidateParentBone.name}'`, { sortText: `0_0_${bx.name}` });
        }
        for (const loc of candidateParentBone.locators || []) {
          prioritized.add(loc.name);
          add(loc.name, "Reference", `Locator in parent bone '${candidateParentBone.name}'`, { sortText: `0_1_${loc.name}` });
        }
      }
      for (const bx of symbols.boxes.concat(symbols.planes)) {
        if (!prioritized.has(bx.name)) {
          add(bx.name, "Field", `Element in bone '${bx.boneName || "?"}'`, { sortText: `1_${bx.name}` });
        }
      }
      return dedupeItems(items);
    }
    if (keyName === "base_box") {
      if (ctx.enclosingBone) {
        for (const bx of ctx.enclosingBone.boxes || []) {
          if (bx.offset < offset) {
            add(bx.name, "Field", `Earlier element in bone '${ctx.enclosingBone.name}'`, { sortText: `0_${bx.name}` });
          }
        }
      }
      return dedupeItems(items);
    }
    if (keyName === "role") {
      catalog.LOCATOR_ROLES.forEach((r, idx) => add(r, "EnumMember", "Semantic locator socket role", { sortText: `0_${idx}_${r}` }));
      return dedupeItems(items);
    }
    if (keyName === "normal" || keyName === "plane_normal" || keyName === "action_axis" || keyName === "hinge_axis" || keyName === "axis") {
      const axes = ctx.activeCall === "mirror" ? catalog.MIRROR_AXES : catalog.AXES_3D;
      axes.forEach((ax) => add(ax, "Constant", `${ax.toUpperCase()} coordinate axis`, { sortText: `0_${ax}` }));
      if (ctx.containerKind === "chain" && keyName === "axis") {
        ["+x", "-x", "+y", "-y", "+z", "-z"].forEach((ax, idx) => add(ax, "Constant", "Signed chain step direction", { sortText: `0_1_${idx}` }));
      }
      return dedupeItems(items);
    }
    if (keyName === "family") {
      for (const fam of catalog.MATERIAL_FAMILIES) {
        add(fam.name, "EnumMember", fam.detail, {
          doc: `${fam.detail}\n\nRecommended parameters: \`${fam.params}\``,
          sortText: `0_${fam.name}`
        });
      }
      return dedupeItems(items);
    }
    if (keyName === "target_encoding") {
      catalog.TARGET_ENCODINGS.forEach((enc, idx) => add(enc, "EnumMember", "@style output encoding", { sortText: `0_${idx}_${enc}` }));
      return dedupeItems(items);
    }
    if (keyName === "quantize_ramp" || keyName === "emissive" || keyName === "detached" || keyName === "loop") {
      add("true", "Keyword", "Boolean literal", { sortText: "0_true" });
      add("false", "Keyword", "Boolean literal", { sortText: "0_false" });
      return dedupeItems(items);
    }
    if (keyName === "origin_bone" || keyName === "insertion_bone" || keyName === "upper" || keyName === "lower" || keyName === "bones") {
      const allowRoot = keyName === "origin_bone";
      for (const b of symbols.expandedBones) {
        if (!allowRoot && b.isRoot) continue;
        add(
          b.name,
          "Struct",
          b.isRoot ? "Root bone" : (b.expanded ? `Expanded bone (${b.origin})` : `Child of '${b.parentBoneName || "root"}'`),
          { sortText: b.expanded ? `0_1_${b.name}` : `0_0_${b.name}` }
        );
      }
      return dedupeItems(items);
    }
    if (keyName === "belly_box") {
      for (const bx of symbols.expandedBoxes) {
        add(bx.name, "Field", `Box in '${bx.boneName}'${bx.expanded ? ` (${bx.origin})` : ""}`, {
          sortText: bx.expanded ? `0_1_${bx.name}` : `0_0_${bx.name}`
        });
      }
      return dedupeItems(items);
    }
    if (keyName === "tendon_coupling") {
      for (const b of symbols.expandedBones) {
        if (b.isRoot) continue;
        add(b.name, "Struct", "Coupled non-root bone", { sortText: `0_0_${b.name}` });
      }
      catalog.AXES_3D.forEach((ax) => add(ax, "Constant", "Coupled rotation axis", { sortText: `0_1_${ax}` }));
      return dedupeItems(items);
    }
    if (keyName === "initial") {
      for (const c of symbols.clips) add(c.name, "Function", "Declared @behavior clip", { sortText: `0_0_${c.name}` });
      for (const st of symbols.states) add(st.name, "Enum", "Declared @behavior state", { sortText: `0_1_${st.name}` });
      return dedupeItems(items);
    }
  }

  // 2. Named arguments inside active function call `fn( ... <cursor> )`
  if (ctx.activeCall) {
    const fn = ctx.activeCall;
    if (fn === "attach") {
      add("parent_box=", "Property", "Reference box/plane/locator in parent bone", { insertText: "parent_box=", sortText: "0_1" });
      add("anchor=", "Property", "Normalized point [-1..1]^3 on parent_box", { insertText: "anchor=[${1:0, 1, 0}]", isSnippet: true, sortText: "0_2" });
      add("align=", "Property", "Normalized point [-1..1]^3 on child primary box (-anchor for flush)", { insertText: "align=[${1:0, -1, 0}]", isSnippet: true, sortText: "0_3" });
      add("offset=", "Property", "Fine pivot offset [-0.5..0.5]^3", { insertText: "offset=[${1:0, 0, 0}]", isSnippet: true, sortText: "0_4" });
      add("detached=", "Property", "Authorize hovering bone without physical contact", { insertText: "detached=${1|false,true|}", isSnippet: true, sortText: "0_5" });
      return dedupeItems(items);
    }
    if (fn === "joint") {
      add("parent=", "Property", "Reference box/locator in parent bone", { insertText: "parent=", sortText: "0_1" });
      add("at=", "Property", "Explicit pivot offset [dx, dy, dz] from parent center", { insertText: "at=[${1:0, 2, 0}]", isSnippet: true, sortText: "0_2" });
      add("align=", "Property", "Child primary box alignment [-1..1]^3", { insertText: "align=[${1:0, -1, 0}]", isSnippet: true, sortText: "0_3" });
      add("detached=", "Property", "Authorize detached joint", { insertText: "detached=${1|false,true|}", isSnippet: true, sortText: "0_4" });
      return dedupeItems(items);
    }
    if (fn === "place") {
      add("base_box=", "Property", "Earlier box/plane in the same bone", { insertText: "base_box=", sortText: "0_1" });
      add("anchor=", "Property", "Normalized point [-1..1]^3 on base_box", { insertText: "anchor=[${1:0, 1, 0}]", isSnippet: true, sortText: "0_2" });
      add("align=", "Property", "Normalized point [-1..1]^3 on this box (-anchor for flush)", { insertText: "align=[${1:0, -1, 0}]", isSnippet: true, sortText: "0_3" });
      add("offset=", "Property", "Local translation offset [dx, dy, dz]", { insertText: "offset=[${1:0, 0, 0}]", isSnippet: true, sortText: "0_4" });
      return dedupeItems(items);
    }
    if (fn === "ramp") {
      add("base_lch=", "Property", "OKLCh [Lightness 0..1, Chroma >=0, Hue deg]", { insertText: "base_lch=[${1:0.55, 0.08, 220}]", isSnippet: true, sortText: "0_1" });
      add("steps=", "Property", "Palette ramp steps [2..256]", { insertText: "steps=${1:8}", isSnippet: true, sortText: "0_2" });
      add("hue_shift=", "Property", "[shadow_deg, highlight_deg]", { insertText: "hue_shift=[${1:-4, 4}]", isSnippet: true, sortText: "0_3" });
      add("emissive=", "Property", "Self-luminous palette flag", { insertText: "emissive=${1|false,true|}", isSnippet: true, sortText: "0_4" });
      return dedupeItems(items);
    }
    if (fn === "process") {
      catalog.MATERIAL_PROCESS_ARGS.forEach((arg, idx) => {
        const snippetVal = arg.name === "family" ? "family=" : `${arg.name}=\${1:${arg.defaultVal}}`;
        add(`${arg.name}=`, "Property", `Range: ${arg.range} (default: ${arg.defaultVal})`, {
          insertText: snippetVal,
          isSnippet: arg.name !== "family",
          sortText: `0_${String(idx).padStart(2, "0")}`
        });
      });
      return dedupeItems(items);
    }
    if (fn === "radial") {
      add("count=", "Property", "Rotational symmetry count >= 2", { insertText: "count=${1:4}", isSnippet: true, sortText: "0_1" });
      add("axis=", "Property", "Rotation axis (x, y, z)", { insertText: "axis=${1|z,y,x|}", isSnippet: true, sortText: "0_2" });
      return dedupeItems(items);
    }
    if (fn === "mirror") {
      add("axis=", "Property", "Reflection plane normal (x or z)", { insertText: "axis=${1|x,z|}", isSnippet: true, sortText: "0_1" });
      add("as=", "Property", "Reflected twin bone name", { insertText: "as=${1:twin_bone}", isSnippet: true, sortText: "0_2" });
      return dedupeItems(items);
    }
    if (catalog.TRACK_LAWS.includes(fn)) {
      const ampRange = fn === "spin" ? "[-1440..1440]" : "[-120..120]";
      add("axis=", "Property", "Rotation axis (x, y, z)", { insertText: "axis=${1|x,y,z|}", isSnippet: true, sortText: "0_1" });
      add("amp_deg=", "Property", `Amplitude in degrees ${ampRange}`, { insertText: `amp_deg=\${1:${fn === "spin" ? "360" : "20"}}`, isSnippet: true, sortText: "0_2" });
      add("freq_hz=", "Property", "Frequency in Hz (0, 20]", { insertText: "freq_hz=${1:0.5}", isSnippet: true, sortText: "0_3" });
      add("phase_deg=", "Property", "Phase offset in degrees", { insertText: "phase_deg=${1:0}", isSnippet: true, sortText: "0_4" });
      add("bias_deg=", "Property", "Static angle bias [-90..90]", { insertText: "bias_deg=${1:0}", isSnippet: true, sortText: "0_5" });
      return dedupeItems(items);
    }
    if (fn === "activation") {
      add("peak=", "Property", "Peak activation [0.0..1.0]", { insertText: "peak=${1:0.85}", isSnippet: true, sortText: "0_1" });
      add("freq_hz=", "Property", "Cycle frequency (0, 20] Hz", { insertText: "freq_hz=${1:0.5}", isSnippet: true, sortText: "0_2" });
      add("phase_deg=", "Property", "Phase offset in degrees", { insertText: "phase_deg=${1:0}", isSnippet: true, sortText: "0_3" });
      add("baseline=", "Property", "Resting baseline tone [0..peak]", { insertText: "baseline=${1:0.05}", isSnippet: true, sortText: "0_4" });
      return dedupeItems(items);
    }
    if (fn === "foot_cycle") {
      add("stride=", "Property", "Horizontal stride length [0..24]", { insertText: "stride=${1:4.5}", isSnippet: true, sortText: "0_1" });
      add("lift=", "Property", "Swing foot lift height [0..16]", { insertText: "lift=${1:1.8}", isSnippet: true, sortText: "0_2" });
      add("phase_deg=", "Property", "Gait phase in degrees (0 vs 180)", { insertText: "phase_deg=${1:0}", isSnippet: true, sortText: "0_3" });
      return dedupeItems(items);
    }
  }

  // 3. Section & Container Body Completions
  if (!ctx.section) {
    for (const sec of catalog.TOP_SECTIONS) {
      add(sec.name, "Keyword", sec.detail, { doc: sec.doc, insertText: sec.insertText, isSnippet: true, sortText: `0_${sec.name}` });
    }
    add("let", "Keyword", "Compile-time numeric constant", { insertText: "let ${1:const_name} = ${2:8}", isSnippet: true, sortText: "1_let" });
    add("def", "Snippet", "Parametric rig bone macro", {
      insertText: "def ${1:macro_name}(${2:span=4}) {\n  box ${3:part} { size: [2, ${2:span}, 2], material: ${4:mat_id} }\n}",
      isSnippet: true,
      sortText: "1_def"
    });
    return dedupeItems(items);
  }

  if (ctx.section === "@meta") {
    add("archetype", "Property", "entity | block_multipart | weapon_3d", { insertText: "archetype: ${1|entity,block_multipart,weapon_3d|}", isSnippet: true, sortText: "0_1" });
    add("target_profile", "Property", "bedrock_entity | geckolib | bbmodel_5 | java_modern | java_legacy", { insertText: "target_profile: ${1|bedrock_entity,geckolib,bbmodel_5,java_modern,java_legacy|}", isSnippet: true, sortText: "0_2" });
    add("texel_density", "Property", "Pixels per block (16, 32, 64)", { insertText: "texel_density: ${1|16,32,64|}", isSnippet: true, sortText: "0_3" });
    add("size_step", "Property", "Sub-unit grid quantum (1.0, 0.5, 0.25)", { insertText: "size_step: ${1|1.0,0.5,0.25,0.125|}", isSnippet: true, sortText: "0_4" });
    add("audit_profile", "Property", "strict (default) | sculpted (ADR 0007)", { insertText: "audit_profile: ${1|strict,sculpted|}", isSnippet: true, doc: catalog.HOVER_DOCUMENTATION.audit_profile, sortText: "0_5" });
    add("dsl_version", "Property", "Block-DSL version string", { insertText: "dsl_version: \"3.2\"", sortText: "0_6" });
    return dedupeItems(items);
  }

  if (ctx.section === "@palette") {
    add("ramp", "Snippet", "Declare perceptual OKLCh ramp", {
      insertText: "${1:slot_name}: ramp(base_lch=[${2:0.55}, ${3:0.08}, ${4:220}], steps=${5:8}, hue_shift=[${6:-4}, ${7:4}])",
      isSnippet: true,
      sortText: "0_ramp"
    });
    add("let", "Keyword", "Local numeric constant", { insertText: "let ${1:hue} = ${2:220}", isSnippet: true, sortText: "1_let" });
    return dedupeItems(items);
  }

  if (ctx.section === "@textures") {
    add("image", "Snippet", "Bind external PNG texture sheet", {
      insertText: "${1:tex_main}: image(\"${2:textures/entity/asset.png}\", ${3:64}, ${4:64})",
      isSnippet: true,
      sortText: "0_image"
    });
    return dedupeItems(items);
  }

  if (ctx.section === "@materials") {
    const bound = new Set(symbols.materials.map((m) => m.name));
    for (const p of symbols.palettes) {
      const isUnbound = !bound.has(p.name);
      add(
        `${p.name}: process(...)`,
        "Color",
        isUnbound ? `Bind physical @materials process for palette '${p.name}'` : `Update process for '${p.name}'`,
        {
          insertText: `${p.name}: process(family=\${1|keratin_scales,brushed_metal,painted_metal,weathered_stone,solid_wood,woven_fabric,polymer_dielectric,patterned_skin,lichen_biofilm,thin_film_iridescence|}, roughness=\${2:0.35}, wear=\${3:0.25})`,
          isSnippet: true,
          sortText: isUnbound ? `0_0_${p.name}` : `0_1_${p.name}`
        }
      );
    }
    return dedupeItems(items);
  }

  if (ctx.section === "@style") {
    add("target_encoding", "Property", "vanilla | labpbr_1_3 | gltf_2_0", { insertText: "target_encoding: ${1|vanilla,labpbr_1_3,gltf_2_0|}", isSnippet: true, sortText: "0_1" });
    add("quantize_ramp", "Property", "Snap shading to discrete OKLCh palette swatches", { insertText: "quantize_ramp: ${1|true,false|}", isSnippet: true, sortText: "0_2" });
    add("bevel_highlight", "Property", "Edge bevel highlight intensity (>= 0)", { insertText: "bevel_highlight: ${1:0.08}", isSnippet: true, sortText: "0_3" });
    add("view_deg", "Property", "Reference specular viewing angle [0..89]", { insertText: "view_deg: ${1:20}", isSnippet: true, sortText: "0_4" });
    return dedupeItems(items);
  }

  if (ctx.section === "@rig") {
    if (ctx.containerKind === "box" || ctx.containerKind === "plane") {
      const defaultMat = symbols.palettes.length > 0 ? symbols.palettes[0].name : "mat_id";
      if (ctx.containerKind === "plane") {
        add("size", "Property", "2D tangential [width, height]", { insertText: "size: [${1:6, 6}]", isSnippet: true, sortText: "0_1" });
        add("normal", "Property", "Zero-thickness quad normal (x, y, z)", { insertText: "normal: ${1|y,x,z|}", isSnippet: true, sortText: "0_2" });
      } else {
        add("size", "Property", "3D nominal [sx, sy, sz]", { insertText: "size: [${1:4, 4, 4}]", isSnippet: true, sortText: "0_1" });
      }
      add("material", "Property", "Palette material slot", { insertText: `material: \${1:${defaultMat}}`, isSnippet: true, sortText: "0_3" });
      add("inflate", "Property", "Physical OBB half-extent expansion (>= 0)", { insertText: "inflate: ${1:0.125}", isSnippet: true, doc: catalog.HOVER_DOCUMENTATION.inflate, sortText: "0_4" });
      add("rotate", "Property", "Local intrinsic XYZ Euler [rx, ry, rz] deg", { insertText: "rotate: [${1:0, 0, 0}]", isSnippet: true, sortText: "0_5" });
      add("pivot", "Property", "Custom local box rotation pivot [px, py, pz]", { insertText: "pivot: [${1:0, 0, 0}]", isSnippet: true, sortText: "0_6" });
      add("blend_with", "Property", "ADR 0007 explicit sculpted miter seam targets", { insertText: "blend_with: [${1:target_box}]", isSnippet: true, doc: catalog.HOVER_DOCUMENTATION.blend_with, sortText: "0_7" });
      add("cullface", "Property", "Hidden face list [down, north, ...]", { insertText: "cullface: [${1:down}]", isSnippet: true, sortText: "0_8" });
      add("uv", "Property", "Explicit box(u, v) atlas origin", { insertText: "uv: box(${1:0}, ${2:0})", isSnippet: true, sortText: "0_9" });
      add("uv_faces", "Property", "Per-face UV rectangle map", { insertText: "uv_faces: { north: [${1:0, 0, 16, 16}] }", isSnippet: true, sortText: "1_0" });
      add("texture", "Property", "External @textures binding", { insertText: "texture: ${1:tex_id}", isSnippet: true, sortText: "1_1" });
      return dedupeItems(items);
    }

    if (ctx.containerKind === "locator") {
      add("offset", "Property", "Bone-local 3D position [x, y, z]", { insertText: "offset: [${1:0, 0, 0}]", isSnippet: true, sortText: "0_1" });
      add("role", "Property", "Semantic socket role", { insertText: "role: ${1|socket,ik_effector,vfx_emitter,camera_focus,hitbox,rider_seat|}", isSnippet: true, sortText: "0_2" });
      return dedupeItems(items);
    }

    if (ctx.containerKind === "section") {
      add("root bone", "Keyword", "Required single root bone of @rig", {
        insertText: "root bone ${1:root_body} {\n  pivot_local: [0, ${2:8}, 0]\n  box ${3:core} { size: [${4:8, 8, 8}], material: ${5:mat_id} }\n}",
        isSnippet: true,
        sortText: "0_root_bone"
      });
      return dedupeItems(items);
    }

    // Inside `root_bone`, `bone`, `chain`, or `macro`
    const defaultMat = symbols.palettes.length > 0 ? symbols.palettes[0].name : "mat_id";
    const boneBoxes = ctx.enclosingBone && ctx.enclosingBone.boxes ? ctx.enclosingBone.boxes : [];
    const lastBox = boneBoxes.length > 0 ? boneBoxes[boneBoxes.length - 1].name : "base_box";

    if (ctx.containerKind === "root_bone") {
      add("pivot_local", "Property", "World-space root pivot [x, y, z]", { insertText: "pivot_local: [0, ${1:12}, 0]", isSnippet: true, sortText: "0_0_pivot" });
    } else {
      add("rest_euler", "Property", "Rest intrinsic XYZ rotation [rx, ry, rz] deg", { insertText: "rest_euler: [${1:0, 0, 0}]", isSnippet: true, sortText: "0_0_rest" });
      add("joint_limits", "Property", "Per-axis angular bounds [[-x,+x], [-y,+y], [-z,+z]]", {
        insertText: "joint_limits: [[${1:-45, 45}], [${2:-30, 30}], [${3:-20, 20}]]",
        isSnippet: true,
        doc: catalog.HOVER_DOCUMENTATION.joint_limits,
        sortText: "0_0_limits"
      });
    }

    if (boneBoxes.length === 0) {
      add("box", "Keyword", "Primary cuboid in bone", {
        insertText: `box \${1:primary_box} { size: [\${2:6, 6, 6}], material: \${3:${defaultMat}} }`,
        isSnippet: true,
        sortText: "0_1_box"
      });
    } else {
      add("box", "Keyword", `Flush-placed cuboid on '${lastBox}'`, {
        insertText: `box \${1:detail_box} place(base_box=\${2:${lastBox}}, anchor=[\${3:0, 1, 0}], align=[\${4:0, -1, 0}]) {\n  size: [\${5:4, 4, 4}], material: \${6:${defaultMat}}\n}`,
        isSnippet: true,
        sortText: "0_1_box"
      });
    }
    add("plane", "Keyword", "Zero-thickness 2D quad (wings, fins, capes)", {
      insertText: `plane \${1:fin_plane} place(base_box=\${2:${lastBox}}, anchor=[\${3:0, 1, 0}], align=[\${4:0, -1, 0}]) {\n  size: [\${5:6, 4}], normal: \${6|x,y,z|}, material: \${7:${defaultMat}}\n}`,
      isSnippet: true,
      sortText: "0_2_plane"
    });
    add("locator", "Keyword", "Semantic 3D attachment/effector/VFX socket", {
      insertText: "locator ${1:socket_pt} { offset: [${2:0, 0, 0}], role: ${3|socket,ik_effector,vfx_emitter,camera_focus,hitbox,rider_seat|} }",
      isSnippet: true,
      sortText: "0_3_locator"
    });
    add("bone", "Keyword", `Child bone attached to '${lastBox}'`, {
      insertText: `bone \${1:child_bone} attach(parent_box=\${2:${lastBox}}, anchor=[\${3:0, 1, 0}], align=[\${4:0, -1, 0}]) {\n  joint_limits: [[-45, 45], [-30, 30], [-20, 20]]\n  box \${5:child_box} { size: [\${6:4, 4, 4}], material: \${7:${defaultMat}} }\n}`,
      isSnippet: true,
      sortText: "0_4_bone"
    });
    add("mirror_x", "Function", "Bilateral X reflection modifier (`twin__child` naming)", {
      insertText: "mirror_x(${1:twin_bone})",
      isSnippet: true,
      doc: catalog.HOVER_DOCUMENTATION.mirror_x,
      sortText: "0_5_mirror_x"
    });
    add("radial", "Function", "Rotational symmetry modifier (`count=K, axis=z`)", {
      insertText: "radial(count=${1:4}, axis=${2|z,y,x|})",
      isSnippet: true,
      doc: catalog.HOVER_DOCUMENTATION.radial,
      sortText: "0_5_radial"
    });
    add("chain", "Keyword", "Serial multi-segment tapered bone chain", {
      insertText: `chain \${1:tail_chain} attach(parent_box=\${2:${lastBox}}, anchor=[0, 0, -1], align=[0, 0, 1]) segments=\${3:4} taper=\${4:0.82} {\n  axis: -z,\n  joint_limits: [[-22, 22], [-28, 28], [-15, 15]]\n  box \${5:seg_box} { size: [4, 3, 5], material: \${6:${defaultMat}} }\n}`,
      isSnippet: true,
      doc: catalog.HOVER_DOCUMENTATION.chain,
      sortText: "0_6_chain"
    });

    // Suggest `use <macro>` for every declared `def` macro in the file!
    for (const [mName] of symbols.macros.entries()) {
      add(`use ${mName}(...)`, "Function", `Instantiate declared macro '${mName}'`, {
        insertText: `use ${mName}(\${1}) as \${2:${mName}_inst} attach(parent_box=\${3:${lastBox}}, anchor=[0, 1, 0], align=[0, -1, 0])`,
        isSnippet: true,
        sortText: `0_7_use_${mName}`
      });
    }

    for (const preset of catalog.FLUSH_CONTACT_PRESETS) {
      add(preset.label, "Snippet", preset.detail, {
        insertText: preset.insertText.replace("base_box}", `${lastBox}}`).replace("parent_box}", `${lastBox}}`),
        isSnippet: true,
        sortText: `1_${preset.label}`
      });
    }
    return dedupeItems(items);
  }

  if (ctx.section === "@muscles") {
    if (ctx.containerKind === "muscle") {
      add("origin_bone", "Property", "Proximal anchor bone", { insertText: "origin_bone: ", sortText: "0_1" });
      add("insertion_bone", "Property", "Actuated non-root child bone", { insertText: "insertion_bone: ", sortText: "0_2" });
      add("action_axis", "Property", "Joint torque axis (x, y, z)", { insertText: "action_axis: ${1|x,y,z|}", isSnippet: true, sortText: "0_3" });
      add("moment_arm_deg", "Property", "Signed lever arm [-90..90] deg", { insertText: "moment_arm_deg: ${1:-35}", isSnippet: true, sortText: "0_4" });
      add("tendon_coupling", "Property", "Biarticular coupling [bone, axis, ratio]", { insertText: "tendon_coupling: [${1:distal_bone}, ${2|x,y,z|}, ${3:-0.4}]", isSnippet: true, sortText: "0_5" });
      add("belly_box", "Property", "Isochoric bulging muscle belly box", { insertText: "belly_box: ", sortText: "0_6" });
      add("bulge_inflate", "Property", "Dynamic inflate amplitude [0..0.35]", { insertText: "bulge_inflate: ${1:0.15}", isSnippet: true, sortText: "0_7" });
    } else {
      add("muscle", "Keyword", "Zajac/Hill muscle-tendon unit", {
        insertText: "muscle ${1:muscle_id} {\n  origin_bone: ${2:origin}, insertion_bone: ${3:insertion}, action_axis: ${4|x,y,z|}, moment_arm_deg: ${5:-35}\n}",
        isSnippet: true,
        doc: catalog.HOVER_DOCUMENTATION["@muscles"],
        sortText: "0_muscle"
      });
    }
    return dedupeItems(items);
  }

  if (ctx.section === "@kinematics") {
    if (ctx.containerKind === "ik_chain") {
      add("bones", "Property", "Ordered parent->child non-root bone chain [2..16]", { insertText: "bones: [${1:thigh_l, shin_l}]", isSnippet: true, sortText: "0_1" });
      add("effector", "Property", "Terminal bone local effector offset", { insertText: "effector: [${1:0, -6, 0.5}]", isSnippet: true, sortText: "0_2" });
      add("hinge_axis", "Property", "Primary knee/elbow hinge axis", { insertText: "hinge_axis: ${1|x,y,z|}", isSnippet: true, sortText: "0_3" });
      add("pole", "Property", "Optional pole vector [px, py, pz]", { insertText: "pole: [${1:0, 0, 1}]", isSnippet: true, sortText: "0_4" });
      add("upper", "Property", "2-bone shorthand upper limb bone", { insertText: "upper: ", sortText: "0_5" });
      add("lower", "Property", "2-bone shorthand lower limb bone", { insertText: "lower: ", sortText: "0_6" });
    } else if (ctx.containerKind === "look_chain") {
      add("bones", "Property", "Distributed gaze bone list [1..12]", { insertText: "bones: [${1:neck, head}]", isSnippet: true, sortText: "0_1" });
      add("weight_decay", "Property", "Distal-to-proximal weight decay [0.05..1.0]", { insertText: "weight_decay: ${1:0.7}", isSnippet: true, sortText: "0_2" });
    } else if (ctx.containerKind === "spring") {
      add("stiffness", "Property", "Spring stiffness [0.1..2000]", { insertText: "stiffness: ${1:36}", isSnippet: true, sortText: "0_1" });
      add("damping", "Property", "Viscous damping [0..500]", { insertText: "damping: ${1:6.0}", isSnippet: true, sortText: "0_2" });
    } else {
      add("ik_chain", "Keyword", "2-bone analytic / CCD N-bone IK chain", {
        insertText: "ik_chain ${1:leg_ik_l} { bones: [${2:thigh_l, shin_l}], effector: [0, -6, 0.5], hinge_axis: x, pole: [0, 0, 1] }",
        isSnippet: true,
        sortText: "0_1_ik"
      });
      add("look_chain", "Keyword", "Distributed multi-bone aim/gaze chain", {
        insertText: "look_chain ${1:head_track} { bones: [${2:neck, head}], weight_decay: 0.7 }",
        isSnippet: true,
        sortText: "0_2_look"
      });
      add("spring", "Keyword", "Inertial secondary-motion spring", {
        insertText: "spring ${1:tail_tip} { stiffness: 36, damping: 6.0 }",
        isSnippet: true,
        sortText: "0_3_spring"
      });
    }
    return dedupeItems(items);
  }

  if (ctx.section === "@behavior") {
    if (ctx.containerKind === "clip") {
      add("duration_s", "Property", "Clip duration in seconds [0.05..60]", { insertText: "duration_s: ${1:2.0}", isSnippet: true, sortText: "0_1" });
      add("loop", "Property", "Loop flag (default true)", { insertText: "loop: ${1|true,false|}", isSnippet: true, sortText: "0_2" });
      add("track", "Keyword", "Analytic wave/flap/hinge/impulse/spin track", {
        insertText: "track ${1:bone_id}: ${2|wave,flap,hinge,impulse,spin|}(axis=${3|x,y,z|}, amp_deg=${4:20}, freq_hz=${5:0.5})",
        isSnippet: true,
        sortText: "0_3_track"
      });
      add("ik_drive", "Keyword", "World-anchored foot_cycle IK drive (< 0.02u slip)", {
        insertText: "ik_drive ${1:ik_chain_id}: foot_cycle(stride=${2:4.5}, lift=${3:1.8}, phase_deg=${4:0})",
        isSnippet: true,
        sortText: "0_4_ik"
      });
      add("muscle_drive", "Keyword", "Zajac/Hill muscle activation drive", {
        insertText: "muscle_drive ${1:muscle_id}: activation(peak=${2:0.85}, freq_hz=${3:0.5}, phase_deg=${4:0}, baseline=${5:0.05})",
        isSnippet: true,
        sortText: "0_5_muscle"
      });
      add("key", "Keyword", "Explicit spline keyframe channel", {
        insertText: "key ${1:bone_id}.${2|rotation,position,scale|} { 0: [0, 0, 0], ${3:1.0}: [${4:15, 0, 0}], ${5|catmullrom,linear|} }",
        isSnippet: true,
        sortText: "0_6_key"
      });
    } else if (ctx.containerKind === "keyframe") {
      catalog.KEYFRAME_INTERPOLATIONS.forEach((interp, idx) => add(interp, "EnumMember", "Keyframe spline interpolation", { sortText: `0_${idx}` }));
    } else if (ctx.containerKind === "state") {
      add("pose_bias", "Keyword", "Additive bone Euler offset [rx, ry, rz]", { insertText: "pose_bias ${1:bone_id}: [${2:15, 0, 0}]", isSnippet: true, sortText: "0_1" });
      add("muscle_tone", "Keyword", "Resting muscle co-contraction tone [0..1]", { insertText: "muscle_tone ${1:muscle_id}: ${2:0.3}", isSnippet: true, sortText: "0_2" });
      add("material_mod", "Keyword", "Dynamic emissive/moisture/wear modulation", {
        insertText: "material_mod ${1:palette_id}: { emissive: ${2:1.2}, moisture: ${3:0.2}, wear_delta: ${4:0.05} }",
        isSnippet: true,
        sortText: "0_3"
      });
    } else if (ctx.containerKind === "material_mod") {
      add("emissive", "Property", "Emissive radiance [0..2]", { insertText: "emissive: ${1:1.2}", isSnippet: true, sortText: "0_1" });
      add("moisture", "Property", "Surface wetness [0..1]", { insertText: "moisture: ${1:0.25}", isSnippet: true, sortText: "0_2" });
      add("wear_delta", "Property", "Incremental wear [0..1]", { insertText: "wear_delta: ${1:0.05}", isSnippet: true, sortText: "0_3" });
    } else {
      add("clip", "Keyword", "Motion clip block", {
        insertText: "clip ${1:idle_loop} {\n  duration_s: ${2:2.0}\n  track ${3:bone_id}: wave(axis=y, amp_deg=12, freq_hz=0.5)\n}",
        isSnippet: true,
        sortText: "0_1_clip"
      });
      add("state", "Keyword", "Morphological / optical state block", {
        insertText: "state ${1:enraged} {\n  pose_bias ${2:bone_id}: [15, 0, 0]\n}",
        isSnippet: true,
        sortText: "0_2_state"
      });
    }
    return dedupeItems(items);
  }

  if (ctx.section === "@controller") {
    const allTargets = symbols.clips.map((c) => c.name).concat(symbols.states.map((s) => s.name));
    const firstState = allTargets[0] || "idle_clip";
    const secondState = allTargets[1] || "active_clip";
    add("initial", "Property", "Initial @behavior clip or state", {
      insertText: `initial: \${1:${firstState}}`,
      isSnippet: true,
      sortText: "0_0_initial"
    });
    add("transition (->)", "Snippet", "State transition rule", {
      insertText: `\${1:${firstState}} -> \${2:${secondState}} when \${3:speed} \${4|>,>=,<,<=,==,!=|} \${5:0.5}, blend \${6:0.25}`,
      isSnippet: true,
      sortText: "0_1_transition"
    });
    for (const tName of allTargets) {
      add(tName, "Function", "Declared @behavior clip/state", { sortText: `1_${tName}` });
    }
    return dedupeItems(items);
  }

  return dedupeItems(items);
}

function dedupeItems(items) {
  const seen = new Set();
  const out = [];
  for (const it of items) {
    const key = `${it.label}::${it.kind}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(it);
    }
  }
  out.sort((a, b) => a.sortText.localeCompare(b.sortText));
  return out;
}

module.exports = {
  provideCompletions
};
