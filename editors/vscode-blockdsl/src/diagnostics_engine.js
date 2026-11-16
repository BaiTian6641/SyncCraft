// @ts-check
"use strict";

const catalog = require("./catalog");

/**
 * Fast, dependency-free instant static diagnostics (< 3 ms on keystroke) so users get
 * immediate feedback even before saving or when Python isn't configured.
 *
 * Checks:
 * 1. Unsupported reserved keywords (`@shade`, `@motion`, `@ergonomics`, `extrude`).
 * 2. Invalid uppercase booleans (`True` / `False`).
 * 3. Missing required sections (`@meta`, `@palette`, `@rig`).
 * 4. Global symbol collisions (including collisions with expanded `mirror_x` / `radial` / `chain` names!).
 * 5. Unknown `material:` palette references.
 * 6. Incompatible `@meta` `entity` + `java_*` or animated `@behavior` `clip` + static `java_*`.
 * 7. Invalid `joint_limits` intervals that exclude `0` (`min > 0` or `max < 0`).
 * 8. Negative `inflate` outside `audit_profile: authored`.
 *
 * @param {ReturnType<import("./symbol_index").indexDocument>} docIndex
 */
function runFastDiagnostics(docIndex) {
  const { masked, lineStarts, scopes, symbols } = docIndex;
  /** @type {Array<{ severity: "error" | "warning", message: string, start: {line: number, character: number}, end: {line: number, character: number}, code: string }>} */
  const diags = [];

  function pushSpan(offset, length, severity, code, message) {
    const startPos = offsetToPos(lineStarts, offset);
    const endPos = offsetToPos(lineStarts, offset + Math.max(1, length));
    diags.push({ severity, code, message, start: startPos, end: endPos });
  }

  // 1. Unsupported reserved keywords
  const unsupRe = /@(?:shade|motion|ergonomics)\b|\bextrude\b/g;
  let m;
  while ((m = unsupRe.exec(masked)) !== null) {
    pushSpan(
      m.index,
      m[0].length,
      "error",
      "unsupported_feature",
      `'${m[0]}' is an unimplemented reserved keyword and raises UnsupportedFeature in SyncCraft Block-DSL v3.2.`
    );
  }

  // 2. Capitalized Python booleans
  const boolCaseRe = /\b(?:True|False)\b/g;
  while ((m = boolCaseRe.exec(masked)) !== null) {
    pushSpan(
      m.index,
      m[0].length,
      "error",
      "boolean_case",
      `Use lowercase '${m[0].toLowerCase()}' in Block-DSL (capitalized '${m[0]}' is invalid).`
    );
  }

  // 3. Required sections check (only when file is non-empty)
  if (masked.trim().length > 0) {
    const presentSections = new Set(scopes.filter((s) => s.kind === "section").map((s) => s.name));
    for (const req of ["@meta", "@palette", "@rig"]) {
      if (!presentSections.has(req)) {
        pushSpan(0, 1, "error", "missing_required_section", `Missing required Block-DSL section '${req}'.`);
      }
    }
  }

  // 4. Global symbol uniqueness (declared + expanded)
  const seenGlobal = new Map();
  for (const decl of symbols.globalDeclarations) {
    const prev = seenGlobal.get(decl.name);
    if (prev) {
      const originDetail = decl.origin ? ` (generated via ${decl.origin})` : "";
      pushSpan(
        decl.offset,
        decl.name.length,
        "error",
        "duplicate_identifier",
        `Duplicate global identifier '${decl.name}'${originDetail}; already declared as ${prev.kind} on line ${prev.position.line + 1}.`
      );
    } else {
      seenGlobal.set(decl.name, decl);
    }
  }

  // 5. Unknown `material:` / `mat:` references inside `@rig`
  const paletteNames = new Set(symbols.palettes.map((p) => p.name));
  if (paletteNames.size > 0) {
    const rigScope = scopes.find((s) => s.kind === "section" && s.name === "@rig");
    if (rigScope) {
      const rigSlice = masked.slice(rigScope.bodyStart, rigScope.bodyEnd);
      const matRefRe = /\b(?:material|mat)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\b/g;
      while ((m = matRefRe.exec(rigSlice)) !== null) {
        const refName = m[1];
        if (!paletteNames.has(refName)) {
          const off = rigScope.bodyStart + m.index + m[0].lastIndexOf(refName);
          pushSpan(off, refName.length, "error", "unknown_material", `Unknown @palette material '${refName}'.`);
        }
      }
    }
  }

  // 6. Profile vs archetype / @behavior compatibility
  const arch = symbols.metaFields.archetype;
  const prof = symbols.metaFields.target_profile;
  const auditProf = symbols.metaFields.audit_profile || "strict";
  const metaScope = scopes.find((s) => s.kind === "section" && s.name === "@meta");
  if (arch === "entity" && (prof === "java_modern" || prof === "java_legacy") && metaScope) {
    pushSpan(
      metaScope.start,
      5,
      "error",
      "incompatible_entity_profile",
      `archetype 'entity' cannot target static '${prof}'; use 'bedrock_entity', 'geckolib', or 'bbmodel_5'.`
    );
  }
  if (symbols.clips.length > 0 && prof && !catalog.ANIMATED_PROFILES.includes(prof)) {
    const firstClip = symbols.clips[0];
    pushSpan(
      firstClip.offset,
      firstClip.name.length,
      "error",
      "animated_clip_on_static_profile",
      `@behavior clips require target_profile in {bedrock_entity, geckolib, bbmodel_5}, not '${prof}'.`
    );
  }

  // 7. `joint_limits` must bracket 0: `-180 <= min <= 0 <= max <= 180`
  const jlRe = /\bjoint_limits\s*:\s*\[\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]\s*,\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]\s*,\s*\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]\s*\]/g;
  while ((m = jlRe.exec(masked)) !== null) {
    const nums = m.slice(1, 7).map(Number);
    for (let ax = 0; ax < 3; ax += 1) {
      const lo = nums[2 * ax];
      const hi = nums[2 * ax + 1];
      if (lo < -180 || lo > 0 || hi < 0 || hi > 180 || lo > hi) {
        pushSpan(
          m.index,
          m[0].length,
          "error",
          "invalid_joint_limits",
          `joint_limits axis ${["X", "Y", "Z"][ax]} [${lo}, ${hi}] must satisfy -180 <= min <= 0 <= max <= 180.`
        );
        break;
      }
    }
  }

  // 8. Negative `inflate` outside `authored`
  if (auditProf !== "authored") {
    const infRe = /\binflate\s*:\s*(-\s*(?:[0-9]+\.?[0-9]*|\.[0-9]+))/g;
    while ((m = infRe.exec(masked)) !== null) {
      const val = parseFloat(m[1].replace(/\s+/g, ""));
      if (val < 0) {
        pushSpan(
          m.index,
          m[0].length,
          "error",
          "negative_inflate",
          `Negative inflate (${val}) is disallowed under audit_profile '${auditProf}' (must be >= 0.0).`
        );
      }
    }
  }

  return diags;
}

function offsetToPos(lineStarts, offset) {
  let lo = 0;
  let hi = lineStarts.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (lineStarts[mid] <= offset) lo = mid + 1;
    else hi = mid - 1;
  }
  const line = Math.max(0, hi);
  return { line, character: Math.max(0, offset - lineStarts[line]) };
}

module.exports = {
  runFastDiagnostics
};
