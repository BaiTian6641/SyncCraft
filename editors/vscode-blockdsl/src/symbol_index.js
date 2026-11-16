// @ts-check
"use strict";

/**
 * Dependency-free Block-DSL v3.2 lexical scanner, scope tracker, and deterministic
 * symbol harvester (`mirror_x`, `mirror`, `radial`, `def`/`use`, `chain` expansion).
 *
 * Designed to run in < 3 ms on 2,000-line `.blockdsl` files inside VS Code's extension
 * host (and testable standalone in JavaScriptCore / Node).
 */

/**
 * Replace `//` and `#` comments and string contents with spaces of identical length
 * so all byte/character offsets and line numbers remain 100% preserved!
 * @param {string} text
 * @returns {string}
 */
function maskCommentsAndStrings(text) {
  const chars = text.split("");
  const n = chars.length;
  let i = 0;
  while (i < n) {
    const ch = chars[i];
    if (ch === "\"") {
      chars[i] = "\"";
      i += 1;
      while (i < n && chars[i] !== "\"" && chars[i] !== "\n") {
        if (chars[i] === "\\" && i + 1 < n) {
          chars[i] = " ";
          chars[i + 1] = " ";
          i += 2;
          continue;
        }
        chars[i] = " ";
        i += 1;
      }
      i += 1;
    } else if (ch === "/" && i + 1 < n && chars[i + 1] === "/") {
      while (i < n && chars[i] !== "\n") {
        chars[i] = " ";
        i += 1;
      }
    } else if (ch === "#") {
      while (i < n && chars[i] !== "\n") {
        chars[i] = " ";
        i += 1;
      }
    } else {
      i += 1;
    }
  }
  return chars.join("");
}

/**
 * Convert character offset to 0-based { line, character } using precomputed line starts.
 * @param {number[]} lineStarts
 * @param {number} offset
 */
function offsetToPosition(lineStarts, offset) {
  let lo = 0;
  let hi = lineStarts.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (lineStarts[mid] <= offset) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  const line = Math.max(0, hi);
  return { line, character: offset - lineStarts[line] };
}

/**
 * Compute line start offsets for fast offset -> { line, character } lookup.
 * @param {string} text
 * @returns {number[]}
 */
function computeLineStarts(text) {
  const starts = [0];
  for (let i = 0; i < text.length; i += 1) {
    if (text.charCodeAt(i) === 10) {
      starts.push(i + 1);
    }
  }
  return starts;
}

/**
 * Clone and prefix a bone tree for `def`/`use` (`_` separator) or `mirror`/`radial` (`__` separator).
 * @param {any} node
 * @param {string} rootName
 * @param {string} descPrefix
 * @param {string} sep
 * @param {string} originNote
 */
function cloneBoneSubtree(node, rootName, descPrefix, sep, originNote) {
  const mappedBoxes = node.boxes.map((b) => ({
    ...b,
    name: `${descPrefix}${sep}${b.name}`,
    expanded: true,
    origin: originNote
  }));
  const mappedLocators = node.locators.map((loc) => ({
    ...loc,
    name: `${descPrefix}${sep}${loc.name}`,
    expanded: true,
    origin: originNote
  }));
  const mappedChildren = node.children.map((child) =>
    cloneBoneSubtree(
      child,
      `${descPrefix}${sep}${child.name}`,
      `${descPrefix}${sep}${child.name}`,
      sep,
      originNote
    )
  );
  return {
    ...node,
    name: rootName,
    boxes: mappedBoxes,
    locators: mappedLocators,
    children: mappedChildren,
    expanded: true,
    origin: originNote,
    symmetryMods: []
  };
}

/**
 * Parse symmetry modifiers (`mirror_x(twin)`, `mirror(axis=z, as=twin)`, `radial(count=K, axis=y)`)
 * from a bone/use/chain header string.
 * @param {string} headerText
 * @param {string} baseBoneName
 */
function parseSymmetryModifiers(headerText, baseBoneName) {
  const mods = [];
  const mxRe = /\bmirror_x\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)/g;
  let m;
  while ((m = mxRe.exec(headerText)) !== null) {
    mods.push({ kind: "mirror_x", twinName: m[1] });
  }
  const mirRe = /\bmirror\s*\(([^)]*)\)/g;
  while ((m = mirRe.exec(headerText)) !== null) {
    const args = m[1];
    const asMatch = /\b(?:as|name)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)/.exec(args);
    let twin = asMatch ? asMatch[1] : null;
    if (!twin) {
      const bareTokens = args
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s && !s.includes("="));
      if (bareTokens.length > 0 && /^[A-Za-z_][A-Za-z0-9_]*$/.test(bareTokens[0])) {
        twin = bareTokens[0];
      }
    }
    mods.push({ kind: "mirror", twinName: twin || `${baseBoneName}_mirror` });
  }
  const radRe = /\bradial\s*\(([^)]*)\)/g;
  while ((m = radRe.exec(headerText)) !== null) {
    const countMatch = /\bcount\s*=\s*(\d+)/.exec(m[1]);
    const count = countMatch ? Math.min(64, Math.max(2, parseInt(countMatch[1], 10))) : 2;
    mods.push({ kind: "radial", count });
  }
  return mods;
}

/**
 * Expand symmetry modifiers on a list of sibling bones (matching `se3_chain.py` / `ast_nodes.py`).
 * @param {any[]} bones
 * @param {{ count: number }} budget
 */
function expandBoneList(bones, budget) {
  const out = [];
  for (const bone of bones) {
    const expandedChildren = expandBoneList(bone.children, budget);
    const baseNode = { ...bone, children: expandedChildren };
    let currentBatch = [baseNode];
    for (const mod of bone.symmetryMods || []) {
      const nextBatch = [];
      for (const item of currentBatch) {
        nextBatch.push(item);
        if (budget.count > 4096) {
          continue;
        }
        if (mod.kind === "mirror_x" || mod.kind === "mirror") {
          const twinName = item === baseNode ? mod.twinName : `${mod.twinName}__${item.name}`;
          budget.count += 1;
          nextBatch.push(
            cloneBoneSubtree(item, twinName, twinName, "__", `${mod.kind} of ${item.name}`)
          );
        } else if (mod.kind === "radial") {
          for (let k = 1; k < mod.count; k += 1) {
            if (budget.count > 4096) {
              break;
            }
            const cloneName = `${item.name}_r${k}`;
            budget.count += 1;
            nextBatch.push(
              cloneBoneSubtree(item, cloneName, cloneName, "__", `radial[${k}/${mod.count}] of ${item.name}`)
            );
          }
        }
      }
      currentBatch = nextBatch;
    }
    for (const item of currentBatch) {
      out.push(item);
    }
  }
  return out;
}

/**
 * Index an entire Block-DSL document: sections, brace scopes, declared symbols, and
 * expanded rig identifiers (`mirror_x`, `mirror`, `radial`, `def`/`use`, `chain`).
 * @param {string} text
 */
function indexDocument(text) {
  const masked = maskCommentsAndStrings(text);
  const lineStarts = computeLineStarts(text);

  /** @type {Array<{kind: string, name: string, start: number, end: number, bodyStart: number, bodyEnd: number, parentBone: any, meta: any}>} */
  const scopes = [];
  const stack = [];

  // Scan top-level and nested `{ ... }` blocks with header classification
  let headerStart = 0;
  for (let i = 0; i < masked.length; i += 1) {
    const ch = masked[i];
    if (ch === "{") {
      const rawHeader = masked.slice(headerStart, i).trim();
      const parentScope = stack.length > 0 ? stack[stack.length - 1] : null;
      const scopeInfo = classifyBlockHeader(rawHeader, headerStart, i, parentScope, masked);
      stack.push(scopeInfo);
      headerStart = i + 1;
    } else if (ch === "}") {
      const closed = stack.pop();
      if (closed) {
        closed.bodyEnd = i;
        closed.end = i + 1;
        scopes.push(closed);
      }
      headerStart = i + 1;
    } else if (ch === ";" || ch === "\n") {
      // Advance headerStart across standalone single-line statements without `{`
      const seg = masked.slice(headerStart, i).trim();
      if (
        seg &&
        /^(?:param\b|let\b|budget\b|envelope\b|material_slots\b|uv_policy\b|use\b|bind\b|track\b|slide\b|muscle_drive\b|ik_drive\b|pose_bias\b|muscle_tone\b|initial\b|[A-Za-z_][A-Za-z0-9_]*\s*:\s*(?:ramp|image|process)\b|[A-Za-z_][A-Za-z0-9_]*\s*->)/.test(seg) &&
        !seg.includes("{")
      ) {
        // Only reset if parentheses are balanced (multi-line call stays intact)
        const openParens = (seg.match(/\(/g) || []).length;
        const closeParens = (seg.match(/\)/g) || []).length;
        if (openParens <= closeParens) {
          headerStart = i + 1;
        }
      }
    }
  }
  // Unclosed blocks while user is actively typing!
  while (stack.length > 0) {
    const unclosed = stack.pop();
    unclosed.bodyEnd = masked.length;
    unclosed.end = masked.length;
    scopes.push(unclosed);
  }
  scopes.sort((a, b) => a.start - b.start);

  const symbols = {
    metaFields: {},
    lets: [],
    macros: new Map(),
    components: new Map(),
    instances: [],
    textures: [],
    palettes: [],
    materials: [],
    bones: [],
    boxes: [],
    planes: [],
    locators: [],
    muscles: [],
    ikChains: [],
    lookChains: [],
    springs: [],
    clips: [],
    states: [],
    expandedBones: [],
    expandedBoxes: [],
    expandedLocators: [],
    globalDeclarations: []
  };

  // Extract `@meta` key-value fields
  const metaScope = scopes.find((s) => s.kind === "section" && s.name === "@meta");
  if (metaScope) {
    const metaBody = text.slice(metaScope.bodyStart, metaScope.bodyEnd);
    const kvRe = /\b(archetype|target_profile|texel_density|units_per_block|dsl_version|size_step|audit_profile)\s*:\s*("[^"]*"|[A-Za-z0-9_.\-]+)/g;
    let m;
    while ((m = kvRe.exec(metaBody)) !== null) {
      symbols.metaFields[m[1]] = m[2].replace(/^"|"$/g, "");
    }
  }

  // Extract `let` and `param` bindings
  const letRe = /\b(?:let|param)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=/g;
  let m;
  while ((m = letRe.exec(masked)) !== null) {
    const nameOffset = m.index + m[0].indexOf(m[1]);
    symbols.lets.push({
      name: m[1],
      kind: "let",
      offset: nameOffset,
      position: offsetToPosition(lineStarts, nameOffset)
    });
  }

  // Extract `@textures`, `@palette`, and `@materials` entries
  for (const sec of scopes) {
    if (sec.kind !== "section") {
      continue;
    }
    const sub = masked.slice(sec.bodyStart, sec.bodyEnd);
    if (sec.name === "@textures") {
      const re = /\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*image\s*\(/g;
      while ((m = re.exec(sub)) !== null) {
        const off = sec.bodyStart + m.index;
        const item = { name: m[1], kind: "texture", offset: off, position: offsetToPosition(lineStarts, off) };
        symbols.textures.push(item);
      }
    } else if (sec.name === "@palette") {
      const re = /\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*ramp\s*\(/g;
      while ((m = re.exec(sub)) !== null) {
        const off = sec.bodyStart + m.index;
        const item = { name: m[1], kind: "palette", offset: off, position: offsetToPosition(lineStarts, off) };
        symbols.palettes.push(item);
        symbols.globalDeclarations.push(item);
      }
    } else if (sec.name === "@materials") {
      const re = /\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*process\s*\(/g;
      while ((m = re.exec(sub)) !== null) {
        const off = sec.bodyStart + m.index;
        symbols.materials.push({ name: m[1], kind: "material_process", offset: off, position: offsetToPosition(lineStarts, off) });
      }
    }
  }

  // Build hierarchical bone/macro AST from scopes so we can expand `use`, `chain`, `mirror_x`, `mirror`, `radial`
  const boneMap = new Map();
  const rootBones = [];

  for (const sc of scopes) {
    if (sc.kind === "macro") {
      symbols.macros.set(sc.name, {
        name: sc.name,
        kind: "macro",
        offset: sc.nameOffset,
        position: offsetToPosition(lineStarts, sc.nameOffset),
        boxes: [],
        locators: [],
        children: [],
        scope: sc
      });
    } else if (sc.kind === "component") {
      const compBody = masked.slice(sc.bodyStart, sc.bodyEnd);
      const msMatch = /\bmaterial_slots\s*:\s*\[([^\]]*)\]/.exec(compBody);
      const slots = msMatch
        ? msMatch[1].split(",").map((s) => s.trim().split("=")[0].trim()).filter(Boolean)
        : [];
      const cDef = {
        name: sc.name,
        kind: "component",
        offset: sc.nameOffset,
        position: offsetToPosition(lineStarts, sc.nameOffset),
        boxes: [],
        locators: [],
        frames: [],
        materialSlots: slots,
        children: [],
        scope: sc
      };
      symbols.components.set(sc.name, cDef);
      symbols.globalDeclarations.push({
        name: sc.name,
        kind: "component",
        offset: sc.nameOffset,
        position: offsetToPosition(lineStarts, sc.nameOffset)
      });
    } else if (sc.kind === "instance") {
      const instRec = {
        name: sc.name,
        kind: "instance",
        componentName: sc.meta.componentName || "",
        symmetryMods: sc.meta.symmetryMods || [],
        offset: sc.nameOffset,
        position: offsetToPosition(lineStarts, sc.nameOffset)
      };
      symbols.instances.push(instRec);
      symbols.globalDeclarations.push(instRec);
    } else if (sc.kind === "bone" || sc.kind === "root_bone") {
      const bNode = {
        name: sc.name,
        kind: sc.kind,
        isRoot: sc.kind === "root_bone",
        offset: sc.nameOffset,
        position: offsetToPosition(lineStarts, sc.nameOffset),
        boxes: [],
        locators: [],
        children: [],
        symmetryMods: sc.meta.symmetryMods || [],
        parentBoneName: null,
        scope: sc
      };
      boneMap.set(sc, bNode);
      if (!sc.inMacro) {
        symbols.bones.push(bNode);
        symbols.globalDeclarations.push(bNode);
      }
    } else if (sc.kind === "chain") {
      const cNode = {
        name: sc.name,
        kind: "chain",
        offset: sc.nameOffset,
        position: offsetToPosition(lineStarts, sc.nameOffset),
        segments: sc.meta.segments || 1,
        symmetryMods: sc.meta.symmetryMods || [],
        boxes: [],
        locators: [],
        scope: sc
      };
      boneMap.set(sc, cNode);
    }
  }

  // Pass 1: Attach boxes, planes, locators, frames, and non-rig domain symbols so `chain`, `macro`, & `component`
  // templates already contain their elements before Pass 2 expands `chain`, `use`, and `@assembly` instances!
  for (const sc of scopes) {
    if (sc.kind === "box" || sc.kind === "plane" || sc.kind === "locator" || sc.kind === "frame") {
      const owner = findEnclosingOwner(sc, scopes, boneMap, symbols.macros, symbols.components);
      const item = {
        name: sc.name,
        kind: sc.kind,
        offset: sc.nameOffset,
        position: offsetToPosition(lineStarts, sc.nameOffset),
        boneName: owner ? owner.name : null
      };
      if (owner) {
        if (sc.kind === "frame") {
          if (owner.frames) owner.frames.push(item);
        } else if (sc.kind === "locator") {
          owner.locators.push(item);
        } else {
          owner.boxes.push(item);
        }
      }
      if (!sc.inMacro && sc.kind !== "frame" && (!owner || owner.kind !== "chain")) {
        if (sc.kind === "box") symbols.boxes.push(item);
        else if (sc.kind === "plane") symbols.planes.push(item);
        else symbols.locators.push(item);
        symbols.globalDeclarations.push(item);
      }
    } else if (sc.kind === "muscle") {
      const item = { name: sc.name, kind: "muscle", offset: sc.nameOffset, position: offsetToPosition(lineStarts, sc.nameOffset) };
      symbols.muscles.push(item);
      symbols.globalDeclarations.push(item);
    } else if (sc.kind === "ik_chain") {
      const item = { name: sc.name, kind: "ik_chain", offset: sc.nameOffset, position: offsetToPosition(lineStarts, sc.nameOffset) };
      symbols.ikChains.push(item);
      symbols.globalDeclarations.push(item);
    } else if (sc.kind === "look_chain") {
      const item = { name: sc.name, kind: "look_chain", offset: sc.nameOffset, position: offsetToPosition(lineStarts, sc.nameOffset) };
      symbols.lookChains.push(item);
      symbols.globalDeclarations.push(item);
    } else if (sc.kind === "spring") {
      symbols.springs.push({ name: sc.name, kind: "spring", offset: sc.nameOffset, position: offsetToPosition(lineStarts, sc.nameOffset) });
    } else if (sc.kind === "clip") {
      const item = { name: sc.name, kind: "clip", offset: sc.nameOffset, position: offsetToPosition(lineStarts, sc.nameOffset) };
      symbols.clips.push(item);
      symbols.globalDeclarations.push(item);
    } else if (sc.kind === "state") {
      const item = { name: sc.name, kind: "state", offset: sc.nameOffset, position: offsetToPosition(lineStarts, sc.nameOffset) };
      symbols.states.push(item);
      symbols.globalDeclarations.push(item);
    }
  }

  // Pass 2: Wire bone & chain parent-child hierarchy (now with `cNode.boxes` populated!)
  for (const sc of scopes) {
    if (sc.kind === "bone" || sc.kind === "root_bone") {
      const bNode = boneMap.get(sc);
      const owner = findEnclosingOwner(sc, scopes, boneMap, symbols.macros, symbols.components);
      if (owner && (owner.kind === "bone" || owner.kind === "root_bone" || owner.kind === "macro" || owner.kind === "component")) {
        bNode.parentBoneName = owner.name;
        owner.children.push(bNode);
      } else if (!sc.inMacro) {
        rootBones.push(bNode);
      }
    } else if (sc.kind === "chain") {
      const cNode = boneMap.get(sc);
      const owner = findEnclosingOwner(sc, scopes, boneMap, symbols.macros, symbols.components);
      if (owner && owner.children) {
        let firstSeg = null;
        let prevSeg = null;
        for (let s = 1; s <= cNode.segments; s += 1) {
          const segBoneName = `${cNode.name}_${s}`;
          const segBoxes = cNode.boxes.map((b) => ({
            ...b,
            name: `${segBoneName}_${b.name}`,
            expanded: true,
            origin: `chain ${cNode.name} seg ${s}`
          }));
          const segLocs = cNode.locators.map((l) => ({
            ...l,
            name: `${segBoneName}_${l.name}`,
            expanded: true,
            origin: `chain ${cNode.name} seg ${s}`
          }));
          const segBone = {
            name: segBoneName,
            kind: "bone",
            isRoot: false,
            offset: cNode.offset,
            position: cNode.position,
            boxes: segBoxes,
            locators: segLocs,
            children: [],
            symmetryMods: s === 1 ? cNode.symmetryMods : [],
            expanded: true,
            origin: `chain ${cNode.name}[${s}/${cNode.segments}]`
          };
          if (!firstSeg) firstSeg = segBone;
          if (prevSeg) prevSeg.children.push(segBone);
          prevSeg = segBone;
        }
        if (firstSeg) {
          owner.children.push(firstSeg);
        }
      }
    }
  }

  // Handle `use <macro>(...) as <alias> [attach|joint] [symmetry_mod*]` statements inside bones
  const useRe = /\buse\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*as\s+([A-Za-z_][A-Za-z0-9_]*)([^\n;{]*)/g;
  while ((m = useRe.exec(masked)) !== null) {
    const macroName = m[1];
    const aliasName = m[2];
    const tail = m[3] || "";
    const useOffset = m.index + m[0].indexOf(aliasName);
    const macroDef = symbols.macros.get(macroName);
    const enclosingScope = findInnermostScopeAt(scopes, m.index, ["bone", "root_bone"]);
    const ownerBone = enclosingScope ? boneMap.get(enclosingScope) : null;
    if (ownerBone) {
      const symMods = parseSymmetryModifiers(tail, aliasName);
      if (macroDef) {
        const instantiated = cloneBoneSubtree(macroDef, aliasName, aliasName, "_", `use ${macroName} as ${aliasName}`);
        instantiated.offset = useOffset;
        instantiated.position = offsetToPosition(lineStarts, useOffset);
        instantiated.symmetryMods = symMods;
        ownerBone.children.push(instantiated);
      } else {
        ownerBone.children.push({
          name: aliasName,
          kind: "bone",
          isRoot: false,
          offset: useOffset,
          position: offsetToPosition(lineStarts, useOffset),
          boxes: [],
          locators: [],
          children: [],
          symmetryMods: symMods,
          expanded: true,
          origin: `use ${macroName} as ${aliasName}`
        });
      }
    }
  }

  // Deterministically expand all `mirror_x`, `mirror`, `radial`, `use`, and `chain` nodes
  const expandedRoots = expandBoneList(rootBones, { count: 0 });
  function collectExpanded(nodes, parentName) {
    for (const b of nodes) {
      symbols.expandedBones.push({
        name: b.name,
        isRoot: !!b.isRoot,
        parentBoneName: parentName,
        expanded: !!b.expanded,
        origin: b.origin || "declared",
        offset: b.offset,
        position: b.position,
        boxes: b.boxes,
        locators: b.locators
      });
      if (b.expanded) {
        symbols.globalDeclarations.push({
          name: b.name,
          kind: "expanded_bone",
          offset: b.offset,
          position: b.position,
          origin: b.origin
        });
      }
      for (const bx of b.boxes) {
        symbols.expandedBoxes.push({ ...bx, boneName: b.name });
        if (bx.expanded) {
          symbols.globalDeclarations.push({
            name: bx.name,
            kind: "expanded_box",
            offset: bx.offset,
            position: bx.position,
            origin: bx.origin
          });
        }
      }
      for (const loc of b.locators) {
        symbols.expandedLocators.push({ ...loc, boneName: b.name });
        if (loc.expanded) {
          symbols.globalDeclarations.push({
            name: loc.name,
            kind: "expanded_locator",
            offset: loc.offset,
            position: loc.position,
            origin: loc.origin
          });
        }
      }
      collectExpanded(b.children, b.name);
    }
  }
  collectExpanded(expandedRoots, null);

  // Expand v4.0 `@assembly` component instances (including `mirror_x(twin)` clones & dotted refs)
  for (let iIdx = 0; iIdx < symbols.instances.length; iIdx += 1) {
    const inst = symbols.instances[iIdx];
    const compDef = symbols.components.get(inst.componentName);
    const variantNames = [{ name: inst.name, expanded: false, origin: `instance ${inst.name} : ${inst.componentName}` }];
    for (const mod of inst.symmetryMods || []) {
      if ((mod.kind === "mirror_x" || mod.kind === "mirror") && mod.twinName) {
        variantNames.push({
          name: mod.twinName,
          expanded: true,
          origin: `${mod.kind} of instance ${inst.name}`
        });
        symbols.globalDeclarations.push({
          name: mod.twinName,
          kind: "expanded_instance",
          offset: inst.offset,
          position: inst.position,
          origin: `${mod.kind} of ${inst.name}`
        });
      }
    }
    for (const v of variantNames) {
      const instBoxes = compDef
        ? compDef.boxes.map((bx) => ({
            ...bx,
            name: `${v.name}__${bx.name}`,
            dottedName: `${v.name}.${bx.name}`,
            expanded: true,
            origin: v.origin
          }))
        : [];
      symbols.expandedBones.push({
        name: v.name,
        isRoot: iIdx === 0 && !v.expanded,
        parentBoneName: iIdx === 0 && !v.expanded ? null : "assembly",
        expanded: v.expanded,
        origin: v.origin,
        offset: inst.offset,
        position: inst.position,
        boxes: instBoxes,
        locators: []
      });
      for (const bx of instBoxes) {
        symbols.expandedBoxes.push({ ...bx, boneName: v.name });
        symbols.expandedBoxes.push({ ...bx, name: bx.dottedName, boneName: v.name });
      }
      if (compDef) {
        for (const ch of compDef.children || []) {
          const chFull = `${v.name}__${ch.name}`;
          const chDot = `${v.name}.${ch.name}`;
          symbols.expandedBones.push({
            name: chDot,
            isRoot: false,
            parentBoneName: v.name,
            expanded: true,
            origin: `${v.origin} child bone`,
            offset: ch.offset,
            position: ch.position,
            boxes: [],
            locators: []
          });
          symbols.expandedBones.push({
            name: chFull,
            isRoot: false,
            parentBoneName: v.name,
            expanded: true,
            origin: `${v.origin} child bone`,
            offset: ch.offset,
            position: ch.position,
            boxes: [],
            locators: []
          });
        }
      }
    }
  }

  return {
    text,
    masked,
    lineStarts,
    scopes,
    boneMap,
    symbols
  };
}

/**
 * Classify a `{` block header into its semantic AST container kind.
 */
function classifyBlockHeader(rawHeader, headerStart, braceOffset, parentScope, masked) {
  const inMacro = !!(parentScope && (parentScope.kind === "macro" || parentScope.kind === "component" || parentScope.inMacro));
  // Take the last statement segment if multiple statements preceded `{`
  const tailIdx = Math.max(rawHeader.lastIndexOf(";"), rawHeader.lastIndexOf("}"));
  const clean = (tailIdx >= 0 ? rawHeader.slice(tailIdx + 1) : rawHeader).trim();
  const cleanOffsetInMasked = masked.lastIndexOf(clean, braceOffset);
  const baseOff = cleanOffsetInMasked >= 0 ? cleanOffsetInMasked : headerStart;

  let m;
  if ((m = /^@component\s+([A-Za-z_][A-Za-z0-9_]*)\b/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[1]);
    return { kind: "component", name: m[1], start: baseOff, nameOffset: nOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro: true, meta: {} };
  }
  if ((m = /^@assembly\s+([A-Za-z_][A-Za-z0-9_]*)\b/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[1]);
    return { kind: "section", name: "@assembly", start: baseOff, nameOffset: nOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro: false, meta: { assemblyName: m[1] } };
  }
  if ((m = /^(@[A-Za-z_][A-Za-z0-9_]*)\b/.exec(clean)) !== null) {
    return { kind: "section", name: m[1], start: baseOff, nameOffset: baseOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro: false, meta: {} };
  }
  if ((m = /\bsubassembly\s+([A-Za-z_][A-Za-z0-9_]*)\b/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[1]);
    return { kind: "subassembly", name: m[1], start: baseOff, nameOffset: nOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro: false, meta: {} };
  }
  if ((m = /\binstance\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)([\s\S]*)$/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[1]);
    return {
      kind: "instance",
      name: m[1],
      start: baseOff,
      nameOffset: nOff,
      bodyStart: braceOffset + 1,
      bodyEnd: braceOffset + 1,
      end: braceOffset + 1,
      inMacro: false,
      meta: { componentName: m[2], symmetryMods: parseSymmetryModifiers(m[3] || "", m[1]) }
    };
  }
  if ((m = /\bconnect\s+([A-Za-z_][A-Za-z0-9_.]*)\s+to\s+([A-Za-z_][A-Za-z0-9_.]*)/.exec(clean)) !== null) {
    return { kind: "connect", name: `${m[1]}->${m[2]}`, start: baseOff, nameOffset: baseOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro: false, meta: { childRef: m[1], parentRef: m[2] } };
  }
  if ((m = /\b(frame|control)\s+([A-Za-z_][A-Za-z0-9_]*)\b/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[2]);
    return { kind: m[1], name: m[2], start: baseOff, nameOffset: nOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro, meta: {} };
  }
  if ((m = /\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[1]);
    return { kind: "macro", name: m[1], start: baseOff, nameOffset: nOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro: true, meta: {} };
  }
  if ((m = /\broot\s+bone\s+([A-Za-z_][A-Za-z0-9_]*)\b/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[1]);
    return { kind: "root_bone", name: m[1], start: baseOff, nameOffset: nOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro, meta: { symmetryMods: [] } };
  }
  if ((m = /\bbone\s+([A-Za-z_][A-Za-z0-9_]*)([\s\S]*)$/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[1]);
    return {
      kind: "bone",
      name: m[1],
      start: baseOff,
      nameOffset: nOff,
      bodyStart: braceOffset + 1,
      bodyEnd: braceOffset + 1,
      end: braceOffset + 1,
      inMacro,
      meta: { symmetryMods: parseSymmetryModifiers(m[2] || "", m[1]) }
    };
  }
  if ((m = /\bchain\s+([A-Za-z_][A-Za-z0-9_]*)([\s\S]*)$/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[1]);
    const segMatch = /\bsegments\s*(?:=|:)\s*(\d+)/.exec(m[2] || "");
    const segments = segMatch ? Math.min(32, Math.max(1, parseInt(segMatch[1], 10))) : 1;
    return {
      kind: "chain",
      name: m[1],
      start: baseOff,
      nameOffset: nOff,
      bodyStart: braceOffset + 1,
      bodyEnd: braceOffset + 1,
      end: braceOffset + 1,
      inMacro,
      meta: { segments, symmetryMods: parseSymmetryModifiers(m[2] || "", m[1]) }
    };
  }
  if ((m = /\b(box|plane|locator)\s+([A-Za-z_][A-Za-z0-9_]*)\b/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[2]);
    return { kind: m[1], name: m[2], start: baseOff, nameOffset: nOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro, meta: {} };
  }
  if ((m = /\b(muscle|ik_chain|look_chain|spring|clip|state)\s+([A-Za-z_][A-Za-z0-9_]*)\b/.exec(clean)) !== null) {
    const nOff = baseOff + m.index + m[0].indexOf(m[2]);
    return { kind: m[1], name: m[2], start: baseOff, nameOffset: nOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro: false, meta: {} };
  }
  if ((m = /\bkey\s+([A-Za-z_][A-Za-z0-9_]*)\.(rotation|position|scale)\b/.exec(clean)) !== null) {
    return { kind: "keyframe", name: `${m[1]}.${m[2]}`, start: baseOff, nameOffset: baseOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro: false, meta: {} };
  }
  if ((m = /\bmaterial_mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*:/.exec(clean)) !== null) {
    return { kind: "material_mod", name: m[1], start: baseOff, nameOffset: baseOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro: false, meta: {} };
  }
  if ((m = /\buv_faces\s*:/.exec(clean)) !== null) {
    return { kind: "uv_faces", name: "uv_faces", start: baseOff, nameOffset: baseOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro, meta: {} };
  }
  return { kind: "unknown", name: "", start: baseOff, nameOffset: baseOff, bodyStart: braceOffset + 1, bodyEnd: braceOffset + 1, end: braceOffset + 1, inMacro, meta: {} };
}

function findInnermostScopeAt(scopes, offset, allowedKinds) {
  let best = null;
  for (const sc of scopes) {
    if (offset >= sc.bodyStart && offset <= sc.bodyEnd) {
      if (!allowedKinds || allowedKinds.includes(sc.kind)) {
        if (!best || sc.bodyStart >= best.bodyStart) {
          best = sc;
        }
      }
    }
  }
  return best;
}

function findEnclosingOwner(childScope, scopes, boneMap, macroMap, componentMap) {
  let best = null;
  for (const sc of scopes) {
    if (sc === childScope) continue;
    if (childScope.start >= sc.bodyStart && childScope.start <= sc.bodyEnd) {
      if (sc.kind === "bone" || sc.kind === "root_bone" || sc.kind === "chain" || sc.kind === "macro" || sc.kind === "component") {
        if (!best || sc.bodyStart >= best.bodyStart) {
          best = sc;
        }
      }
    }
  }
  if (!best) return null;
  if (best.kind === "macro") return macroMap.get(best.name) || null;
  if (best.kind === "component") return (componentMap && componentMap.get(best.name)) || null;
  return boneMap.get(best) || null;
}

/**
 * Determine the exact cursor context at `offset` for context-aware completion.
 * @param {ReturnType<typeof indexDocument>} docIndex
 * @param {number} offset
 */
function getCursorContext(docIndex, offset) {
  const { text, masked, scopes, boneMap, symbols } = docIndex;
  const sectionScope = findInnermostScopeAt(scopes, offset, ["section"]);
  const compScope = findInnermostScopeAt(scopes, offset, ["component"]);
  const containerScope = findInnermostScopeAt(scopes, offset, null);
  const boneScope = findInnermostScopeAt(scopes, offset, ["bone", "root_bone", "chain"]);
  const enclosingBone = boneScope ? boneMap.get(boneScope) : (compScope ? symbols.components.get(compScope.name) : null);

  // Find parent bone of `boneScope` if cursor is in a child bone header (between `bone X` and `{`)
  let headerBoneParent = null;
  if (boneScope) {
    const parentBoneScope = findInnermostScopeAt(scopes, boneScope.start - 1, ["bone", "root_bone"]);
    if (parentBoneScope) {
      headerBoneParent = boneMap.get(parentBoneScope) || null;
    }
  }

  const lineStart = text.lastIndexOf("\n", Math.max(0, offset - 1)) + 1;
  const linePrefix = text.slice(lineStart, offset);
  const maskedPrefix = masked.slice(Math.max(0, offset - 240), offset);

  // Detect active function call `fn_name( ... <cursor>` if inside unclosed `(`
  let activeCall = null;
  let depth = 0;
  for (let i = maskedPrefix.length - 1; i >= 0; i -= 1) {
    const ch = maskedPrefix[i];
    if (ch === ")" || ch === "}" || ch === "]") depth += 1;
    else if (ch === "{") break;
    else if (ch === "(") {
      if (depth === 0) {
        const beforeParen = maskedPrefix.slice(0, i);
        const fnMatch = /\b([A-Za-z_][A-Za-z0-9_]*)\s*$/.exec(beforeParen);
        if (fnMatch) {
          activeCall = fnMatch[1];
        }
        break;
      }
      depth -= 1;
    }
  }

  // Detect `key: <cursor>`, `arg=<cursor>`, or `list_key: [a, <cursor>`
  let valueTarget = null;
  const assignMatch = /\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[A-Za-z0-9_."\-]*$/.exec(linePrefix);
  const colonMatch = /\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[A-Za-z0-9_."\-]*$/.exec(linePrefix);
  const listMatch = /\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\[[^\]]*$/.exec(maskedPrefix);
  const driveTargetMatch = /\b(track|slide|muscle_drive|ik_drive|pose_bias|muscle_tone|material_mod|spring|prismatic_joint)\s+[A-Za-z0-9_.]*$/.exec(linePrefix);
  const keyTargetMatch = /\bkey\s+([A-Za-z0-9_.]*)(\.)?([A-Za-z0-9_]*)$/.exec(linePrefix);

  if (assignMatch) {
    valueTarget = { kind: "arg", name: assignMatch[1] };
  } else if (listMatch) {
    valueTarget = { kind: "list_prop", name: listMatch[1] };
  } else if (keyTargetMatch) {
    valueTarget = {
      kind: keyTargetMatch[2] ? "key_channel" : "key_bone",
      bone: keyTargetMatch[1]
    };
  } else if (driveTargetMatch) {
    valueTarget = { kind: "drive_subject", keyword: driveTargetMatch[1] };
  } else if (colonMatch) {
    valueTarget = { kind: "prop", name: colonMatch[1] };
  }

  return {
    offset,
    linePrefix,
    section: sectionScope ? sectionScope.name : (compScope ? "@component" : null),
    componentName: compScope ? compScope.name : null,
    containerKind: containerScope ? containerScope.kind : null,
    containerName: containerScope ? containerScope.name : null,
    enclosingBone,
    headerBoneParent,
    activeCall,
    valueTarget
  };
}

module.exports = {
  maskCommentsAndStrings,
  computeLineStarts,
  offsetToPosition,
  indexDocument,
  getCursorContext
};
