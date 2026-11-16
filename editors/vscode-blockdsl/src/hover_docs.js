// @ts-check
"use strict";

const catalog = require("./catalog");

/**
 * Return Markdown hover documentation for the token at `offset`, combining built-in
 * Block-DSL v3.2 language/physics reference cards with live symbol origin info from
 * the active document (including `mirror_x`, `radial`, `use`, and `chain` expansions!).
 *
 * @param {ReturnType<import("./symbol_index").indexDocument>} docIndex
 * @param {number} offset
 * @returns {{ word: string, markdown: string } | null}
 */
function provideHover(docIndex, offset) {
  const { text, symbols } = docIndex;
  const token = extractTokenAt(text, offset);
  if (!token) {
    return null;
  }

  if (catalog.UNSUPPORTED_KEYWORDS.includes(token)) {
    return {
      word: token,
      markdown: `**\`${token}\`** — **Unsupported Reserved Keyword**\n\nRaises \`UnsupportedFeature\` (\`DSLValidationError\`) in SyncCraft Block-DSL v3.2. Use \`@materials\`/\`@style\` instead of \`@shade\`, and \`@muscles\`/\`@kinematics\`/\`@behavior\` instead of \`@motion\`.`
    };
  }

  if (Object.prototype.hasOwnProperty.call(catalog.HOVER_DOCUMENTATION, token)) {
    return {
      word: token,
      markdown: catalog.HOVER_DOCUMENTATION[token]
    };
  }

  const matFam = catalog.MATERIAL_FAMILIES.find((f) => f.name === token);
  if (matFam) {
    return {
      word: token,
      markdown: `**\`${matFam.name}\`** — Physical \`@materials\` Family\n\n${matFam.detail}.\n\nRecommended parameters:\n\`\`\`blockdsl\nprocess(family=${matFam.name}, ${matFam.params})\n\`\`\``
    };
  }

  const matArg = catalog.MATERIAL_PROCESS_ARGS.find((a) => a.name === token);
  if (matArg) {
    return {
      word: token,
      markdown: `**\`${matArg.name}\`** — \`@materials\` \`process(...)\` Parameter\n\n- **Valid Range**: \`${matArg.range}\`\n- **Default**: \`${matArg.defaultVal}\``
    };
  }

  // User-declared or deterministically expanded symbol lookup
  const expBone = symbols.expandedBones.find((b) => b.name === token);
  if (expBone) {
    const childBoxCount = (expBone.boxes || []).length;
    return {
      word: token,
      markdown: `**\`${expBone.name}\`** — ${expBone.isRoot ? "Root `bone`" : (expBone.expanded ? "Expanded `bone`" : "Child `bone`")}\n\n- **Origin**: ${expBone.origin} (line ${(expBone.position ? expBone.position.line : 0) + 1})\n- **Parent Bone**: \`${expBone.parentBoneName || "<world>"}\`\n- **Direct Elements**: \`${childBoxCount}\` box/plane element(s)`
    };
  }

  const expBox = symbols.expandedBoxes.find((b) => b.name === token);
  if (expBox) {
    return {
      word: token,
      markdown: `**\`${expBox.name}\`** — ${expBox.expanded ? `Expanded \`${expBox.kind}\`` : `Declared \`${expBox.kind}\``}\n\n- **Enclosing Bone**: \`${expBox.boneName}\`\n- **Origin**: ${expBox.origin || "declared"} (line ${(expBox.position ? expBox.position.line : 0) + 1})`
    };
  }

  const pal = symbols.palettes.find((p) => p.name === token);
  if (pal) {
    const hasMat = symbols.materials.some((m) => m.name === token);
    return {
      word: token,
      markdown: `**\`${pal.name}\`** — Declared \`@palette\` Ramp (line ${pal.position.line + 1})\n\n- **Physical \`@materials\` Process Bound**: \`${hasMat ? "yes" : "no (using OKLCh ramp only)"}\``
    };
  }

  const mus = symbols.muscles.find((m) => m.name === token);
  if (mus) {
    return {
      word: token,
      markdown: `**\`${mus.name}\`** — Declared \`@muscles\` Zajac/Hill Actuator (line ${mus.position.line + 1})`
    };
  }

  const ik = symbols.ikChains.find((c) => c.name === token);
  if (ik) {
    return {
      word: token,
      markdown: `**\`${ik.name}\`** — Declared \`@kinematics\` \`ik_chain\` (line ${ik.position.line + 1})`
    };
  }

  return null;
}

function extractTokenAt(text, offset) {
  if (offset < 0 || offset > text.length) return null;
  let s = offset;
  let e = offset;
  const isWordChar = (ch) => /[A-Za-z0-9_@]/.test(ch);
  while (s > 0 && isWordChar(text[s - 1])) s -= 1;
  while (e < text.length && /[A-Za-z0-9_]/.test(text[e])) e += 1;
  const tok = text.slice(s, e);
  return tok || null;
}

module.exports = {
  provideHover,
  extractTokenAt
};
