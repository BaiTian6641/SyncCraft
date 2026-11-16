"""Comprehensive tests for the SyncCraft Block-DSL VS Code extension (`editors/vscode-blockdsl`).

Verifies:
1. 1:1 schema & enum parity between `src/synccraft/dsl/ast_nodes.py` + `grammar.lark`
   and `editors/vscode-blockdsl/src/catalog.js` + `syntaxes/blockdsl.tmLanguage.json`.
2. TextMate grammar regex compilation & token scope matching across every Block-DSL v3.2
   construct (including `sculpted`, `blend_with`, `spin`, `mirror_x`, `radial`, `chain`,
   `def`/`use`, `@muscles`, `@kinematics`, `@behavior`, `@controller`, and `invalid.illegal.*`
   scopes on `@shade`/`@motion`/`@ergonomics`/`extrude`/`True`).
3. Real JavaScript execution (via macOS JavaScriptCore `jsc` or `node` if available) of
   `symbol_index.js`, `completion_engine.js`, `hover_docs.js`, and `diagnostics_engine.js`
   on both synthetic edge cases and flagship 62-box / 74-box exemplars.
4. Live Python compiler/SAT audit & canonical formatter bridge (`extension.js` embedded script).
5. Deterministic `.vsix` OPC packaging (`scripts/package_vscode_extension.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

from synccraft.dsl import ast_nodes
from synccraft.world.assembly_world_tasks import (
    MODULAR_ARTILLERY_WALKER_V4_DSL,
    MODULAR_SCULPTED_DRAGON_V4_DSL,
)
from synccraft.world.exemplars import COLOSSAL_WALKING_CITADEL_DSL, SCULPTED_ELDER_WYVERN_DSL

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = REPO_ROOT / "editors" / "vscode-blockdsl"
JSC_PATH = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")


def _find_js_runtime() -> tuple[str, str] | None:
    node_bin = shutil.which("node")
    if node_bin:
        return ("node", node_bin)
    if JSC_PATH.exists():
        return ("jsc", str(JSC_PATH))
    return None


class VSCodeBlockDSLExtensionTests(unittest.TestCase):
    def test_manifests_snippets_and_grammar_regexes_are_valid(self) -> None:
        pkg = json.loads((EXT_DIR / "package.json").read_text(encoding="utf-8"))
        lang_cfg = json.loads((EXT_DIR / "language-configuration.json").read_text(encoding="utf-8"))
        tm = json.loads((EXT_DIR / "syntaxes" / "blockdsl.tmLanguage.json").read_text(encoding="utf-8"))
        md_inj = json.loads((EXT_DIR / "syntaxes" / "blockdsl.markdown.injection.json").read_text(encoding="utf-8"))
        snippets = json.loads((EXT_DIR / "snippets" / "blockdsl.code-snippets").read_text(encoding="utf-8"))

        self.assertEqual(pkg["name"], "synccraft-blockdsl")
        self.assertEqual(pkg["contributes"]["languages"][0]["id"], "blockdsl")
        self.assertIn(".blockdsl", pkg["contributes"]["languages"][0]["extensions"])
        self.assertIn(".bdsl", pkg["contributes"]["languages"][0]["extensions"])
        self.assertEqual(lang_cfg["comments"]["lineComment"], "//")
        self.assertEqual(tm["scopeName"], "source.blockdsl")
        self.assertEqual(md_inj["scopeName"], "markdown.blockdsl.codeblock")
        self.assertGreaterEqual(len(snippets), 10)

        # Compile every single regex pattern in the TextMate grammar
        compiled_count = 0
        for repo_entry in tm["repository"].values():
            for pat in repo_entry.get("patterns", []):
                for key in ("match", "begin", "end", "while"):
                    if key in pat:
                        re.compile(pat[key], re.MULTILINE)
                        compiled_count += 1
        self.assertGreaterEqual(compiled_count, 35)

    def test_textmate_grammar_and_lark_parity_with_python_ast_constants(self) -> None:
        tm_text = (EXT_DIR / "syntaxes" / "blockdsl.tmLanguage.json").read_text(encoding="utf-8")
        lark_text = (REPO_ROOT / "src" / "synccraft" / "dsl" / "grammar.lark").read_text(encoding="utf-8")
        catalog_text = (EXT_DIR / "src" / "catalog.js").read_text(encoding="utf-8")

        self.assertIn('"blend_with" ":" ident_list', lark_text)

        constant_sets = [
            ast_nodes.ARCHETYPES,
            ast_nodes.PROFILES,
            ast_nodes.AUDIT_PROFILES,
            ast_nodes.FACES,
            ast_nodes.AXES_3D,
            ast_nodes.LOCATOR_ROLES,
            ast_nodes.TRACK_LAWS,
            ast_nodes.SLIDE_LAWS,
            ast_nodes.KEYFRAME_CHANNELS,
            ast_nodes.KEYFRAME_INTERPOLATIONS,
            ast_nodes.MATERIAL_FAMILIES_DSL,
            ast_nodes.TARGET_ENCODINGS_DSL,
        ]
        for group in constant_sets:
            for token in group:
                self.assertIn(token, tm_text, f"Missing '{token}' in blockdsl.tmLanguage.json")
                self.assertIn(token, catalog_text, f"Missing '{token}' in src/catalog.js")

        for sec in ("@meta", "@layout", "@textures", "@palette", "@component", "@assembly", "@rig", "@materials", "@style", "@muscles", "@kinematics", "@behavior", "@controller"):
            self.assertIn(sec[1:], tm_text)
            self.assertIn(sec, catalog_text)

        for unsup in ("shade", "motion", "ergonomics", "extrude"):
            self.assertIn(unsup, tm_text)

    def test_textmate_token_scopes_match_all_v32_constructs(self) -> None:
        tm = json.loads((EXT_DIR / "syntaxes" / "blockdsl.tmLanguage.json").read_text(encoding="utf-8"))

        def matching_scopes(line: str) -> set[str]:
            found: set[str] = set()
            for repo_entry in tm["repository"].values():
                for pat in repo_entry.get("patterns", []):
                    if "match" in pat:
                        m = re.search(pat["match"], line, re.MULTILINE)
                        if m:
                            if "name" in pat:
                                found.add(pat["name"])
                            for cap in pat.get("captures", {}).values():
                                if "name" in cap:
                                    found.add(cap["name"])
            return found

        self.assertIn("keyword.control.directive.section.blockdsl", matching_scopes("@meta { archetype: entity, audit_profile: sculpted }"))
        self.assertIn("support.constant.audit-profile.blockdsl", matching_scopes("@meta { archetype: entity, audit_profile: sculpted }"))
        self.assertIn("support.type.property-name.blockdsl", matching_scopes("box prow_l { size: [8, 7, 6], material: armor, blend_with: [prow_r__prow_l] }"))
        self.assertIn("support.function.motion-law.blockdsl", matching_scopes("track flywheel_l: spin(axis=x, amp_deg=360, freq_hz=0.5)"))
        self.assertIn("support.function.symmetry.blockdsl", matching_scopes("bone wing_l attach(parent_box=chest, anchor=[-1,0,0], align=[1,0,0]) mirror_x(wing_r) {"))
        self.assertIn("keyword.operator.arrow.blockdsl", matching_scopes("fortress_march -> boiler_overdrive when steam_pressure >= 0.85, blend 0.3"))
        self.assertIn("invalid.illegal.unsupported-section.blockdsl", matching_scopes("@shade { mode: toon }"))
        self.assertIn("invalid.illegal.unsupported-keyword.blockdsl", matching_scopes("extrude blade_outline { depth: 1 }"))
        self.assertIn("invalid.illegal.boolean-case.blockdsl", matching_scopes("loop: True"))

    def test_javascript_intelligence_engine_completions_hover_and_diagnostics(self) -> None:
        rt = _find_js_runtime()
        if rt is None:
            self.skipTest("Neither node nor macOS JavaScriptCore jsc is installed on this host")
        kind, binary = rt

        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "run_js_tests.js"
            citadel_json = json.dumps(COLOSSAL_WALKING_CITADEL_DSL)
            wyvern_json = json.dumps(SCULPTED_ELDER_WYVERN_DSL)
            dragon_v4_json = json.dumps(MODULAR_SCULPTED_DRAGON_V4_DSL)
            walker_v4_json = json.dumps(MODULAR_ARTILLERY_WALKER_V4_DSL)
            ext_dir_json = json.dumps(str(EXT_DIR))

            if kind == "jsc":
                loader_preamble = f"""
const EXT_DIR = {ext_dir_json};
const __cache = {{}};
function require(rel) {{
  const clean = rel.replace(/^\\.\\//, "");
  const full = EXT_DIR + "/src/" + clean.replace(/^src\\//, "") + (clean.endsWith(".js") ? "" : ".js");
  if (__cache[full]) return __cache[full].exports;
  const mod = {{ exports: {{}} }};
  __cache[full] = mod;
  const code = readFile(full);
  const fn = new Function("module", "exports", "require", code);
  fn(mod, mod.exports, require);
  return mod.exports;
}}
const outPrint = print;
"""
            else:
                loader_preamble = f"""
const path = require("path");
const EXT_DIR = {ext_dir_json};
const origReq = require;
function customReq(rel) {{
  return origReq(path.join(EXT_DIR, "src", rel.replace(/^\\.\\//, "")));
}}
require = customReq;
const outPrint = console.log;
"""

            harness.write_text(
                loader_preamble
                + f"""
const {{ indexDocument }} = require("./symbol_index");
const {{ provideCompletions }} = require("./completion_engine");
const {{ provideHover }} = require("./hover_docs");
const {{ runFastDiagnostics }} = require("./diagnostics_engine");

function assert(cond, msg) {{
  if (!cond) throw new Error("Assertion failed: " + msg);
}}

// 1. Index full 74-box Colossal Walking Citadel & 62-box Sculpted Elder Wyvern
const citadelDsl = {citadel_json};
const wyvernDsl = {wyvern_json};

const citIdx = indexDocument(citadelDsl);
assert(runFastDiagnostics(citIdx).length === 0, "Citadel must have 0 fast lint errors");
const wyvIdx = indexDocument(wyvernDsl);
assert(runFastDiagnostics(wyvIdx).length === 0, "Wyvern must have 0 fast lint errors");

// Verify Block-DSL v4.0 assemblies (@layout, @component, @assembly) also index cleanly with 0 fast lint errors!
const dragonV4Dsl = {dragon_v4_json};
const walkerV4Dsl = {walker_v4_json};
const drgV4Idx = indexDocument(dragonV4Dsl);
assert(runFastDiagnostics(drgV4Idx).length === 0, "Dragon v4.0 must have 0 fast lint errors, got: " + JSON.stringify(runFastDiagnostics(drgV4Idx)));
const wlkV4Idx = indexDocument(walkerV4Dsl);
assert(runFastDiagnostics(wlkV4Idx).length === 0, "Walker v4.0 must have 0 fast lint errors, got: " + JSON.stringify(runFastDiagnostics(wlkV4Idx)));
assert(drgV4Idx.symbols.expandedBones.some((b) => b.name === "foreleg_r.shin"), "Dragon v4.0 must expand mirrored dotted bone foreleg_r.shin");
const hSkull = provideHover(drgV4Idx, dragonV4Dsl.indexOf("DragonSkull("));
assert(hSkull && hSkull.markdown.includes("horn_socket_l"), "@component DragonSkull hover must list public frames");

// Check deterministic expansion of mirror_x and radial(count=4)
const expBoneNames = citIdx.symbols.expandedBones.map((b) => b.name);
assert(expBoneNames.includes("prow_wing_r"), "mirror_x root prow_wing_r missing");
assert(expBoneNames.includes("leg_fore_hip_r__leg_fore_shin_l"), "nested mirror_x shin missing");
assert(expBoneNames.includes("dome_rib_r1") && expBoneNames.includes("dome_rib_r3"), "radial dome ribs missing");

const expBoxNames = citIdx.symbols.expandedBoxes.map((b) => b.name);
assert(expBoxNames.includes("prow_wing_r__prow_plate_l"), "mirrored prow_plate_l missing");
assert(expBoxNames.includes("dome_rib_r2__arch_slab"), "radial arch_slab missing");
assert(wyvIdx.symbols.expandedBones.some((b) => b.name === "tail_seg_4"), "chain tail_seg_4 missing");
assert(wyvIdx.symbols.expandedBoxes.some((b) => b.name.startsWith("tail_seg_4_")), "chain tail_seg_4 expanded box missing");

// 2. Context-aware auto-completion probes
const probeDoc = [
  "def fin_macro(h=3) {{ box blade {{ size: [1, h, 2], material: hull_mat }} }}",
  "@meta {{ archetype: entity, target_profile: geckolib, texel_density: 16, audit_profile: ",
  "}}",
  "@palette {{",
  "  hull_mat: ramp(base_lch=[0.5, 0.08, 220], steps=8)",
  "  glow_mat: ramp(base_lch=[0.7, 0.18, 40], steps=6, emissive=true)",
  "}}",
  "@rig {{",
  "  root bone chassis {{",
  "    box hull_core {{ size: [8, 6, 10], material: ",
  "    }}",
  "    bone prow_l attach(parent_box=",
  "    ) mirror_x(prow_r) {{",
  "      box plate_l {{ size: [4, 4, 4], material: hull_mat, blend_with: [",
  "      ] }}",
  "    }}",
  "  }}",
  "}}",
  "@materials {{",
  "  ",
  "}}",
  "@kinematics {{",
  "  ik_chain leg_ik {{ bones: [",
  "  ], effector: [0, -5, 0], hinge_axis: x }}",
  "}}",
  "@behavior {{",
  "  clip march {{",
  "    duration_s: 2.0",
  "    track prow_l: ",
  "  }}",
  "}}"
].join("\\n");

const pIdx = indexDocument(probeDoc);

const auditOff = probeDoc.indexOf("audit_profile: ") + "audit_profile: ".length;
const auditLabels = provideCompletions(pIdx, auditOff).map((i) => i.label);
assert(auditLabels.includes("strict") && auditLabels.includes("sculpted"), "audit_profile completions missing sculpted");

const matOff = probeDoc.indexOf("material: ") + "material: ".length;
const matLabels = provideCompletions(pIdx, matOff).map((i) => i.label);
assert(matLabels.includes("hull_mat") && matLabels.includes("glow_mat"), "material completions missing palette names");

const parentOff = probeDoc.indexOf("parent_box=") + "parent_box=".length;
const parentLabels = provideCompletions(pIdx, parentOff).map((i) => i.label);
assert(parentLabels[0] === "hull_core", "parent_box= must prioritize enclosing bone box hull_core");

const blendOff = probeDoc.indexOf("blend_with: [") + "blend_with: [".length;
const blendItems = provideCompletions(pIdx, blendOff);
assert(blendItems[0].label === "prow_r__plate_l", "blend_with must rank mirrored twin prow_r__plate_l #1, got " + blendItems[0].label);

const matSecOff = probeDoc.indexOf("@materials {{\\n  ") + "@materials {{\\n  ".length;
const matSecLabels = provideCompletions(pIdx, matSecOff).map((i) => i.label);
assert(matSecLabels.includes("hull_mat: process(...)"), "@materials should suggest unbound palette process bindings");

const ikBonesOff = probeDoc.indexOf("bones: [") + "bones: [".length;
const ikBoneLabels = provideCompletions(pIdx, ikBonesOff).map((i) => i.label);
assert(ikBoneLabels.includes("prow_l") && ikBoneLabels.includes("prow_r") && !ikBoneLabels.includes("chassis"), "ik_chain bones must include non-root + mirrored bones and exclude root");

const trackLawOff = probeDoc.indexOf("track prow_l: ") + "track prow_l: ".length;
const trackLaws = provideCompletions(pIdx, trackLawOff).map((i) => i.label);
assert(trackLaws.includes("spin") && trackLaws.includes("wave"), "track laws must include spin and wave");

// 3. Hover documentation checks
const spinHoverOff = citadelDsl.indexOf("spin(axis=x");
const hSpin = provideHover(citIdx, spinHoverOff);
assert(hSpin && hSpin.markdown.includes("1440"), "spin hover missing [-1440..1440] doc");

const twinHoverOff = citadelDsl.indexOf("prow_wing_r__prow_plate_l");
const hTwin = provideHover(citIdx, twinHoverOff);
assert(hTwin && hTwin.markdown.includes("prow_wing_r") && hTwin.markdown.includes("mirror_x"), "expanded box hover missing mirror_x origin");

// 4. Negative diagnostics checks
const badDoc = [
  "@meta {{ archetype: entity, target_profile: java_modern, texel_density: 16 }}",
  "@shade {{ mode: flat }}",
  "@palette {{ p1: ramp(base_lch=[0.5, 0.1, 100], steps=4, emissive=True) }}",
  "@rig {{",
  "  root bone r {{",
  "    box b1 {{ size: [4, 4, 4], material: unknown_mat, inflate: -0.2 }}",
  "    bone c attach(parent_box=b1, anchor=[0,1,0], align=[0,-1,0]) {{",
  "      joint_limits: [[10, 45], [-10, 10], [-10, 10]]",
  "      box p1 {{ size: [2, 2, 2], material: p1 }}",
  "    }}",
  "  }}",
  "}}"
].join("\\n");
const badDiags = runFastDiagnostics(indexDocument(badDoc));
const codes = new Set(badDiags.map((d) => d.code));
for (const expected of ["unsupported_feature", "boolean_case", "duplicate_identifier", "unknown_material", "incompatible_entity_profile", "invalid_joint_limits", "negative_inflate"]) {{
  assert(codes.has(expected), "Expected diagnostic code missing: " + expected);
}}

outPrint("JS_ENGINE_TESTS_PASSED");
""",
                encoding="utf-8",
            )

            proc = subprocess.run([binary, str(harness)], capture_output=True, text=True, check=False, timeout=15)
            self.assertEqual(proc.returncode, 0, msg=f"JS test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
            self.assertIn("JS_ENGINE_TESTS_PASSED", proc.stdout)

    def test_embedded_python_audit_and_formatter_bridge_plus_vsix_packaging(self) -> None:
        ext_js = (EXT_DIR / "extension.js").read_text(encoding="utf-8")
        m = re.search(r"const PYTHON_AUDIT_SCRIPT = \[(.*?)\]\.join\(\"\\n\"\);", ext_js, re.S)
        self.assertIsNotNone(m)
        lines = json.loads("[" + m.group(1) + "]")
        py_bridge = "\n".join(lines)

        env = {"PYTHONPATH": str(REPO_ROOT / "src")}
        # 1. Audit passing sculpted citadel
        aud_proc = subprocess.run(
            [sys.executable, "-c", py_bridge, "audit", "1"],
            input=COLOSSAL_WALKING_CITADEL_DSL,
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=20,
        )
        aud_res = json.loads(aud_proc.stdout)
        self.assertTrue(aud_res["ok"])
        self.assertTrue(aud_res["passed"])
        self.assertEqual(aud_res["audit_profile"], "sculpted")
        self.assertEqual(aud_res["boxes"], 74)
        self.assertEqual(aud_res["sculpted_overlaps"], 40)

        # 2. Format mode roundtrip
        fmt_proc = subprocess.run(
            [sys.executable, "-c", py_bridge, "format", "0"],
            input=(REPO_ROOT / "examples" / "mirrored_wings.blockdsl").read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=10,
        )
        fmt_res = json.loads(fmt_proc.stdout)
        self.assertTrue(fmt_res["ok"])
        self.assertIn("mirror_x(right_wing)", fmt_res["formatted"])

        # 3. Package .vsix into temp directory and inspect OPC archive
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "package_vscode_extension",
            REPO_ROOT / "scripts" / "package_vscode_extension.py",
        )
        assert spec is not None and spec.loader is not None
        pkg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pkg_mod)
        package_vsix = pkg_mod.package_vsix

        with tempfile.TemporaryDirectory() as tmp:
            vsix_path = package_vsix(Path(tmp), regen_icon=True)
            self.assertTrue(vsix_path.exists())
            with zipfile.ZipFile(vsix_path, "r") as zf:
                names = set(zf.namelist())
                for required in (
                    "[Content_Types].xml",
                    "extension.vsixmanifest",
                    "extension/package.json",
                    "extension/extension.js",
                    "extension/src/catalog.js",
                    "extension/src/symbol_index.js",
                    "extension/src/completion_engine.js",
                    "extension/src/hover_docs.js",
                    "extension/src/diagnostics_engine.js",
                    "extension/syntaxes/blockdsl.tmLanguage.json",
                    "extension/syntaxes/blockdsl.markdown.injection.json",
                    "extension/snippets/blockdsl.code-snippets",
                    "extension/images/icon.png",
                ):
                    self.assertIn(required, names)


if __name__ == "__main__":
    unittest.main()
