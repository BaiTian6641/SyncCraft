"""Block-DSL v3.0 / v3.1 / v3.2 contract & single-grammar-truth tests; zero network dependency."""

import ast
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import random
import re
import unittest

from synccraft.dsl import (
    Attach,
    Bone,
    Box,
    ControllerDecl,
    DSLParseError,
    DSLValidationError,
    IKChainDecl,
    Joint,
    KeyframeTrackDecl,
    Meta,
    Palette,
    Place,
    Plane,
    Program,
    RadialSpec,
    SpringDecl,
    TextureDecl,
    UnsupportedFeature,
    format_program,
    parse_program,
    validate_program,
)

ROOT = Path(__file__).resolve().parents[1]
MINIMAL = """@meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
@palette { metal: ramp(base_lch=[0.7,0.05,240], steps=4) }
@rig { root bone body { pivot_local: [8,8,8]
  box shaft { size: [2,6,2], material: metal }
} }
"""


class TestDSL(unittest.TestCase):
    def test_examples_and_canonical_roundtrip(self):
        paths = sorted((ROOT / "examples").glob("*.blockdsl"))
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            with self.subTest(path=path.name):
                first = parse_program(path.read_text())
                canonical = format_program(first)
                second = parse_program(canonical)
                self.assertEqual(first, second)
                self.assertEqual(canonical, format_program(second))

    def test_seeded_500_roundtrips(self):
        rng = random.Random(303)
        for _ in range(500):
            program = Program(
                Meta("block_multipart", "bbmodel_5", 16),
                [Palette("material_a", (rng.random(), rng.random() * 0.2, rng.random() * 360), rng.randint(2, 8))],
                Bone(
                    "root_a",
                    pivot_local=(rng.uniform(-8, 8), 0, 0),
                    items=[
                        Box("base_a", (rng.randint(1, 8), rng.randint(1, 8), rng.randint(1, 8)), "material_a"),
                        Box("cap_a", (1, 2, 3), "material_a", Place("base_a", (0, 1, 0), (0, -1, 0))),
                        Bone(
                            "child_a",
                            attach=Attach("base_a", (1, 0, 0), (-1, 0, 0)),
                            rest_euler=(0, 0, rng.choice([-45, 0, 22.5, 45])),
                            items=[Box("tip_a", (2, 2, 2), "material_a")],
                        ),
                    ],
                ),
            )
            self.assertEqual(program, parse_program(format_program(program)))

    def test_rejects_noninteger_and_nonpositive_sizes(self):
        for size in ("[2.0,6,2]", "[2e0,6,2]", "[2,-6,2]", "[2,0,2]"):
            with self.subTest(size=size), self.assertRaises((DSLParseError, DSLValidationError)):
                parse_program(MINIMAL.replace("[2,6,2]", size))

    def test_rejects_nonfinite_number(self):
        with self.assertRaises(DSLValidationError):
            parse_program(MINIMAL.replace("[8,8,8]", "[1e999,8,8]"))

    def test_rejects_undefined_material_and_profile(self):
        with self.assertRaises(DSLValidationError):
            parse_program(MINIMAL.replace("material: metal", "material: missing"))
        with self.assertRaises(DSLValidationError):
            parse_program(MINIMAL.replace("bbmodel_5", "bogus"))
        with self.assertRaises(DSLValidationError):
            parse_program(MINIMAL.replace("weapon_3d", "entity").replace("bbmodel_5", "java_modern"))

    def test_rejects_duplicates(self):
        with self.assertRaises(DSLValidationError):
            parse_program(MINIMAL.replace("box shaft", "box body"))

    def test_rejects_place_forward_reference(self):
        text = MINIMAL.replace("box shaft", "box shaft place(base_box=later, anchor=[0,1,0], align=[0,-1,0])")
        with self.assertRaises(DSLValidationError):
            parse_program(text)

    def test_rejects_second_implicit_box(self):
        text = MINIMAL.replace("\n} }", "\nbox second { size: [1,1,1], material: metal }\n} }")
        with self.assertRaises(DSLValidationError):
            parse_program(text)

    def test_attach_scope_and_connected_offset(self):
        p = parse_program(MINIMAL)
        child = Bone("joint", attach=Attach("shaft", (0, 1, 0), (0, -1, 0), (0, 0.6, 0)), items=[Box("child", (1, 1, 1), "metal")])
        with self.assertRaises(DSLValidationError):
            validate_program(replace(p, root=replace(p.root, items=p.root.items + [child])))
        detached = replace(child, attach=replace(child.attach, detached=True))
        validate_program(replace(p, root=replace(p.root, items=p.root.items + [detached])))
        with self.assertRaises(DSLValidationError):
            validate_program(replace(p, root=replace(p.root, items=[detached] + p.root.items)))

    def test_unit_anchor_and_origin_place(self):
        p = parse_program(MINIMAL)
        original = p.root.items[0]
        for place in (Place(None, (1, 0, 0), (0, 0, 0)), Place(None, (0, 0, 0), (1.1, 0, 0))):
            with self.subTest(place=place), self.assertRaises(DSLValidationError):
                validate_program(replace(p, root=replace(p.root, items=[replace(original, place=place)])))

    def test_mirror_generated_identifiers_reserved(self):
        p = parse_program((ROOT / "examples/mirrored_wings.blockdsl").read_text())
        duplicate = Box("right_wing__left_panel", (1, 1, 1), "slate", Place("hub", (0, 1, 0), (0, -1, 0)))
        with self.assertRaises(DSLValidationError):
            validate_program(replace(p, root=replace(p.root, items=p.root.items + [duplicate])))

    def test_deferred_features_fail_explicitly(self):
        for section in ("shade", "motion", "ergonomics"):
            with self.subTest(section=section), self.assertRaises(UnsupportedFeature):
                parse_program(MINIMAL + f"@{section} {{ }}")
        with self.assertRaises(UnsupportedFeature):
            parse_program(MINIMAL.replace("box shaft", "extrude shaft"))

    def test_rejects_extra_fields_and_code(self):
        with self.assertRaises(DSLParseError):
            parse_program(MINIMAL.replace("material: metal", "material: metal, undocumented: 1"))
        with self.assertRaises(DSLParseError):
            parse_program(MINIMAL + "__import__('os').system('echo bad')")

    def test_error_location(self):
        with self.assertRaises(DSLParseError) as error:
            parse_program("\n\n?")
        self.assertEqual((error.exception.line, error.exception.column), (3, 1))

    def test_comments(self):
        self.assertEqual(parse_program(MINIMAL), parse_program("// hello\n" + MINIMAL + "# tail\n"))

    def test_direct_ast_canonical_numbers_have_stable_hash_input(self):
        program = parse_program(MINIMAL)
        program = replace(
            program,
            root=replace(
                program.root,
                pivot_local=(8, 8, 8),
                items=[
                    replace(program.root.items[0], inflate=0),
                ],
            ),
        )
        canonical = format_program(program)
        self.assertEqual(canonical, format_program(parse_program(canonical)))

    def test_direct_ast_invalid_types_and_unrepresentable_integers(self):
        program = parse_program(MINIMAL)
        for vector in ((True, 0, 0), (10**500, 0, 0), [0, 0, 0], None):
            with self.subTest(vector_type=type(vector)), self.assertRaises(DSLValidationError):
                validate_program(replace(program, root=replace(program.root, pivot_local=vector)))
        for size in ((True, 2, 2), (10**500, 2, 2)):
            with self.assertRaises(DSLValidationError):
                validate_program(replace(program, root=replace(program.root, items=[replace(program.root.items[0], size=size)])))
        with self.assertRaises(DSLValidationError):
            validate_program(replace(program, meta=replace(program.meta, texel_density=10**500)))

    def test_wrong_direct_ast_reference_type_has_validation_error(self):
        program = parse_program(MINIMAL)
        invalid = replace(program.root.items[0], place=Place([], (0, 0, 0), (0, 0, 0)))
        with self.assertRaises(DSLValidationError):
            validate_program(replace(program, root=replace(program.root, items=[invalid])))

    def test_v32_compact_canonical_formatter_cuts_default_boilerplate_m2(self):
        verbose_legacy = """@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16 }
@palette {
  hide: ramp(base_lch=[0.50, 0.10, 35.0], steps=5, hue_shift=[0.0, 0.0], emissive=false)
  steel: ramp(base_lch=[0.70, 0.03, 240.0], steps=4, hue_shift=[0.0, 0.0], emissive=false)
}
@rig {
  root bone root {
    pivot_local: [0.0, 12.0, 0.0]
    box torso { size: [8, 8, 12], material: hide, inflate: 0.0, rotate: [0.0, 0.0, 0.0] }
    box saddle place(base_box=torso, anchor=[0.0, 1.0, 0.0], align=[0.0, -1.0, 0.0], offset=[0.0, 0.0, 0.0]) { size: [6, 2, 8], material: steel, inflate: 0.0, rotate: [0.0, 0.0, 0.0] }
    bone head attach(parent_box=torso, anchor=[0.0, 0.3, -1.0], align=[0.0, 0.0, 1.0], offset=[0.0, 0.0, 0.0], detached=false) {
      rest_euler: [0.0, 0.0, 0.0]
      box skull { size: [6, 6, 6], material: hide, inflate: 0.0, rotate: [0.0, 0.0, 0.0] }
    }
    bone leg_fl attach(parent_box=torso, anchor=[1.0, -1.0, -0.6], align=[-1.0, 1.0, 0.0], offset=[0.0, 0.0, 0.0], detached=false) mirror_x(leg_fr) {
      rest_euler: [0.0, 0.0, 0.0]
      box limb_fl { size: [3, 8, 3], material: hide, inflate: 0.0, rotate: [0.0, 0.0, 0.0] }
    }
  }
}
@materials {
  hide: process(family=patterned_skin, roughness=0.35, metalness=0.0, coating=0.0, ior=1.5, scale=1.0, seed=101, anisotropy=1.0, subsurface=0.0, porosity=0.08, emissive=0.0, rd_feed=0.036, rd_kill=0.062, film_nm=380.0, wear=0.35, oxidation=0.25, biofilm=0.20)
  steel: process(family=brushed_metal, roughness=0.25, metalness=0.9, coating=0.0, ior=1.5, scale=1.0, seed=101, anisotropy=1.0, subsurface=0.0, porosity=0.08, emissive=0.0, rd_feed=0.036, rd_kill=0.062, film_nm=380.0, wear=0.35, oxidation=0.25, biofilm=0.20)
}
"""
        prog = parse_program(verbose_legacy)
        compact = format_program(prog)
        self.assertNotIn("offset=[0", compact)
        self.assertNotIn("detached=false", compact)
        self.assertNotIn("inflate: 0", compact)
        self.assertNotIn("rotate: [0", compact)
        self.assertNotIn("rest_euler: [0", compact)
        self.assertNotIn("hue_shift=[0", compact)
        self.assertNotIn("rd_feed=", compact)
        reduction = 1.0 - (len(compact) / float(len(verbose_legacy)))
        self.assertGreaterEqual(reduction, 0.17)
        reparsed = parse_program(compact)
        self.assertEqual(prog, reparsed)
        self.assertEqual(compact, format_program(reparsed))

    def test_v32_versioned_meta_size_step_and_audit_profile(self):
        default_prog = parse_program(MINIMAL)
        self.assertEqual(default_prog.meta.dsl_version, "3.2")
        self.assertEqual(default_prog.meta.size_step, 1.0)
        self.assertEqual(default_prog.meta.audit_profile, "strict")

        for step, valid_vec, bad_vec in (
            (0.5, "[2.5, 4, 1.5]", "[2.25, 4, 1.5]"),
            (0.25, "[1.25, 0.75, 2.0]", "[1.125, 0.75, 2.0]"),
            (0.125, "[0.125, 1.375, 0.5]", "[0.1, 1.0, 0.5]"),
        ):
            src = (
                f'@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "3.2", size_step: {step}, audit_profile: authored }}\n'
                "@palette { p: ramp(base_lch=[0.6, 0.1, 50], steps=4) }\n"
                f"@rig {{ root bone r {{ pivot_local: [0, 8, 0] box b {{ size: {valid_vec}, material: p }} }} }}\n"
            )
            parsed = parse_program(src)
            self.assertEqual(parsed.meta.size_step, step)
            self.assertEqual(parsed.meta.audit_profile, "authored")
            self.assertEqual(parsed, parse_program(format_program(parsed)))

            with self.assertRaises(DSLValidationError):
                parse_program(src.replace(valid_vec, bad_vec))

        with self.assertRaises(DSLValidationError):
            parse_program(MINIMAL.replace("texel_density: 16", "texel_density: 16, size_step: 0.2"))

    def test_v32_textures_planes_rotated_boxes_uv_and_joint_roundtrip_h6(self):
        src = """@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, size_step: 0.5, audit_profile: authored }
@palette {
  fur: ramp(base_lch=[0.62, 0.14, 48], steps=5)
}
@textures {
  atlas_main: image("textures/entity/kit_fox.png", 64, 32)
}
@rig {
  root bone body {
    pivot_local: [0, 7, 0]
    box torso { size: [5.5, 5, 9], material: fur, uv: box(0, 0), texture: atlas_main }
    box snout place(base_box=torso, anchor=[0, 0.2, -1], align=[0, 0, 1]) {
      size: [2.5, 2, 3.5], material: fur, rotate: [12.5, 0, 0], pivot: [0, 1, -4.5],
      uv_faces: { north: [16, 0, 21, 4], up: [21, 0, 26, 7] }, texture: atlas_main
    }
    bone ear_l joint(parent=torso, at=[2.25, 2.5, -3.5]) mirror_x(ear_r) {
      rest_euler: [8, 15, -18]
      plane ear_blade { size: [2.5, 4], normal: z, mat: fur, uv: box(32, 0), texture: atlas_main }
    }
  }
}
"""
        prog = parse_program(src)
        self.assertEqual(prog.textures, (TextureDecl("atlas_main", "textures/entity/kit_fox.png", 64, 32),))
        ear_bone = prog.root.items[2]
        self.assertIsInstance(ear_bone, Bone)
        self.assertTrue(ear_bone.attach.joint_mode)
        ear_plane = ear_bone.items[0]
        self.assertEqual(ear_plane.plane_normal, "z")
        self.assertEqual(ear_plane.size, (2.5, 4.0, 0.0))

        canon = format_program(prog)
        self.assertIn("plane ear_blade", canon)
        self.assertIn("joint(parent=torso, at=[2.25, 2.5, -3.5])", canon)
        self.assertEqual(prog, parse_program(canon))
        # Verify JSON serializability via dataclasses.asdict (used by CLI)
        serialized = json.dumps(asdict(prog))
        self.assertIn("textures/entity/kit_fox.png", serialized)

    def test_v32_let_def_use_chain_mirror_and_radial_zero_eval(self):
        # Verify zero eval/exec calls exist in DSL parser or AST validator source trees
        for rel in ("src/synccraft/dsl/parser.py", "src/synccraft/dsl/ast_nodes.py"):
            tree = ast.parse((ROOT / rel).read_text())
            forbidden = [
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"eval", "exec", "compile"}
            ]
            self.assertEqual(forbidden, [], f"Forbidden dynamic execution in {rel}: {forbidden}")

        src = """
let hub_w = 4 + 2
let pod_len = (hub_w * 2) - 4

def thrusters(length, width=2) {
  rest_euler: [-10, 0, 0]
  box housing { size: [width, width, length], material: alloy }
  locator nozzle { offset: [0, 0, -length / 2], role: vfx_emitter }
}

@meta { archetype: entity, target_profile: geckolib, texel_density: 16 }
@palette { alloy: ramp(base_lch=[0.65, 0.03, 230], steps=4) }
@rig {
  root bone core {
    pivot_local: [0, 14, 0]
    box hub { size: [hub_w, 4, hub_w], material: alloy }
    use thrusters(pod_len, width=2) as pod_front attach(parent_box=hub, anchor=[0, -0.5, -1], align=[0, 0, 1]) mirror(axis=z, as=pod_rear)
    bone spoke attach(parent_box=hub, anchor=[1, 0, 0], align=[-1, 0, 0]) radial(count=4, axis=y) {
      box fin { size: [3, 1, 2], material: alloy }
    }
    chain tail attach(parent_box=hub, anchor=[0, 0, 1], align=[0, 0, -1]) segments=3 taper=0.8 {
      axis: +z
      rest_euler: [-5, 0, 0]
      box seg { size: [4, 4, 3], material: alloy }
    }
  }
}
"""
        prog = parse_program(src)
        hub = prog.root.items[0]
        self.assertEqual(hub.size, (6, 4, 6))
        pod = prog.root.items[1]
        self.assertIsInstance(pod, Bone)
        self.assertEqual(pod.name, "pod_front")
        self.assertEqual(pod.mirror_axis, "z")
        self.assertEqual(pod.mirror_name, "pod_rear")
        spoke = prog.root.items[2]
        self.assertEqual(spoke.radial, RadialSpec(count=4, axis="y"))
        tail_1 = prog.root.items[3]
        self.assertEqual(tail_1.name, "tail_1")
        self.assertEqual(tail_1.items[0].size, (4, 4, 3))
        tail_2 = tail_1.items[1]
        self.assertEqual(tail_2.name, "tail_2")
        self.assertEqual(tail_2.items[0].size, (3, 3, 2))

        canon = format_program(prog)
        self.assertEqual(prog, parse_program(canon))

        # Division by zero in let must raise DSLParseError cleanly
        with self.assertRaises(DSLParseError):
            parse_program("let bad = 4 / 0\n" + MINIMAL)

    def test_v32_nbone_ik_keyframes_springs_and_controller_ast(self):
        src = """@meta { archetype: entity, target_profile: geckolib, texel_density: 16 }
@palette { shell: ramp(base_lch=[0.5, 0.12, 160], steps=5) }
@rig {
  root bone root {
    pivot_local: [0, 12, 0]
    box carapace { size: [8, 6, 10], material: shell }
    bone cape attach(parent_box=carapace, anchor=[0, 0.8, 1], align=[0, 1, -1]) {
      box drape { size: [6, 8, 1], material: shell }
    }
    chain tentacle attach(parent_box=carapace, anchor=[0, -0.5, -1], align=[0, 0, 1]) segments=3 taper=0.85 {
      joint_limits: [[-55, 55], [-40, 40], [-25, 25]]
      box link { size: [2, 2, 4], material: shell }
    }
  }
}
@kinematics {
  ik_chain reach_ik { bones: [tentacle_1, tentacle_2, tentacle_3], effector: [0, 0, -3], pole: [0, 6, -5] }
}
@behavior {
  clip idle {
    duration_s: 2.0
    key cape.rotation { 0.0: [0, 0, 0], 1.0: [8, 0, 0], 2.0: [0, 0, 0], catmullrom }
  }
  clip walk {
    duration_s: 1.2
    ik_drive reach_ik: foot_cycle(stride=2.0, lift=1.0)
    key root.position { 0.0: [0, 0, 0], 0.6: [0, 0.5, 0], 1.2: [0, 0, 0] }
    key cape.scale { 0.0: [1, 1, 1], 1.2: [1, 1.05, 1] }
  }
  spring cape { stiffness: 40, damping: 6 }
}
@controller {
  initial: idle
  idle -> walk when speed > 0.1, blend 0.2
  walk -> idle when speed <= 0.05, blend 0.25
}
"""
        prog = parse_program(src)
        self.assertIsInstance(prog.kinematics.ik_chains[0], IKChainDecl)
        self.assertEqual(prog.kinematics.ik_chains[0].bones, ("tentacle_1", "tentacle_2", "tentacle_3"))
        self.assertEqual(prog.kinematics.ik_chains[0].pole, (0.0, 6.0, -5.0))
        self.assertEqual(prog.behavior.springs, (SpringDecl("cape", 40.0, 6.0),))
        self.assertIsInstance(prog.behavior.clips[0].keyframes[0], KeyframeTrackDecl)
        self.assertIsInstance(prog.controller, ControllerDecl)
        self.assertEqual(len(prog.controller.transitions), 2)
        self.assertEqual(prog, parse_program(format_program(prog)))

    def test_single_grammar_truth_lark_and_parser_parity_m1(self):
        grammar_path = ROOT / "src/synccraft/dsl/grammar.lark"
        grammar_text = grammar_path.read_text()

        # 1. Structural completeness: every referenced rule/terminal in grammar.lark must be defined & reachable
        defined_symbols: set[str] = {"WS"}
        rule_bodies: dict[str, str] = {}
        current_lhs: str | None = None
        for raw_line in grammar_text.splitlines():
            line = raw_line.split("//", 1)[0].strip()
            if not line or line.startswith("%"):
                continue
            m = re.match(r"^\??([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$", line)
            if m:
                current_lhs = m.group(1)
                defined_symbols.add(current_lhs)
                rule_bodies[current_lhs] = m.group(2)
            elif current_lhs is not None and line.startswith("|"):
                rule_bodies[current_lhs] += " " + line[1:]

        self.assertIn("start", rule_bodies)
        referenced_by: dict[str, set[str]] = {}
        for lhs, rhs in rule_bodies.items():
            # Strip regexes and quoted literals before scanning symbol references
            cleaned = re.sub(r"/(?:[^/\\]|\\.)+/", " ", rhs)
            cleaned = re.sub(r'"(?:[^"\\]|\\.)*"', " ", cleaned)
            refs = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", cleaned))
            undefined = refs - defined_symbols
            self.assertEqual(undefined, set(), f"Undefined symbol(s) in grammar.lark rule {lhs}: {undefined}")
            referenced_by[lhs] = refs

        # Reachability from start across nonterminals
        reachable: set[str] = set()
        queue = ["start"]
        while queue:
            sym = queue.pop()
            if sym in reachable:
                continue
            reachable.add(sym)
            for nxt in referenced_by.get(sym, ()):
                if nxt in rule_bodies and nxt not in reachable:
                    queue.append(nxt)
        unreachable_rules = {r for r in rule_bodies if r.islower()} - reachable
        self.assertEqual(unreachable_rules, set(), f"Unreachable nonterminals in grammar.lark: {unreachable_rules}")

        # 2. Keyword & section parity between grammar.lark and Block-DSL v3.2 surface syntax
        required_literals = {
            "@meta", "@textures", "@palette", "@rig", "@materials", "@style",
            "@muscles", "@kinematics", "@behavior", "@controller",
            "dsl_version", "size_step", "audit_profile", "image",
            "let", "def", "use", "as", "chain", "segments", "taper",
            "box", "plane", "normal", "rotate", "pivot", "uv", "uv_faces", "texture",
            "attach", "joint", "mirror_x", "mirror", "radial",
            "ik_chain", "bones", "pole", "look_chain", "spring", "stiffness", "damping",
            "clip", "track", "key", "muscle_drive", "ik_drive", "state",
            "initial", "when", "blend", "->",
        }
        quoted_literals = set(re.findall(r'"([^"]+)"', grammar_text))
        missing = required_literals - quoted_literals
        self.assertEqual(missing, set(), f"grammar.lark missing v3.2 literals: {missing}")

        # 3. If lark is installed in the environment, execute live parse-tree parity on v3.0/v3.1/v3.2 fixtures
        if importlib.util.find_spec("lark") is not None:
            from lark import Lark  # type: ignore[import-not-found]

            lark_parser = Lark(grammar_text, start="start")
            valid_samples = [
                MINIMAL,
                *(p.read_text() for p in sorted((ROOT / "examples").glob("*.blockdsl"))),
                format_program(
                    Program(
                        Meta("entity", "bedrock_entity", 16, size_step=0.5, audit_profile="authored"),
                        [Palette("fur", (0.6, 0.1, 45.0), 4)],
                        Bone(
                            "root",
                            pivot_local=(0.0, 8.0, 0.0),
                            items=[
                                Box("body", (4.0, 4.0, 6.0), "fur", uv=(0.0, 0.0), texture="tex"),
                                Bone(
                                    "ear",
                                    attach=Joint("body", at=(2.0, 2.0, -2.0)),
                                    items=[Plane("ear_p", (2.0, 3.0), "z", "fur")],
                                ),
                            ],
                        ),
                        textures=(TextureDecl("tex", "textures/entity/test.png", 64, 32),),
                    )
                ),
            ]
            for sample in valid_samples:
                lark_parser.parse(sample)
                parse_program(sample)


if __name__ == "__main__":
    unittest.main()
