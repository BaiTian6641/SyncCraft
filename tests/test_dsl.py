"""Static subset contract tests; no tokenizer/model/network dependency."""
from dataclasses import replace
from pathlib import Path
import random
import unittest

from synccraft.dsl import (
    Attach, Bone, Box, DSLParseError, DSLValidationError, Meta, Palette,
    Place, Program, UnsupportedFeature, format_program, parse_program,
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
                Bone("root_a", pivot_local=(rng.uniform(-8, 8), 0, 0), items=[
                    Box("base_a", (rng.randint(1,8), rng.randint(1,8), rng.randint(1,8)), "material_a"),
                    Box("cap_a", (1,2,3), "material_a", Place("base_a", (0,1,0), (0,-1,0))),
                    Bone("child_a", attach=Attach("base_a", (1,0,0), (-1,0,0)),
                         rest_euler=(0,0,rng.choice([-45,0,22.5,45])),
                         items=[Box("tip_a", (2,2,2), "material_a")]),
                ]),
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
        child = Bone("joint", attach=Attach("shaft", (0,1,0), (0,-1,0), (0,0.6,0)), items=[Box("child", (1,1,1), "metal")])
        with self.assertRaises(DSLValidationError):
            validate_program(replace(p, root=replace(p.root, items=p.root.items + [child])))
        detached = replace(child, attach=replace(child.attach, detached=True))
        validate_program(replace(p, root=replace(p.root, items=p.root.items + [detached])))
        with self.assertRaises(DSLValidationError):
            validate_program(replace(p, root=replace(p.root, items=[detached] + p.root.items)))

    def test_unit_anchor_and_origin_place(self):
        p = parse_program(MINIMAL)
        original = p.root.items[0]
        for place in (Place(None, (1,0,0), (0,0,0)), Place(None, (0,0,0), (1.1,0,0))):
            with self.assertRaises(DSLValidationError):
                validate_program(replace(p, root=replace(p.root, items=[replace(original, place=place)])))

    def test_mirror_generated_identifiers_reserved(self):
        p = parse_program((ROOT / "examples/mirrored_wings.blockdsl").read_text())
        duplicate = Box("right_wing__left_panel", (1,1,1), "slate", Place("hub", (0,1,0), (0,-1,0)))
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
        self.assertEqual((error.exception.line, error.exception.column), (3,1))

    def test_comments(self):
        self.assertEqual(parse_program(MINIMAL), parse_program("// hello\n" + MINIMAL + "# tail\n"))

    def test_direct_ast_canonical_numbers_have_stable_hash_input(self):
        program = parse_program(MINIMAL)
        program = replace(program, root=replace(program.root, pivot_local=(8, 8, 8), items=[
            replace(program.root.items[0], inflate=0),
        ]))
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
        invalid = replace(program.root.items[0], place=Place([], (0,0,0), (0,0,0)))
        with self.assertRaises(DSLValidationError):
            validate_program(replace(program, root=replace(program.root, items=[invalid])))


if __name__ == "__main__":
    unittest.main()
