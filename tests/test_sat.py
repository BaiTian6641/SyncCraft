"""Exact SAT detection, planar Z-fighting checks, M4 hinge bounds, and strict vs authored audit tests."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from synccraft.audit.sync_audit import (
    MAX_HINGE_ALLOWANCE_UNITS,
    audit_scene,
    compute_delta_hinge,
)
from synccraft.dsl.parser import parse_program
from synccraft.kinematics.sat_potential import sat_obb
from synccraft.kinematics.se3_chain import Scene, WorldBox, compile_geometry, euler_xyz_matrix


def box(name, center=(0, 0, 0), size=(2, 2, 2), euler=(0, 0, 0), inflate=0.0, plane_normal=None):
    return WorldBox(
        name,
        "root",
        np.array(center, dtype=float),
        euler_xyz_matrix(euler),
        np.array(size, dtype=float),
        inflate,
        np.zeros(3),
        "steel",
        plane_normal=plane_normal,
    )


class SATTests(unittest.TestCase):
    def test_identical_parallel_boxes_detect_collision(self):
        result = sat_obb(box("a"), box("b"))
        self.assertTrue(result.collision)
        self.assertAlmostEqual(result.penetration_depth, 2.0)
        self.assertGreater(result.surrogate_energy, 0)
        self.assertLess(result.tested_axes, 15)  # Zero cross axes are omitted.

    def test_parallel_zero_contact_and_separation(self):
        for center in ([2, 0, 0], [3, 0, 0]):
            result = sat_obb(box("a"), box("b", center))
            self.assertFalse(result.collision)
            self.assertEqual(result.penetration_depth, 0)
            self.assertEqual(result.surrogate_energy, 0)

    def test_rotated_flush_contacts_1000_deterministic_cases(self):
        rng = np.random.default_rng(20261113)
        for _ in range(1000):
            euler = rng.uniform(-180, 180, 3)
            a = box("a", rng.uniform(-30, 30, 3), rng.integers(1, 12, 3), euler)
            sizes = rng.integers(1, 12, 3)
            center = a.center + a.rotation[:, 0] * (a.size[0] + sizes[0]) / 2
            b = box("b", center, sizes, euler)
            result = sat_obb(a, b)
            self.assertFalse(result.collision)
            self.assertEqual(result.penetration_depth, 0)
            self.assertEqual(result.surrogate_energy, 0)
        # No claim about PyTorch gradients follows from this CPU regression.

    def test_positive_point_two_overlap_and_inflation(self):
        result = sat_obb(box("a"), box("b", [1.8, 0, 0]))
        self.assertTrue(result.collision)
        self.assertAlmostEqual(result.penetration_depth, 0.2)
        inflated = sat_obb(box("a", inflate=0.05), box("b", [2, 0, 0], inflate=0.05))
        self.assertTrue(inflated.collision)
        self.assertAlmostEqual(inflated.penetration_depth, 0.1)

    def test_oblique_axes_find_separation_despite_aabb_overlap(self):
        a = box("a", size=(8, 1, 1), euler=(0, 0, 45))
        b = box("b", center=a.rotation[:, 1] * 1.2, size=(8, 1, 1), euler=(0, 0, 45))
        ac, bc = a.corners(), b.corners()
        self.assertTrue(np.all(ac.min(0) <= bc.max(0)))
        self.assertTrue(np.all(bc.min(0) <= ac.max(0)))
        self.assertFalse(sat_obb(a, b).collision)

    def test_softmin_can_miss_positive_overlap_audit_does_not(self):
        a, b = box("a"), box("b", [1.99, 1.99, 1.99])
        result = sat_obb(a, b)
        self.assertTrue(result.collision)
        self.assertLess(result.softmin_overlap, 0)
        self.assertEqual(result.surrogate_energy, 0)
        scene = Scene((a, b), (), frozenset(), SimpleNamespace(target_profile="bbmodel_5"))
        self.assertFalse(audit_scene(scene)["passed"])

    def test_audit_reports_declared_exemptions_without_sibling_exemption(self):
        a, b, c = box("a"), box("b", [0.5, 0, 0]), box("c", [0, 0.5, 0])
        edge = ("a", "b")
        scene = Scene((a, b, c), (), frozenset({edge}), SimpleNamespace(target_profile="bbmodel_5"), {edge: "place"})
        report = audit_scene(scene)
        self.assertFalse(report["passed"])
        self.assertFalse(report["complete_for_release"])
        self.assertEqual(len(report["declared_contact_pairs"]), 1)
        self.assertEqual(len(report["collision_pairs"]), 2)
        self.assertGreater(report["declared_contact_pairs"][0]["penetration_depth"], 0)

    def test_invalid_obb_rejected(self):
        invalid = box("bad")
        invalid.rotation[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            sat_obb(invalid, box("ok"))

    def test_zero_thickness_plane_sat_flush_piercing_and_transverse_intersection(self):
        ear = box("ear", center=(2.0, 2.0, 0.0), size=(4.0, 4.0, 0.0), plane_normal="z")
        head = box("head", center=(0.0, 0.0, 2.0), size=(4.0, 4.0, 4.0))
        # Touching at z=0: zero SAT penetration
        flush_res = sat_obb(ear, head)
        self.assertFalse(flush_res.collision)
        self.assertEqual(flush_res.penetration_depth, 0.0)

        # Piercing through center of 3D box: positive SAT overlap
        pierce_res = sat_obb(box("blade", center=(0.0, 0.0, 0.0), size=(6.0, 6.0, 0.0), plane_normal="z"), box("block", size=(4.0, 4.0, 4.0)))
        self.assertTrue(pierce_res.collision)
        self.assertAlmostEqual(pierce_res.penetration_depth, 2.0)

        # Two perpendicular zero-thickness planes crossing at origin collide
        p1 = box("p1", size=(4.0, 4.0, 0.0), plane_normal="z")
        p2 = box("p2", size=(0.0, 4.0, 4.0), plane_normal="x")
        cross_res = sat_obb(p1, p2)
        self.assertTrue(cross_res.collision)
        self.assertAlmostEqual(cross_res.penetration_depth, 2.0)

    def test_coplanar_z_fighting_detected_in_strict_and_authored_profiles(self):
        # 1. Two coplanar overlapping 0-thickness planes -> caught under both strict & authored
        p_a = box("p_a", center=(0.0, 0.0, 1.0), size=(4.0, 4.0, 0.0), plane_normal="z")
        p_b = box("p_b", center=(1.0, 0.5, 1.0), size=(4.0, 4.0, 0.0), plane_normal="z")
        sc_planes = Scene((p_a, p_b), (), frozenset(), SimpleNamespace(target_profile="bedrock_entity", audit_profile="authored"))
        rep_auth = audit_scene(sc_planes)
        rep_strict = audit_scene(sc_planes, audit_profile="strict")
        self.assertFalse(rep_auth["passed"])
        self.assertFalse(rep_strict["passed"])
        self.assertEqual(len(rep_auth["z_fighting_pairs"]), 1)

        # 2. Zero-thickness decal flat on non-culled 3D box face with inflate=0 -> caught; with inflate=0.05 -> clean
        cube = box("cube", center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0))
        decal_bad = box("decal_bad", center=(0.0, 0.0, 2.0), size=(2.0, 2.0, 0.0), plane_normal="z")
        sc_decal_bad = Scene((cube, decal_bad), (), frozenset({("cube", "decal_bad")}), SimpleNamespace(target_profile="bedrock_entity", audit_profile="authored"), {("cube", "decal_bad"): "place"})
        self.assertFalse(audit_scene(sc_decal_bad)["passed"])

        decal_offset = box("decal_ok", center=(0.0, 0.0, 2.05), size=(2.0, 2.0, 0.0), plane_normal="z")
        sc_decal_ok = Scene((cube, decal_offset), (), frozenset(), SimpleNamespace(target_profile="bedrock_entity", audit_profile="authored"))
        self.assertTrue(audit_scene(sc_decal_ok)["passed"])

        # 3. Duplicate coincident 3D boxes -> caught even under authored
        dup_1 = box("dup_1", center=(1.0, 2.0, 3.0), size=(4.0, 6.0, 2.0))
        dup_2 = box("dup_2", center=(1.0, 2.0, 3.0), size=(4.0, 6.0, 2.0))
        sc_dup = Scene((dup_1, dup_2), (), frozenset(), SimpleNamespace(target_profile="bedrock_entity", audit_profile="authored"))
        self.assertFalse(audit_scene(sc_dup)["passed"])
        self.assertEqual(rep_auth["z_fighting_pairs"][0]["violation"], "coplanar_z_fighting")

    def test_authored_audit_profile_7_bone_9_cube_fractional_planar_rotated_fixture_h6(self):
        # 7-bone (body, head, ear_l, ear_r, tail, leg_l, leg_r), 9-primitive authored rig with:
        # - fractional sizes on size_step: 0.25
        # - zero-thickness ear planes (ear_l, ear_r)
        # - rotated snout & angled cheek cubes with intentional authored joint overlaps
        authored_dsl = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, size_step: 0.25, audit_profile: authored }
        @palette { pelt: ramp(base_lch=[0.64, 0.14, 52], steps=5) }
        @rig {
          root bone body {
            pivot_local: [0, 6.5, 0]
            box torso { size: [4.5, 4.25, 7.5], material: pelt }
            box chest_tuft place(base_box=torso, anchor=[0, 0.2, -1], align=[0, 0, 0], offset=[0, 0, 0.5]) {
              size: [3.5, 3.0, 2.25], material: pelt, rotate: [14, 0, 0]
            }
            bone head joint(parent=torso, at=[0, 1.75, -3.25]) {
              rest_euler: [-6, 0, 0]
              box skull { size: [4.0, 3.75, 4.0], material: pelt }
              box snout place(base_box=skull, anchor=[0, -0.25, -1], align=[0, 0, 0.5]) {
                size: [2.25, 1.75, 2.5], material: pelt, rotate: [10, 0, 0]
              }
              bone ear_l joint(parent=skull, at=[1.5, 1.75, -0.5], align=[0, -1, 0]) mirror_x(ear_r) {
                rest_euler: [12, 18, -22]
                plane ear_fin { size: [2.25, 3.5], normal: z, material: pelt }
              }
            }
            bone tail joint(parent=torso, at=[0, 1.25, 3.25]) {
              rest_euler: [28, 0, 0]
              box plume { size: [2.25, 2.25, 5.5], material: pelt }
            }
            bone leg_l joint(parent=torso, at=[1.5, -1.5, -2.0]) mirror_x(leg_r) {
              box paw { size: [1.75, 4.5, 1.75], material: pelt }
            }
          }
        }
        """
        prog = parse_program(authored_dsl)
        scene = compile_geometry(prog)
        self.assertEqual(len(scene.bones), 7)
        self.assertEqual(len(scene.boxes), 9)

        rep_authored = audit_scene(scene)
        self.assertTrue(rep_authored["passed"], msg=f"Authored fixture failed: {rep_authored}")
        self.assertEqual(rep_authored["audit_profile"], "authored")
        self.assertGreater(len(rep_authored["authored_exempt_pairs"]), 0)

        # Under strict P1-02 audit, deep authored joint overlaps are rejected
        rep_strict = audit_scene(scene, audit_profile="strict")
        self.assertFalse(rep_strict["passed"])

        # Non-adjacent bone collision (e.g. tail plume colliding with ear_l plane across cousin/uncle bones) fails authored audit!
        bad_boxes = list(scene.boxes)
        ear_box = scene.box_by_name["ear_fin"]
        for idx, bx in enumerate(bad_boxes):
            if bx.name == "plume":
                bad_boxes[idx] = WorldBox(
                    bx.name, bx.bone_name, ear_box.center.copy(), bx.rotation, bx.size, bx.inflate, bx.local_center, bx.material
                )
        non_adj_scene = Scene(tuple(bad_boxes), scene.bones, scene.adjacency, scene.meta, scene.adjacency_reasons, scene.locators)
        rep_bad = audit_scene(non_adj_scene)
        self.assertFalse(rep_bad["passed"])
        self.assertTrue(any(set(c["pair"]) == {"ear_fin", "plume"} for c in rep_bad["collision_pairs"]))

    def test_m4_bounded_delta_hinge_allowance_caps_thick_blocks(self):
        big = np.array([16.0, 16.0, 16.0])
        self.assertEqual(compute_delta_hinge(0.0, big, big), 0.0)
        self.assertEqual(compute_delta_hinge(math.radians(65.0), big, big), 0.0)
        capped = compute_delta_hinge(math.radians(45.0), big, big)
        self.assertEqual(capped, MAX_HINGE_ALLOWANCE_UNITS)
        self.assertEqual(MAX_HINGE_ALLOWANCE_UNITS, 2.5)


if __name__ == "__main__":
    unittest.main()
