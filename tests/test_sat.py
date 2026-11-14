"""Exact SAT detection differs from the lower-bound optimization surrogate."""

import unittest
from types import SimpleNamespace

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.kinematics.se3_chain import Scene, WorldBox, euler_xyz_matrix
from synccraft.kinematics.sat_potential import sat_obb


def box(name, center=(0, 0, 0), size=(2, 2, 2), euler=(0, 0, 0), inflate=0.0):
    return WorldBox(name, "root", np.array(center, dtype=float), euler_xyz_matrix(euler),
                    np.array(size, dtype=float), inflate, np.zeros(3), "steel")


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
        # Parallel diagonal beams have overlapping axis-aligned bounds but a gap
        # in their narrow local direction.
        a = box("a", size=(8, 1, 1), euler=(0, 0, 45))
        b = box("b", center=a.rotation[:, 1] * 1.2, size=(8, 1, 1), euler=(0, 0, 45))
        ac, bc = a.corners(), b.corners()
        self.assertTrue(np.all(ac.min(0) <= bc.max(0)))
        self.assertTrue(np.all(bc.min(0) <= ac.max(0)))
        self.assertFalse(sat_obb(a, b).collision)

    def test_softmin_can_miss_positive_overlap_audit_does_not(self):
        # Close to a corner, repeated small positive candidate-axis overlaps make
        # the unnormalised soft minimum negative, despite a true collision.
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
        scene = Scene((a, b, c), (), frozenset({edge}),
                      SimpleNamespace(target_profile="bbmodel_5"), {edge: "place"})
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


if __name__ == "__main__":
    unittest.main()
