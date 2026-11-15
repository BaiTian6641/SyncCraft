"""Reference SE(3) geometry, N-bone IK, keyframe/spring, and Zajac/Hill biomechanics regressions."""

from dataclasses import replace
import math
import unittest

import numpy as np

from synccraft.data_gen.ik_motion_templates import solve_n_bone_ik
from synccraft.dsl.parser import parse_program
from synccraft.kinematics.musculoskeletal import (
    HILL_ACTIVE_WIDTH_DEFAULT,
    MUSCULOSKELETAL_FIDELITY_LABEL,
    evaluate_behavior_clip,
    evaluate_controller_transition,
    evaluate_muscles_at_activations,
    hill_active_force_length,
    hill_passive_force_length,
)
from synccraft.kinematics.se3_chain import compile_geometry, euler_xyz_matrix


def program_with(rig: str):
    return parse_program(
        "@meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }\n"
        "@palette { steel: ramp(base_lch=[0.7,0.02,250], steps=4) }\n"
        "@rig {" + rig + "}"
    )


class KinematicsTests(unittest.TestCase):
    def test_euler_rotation_order_and_proper_rotation(self):
        np.testing.assert_allclose(
            euler_xyz_matrix([90, 0, 90]) @ [0, 1, 0], [0, 0, 1], atol=1e-14
        )
        rotation = euler_xyz_matrix([17, 31, -28])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_root_and_same_bone_place(self):
        scene = compile_geometry(
            program_with(
                """
          root bone assembly {
            pivot_local: [8,8,8]
            box grip { size:[2,6,2], material:steel }
            box pommel place(base_box=grip, anchor=[0,-1,0], align=[0,1,0]) {
              size:[4,2,4], material:steel
            }
          }
        """
            )
        )
        np.testing.assert_allclose(scene.box_by_name["grip"].center, [8, 8, 8])
        np.testing.assert_allclose(scene.box_by_name["pommel"].center, [8, 4, 8])
        self.assertEqual(scene.adjacency, frozenset({("grip", "pommel")}))

    def test_child_joint_and_rotated_primitive_alignment(self):
        scene = compile_geometry(
            program_with(
                """
          root bone assembly {
            pivot_local: [8,8,8]
            box base { size:[4,2,2], material:steel }
            bone arm attach(parent_box=base,anchor=[0,1,0],align=[0,-1,0]) {
              rest_euler:[0,0,90]
              box tip { size:[2,4,2], material:steel }
            }
          }
        """
            )
        )
        bone = scene.bone_by_name["arm"]
        tip = scene.box_by_name["tip"]
        np.testing.assert_allclose(bone.pivot_world, [8, 9, 8])
        np.testing.assert_allclose(tip.local_center, [0, 2, 0])
        np.testing.assert_allclose(tip.center, [6, 9, 8], atol=1e-14)
        contact = tip.center + tip.rotation @ (tip.size / 2 * [0, -1, 0])
        np.testing.assert_allclose(contact, bone.pivot_world, atol=1e-14)

    def test_nested_parent_rotation_is_shared(self):
        scene = compile_geometry(
            program_with(
                """
          root bone assembly {
            pivot_local:[0,0,0]
            box base {size:[2,2,2],material:steel}
            bone arm attach(parent_box=base,anchor=[1,0,0],align=[-1,0,0]) {
              rest_euler:[0,0,90]
              box first {size:[2,2,2],material:steel}
              bone end attach(parent_box=first,anchor=[1,0,0],align=[-1,0,0]) {
                rest_euler:[0,90,0]
                box last {size:[2,2,2],material:steel}
              }
            }
          }
        """
            )
        )
        end = scene.bone_by_name["end"]
        np.testing.assert_allclose(end.pivot_world, [1, 2, 0], atol=1e-14)
        expected = euler_xyz_matrix([0, 0, 90]) @ euler_xyz_matrix([0, 90, 0])
        np.testing.assert_allclose(end.rotation, expected, atol=1e-14)

    def test_mirror_generates_proper_reflected_subtree(self):
        scene = compile_geometry(
            program_with(
                """
          root bone assembly {
            pivot_local:[8,8,8]
            box base {size:[4,2,2],material:steel}
            bone left attach(parent_box=base,anchor=[1,0,0],align=[-1,0,0]) mirror_x(right) {
              rest_euler:[13,17,22]
              box wing {size:[4,2,2],material:steel}
              box tip place(base_box=wing,anchor=[1,0,0],align=[-1,0,0]) {
                size:[2,1,1],material:steel
              }
              bone feather attach(parent_box=wing,anchor=[0,1,0],align=[0,-1,0]) {
                rest_euler:[7,11,19]
                box plume {size:[2,3,1],material:steel}
              }
            }
          }
        """
            )
        )
        s = np.diag([-1.0, 1.0, 1.0])
        reference = np.array([8, 8, 8])
        for original in ("wing", "tip", "plume"):
            left = scene.box_by_name[original]
            right = scene.box_by_name["right__" + original]
            np.testing.assert_allclose(right.center, reference + s @ (left.center - reference), atol=1e-13)
            np.testing.assert_allclose(right.rotation, s @ left.rotation @ s, atol=1e-13)
            self.assertAlmostEqual(np.linalg.det(right.rotation), 1)
            reflected_corners = (left.corners() - reference) @ s + reference
            left_set = {tuple(np.round(v, 10)) for v in reflected_corners}
            right_set = {tuple(np.round(v, 10)) for v in right.corners()}
            self.assertEqual(left_set, right_set)
        self.assertIn(("base", "right__wing"), scene.adjacency)
        self.assertNotIn(("right__wing", "wing"), scene.adjacency)

    def test_detached_attachment_not_exempt(self):
        scene = compile_geometry(
            program_with(
                """
          root bone assembly {
            pivot_local:[0,0,0]
            box base {size:[2,2,2],material:steel}
            bone detached_part attach(parent_box=base,anchor=[1,0,0],align=[-1,0,0],offset=[2,0,0],detached=true) {
              box floating {size:[2,2,2],material:steel}
            }
          }
        """
            )
        )
        self.assertNotIn(("base", "floating"), scene.adjacency)

    def test_direct_ast_calls_cannot_bypass_validation(self):
        program = program_with(
            """
          root bone assembly { pivot_local:[0,0,0]
            box base {size:[2,2,2],material:steel}
          }
        """
        )
        invalid_box = replace(program.root.items[0], size=(True, 2, 2))
        invalid_root = replace(program.root, items=[invalid_box])
        with self.assertRaises(ValueError):
            compile_geometry(replace(program, root=invalid_root))

    def test_box_local_rotate_and_pivot_compile_exact_world_corners_without_extra_bones(self):
        prog = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16 }
            @palette { p: ramp(base_lch=[0.6, 0.1, 40], steps=4) }
            @rig {
              root bone body {
                pivot_local: [0, 10, 0]
                box torso { size: [6, 6, 8], material: p }
                bone head joint(parent=torso, at=[0, 3, -4]) {
                  rest_euler: [10, -20, 5]
                  box skull { size: [4, 4, 4], material: p }
                  box horn_l place(base_box=skull, anchor=[1, 1, 0], align=[-1, -1, 0]) {
                    size: [2, 4, 2], material: p, rotate: [25, -30, 15], pivot: [2, 2, 0]
                  }
                }
              }
            }
            """
        )
        scene = compile_geometry(prog)
        # No synthetic helper bones are created for rotated cubes
        self.assertEqual([b.name for b in scene.bones], ["body", "head"])
        head_b = scene.bone_by_name["head"]
        horn = scene.box_by_name["horn_l"]
        r_box = euler_xyz_matrix([25, -30, 15])
        unrot_local = np.array([2.0, 2.0, 0.0]) + np.array([1.0, 2.0, 0.0])
        piv_local = np.array([2.0, 2.0, 0.0])
        expected_local = piv_local + r_box @ (unrot_local - piv_local)
        expected_world_center = head_b.pivot_world + head_b.rotation @ expected_local
        expected_world_rot = head_b.rotation @ r_box
        np.testing.assert_allclose(horn.center, expected_world_center, atol=1e-13)
        np.testing.assert_allclose(horn.rotation, expected_world_rot, atol=1e-13)
        self.assertAlmostEqual(float(np.linalg.det(horn.rotation)), 1.0, places=12)

    def test_fractional_size_step_and_zero_thickness_plane_geometry(self):
        prog = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, size_step: 0.25 }
            @palette { fin_mat: ramp(base_lch=[0.7, 0.12, 200], steps=4) }
            @rig {
              root bone root {
                pivot_local: [0, 6, 0]
                box body { size: [2.25, 1.5, 3.75], material: fin_mat }
                plane dorsal place(base_box=body, anchor=[0, 1, 0], align=[0, -1, 0]) {
                  size: [1.75, 2.5], normal: x, material: fin_mat
                }
              }
            }
            """
        )
        scene = compile_geometry(prog)
        body = scene.box_by_name["body"]
        dorsal = scene.box_by_name["dorsal"]
        np.testing.assert_allclose(body.size, [2.25, 1.5, 3.75])
        np.testing.assert_allclose(dorsal.size, [0.0, 1.75, 2.5])
        np.testing.assert_allclose(dorsal.half_extents, [0.0, 0.875, 1.25])
        corners = dorsal.corners(include_inflate=False)
        np.testing.assert_allclose(corners[:, 0], 0.0, atol=1e-14)
        self.assertAlmostEqual(float(corners[:, 1].max() - corners[:, 1].min()), 1.75, places=12)

    def test_joint_attachment_mirror_z_and_radial_so3_parity(self):
        prog = parse_program(
            """
            @meta { archetype: entity, target_profile: geckolib, texel_density: 16 }
            @palette { m: ramp(base_lch=[0.6, 0.08, 120], steps=4) }
            @rig {
              root bone root {
                pivot_local: [0, 10, 0]
                box hub { size: [4, 4, 4], material: m }
                bone leg_f joint(parent=hub, at=[0, -2, -2]) mirror(axis=z, as=leg_b) {
                  rest_euler: [18, 12, -9]
                  box shin { size: [2, 4, 2], material: m }
                }
                bone petal attach(parent_box=hub, anchor=[1, 0, 0], align=[-1, 0, 0]) radial(count=6, axis=y) {
                  box blade { size: [3, 1, 1], material: m }
                }
              }
            }
            """
        )
        scene = compile_geometry(prog)
        sz = np.diag([1.0, 1.0, -1.0])
        ref = np.array([0.0, 10.0, 0.0])
        front = scene.box_by_name["shin"]
        back = scene.box_by_name["leg_b__shin"]
        np.testing.assert_allclose(back.center, ref + sz @ (front.center - ref), atol=1e-13)
        np.testing.assert_allclose(back.rotation, sz @ front.rotation @ sz, atol=1e-13)
        self.assertAlmostEqual(float(np.linalg.det(back.rotation)), 1.0, places=12)

        petal_names = ["petal"] + [f"petal_r{k}" for k in range(1, 6)]
        radii = []
        angles = []
        for pname in petal_names:
            pb = scene.bone_by_name[pname]
            self.assertAlmostEqual(float(np.linalg.det(pb.rotation)), 1.0, places=12)
            rel = pb.pivot_world - ref
            radii.append(math.hypot(float(rel[0]), float(rel[2])))
            angles.append(math.degrees(math.atan2(-float(rel[2]), float(rel[0]))) % 360.0)
        np.testing.assert_allclose(radii, [2.0] * 6, atol=1e-12)
        np.testing.assert_allclose(angles, [0.0, 60.0, 120.0, 180.0, 240.0, 300.0], atol=1e-10)

    def test_solve_n_bone_ik_four_bone_chain_joint_limits_and_pole_steering(self):
        prog = parse_program(
            """
            @meta { archetype: entity, target_profile: geckolib, texel_density: 16 }
            @palette { p: ramp(base_lch=[0.55, 0.1, 180], steps=4) }
            @rig {
              root bone root {
                pivot_local: [0, 16, 0]
                box mount { size: [4, 2, 4], material: p }
                chain arm attach(parent_box=mount, anchor=[0, -1, 0], align=[0, 1, 0]) segments=4 taper=1.0 {
                  axis: -y
                  joint_limits: [[-75, 75], [-60, 60], [-60, 60]]
                  box seg { size: [2, 3, 2], material: p }
                }
              }
            }
            """
        )
        bones = ("arm_1", "arm_2", "arm_3", "arm_4")
        target = (3.0, 7.0, -2.5)
        sol_pos_z = solve_n_bone_ik(prog, bones, (0.0, -3.0, 0.0), target, pole_world=(0.0, 10.0, 8.0))
        sol_neg_z = solve_n_bone_ik(prog, bones, (0.0, -3.0, 0.0), target, pole_world=(0.0, 10.0, -10.0))

        self.assertTrue(sol_pos_z.reachable)
        self.assertLess(sol_pos_z.position_error, 1e-3)
        self.assertLess(sol_neg_z.position_error, 1e-3)
        for b in bones:
            rx, ry, rz = sol_pos_z.pose_euler_deltas[b]
            self.assertTrue(-75.0 - 1e-6 <= rx <= 75.0 + 1e-6)
            self.assertTrue(-60.0 - 1e-6 <= ry <= 60.0 + 1e-6)
            self.assertTrue(-60.0 - 1e-6 <= rz <= 60.0 + 1e-6)

        # Verify pole vector steers interior joint Z coordinate in opposite half-spaces
        sc_pos = compile_geometry(prog, pose_euler_deltas=sol_pos_z.pose_euler_deltas)
        sc_neg = compile_geometry(prog, pose_euler_deltas=sol_neg_z.pose_euler_deltas)
        mid_z_pos = float(sc_pos.bone_by_name["arm_3"].pivot_world[2])
        mid_z_neg = float(sc_neg.bone_by_name["arm_3"].pivot_world[2])
        self.assertGreater(mid_z_pos, mid_z_neg + 0.5)

    def test_zajac_hill_force_length_curves_and_passive_elastic_equilibrium_h8(self):
        # 1. Analytical Zajac (1989) reference points on f_L(L_tilde) and f_PE(L_tilde)
        w = HILL_ACTIVE_WIDTH_DEFAULT
        self.assertAlmostEqual(hill_active_force_length(1.0, width=w), 1.0, places=14)
        self.assertAlmostEqual(hill_active_force_length(1.0 + w, width=w), math.exp(-1.0), places=14)
        self.assertAlmostEqual(hill_active_force_length(1.0 - w, width=w), math.exp(-1.0), places=14)
        self.assertAlmostEqual(
            hill_active_force_length(0.82, width=w),
            hill_active_force_length(1.18, width=w),
            places=14,
        )
        self.assertEqual(hill_passive_force_length(0.75), 0.0)
        self.assertEqual(hill_passive_force_length(1.0), 0.0)
        self.assertAlmostEqual(hill_passive_force_length(1.25), 0.0625, places=14)
        self.assertAlmostEqual(hill_passive_force_length(1.50), 0.25, places=14)

        # 2. Fidelity label & passive parallel elastic restoring torque at zero neural activation
        rig = parse_program(
            """
            @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16 }
            @palette { muscle_p: ramp(base_lch=[0.55, 0.16, 25], steps=4) }
            @rig {
              root bone root {
                pivot_local: [0, 14, 0]
                box shoulder { size: [4, 4, 4], material: muscle_p }
                bone forearm attach(parent_box=shoulder, anchor=[0, -1, 0], align=[0, 1, 0]) {
                  joint_limits: [[-80, 80], [-15, 15], [-15, 15]]
                  box ulna { size: [3, 6, 3], material: muscle_p }
                }
              }
            }
            @muscles {
              muscle flexor { origin_bone: root, insertion_bone: forearm, action_axis: x, moment_arm_deg: 36, belly_box: shoulder, bulge_inflate: 0.12 }
              muscle extensor { origin_bone: root, insertion_bone: forearm, action_axis: x, moment_arm_deg: -36, belly_box: shoulder, bulge_inflate: 0.10 }
            }
            """
        )
        res_active = evaluate_muscles_at_activations(rig, {"flexor": 0.8, "extensor": 0.0})
        self.assertEqual(res_active.fidelity_label, MUSCULOSKELETAL_FIDELITY_LABEL)
        self.assertEqual(res_active.fidelity_label, "reduced_order_qualitative")
        self.assertEqual(res_active.to_dict()["fidelity_label"], "reduced_order_qualitative")
        self.assertGreater(res_active.pose_euler_deltas["forearm"][0], 15.0)
        tel_by_name = {t.name: t for t in res_active.muscle_telemetry}
        # Antagonist extensor is stretched > 1.0 under flexion, generating positive f_PE
        self.assertGreater(tel_by_name["extensor"].passive_force_fpe, 0.0)
        self.assertLess(tel_by_name["flexor"].active_force_fl, 1.0)

        # Stretch flexor externally (forearm hyper-extended to -60 deg) with 0 neural activation:
        # passive parallel elastic curve f_PE > 0 pulls joint back toward flexion (> -60 deg)!
        res_passive = evaluate_muscles_at_activations(
            rig,
            {"flexor": 0.0, "extensor": 0.0},
            base_pose_deltas={"forearm": (-60.0, 0.0, 0.0)},
        )
        self.assertGreater(res_passive.pose_euler_deltas["forearm"][0], -60.0 + 0.05)

    def test_behavior_catmullrom_keyframes_damped_spring_and_controller_transitions(self):
        prog = parse_program(
            """
            @meta { archetype: entity, target_profile: geckolib, texel_density: 16 }
            @palette { cloth: ramp(base_lch=[0.5, 0.15, 280], steps=4) }
            @rig {
              root bone torso {
                pivot_local: [0, 12, 0]
                box chest { size: [6, 8, 4], material: cloth }
                bone head attach(parent_box=chest, anchor=[0, 1, 0], align=[0, -1, 0]) {
                  box skull { size: [4, 4, 4], material: cloth }
                  bone plume attach(parent_box=skull, anchor=[0, 1, 0], align=[0, -1, 0]) {
                    joint_limits: [[-45, 45], [-45, 45], [-45, 45]]
                    box crest { size: [1, 4, 2], material: cloth }
                  }
                }
              }
            }
            @behavior {
              clip idle {
                duration_s: 1.0
                key head.rotation { 0.0: [0, 0, 0], 0.5: [25, 0, 0], 1.0: [0, 0, 0], catmullrom }
              }
              clip sprint {
                duration_s: 0.8
                track head: wave(axis=x, amp_deg=24, freq_hz=1.25)
                key torso.position { 0.0: [0, 0, 0], 0.4: [0, 1.5, 0], 0.8: [0, 0, 0] }
                key head.scale { 0.0: [1, 1, 1], 0.4: [1.1, 1.1, 1.1], 0.8: [1, 1, 1] }
              }
              spring plume { stiffness: 35, damping: 5 }
            }
            @controller {
              initial: idle
              idle -> sprint when speed > 0.2, blend 0.15
              sprint -> idle when speed <= 0.1, blend 0.25
            }
            """
        )
        frames = evaluate_behavior_clip(prog, "sprint", num_frames=10)
        self.assertEqual(len(frames), 10)
        # Spring bone 'plume' exhibits passive follow-through excursion driven by oscillating parent 'head'
        plume_rx = [abs(fr.pose_euler_deltas.get("plume", (0.0, 0.0, 0.0))[0]) for fr in frames]
        self.assertGreater(max(plume_rx), 0.5)
        # Position & scale keyframes take effect on sampled scene
        max_chest_y = max(float(fr.scene.box_by_name["chest"].center[1]) for fr in frames)
        self.assertGreater(max_chest_y, 12.5)
        max_skull_w = max(float(fr.scene.box_by_name["skull"].size[0]) for fr in frames)
        self.assertGreater(max_skull_w, 4.1)

        assert prog.controller is not None
        st0, bl0 = evaluate_controller_transition(prog.controller, None, {"speed": 0.05})
        self.assertEqual((st0, bl0), ("idle", 0.0))
        st1, bl1 = evaluate_controller_transition(prog.controller, st0, {"speed": 0.45})
        self.assertEqual((st1, bl1), ("sprint", 0.15))
        st2, bl2 = evaluate_controller_transition(prog.controller, st1, {"speed": 0.02})
        self.assertEqual((st2, bl2), ("idle", 0.25))


if __name__ == "__main__":
    unittest.main()
