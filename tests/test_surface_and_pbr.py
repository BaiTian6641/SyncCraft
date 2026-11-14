"""Comprehensive unit tests for P1-03a/b/c stateful surfaces, PDE conservation, and multi-target PBR."""

from __future__ import annotations

import unittest
import numpy as np

from synccraft.backends import export_bbmodel, export_gltf_pbr, export_labpbr_1_3
from synccraft.dsl import (
    Attach,
    Bone,
    Box,
    DSLParseError,
    DSLValidationError,
    MaterialDecl,
    Meta,
    Palette,
    Place,
    Program,
    StyleDecl,
    UnsupportedFeature,
    format_program,
    parse_program,
    validate_program,
)
from synccraft.shading import (
    allocate_uv_layout,
    downsample_optical_fields,
    evaluate_world_optical_pipeline,
    shade_asset,
)
from synccraft.surface import (
    FIDELITY_BY_FAMILY,
    MATERIAL_FAMILIES,
    EnvironmentState,
    MaterialProgram,
    MaterialSpec,
    StampOverride,
    StyleConfig,
    WorldState,
    build_surface_domain,
    evaluate_optical_fields,
    evaluate_posed_surface_kinematics,
    initialize_surface_state,
    linear_srgb_to_oklch,
    linear_to_srgb_u8,
    manifold_divergence_operator,
    oklch_to_linear_srgb,
    srgb_u8_to_linear,
    step_pure_diffusion,
)


class TestSurfaceDomainAndPDEs(unittest.TestCase):
    def _mirrored_sample_program(self) -> Program:
        return validate_program(
            Program(
                Meta("entity", "bedrock_entity", 16),
                [
                    Palette("skin", (0.65, 0.10, 70.0), 6),
                    Palette("bronze", (0.62, 0.11, 65.0), 6),
                ],
                Bone(
                    "root",
                    pivot_local=(0.0, 8.0, 0.0),
                    items=[
                        Box("torso_lower", (4, 4, 4), "skin"),
                        Box("torso_upper", (4, 4, 4), "skin", place=Place("torso_lower", (0, 1, 0), (0, -1, 0))),
                        Box("harness", (4, 2, 2), "bronze", place=Place("torso_upper", (0, 0, -1), (0, 0, 1))),
                        Bone(
                            "left_arm",
                            attach=Attach("torso_upper", (-1, 0, 0), (1, 0, 0)),
                            mirror_name="right_arm",
                            items=[Box("arm_box", (2, 4, 2), "skin")],
                        ),
                    ],
                ),
            )
        )

    def test_three_relationships_kept_strictly_distinct(self) -> None:
        prog = self._mirrored_sample_program()
        # Without weld_group, each box is its own 6-face closed manifold component (5 boxes -> 5 components)
        dom_unwelded = build_surface_domain(prog, subdivisions_per_unit=4)
        self.assertEqual(len(dom_unwelded.connected_components), 5)
        self.assertEqual(len(dom_unwelded.symmetry_pairs), 6)

        # Mirrored arm_box and right_arm__arm_box must NEVER share a manifold seam
        comp_by_box = {}
        for idx, comp in enumerate(dom_unwelded.connected_components):
            for box_name, _ in comp:
                comp_by_box[box_name] = idx
        self.assertNotEqual(comp_by_box["arm_box"], comp_by_box["right_arm__arm_box"])
        self.assertNotEqual(comp_by_box["torso_upper"], comp_by_box["harness"])

        # With opt-in weld_group on 'skin', flush torso_lower + torso_upper weld on exterior coplanar faces,
        # while bronze 'harness' touching skin remains strictly disconnected!
        mp_welded = MaterialProgram(
            materials={
                "skin": MaterialSpec("skin", family="patterned_skin", weld_group="body_pelt"),
                "bronze": MaterialSpec("bronze", family="brushed_metal", weld_group="armor"),
            }
        )
        dom_welded = build_surface_domain(prog, mp_welded, subdivisions_per_unit=4)
        welded_comp_by_box: dict[str, set[int]] = {}
        for idx, comp in enumerate(dom_welded.connected_components):
            for box_name, _ in comp:
                welded_comp_by_box.setdefault(box_name, set()).add(idx)
        # Exterior faces of torso_lower and torso_upper share a connected component, never harness or right_arm
        self.assertTrue(welded_comp_by_box["torso_lower"] & welded_comp_by_box["torso_upper"])
        self.assertFalse(welded_comp_by_box["torso_upper"] & welded_comp_by_box["harness"])

    def test_manifold_laplacian_zero_on_constants_and_conserves_mass(self) -> None:
        prog = self._mirrored_sample_program()
        mp = MaterialProgram(
            materials={
                "skin": MaterialSpec("skin", family="patterned_skin", weld_group="body_pelt"),
                "bronze": MaterialSpec("bronze", family="brushed_metal"),
            }
        )
        dom = build_surface_domain(prog, mp, subdivisions_per_unit=4)

        # 1. Constant field -> Laplacian is zero to machine precision
        const_field = {fk: np.full(g.shape, 0.73, dtype=np.float64) for fk, g in dom.faces.items()}
        for aniso in (1.0, 2.3):
            laps_c = manifold_divergence_operator(dom, const_field, anisotropy=aniso)
            peak_err = max(float(np.max(np.abs(v))) for v in laps_c.values())
            self.assertLess(peak_err, 1e-13)

        # 2. Random discontinuous field -> integral of divergence across all faces is zero (conservative flux)
        rng = np.random.default_rng(99)
        rand_field = {fk: rng.uniform(0.0, 1.0, size=g.shape) for fk, g in dom.faces.items()}
        for aniso in (1.0, 1.85):
            laps_r = manifold_divergence_operator(dom, rand_field, grain_axis_bone=np.array((0.3, 0.9, 0.2)), anisotropy=aniso)
            total_div = sum(float(v.sum()) for v in laps_r.values())
            self.assertAlmostEqual(total_div, 0.0, places=11)

        # 3. Pure diffusion conserves total mass and never leaks across mirror_x limbs
        init_mass_field = {fk: np.zeros(g.shape, dtype=np.float64) for fk, g in dom.faces.items()}
        init_mass_field[("arm_box", "north")][:] = 1.0
        before_mass = sum(float(v.sum()) for v in init_mass_field.values()) * (dom.dx ** 2)

        diffused = step_pure_diffusion(dom, init_mass_field, diffusivity=0.12, total_time=2.5)
        after_mass = sum(float(v.sum()) for v in diffused.values()) * (dom.dx ** 2)
        self.assertAlmostEqual(before_mass, after_mass, places=11)
        # Wrapped around 90-deg edges of left arm_box onto east/west/up/down, but ZERO on right_arm__arm_box or bronze harness!
        self.assertGreater(float(diffused[("arm_box", "west")].max()), 0.01)
        self.assertEqual(float(diffused[("right_arm__arm_box", "north")].max()), 0.0)
        self.assertEqual(float(diffused[("harness", "north")].max()), 0.0)


class TestMaterialCatalogAndPBRTargets(unittest.TestCase):
    def test_all_ten_families_and_color_roundtrips(self) -> None:
        # OKLCH <-> Linear sRGB roundtrip for strictly in-gamut colors
        lch_orig = np.array([[0.58, 0.08, 45.0], [0.55, 0.06, 225.0]], dtype=np.float64)
        lin = oklch_to_linear_srgb(lch_orig)
        self.assertGreater(float(lin.min()), 0.0)
        self.assertLess(float(lin.max()), 1.0)
        lch_rec = linear_srgb_to_oklch(lin)
        np.testing.assert_allclose(lch_orig, lch_rec, atol=1e-4)

        u8 = linear_to_srgb_u8(lin)
        lin_rec = srgb_u8_to_linear(u8)
        np.testing.assert_allclose(lin, lin_rec, atol=0.01)

        prog = parse_program(
            "@meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }\n"
            "@palette { mat: ramp(base_lch=[0.6, 0.08, 210], steps=6, emissive=false) }\n"
            "@rig { root bone r { pivot_local: [0,0,0] box b { size: [4,4,4], material: mat } } }\n"
        )
        self.assertEqual(len(MATERIAL_FAMILIES), 10)
        for fam in sorted(MATERIAL_FAMILIES):
            spec = MaterialSpec("mat", family=fam, coating_thickness=0.5)
            self.assertIn(spec.fidelity_label, FIDELITY_BY_FAMILY.values())
            mp = MaterialProgram({"mat": spec}, stamps=(StampOverride("b", "north", 1.0, 1.0, 2.0, 2.0, (0.85, 0.12, 80.0), emissive=0.7),))
            dom = build_surface_domain(prog, mp, subdivisions_per_unit=4)
            st = initialize_surface_state(dom)
            _, kin = evaluate_posed_surface_kinematics(dom, prog)
            opt = evaluate_optical_fields(prog, dom, mp, st, kin)
            p = opt.patches[("b", "north")]
            self.assertEqual(p.base_color_linear.shape, (16, 16, 3))
            norms = np.linalg.norm(p.tangent_normal, axis=-1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)

            # Export across all 3 targets
            ws = WorldState(prog, mp, st)
            van_desc, van_png = shade_asset(ws)
            bb = export_bbmodel(dom.rest_scene, atlas=van_desc)
            self.assertEqual(bb["meta"]["format_version"], "5.0")
            self.assertGreater(len(van_png), 100)

            lab = export_labpbr_1_3(ws)
            self.assertGreaterEqual(lab["channel_audit"]["normal_alpha_min"], 1)
            self.assertLessEqual(lab["channel_audit"]["normal_alpha_max"], 255)
            # Emissive stamp has alpha <= 254, non-emissive south face has alpha == 255
            self.assertGreater(lab["channel_audit"]["specular_alpha_emissive_pixels"], 0)
            self.assertGreater(lab["channel_audit"]["specular_alpha_non_emissive_pixels"], 0)
            self.assertEqual(lab["verification_status"], "serialized_and_channel_audited_not_runtime_shader_verified")

            gltf = export_gltf_pbr(ws)
            self.assertEqual(gltf["gltf"]["asset"]["version"], "2.0")
            # glTF normal B encodes +Z outward (> 200 on flat regions), distinct from LabPBR B (cavity AO)
            self.assertGreater(float(gltf["arrays"]["normal_gl"][..., 2].mean()), 220.0)
            if fam == "thin_film_iridescence":
                self.assertIn("KHR_materials_iridescence", gltf["gltf"]["extensionsUsed"])

    def test_area_downsampler_conserves_integral_and_rejects_fractional_density(self) -> None:
        prog = parse_program(
            "@meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }\n"
            "@palette { mat: ramp(base_lch=[0.55, 0.09, 60], steps=6) }\n"
            "@rig { root bone r { pivot_local: [0,0,0] box b { size: [4,4,4], material: mat } } }\n"
        )
        mp = MaterialProgram({"mat": MaterialSpec("mat", family="solid_wood", seed=77)})
        ws = WorldState(prog, mp, initialize_surface_state(build_surface_domain(prog, mp, subdivisions_per_unit=8)))
        dom, _, _, sol_opt, _ = evaluate_world_optical_pipeline(ws, subdivisions_per_unit=8)
        sol_mean = float(sol_opt.patches[("b", "north")].base_color_linear.mean())

        for td in (8, 16, 32, 64):
            ds = downsample_optical_fields(sol_opt, dom, td)
            px_mean = float(ds.patches[("b", "north")].base_color_linear.mean())
            self.assertAlmostEqual(sol_mean, px_mean, places=12)

        with self.assertRaises(ValueError):
            downsample_optical_fields(sol_opt, dom, 10)  # 4 * 10/16 = 2.5 fractional texels

    def test_optional_materials_and_style_dsl_roundtrip_and_validation(self) -> None:
        dsl_with_mat = """@meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
@palette {
  coat: ramp(base_lch=[0.5, 0.12, 240.0], steps=6, hue_shift=[-8.0, 6.0], emissive=false)
}
@rig {
  root bone hilt {
    pivot_local: [0.0, 0.0, 0.0]
    box core { size: [2, 6, 2], material: coat, inflate: 0.0 }
  }
}
@materials {
  coat: process(family=painted_metal, roughness=0.3, metalness=0.0, coating=0.6, ior=1.5, scale=1.2, seed=42)
}
@style { target_encoding: labpbr_1_3, quantize_ramp: false, bevel_highlight: 0.05, view_deg: 22.0 }
"""
        parsed = parse_program(dsl_with_mat)
        self.assertEqual(len(parsed.materials), 1)
        self.assertEqual(parsed.materials[0].family, "painted_metal")
        self.assertIsNotNone(parsed.style)
        assert parsed.style is not None
        self.assertEqual(parsed.style.target_encoding, "labpbr_1_3")

        canon = format_program(parsed)
        self.assertEqual(parsed, parse_program(canon))

        # Invalid family or undeclared palette rejected
        with self.assertRaises(DSLValidationError):
            parse_program(dsl_with_mat.replace("painted_metal", "magic_plasma"))
        with self.assertRaises(DSLValidationError):
            parse_program(dsl_with_mat.replace("coat: process", "ghost: process"))
        # Deferred sections still fail explicitly
        for deferred in ("shade", "motion", "ergonomics"):
            with self.assertRaises(UnsupportedFeature):
                parse_program(dsl_with_mat + f"\n@{deferred} {{ }}\n")


if __name__ == "__main__":
    unittest.main()
