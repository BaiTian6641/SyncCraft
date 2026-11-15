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


class TestScientificOpticalAndSurfaceValidation(unittest.TestCase):
    """Reference physics & numerical validation suite for M5 and M6."""

    def test_m5_three_tier_fidelity_labels_and_cited_physical_constants(self) -> None:
        from synccraft.surface import (
            ARCHARD_DIRECT_RUBBING_COUPLING,
            ARCHARD_SWEPT_CONTACT_COUPLING,
            ARCHARD_WEAR_NORMALIZATION,
            DIELECTRIC_IOR_REFERENCE_TABLE,
            FIDELITY_BY_FAMILY,
            FIDELITY_TIERS,
            MATERIAL_FAMILIES,
            MaterialSpec,
            StyleConfig,
            ior_to_dielectric_f0,
        )

        self.assertEqual(
            FIDELITY_TIERS,
            frozenset({"physical_law_validated", "reduced_order_qualitative", "stylistic"}),
        )
        self.assertEqual(set(FIDELITY_BY_FAMILY.keys()), set(MATERIAL_FAMILIES))
        self.assertTrue(set(FIDELITY_BY_FAMILY.values()).issubset(FIDELITY_TIERS))
        for fam in MATERIAL_FAMILIES:
            spec = MaterialSpec(name=f"m_{fam}", family=fam)
            self.assertIn(spec.fidelity_label, FIDELITY_TIERS)
        self.assertEqual(MaterialSpec("irid", family="thin_film_iridescence").fidelity_label, "physical_law_validated")
        self.assertEqual(MaterialSpec("poly", family="polymer_dielectric").fidelity_label, "physical_law_validated")
        self.assertEqual(MaterialSpec("skin", family="patterned_skin").fidelity_label, "reduced_order_qualitative")
        self.assertEqual(StyleConfig().fidelity_label, "stylistic")
        self.assertEqual(StyleConfig().to_dict()["fidelity_label"], "stylistic")

        # Every entry in DIELECTRIC_IOR_REFERENCE_TABLE obeys Fresnel F0 = ((n - 1)/(n + 1))^2
        self.assertGreaterEqual(len(DIELECTRIC_IOR_REFERENCE_TABLE), 5)
        for key, entry in DIELECTRIC_IOR_REFERENCE_TABLE.items():
            ior = float(entry["ior"])
            f0 = float(entry["f0"])
            self.assertGreater(ior, 1.0, msg=key)
            self.assertAlmostEqual(ior_to_dielectric_f0(ior), f0, places=6, msg=key)
            self.assertTrue(str(entry["source"]).strip(), msg=key)

        # Archard (1953) dimensional wear normalization reproduces 0.45 and 0.22 couplings
        c_rub_calc = (
            float(ARCHARD_WEAR_NORMALIZATION["reference_specific_wear_rate_inv_pa"])
            * float(ARCHARD_WEAR_NORMALIZATION["reference_contact_pressure_pa"])
            * float(ARCHARD_WEAR_NORMALIZATION["model_unit_meters"])
            / float(ARCHARD_WEAR_NORMALIZATION["reference_coating_thickness_m"])
        )
        self.assertAlmostEqual(c_rub_calc, ARCHARD_DIRECT_RUBBING_COUPLING, places=6)
        self.assertAlmostEqual(ARCHARD_DIRECT_RUBBING_COUPLING, 0.45, places=6)
        self.assertAlmostEqual(ARCHARD_SWEPT_CONTACT_COUPLING, 0.22, places=6)
        self.assertIn("Archard", str(ARCHARD_WEAR_NORMALIZATION["source"]))

    def test_m6_cook_torrance_ggx_white_furnace_and_helmholtz_reciprocity(self) -> None:
        from synccraft.shading import (
            compute_ggx_hemispherical_reflectance,
            evaluate_cook_torrance_ggx_brdf,
        )

        # 1. White-furnace passivity: hemispherical reflectance <= 1.0 across all roughnesses & view angles
        roughnesses = (0.02, 0.05, 0.15, 0.35, 0.65, 1.0)
        mu_views = (0.05, 0.20, 0.50, 0.80, 1.00)
        for rp in roughnesses:
            for mu_o in mu_views:
                rho_cond = compute_ggx_hemispherical_reflectance(
                    mu_o, rp, metalness=1.0, base_color_linear=(1.0, 1.0, 1.0), n_quad=64
                )
                self.assertTrue(np.all(rho_cond > 0.0), msg=f"conductor rp={rp}, mu_o={mu_o}")
                self.assertTrue(
                    np.all(rho_cond <= 1.0 + 1e-7),
                    msg=f"White-furnace violation on conductor rp={rp}, mu_o={mu_o}: {rho_cond}",
                )
                rho_diel = compute_ggx_hemispherical_reflectance(
                    mu_o, rp, metalness=0.0, base_color_linear=(1.0, 1.0, 1.0), dielectric_f0=0.04, n_quad=64
                )
                self.assertTrue(
                    np.all(rho_diel <= 1.0 + 1e-7),
                    msg=f"White-furnace violation on dielectric rp={rp}, mu_o={mu_o}: {rho_diel}",
                )

        # Smooth mirror-like white conductor approaches unit reflectance at non-grazing incidence
        rho_mirror = compute_ggx_hemispherical_reflectance(
            0.85, 0.02, metalness=1.0, base_color_linear=(1.0, 1.0, 1.0), n_quad=64
        )
        np.testing.assert_allclose(rho_mirror, np.ones(3), atol=1e-3)

        # 2. Exact Helmholtz reciprocity: f_r(wi, wo) == f_r(wo, wi)
        rng = np.random.default_rng(20261114)
        n_samples = 200
        wi = rng.normal(size=(n_samples, 3))
        wo = rng.normal(size=(n_samples, 3))
        wi[:, 2] = np.abs(wi[:, 2]) + 0.02
        wo[:, 2] = np.abs(wo[:, 2]) + 0.02
        wi /= np.linalg.norm(wi, axis=-1, keepdims=True)
        wo /= np.linalg.norm(wo, axis=-1, keepdims=True)
        nrm = np.broadcast_to(np.array((0.0, 0.0, 1.0)), wi.shape)

        for cfg in (
            {"perceptual_roughness": 0.25, "metalness": 0.0, "base_color_linear": (0.8, 0.4, 0.2), "dielectric_f0": 0.04},
            {"perceptual_roughness": 0.12, "metalness": 1.0, "base_color_linear": (0.95, 0.78, 0.35), "dielectric_f0": 0.04},
            {
                "perceptual_roughness": 0.30,
                "metalness": 0.35,
                "base_color_linear": (0.3, 0.6, 0.9),
                "dielectric_f0": 0.0465,
                "iridescence_thickness_nm": 380.0,
                "iridescence_ior": 1.45,
            },
        ):
            f_io = evaluate_cook_torrance_ggx_brdf(wi, wo, nrm, **cfg)
            f_oi = evaluate_cook_torrance_ggx_brdf(wo, wi, nrm, **cfg)
            np.testing.assert_allclose(f_io, f_oi, rtol=0.0, atol=1e-13)

    def test_m6_airy_single_layer_thin_film_analytic_path_difference_extrema(self) -> None:
        import math
        from synccraft.surface import evaluate_airy_thin_film_linear_rgb

        wavelengths = np.array((650.0, 532.0, 460.0), dtype=np.float64)
        n_film = 1.46
        r01_0 = ((1.0 - n_film) / (1.0 + n_film)) ** 2  # normal-incidence air->film R01

        for theta_deg in (0.0, 45.0):
            cos_i = math.cos(math.radians(theta_deg))
            sin2_i = max(0.0, 1.0 - cos_i * cos_i)
            cos_t = math.sqrt(1.0 - sin2_i / (n_film * n_film))
            R01 = r01_0 + (1.0 - r01_0) * ((1.0 - cos_i) ** 5)
            r1 = math.sqrt(R01)
            sub_rgb = np.array((0.25, 0.16, 0.36), dtype=np.float64)
            r2 = np.sqrt(sub_rgb)

            r_max_analytic = ((r1 + r2) / (1.0 + r1 * r2)) ** 2
            r_min_analytic = ((r1 - r2) / (1.0 - r1 * r2)) ** 2

            for ch_idx, lam in enumerate(wavelengths):
                # Quarter-wave optical thickness 2 n_film d cos(theta_t) = (2m + 1) * lam / 2
                # => Δ = 2 n_film d cos(theta_t) + lam/2 = (m + 1) * lam => constructive maximum!
                for m_order in (0, 1):
                    d_qw = ((2 * m_order + 1) * lam) / (4.0 * n_film * cos_t)
                    rgb_qw = evaluate_airy_thin_film_linear_rgb(np.array(d_qw), n_film, cos_i, sub_rgb)
                    self.assertAlmostEqual(float(rgb_qw[ch_idx]), float(r_max_analytic[ch_idx]), places=12)

                # Half-wave optical thickness 2 n_film d cos(theta_t) = m * lam
                # => Δ = (m + 0.5) * lam => destructive minimum!
                for m_order in (0, 1, 2):
                    d_hw = (m_order * lam) / (2.0 * n_film * cos_t)
                    rgb_hw = evaluate_airy_thin_film_linear_rgb(np.array(d_hw), n_film, cos_i, sub_rgb)
                    self.assertAlmostEqual(float(rgb_hw[ch_idx]), float(r_min_analytic[ch_idx]), places=12)

            # Matched-amplitude anti-reflection check (r2 == r1 at half-wave gives exact 0 reflectance)
            ar_sub = np.full(3, R01, dtype=np.float64)
            d_ar = wavelengths[1] / (2.0 * n_film * cos_t)  # half-wave for 532 nm
            rgb_ar = evaluate_airy_thin_film_linear_rgb(np.array(d_ar), n_film, cos_i, ar_sub)
            self.assertAlmostEqual(float(rgb_ar[1]), 0.0, places=12)

    def test_m6_gray_scott_pearson_1993_pattern_map_regimes(self) -> None:
        from synccraft.surface import (
            FacePersistentPatch,
            MaterialSpec,
            PersistentSurfaceState,
            build_surface_domain,
            classify_gray_scott_regime,
            integrate_gray_scott_on_state,
        )

        prog = parse_program(
            "@meta { archetype: entity, target_profile: bbmodel_5, texel_density: 16 }\n"
            "@palette { skin: ramp(base_lch=[0.68, 0.07, 115], steps=6) }\n"
            "@rig { root bone body { pivot_local: [0,0,0] box slab { size: [12, 12, 12], material: skin } } }\n"
        )
        dom = build_surface_domain(prog, subdivisions_per_unit=4)
        # Pearson (1993) 2:1 diffusivity ratio scaled to manifold grid spacing dx = 0.25 u:
        du_p = 0.16 * (dom.dx ** 2)
        dv_p = 0.08 * (dom.dx ** 2)

        def eval_regime(feed: float, kill: float) -> dict[str, float | int | str]:
            spec = MaterialSpec(
                name="skin",
                family="patterned_skin",
                rd_feed=feed,
                rd_kill=kill,
                rd_du=du_p,
                rd_dv=dv_p,
                rd_initial_time=800.0,
                anisotropy=1.0,
                seed=42,
            )
            st = PersistentSurfaceState(
                dom.subdivisions_per_unit,
                {fk: FacePersistentPatch.zeros(g.shape) for fk, g in dom.faces.items()},
            )
            integrate_gray_scott_on_state(dom, st, spec, {("slab", "up")}, duration=800.0, dorsal_modulation=0.0)
            return classify_gray_scott_regime(st.patches[("slab", "up")].rd_v, threshold=0.16)

        hom_low = eval_regime(0.012, 0.065)
        hom_high = eval_regime(0.095, 0.060)
        spots = eval_regime(0.030, 0.062)
        stripes = eval_regime(0.054, 0.062)

        self.assertEqual(hom_low["regime"], "homogeneous")
        self.assertLess(float(hom_low["v_std"]), 1e-4)
        self.assertEqual(hom_high["regime"], "homogeneous")
        self.assertLess(float(hom_high["v_std"]), 1e-4)

        self.assertEqual(spots["regime"], "spots")
        self.assertGreaterEqual(int(spots["n_components"]), 3)

        self.assertEqual(stripes["regime"], "stripes_labyrinth")
        self.assertGreater(float(stripes["active_fraction"]), float(spots["active_fraction"]))
        self.assertGreater(float(stripes["max_component_fraction"]), float(spots["max_component_fraction"]))

    def test_m6_16px_texel_survival_sensitivity_report(self) -> None:
        from synccraft.backends.labpbr_exporter import export_labpbr_1_3
        from synccraft.shading import compile_vanilla_atlas
        from synccraft.surface import (
            MaterialProgram,
            MaterialSpec,
            StyleConfig,
            WorldState,
            build_surface_domain,
            initialize_surface_state,
        )

        prog = parse_program(
            "@meta { archetype: entity, target_profile: bbmodel_5, texel_density: 16 }\n"
            "@palette { body_mat: ramp(base_lch=[0.58, 0.11, 225], steps=5) }\n"
            "@rig { root bone root_b { pivot_local: [0,0,0] box hull { size: [6, 6, 6], material: body_mat } } }\n"
        )

        def render_16px(spec: MaterialSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            import io
            from PIL import Image

            mp = MaterialProgram({"body_mat": spec})
            dom = build_surface_domain(prog, mp, subdivisions_per_unit=4)
            st = initialize_surface_state(dom)
            # Apply mild deterministic wear on north face so coating_thickness sensitivity is observable
            st.patches[("hull", "north")].wear[:, :12] = 0.30
            ws = WorldState(
                program=prog,
                material_program=mp,
                surface_state=st,
                style_config=StyleConfig(quantize_to_palette_ramp=True),
            )
            _, vanilla_png = compile_vanilla_atlas(ws)
            vanilla = np.asarray(Image.open(io.BytesIO(vanilla_png)).convert("RGBA"), dtype=np.uint8)
            labpbr = export_labpbr_1_3(ws)
            return vanilla, labpbr["arrays"]["atlas_s"], labpbr["arrays"]["atlas_n"]

        base_metal = MaterialSpec("body_mat", family="painted_metal", coating_thickness=0.70, perceptual_roughness=0.45, dielectric_ior=1.50, feature_scale=1.0, seed=101)
        v_base, s_base, n_base = render_16px(base_metal)

        probes: list[tuple[str, MaterialSpec, tuple[np.ndarray, np.ndarray, np.ndarray]]] = [
            (
                "coating_thickness_macro",
                MaterialSpec("body_mat", family="painted_metal", coating_thickness=0.18, perceptual_roughness=0.45, dielectric_ior=1.50, feature_scale=1.0, seed=101),
                (v_base, s_base, n_base),
            ),
            (
                "perceptual_roughness_macro",
                MaterialSpec("body_mat", family="painted_metal", coating_thickness=0.70, perceptual_roughness=0.06, dielectric_ior=1.50, feature_scale=1.0, seed=101),
                (v_base, s_base, n_base),
            ),
            (
                "dielectric_ior_macro",
                MaterialSpec("body_mat", family="painted_metal", coating_thickness=0.70, perceptual_roughness=0.45, dielectric_ior=2.10, feature_scale=1.0, seed=101),
                (v_base, s_base, n_base),
            ),
            (
                "feature_scale_macro",
                MaterialSpec("body_mat", family="painted_metal", coating_thickness=0.18, perceptual_roughness=0.45, dielectric_ior=1.50, feature_scale=2.5, seed=101),
                render_16px(MaterialSpec("body_mat", family="painted_metal", coating_thickness=0.18, perceptual_roughness=0.45, dielectric_ior=1.50, feature_scale=1.0, seed=101)),
            ),
            (
                "thin_film_thickness_nm_macro",
                MaterialSpec("body_mat", family="thin_film_iridescence", film_thickness_nm=460.0, perceptual_roughness=0.18, seed=101),
                render_16px(MaterialSpec("body_mat", family="thin_film_iridescence", film_thickness_nm=240.0, perceptual_roughness=0.18, seed=101)),
            ),
            (
                "countershade_strength_macro",
                MaterialSpec("body_mat", family="keratin_scales", countershade_strength=0.85, seed=101),
                render_16px(MaterialSpec("body_mat", family="keratin_scales", countershade_strength=0.0, seed=101)),
            ),
            (
                "sub_lsb_roughness_micro",
                MaterialSpec("body_mat", family="painted_metal", coating_thickness=0.70, perceptual_roughness=0.4501, dielectric_ior=1.50, feature_scale=1.0, seed=101),
                (v_base, s_base, n_base),
            ),
        ]

        report: dict[str, dict[str, int | bool]] = {}
        for pname, p_spec, (ref_v, ref_s, ref_n) in probes:
            cur_v, cur_s, cur_n = render_16px(p_spec)
            dv = int(np.count_nonzero(np.any(cur_v != ref_v, axis=-1)))
            ds = int(np.count_nonzero(np.any(cur_s != ref_s, axis=-1)))
            dn = int(np.count_nonzero(np.any(cur_n != ref_n, axis=-1)))
            report[pname] = {
                "vanilla_quantized_changed_texels": dv,
                "labpbr_s_changed_texels": ds,
                "labpbr_n_changed_texels": dn,
                "survives_vanilla_16px": dv > 0,
                "survives_any_16px_target": (dv + ds + dn) > 0,
            }

        # Macro physical parameters survive to 16-px texels (vanilla and/or LabPBR _s/_n)
        for macro_key in (
            "coating_thickness_macro",
            "perceptual_roughness_macro",
            "dielectric_ior_macro",
            "feature_scale_macro",
            "thin_film_thickness_nm_macro",
            "countershade_strength_macro",
        ):
            self.assertTrue(report[macro_key]["survives_any_16px_target"], msg=f"{macro_key}: {report[macro_key]}")

        # Coating wear, countershading, and thin-film shift survive even 5-step quantized vanilla 16-px atlas
        self.assertTrue(report["coating_thickness_macro"]["survives_vanilla_16px"])
        self.assertTrue(report["countershade_strength_macro"]["survives_vanilla_16px"])
        self.assertTrue(report["thin_film_thickness_nm_macro"]["survives_vanilla_16px"])
        # Honest quantization floor: 1e-4 roughness tweak collapses under 5-step OKLab ramp quantization
        self.assertFalse(report["sub_lsb_roughness_micro"]["survives_vanilla_16px"])


if __name__ == "__main__":
    unittest.main()
