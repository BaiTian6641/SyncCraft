"""Unit & Integration Tests for TexGraph v11 Per-UV / Per-Body-Part Pixel Composer Region Nodes."""
from __future__ import annotations

import numpy as np

from synccraft.texgraph import (
    DEFAULT_CHRONO_VARIANTS,
    TexGraph,
    apply_variant_to_graph,
    bake_bbmodel_maps,
    build_sample_monolith_bbmodel,
    build_uv_region_overlay,
    export_node_schema_json,
    parse_texdsl,
    region_base_fill,
    region_bevel_and_ao,
    region_composite,
    region_restrained_noise,
    uv_island_coords,
)
from synccraft.texgraph.studio_server import StudioSession


def test_schema_v4_149_nodes_100_percent_coverage() -> None:
    schema = export_node_schema_json()
    assert schema["node_count"] >= 149
    assert schema["param_schema_coverage_pct"] == 100.0
    assert "Region & Per-UV Pixel Composer" in schema["categories"]
    region_ops = {n["op"] for n in schema["nodes_by_category"]["Region & Per-UV Pixel Composer"]}
    assert region_ops == {
        "uv_island_coords",
        "region_base_fill",
        "region_restrained_noise",
        "region_bevel_and_ao",
        "region_composite",
    }


def test_mesh_baker_island_metrics_and_inner_area() -> None:
    bb = build_sample_monolith_bbmodel()
    mc = bake_bbmodel_maps(bb)
    assert mc.island_edge_px.shape == (mc.height, mc.width)
    assert mc.island_size_min.shape == (mc.height, mc.width)
    assert mc.island_size_wh.shape == (mc.height, mc.width, 2)
    assert mc.island_inner_area.shape == (mc.height, mc.width)
    assert mc.face_is_up.shape == (mc.height, mc.width)
    assert mc.face_is_side.shape == (mc.height, mc.width)

    # Every outermost border pixel (edge_px == 0) must have inner_area == 0.0!
    border_mask = mc.island_mask & (mc.island_edge_px < 0.5)
    assert np.any(border_mask)
    assert float(mc.island_inner_area[border_mask].max()) == 0.0

    # Large islands (>=4px) must have interior pixels with inner_area == 1.0!
    interior_large = mc.island_mask & (mc.island_edge_px >= 1.0) & (mc.island_size_min >= 4.0)
    assert np.any(interior_large)
    assert float(mc.island_inner_area[interior_large].min()) == 1.0

    # Color-coded UV region overlay renders valid RGBA uint8
    ov = build_uv_region_overlay(mc)
    assert ov.shape == (mc.height, mc.width, 4)
    assert ov.dtype == np.uint8


def test_per_part_isolation_inner_area_restraint_and_border_trim() -> None:
    bb = build_sample_monolith_bbmodel()
    mc = bake_bbmodel_maps(bb)
    H, W = mc.height, mc.width

    # Pick first element name from monolith to verify strict per-part masking
    target_name = mc.elem_names[0]
    m_target = uv_island_coords(H, W, mc, channel="part_mask", part_pattern=target_name)
    m_other = uv_island_coords(H, W, mc, channel="part_mask", part_pattern=f"!{target_name}")
    assert np.any(m_target > 0.5)
    assert np.any(m_other > 0.5)
    assert float((m_target * m_other).max()) == 0.0

    # Base fill with border-exact bottom trim
    base = region_base_fill(
        H, W, mc,
        stops=["#20223A", "#7A82B0", "#EEF2FF"],
        trim_edge="bottom",
        trim_color="#ECC560",
        trim_width_px=2.0,
        trim_faces="side",
    )
    assert base["albedo"].shape == (H, W, 4)
    assert np.any(base["trim_mask"] > 0.5)
    # Trim must NEVER bleed onto `up` faces when trim_faces="side"!
    assert float(base["trim_mask"][mc.face_is_up > 0.5].max()) == 0.0

    # Restrained noise with edge_restraint=1.0 MUST leave 1px border pixels 100% untouched!
    noisy = region_restrained_noise(
        base, mc,
        pattern="clump_scales",
        amplitude=0.18,
        edge_restraint=1.0,
        min_island_px=4.0,
        protect_trim=True,
        seed=99,
    )
    border_px = mc.island_mask & (mc.island_edge_px < 0.5)
    diff_border = np.abs(noisy["albedo"][border_px, :3] - base["albedo"][border_px, :3]).max()
    assert float(diff_border) < 1e-5

    # Interior of >=4px non-trim islands MUST receive bounded restrained noise breakup
    interior_px = mc.island_mask & (mc.island_inner_area > 0.5) & (base["trim_mask"] < 0.5)
    assert np.any(interior_px)
    diff_interior = np.abs(noisy["albedo"][interior_px, :3] - base["albedo"][interior_px, :3]).max()
    assert 0.005 < float(diff_interior) <= 0.25

    # Bevel & hue-shifted AO preserves alpha strictly
    beveled = region_bevel_and_ao(noisy, mc, highlight_strength=0.18, shadow_strength=0.22, ao_strength=0.28)
    assert np.allclose(beveled["albedo"][..., 3], base["albedo"][..., 3])

    # Per-part region layer composited onto base leaves non-target parts untouched (outside 1px seam)!
    part_layer = region_base_fill(H, W, mc, part_pattern=target_name, stops=["#3A1414", "#B84030", "#FFD490"])
    comp = region_composite(
        beveled, part_layer, mc,
        seam_ao_strength=0.30,
        palette_snap_stops=["#3A1414", "#B84030", "#FFD490"],
        palette_snap_mix=0.45,
        preserve_alpha=True,
    )
    assert np.allclose(comp["albedo"][..., 3], beveled["albedo"][..., 3])
    far_other = (m_other > 0.5) & (comp["seam_mask"] < 0.01)
    assert np.allclose(comp["albedo"][far_other, :3], beveled["albedo"][far_other, :3])


def test_texdsl_per_region_pipeline_and_variant_morph() -> None:
    bb = build_sample_monolith_bbmodel()
    mc = bake_bbmodel_maps(bb)
    target_name = mc.elem_names[0]
    src = f"""
    @texture {{ name: "region_test" size: [{mc.width}, {mc.height}] domain: "entity" seed: 77 }}
    @palette {{
      scales: ["#363452", "#888AB4", "#EAEEF9", "#FAFBFF"]
      brass:  ["#462614", "#CA9035", "#FFF2B2"]
    }}
    @pass main "Per-Region Pixel Composer" {{
      scale_base = region_base_fill(stops=@scales, top_boost=0.16, side_grad=0.18)
      scale_tex  = region_restrained_noise(scale_base, pattern="clump_scales", amplitude=0.12, edge_restraint=0.9)
      scale_mat  = region_bevel_and_ao(scale_tex, highlight_strength=0.15, shadow_strength=0.20)
      brass_base = region_base_fill(part_pattern="{target_name}", stops=@brass, metallic_base=0.92, roughness_base=0.18)
      brass_mat  = region_restrained_noise(brass_base, pattern="brushed_metal", amplitude=0.12)
      merged     = region_composite(scale_mat, brass_mat, seam_ao_strength=0.32, palette_snap_stops=@brass, palette_snap_mix=0.35)
    }}
    @output material(merged, normal_strength=2.2)
    """
    g = parse_texdsl(src, mesh_context=mc)
    val = g.validate_graph()
    assert val.ok, val.errors
    ev = g.evaluate(build_pbr=True)
    assert ev.output.shape == (mc.height, mc.width, 4)
    assert ev.pbr_maps is not None

    # Verify Chronal Variant morphing seamlessly updates `region_base_fill` stops!
    g_solar = apply_variant_to_graph(g, DEFAULT_CHRONO_VARIANTS["solar_aeon"])
    ev_solar = g_solar.evaluate(build_pbr=True)
    diff = float(np.abs(ev_solar.output[..., :3] - ev.output[..., :3]).mean())
    assert diff > 0.04


def test_studio_session_exposes_uv_regions_and_inner_area() -> None:
    sess = StudioSession(initial_sample="monolith_v10_3d", size=64)
    ev = sess.evaluate({"include_thumbs": False, "include_qa": True})
    assert ev["ok"] is True
    assert "uv_regions" in ev["channels"]
    assert "inner_area" in ev["channels"]
    assert ev["channels"]["uv_regions"].startswith("data:image/png;base64,")
    assert ev["channels"]["inner_area"].startswith("data:image/png;base64,")
