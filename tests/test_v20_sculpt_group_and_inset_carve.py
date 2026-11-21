"""Verification tests for v20 sculpt_group & inset_carve compound cuboid mass-packing."""
from __future__ import annotations

from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl import format_program, parse_program
from synccraft.kinematics.se3_chain import compile_geometry

META_AND_PALETTES = """
@meta {
  archetype: entity,
  target_profile: bedrock_entity,
  texel_density: 16,
  units_per_block: 16,
  dsl_version: "4.1",
  size_step: 0.5,
  audit_profile: sculpted
}
@palette {
  silver: ramp(base_lch=[0.82, 0.04, 240], steps=5, hue_shift=[20, -15])
  gold:   ramp(base_lch=[0.76, 0.14, 82],  steps=5, hue_shift=[18, -12])
}
"""


def test_inset_carve_and_sibling_sculpt_group_pass_and_roundtrip() -> None:
    dsl = META_AND_PALETTES + """
@rig {
  root bone chest {
    pivot_local: [0, 12, 0]
    sculpt_group: thorax_pack
    box chest_core { size: [14, 12, 16], material: silver }
    box chest_stepped_chamfer inset_carve(base_box=chest_core, anchor=[0, 1, 0], protrude=1.5) {
      size: [11, 8, 13], material: gold
    }
    bone deltoid_l attach(parent_box=chest_core, anchor=[-1, 0.4, -0.4], align=[0, 0, 0]) mirror_x(deltoid_r) {
      rest_euler: [12, -15, -25]
      sculpt_group: shoulder_pack
      box deltoid_mass { size: [6, 8, 9], material: silver, sculpt_group: thorax_pack }
      box scute_ridge inset_carve(base_box=deltoid_mass, anchor=[-1, 0.2, 0], protrude=1.5) {
        size: [5, 4, 7], material: gold, rotate: [-35, 0, -12], sculpt_group: shoulder_pack
      }
    }
    bone trapezius_l attach(parent_box=chest_core, anchor=[-0.8, 0.8, -0.3], align=[0, 0, 0]) mirror_x(trapezius_r) {
      rest_euler: [18, -10, -32]
      sculpt_group: [shoulder_pack, thorax_pack]
      box trap_wedge { size: [6, 6, 8], material: gold }
    }
  }
}
"""
    prog = parse_program(dsl)
    rt = format_program(prog)
    assert "inset_carve(" in rt
    assert "sculpt_group:" in rt
    prog2 = parse_program(rt)
    scene = compile_geometry(prog2)
    aud = audit_scene(scene, audit_profile="sculpted")
    assert aud["passed"], f"Expected sculpted compound mass-packing to pass, got: {aud}"
    reasons = {p["exemption_reason"] for p in aud["sculpted_overlap_pairs"]}
    assert "sculpted_anatomical_cluster" in reasons or "declared_inset_carve" in reasons


def test_completely_swallowed_box_and_cross_midline_still_rejected() -> None:
    swallowed_dsl = META_AND_PALETTES + """
@rig {
  root bone chest {
    pivot_local: [0, 10, 0]
    sculpt_group: thorax_pack
    box outer_hull { size: [12, 12, 12], material: silver }
    box dead_hidden place(base_box=outer_hull, anchor=[0, 0, 0], align=[0, 0, 0]) {
      size: [4, 4, 4], material: silver
    }
  }
}
"""
    aud_sw = audit_scene(compile_geometry(parse_program(swallowed_dsl)), audit_profile="sculpted")
    assert not aud_sw["passed"]
    assert any(v["violation"] == "swallowed_redundant_box" for v in aud_sw["contact_violations"])

    cross_dsl = META_AND_PALETTES + """
@rig {
  root bone pelvis {
    pivot_local: [0, 10, 0]
    box core { size: [8, 6, 8], material: silver }
    bone wing_l attach(parent_box=core, anchor=[-0.2, 1, 0], align=[0, -1, 0]) mirror_x(wing_r) {
      sculpt_group: wing_blade
      box spar { size: [8, 8, 6], material: silver }
    }
  }
}
"""
    aud_cr = audit_scene(compile_geometry(parse_program(cross_dsl)), audit_profile="sculpted")
    assert not aud_cr["passed"]
    assert any(c["pair"] == ["spar", "wing_r__spar"] for c in aud_cr["collision_pairs"])


def test_per_element_uv_scale_focal_density_roundtrip_and_export() -> None:
    from pathlib import Path
    from synccraft.backends import export_bbmodel, export_bedrock
    from synccraft.dsl.ast_nodes import DSLValidationError
    from synccraft.shading.uv_packer import allocate_uv_layout, build_atlas

    dsl = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 32, size_step: 0.5, audit_profile: sculpted }
@palette {
  hide: ramp(base_lch=[0.5, 0.08, 195], steps=5)
  eye:  ramp(base_lch=[0.8, 0.16, 80], steps=5)
}
@rig {
  root bone head {
    pivot_local: [0, 6, 0]
    box skull { size: [8, 6, 8], material: hide }
    box eye_l place(base_box=skull, anchor=[-1, 0.2, 0.2], align=[0.5, 0, 0]) {
      size: [1.5, 2, 3], material: eye, uv_scale: 2
    }
  }
}
"""
    prog = parse_program(dsl)
    rt = format_program(prog)
    assert "uv_scale: 2" in rt
    assert parse_program(rt) == prog

    scene = compile_geometry(prog)
    layout = allocate_uv_layout(prog, scene)
    sk_r = layout["faces"]["skull"]["west"]
    ey_r = layout["faces"]["eye_l"]["west"]
    assert (sk_r[2] - sk_r[0], sk_r[3] - sk_r[1]) == (16, 12)
    assert (ey_r[2] - ey_r[0], ey_r[3] - ey_r[1]) == (12, 8)

    atlas, _ = build_atlas(prog, scene)
    bb = export_bbmodel(scene, atlas=atlas, program=prog)
    bed = export_bedrock(scene, atlas=atlas, program=prog)
    assert bed["serialized_vertex_max_error"] <= 1e-6
    assert any(el["name"] == "eye_l" for el in bb["elements"])

    try:
        from lark import Lark
        grammar = (Path(__file__).resolve().parents[1] / "src/synccraft/dsl/grammar.lark").read_text()
        Lark(grammar, start="start", parser="earley").parse(dsl)
        Lark(grammar, start="start", parser="earley").parse(rt)
    except ImportError:
        pass

    try:
        parse_program(dsl.replace("uv_scale: 2", "uv_scale: 16"))
        raise AssertionError("Expected DSLValidationError for uv_scale > 8.0")
    except DSLValidationError:
        pass
