"""Unified `.syncproj` Asset Project Manifest & Batch Variant/Biome Morph Factory (`SyncCraft v49.0 / Milestone v12`).

Links 3D sculpted Blockbench geometry (`.blockdsl` + `.bbmodel`), declarative `.texdsl` texture graphs,
human artist exposed-parameter locks (`artist_overrides`), multi-engine PBR export targets, and
declarative multi-variant/biome tables into one clean version-controlled manifest (`schema: "synccraft.project/1.0"`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .ai_copilot import BUILT_IN_STYLE_PRESETS, apply_surgical_patch
from .core import TexGraph
from .mesh_baker import MeshBakeContext, bake_bbmodel_maps
from .quality import evaluate_texture_quality
from .texdsl import export_texdsl, parse_texdsl


DEFAULT_CHRONO_VARIANTS: dict[str, dict[str, Any]] = {
    "chronos_prime": {
        "title": "Chronos Prime (Time × Amethyst × Gemini 4)",
        "style_preset": "chronos_amethyst",
        "exposed_overrides": {"shadow_strength": 0.54, "emissive_bounce": 0.56},
        "eye_iris": "#A855F7",
        "eye_iris_inner": "#38F2FF",
        "eye_iris_outer": "#F5C451",
        "rune_color": "#A855F7",
        "rune_core_color": "#4FF8FF",
    },
    "solar_aeon": {
        "title": "Solar Aeon (Radiant Amber-Gold Timekeeper)",
        "style_preset": "solar_aeon",
        "exposed_overrides": {"shadow_strength": 0.50, "emissive_bounce": 0.62},
        "eye_iris": "#F59E0B",
        "eye_iris_inner": "#FEF08A",
        "eye_iris_outer": "#DC2626",
        "rune_color": "#F59E0B",
        "rune_core_color": "#FFFBEB",
    },
    "void_eclipse": {
        "title": "Void Eclipse (Planar Rift Singularity)",
        "style_preset": "void_eclipse",
        "exposed_overrides": {"shadow_strength": 0.64, "emissive_bounce": 0.64},
        "eye_iris": "#EC4899",
        "eye_iris_inner": "#FFD6EC",
        "eye_iris_outer": "#6D28D9",
        "rune_color": "#EC4899",
        "rune_core_color": "#FFE4F3",
    },
    "frost_singularity": {
        "title": "Frost Singularity (Glacial Chronomancer)",
        "style_preset": "frost_singularity",
        "exposed_overrides": {"shadow_strength": 0.48, "emissive_bounce": 0.52},
        "eye_iris": "#38BDF8",
        "eye_iris_inner": "#E0F9FF",
        "eye_iris_outer": "#2563EB",
        "rune_color": "#38BDF8",
        "rune_core_color": "#E0FCFF",
    },
}


@dataclass
class SyncProject:
    """Unified `.syncproj` manifest linking `.blockdsl`, `.bbmodel`, `.texdsl`, artist locks & variants."""

    name: str
    blockdsl_path: str | None = None
    bbmodel_path: str | None = None
    texdsl_path: str | None = None
    pbr_preset: str = "studio_hero"
    target_engines: list[str] = field(
        default_factory=lambda: ["synccraft_special_pbr", "labpbr_1_3", "bedrock_rtx", "unreal_orm", "unity_maskmap", "vanilla_baked"]
    )
    artist_overrides: dict[str, Any] = field(default_factory=dict)
    variants: dict[str, dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CHRONO_VARIANTS))
    base_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "synccraft.project/1.0",
            "name": self.name,
            "geometry": {
                "blockdsl": self.blockdsl_path,
                "bbmodel": self.bbmodel_path,
            },
            "texture": {
                "texdsl": self.texdsl_path,
                "pbr_preset": self.pbr_preset,
                "target_engines": list(self.target_engines),
            },
            "artist_overrides": copy.deepcopy(self.artist_overrides),
            "variants": copy.deepcopy(self.variants),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.base_dir = p.parent
        return p

    @classmethod
    def load(cls, path: str | Path) -> "SyncProject":
        p = Path(path).resolve()
        raw = json.loads(p.read_text(encoding="utf-8"))
        if raw.get("schema") != "synccraft.project/1.0":
            raise ValueError(f"Unsupported .syncproj schema: {raw.get('schema')!r}")
        geom = raw.get("geometry") or {}
        tex = raw.get("texture") or {}
        return cls(
            name=str(raw.get("name", p.stem)),
            blockdsl_path=geom.get("blockdsl"),
            bbmodel_path=geom.get("bbmodel"),
            texdsl_path=tex.get("texdsl"),
            pbr_preset=str(tex.get("pbr_preset", "studio_hero")),
            target_engines=list(tex.get("target_engines") or []),
            artist_overrides=dict(raw.get("artist_overrides") or {}),
            variants=dict(raw.get("variants") or {}),
            base_dir=p.parent,
        )

    def _resolve(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        if not p.is_absolute() and self.base_dir is not None:
            p = (self.base_dir / p).resolve()
        return p


def apply_variant_to_graph(
    graph: TexGraph,
    variant_spec: dict[str, Any],
    *,
    artist_overrides: dict[str, Any] | None = None,
) -> TexGraph:
    """Clone `graph` and apply a declarative biome/chronal variant cleanly (honoring artist overrides!)."""
    g_var = TexGraph.from_recipe(graph.to_recipe(), mesh_context=graph.mesh_context)
    g_var.domain = graph.domain

    # 1. Apply style preset palettes to scale / primary & brass / metal material nodes
    preset_key = variant_spec.get("style_preset")
    if preset_key and preset_key in BUILT_IN_STYLE_PRESETS:
        st = BUILT_IN_STYLE_PRESETS[preset_key]
        for nid, sp in g_var.nodes.items():
            if sp.op in ("aquatic_scales_skin", "cobblestone_masonry", "orc_flesh_skin", "organic_fur_pelt"):
                for pk in ("base_stops", "dorsal_stops", "stone_stops", "stops", "coat_stops"):
                    if pk in sp.params:
                        if pk == "base_stops":
                            n = len(st["stops"])
                            sp.params[pk] = [[round(i / max(1, n - 1), 3), hx] for i, hx in enumerate(st["stops"])]
                        else:
                            sp.params[pk] = list(st["stops"])
                        break
            elif sp.op == "forged_battle_metal" and ("brass" in nid or "gold" in nid):
                if "stops" in sp.params:
                    sp.params["stops"] = list(st["metal_stops"])

    # 2. Apply eye / rune emissive palette updates on landmark detail nodes
    for nid, sp in g_var.nodes.items():
        if sp.op == "paint_entity_eyes_and_parts":
            for k in ("eye_iris", "eye_iris_inner", "eye_iris_outer", "rune_color", "rune_core_color"):
                if k in variant_spec:
                    sp.params[k] = variant_spec[k]

    # 3. Apply variant exposed parameter overrides
    for pub, val in (variant_spec.get("exposed_overrides") or {}).items():
        if pub in g_var.exposed_params:
            g_var.set_exposed(pub, val)

    # 4. Re-apply human artist overrides LAST so artist locks always win!
    for k, val in (artist_overrides or {}).items():
        if "." in k:
            nid_k, p_k = k.split(".", 1)
            if nid_k in g_var.nodes:
                g_var.nodes[nid_k].params[p_k] = val
        elif k in g_var.exposed_params:
            g_var.set_exposed(k, val)

    g_var._static_cache.clear()
    return g_var


def generate_variant_pack(
    graph: TexGraph,
    variants: dict[str, dict[str, Any]],
    out_dir: str | Path,
    *,
    bbmodel_doc: dict[str, Any] | None = None,
    artist_overrides: dict[str, Any] | None = None,
    pbr_preset: str = "studio_hero",
    qa_gate: float = 80.0,
) -> dict[str, Any]:
    """Batch Variant & Biome Morph Factory — builds multi-engine PBR + `.bbmodel` variants with automated QA gating!"""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    mc = graph.mesh_context
    dom = "entity" if mc is not None else (getattr(graph, "domain", None) or "block")

    results: list[dict[str, Any]] = []
    all_passed = True

    for v_name, v_spec in variants.items():
        v_dir = out_root / v_name
        v_dir.mkdir(parents=True, exist_ok=True)
        g_v = apply_variant_to_graph(graph, v_spec, artist_overrides=artist_overrides)
        g_v.name = f"{graph.name}_{v_name}"
        ev = g_v.evaluate(phase=0.0, build_pbr=True, allow_orphan_errors=True)
        qa = evaluate_texture_quality(
            ev.output,
            domain=dom,
            tileable=g_v.tileable,
            mesh_ctx=mc,
            pbr_channels=ev.pbr_channels,
            pbr_maps=ev.pbr_maps,
        )
        passed = float(qa.score) >= float(qa_gate)
        if not passed:
            all_passed = False

        pkg = g_v.export_baked_pbr(
            v_dir,
            stem=f"{graph.name}_{v_name}",
            preset=pbr_preset,
            bbmodel_doc=bbmodel_doc,
            embed_maps_in_manifest=False,
        )
        (v_dir / f"{graph.name}_{v_name}.texdsl").write_text(export_texdsl(g_v), encoding="utf-8")

        results.append(
            {
                "variant": v_name,
                "title": str(v_spec.get("title", v_name)),
                "qa_score": round(float(qa.score), 1),
                "qa_passed": passed,
                "out_dir": str(v_dir),
                "files_count": len(pkg["files"]) + 1,
                "baked_png": pkg["files"].get("baked_albedo"),
                "baked_bbmodel": pkg["files"].get("baked_bbmodel"),
            }
        )

    summary = {
        "schema": "synccraft.variant_pack/1.0",
        "base_graph": graph.name,
        "qa_gate": float(qa_gate),
        "all_passed": all_passed,
        "variant_count": len(results),
        "variants": results,
    }
    (out_root / "variant_pack_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
