"""Typed Block-DSL v3 static starter AST. All public angles are degrees."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import TypeAlias

Vec3: TypeAlias = tuple[float, float, float]
Vec3i: TypeAlias = tuple[int, int, int]
ARCHETYPES = frozenset({"entity", "weapon_3d", "block_multipart", "item_extruded"})
PROFILES = frozenset({"bedrock_entity", "geckolib", "java_modern", "java_legacy", "bbmodel_5"})
FACES = frozenset({"north", "south", "east", "west", "up", "down"})
MATERIAL_FAMILIES_DSL = frozenset({
    "painted_metal",
    "brushed_metal",
    "polymer_dielectric",
    "woven_fabric",
    "solid_wood",
    "weathered_stone",
    "keratin_scales",
    "patterned_skin",
    "lichen_biofilm",
    "thin_film_iridescence",
})
TARGET_ENCODINGS_DSL = frozenset({"vanilla", "labpbr_1_3", "gltf_2_0"})


class DSLValidationError(ValueError):
    """A complete declaration violates the static starter contract."""


class UnsupportedFeature(DSLValidationError):
    """A recognized v3 feature has no implemented starter semantics."""


@dataclass(frozen=True)
class Meta:
    archetype: str
    target_profile: str
    texel_density: int


@dataclass(frozen=True)
class Palette:
    name: str
    base_lch: Vec3
    steps: int
    hue_shift: tuple[float, float] = (0.0, 0.0)
    emissive: bool = False


@dataclass(frozen=True)
class Attach:
    parent_box: str
    anchor: Vec3
    align: Vec3
    offset: Vec3 = (0.0, 0.0, 0.0)
    detached: bool = False


@dataclass(frozen=True)
class Place:
    base_box: str | None
    anchor: Vec3
    align: Vec3
    offset: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Box:
    name: str
    size: Vec3i
    material: str
    place: Place | None = None
    inflate: float = 0.0
    cullface: tuple[str, ...] = ()


@dataclass(frozen=True)
class Bone:
    name: str
    pivot_local: Vec3 = (0.0, 0.0, 0.0)
    rest_euler: Vec3 = (0.0, 0.0, 0.0)
    attach: Attach | None = None
    mirror_name: str | None = None
    items: list[Box | Bone] = field(default_factory=list)


@dataclass(frozen=True)
class ShadeOp:
    """Reserved future AST boundary; parser rejects @shade in this release."""
    name: str
    arguments: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Ergonomics:
    grip_box: str
    hand_mode: str
    gui_fill: float
    fp_tilt_deg: Vec3


@dataclass(frozen=True)
class MaterialDecl:
    """Compact optional @materials AST node binding a palette material to a physical family."""
    name: str
    family: str
    roughness: float = 0.35
    metalness: float = 0.0
    coating: float = 0.0
    ior: float = 1.5
    scale: float = 1.0
    seed: int = 101


@dataclass(frozen=True)
class StyleDecl:
    """Compact optional @style AST node configuring stylized bake and PBR target encoding."""
    target_encoding: str = "vanilla"
    quantize_ramp: bool = True
    bevel_highlight: float = 0.08
    view_deg: float = 20.0


@dataclass(frozen=True)
class Program:
    meta: Meta
    palettes: list[Palette]
    root: Bone
    shade: tuple[ShadeOp, ...] = ()
    ergonomics: Ergonomics | None = None
    materials: tuple[MaterialDecl, ...] = ()
    style: StyleDecl | None = None


def _finite_number(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # Python integers can exceed the finite float64 geometry domain.
        return False


def _vec(value: tuple, label: str, length: int = 3, unit: bool = False) -> None:
    if not isinstance(value, tuple) or len(value) != length or any(not _finite_number(x) for x in value):
        raise DSLValidationError(f"{label}: expected {length} finite numbers")
    if unit and any(abs(x) > 1.0 for x in value):
        raise DSLValidationError(f"{label}: anchor/align components must lie in [-1, 1]")


def validate_program(program: Program) -> Program:
    """Validate symbol scope, primitive types, and profile compatibility.

    World-coordinate bounds/rotation legality are checked after FK, not here.
    Returns the input to allow ``validate_program(parse(...))`` composition.
    """
    if not isinstance(program, Program) or not isinstance(program.meta, Meta) or not isinstance(program.root, Bone):
        raise DSLValidationError("expected Program with typed Meta and root Bone")
    meta = program.meta
    if not isinstance(meta.archetype, str) or not isinstance(meta.target_profile, str) or meta.archetype not in ARCHETYPES or meta.target_profile not in PROFILES:
        raise DSLValidationError("unknown archetype or target_profile")
    if type(meta.texel_density) is not int or meta.texel_density <= 0 or not _finite_number(meta.texel_density):
        raise DSLValidationError("texel_density must be a positive integer")
    if meta.archetype == "entity" and meta.target_profile.startswith("java_"):
        raise DSLValidationError("entity cannot target vanilla Java item/block geometry")
    if program.shade or program.ergonomics is not None:
        raise UnsupportedFeature("@shade and @ergonomics require future implemented rendering/grip semantics")
    if not isinstance(program.palettes, list) or not program.palettes:
        raise DSLValidationError("@palette must declare at least one material")
    used: set[str] = set()
    materials: set[str] = set()
    generated: list[str] = []

    def claim(name: str) -> None:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise DSLValidationError(f"invalid identifier: {name!r}")
        if name in used:
            raise DSLValidationError(f"duplicate identifier: {name}")
        used.add(name)

    for palette in program.palettes:
        if not isinstance(palette, Palette):
            raise DSLValidationError("palettes must contain Palette objects")
        claim(palette.name)
        materials.add(palette.name)
        _vec(palette.base_lch, f"{palette.name}.base_lch")
        _vec(palette.hue_shift, f"{palette.name}.hue_shift", 2)
        lightness, chroma, _ = palette.base_lch
        if not 0 <= lightness <= 1 or chroma < 0:
            raise DSLValidationError("OKLCH requires L in [0,1], C >= 0, hue in degrees")
        if type(palette.steps) is not int or not 2 <= palette.steps <= 256:
            raise DSLValidationError("palette steps must be an integer in [2,256]")
        if type(palette.emissive) is not bool:
            raise DSLValidationError("emissive must be a boolean")

    def descendant_names(node: Bone):
        for item in node.items:
            if not isinstance(item, (Box, Bone)):
                raise UnsupportedFeature(f"unimplemented node type {type(item).__name__}")
            yield item.name
            if isinstance(item, Bone):
                yield from descendant_names(item)

    count = 0

    def walk(bone: Bone, parent_boxes: set[str] | None, depth: int, inside_mirror: bool = False) -> None:
        nonlocal count
        if depth > 64:
            raise DSLValidationError("maximum supported bone depth is 64")
        count += 1
        if count > 4096:
            raise DSLValidationError("maximum supported declarations is 4096")
        claim(bone.name)
        _vec(bone.pivot_local, f"{bone.name}.pivot_local")
        _vec(bone.rest_euler, f"{bone.name}.rest_euler")
        if not isinstance(bone.items, list) or not bone.items:
            raise DSLValidationError(f"bone {bone.name} must contain at least one primitive")
        if parent_boxes is None:
            if bone.attach is not None or bone.mirror_name is not None or any(bone.rest_euler):
                raise DSLValidationError("root has pivot_local only; no attach, mirror, or rest_euler")
        else:
            if not isinstance(bone.attach, Attach) or not isinstance(bone.attach.parent_box, str) or bone.attach.parent_box not in parent_boxes:
                raise DSLValidationError(f"bone {bone.name}: parent_box must name a preceding box in its direct parent bone")
            attach = bone.attach
            _vec(attach.anchor, f"{bone.name}.attach.anchor", unit=True)
            _vec(attach.align, f"{bone.name}.attach.align", unit=True)
            _vec(attach.offset, f"{bone.name}.attach.offset")
            if type(attach.detached) is not bool:
                raise DSLValidationError("detached must be a boolean")
            if not attach.detached and any(abs(x) > 0.5 for x in attach.offset):
                raise DSLValidationError("connected attach offset must lie in [-0.5,0.5] on every axis")
            if any(bone.pivot_local):
                raise DSLValidationError("child pivot is derived from attach; pivot_local must remain zero")
        if bone.mirror_name:
            if inside_mirror:
                raise UnsupportedFeature("nested mirror_x is deferred until expansion semantics are frozen")
            generated.append(bone.mirror_name)
            generated.extend(f"{bone.mirror_name}__{name}" for name in descendant_names(bone))
        local_boxes: set[str] = set()
        for item in bone.items:
            if isinstance(item, Bone):
                walk(item, local_boxes.copy(), depth + 1, inside_mirror or bool(bone.mirror_name))
                continue
            if not isinstance(item, Box):
                raise UnsupportedFeature(f"unimplemented node type {type(item).__name__}")
            count += 1
            if count > 4096:
                raise DSLValidationError("maximum supported declarations is 4096")
            claim(item.name)
            if not isinstance(item.size, tuple) or len(item.size) != 3 or any(type(x) is not int or x <= 0 or not _finite_number(x) for x in item.size):
                raise DSLValidationError(f"box {item.name}: size must contain three positive integers")
            if not isinstance(item.material, str) or item.material not in materials:
                raise DSLValidationError(f"box {item.name}: undeclared material {item.material}")
            if not _finite_number(item.inflate) or item.inflate < 0:
                raise DSLValidationError("inflate must be a nonnegative finite number in this starter")
            if not isinstance(item.cullface, tuple) or any(not isinstance(x, str) for x in item.cullface) or len(set(item.cullface)) != len(item.cullface) or any(x not in FACES for x in item.cullface):
                raise DSLValidationError("cullface must contain unique valid face names")
            if item.place is None:
                if local_boxes:
                    raise DSLValidationError(f"box {item.name}: subsequent same-bone boxes require an explicit place clause")
            else:
                place = item.place
                if not isinstance(place, Place) or (place.base_box is not None and not isinstance(place.base_box, str)):
                    raise DSLValidationError("place must be a Place object with a string or None base_box")
                _vec(place.anchor, f"{item.name}.place.anchor", unit=True)
                _vec(place.align, f"{item.name}.place.align", unit=True)
                _vec(place.offset, f"{item.name}.place.offset")
                if place.base_box is not None and place.base_box not in local_boxes:
                    raise DSLValidationError(f"box {item.name}: base_box must name a preceding box in the same bone")
                if place.base_box is None and any(place.anchor):
                    raise UnsupportedFeature("place without base_box requires anchor=[0,0,0]; nonzero origin-relative anchors are unspecified")
            local_boxes.add(item.name)
        if not local_boxes:
            raise DSLValidationError(f"bone {bone.name}: starter requires at least one direct box")

    walk(program.root, None, 0)
    for name in generated:
        claim(name)

    if not isinstance(program.materials, tuple):
        raise DSLValidationError("materials must be a tuple of MaterialDecl objects")
    seen_mat_decls: set[str] = set()
    for mdecl in program.materials:
        if not isinstance(mdecl, MaterialDecl):
            raise DSLValidationError("materials must contain MaterialDecl objects")
        if not isinstance(mdecl.name, str) or mdecl.name not in materials:
            raise DSLValidationError(f"@materials references undeclared palette material {mdecl.name!r}")
        if mdecl.name in seen_mat_decls:
            raise DSLValidationError(f"duplicate @materials declaration for {mdecl.name!r}")
        seen_mat_decls.add(mdecl.name)
        if not isinstance(mdecl.family, str) or mdecl.family not in MATERIAL_FAMILIES_DSL:
            raise DSLValidationError(f"unknown material family {mdecl.family!r}")
        if not _finite_number(mdecl.roughness) or not (0.02 <= mdecl.roughness <= 1.0):
            raise DSLValidationError("roughness must lie in [0.02, 1.0]")
        if not _finite_number(mdecl.metalness) or not (0.0 <= mdecl.metalness <= 1.0):
            raise DSLValidationError("metalness must lie in [0.0, 1.0]")
        if not _finite_number(mdecl.coating) or mdecl.coating < 0.0:
            raise DSLValidationError("coating must be >= 0.0")
        if not _finite_number(mdecl.ior) or mdecl.ior < 1.0:
            raise DSLValidationError("ior must be >= 1.0")
        if not _finite_number(mdecl.scale) or mdecl.scale <= 0.0:
            raise DSLValidationError("scale must be > 0.0")
        if type(mdecl.seed) is not int or not _finite_number(mdecl.seed):
            raise DSLValidationError("seed must be a finite integer")

    if program.style is not None:
        sdecl = program.style
        if not isinstance(sdecl, StyleDecl):
            raise DSLValidationError("style must be a StyleDecl or None")
        if not isinstance(sdecl.target_encoding, str) or sdecl.target_encoding not in TARGET_ENCODINGS_DSL:
            raise DSLValidationError(f"unknown target_encoding {sdecl.target_encoding!r}")
        if type(sdecl.quantize_ramp) is not bool:
            raise DSLValidationError("quantize_ramp must be a boolean")
        if not _finite_number(sdecl.bevel_highlight) or sdecl.bevel_highlight < 0.0:
            raise DSLValidationError("bevel_highlight must be >= 0.0")
        if not _finite_number(sdecl.view_deg) or not (0.0 <= sdecl.view_deg <= 89.0):
            raise DSLValidationError("view_deg must lie in [0.0, 89.0]")

    return program
