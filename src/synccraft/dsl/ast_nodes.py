"""Typed Block-DSL v3.1 AST & semantic validator (ADR 0001 / 0003 / 0006).

All public angles are degrees. Supports arbitrary-depth nested ``mirror_x``,
``joint_limits`` & ``locator`` nodes in ``@rig``, extended physical/RD ``@materials``
parameters, and optional ``@muscles``, ``@kinematics``, and ``@behavior`` sections,
while preserving ``UnsupportedFeature`` on unimplemented ``@shade``, ``@motion``,
``@ergonomics``, and ``extrude`` boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import TypeAlias

Vec3: TypeAlias = tuple[float, float, float]
Vec3i: TypeAlias = tuple[int, int, int]
JointLimits: TypeAlias = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

ARCHETYPES = frozenset({"entity", "weapon_3d", "block_multipart", "item_extruded"})
PROFILES = frozenset({"bedrock_entity", "geckolib", "java_modern", "java_legacy", "bbmodel_5"})
ANIMATED_PROFILES = frozenset({"bedrock_entity", "geckolib", "bbmodel_5"})
FACES = frozenset({"north", "south", "east", "west", "up", "down"})
AXES_3D = frozenset({"x", "y", "z"})
LOCATOR_ROLES = frozenset({"socket", "ik_effector", "vfx_emitter", "camera_focus", "hitbox", "rider_seat"})
TRACK_LAWS = frozenset({"wave", "flap", "hinge", "impulse"})
MATERIAL_FAMILIES_DSL_LIST = (
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
)
MATERIAL_FAMILIES_DSL = frozenset(MATERIAL_FAMILIES_DSL_LIST)
TARGET_ENCODINGS_DSL = frozenset({"vanilla", "labpbr_1_3", "gltf_2_0"})

MAX_EXPANDED_MIRROR_NODES = 8192


class DSLValidationError(ValueError):
    """A complete declaration violates the Block-DSL semantic contract."""


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
class Locator:
    """Named 3D socket / IK-effector / emitter / seat node attached to a Bone frame."""

    name: str
    offset: Vec3 = (0.0, 0.0, 0.0)
    role: str = "socket"


@dataclass(frozen=True)
class Bone:
    name: str
    pivot_local: Vec3 = (0.0, 0.0, 0.0)
    rest_euler: Vec3 = (0.0, 0.0, 0.0)
    attach: Attach | None = None
    mirror_name: str | None = None
    items: list[Box | Locator | Bone] = field(default_factory=list)
    joint_limits: JointLimits | None = None


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
    anisotropy: float = 1.0
    subsurface: float = 0.0
    porosity: float = 0.08
    emissive: float = 0.0
    rd_feed: float = 0.036
    rd_kill: float = 0.062
    film_nm: float = 380.0
    wear: float = 0.35
    oxidation: float = 0.25
    biofilm: float = 0.20
    weld_group: str | None = None


@dataclass(frozen=True)
class StyleDecl:
    """Compact optional @style AST node configuring stylized bake and PBR target encoding."""

    target_encoding: str = "vanilla"
    quantize_ramp: bool = True
    bevel_highlight: float = 0.08
    view_deg: float = 20.0


@dataclass(frozen=True)
class TendonCoupling:
    """Biarticular mechanical tendon coupling from primary insertion bone to a secondary bone."""

    bone: str
    axis: str
    ratio: float


@dataclass(frozen=True)
class MuscleDecl:
    """Reduced-order Hill-type muscle-tendon unit with optional tendon pantograph & isochoric bulge."""

    name: str
    origin_bone: str
    insertion_bone: str
    action_axis: str
    moment_arm_deg: float
    tendon_coupling: TendonCoupling | None = None
    belly_box: str | None = None
    bulge_inflate: float = 0.0


@dataclass(frozen=True)
class IKChainDecl:
    """Analytic 2-bone limb IK descriptor in @kinematics."""

    name: str
    upper: str
    lower: str
    effector: Vec3
    hinge_axis: str = "x"


@dataclass(frozen=True)
class LookChainDecl:
    """Multi-bone distributed gaze/posture chain (e.g. neck_1 -> neck_2 -> head)."""

    name: str
    bones: tuple[str, ...]
    weight_decay: float = 0.75


@dataclass(frozen=True)
class KinematicsDecl:
    """Typed @kinematics section binding IK and multi-bone look chains to @rig."""

    ik_chains: tuple[IKChainDecl, ...] = ()
    look_chains: tuple[LookChainDecl, ...] = ()


@dataclass(frozen=True)
class TrackDecl:
    """Analytic bone rotation channel inside a @behavior clip."""

    bone: str
    law: str
    axis: str
    amp_deg: float
    freq_hz: float
    phase_deg: float = 0.0
    bias_deg: float = 0.0


@dataclass(frozen=True)
class MuscleDriveDecl:
    """Periodic/phasic neural excitation curve driving a declared @muscles unit."""

    muscle: str
    peak: float
    freq_hz: float
    phase_deg: float = 0.0
    baseline: float = 0.0


@dataclass(frozen=True)
class IKDriveDecl:
    """Closed-loop stance-locked gait cycle driving a declared @kinematics ik_chain."""

    chain: str
    stride: float
    lift: float
    phase_deg: float = 0.0


@dataclass(frozen=True)
class ClipDecl:
    """Named multi-channel biomechanical/mechanical animation clip in @behavior."""

    name: str
    duration_s: float
    loop: bool = True
    tracks: tuple[TrackDecl, ...] = ()
    muscle_drives: tuple[MuscleDriveDecl, ...] = ()
    ik_drives: tuple[IKDriveDecl, ...] = ()


@dataclass(frozen=True)
class MaterialModDecl:
    """State-coupled modulation of surface optical/persistent fields."""

    material: str
    emissive: float = 0.0
    moisture: float = 0.0
    wear_delta: float = 0.0


@dataclass(frozen=True)
class BehaviorStateDecl:
    """Named morphological + muscular + surface state (e.g. enraged, charged, weathered)."""

    name: str
    pose_biases: tuple[tuple[str, Vec3], ...] = ()
    muscle_tones: tuple[tuple[str, float], ...] = ()
    material_mods: tuple[MaterialModDecl, ...] = ()


@dataclass(frozen=True)
class BehaviorDecl:
    """Typed @behavior section coupling movement clips and reactive entity/asset states."""

    clips: tuple[ClipDecl, ...] = ()
    states: tuple[BehaviorStateDecl, ...] = ()


@dataclass(frozen=True)
class Program:
    meta: Meta
    palettes: list[Palette]
    root: Bone
    shade: tuple[ShadeOp, ...] = ()
    ergonomics: Ergonomics | None = None
    materials: tuple[MaterialDecl, ...] = ()
    style: StyleDecl | None = None
    muscles: tuple[MuscleDecl, ...] = ()
    kinematics: KinematicsDecl | None = None
    behavior: BehaviorDecl | None = None


def _finite_number(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_ident(name: object) -> bool:
    return isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _vec(value: tuple, label: str, length: int = 3, unit: bool = False) -> None:
    if not isinstance(value, tuple) or len(value) != length or any(not _finite_number(x) for x in value):
        raise DSLValidationError(f"{label}: expected {length} finite numbers")
    if unit and any(abs(x) > 1.0 for x in value):
        raise DSLValidationError(f"{label}: anchor/align components must lie in [-1, 1]")


def _validate_joint_limits(limits: JointLimits, label: str) -> None:
    if not isinstance(limits, tuple) or len(limits) != 3:
        raise DSLValidationError(f"{label}: expected 3 [min_deg, max_deg] pairs")
    for idx, pair in enumerate(limits):
        if not isinstance(pair, tuple) or len(pair) != 2 or any(not _finite_number(v) for v in pair):
            raise DSLValidationError(f"{label}[{idx}]: expected (min_deg, max_deg) finite pair")
        lo, hi = float(pair[0]), float(pair[1])
        if lo < -180.0 or hi > 180.0 or lo > 0.0 or hi < 0.0 or lo > hi:
            raise DSLValidationError(
                f"{label}[{idx}]: joint_limits must satisfy -180 <= min_deg <= 0 <= max_deg <= 180"
            )


def validate_program(program: Program) -> Program:
    """Validate symbol scope, primitive types, hierarchy expansion, and profile compatibility.

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
        if not _valid_ident(name):
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

    def descendant_expanded_names(node: Bone, budget_counter: list[int]) -> list[str]:
        out: list[str] = []
        for item in node.items:
            if not isinstance(item, (Box, Locator, Bone)):
                raise UnsupportedFeature(f"unimplemented node type {type(item).__name__}")
            out.append(item.name)
            budget_counter[0] += 1
            if budget_counter[0] > MAX_EXPANDED_MIRROR_NODES:
                raise DSLValidationError(
                    f"expanded mirror_x hierarchy exceeds {MAX_EXPANDED_MIRROR_NODES} node limit"
                )
            if isinstance(item, Bone):
                sub_names = descendant_expanded_names(item, budget_counter)
                out.extend(sub_names)
                if item.mirror_name:
                    if not _valid_ident(item.mirror_name):
                        raise DSLValidationError(f"invalid identifier: {item.mirror_name!r}")
                    out.append(item.mirror_name)
                    budget_counter[0] += 1 + len(sub_names)
                    if budget_counter[0] > MAX_EXPANDED_MIRROR_NODES:
                        raise DSLValidationError(
                            f"expanded mirror_x hierarchy exceeds {MAX_EXPANDED_MIRROR_NODES} node limit"
                        )
                    out.extend(f"{item.mirror_name}__{sn}" for sn in sub_names)
        return out

    count = 0
    has_any_box = False

    def walk(
        bone: Bone,
        parent_boxes: set[str] | None,
        parent_locators: set[str] | None,
        depth: int,
    ) -> None:
        nonlocal count, has_any_box
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
            if bone.attach is not None or bone.mirror_name is not None or any(bone.rest_euler) or bone.joint_limits is not None:
                raise DSLValidationError("root has pivot_local only; no attach, mirror, rest_euler, or joint_limits")
        else:
            valid_parents = (parent_boxes or set()) | (parent_locators or set())
            if (
                not isinstance(bone.attach, Attach)
                or not isinstance(bone.attach.parent_box, str)
                or bone.attach.parent_box not in valid_parents
            ):
                raise DSLValidationError(
                    f"bone {bone.name}: parent_box must name a preceding box or locator in its direct parent bone"
                )
            attach = bone.attach
            _vec(attach.anchor, f"{bone.name}.attach.anchor", unit=True)
            _vec(attach.align, f"{bone.name}.attach.align", unit=True)
            _vec(attach.offset, f"{bone.name}.attach.offset")
            if parent_locators and attach.parent_box in parent_locators and any(attach.anchor):
                raise DSLValidationError(
                    f"bone {bone.name}: attach to a dimensionless parent locator requires anchor=[0,0,0]"
                )
            if type(attach.detached) is not bool:
                raise DSLValidationError("detached must be a boolean")
            if not attach.detached and any(abs(x) > 0.5 for x in attach.offset):
                raise DSLValidationError("connected attach offset must lie in [-0.5,0.5] on every axis")
            if any(bone.pivot_local):
                raise DSLValidationError("child pivot is derived from attach; pivot_local must remain zero")
            if bone.joint_limits is not None:
                _validate_joint_limits(bone.joint_limits, f"{bone.name}.joint_limits")
        if bone.mirror_name:
            budget = [len(generated)]
            sub_expanded = descendant_expanded_names(bone, budget)
            if len(generated) + 1 + len(sub_expanded) > MAX_EXPANDED_MIRROR_NODES:
                raise DSLValidationError(
                    f"expanded mirror_x hierarchy exceeds {MAX_EXPANDED_MIRROR_NODES} node limit"
                )
            generated.append(bone.mirror_name)
            generated.extend(f"{bone.mirror_name}__{name}" for name in sub_expanded)

        local_boxes: set[str] = set()
        local_locators: set[str] = set()
        for item in bone.items:
            if isinstance(item, Bone):
                walk(item, local_boxes.copy(), local_locators.copy(), depth + 1)
                continue
            if isinstance(item, Locator):
                count += 1
                if count > 4096:
                    raise DSLValidationError("maximum supported declarations is 4096")
                claim(item.name)
                _vec(item.offset, f"{item.name}.offset")
                if not isinstance(item.role, str) or item.role not in LOCATOR_ROLES:
                    raise DSLValidationError(f"locator {item.name}: unknown role {item.role!r}")
                local_locators.add(item.name)
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
            has_any_box = True
        if not local_boxes and not local_locators:
            raise DSLValidationError(f"bone {bone.name}: requires at least one direct box or locator")

    walk(program.root, None, None, 0)
    if not has_any_box:
        raise DSLValidationError("@rig must contain at least one box primitive")
    for name in generated:
        claim(name)

    # Build expanded bone parent map & expanded box set for @muscles/@kinematics/@behavior validation
    expanded_bone_parent: dict[str, str | None] = {}
    expanded_boxes: set[str] = set()

    def _collect_expanded_graph(
        bone: Bone,
        parent_name: str | None,
        prefix: str | None = None,
        renamed_root: str | None = None,
    ) -> None:
        bname = renamed_root or (f"{prefix}__{bone.name}" if prefix else bone.name)
        expanded_bone_parent[bname] = parent_name
        for it in bone.items:
            if isinstance(it, Box):
                expanded_boxes.add(f"{prefix}__{it.name}" if prefix else it.name)
            elif isinstance(it, Bone):
                _collect_expanded_graph(it, bname, prefix=prefix)
                if it.mirror_name:
                    m_root = f"{prefix}__{it.mirror_name}" if prefix else it.mirror_name
                    _collect_expanded_graph(it, bname, prefix=m_root, renamed_root=m_root)

    _collect_expanded_graph(program.root, None)
    root_bone_name = program.root.name

    # Validate @materials
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
        if not _finite_number(mdecl.anisotropy) or mdecl.anisotropy <= 0.0:
            raise DSLValidationError("anisotropy must be > 0.0")
        if not _finite_number(mdecl.subsurface) or not (0.0 <= mdecl.subsurface <= 1.0):
            raise DSLValidationError("subsurface must lie in [0.0, 1.0]")
        if not _finite_number(mdecl.porosity) or not (0.0 <= mdecl.porosity <= 1.0):
            raise DSLValidationError("porosity must lie in [0.0, 1.0]")
        if not _finite_number(mdecl.emissive) or not (0.0 <= mdecl.emissive <= 2.0):
            raise DSLValidationError("emissive must lie in [0.0, 2.0]")
        if not _finite_number(mdecl.rd_feed) or not (0.002 <= mdecl.rd_feed <= 0.2):
            raise DSLValidationError("rd_feed must lie in [0.002, 0.2]")
        if not _finite_number(mdecl.rd_kill) or not (0.002 <= mdecl.rd_kill <= 0.2):
            raise DSLValidationError("rd_kill must lie in [0.002, 0.2]")
        if not _finite_number(mdecl.film_nm) or mdecl.film_nm < 0.0:
            raise DSLValidationError("film_nm must be >= 0.0")
        if not _finite_number(mdecl.wear) or mdecl.wear < 0.0:
            raise DSLValidationError("wear must be >= 0.0")
        if not _finite_number(mdecl.oxidation) or mdecl.oxidation < 0.0:
            raise DSLValidationError("oxidation must be >= 0.0")
        if not _finite_number(mdecl.biofilm) or mdecl.biofilm < 0.0:
            raise DSLValidationError("biofilm must be >= 0.0")
        if mdecl.weld_group is not None and not _valid_ident(mdecl.weld_group):
            raise DSLValidationError(f"invalid weld_group identifier {mdecl.weld_group!r}")

    # Validate @style
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

    # Validate @muscles
    if not isinstance(program.muscles, tuple):
        raise DSLValidationError("muscles must be a tuple of MuscleDecl objects")
    declared_muscles: set[str] = set()
    for mus in program.muscles:
        if not isinstance(mus, MuscleDecl):
            raise DSLValidationError("muscles must contain MuscleDecl objects")
        claim(mus.name)
        declared_muscles.add(mus.name)
        if mus.origin_bone not in expanded_bone_parent:
            raise DSLValidationError(f"muscle {mus.name}: unknown origin_bone {mus.origin_bone!r}")
        if mus.insertion_bone not in expanded_bone_parent or mus.insertion_bone == root_bone_name:
            raise DSLValidationError(f"muscle {mus.name}: insertion_bone {mus.insertion_bone!r} must be a non-root bone")
        if mus.origin_bone == mus.insertion_bone:
            raise DSLValidationError(f"muscle {mus.name}: origin_bone and insertion_bone must differ")
        if mus.action_axis not in AXES_3D:
            raise DSLValidationError(f"muscle {mus.name}: action_axis must be x, y, or z")
        if not _finite_number(mus.moment_arm_deg) or abs(mus.moment_arm_deg) > 90.0:
            raise DSLValidationError(f"muscle {mus.name}: moment_arm_deg must lie in [-90, 90]")
        if mus.tendon_coupling is not None:
            tc = mus.tendon_coupling
            if not isinstance(tc, TendonCoupling):
                raise DSLValidationError(f"muscle {mus.name}: tendon_coupling must be TendonCoupling")
            if tc.bone not in expanded_bone_parent or tc.bone == root_bone_name or tc.bone == mus.insertion_bone:
                raise DSLValidationError(f"muscle {mus.name}: tendon_coupling bone {tc.bone!r} must be a distinct non-root bone")
            if tc.axis not in AXES_3D:
                raise DSLValidationError(f"muscle {mus.name}: tendon_coupling axis must be x, y, or z")
            if not _finite_number(tc.ratio) or abs(tc.ratio) > 2.5:
                raise DSLValidationError(f"muscle {mus.name}: tendon_coupling ratio must lie in [-2.5, 2.5]")
        if not _finite_number(mus.bulge_inflate) or not (0.0 <= mus.bulge_inflate <= 0.35):
            raise DSLValidationError(f"muscle {mus.name}: bulge_inflate must lie in [0.0, 0.35]")
        if mus.belly_box is not None:
            if mus.belly_box not in expanded_boxes:
                raise DSLValidationError(f"muscle {mus.name}: unknown belly_box {mus.belly_box!r}")
        elif mus.bulge_inflate > 0.0:
            raise DSLValidationError(f"muscle {mus.name}: bulge_inflate > 0 requires belly_box")

    # Validate @kinematics
    declared_ik_chains: set[str] = set()
    if program.kinematics is not None:
        kdecl = program.kinematics
        if not isinstance(kdecl, KinematicsDecl):
            raise DSLValidationError("kinematics must be a KinematicsDecl or None")
        if not kdecl.ik_chains and not kdecl.look_chains:
            raise DSLValidationError("@kinematics must declare at least one ik_chain or look_chain")
        for ikc in kdecl.ik_chains:
            if not isinstance(ikc, IKChainDecl):
                raise DSLValidationError("ik_chains must contain IKChainDecl objects")
            claim(ikc.name)
            declared_ik_chains.add(ikc.name)
            if ikc.upper not in expanded_bone_parent or ikc.upper == root_bone_name:
                raise DSLValidationError(f"ik_chain {ikc.name}: upper bone {ikc.upper!r} must be a non-root bone")
            if ikc.lower not in expanded_bone_parent or expanded_bone_parent[ikc.lower] != ikc.upper:
                raise DSLValidationError(f"ik_chain {ikc.name}: lower bone {ikc.lower!r} must be a direct child of {ikc.upper!r}")
            _vec(ikc.effector, f"{ikc.name}.effector")
            if math.sqrt(sum(float(v) * float(v) for v in ikc.effector)) <= 1e-4:
                raise DSLValidationError(f"ik_chain {ikc.name}: effector offset must be nonzero")
            if ikc.hinge_axis not in AXES_3D:
                raise DSLValidationError(f"ik_chain {ikc.name}: hinge_axis must be x, y, or z")
        for lkc in kdecl.look_chains:
            if not isinstance(lkc, LookChainDecl):
                raise DSLValidationError("look_chains must contain LookChainDecl objects")
            claim(lkc.name)
            if not isinstance(lkc.bones, tuple) or not (1 <= len(lkc.bones) <= 12) or len(set(lkc.bones)) != len(lkc.bones):
                raise DSLValidationError(f"look_chain {lkc.name}: bones must contain 1..12 unique bone names")
            for bn in lkc.bones:
                if bn not in expanded_bone_parent or bn == root_bone_name:
                    raise DSLValidationError(f"look_chain {lkc.name}: unknown or root bone {bn!r}")
            if not _finite_number(lkc.weight_decay) or not (0.05 <= lkc.weight_decay <= 1.0):
                raise DSLValidationError(f"look_chain {lkc.name}: weight_decay must lie in [0.05, 1.0]")

    # Validate @behavior
    if program.behavior is not None:
        bdecl = program.behavior
        if not isinstance(bdecl, BehaviorDecl):
            raise DSLValidationError("behavior must be a BehaviorDecl or None")
        if not bdecl.clips and not bdecl.states:
            raise DSLValidationError("@behavior must declare at least one clip or state")
        if bdecl.clips and meta.target_profile not in ANIMATED_PROFILES:
            raise DSLValidationError(
                f"@behavior clips are not supported on static vanilla profile {meta.target_profile!r}; "
                f"use one of {sorted(ANIMATED_PROFILES)}"
            )
        for clip in bdecl.clips:
            if not isinstance(clip, ClipDecl):
                raise DSLValidationError("clips must contain ClipDecl objects")
            claim(clip.name)
            if not _finite_number(clip.duration_s) or not (0.05 <= clip.duration_s <= 60.0):
                raise DSLValidationError(f"clip {clip.name}: duration_s must lie in [0.05, 60.0]")
            if type(clip.loop) is not bool:
                raise DSLValidationError(f"clip {clip.name}: loop must be a boolean")
            if not clip.tracks and not clip.muscle_drives and not clip.ik_drives:
                raise DSLValidationError(f"clip {clip.name}: must contain at least one track, muscle_drive, or ik_drive")
            for tr in clip.tracks:
                if not isinstance(tr, TrackDecl):
                    raise DSLValidationError("tracks must contain TrackDecl objects")
                if tr.bone not in expanded_bone_parent or tr.bone == root_bone_name:
                    raise DSLValidationError(f"clip {clip.name}: track references unknown or root bone {tr.bone!r}")
                if tr.law not in TRACK_LAWS:
                    raise DSLValidationError(f"clip {clip.name}: unknown track law {tr.law!r}")
                if tr.axis not in AXES_3D:
                    raise DSLValidationError(f"clip {clip.name}: track axis must be x, y, or z")
                if not _finite_number(tr.amp_deg) or abs(tr.amp_deg) > 120.0:
                    raise DSLValidationError(f"clip {clip.name}: track amp_deg must lie in [-120, 120]")
                if not _finite_number(tr.freq_hz) or not (0.0 < tr.freq_hz <= 20.0):
                    raise DSLValidationError(f"clip {clip.name}: track freq_hz must lie in (0, 20]")
                if not _finite_number(tr.phase_deg) or not _finite_number(tr.bias_deg) or abs(tr.bias_deg) > 90.0:
                    raise DSLValidationError(f"clip {clip.name}: invalid phase_deg or bias_deg")
            for md in clip.muscle_drives:
                if not isinstance(md, MuscleDriveDecl):
                    raise DSLValidationError("muscle_drives must contain MuscleDriveDecl objects")
                if md.muscle not in declared_muscles:
                    raise DSLValidationError(f"clip {clip.name}: muscle_drive references undeclared muscle {md.muscle!r}")
                if not _finite_number(md.peak) or not (0.0 <= md.peak <= 1.0):
                    raise DSLValidationError(f"clip {clip.name}: muscle_drive peak must lie in [0, 1]")
                if not _finite_number(md.baseline) or not (0.0 <= md.baseline <= md.peak):
                    raise DSLValidationError(f"clip {clip.name}: muscle_drive baseline must lie in [0, peak]")
                if not _finite_number(md.freq_hz) or not (0.0 < md.freq_hz <= 20.0) or not _finite_number(md.phase_deg):
                    raise DSLValidationError(f"clip {clip.name}: invalid muscle_drive freq_hz or phase_deg")
            for ikd in clip.ik_drives:
                if not isinstance(ikd, IKDriveDecl):
                    raise DSLValidationError("ik_drives must contain IKDriveDecl objects")
                if ikd.chain not in declared_ik_chains:
                    raise DSLValidationError(f"clip {clip.name}: ik_drive references undeclared ik_chain {ikd.chain!r}")
                if not _finite_number(ikd.stride) or not (0.0 <= ikd.stride <= 24.0):
                    raise DSLValidationError(f"clip {clip.name}: ik_drive stride must lie in [0, 24]")
                if not _finite_number(ikd.lift) or not (0.0 <= ikd.lift <= 16.0) or not _finite_number(ikd.phase_deg):
                    raise DSLValidationError(f"clip {clip.name}: invalid ik_drive lift or phase_deg")
        for st in bdecl.states:
            if not isinstance(st, BehaviorStateDecl):
                raise DSLValidationError("states must contain BehaviorStateDecl objects")
            claim(st.name)
            if not st.pose_biases and not st.muscle_tones and not st.material_mods:
                raise DSLValidationError(f"state {st.name}: must declare at least one pose_bias, muscle_tone, or material_mod")
            for bname, pb in st.pose_biases:
                if bname not in expanded_bone_parent or bname == root_bone_name:
                    raise DSLValidationError(f"state {st.name}: pose_bias references unknown or root bone {bname!r}")
                _vec(pb, f"{st.name}.pose_bias.{bname}")
            for mname, tone in st.muscle_tones:
                if mname not in declared_muscles:
                    raise DSLValidationError(f"state {st.name}: muscle_tone references undeclared muscle {mname!r}")
                if not _finite_number(tone) or not (0.0 <= tone <= 1.0):
                    raise DSLValidationError(f"state {st.name}: muscle_tone must lie in [0, 1]")
            for mm in st.material_mods:
                if not isinstance(mm, MaterialModDecl) or mm.material not in materials:
                    raise DSLValidationError(f"state {st.name}: material_mod references undeclared palette material")
                if not _finite_number(mm.emissive) or not (0.0 <= mm.emissive <= 2.0):
                    raise DSLValidationError(f"state {st.name}: material_mod emissive must lie in [0, 2]")
                if not _finite_number(mm.moisture) or not (0.0 <= mm.moisture <= 1.0):
                    raise DSLValidationError(f"state {st.name}: material_mod moisture must lie in [0, 1]")
                if not _finite_number(mm.wear_delta) or not (0.0 <= mm.wear_delta <= 1.0):
                    raise DSLValidationError(f"state {st.name}: material_mod wear_delta must lie in [0, 1]")

    return program
