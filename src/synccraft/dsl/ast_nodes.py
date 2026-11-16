"""Typed Block-DSL v3.2 AST & semantic validator (ADR 0001 / 0003 / 0006).

All public angles are degrees. Supports versioned ``@meta`` (``dsl_version``,
``size_step``, ``audit_profile``), ``@textures``, fractional box sizes on declared
``size_step`` grids, zero-thickness planar quads (``plane`` / ``Box.plane_normal``),
per-box local ``rotate`` & ``pivot``, UV metadata (``uv``, ``uv_faces``, ``texture``),
first-class ``joint(...)`` attachments, ``mirror(axis=x|z)`` & ``radial(count=K, axis=...)``
expansions, generalized $N$-bone ``IKChainDecl``, multi-keyframe ``key`` tracks,
2-parameter damped ``spring`` dynamics, and ``@controller`` state transitions, while
preserving 100% backward compatibility with v3.0/v3.1 AST constructors and explicit
``UnsupportedFeature`` errors on ``@shade``, ``@motion``, ``@ergonomics``, and ``extrude``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, TypeAlias

Vec3: TypeAlias = tuple[float, float, float]
Vec3i: TypeAlias = tuple[int, int, int]
Vec3Size: TypeAlias = tuple[int | float, int | float, int | float]
UVRect: TypeAlias = tuple[float, float, float, float] | tuple[float, float, float, float, int]
JointLimits: TypeAlias = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
UV_ROTATIONS = frozenset({0, 90, 180, 270})

ARCHETYPES = frozenset({"entity", "weapon_3d", "block_multipart", "item_extruded"})
PROFILES = frozenset({"bedrock_entity", "geckolib", "java_modern", "java_legacy", "bbmodel_5"})
ANIMATED_PROFILES = frozenset({"bedrock_entity", "geckolib", "bbmodel_5"})
FACES = frozenset({"north", "south", "east", "west", "up", "down"})
FACE_ORDER = ("north", "south", "east", "west", "up", "down")
AXES_3D = frozenset({"x", "y", "z"})
MIRROR_AXES = frozenset({"x", "z"})
ALLOWED_SIZE_STEPS = frozenset({1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.1, 0.05, 0.01, 0.001})
AUDIT_PROFILES = frozenset({"strict", "sculpted", "authored"})
DSL_VERSIONS = frozenset({"3.0", "3.1", "3.2", "4.0"})
LOCATOR_ROLES = frozenset({"socket", "ik_effector", "vfx_emitter", "camera_focus", "hitbox", "rider_seat"})
TRACK_LAWS = frozenset({"wave", "flap", "hinge", "impulse", "spin"})
SLIDE_LAWS = frozenset({"piston", "wave", "hinge", "impulse"})
KEYFRAME_CHANNELS = frozenset({"rotation", "position", "scale"})
KEYFRAME_INTERPOLATIONS = frozenset({"linear", "catmullrom"})
CONTROLLER_OPS = frozenset({">", "<", ">=", "<=", "==", "!="})
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
    dsl_version: str = "3.2"
    size_step: float = 1.0
    audit_profile: str = "strict"
    units_per_block: int = 16


@dataclass(frozen=True)
class TextureDecl:
    """Declared texture sheet binding in @textures."""

    name: str
    path: str
    width: int
    height: int


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
    anchor: Vec3 = (0.0, 0.0, 0.0)
    align: Vec3 = (0.0, 0.0, 0.0)
    offset: Vec3 = (0.0, 0.0, 0.0)
    detached: bool = False
    joint_mode: bool = False


def Joint(
    parent: str,
    at: Vec3 = (0.0, 0.0, 0.0),
    *,
    align: Vec3 = (0.0, 0.0, 0.0),
    detached: bool = False,
) -> Attach:
    """Construct a first-class joint(parent=..., at=[x, y, z]) Bone attachment."""
    return Attach(
        parent_box=parent,
        anchor=(0.0, 0.0, 0.0),
        align=align,
        offset=at,
        detached=detached,
        joint_mode=True,
    )


@dataclass(frozen=True)
class Place:
    base_box: str | None
    anchor: Vec3
    align: Vec3
    offset: Vec3 = (0.0, 0.0, 0.0)


def _normalize_single_uv_rect(vals: Any) -> UVRect | None:
    if not isinstance(vals, (tuple, list)) or len(vals) not in (4, 5):
        return None
    if len(vals) == 5:
        rot_v = vals[4]
        if not _finite_number(rot_v) or float(rot_v) != round(float(rot_v)):
            return None
        rot_i = int(round(float(rot_v))) % 360
        if rot_i not in UV_ROTATIONS:
            return None
        if rot_i != 0:
            return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), rot_i)
    return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))


def _normalize_uv_faces(raw: Any) -> Any:
    if raw is None or raw == ():
        return ()
    if isinstance(raw, dict):
        items: list[tuple[str, UVRect]] = []
        order_idx = {f: idx for idx, f in enumerate(FACE_ORDER)}
        for k in sorted(raw.keys(), key=lambda key: (order_idx.get(str(key), 99), str(key))):
            norm_r = _normalize_single_uv_rect(raw[k])
            if norm_r is not None:
                items.append((str(k), norm_r))
            else:
                return raw
        return tuple(items)
    if isinstance(raw, (list, tuple)):
        out: list[tuple[str, UVRect]] = []
        for entry in raw:
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                norm_r = _normalize_single_uv_rect(entry[1])
                if norm_r is not None:
                    out.append((str(entry[0]), norm_r))
                else:
                    return raw
            else:
                return raw
        return tuple(out)
    return raw


@dataclass(frozen=True)
class Box:
    name: str
    size: Vec3Size
    material: str
    place: Place | None = None
    inflate: float = 0.0
    cullface: tuple[str, ...] = ()
    rotate: Vec3 = (0.0, 0.0, 0.0)
    pivot: Vec3 | None = None
    plane_normal: str | None = None
    uv: tuple[float, float] | None = None
    uv_faces: tuple[tuple[str, UVRect], ...] = ()
    texture: str | None = None
    blend_with: tuple[str, ...] = ()
    uv_share_key: str | None = None
    component_rest_pos: Vec3 | None = None
    component_rest_euler: Vec3 | None = None
    explicit_local_center: Vec3 | None = None
    explicit_contact_with: tuple[str, ...] = ()
    instance_name: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.uv_faces, (dict, list)) or self.uv_faces is None:
            object.__setattr__(self, "uv_faces", _normalize_uv_faces(self.uv_faces))
        if isinstance(self.blend_with, list) or self.blend_with is None:
            object.__setattr__(self, "blend_with", tuple(self.blend_with or ()))
        if isinstance(self.explicit_contact_with, list) or self.explicit_contact_with is None:
            object.__setattr__(self, "explicit_contact_with", tuple(self.explicit_contact_with or ()))

    @property
    def uv_faces_dict(self) -> dict[str, UVRect]:
        return {face: rect for face, rect in self.uv_faces}


def Plane(
    name: str,
    size: tuple[int | float, ...],
    normal: str,
    material: str,
    place: Place | None = None,
    inflate: float = 0.0,
    cullface: tuple[str, ...] = (),
    rotate: Vec3 = (0.0, 0.0, 0.0),
    pivot: Vec3 | None = None,
    uv: tuple[float, float] | None = None,
    uv_faces: Any = (),
    texture: str | None = None,
    blend_with: tuple[str, ...] = (),
) -> Box:
    """Construct a zero-thickness planar quad Box with 0 physical thickness along ``normal``."""
    if len(size) == 2:
        w, h = size[0], size[1]
        zero: int | float = 0 if (type(w) is int and type(h) is int) else 0.0
        if normal == "x":
            size_3d: Vec3Size = (zero, w, h)
        elif normal == "y":
            size_3d = (w, zero, h)
        elif normal == "z":
            size_3d = (w, h, zero)
        else:
            size_3d = (w, h, zero)
    elif len(size) == 3:
        size_3d = (size[0], size[1], size[2])
    else:
        raise DSLValidationError(f"plane {name}: size must have 2 (or 3 with normal=0) components")
    return Box(
        name=name,
        size=size_3d,
        material=material,
        place=place,
        inflate=inflate,
        cullface=cullface,
        rotate=rotate,
        pivot=pivot,
        plane_normal=normal,
        uv=uv,
        uv_faces=uv_faces,
        texture=texture,
        blend_with=blend_with,
    )


@dataclass(frozen=True)
class Locator:
    """Named oriented 3D socket / IK-effector / emitter / seat frame attached to a Bone."""

    name: str
    offset: Vec3 = (0.0, 0.0, 0.0)
    role: str = "socket"
    rot: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RadialSpec:
    """Cyclic symmetry expansion around a parent axis (count >= 2)."""

    count: int
    axis: str = "y"


@dataclass(frozen=True)
class Bone:
    name: str
    pivot_local: Vec3 = (0.0, 0.0, 0.0)
    rest_euler: Vec3 = (0.0, 0.0, 0.0)
    attach: Attach | None = None
    mirror_name: str | None = None
    items: list[Box | Locator | Bone] = field(default_factory=list)
    joint_limits: JointLimits | None = None
    mirror_axis: str = "x"
    radial: RadialSpec | None = None


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
    """Generalized N-bone (N >= 2) limb/tentacle IK descriptor in @kinematics."""

    name: str
    upper: str = ""
    lower: str = ""
    effector: Vec3 = (0.0, -1.0, 0.0)
    hinge_axis: str = "x"
    bones: tuple[str, ...] = ()
    pole: Vec3 | None = None

    def __post_init__(self) -> None:
        if not self.bones and self.upper and self.lower:
            object.__setattr__(self, "bones", (self.upper, self.lower))
        elif isinstance(self.bones, (tuple, list)) and len(self.bones) >= 2:
            if isinstance(self.bones, list):
                object.__setattr__(self, "bones", tuple(self.bones))
            if not self.upper:
                object.__setattr__(self, "upper", str(self.bones[0]))
            if not self.lower:
                object.__setattr__(self, "lower", str(self.bones[-1]))


@dataclass(frozen=True)
class LookChainDecl:
    """Multi-bone distributed gaze/posture chain (e.g. neck_1 -> neck_2 -> head)."""

    name: str
    bones: tuple[str, ...]
    weight_decay: float = 0.75


@dataclass(frozen=True)
class SpringDecl:
    """2-parameter damped harmonic secondary follow-through spring on a bone."""

    bone: str
    stiffness: float = 40.0
    damping: float = 6.0


@dataclass(frozen=True)
class PrismaticJointDecl:
    """1-DOF linear slider/piston mechanical constraint on a non-root bone (ADR 0008)."""

    bone: str
    axis: str = "z"
    limits: tuple[float, float] = (-4.0, 4.0)
    axis_vec: Vec3 | None = None


@dataclass(frozen=True)
class KinematicsDecl:
    """Typed @kinematics section binding IK, look, secondary spring, and prismatic joints to @rig."""

    ik_chains: tuple[IKChainDecl, ...] = ()
    look_chains: tuple[LookChainDecl, ...] = ()
    springs: tuple[SpringDecl, ...] = ()
    prismatic_joints: tuple[PrismaticJointDecl, ...] = ()


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
class KeyframeTrackDecl:
    """Multi-keyframe bone channel (rotation | position | scale) inside a @behavior clip."""

    bone: str
    channel: str
    keys: tuple[tuple[float, Vec3], ...]
    interpolation: str = "linear"


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
class SlideDriveDecl:
    """1-DOF linear translation drive inside a @behavior clip (piston, slider, recoil)."""

    bone: str
    law: str
    axis: str
    amp_units: float
    freq_hz: float
    phase_deg: float = 0.0
    bias_units: float = 0.0


@dataclass(frozen=True)
class ClipDecl:
    """Named multi-channel biomechanical/mechanical/keyframe animation clip in @behavior."""

    name: str
    duration_s: float
    loop: bool = True
    tracks: tuple[TrackDecl, ...] = ()
    muscle_drives: tuple[MuscleDriveDecl, ...] = ()
    ik_drives: tuple[IKDriveDecl, ...] = ()
    keyframes: tuple[KeyframeTrackDecl, ...] = ()
    slide_drives: tuple[SlideDriveDecl, ...] = ()


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
    """Typed @behavior section coupling movement clips, springs, and reactive entity states."""

    clips: tuple[ClipDecl, ...] = ()
    states: tuple[BehaviorStateDecl, ...] = ()
    springs: tuple[SpringDecl, ...] = ()


@dataclass(frozen=True)
class ControllerTransitionDecl:
    """Reactive clip/state transition rule in @controller."""

    from_state: str
    to_state: str
    parameter: str
    operator: str
    threshold: float
    blend_s: float = 0.2


@dataclass(frozen=True)
class ControllerDecl:
    """Typed @controller state-machine section."""

    transitions: tuple[ControllerTransitionDecl, ...] = ()
    initial: str | None = None


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
    textures: tuple[TextureDecl, ...] = ()
    controller: ControllerDecl | None = None
    assembly_spec: Any = None
    source_map: Any = None


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)  # type: ignore[arg-type]
    except OverflowError:
        return False


def _valid_ident(name: object) -> bool:
    return isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _vec(value: Any, label: str, length: int = 3, unit: bool = False) -> None:
    if not isinstance(value, tuple) or len(value) != length or any(not _finite_number(x) for x in value):
        raise DSLValidationError(f"{label}: expected {length} finite numbers")
    if unit and any(abs(float(x)) > 1.0 for x in value):
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


def _validate_size_component(val: object, size_step: float, box_name: str) -> None:
    if not _finite_number(val):
        raise DSLValidationError(f"box {box_name}: size must contain finite numbers")
    if size_step == 1.0:
        if type(val) is not int or val <= 0:
            raise DSLValidationError(f"box {box_name}: size must contain three positive integers when size_step == 1.0")
    else:
        fval = float(val)  # type: ignore[arg-type]
        if fval <= 0.0:
            raise DSLValidationError(f"box {box_name}: non-planar size components must be strictly positive")
        steps = round(fval / size_step)
        if steps < 1 or abs(steps * size_step - fval) > 1e-9:
            raise DSLValidationError(
                f"box {box_name}: size component {val!r} is not aligned to meta.size_step={size_step}"
            )


def _validate_spring(sp: SpringDecl, expanded_bone_parent: dict[str, str | None], root_bone_name: str) -> None:
    if not isinstance(sp, SpringDecl):
        raise DSLValidationError("springs must contain SpringDecl objects")
    if sp.bone not in expanded_bone_parent or sp.bone == root_bone_name:
        raise DSLValidationError(f"spring references unknown or root bone {sp.bone!r}")
    if not _finite_number(sp.stiffness) or not (0.1 <= float(sp.stiffness) <= 2000.0):
        raise DSLValidationError(f"spring {sp.bone}: stiffness must lie in [0.1, 2000.0]")
    if not _finite_number(sp.damping) or not (0.0 <= float(sp.damping) <= 500.0):
        raise DSLValidationError(f"spring {sp.bone}: damping must lie in [0.0, 500.0]")


def validate_program(program: Program) -> Program:
    """Validate symbol scope, primitive types, hierarchy expansion, and profile compatibility.

    World-coordinate bounds/rotation legality are checked after FK, not here.
    Returns the input to allow ``validate_program(parse(...))`` composition.
    """
    if not isinstance(program, Program) or not isinstance(program.meta, Meta) or not isinstance(program.root, Bone):
        raise DSLValidationError("expected Program with typed Meta and root Bone")
    meta = program.meta
    if (
        not isinstance(meta.archetype, str)
        or not isinstance(meta.target_profile, str)
        or meta.archetype not in ARCHETYPES
        or meta.target_profile not in PROFILES
    ):
        raise DSLValidationError("unknown archetype or target_profile")
    if type(meta.texel_density) is not int or meta.texel_density <= 0 or not _finite_number(meta.texel_density):
        raise DSLValidationError("texel_density must be a positive integer")
    if (
        type(meta.units_per_block) is not int
        or not (1 <= meta.units_per_block <= 512)
        or not _finite_number(meta.units_per_block)
    ):
        raise DSLValidationError("units_per_block must be a positive integer in [1, 512]")
    if not isinstance(meta.dsl_version, str) or meta.dsl_version not in DSL_VERSIONS:
        raise DSLValidationError(f"unsupported dsl_version {meta.dsl_version!r}; expected one of {sorted(DSL_VERSIONS)}")
    if not _finite_number(meta.size_step) or float(meta.size_step) not in ALLOWED_SIZE_STEPS:
        raise DSLValidationError(f"meta.size_step must be one of {sorted(ALLOWED_SIZE_STEPS)}")
    size_step = float(meta.size_step)
    if not isinstance(meta.audit_profile, str) or meta.audit_profile not in AUDIT_PROFILES:
        raise DSLValidationError(f"meta.audit_profile must be one of {sorted(AUDIT_PROFILES)}")

    if meta.archetype == "entity" and meta.target_profile.startswith("java_"):
        raise DSLValidationError("entity cannot target vanilla Java item/block geometry")
    if program.shade or program.ergonomics is not None:
        raise UnsupportedFeature("@shade and @ergonomics require future implemented rendering/grip semantics")
    if not isinstance(program.palettes, list) or not program.palettes:
        raise DSLValidationError("@palette must declare at least one material")

    # Validate @textures
    if not isinstance(program.textures, tuple):
        raise DSLValidationError("textures must be a tuple of TextureDecl objects")
    declared_textures: set[str] = set()
    for tdecl in program.textures:
        if not isinstance(tdecl, TextureDecl):
            raise DSLValidationError("textures must contain TextureDecl objects")
        if not _valid_ident(tdecl.name):
            raise DSLValidationError(f"invalid texture identifier: {tdecl.name!r}")
        if tdecl.name in declared_textures:
            raise DSLValidationError(f"duplicate @textures declaration for {tdecl.name!r}")
        if not isinstance(tdecl.path, str) or not tdecl.path.strip():
            raise DSLValidationError(f"texture {tdecl.name}: path must be a non-empty string")
        if (
            type(tdecl.width) is not int
            or type(tdecl.height) is not int
            or tdecl.width <= 0
            or tdecl.height <= 0
            or not _finite_number(tdecl.width)
            or not _finite_number(tdecl.height)
        ):
            raise DSLValidationError(f"texture {tdecl.name}: width and height must be positive integers")
        declared_textures.add(tdecl.name)

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

    def _clone_roots_for_bone(item: Bone) -> list[str]:
        roots: list[str] = []
        if item.mirror_name:
            if not _valid_ident(item.mirror_name):
                raise DSLValidationError(f"invalid identifier: {item.mirror_name!r}")
            roots.append(item.mirror_name)
        if item.radial is not None:
            if not isinstance(item.radial, RadialSpec):
                raise DSLValidationError(f"bone {item.name}: radial must be a RadialSpec")
            if type(item.radial.count) is not int or not (2 <= item.radial.count <= 64):
                raise DSLValidationError(f"bone {item.name}: radial count must be an integer in [2, 64]")
            if item.radial.axis not in AXES_3D:
                raise DSLValidationError(f"bone {item.name}: radial axis must be x, y, or z")
            for k in range(1, item.radial.count):
                roots.append(f"{item.name}_r{k}")
                if item.mirror_name:
                    roots.append(f"{item.mirror_name}_r{k}")
        return roots

    def descendant_expanded_names(node: Bone, budget_counter: list[int]) -> list[str]:
        out: list[str] = []
        for item in node.items:
            if not isinstance(item, (Box, Locator, Bone)):
                raise UnsupportedFeature(f"unimplemented node type {type(item).__name__}")
            out.append(item.name)
            budget_counter[0] += 1
            if budget_counter[0] > MAX_EXPANDED_MIRROR_NODES:
                raise DSLValidationError(
                    f"expanded mirror/radial hierarchy exceeds {MAX_EXPANDED_MIRROR_NODES} node limit"
                )
            if isinstance(item, Bone):
                sub_names = descendant_expanded_names(item, budget_counter)
                out.extend(sub_names)
                for clone_root in _clone_roots_for_bone(item):
                    out.append(clone_root)
                    budget_counter[0] += 1 + len(sub_names)
                    if budget_counter[0] > MAX_EXPANDED_MIRROR_NODES:
                        raise DSLValidationError(
                            f"expanded mirror/radial hierarchy exceeds {MAX_EXPANDED_MIRROR_NODES} node limit"
                        )
                    out.extend(f"{clone_root}__{sn}" for sn in sub_names)
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
        if not isinstance(bone.mirror_axis, str) or bone.mirror_axis not in MIRROR_AXES:
            raise DSLValidationError(f"bone {bone.name}: mirror_axis must be 'x' or 'z'")
        if not isinstance(bone.items, list) or not bone.items:
            raise DSLValidationError(f"bone {bone.name} must contain at least one primitive")
        if parent_boxes is None:
            if (
                bone.attach is not None
                or bone.mirror_name is not None
                or bone.radial is not None
                or any(bone.rest_euler)
                or bone.joint_limits is not None
            ):
                raise DSLValidationError("root has pivot_local only; no attach, mirror, radial, rest_euler, or joint_limits")
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
            if type(attach.detached) is not bool or type(attach.joint_mode) is not bool:
                raise DSLValidationError("detached and joint_mode must be booleans")
            if not attach.detached and not attach.joint_mode and any(abs(float(x)) > 0.5 for x in attach.offset):
                raise DSLValidationError("connected attach offset must lie in [-0.5,0.5] on every axis")
            if any(bone.pivot_local):
                raise DSLValidationError("child pivot is derived from attach; pivot_local must remain zero")
            if bone.joint_limits is not None:
                _validate_joint_limits(bone.joint_limits, f"{bone.name}.joint_limits")

        clone_roots = _clone_roots_for_bone(bone) if parent_boxes is not None else []
        if clone_roots:
            budget = [len(generated)]
            sub_expanded = descendant_expanded_names(bone, budget)
            needed = len(clone_roots) * (1 + len(sub_expanded))
            if len(generated) + needed > MAX_EXPANDED_MIRROR_NODES:
                raise DSLValidationError(
                    f"expanded mirror/radial hierarchy exceeds {MAX_EXPANDED_MIRROR_NODES} node limit"
                )
            for cr in clone_roots:
                generated.append(cr)
                generated.extend(f"{cr}__{name}" for name in sub_expanded)

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
                _vec(item.rot, f"{item.name}.rot")
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
            if not isinstance(item.size, tuple) or len(item.size) != 3:
                raise DSLValidationError(f"box {item.name}: size must be a 3-tuple")
            if item.plane_normal is None:
                for comp in item.size:
                    _validate_size_component(comp, size_step, item.name)
            else:
                if not isinstance(item.plane_normal, str) or item.plane_normal not in AXES_3D:
                    raise DSLValidationError(f"plane {item.name}: plane_normal must be x, y, or z")
                n_idx = {"x": 0, "y": 1, "z": 2}[item.plane_normal]
                n_val = item.size[n_idx]
                if not _finite_number(n_val) or float(n_val) != 0.0:
                    raise DSLValidationError(
                        f"plane {item.name}: size along normal axis {item.plane_normal!r} must be 0"
                    )
                for idx_c, comp in enumerate(item.size):
                    if idx_c != n_idx:
                        _validate_size_component(comp, size_step, item.name)
            if not isinstance(item.material, str) or item.material not in materials:
                raise DSLValidationError(f"box {item.name}: undeclared material {item.material}")
            if not _finite_number(item.inflate):
                raise DSLValidationError("inflate must be a nonnegative finite number in this starter")
            if float(item.inflate) < 0.0:
                if meta.audit_profile != "authored":
                    raise DSLValidationError("inflate must be a nonnegative finite number in this starter")
                p_idx_inf = {"x": 0, "y": 1, "z": 2}[item.plane_normal] if item.plane_normal is not None else -1
                for idx_inf, sz_inf in enumerate(item.size):
                    if idx_inf != p_idx_inf and 0.5 * float(sz_inf) + float(item.inflate) <= 0.0:
                        raise DSLValidationError(
                            f"box {item.name}: negative inflate {item.inflate} collapses non-planar size {item.size}"
                        )
            _vec(item.rotate, f"{item.name}.rotate")
            if item.pivot is not None:
                _vec(item.pivot, f"{item.name}.pivot")
            if item.uv is not None:
                _vec(item.uv, f"{item.name}.uv", 2)
            if not isinstance(item.uv_faces, tuple):
                raise DSLValidationError(f"box {item.name}: uv_faces must be a tuple of (face, rect) pairs")
            seen_uv_faces: set[str] = set()
            for uv_entry in item.uv_faces:
                if not isinstance(uv_entry, tuple) or len(uv_entry) != 2:
                    raise DSLValidationError(f"box {item.name}: invalid uv_faces entry")
                fname, frect = uv_entry
                if not isinstance(fname, str) or fname not in FACES or fname in seen_uv_faces:
                    raise DSLValidationError(f"box {item.name}: duplicate or invalid uv_faces face {fname!r}")
                seen_uv_faces.add(fname)
                if not isinstance(frect, tuple) or len(frect) not in (4, 5):
                    raise DSLValidationError(f"box {item.name}.uv_faces.{fname}: expected 4 or 5 numbers")
                _vec(frect[:4], f"{item.name}.uv_faces.{fname}", 4)
                if len(frect) == 5:
                    rot_uv = frect[4]
                    if type(rot_uv) is not int or rot_uv not in UV_ROTATIONS:
                        raise DSLValidationError(
                            f"box {item.name}.uv_faces.{fname}: UV rotation must be an integer in {{0, 90, 180, 270}}"
                        )
            if item.texture is not None:
                if not _valid_ident(item.texture):
                    raise DSLValidationError(f"box {item.name}: invalid texture identifier {item.texture!r}")
                if declared_textures and item.texture not in declared_textures:
                    raise DSLValidationError(f"box {item.name}: undeclared texture {item.texture!r}")
            if (
                not isinstance(item.cullface, tuple)
                or any(not isinstance(x, str) for x in item.cullface)
                or len(set(item.cullface)) != len(item.cullface)
                or any(x not in FACES for x in item.cullface)
            ):
                raise DSLValidationError("cullface must contain unique valid face names")
            if (
                not isinstance(item.blend_with, tuple)
                or any(not _valid_ident(x) for x in item.blend_with)
                or len(set(item.blend_with)) != len(item.blend_with)
                or item.name in item.blend_with
            ):
                raise DSLValidationError(f"box {item.name}: blend_with must contain unique valid partner box identifiers")
            if item.explicit_local_center is not None:
                _vec(item.explicit_local_center, f"{item.name}.explicit_local_center")
            if item.component_rest_pos is not None:
                _vec(item.component_rest_pos, f"{item.name}.component_rest_pos")
            if item.component_rest_euler is not None:
                _vec(item.component_rest_euler, f"{item.name}.component_rest_euler")
            if item.place is None:
                if local_boxes and item.explicit_local_center is None:
                    raise DSLValidationError(f"box {item.name}: subsequent same-bone boxes require an explicit place clause")
            else:
                place = item.place
                if not isinstance(place, Place) or (place.base_box is not None and not isinstance(place.base_box, str)):
                    raise DSLValidationError("place must be a Place object with a string or None base_box")
                _vec(place.anchor, f"{item.name}.place.anchor", unit=True)
                _vec(place.align, f"{item.name}.place.align", unit=True)
                _vec(place.offset, f"{item.name}.place.offset")
                if place.base_box is not None:
                    if place.base_box not in local_boxes and place.base_box not in local_locators:
                        raise DSLValidationError(
                            f"box {item.name}: base_box must name a preceding box or locator in the same bone"
                        )
                    if place.base_box in local_locators and any(place.anchor):
                        raise DSLValidationError(
                            f"box {item.name}: place relative to locator {place.base_box!r} requires anchor=[0,0,0]"
                        )
                elif any(place.anchor):
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
    expanded_radial_counts: dict[str, int] = {}
    expanded_blend_specs: list[tuple[str | None, str, tuple[str, ...]]] = []

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
                full_bname = f"{prefix}__{it.name}" if prefix else it.name
                expanded_boxes.add(full_bname)
                if it.blend_with:
                    expanded_blend_specs.append((prefix, full_bname, it.blend_with))
            elif isinstance(it, Bone):
                if isinstance(it.radial, RadialSpec):
                    base_r = f"{prefix}__{it.name}" if prefix else it.name
                    expanded_radial_counts[base_r] = it.radial.count
                _collect_expanded_graph(it, bname, prefix=prefix)
                for cr in _clone_roots_for_bone(it):
                    c_root = f"{prefix}__{cr}" if prefix else cr
                    _collect_expanded_graph(it, bname, prefix=c_root, renamed_root=c_root)

    _collect_expanded_graph(program.root, None)
    for pref, full_bname, bw_targets in expanded_blend_specs:
        for tgt in bw_targets:
            resolved_tgt = f"{pref}__{tgt}" if (pref and f"{pref}__{tgt}" in expanded_boxes) else tgt
            if pref and resolved_tgt == tgt:
                m_src = re.match(r"^(.*)_r(\d+)$", pref)
                if m_src and m_src.group(1) in expanded_radial_counts:
                    r_root = m_src.group(1)
                    k_cur = int(m_src.group(2))
                    r_cnt = expanded_radial_counts[r_root]
                    leaf_r = r_root.split("__")[-1]
                    m_tgt = re.match(rf"^{re.escape(leaf_r)}_r(\d+)__(.+)$", tgt)
                    if m_tgt:
                        k_next = (k_cur + int(m_tgt.group(1))) % r_cnt
                        outer_p = r_root[: -(len(leaf_r) + 2)] if "__" in r_root else None
                        resolved_tgt = (
                            (f"{outer_p}__{m_tgt.group(2)}" if outer_p else m_tgt.group(2))
                            if k_next == 0
                            else f"{r_root}_r{k_next}__{m_tgt.group(2)}"
                        )
                else:
                    leaf_p = pref.split("__")[-1]
                    if tgt.startswith(f"{leaf_p}__"):
                        outer_p = pref[: -(len(leaf_p) + 2)] if "__" in pref else None
                        base_item = tgt[len(leaf_p) + 2 :]
                        cand_inv = f"{outer_p}__{base_item}" if outer_p else base_item
                        if cand_inv in expanded_boxes:
                            resolved_tgt = cand_inv
            if resolved_tgt not in expanded_boxes or resolved_tgt == full_bname:
                raise DSLValidationError(f"box {full_bname}: blend_with references unknown or self box {tgt!r}")
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
        if not kdecl.ik_chains and not kdecl.look_chains and not kdecl.springs and not kdecl.prismatic_joints:
            raise DSLValidationError(
                "@kinematics must declare at least one ik_chain, look_chain, spring, or prismatic_joint"
            )
        for pj in kdecl.prismatic_joints:
            if not isinstance(pj, PrismaticJointDecl):
                raise DSLValidationError("prismatic_joints must contain PrismaticJointDecl objects")
            if pj.bone not in expanded_bone_parent or pj.bone == root_bone_name:
                raise DSLValidationError(f"prismatic_joint {pj.bone!r}: must reference a non-root bone")
            if pj.axis not in AXES_3D:
                raise DSLValidationError(f"prismatic_joint {pj.bone!r}: axis must be x, y, or z")
            if (
                not isinstance(pj.limits, tuple)
                or len(pj.limits) != 2
                or not _finite_number(pj.limits[0])
                or not _finite_number(pj.limits[1])
                or not (-128.0 <= float(pj.limits[0]) <= 0.0 <= float(pj.limits[1]) <= 128.0)
            ):
                raise DSLValidationError(
                    f"prismatic_joint {pj.bone!r}: limits must satisfy -128 <= min <= 0 <= max <= 128"
                )
            if pj.axis_vec is not None:
                _vec(pj.axis_vec, f"prismatic_joint.{pj.bone}.axis_vec")
        for ikc in kdecl.ik_chains:
            if not isinstance(ikc, IKChainDecl):
                raise DSLValidationError("ik_chains must contain IKChainDecl objects")
            claim(ikc.name)
            declared_ik_chains.add(ikc.name)
            chain_bones = ikc.bones
            if (
                not isinstance(chain_bones, tuple)
                or not (2 <= len(chain_bones) <= 16)
                or len(set(chain_bones)) != len(chain_bones)
            ):
                raise DSLValidationError(f"ik_chain {ikc.name}: bones must contain 2..16 unique bones")
            for idx_b, b_curr in enumerate(chain_bones):
                if b_curr not in expanded_bone_parent or b_curr == root_bone_name:
                    raise DSLValidationError(f"ik_chain {ikc.name}: bone {b_curr!r} must be a non-root bone")
                if idx_b > 0 and expanded_bone_parent[b_curr] != chain_bones[idx_b - 1]:
                    raise DSLValidationError(
                        f"ik_chain {ikc.name}: bone {b_curr!r} must be a direct child of {chain_bones[idx_b - 1]!r}"
                    )
            _vec(ikc.effector, f"{ikc.name}.effector")
            if math.sqrt(sum(float(v) * float(v) for v in ikc.effector)) <= 1e-4:
                raise DSLValidationError(f"ik_chain {ikc.name}: effector offset must be nonzero")
            if ikc.hinge_axis not in AXES_3D:
                raise DSLValidationError(f"ik_chain {ikc.name}: hinge_axis must be x, y, or z")
            if ikc.pole is not None:
                _vec(ikc.pole, f"{ikc.name}.pole")
                if math.sqrt(sum(float(v) * float(v) for v in ikc.pole)) <= 1e-6:
                    raise DSLValidationError(f"ik_chain {ikc.name}: pole vector must be nonzero")
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
        for sp in kdecl.springs:
            _validate_spring(sp, expanded_bone_parent, root_bone_name)

    # Validate @behavior
    known_behavior_targets: set[str] = set()
    if program.behavior is not None:
        bdecl = program.behavior
        if not isinstance(bdecl, BehaviorDecl):
            raise DSLValidationError("behavior must be a BehaviorDecl or None")
        if not bdecl.clips and not bdecl.states and not bdecl.springs:
            raise DSLValidationError("@behavior must declare at least one clip, state, or spring")
        if bdecl.clips and meta.target_profile not in ANIMATED_PROFILES:
            raise DSLValidationError(
                f"@behavior clips are not supported on static vanilla profile {meta.target_profile!r}; "
                f"use one of {sorted(ANIMATED_PROFILES)}"
            )
        for sp in bdecl.springs:
            _validate_spring(sp, expanded_bone_parent, root_bone_name)
        for clip in bdecl.clips:
            if not isinstance(clip, ClipDecl):
                raise DSLValidationError("clips must contain ClipDecl objects")
            claim(clip.name)
            known_behavior_targets.add(clip.name)
            if not _finite_number(clip.duration_s) or not (0.05 <= clip.duration_s <= 60.0):
                raise DSLValidationError(f"clip {clip.name}: duration_s must lie in [0.05, 60.0]")
            if type(clip.loop) is not bool:
                raise DSLValidationError(f"clip {clip.name}: loop must be a boolean")
            if (
                not clip.tracks
                and not clip.muscle_drives
                and not clip.ik_drives
                and not clip.keyframes
                and not clip.slide_drives
            ):
                raise DSLValidationError(
                    f"clip {clip.name}: must contain at least one track, slide, key, muscle_drive, or ik_drive"
                )
            for sd in clip.slide_drives:
                if not isinstance(sd, SlideDriveDecl):
                    raise DSLValidationError("slide_drives must contain SlideDriveDecl objects")
                if sd.bone not in expanded_bone_parent or sd.bone == root_bone_name:
                    raise DSLValidationError(f"clip {clip.name}: slide references unknown or root bone {sd.bone!r}")
                if sd.law not in SLIDE_LAWS:
                    raise DSLValidationError(f"clip {clip.name}: unknown slide law {sd.law!r}")
                if sd.axis not in AXES_3D:
                    raise DSLValidationError(f"clip {clip.name}: slide axis must be x, y, or z")
                if not _finite_number(sd.amp_units) or abs(float(sd.amp_units)) > 64.0:
                    raise DSLValidationError(f"clip {clip.name}: slide amp_units must lie in [-64, 64]")
                if not _finite_number(sd.freq_hz) or not (0.0 < float(sd.freq_hz) <= 20.0):
                    raise DSLValidationError(f"clip {clip.name}: slide freq_hz must lie in (0, 20]")
                if not _finite_number(sd.phase_deg) or not _finite_number(sd.bias_units) or abs(float(sd.bias_units)) > 64.0:
                    raise DSLValidationError(f"clip {clip.name}: invalid slide phase_deg or bias_units")
            for tr in clip.tracks:
                if not isinstance(tr, TrackDecl):
                    raise DSLValidationError("tracks must contain TrackDecl objects")
                if tr.bone not in expanded_bone_parent or tr.bone == root_bone_name:
                    raise DSLValidationError(f"clip {clip.name}: track references unknown or root bone {tr.bone!r}")
                if tr.law not in TRACK_LAWS:
                    raise DSLValidationError(f"clip {clip.name}: unknown track law {tr.law!r}")
                if tr.axis not in AXES_3D:
                    raise DSLValidationError(f"clip {clip.name}: track axis must be x, y, or z")
                max_amp_deg = 1440.0 if tr.law == "spin" else 120.0
                if not _finite_number(tr.amp_deg) or abs(tr.amp_deg) > max_amp_deg:
                    raise DSLValidationError(
                        f"clip {clip.name}: track amp_deg must lie in [-{int(max_amp_deg)}, {int(max_amp_deg)}]"
                    )
                if not _finite_number(tr.freq_hz) or not (0.0 < tr.freq_hz <= 20.0):
                    raise DSLValidationError(f"clip {clip.name}: track freq_hz must lie in (0, 20]")
                if not _finite_number(tr.phase_deg) or not _finite_number(tr.bias_deg) or abs(tr.bias_deg) > 90.0:
                    raise DSLValidationError(f"clip {clip.name}: invalid phase_deg or bias_deg")
            for kf in clip.keyframes:
                if not isinstance(kf, KeyframeTrackDecl):
                    raise DSLValidationError("keyframes must contain KeyframeTrackDecl objects")
                if kf.bone not in expanded_bone_parent:
                    raise DSLValidationError(f"clip {clip.name}: keyframe references unknown bone {kf.bone!r}")
                if kf.channel not in KEYFRAME_CHANNELS:
                    raise DSLValidationError(f"clip {clip.name}: unknown keyframe channel {kf.channel!r}")
                if kf.interpolation not in KEYFRAME_INTERPOLATIONS:
                    raise DSLValidationError(f"clip {clip.name}: unknown keyframe interpolation {kf.interpolation!r}")
                if not isinstance(kf.keys, tuple) or len(kf.keys) < 1:
                    raise DSLValidationError(f"clip {clip.name}: keyframe {kf.bone}.{kf.channel} requires >= 1 key")
                prev_t = -1e-9
                for k_entry in kf.keys:
                    if not isinstance(k_entry, tuple) or len(k_entry) != 2 or not _finite_number(k_entry[0]):
                        raise DSLValidationError(f"clip {clip.name}: invalid keyframe timestamp entry")
                    t_val = float(k_entry[0])
                    if t_val < -1e-6 or t_val > clip.duration_s + 1e-4 or t_val < prev_t - 1e-9:
                        raise DSLValidationError(
                            f"clip {clip.name}: keyframe timestamps on {kf.bone}.{kf.channel} must be non-decreasing in [0, duration_s]"
                        )
                    prev_t = t_val
                    _vec(k_entry[1], f"{clip.name}.{kf.bone}.{kf.channel}[{t_val}]")
                    if kf.channel == "scale" and any(float(s) <= 0.0 for s in k_entry[1]):
                        raise DSLValidationError(f"clip {clip.name}: scale keyframes must be positive")
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
            known_behavior_targets.add(st.name)
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

    # Validate @controller
    if program.controller is not None:
        cdecl = program.controller
        if not isinstance(cdecl, ControllerDecl):
            raise DSLValidationError("controller must be a ControllerDecl or None")
        if not cdecl.transitions:
            raise DSLValidationError("@controller must declare at least one state transition")
        if cdecl.initial is not None:
            if not _valid_ident(cdecl.initial):
                raise DSLValidationError(f"@controller: invalid initial state {cdecl.initial!r}")
            if known_behavior_targets and cdecl.initial not in known_behavior_targets:
                raise DSLValidationError(f"@controller: initial state {cdecl.initial!r} not found in @behavior")
        for tr_c in cdecl.transitions:
            if not isinstance(tr_c, ControllerTransitionDecl):
                raise DSLValidationError("@controller transitions must be ControllerTransitionDecl objects")
            if not _valid_ident(tr_c.from_state) or not _valid_ident(tr_c.to_state) or not _valid_ident(tr_c.parameter):
                raise DSLValidationError("@controller transition requires valid from_state, to_state, and parameter identifiers")
            if known_behavior_targets:
                if tr_c.from_state not in known_behavior_targets or tr_c.to_state not in known_behavior_targets:
                    raise DSLValidationError(
                        f"@controller transition {tr_c.from_state!r} -> {tr_c.to_state!r} references undeclared @behavior clip/state"
                    )
            if tr_c.operator not in CONTROLLER_OPS:
                raise DSLValidationError(f"@controller: unsupported comparison operator {tr_c.operator!r}")
            if not _finite_number(tr_c.threshold):
                raise DSLValidationError("@controller: threshold must be finite")
            if not _finite_number(tr_c.blend_s) or not (0.0 <= float(tr_c.blend_s) <= 10.0):
                raise DSLValidationError("@controller: blend duration must lie in [0.0, 10.0]")

    return program
