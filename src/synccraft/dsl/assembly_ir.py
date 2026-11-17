"""Typed Assembly, Component & Top-Down Layout Intermediate Representation (ADR 0008 / Block-DSL v4.0).

Separates four concerns cleanly above SyncCraft's shared ``Program`` / ``Scene`` engine:
1. ``@layout``: Acyclic top-down master parameter DAG + non-mutating ``budget`` / ``envelope`` contracts.
2. ``@component``: Reusable parametric submodels with local namespaces, ``material_slots``,
   oriented 6-DOF ``frame`` interfaces, optional internal sub-rigs, and ``uv_policy``.
3. ``@assembly``: Organizational ``subassembly`` hierarchy, ``instance`` occurrences (with
   complete ``mirror_x`` reflection), explicit mechanical ``connect`` mounts
   (``fixed`` | ``revolute`` | ``prismatic``), and optional ``bind`` motion overrides.
4. ``SourceMap``: Persistent ``stable_id`` <-> ``readable_path`` <-> ``runtime_bone_name`` /
   ``exported_boxes`` mapping with explicit ``assembly_provenance`` (``"authored"`` vs
   ``"inferred_subtree"``).

Zero Python ``eval`` / ``exec`` is used anywhere in expression or DAG evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import math
import re
from typing import Any

import numpy as np

from .ast_nodes import (
    AXES_3D,
    FACES,
    LOCATOR_ROLES,
    MIRROR_AXES,
    BehaviorDecl,
    BehaviorStateDecl,
    Bone,
    Box,
    ClipDecl,
    ControllerDecl,
    DSLValidationError,
    IKChainDecl,
    Joint,
    JointLimits,
    KeyframeTrackDecl,
    KinematicsDecl,
    Locator,
    LookChainDecl,
    MaterialDecl,
    Meta,
    MuscleDecl,
    Palette,
    Place,
    Plane,
    PrismaticJointDecl,
    Program,
    SlideDriveDecl,
    SpringDecl,
    StampDecl,
    StyleDecl,
    TendonCoupling,
    TextureDecl,
    TrackDecl,
    UVRect,
    Vec3,
    _is_continuous_limit_pair,
    EdgeCutoutDecl,
    format_edge_cutout,
    validate_program,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synccraft.kinematics.se3_chain import Scene


def euler_xyz_matrix(deg: Vec3) -> np.ndarray:
    from synccraft.kinematics.se3_chain import euler_xyz_matrix as _fn

    return _fn(deg)


def _recover_euler_xyz_degrees(rot: np.ndarray) -> Vec3:
    from synccraft.kinematics.se3_chain import _recover_euler_xyz_degrees as _fn

    return _fn(rot)


def _angle_sign_from_reflection(sign: np.ndarray) -> np.ndarray:
    from synccraft.kinematics.se3_chain import _angle_sign_from_reflection as _fn

    return _fn(sign)


def compile_geometry(program: Program) -> Scene:
    from synccraft.kinematics.se3_chain import compile_geometry as _fn

    return _fn(program)

CONNECTION_TYPES = frozenset({"fixed", "revolute", "prismatic"})
MATE_MODES = frozenset({"align", "oppose"})
UV_POLICIES = frozenset({"unique", "shared"})
ASSEMBLY_PROVENANCES = frozenset({"authored", "inferred_subtree"})
SAFE_EXPR_FUNCS = frozenset({"min", "max", "abs", "round"})

_EXPR_TOKEN_RE = re.compile(
    r"(?P<SPACE>\s+)|"
    r"(?P<NUMBER>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<OP>[()+\-*/,])"
)


def _tokenize_expr(expr: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _EXPR_TOKEN_RE.match(expr, pos)
        if not m:
            raise DSLValidationError(f"Invalid character {expr[pos]!r} in numeric expression {expr!r}")
        kind = str(m.lastgroup)
        if kind != "SPACE":
            tokens.append((kind, m.group(0)))
        pos = m.end()
    return tokens


def extract_expr_vars(expr: str | int | float) -> set[str]:
    """Return variable identifiers referenced in a symbolic numeric expression."""
    if isinstance(expr, (int, float)):
        return set()
    toks = _tokenize_expr(str(expr))
    out: set[str] = set()
    for idx, (kind, val) in enumerate(toks):
        if kind == "IDENT":
            is_call = idx + 1 < len(toks) and toks[idx + 1] == ("OP", "(")
            if is_call and val in SAFE_EXPR_FUNCS:
                continue
            out.add(val)
    return out


def eval_symbolic_expr(expr: str | int | float, env: Mapping[str, int | float]) -> int | float:
    """Evaluate a bounded arithmetic expression against ``env`` without ``eval``."""
    if type(expr) in (int, float):
        if not math.isfinite(float(expr)):
            raise DSLValidationError(f"Non-finite numeric value {expr!r}")
        return expr  # type: ignore[return-value]
    toks = _tokenize_expr(str(expr))
    if not toks:
        raise DSLValidationError("Empty numeric expression")
    idx = 0

    def peek() -> tuple[str, str] | None:
        return toks[idx] if idx < len(toks) else None

    def take() -> tuple[str, str]:
        nonlocal idx
        t = toks[idx]
        idx += 1
        return t

    def parse_add(depth: int) -> int | float:
        if depth > 32:
            raise DSLValidationError("Expression nesting exceeds 32")
        val = parse_mul(depth + 1)
        while (p := peek()) is not None and p[0] == "OP" and p[1] in ("+", "-"):
            op = take()[1]
            rhs = parse_mul(depth + 1)
            val = val + rhs if op == "+" else val - rhs
        return val

    def parse_mul(depth: int) -> int | float:
        val = parse_unary(depth + 1)
        while (p := peek()) is not None and p[0] == "OP" and p[1] in ("*", "/"):
            op = take()[1]
            rhs = parse_unary(depth + 1)
            if op == "*":
                val = val * rhs
            else:
                if abs(float(rhs)) <= 1e-15:
                    raise DSLValidationError(f"Division by zero in expression {expr!r}")
                res_f = float(val) / float(rhs)
                val = int(round(res_f)) if (type(val) is int and type(rhs) is int and res_f.is_integer()) else res_f
        return val

    def parse_unary(depth: int) -> int | float:
        p = peek()
        if p is not None and p[0] == "OP" and p[1] in ("+", "-"):
            op = take()[1]
            v = parse_unary(depth + 1)
            return v if op == "+" else -v
        return parse_atom(depth + 1)

    def parse_atom(depth: int) -> int | float:
        p = peek()
        if p is None:
            raise DSLValidationError(f"Unexpected end of expression {expr!r}")
        kind, raw = take()
        if kind == "NUMBER":
            if "." in raw or "e" in raw or "E" in raw:
                fv = float(raw)
                return fv
            return int(raw)
        if kind == "IDENT":
            nxt = peek()
            if nxt is not None and nxt == ("OP", "("):
                if raw not in SAFE_EXPR_FUNCS:
                    raise DSLValidationError(f"Unsupported function {raw!r} in expression {expr!r}")
                take()  # '('
                args: list[int | float] = [parse_add(depth + 1)]
                while (c := peek()) is not None and c == ("OP", ","):
                    take()
                    args.append(parse_add(depth + 1))
                end_p = peek()
                if end_p != ("OP", ")"):
                    raise DSLValidationError(f"Expected ')' in function call {raw}(...) in {expr!r}")
                take()
                if raw == "min":
                    if len(args) < 2:
                        raise DSLValidationError("min(...) requires at least 2 arguments")
                    return min(args)
                if raw == "max":
                    if len(args) < 2:
                        raise DSLValidationError("max(...) requires at least 2 arguments")
                    return max(args)
                if raw == "abs":
                    if len(args) != 1:
                        raise DSLValidationError("abs(...) requires 1 argument")
                    return abs(args[0])
                if raw == "round":
                    if len(args) != 1:
                        raise DSLValidationError("round(...) requires 1 argument")
                    return int(round(float(args[0])))
            if raw not in env:
                raise DSLValidationError(f"Unknown parameter or constant {raw!r} in expression {expr!r}")
            return env[raw]
        if kind == "OP" and raw == "(":
            inner = parse_add(depth + 1)
            if peek() != ("OP", ")"):
                raise DSLValidationError(f"Unclosed '(' in expression {expr!r}")
            take()
            return inner
        raise DSLValidationError(f"Unexpected token {raw!r} in expression {expr!r}")

    out = parse_add(0)
    if idx != len(toks):
        raise DSLValidationError(f"Trailing token {toks[idx][1]!r} in expression {expr!r}")
    if not math.isfinite(float(out)):
        raise DSLValidationError(f"Non-finite result in expression {expr!r}")
    return out


def _eval_vec3_expr(vec_exprs: tuple[str | int | float, ...], env: Mapping[str, int | float]) -> Vec3:
    if len(vec_exprs) != 3:
        raise DSLValidationError(f"Expected 3-vector expression, got {vec_exprs!r}")
    return (
        float(eval_symbolic_expr(vec_exprs[0], env)),
        float(eval_symbolic_expr(vec_exprs[1], env)),
        float(eval_symbolic_expr(vec_exprs[2], env)),
    )


def _quantize_size_value(val: int | float, size_step: float) -> int | float:
    f = float(val)
    if abs(size_step - 1.0) <= 1e-12:
        r = round(f)
        if abs(f - r) <= 1e-7:
            return int(r)
        return max(1, int(round(f)))
    steps = max(1, int(round(f / size_step)))
    q = round(steps * size_step, 6)
    return int(round(q)) if abs(q - round(q)) <= 1e-9 else q


# ---------------------------------------------------------------------------
# Typed Assembly IR Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutParamDecl:
    name: str
    default_expr: str
    min_val: float | None = None
    max_val: float | None = None


@dataclass(frozen=True)
class LayoutLetDecl:
    name: str
    expr: str


@dataclass(frozen=True)
class BudgetSpec:
    max_boxes: int | None = None
    max_bones: int | None = None
    max_atlas_px: int | None = None


@dataclass(frozen=True)
class EnvelopeSpec:
    max_span: Vec3 | None = None
    min_ground_y: float | None = None
    grounded: bool = False
    ground_y: float = 0.0
    ground_tol: float = 0.15


@dataclass(frozen=True)
class LayoutDecl:
    params: tuple[LayoutParamDecl, ...] = ()
    lets: tuple[LayoutLetDecl, ...] = ()
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    envelope: EnvelopeSpec = field(default_factory=EnvelopeSpec)


@dataclass(frozen=True)
class ComponentParamDecl:
    name: str
    default_expr: str
    min_val: float | None = None
    max_val: float | None = None


@dataclass(frozen=True)
class AttachmentFrameDecl:
    name: str
    pos: tuple[str, str, str] = ("0", "0", "0")
    rot: tuple[str, str, str] = ("0", "0", "0")
    on_box: str | None = None
    anchor: tuple[str, str, str] = ("0", "0", "0")
    offset: tuple[str, str, str] = ("0", "0", "0")
    role: str = "socket"
    export_locator: bool = False


@dataclass(frozen=True)
class ComponentControlDecl:
    name: str
    target: str
    channel: str = "rotation"
    axis: str = "x"
    limits: tuple[float, float] = (-45.0, 45.0)


@dataclass(frozen=True)
class ComponentBoxTemplate:
    name: str
    is_plane: bool
    size_exprs: tuple[str, ...]
    material_slot: str
    place_base: str | None = None
    place_anchor: tuple[str, str, str] = ("0", "0", "0")
    place_align: tuple[str, str, str] = ("0", "0", "0")
    place_offset: tuple[str, str, str] = ("0", "0", "0")
    has_place: bool = False
    inflate_expr: str = "0"
    rotate_exprs: tuple[str, str, str] = ("0", "0", "0")
    pivot_exprs: tuple[str, str, str] | None = None
    cullface: tuple[str, ...] = ()
    blend_with: tuple[str, ...] = ()
    plane_normal: str | None = None
    uv: tuple[float, float] | None = None
    uv_faces: tuple[tuple[str, UVRect], ...] = ()
    texture: str | None = None
    edge: EdgeCutoutDecl | None = None
    layer: int = 0
    prism_group: str | None = None


@dataclass(frozen=True)
class ComponentLocatorTemplate:
    name: str
    offset_exprs: tuple[str, str, str] = ("0", "0", "0")
    rot_exprs: tuple[str, str, str] = ("0", "0", "0")
    role: str = "socket"


@dataclass(frozen=True)
class ComponentBoneTemplate:
    name: str
    attach_parent: str
    attach_anchor: tuple[str, str, str] = ("0", "0", "0")
    attach_align: tuple[str, str, str] = ("0", "0", "0")
    attach_offset: tuple[str, str, str] = ("0", "0", "0")
    attach_joint_mode: bool = False
    attach_detached: bool = False
    attach_seat: float = 0.0
    rest_euler_exprs: tuple[str, str, str] = ("0", "0", "0")
    joint_limits: JointLimits | None = None
    boxes: tuple[ComponentBoxTemplate, ...] = ()
    locators: tuple[ComponentLocatorTemplate, ...] = ()
    children: tuple[ComponentBoneTemplate, ...] = ()


@dataclass(frozen=True)
class ComponentDef:
    name: str
    params: tuple[ComponentParamDecl, ...] = ()
    material_slots: tuple[tuple[str, str | None], ...] = ()
    uv_policy: str = "unique"
    lets: tuple[LayoutLetDecl, ...] = ()
    root_boxes: tuple[ComponentBoxTemplate, ...] = ()
    root_locators: tuple[ComponentLocatorTemplate, ...] = ()
    child_bones: tuple[ComponentBoneTemplate, ...] = ()
    frames: tuple[AttachmentFrameDecl, ...] = ()
    controls: tuple[ComponentControlDecl, ...] = ()


@dataclass(frozen=True)
class ComponentInstanceDecl:
    name: str
    component_name: str
    stable_id: str | None = None
    param_exprs: tuple[tuple[str, str], ...] = ()
    material_bindings: tuple[tuple[str, str], ...] = ()
    uv_policy: str | None = None
    mirror_name: str | None = None
    mirror_axis: str = "x"
    subassembly_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectionDecl:
    child_instance: str
    child_frame: str
    parent_instance: str
    parent_frame: str
    conn_type: str = "fixed"
    axis: str = "x"
    limits: tuple[float, float] | None = None
    mode: str = "align"
    offset_pos: tuple[str, str, str] = ("0", "0", "0")
    offset_rot: tuple[str, str, str] = ("0", "0", "0")
    sculpted_blend: bool = False
    preserve_bone: bool = False


@dataclass(frozen=True)
class BindDecl:
    instance_name: str
    target_bone: str


@dataclass(frozen=True)
class SubassemblyDecl:
    name: str
    instances: tuple[ComponentInstanceDecl, ...] = ()
    subassemblies: tuple[SubassemblyDecl, ...] = ()
    connections: tuple[ConnectionDecl, ...] = ()
    binds: tuple[BindDecl, ...] = ()


@dataclass(frozen=True)
class AssemblyDecl:
    name: str
    root_pivot: tuple[str, str, str] = ("0", "0", "0")
    root_instance: str | None = None
    instances: tuple[ComponentInstanceDecl, ...] = ()
    subassemblies: tuple[SubassemblyDecl, ...] = ()
    connections: tuple[ConnectionDecl, ...] = ()
    binds: tuple[BindDecl, ...] = ()


@dataclass(frozen=True)
class AssemblySpec:
    layout: LayoutDecl | None = None
    components: tuple[ComponentDef, ...] = ()
    assembly: AssemblyDecl | None = None
    assembly_provenance: str = "authored"


@dataclass(frozen=True)
class SourceMapEntry:
    stable_id: str
    instance_name: str
    readable_path: str
    component_name: str
    mirrored: bool
    bound_params: tuple[tuple[str, float | int], ...]
    bound_materials: tuple[tuple[str, str], ...]
    uv_policy: str
    runtime_bone_name: str
    exported_boxes: tuple[str, ...]
    exported_locators: tuple[str, ...]
    exported_sub_bones: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "instance_name": self.instance_name,
            "readable_path": self.readable_path,
            "component_name": self.component_name,
            "mirrored": self.mirrored,
            "bound_params": {k: v for k, v in self.bound_params},
            "bound_materials": {k: v for k, v in self.bound_materials},
            "uv_policy": self.uv_policy,
            "runtime_bone_name": self.runtime_bone_name,
            "exported_boxes": list(self.exported_boxes),
            "exported_locators": list(self.exported_locators),
            "exported_sub_bones": {k: v for k, v in self.exported_sub_bones},
        }


@dataclass(frozen=True)
class SourceMap:
    assembly_name: str
    assembly_provenance: str
    layout_values: tuple[tuple[str, float | int], ...]
    entries: tuple[SourceMapEntry, ...]
    dotted_bone_map: tuple[tuple[str, str], ...]
    dotted_box_map: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "synccraft.source-map/4.0",
            "assembly_name": self.assembly_name,
            "assembly_provenance": self.assembly_provenance,
            "layout_values": {k: v for k, v in self.layout_values},
            "instances": [e.to_dict() for e in self.entries],
            "dotted_bone_map": {k: v for k, v in self.dotted_bone_map},
            "dotted_box_map": {k: v for k, v in self.dotted_box_map},
        }


# ---------------------------------------------------------------------------
# Deterministic Layout DAG Evaluation
# ---------------------------------------------------------------------------


def evaluate_layout_dag(
    layout: LayoutDecl | None,
    overrides: Mapping[str, int | float] | None = None,
) -> dict[str, int | float]:
    """Evaluate ``@layout`` parameters and ``let`` expressions in topological DAG order."""
    if layout is None:
        if overrides:
            raise DSLValidationError(f"Cannot apply layout overrides {sorted(overrides)} when @layout is absent")
        return {}

    overrides_map = dict(overrides or {})
    nodes: dict[str, tuple[str, str, float | None, float | None]] = {}
    for p in layout.params:
        if p.name in nodes:
            raise DSLValidationError(f"Duplicate @layout identifier {p.name!r}")
        nodes[p.name] = ("param", p.default_expr, p.min_val, p.max_val)
    for ldecl in layout.lets:
        if ldecl.name in nodes:
            raise DSLValidationError(f"Duplicate @layout identifier {ldecl.name!r}")
        if ldecl.name in overrides_map:
            raise DSLValidationError(f"Cannot override derived @layout let {ldecl.name!r}; override a param instead")
        nodes[ldecl.name] = ("let", ldecl.expr, None, None)

    for ov_k in overrides_map:
        if ov_k not in nodes:
            raise DSLValidationError(f"Unknown @layout parameter override {ov_k!r}")

    deps: dict[str, set[str]] = {}
    for name, (kind, expr_s, _, _) in nodes.items():
        if kind == "param" and name in overrides_map:
            deps[name] = set()
        else:
            referenced = extract_expr_vars(expr_s)
            unknown = referenced - set(nodes.keys())
            if unknown:
                raise DSLValidationError(f"@layout {name!r} references undeclared variable(s): {sorted(unknown)}")
            deps[name] = referenced

    # 3-color DFS topological sort with exact cycle reporting
    state: dict[str, int] = {k: 0 for k in nodes}  # 0=unvisited, 1=visiting, 2=done
    order: list[str] = []
    stack_trace: list[str] = []

    def dfs(u: str) -> None:
        state[u] = 1
        stack_trace.append(u)
        for v in sorted(deps[u]):
            if state[v] == 1:
                cyc_idx = stack_trace.index(v)
                cycle_str = " -> ".join([*stack_trace[cyc_idx:], v])
                raise DSLValidationError(f"Cyclic @layout dependency detected: {cycle_str}")
            if state[v] == 0:
                dfs(v)
        stack_trace.pop()
        state[u] = 2
        order.append(u)

    for k in nodes:
        if state[k] == 0:
            dfs(k)

    env: dict[str, int | float] = {}
    for k in order:
        kind, expr_s, lo, hi = nodes[k]
        val = overrides_map[k] if (kind == "param" and k in overrides_map) else eval_symbolic_expr(expr_s, env)
        if lo is not None and float(val) < float(lo) - 1e-9:
            raise DSLValidationError(f"@layout param {k!r}={val} violates min bound {lo}")
        if hi is not None and float(val) > float(hi) + 1e-9:
            raise DSLValidationError(f"@layout param {k!r}={val} violates max bound {hi}")
        env[k] = val
    return env


# ---------------------------------------------------------------------------
# Component Evaluation & Assembly SE(3) Lowering
# ---------------------------------------------------------------------------


def _component_uv_faces(
    faces: tuple[tuple[str, UVRect], ...], sign: np.ndarray,
) -> tuple[tuple[str, UVRect], ...]:
    """Preserve painted UVs, using the direct-rig mirror convention exactly once."""
    from synccraft.kinematics.se3_chain import _stored_uv_faces

    return tuple(_stored_uv_faces(dict(faces), sign).items())


@dataclass
class _EvaluatedBox:
    local_name: str
    export_name: str
    size: np.ndarray
    material: str
    center_comp: np.ndarray
    rot_comp: np.ndarray
    euler_comp: Vec3
    inflate: float
    cullface: tuple[str, ...]
    blend_with_local: tuple[str, ...]
    plane_normal: str | None
    uv_share_key: str | None
    uv: tuple[float, float] | None
    uv_faces: tuple[tuple[str, UVRect], ...]
    texture: str | None
    place_base_local: str | None = None
    edge: EdgeCutoutDecl | None = None
    layer: int = 0
    prism_group: str | None = None


@dataclass
class _EvaluatedFrame:
    name: str
    pos_comp: np.ndarray
    rot_comp: np.ndarray
    on_box_local: str | None
    role: str
    export_locator: bool


def _flatten_assembly_tree(
    asm: AssemblyDecl,
) -> tuple[list[ComponentInstanceDecl], list[ConnectionDecl], list[BindDecl]]:
    instances: list[ComponentInstanceDecl] = []
    conns: list[ConnectionDecl] = list(asm.connections)
    binds: list[BindDecl] = list(asm.binds)

    for inst in asm.instances:
        instances.append(replace(inst, subassembly_path=()))

    def _walk_sub(sub: SubassemblyDecl, prefix_path: tuple[str, ...]) -> None:
        cur_path = (*prefix_path, sub.name)
        for inst in sub.instances:
            instances.append(replace(inst, subassembly_path=cur_path))
        conns.extend(sub.connections)
        binds.extend(sub.binds)
        for ch_sub in sub.subassemblies:
            _walk_sub(ch_sub, cur_path)

    for sub in asm.subassemblies:
        _walk_sub(sub, ())

    return instances, conns, binds


def _swap_lr_frame_name(frame_name: str) -> str | None:
    """Infer bilateral counterpart socket name (e.g. ``shoulder_l`` -> ``shoulder_r``)."""
    pairs = (
        ("_l", "_r"),
        ("_left", "_right"),
        ("left_", "right_"),
        ("_fl", "_fr"),
        ("_hl", "_hr"),
        ("_bl", "_br"),
    )
    for left_tok, right_tok in pairs:
        if frame_name.endswith(left_tok):
            return frame_name[: -len(left_tok)] + right_tok
        if frame_name.endswith(right_tok):
            return frame_name[: -len(right_tok)] + left_tok
        if frame_name.startswith(left_tok):
            return right_tok + frame_name[len(left_tok) :]
        if frame_name.startswith(right_tok):
            return left_tok + frame_name[len(right_tok) :]
    return None


def _evaluate_component_instance(
    comp: ComponentDef,
    inst: ComponentInstanceDecl,
    *,
    is_mirrored: bool,
    mirror_sign: np.ndarray,
    layout_env: Mapping[str, int | float],
    palette_names: set[str],
    size_step: float,
) -> tuple[
    dict[str, int | float],
    dict[str, str],
    str,
    list[_EvaluatedBox],
    list[tuple[str, np.ndarray, np.ndarray, str]],
    dict[str, _EvaluatedFrame],
]:
    """Evaluate one ``@component`` occurrence in its local frame (applying ``mirror_sign`` when reflected)."""
    angle_sign = _angle_sign_from_reflection(mirror_sign)

    # 1. Bind component parameters
    provided_params = {k: eval_symbolic_expr(v_expr, layout_env) for k, v_expr in inst.param_exprs}
    declared_param_names = {p.name for p in comp.params}
    unknown_p = set(provided_params) - declared_param_names
    if unknown_p:
        raise DSLValidationError(
            f"instance {inst.name!r} passes unknown parameter(s) {sorted(unknown_p)} to @component {comp.name!r}"
        )

    comp_env: dict[str, int | float] = dict(layout_env)
    bound_params_clean: dict[str, int | float] = {}
    for p in comp.params:
        val = provided_params[p.name] if p.name in provided_params else eval_symbolic_expr(p.default_expr, comp_env)
        if p.min_val is not None and float(val) < float(p.min_val) - 1e-9:
            raise DSLValidationError(
                f"instance {inst.name!r}: param {p.name!r}={val} < min {p.min_val} on {comp.name!r}"
            )
        if p.max_val is not None and float(val) > float(p.max_val) + 1e-9:
            raise DSLValidationError(
                f"instance {inst.name!r}: param {p.name!r}={val} > max {p.max_val} on {comp.name!r}"
            )
        comp_env[p.name] = val
        bound_params_clean[p.name] = val

    for ldecl in comp.lets:
        comp_env[ldecl.name] = eval_symbolic_expr(ldecl.expr, comp_env)

    # 2. Bind material_slots
    inst_mats = dict(inst.material_bindings)
    slot_map: dict[str, str] = {}
    if comp.material_slots:
        declared_slots = {s for s, _ in comp.material_slots}
        extra_m = set(inst_mats) - declared_slots
        if extra_m:
            raise DSLValidationError(
                f"instance {inst.name!r} binds undeclared material slot(s) {sorted(extra_m)} on {comp.name!r}"
            )
        for s_name, s_default in comp.material_slots:
            chosen = inst_mats.get(s_name, s_default)
            if chosen is None:
                raise DSLValidationError(
                    f"instance {inst.name!r} must bind material slot {s_name!r} for @component {comp.name!r}"
                )
            if chosen not in palette_names:
                raise DSLValidationError(
                    f"instance {inst.name!r}: bound material {chosen!r} for slot {s_name!r} is not in @palette"
                )
            slot_map[s_name] = chosen
    elif inst_mats:
        raise DSLValidationError(
            f"instance {inst.name!r} specifies materials, but @component {comp.name!r} declares no material_slots"
        )

    eff_uv_policy = inst.uv_policy or comp.uv_policy
    if eff_uv_policy not in UV_POLICIES:
        raise DSLValidationError(f"Unsupported uv_policy {eff_uv_policy!r}; expected 'unique' or 'shared'")

    param_sig = ",".join(f"{k}={bound_params_clean[k]}" for k in sorted(bound_params_clean))
    mat_sig = ",".join(f"{k}={slot_map[k]}" for k in sorted(slot_map))

    # 3. Evaluate root boxes & planes in component coordinates
    eval_boxes: list[_EvaluatedBox] = []
    by_local_box: dict[str, _EvaluatedBox] = {}
    by_local_loc: dict[str, np.ndarray] = {}

    for b_idx, bt in enumerate(comp.root_boxes):
        mat_resolved = slot_map.get(bt.material_slot, bt.material_slot)
        if mat_resolved not in palette_names:
            raise DSLValidationError(
                f"@component {comp.name!r} box {bt.name!r}: material {mat_resolved!r} is neither a bound slot nor @palette"
            )
        if bt.is_plane:
            if len(bt.size_exprs) != 2 or bt.plane_normal not in AXES_3D:
                raise DSLValidationError(f"@component {comp.name!r} plane {bt.name!r}: invalid 2D size or normal")
            w_q = _quantize_size_value(eval_symbolic_expr(bt.size_exprs[0], comp_env), size_step)
            h_q = _quantize_size_value(eval_symbolic_expr(bt.size_exprs[1], comp_env), size_step)
            z0: int | float = 0 if (type(w_q) is int and type(h_q) is int) else 0.0
            if bt.plane_normal == "x":
                sz_vec = np.array((float(z0), float(w_q), float(h_q)), dtype=np.float64)
            elif bt.plane_normal == "y":
                sz_vec = np.array((float(w_q), float(z0), float(h_q)), dtype=np.float64)
            else:
                sz_vec = np.array((float(w_q), float(h_q), float(z0)), dtype=np.float64)
        else:
            sz_vals = [_quantize_size_value(eval_symbolic_expr(se, comp_env), size_step) for se in bt.size_exprs]
            sz_vec = np.array([float(v) for v in sz_vals], dtype=np.float64)

        if not bt.has_place:
            if b_idx > 0:
                raise DSLValidationError(
                    f"@component {comp.name!r} box {bt.name!r}: 2nd+ boxes require place(...)"
                )
            unrot_ctr = np.zeros(3, dtype=np.float64)
        else:
            anc = np.asarray(_eval_vec3_expr(bt.place_anchor, comp_env), dtype=np.float64) * mirror_sign
            alg = np.asarray(_eval_vec3_expr(bt.place_align, comp_env), dtype=np.float64) * mirror_sign
            off = np.asarray(_eval_vec3_expr(bt.place_offset, comp_env), dtype=np.float64) * mirror_sign
            if bt.place_base is not None:
                if bt.place_base not in by_local_box:
                    raise DSLValidationError(
                        f"@component {comp.name!r} box {bt.name!r}: unknown earlier base_box {bt.place_base!r}"
                    )
                ref_b = by_local_box[bt.place_base]
                ref_pt = ref_b.center_comp + ref_b.rot_comp @ (0.5 * ref_b.size * anc)
            else:
                ref_pt = np.zeros(3, dtype=np.float64)
            unrot_ctr = ref_pt - 0.5 * sz_vec * alg + off

        eul_raw = np.asarray(_eval_vec3_expr(bt.rotate_exprs, comp_env), dtype=np.float64) * angle_sign
        r_loc = euler_xyz_matrix(eul_raw)
        if bt.pivot_exprs is not None:
            piv_c = np.asarray(_eval_vec3_expr(bt.pivot_exprs, comp_env), dtype=np.float64) * mirror_sign
            ctr_c = piv_c + r_loc @ (unrot_ctr - piv_c)
        else:
            ctr_c = unrot_ctr

        inf_v = float(eval_symbolic_expr(bt.inflate_expr, comp_env))
        cull = bt.cullface
        if is_mirrored and cull:
            c_swapped: list[str] = []
            for fc in cull:
                if float(mirror_sign[0]) < 0.0 and fc in ("east", "west"):
                    c_swapped.append("west" if fc == "east" else "east")
                elif float(mirror_sign[2]) < 0.0 and fc in ("north", "south"):
                    c_swapped.append("north" if fc == "south" else "south")
                else:
                    c_swapped.append(fc)
            cull = tuple(c_swapped)

        share_key = (
            f"{comp.name}::{param_sig}::{mat_sig}::{bt.name}"
            if (eff_uv_policy == "shared" and not is_mirrored)
            else None
        )
        eb = _EvaluatedBox(
            local_name=bt.name,
            export_name=f"{inst.name}__{bt.name}",
            size=sz_vec,
            material=mat_resolved,
            center_comp=ctr_c,
            rot_comp=r_loc,
            euler_comp=(float(eul_raw[0]), float(eul_raw[1]), float(eul_raw[2])),
            inflate=inf_v,
            cullface=cull,
            blend_with_local=bt.blend_with,
            plane_normal=bt.plane_normal,
            uv_share_key=share_key,
            uv=bt.uv,
            uv_faces=_component_uv_faces(bt.uv_faces, mirror_sign),
            texture=bt.texture,
            place_base_local=bt.place_base if bt.has_place else None,
            edge=bt.edge,
            layer=bt.layer,
            prism_group=f"{inst.name}__{bt.prism_group}" if bt.prism_group else None,
        )
        eval_boxes.append(eb)
        by_local_box[bt.name] = eb

    # 4. Evaluate component root locators
    eval_locs: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for lt in comp.root_locators:
        off_l = np.asarray(_eval_vec3_expr(lt.offset_exprs, comp_env), dtype=np.float64) * mirror_sign
        rot_l_deg = np.asarray(_eval_vec3_expr(lt.rot_exprs, comp_env), dtype=np.float64) * angle_sign
        r_l = euler_xyz_matrix(rot_l_deg)
        eval_locs.append((f"{inst.name}__{lt.name}", off_l, r_l, lt.role))
        by_local_loc[lt.name] = off_l

    # 5. Evaluate public oriented attachment frames
    eval_frames: dict[str, _EvaluatedFrame] = {}
    for fr in comp.frames:
        if fr.role not in LOCATOR_ROLES:
            raise DSLValidationError(f"@component {comp.name!r} frame {fr.name!r}: invalid role {fr.role!r}")
        rot_f_deg = np.asarray(_eval_vec3_expr(fr.rot, comp_env), dtype=np.float64) * angle_sign
        r_f = euler_xyz_matrix(rot_f_deg)
        if fr.on_box is not None:
            if fr.on_box not in by_local_box:
                raise DSLValidationError(
                    f"@component {comp.name!r} frame {fr.name!r}: unknown local box {fr.on_box!r}"
                )
            ref_bx = by_local_box[fr.on_box]
            anc_f = np.asarray(_eval_vec3_expr(fr.anchor, comp_env), dtype=np.float64) * mirror_sign
            off_f = np.asarray(_eval_vec3_expr(fr.offset, comp_env), dtype=np.float64) * mirror_sign
            pos_f = ref_bx.center_comp + ref_bx.rot_comp @ (0.5 * ref_bx.size * anc_f) + off_f
        else:
            pos_f = np.asarray(_eval_vec3_expr(fr.pos, comp_env), dtype=np.float64) * mirror_sign
        eval_frames[fr.name] = _EvaluatedFrame(
            name=fr.name,
            pos_comp=pos_f,
            rot_comp=r_f,
            on_box_local=fr.on_box,
            role=fr.role,
            export_locator=fr.export_locator or fr.role != "socket",
        )

    return bound_params_clean, slot_map, eff_uv_policy, eval_boxes, eval_locs, eval_frames


def _lower_component_sub_bone(
    btmpl: ComponentBoneTemplate,
    inst_name: str,
    *,
    is_mirrored: bool,
    mirror_sign: np.ndarray,
    comp_env: Mapping[str, int | float],
    slot_map: Mapping[str, str],
    palette_names: set[str],
    size_step: float,
    uv_policy: str,
    comp_name: str,
    param_sig: str,
    mat_sig: str,
    parent_box_export_map: Mapping[str, str],
    dotted_bone_map: dict[str, str],
    dotted_box_map: dict[str, str],
    sub_bone_records: list[tuple[str, str]],
    exported_boxes_acc: list[str],
    exported_locs_acc: list[str],
) -> Bone:
    """Recursively instantiate internal articulated child bones inside a ``@component``."""
    angle_sign = _angle_sign_from_reflection(mirror_sign)
    exp_bone_name = f"{inst_name}__{btmpl.name}"
    dotted_bone_map[f"{inst_name}.{btmpl.name}"] = exp_bone_name
    sub_bone_records.append((btmpl.name, exp_bone_name))

    if btmpl.attach_parent not in parent_box_export_map:
        raise DSLValidationError(
            f"@component {comp_name!r} internal bone {btmpl.name!r}: unknown parent box {btmpl.attach_parent!r}"
        )
    parent_exp = parent_box_export_map[btmpl.attach_parent]
    anc = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(btmpl.attach_anchor, comp_env)) * mirror_sign))
    alg = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(btmpl.attach_align, comp_env)) * mirror_sign))
    off = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(btmpl.attach_offset, comp_env)) * mirror_sign))
    rest_e = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(btmpl.rest_euler_exprs, comp_env)) * angle_sign))

    j_lims = btmpl.joint_limits
    if j_lims is not None and is_mirrored:
        new_lims: list[tuple[float, float]] = []
        for ax_i in range(3):
            lo, hi = j_lims[ax_i]
            if _is_continuous_limit_pair((lo, hi)):
                new_lims.append((float(lo), float(hi)))
            elif float(angle_sign[ax_i]) < 0.0:
                new_lims.append((-float(hi), -float(lo)))
            else:
                new_lims.append((float(lo), float(hi)))
        j_lims = (new_lims[0], new_lims[1], new_lims[2])

    att = (
        Joint(parent=parent_exp, at=off, align=alg, detached=btmpl.attach_detached)  # type: ignore[arg-type]
        if btmpl.attach_joint_mode
        else replace(
            Joint(parent=parent_exp, at=off, align=alg, detached=btmpl.attach_detached),  # type: ignore[arg-type]
            anchor=anc,  # type: ignore[arg-type]
            joint_mode=False,
            seat=btmpl.attach_seat,
        )
    )

    local_exp_map: dict[str, str] = {}
    bone_items: list[Box | Locator | Bone] = []
    for bx_t in btmpl.boxes:
        exp_bname = f"{exp_bone_name}__{bx_t.name}"
        local_exp_map[bx_t.name] = exp_bname
        dotted_box_map[f"{inst_name}.{bx_t.name}"] = exp_bname
        exported_boxes_acc.append(exp_bname)
        mat_res = slot_map.get(bx_t.material_slot, bx_t.material_slot)
        if mat_res not in palette_names:
            raise DSLValidationError(f"@component {comp_name!r} box {bx_t.name!r}: unknown material {mat_res!r}")

        pl_obj: Place | None = None
        if bx_t.has_place:
            b_ref = local_exp_map.get(bx_t.place_base or "", None) if bx_t.place_base else None
            p_anc = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(bx_t.place_anchor, comp_env)) * mirror_sign))
            p_alg = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(bx_t.place_align, comp_env)) * mirror_sign))
            p_off = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(bx_t.place_offset, comp_env)) * mirror_sign))
            pl_obj = Place(base_box=b_ref, anchor=p_anc, align=p_alg, offset=p_off)  # type: ignore[arg-type]

        rot_b = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(bx_t.rotate_exprs, comp_env)) * angle_sign))
        piv_b = (
            tuple(float(v) for v in (np.asarray(_eval_vec3_expr(bx_t.pivot_exprs, comp_env)) * mirror_sign))
            if bx_t.pivot_exprs is not None
            else None
        )
        inf_b = float(eval_symbolic_expr(bx_t.inflate_expr, comp_env))
        sh_key = (
            f"{comp_name}::{param_sig}::{mat_sig}::{btmpl.name}::{bx_t.name}"
            if (uv_policy == "shared" and not is_mirrored)
            else None
        )
        if bx_t.is_plane:
            w_q = _quantize_size_value(eval_symbolic_expr(bx_t.size_exprs[0], comp_env), size_step)
            h_q = _quantize_size_value(eval_symbolic_expr(bx_t.size_exprs[1], comp_env), size_step)
            p_box = Plane(
                name=exp_bname,
                size=(w_q, h_q),
                normal=bx_t.plane_normal or "y",
                material=mat_res,
                place=pl_obj,
                inflate=inf_b,
                cullface=bx_t.cullface,
                rotate=rot_b,  # type: ignore[arg-type]
                pivot=piv_b,  # type: ignore[arg-type]
                uv=bx_t.uv,
                uv_faces=_component_uv_faces(bx_t.uv_faces, mirror_sign),
                texture=bx_t.texture,
                edge=bx_t.edge,
                layer=bx_t.layer,
            )
            bone_items.append(replace(p_box, uv_share_key=sh_key, instance_name=inst_name))
        else:
            sz_q = tuple(_quantize_size_value(eval_symbolic_expr(s, comp_env), size_step) for s in bx_t.size_exprs)
            bone_items.append(
                Box(
                    name=exp_bname,
                    size=sz_q,  # type: ignore[arg-type]
                    material=mat_res,
                    place=pl_obj,
                    inflate=inf_b,
                    cullface=bx_t.cullface,
                    rotate=rot_b,  # type: ignore[arg-type]
                    pivot=piv_b,  # type: ignore[arg-type]
                    uv=bx_t.uv,
                    uv_faces=_component_uv_faces(bx_t.uv_faces, mirror_sign),
                    texture=bx_t.texture,
                    uv_share_key=sh_key,
                    instance_name=inst_name,
                    prism_group=f"{exp_bone_name}__{bx_t.prism_group}" if bx_t.prism_group else None,
                )
            )

    for loc_t in btmpl.locators:
        exp_lname = f"{exp_bone_name}__{loc_t.name}"
        exported_locs_acc.append(exp_lname)
        l_off = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(loc_t.offset_exprs, comp_env)) * mirror_sign))
        l_rot = tuple(float(v) for v in (np.asarray(_eval_vec3_expr(loc_t.rot_exprs, comp_env)) * angle_sign))
        bone_items.append(Locator(name=exp_lname, offset=l_off, role=loc_t.role, rot=l_rot))  # type: ignore[arg-type]

    for ch_b in btmpl.children:
        bone_items.append(
            _lower_component_sub_bone(
                ch_b,
                inst_name,
                is_mirrored=is_mirrored,
                mirror_sign=mirror_sign,
                comp_env=comp_env,
                slot_map=slot_map,
                palette_names=palette_names,
                size_step=size_step,
                uv_policy=uv_policy,
                comp_name=comp_name,
                param_sig=param_sig,
                mat_sig=mat_sig,
                parent_box_export_map=local_exp_map,
                dotted_bone_map=dotted_bone_map,
                dotted_box_map=dotted_box_map,
                sub_bone_records=sub_bone_records,
                exported_boxes_acc=exported_boxes_acc,
                exported_locs_acc=exported_locs_acc,
            )
        )

    return Bone(
        name=exp_bone_name,
        rest_euler=rest_e,  # type: ignore[arg-type]
        attach=att,
        items=bone_items,
        joint_limits=j_lims,
    )


def resolve_assembly_program(
    *,
    meta: Meta,
    palettes: list[Palette],
    assembly_spec: AssemblySpec,
    textures: tuple[TextureDecl, ...] = (),
    materials: tuple[MaterialDecl, ...] = (),
    style: StyleDecl | None = None,
    muscles: tuple[MuscleDecl, ...] = (),
    kinematics: KinematicsDecl | None = None,
    behavior: BehaviorDecl | None = None,
    controller: ControllerDecl | None = None,
    stamps: tuple[StampDecl, ...] = (),
    layout_overrides: Mapping[str, int | float] | None = None,
) -> Program:
    """Lower a typed v4.0 ``AssemblySpec`` into a resolved ``Program`` + ``SourceMap``."""
    if assembly_spec.assembly is None:
        raise DSLValidationError("AssemblySpec requires an @assembly block")

    asm = assembly_spec.assembly
    layout_env = evaluate_layout_dag(assembly_spec.layout, overrides=layout_overrides)
    comp_by_name: dict[str, ComponentDef] = {}
    for comp in assembly_spec.components:
        if comp.name in comp_by_name:
            raise DSLValidationError(f"Duplicate @component definition {comp.name!r}")
        comp_by_name[comp.name] = comp

    palette_names = {p.name for p in palettes}
    raw_instances, raw_conns, raw_binds = _flatten_assembly_tree(asm)
    if not raw_instances:
        raise DSLValidationError(f"@assembly {asm.name!r} must declare at least one component instance")

    # Expand `mirror_x(twin)` / `mirror(...)` on instances
    expanded_instances: list[tuple[ComponentInstanceDecl, bool, np.ndarray]] = []
    mirror_pair_map: dict[str, tuple[str, str]] = {}  # orig -> (twin, axis)
    seen_inst_names: set[str] = set()

    for inst in raw_instances:
        if inst.name in seen_inst_names:
            raise DSLValidationError(f"Duplicate assembly instance {inst.name!r}")
        if inst.component_name not in comp_by_name:
            raise DSLValidationError(
                f"instance {inst.name!r} references undeclared @component {inst.component_name!r}"
            )
        seen_inst_names.add(inst.name)
        expanded_instances.append((inst, False, np.ones(3, dtype=np.float64)))

        if inst.mirror_name is not None:
            if inst.mirror_name in seen_inst_names:
                raise DSLValidationError(f"Mirrored instance name {inst.mirror_name!r} collides with existing instance")
            if inst.mirror_axis not in MIRROR_AXES:
                raise DSLValidationError(f"instance {inst.name!r}: mirror axis must be 'x' or 'z'")
            seen_inst_names.add(inst.mirror_name)
            m_sign = (
                np.array((-1.0, 1.0, 1.0), dtype=np.float64)
                if inst.mirror_axis == "x"
                else np.array((1.0, 1.0, -1.0), dtype=np.float64)
            )
            twin_id = f"{inst.stable_id}__mirror" if inst.stable_id else None
            twin_inst = ComponentInstanceDecl(
                name=inst.mirror_name,
                component_name=inst.component_name,
                stable_id=twin_id,
                param_exprs=inst.param_exprs,
                material_bindings=inst.material_bindings,
                uv_policy=inst.uv_policy,
                mirror_name=None,
                mirror_axis=inst.mirror_axis,
                subassembly_path=inst.subassembly_path,
            )
            expanded_instances.append((twin_inst, True, m_sign))
            mirror_pair_map[inst.name] = (inst.mirror_name, inst.mirror_axis)

    inst_lookup = {inst.name: (inst, is_m, m_s) for inst, is_m, m_s in expanded_instances}

    # Evaluate every component instance locally first so its frames & boxes are ready
    eval_cache: dict[str, tuple[dict[str, int | float], dict[str, str], str, list[_EvaluatedBox], list[tuple[str, np.ndarray, np.ndarray, str]], dict[str, _EvaluatedFrame]]] = {}
    for inst, is_m, m_s in expanded_instances:
        eval_cache[inst.name] = _evaluate_component_instance(
            comp_by_name[inst.component_name],
            inst,
            is_mirrored=is_m,
            mirror_sign=m_s,
            layout_env=layout_env,
            palette_names=palette_names,
            size_step=float(meta.size_step),
        )

    # Expand mirrored connections automatically for `mirror_x` instances
    all_conns: list[ConnectionDecl] = list(raw_conns)
    explicit_child_set = {c.child_instance for c in raw_conns}
    for conn in raw_conns:
        if conn.child_instance in mirror_pair_map:
            twin_child, m_ax = mirror_pair_map[conn.child_instance]
            if twin_child in explicit_child_set:
                continue
            if conn.parent_instance in mirror_pair_map:
                twin_parent = mirror_pair_map[conn.parent_instance][0]
                twin_p_frame = conn.parent_frame
            else:
                twin_parent = conn.parent_instance
                parent_frames = eval_cache[twin_parent][5]
                swapped_f = _swap_lr_frame_name(conn.parent_frame)
                if swapped_f and swapped_f in parent_frames:
                    twin_p_frame = swapped_f
                else:
                    twin_p_frame = conn.parent_frame

            m_sign_c = np.array((-1.0, 1.0, 1.0) if m_ax == "x" else (1.0, 1.0, -1.0), dtype=np.float64)
            ang_sign_c = _angle_sign_from_reflection(m_sign_c)
            ax_idx = {"x": 0, "y": 1, "z": 2}[conn.axis]
            new_lims = conn.limits
            if new_lims is not None and not _is_continuous_limit_pair(new_lims):
                flip_lim = (
                    float(ang_sign_c[ax_idx]) < 0.0
                    if conn.conn_type == "revolute"
                    else float(m_sign_c[ax_idx]) < 0.0
                )
                if flip_lim:
                    new_lims = (-float(new_lims[1]), -float(new_lims[0]))

            off_p = _eval_vec3_expr(conn.offset_pos, layout_env)
            off_r = _eval_vec3_expr(conn.offset_rot, layout_env)
            m_off_p = tuple(str(float(off_p[i]) * float(m_sign_c[i])) for i in range(3))
            m_off_r = tuple(str(float(off_r[i]) * float(ang_sign_c[i])) for i in range(3))
            all_conns.append(
                replace(
                    conn,
                    child_instance=twin_child,
                    parent_instance=twin_parent,
                    parent_frame=twin_p_frame,
                    limits=new_lims,
                    offset_pos=m_off_p,  # type: ignore[arg-type]
                    offset_rot=m_off_r,  # type: ignore[arg-type]
                )
            )

    conn_by_child: dict[str, ConnectionDecl] = {}
    for c in all_conns:
        if c.conn_type not in CONNECTION_TYPES:
            raise DSLValidationError(f"Unsupported connection type {c.conn_type!r}; expected {sorted(CONNECTION_TYPES)}")
        if c.mode not in MATE_MODES:
            raise DSLValidationError(f"Unsupported connection mode {c.mode!r}; expected 'align' or 'oppose'")
        if c.child_instance not in inst_lookup:
            raise DSLValidationError(f"connect references unknown child instance {c.child_instance!r}")
        if c.parent_instance not in inst_lookup:
            raise DSLValidationError(f"connect references unknown parent instance {c.parent_instance!r}")
        if c.child_instance == c.parent_instance:
            raise DSLValidationError(f"connect cannot self-attach instance {c.child_instance!r}")
        if c.child_instance in conn_by_child:
            raise DSLValidationError(f"Instance {c.child_instance!r} has multiple incoming connect statements")
        conn_by_child[c.child_instance] = c

    unconnected = [inst.name for inst, _, _ in expanded_instances if inst.name not in conn_by_child]
    if asm.root_instance is not None:
        if asm.root_instance not in inst_lookup:
            raise DSLValidationError(f"@assembly root_instance {asm.root_instance!r} not found")
        if asm.root_instance in conn_by_child:
            raise DSLValidationError(f"@assembly root_instance {asm.root_instance!r} must not have an incoming connect")
        root_inst_name = asm.root_instance
        extra_unconnected = [u for u in unconnected if u != root_inst_name]
        if extra_unconnected:
            raise DSLValidationError(
                f"@assembly {asm.name!r} has disconnected component instance(s): {extra_unconnected}"
            )
    else:
        if len(unconnected) != 1:
            raise DSLValidationError(
                f"@assembly {asm.name!r} must have exactly 1 root (unconnected) instance; found {unconnected}"
            )
        root_inst_name = unconnected[0]

    # Topological order over connection tree (rejects cycles!)
    children_of: dict[str, list[str]] = {inst.name: [] for inst, _, _ in expanded_instances}
    for ch_name, c in conn_by_child.items():
        children_of[c.parent_instance].append(ch_name)

    topo_instances: list[str] = []
    visited_inst: set[str] = set()

    def _walk_conn_tree(curr: str, stack_set: set[str]) -> None:
        if curr in stack_set:
            raise DSLValidationError(f"Cyclic assembly connection involving {curr!r}")
        visited_inst.add(curr)
        topo_instances.append(curr)
        nxt_stack = stack_set | {curr}
        for ch in children_of[curr]:
            _walk_conn_tree(ch, nxt_stack)

    _walk_conn_tree(root_inst_name, set())
    if len(visited_inst) != len(expanded_instances):
        missing = sorted(seen_inst_names - visited_inst)
        raise DSLValidationError(f"Cyclic or unreachable assembly instance(s): {missing}")

    # Identify instances explicitly referenced as moving bones in `bind` or motion sections
    bind_map = {b.instance_name: b.target_bone for b in raw_binds}
    motion_referenced_instances: set[str] = set()
    for mdecl in muscles:
        motion_referenced_instances.add(mdecl.origin_bone.split(".")[0])
        motion_referenced_instances.add(mdecl.insertion_bone.split(".")[0])
        if mdecl.tendon_coupling:
            motion_referenced_instances.add(mdecl.tendon_coupling.bone.split(".")[0])
    if kinematics is not None:
        for ik in kinematics.ik_chains:
            for bn in ik.bones:
                motion_referenced_instances.add(bn.split(".")[0])
        for lk in kinematics.look_chains:
            for bn in lk.bones:
                motion_referenced_instances.add(bn.split(".")[0])
        for sp in kinematics.springs:
            motion_referenced_instances.add(sp.bone.split(".")[0])
    if behavior is not None:
        for sp in behavior.springs:
            motion_referenced_instances.add(sp.bone.split(".")[0])
        for cl in behavior.clips:
            for tr in cl.tracks:
                motion_referenced_instances.add(tr.bone.split(".")[0])
            for sd in cl.slide_drives:
                motion_referenced_instances.add(sd.bone.split(".")[0])
            for kf in cl.keyframes:
                motion_referenced_instances.add(kf.bone.split(".")[0])
        for st in behavior.states:
            for b_st, _ in st.pose_biases:
                motion_referenced_instances.add(b_st.split(".")[0])

    # Evaluate exact world transforms ^W T_C = ^W T_P ^P T_A M(0) (^C T_B)^-1
    root_piv_vec = np.asarray(_eval_vec3_expr(asm.root_pivot, layout_env), dtype=np.float64)
    inst_world_pos: dict[str, np.ndarray] = {root_inst_name: root_piv_vec.copy()}
    inst_world_rot: dict[str, np.ndarray] = {root_inst_name: np.eye(3, dtype=np.float64)}
    joint_world_piv: dict[str, np.ndarray] = {root_inst_name: root_piv_vec.copy()}
    joint_world_rot: dict[str, np.ndarray] = {root_inst_name: np.eye(3, dtype=np.float64)}

    runtime_bone_of: dict[str, str] = {root_inst_name: root_inst_name}
    bone_world_piv: dict[str, np.ndarray] = {root_inst_name: root_piv_vec.copy()}
    bone_world_rot: dict[str, np.ndarray] = {root_inst_name: np.eye(3, dtype=np.float64)}
    bone_parent_bone: dict[str, str | None] = {root_inst_name: None}
    bone_joint_limits: dict[str, JointLimits | None] = {root_inst_name: None}
    generated_prismatic_joints: list[PrismaticJointDecl] = []

    #Mutable bone buckets (boxes, locators, sub-bones) assembled in deterministic topological order
    bone_boxes_map: dict[str, list[Box]] = {root_inst_name: []}
    bone_locs_map: dict[str, list[Locator]] = {root_inst_name: []}
    bone_children_order: dict[str, list[str]] = {root_inst_name: []}
    bone_component_subbones: dict[str, list[Bone]] = {root_inst_name: []}

    dotted_bone_map: dict[str, str] = {root_inst_name: root_inst_name}
    dotted_box_map: dict[str, str] = {}
    source_entries: list[SourceMapEntry] = []

    for inst_name in topo_instances:
        inst, is_m, m_sign = inst_lookup[inst_name]
        comp = comp_by_name[inst.component_name]
        bound_params, slot_map, eff_uv, ev_boxes, ev_locs, ev_frames = eval_cache[inst_name]

        conn_contact_target_box: str | None = None
        conn_contact_child_local: str | None = None
        conn_sculpted_blend: bool = False

        if inst_name != root_inst_name:
            conn = conn_by_child[inst_name]
            p_frames = eval_cache[conn.parent_instance][5]
            if conn.parent_frame not in p_frames:
                raise DSLValidationError(
                    f"connect {inst_name}.{conn.child_frame} -> {conn.parent_instance}.{conn.parent_frame}: "
                    f"parent frame {conn.parent_frame!r} not found (available: {sorted(p_frames)})"
                )
            if conn.child_frame not in ev_frames:
                raise DSLValidationError(
                    f"connect {inst_name}.{conn.child_frame}: child frame {conn.child_frame!r} not found "
                    f"(available: {sorted(ev_frames)})"
                )
            fa = p_frames[conn.parent_frame]
            fb = ev_frames[conn.child_frame]

            p_w_pos = inst_world_pos[conn.parent_instance]
            p_w_rot = inst_world_rot[conn.parent_instance]

            r_mode = np.diag((-1.0, 1.0, -1.0)) if conn.mode == "oppose" else np.eye(3, dtype=np.float64)
            t_trim = np.asarray(_eval_vec3_expr(conn.offset_pos, layout_env), dtype=np.float64)
            r_trim = euler_xyz_matrix(_eval_vec3_expr(conn.offset_rot, layout_env))

            p_joint_w = p_w_pos + p_w_rot @ (fa.pos_comp + fa.rot_comp @ t_trim)
            r_joint_w = p_w_rot @ fa.rot_comp @ r_mode @ r_trim

            r_c_w = r_joint_w @ fb.rot_comp.T
            p_c_w = p_joint_w - r_c_w @ fb.pos_comp

            inst_world_pos[inst_name] = p_c_w
            inst_world_rot[inst_name] = r_c_w
            joint_world_piv[inst_name] = p_joint_w
            joint_world_rot[inst_name] = r_joint_w

            if fa.on_box_local is not None and fb.on_box_local is not None:
                conn_contact_target_box = f"{conn.parent_instance}__{fa.on_box_local}"
                conn_contact_child_local = fb.on_box_local
                conn_sculpted_blend = conn.sculpted_blend

            parent_rb = runtime_bone_of[conn.parent_instance]
            if inst_name in bind_map:
                target_rb = bind_map[inst_name]
                if target_rb == parent_rb:
                    allocates_new_bone = False
                elif target_rb == inst_name:
                    allocates_new_bone = True
                else:
                    raise DSLValidationError(
                        f"bind {inst_name} to {target_rb!r}: target must be parent bone {parent_rb!r} or own bone {inst_name!r}"
                    )
            else:
                allocates_new_bone = (
                    conn.conn_type in ("revolute", "prismatic")
                    or conn.preserve_bone
                    or inst_name in motion_referenced_instances
                )

            if allocates_new_bone:
                runtime_bone_of[inst_name] = inst_name
                bone_world_piv[inst_name] = p_joint_w
                bone_world_rot[inst_name] = r_joint_w
                bone_parent_bone[inst_name] = parent_rb
                bone_boxes_map[inst_name] = []
                bone_locs_map[inst_name] = []
                bone_children_order[inst_name] = []
                bone_component_subbones[inst_name] = []
                bone_children_order[parent_rb].append(inst_name)

                if conn.conn_type == "revolute":
                    lo_d, hi_d = conn.limits if conn.limits is not None else (-45.0, 45.0)
                    lims_3 = [
                        (float(lo_d), float(hi_d)) if ax == conn.axis else (0.0, 0.0)
                        for ax in ("x", "y", "z")
                    ]
                    bone_joint_limits[inst_name] = (lims_3[0], lims_3[1], lims_3[2])
                elif conn.conn_type == "prismatic":
                    bone_joint_limits[inst_name] = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
                    lo_u, hi_u = conn.limits if conn.limits is not None else (-4.0, 4.0)
                    e_ax = np.zeros(3, dtype=np.float64)
                    e_ax[{"x": 0, "y": 1, "z": 2}[conn.axis]] = 1.0
                    dir_in_parent = bone_world_rot[parent_rb].T @ (r_joint_w @ e_ax)
                    generated_prismatic_joints.append(
                        PrismaticJointDecl(
                            bone=inst_name,
                            axis=conn.axis,
                            limits=(float(lo_u), float(hi_u)),
                            axis_vec=(float(dir_in_parent[0]), float(dir_in_parent[1]), float(dir_in_parent[2])),
                        )
                    )
                else:
                    bone_joint_limits[inst_name] = None
            else:
                runtime_bone_of[inst_name] = parent_rb

        rb_name = runtime_bone_of[inst_name]
        dotted_bone_map[inst_name] = rb_name
        rb_piv = bone_world_piv[rb_name]
        rb_rot = bone_world_rot[rb_name]
        c_w_pos = inst_world_pos[inst_name]
        c_w_rot = inst_world_rot[inst_name]

        local_to_export_box: dict[str, str] = {eb.local_name: eb.export_name for eb in ev_boxes}
        exported_boxes_acc: list[str] = []
        exported_locs_acc: list[str] = []

        # Emit boxes transformed into `rb_name` runtime bone frame
        for eb in ev_boxes:
            box_w_ctr = c_w_pos + c_w_rot @ eb.center_comp
            box_w_rot = c_w_rot @ eb.rot_comp
            ctr_in_rb = rb_rot.T @ (box_w_ctr - rb_piv)
            rot_in_rb = rb_rot.T @ box_w_rot
            eul_in_rb = _recover_euler_xyz_degrees(rot_in_rb)

            blend_targets: list[str] = [
                local_to_export_box[bw] if bw in local_to_export_box else bw
                for bw in eb.blend_with_local
            ]
            contact_targets: list[str] = []
            if eb.place_base_local is not None and eb.place_base_local in local_to_export_box:
                contact_targets.append(local_to_export_box[eb.place_base_local])
            if conn_contact_target_box is not None and eb.local_name == conn_contact_child_local:
                if conn_contact_target_box not in contact_targets:
                    contact_targets.append(conn_contact_target_box)
                if conn_sculpted_blend and conn_contact_target_box not in blend_targets:
                    blend_targets.append(conn_contact_target_box)

            if eb.plane_normal is not None and eb.layer != 0:
                n_ax = {"x": 0, "y": 1, "z": 2}[eb.plane_normal]
                n_w = box_w_rot[:, n_ax]
                ctr_in_rb = ctr_in_rb + 0.25 * float(eb.layer) * (rb_rot.T @ n_w)

            sz_tuple: tuple[int | float, int | float, int | float] = tuple(
                int(round(float(v))) if abs(float(v) - round(float(v))) <= 1e-9 else round(float(v), 6)
                for v in eb.size
            )  # type: ignore[assignment]

            ast_box = Box(
                name=eb.export_name,
                size=sz_tuple,
                material=eb.material,
                place=None,
                inflate=eb.inflate,
                cullface=eb.cullface,
                rotate=(float(eul_in_rb[0]), float(eul_in_rb[1]), float(eul_in_rb[2])),
                pivot=None,
                plane_normal=eb.plane_normal,
                blend_with=tuple(blend_targets),
                uv_share_key=eb.uv_share_key,
                component_rest_pos=(float(eb.center_comp[0]), float(eb.center_comp[1]), float(eb.center_comp[2])),
                component_rest_euler=eb.euler_comp,
                explicit_local_center=(float(ctr_in_rb[0]), float(ctr_in_rb[1]), float(ctr_in_rb[2])),
                explicit_contact_with=tuple(contact_targets),
                instance_name=inst_name,
                uv=eb.uv,
                uv_faces=eb.uv_faces,
                texture=eb.texture,
                edge=eb.edge,
                layer=0,
                prism_group=eb.prism_group,
            )
            bone_boxes_map[rb_name].append(ast_box)
            dotted_box_map[f"{inst_name}.{eb.local_name}"] = eb.export_name
            exported_boxes_acc.append(eb.export_name)

        # Emit component root locators + exported public attachment frames
        for exp_lname, off_c, rot_c, l_role in ev_locs:
            lw_pos = c_w_pos + c_w_rot @ off_c
            lw_rot = c_w_rot @ rot_c
            l_off_rb = rb_rot.T @ (lw_pos - rb_piv)
            l_eul_rb = _recover_euler_xyz_degrees(rb_rot.T @ lw_rot)
            bone_locs_map[rb_name].append(
                Locator(
                    name=exp_lname,
                    offset=(float(l_off_rb[0]), float(l_off_rb[1]), float(l_off_rb[2])),
                    role=l_role,
                    rot=(float(l_eul_rb[0]), float(l_eul_rb[1]), float(l_eul_rb[2])),
                )
            )
            exported_locs_acc.append(exp_lname)

        for fr_obj in ev_frames.values():
            if not fr_obj.export_locator:
                continue
            exp_fname = f"{inst_name}__{fr_obj.name}"
            fw_pos = c_w_pos + c_w_rot @ fr_obj.pos_comp
            fw_rot = c_w_rot @ fr_obj.rot_comp
            f_off_rb = rb_rot.T @ (fw_pos - rb_piv)
            f_eul_rb = _recover_euler_xyz_degrees(rb_rot.T @ fw_rot)
            bone_locs_map[rb_name].append(
                Locator(
                    name=exp_fname,
                    offset=(float(f_off_rb[0]), float(f_off_rb[1]), float(f_off_rb[2])),
                    role=fr_obj.role,
                    rot=(float(f_eul_rb[0]), float(f_eul_rb[1]), float(f_eul_rb[2])),
                )
            )
            exported_locs_acc.append(exp_fname)

        # Instantiate any internal articulated child bones inside the component
        sub_bone_records: list[tuple[str, str]] = []
        if comp.child_bones:
            comp_full_env = {**layout_env, **bound_params}
            for ldecl in comp.lets:
                comp_full_env[ldecl.name] = eval_symbolic_expr(ldecl.expr, comp_full_env)
            param_sig = ",".join(f"{k}={bound_params[k]}" for k in sorted(bound_params))
            mat_sig = ",".join(f"{k}={slot_map[k]}" for k in sorted(slot_map))
            for cb_tmpl in comp.child_bones:
                bone_component_subbones[rb_name].append(
                    _lower_component_sub_bone(
                        cb_tmpl,
                        inst_name,
                        is_mirrored=is_m,
                        mirror_sign=m_sign,
                        comp_env=comp_full_env,
                        slot_map=slot_map,
                        palette_names=palette_names,
                        size_step=float(meta.size_step),
                        uv_policy=eff_uv,
                        comp_name=comp.name,
                        param_sig=param_sig,
                        mat_sig=mat_sig,
                        parent_box_export_map=local_to_export_box,
                        dotted_bone_map=dotted_bone_map,
                        dotted_box_map=dotted_box_map,
                        sub_bone_records=sub_bone_records,
                        exported_boxes_acc=exported_boxes_acc,
                        exported_locs_acc=exported_locs_acc,
                    )
                )

        for ctrl in comp.controls:
            tgt_resolved = dotted_bone_map.get(f"{inst_name}.{ctrl.target}", rb_name)
            dotted_bone_map[f"{inst_name}.{ctrl.name}"] = tgt_resolved

        readable_parts = [asm.name, *inst.subassembly_path, inst_name]
        readable_path = ".".join(readable_parts)
        stable_id = inst.stable_id or f"sid_{hashlib.sha256(readable_path.encode('utf-8')).hexdigest()[:10]}"
        source_entries.append(
            SourceMapEntry(
                stable_id=stable_id,
                instance_name=inst_name,
                readable_path=readable_path,
                component_name=comp.name,
                mirrored=is_m,
                bound_params=tuple(sorted(bound_params.items())),
                bound_materials=tuple(sorted(slot_map.items())),
                uv_policy=eff_uv,
                runtime_bone_name=rb_name,
                exported_boxes=tuple(exported_boxes_acc),
                exported_locators=tuple(exported_locs_acc),
                exported_sub_bones=tuple(sub_bone_records),
            )
        )

    # Assemble the lowered runtime Bone tree
    def _build_runtime_bone(b_name: str) -> Bone:
        parent_b = bone_parent_bone[b_name]
        b_boxes = bone_boxes_map[b_name]
        b_locs = bone_locs_map[b_name]
        if not b_boxes and not b_locs:
            # Ensure runtime bone satisfies non-empty primitive rule if an empty carrier bone was requested
            b_locs = [Locator(name=f"{b_name}_pivot", offset=(0.0, 0.0, 0.0), role="socket")]

        items_out: list[Box | Locator | Bone] = [*b_boxes, *b_locs, *bone_component_subbones[b_name]]
        for ch_rb in bone_children_order[b_name]:
            items_out.append(_build_runtime_bone(ch_rb))

        if parent_b is None:
            piv = bone_world_piv[b_name]
            return Bone(
                name=b_name,
                pivot_local=(float(piv[0]), float(piv[1]), float(piv[2])),
                items=items_out,
            )

        p_piv = bone_world_piv[parent_b]
        p_rot = bone_world_rot[parent_b]
        c_piv = bone_world_piv[b_name]
        c_rot = bone_world_rot[b_name]

        desired_trans_in_p = p_rot.T @ (c_piv - p_piv)
        p_boxes = bone_boxes_map[parent_b]
        if p_boxes:
            anchor_box = p_boxes[0]
            ref_ctr = np.asarray(anchor_box.explicit_local_center or (0.0, 0.0, 0.0), dtype=np.float64)
            at_vec = desired_trans_in_p - ref_ctr
            att_parent_name = anchor_box.name
        else:
            anchor_loc = bone_locs_map[parent_b][0]
            ref_ctr = np.asarray(anchor_loc.offset, dtype=np.float64)
            at_vec = desired_trans_in_p - ref_ctr
            att_parent_name = anchor_loc.name

        rel_eul = _recover_euler_xyz_degrees(p_rot.T @ c_rot)
        return Bone(
            name=b_name,
            rest_euler=(float(rel_eul[0]), float(rel_eul[1]), float(rel_eul[2])),
            attach=Joint(
                parent=att_parent_name,
                at=(float(at_vec[0]), float(at_vec[1]), float(at_vec[2])),
                align=(0.0, 0.0, 0.0),
                detached=True,  # Contact edges are attached explicitly via `explicit_contact_with`!
            ),
            items=items_out,
            joint_limits=bone_joint_limits.get(b_name),
        )

    root_bone = _build_runtime_bone(root_inst_name)

    # Remap dotted references (`leg_fl.shin`, `skull.cranium`) in @muscles, @kinematics, @behavior
    def _res_bone(ref: str) -> str:
        return dotted_bone_map.get(ref, ref)

    def _res_box(ref: str) -> str:
        return dotted_box_map.get(ref, ref)

    remapped_muscles: tuple[MuscleDecl, ...] = tuple(
        MuscleDecl(
            name=m.name,
            origin_bone=_res_bone(m.origin_bone),
            insertion_bone=_res_bone(m.insertion_bone),
            action_axis=m.action_axis,
            moment_arm_deg=m.moment_arm_deg,
            tendon_coupling=(
                TendonCoupling(
                    bone=_res_bone(m.tendon_coupling.bone),
                    axis=m.tendon_coupling.axis,
                    ratio=m.tendon_coupling.ratio,
                )
                if m.tendon_coupling is not None
                else None
            ),
            belly_box=_res_box(m.belly_box) if m.belly_box is not None else None,
            bulge_inflate=m.bulge_inflate,
        )
        for m in muscles
    )

    remapped_kinematics: KinematicsDecl | None = None
    if kinematics is not None or generated_prismatic_joints:
        ik_list = tuple(
            IKChainDecl(
                name=ik.name,
                bones=tuple(_res_bone(b) for b in ik.bones),
                effector=ik.effector,
                hinge_axis=ik.hinge_axis,
                pole=ik.pole,
            )
            for ik in (kinematics.ik_chains if kinematics else ())
        )
        lk_list = tuple(
            LookChainDecl(
                name=lk.name,
                bones=tuple(_res_bone(b) for b in lk.bones),
                weight_decay=lk.weight_decay,
            )
            for lk in (kinematics.look_chains if kinematics else ())
        )
        sp_list = tuple(
            SpringDecl(bone=_res_bone(sp.bone), stiffness=sp.stiffness, damping=sp.damping)
            for sp in (kinematics.springs if kinematics else ())
        )
        pj_existing = tuple(
            PrismaticJointDecl(
                bone=_res_bone(pj.bone),
                axis=pj.axis,
                limits=pj.limits,
                axis_vec=pj.axis_vec,
            )
            for pj in (kinematics.prismatic_joints if kinematics else ())
        )
        all_pjs = (*pj_existing, *generated_prismatic_joints)
        if ik_list or lk_list or sp_list or all_pjs:
            remapped_kinematics = KinematicsDecl(
                ik_chains=ik_list,
                look_chains=lk_list,
                springs=sp_list,
                prismatic_joints=all_pjs,
            )

    remapped_behavior: BehaviorDecl | None = None
    if behavior is not None:
        remapped_clips = tuple(
            ClipDecl(
                name=cl.name,
                duration_s=cl.duration_s,
                loop=cl.loop,
                tracks=tuple(replace(tr, bone=_res_bone(tr.bone)) for tr in cl.tracks),
                muscle_drives=cl.muscle_drives,
                ik_drives=cl.ik_drives,
                keyframes=tuple(replace(kf, bone=_res_bone(kf.bone)) for kf in cl.keyframes),
                slide_drives=tuple(replace(sd, bone=_res_bone(sd.bone)) for sd in cl.slide_drives),
                gaits=cl.gaits,
            )
            for cl in behavior.clips
        )
        remapped_states = tuple(
            BehaviorStateDecl(
                name=st.name,
                pose_biases=tuple((_res_bone(bn), vec) for bn, vec in st.pose_biases),
                muscle_tones=st.muscle_tones,
                material_mods=st.material_mods,
            )
            for st in behavior.states
        )
        remapped_sp_b = tuple(
            SpringDecl(bone=_res_bone(sp.bone), stiffness=sp.stiffness, damping=sp.damping)
            for sp in behavior.springs
        )
        remapped_behavior = BehaviorDecl(
            clips=remapped_clips,
            states=remapped_states,
            springs=remapped_sp_b,
        )

    source_map = SourceMap(
        assembly_name=asm.name,
        assembly_provenance=assembly_spec.assembly_provenance,
        layout_values=tuple(sorted(layout_env.items())),
        entries=tuple(source_entries),
        dotted_bone_map=tuple(sorted(dotted_bone_map.items())),
        dotted_box_map=tuple(sorted(dotted_box_map.items())),
    )

    remapped_stamps = tuple(
        replace(stp, target_box=dotted_box_map.get(stp.target_box, stp.target_box))
        for stp in stamps
    )

    program = Program(
        meta=meta,
        palettes=palettes,
        root=root_bone,
        materials=materials,
        style=style,
        muscles=remapped_muscles,
        kinematics=remapped_kinematics,
        behavior=remapped_behavior,
        textures=textures,
        controller=controller,
        assembly_spec=assembly_spec,
        source_map=source_map,
        stamps=remapped_stamps,
    )
    return validate_program(program)


# ---------------------------------------------------------------------------
# Parametric Overrides, Contract Audits & Blockbench Edit Reconciliation
# ---------------------------------------------------------------------------


def override_layout_params(
    program: Program,
    param_overrides: Mapping[str, int | float],
) -> Program:
    """Re-evaluate a v4.0 assembly ``Program`` with updated top-down ``@layout`` parameters."""
    if program.assembly_spec is None or not isinstance(program.assembly_spec, AssemblySpec):
        raise DSLValidationError("Program has no AssemblySpec to override @layout parameters on")

    spec: AssemblySpec = program.assembly_spec
    if spec.layout is None:
        raise DSLValidationError("AssemblySpec has no @layout section")

    updated_params = tuple(
        replace(p, default_expr=str(param_overrides[p.name])) if p.name in param_overrides else p
        for p in spec.layout.params
    )
    unknown = set(param_overrides) - {p.name for p in spec.layout.params}
    if unknown:
        raise DSLValidationError(f"Unknown @layout parameter override(s): {sorted(unknown)}")

    new_spec = replace(spec, layout=replace(spec.layout, params=updated_params))
    # Restore un-remapped dotted motion declarations by inverting `dotted_bone_map` if needed,
    # or re-resolving directly since resolved export bone names remain valid in `dotted_bone_map`!
    inv_pjs: tuple[PrismaticJointDecl, ...] = ()
    if program.kinematics is not None:
        auto_bones = {
            c.child_instance
            for c in _flatten_assembly_tree(spec.assembly)[1]  # type: ignore[arg-type]
            if c.conn_type == "prismatic"
        }
        # Also exclude mirrored prismatic twins
        for inst_d in _flatten_assembly_tree(spec.assembly)[0]:  # type: ignore[arg-type]
            if inst_d.mirror_name and inst_d.name in auto_bones:
                auto_bones.add(inst_d.mirror_name)
        inv_pjs = tuple(pj for pj in program.kinematics.prismatic_joints if pj.bone not in auto_bones)
        clean_kin = (
            replace(program.kinematics, prismatic_joints=inv_pjs)
            if (program.kinematics.ik_chains or program.kinematics.look_chains or program.kinematics.springs or inv_pjs)
            else None
        )
    else:
        clean_kin = None

    return resolve_assembly_program(
        meta=program.meta,
        palettes=list(program.palettes),
        assembly_spec=new_spec,
        textures=program.textures,
        materials=program.materials,
        style=program.style,
        muscles=program.muscles,
        kinematics=clean_kin,
        behavior=program.behavior,
        controller=program.controller,
        stamps=program.stamps,
    )


def swap_component_definition(
    program: Program,
    new_component: ComponentDef,
) -> Program:
    """Surgically replace a single ``@component`` definition while keeping all other instances invariant."""
    if program.assembly_spec is None or not isinstance(program.assembly_spec, AssemblySpec):
        raise DSLValidationError("Program has no AssemblySpec")
    spec: AssemblySpec = program.assembly_spec
    found = any(c.name == new_component.name for c in spec.components)
    if not found:
        raise DSLValidationError(f"@component {new_component.name!r} not found in AssemblySpec")
    updated_comps = tuple(new_component if c.name == new_component.name else c for c in spec.components)
    new_spec = replace(spec, components=updated_comps)

    auto_bones: set[str] = set()
    if spec.assembly is not None:
        conns = _flatten_assembly_tree(spec.assembly)[1]
        insts = _flatten_assembly_tree(spec.assembly)[0]
        auto_bones = {c.child_instance for c in conns if c.conn_type == "prismatic"}
        for inst_d in insts:
            if inst_d.mirror_name and inst_d.name in auto_bones:
                auto_bones.add(inst_d.mirror_name)
    clean_kin = None
    if program.kinematics is not None:
        inv_pjs = tuple(pj for pj in program.kinematics.prismatic_joints if pj.bone not in auto_bones)
        if program.kinematics.ik_chains or program.kinematics.look_chains or program.kinematics.springs or inv_pjs:
            clean_kin = replace(program.kinematics, prismatic_joints=inv_pjs)

    return resolve_assembly_program(
        meta=program.meta,
        palettes=list(program.palettes),
        assembly_spec=new_spec,
        textures=program.textures,
        materials=program.materials,
        style=program.style,
        muscles=program.muscles,
        kinematics=clean_kin,
        behavior=program.behavior,
        controller=program.controller,
        stamps=program.stamps,
    )


def audit_assembly_contracts(program: Program, scene: Scene | None = None) -> dict[str, Any]:
    """Audit ``@layout`` ``budget`` and ``envelope`` assertions plus ``SourceMap`` integrity."""
    compiled = scene if scene is not None else compile_geometry(program)
    violations: list[dict[str, Any]] = []

    all_corners = np.concatenate([b.corners(include_inflate=True) for b in compiled.boxes], axis=0)
    mins = all_corners.min(axis=0)
    maxs = all_corners.max(axis=0)
    span = maxs - mins

    spec: AssemblySpec | None = program.assembly_spec if isinstance(program.assembly_spec, AssemblySpec) else None
    if spec is not None and spec.layout is not None:
        bud = spec.layout.budget
        if bud.max_boxes is not None and len(compiled.boxes) > bud.max_boxes:
            violations.append({
                "kind": "budget_max_boxes_exceeded",
                "actual": len(compiled.boxes),
                "limit": bud.max_boxes,
            })
        if bud.max_bones is not None and len(compiled.bones) > bud.max_bones:
            violations.append({
                "kind": "budget_max_bones_exceeded",
                "actual": len(compiled.bones),
                "limit": bud.max_bones,
            })
        env = spec.layout.envelope
        if env.max_span is not None:
            for ax_i, ax_name in enumerate(("x", "y", "z")):
                if float(span[ax_i]) > float(env.max_span[ax_i]) + 1e-6:
                    violations.append({
                        "kind": "envelope_max_span_exceeded",
                        "axis": ax_name,
                        "actual": round(float(span[ax_i]), 4),
                        "limit": float(env.max_span[ax_i]),
                    })
        if env.min_ground_y is not None and float(mins[1]) < float(env.min_ground_y) - 1e-6:
            violations.append({
                "kind": "envelope_min_ground_y_violated",
                "actual_min_y": round(float(mins[1]), 4),
                "limit": float(env.min_ground_y),
            })
        if env.grounded and abs(float(mins[1]) - float(env.ground_y)) > float(env.ground_tol) + 1e-6:
            violations.append({
                "kind": "envelope_grounded_violated",
                "actual_min_y": round(float(mins[1]), 4),
                "ground_y": float(env.ground_y),
                "tolerance": float(env.ground_tol),
            })

    return {
        "passed": len(violations) == 0,
        "box_count": len(compiled.boxes),
        "bone_count": len(compiled.bones),
        "world_aabb_min": [round(float(x), 4) for x in mins],
        "world_aabb_max": [round(float(x), 4) for x in maxs],
        "world_span": [round(float(x), 4) for x in span],
        "assembly_provenance": spec.assembly_provenance if spec is not None else "direct_rig",
        "violations": violations,
    }


def reconcile_exported_pose_to_assembly(
    program: Program,
    bone_euler_deltas_deg: Mapping[str, Vec3],
) -> Program:
    """Reconcile Blockbench/runtime bone orientation tweaks back into ``connect { offset_rot }`` via ``SourceMap``."""
    if program.assembly_spec is None or not isinstance(program.assembly_spec, AssemblySpec):
        raise DSLValidationError("Program has no AssemblySpec for reconciliation")
    spec: AssemblySpec = program.assembly_spec
    if spec.assembly is None:
        raise DSLValidationError("AssemblySpec has no @assembly")

    layout_env = evaluate_layout_dag(spec.layout)

    def _patch_conns(conns: tuple[ConnectionDecl, ...]) -> tuple[ConnectionDecl, ...]:
        out_c: list[ConnectionDecl] = []
        for c in conns:
            if c.child_instance in bone_euler_deltas_deg:
                cur_r = _eval_vec3_expr(c.offset_rot, layout_env)
                d_r = bone_euler_deltas_deg[c.child_instance]
                r_comb = _recover_euler_xyz_degrees(euler_xyz_matrix(cur_r) @ euler_xyz_matrix(d_r))
                out_c.append(
                    replace(
                        c,
                        offset_rot=(
                            f"{round(float(r_comb[0]), 4)}",
                            f"{round(float(r_comb[1]), 4)}",
                            f"{round(float(r_comb[2]), 4)}",
                        ),
                    )
                )
            else:
                out_c.append(c)
        return tuple(out_c)

    def _patch_sub(sub: SubassemblyDecl) -> SubassemblyDecl:
        return replace(
            sub,
            connections=_patch_conns(sub.connections),
            subassemblies=tuple(_patch_sub(s) for s in sub.subassemblies),
        )

    patched_asm = replace(
        spec.assembly,
        connections=_patch_conns(spec.assembly.connections),
        subassemblies=tuple(_patch_sub(s) for s in spec.assembly.subassemblies),
    )
    new_spec = replace(spec, assembly=patched_asm)
    return swap_component_definition(
        replace(program, assembly_spec=new_spec),
        spec.components[0],
    )


def format_assembly_sections(spec: AssemblySpec) -> list[str]:
    """Serialize ``@layout``, ``@component``s, and ``@assembly`` in canonical Block-DSL v4.0 text."""
    lines: list[str] = []

    def _fmt_vec_e(ve: tuple[str, ...]) -> str:
        return "[" + ", ".join(str(x) for x in ve) + "]"

    def _is_zero_ve(ve: tuple[str, ...]) -> bool:
        return all(str(x).strip() in ("0", "0.0", "-0", "-0.0") for x in ve)

    def _fmt_num(val: int | float) -> str:
        f = float(val)
        if f == 0.0:
            return "0"
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        return str(f)

    if spec.layout is not None:
        lay = spec.layout
        lines.append("@layout {")
        for p in lay.params:
            bounds = (
                f" [{_fmt_num(p.min_val)}, {_fmt_num(p.max_val)}]"
                if (p.min_val is not None and p.max_val is not None)
                else ""
            )
            lines.append(f"  param {p.name} = {p.default_expr}{bounds}")
        for ld in lay.lets:
            lines.append(f"  let {ld.name} = {ld.expr}")
        if lay.budget.max_boxes is not None:
            lines.append(f"  budget max_boxes = {lay.budget.max_boxes}")
        if lay.budget.max_bones is not None:
            lines.append(f"  budget max_bones = {lay.budget.max_bones}")
        if lay.budget.max_atlas_px is not None:
            lines.append(f"  budget max_atlas_px = {lay.budget.max_atlas_px}")
        if lay.envelope.max_span is not None:
            sp = lay.envelope.max_span
            lines.append(f"  envelope max_span = [{_fmt_num(sp[0])}, {_fmt_num(sp[1])}, {_fmt_num(sp[2])}]")
        if lay.envelope.min_ground_y is not None:
            lines.append(f"  envelope min_ground_y = {_fmt_num(lay.envelope.min_ground_y)}")
        if lay.envelope.grounded:
            lines.append("  envelope grounded = true")
            if abs(float(lay.envelope.ground_y)) > 1e-9:
                lines.append(f"  envelope ground_y = {_fmt_num(lay.envelope.ground_y)}")
            if abs(float(lay.envelope.ground_tol) - 0.15) > 1e-9:
                lines.append(f"  envelope ground_tol = {_fmt_num(lay.envelope.ground_tol)}")
        lines.append("}")

    def _write_comp_box(bt: ComponentBoxTemplate, pad: str) -> None:
        kw = "plane" if bt.is_plane else "box"
        hdr = f"{pad}{kw} {bt.name}"
        if bt.has_place:
            p_args: list[str] = []
            if bt.place_base:
                p_args.append(f"base_box={bt.place_base}")
            p_args.append(f"anchor={_fmt_vec_e(bt.place_anchor)}")
            p_args.append(f"align={_fmt_vec_e(bt.place_align)}")
            if not _is_zero_ve(bt.place_offset):
                p_args.append(f"offset={_fmt_vec_e(bt.place_offset)}")
            hdr += f" place({', '.join(p_args)})"
        bf: list[str] = [f"size: {_fmt_vec_e(bt.size_exprs)}"]
        if bt.is_plane and bt.plane_normal:
            bf.append(f"normal: {bt.plane_normal}")
        bf.append(f"material: {bt.material_slot}")
        if bt.inflate_expr.strip() not in ("0", "0.0"):
            bf.append(f"inflate: {bt.inflate_expr}")
        if not _is_zero_ve(bt.rotate_exprs):
            bf.append(f"rotate: {_fmt_vec_e(bt.rotate_exprs)}")
        if bt.pivot_exprs is not None:
            bf.append(f"pivot: {_fmt_vec_e(bt.pivot_exprs)}")
        if bt.cullface:
            bf.append(f"cullface: [{', '.join(bt.cullface)}]")
        if bt.blend_with:
            bf.append(f"blend_with: [{', '.join(bt.blend_with)}]")
        if bt.uv is not None:
            bf.append(f"uv: box({_fmt_num(bt.uv[0])}, {_fmt_num(bt.uv[1])})")
        if bt.uv_faces:
            uvf = [f"{face}: [{', '.join(_fmt_num(x) for x in rect)}]" for face, rect in bt.uv_faces]
            bf.append("uv_faces: { " + ", ".join(uvf) + " }")
        if bt.texture is not None:
            bf.append(f"texture: {bt.texture}")
        if bt.edge is not None:
            bf.append(f"edge: {format_edge_cutout(bt.edge)}")
        if bt.layer != 0:
            bf.append(f"layer: {bt.layer}")
        if bt.prism_group is not None:
            bf.append(f"prism_group: {bt.prism_group}")
        lines.append(f"{hdr} {{ {', '.join(bf)} }}")

    def _write_comp_bone(cb: ComponentBoneTemplate, indent: int) -> None:
        pad = " " * indent
        if cb.attach_joint_mode:
            ja = [f"parent={cb.attach_parent}"]
            if not _is_zero_ve(cb.attach_offset):
                ja.append(f"at={_fmt_vec_e(cb.attach_offset)}")
            if not _is_zero_ve(cb.attach_align):
                ja.append(f"align={_fmt_vec_e(cb.attach_align)}")
            if cb.attach_detached:
                ja.append("detached=true")
            att_s = f"joint({', '.join(ja)})"
        else:
            aa = [
                f"parent_box={cb.attach_parent}",
                f"anchor={_fmt_vec_e(cb.attach_anchor)}",
                f"align={_fmt_vec_e(cb.attach_align)}",
            ]
            if not _is_zero_ve(cb.attach_offset):
                aa.append(f"offset={_fmt_vec_e(cb.attach_offset)}")
            if cb.attach_detached:
                aa.append("detached=true")
            if cb.attach_seat > 1e-9:
                aa.append(f"seat={_fmt_num(cb.attach_seat)}")
            att_s = f"attach({', '.join(aa)})"
        lines.append(f"{pad}bone {cb.name} {att_s} {{")
        if not _is_zero_ve(cb.rest_euler_exprs):
            lines.append(f"{pad}  rest_euler: {_fmt_vec_e(cb.rest_euler_exprs)}")
        if cb.joint_limits is not None:
            if all(_is_continuous_limit_pair(p) for p in cb.joint_limits):
                lines.append(f"{pad}  joint_limits: continuous")
            else:
                jl_s = "[" + ", ".join("continuous" if _is_continuous_limit_pair(p) else f"[{_fmt_num(p[0])}, {_fmt_num(p[1])}]" for p in cb.joint_limits) + "]"
                lines.append(f"{pad}  joint_limits: {jl_s}")
        for bx in cb.boxes:
            _write_comp_box(bx, pad + "  ")
        for lc in cb.locators:
            r_p = f", rot: {_fmt_vec_e(lc.rot_exprs)}" if not _is_zero_ve(lc.rot_exprs) else ""
            rl_p = f", role: {lc.role}" if lc.role != "socket" else ""
            lines.append(f"{pad}  locator {lc.name} {{ offset: {_fmt_vec_e(lc.offset_exprs)}{r_p}{rl_p} }}")
        for sub_cb in cb.children:
            _write_comp_bone(sub_cb, indent + 2)
        lines.append(f"{pad}}}")

    for comp in spec.components:
        lines.append(f"@component {comp.name} {{")
        for p in comp.params:
            b_s = (
                f" [{_fmt_num(p.min_val)}, {_fmt_num(p.max_val)}]"
                if (p.min_val is not None and p.max_val is not None)
                else ""
            )
            lines.append(f"  param {p.name} = {p.default_expr}{b_s}")
        if comp.material_slots:
            ms_parts = [f"{sn}={sd}" if sd else sn for sn, sd in comp.material_slots]
            lines.append(f"  material_slots: [{', '.join(ms_parts)}]")
        if comp.uv_policy != "unique":
            lines.append(f"  uv_policy: {comp.uv_policy}")
        for ld in comp.lets:
            lines.append(f"  let {ld.name} = {ld.expr}")
        for bx in comp.root_boxes:
            _write_comp_box(bx, "  ")
        for lc in comp.root_locators:
            r_p = f", rot: {_fmt_vec_e(lc.rot_exprs)}" if not _is_zero_ve(lc.rot_exprs) else ""
            rl_p = f", role: {lc.role}" if lc.role != "socket" else ""
            lines.append(f"  locator {lc.name} {{ offset: {_fmt_vec_e(lc.offset_exprs)}{r_p}{rl_p} }}")
        for cb in comp.child_bones:
            _write_comp_bone(cb, 2)
        for fr in comp.frames:
            fp: list[str] = []
            if fr.on_box is not None:
                fp.append(f"on_box: {fr.on_box}")
                fp.append(f"anchor: {_fmt_vec_e(fr.anchor)}")
                if not _is_zero_ve(fr.offset):
                    fp.append(f"offset: {_fmt_vec_e(fr.offset)}")
            else:
                fp.append(f"pos: {_fmt_vec_e(fr.pos)}")
            if not _is_zero_ve(fr.rot):
                fp.append(f"rot: {_fmt_vec_e(fr.rot)}")
            if fr.role != "socket":
                fp.append(f"role: {fr.role}")
            if fr.export_locator and fr.role == "socket":
                fp.append("export_locator: true")
            lines.append(f"  frame {fr.name} {{ {', '.join(fp)} }}")
        for ctrl in comp.controls:
            lines.append(
                f"  control {ctrl.name} {{ target: {ctrl.target}, channel: {ctrl.channel}, "
                f"axis: {ctrl.axis}, limits: [{_fmt_num(ctrl.limits[0])}, {_fmt_num(ctrl.limits[1])}] }}"
            )
        lines.append("}")

    if spec.assembly is not None:
        asm = spec.assembly
        lines.append(f"@assembly {asm.name} {{")
        if not _is_zero_ve(asm.root_pivot):
            lines.append(f"  root_pivot: {_fmt_vec_e(asm.root_pivot)}")
        if asm.root_instance is not None:
            lines.append(f"  root_instance: {asm.root_instance}")

        def _write_inst(inst: ComponentInstanceDecl, pad: str) -> None:
            p_call = ", ".join(f"{k}={v}" for k, v in inst.param_exprs)
            mir_s = ""
            if inst.mirror_name:
                mir_s = (
                    f" mirror_x({inst.mirror_name})"
                    if inst.mirror_axis == "x"
                    else f" mirror(axis={inst.mirror_axis}, as={inst.mirror_name})"
                )
            body_lines: list[str] = []
            if inst.stable_id is not None:
                body_lines.append(f'{pad}  id: "{inst.stable_id}"')
            if inst.material_bindings:
                mb = ", ".join(f"{k}: {v}" for k, v in inst.material_bindings)
                body_lines.append(f"{pad}  materials: {{ {mb} }}")
            if inst.uv_policy is not None:
                body_lines.append(f"{pad}  uv_policy: {inst.uv_policy}")
            if body_lines:
                lines.append(f"{pad}instance {inst.name} : {inst.component_name}({p_call}){mir_s} {{")
                lines.extend(body_lines)
                lines.append(f"{pad}}}")
            else:
                lines.append(f"{pad}instance {inst.name} : {inst.component_name}({p_call}){mir_s} {{}}")

        def _write_conn(c: ConnectionDecl, pad: str) -> None:
            cp = [f"type: {c.conn_type}"]
            if c.conn_type in ("revolute", "prismatic"):
                cp.append(f"axis: {c.axis}")
                if c.limits is not None:
                    if _is_continuous_limit_pair(c.limits):
                        cp.append("limits: continuous")
                    else:
                        cp.append(f"limits: [{_fmt_num(c.limits[0])}, {_fmt_num(c.limits[1])}]")
            if c.mode != "align":
                cp.append(f"mode: {c.mode}")
            if not _is_zero_ve(c.offset_pos):
                cp.append(f"offset_pos: {_fmt_vec_e(c.offset_pos)}")
            if not _is_zero_ve(c.offset_rot):
                cp.append(f"offset_rot: {_fmt_vec_e(c.offset_rot)}")
            if c.sculpted_blend:
                cp.append("sculpted_blend: true")
            if c.preserve_bone:
                cp.append("preserve_bone: true")
            lines.append(
                f"{pad}connect {c.child_instance}.{c.child_frame} to {c.parent_instance}.{c.parent_frame} {{ {', '.join(cp)} }}"
            )

        def _write_sub(sub: SubassemblyDecl, indent: int) -> None:
            pad = " " * indent
            lines.append(f"{pad}subassembly {sub.name} {{")
            for it in sub.instances:
                _write_inst(it, pad + "  ")
            for ch_s in sub.subassemblies:
                _write_sub(ch_s, indent + 2)
            for cn in sub.connections:
                _write_conn(cn, pad + "  ")
            for bd in sub.binds:
                lines.append(f"{pad}  bind {bd.instance_name} to {bd.target_bone}")
            lines.append(f"{pad}}}")

        for it in asm.instances:
            _write_inst(it, "  ")
        for sb in asm.subassemblies:
            _write_sub(sb, 2)
        for cn in asm.connections:
            _write_conn(cn, "  ")
        for bd in asm.binds:
            lines.append(f"  bind {bd.instance_name} to {bd.target_bone}")
        lines.append("}")

    return lines

