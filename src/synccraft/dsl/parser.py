"""Dependency-free, bounded recursive-descent parser & compact canonical formatter for Block-DSL v3.2.

No Python evaluation (`eval`/`exec`), code execution, imports, or permissive unknown fields.
Accepts both legacy verbose v3.0/v3.1 programs and compact Block-DSL v3.2 constructs:
- Versioned ``@meta`` (``dsl_version``, ``size_step``, ``audit_profile``)
- ``@textures { <id>: image("<path>", <w>, <h>) }``
- Compile-time ``let <name> = <num_expr>`` constants (bounded arithmetic, zero ``eval``)
- Reusable ``def <macro>(params...) { ... }`` + ``use <macro>(args...) as <prefix> ...``
- Segmented ``chain <name> [attach|joint] segments=K [taper=0.85] { ... }`` lowering
- Zero-thickness ``plane <name> { size: [w, h], normal: x|y|z, material: ... }``
- Per-box ``rotate``, ``pivot``, ``uv: box(u, v)``, ``uv_faces``, ``texture``, fractional ``size``
- First-class ``joint(parent=..., at=[x, y, z])`` alongside ``attach(...)``
- ``mirror(axis=x|z)`` and ``radial(count=K, axis=y)`` symmetry expansions
- Generalized N-bone ``ik_chain { bones: [...], effector: [...], pole: [...] }``
- ``key <bone>.<rotation|position|scale> { ... }``, ``spring <bone> { ... }``, and ``@controller``
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
from typing import Any

from .ast_nodes import (
    ALLOWED_SIZE_STEPS,
    AXES_3D,
    CONTROLLER_OPS,
    DSLValidationError,
    FACE_ORDER,
    FACES,
    KEYFRAME_INTERPOLATIONS,
    Attach,
    BehaviorDecl,
    BehaviorStateDecl,
    Bone,
    Box,
    ClipDecl,
    ControllerDecl,
    ControllerTransitionDecl,
    IKChainDecl,
    IKDriveDecl,
    Joint,
    JointLimits,
    KeyframeTrackDecl,
    KinematicsDecl,
    Locator,
    LookChainDecl,
    MaterialDecl,
    MaterialModDecl,
    Meta,
    MuscleDecl,
    MuscleDriveDecl,
    Palette,
    Place,
    Plane,
    Program,
    RadialSpec,
    SpringDecl,
    StyleDecl,
    TendonCoupling,
    TextureDecl,
    TrackDecl,
    UVRect,
    UnsupportedFeature,
    Vec3,
    validate_program,
)


class DSLParseError(ValueError):
    def __init__(self, message: str, line: int, column: int):
        self.line, self.column = line, column
        super().__init__(f"line {line}, column {column}: {message}")


@dataclass(frozen=True)
class _Token:
    value: str
    kind: str
    line: int
    column: int


@dataclass(frozen=True)
class _MacroDef:
    name: str
    params: tuple[tuple[str, int | float | None], ...]
    body_tokens: tuple[_Token, ...]
    defined_env: dict[str, int | float]


_LEXER = re.compile(
    r"(?P<SPACE>\s+)|(?P<COMMENT>//[^\n]*|\#[^\n]*)|"
    r'(?P<STRING>"(?:[^"\\]|\\.)*")|'
    r"(?P<ARROW>->)|(?P<CMP>>=|<=|==|!=|>|<)|"
    r"(?P<NUMBER>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)|(?P<PUNCT>[@{}()\[\]:,=.;*/+-])"
)


def _tokens(text: str) -> list[_Token]:
    if len(text.encode("utf-8")) > 2_000_000:
        raise DSLParseError("input exceeds 2 MB starter limit", 1, 1)
    result: list[_Token] = []
    pos, line, column = 0, 1, 1
    while pos < len(text):
        match = _LEXER.match(text, pos)
        if not match:
            raise DSLParseError(f"unexpected character {text[pos]!r}", line, column)
        word = match.group(0)
        if match.lastgroup not in {"SPACE", "COMMENT"}:
            result.append(_Token(word, str(match.lastgroup), line, column))
        newlines = word.count("\n")
        if newlines:
            line += newlines
            column = len(word.rsplit("\n", 1)[1]) + 1
        else:
            column += len(word)
        pos = match.end()
        if len(result) > 200_000:
            raise DSLParseError("input exceeds 200,000-token starter limit", line, column)
    result.append(_Token("<EOF>", "EOF", line, column))
    return result


def _prefix_macro_subtree(
    node: Bone,
    prefix: str,
    *,
    is_macro_root: bool = False,
    local_rename_map: dict[str, str] | None = None,
) -> Bone:
    """Prefix internal macro identifiers with ``{prefix}_`` and remap local references."""
    parent_map = dict(local_rename_map or {})
    own_map: dict[str, str] = {}

    for it in node.items:
        if isinstance(it, (Box, Locator)):
            own_map[it.name] = f"{prefix}_{it.name}"

    new_items: list[Box | Locator | Bone] = []
    for it in node.items:
        if isinstance(it, Box):
            new_place = it.place
            if new_place is not None and new_place.base_box is not None:
                mapped_base = own_map.get(new_place.base_box, new_place.base_box)
                new_place = replace(new_place, base_box=mapped_base)
            new_items.append(replace(it, name=own_map[it.name], place=new_place))
        elif isinstance(it, Locator):
            new_items.append(replace(it, name=own_map[it.name]))
        elif isinstance(it, Bone):
            new_items.append(
                _prefix_macro_subtree(
                    it,
                    prefix,
                    is_macro_root=False,
                    local_rename_map=own_map,
                )
            )

    new_name = prefix if is_macro_root else f"{prefix}_{node.name}"
    new_mirror = (
        node.mirror_name
        if is_macro_root
        else (f"{prefix}_{node.mirror_name}" if node.mirror_name else None)
    )
    new_attach = node.attach
    if not is_macro_root and new_attach is not None:
        mapped_parent = parent_map.get(new_attach.parent_box, new_attach.parent_box)
        new_attach = replace(new_attach, parent_box=mapped_parent)

    return replace(
        node,
        name=new_name,
        attach=new_attach,
        mirror_name=new_mirror,
        items=new_items,
    )


class _Parser:
    def __init__(
        self,
        text: str | None = None,
        *,
        tokens: list[_Token] | None = None,
        env: dict[str, int | float] | None = None,
        macros: dict[str, _MacroDef] | None = None,
        size_step: float = 1.0,
        macro_depth: int = 0,
    ):
        self.tokens = tokens if tokens is not None else _tokens(text or "")
        self.index = 0
        self.env: dict[str, int | float] = dict(env or {})
        self.macros: dict[str, _MacroDef] = dict(macros or {})
        self.size_step: float = size_step
        self.macro_depth: int = macro_depth

    @property
    def token(self) -> _Token:
        return self.tokens[self.index]

    def peek(self, offset: int = 1) -> _Token:
        idx = min(len(self.tokens) - 1, self.index + offset)
        return self.tokens[idx]

    def at(self, value: str) -> bool:
        return self.token.value == value

    def fail(self, message: str):
        raise DSLParseError(message, self.token.line, self.token.column)

    def take(self, value: str) -> str:
        if not self.at(value):
            self.fail(f"expected {value!r}, got {self.token.value!r}")
        self.index += 1
        return value

    def maybe_take(self, value: str) -> bool:
        if self.at(value):
            self.index += 1
            return True
        return False

    def ident(self) -> str:
        if self.token.kind != "IDENT":
            self.fail(f"expected identifier, got {self.token.value!r}")
        value = self.token.value
        self.index += 1
        return value

    def string_literal(self) -> str:
        if self.token.kind != "STRING":
            self.fail(f"expected quoted string literal, got {self.token.value!r}")
        raw = self.token.value
        self.index += 1
        try:
            parsed = json.loads(raw)
        except Exception:
            self.fail(f"invalid string literal {raw}")
        if not isinstance(parsed, str):
            self.fail("expected string literal")
        return parsed

    def _expr_add(self, depth: int) -> tuple[int | float, bool]:
        if depth > 32:
            self.fail("arithmetic expression nesting exceeds 32 levels")
        val, is_pure_float_lit = self._expr_mul(depth + 1)
        while self.token.kind == "PUNCT" and self.token.value in ("+", "-"):
            op = self.token.value
            self.index += 1
            rhs, _ = self._expr_mul(depth + 1)
            is_pure_float_lit = False
            if op == "+":
                val = val + rhs
            else:
                val = val - rhs
        return val, is_pure_float_lit

    def _expr_mul(self, depth: int) -> tuple[int | float, bool]:
        val, is_pure_float_lit = self._expr_unary(depth + 1)
        while self.token.kind == "PUNCT" and self.token.value in ("*", "/"):
            op = self.token.value
            op_tok = self.token
            self.index += 1
            rhs, _ = self._expr_unary(depth + 1)
            is_pure_float_lit = False
            if op == "*":
                val = val * rhs
            else:
                if float(rhs) == 0.0:
                    raise DSLParseError("division by zero in numeric expression", op_tok.line, op_tok.column)
                div_val = float(val) / float(rhs)
                if type(val) is int and type(rhs) is int and div_val.is_integer():
                    val = int(div_val)
                else:
                    val = div_val
        return val, is_pure_float_lit

    def _expr_unary(self, depth: int) -> tuple[int | float, bool]:
        if depth > 32:
            self.fail("arithmetic expression nesting exceeds 32 levels")
        if self.token.kind == "PUNCT" and self.token.value in ("+", "-"):
            sign = -1 if self.token.value == "-" else 1
            self.index += 1
            sub, is_pure_float_lit = self._expr_unary(depth + 1)
            return (sign * sub), is_pure_float_lit
        return self._expr_primary(depth + 1)

    def _expr_primary(self, depth: int) -> tuple[int | float, bool]:
        tok = self.token
        if tok.kind == "NUMBER":
            self.index += 1
            if re.fullmatch(r"\d+", tok.value):
                return int(tok.value), False
            return float(tok.value), True
        if tok.kind == "IDENT" and tok.value not in ("true", "false"):
            if tok.value not in self.env:
                raise DSLValidationError(f"undefined compile-time constant or parameter {tok.value!r}")
            self.index += 1
            bound = self.env[tok.value]
            return bound, False
        if tok.value == "(":
            self.index += 1
            inner, _ = self._expr_add(depth + 1)
            self.take(")")
            return inner, False
        self.fail(f"expected number or constant expression, got {tok.value!r}")
        raise AssertionError("unreachable")

    def number(self, integer: bool = False) -> int | float:
        start_tok = self.token
        val, is_pure_float_lit = self._expr_add(0)
        if integer:
            if is_pure_float_lit:
                raise DSLParseError(
                    "expected integer literal; fractional and exponent spellings are not integer sizes",
                    start_tok.line,
                    start_tok.column,
                )
            if type(val) is int:
                return val
            if isinstance(val, float) and math.isfinite(val) and val.is_integer():
                return int(round(val))
            raise DSLParseError(
                "expected integer value; expression evaluated to non-integer",
                start_tok.line,
                start_tok.column,
            )
        return float(val)

    def size_number(self) -> int | float:
        """Parse a box/plane size component honoring current ``self.size_step``."""
        if self.size_step == 1.0:
            return int(self.number(integer=True))
        val = float(self.number(integer=False))
        return val

    def boolean(self) -> bool:
        if not self.at("true") and not self.at("false"):
            self.fail("expected true or false")
        result = self.at("true")
        self.index += 1
        return result

    def vector(self, count: int = 3, integer: bool = False):
        self.take("[")
        result = []
        for i in range(count):
            if i:
                self.take(",")
            result.append(self.number(integer))
        self.take("]")
        return tuple(result)

    def size_vector(self, allow_2d: bool = False) -> tuple[int | float, ...]:
        self.take("[")
        first = self.size_number()
        self.take(",")
        second = self.size_number()
        if allow_2d and self.at("]"):
            self.take("]")
            return (first, second)
        self.take(",")
        third = self.size_number()
        self.take("]")
        return (first, second, third)

    def ident_list(self) -> tuple[str, ...]:
        self.take("[")
        items = [self.ident()]
        while self.at(","):
            self.take(",")
            items.append(self.ident())
        self.take("]")
        return tuple(items)

    def joint_limits_matrix(self) -> JointLimits:
        self.take("[")
        rx = self.vector(2)
        self.take(",")
        ry = self.vector(2)
        self.take(",")
        rz = self.vector(2)
        self.take("]")
        return ((float(rx[0]), float(rx[1])), (float(ry[0]), float(ry[1])), (float(rz[0]), float(rz[1])))

    def field(self, name: str, separator: str = ":") -> None:
        self.take(name)
        self.take(separator)

    def section(self, name: str) -> None:
        self.take("@")
        self.take(name)
        self.take("{")

    def _consume_preamble_decls(self, allow_def: bool = True) -> None:
        while self.at("let") or (allow_def and self.at("def")):
            if self.at("let"):
                self.let_decl()
            else:
                self.macro_def()

    def let_decl(self) -> None:
        self.take("let")
        name = self.ident()
        self.take("=")
        val, _ = self._expr_add(0)
        if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise DSLValidationError(f"let {name}: value must be finite")
        self.env[name] = val
        self.maybe_take(";")

    def macro_def(self) -> None:
        self.take("def")
        name = self.ident()
        if name in self.macros:
            self.fail(f"duplicate macro definition {name!r}")
        self.take("(")
        params: list[tuple[str, int | float | None]] = []
        seen_p: set[str] = set()
        if not self.at(")"):
            while True:
                pname = self.ident()
                if pname in seen_p:
                    self.fail(f"duplicate macro parameter {pname!r}")
                seen_p.add(pname)
                default_val: int | float | None = None
                if self.maybe_take("="):
                    dv, _ = self._expr_add(0)
                    default_val = dv
                params.append((pname, default_val))
                if not self.maybe_take(","):
                    break
        self.take(")")
        open_tok = self.token
        self.take("{")
        depth_b = 1
        body_start = self.index
        while depth_b > 0:
            if self.token.kind == "EOF":
                raise DSLParseError(f"unclosed macro body for {name!r}", open_tok.line, open_tok.column)
            if self.at("{"):
                depth_b += 1
            elif self.at("}"):
                depth_b -= 1
                if depth_b == 0:
                    break
            self.index += 1
        body_end = self.index
        self.take("}")
        self.macros[name] = _MacroDef(
            name=name,
            params=tuple(params),
            body_tokens=tuple(self.tokens[body_start:body_end]),
            defined_env=dict(self.env),
        )

    def _parse_meta_body(self) -> Meta:
        archetype: str | None = None
        profile: str | None = None
        density: int | None = None
        dsl_version = "3.2"
        size_step = 1.0
        audit_profile = "strict"
        seen_fields: set[str] = set()
        while not self.at("}"):
            key = self.ident()
            if key in seen_fields:
                self.fail(f"duplicate @meta field {key!r}")
            seen_fields.add(key)
            self.take(":")
            if key == "archetype":
                archetype = self.ident()
            elif key == "target_profile":
                profile = self.ident()
            elif key == "texel_density":
                density = int(self.number(integer=True))
            elif key == "dsl_version":
                if self.token.kind == "STRING":
                    dsl_version = self.string_literal()
                elif self.token.kind == "NUMBER":
                    dsl_version = self.token.value
                    self.index += 1
                else:
                    dsl_version = self.ident()
            elif key == "size_step":
                size_step = float(self.number(integer=False))
                if size_step not in ALLOWED_SIZE_STEPS:
                    raise DSLValidationError(f"meta.size_step must be one of {sorted(ALLOWED_SIZE_STEPS)}")
            elif key == "audit_profile":
                audit_profile = self.string_literal() if self.token.kind == "STRING" else self.ident()
            else:
                self.fail(f"unexpected @meta field {key!r}")
            if not self.maybe_take(","):
                break
        self.take("}")
        for req_name, req_val in (
            ("archetype", archetype),
            ("target_profile", profile),
            ("texel_density", density),
        ):
            if req_val is None:
                self.fail(f"missing required @meta field {req_name!r}")
        self.size_step = size_step
        return Meta(
            archetype=archetype,  # type: ignore[arg-type]
            target_profile=profile,  # type: ignore[arg-type]
            texel_density=density,  # type: ignore[arg-type]
            dsl_version=dsl_version,
            size_step=size_step,
            audit_profile=audit_profile,
        )

    def textures_decl(self) -> list[TextureDecl]:
        self.take("{")
        decls: list[TextureDecl] = []
        while not self.at("}"):
            tname = self.ident()
            self.take(":")
            self.take("image")
            self.take("(")
            path = self.string_literal()
            self.take(",")
            width = int(self.number(integer=True))
            self.take(",")
            height = int(self.number(integer=True))
            self.take(")")
            self.maybe_take(",")
            decls.append(TextureDecl(name=tname, path=path, width=width, height=height))
        self.take("}")
        return decls

    def program(self) -> Program:
        self._consume_preamble_decls(allow_def=True)
        self.section("meta")
        meta = self._parse_meta_body()

        self._consume_preamble_decls(allow_def=True)
        textures: list[TextureDecl] | None = None
        if self.at("@") and self.peek().value == "textures":
            self.take("@")
            self.take("textures")
            textures = self.textures_decl()
            self._consume_preamble_decls(allow_def=True)

        self.section("palette")
        palettes = []
        while not self.at("}"):
            if self.at("let"):
                self.let_decl()
                continue
            palettes.append(self.palette())
        self.take("}")

        self._consume_preamble_decls(allow_def=True)
        if self.at("@") and self.peek().value == "textures":
            if textures is not None:
                self.fail("duplicate @textures section")
            self.take("@")
            self.take("textures")
            textures = self.textures_decl()
            self._consume_preamble_decls(allow_def=True)

        self.section("rig")
        self._consume_preamble_decls(allow_def=True)
        self.take("root")
        root = self.bone(root=True, depth=0)
        self._consume_preamble_decls(allow_def=True)
        self.take("}")

        materials: list[MaterialDecl] | None = None
        style: StyleDecl | None = None
        muscles: list[MuscleDecl] | None = None
        kinematics: KinematicsDecl | None = None
        behavior: BehaviorDecl | None = None
        controller: ControllerDecl | None = None

        while not self.at("<EOF>"):
            if self.at("let") or self.at("def"):
                self._consume_preamble_decls(allow_def=True)
                continue
            if not self.at("@"):
                break
            self.take("@")
            feature = self.ident()
            if feature in {"shade", "ergonomics", "motion"}:
                raise UnsupportedFeature(
                    f"@{feature} is reserved by v3 but not implemented in the static starter; "
                    f"remove it only for an explicitly simplified asset"
                )
            if feature == "textures":
                if textures is not None:
                    self.fail("duplicate @textures section")
                textures = self.textures_decl()
            elif feature == "materials":
                if materials is not None:
                    self.fail("duplicate @materials section")
                self.take("{")
                materials = []
                while not self.at("}"):
                    if self.at("let"):
                        self.let_decl()
                        continue
                    materials.append(self.material_decl())
                self.take("}")
            elif feature == "style":
                if style is not None:
                    self.fail("duplicate @style section")
                style = self.style_decl()
            elif feature == "muscles":
                if muscles is not None:
                    self.fail("duplicate @muscles section")
                self.take("{")
                muscles = []
                while not self.at("}"):
                    if self.at("let"):
                        self.let_decl()
                        continue
                    muscles.append(self.muscle_decl())
                self.take("}")
            elif feature == "kinematics":
                if kinematics is not None:
                    self.fail("duplicate @kinematics section")
                kinematics = self.kinematics_decl()
            elif feature == "behavior":
                if behavior is not None:
                    self.fail("duplicate @behavior section")
                behavior = self.behavior_decl()
            elif feature == "controller":
                if controller is not None:
                    self.fail("duplicate @controller section")
                controller = self.controller_decl()
            else:
                self.fail(f"unknown section @{feature}")
        self.take("<EOF>")
        return validate_program(
            Program(
                meta=meta,
                palettes=palettes,
                root=root,
                materials=tuple(materials) if materials else (),
                style=style,
                muscles=tuple(muscles) if muscles else (),
                kinematics=kinematics,
                behavior=behavior,
                textures=tuple(textures) if textures else (),
                controller=controller,
            )
        )

    def material_decl(self) -> MaterialDecl:
        name = self.ident()
        self.take(":")
        self.take("process")
        self.take("(")
        self.field("family", "=")
        family = self.ident()
        kwargs: dict[str, object] = {}
        allowed = {
            "roughness",
            "metalness",
            "coating",
            "ior",
            "scale",
            "seed",
            "anisotropy",
            "subsurface",
            "porosity",
            "emissive",
            "rd_feed",
            "rd_kill",
            "film_nm",
            "wear",
            "oxidation",
            "biofilm",
            "weld_group",
        }
        while self.at(","):
            self.take(",")
            key = self.ident()
            if key not in allowed or key in kwargs:
                self.fail(f"unexpected or duplicate @materials field {key!r}")
            self.take("=")
            if key == "weld_group":
                wg = self.ident()
                kwargs[key] = None if wg == "none" else wg
            else:
                kwargs[key] = self.number(integer=(key == "seed"))
        self.take(")")
        return MaterialDecl(
            name=name,
            family=family,
            roughness=float(kwargs.get("roughness", 0.35)),  # type: ignore[arg-type]
            metalness=float(kwargs.get("metalness", 0.0)),  # type: ignore[arg-type]
            coating=float(kwargs.get("coating", 0.0)),  # type: ignore[arg-type]
            ior=float(kwargs.get("ior", 1.5)),  # type: ignore[arg-type]
            scale=float(kwargs.get("scale", 1.0)),  # type: ignore[arg-type]
            seed=int(kwargs.get("seed", 101)),  # type: ignore[arg-type]
            anisotropy=float(kwargs.get("anisotropy", 1.0)),  # type: ignore[arg-type]
            subsurface=float(kwargs.get("subsurface", 0.0)),  # type: ignore[arg-type]
            porosity=float(kwargs.get("porosity", 0.08)),  # type: ignore[arg-type]
            emissive=float(kwargs.get("emissive", 0.0)),  # type: ignore[arg-type]
            rd_feed=float(kwargs.get("rd_feed", 0.036)),  # type: ignore[arg-type]
            rd_kill=float(kwargs.get("rd_kill", 0.062)),  # type: ignore[arg-type]
            film_nm=float(kwargs.get("film_nm", 380.0)),  # type: ignore[arg-type]
            wear=float(kwargs.get("wear", 0.35)),  # type: ignore[arg-type]
            oxidation=float(kwargs.get("oxidation", 0.25)),  # type: ignore[arg-type]
            biofilm=float(kwargs.get("biofilm", 0.20)),  # type: ignore[arg-type]
            weld_group=kwargs.get("weld_group"),  # type: ignore[arg-type]
        )

    def style_decl(self) -> StyleDecl:
        self.take("{")
        self.field("target_encoding")
        target_enc = self.ident()
        kwargs: dict[str, object] = {}
        allowed = {"quantize_ramp", "bevel_highlight", "view_deg"}
        while self.at(","):
            self.take(",")
            key = self.ident()
            if key not in allowed or key in kwargs:
                self.fail(f"unexpected or duplicate @style field {key!r}")
            self.take(":")
            if key == "quantize_ramp":
                kwargs[key] = self.boolean()
            else:
                kwargs[key] = self.number()
        self.take("}")
        return StyleDecl(
            target_encoding=target_enc,
            quantize_ramp=bool(kwargs.get("quantize_ramp", True)),
            bevel_highlight=float(kwargs.get("bevel_highlight", 0.08)),  # type: ignore[arg-type]
            view_deg=float(kwargs.get("view_deg", 20.0)),  # type: ignore[arg-type]
        )

    def muscle_decl(self) -> MuscleDecl:
        self.take("muscle")
        name = self.ident()
        self.take("{")
        self.field("origin_bone")
        origin_bone = self.ident()
        self.take(",")
        self.field("insertion_bone")
        insertion_bone = self.ident()
        self.take(",")
        self.field("action_axis")
        action_axis = self.ident()
        self.take(",")
        self.field("moment_arm_deg")
        moment_arm_deg = self.number()
        tendon: TendonCoupling | None = None
        belly_box: str | None = None
        bulge_inflate = 0.0
        seen_opt: set[str] = set()
        while self.at(","):
            self.take(",")
            key = self.ident()
            if key in seen_opt:
                self.fail(f"duplicate @muscles field {key!r}")
            seen_opt.add(key)
            self.take(":")
            if key == "tendon_coupling":
                self.take("[")
                tc_bone = self.ident()
                self.take(",")
                tc_axis = self.ident()
                self.take(",")
                tc_ratio = self.number()
                self.take("]")
                tendon = TendonCoupling(tc_bone, tc_axis, float(tc_ratio))
            elif key == "belly_box":
                belly_box = self.ident()
            elif key == "bulge_inflate":
                bulge_inflate = float(self.number())
            else:
                self.fail(f"unexpected @muscles field {key!r}")
        self.take("}")
        return MuscleDecl(
            name=name,
            origin_bone=origin_bone,
            insertion_bone=insertion_bone,
            action_axis=action_axis,
            moment_arm_deg=float(moment_arm_deg),
            tendon_coupling=tendon,
            belly_box=belly_box,
            bulge_inflate=bulge_inflate,
        )

    def spring_decl(self) -> SpringDecl:
        self.take("spring")
        bone = self.ident()
        self.take("{")
        stiffness = 40.0
        damping = 6.0
        seen: set[str] = set()
        while not self.at("}"):
            if seen:
                self.maybe_take(",")
                if self.at("}"):
                    break
            k = self.ident()
            if k in seen:
                self.fail(f"duplicate spring field {k!r}")
            seen.add(k)
            self.take(":")
            if k == "stiffness":
                stiffness = float(self.number())
            elif k == "damping":
                damping = float(self.number())
            else:
                self.fail(f"unexpected spring field {k!r}")
        self.take("}")
        return SpringDecl(bone=bone, stiffness=stiffness, damping=damping)

    def kinematics_decl(self) -> KinematicsDecl:
        self.take("{")
        ik_chains: list[IKChainDecl] = []
        look_chains: list[LookChainDecl] = []
        springs: list[SpringDecl] = []
        while not self.at("}"):
            if self.at("let"):
                self.let_decl()
                continue
            if self.at("ik_chain"):
                self.take("ik_chain")
                name = self.ident()
                self.take("{")
                upper = ""
                lower = ""
                bones: tuple[str, ...] = ()
                effector: Vec3 | None = None
                hinge_axis = "x"
                pole: Vec3 | None = None
                seen_k: set[str] = set()
                while not self.at("}"):
                    if seen_k:
                        self.take(",")
                    k = self.ident()
                    if k in seen_k:
                        self.fail(f"duplicate ik_chain field {k!r}")
                    seen_k.add(k)
                    self.take(":")
                    if k == "upper":
                        upper = self.ident()
                    elif k == "lower":
                        lower = self.ident()
                    elif k == "bones":
                        bones = self.ident_list()
                    elif k == "effector":
                        effector = self.vector(3)
                    elif k == "hinge_axis":
                        hinge_axis = self.ident()
                    elif k == "pole":
                        pole = self.vector(3)
                    else:
                        self.fail(f"unexpected ik_chain field {k!r}")
                self.take("}")
                if effector is None:
                    self.fail(f"ik_chain {name}: missing effector field")
                if not bones and (not upper or not lower):
                    self.fail(f"ik_chain {name}: specify either bones: [...] or upper/lower")
                ik_chains.append(
                    IKChainDecl(
                        name=name,
                        upper=upper,
                        lower=lower,
                        effector=effector,
                        hinge_axis=hinge_axis,
                        bones=bones,
                        pole=pole,
                    )
                )
            elif self.at("look_chain"):
                self.take("look_chain")
                name = self.ident()
                self.take("{")
                self.field("bones")
                bones = self.ident_list()
                weight_decay = 0.75
                if self.at(","):
                    self.take(",")
                    self.field("weight_decay")
                    weight_decay = float(self.number())
                self.take("}")
                look_chains.append(LookChainDecl(name=name, bones=bones, weight_decay=weight_decay))
            elif self.at("spring"):
                springs.append(self.spring_decl())
            else:
                self.fail("expected ik_chain, look_chain, spring, or closing brace in @kinematics")
        self.take("}")
        return KinematicsDecl(ik_chains=tuple(ik_chains), look_chains=tuple(look_chains), springs=tuple(springs))

    def behavior_decl(self) -> BehaviorDecl:
        self.take("{")
        clips: list[ClipDecl] = []
        states: list[BehaviorStateDecl] = []
        springs: list[SpringDecl] = []
        while not self.at("}"):
            if self.at("let"):
                self.let_decl()
                continue
            if self.at("clip"):
                clips.append(self.clip_decl())
            elif self.at("state"):
                states.append(self.behavior_state_decl())
            elif self.at("spring"):
                springs.append(self.spring_decl())
            else:
                self.fail("expected clip, state, spring, or closing brace in @behavior")
        self.take("}")
        return BehaviorDecl(clips=tuple(clips), states=tuple(states), springs=tuple(springs))

    def keyframe_track_decl(self) -> KeyframeTrackDecl:
        self.take("key")
        bone = self.ident()
        self.take(".")
        channel = self.ident()
        self.take("{")
        keys: list[tuple[float, Vec3]] = []
        interp = "linear"
        while not self.at("}"):
            if self.token.kind == "IDENT" and self.token.value in KEYFRAME_INTERPOLATIONS:
                interp = self.ident()
                self.maybe_take(",")
                break
            t_val = float(self.number())
            self.take(":")
            vec = self.vector(3)
            keys.append((t_val, (float(vec[0]), float(vec[1]), float(vec[2]))))
            self.maybe_take(",")
        self.take("}")
        return KeyframeTrackDecl(bone=bone, channel=channel, keys=tuple(keys), interpolation=interp)

    def clip_decl(self) -> ClipDecl:
        self.take("clip")
        name = self.ident()
        self.take("{")
        self.field("duration_s")
        duration_s = float(self.number())
        loop = True
        if self.at(","):
            self.take(",")
            self.field("loop")
            loop = self.boolean()
        tracks: list[TrackDecl] = []
        muscle_drives: list[MuscleDriveDecl] = []
        ik_drives: list[IKDriveDecl] = []
        keyframes: list[KeyframeTrackDecl] = []
        while not self.at("}"):
            if self.at("track"):
                self.take("track")
                bone = self.ident()
                self.take(":")
                law = self.ident()
                self.take("(")
                self.field("axis", "=")
                axis = self.ident()
                self.take(",")
                self.field("amp_deg", "=")
                amp_deg = float(self.number())
                self.take(",")
                self.field("freq_hz", "=")
                freq_hz = float(self.number())
                phase_deg = 0.0
                bias_deg = 0.0
                seen_k: set[str] = set()
                while self.at(","):
                    self.take(",")
                    k = self.ident()
                    if k in seen_k:
                        self.fail(f"duplicate track parameter {k!r}")
                    seen_k.add(k)
                    self.take("=")
                    if k == "phase_deg":
                        phase_deg = float(self.number())
                    elif k == "bias_deg":
                        bias_deg = float(self.number())
                    else:
                        self.fail(f"unexpected track parameter {k!r}")
                self.take(")")
                tracks.append(
                    TrackDecl(
                        bone=bone,
                        law=law,
                        axis=axis,
                        amp_deg=amp_deg,
                        freq_hz=freq_hz,
                        phase_deg=phase_deg,
                        bias_deg=bias_deg,
                    )
                )
            elif self.at("key"):
                keyframes.append(self.keyframe_track_decl())
            elif self.at("muscle_drive"):
                self.take("muscle_drive")
                muscle = self.ident()
                self.take(":")
                self.take("activation")
                self.take("(")
                self.field("peak", "=")
                peak = float(self.number())
                self.take(",")
                self.field("freq_hz", "=")
                freq_hz = float(self.number())
                phase_deg = 0.0
                baseline = 0.0
                seen_k = set()
                while self.at(","):
                    self.take(",")
                    k = self.ident()
                    if k in seen_k:
                        self.fail(f"duplicate muscle_drive parameter {k!r}")
                    seen_k.add(k)
                    self.take("=")
                    if k == "phase_deg":
                        phase_deg = float(self.number())
                    elif k == "baseline":
                        baseline = float(self.number())
                    else:
                        self.fail(f"unexpected muscle_drive parameter {k!r}")
                self.take(")")
                muscle_drives.append(
                    MuscleDriveDecl(
                        muscle=muscle, peak=peak, freq_hz=freq_hz, phase_deg=phase_deg, baseline=baseline
                    )
                )
            elif self.at("ik_drive"):
                self.take("ik_drive")
                chain = self.ident()
                self.take(":")
                self.take("foot_cycle")
                self.take("(")
                self.field("stride", "=")
                stride = float(self.number())
                self.take(",")
                self.field("lift", "=")
                lift = float(self.number())
                phase_deg = 0.0
                if self.at(","):
                    self.take(",")
                    self.field("phase_deg", "=")
                    phase_deg = float(self.number())
                self.take(")")
                ik_drives.append(IKDriveDecl(chain=chain, stride=stride, lift=lift, phase_deg=phase_deg))
            else:
                self.fail("expected track, key, muscle_drive, ik_drive, or closing brace in clip")
        self.take("}")
        return ClipDecl(
            name=name,
            duration_s=duration_s,
            loop=loop,
            tracks=tuple(tracks),
            muscle_drives=tuple(muscle_drives),
            ik_drives=tuple(ik_drives),
            keyframes=tuple(keyframes),
        )

    def behavior_state_decl(self) -> BehaviorStateDecl:
        self.take("state")
        name = self.ident()
        self.take("{")
        pose_biases: list[tuple[str, tuple[float, float, float]]] = []
        muscle_tones: list[tuple[str, float]] = []
        material_mods: list[MaterialModDecl] = []
        while not self.at("}"):
            if self.at("pose_bias"):
                self.take("pose_bias")
                bname = self.ident()
                self.take(":")
                vec = self.vector(3)
                pose_biases.append((bname, vec))
            elif self.at("muscle_tone"):
                self.take("muscle_tone")
                mname = self.ident()
                self.take(":")
                tone = float(self.number())
                muscle_tones.append((mname, tone))
            elif self.at("material_mod"):
                self.take("material_mod")
                mat = self.ident()
                self.take(":")
                self.take("{")
                self.field("emissive")
                emissive = float(self.number())
                self.take(",")
                self.field("moisture")
                moisture = float(self.number())
                self.take(",")
                self.field("wear_delta")
                wear_delta = float(self.number())
                self.take("}")
                material_mods.append(
                    MaterialModDecl(material=mat, emissive=emissive, moisture=moisture, wear_delta=wear_delta)
                )
            else:
                self.fail("expected pose_bias, muscle_tone, material_mod, or closing brace in state")
        self.take("}")
        return BehaviorStateDecl(
            name=name,
            pose_biases=tuple(pose_biases),
            muscle_tones=tuple(muscle_tones),
            material_mods=tuple(material_mods),
        )

    def controller_decl(self) -> ControllerDecl:
        self.take("{")
        initial: str | None = None
        transitions: list[ControllerTransitionDecl] = []
        if self.at("initial") and self.peek().value == ":":
            self.take("initial")
            self.take(":")
            initial = self.ident()
            self.maybe_take(",")
        while not self.at("}"):
            src = self.ident()
            if self.token.kind != "ARROW":
                self.fail(f"expected '->' in @controller transition, got {self.token.value!r}")
            self.index += 1
            dst = self.ident()
            self.take("when")
            param = self.ident()
            if self.token.kind != "CMP" or self.token.value not in CONTROLLER_OPS:
                self.fail(f"expected comparison operator in @controller, got {self.token.value!r}")
            op = self.token.value
            self.index += 1
            thresh = float(self.number())
            blend_s = 0.2
            if self.maybe_take(","):
                if self.at("blend"):
                    self.take("blend")
                    if self.at(":") or self.at("="):
                        self.index += 1
                    blend_s = float(self.number())
                    self.maybe_take(",")
            elif self.at("blend"):
                self.take("blend")
                if self.at(":") or self.at("="):
                    self.index += 1
                blend_s = float(self.number())
            self.maybe_take(";")
            transitions.append(
                ControllerTransitionDecl(
                    from_state=src,
                    to_state=dst,
                    parameter=param,
                    operator=op,
                    threshold=thresh,
                    blend_s=blend_s,
                )
            )
        self.take("}")
        return ControllerDecl(transitions=tuple(transitions), initial=initial)

    def palette(self) -> Palette:
        name = self.ident()
        self.take(":")
        self.take("ramp")
        self.take("(")
        self.field("base_lch", "=")
        lch = self.vector()
        self.take(",")
        self.field("steps", "=")
        steps = int(self.number(integer=True))
        hue_shift, emissive = (0.0, 0.0), False
        if self.at(","):
            self.take(",")
            if self.at("hue_shift"):
                self.field("hue_shift", "=")
                hue_shift = self.vector(2)
                if self.at(","):
                    self.take(",")
                    self.field("emissive", "=")
                    emissive = self.boolean()
            else:
                self.field("emissive", "=")
                emissive = self.boolean()
        self.take(")")
        return Palette(name, lch, steps, hue_shift, emissive)

    def attach_or_joint(self) -> Attach:
        if self.at("joint"):
            self.take("joint")
            self.take("(")
            self.field("parent", "=")
            parent = self.ident()
            at_vec: Vec3 = (0.0, 0.0, 0.0)
            align_vec: Vec3 = (0.0, 0.0, 0.0)
            detached = False
            seen_j: set[str] = set()
            while self.maybe_take(","):
                k = self.ident()
                if k in seen_j:
                    self.fail(f"duplicate joint(...) field {k!r}")
                seen_j.add(k)
                self.take("=")
                if k == "at":
                    v = self.vector(3)
                    at_vec = (float(v[0]), float(v[1]), float(v[2]))
                elif k == "align":
                    v = self.vector(3)
                    align_vec = (float(v[0]), float(v[1]), float(v[2]))
                elif k == "detached":
                    detached = self.boolean()
                else:
                    self.fail(f"unexpected joint(...) field {k!r}")
            self.take(")")
            return Joint(parent=parent, at=at_vec, align=align_vec, detached=detached)

        self.take("attach")
        self.take("(")
        self.field("parent_box", "=")
        parent_box = self.ident()
        self.take(",")
        self.field("anchor", "=")
        anchor = self.vector()
        self.take(",")
        self.field("align", "=")
        align = self.vector()
        offset, detached = (0.0, 0.0, 0.0), False
        if self.at(","):
            self.take(",")
            if self.at("offset"):
                self.field("offset", "=")
                offset = self.vector()
                if self.at(","):
                    self.take(",")
                    self.field("detached", "=")
                    detached = self.boolean()
            else:
                self.field("detached", "=")
                detached = self.boolean()
        self.take(")")
        return Attach(parent_box, anchor, align, offset, detached)

    def _parse_bone_symmetry_modifiers(
        self, bone_name: str
    ) -> tuple[str | None, str, RadialSpec | None]:
        mirror_name: str | None = None
        mirror_axis = "x"
        radial: RadialSpec | None = None
        while self.at("mirror_x") or self.at("mirror") or self.at("radial"):
            if self.at("mirror_x"):
                if mirror_name is not None:
                    self.fail("duplicate mirror modifier on bone")
                self.take("mirror_x")
                self.take("(")
                mirror_name = self.ident()
                mirror_axis = "x"
                self.take(")")
            elif self.at("mirror"):
                if mirror_name is not None:
                    self.fail("duplicate mirror modifier on bone")
                self.take("mirror")
                self.take("(")
                axis_val = "x"
                explicit_name: str | None = None
                while not self.at(")"):
                    if self.peek().value == "=":
                        k = self.ident()
                        self.take("=")
                        if k == "axis":
                            axis_val = self.ident()
                        elif k in ("as", "name"):
                            explicit_name = self.ident()
                        else:
                            self.fail(f"unexpected mirror(...) argument {k!r}")
                    else:
                        explicit_name = self.ident()
                    if not self.maybe_take(","):
                        break
                self.take(")")
                mirror_axis = axis_val
                mirror_name = explicit_name or f"{bone_name}_mirror"
            elif self.at("radial"):
                if radial is not None:
                    self.fail("duplicate radial modifier on bone")
                self.take("radial")
                self.take("(")
                count_val: int | None = None
                rad_axis = "y"
                while not self.at(")"):
                    k = self.ident()
                    self.take("=")
                    if k == "count":
                        count_val = int(self.number(integer=True))
                    elif k == "axis":
                        rad_axis = self.ident()
                    else:
                        self.fail(f"unexpected radial(...) argument {k!r}")
                    if not self.maybe_take(","):
                        break
                self.take(")")
                if count_val is None:
                    self.fail("radial(...) requires count=K")
                radial = RadialSpec(count=count_val, axis=rad_axis)
        return mirror_name, mirror_axis, radial

    def place(self) -> Place:
        self.take("place")
        self.take("(")
        base = None
        if self.at("base_box"):
            self.field("base_box", "=")
            base = self.ident()
            self.take(",")
        self.field("anchor", "=")
        anchor = self.vector()
        self.take(",")
        self.field("align", "=")
        align = self.vector()
        offset = (0.0, 0.0, 0.0)
        if self.at(","):
            self.take(",")
            self.field("offset", "=")
            offset = self.vector()
        self.take(")")
        return Place(base, anchor, align, offset)

    def _parse_uv_faces_dict(self) -> tuple[tuple[str, UVRect], ...]:
        self.take("{")
        entries: dict[str, UVRect] = {}
        while not self.at("}"):
            fname = self.ident()
            if fname in entries:
                self.fail(f"duplicate face {fname!r} in uv_faces")
            self.take(":")
            self.take("[")
            u0 = float(self.number())
            self.take(",")
            v0 = float(self.number())
            self.take(",")
            u1 = float(self.number())
            self.take(",")
            v1 = float(self.number())
            if self.maybe_take(","):
                rot_val = int(self.number(integer=True)) % 360
                self.take("]")
                if rot_val not in (0, 90, 180, 270):
                    raise DSLValidationError(
                        f"uv_faces.{fname}: UV rotation must be one of 0, 90, 180, 270; got {rot_val}"
                    )
                entries[fname] = (u0, v0, u1, v1) if rot_val == 0 else (u0, v0, u1, v1, rot_val)
            else:
                self.take("]")
                entries[fname] = (u0, v0, u1, v1)
            self.maybe_take(",")
        self.take("}")
        order_idx = {f: idx for idx, f in enumerate(FACE_ORDER)}
        return tuple(
            (f, entries[f]) for f in sorted(entries, key=lambda key: (order_idx.get(key, 99), key))
        )

    def _parse_box_or_plane_fields(
        self,
        name: str,
        place: Place | None,
        *,
        is_plane_kw: bool,
    ) -> Box:
        self.take("{")
        size_raw: tuple[int | float, ...] | None = None
        material: str | None = None
        inflate = 0.0
        cullface: tuple[str, ...] = ()
        rotate: Vec3 = (0.0, 0.0, 0.0)
        pivot: Vec3 | None = None
        plane_normal: str | None = None
        uv: tuple[float, float] | None = None
        uv_faces: tuple[tuple[str, UVRect], ...] = ()
        texture: str | None = None
        blend_with: tuple[str, ...] = ()
        seen_f: set[str] = set()

        while not self.at("}"):
            if seen_f:
                self.take(",")
                if self.at("}"):
                    break
            key = self.ident()
            canon_key = "material" if key == "mat" else ("plane_normal" if key == "normal" else key)
            if canon_key in seen_f:
                self.fail(f"duplicate box/plane field {key!r}")
            seen_f.add(canon_key)
            self.take(":")
            if canon_key == "size":
                size_raw = self.size_vector(allow_2d=is_plane_kw)
            elif canon_key == "material":
                material = self.ident()
            elif canon_key == "inflate":
                inflate = float(self.number())
            elif canon_key == "cullface":
                cullface = self.ident_list()
            elif canon_key == "blend_with":
                blend_with = self.ident_list()
            elif canon_key == "rotate":
                rv = self.vector(3)
                rotate = (float(rv[0]), float(rv[1]), float(rv[2]))
            elif canon_key == "pivot":
                pv = self.vector(3)
                pivot = (float(pv[0]), float(pv[1]), float(pv[2]))
            elif canon_key == "plane_normal":
                plane_normal = self.ident()
            elif canon_key == "uv":
                if self.at("box"):
                    self.take("box")
                    self.take("(")
                    u0 = float(self.number())
                    self.take(",")
                    v0 = float(self.number())
                    self.take(")")
                    uv = (u0, v0)
                else:
                    uv_v = self.vector(2)
                    uv = (float(uv_v[0]), float(uv_v[1]))
            elif canon_key == "uv_faces":
                uv_faces = self._parse_uv_faces_dict()
            elif canon_key == "texture":
                texture = self.ident()
            elif canon_key == "place" and place is None:
                place = self.place()
            else:
                self.fail(f"unexpected box/plane field {key!r}")
        self.take("}")

        if size_raw is None or material is None:
            self.fail(f"{'plane' if is_plane_kw else 'box'} {name}: size and material are required")
        if is_plane_kw:
            if plane_normal is None:
                self.fail(f"plane {name}: normal: x|y|z is required")
            return Plane(
                name=name,
                size=size_raw,
                normal=plane_normal,
                material=material,
                place=place,
                inflate=inflate,
                cullface=cullface,
                rotate=rotate,
                pivot=pivot,
                uv=uv,
                uv_faces=uv_faces,
                texture=texture,
                blend_with=blend_with,
            )
        if len(size_raw) != 3:
            self.fail(f"box {name}: size must have 3 components")
        return Box(
            name=name,
            size=(size_raw[0], size_raw[1], size_raw[2]),
            material=material,
            place=place,
            inflate=inflate,
            cullface=cullface,
            rotate=rotate,
            pivot=pivot,
            plane_normal=plane_normal,
            uv=uv,
            uv_faces=uv_faces,
            texture=texture,
            blend_with=blend_with,
        )

    def box(self) -> Box:
        self.take("box")
        name = self.ident()
        place = self.place() if self.at("place") else None
        return self._parse_box_or_plane_fields(name, place, is_plane_kw=False)

    def plane(self) -> Box:
        self.take("plane")
        name = self.ident()
        place = self.place() if self.at("place") else None
        return self._parse_box_or_plane_fields(name, place, is_plane_kw=True)

    def locator(self) -> Locator:
        self.take("locator")
        name = self.ident()
        self.take("{")
        self.field("offset")
        offset = self.vector(3)
        role = "socket"
        if self.at(","):
            self.take(",")
            self.field("role")
            role = self.ident()
        self.take("}")
        return Locator(name=name, offset=offset, role=role)

    def use_macro(self, depth: int) -> Bone:
        if self.macro_depth >= 8:
            self.fail("macro expansion depth exceeds 8")
        self.take("use")
        mname = self.ident()
        if mname not in self.macros:
            raise DSLValidationError(f"undefined macro {mname!r}")
        mdef = self.macros[mname]
        self.take("(")
        pos_args: list[int | float] = []
        kw_args: dict[str, int | float] = {}
        if not self.at(")"):
            while True:
                if self.token.kind == "IDENT" and self.peek().value == "=":
                    pname = self.ident()
                    self.take("=")
                    pv, _ = self._expr_add(0)
                    if pname in kw_args:
                        self.fail(f"duplicate macro argument {pname!r}")
                    kw_args[pname] = pv
                else:
                    if kw_args:
                        self.fail("positional macro argument cannot follow keyword argument")
                    pv, _ = self._expr_add(0)
                    pos_args.append(pv)
                if not self.maybe_take(","):
                    break
        self.take(")")
        self.take("as")
        prefix = self.ident()
        attach = self.attach_or_joint()
        mirror_name, mirror_axis, radial = self._parse_bone_symmetry_modifiers(prefix)

        call_env = dict(mdef.defined_env)
        if len(pos_args) > len(mdef.params):
            self.fail(f"macro {mname!r} takes at most {len(mdef.params)} arguments, got {len(pos_args)}")
        for idx_p, (pname, pdef) in enumerate(mdef.params):
            if idx_p < len(pos_args):
                if pname in kw_args:
                    self.fail(f"multiple values for macro parameter {pname!r}")
                call_env[pname] = pos_args[idx_p]
            elif pname in kw_args:
                call_env[pname] = kw_args.pop(pname)
            elif pdef is not None:
                call_env[pname] = pdef
            else:
                self.fail(f"missing required argument {pname!r} in call to macro {mname!r}")
        if kw_args:
            self.fail(f"unexpected keyword arguments {sorted(kw_args)} in call to macro {mname!r}")

        eof_tok = mdef.body_tokens[-1] if mdef.body_tokens else self.token
        sub_tokens = [*mdef.body_tokens, _Token("}", "PUNCT", eof_tok.line, eof_tok.column), _Token("<EOF>", "EOF", eof_tok.line, eof_tok.column)]
        sub_parser = _Parser(
            tokens=sub_tokens,
            env=call_env,
            macros={k: v for k, v in self.macros.items() if k != mname},
            size_step=self.size_step,
            macro_depth=self.macro_depth + 1,
        )
        sub_parser._consume_preamble_decls(allow_def=False)
        if sub_parser.at("bone") and sub_parser.peek(2).value == "{":
            sub_parser.take("bone")
            _inner_bname = sub_parser.ident()
            sub_parser.take("{")
            raw_bone = sub_parser._parse_bone_interior(
                name=_inner_bname,
                root=False,
                attach=attach,
                mirror=mirror_name,
                mirror_axis=mirror_axis,
                radial=radial,
                depth=depth + 1,
            )
            sub_parser.take("}")
        else:
            raw_bone = sub_parser._parse_bone_interior(
                name=prefix,
                root=False,
                attach=attach,
                mirror=mirror_name,
                mirror_axis=mirror_axis,
                radial=radial,
                depth=depth + 1,
            )
        return _prefix_macro_subtree(raw_bone, prefix, is_macro_root=True)

    def chain_decl(self, enclosing_items: list[Box | Locator | Bone], depth: int) -> Bone:
        self.take("chain")
        name = self.ident()
        attach: Attach | None = None
        segments: int | None = None
        taper = 1.0
        mirror_name: str | None = None
        mirror_axis = "x"
        radial: RadialSpec | None = None

        while not self.at("{"):
            if self.at("attach") or self.at("joint"):
                if attach is not None:
                    self.fail(f"chain {name}: duplicate attach/joint")
                attach = self.attach_or_joint()
            elif self.at("mirror_x") or self.at("mirror") or self.at("radial"):
                m_n, m_ax, rad = self._parse_bone_symmetry_modifiers(name)
                if m_n is not None:
                    mirror_name, mirror_axis = m_n, m_ax
                if rad is not None:
                    radial = rad
            elif self.at("segments"):
                self.take("segments")
                if self.at("=") or self.at(":"):
                    self.index += 1
                segments = int(self.number(integer=True))
                self.maybe_take(",")
            elif self.at("taper"):
                self.take("taper")
                if self.at("=") or self.at(":"):
                    self.index += 1
                taper = float(self.number())
                self.maybe_take(",")
            else:
                self.fail(f"unexpected header token {self.token.value!r} in chain {name}")

        if segments is None or not (1 <= segments <= 32):
            raise DSLValidationError(f"chain {name}: segments must be an integer in [1, 32]")
        if not math.isfinite(taper) or not (0.2 <= taper <= 2.0):
            raise DSLValidationError(f"chain {name}: taper must lie in [0.2, 2.0]")

        if attach is None:
            prev_boxes = [it for it in enclosing_items if isinstance(it, Box)]
            if not prev_boxes:
                self.fail(f"chain {name}: requires attach(...) or joint(...) when no preceding box exists in bone")
            attach = Attach(parent_box=prev_boxes[-1].name, anchor=(0.0, 0.0, -1.0), align=(0.0, 0.0, 1.0))

        self.take("{")
        saved_env = dict(self.env)
        self._consume_preamble_decls(allow_def=False)

        explicit_step_anchor: Vec3 | None = None
        rest: Vec3 = (0.0, 0.0, 0.0)
        joint_limits: JointLimits | None = None
        while self.at("axis") or self.at("rest_euler") or self.at("joint_limits") or self.at("let"):
            if self.at("let"):
                self.let_decl()
                continue
            if self.at("axis"):
                self.field("axis")
                sign_ax = -1.0 if self.maybe_take("-") else (1.0 if self.maybe_take("+") else 1.0)
                ax_ident = self.ident()
                if ax_ident not in AXES_3D:
                    raise DSLValidationError(f"chain {name}: axis must be [+-]x|y|z")
                explicit_step_anchor = (
                    sign_ax if ax_ident == "x" else 0.0,
                    sign_ax if ax_ident == "y" else 0.0,
                    sign_ax if ax_ident == "z" else 0.0,
                )
                self.maybe_take(",")
            elif self.at("rest_euler"):
                self.field("rest_euler")
                rv = self.vector(3)
                rest = (float(rv[0]), float(rv[1]), float(rv[2]))
                self.maybe_take(",")
            elif self.at("joint_limits"):
                self.field("joint_limits")
                joint_limits = self.joint_limits_matrix()
                self.maybe_take(",")

        tmpl_items: list[Box | Locator] = []
        while not self.at("}"):
            if self.at("let"):
                self.let_decl()
                continue
            if self.at("box"):
                tmpl_items.append(self.box())
            elif self.at("plane"):
                tmpl_items.append(self.plane())
            elif self.at("locator"):
                tmpl_items.append(self.locator())
            else:
                self.fail(f"chain {name}: expected box, plane, locator, or closing brace")
        self.take("}")
        self.env = saved_env

        tmpl_boxes = [it for it in tmpl_items if isinstance(it, Box)]
        if not tmpl_boxes:
            raise DSLValidationError(f"chain {name}: template body must contain at least one box or plane")

        if explicit_step_anchor is not None:
            step_anchor = explicit_step_anchor
        else:
            al = attach.align
            nz = [i for i in range(3) if abs(float(al[i])) == 1.0]
            if len(nz) == 1 and sum(abs(float(x)) for x in al) == 1.0:
                step_anchor = (-float(al[0]), -float(al[1]), -float(al[2]))
            else:
                step_anchor = (0.0, 0.0, -1.0)
        step_align = (-step_anchor[0], -step_anchor[1], -step_anchor[2])

        def _scale_size_val(orig: int | float, scale: float) -> int | float:
            if float(orig) == 0.0:
                return orig
            if self.size_step == 1.0:
                return max(1, int(round(float(orig) * scale)))
            steps = max(1, int(round((float(orig) * scale) / self.size_step)))
            return round(steps * self.size_step, 6)

        built_next: Bone | None = None
        for idx_s in range(segments - 1, -1, -1):
            seg_num = idx_s + 1
            seg_bname = f"{name}_{seg_num}"
            scale_k = taper ** idx_s
            rename_local = {it.name: f"{seg_bname}_{it.name}" for it in tmpl_items}
            seg_items: list[Box | Locator | Bone] = []
            for it in tmpl_items:
                if isinstance(it, Box):
                    s_new = (
                        _scale_size_val(it.size[0], scale_k),
                        _scale_size_val(it.size[1], scale_k),
                        _scale_size_val(it.size[2], scale_k),
                    )
                    pl_new = it.place
                    if pl_new is not None:
                        mapped_b = rename_local.get(pl_new.base_box, pl_new.base_box) if pl_new.base_box else None
                        off_new = (
                            float(pl_new.offset[0]) * scale_k,
                            float(pl_new.offset[1]) * scale_k,
                            float(pl_new.offset[2]) * scale_k,
                        )
                        pl_new = Place(mapped_b, pl_new.anchor, pl_new.align, off_new)
                    bw_new = tuple(rename_local.get(bw, bw) for bw in it.blend_with) if it.blend_with else ()
                    seg_items.append(replace(it, name=rename_local[it.name], size=s_new, place=pl_new, blend_with=bw_new))
                elif isinstance(it, Locator):
                    off_l = (
                        float(it.offset[0]) * scale_k,
                        float(it.offset[1]) * scale_k,
                        float(it.offset[2]) * scale_k,
                    )
                    seg_items.append(Locator(name=rename_local[it.name], offset=off_l, role=it.role))
            if built_next is not None:
                seg_items.append(built_next)

            if idx_s == 0:
                seg_att = attach
                seg_mir = mirror_name
                seg_mir_ax = mirror_axis
                seg_rad = radial
            else:
                prev_seg_primary = f"{name}_{seg_num - 1}_{tmpl_boxes[0].name}"
                seg_att = Attach(parent_box=prev_seg_primary, anchor=step_anchor, align=step_align)
                seg_mir = None
                seg_mir_ax = "x"
                seg_rad = None

            built_next = Bone(
                name=seg_bname,
                rest_euler=rest,
                attach=seg_att,
                mirror_name=seg_mir,
                items=seg_items,
                joint_limits=joint_limits,
                mirror_axis=seg_mir_ax,
                radial=seg_rad,
            )

        assert built_next is not None
        return built_next

    def _parse_bone_interior(
        self,
        *,
        name: str,
        root: bool,
        attach: Attach | None,
        mirror: str | None,
        mirror_axis: str,
        radial: RadialSpec | None,
        depth: int,
    ) -> Bone:
        saved_env = dict(self.env)
        pivot: Vec3 = (0.0, 0.0, 0.0)
        rest: Vec3 = (0.0, 0.0, 0.0)
        joint_limits: JointLimits | None = None
        self._consume_preamble_decls(allow_def=False)
        if root:
            if self.at("pivot_local"):
                self.field("pivot_local")
                pv = self.vector()
                pivot = (float(pv[0]), float(pv[1]), float(pv[2]))
        else:
            if self.at("rest_euler"):
                self.field("rest_euler")
                rv = self.vector()
                rest = (float(rv[0]), float(rv[1]), float(rv[2]))
            if self.at("joint_limits"):
                self.field("joint_limits")
                joint_limits = self.joint_limits_matrix()
        items: list[Box | Locator | Bone] = []
        while not self.at("}"):
            if self.at("let"):
                self.let_decl()
            elif self.at("box"):
                items.append(self.box())
            elif self.at("plane"):
                items.append(self.plane())
            elif self.at("locator"):
                items.append(self.locator())
            elif self.at("bone"):
                items.append(self.bone(depth=depth + 1))
            elif self.at("use"):
                items.append(self.use_macro(depth=depth))
            elif self.at("chain"):
                items.append(self.chain_decl(items, depth=depth))
            elif self.at("extrude"):
                raise UnsupportedFeature(
                    "extrude voxelization/greedy merging is a queued Phase 1 task; no bounding-box approximation is performed"
                )
            else:
                self.fail("expected box, plane, locator, bone, use, chain, or closing brace")
        self.env = saved_env
        return Bone(
            name=name,
            pivot_local=pivot,
            rest_euler=rest,
            attach=attach,
            mirror_name=mirror,
            items=items,
            joint_limits=joint_limits,
            mirror_axis=mirror_axis,
            radial=radial,
        )

    def bone(self, root: bool = False, depth: int = 0) -> Bone:
        if depth > 64:
            self.fail("maximum supported bone depth is 64")
        self.take("bone")
        name = self.ident()
        attach: Attach | None = None
        mirror: str | None = None
        mirror_axis = "x"
        radial: RadialSpec | None = None
        if not root:
            attach = self.attach_or_joint()
            mirror, mirror_axis, radial = self._parse_bone_symmetry_modifiers(name)
        self.take("{")
        node = self._parse_bone_interior(
            name=name,
            root=root,
            attach=attach,
            mirror=mirror,
            mirror_axis=mirror_axis,
            radial=radial,
            depth=depth,
        )
        self.take("}")
        return node


def parse_program(text: str) -> Program:
    """Parse and semantically validate a Block-DSL v3.0 / v3.1 / v3.2 program."""
    return _Parser(text).program()


def format_program(program: Program) -> str:
    """Compact canonical deterministic serialization of the Block-DSL v3.2 AST.

    Omits default boilerplate (`offset=[0, 0, 0]`, `detached=false`, `inflate: 0.0`,
    `rotate: [0, 0, 0]`, `rest_euler: [0, 0, 0]`, and default `@materials` kwargs)
    while guaranteeing exact AST roundtrip `parse_program(format_program(p)) == p`.
    """
    validate_program(program)

    def fmt_num(val: int | float) -> str:
        f = float(val)
        if f == 0.0:
            return "0"
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
        return str(f)

    def vector(v: Any, integer: bool = False) -> str:
        if integer:
            return "[" + ", ".join(str(int(x)) for x in v) + "]"
        return "[" + ", ".join(fmt_num(x) for x in v) + "]"

    def is_zero_vec(v: Any) -> bool:
        return all(float(x) == 0.0 for x in v)

    def boolstr(v: bool) -> str:
        return "true" if v else "false"

    m = program.meta
    meta_parts = [
        f"archetype: {m.archetype}",
        f"target_profile: {m.target_profile}",
        f"texel_density: {m.texel_density}",
    ]
    if m.dsl_version != "3.2":
        meta_parts.append(f'dsl_version: "{m.dsl_version}"')
    if float(m.size_step) != 1.0:
        meta_parts.append(f"size_step: {fmt_num(m.size_step)}")
    if m.audit_profile != "strict":
        meta_parts.append(f"audit_profile: {m.audit_profile}")

    lines = [
        f"@meta {{ {', '.join(meta_parts)} }}",
        "@palette {",
    ]
    for p in program.palettes:
        p_args = [f"base_lch={vector(p.base_lch)}", f"steps={p.steps}"]
        if not is_zero_vec(p.hue_shift):
            p_args.append(f"hue_shift={vector(p.hue_shift)}")
        if p.emissive:
            p_args.append("emissive=true")
        lines.append(f"  {p.name}: ramp({', '.join(p_args)})")
    lines.append("}")

    if program.textures:
        lines.append("@textures {")
        for td in program.textures:
            lines.append(f"  {td.name}: image({json.dumps(td.path)}, {td.width}, {td.height})")
        lines.append("}")

    lines.append("@rig {")
    int_sizes = float(m.size_step) == 1.0

    def write_bone(bone: Bone, indent: int, root: bool = False) -> None:
        pad = " " * indent
        start = f"{pad}{'root ' if root else ''}bone {bone.name}"
        if bone.attach:
            a = bone.attach
            if a.joint_mode:
                j_args = [f"parent={a.parent_box}"]
                if not is_zero_vec(a.offset):
                    j_args.append(f"at={vector(a.offset)}")
                if not is_zero_vec(a.align):
                    j_args.append(f"align={vector(a.align)}")
                if a.detached:
                    j_args.append("detached=true")
                start += f" joint({', '.join(j_args)})"
            else:
                a_args = [
                    f"parent_box={a.parent_box}",
                    f"anchor={vector(a.anchor)}",
                    f"align={vector(a.align)}",
                ]
                if not is_zero_vec(a.offset):
                    a_args.append(f"offset={vector(a.offset)}")
                if a.detached:
                    a_args.append("detached=true")
                start += f" attach({', '.join(a_args)})"
        if bone.mirror_name:
            if bone.mirror_axis == "x" and bone.mirror_name != f"{bone.name}_mirror":
                start += f" mirror_x({bone.mirror_name})"
            elif bone.mirror_name == f"{bone.name}_mirror":
                start += f" mirror(axis={bone.mirror_axis})"
            else:
                start += f" mirror(axis={bone.mirror_axis}, as={bone.mirror_name})"
        if bone.radial is not None:
            r_args = [f"count={bone.radial.count}"]
            if bone.radial.axis != "y":
                r_args.append(f"axis={bone.radial.axis}")
            start += f" radial({', '.join(r_args)})"
        lines.append(start + " {")
        if root:
            lines.append(f"{pad}  pivot_local: {vector(bone.pivot_local)}")
        else:
            if not is_zero_vec(bone.rest_euler):
                lines.append(f"{pad}  rest_euler: {vector(bone.rest_euler)}")
            if bone.joint_limits is not None:
                jl = bone.joint_limits
                jl_str = "[" + ", ".join(vector(pair) for pair in jl) + "]"
                lines.append(f"{pad}  joint_limits: {jl_str}")
        for item in bone.items:
            if isinstance(item, Bone):
                write_bone(item, indent + 2)
                continue
            if isinstance(item, Locator):
                role_part = f", role: {item.role}" if item.role != "socket" else ""
                lines.append(f"{pad}  locator {item.name} {{ offset: {vector(item.offset)}{role_part} }}")
                continue

            kw = "plane" if item.plane_normal is not None else "box"
            b_start = f"{pad}  {kw} {item.name}"
            if item.place:
                p = item.place
                p_args: list[str] = []
                if p.base_box:
                    p_args.append(f"base_box={p.base_box}")
                p_args.append(f"anchor={vector(p.anchor)}")
                p_args.append(f"align={vector(p.align)}")
                if not is_zero_vec(p.offset):
                    p_args.append(f"offset={vector(p.offset)}")
                b_start += f" place({', '.join(p_args)})"

            b_fields: list[str] = []
            if item.plane_normal is not None:
                n_idx = {"x": 0, "y": 1, "z": 2}[item.plane_normal]
                sz_2d = [item.size[i] for i in range(3) if i != n_idx]
                b_fields.append(f"size: {vector(sz_2d, integer=int_sizes)}")
                b_fields.append(f"normal: {item.plane_normal}")
            else:
                b_fields.append(f"size: {vector(item.size, integer=int_sizes)}")
            b_fields.append(f"material: {item.material}")
            if float(item.inflate) != 0.0:
                b_fields.append(f"inflate: {fmt_num(item.inflate)}")
            if not is_zero_vec(item.rotate):
                b_fields.append(f"rotate: {vector(item.rotate)}")
            if item.pivot is not None:
                b_fields.append(f"pivot: {vector(item.pivot)}")
            if item.uv is not None:
                b_fields.append(f"uv: box({fmt_num(item.uv[0])}, {fmt_num(item.uv[1])})")
            if item.uv_faces:
                uvf_parts = [f"{fn}: {vector(frect)}" for fn, frect in item.uv_faces]
                b_fields.append(f"uv_faces: {{ {', '.join(uvf_parts)} }}")
            if item.texture is not None:
                b_fields.append(f"texture: {item.texture}")
            if item.cullface:
                b_fields.append(f"cullface: [{', '.join(item.cullface)}]")
            if item.blend_with:
                b_fields.append(f"blend_with: [{', '.join(item.blend_with)}]")
            lines.append(f"{b_start} {{ {', '.join(b_fields)} }}")
        lines.append(pad + "}")

    write_bone(program.root, 2, True)
    lines.append("}")

    if program.materials:
        lines.append("@materials {")
        mat_defaults: tuple[tuple[str, float | int], ...] = (
            ("roughness", 0.35),
            ("metalness", 0.0),
            ("coating", 0.0),
            ("ior", 1.5),
            ("scale", 1.0),
            ("seed", 101),
            ("anisotropy", 1.0),
            ("subsurface", 0.0),
            ("porosity", 0.08),
            ("emissive", 0.0),
            ("rd_feed", 0.036),
            ("rd_kill", 0.062),
            ("film_nm", 380.0),
            ("wear", 0.35),
            ("oxidation", 0.25),
            ("biofilm", 0.20),
        )
        for md in program.materials:
            m_args = [f"family={md.family}"]
            for attr, dval in mat_defaults:
                cur_v = getattr(md, attr)
                if attr == "seed":
                    if int(cur_v) != int(dval):
                        m_args.append(f"seed={int(cur_v)}")
                elif float(cur_v) != float(dval):
                    m_args.append(f"{attr}={fmt_num(cur_v)}")
            if md.weld_group is not None:
                m_args.append(f"weld_group={md.weld_group}")
            lines.append(f"  {md.name}: process({', '.join(m_args)})")
        lines.append("}")

    if program.style is not None:
        sd = program.style
        s_parts = [f"target_encoding: {sd.target_encoding}"]
        if not sd.quantize_ramp:
            s_parts.append("quantize_ramp: false")
        if float(sd.bevel_highlight) != 0.08:
            s_parts.append(f"bevel_highlight: {fmt_num(sd.bevel_highlight)}")
        if float(sd.view_deg) != 20.0:
            s_parts.append(f"view_deg: {fmt_num(sd.view_deg)}")
        lines.append(f"@style {{ {', '.join(s_parts)} }}")

    if program.muscles:
        lines.append("@muscles {")
        for mus in program.muscles:
            parts = [
                f"origin_bone: {mus.origin_bone}",
                f"insertion_bone: {mus.insertion_bone}",
                f"action_axis: {mus.action_axis}",
                f"moment_arm_deg: {fmt_num(mus.moment_arm_deg)}",
            ]
            if mus.tendon_coupling is not None:
                tc = mus.tendon_coupling
                parts.append(f"tendon_coupling: [{tc.bone}, {tc.axis}, {fmt_num(tc.ratio)}]")
            if mus.belly_box is not None:
                parts.append(f"belly_box: {mus.belly_box}")
                if float(mus.bulge_inflate) != 0.0:
                    parts.append(f"bulge_inflate: {fmt_num(mus.bulge_inflate)}")
            lines.append(f"  muscle {mus.name} {{ {', '.join(parts)} }}")
        lines.append("}")

    if program.kinematics is not None:
        lines.append("@kinematics {")
        for ikc in program.kinematics.ik_chains:
            if len(ikc.bones) == 2 and ikc.pole is None:
                ik_parts = [
                    f"upper: {ikc.upper}",
                    f"lower: {ikc.lower}",
                    f"effector: {vector(ikc.effector)}",
                ]
            else:
                b_str = "[" + ", ".join(ikc.bones) + "]"
                ik_parts = [f"bones: {b_str}", f"effector: {vector(ikc.effector)}"]
            if ikc.hinge_axis != "x":
                ik_parts.append(f"hinge_axis: {ikc.hinge_axis}")
            if ikc.pole is not None:
                ik_parts.append(f"pole: {vector(ikc.pole)}")
            lines.append(f"  ik_chain {ikc.name} {{ {', '.join(ik_parts)} }}")
        for lkc in program.kinematics.look_chains:
            b_list = "[" + ", ".join(lkc.bones) + "]"
            wd_part = f", weight_decay: {fmt_num(lkc.weight_decay)}" if float(lkc.weight_decay) != 0.75 else ""
            lines.append(f"  look_chain {lkc.name} {{ bones: {b_list}{wd_part} }}")
        for sp in program.kinematics.springs:
            lines.append(
                f"  spring {sp.bone} {{ stiffness: {fmt_num(sp.stiffness)}, damping: {fmt_num(sp.damping)} }}"
            )
        lines.append("}")

    if program.behavior is not None:
        lines.append("@behavior {")
        for clip in program.behavior.clips:
            lines.append(f"  clip {clip.name} {{")
            loop_part = f", loop: {boolstr(clip.loop)}" if not clip.loop else ""
            lines.append(f"    duration_s: {fmt_num(clip.duration_s)}{loop_part}")
            for tr in clip.tracks:
                tr_args = [
                    f"axis={tr.axis}",
                    f"amp_deg={fmt_num(tr.amp_deg)}",
                    f"freq_hz={fmt_num(tr.freq_hz)}",
                ]
                if float(tr.phase_deg) != 0.0:
                    tr_args.append(f"phase_deg={fmt_num(tr.phase_deg)}")
                if float(tr.bias_deg) != 0.0:
                    tr_args.append(f"bias_deg={fmt_num(tr.bias_deg)}")
                lines.append(f"    track {tr.bone}: {tr.law}({', '.join(tr_args)})")
            for kf in clip.keyframes:
                kf_entries = [f"{fmt_num(t_k)}: {vector(v_k)}" for t_k, v_k in kf.keys]
                if kf.interpolation != "linear":
                    kf_entries.append(kf.interpolation)
                lines.append(f"    key {kf.bone}.{kf.channel} {{ {', '.join(kf_entries)} }}")
            for md_drive in clip.muscle_drives:
                md_args = [f"peak={fmt_num(md_drive.peak)}", f"freq_hz={fmt_num(md_drive.freq_hz)}"]
                if float(md_drive.phase_deg) != 0.0:
                    md_args.append(f"phase_deg={fmt_num(md_drive.phase_deg)}")
                if float(md_drive.baseline) != 0.0:
                    md_args.append(f"baseline={fmt_num(md_drive.baseline)}")
                lines.append(f"    muscle_drive {md_drive.muscle}: activation({', '.join(md_args)})")
            for ikd in clip.ik_drives:
                ikd_args = [f"stride={fmt_num(ikd.stride)}", f"lift={fmt_num(ikd.lift)}"]
                if float(ikd.phase_deg) != 0.0:
                    ikd_args.append(f"phase_deg={fmt_num(ikd.phase_deg)}")
                lines.append(f"    ik_drive {ikd.chain}: foot_cycle({', '.join(ikd_args)})")
            lines.append("  }")
        for sp in program.behavior.springs:
            lines.append(
                f"  spring {sp.bone} {{ stiffness: {fmt_num(sp.stiffness)}, damping: {fmt_num(sp.damping)} }}"
            )
        for st in program.behavior.states:
            lines.append(f"  state {st.name} {{")
            for bname, pb in st.pose_biases:
                lines.append(f"    pose_bias {bname}: {vector(pb)}")
            for mname, tone in st.muscle_tones:
                lines.append(f"    muscle_tone {mname}: {fmt_num(tone)}")
            for mm in st.material_mods:
                lines.append(
                    f"    material_mod {mm.material}: {{ emissive: {fmt_num(mm.emissive)}, "
                    f"moisture: {fmt_num(mm.moisture)}, wear_delta: {fmt_num(mm.wear_delta)} }}"
                )
            lines.append("  }")
        lines.append("}")

    if program.controller is not None:
        lines.append("@controller {")
        if program.controller.initial is not None:
            lines.append(f"  initial: {program.controller.initial}")
        for ctr in program.controller.transitions:
            lines.append(
                f"  {ctr.from_state} -> {ctr.to_state} when {ctr.parameter} {ctr.operator} {fmt_num(ctr.threshold)}, blend {fmt_num(ctr.blend_s)}"
            )
        lines.append("}")

    return "\n".join(lines) + "\n"
