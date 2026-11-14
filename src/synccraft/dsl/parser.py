"""Dependency-free, bounded recursive-descent parser for the static v3 subset.

No Python evaluation, code execution, imports, or permissive unknown fields.
The equivalent grammar is shipped as grammar.lark for later Lark/XGrammar work.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .ast_nodes import (
    Attach,
    Bone,
    Box,
    MaterialDecl,
    Meta,
    Palette,
    Place,
    Program,
    StyleDecl,
    UnsupportedFeature,
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


_LEXER = re.compile(
    r"(?P<SPACE>\s+)|(?P<COMMENT>//[^\n]*|\#[^\n]*)|"
    r"(?P<NUMBER>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)|(?P<PUNCT>[@{}()\[\]:,=])"
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


class _Parser:
    def __init__(self, text: str):
        self.tokens = _tokens(text)
        self.index = 0

    @property
    def token(self) -> _Token:
        return self.tokens[self.index]

    def at(self, value: str) -> bool:
        return self.token.value == value

    def fail(self, message: str):
        raise DSLParseError(message, self.token.line, self.token.column)

    def take(self, value: str) -> str:
        if not self.at(value):
            self.fail(f"expected {value!r}, got {self.token.value!r}")
        self.index += 1
        return value

    def ident(self) -> str:
        if self.token.kind != "IDENT":
            self.fail("expected identifier")
        value = self.token.value
        self.index += 1
        return value

    def number(self, integer: bool = False):
        if self.token.kind != "NUMBER":
            self.fail("expected integer" if integer else "expected number")
        value = self.token.value
        if integer and not re.fullmatch(r"[+-]?\d+", value):
            self.fail("expected integer literal; fractional and exponent spellings are not integer sizes")
        self.index += 1
        return int(value) if integer else float(value)

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

    def field(self, name: str, separator: str = ":") -> None:
        self.take(name)
        self.take(separator)

    def section(self, name: str) -> None:
        self.take("@")
        self.take(name)
        self.take("{")

    def program(self) -> Program:
        self.section("meta")
        self.field("archetype")
        archetype = self.ident()
        self.take(",")
        self.field("target_profile")
        profile = self.ident()
        self.take(",")
        self.field("texel_density")
        density = self.number(integer=True)
        self.take("}")
        self.section("palette")
        palettes = []
        while not self.at("}"):
            palettes.append(self.palette())
        self.take("}")
        self.section("rig")
        self.take("root")
        root = self.bone(root=True, depth=0)
        self.take("}")
        materials: list[MaterialDecl] | None = None
        style: StyleDecl | None = None
        while self.at("@"):
            self.take("@")
            feature = self.ident()
            if feature in {"shade", "ergonomics", "motion"}:
                raise UnsupportedFeature(f"@{feature} is reserved by v3 but not implemented in the static starter; remove it only for an explicitly simplified asset")
            if feature == "materials":
                if materials is not None:
                    self.fail("duplicate @materials section")
                self.take("{")
                materials = []
                while not self.at("}"):
                    materials.append(self.material_decl())
                self.take("}")
            elif feature == "style":
                if style is not None:
                    self.fail("duplicate @style section")
                style = self.style_decl()
            else:
                self.fail(f"unknown section @{feature}")
        self.take("<EOF>")
        return validate_program(
            Program(
                Meta(archetype, profile, density),
                palettes,
                root,
                materials=tuple(materials) if materials else (),
                style=style,
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
        allowed = {"roughness", "metalness", "coating", "ior", "scale", "seed"}
        while self.at(","):
            self.take(",")
            key = self.ident()
            if key not in allowed or key in kwargs:
                self.fail(f"unexpected or duplicate @materials field {key!r}")
            self.take("=")
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

    def palette(self) -> Palette:
        name = self.ident()
        self.take(":")
        self.take("ramp")
        self.take("(")
        self.field("base_lch", "=")
        lch = self.vector()
        self.take(",")
        self.field("steps", "=")
        steps = self.number(integer=True)
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

    def attach(self) -> Attach:
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

    def box(self) -> Box:
        self.take("box")
        name = self.ident()
        place = self.place() if self.at("place") else None
        self.take("{")
        self.field("size")
        size = self.vector(integer=True)
        self.take(",")
        self.field("material")
        material = self.ident()
        inflate, faces = 0.0, []
        if self.at(","):
            self.take(",")
            if self.at("inflate"):
                self.field("inflate")
                inflate = self.number()
                if self.at(","):
                    self.take(",")
                else:
                    self.take("}")
                    return Box(name, size, material, place, inflate)
            self.field("cullface")
            self.take("[")
            faces.append(self.ident())
            while self.at(","):
                self.take(",")
                faces.append(self.ident())
            self.take("]")
        self.take("}")
        return Box(name, size, material, place, inflate, tuple(faces))

    def bone(self, root: bool = False, depth: int = 0) -> Bone:
        if depth > 64:
            self.fail("maximum supported bone depth is 64")
        self.take("bone")
        name = self.ident()
        attach, mirror = None, None
        pivot, rest = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        if not root:
            attach = self.attach()
            if self.at("mirror_x"):
                self.take("mirror_x")
                self.take("(")
                mirror = self.ident()
                self.take(")")
        self.take("{")
        if root:
            self.field("pivot_local")
            pivot = self.vector()
        elif self.at("rest_euler"):
            self.field("rest_euler")
            rest = self.vector()
        items = []
        while not self.at("}"):
            if self.at("box"):
                items.append(self.box())
            elif self.at("bone"):
                items.append(self.bone(depth=depth + 1))
            elif self.at("extrude"):
                raise UnsupportedFeature("extrude voxelization/greedy merging is a queued Phase 1 task; no bounding-box approximation is performed")
            else:
                self.fail("expected box, bone, or closing brace")
        self.take("}")
        return Bone(name, pivot, rest, attach, mirror, items)


def parse_program(text: str) -> Program:
    """Parse and semantically validate a static starter program."""
    return _Parser(text).program()


def format_program(program: Program) -> str:
    """Canonical deterministic serialization of the implemented AST subset."""
    validate_program(program)

    def vector(v, integer=False):
        # Direct Python AST callers may use integers for floating fields. Emit
        # the same representation as a text parse so canonical hashes agree.
        return "[" + ", ".join(str(x) if integer else str(float(x)) for x in v) + "]"

    def boolstr(v):
        return "true" if v else "false"

    m = program.meta
    lines = [f"@meta {{ archetype: {m.archetype}, target_profile: {m.target_profile}, texel_density: {m.texel_density} }}", "@palette {"]
    for p in program.palettes:
        lines.append(f"  {p.name}: ramp(base_lch={vector(p.base_lch)}, steps={p.steps}, hue_shift={vector(p.hue_shift)}, emissive={boolstr(p.emissive)})")
    lines += ["}", "@rig {"]

    def write_bone(bone, indent, root=False):
        pad = " " * indent
        start = f"{pad}{'root ' if root else ''}bone {bone.name}"
        if bone.attach:
            a = bone.attach
            start += f" attach(parent_box={a.parent_box}, anchor={vector(a.anchor)}, align={vector(a.align)}, offset={vector(a.offset)}, detached={boolstr(a.detached)})"
        if bone.mirror_name:
            start += f" mirror_x({bone.mirror_name})"
        lines.append(start + " {")
        if root:
            lines.append(f"{pad}  pivot_local: {vector(bone.pivot_local)}")
        else:
            lines.append(f"{pad}  rest_euler: {vector(bone.rest_euler)}")
        for item in bone.items:
            if isinstance(item, Bone):
                write_bone(item, indent + 2)
                continue
            start = f"{pad}  box {item.name}"
            if item.place:
                p = item.place
                base = f"base_box={p.base_box}, " if p.base_box else ""
                start += f" place({base}anchor={vector(p.anchor)}, align={vector(p.align)}, offset={vector(p.offset)})"
            suffix = f", cullface: [{', '.join(item.cullface)}]" if item.cullface else ""
            lines.append(start + f" {{ size: {vector(item.size, integer=True)}, material: {item.material}, inflate: {float(item.inflate)}{suffix} }}")
        lines.append(pad + "}")

    write_bone(program.root, 2, True)
    lines.append("}")
    if program.materials:
        lines.append("@materials {")
        for md in program.materials:
            lines.append(
                f"  {md.name}: process(family={md.family}, roughness={float(md.roughness)}, "
                f"metalness={float(md.metalness)}, coating={float(md.coating)}, "
                f"ior={float(md.ior)}, scale={float(md.scale)}, seed={int(md.seed)})"
            )
        lines.append("}")
    if program.style is not None:
        sd = program.style
        lines.append(
            f"@style {{ target_encoding: {sd.target_encoding}, quantize_ramp: {boolstr(sd.quantize_ramp)}, "
            f"bevel_highlight: {float(sd.bevel_highlight)}, view_deg: {float(sd.view_deg)} }}"
        )
    return "\n".join(lines) + "\n"
