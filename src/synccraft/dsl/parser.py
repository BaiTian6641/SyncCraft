"""Dependency-free, bounded recursive-descent parser & canonical formatter for Block-DSL v3.1.

No Python evaluation, code execution, imports, or permissive unknown fields.
The equivalent grammar is shipped as grammar.lark for later Lark/XGrammar work.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .ast_nodes import (
    Attach,
    BehaviorDecl,
    BehaviorStateDecl,
    Bone,
    Box,
    ClipDecl,
    IKChainDecl,
    IKDriveDecl,
    JointLimits,
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
    Program,
    StyleDecl,
    TendonCoupling,
    TrackDecl,
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
        muscles: list[MuscleDecl] | None = None
        kinematics: KinematicsDecl | None = None
        behavior: BehaviorDecl | None = None

        while self.at("@"):
            self.take("@")
            feature = self.ident()
            if feature in {"shade", "ergonomics", "motion"}:
                raise UnsupportedFeature(
                    f"@{feature} is reserved by v3 but not implemented in the static starter; "
                    f"remove it only for an explicitly simplified asset"
                )
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
            elif feature == "muscles":
                if muscles is not None:
                    self.fail("duplicate @muscles section")
                self.take("{")
                muscles = []
                while not self.at("}"):
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
                muscles=tuple(muscles) if muscles else (),
                kinematics=kinematics,
                behavior=behavior,
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

    def kinematics_decl(self) -> KinematicsDecl:
        self.take("{")
        ik_chains: list[IKChainDecl] = []
        look_chains: list[LookChainDecl] = []
        while not self.at("}"):
            if self.at("ik_chain"):
                self.take("ik_chain")
                name = self.ident()
                self.take("{")
                self.field("upper")
                upper = self.ident()
                self.take(",")
                self.field("lower")
                lower = self.ident()
                self.take(",")
                self.field("effector")
                effector = self.vector(3)
                hinge_axis = "x"
                if self.at(","):
                    self.take(",")
                    self.field("hinge_axis")
                    hinge_axis = self.ident()
                self.take("}")
                ik_chains.append(IKChainDecl(name=name, upper=upper, lower=lower, effector=effector, hinge_axis=hinge_axis))
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
            else:
                self.fail("expected ik_chain, look_chain, or closing brace in @kinematics")
        self.take("}")
        return KinematicsDecl(ik_chains=tuple(ik_chains), look_chains=tuple(look_chains))

    def behavior_decl(self) -> BehaviorDecl:
        self.take("{")
        clips: list[ClipDecl] = []
        states: list[BehaviorStateDecl] = []
        while not self.at("}"):
            if self.at("clip"):
                clips.append(self.clip_decl())
            elif self.at("state"):
                states.append(self.behavior_state_decl())
            else:
                self.fail("expected clip, state, or closing brace in @behavior")
        self.take("}")
        return BehaviorDecl(clips=tuple(clips), states=tuple(states))

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
                tracks.append(TrackDecl(bone=bone, law=law, axis=axis, amp_deg=amp_deg, freq_hz=freq_hz, phase_deg=phase_deg, bias_deg=bias_deg))
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
                muscle_drives.append(MuscleDriveDecl(muscle=muscle, peak=peak, freq_hz=freq_hz, phase_deg=phase_deg, baseline=baseline))
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
                self.fail("expected track, muscle_drive, ik_drive, or closing brace in clip")
        self.take("}")
        return ClipDecl(
            name=name,
            duration_s=duration_s,
            loop=loop,
            tracks=tuple(tracks),
            muscle_drives=tuple(muscle_drives),
            ik_drives=tuple(ik_drives),
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
                material_mods.append(MaterialModDecl(material=mat, emissive=emissive, moisture=moisture, wear_delta=wear_delta))
            else:
                self.fail("expected pose_bias, muscle_tone, material_mod, or closing brace in state")
        self.take("}")
        return BehaviorStateDecl(
            name=name,
            pose_biases=tuple(pose_biases),
            muscle_tones=tuple(muscle_tones),
            material_mods=tuple(material_mods),
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

    def bone(self, root: bool = False, depth: int = 0) -> Bone:
        if depth > 64:
            self.fail("maximum supported bone depth is 64")
        self.take("bone")
        name = self.ident()
        attach, mirror = None, None
        pivot, rest = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        joint_limits: JointLimits | None = None
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
        else:
            if self.at("rest_euler"):
                self.field("rest_euler")
                rest = self.vector()
            if self.at("joint_limits"):
                self.field("joint_limits")
                joint_limits = self.joint_limits_matrix()
        items = []
        while not self.at("}"):
            if self.at("box"):
                items.append(self.box())
            elif self.at("locator"):
                items.append(self.locator())
            elif self.at("bone"):
                items.append(self.bone(depth=depth + 1))
            elif self.at("extrude"):
                raise UnsupportedFeature(
                    "extrude voxelization/greedy merging is a queued Phase 1 task; no bounding-box approximation is performed"
                )
            else:
                self.fail("expected box, locator, bone, or closing brace")
        self.take("}")
        return Bone(name, pivot, rest, attach, mirror, items, joint_limits=joint_limits)


def parse_program(text: str) -> Program:
    """Parse and semantically validate a Block-DSL v3.1 program."""
    return _Parser(text).program()


def format_program(program: Program) -> str:
    """Canonical deterministic serialization of the implemented AST."""
    validate_program(program)

    def vector(v, integer=False):
        return "[" + ", ".join(str(x) if integer else str(float(x)) for x in v) + "]"

    def boolstr(v):
        return "true" if v else "false"

    m = program.meta
    lines = [
        f"@meta {{ archetype: {m.archetype}, target_profile: {m.target_profile}, texel_density: {m.texel_density} }}",
        "@palette {",
    ]
    for p in program.palettes:
        lines.append(
            f"  {p.name}: ramp(base_lch={vector(p.base_lch)}, steps={p.steps}, hue_shift={vector(p.hue_shift)}, emissive={boolstr(p.emissive)})"
        )
    lines += ["}", "@rig {"]

    def write_bone(bone: Bone, indent: int, root: bool = False):
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
            if bone.joint_limits is not None:
                jl = bone.joint_limits
                jl_str = "[" + ", ".join(vector(pair) for pair in jl) + "]"
                lines.append(f"{pad}  joint_limits: {jl_str}")
        for item in bone.items:
            if isinstance(item, Bone):
                write_bone(item, indent + 2)
                continue
            if isinstance(item, Locator):
                lines.append(f"{pad}  locator {item.name} {{ offset: {vector(item.offset)}, role: {item.role} }}")
                continue
            start = f"{pad}  box {item.name}"
            if item.place:
                p = item.place
                base = f"base_box={p.base_box}, " if p.base_box else ""
                start += f" place({base}anchor={vector(p.anchor)}, align={vector(p.align)}, offset={vector(p.offset)})"
            suffix = f", cullface: [{', '.join(item.cullface)}]" if item.cullface else ""
            lines.append(
                start + f" {{ size: {vector(item.size, integer=True)}, material: {item.material}, inflate: {float(item.inflate)}{suffix} }}"
            )
        lines.append(pad + "}")

    write_bone(program.root, 2, True)
    lines.append("}")

    if program.materials:
        lines.append("@materials {")
        for md in program.materials:
            extras: list[str] = []
            if float(md.anisotropy) != 1.0:
                extras.append(f"anisotropy={float(md.anisotropy)}")
            if float(md.subsurface) != 0.0:
                extras.append(f"subsurface={float(md.subsurface)}")
            if float(md.porosity) != 0.08:
                extras.append(f"porosity={float(md.porosity)}")
            if float(md.emissive) != 0.0:
                extras.append(f"emissive={float(md.emissive)}")
            if float(md.rd_feed) != 0.036:
                extras.append(f"rd_feed={float(md.rd_feed)}")
            if float(md.rd_kill) != 0.062:
                extras.append(f"rd_kill={float(md.rd_kill)}")
            if float(md.film_nm) != 380.0:
                extras.append(f"film_nm={float(md.film_nm)}")
            if float(md.wear) != 0.35:
                extras.append(f"wear={float(md.wear)}")
            if float(md.oxidation) != 0.25:
                extras.append(f"oxidation={float(md.oxidation)}")
            if float(md.biofilm) != 0.20:
                extras.append(f"biofilm={float(md.biofilm)}")
            if md.weld_group is not None:
                extras.append(f"weld_group={md.weld_group}")
            extra_str = (", " + ", ".join(extras)) if extras else ""
            lines.append(
                f"  {md.name}: process(family={md.family}, roughness={float(md.roughness)}, "
                f"metalness={float(md.metalness)}, coating={float(md.coating)}, "
                f"ior={float(md.ior)}, scale={float(md.scale)}, seed={int(md.seed)}{extra_str})"
            )
        lines.append("}")

    if program.style is not None:
        sd = program.style
        lines.append(
            f"@style {{ target_encoding: {sd.target_encoding}, quantize_ramp: {boolstr(sd.quantize_ramp)}, "
            f"bevel_highlight: {float(sd.bevel_highlight)}, view_deg: {float(sd.view_deg)} }}"
        )

    if program.muscles:
        lines.append("@muscles {")
        for mus in program.muscles:
            parts = [
                f"origin_bone: {mus.origin_bone}",
                f"insertion_bone: {mus.insertion_bone}",
                f"action_axis: {mus.action_axis}",
                f"moment_arm_deg: {float(mus.moment_arm_deg)}",
            ]
            if mus.tendon_coupling is not None:
                tc = mus.tendon_coupling
                parts.append(f"tendon_coupling: [{tc.bone}, {tc.axis}, {float(tc.ratio)}]")
            if mus.belly_box is not None:
                parts.append(f"belly_box: {mus.belly_box}")
                parts.append(f"bulge_inflate: {float(mus.bulge_inflate)}")
            lines.append(f"  muscle {mus.name} {{ {', '.join(parts)} }}")
        lines.append("}")

    if program.kinematics is not None:
        lines.append("@kinematics {")
        for ikc in program.kinematics.ik_chains:
            lines.append(
                f"  ik_chain {ikc.name} {{ upper: {ikc.upper}, lower: {ikc.lower}, "
                f"effector: {vector(ikc.effector)}, hinge_axis: {ikc.hinge_axis} }}"
            )
        for lkc in program.kinematics.look_chains:
            b_list = "[" + ", ".join(lkc.bones) + "]"
            lines.append(f"  look_chain {lkc.name} {{ bones: {b_list}, weight_decay: {float(lkc.weight_decay)} }}")
        lines.append("}")

    if program.behavior is not None:
        lines.append("@behavior {")
        for clip in program.behavior.clips:
            lines.append(f"  clip {clip.name} {{")
            lines.append(f"    duration_s: {float(clip.duration_s)}, loop: {boolstr(clip.loop)}")
            for tr in clip.tracks:
                lines.append(
                    f"    track {tr.bone}: {tr.law}(axis={tr.axis}, amp_deg={float(tr.amp_deg)}, "
                    f"freq_hz={float(tr.freq_hz)}, phase_deg={float(tr.phase_deg)}, bias_deg={float(tr.bias_deg)})"
                )
            for md_drive in clip.muscle_drives:
                lines.append(
                    f"    muscle_drive {md_drive.muscle}: activation(peak={float(md_drive.peak)}, "
                    f"freq_hz={float(md_drive.freq_hz)}, phase_deg={float(md_drive.phase_deg)}, baseline={float(md_drive.baseline)})"
                )
            for ikd in clip.ik_drives:
                lines.append(
                    f"    ik_drive {ikd.chain}: foot_cycle(stride={float(ikd.stride)}, "
                    f"lift={float(ikd.lift)}, phase_deg={float(ikd.phase_deg)})"
                )
            lines.append("  }")
        for st in program.behavior.states:
            lines.append(f"  state {st.name} {{")
            for bname, pb in st.pose_biases:
                lines.append(f"    pose_bias {bname}: {vector(pb)}")
            for mname, tone in st.muscle_tones:
                lines.append(f"    muscle_tone {mname}: {float(tone)}")
            for mm in st.material_mods:
                lines.append(
                    f"    material_mod {mm.material}: {{ emissive: {float(mm.emissive)}, "
                    f"moisture: {float(mm.moisture)}, wear_delta: {float(mm.wear_delta)} }}"
                )
            lines.append("  }")
        lines.append("}")

    return "\n".join(lines) + "\n"
