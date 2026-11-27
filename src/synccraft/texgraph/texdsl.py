"""Declarative `.texdsl` Texture Programming Language for `synccraft.texgraph` (`TexGraph v8`).

Eliminates writing 300-line Python scripts for every new Blockbench texture! Author clean,
human-readable, version-controllable 30-60 line `.texdsl` files for blocks, items, multi-material
3D characters (Orcs, Knights), part-aware animals, and animated PBR `.mcmeta` flipbooks.

Syntax overview::

    @texture {
      name: "ironjaw_orc_warlord"
      size: [256, 256]
      domain: "entity"
      seed: 808
      tileable: false
      frames: 8
      frametime: 3
      interpolate: false
      frames_schedule: [{index: 0, time: 18}, 1, 2, 3, 4, 5, 6, 7]
    }

    @palette {
      orc_skin: ["#172A26", "#244834", "#3F743C", "#6D9C46", "#A7C459", "#D4DF86"]
      steel:    [[0.0, "#1B1726"], [0.5, "#51576A"], [1.0, "#EEF5FC"]]
    }

    @pass bake "Pass 1: 3D Mesh Bake & Part Masks" {
      ao      = mesh_channel("baked_ao")
      m_metal = select(pattern="pauldron|chest_plate|axe_head")
      m_both  = m_metal | select(zone="horns")
      m_body  = !m_both
      preview m_metal
    }

    @pass shade "Pass 2: Materials & Protected Facial Landmarks" {
      skin    = orc_flesh_skin(stops=@orc_skin, scar_amount=0.35)
      metal   = forged_battle_metal(stops=@steel, scratches=0.5)
      merged  = material_height_blend(skin, metal, mask=m_metal, contrast=8.5, seam_ao_strength=0.45)
      detail  = paint_entity_eyes_and_parts(merged, eye_iris="#FF6B1A", eye_glow=0.88)
      final   = material_pad_islands(material_grade(detail, ao_bake=ao, ao_darken=0.35), pad=2)
    }

    @animate {
      drive detail.eye_glow  = sine(min_val=0.50, max_val=1.0, freq=1.0)
      drive detail.eye_blink = keyframes([[0.0, 0.0], [0.75, 0.0], [0.85, 0.95], [0.95, 0.0], [1.0, 0.0]])
    }

    @expose {
      scar_intensity: skin.scar_amount [0.0..1.0 step=0.05]
    }

    @output material(final, normal_strength=2.5)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .mesh_baker import MeshBakeContext
from .pipeline import SUBGRAPH_REGISTRY
from .production import PALETTE_BANK
from .registry import NODE_CATALOG


class TexDslError(ValueError):
    """Syntax or semantic error raised while compiling a `.texdsl` source."""

    def __init__(self, message: str, line: int = 1, col: int = 1) -> None:
        super().__init__(f".texdsl error at line {line}, col {col}: {message}")
        self.line = line
        self.col = col


@dataclass(frozen=True)
class _Tok:
    kind: str  # IDENT, NUMBER, STRING, AT, DOT, DOTDOT, LBRACE, RBRACE, LBRACK, RBRACK, LPAREN, RPAREN, COLON, COMMA, EQ, OP, EOF
    val: str
    line: int
    col: int


@dataclass(frozen=True)
class _NodeRef:
    ref: str  # e.g. "skin" or "plates.height"


@dataclass(frozen=True)
class _PaletteRef:
    name: str  # e.g. "orc_skin" or "bank.husky_espresso" or "orc_skin.hex"


def _tokenize(src: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i = 0
    n = len(src)
    line = 1
    col = 1

    def _adv(k: int = 1) -> None:
        nonlocal i, line, col
        for _ in range(k):
            if i < n and src[i] == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1

    while i < n:
        ch = src[i]
        # Whitespace & semicolons
        if ch in " \t\r\n;":
            _adv()
            continue
        # Line comments (# or //)
        if ch == "#" or (ch == "/" and i + 1 < n and src[i + 1] == "/"):
            while i < n and src[i] != "\n":
                _adv()
            continue
        # Block comments /* ... */
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            _adv(2)
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                _adv()
            if i + 1 >= n:
                raise TexDslError("Unterminated /* block comment", line, col)
            _adv(2)
            continue

        sl, sc = line, col
        # Strings ("..." or '...')
        if ch in ('"', "'"):
            quote = ch
            _adv()
            chars: list[str] = []
            while i < n and src[i] != quote:
                if src[i] == "\\":
                    _adv()
                    if i >= n:
                        raise TexDslError("Unterminated string escape", sl, sc)
                    esc = src[i]
                    chars.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}.get(esc, esc))
                    _adv()
                else:
                    chars.append(src[i])
                    _adv()
            if i >= n:
                raise TexDslError("Unterminated string literal", sl, sc)
            _adv()
            toks.append(_Tok("STRING", "".join(chars), sl, sc))
            continue

        # Numbers (int or float, optionally scientific; unary '-' handled in parser)
        if ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            start = i
            has_dot = False
            while i < n and (src[i].isdigit() or (src[i] == "." and not has_dot and not (i + 1 < n and src[i + 1] == "."))):
                if src[i] == ".":
                    has_dot = True
                _adv()
            if i < n and src[i] in "eE":
                _adv()
                if i < n and src[i] in "+-":
                    _adv()
                while i < n and src[i].isdigit():
                    _adv()
            toks.append(_Tok("NUMBER", src[start:i], sl, sc))
            continue

        # Identifiers / keywords
        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (src[i].isalnum() or src[i] == "_"):
                _adv()
            toks.append(_Tok("IDENT", src[start:i], sl, sc))
            continue

        # Two-char punctuation
        if ch == "." and i + 1 < n and src[i + 1] == ".":
            toks.append(_Tok("DOTDOT", "..", sl, sc))
            _adv(2)
            continue

        single = {
            "@": "AT",
            ".": "DOT",
            "{": "LBRACE",
            "}": "RBRACE",
            "[": "LBRACK",
            "]": "RBRACK",
            "(": "LPAREN",
            ")": "RPAREN",
            ":": "COLON",
            ",": "COMMA",
            "=": "EQ",
            "+": "OP",
            "-": "OP",
            "*": "OP",
            "/": "OP",
            "|": "OP",
            "&": "OP",
            "!": "OP",
            "~": "OP",
        }
        if ch in single:
            toks.append(_Tok(single[ch], ch, sl, sc))
            _adv()
            continue

        raise TexDslError(f"Unexpected character {ch!r}", sl, sc)

    toks.append(_Tok("EOF", "", line, col))
    return toks


def _expand_palette_stops(raw: Any, *, line: int = 1, col: int = 1) -> list[tuple[float, str]]:
    """Normalize palette definitions: ['#a','#b'] or [[0.0,'#a'],[1.0,'#b']] or {0.0:'#a'}."""
    if isinstance(raw, dict):
        items = sorted(((float(k), str(v)) for k, v in raw.items()), key=lambda kv: kv[0])
        return items
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise TexDslError("Palette cannot be empty", line, col)
        if isinstance(raw[0], str):
            n = len(raw)
            if n == 1:
                return [(0.0, str(raw[0])), (1.0, str(raw[0]))]
            return [(round(float(idx) / float(n - 1), 4), str(c)) for idx, c in enumerate(raw)]
        return [(float(pair[0]), str(pair[1])) for pair in raw]
    raise TexDslError(f"Invalid palette definition: {raw!r}", line, col)


# Friendly aliases so artists can write natural concise op names in .texdsl
_OP_ALIASES: dict[str, str] = {
    "blur": "gaussian_blur",
    "sharpen": "unsharp_mask",
    "quantize": "palette_quantize",
    "shade": "directional_shade",
    "scales": "imbricated_scales",
    "feathers": "feather_barbs",
    "fur": "fur_tufts",
    "vibrance": "vibrance_grade",
    "twirl": "twirl_distort",
    "wave": "wave_distort",
    "morph": "morph_op",
    "sdf": "sdf_from_mask",
    "bloom": "bloom_glow",
    "painterly_brush": "painterly_brush_field",
    "bevel": "bevel_emboss",
    "ao": "ao_from_height",
}

# Positional parameter signatures for ops whose first positional arg(s) are params rather than node inputs
_POSITIONAL_PARAMS: dict[str, tuple[str, ...]] = {
    "constant": ("value",),
    "gradient": ("direction",),
    "mesh_channel": ("channel",),
    "gradient_field": ("kind",),
    "per_island_gradient": ("direction",),
    "body_flow_gradient": ("axis",),
    "sdf_shape": ("shape",),
    "sprite_mask": ("shape",),
    "anim_driver": ("kind",),
    "value_noise": ("scale",),
    "perlin": ("scale",),
    "fbm": ("scale", "octaves", "mode"),
    "voronoi": ("scale", "output"),
    "gabor": ("scale", "angle_deg"),
    "noise_3d": ("scale", "octaves", "mode"),
    "grid": ("cols", "rows"),
    "brick": ("cols", "rows", "style"),
    "hex_grid": ("cols",),
    "tri_grid": ("cols",),
    "diamond": ("cols", "rows", "style"),
    "truchet": ("cols", "style"),
    "mosaic": ("scale",),
    "imbricated_scales": ("cols", "rows", "shape"),
    "gray_scott": ("preset", "steps"),
    "cellular_automata": ("preset", "steps"),
    "brushstrokes": ("angle_deg",),
    "layered_plates": ("cols", "rows"),
    "temple_stone": ("cols", "rows"),
    "wood_planks": ("planks",),
    "foliage_clusters": ("cols", "rows"),
}

# Canonical mesh_channel short aliases
_MESH_CHANNEL_ALIASES: dict[str, str] = {
    "ao": "baked_ao",
    "bevel": "edge_bevel",
    "cavity": "joint_cavity",
    "mask": "island_mask",
}


class _TexDslParser:
    def __init__(self, toks: list[_Tok], mesh_context: MeshBakeContext | None = None) -> None:
        self.toks = toks
        self.pos = 0
        self.mesh_context = mesh_context
        self.header: dict[str, Any] = {
            "name": "texdsl_graph",
            "width": 32,
            "height": 32,
            "tileable": False,
            "seed": 42,
            "domain": "block",
            "frames": 1,
            "frametime": 2,
            "interpolate": False,
            "frames_schedule": None,
        }
        self.palettes: dict[str, list[tuple[float, str]]] = {}
        self.aliases: dict[str, str] = {}
        self.graph: Any = None  # created lazily once header is known
        self._tmp_n = 0

    def _cur(self) -> _Tok:
        return self.toks[self.pos]

    def _peek(self, offset: int = 1) -> _Tok:
        idx = min(len(self.toks) - 1, self.pos + offset)
        return self.toks[idx]

    def _eat(self, kind: str | None = None, val: str | None = None) -> _Tok:
        t = self._cur()
        if kind is not None and t.kind != kind:
            raise TexDslError(f"Expected {kind}, got {t.kind} ({t.val!r})", t.line, t.col)
        if val is not None and t.val != val:
            raise TexDslError(f"Expected {val!r}, got {t.val!r}", t.line, t.col)
        self.pos += 1
        return t

    def _ensure_graph(self) -> Any:
        if self.graph is None:
            from .core import TexGraph

            w = int(self.header["width"])
            h = int(self.header["height"])
            if self.mesh_context is not None:
                w = self.mesh_context.width
                h = self.mesh_context.height
            self.graph = TexGraph(
                w,
                h,
                tileable=bool(self.header["tileable"]),
                seed=int(self.header["seed"]),
                name=str(self.header["name"]),
                mesh_context=self.mesh_context,
            )
            self.graph.domain = str(self.header.get("domain", "block"))
            self.graph.animation_config = {
                "frames": int(self.header.get("frames", 1)),
                "frametime": int(self.header.get("frametime", 2)),
                "interpolate": bool(self.header.get("interpolate", False)),
                "frames_schedule": self.header.get("frames_schedule"),
            }
        return self.graph

    def _next_tmp(self, prefix: str = "expr") -> str:
        self._tmp_n += 1
        return f"_{prefix}_{self._tmp_n:03d}"

    def parse(self) -> Any:
        while self._cur().kind != "EOF":
            if self._cur().kind == "AT":
                self._parse_directive()
            else:
                self._ensure_graph()
                self._parse_statement()
        g = self._ensure_graph()
        if g.output_node is None and g.output_material_node is None and g.node_order:
            last_id = g.node_order[-1]
            meta = NODE_CATALOG.get(g.nodes[last_id].op)
            if meta is not None and meta.output_type == "MATERIAL":
                g.set_output_material(last_id)
            else:
                g.set_output(last_id)
        return g

    # --- Top-level Directives ---
    def _parse_directive(self) -> None:
        self._eat("AT")
        kw = self._eat("IDENT")
        name = kw.val
        if name == "texture":
            self._parse_texture_header()
        elif name == "palette":
            self._parse_palette_block()
        elif name == "pass":
            self._parse_pass_block()
        elif name == "animate":
            self._parse_animate_block()
        elif name == "expose":
            self._parse_expose_block()
        elif name == "output":
            self._parse_output_directive()
        else:
            raise TexDslError(f"Unknown directive @{name}", kw.line, kw.col)

    def _parse_texture_header(self) -> None:
        if self.graph is not None:
            raise TexDslError("@texture header must appear before node statements", self._cur().line, self._cur().col)
        obj = self._parse_map_literal()
        for k, v in obj.items():
            if k == "size":
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    self.header["width"] = int(v[0])
                    self.header["height"] = int(v[1])
                else:
                    s = int(v)
                    self.header["width"] = s
                    self.header["height"] = s
            elif k in ("width", "height", "seed", "frames", "frametime"):
                self.header[k] = int(v)
            elif k in ("tileable", "interpolate"):
                self.header[k] = bool(v)
            elif k in ("name", "domain"):
                self.header[k] = str(v)
            elif k in ("frames_schedule", "schedule"):
                self.header["frames_schedule"] = v
            else:
                self.header[k] = v

    def _parse_palette_block(self) -> None:
        obj = self._parse_map_literal()
        for k, v in obj.items():
            self.palettes[str(k)] = _expand_palette_stops(v, line=self._cur().line, col=self._cur().col)

    def _parse_pass_block(self) -> None:
        g = self._ensure_graph()
        pid = self._eat("IDENT").val
        title = pid
        if self._cur().kind == "STRING":
            title = self._eat("STRING").val
        self._eat("LBRACE")
        with g.pass_(pid, title=title):
            while self._cur().kind not in ("RBRACE", "EOF"):
                self._parse_statement()
        self._eat("RBRACE")

    def _parse_animate_block(self) -> None:
        g = self._ensure_graph()
        self._eat("LBRACE")
        while self._cur().kind not in ("RBRACE", "EOF"):
            self._eat("IDENT", "drive")
            nid = self._eat("IDENT").val
            self._eat("DOT")
            pkey = self._eat("IDENT").val
            self._eat("EQ")
            kind_tok = self._eat("IDENT")
            kind = kind_tok.val
            self._eat("LPAREN")
            pos_args, kw_args = self._parse_call_args_literals()
            self._eat("RPAREN")
            drv_kw: dict[str, Any] = dict(kw_args)
            if kind == "keyframes" and pos_args:
                drv_kw["keyframes"] = pos_args[0]
            elif pos_args:
                # Allow concise positional sine(0.4, 1.0, 1.0) -> min_val, max_val, freq
                for pname, pval in zip(("min_val", "max_val", "freq", "phase_offset"), pos_args):
                    drv_kw.setdefault(pname, pval)
            g.bind_driver(self._resolve_alias(nid).split(".", 1)[0], pkey, kind=kind, **drv_kw)
        self._eat("RBRACE")

    def _parse_expose_block(self) -> None:
        g = self._ensure_graph()
        self._eat("LBRACE")
        while self._cur().kind not in ("RBRACE", "EOF"):
            pub = self._eat("IDENT").val
            self._eat("COLON")
            nid = self._eat("IDENT").val
            self._eat("DOT")
            pkey = self._eat("IDENT").val
            min_v = max_v = step_v = None
            if self._cur().kind == "LBRACK":
                self._eat("LBRACK")
                min_v = float(self._parse_literal())
                self._eat("DOTDOT")
                max_v = float(self._parse_literal())
                if self._cur().kind == "IDENT" and self._cur().val == "step":
                    self._eat("IDENT")
                    self._eat("EQ")
                    step_v = float(self._parse_literal())
                self._eat("RBRACK")
            if self._cur().kind == "COMMA":
                self._eat("COMMA")
            g.expose_param(pub, nid, pkey, min_val=min_v, max_val=max_v, step=step_v)
        self._eat("RBRACE")

    def _parse_output_directive(self) -> None:
        g = self._ensure_graph()
        mode_tok = self._eat("IDENT")
        mode = mode_tok.val
        self._eat("LPAREN")
        pos_args, kw_args = self._parse_call_args_expr()
        self._eat("RPAREN")

        if mode == "material":
            if not pos_args or not isinstance(pos_args[0], _NodeRef):
                raise TexDslError("@output material(...) requires a MATERIAL node reference", mode_tok.line, mode_tok.col)
            g.set_output_material(
                self._resolve_alias(pos_args[0].ref),
                normal_strength=kw_args.get("normal_strength"),
                dielectric_f0=kw_args.get("dielectric_f0"),
            )
            return

        if mode in ("rgba", "albedo", "color"):
            if not pos_args or not isinstance(pos_args[0], _NodeRef):
                raise TexDslError(f"@output {mode}(...) requires an output node reference", mode_tok.line, mode_tok.col)

            def _ref(k: str) -> str | None:
                v = kw_args.get(k)
                if v is None:
                    return None
                return v.ref if isinstance(v, _NodeRef) else str(v)

            g.set_output(
                pos_args[0].ref,
                emissive_node=_ref("emissive"),
                height_node=_ref("height"),
                normal_node=_ref("normal"),
                roughness_node=_ref("roughness"),
                metallic_node=_ref("metallic"),
                ao_node=_ref("ao"),
                sss_node=_ref("sss"),
                porosity_node=_ref("porosity"),
                normal_strength=kw_args.get("normal_strength"),
                dielectric_f0=kw_args.get("dielectric_f0"),
            )
            return

        raise TexDslError(f"Unknown @output mode {mode!r}; expected 'material' or 'rgba'", mode_tok.line, mode_tok.col)

    def _rename_node(self, old_id: str, new_id: str) -> None:
        g = self._ensure_graph()
        if old_id == new_id:
            return
        if new_id in g.nodes:
            raise TexDslError(f"Duplicate node name {new_id!r}", self._cur().line, self._cur().col)
        spec = g.nodes.pop(old_id)
        spec.id = new_id
        g.nodes[new_id] = spec
        g.node_order[g.node_order.index(old_id)] = new_id
        if g._active_pass is not None and old_id in g._active_pass.node_ids:
            g._active_pass.node_ids[g._active_pass.node_ids.index(old_id)] = new_id
        for grp in g.groups:
            if old_id in grp["nodes"]:
                grp["nodes"][grp["nodes"].index(old_id)] = new_id

    def _resolve_alias(self, ref: str) -> str:
        if "." in ref:
            base, port = ref.split(".", 1)
            mapped = self.aliases.get(base, base)
            return f"{mapped}.{port}" if "." not in mapped else mapped
        return self.aliases.get(ref, ref)

    # --- Statements inside @pass or top-level ---
    def _parse_statement(self) -> None:
        g = self._ensure_graph()
        t = self._cur()
        if t.kind == "IDENT" and t.val == "preview" and self._peek().kind != "EQ":
            self._eat("IDENT")
            ref_tok = self._parse_expr(target_id=None)
            if not isinstance(ref_tok, _NodeRef):
                raise TexDslError("preview requires a node reference", t.line, t.col)
            g.set_pass_preview(self._resolve_alias(ref_tok.ref))
            return

        lhs = self._eat("IDENT")
        self._eat("EQ")
        before_n = len(g.node_order)
        res = self._parse_expr(target_id=None)
        if not isinstance(res, _NodeRef):
            # Assigning a literal directly creates a constant node
            g.add("constant", id=lhs.val, value=res)
        else:
            ref = self._resolve_alias(res.ref)
            if len(g.node_order) > before_n and "." not in ref and ref == g.node_order[-1]:
                # Rename the final node produced by this expression directly to lhs.val!
                self._rename_node(ref, lhs.val)
            else:
                # Zero-cost alias (preserves MATERIAL bus & multi-port refs cleanly!)
                self.aliases[lhs.val] = ref

    # --- Expression Parser (Precedence: | -> & -> +/- -> * -> unary !/- -> atom) ---
    def _parse_expr(self, target_id: str | None) -> Any:
        return self._parse_or(target_id=target_id)

    def _parse_or(self, target_id: str | None) -> Any:
        left = self._parse_and(target_id=None)
        while self._cur().kind == "OP" and self._cur().val == "|":
            self._eat("OP")
            right = self._parse_and(target_id=None)
            is_last = not (self._cur().kind == "OP" and self._cur().val == "|")
            left = self._emit_binop("max", left, right, target_id=(target_id if is_last else None))
        if target_id and isinstance(left, _NodeRef) and left.ref != target_id:
            return left
        return left

    def _parse_and(self, target_id: str | None) -> Any:
        left = self._parse_add(target_id=None)
        while self._cur().kind == "OP" and self._cur().val == "&":
            self._eat("OP")
            right = self._parse_add(target_id=None)
            is_last = not (self._cur().kind == "OP" and self._cur().val == "&")
            left = self._emit_binop("min", left, right, target_id=(target_id if is_last else None))
        return left

    def _parse_add(self, target_id: str | None) -> Any:
        left = self._parse_mul(target_id=None)
        while self._cur().kind == "OP" and self._cur().val in ("+", "-"):
            op_ch = self._eat("OP").val
            right = self._parse_mul(target_id=None)
            is_last = not (self._cur().kind == "OP" and self._cur().val in ("+", "-"))
            left = self._emit_binop("add" if op_ch == "+" else "sub", left, right, target_id=(target_id if is_last else None))
        return left

    def _parse_mul(self, target_id: str | None) -> Any:
        # Look ahead: if this is a top-level bare call/atom with no trailing operator, pass target_id down!
        left = self._parse_unary(target_id=None)
        while self._cur().kind == "OP" and self._cur().val == "*":
            self._eat("OP")
            right = self._parse_unary(target_id=None)
            is_last = not (self._cur().kind == "OP" and self._cur().val == "*")
            left = self._emit_binop("mul", left, right, target_id=(target_id if is_last else None))
        return left

    def _parse_unary(self, target_id: str | None) -> Any:
        if self._cur().kind == "OP" and self._cur().val in ("!", "~"):
            self._eat("OP")
            sub = self._parse_unary(target_id=None)
            ref = self._as_node_ref(sub)
            g = self._ensure_graph()
            nid = target_id or self._next_tmp("inv")
            g.add("levels", id=nid, inputs={"src": ref}, invert=True)
            return _NodeRef(nid)
        if self._cur().kind == "OP" and self._cur().val == "-":
            self._eat("OP")
            tok = self._eat("NUMBER")
            return -self._parse_num_str(tok.val)
        return self._parse_atom(target_id=target_id)

    def _parse_atom(self, target_id: str | None) -> Any:
        t = self._cur()
        if t.kind == "LPAREN":
            self._eat("LPAREN")
            val = self._parse_expr(target_id=target_id)
            self._eat("RPAREN")
            return val
        if t.kind == "AT":
            return self._resolve_palette_ref()
        if t.kind == "STRING":
            self._eat("STRING")
            return t.val
        if t.kind == "NUMBER":
            self._eat("NUMBER")
            return self._parse_num_str(t.val)
        if t.kind == "LBRACK":
            return self._parse_list_literal()
        if t.kind == "LBRACE":
            return self._parse_map_literal()
        if t.kind == "IDENT":
            # Booleans / null
            if t.val == "true":
                self._eat("IDENT")
                return True
            if t.val == "false":
                self._eat("IDENT")
                return False
            if t.val in ("null", "none", "None"):
                self._eat("IDENT")
                return None

            # `use subgraph_name(...)`
            if t.val == "use" and self._peek().kind == "IDENT":
                self._eat("IDENT")
                return self._parse_subgraph_call(target_id=target_id)

            # Function / Node call: `op_name(...)`
            if self._peek().kind == "LPAREN":
                return self._parse_node_call(target_id=target_id)

            # Node reference or multi-port reference: `node_id` or `node_id.port`
            self._eat("IDENT")
            ref = t.val
            if self._cur().kind == "DOT":
                self._eat("DOT")
                port = self._eat("IDENT").val
                ref = f"{ref}.{port}"
            return _NodeRef(self._resolve_alias(ref))

        raise TexDslError(f"Unexpected token {t.kind} ({t.val!r}) in expression", t.line, t.col)

    def _parse_subgraph_call(self, *, target_id: str | None) -> _NodeRef:
        g = self._ensure_graph()
        sg_tok = self._eat("IDENT")
        sg_name = sg_tok.val
        if sg_name not in SUBGRAPH_REGISTRY:
            raise TexDslError(f"Unknown subgraph {sg_name!r}; available: {sorted(SUBGRAPH_REGISTRY)}", sg_tok.line, sg_tok.col)
        self._eat("LPAREN")
        _, kw_args = self._parse_call_args_expr()
        self._eat("RPAREN")
        ins: dict[str, str] = {}
        params: dict[str, Any] = {}
        for k, v in kw_args.items():
            if isinstance(v, _NodeRef):
                ins[k] = v.ref
            else:
                params[k] = v
        pref = target_id or self._next_tmp(sg_name)
        out_id = g.use_subgraph(sg_name, prefix=pref, inputs=ins, **params)
        if target_id and out_id != target_id:
            # Rename or alias last created node cleanly if target_id doesn't exist yet
            if target_id not in g.nodes:
                spec = g.nodes.pop(out_id)
                spec.id = target_id
                g.nodes[target_id] = spec
                g.node_order[g.node_order.index(out_id)] = target_id
                if g._active_pass is not None and out_id in g._active_pass.node_ids:
                    g._active_pass.node_ids[g._active_pass.node_ids.index(out_id)] = target_id
                for grp in g.groups:
                    if out_id in grp["nodes"]:
                        grp["nodes"][grp["nodes"].index(out_id)] = target_id
                out_id = target_id
        return _NodeRef(out_id)

    def _parse_node_call(self, *, target_id: str | None) -> _NodeRef:
        g = self._ensure_graph()
        op_tok = self._eat("IDENT")
        raw_op = op_tok.val
        self._eat("LPAREN")
        pos_args, kw_args = self._parse_call_args_expr()
        self._eat("RPAREN")

        # Ergonomic `select(pattern=..., zone=..., faces=...)` shorthand!
        if raw_op == "select":
            return self._emit_select(pos_args, kw_args, target_id=target_id, line=op_tok.line, col=op_tok.col)

        op = _OP_ALIASES.get(raw_op, raw_op)
        meta = NODE_CATALOG.get(op)
        inputs: dict[str, str] = {}
        params: dict[str, Any] = {}

        # 1. Map positional arguments:
        #    If op has explicit _POSITIONAL_PARAMS and no node-ref inputs at front, map to params;
        #    otherwise map positional args to NODE_CATALOG[op].inputs in declared port order!
        if op in _POSITIONAL_PARAMS and (meta is None or not meta.inputs):
            pnames = _POSITIONAL_PARAMS[op]
            for idx, val in enumerate(pos_args):
                if idx >= len(pnames):
                    raise TexDslError(f"Too many positional args for {raw_op}(...)", op_tok.line, op_tok.col)
                if isinstance(val, _NodeRef):
                    inputs[pnames[idx]] = val.ref
                else:
                    params[pnames[idx]] = val
        elif meta is not None and meta.inputs:
            port_names = [p.name for p in meta.inputs]
            for idx, val in enumerate(pos_args):
                if idx < len(port_names):
                    if isinstance(val, _NodeRef):
                        inputs[port_names[idx]] = val.ref
                    elif op == "material_make" and port_names[idx] != "albedo":
                        params[port_names[idx]] = val
                    else:
                        # Scalar literal passed to a node input port -> auto-promote to constant node!
                        inputs[port_names[idx]] = self._as_node_ref(val)
                else:
                    # Overflow positional args go to _POSITIONAL_PARAMS if registered
                    extra_idx = idx - len(port_names)
                    pnames = _POSITIONAL_PARAMS.get(op, ())
                    if extra_idx < len(pnames):
                        params[pnames[extra_idx]] = val
                    else:
                        raise TexDslError(f"Too many positional args for {raw_op}(...)", op_tok.line, op_tok.col)
        else:
            for idx, val in enumerate(pos_args):
                if isinstance(val, _NodeRef):
                    inputs["src" if idx == 0 else f"in_{idx}"] = val.ref
                else:
                    params[f"arg_{idx}"] = val

        # 2. Map keyword arguments cleanly to inputs vs params!
        known_in_ports = {p.name for p in meta.inputs} if meta is not None else set()
        for k, v in kw_args.items():
            if isinstance(v, _NodeRef):
                inputs[k] = v.ref
            elif k in known_in_ports and op not in ("material_make",):
                # Allow scalar literal passed to an input port (e.g., mask=1.0)
                if v is not None:
                    inputs[k] = self._as_node_ref(v)
            else:
                params[k] = v

        # Canonical channel alias for mesh_channel("ao") -> "baked_ao"
        if op == "mesh_channel" and "channel" in params:
            ch = str(params["channel"])
            params["channel"] = _MESH_CHANNEL_ALIASES.get(ch, ch)

        # Auto-rewrite `math(..., op="mul")` -> `math_op="mul"`
        if op == "math" and "op" in params:
            params["math_op"] = params.pop("op")
        if op == "morph_op" and "op" in params:
            params["morph_op"] = params.pop("op")

        nid = target_id or self._next_tmp(op)
        g.add(op, id=nid, inputs=inputs, **params)
        return _NodeRef(nid)

    def _emit_select(self, pos_args: list[Any], kw_args: dict[str, Any], *, target_id: str | None, line: int, col: int) -> _NodeRef:
        g = self._ensure_graph()
        pat = kw_args.get("pattern") or (pos_args[0] if pos_args else None)
        zone = kw_args.get("zone")
        faces = kw_args.get("faces")
        parts: list[str] = []
        if pat is not None:
            parts.append(g.add("mesh_channel", id=(target_id if (target_id and not zone and not faces) else self._next_tmp("sel_el")), channel="element", pattern=str(pat)))
        if zone is not None:
            parts.append(g.add("mesh_channel", id=(target_id if (target_id and not pat and not faces) else self._next_tmp("sel_zn")), channel="zone", zone=str(zone)))
        if faces is not None:
            parts.append(g.add("mesh_channel", id=(target_id if (target_id and not pat and not zone) else self._next_tmp("sel_fc")), channel="face", faces=str(faces)))
        if not parts:
            raise TexDslError("select(...) requires at least one of pattern=, zone=, or faces=", line, col)
        cur = parts[0]
        for idx, nxt in enumerate(parts[1:], start=1):
            is_last = idx == len(parts) - 1
            nid = target_id if (is_last and target_id) else self._next_tmp("sel_and")
            cur = g.add("math", id=nid, inputs={"a": cur, "b": nxt}, math_op="mul")
        return _NodeRef(cur)

    def _emit_binop(self, math_op: str, left: Any, right: Any, *, target_id: str | None) -> _NodeRef:
        g = self._ensure_graph()
        a_ref = self._as_node_ref(left)
        b_ref = self._as_node_ref(right)
        nid = target_id or self._next_tmp(math_op)
        g.add("math", id=nid, inputs={"a": a_ref, "b": b_ref}, math_op=math_op)
        return _NodeRef(nid)

    def _as_node_ref(self, val: Any) -> str:
        if isinstance(val, _NodeRef):
            return val.ref
        g = self._ensure_graph()
        cid = self._next_tmp("const")
        g.add("constant", id=cid, value=val)
        return cid

    def _resolve_palette_ref(self) -> Any:
        at = self._eat("AT")
        first = self._eat("IDENT").val
        parts = [first]
        while self._cur().kind == "DOT":
            self._eat("DOT")
            parts.append(self._eat("IDENT").val)
        if parts[0] == "bank" and len(parts) >= 2:
            bkey = parts[1]
            if bkey not in PALETTE_BANK:
                raise TexDslError(f"Unknown PALETTE_BANK entry @bank.{bkey}; choices: {sorted(PALETTE_BANK)}", at.line, at.col)
            stops = list(PALETTE_BANK[bkey])
            if len(parts) == 3 and parts[2] == "hex":
                return [c for _, c in stops]
            return stops
        if parts[0] not in self.palettes:
            if parts[0] in PALETTE_BANK:
                stops = list(PALETTE_BANK[parts[0]])
            else:
                raise TexDslError(f"Unknown palette @{parts[0]}", at.line, at.col)
        else:
            stops = list(self.palettes[parts[0]])
        if len(parts) == 2 and parts[1] == "hex":
            return [c for _, c in stops]
        return stops

    # --- Argument & Literal Parsers ---
    def _parse_call_args_expr(self) -> tuple[list[Any], dict[str, Any]]:
        pos_args: list[Any] = []
        kw_args: dict[str, Any] = {}
        while self._cur().kind not in ("RPAREN", "EOF"):
            if self._cur().kind == "IDENT" and self._peek().kind == "EQ":
                k = self._eat("IDENT").val
                self._eat("EQ")
                kw_args[k] = self._parse_expr(target_id=None)
            else:
                pos_args.append(self._parse_expr(target_id=None))
            if self._cur().kind == "COMMA":
                self._eat("COMMA")
            else:
                break
        return pos_args, kw_args

    def _parse_call_args_literals(self) -> tuple[list[Any], dict[str, Any]]:
        pos_args: list[Any] = []
        kw_args: dict[str, Any] = {}
        while self._cur().kind not in ("RPAREN", "EOF"):
            if self._cur().kind == "IDENT" and self._peek().kind == "EQ":
                k = self._eat("IDENT").val
                self._eat("EQ")
                kw_args[k] = self._parse_literal()
            else:
                pos_args.append(self._parse_literal())
            if self._cur().kind == "COMMA":
                self._eat("COMMA")
            else:
                break
        return pos_args, kw_args

    def _parse_literal(self) -> Any:
        t = self._cur()
        if t.kind == "OP" and t.val == "-":
            self._eat("OP")
            return -self._parse_num_str(self._eat("NUMBER").val)
        if t.kind == "NUMBER":
            self._eat("NUMBER")
            return self._parse_num_str(t.val)
        if t.kind == "STRING":
            self._eat("STRING")
            return t.val
        if t.kind == "IDENT":
            self._eat("IDENT")
            if t.val == "true":
                return True
            if t.val == "false":
                return False
            if t.val in ("null", "none", "None"):
                return None
            return t.val
        if t.kind == "LBRACK":
            return self._parse_list_literal()
        if t.kind == "LBRACE":
            return self._parse_map_literal()
        if t.kind == "AT":
            return self._resolve_palette_ref()
        raise TexDslError(f"Expected literal value, got {t.kind} ({t.val!r})", t.line, t.col)

    def _parse_list_literal(self) -> list[Any]:
        self._eat("LBRACK")
        items: list[Any] = []
        while self._cur().kind not in ("RBRACK", "EOF"):
            items.append(self._parse_literal())
            if self._cur().kind == "COMMA":
                self._eat("COMMA")
            else:
                break
        self._eat("RBRACK")
        return items

    def _parse_map_literal(self) -> dict[str, Any]:
        self._eat("LBRACE")
        out: dict[str, Any] = {}
        while self._cur().kind not in ("RBRACE", "EOF"):
            k_tok = self._cur()
            if k_tok.kind in ("IDENT", "STRING", "NUMBER"):
                self._eat()
                key = k_tok.val
            else:
                raise TexDslError(f"Expected map key, got {k_tok.kind} ({k_tok.val!r})", k_tok.line, k_tok.col)
            self._eat("COLON")
            out[key] = self._parse_literal()
            if self._cur().kind == "COMMA":
                self._eat("COMMA")
        self._eat("RBRACE")
        return out

    @staticmethod
    def _parse_num_str(s: str) -> int | float:
        return float(s) if ("." in s or "e" in s or "E" in s) else int(s)


# ---------------------------------------------------------------------------
# Public API: parse_texdsl / load_texdsl / export_texdsl
# ---------------------------------------------------------------------------


def parse_texdsl(source: str, *, mesh_context: MeshBakeContext | None = None) -> Any:
    """Compile a declarative `.texdsl` texture program directly into a runnable `TexGraph` DAG."""
    toks = _tokenize(source)
    parser = _TexDslParser(toks, mesh_context=mesh_context)
    return parser.parse()


def load_texdsl(path: Path | str, *, mesh_context: MeshBakeContext | None = None) -> Any:
    """Load and compile a `.texdsl` file from disk into a `TexGraph` DAG."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_texdsl(text, mesh_context=mesh_context)


def _fmt_lit(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return repr(val)
    if isinstance(val, str):
        esc = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{esc}"'
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_fmt_lit(x) for x in val) + "]"
    if isinstance(val, dict):
        return "{" + ", ".join(f"{k}: {_fmt_lit(v)}" for k, v in val.items()) + "}"
    return repr(val)


def export_texdsl(graph: Any) -> str:
    """Decompile any `TexGraph` DAG losslessly into clean, readable `.texdsl` source code!"""
    lines: list[str] = []
    anim_cfg = getattr(graph, "animation_config", {}) or {}
    lines.append("@texture {")
    lines.append(f'  name: {_fmt_lit(graph.name)}')
    lines.append(f"  size: [{graph.width}, {graph.height}]")
    lines.append(f"  tileable: {'true' if graph.tileable else 'false'}")
    lines.append(f"  seed: {int(graph.seed)}")
    if getattr(graph, "domain", None):
        lines.append(f'  domain: {_fmt_lit(graph.domain)}')
    if int(anim_cfg.get("frames", 1)) > 1:
        lines.append(f"  frames: {int(anim_cfg['frames'])}")
        lines.append(f"  frametime: {int(anim_cfg.get('frametime', 2))}")
        lines.append(f"  interpolate: {'true' if anim_cfg.get('interpolate') else 'false'}")
        if anim_cfg.get("frames_schedule"):
            lines.append(f"  frames_schedule: {_fmt_lit(anim_cfg['frames_schedule'])}")
    lines.append("}")
    lines.append("")

    def _emit_node_stmt(nid: str, indent: str = "  ") -> str:
        sp = graph.nodes[nid]
        args: list[str] = []
        for k, ref in sp.inputs.items():
            args.append(f"{k}={ref}")
        for k, v in sp.params.items():
            args.append(f"{k}={_fmt_lit(v)}")
        return f"{indent}{nid} = {sp.op}({', '.join(args)})"

    emitted: set[str] = set()
    if graph.passes:
        for ps in graph.passes:
            lines.append(f'@pass {ps.name} {_fmt_lit(ps.title)} {{')
            for nid in ps.node_ids:
                if nid in graph.nodes:
                    lines.append(_emit_node_stmt(nid))
                    emitted.add(nid)
            if ps.preview_ref:
                lines.append(f"  preview {ps.preview_ref}")
            lines.append("}")
            lines.append("")

    remaining = [nid for nid in graph.node_order if nid not in emitted]
    if remaining:
        lines.append('@pass main "Main Graph" {')
        for nid in remaining:
            lines.append(_emit_node_stmt(nid))
        lines.append("}")
        lines.append("")

    if graph.param_drivers:
        lines.append("@animate {")
        for d in graph.param_drivers:
            kind = d.get("kind", "sine")
            parts: list[str] = []
            if d.get("keyframes"):
                parts.append(f"keyframes={_fmt_lit(d['keyframes'])}")
            else:
                for k in ("min_val", "max_val", "freq", "phase_offset", "duty", "octaves"):
                    if k in d:
                        parts.append(f"{k}={_fmt_lit(d[k])}")
            lines.append(f"  drive {d['node_id']}.{d['param_key']} = {kind}({', '.join(parts)})")
        lines.append("}")
        lines.append("")

    if graph.exposed_params:
        lines.append("@expose {")
        for pub, info in graph.exposed_params.items():
            rng = ""
            if info.get("min") is not None and info.get("max") is not None:
                st = f" step={info['step']}" if info.get("step") is not None else ""
                rng = f" [{info['min']}..{info['max']}{st}]"
            lines.append(f"  {pub}: {info['node_id']}.{info['param_key']}{rng}")
        lines.append("}")
        lines.append("")

    if graph.output_material_node:
        ns = graph.pbr_config.get("normal_strength", 2.4)
        f0 = graph.pbr_config.get("dielectric_f0", 0.04)
        lines.append(f"@output material({graph.output_material_node}, normal_strength={ns}, dielectric_f0={f0})")
    elif graph.output_node:
        oargs = [graph.output_node]
        if graph.emissive_node:
            oargs.append(f"emissive={graph.emissive_node}")
        for ch, ref in graph.pbr_nodes.items():
            oargs.append(f"{ch}={ref}")
        lines.append(f"@output rgba({', '.join(oargs)})")

    return "\n".join(lines).rstrip() + "\n"
