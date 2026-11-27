"""UI-Ready Typed Node Registry & Schema Exporter for `synccraft.texgraph`.

Provides full Pixel-Composer-parity metadata for every TexGraph node:
- Category, display name, description, typed input/output ports (`SCALAR|RGBA|MASK|HEIGHT|NORMAL`)
- Parameter widget metadata (`slider`, `int`, `select`, `color`, `gradient_stops`, `curve`, `checkbox`, `text`)
- Serializable JSON catalog (`export_node_schema_json()`) ready to drive any future visual Node-Graph UI!
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class PortSchema:
    name: str
    dtype: str  # "SCALAR" | "RGBA" | "MASK" | "HEIGHT" | "NORMAL" | "ANY"
    optional: bool = False


@dataclass(frozen=True)
class ParamSchema:
    name: str
    widget: str  # "slider" | "int" | "select" | "color" | "gradient_stops" | "palette" | "curve" | "checkbox" | "text" | "points" | "vec2" | "vec3"
    default: Any
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    group: str = "General"
    animatable: bool = True
    tooltip: str = ""
    visible_if: dict[str, Any] | None = None


PORT_COLORS: dict[str, str] = {
    "SCALAR": "#94A3B8",
    "MASK": "#F8FAFC",
    "HEIGHT": "#38BDF8",
    "RGBA": "#F59E0B",
    "NORMAL": "#818CF8",
    "MATERIAL": "#10B981",
    "ANY": "#E879F9",
}

PORT_COMPATIBILITY: dict[str, tuple[str, ...]] = {
    "SCALAR": ("SCALAR", "MASK", "HEIGHT", "ANY"),
    "MASK": ("MASK", "SCALAR", "HEIGHT", "ANY"),
    "HEIGHT": ("HEIGHT", "SCALAR", "MASK", "ANY"),
    "RGBA": ("RGBA", "ANY"),
    "NORMAL": ("NORMAL", "RGBA", "ANY"),
    "MATERIAL": ("MATERIAL", "RGBA", "ANY"),
    "ANY": ("SCALAR", "MASK", "HEIGHT", "RGBA", "NORMAL", "MATERIAL", "ANY"),
}


@dataclass(frozen=True)
class NodeMeta:
    op: str
    category: str
    display_name: str
    description: str
    inputs: tuple[PortSchema, ...] = ()
    output_type: str = "SCALAR"
    outputs: tuple[PortSchema, ...] = ()
    params: tuple[ParamSchema, ...] = ()
    time_aware: bool = False


NODE_CATALOG: dict[str, NodeMeta] = {
    # --- Gradients & Smooth Paths ---
    "constant": NodeMeta("constant", "Gradients & Paths", "Constant", "Uniform scalar or hex color fill.", (), "ANY", params=(ParamSchema("value", "text", 0.5, tooltip="Numeric float or #RRGGBB hex"),)),
    "gradient": NodeMeta("gradient", "Gradients & Paths", "Simple Gradient", "Basic directional/radial scalar gradient.", (), "SCALAR", params=(ParamSchema("direction", "select", "vertical", choices=("vertical", "top_to_bottom", "horizontal", "left_to_right", "diagonal", "radial")),)),
    "gradient_field": NodeMeta("gradient_field", "Gradients & Paths", "Studio Gradient Field", "Linear, radial, conical, diamond, box or reflected smooth gradient with easing.", (), "SCALAR", params=(
        ParamSchema("kind", "select", "linear", choices=("linear", "radial", "radial_out", "diamond", "box", "conical", "reflected")),
        ParamSchema("angle_deg", "slider", 90.0, -180.0, 180.0, 1.0),
        ParamSchema("radius", "slider", 0.55, 0.05, 2.0, 0.01),
        ParamSchema("ease", "select", "smoothstep", choices=("linear", "smoothstep", "smootherstep", "ease_in", "ease_out")),
    )),
    "multi_stop_gradient": NodeMeta("multi_stop_gradient", "Gradients & Paths", "Multi-Stop OKLab Gradient", "Silky smooth perceptual OKLab multi-stop color ramp with smooth-to-pixel blend.", (PortSchema("src", "SCALAR"), PortSchema("alpha_mask", "MASK", True)), "RGBA", params=(
        ParamSchema("stops", "gradient_stops", [(0.0, "#190E33"), (0.5, "#7C3AED"), (1.0, "#A5F3FC")]),
        ParamSchema("space", "select", "oklab", choices=("oklab", "srgb")),
        ParamSchema("smooth_blend", "slider", 1.0, 0.0, 1.0, 0.05),
        ParamSchema("quantize_steps", "int", 0, 0, 32, 1),
    )),
    "cosine_gradient": NodeMeta("cosine_gradient", "Gradients & Paths", "Cosine Palette (Gradient Cos)", "Iridescent procedural cosine color ramp.", (PortSchema("src", "SCALAR"),), "RGBA", params=(
        ParamSchema("a", "vec3", [0.5, 0.5, 0.5]),
        ParamSchema("b", "vec3", [0.5, 0.5, 0.5]),
        ParamSchema("c", "vec3", [1.0, 1.0, 1.0]),
        ParamSchema("d", "vec3", [0.0, 0.33, 0.67]),
    ), time_aware=True),
    "spline_path": NodeMeta("spline_path", "Gradients & Paths", "Spline Path Gradient", "Smooth Catmull-Rom curve ribbon, along-path gradient, or distance field.", (), "SCALAR", params=(
        ParamSchema("points", "points", [(0.1, 0.2), (0.5, 0.7), (0.9, 0.3)]),
        ParamSchema("stroke_width", "slider", 0.12, 0.01, 0.5, 0.01),
        ParamSchema("output", "select", "ribbon", choices=("ribbon", "along", "sdf")),
    )),
    "per_island_gradient": NodeMeta("per_island_gradient", "Gradients & Paths", "Per-Island Face Gradient", "Smooth top-to-bottom, convex center-out, or edge-rim gradient per Blockbench face.", (), "SCALAR", params=(
        ParamSchema("direction", "select", "top_to_bottom", choices=("top_to_bottom", "bottom_to_top", "left_to_right", "diag", "center_out", "edge_rim")),
        ParamSchema("curve", "slider", 1.0, 0.2, 3.0, 0.05),
    )),
    "body_flow_gradient": NodeMeta("body_flow_gradient", "Gradients & Paths", "3D Body-Flow Gradient", "Seamless creature-space dorsal-ventral, head-to-tail, root-to-tip, or half-Lambert light wrap.", (), "SCALAR", params=(
        ParamSchema("axis", "select", "dorsal_ventral", choices=("dorsal_ventral", "head_to_tail", "center_to_tips", "top_light_wrap")),
        ParamSchema("bias", "slider", 0.0, -0.5, 0.5, 0.02),
        ParamSchema("gamma", "slider", 1.0, 0.3, 3.0, 0.05),
    )),
    # --- Noise ---
    "value_noise": NodeMeta("value_noise", "Noise", "Value Noise", "Tileable bicubic value noise.", (), "SCALAR", params=(ParamSchema("scale", "slider", 8.0, 1.0, 64.0, 0.5),)),
    "perlin": NodeMeta("perlin", "Noise", "Perlin Noise", "Tileable gradient Perlin noise with seamless time loop.", (), "SCALAR", params=(ParamSchema("scale", "slider", 8.0, 1.0, 64.0, 0.5),), time_aware=True),
    "fbm": NodeMeta("fbm", "Noise", "fBm / Ridged / Billow", "Multi-octave fractal Brownian motion noise.", (), "SCALAR", params=(ParamSchema("scale", "slider", 6.0, 1.0, 48.0, 0.5), ParamSchema("octaves", "int", 4, 1, 8, 1), ParamSchema("mode", "select", "fbm", choices=("fbm", "ridged", "billow", "turbulence"))), time_aware=True),
    "voronoi": NodeMeta("voronoi", "Noise", "Voronoi / Cellular", "Tileable Worley cellular F1, F2, cracks, edge, or cell ID.", (), "SCALAR", params=(ParamSchema("scale", "slider", 6.0, 1.0, 48.0, 0.5), ParamSchema("output", "select", "edge", choices=("f1", "f2", "f2_minus_f1", "edge", "cracks", "cell_id"))), time_aware=True),
    "gabor": NodeMeta("gabor", "Noise", "Anisotropic Gabor", "Directional brushed metal, striations, and bone flutes.", (), "SCALAR", params=(ParamSchema("scale", "slider", 12.0, 2.0, 64.0, 1.0), ParamSchema("angle_deg", "slider", 0.0, -180.0, 180.0, 1.0))),
    "domain_warp": NodeMeta("domain_warp", "Noise", "Domain / Curl Warp", "Organic vector-field distortion warp.", (PortSchema("src", "ANY"),), "ANY", params=(ParamSchema("strength", "slider", 0.08, 0.0, 0.5, 0.01), ParamSchema("mode", "select", "fbm", choices=("fbm", "curl"))), time_aware=True),
    "noise_3d": NodeMeta("noise_3d", "Noise", "3D Rest-Space Noise", "Seamless 3D world-space noise across creature UV seams.", (), "SCALAR", params=(ParamSchema("scale", "slider", 0.22, 0.02, 2.0, 0.02), ParamSchema("mode", "select", "fbm", choices=("fbm", "ridged", "billow"))), time_aware=True),
    # --- Geometric & Structural Patterns ---
    "grid": NodeMeta("grid", "Geometric Patterns", "Grid Tiles", "Beveled rectangular tiles & mortar.", (), "HEIGHT"),
    "brick": NodeMeta("brick", "Geometric Patterns", "Brick Masonry", "Running, stacked, herringbone, or basketweave brickwork.", (), "HEIGHT"),
    "hex_grid": NodeMeta("hex_grid", "Geometric Patterns", "Hexagon Grid", "Beveled hexagonal tiles & conduits.", (), "HEIGHT"),
    "tri_grid": NodeMeta("tri_grid", "Geometric Patterns", "Triangle Grid", "Faceted triangular mosaic.", (), "HEIGHT"),
    "diamond": NodeMeta("diamond", "Geometric Patterns", "Diamond / Chevron", "Quilted diamond or herringbone chevron.", (), "HEIGHT"),
    "truchet": NodeMeta("truchet", "Geometric Patterns", "Truchet Curves", "Interlocking calligraphic arcs & tiles.", (), "SCALAR"),
    "mosaic": NodeMeta("mosaic", "Geometric Patterns", "Voronoi Mosaic", "Beveled irregular stone/scute mosaic.", (), "HEIGHT"),
    "radial_rings": NodeMeta("radial_rings", "Geometric Patterns", "Radial Rings", "Concentric mandala/astrolabe rings & spokes.", (), "SCALAR", time_aware=True),
    "energy_veins": NodeMeta("energy_veins", "Geometric Patterns", "Energy Veins", "Animated lightning, circuit, or nebula energy streams.", (), "MASK", time_aware=True),
    # --- Painterly & Biological Craft ---
    "brushstrokes": NodeMeta("brushstrokes", "Painterly & Bio", "Directional Brushstrokes", "Long smooth overlapping hand-painted studio strokes.", (), "SCALAR", params=(ParamSchema("angle_deg", "slider", 0.0, -180.0, 180.0, 1.0), ParamSchema("contrast", "slider", 0.35, 0.05, 1.0, 0.05))),
    "layered_plates": NodeMeta("layered_plates", "Painterly & Bio", "Chunky Scale / Armor Plates", "Calm Blockbench staggered scale/armor plates with 1px lit rim + under-seam.", (), "HEIGHT"),
    "smooth_wing_rgba": NodeMeta("smooth_wing_rgba", "Painterly & Bio", "Master Dragon Wing Membrane", "Backlit translucent smooth-gradient dragon wing with bone AO, tension folds & organic scalloped edge.", (PortSchema("local_u", "SCALAR"), PortSchema("local_v", "SCALAR")), "RGBA"),
    "sculpted_pinion_wing": NodeMeta(
        "sculpted_pinion_wing",
        "Painterly & Bio",
        "Sculpted Pinion-Blade Wing (MATERIAL)",
        "Aerodynamic overlapping metallic obsidian pinion blades + brushed gold astrolabe arc + Twin-Spectrum neon quill filaments (single-eval full MATERIAL bus).",
        (PortSchema("span_u", "SCALAR"), PortSchema("trail_v", "SCALAR"), PortSchema("face_is_down", "SCALAR", True), PortSchema("wave", "SCALAR", True)),
        "MATERIAL",
        outputs=(
            PortSchema("albedo", "RGBA"), PortSchema("alpha", "MASK"), PortSchema("height", "HEIGHT"),
            PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"),
            PortSchema("ao", "SCALAR"), PortSchema("emissive", "RGBA"), PortSchema("sss", "SCALAR"),
            PortSchema("glow_mask", "MASK"), PortSchema("gold_mask", "MASK"),
        ),
        params=(
            ParamSchema("vanes", "int", 5, 2, 10, 1),
            ParamSchema("sweep", "slider", 0.84, 0.2, 1.4, 0.02),
            ParamSchema("envelope_style", "select", "primary", choices=("primary", "distal_tip", "inner_curtain", "forearm_covert")),
            ParamSchema("smooth_blend", "slider", 0.70, 0.0, 1.0, 0.05),
            ParamSchema("gold_arc", "checkbox", True),
            ParamSchema("quill_glow_strength", "slider", 0.92, 0.0, 1.2, 0.05),
        ),
    ),
    "smooth_wing_multi": NodeMeta(
        "smooth_wing_multi",
        "Painterly & Bio",
        "Backlit Dragon Membrane (Multi-Port)",
        "Single-evaluation multi-port backlit translucent smooth-gradient dragon wing membrane.",
        (PortSchema("local_u", "SCALAR"), PortSchema("local_v", "SCALAR")),
        "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("alpha", "MASK"), PortSchema("rim_mask", "MASK"), PortSchema("glow_mask", "MASK"), PortSchema("height", "HEIGHT")),
    ),
    "gradient_eye_multi": NodeMeta(
        "gradient_eye_multi",
        "Painterly & Bio",
        "Non-Standard Gradient Eye (Multi-Port)",
        "Single-evaluation mirror-symmetric luminous multi-stop gradient eye emitting albedo + emissive_mask.",
        (PortSchema("local_u", "SCALAR"), PortSchema("local_v", "SCALAR"), PortSchema("side_sign", "SCALAR", True)),
        "RGBA",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("emissive_mask", "MASK")),
    ),
    "emissive_composite": NodeMeta(
        "emissive_composite",
        "PBR & Normal",
        "Emissive Composite (Multi-Port)",
        "Single-evaluation emissive bloom bleed composite returning both albedo and isolated emissive ports.",
        (PortSchema("albedo", "RGBA"), PortSchema("mask", "MASK"), PortSchema("color", "RGBA", True)),
        "RGBA",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("emissive", "RGBA")),
    ),
    # --- Industry-Standard Material Bus & Multi-Pass Pipeline ---
    "material_make": NodeMeta(
        "material_make",
        "Material Bus (PBR)",
        "Make Material Bus",
        "Packs albedo + height/normal/roughness/metallic/ao/emissive/sss/porosity into a unified MATERIAL bus.",
        (
            PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT", True), PortSchema("normal", "NORMAL", True),
            PortSchema("roughness", "SCALAR", True), PortSchema("metallic", "SCALAR", True), PortSchema("ao", "SCALAR", True),
            PortSchema("emissive", "RGBA", True), PortSchema("sss", "SCALAR", True), PortSchema("porosity", "SCALAR", True),
        ),
        "MATERIAL",
        outputs=(
            PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"),
            PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"),
            PortSchema("emissive", "RGBA"), PortSchema("sss", "SCALAR"), PortSchema("porosity", "SCALAR"),
        ),
        params=(
            ParamSchema("roughness", "slider", 0.55, 0.04, 1.0, 0.02),
            ParamSchema("metallic", "slider", 0.0, 0.0, 1.0, 0.02),
            ParamSchema("normal_strength", "slider", 2.4, 0.2, 6.0, 0.1),
        ),
    ),
    "material_blend": NodeMeta(
        "material_blend",
        "Material Bus (PBR)",
        "Blend Material Buses",
        "Composites two full PBR MATERIAL buses simultaneously across albedo, normal, height, roughness, metallic, AO, SSS & emissive.",
        (PortSchema("base", "MATERIAL"), PortSchema("top", "MATERIAL"), PortSchema("mask", "MASK", True)),
        "MATERIAL",
        outputs=(
            PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"),
            PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"),
            PortSchema("emissive", "RGBA"), PortSchema("sss", "SCALAR"), PortSchema("porosity", "SCALAR"),
            PortSchema("blend_mask", "MASK"),
        ),
        params=(
            ParamSchema("opacity", "slider", 1.0, 0.0, 1.0, 0.05),
            ParamSchema("normal_mode", "select", "lerp", choices=("lerp", "rnm")),
        ),
    ),
    "material_height_blend": NodeMeta(
        "material_height_blend",
        "Material Bus (PBR)",
        "Height-Blend Material (Substance)",
        "Physical height-driven PBR material composite: higher relief pokes cleanly through cracks/seams with automatic contact AO.",
        (PortSchema("base", "MATERIAL"), PortSchema("top", "MATERIAL"), PortSchema("mask", "MASK", True)),
        "MATERIAL",
        outputs=(
            PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"),
            PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"),
            PortSchema("emissive", "RGBA"), PortSchema("sss", "SCALAR"), PortSchema("porosity", "SCALAR"),
            PortSchema("blend_mask", "MASK"), PortSchema("seam_mask", "MASK"),
        ),
        params=(
            ParamSchema("height_offset", "slider", 0.0, -0.5, 0.5, 0.02),
            ParamSchema("contrast", "slider", 6.0, 0.5, 16.0, 0.25),
            ParamSchema("seam_ao_strength", "slider", 0.25, 0.0, 0.75, 0.05),
        ),
    ),
    "material_grade": NodeMeta(
        "material_grade",
        "Material Bus (PBR)",
        "Grade Material Bus (Pass 4)",
        "Applies 3D contact AO, Color Balance, OKLCH, Vibrance, Despeckle & RNM detail normals cleanly onto a MATERIAL bus.",
        (PortSchema("mat", "MATERIAL"), PortSchema("ao_bake", "SCALAR", True), PortSchema("detail_normal", "NORMAL", True)),
        "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("normal", "NORMAL"), PortSchema("ao", "SCALAR")),
    ),
    "material_set_alpha": NodeMeta(
        "material_set_alpha",
        "Material Bus (PBR)",
        "Material Alpha Cutout",
        "Applies hard or soft alpha cutout cleanly to both albedo and emissive inside a MATERIAL bus.",
        (PortSchema("mat", "MATERIAL"), PortSchema("alpha", "MASK"), PortSchema("mask", "MASK", True)),
        "MATERIAL",
    ),
    "material_pad_islands": NodeMeta(
        "material_pad_islands",
        "Material Bus (PBR)",
        "Material Island Bleed Pad",
        "Clamp-bleeds EVERY PBR channel of a MATERIAL bus cleanly into UV island padding.",
        (PortSchema("mat", "MATERIAL"),),
        "MATERIAL",
    ),
    "gradient_eye_rgba": NodeMeta("gradient_eye_rgba", "Painterly & Bio", "Non-Standard Gradient Eye", "Fierce mythic multi-stop luminous gradient dragon eye (mirror-symmetric L/R).", (PortSchema("local_u", "SCALAR"), PortSchema("local_v", "SCALAR"), PortSchema("side_sign", "SCALAR", True)), "RGBA"),
    "eye_paint_rgba": NodeMeta("eye_paint_rgba", "Painterly & Bio", "7-Layer Iris/Slit Eye", "Detailed 7-layer slit/round iris creature eye.", (PortSchema("local_u", "SCALAR"), PortSchema("local_v", "SCALAR"), PortSchema("side_sign", "SCALAR", True)), "RGBA"),
    "imbricated_scales": NodeMeta("imbricated_scales", "Painterly & Bio", "Imbricated Dragon Scales", "Overlapping keeled dragon/fish/reptile scales.", (), "HEIGHT"),
    "feather_barbs": NodeMeta("feather_barbs", "Painterly & Bio", "Feather Barbs", "Herringbone rachis & angled plume barbs.", (), "HEIGHT"),
    "fur_tufts": NodeMeta("fur_tufts", "Painterly & Bio", "Fur Tufts", "Layered directional drooping fur locks.", (), "HEIGHT"),
    "temple_stone": NodeMeta("temple_stone", "Painterly & Bio", "Temple Carved Stone", "Sunlit stepped ruin masonry with optional carved step-fret relief.", (), "HEIGHT"),
    "wood_planks": NodeMeta("wood_planks", "Painterly & Bio", "Stylized Wood Planks", "Warm sunlit Blockbench timber planks & silky wood grain.", (), "HEIGHT"),
    "foliage_clusters": NodeMeta("foliage_clusters", "Painterly & Bio", "Foliage Leaf Clusters", "Clean painterly 3-tone canopy clusters.", (), "HEIGHT"),
    # --- Simulation & Algorithmic Denoise ---
    "gray_scott": NodeMeta("gray_scott", "Simulation", "Gray-Scott Turing RD", "Reaction-diffusion coral, labyrinth, spots & nebula patterns.", (PortSchema("seed_mask", "MASK", True),), "SCALAR"),
    "cellular_automata": NodeMeta("cellular_automata", "Simulation", "Cellular Automata", "Totalistic cave, coral & crystal growth rules.", (PortSchema("initial_mask", "MASK", True),), "SCALAR"),
    "iterate": NodeMeta("iterate", "Simulation", "Iterate Process", "Iterative erosion, sharpening, ridge enhancement or moss spread.", (PortSchema("src", "SCALAR"),), "SCALAR"),
    "weathering_drip": NodeMeta("weathering_drip", "Simulation", "Weathering Drip", "Gravity-driven cavity streaks & patina.", (PortSchema("src", "SCALAR"), PortSchema("cavity", "MASK", True)), "SCALAR"),
    "kuwahara": NodeMeta("kuwahara", "Denoise & Polish", "Anisotropic Kuwahara", "Edge-preserving painterly pixel-cluster filter.", (PortSchema("src", "ANY"),), "ANY"),
    "bilateral": NodeMeta("bilateral", "Denoise & Polish", "Bilateral Denoise", "Edge-preserving smooth bilateral filter.", (PortSchema("src", "ANY"),), "ANY"),
    "tv_denoise": NodeMeta("tv_denoise", "Denoise & Polish", "Chambolle TV-L2 Denoise", "Total-variation algorithmic denoiser preserving sharp silhouettes.", (PortSchema("src", "ANY"),), "ANY"),
    "nlm_denoise": NodeMeta("nlm_denoise", "Denoise & Polish", "Non-Local Means Denoise", "Self-similar patch denoiser.", (PortSchema("src", "ANY"),), "ANY"),
    "despeckle": NodeMeta("despeckle", "Denoise & Polish", "Morphological Despeckle", "Eliminates isolated orphan 1px noise confetti.", (PortSchema("src", "RGBA"),), "RGBA"),
    "cluster_mode": NodeMeta("cluster_mode", "Denoise & Polish", "Pixel Cluster Mode", "Tightens 2px organized pixel clusters.", (PortSchema("src", "RGBA"),), "RGBA"),
    # --- Transform, Filter & SDF ---
    "tile_transform": NodeMeta("tile_transform", "Transform & Distort", "Tile Transform", "2D Scale, offset, and rotation.", (PortSchema("src", "ANY"),), "ANY"),
    "polar_warp": NodeMeta("polar_warp", "Transform & Distort", "Polar Coordinates", "Cartesian <-> Polar coordinate remap.", (PortSchema("src", "ANY"),), "ANY"),
    "twirl_distort": NodeMeta("twirl_distort", "Transform & Distort", "Twirl Vortex", "Smooth angular vortex distortion.", (PortSchema("src", "ANY"),), "ANY"),
    "wave_distort": NodeMeta("wave_distort", "Transform & Distort", "Wave Displace", "Animated sinusoidal wave distortion.", (PortSchema("src", "ANY"),), "ANY", (), time_aware=True),
    "gaussian_blur": NodeMeta("gaussian_blur", "Filter & SDF", "Gaussian Blur", "Fast separable 2D smooth blur.", (PortSchema("src", "ANY"),), "ANY"),
    "directional_blur": NodeMeta("directional_blur", "Filter & SDF", "Directional Motion Blur", "Angled streak blur.", (PortSchema("src", "ANY"),), "ANY"),
    "unsharp_mask": NodeMeta("unsharp_mask", "Filter & SDF", "Unsharp Mask", "Crisp edge-contrast sharpening.", (PortSchema("src", "ANY"),), "ANY"),
    "sobel_edges": NodeMeta("sobel_edges", "Filter & SDF", "Sobel Edge Detect", "Gradient edge magnitude detector.", (PortSchema("src", "ANY"),), "SCALAR"),
    "sdf_from_mask": NodeMeta("sdf_from_mask", "Filter & SDF", "Distance Field (SDF)", "2D signed distance field from any mask.", (PortSchema("mask", "MASK"),), "SCALAR"),
    "morph_op": NodeMeta("morph_op", "Filter & SDF", "Morphology (Erode/Dilate)", "Erode, dilate, open, or close mask/image.", (PortSchema("src", "ANY"),), "ANY"),
    "bloom_glow": NodeMeta("bloom_glow", "Filter & SDF", "Studio Bloom Glow", "Multi-radius soft emissive highlight bloom.", (PortSchema("src", "RGBA"),), "RGBA"),
    # --- Color, Grading & Compositing ---
    "levels": NodeMeta("levels", "Color & Grading", "Levels / Gamma", "Input/output black-white points, gamma & invert.", (PortSchema("src", "SCALAR"),), "SCALAR"),
    "curves": NodeMeta("curves", "Color & Grading", "Tone Curve", "Piecewise smooth tone curve remap.", (PortSchema("src", "ANY"),), "ANY"),
    "color_balance": NodeMeta("color_balance", "Color & Grading", "Color Balance (S/M/H)", "Shadows, Midtones, Highlights atmospheric grading.", (PortSchema("src", "RGBA"),), "RGBA"),
    "oklch_adjust": NodeMeta("oklch_adjust", "Color & Grading", "OKLCH Hue/Chroma/Light", "Perceptual hue, chroma & lightness shift.", (PortSchema("src", "RGBA"),), "RGBA"),
    "posterize_soft": NodeMeta("posterize_soft", "Color & Grading", "Soft Posterize", "Hybrid slider between silky smooth and stepped pixel bands.", (PortSchema("src", "ANY"),), "ANY"),
    "hue_ramp": NodeMeta("hue_ramp", "Color & Grading", "Hue-Shifted OKLCH Ramp", "Automatic cool-shadow to warm-highlight pixel-art ramp.", (PortSchema("src", "SCALAR"), PortSchema("alpha_mask", "MASK", True)), "RGBA"),
    "palette_quantize": NodeMeta("palette_quantize", "Color & Grading", "OKLab Palette Quantize", "Quantize RGBA to exact artist hex palette with Bayer/Floyd dithering.", (PortSchema("src", "RGBA"),), "RGBA"),
    "vibrance_grade": NodeMeta("vibrance_grade", "Color & Grading", "Vibrance & Clean Contrast", "Chroma-preserving perceptual vibrance boost.", (PortSchema("src", "RGBA"),), "RGBA"),
    "selout": NodeMeta("selout", "Color & Grading", "Selective Outlining (Selout)", "Tinted silhouette & internal pixel-art outlines.", (PortSchema("src", "RGBA"),), "RGBA"),
    "blend": NodeMeta("blend", "Compositing", "Blend Layers", "Normal, multiply, screen, overlay, soft_light, hard_light, add, sub, max, min.", (PortSchema("base", "ANY"), PortSchema("top", "ANY"), PortSchema("mask", "MASK", True)), "ANY"),
    "math": NodeMeta("math", "Compositing", "Math Combine", "Mul, add, sub, max, min, lerp scalar fields.", (PortSchema("a", "SCALAR"), PortSchema("b", "SCALAR")), "SCALAR"),
    "set_alpha": NodeMeta("set_alpha", "Compositing", "Set Alpha Cutout", "Apply hard or soft alpha mask globally or per region.", (PortSchema("src", "RGBA"), PortSchema("alpha", "MASK"), PortSchema("mask", "MASK", True)), "RGBA"),
    # --- 3D Mesh Baker, PBR & Animation ---
    "mesh_channel": NodeMeta("mesh_channel", "3D Mesh Baker", "3D Mesh Attribute", "Reads baked 3D AO, bevel, cavity, normal, zone, element, outward_eye, face, local UV.", (), "SCALAR"),
    "directional_shade": NodeMeta("directional_shade", "PBR & Normal", "Directional 3D/2D Shade", "Combines height relief, 3D normals, baked AO, edge bevel & rim light.", (PortSchema("height", "HEIGHT"),), "SCALAR"),
    "height_to_normal": NodeMeta("height_to_normal", "PBR & Normal", "Height -> Tangent Normal", "Isotropic Scharr 3x3 normal map synthesis (OpenGL/LabPBR).", (PortSchema("height", "HEIGHT"),), "NORMAL"),
    "normal_blend": NodeMeta("normal_blend", "PBR & Normal", "Normal Blend (RNM)", "Reoriented Normal Mapping combining macro & micro detail normals.", (PortSchema("base", "NORMAL"), PortSchema("detail", "NORMAL")), "NORMAL"),
    "ao_from_height": NodeMeta("ao_from_height", "PBR & Normal", "Heightfield HBAO", "8-direction horizon-based AO from height.", (PortSchema("height", "HEIGHT"),), "SCALAR"),
    "emissive_albedo": NodeMeta("emissive_albedo", "PBR & Normal", "Emissive Composite", "Composites emissive glow & bleed onto albedo.", (PortSchema("albedo", "RGBA"), PortSchema("mask", "MASK"), PortSchema("color", "RGBA", True)), "RGBA"),
    "pad_islands": NodeMeta("pad_islands", "3D Mesh Baker", "UV Island Bleed Pad", "Clamp-bleeds island edges into UV padding.", (PortSchema("src", "ANY"),), "ANY"),
    "scroll": NodeMeta("scroll", "Animation", "Seamless Scroll", "Looping UV translation across flipbook phase.", (PortSchema("src", "ANY"),), "ANY", (), time_aware=True),
    "pulse": NodeMeta("pulse", "Animation", "Sine Pulse", "Looping sinusoidal intensity oscillation.", (PortSchema("src", "ANY"),), "ANY", (), time_aware=True),
    "anim_driver": NodeMeta(
        "anim_driver",
        "Animation",
        "Animation Driver (Wavetable/Wiggler/Curve)",
        "Pixel-Composer-style timeline driver emitting scalar wave, seamless loop wiggler, or eased keyframe curve.",
        (),
        "SCALAR",
        params=(
            ParamSchema("kind", "select", "sine", choices=("sine", "triangle", "sawtooth", "square", "wiggler", "keyframes")),
            ParamSchema("min_val", "slider", 0.0, -2.0, 2.0, 0.05),
            ParamSchema("max_val", "slider", 1.0, -2.0, 2.0, 0.05),
            ParamSchema("freq", "slider", 1.0, 0.25, 8.0, 0.25),
        ),
        time_aware=True,
    ),
    "iterate_inline": NodeMeta(
        "iterate_inline",
        "Simulation",
        "Iterate Inline Loop",
        "Multi-step recursive operator accumulation (warp_cascade, scale_echo, trail_stack, kuwahara_stack, erode_weather, moss_spread).",
        (PortSchema("src", "ANY"),),
        "ANY",
        params=(
            ParamSchema("mode", "select", "warp_cascade", choices=("warp_cascade", "scale_echo", "trail_stack", "kuwahara_stack", "erode_weather", "ridge_sharpen", "moss_spread")),
            ParamSchema("steps", "int", 5, 1, 16, 1),
            ParamSchema("strength", "slider", 0.35, 0.05, 1.0, 0.05),
            ParamSchema("decay", "slider", 0.82, 0.2, 1.0, 0.02),
        ),
        time_aware=True,
    ),
    "feedback_loop": NodeMeta(
        "feedback_loop",
        "Simulation",
        "Feedback Inline Buffer",
        "Stateful frame-to-frame temporal feedback loop with warm-up pre-roll for seamless motion trails & glowing wakes.",
        (PortSchema("src", "ANY"),),
        "ANY",
        params=(
            ParamSchema("decay", "slider", 0.80, 0.1, 0.98, 0.02),
            ParamSchema("drift_x_px", "slider", 0.0, -8.0, 8.0, 1.0),
            ParamSchema("drift_y_px", "slider", -1.0, -8.0, 8.0, 1.0),
            ParamSchema("blur_radius", "int", 1, 0, 4, 1),
            ParamSchema("pre_roll", "int", 6, 0, 16, 1),
        ),
        time_aware=True,
    ),
    "particle_sim": NodeMeta(
        "particle_sim",
        "Simulation",
        "Pixel Particle System",
        "Deterministic looping pixel-art particle emitter with gravity, curl turbulence, vortex attractor & trails.",
        (PortSchema("emitter_mask", "MASK", True),),
        "RGBA",
        outputs=(PortSchema("rgba", "RGBA"), PortSchema("density", "MASK"), PortSchema("heat", "SCALAR")),
        params=(
            ParamSchema("count", "int", 48, 4, 256, 4),
            ParamSchema("emitter", "select", "bottom", choices=("bottom", "center", "ring", "full")),
            ParamSchema("vel_y", "slider", -0.55, -2.0, 2.0, 0.05),
            ParamSchema("turbulence", "slider", 0.22, 0.0, 1.0, 0.02),
            ParamSchema("vortex", "slider", 0.0, -3.0, 3.0, 0.1),
        ),
        time_aware=True,
    ),
    "fluid_smoke_sim": NodeMeta(
        "fluid_smoke_sim",
        "Simulation",
        "2D Smoke / Flame / Plasma Fluid Sim",
        "Semi-Lagrangian buoyancy + curl-vorticity 2D pixel fluid & flame simulation (tileable & seamlessly loopable).",
        (PortSchema("source_mask", "MASK", True),),
        "RGBA",
        outputs=(PortSchema("rgba", "RGBA"), PortSchema("density", "MASK"), PortSchema("temperature", "SCALAR")),
        params=(
            ParamSchema("steps", "int", 16, 4, 48, 2),
            ParamSchema("buoyancy", "slider", 1.65, -3.0, 4.0, 0.05),
            ParamSchema("curl_strength", "slider", 1.85, 0.0, 4.0, 0.05),
            ParamSchema("dissipation", "slider", 0.93, 0.60, 0.99, 0.01),
        ),
        time_aware=True,
    ),
    "organic_domain_warp_3d": NodeMeta(
        "organic_domain_warp_3d",
        "Painterly & Bio",
        "3D Organic Domain Warp",
        "Continuous cross-seam 3D vector domain warping & pigment mottle flow field for natural organic texturing.",
        (),
        "SCALAR",
        outputs=(PortSchema("mottle", "SCALAR"), PortSchema("warp_x", "SCALAR"), PortSchema("warp_y", "SCALAR"), PortSchema("flow", "SCALAR")),
        params=(
            ParamSchema("scale", "slider", 3.5, 0.5, 16.0, 0.25),
            ParamSchema("warp_strength", "slider", 0.42, 0.0, 1.0, 0.02),
            ParamSchema("octaves", "int", 3, 1, 5, 1),
        ),
    ),
    "painterly_brush_field": NodeMeta(
        "painterly_brush_field",
        "Painterly & Bio",
        "Painterly Brush & Mottle Field",
        "Anisotropic domain-warped hand-painted Blockbench brushstrokes + organic pigment mottling & hue-temperature drift.",
        (),
        "SCALAR",
        outputs=(PortSchema("painterly", "SCALAR"), PortSchema("brush", "SCALAR"), PortSchema("mottle", "SCALAR"), PortSchema("hue_drift", "SCALAR")),
        params=(
            ParamSchema("angle_deg", "slider", -16.0, -180.0, 180.0, 1.0),
            ParamSchema("stroke_scale", "slider", 12.0, 2.0, 48.0, 1.0),
            ParamSchema("organic_distortion", "slider", 0.58, 0.0, 1.0, 0.02),
            ParamSchema("mottle_weight", "slider", 0.48, 0.0, 1.0, 0.02),
        ),
    ),
    "anatomical_muscle_scales": NodeMeta(
        "anatomical_muscle_scales",
        "Painterly & Bio",
        "Organic Painterly Dragon Hide & Scales (MATERIAL)",
        "3D domain-warped organic dragon scales, painterly brushwork, hue-temperature drift, muscle cavities & ventral belly scutes.",
        (),
        "MATERIAL",
        outputs=(
            PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"),
            PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"),
            PortSchema("muscle_crevice", "MASK"), PortSchema("crown_highlight", "MASK"),
        ),
        params=(
            ParamSchema("organic_distortion", "slider", 0.64, 0.0, 1.0, 0.02),
            ParamSchema("softness", "slider", 0.58, 0.0, 1.0, 0.02),
            ParamSchema("hue_variation", "slider", 0.32, 0.0, 1.0, 0.02),
            ParamSchema("scale_relief", "slider", 0.44, 0.0, 1.0, 0.02),
        ),
    ),
    "colossal_dragon_membrane": NodeMeta(
        "colossal_dragon_membrane",
        "Painterly & Bio",
        "Dual-Layer Dragon Wing Membrane (MATERIAL)",
        "Zero-Blink 0.5u dual-layer dorsal/ventral wing membrane with organic domain-warped tendon ribs, painterly mottling & backlit SSS.",
        (PortSchema("span_u", "SCALAR"), PortSchema("trail_v", "SCALAR"), PortSchema("face_is_down", "SCALAR", True)),
        "MATERIAL",
        outputs=(
            PortSchema("albedo", "RGBA"), PortSchema("alpha", "MASK"), PortSchema("height", "HEIGHT"),
            PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("ao", "SCALAR"),
            PortSchema("sss", "SCALAR"), PortSchema("tendon_mask", "MASK"), PortSchema("tatter_rim", "MASK"),
        ),
    ),
    "sprite_mask": NodeMeta("sprite_mask", "Sprites & Masks", "Item Sprite Mask", "Procedural pixel-art item silhouette generator.", (), "MASK"),
    # --- TexGraph v7 Industry Operators (Substance Designer + Pixel Composer Parity) ---
    "sdf_shape": NodeMeta(
        "sdf_shape", "Sprites & Masks", "2D SDF Shape Primitive",
        "Analytical 2D SDF shapes & Minecraft item silhouettes (circle, rounded_box, star, leaf, feather, scale, sword, axe, pickaxe, potion, shield, gem, book, apple).",
        (), "SCALAR",
        outputs=(PortSchema("sdf", "SCALAR"), PortSchema("mask", "MASK"), PortSchema("bevel", "HEIGHT"), PortSchema("outline", "MASK")),
        params=(ParamSchema("shape", "select", "rounded_box", choices=("circle", "box", "rounded_box", "polygon", "star", "capsule", "crescent", "teardrop", "gear", "leaf", "feather", "scale", "sword", "axe", "pickaxe", "potion", "shield", "gem", "book", "apple")),),
    ),
    "sdf_boolean": NodeMeta(
        "sdf_boolean", "Sprites & Masks", "2D SDF Boolean",
        "Smooth or sharp union, subtract, intersect & xor on 2D SDF fields.",
        (PortSchema("a", "SCALAR"), PortSchema("b", "SCALAR")), "SCALAR",
        outputs=(PortSchema("sdf", "SCALAR"), PortSchema("mask", "MASK"), PortSchema("bevel", "HEIGHT")),
    ),
    "tile_sampler": NodeMeta(
        "tile_sampler", "Scatter & Layout", "Tile Sampler / Scatter",
        "Substance Tile Sampler / Pixel Composer Scatter: grid or jittered stamp scattering with scale/rotation/luminance jitter, driver maps & seamless wrap.",
        (PortSchema("stamp", "ANY", True), PortSchema("density_map", "MASK", True), PortSchema("scale_map", "SCALAR", True), PortSchema("rot_map", "SCALAR", True)),
        "HEIGHT",
        outputs=(PortSchema("height", "HEIGHT"), PortSchema("mask", "MASK"), PortSchema("id_map", "SCALAR"), PortSchema("rgba", "RGBA")),
    ),
    "flood_fill": NodeMeta(
        "flood_fill", "Scatter & Layout", "Flood Fill (Per-Island)",
        "Labels isolated mask islands (seamless wrap aware) and outputs per-island random grayscale, UV gradients, random tilt slopes & border distance.",
        (PortSchema("mask", "MASK"),), "SCALAR",
        outputs=(PortSchema("island_id", "SCALAR"), PortSchema("random_gray", "SCALAR"), PortSchema("gradient_u", "SCALAR"), PortSchema("gradient_v", "SCALAR"), PortSchema("random_slope", "SCALAR"), PortSchema("edge_dist", "HEIGHT")),
    ),
    "slope_blur": NodeMeta(
        "slope_blur", "Filter & SDF", "Slope Blur",
        "Advects height/mask uphill, downhill, or smoothly along a secondary slope map for chipped rock, impasto & organic wear.",
        (PortSchema("src", "ANY"), PortSchema("slope_map", "SCALAR")), "ANY",
        params=(ParamSchema("intensity", "slider", 1.6, 0.0, 8.0, 0.1), ParamSchema("mode", "select", "min", choices=("min", "max", "blur"))),
    ),
    "histogram_scan": NodeMeta("histogram_scan", "Color & Grading", "Histogram Scan", "Precision position + contrast tonal mask slicing.", (PortSchema("src", "SCALAR"),), "MASK"),
    "histogram_select": NodeMeta("histogram_select", "Color & Grading", "Histogram Select", "Isolates a centered grayscale value band.", (PortSchema("src", "SCALAR"),), "MASK"),
    "auto_levels": NodeMeta("auto_levels", "Color & Grading", "Auto Levels", "Robust percentile dynamic range normalization.", (PortSchema("src", "SCALAR"),), "SCALAR"),
    "curvature": NodeMeta(
        "curvature", "PBR & Normal", "Curvature from Height",
        "Multi-scale convex ridge & concave crevice detector.",
        (PortSchema("height", "HEIGHT"),), "SCALAR",
        outputs=(PortSchema("curvature", "SCALAR"), PortSchema("convex", "MASK"), PortSchema("concave", "MASK")),
    ),
    "edge_wear": NodeMeta(
        "edge_wear", "PBR & Normal", "Edge Wear & Crevice Grime",
        "Procedural chipped edge wear and crevice dirt/grime masks from surface height.",
        (PortSchema("height", "HEIGHT"),), "MASK",
        outputs=(PortSchema("wear", "MASK"), PortSchema("grime", "MASK"), PortSchema("curvature", "SCALAR")),
    ),
    "displace_uv": NodeMeta(
        "displace_uv", "Transform & Distort", "Displace UV",
        "Pixel Composer Displace node: bilinear or crisp pixel-snapped UV displacement.",
        (PortSchema("src", "ANY"), PortSchema("disp_map", "SCALAR"), PortSchema("disp_y_map", "SCALAR", True)), "ANY",
    ),
    "normal_light_2d": NodeMeta(
        "normal_light_2d", "PBR & Normal", "2D Normal Light",
        "Pixel Composer Normal Light: real-time directional 2D relighting of pixel art albedo + normal map.",
        (PortSchema("albedo", "RGBA"), PortSchema("normal", "NORMAL")), "RGBA",
    ),
    "directional_selout": NodeMeta(
        "directional_selout", "Color & Grading", "Directional Selout",
        "Brandon James Greer directional selective outline: warm-lit top-left edges shifting to cool-deep shadow undersides.",
        (PortSchema("src", "RGBA"),), "RGBA",
    ),
    "cobblestone_masonry": NodeMeta(
        "cobblestone_masonry", "World Blocks (PBR)", "Cobblestone Masonry (MATERIAL)",
        "Seamless Flood-Fill + Slope-Blur + Edge-Wear cobblestone & mossy stone masonry MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("mortar_mask", "MASK")),
    ),
    "bark_and_timber": NodeMeta(
        "bark_and_timber", "World Blocks (PBR)", "Furrowed Bark & Timber (MATERIAL)",
        "Seamless Slope-Blur organic tree bark with crevice moss/lichen MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("ao", "SCALAR")),
    ),
    "crystalline_ore_vein": NodeMeta(
        "crystalline_ore_vein", "World Blocks (PBR)", "Crystalline Ore Vein (MATERIAL)",
        "Seamless chipped host basalt/deepslate embedded with faceted mineral ore crystals MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("emissive", "RGBA"), PortSchema("ore_mask", "MASK")),
    ),
    "caustic_water": NodeMeta(
        "caustic_water", "World Blocks (PBR)", "Animated Caustic Water (MATERIAL)",
        "Seamless looping dual-layer Voronoi caustic water surface MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("sss", "SCALAR")),
        time_aware=True,
    ),
    "organic_fur_pelt": NodeMeta(
        "organic_fur_pelt", "Painterly & Bio", "Organic Mammal Fur Pelt (MATERIAL)",
        "Directional 2-4px tufted mammal fur with jagged countershading & saddle/tabby/spotted markings MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("belly_mask", "MASK")),
    ),
    "layered_plumage": NodeMeta(
        "layered_plumage", "Painterly & Bio", "Layered Avian Plumage (MATERIAL)",
        "Overlapping scalloped bird/griffin/phoenix feathers with rachis quill & barred/speckled/iridescent markings MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR")),
    ),
    "chitin_carapace": NodeMeta(
        "chitin_carapace", "Painterly & Bio", "Segmented Chitin Carapace (MATERIAL)",
        "Segmented scorpion/beetle/arthropod carapace with cool plum shadows & Fresnel iridescence MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR")),
    ),
    "aquatic_scales_skin": NodeMeta(
        "aquatic_scales_skin", "Painterly & Bio", "Aquatic Scales & Moist Skin (MATERIAL)",
        "Wet glossy fish cycloid scales or moist aposematic dart-frog/koi amphibian skin MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("sss", "SCALAR")),
    ),
    "elemental_golem_surface": NodeMeta(
        "elemental_golem_surface", "Painterly & Bio", "Elemental Golem Construct (MATERIAL)",
        "Flood-Fill + Slope-Blur fractured stone/obsidian slabs with chipped wear & glowing crevice fissures MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("emissive", "RGBA"), PortSchema("fissure_mask", "MASK")),
    ),
    # --- TexGraph v8 Character/Orc Materials & Part-Aware Entity Landmark Compositor ---
    "orc_flesh_skin": NodeMeta(
        "orc_flesh_skin", "Character & Gear (PBR)", "Orc & Monster Skin (MATERIAL)",
        "Hue-shifted olive/emerald orc/goblin skin with warm subsurface blush, battle scars & optional tribal war paint MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("sss", "SCALAR"), PortSchema("scar_mask", "MASK"), PortSchema("subsurface_mask", "MASK")),
    ),
    "stitched_leather": NodeMeta(
        "stitched_leather", "Character & Gear (PBR)", "Stitched Leather & Straps (MATERIAL)",
        "Warm desaturated brown leather with concentric stepped rim shading, micro-stitch dots & worn fold creases MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("stitch_mask", "MASK")),
    ),
    "forged_battle_metal": NodeMeta(
        "forged_battle_metal", "Character & Gear (PBR)", "Forged Battle Metal & Armor (MATERIAL)",
        "Cool charcoal-indigo shadowed forged steel with 1 px bevel catchlights, battle scratches, rivets & cavity rust MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("rust_mask", "MASK"), PortSchema("scratch_mask", "MASK")),
    ),
    "woven_battle_cloth": NodeMeta(
        "woven_battle_cloth", "Character & Gear (PBR)", "Woven Battle Cloth & Rope (MATERIAL)",
        "Muted battle cloth with soft 2-3 px vertical drape folds, embroidered hem trim, or twisted diagonal rope bindings MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("ao", "SCALAR")),
    ),
    "keratin_horn_bone": NodeMeta(
        "keratin_horn_bone", "Character & Gear (PBR)", "Keratin, Horn, Ivory & Claws (MATERIAL)",
        "Dark-rooted keratin stepping smoothly into striated warm ivory distal tips for tusks, horns, teeth, beaks, claws & hooves MaterialBus.",
        (), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("sss", "SCALAR")),
    ),
    "paint_entity_eyes_and_parts": NodeMeta(
        "paint_entity_eyes_and_parts", "Character & Gear (PBR)", "Part-Aware Eyes & Anatomical Landmarks (MATERIAL)",
        "Protects and hand-shades 3D entity eyes (with animated blink & glow), orbital sockets, wet noses, beaks, dark plum mouths, tongues, ivory teeth/tusks, keratin claws/hooves/horns, inner ears, fins & glowing runes!",
        (PortSchema("base", "MATERIAL"),), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("emissive", "RGBA"), PortSchema("sss", "SCALAR"), PortSchema("eye_mask", "MASK")),
    ),
    # --- TexGraph v9 Pixel Composer Atomic Operators & 3D Baked Emissive Bounce ---
    "pixel_dither": NodeMeta(
        "pixel_dither", "Pixel Composer FX", "Pixel Dither (Bayer / Error-Diffuse)",
        "Pixel Composer Dither: ordered Bayer (2x2/4x4/8x8), diamond, blue-noise, or Floyd-Steinberg/Atkinson error-diffusion locked strictly to a discrete OKLab palette.",
        (PortSchema("src", "ANY"),), "RGBA",
    ),
    "palette_posterize": NodeMeta(
        "palette_posterize", "Pixel Composer FX", "Palette Posterize",
        "Pixel Composer Posterize: snap every pixel crisply onto a discrete curated palette in OKLab/RGB/luma space with optional hard alpha cutoff.",
        (PortSchema("src", "ANY"),), "RGBA",
    ),
    "color_replace": NodeMeta(
        "color_replace", "Pixel Composer FX", "Replace Palette / Colors",
        "Pixel Composer Replace Palette: cleanly remap source palette colors to destination colorway palette.",
        (PortSchema("src", "RGBA"),), "RGBA",
    ),
    "level_selector": NodeMeta(
        "level_selector", "Pixel Composer FX", "Level Selector",
        "Pixel Composer Level Selector: isolate a luminance slice [center ± width/2] for gem glints, specular rims & mask extraction.",
        (PortSchema("src", "ANY"),), "MASK",
    ),
    "de_stray": NodeMeta(
        "de_stray", "Pixel Composer FX", "De-Stray (Orphan Cleaner)",
        "Pixel Composer De-Stray: eliminate isolated 1x1 noise specks while preserving coherent artistic clusters & catchlights.",
        (PortSchema("src", "ANY"),), "ANY",
    ),
    "de_corner": NodeMeta(
        "de_corner", "Pixel Composer FX", "De-Corner (Jag Cleaner)",
        "Pixel Composer De-Corner: remove ugly L-shaped stair-step pixel jaggies and line doubles cleanly while staying palette-locked.",
        (PortSchema("src", "ANY"),), "ANY",
    ),
    "pixel_clusters": NodeMeta(
        "pixel_clusters", "Pixel Composer FX", "Directional Pixel Clusters",
        "Hytale/Everbloom/Mickey Joe 2-4 px directional pixel cluster sculptor with palette lock, De-Stray & De-Corner.",
        (PortSchema("src", "ANY"),), "RGBA",
    ),
    "pixel_bevel_directional": NodeMeta(
        "pixel_bevel_directional", "Pixel Composer FX", "Directional Chipped Bevel",
        "Anti-wireframe directional top-left lit bevel + bottom-right cast shadow with asymmetric battle-chip notch breakup (ports: highlight, shadow, chips, signed).",
        (), "MASK",
        outputs=(PortSchema("highlight", "MASK"), PortSchema("shadow", "MASK"), PortSchema("chips", "MASK"), PortSchema("signed", "SCALAR")),
    ),
    "pixel_outline": NodeMeta(
        "pixel_outline", "Pixel Composer FX", "Pixel Outline / Inline",
        "Pixel Composer Outline: crisp 4-way or 8-way outside/inside pixel-art outline.",
        (PortSchema("src", "RGBA"),), "RGBA",
    ),
    "heightmap_shadow": NodeMeta(
        "heightmap_shadow", "Pixel Composer FX", "Heightmap Cast Shadow",
        "Pixel Composer 1.22.4 Heightmap Shadow: directional ray-marched cast shadows across 2D relief.",
        (PortSchema("height", "HEIGHT"),), "SCALAR",
    ),
    "pixel_mirror": NodeMeta(
        "pixel_mirror", "Transform & Distort", "Pixel Mirror / Symmetry",
        "Pixel Composer Mirror: exact pixel-grid X, Y, or 4-way XY symmetry.",
        (PortSchema("src", "ANY"),), "ANY",
    ),
    "pixel_sort": NodeMeta(
        "pixel_sort", "Pixel Composer FX", "Pixel Sort",
        "Pixel Composer Pixel Sort: directional luminance-masked pixel span sorting.",
        (PortSchema("src", "ANY"),), "ANY",
    ),
    "chromatic_aberration": NodeMeta(
        "chromatic_aberration", "Pixel Composer FX", "Chromatic Aberration",
        "Pixel Composer Chromatic Aberration: crisp RGB channel edge shift.",
        (PortSchema("src", "RGBA"),), "RGBA",
    ),
    "vignette": NodeMeta(
        "vignette", "Pixel Composer FX", "Vignette",
        "Pixel Composer Vignette: hue-shifted dark focal frame.",
        (PortSchema("src", "RGBA"),), "RGBA",
    ),
    "bake_emissive_bounce": NodeMeta(
        "bake_emissive_bounce", "PBR & Normal", "3D Baked Emissive Bounce Light (MATERIAL)",
        "Hytale Golem_Firesteel 3D world-space GI bounce: projects warm/colored glow from eyes, runes & magma fissures onto adjacent dark armor, obsidian & skin.",
        (PortSchema("mat", "MATERIAL"),), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("shadow", "SCALAR"), PortSchema("bent_normal", "NORMAL"), PortSchema("thickness", "SCALAR"), PortSchema("emissive", "RGBA"), PortSchema("sss", "SCALAR")),
        params=(
            ParamSchema("radius_units", "slider", 7.5, 1.0, 24.0, 0.5, group="Bounce Light"),
            ParamSchema("intensity", "slider", 0.52, 0.0, 1.5, 0.02, group="Bounce Light"),
            ParamSchema("posterize_steps", "int", 0, 0, 12, 1, group="Stylization"),
        ),
    ),
    # --- TexGraph v10 Seamless UV-Geometric Edge Solving & Precomputed Shadow Special PBR ---
    "triplanar_project": NodeMeta(
        "triplanar_project", "3D Mesh Baker", "Triplanar 3D Projection (UE5/Substance)",
        "Seamless 3D World/Local normal-blended (or UE 5.7 Bayer-dithered) triplanar projection across scalar, RGBA, or full 12-channel MATERIAL bus — eliminates 90-degree UV seams & re-projects tangent normals!",
        (PortSchema("src", "ANY"),), "ANY",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR")),
        params=(
            ParamSchema("scale", "slider", 0.08, 0.005, 2.0, 0.005, group="Projection"),
            ParamSchema("sharpness", "slider", 6.0, 0.5, 24.0, 0.5, group="Projection"),
            ParamSchema("space", "select", "world", choices=("world", "local_norm"), group="Projection"),
            ParamSchema("dither_blend", "checkbox", False, group="Pixel Craft", tooltip="UE 5.7 single-axis dithered triplanar selection"),
            ParamSchema("pixel_snap", "checkbox", True, group="Pixel Craft"),
        ),
    ),
    "uv_seam_heal": NodeMeta(
        "uv_seam_heal", "3D Mesh Baker", "3D UV-Edge Seam Healer",
        "Uses the solved 3D UV-Edge Adjacency graph to seamlessly stitch albedo, height, roughness & world-reoriented normals across shared 3D edges between UV islands!",
        (PortSchema("src", "ANY"),), "ANY",
        params=(
            ParamSchema("strength", "slider", 0.72, 0.0, 1.0, 0.02, group="Seam Healing"),
            ParamSchema("radius_px", "int", 2, 0, 4, 1, group="Seam Healing"),
            ParamSchema("heal_normals", "checkbox", True, group="Seam Healing"),
        ),
    ),
    "block_edge_transition": NodeMeta(
        "block_edge_transition", "World Blocks (PBR)", "Connected Block-Edge Transition",
        "Connected multi-block edge & corner transition solver (bevel_trim, moss_overhang, worn_stone, metal_frame, glow_circuit) driven by 4-bit NESW neighbor masks!",
        (PortSchema("base", "MATERIAL", True),), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("emissive", "RGBA"), PortSchema("edge_mask", "MASK"), PortSchema("corner_mask", "MASK"), PortSchema("transition_h", "HEIGHT")),
        params=(
            ParamSchema("neighbors", "select", "exposed_all", choices=("exposed_all", "connected_all", "top_exposed", "bottom_exposed", "pillar_mid", "beam_mid", "corner_tl", "corner_tr", "corner_bl", "corner_br"), group="Connectivity"),
            ParamSchema("style", "select", "bevel_trim", choices=("bevel_trim", "moss_overhang", "worn_stone", "metal_frame", "glow_circuit"), group="Style"),
            ParamSchema("border_width", "slider", 0.14, 0.03, 0.40, 0.01, group="Geometry"),
            ParamSchema("corner_strength", "slider", 0.65, 0.0, 1.0, 0.05, group="Geometry"),
            ParamSchema("transition_color", "color", "#3E4B5E", group="Colors"),
            ParamSchema("highlight_color", "color", "#B6C8D6", group="Colors"),
            ParamSchema("moss_color", "color", "#4E823E", group="Colors", visible_if={"style": "moss_overhang"}),
            ParamSchema("glow_color", "color", "#38D9F5", group="Colors", visible_if={"style": "glow_circuit"}),
        ),
    ),
    "precompute_shadow_pbr": NodeMeta(
        "precompute_shadow_pbr", "PBR & Normal", "Precomputed Shadow & Special PBR (12-Ch)",
        "SyncCraft Special PBR Pipeline: integrates true 3D ray-traced directional shadows, hemisphere bent-normal IBL & ray-traced thickness SSS into both separated 12-channel PBR maps and optional pre-baked stylized albedo!",
        (PortSchema("mat", "MATERIAL"),), "MATERIAL",
        outputs=(PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"), PortSchema("shadow", "SCALAR"), PortSchema("bent_normal", "NORMAL"), PortSchema("thickness", "SCALAR"), PortSchema("emissive", "RGBA"), PortSchema("sss", "SCALAR"), PortSchema("porosity", "SCALAR")),
        params=(
            ParamSchema("preset", "select", "studio_hero", choices=("studio_hero", "dramatic_boss", "golden_rim", "overcast_soft"), group="Lighting"),
            ParamSchema("shadow_strength", "slider", 0.62, 0.0, 1.2, 0.02, group="Shadows"),
            ParamSchema("bent_normal_strength", "slider", 0.48, 0.0, 1.2, 0.02, group="IBL"),
            ParamSchema("thickness_sss_strength", "slider", 0.42, 0.0, 1.2, 0.02, group="Subsurface"),
            ParamSchema("bake_into_albedo", "slider", 0.45, 0.0, 1.0, 0.02, group="Output", tooltip="0.0 = pure separated PBR channels; 0.45-0.85 = stylized pre-baked lighting"),
            ParamSchema("shadow_tint", "color", "#1F2442", group="Shadows"),
            ParamSchema("sky_tint", "color", "#B6D2F5", group="IBL"),
            ParamSchema("sss_color", "color", "#EA723D", group="Subsurface"),
            ParamSchema("shadow_posterize_steps", "int", 0, 0, 10, 1, group="Stylization"),
        ),
    ),
}


# Comprehensive typed ParamSchema defaults ensuring 100% of all 144 nodes expose GUI inspector widgets!
_PARAM_OVERRIDES: dict[str, tuple[ParamSchema, ...]] = {
    "grid": (ParamSchema("cols", "int", 4, 1, 32, 1), ParamSchema("rows", "int", 4, 1, 32, 1), ParamSchema("mortar", "slider", 0.08, 0.01, 0.4, 0.01)),
    "brick": (ParamSchema("cols", "int", 4, 1, 32, 1), ParamSchema("rows", "int", 8, 1, 32, 1), ParamSchema("style", "select", "running", choices=("running", "stacked", "herringbone", "basketweave"))),
    "hex_grid": (ParamSchema("cols", "int", 4, 1, 24, 1), ParamSchema("mortar", "slider", 0.08, 0.01, 0.35, 0.01)),
    "tri_grid": (ParamSchema("cols", "int", 6, 1, 32, 1),),
    "diamond": (ParamSchema("cols", "int", 6, 1, 32, 1), ParamSchema("rows", "int", 6, 1, 32, 1), ParamSchema("style", "select", "diamond", choices=("diamond", "chevron"))),
    "truchet": (ParamSchema("cols", "int", 6, 1, 24, 1), ParamSchema("style", "select", "arcs", choices=("arcs", "diagonals"))),
    "mosaic": (ParamSchema("scale", "slider", 7.0, 2.0, 32.0, 0.5),),
    "radial_rings": (ParamSchema("freq", "slider", 6.0, 1.0, 24.0, 0.5),),
    "energy_veins": (ParamSchema("scale", "slider", 5.0, 1.0, 24.0, 0.5), ParamSchema("style", "select", "lightning", choices=("lightning", "circuit", "nebula"))),
    "layered_plates": (ParamSchema("cols", "int", 8, 2, 32, 1), ParamSchema("rows", "int", 12, 2, 32, 1), ParamSchema("output", "select", "height", choices=("height", "rim", "seam"))),
    "imbricated_scales": (ParamSchema("cols", "int", 8, 2, 32, 1), ParamSchema("rows", "int", 10, 2, 32, 1), ParamSchema("shape", "select", "dragon", choices=("dragon", "fish", "reptile", "hex"))),
    "feather_barbs": (ParamSchema("cols", "int", 4, 1, 16, 1),),
    "fur_tufts": (ParamSchema("cols", "int", 10, 2, 32, 1), ParamSchema("rows", "int", 8, 2, 32, 1)),
    "temple_stone": (ParamSchema("cols", "int", 2, 1, 16, 1), ParamSchema("rows", "int", 4, 1, 16, 1)),
    "wood_planks": (ParamSchema("planks", "int", 4, 1, 16, 1), ParamSchema("output", "select", "height", choices=("height", "grain", "seam"))),
    "foliage_clusters": (ParamSchema("cols", "int", 5, 2, 16, 1), ParamSchema("rows", "int", 5, 2, 16, 1)),
    "gray_scott": (ParamSchema("preset", "select", "labyrinth", choices=("labyrinth", "coral", "spots", "stripes", "nebula")), ParamSchema("steps", "int", 320, 40, 800, 20)),
    "cellular_automata": (ParamSchema("preset", "select", "cave", choices=("cave", "coral", "crystal")), ParamSchema("steps", "int", 6, 1, 24, 1)),
    "iterate": (ParamSchema("operation", "select", "erode_weather", choices=("erode_weather", "sharpen", "ridge", "moss_spread")), ParamSchema("steps", "int", 6, 1, 24, 1)),
    "weathering_drip": (ParamSchema("strength", "slider", 0.55, 0.0, 1.0, 0.05), ParamSchema("length", "int", 8, 2, 32, 1)),
    "kuwahara": (ParamSchema("radius", "int", 2, 1, 6, 1), ParamSchema("sharpness", "slider", 8.0, 1.0, 16.0, 0.5)),
    "bilateral": (ParamSchema("radius", "int", 2, 1, 6, 1), ParamSchema("sigma_s", "slider", 1.5, 0.3, 6.0, 0.1), ParamSchema("sigma_r", "slider", 0.14, 0.02, 0.5, 0.02)),
    "tv_denoise": (ParamSchema("weight", "slider", 0.08, 0.01, 0.5, 0.01), ParamSchema("iterations", "int", 12, 2, 40, 2)),
    "nlm_denoise": (ParamSchema("patch_radius", "int", 1, 1, 3, 1), ParamSchema("search_radius", "int", 3, 1, 6, 1), ParamSchema("h_param", "slider", 0.08, 0.01, 0.4, 0.01)),
    "despeckle": (ParamSchema("threshold", "slider", 0.14, 0.01, 0.5, 0.01),),
    "cluster_mode": (ParamSchema("iterations", "int", 1, 1, 4, 1),),
    "tile_transform": (ParamSchema("scale_x", "slider", 1.0, 0.25, 8.0, 0.05), ParamSchema("scale_y", "slider", 1.0, 0.25, 8.0, 0.05), ParamSchema("rot_deg", "slider", 0.0, -180.0, 180.0, 1.0)),
    "polar_warp": (ParamSchema("to_polar", "checkbox", True),),
    "twirl_distort": (ParamSchema("strength_deg", "slider", 90.0, -360.0, 360.0, 5.0), ParamSchema("radius", "slider", 0.5, 0.05, 1.5, 0.05)),
    "wave_distort": (ParamSchema("amp_x", "slider", 0.04, 0.0, 0.3, 0.005), ParamSchema("freq_y", "slider", 4.0, 0.5, 24.0, 0.5)),
    "gaussian_blur": (ParamSchema("radius", "int", 2, 1, 12, 1), ParamSchema("sigma", "slider", 0.0, 0.0, 8.0, 0.2)),
    "directional_blur": (ParamSchema("angle_deg", "slider", 0.0, -180.0, 180.0, 5.0), ParamSchema("length", "int", 4, 1, 24, 1)),
    "unsharp_mask": (ParamSchema("radius", "int", 1, 1, 6, 1), ParamSchema("amount", "slider", 0.65, 0.0, 3.0, 0.05)),
    "sobel_edges": (ParamSchema("strength", "slider", 1.5, 0.1, 6.0, 0.1),),
    "sdf_from_mask": (ParamSchema("max_dist_px", "slider", 12.0, 1.0, 64.0, 1.0), ParamSchema("invert", "checkbox", False)),
    "morph_op": (ParamSchema("morph_op", "select", "dilate", choices=("dilate", "erode", "open", "close")), ParamSchema("radius", "int", 1, 1, 6, 1)),
    "bloom_glow": (ParamSchema("threshold", "slider", 0.72, 0.0, 1.0, 0.02), ParamSchema("radius", "int", 3, 1, 12, 1), ParamSchema("intensity", "slider", 0.55, 0.0, 2.0, 0.05)),
    "levels": (ParamSchema("in_min", "slider", 0.0, 0.0, 1.0, 0.01), ParamSchema("in_max", "slider", 1.0, 0.0, 1.0, 0.01), ParamSchema("gamma", "slider", 1.0, 0.1, 4.0, 0.05), ParamSchema("invert", "checkbox", False)),
    "curves": (ParamSchema("points", "curve", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]),),
    "color_balance": (ParamSchema("shadows", "vec3", [0.0, 0.0, 0.0]), ParamSchema("midtones", "vec3", [0.0, 0.0, 0.0]), ParamSchema("highlights", "vec3", [0.0, 0.0, 0.0])),
    "oklch_adjust": (ParamSchema("hue_shift_deg", "slider", 0.0, -180.0, 180.0, 1.0), ParamSchema("chroma_scale", "slider", 1.0, 0.0, 2.5, 0.05), ParamSchema("lightness_offset", "slider", 0.0, -0.5, 0.5, 0.01)),
    "posterize_soft": (ParamSchema("steps", "int", 6, 2, 24, 1), ParamSchema("blend", "slider", 0.5, 0.0, 1.0, 0.05)),
    "hue_ramp": (ParamSchema("base_hex", "color", "#567B48"), ParamSchema("steps", "int", 6, 3, 12, 1), ParamSchema("hue_swing_deg", "slider", 42.0, 0.0, 90.0, 2.0)),
    "palette_quantize": (ParamSchema("palette_hex", "palette", ["#1B1726", "#4D5368", "#949FB5", "#E5EEF7"]), ParamSchema("dither", "select", "none", choices=("none", "bayer2", "bayer4", "floyd"))),
    "vibrance_grade": (ParamSchema("vibrance", "slider", 0.22, -0.5, 1.0, 0.02), ParamSchema("contrast", "slider", 1.06, 0.5, 2.0, 0.02), ParamSchema("shadow_cool", "slider", 0.05, 0.0, 0.3, 0.01)),
    "selout": (ParamSchema("strength", "slider", 0.65, 0.0, 1.0, 0.05),),
    "blend": (ParamSchema("mode", "select", "normal", choices=("normal", "multiply", "screen", "overlay", "soft_light", "hard_light", "add", "sub", "max", "min")), ParamSchema("opacity", "slider", 1.0, 0.0, 1.0, 0.02)),
    "math": (ParamSchema("math_op", "select", "mul", choices=("mul", "add", "sub", "div", "max", "min", "lerp")),),
    "set_alpha": (ParamSchema("threshold", "slider", 0.5, 0.0, 1.0, 0.05),),
    "mesh_channel": (ParamSchema("channel", "select", "baked_ao", choices=("baked_ao", "baked_shadow", "bent_normal", "thickness", "curvature_3d", "uv_seam", "edge_bevel", "joint_cavity", "top_grad", "front_grad", "normal_y", "zone", "element", "face", "outward_eye", "island_mask", "local_u", "local_v")), ParamSchema("pattern", "text", "", visible_if={"channel": "element"}), ParamSchema("zone", "text", "torso", visible_if={"channel": "zone"})),
    "directional_shade": (ParamSchema("light_az_deg", "slider", -38.0, -180.0, 180.0, 2.0), ParamSchema("light_el_deg", "slider", 52.0, 5.0, 90.0, 2.0)),
    "height_to_normal": (ParamSchema("strength", "slider", 2.4, 0.2, 8.0, 0.1), ParamSchema("convention", "select", "opengl", choices=("opengl", "directx"))),
    "normal_blend": (ParamSchema("strength", "slider", 1.0, 0.0, 2.0, 0.05),),
    "ao_from_height": (ParamSchema("radius", "int", 3, 1, 12, 1), ParamSchema("strength", "slider", 1.0, 0.0, 2.0, 0.05)),
    "emissive_albedo": (ParamSchema("glow_bleed_px", "int", 2, 0, 8, 1), ParamSchema("glow_strength", "slider", 0.35, 0.0, 1.5, 0.05)),
    "emissive_composite": (ParamSchema("glow_bleed_px", "int", 2, 0, 8, 1), ParamSchema("glow_strength", "slider", 0.35, 0.0, 1.5, 0.05)),
    "pad_islands": (ParamSchema("pad", "int", 2, 0, 8, 1),),
    "material_pad_islands": (ParamSchema("pad", "int", 2, 0, 8, 1),),
    "material_set_alpha": (ParamSchema("preserve_emissive", "checkbox", False),),
    "material_grade": (ParamSchema("ao_darken", "slider", 0.35, 0.0, 1.0, 0.02), ParamSchema("atmosphere_strength", "slider", 0.14, 0.0, 0.6, 0.02), ParamSchema("vibrance", "slider", 0.14, -0.5, 0.8, 0.02), ParamSchema("contrast", "slider", 1.05, 0.6, 1.6, 0.02), ParamSchema("shadow_cool", "slider", 0.04, 0.0, 0.25, 0.01)),
    "scroll": (ParamSchema("speed_x", "slider", 0.0, -4.0, 4.0, 0.25), ParamSchema("speed_y", "slider", 1.0, -4.0, 4.0, 0.25)),
    "pulse": (ParamSchema("min_val", "slider", 0.45, 0.0, 1.0, 0.05), ParamSchema("max_val", "slider", 1.0, 0.0, 2.0, 0.05), ParamSchema("freq", "slider", 1.0, 0.25, 8.0, 0.25)),
    "sdf_boolean": (ParamSchema("bool_op", "select", "smooth_union", choices=("union", "subtract", "intersect", "xor", "smooth_union", "smooth_subtract", "smooth_intersect")), ParamSchema("smoothness", "slider", 0.04, 0.0, 0.3, 0.005)),
    "tile_sampler": (ParamSchema("count_x", "int", 5, 1, 24, 1), ParamSchema("count_y", "int", 5, 1, 24, 1), ParamSchema("distribution", "select", "jitter_grid", choices=("jitter_grid", "stagger_grid", "random")), ParamSchema("scale_base", "slider", 0.92, 0.2, 3.0, 0.05), ParamSchema("rot_jitter_deg", "slider", 25.0, 0.0, 180.0, 5.0), ParamSchema("density", "slider", 1.0, 0.0, 1.0, 0.02)),
    "flood_fill": (ParamSchema("threshold", "slider", 0.25, 0.01, 0.99, 0.02),),
    "histogram_scan": (ParamSchema("position", "slider", 0.5, 0.0, 1.0, 0.01), ParamSchema("contrast", "slider", 0.6, 0.0, 1.0, 0.02), ParamSchema("invert", "checkbox", False)),
    "histogram_select": (ParamSchema("position", "slider", 0.5, 0.0, 1.0, 0.01), ParamSchema("range_width", "slider", 0.25, 0.02, 1.0, 0.02), ParamSchema("contrast", "slider", 0.5, 0.0, 1.0, 0.02)),
    "auto_levels": (ParamSchema("low_pct", "slider", 1.0, 0.0, 20.0, 0.5), ParamSchema("high_pct", "slider", 99.0, 80.0, 100.0, 0.5)),
    "curvature": (ParamSchema("radius", "int", 2, 1, 8, 1), ParamSchema("strength", "slider", 2.2, 0.2, 8.0, 0.1)),
    "edge_wear": (ParamSchema("wear_amount", "slider", 0.45, 0.0, 1.0, 0.02), ParamSchema("chip_scale", "slider", 8.0, 2.0, 32.0, 0.5), ParamSchema("grime_amount", "slider", 0.55, 0.0, 1.0, 0.02)),
    "displace_uv": (ParamSchema("strength_x", "slider", 2.0, -24.0, 24.0, 0.5), ParamSchema("strength_y", "slider", 2.0, -24.0, 24.0, 0.5), ParamSchema("pixel_snap", "checkbox", False)),
    "normal_light_2d": (ParamSchema("light_az_deg", "slider", -45.0, -180.0, 180.0, 2.0), ParamSchema("light_el_deg", "slider", 48.0, 5.0, 90.0, 2.0), ParamSchema("light_color", "color", "#FFF7E6"), ParamSchema("ambient_strength", "slider", 0.34, 0.0, 1.0, 0.02)),
    "directional_selout": (ParamSchema("light_angle_deg", "slider", -135.0, -180.0, 180.0, 5.0), ParamSchema("lit_darken", "slider", 0.64, 0.1, 1.0, 0.02), ParamSchema("shadow_darken", "slider", 0.30, 0.05, 0.9, 0.02)),
    "pixel_dither": (ParamSchema("palette", "palette", ["#1B1726", "#4D5368", "#949FB5", "#E5EEF7"]), ParamSchema("pattern", "select", "bayer_2x2", choices=("bayer_2x2", "bayer_4x4", "bayer_8x8", "diamond", "blue_noise", "floyd_steinberg", "atkinson")), ParamSchema("contrast", "slider", 1.0, 0.0, 2.5, 0.05), ParamSchema("mode", "select", "color", choices=("color", "alpha"))),
    "palette_posterize": (ParamSchema("palette", "palette", ["#1B1726", "#4D5368", "#949FB5", "#E5EEF7"]), ParamSchema("space", "select", "oklab", choices=("oklab", "rgb", "luma_rank")), ParamSchema("posterize_alpha", "checkbox", True)),
    "color_replace": (ParamSchema("src_palette", "palette", ["#1B1726", "#949FB5"]), ParamSchema("dst_palette", "palette", ["#261414", "#F5D08A"]), ParamSchema("tolerance", "slider", 0.25, 0.01, 1.0, 0.02)),
    "level_selector": (ParamSchema("center", "slider", 0.75, 0.0, 1.0, 0.01), ParamSchema("width", "slider", 0.20, 0.02, 1.0, 0.01), ParamSchema("softness", "slider", 0.05, 0.0, 0.5, 0.01)),
    "de_stray": (ParamSchema("strictness", "select", "medium", choices=("low", "medium", "high")), ParamSchema("iterations", "int", 1, 1, 4, 1)),
    "de_corner": (ParamSchema("strictness", "select", "medium", choices=("low", "medium", "high")), ParamSchema("iterations", "int", 1, 1, 4, 1)),
    "pixel_clusters": (ParamSchema("palette", "palette", ["#1B1726", "#4D5368", "#949FB5", "#E5EEF7"]), ParamSchema("direction", "select", "isotropic", choices=("isotropic", "horizontal", "vertical", "diag_pos", "diag_neg", "omni")), ParamSchema("cluster_size", "int", 2, 1, 4, 1), ParamSchema("dither_strength", "slider", 0.0, 0.0, 1.0, 0.05)),
    "pixel_bevel_directional": (ParamSchema("light_angle_deg", "slider", -135.0, -180.0, 180.0, 5.0), ParamSchema("bevel_width", "slider", 0.11, 0.03, 0.35, 0.01), ParamSchema("chip_amount", "slider", 0.42, 0.0, 1.0, 0.02)),
    "pixel_outline": (ParamSchema("width", "int", 1, 1, 4, 1), ParamSchema("position", "select", "outside", choices=("outside", "inside")), ParamSchema("color", "color", "#161224"), ParamSchema("corner_mode", "select", "4way", choices=("4way", "8way"))),
    "heightmap_shadow": (ParamSchema("light_angle_deg", "slider", -135.0, -180.0, 180.0, 5.0), ParamSchema("elevation_deg", "slider", 34.0, 8.0, 82.0, 2.0), ParamSchema("max_steps", "int", 8, 1, 16, 1), ParamSchema("strength", "slider", 0.72, 0.0, 1.0, 0.02)),
    "pixel_mirror": (ParamSchema("axis", "select", "x", choices=("x", "y", "xy")),),
    "pixel_sort": (ParamSchema("axis", "select", "vertical", choices=("vertical", "horizontal")), ParamSchema("low_thresh", "slider", 0.25, 0.0, 1.0, 0.02), ParamSchema("high_thresh", "slider", 0.85, 0.0, 1.0, 0.02), ParamSchema("reverse", "checkbox", False)),
    "chromatic_aberration": (ParamSchema("shift_x", "int", 1, -6, 6, 1), ParamSchema("shift_y", "int", 0, -6, 6, 1), ParamSchema("strength", "slider", 0.65, 0.0, 1.0, 0.05)),
    "vignette": (ParamSchema("strength", "slider", 0.35, 0.0, 1.0, 0.02), ParamSchema("radius", "slider", 0.72, 0.1, 1.2, 0.02), ParamSchema("color", "color", "#120E1D")),
    "orc_flesh_skin": (ParamSchema("stops", "gradient_stops", [(0.0, "#1B2822"), (0.5, "#4F6E43"), (1.0, "#C5D694")]), ParamSchema("scar_amount", "slider", 0.32, 0.0, 1.0, 0.02), ParamSchema("warpaint_amount", "slider", 0.14, 0.0, 1.0, 0.02), ParamSchema("warpaint_color", "color", "#782224")),
    "stitched_leather": (ParamSchema("stops", "gradient_stops", [(0.0, "#231616"), (0.5, "#6E4630"), (1.0, "#BD936B")]), ParamSchema("stitch_density", "slider", 1.0, 0.0, 2.0, 0.05), ParamSchema("wear", "slider", 0.35, 0.0, 1.0, 0.02)),
    "forged_battle_metal": (ParamSchema("stops", "gradient_stops", [(0.0, "#1A1B24"), (0.5, "#4B5263"), (1.0, "#E0EAF2")]), ParamSchema("scratches", "slider", 0.50, 0.0, 1.0, 0.02), ParamSchema("rivets", "slider", 0.50, 0.0, 1.0, 0.02), ParamSchema("rust", "slider", 0.20, 0.0, 1.0, 0.02)),
    "woven_battle_cloth": (ParamSchema("stops", "gradient_stops", [(0.0, "#240C16"), (0.5, "#882730"), (1.0, "#DF786E")]), ParamSchema("style", "select", "drape", choices=("drape", "rope")), ParamSchema("fold_contrast", "slider", 0.44, 0.05, 1.0, 0.02)),
    "keratin_horn_bone": (ParamSchema("stops", "gradient_stops", [(0.0, "#352826"), (0.6, "#B89F7D"), (1.0, "#F5ECD8")]), ParamSchema("orientation", "select", "tip_up", choices=("tip_up", "tip_down", "tip_out")), ParamSchema("ring_count", "slider", 5.0, 1.0, 14.0, 0.5)),
    "paint_entity_eyes_and_parts": (ParamSchema("eye_facing", "select", "auto", choices=("auto", "front", "lateral", "both")), ParamSchema("eye_style", "select", "gradient", choices=("gradient", "creature")), ParamSchema("eye_iris", "color", "#FF6E1A"), ParamSchema("eye_glow", "slider", 0.85, 0.0, 1.0, 0.02), ParamSchema("eye_blink", "slider", 0.0, 0.0, 1.0, 0.02), ParamSchema("rune_glow", "slider", 0.85, 0.0, 1.0, 0.02)),
    "cobblestone_masonry": (ParamSchema("cols", "int", 4, 2, 12, 1), ParamSchema("rows", "int", 4, 2, 12, 1), ParamSchema("moss_amount", "slider", 0.22, 0.0, 1.0, 0.02)),
    "bark_and_timber": (ParamSchema("furrow_count", "slider", 7.0, 2.0, 20.0, 0.5), ParamSchema("moss_amount", "slider", 0.18, 0.0, 1.0, 0.02)),
    "crystalline_ore_vein": (ParamSchema("ore_density", "slider", 0.35, 0.05, 0.9, 0.02), ParamSchema("glow_strength", "slider", 0.75, 0.0, 1.0, 0.02)),
    "caustic_water": (ParamSchema("scale", "slider", 5.5, 2.0, 16.0, 0.5), ParamSchema("foam_amount", "slider", 0.32, 0.0, 1.0, 0.02)),
    "organic_fur_pelt": (ParamSchema("pattern", "select", "saddle", choices=("solid", "saddle", "tabby", "spots", "dappled")), ParamSchema("tuft_scale", "slider", 10.0, 4.0, 24.0, 0.5), ParamSchema("countershade_height", "slider", 0.44, 0.0, 1.0, 0.02)),
    "layered_plumage": (ParamSchema("marking", "select", "barred", choices=("solid", "barred", "speckled", "iridescent")), ParamSchema("cols", "int", 8, 3, 24, 1), ParamSchema("rows", "int", 10, 3, 24, 1), ParamSchema("iridescence", "slider", 0.18, 0.0, 1.0, 0.02)),
    "chitin_carapace": (ParamSchema("segments", "int", 6, 2, 18, 1), ParamSchema("iridescence", "slider", 0.26, 0.0, 1.0, 0.02), ParamSchema("glossiness", "slider", 0.72, 0.1, 0.95, 0.02)),
    "aquatic_scales_skin": (ParamSchema("mode", "select", "fish_scales", choices=("fish_scales", "amphibian", "koi_calico")), ParamSchema("wetness", "slider", 0.82, 0.1, 1.0, 0.02)),
    "elemental_golem_surface": (ParamSchema("slab_scale", "slider", 4.5, 1.5, 16.0, 0.25), ParamSchema("glow_strength", "slider", 0.85, 0.0, 1.0, 0.02)),
    "sprite_mask": (ParamSchema("shape", "text", "sword"),),
    "smooth_wing_rgba": (ParamSchema("arches", "int", 2, 1, 6, 1), ParamSchema("smooth_blend", "slider", 0.78, 0.0, 1.0, 0.02)),
    "smooth_wing_multi": (ParamSchema("arches", "int", 2, 1, 6, 1), ParamSchema("smooth_blend", "slider", 0.78, 0.0, 1.0, 0.02)),
    "colossal_dragon_membrane": (ParamSchema("tendon_ribs", "int", 3, 1, 8, 1), ParamSchema("backlit_sss", "slider", 0.85, 0.0, 1.0, 0.02)),
    "gradient_eye_rgba": (ParamSchema("pupil_shape", "select", "slit", choices=("slit", "round")), ParamSchema("glow_strength", "slider", 0.85, 0.0, 1.0, 0.02)),
    "gradient_eye_multi": (ParamSchema("pupil_shape", "select", "slit", choices=("slit", "round")), ParamSchema("glow_strength", "slider", 0.85, 0.0, 1.0, 0.02)),
    "eye_paint_rgba": (ParamSchema("pupil_shape", "select", "slit", choices=("slit", "round")),),
}


def _populate_complete_param_schemas() -> None:
    """Guarantee 100% of registered nodes expose clean typed ParamSchema definitions (zero blank GUI inspectors!)."""
    for op, meta in list(NODE_CATALOG.items()):
        if not meta.params:
            p_spec = _PARAM_OVERRIDES.get(op) or (ParamSchema("seed", "int", 42, 0, 99999, 1, group="General"),)
            NODE_CATALOG[op] = NodeMeta(
                op=meta.op,
                category=meta.category,
                display_name=meta.display_name,
                description=meta.description,
                inputs=meta.inputs,
                output_type=meta.output_type,
                outputs=meta.outputs,
                params=p_spec,
                time_aware=meta.time_aware,
            )


_populate_complete_param_schemas()


def export_node_schema_json() -> dict[str, Any]:
    """Return a commercial GUI-ready JSON descriptor (`schema_version: 4.0`, `engine_version: 10.0`)."""
    from .pipeline import MATERIAL_BUS_PORTS, SUBGRAPH_REGISTRY

    by_cat: dict[str, list[dict[str, Any]]] = {}
    with_params = 0
    for op, meta in NODE_CATALOG.items():
        if meta.params:
            with_params += 1
        by_cat.setdefault(meta.category, []).append(asdict(meta))
    subgraphs = [
        {
            "name": name,
            "category": info["category"],
            "description": info["description"],
            "defaults": info["defaults"],
        }
        for name, info in SUBGRAPH_REGISTRY.items()
    ]
    return {
        "schema_version": "4.0",
        "engine_version": "10.0",
        "node_count": len(NODE_CATALOG),
        "subgraph_count": len(subgraphs),
        "param_schema_coverage_pct": round(100.0 * with_params / max(1, len(NODE_CATALOG)), 1),
        "port_colors": dict(PORT_COLORS),
        "port_compatibility": {k: list(v) for k, v in PORT_COMPATIBILITY.items()},
        "material_bus_ports": list(MATERIAL_BUS_PORTS),
        "supports_multi_port": True,
        "supports_material_bus": True,
        "supports_multipass": True,
        "supports_generate_iterate_simulate_animate": True,
        "supports_param_drivers": True,
        "supports_dag_validation": True,
        "supports_incremental_cache": True,
        "supports_texdsl": True,
        "supports_pixel_composer_fx": True,
        "supports_gui_layout": True,
        "supports_uv_edge_solver": True,
        "supports_precomputed_shadow_pbr": True,
        "categories": list(by_cat.keys()),
        "nodes_by_category": by_cat,
        "subgraphs": subgraphs,
    }


def save_node_schema_json(path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(export_node_schema_json(), fh, indent=2)
