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
    widget: str  # "slider" | "int" | "select" | "color" | "gradient_stops" | "curve" | "checkbox" | "text" | "points"
    default: Any
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()


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
    "constant": NodeMeta("constant", "Gradients & Paths", "Constant", "Uniform scalar or hex color fill.", (), "ANY", (ParamSchema("value", "text", 0.5),)),
    "gradient": NodeMeta("gradient", "Gradients & Paths", "Simple Gradient", "Basic directional/radial scalar gradient.", (), "SCALAR", (ParamSchema("direction", "select", "vertical", choices=("vertical", "horizontal", "radial")),)),
    "gradient_field": NodeMeta("gradient_field", "Gradients & Paths", "Studio Gradient Field", "Linear, radial, conical, diamond, box or reflected smooth gradient with easing.", (), "SCALAR", (
        ParamSchema("kind", "select", "linear", choices=("linear", "radial", "radial_out", "diamond", "box", "conical", "reflected")),
        ParamSchema("angle_deg", "slider", 90.0, -180.0, 180.0, 1.0),
        ParamSchema("radius", "slider", 0.55, 0.05, 2.0, 0.01),
        ParamSchema("ease", "select", "smoothstep", choices=("linear", "smoothstep", "smootherstep", "ease_in", "ease_out")),
    )),
    "multi_stop_gradient": NodeMeta("multi_stop_gradient", "Gradients & Paths", "Multi-Stop OKLab Gradient", "Silky smooth perceptual OKLab multi-stop color ramp with smooth-to-pixel blend.", (PortSchema("src", "SCALAR"), PortSchema("alpha_mask", "MASK", True)), "RGBA", (
        ParamSchema("stops", "gradient_stops", [(0.0, "#190E33"), (0.5, "#7C3AED"), (1.0, "#A5F3FC")]),
        ParamSchema("space", "select", "oklab", choices=("oklab", "srgb")),
        ParamSchema("smooth_blend", "slider", 1.0, 0.0, 1.0, 0.05),
        ParamSchema("quantize_steps", "int", 0, 0, 32, 1),
    )),
    "cosine_gradient": NodeMeta("cosine_gradient", "Gradients & Paths", "Cosine Palette (Gradient Cos)", "Iridescent procedural cosine color ramp.", (PortSchema("src", "SCALAR"),), "RGBA", (), time_aware=True),
    "spline_path": NodeMeta("spline_path", "Gradients & Paths", "Spline Path Gradient", "Smooth Catmull-Rom curve ribbon, along-path gradient, or distance field.", (), "SCALAR", (
        ParamSchema("points", "points", [(0.1, 0.2), (0.5, 0.7), (0.9, 0.3)]),
        ParamSchema("stroke_width", "slider", 0.12, 0.01, 0.5, 0.01),
        ParamSchema("output", "select", "ribbon", choices=("ribbon", "along", "sdf")),
    )),
    "per_island_gradient": NodeMeta("per_island_gradient", "Gradients & Paths", "Per-Island Face Gradient", "Smooth top-to-bottom, convex center-out, or edge-rim gradient per Blockbench face.", (), "SCALAR", (
        ParamSchema("direction", "select", "top_to_bottom", choices=("top_to_bottom", "bottom_to_top", "left_to_right", "diag", "center_out", "edge_rim")),
        ParamSchema("curve", "slider", 1.0, 0.2, 3.0, 0.05),
    )),
    "body_flow_gradient": NodeMeta("body_flow_gradient", "Gradients & Paths", "3D Body-Flow Gradient", "Seamless creature-space dorsal-ventral, head-to-tail, root-to-tip, or half-Lambert light wrap.", (), "SCALAR", (
        ParamSchema("axis", "select", "dorsal_ventral", choices=("dorsal_ventral", "head_to_tail", "center_to_tips", "top_light_wrap")),
        ParamSchema("bias", "slider", 0.0, -0.5, 0.5, 0.02),
        ParamSchema("gamma", "slider", 1.0, 0.3, 3.0, 0.05),
    )),
    # --- Noise ---
    "value_noise": NodeMeta("value_noise", "Noise", "Value Noise", "Tileable bicubic value noise.", (), "SCALAR", (ParamSchema("scale", "slider", 8.0, 1.0, 64.0, 0.5),)),
    "perlin": NodeMeta("perlin", "Noise", "Perlin Noise", "Tileable gradient Perlin noise with seamless time loop.", (), "SCALAR", (ParamSchema("scale", "slider", 8.0, 1.0, 64.0, 0.5),), time_aware=True),
    "fbm": NodeMeta("fbm", "Noise", "fBm / Ridged / Billow", "Multi-octave fractal Brownian motion noise.", (), "SCALAR", (ParamSchema("scale", "slider", 6.0, 1.0, 48.0, 0.5), ParamSchema("octaves", "int", 4, 1, 8, 1), ParamSchema("mode", "select", "fbm", choices=("fbm", "ridged", "billow", "turbulence"))), time_aware=True),
    "voronoi": NodeMeta("voronoi", "Noise", "Voronoi / Cellular", "Tileable Worley cellular F1, F2, cracks, edge, or cell ID.", (), "SCALAR", (ParamSchema("scale", "slider", 6.0, 1.0, 48.0, 0.5), ParamSchema("output", "select", "edge", choices=("f1", "f2", "f2_minus_f1", "edge", "cracks", "cell_id"))), time_aware=True),
    "gabor": NodeMeta("gabor", "Noise", "Anisotropic Gabor", "Directional brushed metal, striations, and bone flutes.", (), "SCALAR", (ParamSchema("scale", "slider", 12.0, 2.0, 64.0, 1.0), ParamSchema("angle_deg", "slider", 0.0, -180.0, 180.0, 1.0))),
    "domain_warp": NodeMeta("domain_warp", "Noise", "Domain / Curl Warp", "Organic vector-field distortion warp.", (PortSchema("src", "ANY"),), "ANY", (ParamSchema("strength", "slider", 0.08, 0.0, 0.5, 0.01), ParamSchema("mode", "select", "fbm", choices=("fbm", "curl"))), time_aware=True),
    "noise_3d": NodeMeta("noise_3d", "Noise", "3D Rest-Space Noise", "Seamless 3D world-space noise across creature UV seams.", (), "SCALAR", (ParamSchema("scale", "slider", 0.22, 0.02, 2.0, 0.02), ParamSchema("mode", "select", "fbm", choices=("fbm", "ridged", "billow"))), time_aware=True),
    # --- Geometric & Structural Patterns ---
    "grid": NodeMeta("grid", "Geometric Patterns", "Grid Tiles", "Beveled rectangular tiles & mortar.", (), "HEIGHT"),
    "brick": NodeMeta("brick", "Geometric Patterns", "Brick Masonry", "Running, stacked, herringbone, or basketweave brickwork.", (), "HEIGHT"),
    "hex_grid": NodeMeta("hex_grid", "Geometric Patterns", "Hexagon Grid", "Beveled hexagonal tiles & conduits.", (), "HEIGHT"),
    "tri_grid": NodeMeta("tri_grid", "Geometric Patterns", "Triangle Grid", "Faceted triangular mosaic.", (), "HEIGHT"),
    "diamond": NodeMeta("diamond", "Geometric Patterns", "Diamond / Chevron", "Quilted diamond or herringbone chevron.", (), "HEIGHT"),
    "truchet": NodeMeta("truchet", "Geometric Patterns", "Truchet Curves", "Interlocking calligraphic arcs & tiles.", (), "SCALAR"),
    "mosaic": NodeMeta("mosaic", "Geometric Patterns", "Voronoi Mosaic", "Beveled irregular stone/scute mosaic.", (), "HEIGHT"),
    "radial_rings": NodeMeta("radial_rings", "Geometric Patterns", "Radial Rings", "Concentric mandala/astrolabe rings & spokes.", (), "SCALAR", (), time_aware=True),
    "energy_veins": NodeMeta("energy_veins", "Geometric Patterns", "Energy Veins", "Animated lightning, circuit, or nebula energy streams.", (), "MASK", (), time_aware=True),
    # --- Painterly & Biological Craft ---
    "brushstrokes": NodeMeta("brushstrokes", "Painterly & Bio", "Directional Brushstrokes", "Long smooth overlapping hand-painted studio strokes.", (), "SCALAR", (ParamSchema("angle_deg", "slider", 0.0, -180.0, 180.0, 1.0), ParamSchema("contrast", "slider", 0.35, 0.05, 1.0, 0.05))),
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
    "anatomical_muscle_scales": NodeMeta(
        "anatomical_muscle_scales",
        "Painterly & Bio",
        "Anatomical Scale-to-Muscle Fit (MATERIAL)",
        "Fits chunky Blockbench scales, limb tendon bars, dorsal crowns & belly scutes directly to baked 3D muscle topology.",
        (),
        "MATERIAL",
        outputs=(
            PortSchema("albedo", "RGBA"), PortSchema("height", "HEIGHT"), PortSchema("normal", "NORMAL"),
            PortSchema("roughness", "SCALAR"), PortSchema("metallic", "SCALAR"), PortSchema("ao", "SCALAR"),
            PortSchema("muscle_crevice", "MASK"), PortSchema("crown_highlight", "MASK"),
        ),
    ),
    "colossal_dragon_membrane": NodeMeta(
        "colossal_dragon_membrane",
        "Painterly & Bio",
        "Colossal Dragon Wing Membrane (MATERIAL)",
        "Monumental full-chord bat-dragon membrane with curved cross-bay tendon ribs, diagonal pleats, battle tatters & backlit SSS (100% dorsal/ventral aligned).",
        (PortSchema("span_u", "SCALAR"), PortSchema("trail_v", "SCALAR"), PortSchema("face_is_down", "SCALAR", True)),
        "MATERIAL",
        outputs=(
            PortSchema("albedo", "RGBA"), PortSchema("alpha", "MASK"), PortSchema("height", "HEIGHT"),
            PortSchema("normal", "NORMAL"), PortSchema("roughness", "SCALAR"), PortSchema("ao", "SCALAR"),
            PortSchema("sss", "SCALAR"), PortSchema("tendon_mask", "MASK"), PortSchema("tatter_rim", "MASK"),
        ),
    ),
    "sprite_mask": NodeMeta("sprite_mask", "Sprites & Masks", "Item Sprite Mask", "Procedural pixel-art item silhouette generator.", (), "MASK"),
}


def export_node_schema_json() -> dict[str, Any]:
    """Return a clean UI-ready JSON descriptor grouping all TexGraph nodes & reusable Subgraphs by category."""
    from .pipeline import SUBGRAPH_REGISTRY

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for op, meta in NODE_CATALOG.items():
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
        "schema_version": "3.1",
        "node_count": len(NODE_CATALOG),
        "subgraph_count": len(subgraphs),
        "supports_multi_port": True,
        "supports_material_bus": True,
        "supports_multipass": True,
        "supports_generate_iterate_simulate_animate": True,
        "supports_param_drivers": True,
        "categories": list(by_cat.keys()),
        "nodes_by_category": by_cat,
        "subgraphs": subgraphs,
    }


def save_node_schema_json(path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(export_node_schema_json(), fh, indent=2)
