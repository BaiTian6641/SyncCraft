#!/usr/bin/env python3
"""Deterministic dependency-free VSIX packager & icon generator for `editors/vscode-blockdsl`.

Generates `editors/vscode-blockdsl/images/icon.png` (if missing or `--regen-icon` is set)
and packages `editors/vscode-blockdsl` into a standard Open Packaging Conventions (OPC)
`.vsix` archive (`dist/synccraft-blockdsl-<version>.vsix`) ready for one-click install via
`Extensions: Install from VSIX...` in VS Code, Antigravity, Cursor, or VSCodium — without
requiring `npm` or `@vscode/vsce`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape
import zipfile

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = REPO_ROOT / "editors" / "vscode-blockdsl"
DEFAULT_OUT_DIR = REPO_ROOT / "dist"


def generate_extension_icon(icon_path: Path) -> None:
    """Render a crisp 128x128 isometric constructive 3D block + SE(3) pivot icon."""
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    size = 128
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded dark slate badge background
    draw.rounded_rectangle([4, 4, 123, 123], radius=24, fill=(18, 25, 38, 255), outline=(56, 189, 248, 255), width=3)

    # Isometric cube faces centered at (64, 66)
    cx, cy = 64, 66
    dx, dy, dz = 34, 20, 36
    top_face = [(cx, cy - dz - dy), (cx + dx, cy - dz), (cx, cy - dz + dy), (cx - dx, cy - dz)]
    left_face = [(cx - dx, cy - dz), (cx, cy - dz + dy), (cx, cy + dy), (cx - dx, cy)]
    right_face = [(cx, cy - dz + dy), (cx + dx, cy - dz), (cx + dx, cy), (cx, cy + dy)]

    draw.polygon(top_face, fill=(56, 189, 248, 255))
    draw.polygon(left_face, fill=(14, 116, 144, 255))
    draw.polygon(right_face, fill=(8, 76, 112, 255))

    # Crisp wireframe seams + SE(3) joint axes triad
    for poly in (top_face, left_face, right_face):
        draw.line(poly + [poly[0]], fill=(224, 242, 254, 255), width=2)

    pivot = (cx, cy - dz + dy)
    draw.line([pivot, (cx, cy - dz - 12)], fill=(74, 222, 128, 255), width=3)   # +Y Up
    draw.line([pivot, (cx + 24, cy - dz + dy + 14)], fill=(248, 113, 113, 255), width=3)  # +X
    draw.line([pivot, (cx - 24, cy - dz + dy + 14)], fill=(250, 204, 21, 255), width=3)   # +Z
    draw.ellipse([pivot[0] - 5, pivot[1] - 5, pivot[0] + 5, pivot[1] + 5], fill=(255, 255, 255, 255))

    img.save(icon_path, format="PNG")


def build_content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension=".json" ContentType="application/json"/>\n'
        '  <Default Extension=".code-snippets" ContentType="application/json"/>\n'
        '  <Default Extension=".js" ContentType="application/javascript"/>\n'
        '  <Default Extension=".md" ContentType="text/markdown"/>\n'
        '  <Default Extension=".png" ContentType="image/png"/>\n'
        '  <Default Extension=".txt" ContentType="text/plain"/>\n'
        '  <Default Extension=".vsixmanifest" ContentType="text/xml"/>\n'
        '</Types>\n'
    )


def build_vsix_manifest(pkg: dict) -> str:
    publisher = xml_escape(str(pkg["publisher"]))
    name = xml_escape(str(pkg["name"]))
    version = xml_escape(str(pkg["version"]))
    display_name = xml_escape(str(pkg.get("displayName", name)))
    description = xml_escape(str(pkg.get("description", "")))
    categories = xml_escape(",".join(pkg.get("categories", ["Programming Languages"])))
    tags = xml_escape(",".join(pkg.get("keywords", [])))
    engine = xml_escape(str(pkg.get("engines", {}).get("vscode", "^1.85.0")))

    return f"""<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011" xmlns:d="http://schemas.microsoft.com/developer/vsx-schema-design/2011">
  <Metadata>
    <Identity Language="en-US" Id="{name}" Version="{version}" Publisher="{publisher}" />
    <DisplayName>{display_name}</DisplayName>
    <Description xml:space="preserve">{description}</Description>
    <Tags>{tags}</Tags>
    <Categories>{categories}</Categories>
    <GalleryFlags>Public</GalleryFlags>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{engine}" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionDependencies" Value="" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionPack" Value="" />
      <Property Id="Microsoft.VisualStudio.Code.LocalizedLanguages" Value="" />
      <Property Id="Microsoft.VisualStudio.Services.GitHubFlavoredMarkdown" Value="true" />
    </Properties>
    <License>extension/LICENSE.txt</License>
    <Icon>extension/images/icon.png</Icon>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code" />
  </Installation>
  <Dependencies />
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true" />
    <Asset Type="Microsoft.VisualStudio.Services.Content.Details" Path="extension/README.md" Addressable="true" />
    <Asset Type="Microsoft.VisualStudio.Services.Content.Changelog" Path="extension/CHANGELOG.md" Addressable="true" />
    <Asset Type="Microsoft.VisualStudio.Services.Content.License" Path="extension/LICENSE.txt" Addressable="true" />
    <Asset Type="Microsoft.VisualStudio.Services.Icons.Default" Path="extension/images/icon.png" Addressable="true" />
  </Assets>
</PackageManifest>
"""


def package_vsix(out_dir: Path = DEFAULT_OUT_DIR, *, regen_icon: bool = False) -> Path:
    pkg = json.loads((EXT_DIR / "package.json").read_text(encoding="utf-8"))
    icon_path = EXT_DIR / "images" / "icon.png"
    if regen_icon or not icon_path.exists():
        generate_extension_icon(icon_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    vsix_path = out_dir / f"{pkg['name']}-{pkg['version']}.vsix"

    excluded_names = {".DS_Store", "node_modules", "__pycache__"}
    files_to_pack: list[tuple[Path, str]] = []
    for file_path in sorted(EXT_DIR.rglob("*")):
        if file_path.is_dir():
            continue
        if any(part in excluded_names for part in file_path.parts):
            continue
        if file_path.suffix == ".vsix":
            continue
        rel = file_path.relative_to(EXT_DIR).as_posix()
        files_to_pack.append((file_path, f"extension/{rel}"))

    fixed_dt = (2026, 11, 15, 0, 0, 0)
    with zipfile.ZipFile(vsix_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, xml_body in (
            ("[Content_Types].xml", build_content_types_xml()),
            ("extension.vsixmanifest", build_vsix_manifest(pkg)),
            ("extension/LICENSE.txt", (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")),
        ):
            zi = zipfile.ZipInfo(arcname, date_time=fixed_dt)
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, xml_body.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)

        for src_file, arcname in files_to_pack:
            zi = zipfile.ZipInfo(arcname, date_time=fixed_dt)
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, src_file.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)

    return vsix_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--regen-icon", action="store_true")
    args = parser.parse_args()
    vsix = package_vsix(args.out_dir, regen_icon=args.regen_icon)
    print(json.dumps({"vsix": str(vsix), "bytes": vsix.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
