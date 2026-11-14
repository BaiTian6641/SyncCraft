"""Actual file handoff checks: content, determinism, and no outputs on audit fail."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from synccraft.cli import compile_file, main


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "simple_blade.blockdsl"


class PipelineTests(unittest.TestCase):
    def test_complete_static_handoff_and_deterministic_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            result = compile_file(EXAMPLE, first, target="java_modern")
            compile_file(EXAMPLE, second, target="java_modern")
            expected = {
                "asset.blockdsl", "ast.json", "scene.json", "audit.json", "asset.bbmodel",
                "asset.java.json", "atlas.png", "atlas.json", "build_info.json", "preview.png",
            }
            self.assertEqual({p.name for p in first.iterdir()}, expected)
            for name in expected:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
            self.assertTrue(result["audit_passed"])
            self.assertFalse(result["engine_import_verified"])
            java = json.loads((first / "asset.java.json").read_text())
            # A known handle is 2x6x2, centered at [8,8,8]; this expectation is
            # independent of the JSON writer's schema or its computed scene.
            self.assertEqual(java["elements"][0]["from"], [7, 5, 7])
            self.assertEqual(java["elements"][0]["to"], [9, 11, 9])
            self.assertEqual(len(java["elements"]), 4)
            with Image.open(first / "atlas.png") as atlas:
                self.assertEqual(atlas.mode, "RGBA")
                nontransparent = [rgba for _, rgba in atlas.getcolors(atlas.width * atlas.height) if rgba[3] != 0]
                self.assertEqual(len(set(nontransparent)), 3)
                self.assertTrue(all(rgba[3] == 255 for rgba in nontransparent))
            with Image.open(first / "preview.png") as preview:
                self.assertEqual(preview.size, (1200, 920))
                self.assertGreater(len(preview.getcolors(preview.width * preview.height)), 20)
            info = json.loads((first / "build_info.json").read_text())
            self.assertFalse(info["complete_for_release"])
            self.assertFalse(info["engine_import_verified"])

    def test_default_bbmodel_compile_and_explicit_overwrite_remove_stale_java_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            compile_file(EXAMPLE, output, target="java_modern")
            marker = output / "user-notes.txt"
            marker.write_text("retain my work")
            with self.assertRaisesRegex(ValueError, "not empty"):
                compile_file(EXAMPLE, output)
            compile_file(EXAMPLE, output, overwrite=True)
            self.assertFalse((output / "asset.java.json").exists())
            self.assertEqual(marker.read_text(), "retain my work")
            info = json.loads((output / "build_info.json").read_text())
            self.assertEqual(info["profile"], "bbmodel_5")

    def test_audit_failure_prevents_export_and_preserves_existing_output(self):
        invalid = """@meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
@palette { gray: ramp(base_lch=[0.5,0,0], steps=4) }
@rig { root bone root_node { pivot_local: [8,8,8]
box first { size: [2,2,2], material: gray }
box second place(anchor=[0,0,0], align=[0,0,0]) { size: [2,2,2], material: gray }
} }
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "collision.blockdsl"
            source.write_text(invalid)
            output = root / "output"
            stderr, stdout = io.StringIO(), io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(stdout):
                code = main(["compile", str(source), "--out", str(output)])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            self.assertIn("geometry audit failed", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            output.mkdir()
            marker = output / "asset.bbmodel"
            marker.write_text("previous valid export")
            with redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "geometry audit failed"):
                    compile_file(source, output, overwrite=True)
            self.assertEqual(marker.read_text(), "previous valid export")
            self.assertEqual([p.name for p in output.iterdir()], ["asset.bbmodel"])


if __name__ == "__main__":
    unittest.main()
