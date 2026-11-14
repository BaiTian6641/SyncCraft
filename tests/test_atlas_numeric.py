import unittest

from synccraft.shading.uv_packer import oklch_to_srgb


class AtlasNumericTests(unittest.TestCase):
    def test_extreme_finite_color_is_a_controlled_error(self):
        with self.assertRaisesRegex(ValueError, "numeric range"):
            oklch_to_srgb((0.5, 1e200, 60))


if __name__ == "__main__":
    unittest.main()
