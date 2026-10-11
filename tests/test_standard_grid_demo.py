import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "milestone2" / "standard_grid_demo.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("standard_grid_demo", MODULE_PATH)
standard_grid_demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(standard_grid_demo)


def make_tiles(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        object_dir = tmp_path / f"images_OBJ{index + 1:03d}"
        object_dir.mkdir()
        path = object_dir / f"tile_{index}.png"
        Image.new("RGB", (224, 224), (index, index + 1, index + 2)).save(path)
        paths.append(path)
    return paths


class StandardGridGeometryTests(unittest.TestCase):
    def test_build_standard_2x2_grid_has_exact_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            grid, manifest = standard_grid_demo.build_standard_grid(make_tiles(Path(directory), 4), 2)
        self.assertEqual(grid.size, (448, 448))
        self.assertEqual(
            [(row["x1"], row["y1"], row["x2"], row["y2"]) for row in manifest],
            [
                (0, 0, 224, 224),
                (224, 0, 448, 224),
                (0, 224, 224, 448),
                (224, 224, 448, 448),
            ],
        )
        self.assertEqual(
            [row["expected_object_id"] for row in manifest],
            ["OBJ001", "OBJ002", "OBJ003", "OBJ004"],
        )

    def test_build_standard_3x3_grid_has_exact_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            grid, manifest = standard_grid_demo.build_standard_grid(make_tiles(Path(directory), 9), 3)
        self.assertEqual(grid.size, (672, 672))
        self.assertEqual(len(manifest), 9)
        self.assertEqual(manifest[-1]["x1"], 448)
        self.assertEqual(manifest[-1]["y1"], 448)
        self.assertEqual(manifest[-1]["x2"], 672)
        self.assertEqual(manifest[-1]["y2"], 672)


if __name__ == "__main__":
    unittest.main()
