import tempfile
import unittest
from pathlib import Path

from tools.build_two_class_monster_dataset import (
    CLASS_NAMES,
    clip_boxes,
    load_manifest,
)


class TwoClassMonsterDatasetTests(unittest.TestCase):
    def test_class_order_is_stable(self):
        self.assertEqual(
            CLASS_NAMES, ["zombie_mushroom", "thorn_mushroom"]
        )

    def test_rejects_source_shared_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "frame.png"
            source.touch()
            manifest = root / "manifest.json"
            manifest.write_text(
                """{
                  "sources": [
                    {"id": "a", "source": "frame.png", "split": "train"},
                    {"id": "b", "source": "frame.png", "split": "test"}
                  ]
                }""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Source leakage"):
                load_manifest(manifest)

    def test_crop_keeps_only_mostly_retained_boxes(self):
        boxes = [
            {"class": "zombie_mushroom", "box": [10, 10, 20, 20]},
            {"class": "thorn_mushroom", "box": [90, 10, 20, 20]},
        ]

        clipped = clip_boxes(boxes, [0, 0, 100, 100])

        self.assertEqual(len(clipped), 1)
        self.assertEqual(clipped[0]["class"], "zombie_mushroom")


if __name__ == "__main__":
    unittest.main()
