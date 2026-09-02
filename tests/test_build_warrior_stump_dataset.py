import json
import tempfile
import unittest
from pathlib import Path

from tools.build_warrior_stump_dataset import clip_boxes, clip_crop, load_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "training_data" / "warrior_stump_v1_manifest.json"


class WarriorStumpDatasetTests(unittest.TestCase):
    def test_source_splits_and_annotation_counts_are_frozen(self):
        _, entries = load_manifest(MANIFEST)
        counts = {entry["id"]: len(entry.get("boxes", [])) for entry in entries}

        self.assertEqual(counts["warrior_baseline"], 9)
        self.assertEqual(counts["warrior_motion_02"], 11)
        self.assertEqual(counts["warrior_motion_03"], 12)
        self.assertEqual(counts["town_negative"], 0)
        self.assertEqual(counts["warrior_motion_04"], 10)
        self.assertEqual(counts["warrior_motion_05_test"], 9)

    def test_manifest_keeps_raw_sources_in_one_split(self):
        _, entries = load_manifest(MANIFEST)
        assignments = {}
        for entry in entries:
            source = str(entry["resolved_source"]).lower()
            assignments.setdefault(source, set()).add(entry["split"])

        self.assertTrue(all(len(splits) == 1 for splits in assignments.values()))

    def test_crop_keeps_overlapping_stumps_individually(self):
        boxes = [
            {"class": "stump", "box": [100, 80, 70, 80]},
            {"class": "stump", "box": [135, 82, 70, 80]},
        ]

        clipped = clip_boxes(boxes, (80, 60, 180, 140))

        self.assertEqual(len(clipped), 2)
        self.assertEqual(clipped[0]["box"], [20.0, 20.0, 70.0, 80.0])
        self.assertEqual(clipped[1]["box"], [55.0, 22.0, 70.0, 80.0])

    def test_manifest_is_plain_json_for_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "manifest.json"
            copied.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")

            payload = json.loads(copied.read_text(encoding="utf-8"))

        self.assertEqual(payload["classes"], ["stump"])
        self.assertEqual(payload["version"], 1)

    def test_negative_crop_is_clipped_to_gameplay(self):
        self.assertEqual(clip_crop([90, 80, 40, 40], 100, 100), (90, 80, 10, 20))
        self.assertIsNone(clip_crop([100, 100, 10, 10], 100, 100))


if __name__ == "__main__":
    unittest.main()
