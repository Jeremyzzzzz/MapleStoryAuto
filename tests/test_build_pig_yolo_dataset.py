import json
import unittest
from pathlib import Path

from tools.build_pig_yolo_dataset import load_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "training_data" / "pig_yolo_v1_manifest.json"


class PigDatasetTests(unittest.TestCase):
    def test_source_splits_and_annotation_counts_are_frozen(self):
        _, entries = load_manifest(MANIFEST)
        counts = {entry["id"]: len(entry.get("boxes", [])) for entry in entries}

        self.assertEqual(counts["pig_source_01"], 2)
        self.assertEqual(counts["pig_source_02"], 4)
        self.assertEqual(counts["pig_source_03"], 4)
        self.assertEqual(counts["pig_source_04"], 2)
        self.assertEqual(counts["pig_source_05"], 2)
        self.assertEqual(counts["pig_source_06"], 1)
        self.assertEqual(counts["pig_source_07"], 3)
        self.assertEqual(counts["pig_source_08"], 2)

    def test_raw_frames_do_not_cross_splits(self):
        _, entries = load_manifest(MANIFEST)
        assignments = {}
        for entry in entries:
            source = str(entry["resolved_source"]).lower()
            assignments.setdefault(source, set()).add(entry["split"])

        self.assertTrue(all(len(splits) == 1 for splits in assignments.values()))

    def test_manifest_is_plain_json(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["classes"], ["pig"])
        self.assertEqual(payload["version"], 1)


if __name__ == "__main__":
    unittest.main()
