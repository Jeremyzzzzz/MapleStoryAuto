import unittest

from tools.build_crocodile_dataset import CLASS_NAME, load_manifest


class CrocodileDatasetTests(unittest.TestCase):
    def test_manifest_is_source_grouped_and_has_all_splits(self):
        _, entries = load_manifest("training_data/crocodile_v1_manifest.json")
        self.assertEqual(CLASS_NAME, "crocodile")
        self.assertEqual({entry["split"] for entry in entries}, {"train", "val", "test"})
        self.assertTrue(all(entry["boxes"] for entry in entries))

    def test_raw_source_never_crosses_splits(self):
        _, entries = load_manifest("training_data/crocodile_v1_manifest.json")
        assignments = {}
        for entry in entries:
            assignments.setdefault(str(entry["resolved_source"]).lower(), set()).add(entry["split"])
        self.assertTrue(all(len(splits) == 1 for splits in assignments.values()))

    def test_hard_negative_manifest_has_empty_boxes_and_explicit_crops(self):
        _, entries = load_manifest("training_data/crocodile_hardneg_live_manifest.json")
        self.assertEqual({entry["split"] for entry in entries}, {"train", "val", "test"})
        self.assertTrue(all(not entry["boxes"] for entry in entries))
        self.assertTrue(all(entry["negative_crops"] for entry in entries))


if __name__ == "__main__":
    unittest.main()
