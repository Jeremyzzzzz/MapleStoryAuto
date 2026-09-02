import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.evaluate_monster_yolo import (
    box_iou,
    dataset_class_names,
    load_ground_truth,
    match_frame,
    parse_confidence_overrides,
)


class EvaluateMonsterYoloTests(unittest.TestCase):
    def test_box_iou_for_identical_boxes(self):
        self.assertEqual(box_iou([10, 10, 20, 30], [10, 10, 20, 30]), 1.0)

    def test_matching_is_class_aware_and_one_to_one(self):
        truth = [
            {"class": "zombie_mushroom", "box": [10, 10, 20, 20]},
            {"class": "thorn_mushroom", "box": [50, 10, 20, 20]},
        ]
        predictions = [
            {
                "class": "zombie_mushroom",
                "confidence": 0.9,
                "box": [10, 10, 20, 20],
            },
            {
                "class": "zombie_mushroom",
                "confidence": 0.8,
                "box": [10, 10, 20, 20],
            },
            {
                "class": "zombie_mushroom",
                "confidence": 0.7,
                "box": [50, 10, 20, 20],
            },
        ]

        matches, matched_truth, matched_predictions = match_frame(
            truth, predictions, 0.5
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matched_truth, {0})
        self.assertEqual(matched_predictions, {0})

    def test_parses_per_model_confidence(self):
        self.assertEqual(
            parse_confidence_overrides(["baseline=0.10", "candidate=0.75"]),
            {"baseline": 0.10, "candidate": 0.75},
        )

    def test_dataset_class_names_supports_yaml_id_mapping(self):
        self.assertEqual(dataset_class_names({"names": {0: "Stump"}}), ["stump"])

    def test_loads_single_class_ground_truth_from_dataset_names(self):
        with TemporaryDirectory() as directory:
            label = Path(directory) / "frame.txt"
            label.write_text("0 0.5 0.5 0.2 0.4\n", encoding="ascii")

            boxes = load_ground_truth(label, 100, 50, ["stump"])

        self.assertEqual(boxes[0]["class"], "stump")
        self.assertEqual(boxes[0]["box"], [40.0, 15.0, 20.0, 20.0])


if __name__ == "__main__":
    unittest.main()
