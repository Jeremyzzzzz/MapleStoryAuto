import unittest

import numpy as np

from tools.codex_monster_detector import CodexMonsterDetector
from tools.yolo_monster_viewer import (
    DetectionTracker,
    EntityCoordinateTracker,
)


class FakeCore:
    def detect(self, frame, gameplay_height):
        return [
            {
                "class": "thorn_mushroom",
                "label": "THORN MUSHROOM",
                "label_zh": "thorn_mushroom",
                "confidence": 0.9,
                "box": [20, 10, 40, 40],
                "color": (1, 2, 3),
            },
            {
                "class": "zombie_mushroom",
                "label": "ZOMBIE MUSHROOM",
                "label_zh": "zombie_mushroom",
                "confidence": 0.9,
                "box": [20, 110, 40, 40],
                "color": (4, 5, 6),
            },
        ]


def build_detector(same_level_only):
    detector = CodexMonsterDetector.__new__(CodexMonsterDetector)
    detector.core = FakeCore()
    detector.tracker = DetectionTracker(max_missed=0)
    detector.coord = EntityCoordinateTracker()
    detector.level_band = 60.0
    detector.same_level_only = same_level_only
    detector.ui_y_start = 180
    return detector


class CodexMonsterDetectorTests(unittest.TestCase):
    def test_marks_same_level_without_deleting_full_frame_detection(self):
        detector = build_detector(same_level_only=False)

        detections = detector.detect(
            np.zeros((200, 200, 3), dtype=np.uint8),
            player={"center": [40, 30]},
        )

        self.assertEqual(len(detections), 2)
        self.assertTrue(detections[0]["same_level"])
        self.assertFalse(detections[1]["same_level"])

    def test_same_level_filter_remains_explicit_opt_in(self):
        detector = build_detector(same_level_only=True)

        detections = detector.detect(
            np.zeros((200, 200, 3), dtype=np.uint8),
            player={"center": [40, 30]},
        )

        self.assertEqual(len(detections), 1)
        self.assertTrue(detections[0]["same_level"])


if __name__ == "__main__":
    unittest.main()
