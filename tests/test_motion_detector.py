import unittest

import numpy as np

from tools.live_perception_viewer import MotionDetector


class MotionDetectorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"ui_coords": {"ui_y_start": 300}}

    def test_detects_local_motion_after_warmup(self):
        detector = MotionDetector(
            self.cfg,
            threshold=10,
            min_area=20,
            max_area=2500,
            candidate_score=0.0,
        )
        first = np.zeros((320, 400, 3), dtype=np.uint8)

        self.assertEqual(detector.detect(first, None, []), [])
        detections = []
        for x_position in (280, 286, 292, 298):
            moved = first.copy()
            moved[145:170, x_position : x_position + 30] = (0, 0, 255)
            detections = detector.detect(moved, None, [])

        self.assertTrue(detections)
        x, y, width, height = detections[0]["box"]
        self.assertLess(x, 328)
        self.assertLessEqual(y, 145)
        self.assertGreater(x + width, 298)
        self.assertGreaterEqual(y + height, 170)

    def test_ignores_motion_inside_player_box(self):
        detector = MotionDetector(
            self.cfg,
            threshold=10,
            min_area=20,
            max_area=2500,
            candidate_score=0.0,
        )
        first = np.zeros((320, 400, 3), dtype=np.uint8)
        second = first.copy()
        second[145:170, 280:310] = (0, 0, 255)
        player = {"box": (270, 135, 55, 55)}

        detector.detect(first, player, [])

        self.assertEqual(detector.detect(second, player, []), [])
        self.assertEqual(detector.tracks, [])


if __name__ == "__main__":
    unittest.main()
