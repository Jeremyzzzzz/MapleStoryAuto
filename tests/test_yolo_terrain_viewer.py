import unittest

import numpy as np

from tools.yolo_terrain_viewer import draw_detections, validate_model_classes


class TerrainViewerTests(unittest.TestCase):
    def test_model_must_have_exact_terrain_classes(self):
        validate_model_classes({0: "ladder", 1: "rope", 2: "platform"})
        with self.assertRaises(ValueError):
            validate_model_classes({0: "ladder", 1: "rope"})

    def test_draw_reports_class_counts(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        output, counts = draw_detections(
            frame,
            [
                {
                    "class": "rope",
                    "label": "ROPE",
                    "confidence": 0.9,
                    "box": [20, 30, 8, 50],
                    "color": (255, 170, 50),
                    "track_id": 1,
                }
            ],
            12.0,
        )
        self.assertEqual(counts, {"ladder": 0, "rope": 1, "platform": 0})
        self.assertGreater(int(np.count_nonzero(output)), 0)


if __name__ == "__main__":
    unittest.main()
