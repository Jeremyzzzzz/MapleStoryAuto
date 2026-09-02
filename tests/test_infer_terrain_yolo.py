import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.infer_terrain_yolo import draw_detections, read_image, write_image


class TerrainInferenceTests(unittest.TestCase):
    def test_draw_detections_changes_image(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        output = draw_detections(
            image,
            [
                {
                    "class": "ladder",
                    "confidence": 0.91,
                    "box_xyxy": [10, 12, 35, 60],
                }
            ],
        )
        self.assertGreater(int(np.count_nonzero(output)), 0)

    def test_write_and_read_image_round_trip(self):
        image = np.full((12, 16, 3), 127, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            write_image(path, image)
            restored = read_image(path)
        self.assertEqual(restored.shape, image.shape)
        self.assertEqual(int(restored.mean()), 127)


if __name__ == "__main__":
    unittest.main()
