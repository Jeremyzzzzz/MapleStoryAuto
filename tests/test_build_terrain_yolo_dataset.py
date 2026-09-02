import random
import unittest

import numpy as np

from tools.build_terrain_yolo_dataset import (
    CLASS_NAMES,
    SOURCE_BOXES,
    choose_crop,
    transform_sample,
)


class TerrainDatasetTests(unittest.TestCase):
    def test_has_three_requested_classes(self):
        self.assertEqual(CLASS_NAMES, ["ladder", "rope", "platform"])

    def test_reviewed_annotation_counts(self):
        counts = {
            class_name: sum(box[4] == class_name for box in SOURCE_BOXES)
            for class_name in CLASS_NAMES
        }

        self.assertEqual(counts, {"ladder": 2, "rope": 1, "platform": 5})

    def test_validation_crop_retains_selected_long_ladder(self):
        source = np.zeros((687, 1370, 3), dtype=np.uint8)
        target = SOURCE_BOXES[0]
        crop = choose_crop(target, 1370, 687, random.Random(53), True)
        image, boxes = transform_sample(
            source, SOURCE_BOXES, crop, random.Random(53), validation=True
        )

        self.assertEqual(image.shape, (540, 960, 3))
        self.assertIn("ladder", [box[4] for box in boxes])

    def test_validation_crop_retains_wide_platform(self):
        source = np.zeros((687, 1370, 3), dtype=np.uint8)
        target = SOURCE_BOXES[6]
        crop = choose_crop(target, 1370, 687, random.Random(53), True)
        _, boxes = transform_sample(
            source, SOURCE_BOXES, crop, random.Random(53), validation=True
        )

        self.assertIn("platform", [box[4] for box in boxes])


if __name__ == "__main__":
    unittest.main()
