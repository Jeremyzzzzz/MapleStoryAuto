import random
import unittest

import numpy as np

from tools.build_seven_class_increment_dataset import (
    CLASS_NAMES,
    SOURCE_BOXES,
    choose_crop,
    transform_sample,
)


class SevenClassIncrementDatasetTests(unittest.TestCase):
    def test_preserves_existing_class_ids(self):
        self.assertEqual(
            CLASS_NAMES[:5],
            [
                "slime",
                "red_snail",
                "green_mushroom",
                "stump",
                "flower_mushroom",
            ],
        )
        self.assertEqual(CLASS_NAMES[5:], ["zombie_mushroom", "thorn_mushroom"])

    def test_source_contains_three_zombies_and_two_thorns(self):
        counts = {
            name: sum(box[4] == name for box in SOURCE_BOXES)
            for name in CLASS_NAMES[5:]
        }

        self.assertEqual(counts, {"zombie_mushroom": 3, "thorn_mushroom": 2})

    def test_validation_crop_keeps_selected_target(self):
        source = np.zeros((156, 778, 3), dtype=np.uint8)
        target = SOURCE_BOXES[3]
        crop = choose_crop(target, 778, 156, random.Random(7), True)
        _, boxes = transform_sample(
            source, SOURCE_BOXES, crop, random.Random(7), validation=True
        )

        self.assertIn("zombie_mushroom", [box[4] for box in boxes])


if __name__ == "__main__":
    unittest.main()
