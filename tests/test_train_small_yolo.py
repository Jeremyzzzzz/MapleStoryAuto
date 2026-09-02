import unittest

import numpy as np

from tools.train_small_yolo import serializable_results


class FakeBoxMetrics:
    ap_class_index = np.array([1])
    nt_per_class = np.array([0, 2])
    p = np.array([0.9])
    r = np.array([1.0])
    ap50 = np.array([0.95])
    ap = np.array([0.75])


class FakeMetrics:
    results_dict = {"fitness": 0.75}
    speed = {"inference": 10.0}
    names = {0: "zombie_mushroom", 1: "thorn_mushroom"}
    nt_per_class = np.array([0, 2])
    box = FakeBoxMetrics()


class TrainSmallYoloTests(unittest.TestCase):
    def test_serializes_class_missing_from_validation_split(self):
        result = serializable_results(FakeMetrics())

        self.assertIsNone(
            result["per_class"]["zombie_mushroom"]["precision"]
        )
        self.assertEqual(
            result["per_class"]["thorn_mushroom"]["instances"], 2
        )
        self.assertEqual(
            result["per_class"]["thorn_mushroom"]["recall"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
