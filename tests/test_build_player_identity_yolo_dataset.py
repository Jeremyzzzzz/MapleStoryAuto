import unittest
from pathlib import Path

from tools.build_player_identity_yolo_dataset import (
    centered_box,
    select_temporal_splits,
    yolo_label,
)


class PlayerIdentityDatasetTests(unittest.TestCase):
    def test_temporal_split_keeps_middle_frames_as_holdout(self):
        frames = [Path(f"frame_{index:04d}.jpg") for index in range(74)]
        splits = select_temporal_splits(frames, train_count=40, val_count=10)

        self.assertEqual(splits["train"], frames[:40])
        self.assertEqual(splits["val"], frames[-10:])
        self.assertEqual(splits["holdout"], frames[40:64])
        self.assertFalse(set(splits["train"]) & set(splits["val"]))

    def test_centered_box_clamps_to_image(self):
        self.assertEqual(centered_box((5, 8), 20, 30, 100, 80), [0, 0, 20, 30])
        self.assertEqual(
            centered_box((99, 79), 20, 30, 100, 80), [80, 50, 20, 30]
        )

    def test_yolo_label_uses_normalized_center_and_size(self):
        label = yolo_label([20, 10, 40, 20], image_width=100, image_height=50)
        self.assertEqual(label, "0 0.400000 0.400000 0.400000 0.400000\n")


if __name__ == "__main__":
    unittest.main()
