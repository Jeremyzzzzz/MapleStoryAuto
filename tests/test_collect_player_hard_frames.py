import unittest

import numpy as np

from tools.collect_player_hard_frames import frame_difference


class FrameCollectorTests(unittest.TestCase):
    def test_identical_frames_have_no_difference(self):
        frame = np.full((100, 200, 3), 40, dtype=np.uint8)

        self.assertEqual(frame_difference(frame, frame.copy()), 0.0)

    def test_changed_frame_has_positive_difference(self):
        first = np.zeros((100, 200, 3), dtype=np.uint8)
        second = first.copy()
        second[20:80, 60:140] = (0, 0, 255)

        self.assertGreater(frame_difference(first, second), 0.0)


if __name__ == "__main__":
    unittest.main()
