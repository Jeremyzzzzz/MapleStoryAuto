import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.engine.HealthMonitor import HealthMonitor
from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.utils.common import (
    get_player_location_on_minimap,
    load_yaml,
    override_cfg,
)


class MinimapPlayerDetectionTests(unittest.TestCase):
    def test_returns_none_when_player_color_is_absent(self):
        minimap = np.zeros((20, 20, 3), dtype=np.uint8)

        self.assertIsNone(
            get_player_location_on_minimap(
                minimap,
                minimap_player_color=(0, 255, 255),
                color_tolerance=(20, 20, 20),
            )
        )

    def test_selects_largest_matching_component(self):
        minimap = np.zeros((20, 20, 3), dtype=np.uint8)
        minimap[10:14, 5:9] = (20, 240, 250)
        minimap[2:4, 15:17] = (0, 255, 255)

        location = get_player_location_on_minimap(
            minimap,
            minimap_player_color=(0, 255, 255),
            color_tolerance=(30, 30, 30),
            min_pixels=4,
            max_pixels=30,
        )

        self.assertEqual(location, (6, 12))


class ConfiguredStatusBarTests(unittest.TestCase):
    def setUp(self):
        base = load_yaml("config/config_default.yaml")
        override = load_yaml("config/config_shanda_legacy.yaml")
        self.cfg = override_cfg(base, override)

    def test_reads_configured_hp_mp_exp_regions(self):
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        frame[731:744, 471:569] = (0, 0, 255)
        frame[731:744, 572:597] = (255, 0, 0)
        frame[731:744, 677:731] = (0, 255, 255)

        monitor = HealthMonitor(self.cfg, kb_controller=None)
        monitor.update_frame(frame)
        hp, mp, exp = monitor.get_hp_mp_exp_percent()

        self.assertAlmostEqual(hp, 100.0, places=2)
        self.assertAlmostEqual(mp, 25 / 98 * 100, places=2)
        self.assertAlmostEqual(exp, 50.0, places=2)
        self.assertEqual(
            monitor.loc_size_bars,
            [(471, 731, 98, 13), (572, 731, 98, 13), (677, 731, 108, 13)],
        )


class NametagStateTests(unittest.TestCase):
    def setUp(self):
        base = load_yaml("config/config_default.yaml")
        override = load_yaml("config/config_shanda_legacy.yaml")
        cfg = override_cfg(base, override)
        self.bot = MapleStoryAutoBot(
            SimpleNamespace(disable_viz=False, disable_control=True, is_ui=False)
        )
        self.assertEqual(self.bot.load_config(cfg), 0)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        self.bot.cfg["ui_coords"]["ui_y_start"] = 120
        self.bot.img_frame = frame
        self.bot.img_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bot.img_frame_debug = frame.copy()

    @patch(
        "src.engine.MapleStoryAutoLevelUp.find_pattern_sqdiff",
        return_value=((20, 60), 1.0, False),
    )
    def test_first_miss_returns_none(self, _):
        self.assertIsNone(self.bot.get_player_location_by_nametag())
        self.assertFalse(self.bot.is_player_location_valid)

    @patch(
        "src.engine.MapleStoryAutoLevelUp.find_pattern_sqdiff",
        return_value=((61, 83), 0.0, False),
    )
    def test_first_match_enables_player_location(self, _):
        location = self.bot.get_player_location_by_nametag()

        self.assertEqual(location, (40, 30))
        self.assertTrue(self.bot.is_player_location_valid)


if __name__ == "__main__":
    unittest.main()
