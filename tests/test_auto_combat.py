import cv2
import numpy as np
import random
import unittest
from unittest.mock import patch

from tools.auto_combat import (
    AsyncMonsterDetector,
    ColorVerifiedMonsterDetector,
    CombatExecutor,
    CombatPolicy,
    MinimapNavigator,
    NameOcrPlayerDetector,
    ParamEditor,
    PauseController,
    PlayerLocator,
    RouteFollower,
    TerrainScanner,
    draw_attack_range,
    draw_detections,
    render_info_panel,
)


def make_player(x=100, y=100):
    return {
        "label": "PLAYER",
        "score": 0.9,
        "box": (x, y, 60, 80),
        "center": (x + 30, y + 40),
    }


def make_advisory(status, attack_ready=False, dodge_risk=False,
                  target_box=(200, 100, 40, 40), horizontal=100,
                  vertical=40):
    return {
        "status": status,
        "attack_ready": attack_ready,
        "dodge_risk": dodge_risk,
        "target_label": "STUMP",
        "target_box": target_box,
        "horizontal_distance_px": horizontal,
        "vertical_distance_px": vertical,
        "distance_px": (horizontal ** 2 + vertical ** 2) ** 0.5,
        "suggested_direction": None,
    }


def make_cfg():
    return {
        "key": {
            "directional_attack": "d",
            "add_hp": "1",
            "add_mp": "2",
            "jump": "space",
        },
        "health_monitor": {
            "add_hp_percent": 50,
            "add_mp_percent": 50,
            "add_hp_cooldown": 0.5,
            "add_mp_cooldown": 0.5,
        },
        "combat_advisory": {
            "attack_horizontal_px": 135,
            "attack_vertical_px": 70,
            "dodge_horizontal_px": 90,
            "dodge_vertical_px": 60,
            "immediate_danger_px": 42,
        },
        "auto_combat": {
            "attack_cooldown_min": 0.08,
            "attack_cooldown_max": 0.22,
            "patrol_enabled": True,
            "patrol_min_seconds": 1.0,
            "patrol_max_seconds": 2.0,
            "attack_retry_limit": 4,
            "attack_ignore_seconds": 2.0,
            "jump_cooldown": 1.5,
            "name_bottom_offset": 30,
        },
        "ui_coords": {"ui_y_start": 687},
        "perception_overlay": {"player_box_size": [70, 90]},
        "attack_profiles": {
            "default": {"horizontal": 135, "vertical": 70, "keep_distance": 60},
        },
    }


class MockOcrEngine:
    """Stand-in for rapidocr.RapidOCR(). Returns a (boxes, _) tuple."""

    def __init__(self, results):
        self.results = results

    def __call__(self, image):
        return (self.results, None)


class NameOcrPlayerDetectorTests(unittest.TestCase):
    def _cfg(self):
        cfg = make_cfg()
        return cfg

    def test_finds_player_by_name(self):
        cfg = self._cfg()
        # rapidocr returns a flat list of [box_points, text, confidence].
        ocr_results = [
            [
                [[3.0, 8.0], [102.0, 9.0], [102.0, 51.0], [3.0, 50.0]],
                "超团甜",
                0.99,
            ]
        ]
        detector = NameOcrPlayerDetector(
            cfg, "超团甜", ocr_engine=MockOcrEngine(ocr_results)
        )
        # Small frame: no top-crop and no downscale are triggered.
        frame = np.zeros((200, 800, 3), dtype=np.uint8)
        result = detector.detect(frame)

        self.assertIsNotNone(result)
        self.assertEqual(result["label"], "PLAYER")
        # Name box bottom is y=51; body center is 30px below the name bottom.
        self.assertEqual(result["center"], (52, 81))

    def test_returns_none_when_name_not_found(self):
        cfg = self._cfg()
        ocr_results = [
            [[[0, 0], [50, 0], [50, 20], [0, 20]], "其他玩家", 0.95]
        ]
        detector = NameOcrPlayerDetector(
            cfg, "超团甜", ocr_engine=MockOcrEngine(ocr_results)
        )
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        self.assertIsNone(detector.detect(frame))

    def test_returns_none_when_confidence_too_low(self):
        cfg = self._cfg()
        ocr_results = [
            [[[0, 0], [50, 0], [50, 20], [0, 20]], "超团甜", 0.10]
        ]
        detector = NameOcrPlayerDetector(
            cfg, "超团甜", confidence=0.5, ocr_engine=MockOcrEngine(ocr_results)
        )
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        self.assertIsNone(detector.detect(frame))

    def test_returns_none_on_empty_results(self):
        cfg = self._cfg()
        detector = NameOcrPlayerDetector(
            cfg, "超团甜", ocr_engine=MockOcrEngine([])
        )
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        self.assertIsNone(detector.detect(frame))

    def test_falls_back_to_title_when_name_missing(self):
        """When the tiny nametag is unreadable, the title badge below it
        ("新手冒险家勋章") still locates the player."""
        cfg = self._cfg()
        ocr_results = [
            [
                [[600.0, 430.0], [731.0, 430.0], [731.0, 446.0], [600.0, 446.0]],
                "新手冒险家勋章",
                0.92,
            ]
        ]
        detector = NameOcrPlayerDetector(
            cfg, "超团甜", ocr_engine=MockOcrEngine(ocr_results)
        )
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        result = detector.detect(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "title")
        # y_offset=68 (687*0.1), scale=1.0, title_bottom_offset=22:
        # x=round((600+731)/2)=666, y=446+22+68=536.
        self.assertEqual(result["center"], (666, 536))

    def test_picks_highest_confidence_match(self):
        cfg = self._cfg()
        ocr_results = [
            [[[0, 0], [50, 0], [50, 20], [0, 20]], "超团甜", 0.60],
            [[[100, 0], [200, 0], [200, 20], [100, 20]], "超团甜", 0.90],
        ]
        detector = NameOcrPlayerDetector(
            cfg, "超团甜", ocr_engine=MockOcrEngine(ocr_results)
        )
        # Small frame: no top-crop / downscale are triggered.
        frame = np.zeros((200, 800, 3), dtype=np.uint8)
        result = detector.detect(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result["center"][0], 150)  # the second, higher-score box

    def test_scales_back_coordinates_on_large_frame(self):
        """On a very wide frame the top-crop + downscale must be undone so the
        reported player center matches the raw-frame coordinates."""
        cfg = self._cfg()
        ocr_results = [
            [
                [[3.0, 8.0], [102.0, 9.0], [102.0, 51.0], [3.0, 50.0]],
                "超团甜",
                0.99,
            ]
        ]
        detector = NameOcrPlayerDetector(
            cfg, "超团甜", ocr_engine=MockOcrEngine(ocr_results)
        )
        # Frame wider than 1500: crop (68px) + 0.6x downscale.
        frame = np.zeros((1136, 1942, 3), dtype=np.uint8)
        result = detector.detect(frame)

        self.assertIsNotNone(result)
        # ui_y_start=687 -> y_offset=68; width 1942>1500 -> scale=0.6.
        # x: 52.5/0.6 = 87.5 -> 88; y: 51/0.6 + 30 + 68 = 183.
        self.assertEqual(result["center"], (88, 183))
        self.assertEqual(result["nametag_box"], (5, 81, 165, 72))

    def test_crop_only_on_native_frame(self):
        """Native 1278x750 frames are only top-cropped (no downscale), so the
        small 33px-wide name tag stays readable for RapidOCR."""
        cfg = self._cfg()
        ocr_results = [
            [
                [[3.0, 8.0], [102.0, 9.0], [102.0, 51.0], [3.0, 50.0]],
                "超团甜",
                0.99,
            ]
        ]
        detector = NameOcrPlayerDetector(
            cfg, "超团甜", ocr_engine=MockOcrEngine(ocr_results)
        )
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        result = detector.detect(frame)

        self.assertIsNotNone(result)
        # Only the 10% top-crop applies (68px); scale stays 1.0.
        self.assertEqual(result["center"], (52, 149))
        self.assertEqual(result["nametag_box"], (3, 76, 99, 43))


class CombatPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = CombatPolicy(make_cfg())

    def test_returns_none_when_player_missed(self):
        command, reason = self.policy.decide(None, make_advisory("ATTACK READY"), 100, 100, 1.0)
        self.assertEqual(command, "none")
        self.assertEqual(reason, "player_missed")

    def test_attacks_right_when_target_is_right(self):
        player = make_player(x=100)
        advisory = make_advisory("ATTACK READY", attack_ready=True, target_box=(200, 100, 40, 40))
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "attack_right")

    def test_attacks_left_when_target_is_left(self):
        player = make_player(x=300)
        advisory = make_advisory("ATTACK READY", attack_ready=True, target_box=(250, 100, 40, 40))
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "attack_left")

    def test_respects_attack_cooldown(self):
        player = make_player(x=100)
        advisory = make_advisory("ATTACK READY", attack_ready=True, target_box=(200, 100, 40, 40))
        self.policy.decide(player, advisory, 100, 100, 1.0)
        # The next attack is scheduled in the random interval range; an attack
        # issued before that deadline must be refused.
        next_attack = self.policy.next_attack_time
        self.assertGreaterEqual(next_attack - 1.0, 0.08)
        self.assertLessEqual(next_attack - 1.0, 0.22)
        command, reason = self.policy.decide(player, advisory, 100, 100, next_attack - 0.01)
        self.assertEqual(command, "none")
        self.assertEqual(reason, "attack_cooldown")

    def test_patrol_hunt_keeps_walking_during_attack_cooldown(self):
        policy = CombatPolicy(make_cfg(), mode="patrol_hunt")
        policy.patrol_direction = "right"
        player = make_player(x=400, y=100)
        advisory = make_advisory(
            "ATTACK READY", attack_ready=True, target_box=(470, 100, 40, 40),
        )
        command, reason = policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "attack_right")
        command, reason = policy.decide(player, advisory, 100, 100, 1.01)
        self.assertEqual(command, "move_right")
        self.assertEqual(reason, "attack_cooldown_patrol")

    def test_patrol_hunt_jumps_out_of_pit_when_stalled(self):
        policy = CombatPolicy(make_cfg(), mode="patrol_hunt")
        policy.patrol_direction = "right"
        player = make_player(x=400, y=500)
        none_adv = make_advisory("NO TARGET", target_box=None)
        none_adv["target_box"] = None
        policy.decide(player, none_adv, 100, 100, 10.0)
        policy.decide(player, none_adv, 100, 100, 10.1)
        command, reason = policy.decide(player, none_adv, 100, 100, 11.2)
        self.assertEqual(command, "jump_right")
        self.assertEqual(reason, "jump_out_of_pit")

    def test_patrol_hunt_turns_after_repeated_pit_jumps(self):
        policy = CombatPolicy(make_cfg(), mode="patrol_hunt")
        policy.patrol_direction = "right"
        player = make_player(x=400, y=500)
        none_adv = make_advisory("NO TARGET", target_box=None)
        none_adv["target_box"] = None
        now = 10.0
        last_jump_reason = None
        command = None
        for _ in range(3):
            policy.decide(player, none_adv, 100, 100, now)
            policy.decide(player, none_adv, 100, 100, now + 0.1)
            command, last_jump_reason = policy.decide(
                player, none_adv, 100, 100, now + 1.2,
            )
            now += 3.0
        self.assertEqual(command, "jump_left")
        self.assertEqual(last_jump_reason, "patrol_stuck_turn")
        self.assertEqual(policy.patrol_direction, "left")

    def test_attack_interval_is_randomized(self):
        # Directly exercise the random delay generator: a fixed cadence would
        # produce a single value, randomization must yield several distinct
        # intervals within [0.08, 0.22].
        delays = {round(self.policy._next_attack_delay(), 3) for _ in range(100)}
        self.assertGreater(len(delays), 1)
        for delay in delays:
            self.assertGreaterEqual(delay, 0.08)
            self.assertLessEqual(delay, 0.22)

    def test_gives_up_on_undying_target_and_patrols(self):
        # A drop item is detected as a monster; after several attacks it never
        # dies, so the policy must ignore it and keep patrolling instead of
        # standing in place.
        player = make_player(x=100)
        advisory = make_advisory("ATTACK READY", attack_ready=True, target_box=(190, 100, 40, 40))
        for i in range(4):
            command, reason = self.policy.decide(player, advisory, 100, 100, 10.0 + i * 2.0)
            self.assertEqual(command, "attack_right")
        # 4th attack at t=16 sets ignore_until = 16 + 2 = 18; query inside the
        # ignore window -> patrol instead of attacking again.
        command, reason = self.policy.decide(player, advisory, 100, 100, 17.0)
        self.assertIn(command, ("move_left", "move_right"))
        self.assertEqual(reason, "patrol")

    def test_moving_target_never_counted_as_drop(self):
        # A real monster walks toward the player (staying beyond keep_distance
        # and on the right side); the attack counter must keep resetting so it
        # is never ignored.
        policy = CombatPolicy(make_cfg())
        # target center: 260 -> 200 (player center 130, min engage 60)
        for i in range(6):
            x = 240 - i * 12
            player = make_player(x=100)
            advisory = make_advisory(
                "ATTACK READY", attack_ready=True,
                target_box=(x, 100, 40, 40),
            )
            command, reason = policy.decide(player, advisory, 100, 100, 10.0 + i * 2.0)
            self.assertEqual(command, "attack_right")
        self.assertFalse(policy._is_target_ignored((180, 120), 100.0))

    def test_patrols_when_no_target(self):
        player = make_player(x=100)
        advisory = make_advisory("NO TARGET", target_box=None)
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertIn(command, ("move_left", "move_right"))
        self.assertEqual(reason, "patrol")

    def test_patrol_changes_direction_after_deadline(self):
        player = make_player(x=100)
        advisory = make_advisory("NO TARGET", target_box=None)
        command, reason = self.policy.decide(player, advisory, 100, 100, 10.0)
        first_direction = command
        # Before the deadline, keep walking the same direction.
        command, _ = self.policy.decide(player, advisory, 100, 100, 10.5)
        self.assertEqual(command, first_direction)
        # After the deadline, direction may change (or at least re-roll).
        command, _ = self.policy.decide(player, advisory, 100, 100, 12.1)
        self.assertIn(command, ("move_left", "move_right"))

    def test_approaches_when_not_in_attack_range(self):
        player = make_player(x=100)
        advisory = make_advisory("TRACKING", attack_ready=False, target_box=(400, 100, 40, 40))
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "move_right")

    def test_attacks_instead_of_dodging_when_target_in_range(self):
        # Target is 90px away (within attack range 135, above keep_distance 60):
        # attack should win over dodging.
        player = make_player(x=100)
        advisory = make_advisory(
            "DODGE RISK", attack_ready=True, dodge_risk=True,
            target_box=(180, 100, 40, 40),
        )
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "attack_right")

    def test_jumps_when_target_too_high_to_attack(self):
        # Target is horizontally close but vertically out of attack range.
        player = make_player(x=100)
        advisory = make_advisory(
            "DODGE RISK", attack_ready=False, dodge_risk=True,
            target_box=(130, 220, 40, 40),
        )
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "jump")

    def test_attacks_when_target_right_next_to_player(self):
        # A monster standing right next to the mage is still a valid target
        # for a ranged spell: attack must fire instead of backing off.
        player = make_player(x=100)
        advisory = make_advisory(
            "DODGE RISK", attack_ready=True, dodge_risk=True,
            target_box=(120, 100, 40, 40),  # center (140,120): 10px from player
        )
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "attack_right")

    def test_backs_off_between_attacks_when_too_close(self):
        # After an attack, while cooling down with the monster still right next
        # to us, back off to ranged comfort.
        player = make_player(x=100)
        advisory = make_advisory(
            "DODGE RISK", attack_ready=True, dodge_risk=True,
            target_box=(120, 100, 40, 40),
        )
        self.policy.decide(player, advisory, 100, 100, 1.0)  # attack
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.05)
        self.assertEqual(command, "move_left")
        self.assertEqual(reason, "keep_distance")

    def test_attacks_when_target_at_comfortable_range(self):
        # 90px away (within range 135, above keep_distance 60): attack.
        player = make_player(x=100)
        advisory = make_advisory(
            "ATTACK READY", attack_ready=True, dodge_risk=False,
            target_box=(180, 100, 40, 40),  # center (200,120): 70px away
        )
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "attack_right")

    def test_jump_has_cooldown(self):
        # After a jump, a second jump must wait for the cooldown; the policy
        # backs off horizontally instead of hopping repeatedly.
        player = make_player(x=100)
        advisory = make_advisory(
            "DODGE RISK", attack_ready=False, dodge_risk=True,
            target_box=(130, 220, 40, 40),
        )
        self.policy.decide(player, advisory, 100, 100, 1.0)
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.2)
        self.assertNotEqual(command, "jump")
        self.assertTrue(command.startswith(("move_", "dodge_")))

    def test_uses_per_character_attack_range(self):
        # A character profile with a longer horizontal range attacks targets
        # that the default range cannot reach.
        cfg = make_cfg()
        cfg["attack_profiles"]["超团甜"] = {"horizontal": 260, "vertical": 70}
        policy = CombatPolicy(cfg, player_name="超团甜")
        player = make_player(x=100)
        # Target center at x=360 (230px away): out of default 135 range, in range.
        advisory = make_advisory(
            "TRACKING", attack_ready=False, dodge_risk=False,
            target_box=(340, 100, 40, 40),
        )
        command, reason = policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "attack_right")

    def test_default_profile_used_when_name_unknown(self):
        # Unknown character falls back to the default range: 230px is out of
        # reach, so the policy approaches instead of attacking.
        cfg = make_cfg()
        policy = CombatPolicy(cfg, player_name="未知名")
        player = make_player(x=100)
        advisory = make_advisory(
            "TRACKING", attack_ready=False, dodge_risk=False,
            target_box=(340, 100, 40, 40),
        )
        command, reason = policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "move_right")

    def test_approaches_when_target_near_but_out_of_range(self):
        # 190px away, not in attack range and not imminent: approach.
        player = make_player(x=100)
        advisory = make_advisory(
            "TRACKING", attack_ready=False, dodge_risk=False,
            target_box=(300, 100, 40, 40),
        )
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertEqual(command, "move_right")

    def test_patrols_when_no_target_box(self):
        player = make_player(x=100)
        advisory = make_advisory("TRACKING", attack_ready=False, target_box=None)
        command, reason = self.policy.decide(player, advisory, 100, 100, 1.0)
        self.assertIn(command, ("move_left", "move_right"))
        self.assertEqual(reason, "patrol")

    def test_patrol_can_be_disabled(self):
        cfg = make_cfg()
        cfg["auto_combat"]["patrol_enabled"] = False
        policy = CombatPolicy(cfg)
        player = make_player(x=100)
        command, reason = policy.decide(player, make_advisory("NO TARGET", target_box=None), 100, 100, 1.0)
        self.assertEqual(command, "none")
        self.assertEqual(reason, "no target")


class PlayerLocatorTests(unittest.TestCase):
    def _player(self, x):
        return {
            "label": "PLAYER",
            "score": 0.9,
            "box": (x, 100, 60, 80),
            "center": (x + 30, 140),
        }

    def test_uses_ocr_when_template_misses(self):
        class Tpl:
            def detect(self, frame):
                return None

        class Ocr:
            def detect(self, frame):
                return self._p
            _p = self._player(300)

        locator = PlayerLocator(Tpl(), Ocr(), refresh_frames=2)
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        try:
            # Simulate the async OCR thread publishing a result.
            locator._ocr_result = locator.ocr_detector._p
            locator._ocr_stamp = 1.0
            player = locator.detect(frame, 1.1)
            self.assertIsNotNone(player)
            # detect shifts the whole visual up 35px (player box + attack box).
            self.assertEqual(player["center"], (330, 140 - 35))
        finally:
            locator.stop()

    def test_stale_position_expires(self):
        class Tpl:
            def detect(self, frame):
                return None

        class Ocr:
            def detect(self, frame):
                return None

        locator = PlayerLocator(Tpl(), Ocr(), refresh_frames=1, sticky_seconds=0.5)
        locator.last_player = self._player(100)
        locator.last_seen = 1.0
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        try:
            self.assertIsNotNone(locator.detect(frame, 1.3))  # within 0.5s sticky
            self.assertIsNone(locator.detect(frame, 2.0))     # expired
        finally:
            locator.stop()

    def test_nametag_fast_match_after_seed(self):
        """After OCR seeds a character-sprite template, per-frame matching
        follows the player in real time; a large jump is rejected."""
        class Ocr:
            def detect(self, target):
                return {
                    "label": "PLAYER", "score": 0.99,
                    "box": (494, 310, 70, 90),
                    "nametag_box": (500, 300, 58, 25),
                    "center": (529, 355),
                }

        locator = PlayerLocator(None, Ocr())
        rng = np.random.default_rng(1)
        # Textured background; a bright block fills only part of the sprite
        # box (a pure-color template breaks CCOEFF normalization).
        frame1 = rng.integers(0, 30, (750, 1278, 3), dtype=np.uint8)
        frame1[322:372, 502:542] = 200  # character sprite block
        try:
            locator._ocr_result = {
                "label": "PLAYER", "score": 0.99,
                "box": (494, 310, 70, 90),
                "nametag_box": (500, 300, 58, 25),
                "center": (529, 355),
            }
            locator._ocr_stamp = 1.0
            p0 = locator.detect(frame1, 1.1)
            self.assertIsNotNone(p0)
            self.assertEqual(len(locator.tag_templates), 1)

            # Player moved 30px right (background continuous): follow in real
            # time via the sprite template.
            frame2 = cv2.warpAffine(
                frame1, np.float32([[1, 0, 30], [0, 1, 0]]), (1278, 750)
            )
            p1 = locator.detect(frame2, 1.2)
            self.assertIsNotNone(p1)
            # Nametag-text full-frame match (name_track) is the primary fast
            # path; the sprite template (tag_track) is the secondary fallback.
            self.assertIn(p1["method"], ("name_track", "tag_track"))
            # Follows the 30px move (synthetic random-texture frames shift the
            # match by a few px due to interpolation; real sharp nametag text
            # matches exactly): expect (559, 355) within +-7px.
            self.assertLess(abs(p1["center"][0] - 559), 8)
            # detect lifts the whole visual 35px, so the expected y shifts up.
            self.assertLess(abs(p1["center"][1] - (355 - 35)), 8)

            # Teleport far away: rejected, no tag_track result.
            frame3 = cv2.warpAffine(
                frame1, np.float32([[1, 0, 200], [0, 1, 0]]), (1278, 750)
            )
            p2 = locator.detect(frame3, 1.3)
            self.assertNotEqual(p2.get("method"), "tag_track")
        finally:
            locator.stop()

    @unittest.skip("Blue-weapon localization removed: fishing-net blue collides with rock crevices and decoration icons; PlayerLocator now uses track + OCR only.")
    def test_blue_weapon_localization(self):
        """The blue fishing-net weapon is located by HSV color every frame;
        the weapon->player offset learned at seed time is applied."""
        class Tpl:
            def detect(self, frame):
                return None

        class Ocr:
            def detect(self, frame, near_center=None):
                return None

        locator = PlayerLocator(Tpl(), Ocr(), refresh_frames=1, sticky_seconds=0.2)
        # Seed: player center (500,500), blue weapon block at (528,490)-(552,510).
        frame1 = np.zeros((750, 1278, 3), dtype=np.uint8)
        frame1[490:510, 528:552] = (255, 0, 0)  # pure blue
        locator._seed_weapon(frame1, (500, 500))
        self.assertIsNotNone(locator.weapon_offset)
        try:
            # Player moved: weapon block now at (548,490)-(572,510) (+20px x).
            frame2 = np.zeros((750, 1278, 3), dtype=np.uint8)
            frame2[490:510, 548:572] = (255, 0, 0)
            locator.last_player = {"center": (500, 500), "box": (465, 455, 70, 90)}
            locator.last_seen = 1.0
            p = locator.detect(frame2, 1.1)
            self.assertIsNotNone(p)
            self.assertEqual(p["method"], "weapon")
            # weapon center (560,500) + offset (-40,0) = (520,500).
            self.assertEqual(p["center"], (520, 500))
        finally:
            locator.stop()

    def test_ocr_on_frame_local_crop_restores_coords(self):
        """Local-crop OCR must map its coordinates back to the full frame."""
        class Ocr:
            def detect(self, target, near_center=None):
                return {
                    "label": "PLAYER",
                    "score": 0.99,
                    "box": (30, 40, 70, 90),
                    "nametag_box": (50, 10, 31, 12),
                    "center": (65, 60),
                }

        locator = PlayerLocator(None, Ocr())
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        # Crop around (900, 500): x0=640, y0=360.
        player, new_center = locator._ocr_on_frame(frame, (900, 500))
        self.assertIsNotNone(player)
        self.assertEqual(player["center"], (705, 420))
        self.assertEqual(player["box"], (670, 400, 70, 90))
        self.assertEqual(player["nametag_box"], (690, 370, 31, 12))
        self.assertEqual(new_center, (705, 420))

    def test_ocr_on_frame_full_keeps_coords(self):
        """Without a local center the full frame is OCRed with no offset."""
        class Ocr:
            def detect(self, target, near_center=None):
                return {
                    "label": "PLAYER",
                    "score": 0.99,
                    "box": (100, 100, 70, 90),
                    "nametag_box": (110, 80, 31, 12),
                    "center": (135, 160),
                }

        locator = PlayerLocator(None, Ocr())
        frame = np.zeros((750, 1278, 3), dtype=np.uint8)
        player, new_center = locator._ocr_on_frame(frame, None)
        self.assertIsNotNone(player)
        self.assertEqual(player["center"], (135, 160))
        self.assertEqual(new_center, (135, 160))


class PauseControllerTests(unittest.TestCase):
    def test_toggle_switches_pause_state(self):
        controller = PauseController()
        self.assertFalse(controller.paused)
        self.assertTrue(controller.toggle())
        self.assertTrue(controller.paused)
        self.assertFalse(controller.toggle())
        self.assertFalse(controller.paused)

    def test_quit_request_is_sticky(self):
        controller = PauseController()
        self.assertFalse(controller.is_quit_requested())
        controller.request_quit()
        self.assertTrue(controller.is_quit_requested())


class AttackRangeRenderTests(unittest.TestCase):
    def test_draw_attack_range_renders_around_player(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        player = {
            "label": "PLAYER",
            "center": (200, 150),
            "box": (170, 110, 60, 80),
        }
        policy = CombatPolicy(make_cfg())
        draw_attack_range(frame, player, policy)
        # The attack range box should be drawn (yellow pixels present).
        self.assertTrue(np.any(frame[:, :, 1] > 0))  # green channel of yellow

    def test_draw_attack_range_safe_without_player(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        draw_attack_range(frame, None, None)  # must not raise

    def test_draw_detections_renders_player_and_monsters(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        player = {"label": "PLAYER", "box": (50, 50, 60, 80), "score": 0.9}
        monsters = [
            {"label": "STUMP", "box": (200, 100, 40, 40), "score": 0.8, "method": "yolo"},
        ]
        advisory = {
            "status": "ATTACK READY", "attack_ready": True, "dodge_risk": False,
            "target_box": (200, 100, 40, 40),
        }
        draw_detections(frame, player, monsters, advisory)  # must not raise
        self.assertTrue(np.any(frame[:, :, 2] > 0))  # red/player channel drawn

    def test_render_info_panel_builds_wider_frame(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        policy = CombatPolicy(make_cfg())
        executor = CombatExecutor(make_cfg(), dry_run=True)
        combined = render_info_panel(
            frame,
            None, [], (100.0, 50.0, 10.0), None, policy, executor,
            "none", "no_target", 10, 9.5, False, "超团甜",
        )
        self.assertGreater(combined.shape[1], frame.shape[1])  # panel appended


class ColorVerifyTests(unittest.TestCase):
    def _cfg(self):
        return {
            "combat_advisory": {
                "attack_horizontal_px": 135,
                "attack_vertical_px": 70,
                "dodge_horizontal_px": 90,
                "dodge_vertical_px": 60,
                "immediate_danger_px": 42,
            },
            "auto_combat": {},
            "attack_profiles": {"default": {"horizontal": 135, "vertical": 70}},
            "ui_coords": {"ui_y_start": 687},
            "perception_overlay": {"player_box_size": [70, 90]},
        }

    def test_keeps_red_snail_with_red_pixels(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:90, 50:90] = (0, 0, 220)  # red block
        class FakeYolo:
            ui_y_start = 200
            def detect(self, frame, player):
                return [{
                    "label": "RED SNAIL",
                    "box": (50, 50, 40, 40),
                    "score": 0.5,
                    "method": "yolo",
                }]
        wrapper = ColorVerifiedMonsterDetector(FakeYolo())
        kept = wrapper.detect(frame, None)
        self.assertEqual(len(kept), 1)

    def test_drops_red_snail_without_red_pixels(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:90, 50:90] = (200, 200, 0)  # yellow, not red
        class FakeYolo:
            ui_y_start = 200
            def detect(self, frame, player):
                return [{
                    "label": "RED SNAIL",
                    "box": (50, 50, 40, 40),
                    "score": 0.5,
                    "method": "yolo",
                }]
        wrapper = ColorVerifiedMonsterDetector(FakeYolo())
        kept = wrapper.detect(frame, None)
        self.assertEqual(len(kept), 0)


class MinimapNavigatorTests(unittest.TestCase):
    def _cfg(self):
        return {
            "minimap": {
                "region": [0, 0, 100, 80],
                "player_color": (0, 255, 255),
                "player_color_tolerance": [160, 75, 75],
                "player_min_pixels": 4,
                "player_max_pixels": 80,
                "lookahead_px": 10,
                "step_px": 5,
                "rope_look_px": 40,
            },
        }

    def _frame_with_player_and_ground(self, player=(45, 40)):
        """Frame with dark pit background, yellow ground band right of the
        player (with a small gap so the marker stays a separate component),
        and the yellow player marker dot itself."""
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:] = (10, 10, 10)                    # pit everywhere
        frame[:, player[0] + 10:100] = (20, 130, 130)  # yellow ground, gap 10px
        cv2.circle(frame, player, 2, (0, 255, 255), -1)  # player marker
        return frame

    def test_finds_player_marker(self):
        nav = MinimapNavigator(self._cfg())
        frame = self._frame_with_player_and_ground(player=(45, 40))
        result = nav.scan(frame, 0.0)
        self.assertIsNotNone(result["player"])
        px, py = result["player"]
        self.assertLess(abs(px - 45), 6)
        self.assertLess(abs(py - 40), 6)

    def test_navigates_toward_ground(self):
        # Ground exists immediately right of the player, left is pit: walk
        # right (never into the void).
        nav = MinimapNavigator(self._cfg())
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:] = (10, 10, 10)
        frame[:, 45:100] = (20, 130, 130)   # ground from player x to the right
        cv2.circle(frame, (45, 40), 2, (0, 255, 255), -1)
        result = nav.scan(frame, 0.0)
        cmd, reason = nav.navigate(result, result["player"])
        self.assertEqual(cmd, "move_right")
        self.assertEqual(reason, "minimap_no_ground_left")

    def test_ground_on_both_sides_defers_to_patrol(self):
        # Ground on both sides: the navigator must NOT pick a direction (that
        # would make the character walk endlessly on a scrolling minimap).
        # It returns None so the normal patrol alternates direction.
        nav = MinimapNavigator(self._cfg())
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:] = (10, 10, 10)
        frame[:, 30:70] = (20, 130, 130)    # ground both sides of player
        cv2.circle(frame, (50, 40), 2, (0, 255, 255), -1)
        result = nav.scan(frame, 0.0)
        self.assertIsNone(nav.navigate(result, result["player"]))

    def test_no_navigation_when_surrounded_by_pit(self):
        nav = MinimapNavigator(self._cfg())
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:] = (10, 10, 10)  # all pit
        cv2.circle(frame, (50, 40), 2, (0, 255, 255), -1)
        result = nav.scan(frame, 0.0)
        cmd = nav.navigate(result, result["player"])
        self.assertIsNone(cmd)

    def test_jumps_across_gap(self):
        # Player at x=45 on a small platform (40..48); ground resumes at x=55
        # after a 7px pit gap. lookahead(10) reaches x=55 -> jump right.
        nav = MinimapNavigator(self._cfg())
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:] = (10, 10, 10)
        frame[:, 40:48] = (20, 130, 130)   # small platform under the player
        frame[:, 55:100] = (20, 130, 130)  # ground resumes after the gap
        cv2.circle(frame, (45, 40), 2, (0, 255, 255), -1)
        result = nav.scan(frame, 0.0)
        cmd, reason = nav.navigate(result, result["player"])
        self.assertEqual(cmd, "jump_right")
        self.assertEqual(reason, "minimap_gap")

    def test_climbs_rope_above(self):
        # Player on a platform at y=48; a pit channel above leads to a ground
        # platform at the top (the rope shaft). -> climb_up.
        nav = MinimapNavigator(self._cfg())
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[:] = (10, 10, 10)                    # pit everywhere
        frame[0:12, 40:60] = (20, 130, 130)         # top platform (rope end)
        frame[44:80, 40:60] = (20, 130, 130)        # player's platform
        cv2.circle(frame, (50, 48), 2, (0, 255, 255), -1)
        result = nav.scan(frame, 0.0)
        self.assertIsNotNone(result["player"])
        cmd, reason = nav.navigate(result, result["player"])
        self.assertEqual(cmd, "climb_up")
        self.assertEqual(reason, "minimap_rope")


class RouteFollowerTests(unittest.TestCase):
    def _cfg(self):
        return {
            "route": {
                "search_range": 10,
                "color_code": {
                    "255,0,0": "left none none",      # red: walk left
                    "0,0,255": "right none none",     # blue: walk right
                    "255,127,0": "left none jump",    # orange: jump left
                    "0,255,255": "right none jump",   # cyan: jump right
                    "255,255,0": "none none goal",    # yellow: goal
                },
                "color_code_up_down": {
                    "127,127,127": "none up none",    # gray: climb up
                    "255,255,127": "none down none",  # light yellow: descend
                },
            },
        }

    def _route_with_red_at(self, x, y):
        route = np.zeros((50, 60, 3), dtype=np.uint8)
        route[y, x] = (255, 0, 0)  # BGR red = walk left
        return route

    def test_decides_walk_right_from_blue_pixel(self):
        follower = RouteFollower(self._cfg())
        route = np.zeros((50, 60, 3), dtype=np.uint8)
        # Config "0,0,255" (RGB blue) is 'right none none'; the follower
        # stores keys in BGR, so the route pixel is BGR(255,0,0).
        route[25, 30] = (255, 0, 0)
        follower.img_routes = [route]
        cmd, reason = follower.decide((20, 25))  # player left of the pixel
        self.assertEqual(cmd, "move_right")
        self.assertIn("route_", reason)

    def test_decides_climb_up_from_gray_pixel(self):
        follower = RouteFollower(self._cfg())
        route = np.zeros((50, 60, 3), dtype=np.uint8)
        route[25, 30] = (127, 127, 127)  # gray = climb up
        follower.img_routes = [route]
        cmd, reason = follower.decide((30, 30))
        self.assertEqual(cmd, "climb_up")

    def test_goal_switches_route(self):
        follower = RouteFollower(self._cfg())
        r1 = self._route_with_red_at(30, 25)
        r2 = self._route_with_red_at(40, 25)
        # Config "255,255,0" (RGB yellow) = goal; stored as BGR(0,255,255).
        r2[25, 20] = (0, 255, 255)
        follower.img_routes = [r1, r2]
        follower.idx_route = 1
        cmd, reason = follower.decide((20, 25))
        self.assertEqual(cmd, "none")
        self.assertEqual(reason, "route_switch")
        self.assertEqual(follower.idx_route, 0)

    def test_no_route_returns_none(self):
        follower = RouteFollower(self._cfg())
        follower.img_routes = []
        self.assertIsNone(follower.decide((20, 25)))


class TerrainScannerTests(unittest.TestCase):
    def _make_frame(self, sky_color=(170, 200, 230), rope_color=(35, 50, 60)):
        """Build a synthetic frame with sky background and a vertical rope.

        BGR ordering: sky is light blue-grey, rope is warm brown. A real rope
        is ~14px wide, single connected component, with sparse brown pixels
        (the weave): adjacent columns overlap so the strip stays connected
        while rows are sparse, giving density ~0.3. We mimic that with tall
        bars that stagger 2px per column (18px overlap keeps them connected).
        """
        h, w = 200, 100
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = sky_color
        for i, x in enumerate(range(43, 57)):   # 14 columns
            y = 10 + i * 3                       # stagger 3px per column
            frame[y:y + 26, x] = rope_color      # 26px tall bar
        return frame

    def _executor_cfg(self):
        return {
            "key": {"directional_attack": "d", "jump": "space", "up": "up", "down": "down"},
            "health_monitor": {
                "add_hp_cooldown": 0.5, "add_mp_cooldown": 0.5,
                "add_hp_percent": 20, "add_mp_percent": 20,
            },
        }

    def test_detects_clear_rope(self):
        frame = self._make_frame()
        scanner = TerrainScanner()
        result = scanner.scan(frame, 0.0, (50, 190))
        self.assertEqual(len(result["ropes"]), 1)
        x, y_top, y_bot, xc = result["ropes"][0]
        self.assertLess(abs(xc - 50), 5)
        self.assertGreater(y_bot - y_top, 50)

    def test_ignores_absent_rope(self):
        frame = np.full((200, 100, 3), 200, dtype=np.uint8)  # bright sky
        scanner = TerrainScanner()
        result = scanner.scan(frame, 0.0, (50, 190))
        self.assertEqual(result["ropes"], [])

    def test_executor_supports_climb_up(self):
        executor = CombatExecutor(self._executor_cfg(), dry_run=True)
        executor.execute("climb_up", "test")
        self.assertEqual(executor.held_vert, "up")
        self.assertIn("down_up", executor.pressed_keys)
        # Calling again should be a no-op (already held).
        executor.execute("climb_up", "test")
        self.assertEqual(executor.pressed_keys.count("down_up"), 1)

    def test_executor_releases_vert(self):
        executor = CombatExecutor(self._executor_cfg(), dry_run=True)
        executor.execute("climb_up", "test")
        # Move command cancels vertical.
        executor.execute("move_left", "test")
        self.assertEqual(executor.held_vert, None)
        self.assertIn("up_up", executor.pressed_keys)

    def test_policy_climbs_when_rope_directly_above(self):
        policy = CombatPolicy(make_cfg())
        player = make_player(x=50, y=100)  # center (80, 140)
        # Rope at x=80, y_top=30, y_bot=80 -> bottom is 60px above the player
        # (out of reach) -> the policy must JUMP first, then climb.
        terrain = {"ropes": [(75, 30, 80, 80)], "drops": []}
        result = policy.terrain_decision(player, terrain)
        self.assertIsNotNone(result)
        cmd, reason = result
        self.assertEqual(cmd, "jump")
        self.assertEqual(reason, "jump_to_rope")
        # Within 0.3s after the jump, keep pressing up to grab the rope.
        policy._climb_jump_at = 0.0  # reset for a fresh look
        import time as _t
        policy._climb_jump_at = _t.time()
        result2 = policy.terrain_decision(player, terrain)
        cmd2, reason2 = result2
        self.assertEqual(cmd2, "climb_up")
        self.assertEqual(reason2, "climb_after_jump")
        # Rope bottom within reach (y_bot close to player) -> climb directly.
        terrain_near = {"ropes": [(75, 30, 130, 80)], "drops": []}
        policy._climb_jump_at = 0.0
        result3 = policy.terrain_decision(player, terrain_near)
        cmd3, reason3 = result3
        self.assertEqual(cmd3, "climb_up")
        self.assertIn(reason3, ("reach_rope", "climb_in_progress"))

    def test_policy_walks_toward_rope(self):
        policy = CombatPolicy(make_cfg())
        player = make_player(x=50, y=180)  # center (80, 220)
        # Rope bottom anchor y=200 is within ±100 of player y=220; center x=130.
        terrain = {"ropes": [(120, 150, 200, 130)], "drops": []}
        result = policy.terrain_decision(player, terrain)
        self.assertIsNotNone(result)
        cmd, reason = result
        self.assertEqual(cmd, "move_right")
        self.assertEqual(reason, "approach_rope")


class AsyncMonsterDetectorTests(unittest.TestCase):
    def test_get_latest_returns_recent_results(self):
        class FakeDetector:
            def detect(self, frame, player):
                return [self._m]
            _m = {"label": "STUMP", "box": (200, 100, 40, 40), "score": 0.9, "method": "test"}

        detector = AsyncMonsterDetector(FakeDetector(), max_age=0.8)
        detector._latest = [FakeDetector._m]
        detector._timestamp = 1.0
        result = detector.get_latest(1.4)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["box"][0], 200)

    def test_get_latest_expires_after_max_age(self):
        class FakeDetector:
            def detect(self, frame, player):
                return []

        detector = AsyncMonsterDetector(FakeDetector(), max_age=0.8)
        detector._latest = [{"label": "STUMP", "box": (200, 100, 40, 40), "score": 0.9, "method": "test"}]
        detector._timestamp = 1.0
        self.assertEqual(detector.get_latest(2.0), [])  # stale -> empty


class ParamEditorTests(unittest.TestCase):
    def test_types_value_and_applies_on_enter(self):
        editor = ParamEditor()
        state = {"value": 100.0}
        editor.add(
            ord("1"), "Range H", lambda: state["value"],
            lambda v: state.__setitem__("value", float(v)), 20,
        )
        self.assertTrue(editor.handle_key(ord("1")))  # selects field
        self.assertTrue(editor.handle_key(ord("5")))
        self.assertTrue(editor.handle_key(ord("0")))
        self.assertTrue(editor.handle_key(13))  # Enter
        self.assertEqual(state["value"], 50.0)

    def test_esc_cancels_edit(self):
        editor = ParamEditor()
        state = {"value": 100.0}
        editor.add(
            ord("2"), "Range V", lambda: state["value"],
            lambda v: state.__setitem__("value", float(v)), 10,
        )
        editor.handle_key(ord("2"))
        editor.handle_key(ord("7"))
        editor.handle_key(27)  # Esc
        self.assertEqual(state["value"], 100.0)


class CombatExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = CombatExecutor(make_cfg(), dry_run=True)

    def test_attack_turns_then_attacks(self):
        self.executor.execute("attack_left", "attack")
        self.assertIn("left", self.executor.pressed_keys)
        self.assertIn("d", self.executor.pressed_keys)

    def test_potion_takes_priority(self):
        self.executor.execute("attack_right", "attack", hp_percent=30, mp_percent=90, now=1.0)
        self.assertEqual(self.executor.pressed_keys, ["1"])

    def test_mp_potion_when_low(self):
        self.executor.execute("move_left", "approach", hp_percent=100, mp_percent=10, now=1.0)
        self.assertEqual(self.executor.pressed_keys, ["2"])

    def test_dry_run_records_counts(self):
        self.executor.execute("attack_right", "attack")
        self.assertEqual(self.executor.counts.get("attack_right", 0), 1)

    def test_movement_is_held_not_tapped(self):
        # Same move command twice should not re-send key events.
        self.executor.execute("move_left", "approach")
        self.executor.execute("move_left", "approach")
        self.assertEqual(self.executor.pressed_keys, ["down_left"])

    def test_move_switches_direction_releases_previous(self):
        self.executor.execute("move_left", "approach")
        self.executor.execute("move_right", "approach")
        self.executor.execute("move_right", "approach")
        self.assertEqual(
            self.executor.pressed_keys, ["down_left", "up_left", "down_right"]
        )

    def test_attack_releases_movement_key(self):
        self.executor.execute("move_left", "approach")
        self.executor.execute("attack_right", "attack")
        self.assertEqual(self.executor.pressed_keys, ["down_left", "up_left", "right", "d"])

    def test_patrol_hunt_attack_keeps_movement_key(self):
        executor = CombatExecutor(make_cfg(), dry_run=True, mode="patrol_hunt")
        executor.execute("move_right", "patrol")
        executor.execute("attack_right", "attack")
        self.assertEqual(executor.held_move, "right")
        self.assertNotIn("up_right", executor.pressed_keys)
        self.assertIn("d", executor.pressed_keys)

    def test_none_releases_movement_key(self):
        self.executor.execute("move_right", "approach")
        self.executor.execute("none", "no_target")
        self.assertEqual(self.executor.pressed_keys, ["down_right", "up_right"])

    def test_release_all_is_safe(self):
        self.executor.release_all()  # must not raise in dry-run

    @patch("tools.auto_combat.press_key")
    def test_non_dry_run_emits_keys(self, press_key):
        executor = CombatExecutor(make_cfg(), dry_run=False)
        executor.execute("attack_left", "attack")
        keys = [call.args[0] for call in press_key.call_args_list]
        self.assertIn("left", keys)
        self.assertIn("d", keys)
        executor.release_all()


if __name__ == "__main__":
    unittest.main()
