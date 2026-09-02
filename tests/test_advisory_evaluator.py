import unittest

from tools.live_perception_viewer import AdvisoryEvaluator


def detection(label, x, y, width=40, height=40):
    return {
        "label": label,
        "box": (x, y, width, height),
        "score": 0.9,
        "method": "test",
    }


class AdvisoryEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "combat_advisory": {
                "attack_horizontal_px": 135,
                "attack_vertical_px": 70,
                "dodge_horizontal_px": 90,
                "dodge_vertical_px": 60,
                "immediate_danger_px": 42,
                "approach_speed_px_s": 18,
                "track_match_px": 100,
            }
        }
        self.player = detection("PLAYER", 100, 100, 60, 80)

    def test_reports_attack_ready_for_nearby_target(self):
        evaluator = AdvisoryEvaluator(self.cfg)
        result = evaluator.evaluate(
            self.player,
            [detection("STUMP", 205, 120)],
            timestamp=1.0,
        )

        self.assertEqual(result["status"], "ATTACK READY")
        self.assertTrue(result["attack_ready"])
        self.assertFalse(result["dodge_risk"])

    def test_reports_approaching_dodge_risk_and_opposite_direction(self):
        evaluator = AdvisoryEvaluator(self.cfg)
        evaluator.evaluate(
            self.player,
            [detection("STUMP", 210, 120)],
            timestamp=1.0,
        )
        result = evaluator.evaluate(
            self.player,
            [detection("STUMP", 180, 120)],
            timestamp=2.0,
        )

        self.assertEqual(result["status"], "DODGE RISK")
        self.assertTrue(result["dodge_risk"])
        self.assertEqual(result["suggested_direction"], "LEFT")
        self.assertGreaterEqual(result["approach_speed_px_s"], 18)

    def test_does_not_warn_when_target_is_retreating(self):
        evaluator = AdvisoryEvaluator(self.cfg)
        evaluator.evaluate(
            self.player,
            [detection("BLUE SNAIL", 160, 120)],
            timestamp=1.0,
        )
        result = evaluator.evaluate(
            self.player,
            [detection("BLUE SNAIL", 190, 120)],
            timestamp=2.0,
        )

        self.assertFalse(result["dodge_risk"])
        self.assertIsNone(result["suggested_direction"])

    def test_camera_motion_pauses_and_clears_tracking(self):
        evaluator = AdvisoryEvaluator(self.cfg)
        evaluator.evaluate(
            self.player,
            [detection("RED SNAIL", 200, 120)],
            timestamp=1.0,
        )

        result = evaluator.evaluate(
            self.player,
            [detection("RED SNAIL", 170, 120)],
            timestamp=2.0,
            camera_motion=True,
        )

        self.assertEqual(result["status"], "PAUSED CAMERA")
        self.assertEqual(evaluator.previous_tracks, [])
        self.assertFalse(result["attack_ready"])
        self.assertFalse(result["dodge_risk"])

    def test_selects_nearest_target_by_total_distance(self):
        evaluator = AdvisoryEvaluator(self.cfg)
        result = evaluator.evaluate(
            self.player,
            [
                detection("STUMP", 620, 100),
                detection("RED SNAIL", 180, 250),
            ],
            timestamp=1.0,
        )

        self.assertEqual(result["target_label"], "RED SNAIL")


if __name__ == "__main__":
    unittest.main()
