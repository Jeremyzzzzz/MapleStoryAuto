import unittest

from tools.offline_combat_simulator import (
    OfflineCombatEnvironment,
    RuleCombatPolicy,
    run_episode,
)


class OfflineCombatEnvironmentTests(unittest.TestCase):
    def test_attack_damages_nearby_monster(self):
        environment = OfflineCombatEnvironment(seed=1)
        target = environment.monsters[0]
        target.x = environment.player_x + 50

        _, _, _, events = environment.step("attack")

        self.assertEqual(target.hp, 1)
        self.assertEqual(events[0]["event"], "hit")

    def test_policy_dodges_away_during_attack_cooldown(self):
        environment = OfflineCombatEnvironment(seed=2)
        environment.player_attack_cooldown = 0.3
        environment.monsters[0].x = environment.player_x + 40
        environment.monsters[1].hp = 0
        environment.monsters[2].hp = 0
        policy = RuleCombatPolicy()

        decision = policy.choose(environment.observe())

        self.assertEqual(decision.action, "move_left")
        self.assertEqual(decision.reason, "dodge")

    def test_full_episode_completes_with_attack_and_dodge(self):
        result = run_episode(seed=7, max_steps=900)

        self.assertTrue(result["completed"])
        self.assertEqual(result["kills"], 3)
        self.assertGreater(result["action_counts"]["attack"], 0)
        self.assertGreater(result["decision_reasons"].get("dodge", 0), 0)
        self.assertFalse(result["client_connected"])
        self.assertFalse(result["input_devices_used"])


if __name__ == "__main__":
    unittest.main()
