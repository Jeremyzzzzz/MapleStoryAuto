import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


ACTIONS = ("idle", "move_left", "move_right", "attack")


@dataclass
class Monster:
    monster_id: int
    x: float
    speed: float
    hp: int = 2
    attack_cooldown: float = 0.0


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str
    target_id: int | None


class OfflineCombatEnvironment:
    """Small deterministic combat level with no game-client dependencies."""

    def __init__(self, seed=7, width=1280.0, time_step=0.1):
        self.random = random.Random(seed)
        self.width = float(width)
        self.time_step = float(time_step)
        self.player_speed = 220.0
        self.attack_range = 135.0
        self.contact_range = 35.0
        self.player_attack_cooldown_seconds = 0.45
        self.monster_attack_cooldown_seconds = 0.8
        self.monster_damage = 8
        self.reset()

    def reset(self):
        self.player_x = self.width / 2.0
        self.player_hp = 100
        self.player_attack_cooldown = 0.0
        self.elapsed_seconds = 0.0
        self.kills = 0
        positions = (260.0, 930.0, 1140.0)
        self.monsters = [
            Monster(
                monster_id=index + 1,
                x=position,
                speed=self.random.uniform(55.0, 85.0),
            )
            for index, position in enumerate(positions)
        ]
        return self.observe()

    def living_monsters(self):
        return [monster for monster in self.monsters if monster.hp > 0]

    def observe(self):
        return {
            "player_x": self.player_x,
            "player_hp": self.player_hp,
            "attack_cooldown": self.player_attack_cooldown,
            "monsters": [
                {
                    "id": monster.monster_id,
                    "x": monster.x,
                    "hp": monster.hp,
                    "speed": monster.speed,
                }
                for monster in self.living_monsters()
            ],
        }

    def nearest_monster(self):
        living = self.living_monsters()
        if not living:
            return None
        return min(living, key=lambda monster: abs(monster.x - self.player_x))

    def step(self, action):
        if action not in ACTIONS:
            raise ValueError(f"Unsupported action: {action}")

        dt = self.time_step
        reward = 0.01
        events = []
        self.player_attack_cooldown = max(
            0.0, self.player_attack_cooldown - dt
        )
        for monster in self.living_monsters():
            monster.attack_cooldown = max(0.0, monster.attack_cooldown - dt)

        if action == "move_left":
            self.player_x = max(0.0, self.player_x - self.player_speed * dt)
        elif action == "move_right":
            self.player_x = min(
                self.width, self.player_x + self.player_speed * dt
            )
        elif action == "attack" and self.player_attack_cooldown <= 0.0:
            target = self.nearest_monster()
            if target and abs(target.x - self.player_x) <= self.attack_range:
                target.hp -= 1
                reward += 1.0
                events.append({"event": "hit", "monster_id": target.monster_id})
                self.player_attack_cooldown = (
                    self.player_attack_cooldown_seconds
                )
                if target.hp <= 0:
                    self.kills += 1
                    reward += 10.0
                    events.append(
                        {"event": "kill", "monster_id": target.monster_id}
                    )
            else:
                reward -= 0.1

        for monster in self.living_monsters():
            distance = abs(monster.x - self.player_x)
            if distance > self.contact_range:
                direction = 1.0 if self.player_x > monster.x else -1.0
                monster.x += direction * monster.speed * dt
                monster.x = max(0.0, min(self.width, monster.x))
            elif monster.attack_cooldown <= 0.0:
                self.player_hp = max(0, self.player_hp - self.monster_damage)
                monster.attack_cooldown = self.monster_attack_cooldown_seconds
                reward -= 4.0
                events.append(
                    {
                        "event": "player_hit",
                        "monster_id": monster.monster_id,
                        "damage": self.monster_damage,
                    }
                )

        self.elapsed_seconds += dt
        done = self.player_hp <= 0 or not self.living_monsters()
        return self.observe(), reward, done, events


class RuleCombatPolicy:
    """Closed-loop baseline for attack and dodge behavior in the simulator."""

    def __init__(self, attack_range=135.0, dodge_range=90.0):
        self.attack_range = float(attack_range)
        self.dodge_range = float(dodge_range)

    def choose(self, observation):
        monsters = observation["monsters"]
        if not monsters:
            return PolicyDecision("idle", "complete", None)

        player_x = observation["player_x"]
        target = min(monsters, key=lambda item: abs(item["x"] - player_x))
        signed_distance = target["x"] - player_x
        distance = abs(signed_distance)
        cooldown_ready = observation["attack_cooldown"] <= 1e-9

        if distance <= self.dodge_range and not cooldown_ready:
            action = "move_left" if signed_distance > 0 else "move_right"
            return PolicyDecision(action, "dodge", target["id"])
        if distance <= self.attack_range and cooldown_ready:
            return PolicyDecision("attack", "attack", target["id"])
        if distance <= self.attack_range:
            return PolicyDecision("idle", "cooldown_spacing", target["id"])

        action = "move_right" if signed_distance > 0 else "move_left"
        return PolicyDecision(action, "approach", target["id"])


def run_episode(seed=7, max_steps=900, trace_limit=120):
    environment = OfflineCombatEnvironment(seed=seed)
    policy = RuleCombatPolicy(attack_range=environment.attack_range)
    total_reward = 0.0
    action_counts = {action: 0 for action in ACTIONS}
    reason_counts = {}
    event_counts = {"hit": 0, "kill": 0, "player_hit": 0}
    trace = []

    for step_index in range(max_steps):
        before = environment.observe()
        decision = policy.choose(before)
        after, reward, done, events = environment.step(decision.action)
        total_reward += reward
        action_counts[decision.action] += 1
        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
        for event in events:
            event_counts[event["event"]] += 1
        if events and len(trace) < trace_limit:
            trace.append(
                {
                    "step": step_index,
                    "time_seconds": round(environment.elapsed_seconds, 2),
                    "action": decision.action,
                    "reason": decision.reason,
                    "target_id": decision.target_id,
                    "events": events,
                    "player_hp": after["player_hp"],
                    "living_monsters": len(after["monsters"]),
                }
            )
        if done:
            break

    return {
        "environment": "offline_combat_simulator",
        "client_connected": False,
        "input_devices_used": False,
        "seed": seed,
        "steps": step_index + 1,
        "elapsed_seconds": round(environment.elapsed_seconds, 2),
        "completed": not environment.living_monsters(),
        "player_hp": environment.player_hp,
        "kills": environment.kills,
        "total_reward": round(total_reward, 2),
        "action_counts": action_counts,
        "decision_reasons": reason_counts,
        "event_counts": event_counts,
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run closed-loop attack/dodge logic in an isolated local level."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    result = run_episode(seed=args.seed, max_steps=args.max_steps)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
