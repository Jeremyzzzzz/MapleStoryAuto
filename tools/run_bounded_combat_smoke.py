import argparse
import json
import time

import pygetwindow as gw

from src.engine.HealthMonitor import HealthMonitor
from src.input.GameWindowCapturor import GameWindowCapturor
from src.input.KeyBoardController import key_up, press_key
from src.utils.common import activate_game_window, load_yaml, override_cfg


def load_config(name):
    cfg = load_yaml("config/config_default.yaml")
    return override_cfg(cfg, load_yaml(f"config/config_{name}.yaml"))


def target_is_foreground(window_title):
    active_window = gw.getActiveWindow()
    return bool(active_window and window_title in active_window.title)


def wait_for_frame(capture, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = capture.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.05)
    raise RuntimeError("No game frame received before timeout")


def read_vitals(monitor, frame, cfg):
    if cfg["health_monitor"].get("input_full_frame", False):
        health_frame = frame
    else:
        health_frame = frame[cfg["ui_coords"]["ui_y_start"] :, :]
    monitor.update_frame(health_frame)
    return monitor.get_hp_mp_exp_percent()


def main():
    parser = argparse.ArgumentParser(
        description="Run a foreground-gated, time-bounded combat input smoke test."
    )
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--attack-interval", type=float, default=1.0)
    parser.add_argument("--activate-window", action="store_true")
    parser.add_argument("--auto-consumables", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 1.0 <= args.duration <= 30.0:
        raise ValueError("duration must be between 1 and 30 seconds")
    if not 0.5 <= args.attack_interval <= 5.0:
        raise ValueError("attack interval must be between 0.5 and 5 seconds")

    cfg = load_config(args.cfg)
    keys = cfg["key"]
    health_cfg = cfg["health_monitor"]
    window_title = cfg["game_window"]["title"]
    monitor = HealthMonitor(cfg, kb_controller=None)
    capture = GameWindowCapturor(cfg)
    started = time.time()
    last_attack = float("-inf")
    last_hp = float("-inf")
    last_mp = float("-inf")
    counts = {"attack": 0, "add_hp": 0, "add_mp": 0, "skipped_background": 0}
    last_vitals = {"hp_percent": None, "mp_percent": None, "exp_percent": None}

    try:
        wait_for_frame(capture)
        if args.activate_window:
            activate_game_window(window_title)
            time.sleep(0.3)
            if not target_is_foreground(window_title):
                raise RuntimeError("Unable to activate the configured game window")

        while time.time() - started < args.duration:
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            hp, mp, exp = read_vitals(monitor, frame, cfg)
            last_vitals = {
                "hp_percent": None if hp is None else round(float(hp), 2),
                "mp_percent": None if mp is None else round(float(mp), 2),
                "exp_percent": None if exp is None else round(float(exp), 2),
            }

            now = time.time()
            action = None
            if (
                args.auto_consumables
                and hp is not None
                and hp <= health_cfg["add_hp_percent"]
                and now - last_hp >= health_cfg["add_hp_cooldown"]
            ):
                action = "add_hp"
                last_hp = now
            elif (
                args.auto_consumables
                and mp is not None
                and mp <= health_cfg["add_mp_percent"]
                and now - last_mp >= health_cfg["add_mp_cooldown"]
            ):
                action = "add_mp"
                last_mp = now
            elif now - last_attack >= args.attack_interval:
                action = "attack"
                last_attack = now

            if action is not None:
                if target_is_foreground(window_title):
                    if not args.dry_run:
                        press_key(
                            {
                                "attack": keys["directional_attack"],
                                "add_hp": keys["add_hp"],
                                "add_mp": keys["add_mp"],
                            }[action]
                        )
                    counts[action] += 1
                else:
                    counts["skipped_background"] += 1

            time.sleep(0.05)
    finally:
        capture.stop()
        for key in (keys["directional_attack"], keys["add_hp"], keys["add_mp"]):
            key_up(key)

    print(
        json.dumps(
            {
                "cfg": args.cfg,
                "dry_run": args.dry_run,
                "auto_consumables": args.auto_consumables,
                "duration_seconds": round(time.time() - started, 2),
                "attack_interval_seconds": args.attack_interval,
                "keys": {
                    "attack": keys["directional_attack"],
                    "add_hp": keys["add_hp"],
                    "add_mp": keys["add_mp"],
                },
                "thresholds": {
                    "hp_percent": health_cfg["add_hp_percent"],
                    "mp_percent": health_cfg["add_mp_percent"],
                },
                "actions": counts,
                "last_vitals": last_vitals,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
