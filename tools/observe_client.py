import argparse
import json
import time
from types import SimpleNamespace

import cv2
import numpy as np

from src.engine.HealthMonitor import HealthMonitor
from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.input.GameWindowCapturor import GameWindowCapturor
from src.utils.common import (
    get_minimap_loc_size,
    get_player_location_on_minimap,
    load_yaml,
    override_cfg,
)


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to load image: {path}")
    return image


def save_image(path, image):
    encoded, data = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError(f"Unable to encode image: {path}")
    data.tofile(path)


def create_player_detector(cfg):
    if not cfg["nametag"]["enable"]:
        return None
    bot = MapleStoryAutoBot(
        SimpleNamespace(disable_viz=False, disable_control=True, is_ui=False)
    )
    if bot.load_config(cfg) != 0:
        raise RuntimeError("Unable to load player detector configuration")
    return bot


def observe(frame, cfg, player_detector=None):
    minimap_box = get_minimap_loc_size(frame)
    player_minimap = None
    player_marker_frame = None
    if minimap_box is not None:
        x, y, width, height = minimap_box
        minimap = frame[y : y + height, x : x + width]
        player_minimap = get_player_location_on_minimap(
            minimap,
            minimap_player_color=cfg["minimap"]["player_color"],
            color_tolerance=cfg["minimap"].get("player_color_tolerance", 0),
            min_pixels=cfg["minimap"].get("player_min_pixels", 4),
            max_pixels=cfg["minimap"].get("player_max_pixels", 100),
        )
        if player_minimap is not None:
            player_marker_frame = (
                int(x + player_minimap[0]),
                int(y + player_minimap[1]),
            )

    health = HealthMonitor(cfg, kb_controller=None)
    if cfg["health_monitor"].get("input_full_frame", False):
        health_frame = frame
    else:
        ui_y = cfg["ui_coords"]["ui_y_start"]
        health_frame = frame[ui_y:, :]
    health.update_frame(health_frame)
    hp, mp, exp = health.get_hp_mp_exp_percent()

    player_screen = None
    nametag_box = None
    if player_detector is not None:
        player_detector.img_frame = frame
        player_detector.img_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        player_detector.img_frame_debug = frame.copy()
        player_screen = player_detector.get_player_location_by_nametag()
        if player_screen is not None:
            template_h, template_w = player_detector.img_nametag.shape[:2]
            nametag_box = [
                int(player_detector.loc_nametag[0]),
                int(player_detector.loc_nametag[1]),
                int(template_w),
                int(template_h),
            ]
        player_detector.is_first_frame = False

    return {
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        "minimap_box": list(map(int, minimap_box)) if minimap_box else None,
        "player_minimap": list(map(int, player_minimap)) if player_minimap else None,
        "player_marker_frame": (
            list(player_marker_frame) if player_marker_frame else None
        ),
        "player_screen": list(map(int, player_screen)) if player_screen else None,
        "nametag_box": nametag_box,
        "hp_percent": None if hp is None else round(float(hp), 2),
        "mp_percent": None if mp is None else round(float(mp), 2),
        "exp_percent": None if exp is None else round(float(exp), 2),
        "bar_regions": [list(map(int, region)) for region in health.loc_size_bars],
    }


def annotate(frame, result):
    output = frame.copy()
    if result["minimap_box"]:
        x, y, width, height = result["minimap_box"]
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 255, 0), 2)
    if result["player_marker_frame"]:
        cv2.circle(
            output,
            tuple(result["player_marker_frame"]),
            6,
            (0, 0, 255),
            2,
        )
    if result["nametag_box"]:
        x, y, width, height = result["nametag_box"]
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 1)
    if result["player_screen"]:
        cv2.circle(output, tuple(result["player_screen"]), 6, (255, 0, 255), 2)
    for x, y, width, height in result["bar_regions"]:
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 255, 255), 1)
    return output


def capture_frame(cfg, timeout):
    capture = GameWindowCapturor(cfg)
    try:
        deadline = time.time() + timeout
        frame = None
        while frame is None and time.time() < deadline:
            frame = capture.get_frame()
            time.sleep(0.05)
        if frame is None:
            raise RuntimeError("No frame received before timeout")
        return frame
    finally:
        capture.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument("--image")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--annotated-output")
    args = parser.parse_args()

    cfg = load_yaml("config/config_default.yaml")
    cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))
    player_detector = create_player_detector(cfg)
    frame = load_image(args.image) if args.image else capture_frame(cfg, args.timeout)
    result = observe(frame, cfg, player_detector=player_detector)

    if args.annotated_output:
        save_image(args.annotated_output, annotate(frame, result))

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
