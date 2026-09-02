"""Read-only diagnostic for minimap coordinate discontinuities.

The tool never sends input to the game.  It records the detected minimap
canvas and every eligible yellow component so a coordinate jump can be traced
to either a moving canvas boundary or a switch of marker candidates.
"""

import argparse
import json
import time
from collections import Counter

from tools.yolo_monster_viewer import (
    ReadOnlyWindowCapture,
    _find_minimap_color_markers,
    _minimap_canvas_crop,
    find_visible_window_title,
    load_config,
)


def detect(frame, minimap_cfg):
    crop, canvas_box = _minimap_canvas_crop(frame, minimap_cfg)
    if crop is None:
        return None
    candidates = _find_minimap_color_markers(
        crop,
        minimap_cfg.get("player_color", [0, 255, 255]),
        minimap_cfg.get("marker_color_tolerance", [40, 40, 40]),
        canvas_box,
        min_pixels=minimap_cfg.get("marker_min_pixels", 3),
        max_pixels=minimap_cfg.get("marker_max_pixels", 24),
        max_dimension=minimap_cfg.get("marker_max_dimension", 10),
        min_fill_ratio=minimap_cfg.get("marker_min_fill_ratio", 0.25),
    )
    player = max(
        candidates,
        key=lambda item: (item["pixel_count"], item["fill_ratio"]),
        default=None,
    )
    return {"canvas": canvas_box, "player": player, "candidates": candidates}


def public_marker(marker):
    if marker is None:
        return None
    return {
        "map_px": [round(float(value), 2) for value in marker["map_px"]],
        "map_norm": [round(float(value), 5) for value in marker["map_norm"]],
        "box": [int(value) for value in marker["marker_box_map"]],
        "pixels": int(marker["pixel_count"]),
    }


def is_discontinuous(previous, current, jump_px):
    if previous is None:
        return False
    if previous["canvas"] != current["canvas"]:
        return True
    old_marker = previous["player"]
    new_marker = current["player"]
    if old_marker is None or new_marker is None:
        return old_marker is not new_marker
    return any(
        abs(float(after) - float(before)) > jump_px
        for before, after in zip(old_marker["map_px"], new_marker["map_px"])
    )


def main():
    parser = argparse.ArgumentParser(
        description="Read-only minimap canvas/marker continuity probe."
    )
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument("--window-title-token", default="冒险岛")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--fps-limit", type=float, default=12.0)
    parser.add_argument("--jump-px", type=float, default=3.0)
    args = parser.parse_args()
    if args.duration <= 0 or args.fps_limit <= 0 or args.jump_px <= 0:
        raise ValueError("duration, fps-limit, and jump-px must be positive")

    minimap_cfg = load_config(args.cfg).get("minimap", {})
    title = find_visible_window_title(args.window_title_token)
    capture = ReadOnlyWindowCapture(title)
    samples = []
    events = []
    previous = None
    started = time.monotonic()
    try:
        time.sleep(0.5)
        while time.monotonic() - started < args.duration:
            frame_started = time.monotonic()
            frame = capture.get_frame()
            if frame is not None:
                current = detect(frame, minimap_cfg)
                if current is not None:
                    sample = {
                        "t": round(time.monotonic() - started, 3),
                        "canvas": current["canvas"],
                        "player": public_marker(current["player"]),
                        "candidate_count": len(current["candidates"]),
                        "candidates": [
                            public_marker(marker) for marker in current["candidates"]
                        ],
                    }
                    samples.append(sample)
                    if is_discontinuous(previous, current, args.jump_px):
                        events.append(sample)
                    previous = current
            remaining = 1.0 / args.fps_limit - (time.monotonic() - frame_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        capture.stop()

    canvas_counts = Counter(tuple(sample["canvas"]) for sample in samples)
    output = {
        "title": title,
        "frames": len(samples),
        "canvas_counts": {str(key): count for key, count in canvas_counts.items()},
        "missing_player_frames": sum(sample["player"] is None for sample in samples),
        "discontinuity_count": len(events),
        "discontinuities": events[:20],
        "first_samples": samples[:3],
        "last_samples": samples[-3:],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
