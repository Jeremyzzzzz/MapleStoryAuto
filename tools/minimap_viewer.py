"""Read-only minimap coordinate viewer without a YOLO/PyTorch dependency."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from tools.yolo_monster_viewer import (
    ReadOnlyWindowCapture,
    find_visible_window_title,
    load_config,
    locate_minimap_players,
    MinimapRedMarkerTracker,
    save_image,
)


WINDOW_TITLE = "Minimap Coordinate Detector - OBSERVE ONLY"


def draw_minimap_view(frame, marker, fps, other_players=None):
    output = frame.copy()
    other_players = list(other_players or ())
    status = "P1" if marker is not None else "MISSED"
    canvas_marker = marker or (other_players[0] if other_players else None)
    if canvas_marker is not None:
        x, y, width, height = canvas_marker["canvas_frame_box"]
        cv2.rectangle(
            output, (x, y), (x + width, y + height), (0, 200, 255), 1
        )
    if marker is not None:
        marker_x, marker_y = [int(round(value)) for value in marker["frame_px"]]
        diamond = np.asarray(
            [
                [marker_x, marker_y - 6],
                [marker_x + 6, marker_y],
                [marker_x, marker_y + 6],
                [marker_x - 6, marker_y],
            ],
            dtype=np.int32,
        )
        cv2.polylines(output, [diamond], True, (0, 230, 255), 2, cv2.LINE_AA)
    for index, other in enumerate(other_players, start=1):
        marker_x, marker_y = [int(round(value)) for value in other["frame_px"]]
        cv2.circle(output, (marker_x, marker_y), 5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(
            output,
            (marker_x, marker_y),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            7,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"R{index}",
            (marker_x + 6, marker_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    text = f"MINIMAP OBSERVE ONLY | player {status} | red {len(other_players)} | {fps:.1f} FPS"
    (text_width, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1
    )
    cv2.rectangle(output, (0, 0), (min(output.shape[1], text_width + 20), 34), (25, 25, 25), -1)
    cv2.putText(
        output,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )

    panel = np.full((output.shape[0], 300, 3), 26, dtype=np.uint8)
    cv2.putText(panel, "MINIMAP COORDINATES", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"player {status}", (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    if marker is not None:
        map_x, map_y = marker["map_px"]
        norm_x, norm_y = marker["map_norm"]
        cv2.putText(panel, f"map xy ({map_x:.1f}, {map_y:.1f})", (14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"norm ({norm_x:.3f}, {norm_y:.3f})", (14, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA)
        width, height = marker["canvas_size"]
        cv2.putText(panel, f"canvas {width} x {height} px", (14, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "marker not detected", (14, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 80, 255), 1, cv2.LINE_AA)
    red_y = 176 if marker is not None else 118
    cv2.putText(
        panel,
        f"other players {len(other_players)}",
        (14, red_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 80, 255),
        1,
        cv2.LINE_AA,
    )
    red_y += 24
    for index, other in enumerate(other_players[:8], start=1):
        map_x, map_y = other["map_px"]
        norm_x, norm_y = other["map_norm"]
        cv2.putText(
            panel,
            f"R{index} map ({map_x:.1f}, {map_y:.1f})",
            (14, red_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (0, 80, 255),
            1,
            cv2.LINE_AA,
        )
        red_y += 18
        cv2.putText(
            panel,
            f"norm ({norm_x:.3f}, {norm_y:.3f})",
            (24, red_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (175, 175, 175),
            1,
            cv2.LINE_AA,
        )
        red_y += 22
    if len(other_players) > 8:
        cv2.putText(panel, f"+ {len(other_players) - 8} more", (14, red_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1, cv2.LINE_AA)
    return np.hstack((output, panel))


def main():
    parser = argparse.ArgumentParser(description="Read-only minimap coordinate viewer.")
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument("--window-title-token", default="冒险岛怀旧服")
    parser.add_argument("--fps-limit", type=float, default=12.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--snapshot")
    args = parser.parse_args()
    if args.fps_limit <= 0:
        raise ValueError("fps-limit must be positive")

    cfg = load_config(args.cfg)
    window_title = find_visible_window_title(args.window_title_token)
    capture = ReadOnlyWindowCapture(window_title)
    started = time.time()
    last_frame = started
    fps = 0.0
    latest = None
    red_tracker = MinimapRedMarkerTracker(
        confirm_frames=cfg.get("minimap", {}).get("other_player_confirm_frames", 2),
        max_missed=cfg.get("minimap", {}).get("other_player_max_missed_frames", 1),
        max_distance=cfg.get("minimap", {}).get("other_player_match_distance_px", 8.0),
    )
    try:
        if not args.headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, 1280, 720)
        while True:
            loop_started = time.time()
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            now = time.time()
            instant = 1.0 / max(now - last_frame, 1e-6)
            fps = instant if fps == 0.0 else fps * 0.85 + instant * 0.15
            last_frame = now
            minimap_markers = locate_minimap_players(frame, cfg.get("minimap", {}))
            marker = minimap_markers["player"]
            other_players = red_tracker.update(minimap_markers["other_players"])
            latest = draw_minimap_view(
                frame,
                marker,
                fps,
                other_players,
            )
            if not args.headless:
                cv2.imshow(WINDOW_TITLE, latest)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.duration > 0 and now - started >= args.duration:
                break
            remaining = 1.0 / args.fps_limit - (time.time() - loop_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        capture.stop()
        if not args.headless:
            cv2.destroyAllWindows()
    if latest is not None and args.snapshot:
        save_image(args.snapshot, latest)


if __name__ == "__main__":
    main()
