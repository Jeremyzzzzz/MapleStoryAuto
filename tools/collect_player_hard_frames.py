"""Collect diverse raw gameplay frames for later player-identity labeling.

This is intentionally capture-only. It does not import game-control modules,
send input, focus the client, or change any existing template/model files.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.yolo_monster_viewer import (
    DEFAULT_WINDOW_TOKEN,
    ReadOnlyWindowCapture,
    find_visible_window_title,
)


def frame_difference(first, second):
    """Return a compact grayscale difference score for two BGR frames."""
    if first is None or second is None:
        return float("inf")
    first_small = cv2.resize(first, (192, 112), interpolation=cv2.INTER_AREA)
    second_small = cv2.resize(second, (192, 112), interpolation=cv2.INTER_AREA)
    first_gray = cv2.cvtColor(first_small, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second_small, cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(first_gray, second_gray).mean())


def save_jpeg(path, frame, quality):
    ok, encoded = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError(f"Unable to encode frame: {path}")
    encoded.tofile(str(path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only diverse-frame collector for player identity data."
    )
    parser.add_argument("--window-title-token", default=DEFAULT_WINDOW_TOKEN)
    parser.add_argument("--output")
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--min-difference", type=float, default=7.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    if args.interval <= 0.0:
        raise ValueError("interval must be positive")
    if args.max_frames <= 0:
        raise ValueError("max-frames must be positive")
    if args.min_difference < 0.0:
        raise ValueError("min-difference must be non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be in [1, 100]")

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "training_data" / f"player_identity_capture_{started_at}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.jsonl"
    window_title = find_visible_window_title(args.window_title_token)
    capture = ReadOnlyWindowCapture(window_title)
    started = time.time()
    previous_saved = None
    saved = 0
    sampled = 0
    skipped_similar = 0

    try:
        while saved < args.max_frames and time.time() - started < args.duration:
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            sampled += 1
            difference = frame_difference(previous_saved, frame)
            if previous_saved is not None and difference < args.min_difference:
                skipped_similar += 1
            else:
                captured_at = datetime.now().isoformat(timespec="milliseconds")
                frame_path = output_dir / f"frame_{saved:04d}.jpg"
                save_jpeg(frame_path, frame, args.jpeg_quality)
                record = {
                    "frame": frame_path.name,
                    "captured_at": captured_at,
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "difference_from_previous_saved": round(difference, 3),
                    "label": None,
                }
                with manifest_path.open("a", encoding="utf-8") as manifest:
                    manifest.write(json.dumps(record, ensure_ascii=True) + "\n")
                previous_saved = frame.copy()
                saved += 1
                print(json.dumps(record, ensure_ascii=True), flush=True)
            time.sleep(args.interval)
    finally:
        capture.stop()

    print(
        json.dumps(
            {
                "output": str(output_dir),
                "saved": saved,
                "sampled": sampled,
                "skipped_similar": skipped_similar,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
