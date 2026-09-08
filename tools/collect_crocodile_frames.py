"""Bounded, observation-only capture of crocodile gameplay frames.

The collector saves raw gameplay pixels and a manifest only. It never sends
keyboard or mouse input and it never treats detector output as a label.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.yolo_monster_viewer import (
    ReadOnlyWindowCapture,
    find_visible_window_title,
    load_config,
    resolve_gameplay_height,
)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, data = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError(f"Unable to encode image: {path}")
    data.tofile(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument(
        "--output", type=Path, default=Path("probe_output/crocodile_collection_v1")
    )
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--interval", type=float, default=0.7)
    parser.add_argument("--min-delta", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_frames <= 0 or args.interval <= 0:
        raise ValueError("max-frames and interval must be positive")

    cfg = load_config(args.cfg)
    configured_height = int(cfg["ui_coords"]["ui_y_start"])
    reference_width = cfg["ui_coords"].get("reference_width")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    title = find_visible_window_title("冒险岛怀旧服")
    capture = ReadOnlyWindowCapture(title)
    records = []
    last_saved = None
    try:
        deadline = time.time() + max(30.0, args.max_frames * args.interval * 2.0)
        next_sample = time.time()
        while len(records) < args.max_frames and time.time() < deadline:
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            now = time.time()
            if now < next_sample:
                time.sleep(0.03)
                continue
            next_sample = now + args.interval
            gameplay_height = resolve_gameplay_height(
                frame.shape, configured_height, reference_width
            )
            gameplay = frame[:gameplay_height].copy()
            if last_saved is not None:
                delta = float(
                    np.mean(
                        cv2.absdiff(
                            cv2.resize(gameplay, (320, 180)),
                            cv2.resize(last_saved, (320, 180)),
                        )
                    )
                )
                if delta < args.min_delta and len(records) + 1 < args.max_frames:
                    continue
            index = len(records)
            stem = f"frame_{index:03d}"
            save_image(output / f"{stem}_raw.png", gameplay)
            records.append(
                {
                    "frame": stem,
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "shape": [int(gameplay.shape[1]), int(gameplay.shape[0])],
                    "split": "unassigned",
                    "boxes": [],
                    "label_status": "unreviewed",
                }
            )
            last_saved = gameplay
    finally:
        capture.stop()

    manifest = {
        "version": 1,
        "classes": ["crocodile"],
        "observe_only": True,
        "input_events_sent": False,
        "window_title": title,
        "frames_collected": len(records),
        "records": records,
        "labeling_rule": "Review raw frames manually; no detector proposals are labels.",
    }
    (output / "collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("observe_only", "input_events_sent", "frames_collected", "window_title")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
