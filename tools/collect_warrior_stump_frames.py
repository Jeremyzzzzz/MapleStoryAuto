"""Bounded, observe-only collection of Warrior Tribe stump frames.

The collector deliberately stores raw gameplay frames and model proposals only;
it never treats model output as ground truth and never sends input to the game.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Support both ``python tools/script.py`` and ``python -m tools.script``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.yolo_monster_viewer import (
    ReadOnlyWindowCapture,
    YoloMonsterDetector,
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


def draw_proposals(image, proposals, color, prefix):
    output = image.copy()
    for index, item in enumerate(proposals, 1):
        x, y, width, height = item["box"]
        x2, y2 = x + width, y + height
        cv2.rectangle(output, (x, y), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{prefix}{index} {item['confidence']:.2f}",
            (x, max(16, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def proposal_payload(items):
    return [
        {
            "confidence": round(float(item["confidence"]), 5),
            "box": [int(value) for value in item["box"]],
        }
        for item in items
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument(
        "--model",
        default="training_runs/warrior_stump_hardneg_v2_1280/weights/best.pt",
    )
    parser.add_argument("--output", type=Path, default=Path("probe_output/warrior_stump_collection_20260901"))
    parser.add_argument("--max-frames", type=int, default=18)
    parser.add_argument("--interval", type=float, default=0.8)
    parser.add_argument("--low-confidence", type=float, default=0.05)
    parser.add_argument("--high-confidence", type=float, default=0.70)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.max_frames <= 0 or args.interval <= 0:
        raise ValueError("max-frames and interval must be positive")
    if not 0.0 < args.low_confidence <= args.high_confidence <= 1.0:
        raise ValueError("confidence thresholds must satisfy 0 < low <= high <= 1")

    cfg = load_config(args.cfg)
    configured_height = int(cfg["ui_coords"]["ui_y_start"])
    reference_width = cfg["ui_coords"].get("reference_width")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    low_detector = YoloMonsterDetector(
        args.model,
        args.low_confidence,
        0.45,
        args.device,
        args.image_size,
        {"stump"},
    )
    high_detector = YoloMonsterDetector(
        args.model,
        args.high_confidence,
        0.45,
        args.device,
        args.image_size,
        {"stump"},
    )
    title = find_visible_window_title("冒险岛怀旧服")
    capture = ReadOnlyWindowCapture(title)
    records = []
    last_saved = None
    try:
        deadline = time.time() + max(20.0, args.max_frames * args.interval * 2.0)
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
                # Avoid filling the set with identical idle frames while still
                # retaining every meaningful movement/occlusion change.
                if delta < 1.5 and len(records) + 1 < args.max_frames:
                    continue
            low = low_detector.detect(frame, gameplay_height)
            high = high_detector.detect(frame, gameplay_height)
            index = len(records)
            stem = f"frame_{index:03d}"
            save_image(output / f"{stem}_raw.png", gameplay)
            preview = draw_proposals(gameplay, low, (0, 80, 255), "L")
            preview = draw_proposals(preview, high, (0, 220, 0), "H")
            save_image(output / f"{stem}_proposals.png", preview)
            records.append(
                {
                    "frame": stem,
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "shape": [int(gameplay.shape[1]), int(gameplay.shape[0])],
                    "low_confidence": args.low_confidence,
                    "high_confidence": args.high_confidence,
                    "low_proposals": proposal_payload(low),
                    "high_proposals": proposal_payload(high),
                    "label_status": "unreviewed",
                }
            )
            last_saved = gameplay
    finally:
        capture.stop()
    manifest = {
        "observe_only": True,
        "input_events_sent": False,
        "window_title": title,
        "model": str(Path(args.model).resolve()),
        "frames_collected": len(records),
        "records": records,
        "labeling_rule": "Review raw frames manually; model proposals are not labels.",
    }
    (output / "collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: manifest[k] for k in ("observe_only", "input_events_sent", "frames_collected", "model")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
