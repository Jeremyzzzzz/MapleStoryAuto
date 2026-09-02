"""Read-only live YOLO viewer for ladders, ropes, and platforms.

This module captures pixels from a visible window and draws detections. It has
no keyboard, mouse, game-control, or focus-changing behavior.
"""

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = REPO_ROOT / ".yolo_runtime"
for module_path in (REPO_ROOT, LOCAL_RUNTIME):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import cv2

from tools.infer_terrain_yolo import CLASS_COLORS, write_image
from tools.yolo_monster_viewer import (
    DetectionTracker,
    ReadOnlyWindowCapture,
    find_visible_window_title,
    load_config,
)


WINDOW_TITLE = "YOLO Terrain Detector - OBSERVE ONLY"
DEFAULT_MODEL = "training_runs/terrain_three_class_v1/weights/best.pt"
REQUIRED_CLASSES = {"ladder", "rope", "platform"}


def validate_model_classes(names):
    available = {str(name).strip().lower() for name in names.values()}
    if available != REQUIRED_CLASSES:
        raise ValueError(
            "Terrain model classes must be exactly ladder, rope, platform; "
            f"got: {', '.join(sorted(available))}"
        )


class YoloTerrainDetector:
    def __init__(self, model_path, confidence, iou, device, image_size):
        from ultralytics import YOLO

        resolved_model = Path(model_path)
        if not resolved_model.is_absolute():
            resolved_model = REPO_ROOT / resolved_model
        if not resolved_model.exists():
            raise FileNotFoundError(f"YOLO model not found: {resolved_model}")

        self.model_path = resolved_model.resolve()
        self.model = YOLO(str(self.model_path))
        validate_model_classes(self.model.names)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = str(device)
        self.image_size = int(image_size)

    def detect(self, frame, gameplay_height):
        gameplay = frame[:gameplay_height]
        result = self.model.predict(
            source=gameplay,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        detections = []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for coordinates, score, class_id in zip(boxes, scores, classes):
            class_name = str(result.names[class_id]).strip().lower()
            if class_name not in REQUIRED_CLASSES:
                continue
            x1, y1, x2, y2 = coordinates
            detections.append(
                {
                    "class": class_name,
                    "label": class_name.upper(),
                    "label_zh": class_name,
                    "confidence": float(score),
                    "box": [
                        max(0, int(round(x1))),
                        max(0, int(round(y1))),
                        max(1, int(round(x2 - x1))),
                        max(1, int(round(y2 - y1))),
                    ],
                    "color": CLASS_COLORS[class_name],
                }
            )
        return detections


def draw_detections(frame, detections, fps):
    output = frame.copy()
    counts = {class_name: 0 for class_name in REQUIRED_CLASSES}
    for detection in detections:
        x, y, width, height = detection["box"]
        color = detection["color"]
        counts[detection["class"]] += 1
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        track = f" #{detection['track_id']}" if "track_id" in detection else ""
        label = f"{detection['label']}{track} {detection['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_x = min(x, max(0, output.shape[1] - text_width - 7))
        text_y = max(text_height + 7, y)
        cv2.rectangle(
            output,
            (text_x, text_y - text_height - 7),
            (text_x + text_width + 7, text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (text_x + 3, text_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    status = (
        "OBSERVE ONLY | TERRAIN YOLO+TRACK | "
        f"ladder {counts['ladder']} | rope {counts['rope']} | "
        f"platform {counts['platform']} | {fps:.1f} FPS"
    )
    (status_width, _), _ = cv2.getTextSize(
        status, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1
    )
    cv2.rectangle(
        output,
        (0, 0),
        (min(output.shape[1], status_width + 20), 34),
        (25, 25, 25),
        -1,
    )
    cv2.putText(
        output,
        status,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )
    return output, counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only live YOLO detection for terrain classes."
    )
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument("--window-title-token")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--fps-limit", type=float, default=12.0)
    parser.add_argument("--track-max-missed", type=int, default=6)
    parser.add_argument("--track-smoothing", type=float, default=0.70)
    parser.add_argument("--track-match-iou", type=float, default=0.20)
    parser.add_argument("--track-center-distance", type=float, default=0.80)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--snapshot")
    parser.add_argument("--raw-snapshot")
    parser.add_argument("--summary")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    if args.fps_limit <= 0:
        raise ValueError("fps-limit must be positive")

    cfg = load_config(args.cfg)
    gameplay_height = int(cfg["ui_coords"]["ui_y_start"])
    window_token = args.window_title_token or cfg["game_window"]["title"]
    window_title = find_visible_window_title(window_token)
    detector = YoloTerrainDetector(
        args.model,
        args.confidence,
        args.iou,
        args.device,
        args.image_size,
    )
    tracker = DetectionTracker(
        max_missed=args.track_max_missed,
        smoothing=args.track_smoothing,
        match_iou=args.track_match_iou,
        max_center_distance=args.track_center_distance,
    )
    capture = ReadOnlyWindowCapture(window_title)

    started = time.time()
    last_frame_time = started
    fps = 0.0
    frames = 0
    latest = None
    latest_raw = None
    detections = []
    counts = {class_name: 0 for class_name in REQUIRED_CLASSES}
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

            raw_detections = detector.detect(frame, gameplay_height)
            detections = tracker.update(raw_detections)
            latest_raw = frame.copy()
            now = time.time()
            current_fps = 1.0 / max(now - last_frame_time, 1e-6)
            fps = current_fps if fps == 0 else fps * 0.85 + current_fps * 0.15
            last_frame_time = now
            latest, counts = draw_detections(frame, detections, fps)
            frames += 1

            if not args.headless:
                cv2.imshow(WINDOW_TITLE, latest)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and frames >= args.max_frames:
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
        write_image(Path(args.snapshot), latest)
    if latest_raw is not None and args.raw_snapshot:
        write_image(Path(args.raw_snapshot), latest_raw)

    summary = {
        "observe_only": True,
        "input_events_sent": False,
        "window_title": window_title,
        "model": str(detector.model_path),
        "model_classes": list(detector.model.names.values()),
        "tracking": {
            "enabled": True,
            "max_missed_frames": tracker.max_missed,
            "smoothing": tracker.smoothing,
        },
        "frames": frames,
        "fps": round(fps, 2),
        "counts": counts,
        "detections": [
            {
                "class": detection["class"],
                "confidence": round(detection["confidence"], 4),
                "box": detection["box"],
                "track_id": detection["track_id"],
                "missed_frames": detection["missed_frames"],
            }
            for detection in detections
        ],
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
