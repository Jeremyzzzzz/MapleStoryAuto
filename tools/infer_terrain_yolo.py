"""Offline YOLO inference for ladder, rope, and platform detection.

The command reads one saved image and writes one annotated image plus JSON.
It does not capture windows, open a viewer, or send keyboard/mouse input.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = REPO_ROOT / ".yolo_runtime"
if str(LOCAL_RUNTIME) not in sys.path:
    sys.path.insert(0, str(LOCAL_RUNTIME))

CLASS_COLORS = {
    "ladder": (40, 220, 255),
    "rope": (255, 170, 50),
    "platform": (80, 220, 90),
}


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Unable to encode image: {path}")
    encoded.tofile(str(path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run read-only terrain YOLO inference on a saved image."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("probe_output/new_mushroom_live_raw.png"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("training_runs/terrain_three_class_v1/weights/best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("probe_output/terrain_three_class_preview.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("probe_output/terrain_three_class_summary.json"),
    )
    parser.add_argument("--gameplay-height", type=int, default=687)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def draw_detections(image, detections):
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box_xyxy"]
        class_name = detection["class"]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        point1 = (int(round(x1)), int(round(y1)))
        point2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(output, point1, point2, color, 2)
        label = f"{class_name} {detection['confidence']:.2f}"
        cv2.putText(
            output,
            label,
            (point1[0], max(16, point1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def infer(args):
    source_path = args.source.resolve()
    model_path = args.model.resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if args.gameplay_height <= 0:
        raise ValueError("--gameplay-height must be positive")

    image = read_image(source_path)
    if image.shape[0] < args.gameplay_height:
        raise ValueError(
            f"Source height {image.shape[0]} is smaller than gameplay height "
            f"{args.gameplay_height}"
        )
    gameplay = image[: args.gameplay_height].copy()

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    result = model.predict(
        source=gameplay,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )[0]
    names = result.names
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].cpu().tolist()]
        class_id = int(box.cls[0].item())
        detections.append(
            {
                "class": str(names[class_id]),
                "class_id": class_id,
                "confidence": float(box.conf[0].item()),
                "box_xyxy": [x1, y1, x2, y2],
            }
        )
    detections.sort(key=lambda item: (item["class"], item["box_xyxy"][1]))

    output_image = draw_detections(gameplay, detections)
    output_path = args.output.resolve()
    summary_path = args.summary.resolve()
    write_image(output_path, output_image)
    summary = {
        "source": str(source_path),
        "model": str(model_path),
        "gameplay_size": [int(gameplay.shape[1]), int(gameplay.shape[0])],
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "device": str(args.device),
        "detections": detections,
        "counts": {
            class_name: sum(item["class"] == class_name for item in detections)
            for class_name in ("ladder", "rope", "platform")
        },
        "mode": "offline_static_image_only",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    summary = infer(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
