"""Evaluate YOLO monster models on source-frame holdouts."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = REPO_ROOT / ".yolo_runtime"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def load_image(path):
    image = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, data = cv2.imencode(path.suffix or ".png", image)
    if not encoded:
        raise RuntimeError(f"Unable to encode image: {path}")
    data.tofile(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def dataset_class_names(payload):
    names = payload.get("names")
    if isinstance(names, list):
        output = [str(name).strip().lower() for name in names]
    elif isinstance(names, dict):
        indexed = {int(class_id): name for class_id, name in names.items()}
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise ValueError("Dataset class ids must be contiguous from zero")
        output = [str(indexed[class_id]).strip().lower() for class_id in expected]
    else:
        raise ValueError("Dataset must define names as a list or id mapping")
    if not output or any(not name for name in output) or len(set(output)) != len(output):
        raise ValueError("Dataset class names must be non-empty and unique")
    return output


def resolve_dataset(data_path, split):
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    class_names = dataset_class_names(payload)
    root = Path(payload.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    entries = payload[split]
    if isinstance(entries, (str, Path)):
        entries = [entries]
    images = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            images.append(path.resolve())
        else:
            images.extend(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.suffix.lower() in IMAGE_SUFFIXES
            )
    return root.resolve(), sorted(set(images)), class_names


def label_path_for_image(root, image_path):
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    try:
        image_index = parts.index("images")
    except ValueError as error:
        raise ValueError(
            f"Image is not below an images directory: {image_path}"
        ) from error
    parts[image_index] = "labels"
    return (root.joinpath(*parts)).with_suffix(".txt")


def load_ground_truth(path, image_width, image_height, class_names):
    boxes = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        class_id, center_x, center_y, width, height = line.split()
        class_id = int(class_id)
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"Class id {class_id} is not declared by the dataset")
        width = float(width) * image_width
        height = float(height) * image_height
        center_x = float(center_x) * image_width
        center_y = float(center_y) * image_height
        boxes.append(
            {
                "class": class_names[class_id],
                "box": [
                    center_x - width / 2.0,
                    center_y - height / 2.0,
                    width,
                    height,
                ],
            }
        )
    return boxes


def box_iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = overlap_width * overlap_height
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


def match_frame(ground_truth, predictions, iou_threshold):
    matches = []
    matched_ground_truth = set()
    matched_predictions = set()
    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(ground_truth):
            if prediction["class"] != truth["class"]:
                continue
            iou = box_iou(prediction["box"], truth["box"])
            if iou >= iou_threshold:
                candidates.append(
                    (-prediction["confidence"], -iou, prediction_index, truth_index)
                )
    for _, negative_iou, prediction_index, truth_index in sorted(candidates):
        if (
            prediction_index in matched_predictions
            or truth_index in matched_ground_truth
        ):
            continue
        matched_predictions.add(prediction_index)
        matched_ground_truth.add(truth_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "truth_index": truth_index,
                "iou": round(-negative_iou, 4),
            }
        )
    return matches, matched_ground_truth, matched_predictions


def draw_frame(image, ground_truth, predictions, matches):
    output = image.copy()
    matched_predictions = {item["prediction_index"] for item in matches}
    for truth in ground_truth:
        x, y, width, height = [round(value) for value in truth["box"]]
        cv2.rectangle(output, (x, y), (x + width, y + height), (40, 220, 40), 2)
        cv2.putText(
            output,
            f"GT {truth['class']}",
            (x, max(16, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (40, 220, 40),
            1,
            cv2.LINE_AA,
        )
    for index, prediction in enumerate(predictions):
        x, y, width, height = [round(value) for value in prediction["box"]]
        color = (255, 180, 40) if index in matched_predictions else (40, 40, 255)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 1)
        cv2.putText(
            output,
            f"P {prediction['class']} {prediction['confidence']:.2f}",
            (x, min(output.shape[0] - 4, y + height + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def parse_model(value):
    if "=" in value:
        name, raw_path = value.split("=", 1)
    else:
        raw_path = value
        name = Path(value).stem
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return name, path.resolve()


def parse_confidence_overrides(values):
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "model-confidence must use NAME=VALUE, "
                f"got: {value}"
            )
        name, raw_confidence = value.split("=", 1)
        confidence = float(raw_confidence)
        if not 0.0 < confidence <= 1.0:
            raise ValueError("model confidence must be in (0, 1]")
        overrides[name] = confidence
    return overrides


def evaluate_model(
    model_name,
    model_path,
    root,
    images,
    confidence,
    nms_iou,
    match_iou,
    image_size,
    device,
    preview_root,
    class_names,
):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    totals = {class_name: Counter() for class_name in class_names}
    frames = []
    for image_path in images:
        image = load_image(image_path)
        ground_truth = load_ground_truth(
            label_path_for_image(root, image_path),
            image.shape[1],
            image.shape[0],
            class_names,
        )
        result = model.predict(
            source=image,
            conf=confidence,
            iou=nms_iou,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]
        predictions = []
        if result.boxes is not None:
            for xyxy, score, class_id in zip(
                result.boxes.xyxy.detach().cpu().numpy(),
                result.boxes.conf.detach().cpu().numpy(),
                result.boxes.cls.detach().cpu().numpy().astype(int),
            ):
                class_name = str(result.names[int(class_id)]).strip().lower()
                if class_name not in class_names:
                    continue
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                predictions.append(
                    {
                        "class": class_name,
                        "confidence": float(score),
                        "box": [x1, y1, x2 - x1, y2 - y1],
                    }
                )
        matches, matched_truth, matched_predictions = match_frame(
            ground_truth, predictions, match_iou
        )
        for index, truth in enumerate(ground_truth):
            key = "tp" if index in matched_truth else "fn"
            totals[truth["class"]][key] += 1
        for index, prediction in enumerate(predictions):
            if index not in matched_predictions:
                totals[prediction["class"]]["fp"] += 1
        frame_summary = {
            "image": str(image_path),
            "ground_truth": len(ground_truth),
            "predictions": len(predictions),
            "matches": len(matches),
            "false_positives": len(predictions) - len(matches),
            "false_negatives": len(ground_truth) - len(matches),
        }
        frames.append(frame_summary)
        if preview_root is not None:
            preview = draw_frame(image, ground_truth, predictions, matches)
            save_image(
                preview_root / model_name / f"{image_path.stem}.png", preview
            )

    per_class = {}
    overall = Counter()
    for class_name, counts in totals.items():
        overall.update(counts)
        precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        per_class[class_name] = {
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    precision = overall["tp"] / max(1, overall["tp"] + overall["fp"])
    recall = overall["tp"] / max(1, overall["tp"] + overall["fn"])
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "confidence": confidence,
        "overall": {
            "tp": overall["tp"],
            "fp": overall["fp"],
            "fn": overall["fn"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "per_class": per_class,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument(
        "--model-confidence",
        action="append",
        default=[],
        help="Optional per-model threshold as NAME=VALUE.",
    )
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()

    if str(LOCAL_RUNTIME) not in sys.path:
        sys.path.insert(0, str(LOCAL_RUNTIME))
    data_path = args.data.resolve()
    root, images, class_names = resolve_dataset(data_path, args.split)
    if not images:
        raise ValueError(f"No images found for split: {args.split}")
    preview_root = None if args.preview_dir is None else args.preview_dir.resolve()
    confidence_overrides = parse_confidence_overrides(args.model_confidence)
    report = {
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "split": args.split,
        "source_frame_count": len(images),
        "confidence": args.confidence,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "image_size": args.image_size,
        "device": args.device,
        "class_names": class_names,
        "models": {},
    }
    for raw_model in args.model:
        model_name, model_path = parse_model(raw_model)
        model_confidence = confidence_overrides.get(
            model_name, args.confidence
        )
        report["models"][model_name] = evaluate_model(
            model_name,
            model_path,
            root,
            images,
            model_confidence,
            args.nms_iou,
            args.match_iou,
            args.image_size,
            args.device,
            preview_root,
            class_names,
        )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
