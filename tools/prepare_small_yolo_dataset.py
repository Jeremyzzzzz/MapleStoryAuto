import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.input.GameWindowCapturor import GameWindowCapturor
from src.utils.common import get_mask, load_yaml, override_cfg
from tools.live_perception_viewer import (
    MonsterDetector,
    PlayerDetector,
    intersection_over_union,
    load_image,
    non_maximum_suppression,
    save_image,
)


CLASS_NAMES = ("red_snail", "blue_snail", "stump")
LABEL_TO_CLASS = {"RED SNAIL": 0, "BLUE SNAIL": 1, "STUMP": 2}
CLASS_COLORS = ((60, 60, 240), (240, 170, 30), (20, 130, 230))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a small three-class YOLO dataset from observation-only captures."
    )
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument(
        "--output", default="training_data/maple_three_class_v1", type=Path
    )
    parser.add_argument("--capture-seconds", type=float, default=20.0)
    parser.add_argument("--capture-fps", type=float, default=3.0)
    parser.add_argument("--probe-dir", type=Path, default=Path("../probe_output"))
    parser.add_argument("--probe-glob", default="warrior*.png")
    parser.add_argument("--synthetic-stumps", type=int, default=48)
    parser.add_argument("--min-blue-train-instances", type=int, default=32)
    parser.add_argument("--no-reuse-captures", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def unique_real_images(probe_dir, probe_glob):
    excluded_tokens = ("annotated", "preview", "debug", "nametag", "player_crop")
    images = []
    if not probe_dir.exists():
        return images
    for path in sorted(probe_dir.glob(probe_glob), key=lambda item: item.stat().st_mtime):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(token in path.stem.lower() for token in excluded_tokens):
            continue
        images.append(path)
    return images


def capture_frames(cfg, destination, duration, fps):
    if duration <= 0:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    capture = GameWindowCapturor(cfg)
    saved = []
    interval = 1.0 / max(0.1, fps)
    started = time.monotonic()
    next_frame_at = started
    try:
        while time.monotonic() - started < duration:
            frame = capture.get_frame()
            now = time.monotonic()
            if frame is None or now < next_frame_at:
                time.sleep(0.01)
                continue
            path = destination / f"live_{len(saved):04d}.png"
            save_image(path, frame)
            saved.append(path)
            next_frame_at += interval
    finally:
        capture.stop()
    return saved


class DatasetLabeler:
    def __init__(self, cfg):
        self.player_detector = PlayerDetector(cfg)
        self.monster_detector = MonsterDetector(cfg)
        self.ui_y_start = cfg["ui_coords"]["ui_y_start"]
        self.stump_threshold = cfg["perception_overlay"]["stump_match_threshold"]
        self.max_monsters = cfg["perception_overlay"]["max_monsters"]
        self.stump_templates = self._build_stump_templates(cfg)

    @staticmethod
    def _build_stump_templates(cfg):
        source = load_image(cfg["perception_overlay"]["stump_template"])
        source_mask = get_mask(source, (0, 255, 0))
        templates = []
        for scale in (0.84, 0.90, 0.96, 1.02):
            size = (
                max(1, int(round(source.shape[1] * scale))),
                max(1, int(round(source.shape[0] * scale))),
            )
            sprite = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(source_mask, size, interpolation=cv2.INTER_NEAREST)
            templates.extend(((sprite, mask), (cv2.flip(sprite, 1), cv2.flip(mask, 1))))
        return templates

    def _detect_stumps_multiscale(self, frame):
        gameplay = frame[: self.ui_y_start]
        candidates = []
        for template, mask in self.stump_templates:
            height, width = template.shape[:2]
            if height > gameplay.shape[0] or width > gameplay.shape[1]:
                continue
            result = cv2.matchTemplate(
                gameplay, template, cv2.TM_CCORR_NORMED, mask=mask
            )
            result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
            local_maximum = cv2.dilate(result, np.ones((13, 13), dtype=np.uint8))
            rows, columns = np.where(
                (result >= local_maximum - 1e-7)
                & (result >= self.stump_threshold)
            )
            for y, x in zip(rows, columns):
                candidates.append(
                    {
                        "label": "STUMP",
                        "score": float(result[y, x]),
                        "box": (int(x), int(y), width, height),
                        "method": "template multiscale",
                    }
                )
        return non_maximum_suppression(candidates, threshold=0.22)[
            : self.max_monsters
        ]

    def detect(self, frame):
        player = self.player_detector.detect(frame)
        detections = self.monster_detector._detect_colored_snails(frame, player)
        detections.extend(self._detect_stumps_multiscale(frame))
        if player:
            detections = [
                item
                for item in detections
                if intersection_over_union(item, player) <= 0.05
            ]
        return detections[: self.max_monsters]


def clamp_box(box, image_shape):
    x, y, width, height = box
    image_height, image_width = image_shape[:2]
    x1 = max(0, min(image_width - 1, int(x)))
    y1 = max(0, min(image_height - 1, int(y)))
    x2 = max(x1 + 1, min(image_width, int(x + width)))
    y2 = max(y1 + 1, min(image_height, int(y + height)))
    return x1, y1, x2 - x1, y2 - y1


def yolo_line(class_id, box, image_shape):
    image_height, image_width = image_shape[:2]
    x, y, width, height = clamp_box(box, image_shape)
    center_x = (x + width / 2.0) / image_width
    center_y = (y + height / 2.0) / image_height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width / image_width:.6f} {height / image_height:.6f}"


def add_real_sample(path, labeler, sequence):
    frame = load_image(path)
    detections = labeler.detect(frame)
    labels = []
    preview_detections = []
    for detection in detections:
        class_id = LABEL_TO_CLASS.get(detection["label"])
        if class_id is None:
            continue
        box = clamp_box(detection["box"], frame.shape)
        labels.append((class_id, box))
        preview_detections.append({**detection, "box": box, "class_id": class_id})
    return {
        "kind": "real",
        "source": str(path.resolve()),
        "sequence": sequence,
        "image": frame,
        "labels": labels,
        "preview": preview_detections,
    }


def choose_real_validation(samples):
    positive = [sample for sample in samples if sample["labels"]]
    desired = max(3, int(round(len(positive) * 0.2))) if positive else 0
    selected = set()
    for class_id in range(len(CLASS_NAMES)):
        matches = [
            index
            for index, sample in enumerate(samples)
            if any(label[0] == class_id for label in sample["labels"])
        ]
        if matches:
            selected.add(matches[-1])
    for index in reversed(range(len(samples))):
        if len(selected) >= desired:
            break
        if samples[index]["labels"]:
            selected.add(index)
    return selected


def foreground_sprite(path):
    sprite = load_image(path)
    alpha = get_mask(sprite, (0, 255, 0))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return sprite, alpha


def make_synthetic_stump(sample, sprite_and_alpha, rng, index):
    frame = sample["image"].copy()
    source, source_alpha = sprite_and_alpha
    scale = rng.uniform(0.78, 1.14)
    size = (
        max(1, int(round(source.shape[1] * scale))),
        max(1, int(round(source.shape[0] * scale))),
    )
    sprite = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(source_alpha, size, interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.5:
        sprite = cv2.flip(sprite, 1)
        alpha = cv2.flip(alpha, 1)

    height, width = sprite.shape[:2]
    max_y = max(1, min(frame.shape[0] - height - 1, 620))
    min_y = min(max_y, 210)
    x = rng.randint(100, max(100, frame.shape[1] - width - 20))
    y = rng.randint(min_y, max_y)
    region = frame[y : y + height, x : x + width]
    foreground = alpha > 0
    region[foreground] = sprite[foreground]
    frame[y : y + height, x : x + width] = region
    labels = list(sample["labels"]) + [(2, (x, y, width, height))]
    preview = list(sample["preview"]) + [
        {
            "label": "STUMP",
            "score": 1.0,
            "box": (x, y, width, height),
            "method": "synthetic",
            "class_id": 2,
        }
    ]
    return {
        "kind": "synthetic",
        "source": sample["source"],
        "sequence": index,
        "image": frame,
        "labels": labels,
        "preview": preview,
    }


def augment_sample(sample, rng, index):
    image = sample["image"]
    height, width = image.shape[:2]
    dx = rng.randint(-45, 45)
    dy = rng.randint(-24, 24)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    augmented = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.82, 1.18)
    hsv[:, :, 2] *= rng.uniform(0.86, 1.14)
    augmented = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    labels = []
    preview = []
    for detection in sample["preview"]:
        class_id = detection["class_id"]
        x, y, box_width, box_height = detection["box"]
        box = clamp_box((x + dx, y + dy, box_width, box_height), augmented.shape)
        if box[2] < box_width * 0.65 or box[3] < box_height * 0.65:
            continue
        labels.append((class_id, box))
        preview.append({**detection, "box": box, "method": "rare-class augmentation"})
    return {
        "kind": "augmented",
        "source": sample["source"],
        "sequence": index,
        "image": augmented,
        "labels": labels,
        "preview": preview,
    }


def balance_rare_class(samples, class_id, minimum_instances, rng):
    sources = [
        sample
        for sample in samples
        if any(label[0] == class_id for label in sample["labels"])
    ]
    current = sum(
        1 for sample in samples for label in sample["labels"] if label[0] == class_id
    )
    augmented = []
    attempts = 0
    while sources and current < minimum_instances and attempts < minimum_instances * 4:
        sample = augment_sample(rng.choice(sources), rng, attempts)
        added = sum(1 for label in sample["labels"] if label[0] == class_id)
        if added:
            augmented.append(sample)
            current += added
        attempts += 1
    return augmented


def draw_preview(sample):
    image = sample["image"].copy()
    for detection in sample["preview"]:
        class_id = detection["class_id"]
        color = CLASS_COLORS[class_id]
        x, y, width, height = detection["box"]
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        text = f"{CLASS_NAMES[class_id]} {detection['score']:.2f}"
        cv2.putText(
            image,
            text,
            (x, max(18, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def write_contact_sheet(samples, destination, columns=3, cell_width=426):
    selected = samples[:12]
    if not selected:
        return
    cells = []
    for sample in selected:
        preview = draw_preview(sample)
        scale = cell_width / preview.shape[1]
        cells.append(
            cv2.resize(
                preview,
                (cell_width, int(round(preview.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    cell_height = max(cell.shape[0] for cell in cells)
    rows = (len(cells) + columns - 1) // columns
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 25, np.uint8)
    for index, cell in enumerate(cells):
        row, column = divmod(index, columns)
        sheet[
            row * cell_height : row * cell_height + cell.shape[0],
            column * cell_width : column * cell_width + cell.shape[1],
        ] = cell
    save_image(destination, sheet)


def write_sample(sample, split, index, output):
    stem = f"{sample['kind']}_{index:04d}"
    image_path = output / "images" / split / f"{stem}.jpg"
    label_path = output / "labels" / split / f"{stem}.txt"
    save_image(image_path, sample["image"])
    lines = [
        yolo_line(class_id, box, sample["image"].shape)
        for class_id, box in sample["labels"]
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    return image_path


def main():
    args = parse_args()
    random_generator = random.Random(args.seed)
    output = args.output.resolve()
    if args.clean and output.exists():
        shutil.rmtree(output)
    elif output.exists():
        for generated in (output / "images", output / "labels"):
            if generated.exists():
                shutil.rmtree(generated)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    raw_capture_dir = output / "raw_capture"

    cfg = override_cfg(
        load_yaml("config/config_default.yaml"),
        load_yaml(f"config/config_{args.cfg}.yaml"),
    )
    capture_frames(
        cfg, raw_capture_dir, args.capture_seconds, args.capture_fps
    )
    captured = (
        []
        if args.no_reuse_captures
        else sorted(raw_capture_dir.glob("live_*.png"))
    )
    source_paths = unique_real_images(args.probe_dir, args.probe_glob) + captured
    labeler = DatasetLabeler(cfg)
    real_samples = [
        add_real_sample(path, labeler, index)
        for index, path in enumerate(source_paths)
    ]
    validation_indexes = choose_real_validation(real_samples)
    real_train = [
        sample for index, sample in enumerate(real_samples) if index not in validation_indexes
    ]
    real_val = [
        sample for index, sample in enumerate(real_samples) if index in validation_indexes
    ]
    blue_augmented = balance_rare_class(
        real_train, 1, args.min_blue_train_instances, random_generator
    )

    sprite_paths = sorted(Path("monster/black_axe_stump").glob("*.png"))
    sprites = [foreground_sprite(path) for path in sprite_paths]
    synthetic_samples = []
    backgrounds = real_train or real_samples
    if backgrounds and sprites:
        for index in range(args.synthetic_stumps):
            background = random_generator.choice(backgrounds)
            sprite = random_generator.choice(sprites)
            synthetic_samples.append(
                make_synthetic_stump(background, sprite, random_generator, index)
            )

    train_samples = real_train + blue_augmented + synthetic_samples
    val_samples = real_val
    random_generator.shuffle(train_samples)
    for index, sample in enumerate(train_samples):
        write_sample(sample, "train", index, output)
    for index, sample in enumerate(val_samples):
        write_sample(sample, "val", index, output)

    data_yaml = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )

    def class_counts(samples):
        counts = Counter()
        for sample in samples:
            counts.update(CLASS_NAMES[label[0]] for label in sample["labels"])
        return dict(counts)

    metadata = {
        "dataset": "maple_three_class_v1",
        "observe_only_capture": True,
        "class_names": list(CLASS_NAMES),
        "seed": args.seed,
        "sources": len(source_paths),
        "captured_frames": len(captured),
        "train_images": len(train_samples),
        "validation_images": len(val_samples),
        "synthetic_train_images": len(synthetic_samples),
        "blue_augmented_train_images": len(blue_augmented),
        "train_instances": class_counts(train_samples),
        "validation_instances": class_counts(val_samples),
        "validation_warning": (
            "Validation uses real frames from the same map/session family and is not "
            "an independent cross-map or cross-session test."
        ),
        "source_files": [str(path.resolve()) for path in source_paths],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_contact_sheet(real_samples, output / "real_labels_contact_sheet.jpg")
    write_contact_sheet(blue_augmented, output / "blue_augment_contact_sheet.jpg")
    write_contact_sheet(synthetic_samples, output / "synthetic_contact_sheet.jpg")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
