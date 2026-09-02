"""Build a seven-class YOLO dataset by adding two reviewed mushroom classes."""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DATASET = REPO_ROOT / "training_data" / "five_class_real_v3"
CLASS_NAMES = [
    "slime",
    "red_snail",
    "green_mushroom",
    "stump",
    "flower_mushroom",
    "zombie_mushroom",
    "thorn_mushroom",
]

# Reviewed on the supplied 778x156 PNG. Boxes are x, y, width, height, class.
SOURCE_BOXES = [
    (17, 18, 106, 138, "zombie_mushroom"),
    (95, 14, 133, 142, "zombie_mushroom"),
    (388, 27, 108, 129, "thorn_mushroom"),
    (484, 15, 141, 141, "zombie_mushroom"),
    (636, 25, 111, 131, "thorn_mushroom"),
]


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, data = cv2.imencode(path.suffix or ".jpg", image)
    if not encoded:
        raise RuntimeError(f"Unable to encode image: {path}")
    data.tofile(path)


def intersection_area(box, crop):
    x, y, width, height, _ = box
    crop_x, crop_y, crop_width, crop_height = crop
    overlap_width = max(
        0, min(x + width, crop_x + crop_width) - max(x, crop_x)
    )
    overlap_height = max(
        0, min(y + height, crop_y + crop_height) - max(y, crop_y)
    )
    return overlap_width * overlap_height


def choose_crop(target, image_width, image_height, rng, validation):
    x, _, width, _, _ = target
    target_center_x = x + width / 2.0
    if validation:
        crop_width = min(image_width, 520)
        jitter_x = 0.0
    else:
        crop_width = min(image_width, rng.randint(360, 778))
        jitter_x = rng.uniform(-crop_width * 0.18, crop_width * 0.18)
    crop_x = round(target_center_x - crop_width / 2.0 + jitter_x)
    crop_x = max(0, min(image_width - crop_width, crop_x))
    return crop_x, 0, crop_width, image_height


def transform_sample(image, boxes, crop, rng, validation=False):
    crop_x, crop_y, crop_width, crop_height = crop
    sample = image[
        crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
    ].copy()
    transformed = []
    for box in boxes:
        x, y, width, height, class_name = box
        retained = intersection_area(box, crop) / float(width * height)
        if retained < 0.75:
            continue
        x1 = max(x, crop_x)
        y1 = max(y, crop_y)
        x2 = min(x + width, crop_x + crop_width)
        y2 = min(y + height, crop_y + crop_height)
        transformed.append(
            [
                float(x1 - crop_x),
                float(y1 - crop_y),
                float(x2 - x1),
                float(y2 - y1),
                class_name,
            ]
        )

    if not validation:
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5.0, 5.0)) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.78, 1.22), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.68, 1.25), 0, 255)
        sample = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.random() < 0.5:
            sample = cv2.flip(sample, 1)
            for box in transformed:
                box[0] = crop_width - box[0] - box[2]

        if rng.random() < 0.24:
            kernel = rng.choice((3, 5))
            sample = cv2.GaussianBlur(sample, (kernel, kernel), 0)

        if rng.random() < 0.20:
            noise = np.random.default_rng(rng.randrange(2**32)).normal(
                0.0, rng.uniform(2.0, 7.0), sample.shape
            )
            sample = np.clip(
                sample.astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)

        # Simulate a small foreground obstruction while keeping the full label.
        if transformed and rng.random() < 0.35:
            target_box = rng.choice(transformed)
            x, y, width, height, _ = target_box
            obstruction_width = max(5, round(width * rng.uniform(0.08, 0.22)))
            obstruction_height = max(5, round(height * rng.uniform(0.16, 0.35)))
            obstruction_x = round(x + rng.uniform(0.15, 0.75) * width)
            obstruction_y = round(y + rng.uniform(0.35, 0.75) * height)
            obstruction_x = min(sample.shape[1] - 1, max(0, obstruction_x))
            obstruction_y = min(sample.shape[0] - 1, max(0, obstruction_y))
            obstruction_x2 = min(
                sample.shape[1], obstruction_x + obstruction_width
            )
            obstruction_y2 = min(
                sample.shape[0], obstruction_y + obstruction_height
            )
            color = tuple(
                int(value)
                for value in sample[
                    max(0, obstruction_y - 2), max(0, obstruction_x - 2)
                ]
            )
            cv2.rectangle(
                sample,
                (obstruction_x, obstruction_y),
                (obstruction_x2, obstruction_y2),
                color,
                -1,
            )

    return sample, transformed


def write_yolo_label(path, boxes, image_width, image_height):
    lines = []
    for x, y, width, height, class_name in boxes:
        class_id = CLASS_NAMES.index(class_name)
        center_x = (x + width / 2.0) / image_width
        center_y = (y + height / 2.0) / image_height
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{width / image_width:.6f} {height / image_height:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def render_boxes(image, boxes):
    output = image.copy()
    colors = {
        "zombie_mushroom": (30, 170, 255),
        "thorn_mushroom": (255, 180, 40),
    }
    for x, y, width, height, class_name in boxes:
        point1 = round(x), round(y)
        point2 = round(x + width), round(y + height)
        color = colors[class_name]
        cv2.rectangle(output, point1, point2, color, 2)
        cv2.putText(
            output,
            class_name,
            (point1[0], max(14, point1[1] - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def build_split(output, source, split, samples_per_class, rng):
    targets_by_class = {
        class_name: [
            box for box in SOURCE_BOXES if box[4] == class_name
        ]
        for class_name in ("zombie_mushroom", "thorn_mushroom")
    }
    previews = []
    for class_name, targets in targets_by_class.items():
        for index in range(samples_per_class):
            target = targets[index % len(targets)]
            crop = choose_crop(
                target,
                source.shape[1],
                source.shape[0],
                rng,
                validation=split == "val",
            )
            image, boxes = transform_sample(
                source,
                SOURCE_BOXES,
                crop,
                rng,
                validation=split == "val",
            )
            stem = f"{class_name}_supplied_scene_{index:03d}"
            image_path = output / "images" / split / f"{stem}.jpg"
            label_path = output / "labels" / split / f"{stem}.txt"
            save_image(image_path, image)
            write_yolo_label(
                label_path, boxes, image.shape[1], image.shape[0]
            )
            if index < 4:
                previews.append(render_boxes(image, boxes))
    return previews


def build_contact_sheet(previews):
    tiles = []
    for image in previews:
        scale = 320.0 / image.shape[1]
        height = max(64, round(image.shape[0] * scale))
        resized = cv2.resize(image, (320, height))
        canvas = np.zeros((180, 320, 3), dtype=np.uint8)
        y = (canvas.shape[0] - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0]] = resized
        tiles.append(canvas)
    rows = [np.hstack(tiles[index : index + 4]) for index in range(0, 8, 4)]
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/seven_class_increment_v1"),
    )
    parser.add_argument("--train-per-class", type=int, default=140)
    parser.add_argument("--val-per-class", type=int, default=16)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    source_path = args.source.resolve()
    output = args.output.resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not (BASE_DATASET / "data.yaml").exists():
        raise FileNotFoundError(BASE_DATASET / "data.yaml")

    source = load_image(source_path)
    if source.shape[:2] != (156, 778):
        raise ValueError(
            f"Expected supplied source size 778x156, got "
            f"{source.shape[1]}x{source.shape[0]}"
        )

    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
        "sources",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)
    copied_source = output / "sources" / "new_mushrooms.png"
    shutil.copy2(source_path, copied_source)

    rng = random.Random(args.seed)
    previews = build_split(
        output, source, "train", args.train_per_class, rng
    )
    previews += build_split(
        output, source, "val", args.val_per_class, rng
    )
    save_image(output / "annotation_contact_sheet.jpg", build_contact_sheet(previews))
    save_image(output / "source_annotations.png", render_boxes(source, SOURCE_BOXES))

    data = {
        "train": [
            str((BASE_DATASET / "images" / "train").resolve()),
            str((output / "images" / "train").resolve()),
        ],
        "val": [
            str((BASE_DATASET / "images" / "val").resolve()),
            str((output / "images" / "val").resolve()),
        ],
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )

    metadata = {
        "classes": CLASS_NAMES,
        "base_dataset": str(BASE_DATASET.resolve()),
        "source": str(copied_source.resolve()),
        "source_size": [source.shape[1], source.shape[0]],
        "source_annotations": len(SOURCE_BOXES),
        "annotation_counts": {
            class_name: sum(box[4] == class_name for box in SOURCE_BOXES)
            for class_name in ("zombie_mushroom", "thorn_mushroom")
        },
        "new_train_images": args.train_per_class * 2,
        "new_validation_images": args.val_per_class * 2,
        "seed": args.seed,
        "validation_warning": (
            "The two new classes use one supplied scene for both augmented "
            "training and validation; results are not cross-map validation."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
