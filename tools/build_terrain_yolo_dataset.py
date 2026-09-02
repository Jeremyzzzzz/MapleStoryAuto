"""Build a three-class terrain dataset from one reviewed read-only frame."""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["ladder", "rope", "platform"]

# Reviewed on probe_output/new_mushroom_live_raw.png after cropping at y=687.
# Boxes are x, y, width, height, class name. A platform is one contiguous,
# visible, standable terrain segment rather than each rock/tile in that segment.
SOURCE_BOXES = [
    (493, 113, 53, 369, "ladder"),
    (987, 159, 62, 323, "ladder"),
    (583, 257, 14, 194, "rope"),
    (423, 116, 526, 171, "platform"),
    (955, 163, 163, 128, "platform"),
    (1117, 207, 253, 99, "platform"),
    (359, 442, 897, 104, "platform"),
    (1254, 485, 116, 91, "platform"),
]

COLORS = {
    "ladder": (40, 220, 255),
    "rope": (255, 170, 50),
    "platform": (80, 220, 90),
}


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


def choose_crop(target, image_width, image_height, rng, validation=False):
    x, y, width, height, _ = target
    minimum_width = min(image_width, max(420, round(width * 1.22)))
    minimum_height = min(image_height, max(340, round(height * 1.18)))
    if validation:
        crop_width = min(image_width, max(minimum_width, 1040))
        crop_height = min(image_height, max(minimum_height, 600))
        jitter_x = 0.0
        jitter_y = 0.0
    else:
        crop_width = rng.randint(minimum_width, image_width)
        crop_height = rng.randint(minimum_height, image_height)
        jitter_x = rng.uniform(-0.10, 0.10) * max(0, crop_width - width)
        jitter_y = rng.uniform(-0.08, 0.08) * max(0, crop_height - height)

    center_x = x + width / 2.0 + jitter_x
    center_y = y + height / 2.0 + jitter_y
    crop_x = round(center_x - crop_width / 2.0)
    crop_y = round(center_y - crop_height / 2.0)
    crop_x = max(0, min(image_width - crop_width, crop_x))
    crop_y = max(0, min(image_height - crop_height, crop_y))
    return crop_x, crop_y, crop_width, crop_height


def transform_sample(image, boxes, crop, rng, validation=False):
    crop_x, crop_y, crop_width, crop_height = crop
    sample = image[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ].copy()
    transformed = []
    for box in boxes:
        x, y, width, height, class_name = box
        retained = intersection_area(box, crop) / float(width * height)
        if retained < 0.80:
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

    output_width, output_height = 960, 540
    scale_x = output_width / float(crop_width)
    scale_y = output_height / float(crop_height)
    sample = cv2.resize(
        sample, (output_width, output_height), interpolation=cv2.INTER_LINEAR
    )
    for box in transformed:
        box[0] *= scale_x
        box[1] *= scale_y
        box[2] *= scale_x
        box[3] *= scale_y

    if not validation:
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4.0, 4.0)) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.80, 1.20), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.70, 1.24), 0, 255)
        sample = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.random() < 0.5:
            sample = cv2.flip(sample, 1)
            for box in transformed:
                box[0] = output_width - box[0] - box[2]
        if rng.random() < 0.22:
            kernel = rng.choice((3, 5))
            sample = cv2.GaussianBlur(sample, (kernel, kernel), 0)
        if rng.random() < 0.16:
            noise = np.random.default_rng(rng.randrange(2**32)).normal(
                0.0, rng.uniform(2.0, 6.0), sample.shape
            )
            sample = np.clip(
                sample.astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)
    return sample, transformed


def write_yolo_label(path, boxes, image_width=960, image_height=540):
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
    for x, y, width, height, class_name in boxes:
        point1 = round(x), round(y)
        point2 = round(x + width), round(y + height)
        color = COLORS[class_name]
        cv2.rectangle(output, point1, point2, color, 2)
        cv2.putText(
            output,
            class_name.upper(),
            (point1[0], max(16, point1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def build_split(output, source, split, samples_per_class, rng):
    targets_by_class = {
        class_name: [box for box in SOURCE_BOXES if box[4] == class_name]
        for class_name in CLASS_NAMES
    }
    previews = []
    for class_name in CLASS_NAMES:
        targets = targets_by_class[class_name]
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
            stem = f"{class_name}_current_map_{index:03d}"
            save_image(output / "images" / split / f"{stem}.jpg", image)
            write_yolo_label(
                output / "labels" / split / f"{stem}.txt", boxes
            )
            if index < 3:
                previews.append(render_boxes(image, boxes))
    return previews


def build_contact_sheet(previews):
    tiles = [cv2.resize(image, (480, 270)) for image in previews[:9]]
    rows = [np.hstack(tiles[index : index + 3]) for index in range(0, 9, 3)]
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("probe_output/new_mushroom_live_raw.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/terrain_three_class_v1"),
    )
    parser.add_argument("--gameplay-height", type=int, default=687)
    parser.add_argument("--train-per-class", type=int, default=160)
    parser.add_argument("--val-per-class", type=int, default=18)
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()

    source_path = args.source.resolve()
    output = args.output.resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    full_frame = load_image(source_path)
    if full_frame.shape[1] != 1370 or full_frame.shape[0] < args.gameplay_height:
        raise ValueError(
            f"Expected a 1370-wide frame at least {args.gameplay_height}px high, "
            f"got {full_frame.shape[1]}x{full_frame.shape[0]}"
        )
    source = full_frame[: args.gameplay_height].copy()

    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
        "sources",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output / "sources" / "current_map_full.png")
    save_image(output / "sources" / "current_map_gameplay.png", source)

    rng = random.Random(args.seed)
    previews = build_split(
        output, source, "train", args.train_per_class, rng
    )
    previews += build_split(
        output, source, "val", args.val_per_class, rng
    )
    save_image(output / "source_annotations.png", render_boxes(source, SOURCE_BOXES))
    save_image(output / "annotation_contact_sheet.jpg", build_contact_sheet(previews))

    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "classes": CLASS_NAMES,
        "source": str(source_path),
        "gameplay_size": [source.shape[1], source.shape[0]],
        "annotation_counts": {
            class_name: sum(box[4] == class_name for box in SOURCE_BOXES)
            for class_name in CLASS_NAMES
        },
        "train_images": args.train_per_class * len(CLASS_NAMES),
        "validation_images": args.val_per_class * len(CLASS_NAMES),
        "seed": args.seed,
        "platform_definition": (
            "One contiguous visible standable terrain segment per box."
        ),
        "validation_warning": (
            "Training and validation are augmented crops of one map frame; "
            "metrics only describe this map and are not cross-map evidence."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
