"""Build a five-class YOLO dataset from read-only, manually reviewed frames."""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "slime",
    "red_snail",
    "green_mushroom",
    "stump",
    "flower_mushroom",
]

# Boxes are (x, y, width, height, class_name). They were reviewed against the
# source PNGs; heavily occluded monsters are intentionally omitted.
SOURCE_ANNOTATIONS = {
    "legacy_scene": {
        "path": REPO_ROOT / "monster" / "测试集" / "场景1.png",
        "boxes": [
            (600, 78, 51, 47, "slime"),
            (721, 80, 53, 47, "slime"),
            (267, 300, 45, 45, "red_snail"),
            (324, 283, 50, 64, "stump"),
            (370, 303, 37, 43, "red_snail"),
            (393, 284, 52, 64, "stump"),
            (432, 285, 57, 64, "green_mushroom"),
            (857, 277, 58, 72, "flower_mushroom"),
        ],
    },
    "current_scene": {
        "path": REPO_ROOT
        / "training_data"
        / "five_class_real_v1"
        / "sources"
        / "current_scene.png",
        "boxes": [
            (215, 83, 62, 58, "flower_mushroom"),
            (797, 193, 66, 59, "flower_mushroom"),
            (1028, 195, 59, 59, "stump"),
            (1133, 205, 61, 51, "slime"),
            (1232, 185, 46, 62, "flower_mushroom"),
            (439, 345, 57, 58, "slime"),
            (249, 424, 45, 49, "slime"),
            (272, 422, 55, 52, "slime"),
            (437, 424, 54, 50, "slime"),
            (480, 424, 55, 50, "slime"),
            (714, 422, 57, 49, "slime"),
            (956, 411, 72, 64, "flower_mushroom"),
            (1177, 416, 60, 57, "slime"),
        ],
    },
    "hard_scene": {
        "path": REPO_ROOT / "probe_output" / "hard_classes_current_raw.png",
        "boxes": [
            (149, 214, 40, 39, "red_snail"),
            (174, 211, 42, 43, "red_snail"),
            (381, 202, 60, 64, "stump"),
            (542, 199, 64, 63, "flower_mushroom"),
            (45, 421, 58, 63, "stump"),
            (366, 423, 62, 57, "flower_mushroom"),
            (59, 645, 64, 43, "slime"),
            (334, 628, 66, 59, "slime"),
        ],
    },
    "stump_recall_scene": {
        "path": REPO_ROOT / "probe_output" / "stump_recall_raw.png",
        "boxes": [
            (296, 207, 43, 45, "red_snail"),
            (535, 188, 68, 65, "flower_mushroom"),
            (966, 190, 69, 64, "stump"),
            (266, 416, 60, 58, "slime"),
            (319, 411, 62, 63, "flower_mushroom"),
            (956, 348, 71, 66, "flower_mushroom"),
            (1068, 406, 67, 64, "flower_mushroom"),
        ],
    },
}

GREEN_MUSHROOM_SPRITE = (
    REPO_ROOT / "monster" / "green_mushroom" / "green_mushroom.png"
)


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


def choose_crop(image, target, rng, validation=False):
    image_height, image_width = image.shape[:2]
    target_x, target_y, target_width, target_height, _ = target
    target_center_x = target_x + target_width / 2.0
    target_center_y = target_y + target_height / 2.0

    if validation:
        crop_width = min(image_width, 720)
    else:
        crop_width = min(image_width, rng.randint(480, 820))
    crop_height = min(image_height, max(300, round(crop_width * 0.75)))

    jitter_x = 0 if validation else rng.uniform(-crop_width * 0.22, crop_width * 0.22)
    jitter_y = 0 if validation else rng.uniform(-crop_height * 0.18, crop_height * 0.18)
    crop_x = round(target_center_x - crop_width / 2.0 + jitter_x)
    crop_y = round(target_center_y - crop_height / 2.0 + jitter_y)
    crop_x = max(0, min(image_width - crop_width, crop_x))
    crop_y = max(0, min(image_height - crop_height, crop_y))
    return (crop_x, crop_y, crop_width, crop_height)


def crop_and_transform(image, boxes, crop, rng, validation=False):
    crop_x, crop_y, crop_width, crop_height = crop
    cropped = image[
        crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
    ].copy()
    transformed = []
    for box in boxes:
        x, y, width, height, class_name = box
        retained = intersection_area(box, crop) / float(width * height)
        if retained < 0.75:
            continue
        clipped_x1 = max(x, crop_x)
        clipped_y1 = max(y, crop_y)
        clipped_x2 = min(x + width, crop_x + crop_width)
        clipped_y2 = min(y + height, crop_y + crop_height)
        transformed.append(
            [
                clipped_x1 - crop_x,
                clipped_y1 - crop_y,
                clipped_x2 - clipped_x1,
                clipped_y2 - clipped_y1,
                class_name,
            ]
        )

    output_width, output_height = 640, 480
    scale_x = output_width / float(crop_width)
    scale_y = output_height / float(crop_height)
    cropped = cv2.resize(
        cropped, (output_width, output_height), interpolation=cv2.INTER_LINEAR
    )
    for box in transformed:
        box[0] *= scale_x
        box[1] *= scale_y
        box[2] *= scale_x
        box[3] *= scale_y

    if not validation:
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4.0, 4.0)) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.88, 1.12), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.82, 1.18), 0, 255)
        cropped = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if rng.random() < 0.5:
            cropped = cv2.flip(cropped, 1)
            for box in transformed:
                box[0] = output_width - box[0] - box[2]
    return cropped, transformed


def write_yolo_label(path, boxes, image_width=640, image_height=480):
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
        "slime": (80, 230, 130),
        "red_snail": (70, 70, 255),
        "green_mushroom": (40, 190, 40),
        "stump": (0, 105, 255),
        "flower_mushroom": (210, 80, 230),
    }
    for x, y, width, height, class_name in boxes:
        color = colors[class_name]
        pt1 = (round(x), round(y))
        pt2 = (round(x + width), round(y + height))
        cv2.rectangle(output, pt1, pt2, color, 2)
        cv2.putText(
            output,
            class_name,
            (pt1[0], max(16, pt1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def extract_green_screen_sprite(path):
    sprite = load_image(path)
    background = (
        (sprite[:, :, 1] >= 245)
        & (sprite[:, :, 0] <= 12)
        & (sprite[:, :, 2] <= 12)
    )
    foreground = (~background).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    rows, columns = np.where(foreground > 0)
    if len(rows) == 0:
        raise RuntimeError(f"No foreground extracted from {path}")
    x1, x2 = columns.min(), columns.max() + 1
    y1, y2 = rows.min(), rows.max() + 1
    return sprite[y1:y2, x1:x2], foreground[y1:y2, x1:x2]


def paste_sprite(background, sprite, mask, x, ground_y, height, flip, angle):
    scale = height / float(sprite.shape[0])
    width = max(8, round(sprite.shape[1] * scale))
    resized_sprite = cv2.resize(
        sprite, (width, height), interpolation=cv2.INTER_AREA
    )
    resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
    if flip:
        resized_sprite = cv2.flip(resized_sprite, 1)
        resized_mask = cv2.flip(resized_mask, 1)
    rotation = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    resized_sprite = cv2.warpAffine(
        resized_sprite,
        rotation,
        (width, height),
        borderValue=(0, 0, 0),
    )
    resized_mask = cv2.warpAffine(
        resized_mask, rotation, (width, height), borderValue=0
    )
    y = ground_y - height
    alpha = resized_mask.astype(np.float32)[:, :, None] / 255.0
    region = background[y : y + height, x : x + width].astype(np.float32)
    blended = resized_sprite.astype(np.float32) * alpha + region * (1.0 - alpha)
    background[y : y + height, x : x + width] = blended.astype(np.uint8)
    return [x, y, width, height, "green_mushroom"]


def build_green_mushroom_composites(output, hard_scene, count, rng):
    sprite, mask = extract_green_screen_sprite(GREEN_MUSHROOM_SPRITE)
    # The upper-right section of hard_scene is a clean real platform with no
    # monsters. Its ground line is y=255 in the source frame.
    base = hard_scene[0:480, 638:1278].copy()
    samples = []
    for index in range(count):
        image = base.copy()
        height = rng.randint(38, 61)
        x = rng.randint(35, max(36, image.shape[1] - height - 50))
        box = paste_sprite(
            image,
            sprite,
            mask,
            x=x,
            ground_y=rng.randint(251, 258),
            height=height,
            flip=rng.random() < 0.5,
            angle=rng.uniform(-5.0, 5.0),
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.92, 1.08), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.88, 1.12), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        stem = f"green_mushroom_composite_{index:03d}"
        image_path = output / "images" / "train" / f"{stem}.jpg"
        label_path = output / "labels" / "train" / f"{stem}.txt"
        save_image(image_path, image)
        write_yolo_label(label_path, [box])
        if index < 5:
            samples.append(render_boxes(image, [box]))
    return samples


def build_split(output, sources, split, samples_per_class, rng):
    targets_by_class = {name: [] for name in CLASS_NAMES}
    for source_name, source in sources.items():
        for box in source["boxes"]:
            targets_by_class[box[4]].append((source_name, box))

    samples = []
    for class_name in CLASS_NAMES:
        targets = targets_by_class[class_name]
        if not targets:
            raise RuntimeError(f"No source annotations for class: {class_name}")
        for index in range(samples_per_class):
            source_name, target = targets[index % len(targets)]
            source = sources[source_name]
            validation = split == "val"
            crop = choose_crop(source["image"], target, rng, validation)
            image, boxes = crop_and_transform(
                source["image"], source["boxes"], crop, rng, validation
            )
            stem = f"{class_name}_{source_name}_{index:03d}"
            image_path = output / "images" / split / f"{stem}.jpg"
            label_path = output / "labels" / split / f"{stem}.txt"
            save_image(image_path, image)
            write_yolo_label(label_path, boxes)
            if index < 2:
                samples.append(render_boxes(image, boxes))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/five_class_real_v1"),
    )
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--green-composites", type=int, default=180)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    output = args.output.resolve()
    sources = {}
    for source_name, source in SOURCE_ANNOTATIONS.items():
        if not source["path"].exists():
            raise FileNotFoundError(source["path"])
        sources[source_name] = {
            "image": load_image(source["path"]),
            "boxes": source["boxes"],
        }

    rng = random.Random(args.seed)
    train_samples = build_split(
        output, sources, "train", args.train_per_class, rng
    )
    val_samples = build_split(output, sources, "val", args.val_per_class, rng)
    green_samples = build_green_mushroom_composites(
        output,
        sources["hard_scene"]["image"],
        args.green_composites,
        rng,
    )

    contact_samples = train_samples + val_samples + green_samples
    rows = []
    for index in range(0, len(contact_samples), 5):
        rows.append(np.hstack(contact_samples[index : index + 5]))
    contact_sheet = np.vstack(rows)
    save_image(output / "annotation_contact_sheet.jpg", contact_sheet)

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
        "sources": {
            name: {
                "path": str(SOURCE_ANNOTATIONS[name]["path"]),
                "annotations": len(source["boxes"]),
            }
            for name, source in sources.items()
        },
        "train_images": (
            args.train_per_class * len(CLASS_NAMES) + args.green_composites
        ),
        "validation_images": args.val_per_class * len(CLASS_NAMES),
        "green_mushroom_composites": args.green_composites,
        "seed": args.seed,
        "validation_warning": (
            "Validation crops use the same source scenes as training and "
            "are not an independent cross-map evaluation."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
