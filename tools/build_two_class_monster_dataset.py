"""Build a source-grouped YOLO dataset for the two cave mushrooms.

Train crops are derived only from sources assigned to the train split. Validation
and test images remain whole, independent source frames.
"""

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["zombie_mushroom", "thorn_mushroom"]


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, data = cv2.imencode(path.suffix or ".png", image)
    if not encoded:
        raise RuntimeError(f"Unable to encode image: {path}")
    data.tofile(path)


def resolve_source(manifest_path, source):
    path = Path(source)
    if path.is_absolute():
        return path
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative.resolve()
    return (REPO_ROOT / path).resolve()


def validate_manifest(entries):
    seen_ids = set()
    seen_sources = {}
    for entry in entries:
        source_id = entry["id"]
        split = entry["split"]
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source id: {source_id}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {source_id}: {split}")
        seen_ids.add(source_id)
        source_key = str(entry["resolved_source"]).lower()
        previous_split = seen_sources.get(source_key)
        if previous_split is not None and previous_split != split:
            raise ValueError(
                f"Source leakage: {entry['resolved_source']} appears in "
                f"{previous_split} and {split}"
            )
        seen_sources[source_key] = split
        for box in entry.get("boxes", []):
            if box["class"] not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class for {source_id}: {box['class']}"
                )
            if len(box["box"]) != 4 or min(box["box"][2:]) <= 0:
                raise ValueError(f"Invalid box for {source_id}: {box['box']}")


def load_manifest(path):
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["sources"]
    for entry in entries:
        entry["resolved_source"] = resolve_source(path, entry["source"])
    validate_manifest(entries)
    return payload, entries


def clip_boxes(boxes, crop, min_retained=0.85):
    crop_x, crop_y, crop_width, crop_height = crop
    output = []
    for item in boxes:
        x, y, width, height = [float(value) for value in item["box"]]
        x1 = max(x, crop_x)
        y1 = max(y, crop_y)
        x2 = min(x + width, crop_x + crop_width)
        y2 = min(y + height, crop_y + crop_height)
        retained = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        retained /= max(1.0, width * height)
        if retained < min_retained:
            continue
        output.append(
            {
                "class": item["class"],
                "box": [
                    x1 - crop_x,
                    y1 - crop_y,
                    x2 - x1,
                    y2 - y1,
                ],
            }
        )
    return output


def target_crop(target, image_width, image_height, rng):
    x, y, width, height = [float(value) for value in target["box"]]
    min_crop_width = min(520, image_width)
    max_crop_width = min(1000, image_width)
    min_crop_height = min(240, image_height)
    max_crop_height = min(687, image_height)
    crop_width = rng.randint(min_crop_width, max_crop_width)
    crop_height = rng.randint(min_crop_height, max_crop_height)
    center_x = x + width / 2.0 + rng.uniform(-0.20, 0.20) * crop_width
    center_y = y + height / 2.0 + rng.uniform(-0.20, 0.20) * crop_height
    crop_x = round(center_x - crop_width / 2.0)
    crop_y = round(center_y - crop_height / 2.0)
    crop_x = max(0, min(image_width - crop_width, crop_x))
    crop_y = max(0, min(image_height - crop_height, crop_y))
    return crop_x, crop_y, crop_width, crop_height


def write_label(path, boxes, image_width, image_height):
    lines = []
    for item in boxes:
        x, y, width, height = [float(value) for value in item["box"]]
        class_id = CLASS_NAMES.index(item["class"])
        lines.append(
            f"{class_id} {(x + width / 2.0) / image_width:.6f} "
            f"{(y + height / 2.0) / image_height:.6f} "
            f"{width / image_width:.6f} {height / image_height:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "\n" if lines else ""
    path.write_text("\n".join(lines) + suffix, encoding="ascii")


def write_sample(output, split, stem, image, boxes):
    image_path = output / "images" / split / f"{stem}.png"
    label_path = output / "labels" / split / f"{stem}.txt"
    save_image(image_path, image)
    write_label(label_path, boxes, image.shape[1], image.shape[0])


def build_train_samples(output, entry, image, boxes, rng, crops_per_box):
    count = 0
    write_sample(output, "train", f"{entry['id']}_full", image, boxes)
    count += 1
    for box_index, target in enumerate(boxes):
        for crop_index in range(crops_per_box):
            crop = target_crop(target, image.shape[1], image.shape[0], rng)
            crop_x, crop_y, crop_width, crop_height = crop
            crop_image = image[
                crop_y : crop_y + crop_height,
                crop_x : crop_x + crop_width,
            ]
            crop_boxes = clip_boxes(boxes, crop)
            stem = f"{entry['id']}_b{box_index:02d}_c{crop_index:02d}"
            write_sample(output, "train", stem, crop_image, crop_boxes)
            count += 1
    return count


def render_annotations(image, boxes):
    output = image.copy()
    colors = {
        "zombie_mushroom": (30, 170, 255),
        "thorn_mushroom": (255, 180, 40),
    }
    for item in boxes:
        x, y, width, height = [round(value) for value in item["box"]]
        color = colors[item["class"]]
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            output,
            item["class"],
            (x, max(16, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_data/two_class_real_v2_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/two_class_real_v2"),
    )
    parser.add_argument("--crops-per-box", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if args.crops_per_box < 0:
        raise ValueError("crops-per-box must be non-negative")

    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    payload, entries = load_manifest(manifest_path)
    rng = random.Random(args.seed)
    split_images = Counter()
    split_boxes = Counter()
    class_counts = {split: Counter() for split in ("train", "val", "test")}

    for entry in entries:
        source_path = entry["resolved_source"]
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        image = load_image(source_path)
        gameplay_height = min(
            int(entry.get("gameplay_height", image.shape[0])), image.shape[0]
        )
        image = image[:gameplay_height].copy()
        boxes = entry.get("boxes", [])
        source_copy = output / "sources" / f"{entry['id']}{source_path.suffix}"
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, source_copy)
        save_image(
            output / "annotation_previews" / f"{entry['id']}.png",
            render_annotations(image, boxes),
        )

        split = entry["split"]
        if split == "train":
            generated = build_train_samples(
                output, entry, image, boxes, rng, args.crops_per_box
            )
        else:
            write_sample(output, split, entry["id"], image, boxes)
            generated = 1
        split_images[split] += generated
        split_boxes[split] += len(boxes)
        class_counts[split].update(item["class"] for item in boxes)

    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "version": payload.get("version", 1),
        "classes": CLASS_NAMES,
        "seed": args.seed,
        "crops_per_train_box": args.crops_per_box,
        "source_groups": {
            split: [entry["id"] for entry in entries if entry["split"] == split]
            for split in ("train", "val", "test")
        },
        "generated_images": dict(split_images),
        "source_annotations": dict(split_boxes),
        "source_class_counts": {
            split: dict(class_counts[split]) for split in class_counts
        },
        "leakage_rule": (
            "Every original source frame belongs to exactly one split. "
            "Only train sources produce derived crops."
        ),
        "limitations": payload.get("limitations", []),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
