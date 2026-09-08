"""Build the source-grouped single-class crocodile dataset."""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_warrior_stump_dataset import (  # noqa: E402
    clip_boxes,
    resolve_source,
    save_image,
    target_crop,
    write_sample,
)

CLASS_NAME = "crocodile"


def load_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def load_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("classes") != [CLASS_NAME]:
        raise ValueError(f"Manifest classes must be exactly [{CLASS_NAME!r}]")
    entries = payload.get("sources", [])
    if not entries:
        raise ValueError("Manifest must contain source entries")
    seen_ids = set()
    seen_sources = {}
    for entry in entries:
        source_id = entry["id"]
        split = entry["split"]
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source id: {source_id}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {source_id}: {split}")
        entry["resolved_source"] = resolve_source(path, entry["source"])
        source_key = str(entry["resolved_source"]).lower()
        previous_split = seen_sources.get(source_key)
        if previous_split is not None and previous_split != split:
            raise ValueError(
                f"Source leakage: {entry['resolved_source']} appears in "
                f"{previous_split} and {split}"
            )
        seen_ids.add(source_id)
        seen_sources[source_key] = split
        for item in entry.get("boxes", []):
            if item.get("class") != CLASS_NAME:
                raise ValueError(f"Invalid class in {source_id}: {item}")
            box = item.get("box", [])
            if len(box) != 4 or min(box[2:]) <= 0:
                raise ValueError(f"Invalid box in {source_id}: {box}")
        for crop in entry.get("negative_crops", []):
            if len(crop) != 4 or min(crop[2:]) <= 0:
                raise ValueError(f"Invalid negative crop in {source_id}: {crop}")
    return payload, entries


def merge_manifest_entries(entries, extra_entries):
    """Merge manifests while retaining source-group and id isolation."""
    merged = list(entries)
    seen_ids = {entry["id"] for entry in merged}
    seen_sources = {
        str(entry["resolved_source"]).lower(): entry["split"] for entry in merged
    }
    for entry in extra_entries:
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate source id across manifests: {entry['id']}")
        source_key = str(entry["resolved_source"]).lower()
        previous_split = seen_sources.get(source_key)
        if previous_split is not None and previous_split != entry["split"]:
            raise ValueError(
                f"Source leakage across manifests: {entry['resolved_source']} appears in "
                f"{previous_split} and {entry['split']}"
            )
        seen_ids.add(entry["id"])
        seen_sources[source_key] = entry["split"]
        merged.append(entry)
    return merged


def render_annotations(image, boxes):
    output = image.copy()
    for index, item in enumerate(boxes, 1):
        x, y, width, height = [round(value) for value in item["box"]]
        cv2.rectangle(output, (x, y), (x + width, y + height), (80, 210, 210), 2)
        cv2.putText(
            output,
            f"C{index}",
            (x, max(16, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (80, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return output


def clip_crop(crop, image_width, image_height):
    """Clip an explicit hard-negative crop to the gameplay image."""
    crop_x, crop_y, crop_width, crop_height = [int(round(value)) for value in crop]
    crop_x = max(0, min(image_width, crop_x))
    crop_y = max(0, min(image_height, crop_y))
    crop_x2 = max(crop_x, min(image_width, crop_x + max(0, crop_width)))
    crop_y2 = max(crop_y, min(image_height, crop_y + max(0, crop_height)))
    if crop_x2 <= crop_x or crop_y2 <= crop_y:
        return None
    return crop_x, crop_y, crop_x2 - crop_x, crop_y2 - crop_y


def build_negative_samples(output, entry, image, split):
    """Write explicit background-only crops with empty YOLO labels."""
    generated = 0
    image_height, image_width = image.shape[:2]
    for crop_index, crop_spec in enumerate(entry.get("negative_crops", [])):
        crop = clip_crop(crop_spec, image_width, image_height)
        if crop is None:
            raise ValueError(f"Invalid negative crop in {entry['id']}: {crop_spec}")
        crop_x, crop_y, crop_width, crop_height = crop
        crop_image = image[
            crop_y : crop_y + crop_height,
            crop_x : crop_x + crop_width,
        ].copy()
        stem = f"{entry['id']}_negative_{crop_index:02d}"
        write_sample(output, split, stem, crop_image, [])
        generated += 1
        if entry.get("negative_flip", True):
            write_sample(output, split, f"{stem}_flip", cv2.flip(crop_image, 1), [])
            generated += 1
    return generated


def build_train_samples(output, entry, image, boxes, rng, crops_per_box):
    write_sample(output, "train", f"{entry['id']}_full", image, boxes)
    generated = 1
    for box_index, target in enumerate(boxes):
        for crop_index in range(crops_per_box):
            crop = target_crop(target, image.shape[1], image.shape[0], rng)
            crop_x, crop_y, crop_width, crop_height = crop
            crop_image = image[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width].copy()
            crop_boxes = clip_boxes(boxes, crop)
            if rng.random() < 0.5:
                crop_image = cv2.flip(crop_image, 1)
                for item in crop_boxes:
                    item["box"][0] = crop_width - item["box"][0] - item["box"][2]
            stem = f"{entry['id']}_b{box_index:02d}_c{crop_index:02d}"
            write_sample(output, "train", stem, crop_image, crop_boxes)
            generated += 1
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("training_data/crocodile_v1_manifest.json"))
    parser.add_argument(
        "--extra-manifest",
        type=Path,
        action="append",
        default=[],
        help="Additional source-grouped manifest, typically hard negatives.",
    )
    parser.add_argument("--output", type=Path, default=Path("training_data/crocodile_v1"))
    parser.add_argument("--crops-per-box", type=int, default=5)
    parser.add_argument("--seed", type=int, default=907)
    args = parser.parse_args()
    if args.crops_per_box < 0:
        raise ValueError("crops-per-box must be non-negative")

    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    payload, entries = load_manifest(manifest_path)
    for extra_manifest in args.extra_manifest:
        extra_payload, extra_entries = load_manifest(extra_manifest.resolve())
        if extra_payload.get("classes") != payload.get("classes"):
            raise ValueError("All manifests must declare the same classes")
        entries = merge_manifest_entries(entries, extra_entries)
    rng = random.Random(args.seed)
    split_images = Counter()
    split_boxes = Counter()
    for entry in entries:
        source_path = entry["resolved_source"]
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        frame = load_image(source_path)
        gameplay_height = min(int(entry.get("gameplay_height", frame.shape[0])), frame.shape[0])
        image = frame[:gameplay_height].copy()
        boxes = entry.get("boxes", [])
        source_copy = output / "sources" / f"{entry['id']}{source_path.suffix}"
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, source_copy)
        save_image(output / "annotation_previews" / f"{entry['id']}.png", render_annotations(image, boxes))
        split = entry["split"]
        include_full = bool(entry.get("include_full", True))
        if split == "train" and boxes:
            generated = build_train_samples(output, entry, image, boxes, rng, args.crops_per_box)
        elif include_full:
            write_sample(output, split, entry["id"], image, boxes)
            generated = 1
        else:
            generated = 0
        generated += build_negative_samples(output, entry, image, split)
        split_images[split] += generated
        split_boxes[split] += len(boxes)

    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: CLASS_NAME},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    metadata = {
        "version": payload.get("version", 1),
        "classes": [CLASS_NAME],
        "seed": args.seed,
        "crops_per_train_box": args.crops_per_box,
        "source_groups": {
            split: [entry["id"] for entry in entries if entry["split"] == split]
            for split in ("train", "val", "test")
        },
        "generated_images": dict(split_images),
        "source_annotations": dict(split_boxes),
        "leakage_rule": "Each raw frame belongs to one split; only train sources create crops.",
        "limitations": payload.get("limitations", []),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
