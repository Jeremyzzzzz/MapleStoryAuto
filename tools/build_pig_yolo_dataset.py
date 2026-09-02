"""Build a source-grouped, one-class dataset for offline pig perception."""

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
    render_annotations,
    resolve_source,
    save_image,
    target_crop,
    write_sample,
)


CLASS_NAME = "pig"


def load_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def load_manifest(path):
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("classes") != [CLASS_NAME]:
        raise ValueError(f"Manifest classes must be exactly [{CLASS_NAME!r}]")
    seen_ids = set()
    seen_sources = {}
    entries = payload.get("sources", [])
    if not entries:
        raise ValueError("Manifest must contain source entries")
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
    return payload, entries


def build_train_samples(output, entry, image, boxes, rng, crops_per_box):
    write_sample(output, "train", f"{entry['id']}_full", image, boxes)
    generated = 1
    for box_index, target in enumerate(boxes):
        for crop_index in range(crops_per_box):
            crop = target_crop(target, image.shape[1], image.shape[0], rng)
            crop_x, crop_y, crop_width, crop_height = crop
            crop_image = image[
                crop_y : crop_y + crop_height,
                crop_x : crop_x + crop_width,
            ].copy()
            crop_boxes = clip_boxes(boxes, crop)
            if rng.random() < 0.5:
                crop_image = cv2.flip(crop_image, 1)
                for item in crop_boxes:
                    item["box"][0] = crop_width - item["box"][0] - item["box"][2]
            stem = f"{entry['id']}_b{box_index:02d}_c{crop_index:02d}"
            write_sample(output, "train", stem, crop_image, crop_boxes)
            generated += 1
    return generated


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training_data/pig_yolo_v1_manifest.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("training_data/pig_yolo_v1"))
    parser.add_argument("--crops-per-box", type=int, default=6)
    parser.add_argument("--seed", type=int, default=241)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.crops_per_box < 0:
        raise ValueError("crops-per-box must be non-negative")
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    payload, entries = load_manifest(manifest_path)
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
        if split == "train" and boxes:
            generated = build_train_samples(output, entry, image, boxes, rng, args.crops_per_box)
        else:
            write_sample(output, split, entry["id"], image, boxes)
            generated = 1
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
