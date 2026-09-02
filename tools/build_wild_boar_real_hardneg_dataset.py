"""Build a real-frame and hard-negative wild-boar YOLO dataset.

Positive labels come from high-threshold masked matching against the four
exact in-game sprites.  Training also retains the v1 synthetic train split.
The captured sequence is split chronologically, so validation/test remain
useful diagnostics but are not independent live-game evidence.
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAME = "wild_boar"
REPO_ROOT = Path(__file__).resolve().parents[1]
SPRITE_DIR = REPO_ROOT / "monster" / CLASS_NAME
DEFAULT_CAPTURE = REPO_ROOT / "probe_output" / "wild_boar_real_capture_20260819_v1"
DEFAULT_SYNTHETIC = REPO_ROOT / "training_data" / "wild_boar_synth_v1"
DEFAULT_OUTPUT = REPO_ROOT / "training_data" / "wild_boar_real_hardneg_v4"
GAMEPLAY_HEIGHT = 687
TEMPLATE_THRESHOLD = 0.90

# These regions are UI or fixed map decorations observed to trigger v1.
HARD_NEGATIVE_REGIONS = (
    (0, 0, 165, 250, "minimap_ui"),
    (0, 300, 105, 520, "guide_ui"),
    (1165, 35, 1370, 235, "portal_decor"),
)


def load_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Unable to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def sprite_templates():
    templates = []
    for path in sorted(SPRITE_DIR.glob("*.png")):
        image = load_image(path)
        b, g, r = cv2.split(image)
        green = (
            (g > 120)
            & (g > r.astype(np.int16) + 25)
            & (g > b.astype(np.int16) + 25)
        )
        mask = (~green).astype(np.uint8) * 255
        ys, xs = np.where(mask > 0)
        if len(xs) < 30:
            continue
        image = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        for flipped in (False, True):
            sprite = cv2.flip(image, 1) if flipped else image
            sprite_mask = cv2.flip(mask, 1) if flipped else mask
            templates.append(
                {
                    "name": path.name,
                    "flipped": flipped,
                    "gray": cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY),
                    "mask": sprite_mask,
                }
            )
    if not templates:
        raise RuntimeError(f"No templates found in {SPRITE_DIR}")
    return templates


def box_iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    overlap_h = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = overlap_w * overlap_h
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


def template_labels(image, templates, threshold=TEMPLATE_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates = []
    for template in templates:
        sprite = template["gray"]
        mask = template["mask"]
        result = cv2.matchTemplate(gray, sprite, cv2.TM_CCORR_NORMED, mask=mask)
        result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
        height, width = sprite.shape
        while True:
            _, score, _, location = cv2.minMaxLoc(result)
            if score < threshold:
                break
            x, y = location
            candidates.append(
                {
                    "box": [x, y, width, height],
                    "score": float(score),
                    "template": template["name"],
                    "flipped": template["flipped"],
                }
            )
            result[
                max(0, y - height // 2) : min(result.shape[0], y + height // 2 + 1),
                max(0, x - width // 2) : min(result.shape[1], x + width // 2 + 1),
            ] = -1.0
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if all(box_iou(candidate["box"], item["box"]) < 0.30 for item in kept):
            kept.append(candidate)
    return kept


def crop_around(box, image_shape, rng):
    image_h, image_w = image_shape[:2]
    x, y, width, height = box
    crop_w = min(image_w, rng.randint(300, 560))
    crop_h = min(image_h, rng.randint(220, 410))
    center_x = x + width / 2 + rng.uniform(-0.18, 0.18) * crop_w
    center_y = y + height / 2 + rng.uniform(-0.18, 0.18) * crop_h
    crop_x = max(0, min(image_w - crop_w, int(round(center_x - crop_w / 2))))
    crop_y = max(0, min(image_h - crop_h, int(round(center_y - crop_h / 2))))
    return crop_x, crop_y, crop_w, crop_h


def labels_in_crop(labels, crop):
    crop_x, crop_y, crop_w, crop_h = crop
    clipped = []
    for item in labels:
        x, y, width, height = item["box"]
        x1 = max(x, crop_x)
        y1 = max(y, crop_y)
        x2 = min(x + width, crop_x + crop_w)
        y2 = min(y + height, crop_y + crop_h)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection < 0.55 * width * height:
            continue
        clipped.append([x1 - crop_x, y1 - crop_y, x2 - x1, y2 - y1])
    return clipped


def write_sample(output, split, stem, image, boxes):
    image_path = output / "images" / split / f"{stem}.jpg"
    label_path = output / "labels" / split / f"{stem}.txt"
    save_image(image_path, image)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    lines = []
    for x, y, box_w, box_h in boxes:
        center_x = (x + box_w / 2) / width
        center_y = (y + box_h / 2) / height
        lines.append(
            f"0 {center_x:.6f} {center_y:.6f} "
            f"{box_w / width:.6f} {box_h / height:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")


def copy_synthetic_train(source, output):
    copied = 0
    for image_path in sorted((source / "images" / "train").glob("*.jpg")):
        label_path = source / "labels" / "train" / f"{image_path.stem}.txt"
        target_stem = f"synth_v1_{image_path.stem}"
        target_image = output / "images" / "train" / f"{target_stem}.jpg"
        target_label = output / "labels" / "train" / f"{target_stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)
        copied += 1
    return copied


def scene_hard_negatives(output, rng, count=60):
    scene_dir = REPO_ROOT / "monster" / "测试集"
    scenes = [load_image(path) for path in sorted(scene_dir.glob("*.png"))]
    if not scenes:
        return 0
    written = 0
    for index in range(count):
        image = rng.choice(scenes)
        height, width = image.shape[:2]
        crop_w = rng.randint(min(260, width), width)
        crop_h = rng.randint(min(180, height), height)
        x = rng.randint(0, width - crop_w)
        y = rng.randint(0, height - crop_h)
        crop = image[y : y + crop_h, x : x + crop_w]
        write_sample(output, "train", f"scene_negative_{index:03d}", crop, [])
        written += 1
    return written


def source_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-frames", type=int, default=30)
    parser.add_argument("--val-frames", type=int, default=10)
    parser.add_argument("--crops-per-box", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1117)
    return parser.parse_args()


def main():
    args = parse_args()
    capture = args.capture if args.capture.is_absolute() else REPO_ROOT / args.capture
    synthetic = args.synthetic if args.synthetic.is_absolute() else REPO_ROOT / args.synthetic
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    capture, synthetic, output = capture.resolve(), synthetic.resolve(), output.resolve()
    frames = sorted(capture.glob("*.png"))
    if len(frames) <= args.train_frames + args.val_frames:
        raise ValueError("Capture must leave at least one test frame")
    if args.crops_per_box <= 0:
        raise ValueError("crops-per-box must be positive")
    if output.exists() and any(output.rglob("*")):
        raise FileExistsError(f"Output is not empty: {output}")

    rng = random.Random(args.seed)
    templates = sprite_templates()
    split_frames = {
        "train": frames[: args.train_frames],
        "val": frames[args.train_frames : args.train_frames + args.val_frames],
        "test": frames[args.train_frames + args.val_frames :],
    }
    sample_counts = Counter()
    box_counts = Counter()
    annotations = []

    synthetic_count = copy_synthetic_train(synthetic, output)
    sample_counts["train"] += synthetic_count
    for split, source_paths in split_frames.items():
        for source_index, source_path in enumerate(source_paths):
            image = load_image(source_path)[:GAMEPLAY_HEIGHT]
            labels = template_labels(image, templates)
            annotations.append(
                {
                    "source": str(source_path),
                    "sha256": source_sha256(source_path),
                    "split": split,
                    "detections": labels,
                }
            )
            full_boxes = [item["box"] for item in labels]
            write_sample(
                output,
                split,
                f"real_{source_index:03d}_full",
                image,
                full_boxes,
            )
            sample_counts[split] += 1
            box_counts[split] += len(full_boxes)
            for box_index, item in enumerate(labels):
                crop_total = args.crops_per_box if split == "train" else 1
                for crop_index in range(crop_total):
                    crop_box = crop_around(item["box"], image.shape, rng)
                    x, y, width, height = crop_box
                    crop = image[y : y + height, x : x + width].copy()
                    crop_labels = labels_in_crop(labels, crop_box)
                    if not crop_labels:
                        continue
                    if split == "train" and rng.random() < 0.5:
                        crop = cv2.flip(crop, 1)
                        for box in crop_labels:
                            box[0] = width - box[0] - box[2]
                    stem = f"real_{source_index:03d}_b{box_index:02d}_c{crop_index:02d}"
                    write_sample(output, split, stem, crop, crop_labels)
                    sample_counts[split] += 1
                    box_counts[split] += len(crop_labels)

            if split == "train":
                for region_index, (x1, y1, x2, y2, name) in enumerate(HARD_NEGATIVE_REGIONS):
                    x2 = min(x2, image.shape[1])
                    y2 = min(y2, image.shape[0])
                    if x1 >= x2 or y1 >= y2:
                        continue
                    region = image[y1:y2, x1:x2].copy()
                    if any(
                        box_iou([x1, y1, x2 - x1, y2 - y1], item["box"]) > 0.02
                        for item in labels
                    ):
                        continue
                    write_sample(
                        output,
                        split,
                        f"hardneg_{source_index:03d}_{region_index}_{name}",
                        region,
                        [],
                    )
                    sample_counts[split] += 1

    scene_negative_count = scene_hard_negatives(output, rng)
    sample_counts["train"] += scene_negative_count
    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: CLASS_NAME},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "version": 4,
        "class": CLASS_NAME,
        "seed": args.seed,
        "template_threshold": TEMPLATE_THRESHOLD,
        "capture": str(capture),
        "source_frame_split": {
            split: [path.name for path in paths] for split, paths in split_frames.items()
        },
        "generated_samples": dict(sample_counts),
        "generated_real_boxes": dict(box_counts),
        "retained_synthetic_train_samples": synthetic_count,
        "scene_hard_negative_samples": scene_negative_count,
        "hard_negative_regions": [list(region) for region in HARD_NEGATIVE_REGIONS],
        "annotation_method": "Masked exact-sprite matching; threshold >= 0.90.",
        "limitations": [
            "Chronological splits come from one adjacent live capture sequence and are not independent OOS evidence.",
            "Template labels prioritize precision and can omit partial or heavily occluded wild boars.",
            "Hard-negative regions are specific to observed UI and fixed decorations in this client/map.",
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "source_annotations.json").write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
