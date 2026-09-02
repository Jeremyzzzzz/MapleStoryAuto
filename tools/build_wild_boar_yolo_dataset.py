"""Build a one-class synthetic YOLO dataset for the MapleStory wild boar.

The source sprites are green-screen PNGs.  Poses are kept source-grouped so
the holdout split is not just another crop of the same sprite.  Background
crops are randomized, but the repository currently has only one real scene
source, so they are not an independent live-game holdout.
This dataset is an offline bootstrap only; it is not a substitute for
source-separated screenshots captured in the wild-boar map.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_NAME = "wild_boar"
REPO_ROOT = Path(__file__).resolve().parents[1]
SPRITE_DIR = REPO_ROOT / "monster" / CLASS_NAME
SCENE_DIR = REPO_ROOT / "monster" / "测试集"

IMG_W, IMG_H = 640, 384
SEED = 713
SPRITES = {
    "train": ["wild_boar_1.png", "wild_boar_5.png"],
    "val": ["wild_boar_7.png"],
    "test": ["wild_boar_9.png"],
}


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def extract_sprite(path):
    """Remove the bright green screen and return a tight BGR sprite/mask."""
    image = read_image(path)
    b, g, r = cv2.split(image)
    green = (g > 120) & (g > (r.astype(np.int16) + 25)) & (g > (b.astype(np.int16) + 25))
    mask = (~green).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        # Keep the main sprite and any immediately adjacent anti-aliased pixels.
        main = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        mask = (labels == main).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        raise ValueError(f"No usable foreground extracted from {path}")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return image[y0 : y1 + 1, x0 : x1 + 1].copy(), mask[y0 : y1 + 1, x0 : x1 + 1].copy()


def scene_backgrounds(rng):
    backgrounds = []
    for source in sorted(SCENE_DIR.glob("*.png")):
        image = read_image(source)
        h, w = image.shape[:2]
        # Random crops keep the game art while avoiding one fixed composition.
        for _ in range(30):
            crop_w = rng.randint(max(320, int(w * 0.68)), w)
            crop_h = rng.randint(max(240, int(h * 0.72)), h)
            x = rng.randint(0, max(0, w - crop_w))
            y = rng.randint(0, max(0, h - crop_h))
            crop = image[y : y + crop_h, x : x + crop_w]
            backgrounds.append(cv2.resize(crop, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA))
    # Smooth, varied backgrounds provide clean negatives and prevent memorizing
    # the single available screenshot.
    for _ in range(50):
        base = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        c0 = np.array([rng.randint(25, 175) for _ in range(3)], dtype=np.float32)
        c1 = np.array([rng.randint(40, 220) for _ in range(3)], dtype=np.float32)
        for y in range(IMG_H):
            t = y / max(IMG_H - 1, 1)
            base[y, :] = np.clip(c0 * (1.0 - t) + c1 * t, 0, 255)
        noise = rng.randint(0, 9)
        if noise:
            base = np.clip(base.astype(np.int16) + rng.choices(range(-noise, noise + 1), k=IMG_W * IMG_H * 3)[0], 0, 255).astype(np.uint8)
        backgrounds.append(base)
    return backgrounds


def transform_sprite(sprite, mask, rng):
    h, w = sprite.shape[:2]
    target_h = rng.randint(30, 72)
    scale = target_h / max(h, 1)
    nw, nh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    sp = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_NEAREST)
    mp = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.5:
        sp, mp = cv2.flip(sp, 1), cv2.flip(mp, 1)
    if rng.random() < 0.35:
        angle = rng.uniform(-5.0, 5.0)
        matrix = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), angle, 1.0)
        sp = cv2.warpAffine(sp, matrix, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=(0, 255, 0))
        mp = cv2.warpAffine(mp, matrix, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    return sp, mp


def paste(background, sprite, mask, rng, anchor=None):
    h, w = background.shape[:2]
    sh, sw = sprite.shape[:2]
    if sw >= w or sh >= h:
        return background, None
    if anchor is None:
        x = rng.randint(0, w - sw)
    else:
        x = int(round(anchor[0] + rng.uniform(-0.55, 0.55) * max(sw, 36)))
        x = max(0, min(w - sw, x))
    # Monsters stand on platforms, so bias the lower half but retain variety.
    if anchor is None:
        y = rng.randint(max(0, int(h * 0.28)), h - sh)
    else:
        y = int(round(anchor[1] + rng.uniform(-0.35, 0.35) * max(sh, 28)))
        y = max(0, min(h - sh, y))
    roi = background[y : y + sh, x : x + sw]
    foreground = mask > 0
    roi[foreground] = sprite[foreground]
    return background, (x, y, sw, sh)


def save_sample(output, split, stem, image, box=None):
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{stem}.jpg"
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Unable to encode {image_path}")
    image_path.write_bytes(encoded.tobytes())
    label_path = label_dir / f"{stem}.txt"
    if box is None:
        label_path.write_text("", encoding="ascii")
        return
    boxes = [box] if isinstance(box, tuple) else list(box)
    h, w = image.shape[:2]
    lines = []
    for x, y, bw, bh in boxes:
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
    label_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "training_data" / "wild_boar_synth_v1")
    parser.add_argument("--per-sprite", type=int, default=90)
    parser.add_argument("--negatives", type=int, default=30)
    parser.add_argument("--instances-min", type=int, default=1)
    parser.add_argument("--instances-max", type=int, default=1)
    parser.add_argument(
        "--images-per-split",
        type=int,
        default=0,
        help="Use multi-instance mode and generate this many labeled images per split.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.per_sprite <= 0 or args.negatives < 0:
        raise ValueError("per-sprite must be positive and negatives non-negative")
    if args.instances_min <= 0 or args.instances_max < args.instances_min:
        raise ValueError("instances-max must be >= instances-min >= 1")
    if args.images_per_split < 0:
        raise ValueError("images-per-split must be non-negative")
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output = output.resolve()
    rng = random.Random(args.seed)
    backgrounds = scene_backgrounds(rng)
    if len(backgrounds) < 6:
        raise RuntimeError("Not enough backgrounds")
    rng.shuffle(backgrounds)
    n = len(backgrounds)
    bg_split = {"train": backgrounds[: int(n * 0.7)], "val": backgrounds[int(n * 0.7) : int(n * 0.85)], "test": backgrounds[int(n * 0.85) :]}
    loaded = {}
    for split, names in SPRITES.items():
        loaded[split] = []
        for name in names:
            path = SPRITE_DIR / name
            if not path.exists():
                raise FileNotFoundError(path)
            loaded[split].append((name, *extract_sprite(path)))

    counts = {"train": 0, "val": 0, "test": 0}
    negatives = {"train": 0, "val": 0, "test": 0}
    multi_instance = args.images_per_split > 0 or args.instances_max > 1
    if multi_instance:
        for split, sprites in loaded.items():
            image_count = args.images_per_split or args.per_sprite * len(sprites)
            for index in range(image_count):
                image = rng.choice(bg_split[split]).copy()
                boxes = []
                instance_count = rng.randint(args.instances_min, args.instances_max)
                # Crowd samples place monsters around one platform area, with
                # partial overlap, instead of only scattering them uniformly.
                anchor = (
                    rng.randint(50, IMG_W - 50),
                    rng.randint(int(IMG_H * 0.52), int(IMG_H * 0.82)),
                )
                for _ in range(instance_count):
                    sprite_name, sprite, mask = rng.choice(sprites)
                    transformed, transformed_mask = transform_sprite(sprite, mask, rng)
                    image, box = paste(image, transformed, transformed_mask, rng, anchor=anchor)
                    if box is not None:
                        boxes.append(box)
                if boxes:
                    save_sample(output, split, f"multi_{split}_{index:04d}", image, boxes)
                    counts[split] += 1
    else:
        for split, sprites in loaded.items():
            for sprite_name, sprite, mask in sprites:
                for index in range(args.per_sprite):
                    image = rng.choice(bg_split[split]).copy()
                    transformed, transformed_mask = transform_sprite(sprite, mask, rng)
                    image, box = paste(image, transformed, transformed_mask, rng)
                    if box is not None:
                        save_sample(output, split, f"{Path(sprite_name).stem}_{index:04d}", image, box)
                        counts[split] += 1
    for split, count in (("train", args.negatives), ("val", max(1, args.negatives // 3)), ("test", max(1, args.negatives // 3))):
        for index in range(count):
            save_sample(output, split, f"background_only_{index:04d}", rng.choice(bg_split[split]).copy())
            negatives[split] += 1
    data = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: CLASS_NAME}}
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    metadata = {
        "version": 2 if multi_instance else 1,
        "class": CLASS_NAME,
        "seed": args.seed,
        "source_sprite_split": SPRITES,
        "generated_labeled_images": counts,
        "generated_background_only_images": negatives,
        "instances_per_labeled_image": (
            [args.instances_min, args.instances_max] if multi_instance else [1, 1]
        ),
        "background_sources": [str(p) for p in sorted(SCENE_DIR.glob("*.png"))],
        "limitations": [
            "Synthetic-only bootstrap; validation and test are not independent live-game evidence.",
            "Only four sprite poses are available; real screenshots in land_of_wild_boar are still required.",
            "Real background crops come from one scene source and can overlap visually across splits.",
            "Other monsters in the scene screenshot are intentionally unlabeled hard negatives.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
