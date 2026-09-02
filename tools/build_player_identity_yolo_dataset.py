"""Build a one-class player detector dataset from reviewed gameplay captures.

The source frames are never modified. Exact OCR identity (player name plus the
configured title badge) seeds the existing read-only visual tracker, which
produces one stable player center per selected frame.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = REPO_ROOT / ".yolo_runtime"
for module_path in (REPO_ROOT, LOCAL_RUNTIME):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from tools.yolo_monster_viewer import (  # noqa: E402
    AsyncNameOcrLocator,
    ReadOnlyPlayerDetector,
    load_config,
    resolve_gameplay_height,
)


DEFAULT_SOURCE = (
    REPO_ROOT
    / "training_data"
    / "player_identity_capture_20260815_150345"
)
DEFAULT_OUTPUT = REPO_ROOT / "training_data" / "player_identity_yolo_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the reviewed 50-frame player YOLO dataset."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="shanda_legacy")
    parser.add_argument("--player-name", default="麻超圆")
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--box-width", type=int, default=150)
    parser.add_argument("--box-height", type=int, default=120)
    return parser.parse_args()


def load_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    return image


def save_image(path, image, quality=94):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if suffix in {".jpg", ".jpeg"} else []
    encoded, data = cv2.imencode(suffix, image, params)
    if not encoded:
        raise RuntimeError(f"Unable to encode image: {path}")
    data.tofile(path)


def select_temporal_splits(frame_paths, train_count, val_count):
    """Use disjoint early/late blocks and leave the middle block untouched."""
    required = int(train_count) + int(val_count)
    if train_count < 1 or val_count < 1:
        raise ValueError("train_count and val_count must be positive")
    if len(frame_paths) < required:
        raise ValueError(
            f"Need at least {required} frames, found {len(frame_paths)}"
        )
    train = list(frame_paths[:train_count])
    val = list(frame_paths[-val_count:])
    selected = set(train + val)
    holdout = [path for path in frame_paths if path not in selected]
    return {"train": train, "val": val, "holdout": holdout}


def centered_box(center, width, height, image_width, image_height):
    width = min(int(width), int(image_width))
    height = min(int(height), int(image_height))
    x = int(round(float(center[0]) - width / 2.0))
    y = int(round(float(center[1]) - height / 2.0))
    x = max(0, min(int(image_width) - width, x))
    y = max(0, min(int(image_height) - height, y))
    return [x, y, width, height]


def yolo_label(box, image_width, image_height):
    x, y, width, height = box
    center_x = (x + width / 2.0) / float(image_width)
    center_y = (y + height / 2.0) / float(image_height)
    return (
        f"0 {center_x:.6f} {center_y:.6f} "
        f"{width / float(image_width):.6f} "
        f"{height / float(image_height):.6f}\n"
    )


def make_identity_parser(player_name, template_shape, cfg):
    parser = AsyncNameOcrLocator.__new__(AsyncNameOcrLocator)
    parser.player_name = str(player_name)
    parser.template_height = int(template_shape[0])
    parser.template_width = int(template_shape[1])
    overlay = cfg["perception_overlay"]
    parser.confidence = float(overlay.get("player_ocr_confidence", 0.70))
    parser.title_text = str(overlay.get("player_title_anchor", "")).strip()
    parser.title_to_name_offset_y = 27
    return parser


def build_player_detector(player_name, cfg):
    overlay = cfg["perception_overlay"]
    template_path = REPO_ROOT / "nametag" / f"{player_name}_player.png"
    detector = ReadOnlyPlayerDetector(
        template_path,
        threshold=float(overlay["player_match_threshold"]),
        box_size=overlay["player_box_size"],
        center_offset_y=abs(int(cfg["nametag"]["offset"][1])),
        identity_threshold=float(overlay.get("player_identity_threshold", 0.48)),
        local_identity_threshold=float(
            overlay.get("player_local_identity_threshold", 0.38)
        ),
        identity_margin=float(overlay.get("player_identity_margin", 0.015)),
        glyph_threshold=int(overlay.get("player_glyph_threshold", 130)),
        glyph_weight=float(overlay.get("player_glyph_weight", 0.70)),
        glyph_min_columns=int(overlay.get("player_glyph_min_columns", 2)),
        candidate_count=int(overlay.get("player_candidate_count", 16)),
        lock_radius=float(overlay.get("player_lock_radius", 180.0)),
        reacquire_misses=int(overlay.get("player_reacquire_misses", 12)),
        center_weight=float(overlay.get("player_center_weight", 0.12)),
        max_valid_x=overlay.get("player_max_valid_x"),
        max_valid_y=overlay.get("player_max_valid_y"),
        color_anchor_enabled=bool(
            overlay.get("player_color_anchor_enabled", True)
        ),
        color_anchor_name_offset_y=int(
            overlay.get("player_color_anchor_name_offset_y", 24)
        ),
        color_anchor_min_red_fraction=float(
            overlay.get("player_color_anchor_min_red_fraction", 0.02)
        ),
        require_identity_seed=True,
    )
    return detector, template_path


def render_preview(image, box, source_name, split, identity_source):
    output = image.copy()
    x, y, width, height = box
    cv2.rectangle(output, (x, y), (x + width, y + height), (0, 230, 255), 3)
    label = f"{split} | {source_name} | {identity_source}"
    cv2.putText(
        output,
        label,
        (max(4, x), max(22, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 230, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def build_contact_sheet(previews, columns=5, tile_width=320, tile_height=180):
    tiles = [
        cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        for image in previews
    ]
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def ensure_empty_output(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output already contains files; choose a new directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    frame_paths = sorted(source.glob("frame_*.jpg"))
    splits = select_temporal_splits(
        frame_paths, args.train_count, args.val_count
    )
    ensure_empty_output(output)

    cfg = load_config(args.config)
    detector, template_path = build_player_detector(args.player_name, cfg)
    identity_parser = make_identity_parser(
        args.player_name, detector.template.shape, cfg
    )
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1)
    selected_splits = {
        path: split
        for split in ("train", "val")
        for path in splits[split]
    }
    previews = []
    annotations = []

    # Process in capture order so the title-anchor tracker can bridge OCR misses.
    for frame_id, path in enumerate(frame_paths, start=1):
        split = selected_splits.get(path)
        if split is None:
            continue
        frame = load_image(path)
        ui_cfg = cfg["ui_coords"]
        gameplay_height = resolve_gameplay_height(
            frame.shape,
            ui_cfg["ui_y_start"],
            ui_cfg.get("reference_width"),
        )
        gameplay = frame[:gameplay_height]
        identity = identity_parser._locate_exact_name(ocr(gameplay), frame_id)
        if identity is not None:
            detector.seed_identity(identity, frame)
        detection = detector.detect(frame, gameplay_height)
        if detection is None:
            raise RuntimeError(f"Unable to locate exact player in {path.name}")

        raw_box = detection["box"]
        center = (
            raw_box[0] + raw_box[2] / 2.0,
            raw_box[1] + raw_box[3] / 2.0,
        )
        box = centered_box(
            center,
            args.box_width,
            args.box_height,
            gameplay.shape[1],
            gameplay.shape[0],
        )
        stem = path.stem
        image_path = output / "images" / split / f"{stem}.jpg"
        label_path = output / "labels" / split / f"{stem}.txt"
        save_image(image_path, gameplay)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            yolo_label(box, gameplay.shape[1], gameplay.shape[0]),
            encoding="ascii",
        )
        identity_source = (
            "tracked_anchor"
            if identity is None
            else str(identity.get("identity_source", "ocr"))
        )
        previews.append(
            render_preview(gameplay, box, path.name, split, identity_source)
        )
        annotations.append(
            {
                "source": str(path),
                "split": split,
                "box_xywh": box,
                "center_xy": [round(center[0], 1), round(center[1], 1)],
                "identity_source": identity_source,
                "identity_mode": detection.get("identity_mode"),
                "identity_score": round(
                    float(detection.get("identity_score", 0.0)), 6
                ),
            }
        )

    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "player"},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    metadata = {
        "source": str(source),
        "player_name": args.player_name,
        "title_anchor": identity_parser.title_text,
        "player_template": str(template_path),
        "train_frames": [path.name for path in splits["train"]],
        "val_frames": [path.name for path in splits["val"]],
        "untouched_holdout_frames": [path.name for path in splits["holdout"]],
        "box_size": [args.box_width, args.box_height],
        "annotations": annotations,
        "split_policy": (
            "first 40 frames train, last 10 frames validation, middle block "
            "preserved as untouched temporal replay"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_image(output / "annotation_contact_sheet.jpg", build_contact_sheet(previews))
    print(
        json.dumps(
            {
                "output": str(output),
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "untouched_holdout": len(splits["holdout"]),
                "ocr_name": sum(
                    row["identity_source"] == "ocr_name" for row in annotations
                ),
                "ocr_title": sum(
                    row["identity_source"] == "ocr_title" for row in annotations
                ),
                "tracked_anchor": sum(
                    row["identity_source"] == "tracked_anchor"
                    for row in annotations
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
