"""Read-only YOLO monster viewer.

This module captures pixels from a window and draws detections. It deliberately
has no keyboard, mouse, combat, player-control, or focus-changing imports.
"""

import argparse
import ctypes
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = REPO_ROOT / ".yolo_runtime"
for module_path in (
    REPO_ROOT,
    LOCAL_RUNTIME,
    LOCAL_RUNTIME / "win32",
    LOCAL_RUNTIME / "win32" / "lib",
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

WIN32_DLL_HANDLE = None
if os.name == "nt" and (LOCAL_RUNTIME / "pywin32_system32").exists():
    WIN32_DLL_HANDLE = os.add_dll_directory(str(LOCAL_RUNTIME / "pywin32_system32"))

import cv2
import numpy as np
import yaml
from windows_capture import Frame, InternalCaptureControl, WindowsCapture

try:
    from src.utils.logger import logger
except Exception:  # viewer 独立运行时回退到 stdlib logging
    import logging
    logger = logging.getLogger("yolo_monster_viewer")


WINDOW_TITLE = "YOLO Monster Detector - OBSERVE ONLY"
DEFAULT_MODEL = "training_runs/two_class_real_v2_1280/weights/best.pt"
DEFAULT_WINDOW_TOKEN = "冒险岛怀旧服"

CLASS_INFO = {
    "stump": {"zh": "树妖", "display": "STUMP", "color": (0, 105, 255)},
    "red_snail": {"zh": "红蜗牛", "display": "RED SNAIL", "color": (70, 70, 255)},
    "slime": {"zh": "绿水灵", "display": "GREEN SLIME", "color": (80, 230, 130)},
    "green_slime": {"zh": "绿水灵", "display": "GREEN SLIME", "color": (80, 230, 130)},
    "green_mushroom": {"zh": "绿蘑菇", "display": "GREEN MUSHROOM", "color": (40, 190, 40)},
    "flower_mushroom": {"zh": "花蘑菇", "display": "FLOWER MUSHROOM", "color": (210, 80, 230)},
    "zombie_mushroom": {"zh": "僵尸蘑菇", "display": "ZOMBIE MUSHROOM", "color": (30, 170, 255)},
    "thorn_mushroom": {"zh": "刺蘑菇", "display": "THORN MUSHROOM", "color": (255, 180, 40)},
    "pig": {"zh": "肥肥", "display": "PIG", "color": (255, 120, 200)},
    "wild_boar": {"zh": "黑肥肥", "display": "WILD BOAR", "color": (110, 110, 110)},
}

LABEL_ALIASES = {
    "树妖": "stump",
    "木妖": "stump",
    "stump": "stump",
    "红蜗牛": "red_snail",
    "red_snail": "red_snail",
    "red snail": "red_snail",
    "绿水灵": "slime",
    "slime": "slime",
    "green_slime": "slime",
    "green slime": "slime",
    "绿蘑菇": "green_mushroom",
    "green_mushroom": "green_mushroom",
    "green mushroom": "green_mushroom",
    "花蘑菇": "flower_mushroom",
    "flower_mushroom": "flower_mushroom",
    "flower mushroom": "flower_mushroom",
    "僵尸蘑菇": "zombie_mushroom",
    "zombie_mushroom": "zombie_mushroom",
    "zombie mushroom": "zombie_mushroom",
    "刺蘑菇": "thorn_mushroom",
    "thorn_mushroom": "thorn_mushroom",
    "thorn mushroom": "thorn_mushroom",
    "肥肥": "pig",
    "猪猪": "pig",
    "pig": "pig",
    "黑肥肥": "wild_boar",
    "wild_boar": "wild_boar",
    "wild boar": "wild_boar",
}

REQUIRED_CLASSES = {
    "stump",
    "red_snail",
    "slime",
    "green_mushroom",
    "flower_mushroom",
    "zombie_mushroom",
    "thorn_mushroom",
    "pig",
}
PLAYER_COLOR = (0, 230, 255)
ENTITY_SHORT_LABELS = {
    "player": "PLAYER",
    "stump": "STUMP",
    "red_snail": "R-SNAIL",
    "slime": "SLIME",
    "green_mushroom": "G-MUSH",
    "flower_mushroom": "F-MUSH",
    "zombie_mushroom": "Z-MUSH",
    "thorn_mushroom": "T-MUSH",
    "pig": "PIG",
    "wild_boar": "W-BOAR",
}


def load_config(name):
    config_path = REPO_ROOT / "config" / f"config_{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_gameplay_height(frame_shape, configured_height, reference_width=None):
    """Keep a legacy UI boundary valid after the client window grows.

    The legacy profile was calibrated at a 1370 px capture width.  Capturing
    a wider client scales the game world down the screen too, so a fixed
    ``ui_y_start`` would crop the lower part of the playable area.
    """
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    gameplay_height = int(configured_height)
    if reference_width is not None and int(reference_width) > 0:
        scaled_height = int(
            round(gameplay_height * frame_width / int(reference_width))
        )
        gameplay_height = max(gameplay_height, scaled_height)
    return max(1, min(frame_height, gameplay_height))


def resolve_scaled_region(frame_shape, region, reference_width=None):
    """Scale a reference-width ``[x, y, width, height]`` region to a frame."""
    if region is None or len(region) < 4:
        return None
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    scale = 1.0
    if reference_width is not None and int(reference_width) > 0:
        scale = frame_width / float(reference_width)
    x = int(round(float(region[0]) * scale))
    y = int(round(float(region[1]) * scale))
    width = int(round(float(region[2]) * scale))
    height = int(round(float(region[3]) * scale))
    x = max(0, min(frame_width, x))
    y = max(0, min(frame_height, y))
    width = max(0, min(frame_width - x, width))
    height = max(0, min(frame_height - y, height))
    if width == 0 or height == 0:
        return None
    return [x, y, width, height]


def resolve_minimap_canvas(frame, canvas_box, minimap_cfg):
    """Measure the current minimap content rectangle from its bright UI border."""
    if not minimap_cfg.get("canvas_auto_width", False):
        return canvas_box
    region = minimap_cfg.get("region")
    search_box = resolve_scaled_region(
        frame.shape,
        region,
        minimap_cfg.get("canvas_reference_width"),
    )
    if search_box is None:
        return canvas_box
    canvas_x, canvas_y, fallback_width, fallback_height = canvas_box
    search_x, search_y, search_width, search_height = search_box
    scale = frame.shape[1] / float(
        max(1, int(minimap_cfg.get("canvas_reference_width", frame.shape[1])))
    )
    scan_top = canvas_y
    scan_bottom = min(frame.shape[0], search_y + search_height)
    if scan_bottom <= scan_top:
        return canvas_box

    white = np.min(frame[scan_top:scan_bottom], axis=2) >= int(
        minimap_cfg.get("canvas_border_white", 235)
    )
    white_counts = white.sum(axis=0)
    min_vertical_pixels = max(30, int(round(50 * scale)))
    min_border_width = max(2, int(round(2 * scale)))
    left_bound = canvas_x + max(40, int(round(60 * scale)))
    right_bound = min(frame.shape[1] - 1, search_x + search_width - 1)
    if right_bound <= left_bound:
        return canvas_box
    columns = np.flatnonzero(
        (white_counts >= min_vertical_pixels)
        & (np.arange(frame.shape[1]) >= left_bound)
        & (np.arange(frame.shape[1]) <= right_bound)
    )
    if columns.size == 0:
        return canvas_box

    groups = []
    for column in columns:
        column = int(column)
        if not groups or column > groups[-1][-1] + 1:
            groups.append([column])
        else:
            groups[-1].append(column)
    right_group = next(
        (group for group in groups if len(group) >= min_border_width), None
    )
    if right_group is None:
        return canvas_box
    right_border = right_group[0]
    canvas_width = right_border - canvas_x
    if canvas_width <= 0:
        return canvas_box

    border_margin = max(3, int(round(3 * scale)))
    row_left = max(0, canvas_x - border_margin)
    row_right = min(frame.shape[1], right_border + border_margin)
    bottom_start = canvas_y + max(30, int(round(40 * scale)))
    bottom_limit = min(frame.shape[0], search_y + search_height)
    horizontal = frame[bottom_start:bottom_limit, row_left:row_right]
    if horizontal.size == 0:
        return [canvas_x, canvas_y, canvas_width, fallback_height]
    light_rows = (
        np.min(horizontal, axis=2)
        >= int(minimap_cfg.get("canvas_border_light", 180))
    ).mean(axis=1)
    required_rows = max(2, int(round(2 * scale)))
    row_candidates = np.flatnonzero(light_rows >= 0.90)
    bottom_y = None
    for index in range(len(row_candidates) - required_rows + 1):
        run = row_candidates[index : index + required_rows]
        if int(run[-1]) - int(run[0]) == required_rows - 1:
            bottom_y = bottom_start + int(run[0])
            break
    canvas_height = fallback_height if bottom_y is None else bottom_y - canvas_y
    if canvas_height <= 0:
        canvas_height = fallback_height
    return [canvas_x, canvas_y, canvas_width, canvas_height]


def _minimap_canvas_crop(frame, minimap_cfg):
    """Return the calibrated minimap crop and frame-space canvas box."""
    if frame is None or not minimap_cfg:
        return None, None
    canvas_region = minimap_cfg.get("canvas_region")
    if canvas_region is None:
        return None, None
    canvas_box = resolve_scaled_region(
        frame.shape,
        canvas_region,
        minimap_cfg.get("canvas_reference_width"),
    )
    if canvas_box is None:
        return None, None
    canvas_box = resolve_minimap_canvas(frame, canvas_box, minimap_cfg)
    canvas_x, canvas_y, canvas_width, canvas_height = canvas_box
    crop = frame[
        canvas_y : canvas_y + canvas_height,
        canvas_x : canvas_x + canvas_width,
    ]
    if crop.size == 0:
        return None, None
    return crop, canvas_box


def _find_minimap_color_markers(
    crop,
    color,
    tolerance,
    canvas_box,
    min_pixels=3,
    max_pixels=24,
    max_dimension=10,
    min_fill_ratio=0.25,
):
    """Find compact colored dots and return their map/frame coordinates."""
    color = np.asarray(color, dtype=np.int16)
    tolerance = np.asarray(tolerance, dtype=np.int16)
    if color.size != 3 or tolerance.size != 3:
        return []
    lower = np.clip(color - tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(color + tolerance, 0, 255).astype(np.uint8)
    mask = cv2.inRange(crop, lower, upper)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)
    max_dimension = int(max_dimension)
    min_fill_ratio = float(min_fill_ratio)
    candidates = []
    canvas_x, canvas_y, canvas_width, canvas_height = canvas_box
    for component_index in range(1, component_count):
        x, y, width, height, area = [
            int(value) for value in stats[component_index]
        ]
        fill_ratio = area / float(max(1, width * height))
        if area < min_pixels or area > max_pixels:
            continue
        if width > max_dimension or height > max_dimension:
            continue
        if max(width / float(max(1, height)), height / float(max(1, width))) > 3.0:
            continue
        if fill_ratio < min_fill_ratio:
            continue
        # Component-box center is steadier than a pixel centroid for tiny
        # antialiased diamonds whose lower raster row may be one pixel longer.
        center_x = x + (width - 1) / 2.0
        center_y = y + (height - 1) / 2.0
        candidates.append(
            {
                "map_px": [center_x, center_y],
                "map_norm": [
                    center_x / float(canvas_width),
                    center_y / float(canvas_height),
                ],
                "frame_px": [canvas_x + center_x, canvas_y + center_y],
                "marker_box_map": [x, y, width, height],
                "canvas_frame_box": canvas_box,
                "canvas_size": [canvas_width, canvas_height],
                "pixel_count": int(stats[component_index, cv2.CC_STAT_AREA]),
                "fill_ratio": float(fill_ratio),
            }
        )
    return sorted(candidates, key=lambda item: (item["map_px"][1], item["map_px"][0]))


def _find_minimap_red_players(crop, canvas_box, minimap_cfg):
    """Find saturated compact red player dots without accepting red edges/text.

    The old BGR box was intentionally broad for the yellow marker and was too
    permissive for red: brown map pixels and antialiased UI glyphs can have a
    high red channel without being a red player dot. Red markers are therefore
    gated in HSV and by channel dominance, then required to contain a small
    bright-red core in a compact near-square component.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue_ranges = minimap_cfg.get(
        "other_player_hsv_ranges",
        [[0, 170, 175, 12, 255, 255], [170, 170, 175, 179, 255, 255]],
    )
    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for values in hue_ranges:
        if len(values) != 6:
            continue
        h0, s0, v0, h1, s1, v1 = [int(value) for value in values]
        red_mask |= cv2.inRange(
            hsv,
            np.array([h0, s0, v0], dtype=np.uint8),
            np.array([h1, s1, v1], dtype=np.uint8),
        )

    # Require red to dominate both other channels. This removes brown/orange
    # terrain whose R channel is bright but whose B/G channels are not small.
    bgr = crop.astype(np.int16)
    red_dominant = (
        (bgr[:, :, 2] >= int(minimap_cfg.get("other_player_red_min", 175)))
        & (bgr[:, :, 2] - bgr[:, :, 1] >= int(minimap_cfg.get("other_player_red_green_gap", 120)))
        & (bgr[:, :, 2] - bgr[:, :, 0] >= int(minimap_cfg.get("other_player_red_blue_gap", 120)))
    )
    red_mask[~red_dominant] = 0

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        red_mask, connectivity=8
    )
    min_pixels = int(minimap_cfg.get("other_player_marker_min_pixels", 6))
    max_pixels = int(minimap_cfg.get("other_player_marker_max_pixels", 30))
    min_dimension = int(
        minimap_cfg.get("other_player_marker_min_dimension", 2)
    )
    max_dimension = int(minimap_cfg.get("other_player_marker_max_dimension", 8))
    min_fill_ratio = float(
        minimap_cfg.get("other_player_marker_min_fill_ratio", 0.35)
    )
    min_core_pixels = int(
        minimap_cfg.get("other_player_marker_min_core_pixels", 2)
    )
    canvas_x, canvas_y, canvas_width, canvas_height = canvas_box
    candidates = []
    for component_index in range(1, component_count):
        x, y, width, height, area = [
            int(value) for value in stats[component_index]
        ]
        fill_ratio = area / float(max(1, width * height))
        if area < min_pixels or area > max_pixels:
            continue
        if (
            width < min_dimension
            or height < min_dimension
            or width > max_dimension
            or height > max_dimension
        ):
            continue
        if max(width / float(height), height / float(width)) > 2.0:
            continue
        if fill_ratio < min_fill_ratio:
            continue
        core = red_mask[y : y + height, x : x + width]
        if int(cv2.countNonZero(core)) < min_core_pixels:
            continue
        center_x = x + (width - 1) / 2.0
        center_y = y + (height - 1) / 2.0
        candidates.append(
            {
                "map_px": [center_x, center_y],
                "map_norm": [
                    center_x / float(canvas_width),
                    center_y / float(canvas_height),
                ],
                "frame_px": [canvas_x + center_x, canvas_y + center_y],
                "marker_box_map": [x, y, width, height],
                "canvas_frame_box": canvas_box,
                "canvas_size": [canvas_width, canvas_height],
                "pixel_count": int(area),
                "fill_ratio": float(fill_ratio),
            }
        )
    return sorted(candidates, key=lambda item: (item["map_px"][1], item["map_px"][0]))


class MinimapRedMarkerTracker:
    """Confirm red dots across frames before exposing them to the viewer."""

    def __init__(self, confirm_frames=2, max_missed=1, max_distance=8.0):
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_missed = max(0, int(max_missed))
        self.max_distance = max(1.0, float(max_distance))
        self._tracks = []

    def update(self, markers):
        markers = list(markers or ())
        used = set()
        next_tracks = []
        for track in self._tracks:
            old_x, old_y = track["center"]
            best_index = None
            best_distance = self.max_distance
            for index, marker in enumerate(markers):
                if index in used:
                    continue
                x, y = marker["map_px"]
                distance = float(np.hypot(x - old_x, y - old_y))
                if distance <= best_distance:
                    best_index = index
                    best_distance = distance
            if best_index is None:
                track["missed"] += 1
                if track["missed"] <= self.max_missed:
                    next_tracks.append(track)
                continue
            marker = markers[best_index]
            used.add(best_index)
            track["marker"] = marker
            track["center"] = tuple(marker["map_px"])
            track["hits"] += 1
            track["missed"] = 0
            next_tracks.append(track)

        for index, marker in enumerate(markers):
            if index in used:
                continue
            next_tracks.append(
                {
                    "marker": marker,
                    "center": tuple(marker["map_px"]),
                    "hits": 1,
                    "missed": 0,
                }
            )
        self._tracks = next_tracks
        return [
            track["marker"]
            for track in self._tracks
            if track["hits"] >= self.confirm_frames and track["missed"] == 0
        ]


def locate_minimap_players(frame, minimap_cfg):
    """Locate our yellow marker and all red other-player markers.

    Coordinates are relative to the detected minimap canvas (``map_px`` and
    ``map_norm``) and to the captured frame (``frame_px``). Detection remains
    read-only and is limited to the calibrated canvas, excluding the title/UI
    area surrounding it.
    """
    crop, canvas_box = _minimap_canvas_crop(frame, minimap_cfg)
    if crop is None:
        return {"player": None, "other_players": []}

    player_markers = _find_minimap_color_markers(
        crop,
        minimap_cfg.get("player_color", [0, 255, 255]),
        minimap_cfg.get("marker_color_tolerance", [40, 40, 40]),
        canvas_box,
        min_pixels=minimap_cfg.get("marker_min_pixels", 3),
        max_pixels=minimap_cfg.get("marker_max_pixels", 24),
        max_dimension=minimap_cfg.get("marker_max_dimension", 10),
        min_fill_ratio=minimap_cfg.get("marker_min_fill_ratio", 0.25),
    )
    # Preserve the original single-player selection: the largest compact
    # yellow component is the stable candidate in a frame. This is deliberately
    # separate from red markers, which must all be returned.
    player = max(
        player_markers,
        key=lambda item: (item["pixel_count"], item["fill_ratio"]),
        default=None,
    )
    other_players = _find_minimap_red_players(crop, canvas_box, minimap_cfg)
    return {"player": player, "other_players": other_players}


def locate_minimap_player(frame, minimap_cfg):
    """Locate the compact yellow player marker inside the calibrated minimap."""
    return locate_minimap_players(frame, minimap_cfg)["player"]


def normalize_label(label):
    key = str(label).strip().lower()
    if key not in LABEL_ALIASES:
        supported = ", ".join(info["zh"] for info in CLASS_INFO.values())
        raise ValueError(f"Unknown monster label '{label}'. Supported: {supported}")
    return LABEL_ALIASES[key]


def normalize_model_class(class_name):
    normalized = str(class_name).strip().lower().replace("-", "_")
    return "slime" if normalized == "green_slime" else normalized


def validate_model_classes(names, required_classes=None):
    normalized = {normalize_model_class(name) for name in names.values()}
    required = REQUIRED_CLASSES if required_classes is None else set(required_classes)
    missing = sorted(required - normalized)
    if missing:
        raise ValueError(
            "The selected YOLO model does not contain all requested "
            f"classes. Missing: {', '.join(missing)}"
        )


def box_iou(first, second):
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    overlap_width = max(
        0,
        min(first_x + first_width, second_x + second_width)
        - max(first_x, second_x),
    )
    overlap_height = max(
        0,
        min(first_y + first_height, second_y + second_height)
        - max(first_y, second_y),
    )
    intersection = overlap_width * overlap_height
    union = (
        first_width * first_height
        + second_width * second_height
        - intersection
    )
    return 0.0 if union <= 0 else intersection / float(union)


def deduplicate_detections(detections, iou_threshold=0.50):
    kept = []
    for detection in sorted(
        detections, key=lambda item: item["confidence"], reverse=True
    ):
        duplicate = any(
            detection["class"] == existing["class"]
            and box_iou(detection["box"], existing["box"]) >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept


def reject_edge_clipped_stump(box, gameplay_height, min_height=60, bottom_margin=3):
    """Reject the characteristic partial black-post false positive.

    Background posts at the bottom of Warrior Tribe captures are clipped by the
    gameplay/UI boundary and produce short stump-like boxes. Real stump sprites
    retain their normal height, so this narrow rule removes only the clipped
    shape while leaving ordinary low-confidence filtering to the model.
    """
    _, y, _, height = [float(value) for value in box]
    return (
        float(y + height) >= float(gameplay_height) - float(bottom_margin)
        and float(height) < float(min_height)
    )


def normalized_center_distance(first, second):
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    first_center = (
        first_x + first_width / 2.0,
        first_y + first_height / 2.0,
    )
    second_center = (
        second_x + second_width / 2.0,
        second_y + second_height / 2.0,
    )
    distance = np.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    scale = max(
        1.0,
        first_width,
        first_height,
        second_width,
        second_height,
    )
    return float(distance / scale)


def box_center(box):
    x, y, width, height = box
    return (x + width / 2.0, y + height / 2.0)


class EntityCoordinateTracker:
    """Add screen coordinates and smoothed motion to tracked detections."""

    def __init__(
        self,
        velocity_smoothing=0.45,
        move_threshold_px_s=35.0,
        vertical_threshold_px_s=45.0,
    ):
        if not 0.0 < velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in (0, 1]")
        self.velocity_smoothing = float(velocity_smoothing)
        self.move_threshold_px_s = float(move_threshold_px_s)
        self.vertical_threshold_px_s = float(vertical_threshold_px_s)
        self.states = {}

    def _motion_state(self, velocity_x, velocity_y):
        speed = float(np.hypot(velocity_x, velocity_y))
        if (
            abs(velocity_y) >= self.vertical_threshold_px_s
            and abs(velocity_y) >= abs(velocity_x) * 0.75
        ):
            return "UP" if velocity_y < 0 else "DOWN"
        if speed >= self.move_threshold_px_s:
            return "MOVE"
        return "STILL"

    def update(
        self,
        detections,
        timestamp,
        frame_width,
        gameplay_height,
        prefix="M",
        fixed_entity_id=None,
    ):
        enriched = []
        active_ids = set()
        for detection in detections:
            entity_id = fixed_entity_id or f"{prefix}{detection['track_id']}"
            active_ids.add(entity_id)
            center_x, center_y = box_center(detection["box"])
            previous = self.states.get(entity_id)
            velocity_x = 0.0
            velocity_y = 0.0
            if previous is not None:
                elapsed = max(float(timestamp) - previous["timestamp"], 1e-6)
                measured_x = (center_x - previous["center"][0]) / elapsed
                measured_y = (center_y - previous["center"][1]) / elapsed
                alpha = self.velocity_smoothing
                velocity_x = previous["velocity"][0] * (1.0 - alpha) + measured_x * alpha
                velocity_y = previous["velocity"][1] * (1.0 - alpha) + measured_y * alpha

            motion_state = self._motion_state(velocity_x, velocity_y)
            self.states[entity_id] = {
                "center": (center_x, center_y),
                "velocity": (velocity_x, velocity_y),
                "timestamp": float(timestamp),
            }
            item = dict(detection)
            item.update(
                {
                    "entity_id": entity_id,
                    "center_px": [round(center_x, 1), round(center_y, 1)],
                    "center_norm": [
                        round(center_x / max(1, frame_width), 5),
                        round(center_y / max(1, gameplay_height), 5),
                    ],
                    "velocity_px_s": [
                        round(velocity_x, 1),
                        round(velocity_y, 1),
                    ],
                    "speed_px_s": round(float(np.hypot(velocity_x, velocity_y)), 1),
                    "motion_state": motion_state,
                    "tracking_state": (
                        "PREDICTED"
                        if int(detection.get("missed_frames", 0)) > 0
                        else "DETECTED"
                    ),
                }
            )
            enriched.append(item)

        self.states = {
            entity_id: state
            for entity_id, state in self.states.items()
            if entity_id in active_ids
        }
        return enriched


def attach_player_relative_coordinates(detections, player):
    if player is None:
        return [dict(detection, relative_to_player=None) for detection in detections]
    player_x, player_y = player["center_px"]
    output = []
    for detection in detections:
        center_x, center_y = detection["center_px"]
        delta_x = center_x - player_x
        delta_y = center_y - player_y
        item = dict(detection)
        item["relative_to_player"] = {
            "delta_px": [round(delta_x, 1), round(delta_y, 1)],
            "distance_px": round(float(np.hypot(delta_x, delta_y)), 1),
        }
        output.append(item)
    return output


class DetectionTracker:
    """Smooth detections and bridge brief detector dropouts without input I/O."""

    def __init__(
        self,
        max_missed=4,
        smoothing=0.65,
        match_iou=0.15,
        max_center_distance=1.25,
        min_confirmed_hits=1,
        high_confidence_confirm=None,
        max_width_ratio=1.70,
        max_height_ratio=2.00,
        max_area_ratio=2.50,
        size_cost_weight=0.25,
    ):
        if max_missed < 0:
            raise ValueError("max_missed must be non-negative")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if not 0.0 <= match_iou <= 1.0:
            raise ValueError("match_iou must be in [0, 1]")
        if max_center_distance <= 0.0:
            raise ValueError("max_center_distance must be positive")
        if min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits must be positive")
        if (
            high_confidence_confirm is not None
            and not 0.0 <= high_confidence_confirm <= 1.0
        ):
            raise ValueError("high_confidence_confirm must be in [0, 1]")
        for name, value in (
            ("max_width_ratio", max_width_ratio),
            ("max_height_ratio", max_height_ratio),
            ("max_area_ratio", max_area_ratio),
        ):
            if value is not None and value < 1.0:
                raise ValueError(f"{name} must be at least 1.0 or None")
        if size_cost_weight < 0.0:
            raise ValueError("size_cost_weight must be non-negative")

        self.max_missed = int(max_missed)
        self.smoothing = float(smoothing)
        self.match_iou = float(match_iou)
        self.max_center_distance = float(max_center_distance)
        self.min_confirmed_hits = int(min_confirmed_hits)
        self.high_confidence_confirm = (
            None
            if high_confidence_confirm is None
            else float(high_confidence_confirm)
        )
        self.max_width_ratio = (
            None if max_width_ratio is None else float(max_width_ratio)
        )
        self.max_height_ratio = (
            None if max_height_ratio is None else float(max_height_ratio)
        )
        self.max_area_ratio = (
            None if max_area_ratio is None else float(max_area_ratio)
        )
        self.size_cost_weight = float(size_cost_weight)
        self.tracks = {}
        self.next_track_id = 1

    @staticmethod
    def _predicted_box(track):
        box = track["box"]
        velocity_x, velocity_y = track["velocity"]
        return [box[0] + velocity_x, box[1] + velocity_y, box[2], box[3]]

    @staticmethod
    def _symmetric_ratio(first, second):
        first = max(float(first), 1e-6)
        second = max(float(second), 1e-6)
        return max(first / second, second / first)

    def _size_match(self, predicted_box, measured_box):
        width_ratio = self._symmetric_ratio(predicted_box[2], measured_box[2])
        height_ratio = self._symmetric_ratio(predicted_box[3], measured_box[3])
        predicted_area = predicted_box[2] * predicted_box[3]
        measured_area = measured_box[2] * measured_box[3]
        area_ratio = self._symmetric_ratio(predicted_area, measured_area)
        matches = (
            (self.max_width_ratio is None or width_ratio <= self.max_width_ratio)
            and (
                self.max_height_ratio is None
                or height_ratio <= self.max_height_ratio
            )
            and (self.max_area_ratio is None or area_ratio <= self.max_area_ratio)
        )
        size_cost = (
            abs(float(np.log(width_ratio)))
            + abs(float(np.log(height_ratio)))
        ) / 2.0
        return matches, size_cost

    @staticmethod
    def _to_detection(track):
        detection = {
            key: value
            for key, value in track.items()
            if key not in {"velocity", "age", "missed", "hits", "confirmed"}
        }
        detection["box"] = [
            max(0, int(round(track["box"][0]))),
            max(0, int(round(track["box"][1]))),
            max(1, int(round(track["box"][2]))),
            max(1, int(round(track["box"][3]))),
        ]
        detection["confidence"] = float(track["confidence"])
        detection["missed_frames"] = int(track["missed"])
        detection["confirmation_hits"] = int(track["hits"])
        detection["confirmed"] = bool(track["confirmed"])
        return detection

    def _new_track(self, detection):
        confidence = float(detection["confidence"])
        high_confidence = (
            self.high_confidence_confirm is not None
            and confidence >= self.high_confidence_confirm
        )
        track = dict(detection)
        track.update(
            {
                "track_id": self.next_track_id,
                "box": [float(value) for value in detection["box"]],
                "confidence": confidence,
                "velocity": [0.0, 0.0],
                "age": 1,
                "hits": 1,
                "missed": 0,
                "confirmed": self.min_confirmed_hits <= 1 or high_confidence,
            }
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def update(self, detections):
        detections = [dict(detection) for detection in detections]
        predicted = {
            track_id: self._predicted_box(track)
            for track_id, track in self.tracks.items()
        }
        candidates = []
        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                if track["class"] != detection["class"]:
                    continue
                iou = box_iou(predicted[track_id], detection["box"])
                center_distance = normalized_center_distance(
                    predicted[track_id], detection["box"]
                )
                size_matches, size_cost = self._size_match(
                    predicted[track_id], detection["box"]
                )
                if not size_matches:
                    continue
                if (
                    iou >= self.match_iou
                    or center_distance <= self.max_center_distance
                ):
                    cost = (
                        (1.0 - iou)
                        + center_distance * 0.35
                        + size_cost * self.size_cost_weight
                    )
                    candidates.append(
                        (cost, track_id, detection_index)
                    )

        matched_tracks = set()
        matched_detections = set()
        for _, track_id, detection_index in sorted(candidates):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[detection_index]
            old_box = track["box"]
            predicted_box = predicted[track_id]
            measured_box = [float(value) for value in detection["box"]]
            alpha = self.smoothing
            track["box"] = [
                predicted_value * (1.0 - alpha) + measured_value * alpha
                for predicted_value, measured_value in zip(
                    predicted_box, measured_box
                )
            ]
            measured_velocity = [
                measured_box[0] - old_box[0],
                measured_box[1] - old_box[1],
            ]
            track["velocity"] = [
                old_velocity * 0.5 + new_velocity * 0.5
                for old_velocity, new_velocity in zip(
                    track["velocity"], measured_velocity
                )
            ]
            track["confidence"] = (
                track["confidence"] * (1.0 - alpha)
                + float(detection["confidence"]) * alpha
            )
            for key in (
                "label",
                "label_zh",
                "color",
                "nametag_box",
                "template_score",
                "glyph_score",
                "identity_score",
                "identity_mode",
            ):
                if key in detection:
                    track[key] = detection[key]
            track["age"] += 1
            track["hits"] += 1
            track["missed"] = 0
            if (
                track["hits"] >= self.min_confirmed_hits
                or (
                    self.high_confidence_confirm is not None
                    and float(detection["confidence"])
                    >= self.high_confidence_confirm
                )
            ):
                track["confirmed"] = True
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        expired = []
        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue
            track["box"] = predicted[track_id]
            track["velocity"] = [value * 0.80 for value in track["velocity"]]
            track["confidence"] *= 0.90
            track["age"] += 1
            track["missed"] += 1
            if not track["confirmed"]:
                track["hits"] = 0
            if track["missed"] > self.max_missed:
                expired.append(track_id)
        for track_id in expired:
            del self.tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._new_track(detection)

        return [
            self._to_detection(self.tracks[track_id])
            for track_id in sorted(self.tracks)
            if self.tracks[track_id]["confirmed"]
        ]


def find_visible_window_title(token):
    if os.name != "nt":
        raise RuntimeError("Live window capture is currently supported on Windows only")

    user32 = ctypes.windll.user32
    matches = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def enum_callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if token in title:
            matches.append(title)
        return True

    user32.EnumWindows(callback_type(enum_callback), 0)
    if not matches:
        raise RuntimeError(f"No visible window title contains: {token}")
    return matches[0]


class ReadOnlyWindowCapture:
    def __init__(self, window_title):
        self.frame = None
        self.lock = threading.Lock()
        self.control = None
        self.capture = WindowsCapture(window_name=window_title)
        self.capture.event(self.on_frame_arrived)
        self.capture.event(self.on_closed)
        self.control = self.capture.start_free_threaded()

    def on_frame_arrived(
        self, frame: Frame, capture_control: InternalCaptureControl
    ):
        del capture_control
        with self.lock:
            self.frame = frame.frame_buffer.copy()

    def on_closed(self):
        return None

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def stop(self):
        if self.control is not None:
            self.control.stop()


class ReadOnlyPlayerDetector:
    """Locate one named player without switching to similar nearby nametags."""

    def __init__(
        self,
        template_path,
        threshold=0.30,
        box_size=(70, 90),
        center_offset_y=30,
        identity_threshold=0.48,
        local_identity_threshold=0.38,
        identity_margin=0.015,
        glyph_threshold=130,
        glyph_weight=0.70,
        candidate_count=16,
        lock_radius=180.0,
        reacquire_misses=12,
        center_weight=0.12,
        require_identity_seed=False,
        max_valid_x=None,
        min_valid_y=None,
        max_valid_y=None,
        glyph_min_columns=2,
        color_anchor_enabled=True,
        color_anchor_name_offset_y=24,
        color_anchor_min_red_fraction=0.02,
        color_anchor_local_radius=260.0,
        color_anchor_color_tol=80.0,
        color_anchor_ref_path=None,
        keep_color_anchor_misses=6,
    ):
        resolved = Path(template_path)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved
        if not resolved.exists():
            raise FileNotFoundError(f"Player template not found: {resolved}")
        data = np.fromfile(str(resolved), dtype=np.uint8)
        template = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if template is None:
            raise RuntimeError(f"Unable to read player template: {resolved}")
        self.template_path = resolved.resolve()
        self.template = template
        self.threshold = float(threshold)
        self.box_width = int(box_size[0])
        self.box_height = int(box_size[1])
        self.center_offset_y = int(center_offset_y)
        self.identity_threshold = float(identity_threshold)
        self.local_identity_threshold = float(local_identity_threshold)
        self.identity_margin = float(identity_margin)
        self.glyph_threshold = int(glyph_threshold)
        self.glyph_weight = float(glyph_weight)
        self.candidate_count = int(candidate_count)
        self.lock_radius = float(lock_radius)
        self.reacquire_misses = int(reacquire_misses)
        self.center_weight = float(center_weight)
        self.require_identity_seed = bool(require_identity_seed)
        self.max_valid_x = None if max_valid_x is None else float(max_valid_x)
        self.max_valid_y = None if max_valid_y is None else float(max_valid_y)
        self.glyph_min_columns = max(1, min(3, int(glyph_min_columns)))
        self.color_anchor_enabled = bool(color_anchor_enabled)
        self.color_anchor_name_offset_y = int(color_anchor_name_offset_y)
        self.color_anchor_min_red_fraction = float(color_anchor_min_red_fraction)
        self.color_anchor_local_radius = float(color_anchor_local_radius)
        self.color_anchor_color_tol = float(color_anchor_color_tol)
        self.keep_color_anchor_misses = int(keep_color_anchor_misses)
        # 【蓝条颜色参照】: 候选称号条的蓝色像素均值必须接近自己勋章蓝(参照图/
        # 名字模板中的蓝像素均值作为冷启动参照, 锁定后按实际蓝条 EMA 微调)。
        # 地形/瀑布/岩石的蓝与中级冒险家勋章的同款蓝有明显色差, 色距超
        # color_tol 直接丢弃——否则"红棕色地形(红分高) + 一条蓝条"就能拿到
        # identity_score=1.0, 黄框锁到地形(用户反馈: 置信1.00锁地形, 巡游掉台子)。
        self.color_anchor_ref_bgr = self._template_blue_ref(ref_path=color_anchor_ref_path)
        self._ref_warn_at = 0.0  # "色距偏高仍锁定"告警节流时间戳
        if not 0.0 <= self.identity_threshold <= 1.0:
            raise ValueError("identity_threshold must be in [0, 1]")
        if not 0.0 <= self.local_identity_threshold <= 1.0:
            raise ValueError("local_identity_threshold must be in [0, 1]")
        if self.identity_margin < 0.0:
            raise ValueError("identity_margin must be non-negative")
        if not 0.0 <= self.glyph_weight <= 1.0:
            raise ValueError("glyph_weight must be in [0, 1]")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if self.lock_radius <= 0.0:
            raise ValueError("lock_radius must be positive")
        if self.reacquire_misses < 0:
            raise ValueError("reacquire_misses must be non-negative")
        if self.center_weight < 0.0:
            raise ValueError("center_weight must be non-negative")

        tag_height, tag_width = self.template.shape[:2]
        margin_y = max(1, int(round(tag_height * 0.13)))
        margin_x = max(1, int(round(tag_width * 0.05)))
        self.glyph_slice = (
            slice(margin_y, max(margin_y + 1, tag_height - margin_y)),
            slice(margin_x, max(margin_x + 1, tag_width - margin_x)),
        )
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.template_glyph = (
            template_gray[self.glyph_slice] >= self.glyph_threshold
        )
        self.last_location = None
        self.identity_misses = 0
        self.anchor_template = None
        self.anchor_location = None
        self.anchor_name_offset = None
        self.anchor_identity_score = 0.0
        # Short-term motion prior used only to gate identity reacquisition.
        # The outer DetectionTracker remains responsible for box smoothing
        # and short dropout prediction.
        self.location_velocity = np.zeros(2, dtype=np.float32)

    @staticmethod
    def _red_fraction(image):
        if image is None or image.size == 0:
            return 0.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = (
            cv2.inRange(hsv, (0, 100, 80), (15, 255, 255))
            | cv2.inRange(hsv, (170, 100, 80), (179, 255, 255))
        )
        return float(np.count_nonzero(mask)) / float(mask.size)

    def _template_blue_ref(self, ref_path=None):
        """从参照图(或名字模板)提取蓝色称号条的平均 BGR(蓝色像素均值)作颜色参照。

        参照图是离线截取的自己称号条(勋章), 与游戏内勋章是同款 UI 蓝;
        地形/瀑布的"蓝"与之有明显色差, 用它滤地形误检。参照图无蓝像素时返回
        None(颜色门不生效, 退回旧行为)。"""
        image = None
        if ref_path:
            _rp = Path(ref_path)
            if not _rp.is_absolute():
                _rp = REPO_ROOT / _rp
            if _rp.exists():
                data = np.fromfile(str(_rp), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            image = getattr(self, "template", None)
        if image is None or image.size == 0:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (90, 80, 50), (135, 255, 255))
        if int(mask.sum()) == 0:
            return None
        return tuple(float(v) for v in image[mask > 0].reshape(-1, 3).mean(axis=0))

    @staticmethod
    def _strip_blue_bgr(patch_bgr, patch_hsv):
        """称号条候选区域内"蓝像素"的均值 BGR; 无蓝像素返回 None。"""
        mask = cv2.inRange(patch_hsv, (90, 80, 50), (135, 255, 255))
        if int(mask.sum()) == 0:
            return None
        return tuple(float(v) for v in patch_bgr[mask > 0].reshape(-1, 3).mean(axis=0))

    def _find_color_anchor(self, gameplay):
        """Find the player's long blue title strip without OCR.

        The title strip is a much wider, more stable feature than the tiny
        three-character nametag. Geometry removes ropes/short UI fragments;
        a red-pixel check around the inferred body rejects unrelated blue UI.
        """
        if not self.color_anchor_enabled:
            return None
        height, width = gameplay.shape[:2]
        hsv = cv2.cvtColor(gameplay, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (90, 80, 50), (135, 255, 255))
        # The minimap and bottom HUD contain many blue pixels but cannot be a
        # title strip. Keep the exclusion relative to the 1370px reference.
        left_cut = min(width, max(0, int(round(width * 160.0 / 1370.0))))
        mask[:, :left_cut] = 0
        if self.max_valid_y is not None:
            mask[int(self.max_valid_y) + 1 :] = 0
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            # Rope/effect/宠物遮挡 can split the badge into several short
            # blue runs; bridge horizontal gaps (21px 覆盖宠物挡住的间隙,
            # 宠物体积大, 13px 桥接不够会漏检), never vertical ones.
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)),
        )
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        # ---- 同Y相邻条合并(抗宠物/绳子遮挡) ----
        # 宠物遮住称号条中段时, 连通域分裂成多段窄条(如 49px 两段), 各自
        # 不满足 80~180 宽/宽高比>=4 几何 -> 漏检。将同一行(Y 中心差<=8px)、
        # 水平相邻(间距<=40px)的碎片合并成完整称号条再走几何过滤。
        raw_boxes = []
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
            if h > 30:
                continue
            raw_boxes.append([x, y, w, h])
        raw_boxes.sort(key=lambda b: (b[1], b[0]))
        merged = []
        for b in raw_boxes:
            placed = False
            for m in merged:
                y_overlap = abs((b[1] + b[3] / 2.0) - (m[1] + m[3] / 2.0)) <= 8
                gap = b[0] - (m[0] + m[2])
                if y_overlap and -5 <= gap <= 40:   # 相邻或小重叠
                    x0 = min(m[0], b[0])
                    y0 = min(m[1], b[1])
                    x1 = max(m[0] + m[2], b[0] + b[2])
                    y1 = max(m[1] + m[3], b[1] + b[3])
                    m[0], m[1], m[2], m[3] = x0, y0, x1 - x0, y1 - y0
                    placed = True
                    break
            if not placed:
                merged.append(list(b))
        stats_eff = merged  # 合并后的伪 stats: [x, y, w, h]
        tag_height, tag_width = self.template.shape[:2]
        proposals = []
        for (x, y, box_width, box_height) in stats_eff:
            area = box_width * box_height
            if not (80 <= box_width <= 180 and 8 <= box_height <= 28):
                continue
            if box_width / max(box_height, 1) < 4.0 or area < 180:
                continue
            if self.max_valid_x is not None and x > self.max_valid_x:
                continue
            # 称号条水平中心 = 候选组边界框中心(稳定, 蓝像素质心对杂蓝/遮挡
            # 敏感会把中心拉偏, 导致红分区域偏移检测失败——已回退)。
            center_x = x + box_width / 2.0
            name_location = (
                int(round(center_x - tag_width / 2.0)),
                y - self.color_anchor_name_offset_y,
            )
            if name_location[1] < 0:
                continue
            player_center = (
                int(round(center_x)),
                name_location[1] - self.center_offset_y,
            )
            # 顶部越界检查放宽: 跳跃到高处时玩家框中心可能靠近画面上沿
            # (如 y=36, 框顶=36-45=-9, 角色大部分仍在画面内)。旧条件
            # pc_y < 45 会把高处跳跃的玩家误判为"框出画面"直接拒绝 →
            # 丢框卡背景。改为: 仅当【整个框】都出画面才拒绝(框顶<-90)。
            if (
                player_center[1] < -self.box_height
                or player_center[1] >= height + self.box_height // 2
            ):
                continue
            # 【用户要求: 取消红色身体特征验证, 识别只看蓝条】——红分不再作为
            # 拒绝条件, 也不参与打分(跳跃/边缘时红分骤降会误拒蓝条导致框飞)。
            # 红分仍计算并记录(供日志/可视化参考)。
            x0 = max(0, player_center[0] - self.box_width // 2)
            y0 = max(0, player_center[1] - self.box_height // 2)
            x1 = min(width, x0 + self.box_width)
            y1 = min(height, y0 + self.box_height)
            red_fraction = self._red_fraction(gameplay[y0:y1, x0:x1])
            # 【原版红分硬拒绝(已回退)】: 红分 < 阈值(0.02)丢弃该蓝条候选——
            # 红色身体特征验证滤掉无红色痕迹的 UI/地形蓝条。跳跃/爬绳的瞬时
            # 红分不足由 color_anchor_hold(6帧保持)兜底。
            if red_fraction < self.color_anchor_min_red_fraction:
                continue
            # 【蓝条颜色参照门】: 候选条蓝色像素均值必须接近自己勋章蓝。
            # 地形误检(红棕岩石+瀑布淡蓝)几何/红分都能通过, 只有颜色能滤掉。
            strip_blue = None
            if self.color_anchor_ref_bgr is not None:
                strip_blue = self._strip_blue_bgr(
                    gameplay[y : y + box_height, x : x + box_width],
                    hsv[y : y + box_height, x : x + box_width],
                )
                if strip_blue is not None:
                    _cdist = float(
                        np.abs(
                            np.asarray(strip_blue)
                            - np.asarray(self.color_anchor_ref_bgr)
                        ).sum()
                    )
                    if _cdist > self.color_anchor_color_tol:
                        continue
            fill = float(area) / float(box_width * box_height)
            shape_score = min(1.0, fill / 0.35)
            red_score = min(1.0, red_fraction / 0.12)
            identity_score = 0.65 * shape_score + 0.35 * red_score
            proposals.append(
                {
                    "location": name_location,
                    "template_score": float(shape_score),
                    "glyph_score": float(red_score),
                    "identity_score": float(identity_score),
                    "anchor_box": [x, y, box_width, box_height],
                    "red_fraction": float(red_fraction),
                    "blue_bgr": strip_blue,
                }
            )
        if not proposals:
            return None
        if self.last_location is not None:
            predicted = (
                float(self.last_location[0] + self.location_velocity[0]),
                float(self.last_location[1] + self.location_velocity[1]),
            )
            local_radius = max(self.color_anchor_local_radius, self.lock_radius)
            local = [
                item
                for item in proposals
                if self._location_distance(item["location"], predicted)
                <= local_radius
            ]
            if local:
                proposals = local
                proposals.sort(
                    key=lambda item: (
                        item["identity_score"]
                        - 0.0015
                        * self._location_distance(item["location"], predicted)
                    ),
                    reverse=True,
                )
            else:
                # 跳跃/大幅位移: 称号条移出 local_radius。color_anchor 的红分验证
                # (下方有红色身体特征)已过滤 UI/背景, 全局选依然安全——不能因此
                # 放弃 color_anchor 回退到模板匹配(模板匹配名字框不稳定, 会卡背景)。
                # 下方"远处+低分拒绝"仍兜底, 防止跳跃瞬间误跳远处低分候选。
                proposals.sort(
                    key=lambda item: item["identity_score"], reverse=True)
        else:
            proposals.sort(key=lambda item: item["identity_score"], reverse=True)
        best = proposals[0]
        if (
            self.last_location is not None
            and self._location_distance(best["location"], self.last_location)
            > max(self.color_anchor_local_radius, self.lock_radius)
            and best["identity_score"] < self.identity_threshold
        ):
            return None
        # 参照色 EMA 微调: 吸收"较可靠匹配"(色距 <= tol*0.9)的蓝条——
        # 离线参照图与实时蓝条有亮度差(实测量到 70/80), 只收 0.6 会让参照色
        # 永远停在离线值, 每帧都打"色距偏高"日志; 吸 0.9 后参照色收敛到实时
        # 蓝条, 后续判定更准。轻度遮挡/光影变化时参照色跟着实际蓝条走。
        if (self.color_anchor_ref_bgr is not None and best.get("blue_bgr") is not None):
            _cdist = float(
                np.abs(
                    np.asarray(best["blue_bgr"])
                    - np.asarray(self.color_anchor_ref_bgr)
                ).sum()
            )
            if _cdist <= self.color_anchor_color_tol * 0.9:
                self.color_anchor_ref_bgr = tuple(
                    0.7 * r + 0.3 * b
                    for r, b in zip(self.color_anchor_ref_bgr, best["blue_bgr"])
                )
                self._ref_warn_at = 0.0  # 重新锁定时刷新告警节流计时
            elif (_cdist > self.color_anchor_color_tol * 0.9
                  and (self._ref_warn_at is None
                       or time.time() - self._ref_warn_at > 2.5)):
                # 可疑锁定(色距偏高仍被接受): 打日志便于调参, 每2.5s最多一条
                self._ref_warn_at = time.time()
                logger.info(
                    f"[color_anchor] 蓝条色距偏高仍锁定: blueBGR="
                    f"({best['blue_bgr'][0]:.0f},{best['blue_bgr'][1]:.0f},"
                    f"{best['blue_bgr'][2]:.0f}) 参照=({self.color_anchor_ref_bgr[0]:.0f},"
                    f"{self.color_anchor_ref_bgr[1]:.0f},{self.color_anchor_ref_bgr[2]:.0f}) "
                    f"dist={_cdist:.0f}/{self.color_anchor_color_tol:.0f} "
                    f"置信={best['identity_score']:.2f} 红分={best['red_fraction']:.3f}")
        return best

    def _title_strip_proposals(self, gameplay, ref_color=None, color_tol=60):
        """返回画面中所有满足"蓝色称号条"几何特征的候选框列表
        [[x, y, w, h], ...](不排除任何一条)——供可视化红框显示, 看误检在哪。
        ref_color: 自己称号条的平均 BGR 颜色(B,G,R 三元组)。若提供, 候选条
        平均色与 ref_color 的色距超过 color_tol 则丢弃——地形/UI 的"蓝色"和
        中级冒险家勋章的同款蓝有明显色差, 用颜色参照可滤掉大量地形误检。
        """
        if gameplay is None or gameplay.size == 0:
            return []
        height, width = gameplay.shape[:2]
        hsv = cv2.cvtColor(gameplay, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (90, 80, 50), (135, 255, 255))
        left_cut = min(width, max(0, int(round(width * 160.0 / 1370.0))))
        mask[:, :left_cut] = 0
        if self.max_valid_y is not None:
            mask[int(self.max_valid_y) + 1:] = 0
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3)),
        )
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        # ---- 同Y相邻条合并(与 _find_color_anchor 相同, 抗宠物遮挡) ----
        raw_boxes = []
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
            if h > 30:
                continue
            raw_boxes.append([x, y, w, h])
        raw_boxes.sort(key=lambda b: (b[1], b[0]))
        merged = []
        for b in raw_boxes:
            placed = False
            for m in merged:
                y_overlap = abs((b[1] + b[3] / 2.0) - (m[1] + m[3] / 2.0)) <= 8
                gap = b[0] - (m[0] + m[2])
                if y_overlap and -5 <= gap <= 40:
                    x0 = min(m[0], b[0]); y0 = min(m[1], b[1])
                    x1 = max(m[0] + m[2], b[0] + b[2])
                    y1 = max(m[1] + m[3], b[1] + b[3])
                    m[0], m[1], m[2], m[3] = x0, y0, x1 - x0, y1 - y0
                    placed = True
                    break
            if not placed:
                merged.append(list(b))
        proposals = []
        for (x, y, box_width, box_height) in merged:
            area = box_width * box_height
            if not (80 <= box_width <= 180 and 8 <= box_height <= 28):
                continue
            if box_width / max(box_height, 1) < 4.0 or area < 300:
                continue
            if self.max_valid_x is not None and x > self.max_valid_x:
                continue
            # 颜色参照过滤: 候选条平均色与"自己称号条"色距过大 -> 地形/UI误检
            if ref_color is not None:
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(width, x + box_width)
                y1 = min(height, y + box_height)
                crop = gameplay[y0:y1, x0:x1]
                if crop.size:
                    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
                    dist = float(np.abs(mean_bgr - np.asarray(ref_color, dtype=np.float32)).sum())
                    if dist > color_tol:
                        continue
            proposals.append([x, y, box_width, box_height])
        return proposals

    def count_title_strips(self, gameplay, exclude_center=None, exclude_radius=70,
                           ref_color=None):
        """统计画面中【其他玩家】的蓝色称号条数量(中级冒险家勋章是蓝色横条)。

        复用 color_anchor 的蓝色检测但【不要求红色身体验证】——其他玩家的称号条
        同样满足蓝色几何特征, 数量 >=1 即说明场上有其他玩家, 用于暂停挂机。
        exclude_center: 自己玩家框中心(exclude_radius 内视为自己, 不计入)。
        ref_color: 自己称号条平均BGR(颜色参照, 滤地形/UI误检——地形蓝色与
        勋章同款蓝有色差, 色距超 color_tol 的候选不计)。
        返回 int(其他玩家称号条数量), 0=没有。
        """
        proposals = self._title_strip_proposals(gameplay, ref_color=ref_color)
        strips = 0
        for (x, y, box_width, box_height) in proposals:
            if exclude_center is not None:
                cx = x + box_width / 2.0
                cy = y + box_height / 2.0
                if (abs(cx - exclude_center[0]) <= exclude_radius
                        and abs(cy - exclude_center[1]) <= exclude_radius):
                    continue
            strips += 1
        return strips

    def seed_identity(self, identity, frame=None):
        """Accept an OCR identity and retain its title badge as a local anchor."""
        if isinstance(identity, dict):
            location = identity["location"]
            self.anchor_identity_score = float(
                identity.get("ocr_score", identity.get("identity_score", 0.0))
            )
            anchor_box = identity.get("anchor_box")
            if anchor_box is not None and frame is not None:
                anchor_x, anchor_y, anchor_width, anchor_height = (
                    int(value) for value in anchor_box
                )
                anchor_x0 = max(0, anchor_x)
                anchor_y0 = max(0, anchor_y)
                anchor_x1 = min(frame.shape[1], anchor_x + anchor_width)
                anchor_y1 = min(frame.shape[0], anchor_y + anchor_height)
                anchor = frame[anchor_y0:anchor_y1, anchor_x0:anchor_x1]
                if anchor.size:
                    self.anchor_template = anchor.copy()
                    self.anchor_location = (anchor_x0, anchor_y0)
                    self.anchor_name_offset = (
                        int(location[0]) - anchor_x0,
                        int(location[1]) - anchor_y0,
                    )
        else:
            location = identity
        self.last_location = (int(location[0]), int(location[1]))
        self.identity_misses = 0

    def reset(self):
        """强制重新识别: 清除身份锁定与 OCR 锚点状态。

        解决"识别框卡死在错误位置"的问题 —— 锁定态 last_location 只要
        半径 lock_radius 内仍有低阈值候选(宠物/干扰物误匹配), identity_misses
        就保持 0, 锁定永不失效。清空后下一次 detect 会重新全局搜索玩家,
        并等 OCR 重新确认身份(require_identity_seed 时)。
        """
        self.last_location = None
        self.identity_misses = 0
        self.anchor_template = None
        self.anchor_location = None
        self.anchor_name_offset = None
        self.anchor_identity_score = 0.0
        self.location_velocity[:] = 0.0

    def _build_detection(
        self,
        location,
        template_score,
        glyph_score,
        identity_score,
        identity_mode,
    ):
        tag_height, tag_width = self.template.shape[:2]
        player_x = int(location[0]) + tag_width // 2
        player_y = int(location[1]) - self.center_offset_y
        box_x = max(0, player_x - self.box_width // 2)
        box_y = max(0, player_y - self.box_height // 2)
        return {
            "class": "player",
            "label": "PLAYER",
            "label_zh": "player",
            "confidence": float(identity_score),
            "box": [box_x, box_y, self.box_width, self.box_height],
            "nametag_box": [int(location[0]), int(location[1]), tag_width, tag_height],
            "color": PLAYER_COLOR,
            "template_score": float(template_score),
            "glyph_score": float(glyph_score),
            "identity_score": float(identity_score),
            "identity_mode": identity_mode,
        }

    def _track_ocr_anchor(self, gameplay):
        """Track the OCR-confirmed title badge between expensive OCR passes."""
        if (
            self.anchor_template is None
            or self.anchor_location is None
            or self.anchor_name_offset is None
        ):
            return None
        anchor_height, anchor_width = self.anchor_template.shape[:2]
        anchor_x, anchor_y = self.anchor_location
        # 扩大搜索半径(240px): 玩家移动/宠物"花蘑菇仔"贴身时勋章也有较大位移,
        # 太小会跟丢导致回退到不可靠的模板匹配(模板匹配识别不了真名字框)。
        radius = max(240, int(self.lock_radius))
        x0 = max(0, anchor_x - radius)
        y0 = max(0, anchor_y - radius)
        x1 = min(gameplay.shape[1], anchor_x + anchor_width + radius)
        y1 = min(gameplay.shape[0], anchor_y + anchor_height + radius)
        search = gameplay[y0:y1, x0:x1]
        if search.shape[0] < anchor_height or search.shape[1] < anchor_width:
            return None
        result = cv2.matchTemplate(
            search,
            self.anchor_template,
            cv2.TM_CCOEFF_NORMED,
        )
        _, score, _, local_location = cv2.minMaxLoc(result)
        # 降低匹配阈值(0.75->0.62): 勋章被攻击特效/宠物部分遮挡时仍能跟上,
        # 减少回退到模板匹配的次数(模板匹配对名字框不可靠, 会漂移到背景/宠物)。
        if score < 0.62:
            return None
        next_anchor = (x0 + local_location[0], y0 + local_location[1])
        location = (
            next_anchor[0] + self.anchor_name_offset[0],
            next_anchor[1] + self.anchor_name_offset[1],
        )
        self.anchor_location = next_anchor
        self.last_location = location
        self.identity_misses = 0
        return self._build_detection(
            location,
            template_score=score,
            glyph_score=1.0,
            identity_score=min(1.0, score * max(0.90, self.anchor_identity_score)),
            identity_mode="ocr_anchor",
        )

    @staticmethod
    def _binary_f1(reference, candidate):
        intersection = int(np.logical_and(reference, candidate).sum())
        denominator = int(reference.sum()) + int(candidate.sum())
        if denominator == 0:
            return 1.0
        return 2.0 * intersection / denominator

    def _glyph_score(self, patch):
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        candidate_glyph = (
            patch_gray[self.glyph_slice] >= self.glyph_threshold
        )
        template_glyph = self.template_glyph
        h, w = template_glyph.shape
        col_w = w // 3
        if col_w < 2:
            return self._binary_f1(template_glyph, candidate_glyph)
        # Names are tiny and can be partially occluded, but accepting one
        # matching column lets unrelated UI text pass as the player name.
        # Require multiple supported columns and average the strongest ones.
        scores = []
        for i in range(3):
            x0 = i * col_w
            x1 = w if i == 2 else (i + 1) * col_w
            t_col = template_glyph[:, x0:x1]
            c_col = candidate_glyph[:, x0:x1]
            score = self._binary_f1(t_col, c_col)
            scores.append(score)
        scores.sort(reverse=True)
        supported = [score for score in scores if score >= 0.35]
        if len(supported) < self.glyph_min_columns:
            return float(sum(supported) / max(len(supported), 1) * 0.5)
        return float(sum(scores[: self.glyph_min_columns]) / self.glyph_min_columns)

    def _find_candidates(self, gameplay):
        result = cv2.matchTemplate(
            gameplay,
            self.template,
            cv2.TM_CCOEFF_NORMED,
        )
        work = result.copy()
        tag_height, tag_width = self.template.shape[:2]
        candidates = []
        for _ in range(self.candidate_count):
            _, template_score, _, location = cv2.minMaxLoc(work)
            if template_score < self.threshold:
                break
            x, y = location
            if self.max_valid_y is not None and y > self.max_valid_y:
                x0 = max(0, x - tag_width // 2)
                x1 = min(work.shape[1], x + tag_width // 2 + 1)
                y0 = max(0, y - tag_height // 2)
                y1 = min(work.shape[0], y + tag_height // 2 + 1)
                work[y0:y1, x0:x1] = -1.0
                continue
            # 过滤 x > max_valid_x 的候选: 避开误检区域(如地图蘑菇等背景纹理)
            if self.max_valid_x is not None and x > self.max_valid_x:
                x0 = max(0, x - tag_width // 2)
                x1 = min(work.shape[1], x + tag_width // 2 + 1)
                y0 = max(0, y - tag_height // 2)
                y1 = min(work.shape[0], y + tag_height // 2 + 1)
                work[y0:y1, x0:x1] = -1.0
                continue
            patch = gameplay[y : y + tag_height, x : x + tag_width]
            glyph_score = self._glyph_score(patch)
            identity_score = (
                template_score * (1.0 - self.glyph_weight)
                + glyph_score * self.glyph_weight
            )
            candidates.append(
                {
                    "location": location,
                    "template_score": float(template_score),
                    "glyph_score": float(glyph_score),
                    "identity_score": float(identity_score),
                }
            )

            # Remove the whole nametag neighborhood so one tag contributes
            # one proposal instead of several adjacent correlation peaks.
            x0 = max(0, x - tag_width // 2)
            x1 = min(work.shape[1], x + tag_width // 2 + 1)
            y0 = max(0, y - tag_height // 2)
            y1 = min(work.shape[0], y + tag_height // 2 + 1)
            work[y0:y1, x0:x1] = -1.0
        return candidates

    @staticmethod
    def _location_distance(first, second):
        return float(np.hypot(first[0] - second[0], first[1] - second[1]))

    def _select_candidate(self, candidates, gameplay_shape):
        lock_active = (
            self.last_location is not None
            and self.identity_misses <= self.reacquire_misses
        )
        if self.require_identity_seed and not lock_active:
            return None, "awaiting_ocr"
        minimum_identity = (
            self.local_identity_threshold
            if lock_active
            else self.identity_threshold
        )
        eligible = [
            candidate
            for candidate in candidates
            if candidate["identity_score"] >= minimum_identity
        ]
        if not eligible:
            return None, "local" if lock_active else "global"

        if lock_active:
            nearby = []
            for candidate in eligible:
                distance = self._location_distance(
                    candidate["location"], self.last_location
                )
                if distance > self.lock_radius:
                    continue
                proximity = 1.0 - distance / self.lock_radius
                candidate["selection_score"] = (
                    candidate["identity_score"] + 0.15 * proximity
                )
                nearby.append(candidate)
            if not nearby:
                return None, "local"
            nearby.sort(key=lambda item: item["selection_score"], reverse=True)
            # A local candidate must remain identity-consistent with the
            # previous lock.  Do not keep a lock alive merely because some
            # unrelated nearby text clears the relaxed local threshold.
            if (
                self.last_location is not None
                and self._location_distance(nearby[0]["location"], self.last_location)
                > self.lock_radius * 0.70
                and nearby[0]["identity_score"] < self.identity_threshold
            ):
                return None, "local"
            return nearby[0], "local"

        gameplay_height, gameplay_width = gameplay_shape[:2]
        center = (gameplay_width / 2.0, gameplay_height / 2.0)
        max_center_distance = max(float(np.hypot(*center)), 1.0)
        for candidate in eligible:
            tag_x, tag_y = candidate["location"]
            tag_height, tag_width = self.template.shape[:2]
            tag_center = (tag_x + tag_width / 2.0, tag_y + tag_height / 2.0)
            distance = self._location_distance(tag_center, center)
            center_proximity = 1.0 - min(1.0, distance / max_center_distance)
            candidate["selection_score"] = (
                candidate["identity_score"]
                + self.center_weight * center_proximity
            )
        eligible.sort(key=lambda item: item["selection_score"], reverse=True)
        if (
            len(eligible) > 1
            and eligible[0]["selection_score"] - eligible[1]["selection_score"]
            < self.identity_margin
        ):
            return None, "global"
        return eligible[0], "global"

    def detect(self, frame, gameplay_height):
        gameplay = frame[:gameplay_height]
        if (
            gameplay.shape[0] < self.template.shape[0]
            or gameplay.shape[1] < self.template.shape[1]
        ):
            self.identity_misses += 1
            return None

        anchored = self._track_ocr_anchor(gameplay)
        if anchored is not None:
            # 辅助锚点: 在玩家位置附近找鲜艳红色翅膀(玩家独有, 宠物和地图地形都没有)
            wing_pos = self._find_wing(gameplay, self.last_location)
            if wing_pos is not None:
                anchored["wing_pos"] = wing_pos
            return anchored

        # In no-OCR mode the blue title strip is a wider identity anchor than
        # the tiny name text. Use it before global nametag matching, while the
        # red body check in _find_color_anchor rejects unrelated blue UI.
        color_anchor = self._find_color_anchor(gameplay)
        if color_anchor is not None:
            location = color_anchor["location"]
            if self.last_location is not None:
                delta = np.asarray(location, dtype=np.float32) - np.asarray(
                    self.last_location, dtype=np.float32
                )
                # Smooth only the identity prior. The outer tracker owns box
                # smoothing and prediction shown to the user.
                self.location_velocity = (
                    self.location_velocity * 0.45 + delta * 0.55
                )
            self.last_location = location
            self.identity_misses = 0
            detection = self._build_detection(
                location,
                template_score=color_anchor["template_score"],
                glyph_score=color_anchor["glyph_score"],
                identity_score=color_anchor["identity_score"],
                identity_mode="color_anchor",
            )
            detection["anchor_box"] = color_anchor["anchor_box"]
            detection["red_fraction"] = color_anchor["red_fraction"]
            return detection

        # color_anchor 丢失时: 短暂漏检(跳跃/宠物遮挡)用上一帧位置保持——
        # 立即回退 local 模板匹配会造成黄框漂移(模板匹配不稳定)。
        # 连续丢失超过 keep_color_anchor_misses 帧才回退 local 重新捕获。
        if self.last_location is not None and self.identity_misses < self.keep_color_anchor_misses:
            self.identity_misses += 1
            return self._build_detection(
                self.last_location,
                template_score=0.6,
                glyph_score=0.5,
                identity_score=0.55,
                identity_mode="color_anchor_hold",
            )

        candidates = self._find_candidates(gameplay)
        candidate, identity_mode = self._select_candidate(
            candidates, gameplay.shape
        )
        if candidate is None:
            self.identity_misses += 1
            return None

        location = candidate["location"]
        self.last_location = location
        self.identity_misses = 0
        return self._build_detection(
            location,
            template_score=candidate["template_score"],
            glyph_score=candidate["glyph_score"],
            identity_score=candidate["identity_score"],
            identity_mode=identity_mode,
        )

    def _find_wing(self, gameplay, player_loc):
        """在玩家位置附近找鲜艳红色翅膀(玩家独有, 宠物和地图地形都没有鲜艳红色翅膀)。

        返回翅膀位置 (x, y) 或 None。"翅膀锚点"作为辅助验证, 防止识别框飘到
        半空/地形(那些地方没有翅膀)。
        """
        if player_loc is None:
            return None
        px, py = int(player_loc[0]), int(player_loc[1])
        radius = 60
        x0 = max(0, px - radius)
        y0 = max(0, py - radius)
        x1 = min(gameplay.shape[1], px + radius)
        y1 = min(gameplay.shape[0], py + radius)
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        crop = gameplay[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 鲜艳红色翅膀: H 0-15 或 170-179, S>=100, V>=80
        mask = (
            cv2.inRange(hsv, (0, 100, 80), (15, 255, 255))
            | cv2.inRange(hsv, (170, 100, 80), (179, 255, 255))
        )
        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 30:
                continue
            cx = int(cents[i][0]) + x0
            cy = int(cents[i][1]) + y0
            if best is None or area > best[0]:
                best = (area, cx, cy)
        return (best[1], best[2]) if best else None


class AsyncNameOcrLocator:
    """Periodically confirm an exact player name without blocking video FPS."""

    def __init__(
        self,
        player_name,
        template_shape,
        confidence=0.70,
        submit_interval=0.50,
        refresh_interval=3.0,
        ocr_threads=2,
        title_text=None,
        title_to_name_offset_y=27,
    ):
        self.player_name = str(player_name)
        self.template_height = int(template_shape[0])
        self.template_width = int(template_shape[1])
        self.confidence = float(confidence)
        self.submit_interval = float(submit_interval)
        self.refresh_interval = float(refresh_interval)
        self.ocr_threads = int(ocr_threads)
        self.title_text = "" if title_text is None else str(title_text).strip()
        self.title_to_name_offset_y = int(title_to_name_offset_y)
        self.condition = threading.Condition()
        self.pending_frame = None
        self.pending_id = 0
        self.consumed_id = 0
        self.latest_result = None
        self.last_submit = 0.0
        self.stopped = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def available():
        return importlib.util.find_spec("rapidocr_onnxruntime") is not None

    @staticmethod
    def _entries(result):
        if isinstance(result, tuple) and result:
            return result[0] or []
        if (
            isinstance(result, list)
            and len(result) == 1
            and isinstance(result[0], list)
            and result[0]
            and isinstance(result[0][0], list)
        ):
            return result[0]
        return result or []

    def _locate_exact_name(self, result, frame_id):
        """OCR 主角判定(用户规则 2026-08-20):
          1) 称号匹配: 文本与称号(如"中级冒险家勋章")有 >=2 个相同字符
             -> 称号候选(OCR 识别不全/遮挡也认);
          2) 场上多个称号 -> 用名字区分: 名字与玩家名有 >=1 个字符相同
             且与称号垂直对齐 -> 主角(有一个字符合就是主角);
          3) 检测不到称号 -> 名字与玩家名有 >=2 个字符相同 -> 主角。
          完整名字(3字全中)永远最高优先级。
        """
        names_strong = []   # 与玩家名 >=2 字符相同
        names_weak = []     # 与玩家名 >=1 字符相同(含 strong)
        titles = []         # 与称号文本 >=2 字符相同
        title_text = getattr(self, "title_text", "")
        player_name = self.player_name
        title_to_name_offset_y = getattr(self, "title_to_name_offset_y", 27)
        for entry in self._entries(result):
            if len(entry) < 3:
                continue
            box, text, score = entry[0], str(entry[1]), float(entry[2])
            if score < self.confidence:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            candidate = {
                "box": [
                    int(round(x_min)),
                    int(round(y_min)),
                    max(1, int(round(x_max - x_min))),
                    max(1, int(round(y_max - y_min))),
                ],
                "center": ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
                "ocr_score": score,
                "text": text,
            }
            text_set = set(text)
            # 名字: 完整 3 字全中 -> strong; 否则按字符交集分级
            if player_name in text:
                names_strong.append(candidate)
                names_weak.append(candidate)
            else:
                name_hits = sum(1 for ch in player_name if ch in text_set)
                if len(text) <= 6 and name_hits >= 2:
                    names_strong.append(candidate)
                if len(text) <= 6 and name_hits >= 1:
                    names_weak.append(candidate)
            # 称号: 与称号文本 >=2 个相同字符。称号仅7字, 文本长度<=8
            # (防"欢迎来到冒险岛世界"这类长UI文本含"冒险"2字误判)
            if title_text and len(text) <= 8:
                title_hits = sum(1 for ch in title_text if ch in text_set)
                if title_hits >= 2:
                    titles.append(candidate)

        def name_identity(name, title=None):
            text_center_x, text_center_y = name["center"]
            return {
                "location": (
                    int(round(text_center_x - self.template_width / 2.0)),
                    int(round(text_center_y - self.template_height / 2.0)),
                ),
                "nametag_box": name["box"],
                # anchor_box 用于 seed_identity 保存锚点模板(后续每帧 matchTemplate
                # 跟踪)。有称号用称号框(大, 好跟踪); 无称号(规则3: 名字两字定位)
                # 用名字框自身当锚点——否则 anchor_box=None 建不了锚, 回退到
                # 模板匹配兜底会受 max_valid_x 限制选错位置, 框漂移不跟随。
                "anchor_box": title["box"] if title is not None else name["box"],
                "ocr_score": name["ocr_score"],
                "text": name["text"],
                "identity_source": "ocr_name",
                "frame_id": int(frame_id),
                "timestamp": time.time(),
            }

        def title_identity(title):
            title_x, title_y, title_width, _ = title["box"]
            return {
                "location": (
                    int(round(title_x + title_width / 2.0 - self.template_width / 2.0 - 4.0)),
                    int(round(title_y - title_to_name_offset_y)),
                ),
                "nametag_box": None,
                "anchor_box": title["box"],
                "ocr_score": title["ocr_score"],
                "text": title["text"],
                "identity_source": "ocr_title",
                "frame_id": int(frame_id),
                "timestamp": time.time(),
            }

        def vertically_aligned(name, title):
            name_x, name_y, name_width, name_height = name["box"]
            title_x, title_y, title_width, _ = title["box"]
            name_center_x = name_x + name_width / 2.0
            title_center_x = title_x + title_width / 2.0
            vertical_gap = title_y - (name_y + name_height)
            return (
                -2 <= vertical_gap <= 42
                and abs(name_center_x - title_center_x)
                <= max(title_width * 0.60, self.template_width)
            )

        # ---- 1) 无称号锚配置: 只能用名字(>=2字符) ----
        if not title_text:
            if not names_strong:
                return None
            return name_identity(
                max(names_strong, key=lambda item: item["ocr_score"]))

        # ---- 2) 有称号候选 ----
        if titles:
            if len(titles) == 1:
                title = titles[0]
                # 单称号: 直接按称号定位(称号任意两字即主角)——乱码名字
                # (如"林超"含"超"1字)不得因 1 字匹配而覆盖称号判定。
                # 仅当名字完整 3 字全中且与称号对齐时用名字(位置更精确)。
                for name in names_strong:
                    if player_name in name["text"] and vertically_aligned(name, title):
                        return name_identity(name, title)
                return title_identity(title)
            # 多称号: 必须用名字区分(>=1字符 + 垂直对齐)
            for pool in (names_strong, names_weak):
                for name in pool:
                    for title in titles:
                        if vertically_aligned(name, title):
                            return name_identity(name, title)
            # 无法区分 -> 不冒险跳错人
            return None

        # ---- 3) 无称号候选: 名字>=2字符即主角 ----
        if names_strong:
            return name_identity(
                max(names_strong, key=lambda item: item["ocr_score"]))
        return None


    def submit(self, gameplay):
        now = time.time()
        with self.condition:
            if self.stopped or now - self.last_submit < self.submit_interval:
                return
            self.pending_frame = gameplay.copy()
            self.pending_id += 1
            self.last_submit = now
            self.condition.notify()

    def latest(self, max_age=10.0):
        with self.condition:
            result = None if self.latest_result is None else dict(self.latest_result)
        if result is None or time.time() - result["timestamp"] > max_age:
            return None
        return result

    def _run(self):
        try:
            from rapidocr_onnxruntime import RapidOCR

            ocr = RapidOCR(
                intra_op_num_threads=self.ocr_threads,
                inter_op_num_threads=1,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return

        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.stopped or self.pending_id > self.consumed_id
                )
                if self.stopped:
                    return
                frame = self.pending_frame.copy()
                frame_id = self.pending_id
                self.consumed_id = frame_id
            try:
                result = self._locate_exact_name(ocr(frame), frame_id)
                if result is not None:
                    with self.condition:
                        self.latest_result = result
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            refresh_deadline = time.time() + self.refresh_interval
            with self.condition:
                while not self.stopped:
                    remaining = refresh_deadline - time.time()
                    if remaining <= 0.0:
                        break
                    self.condition.wait(timeout=remaining)
                if self.stopped:
                    return

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=2.0)


class YoloMonsterDetector:
    def __init__(
        self,
        model_path,
        confidence,
        iou,
        device,
        image_size,
        labels,
        stump_edge_min_height=60,
        stump_edge_bottom_margin=3,
    ):
        from ultralytics import YOLO

        resolved_model = Path(model_path)
        if not resolved_model.is_absolute():
            resolved_model = REPO_ROOT / resolved_model
        if not resolved_model.exists():
            raise FileNotFoundError(f"YOLO model not found: {resolved_model}")

        self.model_path = resolved_model.resolve()
        self.model = YOLO(str(self.model_path))
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = device
        self.image_size = int(image_size)
        self.labels = set(labels)
        self.stump_edge_min_height = float(stump_edge_min_height)
        self.stump_edge_bottom_margin = float(stump_edge_bottom_margin)
        validate_model_classes(self.model.names, self.labels)

    def detect(self, frame, gameplay_height):
        gameplay = frame[:gameplay_height]
        result = self.model.predict(
            source=gameplay,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for coordinates, score, class_id in zip(boxes, scores, classes):
            model_class = normalize_model_class(result.names[class_id])
            if model_class not in CLASS_INFO or model_class not in self.labels:
                continue
            x1, y1, x2, y2 = coordinates
            raw_box = [
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                max(1, int(round(x2 - x1))),
                max(1, int(round(y2 - y1))),
            ]
            if model_class == "stump" and reject_edge_clipped_stump(
                raw_box,
                gameplay.shape[0],
                min_height=self.stump_edge_min_height,
                bottom_margin=self.stump_edge_bottom_margin,
            ):
                continue
            info = CLASS_INFO[model_class]
            detections.append(
                {
                    "class": model_class,
                    "label": info["display"],
                    "label_zh": info["zh"],
                    "confidence": float(score),
                    "box": raw_box,
                    "color": info["color"],
                }
            )
        return deduplicate_detections(detections)


def draw_coordinate_entity(output, detection):
    x, y, width, height = detection["box"]
    color = detection["color"]
    center_x, center_y = [int(round(value)) for value in detection["center_px"]]
    cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
    cv2.circle(output, (center_x, center_y), 4, color, -1)

    velocity_x, velocity_y = detection["velocity_px_s"]
    if detection["speed_px_s"] >= 10.0:
        vector_scale = 0.16
        end_x = int(round(center_x + velocity_x * vector_scale))
        end_y = int(round(center_y + velocity_y * vector_scale))
        cv2.arrowedLine(
            output,
            (center_x, center_y),
            (end_x, end_y),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.35,
        )

    text = detection["entity_id"]
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1
    )
    text_x = min(x, max(0, output.shape[1] - text_width - 7))
    text_y = max(text_height + 7, y)
    cv2.rectangle(
        output,
        (text_x, text_y - text_height - 7),
        (text_x + text_width + 7, text_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        output,
        text,
        (text_x + 3, text_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def build_coordinate_panel(
    height,
    player,
    detections,
    fps,
    minimap_player=None,
    minimap_other_players=None,
    width=340,
):
    panel = np.full((height, width, 3), 26, dtype=np.uint8)
    cv2.putText(
        panel,
        "ENTITY COORDINATES",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"screen px | origin top-left | {fps:.1f} FPS",
        (14, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if minimap_player is None:
        cv2.putText(
            panel,
            "MINIMAP PLAYER MISSED",
            (14, 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        y = 98
    else:
        map_x, map_y = minimap_player["map_px"]
        norm_x, norm_y = minimap_player["map_norm"]
        cv2.putText(
            panel,
            "MINIMAP PLAYER P1",
            (14, 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"map xy ({map_x:.1f}, {map_y:.1f})",
            (24, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"norm ({norm_x:.3f}, {norm_y:.3f})",
            (24, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (175, 175, 175),
            1,
            cv2.LINE_AA,
        )
        y = 136

    minimap_other_players = list(minimap_other_players or ())
    cv2.putText(
        panel,
        f"MINIMAP OTHER PLAYERS {len(minimap_other_players)}",
        (14, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 80, 255),
        1,
        cv2.LINE_AA,
    )
    y += 18
    for index, other in enumerate(minimap_other_players[:6], start=1):
        map_x, map_y = other["map_px"]
        norm_x, norm_y = other["map_norm"]
        cv2.putText(
            panel,
            f"R{index} map ({map_x:.1f}, {map_y:.1f})",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (0, 80, 255),
            1,
            cv2.LINE_AA,
        )
        y += 16
        cv2.putText(
            panel,
            f"norm ({norm_x:.3f}, {norm_y:.3f})",
            (34, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (175, 175, 175),
            1,
            cv2.LINE_AA,
        )
        y += 20
    if len(minimap_other_players) > 6:
        cv2.putText(
            panel,
            f"+ {len(minimap_other_players) - 6} more red markers",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        y += 20

    entities = ([] if player is None else [player]) + list(detections)
    for index, entity in enumerate(entities[:12]):
        color = entity["color"]
        short_label = ENTITY_SHORT_LABELS.get(entity["class"], entity["class"])
        predicted = " PRED" if entity["tracking_state"] == "PREDICTED" else ""
        cv2.putText(
            panel,
            f"{entity['entity_id']}  {short_label}  {entity['motion_state']}{predicted}",
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            1,
            cv2.LINE_AA,
        )
        center_x, center_y = entity["center_px"]
        velocity_x, velocity_y = entity["velocity_px_s"]
        y += 20
        cv2.putText(
            panel,
            f"xy ({center_x:.0f}, {center_y:.0f})   v ({velocity_x:+.0f}, {velocity_y:+.0f})",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        relative = entity.get("relative_to_player")
        if relative is not None:
            delta_x, delta_y = relative["delta_px"]
            y += 18
            cv2.putText(
                panel,
                f"rel ({delta_x:+.0f}, {delta_y:+.0f})   d {relative['distance_px']:.0f}",
                (24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (175, 175, 175),
                1,
                cv2.LINE_AA,
            )
        y += 25
        if y > height - 30:
            remaining = len(entities) - index - 1
            if remaining > 0:
                cv2.putText(
                    panel,
                    f"+ {remaining} more in JSON summary",
                    (14, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (150, 150, 150),
                    1,
                    cv2.LINE_AA,
                )
            break
    return panel


def draw_detections(
    frame,
    detections,
    fps,
    player=None,
    minimap_player=None,
    selected_labels=None,
    minimap_other_players=None,
):
    output = frame.copy()
    counts = {label: 0 for label in REQUIRED_CLASSES}
    for label in selected_labels or ():
        counts.setdefault(label, 0)
    minimap_other_players = list(minimap_other_players or ())
    canvas_marker = minimap_player or (
        minimap_other_players[0] if minimap_other_players else None
    )
    if canvas_marker is not None:
        canvas_x, canvas_y, canvas_width, canvas_height = canvas_marker[
            "canvas_frame_box"
        ]
        cv2.rectangle(
            output,
            (canvas_x, canvas_y),
            (canvas_x + canvas_width, canvas_y + canvas_height),
            (0, 200, 255),
            1,
        )
    if minimap_player is not None:
        marker_x, marker_y = [
            int(round(value)) for value in minimap_player["frame_px"]
        ]
        diamond = np.asarray(
            [
                [marker_x, marker_y - 6],
                [marker_x + 6, marker_y],
                [marker_x, marker_y + 6],
                [marker_x - 6, marker_y],
            ],
            dtype=np.int32,
        )
        cv2.polylines(output, [diamond], True, (0, 230, 255), 2, cv2.LINE_AA)
    for index, other in enumerate(minimap_other_players, start=1):
        marker_x, marker_y = [
            int(round(value)) for value in other["frame_px"]
        ]
        cv2.circle(output, (marker_x, marker_y), 5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(
            output,
            (marker_x, marker_y),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            7,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"R{index}",
            (marker_x + 6, marker_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    if player is not None:
        draw_coordinate_entity(output, player)
        if "nametag_box" in player:
            tag_x, tag_y, tag_width, tag_height = player["nametag_box"]
            cv2.rectangle(
                output,
                (tag_x, tag_y),
                (tag_x + tag_width, tag_y + tag_height),
                PLAYER_COLOR,
                1,
            )
    for detection in detections:
        counts.setdefault(detection["class"], 0)
        counts[detection["class"]] += 1
        draw_coordinate_entity(output, detection)

    player_status = "P1" if player is not None else "MISSED"
    minimap_status = "P1" if minimap_player is not None else "MISSED"
    status = (
        f"OBSERVE ONLY | SCREEN XY | player {player_status} | "
        f"map {minimap_status} red {len(minimap_other_players)} | {len(detections)} monsters | "
        f"{fps:.1f} FPS | origin top-left"
    )
    (status_width, _), _ = cv2.getTextSize(
        status, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1
    )
    cv2.rectangle(
        output,
        (0, 0),
        (min(output.shape[1], status_width + 20), 34),
        (25, 25, 25),
        -1,
    )
    cv2.putText(
        output,
        status,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )
    panel = build_coordinate_panel(
        output.shape[0],
        player,
        detections,
        fps,
        minimap_player,
        minimap_other_players,
    )
    return np.hstack((output, panel)), counts


def save_image(path, image):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded, data = cv2.imencode(output_path.suffix or ".png", image)
    if not encoded:
        raise RuntimeError(f"Unable to encode image: {output_path}")
    data.tofile(output_path)


def serialize_entity(detection):
    if detection is None:
        return None
    payload = {
        "entity_id": detection["entity_id"],
        "class": detection["class"],
        "label_zh": detection["label_zh"],
        "confidence": round(detection["confidence"], 4),
        "box": detection["box"],
        "center_px": detection["center_px"],
        "center_norm": detection["center_norm"],
        "velocity_px_s": detection["velocity_px_s"],
        "speed_px_s": detection["speed_px_s"],
        "motion_state": detection["motion_state"],
        "tracking_state": detection["tracking_state"],
        "track_id": detection["track_id"],
        "missed_frames": detection["missed_frames"],
    }
    if "nametag_box" in detection:
        payload["nametag_box"] = detection["nametag_box"]
    for score_name in ("template_score", "glyph_score", "identity_score"):
        if score_name in detection:
            payload[score_name] = round(float(detection[score_name]), 4)
    if "identity_mode" in detection:
        payload["identity_mode"] = detection["identity_mode"]
    if "relative_to_player" in detection:
        payload["relative_to_player"] = detection["relative_to_player"]
    return payload


def serialize_minimap_player(player):
    if player is None:
        return None
    return {
        "map_px": [round(float(value), 3) for value in player["map_px"]],
        "map_norm": [round(float(value), 6) for value in player["map_norm"]],
        "frame_px": [round(float(value), 3) for value in player["frame_px"]],
        "marker_box_map": [int(value) for value in player["marker_box_map"]],
        "canvas_frame_box": [
            int(value) for value in player["canvas_frame_box"]
        ],
        "canvas_size": [int(value) for value in player["canvas_size"]],
        "pixel_count": int(player["pixel_count"]),
        "fill_ratio": round(float(player["fill_ratio"]), 6),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read-only YOLO detection for selected MapleStory monster classes."
    )
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument(
        "--window-title-token",
        default=DEFAULT_WINDOW_TOKEN,
        help="Visible window-title substring to capture without focusing it.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument(
        "--stump-edge-min-height",
        type=float,
        default=60.0,
        help="Reject stump boxes shorter than this when clipped at the gameplay bottom.",
    )
    parser.add_argument(
        "--stump-edge-bottom-margin",
        type=float,
        default=3.0,
        help="Bottom-boundary tolerance for the clipped-stump guard.",
    )
    parser.add_argument("--fps-limit", type=float, default=12.0)
    parser.add_argument(
        "--gameplay-height",
        type=int,
        help="Override the gameplay/UI boundary for the captured client height.",
    )
    parser.add_argument(
        "--track-max-missed",
        type=int,
        default=4,
        help="Keep a predicted box for this many consecutive missed frames.",
    )
    parser.add_argument(
        "--track-smoothing",
        type=float,
        default=0.65,
        help="Detection weight for box smoothing; lower values are steadier.",
    )
    parser.add_argument("--track-match-iou", type=float, default=0.15)
    parser.add_argument("--track-center-distance", type=float, default=1.25)
    parser.add_argument("--track-max-width-ratio", type=float, default=1.70)
    parser.add_argument("--track-max-height-ratio", type=float, default=2.00)
    parser.add_argument("--track-max-area-ratio", type=float, default=2.50)
    parser.add_argument("--track-size-cost-weight", type=float, default=0.25)
    parser.add_argument(
        "--track-min-hits",
        type=int,
        default=2,
        help="Require this many matched frames before showing a new track.",
    )
    parser.add_argument(
        "--track-high-confidence",
        type=float,
        default=0.75,
        help="Show a new track immediately when confidence reaches this value.",
    )
    parser.add_argument(
        "--player-name",
        help="Use nametag/<name>_player.png for read-only player localization.",
    )
    parser.add_argument("--player-template")
    parser.add_argument("--player-threshold", type=float)
    parser.add_argument("--player-identity-threshold", type=float)
    parser.add_argument("--player-local-identity-threshold", type=float)
    parser.add_argument("--player-identity-margin", type=float)
    parser.add_argument("--player-lock-radius", type=float)
    parser.add_argument("--player-reacquire-misses", type=int)
    parser.add_argument("--player-ocr-confidence", type=float)
    parser.add_argument("--player-ocr-max-age", type=float, default=10.0)
    parser.add_argument("--no-player-ocr", action="store_true")
    parser.add_argument("--player-track-max-missed", type=int, default=8)
    parser.add_argument("--no-player", action="store_true")
    parser.add_argument(
        "--monster-labels",
        nargs="+",
        default=["僵尸蘑菇", "刺蘑菇"],
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--snapshot")
    parser.add_argument("--raw-snapshot")
    parser.add_argument("--summary")
    return parser.parse_args()


def resolve_player_template(args, cfg):
    if args.player_template:
        return Path(args.player_template)
    if args.player_name:
        safe_name = Path(str(args.player_name)).name.replace("/", "").replace("\\", "")
        return Path("nametag") / f"{safe_name}_player.png"
    return Path("nametag") / f"{cfg['nametag']['name']}.png"


def main():
    args = parse_args()
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("confidence must be in (0, 1]")
    if args.fps_limit <= 0:
        raise ValueError("fps-limit must be positive")

    cfg = load_config(args.cfg)
    minimap_cfg = cfg.get("minimap", {})
    labels = [normalize_label(label) for label in args.monster_labels]
    window_token = args.window_title_token
    configured_gameplay_height = (
        int(args.gameplay_height)
        if args.gameplay_height is not None
        else int(cfg["ui_coords"]["ui_y_start"])
    )
    if configured_gameplay_height <= 0:
        raise ValueError("gameplay-height must be positive")
    gameplay_reference_width = cfg["ui_coords"].get("reference_width")
    gameplay_height = configured_gameplay_height
    window_title = find_visible_window_title(window_token)
    detector = YoloMonsterDetector(
        args.model,
        args.confidence,
        args.iou,
        args.device,
        args.image_size,
        labels,
        stump_edge_min_height=args.stump_edge_min_height,
        stump_edge_bottom_margin=args.stump_edge_bottom_margin,
    )
    tracker = DetectionTracker(
        max_missed=args.track_max_missed,
        smoothing=args.track_smoothing,
        match_iou=args.track_match_iou,
        max_center_distance=args.track_center_distance,
        min_confirmed_hits=args.track_min_hits,
        high_confidence_confirm=args.track_high_confidence,
        max_width_ratio=args.track_max_width_ratio,
        max_height_ratio=args.track_max_height_ratio,
        max_area_ratio=args.track_max_area_ratio,
        size_cost_weight=args.track_size_cost_weight,
    )
    monster_coordinates = EntityCoordinateTracker()
    player_detector = None
    player_tracker = None
    player_coordinates = None
    player_ocr = None
    last_ocr_frame_id = 0
    if not args.no_player:
        overlay_cfg = cfg["perception_overlay"]
        player_template = resolve_player_template(args, cfg)
        player_threshold = (
            float(args.player_threshold)
            if args.player_threshold is not None
            else float(overlay_cfg["player_match_threshold"])
        )
        player_detector = ReadOnlyPlayerDetector(
            player_template,
            threshold=player_threshold,
            box_size=overlay_cfg["player_box_size"],
            center_offset_y=abs(int(cfg["nametag"]["offset"][1])),
            identity_threshold=(
                float(args.player_identity_threshold)
                if args.player_identity_threshold is not None
                else float(overlay_cfg.get("player_identity_threshold", 0.48))
            ),
            local_identity_threshold=(
                float(args.player_local_identity_threshold)
                if args.player_local_identity_threshold is not None
                else float(
                    overlay_cfg.get("player_local_identity_threshold", 0.38)
                )
            ),
            identity_margin=(
                float(args.player_identity_margin)
                if args.player_identity_margin is not None
                else float(overlay_cfg.get("player_identity_margin", 0.015))
            ),
            glyph_threshold=int(overlay_cfg.get("player_glyph_threshold", 130)),
            glyph_weight=float(overlay_cfg.get("player_glyph_weight", 0.70)),
            glyph_min_columns=int(overlay_cfg.get("player_glyph_min_columns", 2)),
            candidate_count=int(overlay_cfg.get("player_candidate_count", 16)),
            lock_radius=(
                float(args.player_lock_radius)
                if args.player_lock_radius is not None
                else float(overlay_cfg.get("player_lock_radius", 180.0))
            ),
            reacquire_misses=(
                int(args.player_reacquire_misses)
                if args.player_reacquire_misses is not None
                else int(overlay_cfg.get("player_reacquire_misses", 12))
            ),
            center_weight=float(overlay_cfg.get("player_center_weight", 0.12)),
            max_valid_x=overlay_cfg.get("player_max_valid_x"),
            max_valid_y=overlay_cfg.get("player_max_valid_y"),
            require_identity_seed=(
                bool(args.player_name)
                and not args.no_player_ocr
                and AsyncNameOcrLocator.available()
            ),
            color_anchor_enabled=bool(
                overlay_cfg.get("player_color_anchor_enabled", True)
            ),
            color_anchor_name_offset_y=int(
                overlay_cfg.get("player_color_anchor_name_offset_y", 24)
            ),
            color_anchor_min_red_fraction=float(
                overlay_cfg.get("player_color_anchor_min_red_fraction", 0.02)
            ),
            color_anchor_local_radius=float(
                overlay_cfg.get("player_color_anchor_local_radius", 260.0)
            ),
            color_anchor_color_tol=float(
                overlay_cfg.get("player_color_anchor_color_tol", 80.0)
            ),
            color_anchor_ref_path=overlay_cfg.get("player_color_anchor_ref_file"),
        )
        if (
            args.player_name
            and not args.no_player_ocr
            and AsyncNameOcrLocator.available()
        ):
            player_ocr = AsyncNameOcrLocator(
                args.player_name,
                player_detector.template.shape,
                confidence=(
                    float(args.player_ocr_confidence)
                    if args.player_ocr_confidence is not None
                    else float(overlay_cfg.get("player_ocr_confidence", 0.70))
                ),
                submit_interval=float(
                    overlay_cfg.get("player_ocr_submit_interval", 0.50)
                ),
                refresh_interval=float(
                    overlay_cfg.get("player_ocr_refresh_interval", 3.0)
                ),
                ocr_threads=int(overlay_cfg.get("player_ocr_threads", 2)),
                title_text=overlay_cfg.get("player_title_anchor"),
            )
        player_tracker = DetectionTracker(
            max_missed=args.player_track_max_missed,
            smoothing=0.55,
            match_iou=0.05,
            max_center_distance=1.80,
        )
        player_coordinates = EntityCoordinateTracker(
            velocity_smoothing=0.50,
            move_threshold_px_s=30.0,
            vertical_threshold_px_s=40.0,
        )
    capture = ReadOnlyWindowCapture(window_title)

    started = time.time()
    last_frame_time = started
    fps = 0.0
    frames = 0
    latest = None
    latest_raw = None
    detections = []
    player = None
    minimap_player = None
    minimap_other_players = []
    red_marker_tracker = MinimapRedMarkerTracker(
        confirm_frames=minimap_cfg.get("other_player_confirm_frames", 2),
        max_missed=minimap_cfg.get("other_player_max_missed_frames", 1),
        max_distance=minimap_cfg.get("other_player_match_distance_px", 8.0),
    )
    counts = {label: 0 for label in REQUIRED_CLASSES}
    try:
        if not args.headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

        while True:
            loop_started = time.time()
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            frame_timestamp = time.time()
            minimap_markers = locate_minimap_players(frame, minimap_cfg)
            minimap_player = minimap_markers["player"]
            minimap_other_players = red_marker_tracker.update(
                minimap_markers["other_players"]
            )
            gameplay_height = resolve_gameplay_height(
                frame.shape,
                configured_gameplay_height,
                gameplay_reference_width,
            )
            raw_detections = detector.detect(frame, gameplay_height)
            detections = tracker.update(raw_detections)
            frame_width = int(frame.shape[1])
            detections = monster_coordinates.update(
                detections,
                frame_timestamp,
                frame_width,
                gameplay_height,
                prefix="M",
            )

            player = None
            if player_detector is not None:
                if player_ocr is not None:
                    player_ocr.submit(frame[:gameplay_height])
                    ocr_identity = player_ocr.latest(args.player_ocr_max_age)
                    if (
                        ocr_identity is not None
                        and ocr_identity["frame_id"] > last_ocr_frame_id
                    ):
                        player_detector.seed_identity(ocr_identity, frame)
                        last_ocr_frame_id = ocr_identity["frame_id"]
                raw_player = player_detector.detect(frame, gameplay_height)
                tracked_players = player_tracker.update(
                    [] if raw_player is None else [raw_player]
                )
                if tracked_players:
                    best_player = min(
                        tracked_players,
                        key=lambda item: (
                            item["missed_frames"],
                            -item["confidence"],
                        ),
                    )
                    enriched_players = player_coordinates.update(
                        [best_player],
                        frame_timestamp,
                        frame_width,
                        gameplay_height,
                        fixed_entity_id="P1",
                    )
                    player = enriched_players[0]
                else:
                    player_coordinates.update(
                        [],
                        frame_timestamp,
                        frame_width,
                        gameplay_height,
                        fixed_entity_id="P1",
                    )
            detections = attach_player_relative_coordinates(detections, player)
            latest_raw = frame.copy()
            now = time.time()
            instantaneous_fps = 1.0 / max(now - last_frame_time, 1e-6)
            fps = instantaneous_fps if fps == 0 else fps * 0.85 + instantaneous_fps * 0.15
            last_frame_time = now
            latest, counts = draw_detections(
                frame,
                detections,
                fps,
                player,
                minimap_player,
                labels,
                minimap_other_players,
            )
            frames += 1

            if not args.headless:
                cv2.imshow(WINDOW_TITLE, latest)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.max_frames and frames >= args.max_frames:
                break
            if args.duration > 0 and now - started >= args.duration:
                break

            remaining = 1.0 / args.fps_limit - (time.time() - loop_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        capture.stop()
        if player_ocr is not None:
            player_ocr.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    if latest is not None and args.snapshot:
        save_image(args.snapshot, latest)
    if latest_raw is not None and args.raw_snapshot:
        save_image(args.raw_snapshot, latest_raw)

    summary = {
        "observe_only": True,
        "input_events_sent": False,
        "window_title": window_title,
        "model": str(detector.model_path),
        "model_classes": list(detector.model.names.values()),
        "selected_classes": [CLASS_INFO[label]["zh"] for label in labels],
        "coordinate_system": {
            "type": "captured_window_screen_pixels",
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "gameplay_size": [
                None if latest_raw is None else int(latest_raw.shape[1]),
                gameplay_height,
            ],
            "world_coordinates": False,
        },
        "player_template": (
            None
            if player_detector is None
            else str(player_detector.template_path)
        ),
        "player_identity": {
            "ocr_enabled": player_ocr is not None,
            "ocr_error": None if player_ocr is None else player_ocr.error,
            "identity_seeded": last_ocr_frame_id > 0,
        },
        "tracking": {
            "enabled": True,
            "max_missed_frames": tracker.max_missed,
            "smoothing": tracker.smoothing,
            "match_iou": tracker.match_iou,
            "max_center_distance": tracker.max_center_distance,
            "max_width_ratio": tracker.max_width_ratio,
            "max_height_ratio": tracker.max_height_ratio,
            "max_area_ratio": tracker.max_area_ratio,
            "size_cost_weight": tracker.size_cost_weight,
            "min_confirmed_hits": tracker.min_confirmed_hits,
            "high_confidence_confirm": tracker.high_confidence_confirm,
        },
        "frames": frames,
        "fps": round(fps, 2),
            "player": serialize_entity(player),
            "minimap": {
            "coordinate_system": {
                "type": "minimap_canvas_pixels",
                "origin": "top_left",
                "x_axis": "right",
                "y_axis": "down",
                "normalized_range": [0.0, 1.0],
                "world_coordinates": False,
            },
            "player": serialize_minimap_player(minimap_player),
            "other_players": [
                serialize_minimap_player(marker)
                for marker in minimap_other_players
            ],
        },
        "counts": {
            CLASS_INFO[label]["zh"]: counts[label]
            for label in sorted(counts)
        },
        "detections": [serialize_entity(detection) for detection in detections],
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
