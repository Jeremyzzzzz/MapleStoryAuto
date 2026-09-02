import argparse
import json
import os
import sys
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

from src.engine.HealthMonitor import HealthMonitor
from src.input.GameWindowCapturor import GameWindowCapturor
from src.utils.common import get_mask, load_yaml, override_cfg


WINDOW_TITLE = "MapleStory Perception - OBSERVE ONLY"
PLAYER_COLOR = (0, 255, 255)
STUMP_COLOR = (0, 80, 255)
BLUE_SNAIL_COLOR = (255, 180, 0)
RED_SNAIL_COLOR = (80, 80, 255)
MOTION_COLOR = (220, 60, 220)
ADVISORY_READY_COLOR = (80, 220, 80)
ADVISORY_RISK_COLOR = (50, 70, 255)
ADVISORY_IDLE_COLOR = (190, 190, 190)


def load_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to load image: {path}")
    return image


def save_image(path, image):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded, data = cv2.imencode(output_path.suffix or ".png", image)
    if not encoded:
        raise RuntimeError(f"Unable to encode image: {path}")
    data.tofile(output_path)


def intersection_over_union(first, second):
    ax, ay, aw, ah = first["box"]
    bx, by, bw, bh = second["box"]
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / float(union)


def non_maximum_suppression(detections, threshold=0.25):
    kept = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if all(intersection_over_union(detection, other) <= threshold for other in kept):
            kept.append(detection)
    return kept


def intersection_fraction(first, second):
    ax, ay, aw, ah = first["box"]
    bx, by, bw, bh = second["box"]
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    smaller_area = min(aw * ah, bw * bh)
    return 0.0 if smaller_area <= 0 else intersection / float(smaller_area)


class PlayerDetector:
    def __init__(self, cfg):
        self.template = load_image(f"nametag/{cfg['nametag']['name']}.png")
        self.threshold = cfg["perception_overlay"]["player_match_threshold"]
        self.box_width, self.box_height = cfg["perception_overlay"]["player_box_size"]
        self.ui_y_start = cfg["ui_coords"]["ui_y_start"]

    def detect(self, frame):
        gameplay = frame[: self.ui_y_start]
        result = cv2.matchTemplate(gameplay, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)
        if score < self.threshold:
            return None

        tag_height, tag_width = self.template.shape[:2]
        player_x = location[0] + tag_width // 2
        player_y = location[1] - 30
        box_x = max(0, player_x - self.box_width // 2)
        box_y = max(0, player_y - self.box_height // 2)
        return {
            "label": "PLAYER",
            "score": float(score),
            "box": (box_x, box_y, self.box_width, self.box_height),
            "nametag_box": (location[0], location[1], tag_width, tag_height),
            "center": (player_x, player_y),
        }


class MonsterDetector:
    def __init__(self, cfg):
        overlay_cfg = cfg["perception_overlay"]
        template = load_image(overlay_cfg["stump_template"])
        mask = get_mask(template, (0, 255, 0))
        scale = overlay_cfg["stump_scale"]
        size = (
            max(1, int(round(template.shape[1] * scale))),
            max(1, int(round(template.shape[0] * scale))),
        )
        template = cv2.resize(template, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        self.stump_templates = [(template, mask), (cv2.flip(template, 1), cv2.flip(mask, 1))]
        self.stump_threshold = overlay_cfg["stump_match_threshold"]
        self.max_monsters = overlay_cfg["max_monsters"]
        self.ui_y_start = cfg["ui_coords"]["ui_y_start"]

    def _detect_stumps(self, frame):
        gameplay = frame[: self.ui_y_start]
        detections = []
        for template, mask in self.stump_templates:
            height, width = template.shape[:2]
            result = cv2.matchTemplate(
                gameplay,
                template,
                cv2.TM_CCORR_NORMED,
                mask=mask,
            )
            result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
            local_maximum = cv2.dilate(result, np.ones((15, 15), dtype=np.uint8))
            rows, columns = np.where(
                (result >= local_maximum - 1e-7) & (result >= self.stump_threshold)
            )
            for y, x in zip(rows, columns):
                detections.append(
                    {
                        "label": "STUMP",
                        "score": float(result[y, x]),
                        "box": (int(x), int(y), width, height),
                        "color": STUMP_COLOR,
                        "method": "template",
                    }
                )
        return non_maximum_suppression(detections)[: self.max_monsters]

    def _detect_colored_snails(self, frame, player):
        gameplay = frame[: self.ui_y_start]
        hsv = cv2.cvtColor(gameplay, cv2.COLOR_BGR2HSV)
        masks = {
            "BLUE SNAIL": cv2.inRange(hsv, (75, 120, 70), (100, 255, 255)),
            "RED SNAIL": cv2.bitwise_or(
                cv2.inRange(hsv, (0, 180, 100), (8, 255, 255)),
                cv2.inRange(hsv, (170, 180, 100), (179, 255, 255)),
            ),
        }
        colors = {"BLUE SNAIL": BLUE_SNAIL_COLOR, "RED SNAIL": RED_SNAIL_COLOR}
        detections = []
        for label, mask in masks.items():
            # Static overlays and the portal occupy these screen regions.
            mask[:, :100] = 0
            mask[:230, :500] = 0
            mask[:125, 1050:] = 0
            mask[600:, 850:1030] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for index in range(1, count):
                x, y, width, height, area = map(int, stats[index])
                if not 12 <= area <= 500 or width > 35 or height > 35:
                    continue
                box = (
                    max(0, x - 14),
                    max(0, y - 10),
                    min(gameplay.shape[1] - max(0, x - 14), width + 28),
                    min(gameplay.shape[0] - max(0, y - 10), height + 22),
                )
                detection = {
                    "label": label,
                    "score": min(0.99, 0.55 + area / 1000.0),
                    "box": box,
                    "color": colors[label],
                    "method": "color candidate",
                }
                if player and intersection_over_union(detection, player) > 0.05:
                    continue
                detections.append(detection)
        return non_maximum_suppression(detections, threshold=0.15)

    def detect(self, frame, player):
        detections = self._detect_stumps(frame)
        detections.extend(self._detect_colored_snails(frame, player))
        if player:
            detections = [
                item for item in detections if intersection_over_union(item, player) <= 0.05
            ]
        return detections[: self.max_monsters]


# --------------------------------------------------------------------------
# 实体跟踪 + 坐标(codex 移植): 检测框平滑、跨帧跟踪、速度/运动状态、相对坐标
# 从 tools/yolo_monster_viewer.py 的 DetectionTracker / EntityCoordinateTracker
# 移植, 适配 auto_combat 的字段(label/box(x,y,w,h)/score)
# --------------------------------------------------------------------------
def box_center(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def box_iou(first, second):
    fx, fy, fw, fh = first
    sx, sy, sw, sh = second
    ow = max(0, min(fx + fw, sx + sw) - max(fx, sx))
    oh = max(0, min(fy + fh, sy + sh) - max(fy, sy))
    inter = ow * oh
    union = fw * fh + sw * sh - inter
    return 0.0 if union <= 0 else inter / float(union)


def normalized_center_distance(first, second):
    fc = box_center(first)
    sc = box_center(second)
    return np.hypot(fc[0] - sc[0], fc[1] - sc[1])


class DetectionTracker:
    """平滑检测框 + 跨帧跟踪 + 漏检桥接(codex 移植, 适配 auto_combat 字段)。"""

    def __init__(self, max_missed=4, smoothing=0.65, match_iou=0.15,
                 max_center_distance=1.25):
        self.max_missed = int(max_missed)
        self.smoothing = float(smoothing)
        self.match_iou = float(match_iou)
        self.max_center_distance = float(max_center_distance)
        self.tracks = {}
        self.next_track_id = 1

    def _predicted_box(self, track):
        bx, by, _, _ = track["box"]
        vx, vy = track["velocity"]
        return [bx + vx, by + vy, track["box"][2], track["box"][3]]

    def _to_detection(self, track):
        d = {k: v for k, v in track.items() if k not in ("velocity", "age", "missed")}
        d["box"] = (
            max(0, int(round(track["box"][0]))),
            max(0, int(round(track["box"][1]))),
            max(1, int(round(track["box"][2]))),
            max(1, int(round(track["box"][3]))),
        )
        d["score"] = float(track["score"])
        d["missed_frames"] = int(track["missed"])
        return d

    def _new_track(self, det):
        t = dict(det)
        t.update({
            "track_id": self.next_track_id,
            "box": [float(v) for v in det["box"]],
            "score": float(det["score"]),
            "velocity": [0.0, 0.0],
            "age": 1,
            "missed": 0,
        })
        self.tracks[self.next_track_id] = t
        self.next_track_id += 1

    def update(self, detections):
        detections = [dict(d) for d in detections]
        predicted = {tid: self._predicted_box(t) for tid, t in self.tracks.items()}
        candidates = []
        for tid, t in self.tracks.items():
            for di, det in enumerate(detections):
                if t["label"] != det.get("label"):
                    continue
                iou = box_iou(predicted[tid], det["box"])
                cd = normalized_center_distance(predicted[tid], det["box"])
                if iou >= self.match_iou or cd <= self.max_center_distance:
                    candidates.append(((1.0 - iou) + cd * 0.35, tid, di))
        matched_t = set()
        matched_d = set()
        for _, tid, di in sorted(candidates):
            if tid in matched_t or di in matched_d:
                continue
            t = self.tracks[tid]
            det = detections[di]
            old_box = t["box"]
            pbox = predicted[tid]
            mbox = [float(v) for v in det["box"]]
            a = self.smoothing
            t["box"] = [pv * (1 - a) + mv * a for pv, mv in zip(pbox, mbox)]
            mv_xy = [mbox[0] - old_box[0], mbox[1] - old_box[1]]
            t["velocity"] = [ov * 0.5 + nv * 0.5 for ov, nv in zip(t["velocity"], mv_xy)]
            t["score"] = t["score"] * (1 - a) + float(det["score"]) * a
            for k in ("label", "color", "has_hp_bar"):
                if k in det:
                    t[k] = det[k]
            t["age"] += 1
            t["missed"] = 0
            matched_t.add(tid)
            matched_d.add(di)
        expired = []
        for tid, t in self.tracks.items():
            if tid in matched_t:
                continue
            t["box"] = predicted[tid]
            t["velocity"] = [v * 0.8 for v in t["velocity"]]
            t["score"] *= 0.9
            t["age"] += 1
            t["missed"] += 1
            if t["missed"] > self.max_missed:
                expired.append(tid)
        for tid in expired:
            del self.tracks[tid]
        for di, det in enumerate(detections):
            if di not in matched_d:
                self._new_track(det)
        return [self._to_detection(self.tracks[tid]) for tid in sorted(self.tracks)]


class EntityCoordinateTracker:
    """跨帧计算速度 + 运动状态(codex 移植)。"""

    def __init__(self, velocity_smoothing=0.5, move_threshold_px_s=20.0,
                 vertical_threshold_px_s=15.0):
        self.velocity_smoothing = float(velocity_smoothing)
        self.move_threshold_px_s = float(move_threshold_px_s)
        self.vertical_threshold_px_s = float(vertical_threshold_px_s)
        self.states = {}

    def _motion_state(self, vx, vy):
        speed = float(np.hypot(vx, vy))
        if abs(vy) >= self.vertical_threshold_px_s and abs(vy) >= abs(vx) * 0.75:
            return "UP" if vy < 0 else "DOWN"
        if speed >= self.move_threshold_px_s:
            return "MOVE"
        return "STILL"

    def update(self, detections, timestamp, frame_width, gameplay_height,
               prefix="M", fixed_entity_id=None):
        enriched = []
        active = set()
        for det in detections:
            eid = fixed_entity_id or f"{prefix}{det['track_id']}"
            active.add(eid)
            cx, cy = box_center(det["box"])
            prev = self.states.get(eid)
            vx = vy = 0.0
            if prev is not None:
                elapsed = max(float(timestamp) - prev["timestamp"], 1e-6)
                mx = (cx - prev["center"][0]) / elapsed
                my = (cy - prev["center"][1]) / elapsed
                a = self.velocity_smoothing
                vx = prev["velocity"][0] * (1 - a) + mx * a
                vy = prev["velocity"][1] * (1 - a) + my * a
            ms = self._motion_state(vx, vy)
            self.states[eid] = {
                "center": (cx, cy), "velocity": (vx, vy), "timestamp": float(timestamp),
            }
            item = dict(det)
            item.update({
                "entity_id": eid,
                "center_px": [round(cx, 1), round(cy, 1)],
                "center_norm": [
                    round(cx / max(1, frame_width), 5),
                    round(cy / max(1, gameplay_height), 5),
                ],
                "velocity_px_s": [round(vx, 1), round(vy, 1)],
                "speed_px_s": round(float(np.hypot(vx, vy)), 1),
                "motion_state": ms,
                "tracking_state": "PREDICTED" if int(det.get("missed_frames", 0)) > 0 else "DETECTED",
            })
            enriched.append(item)
        self.states = {eid: s for eid, s in self.states.items() if eid in active}
        return enriched


def attach_player_relative_coordinates(detections, player):
    if player is None:
        return [dict(d, relative_to_player=None) for d in detections]
    px, py = player["center_px"]
    out = []
    for det in detections:
        cx, cy = det["center_px"]
        dx, dy = cx - px, cy - py
        out.append(dict(det, relative_to_player={
            "dx": round(dx, 1), "dy": round(dy, 1),
            "distance": round(float(np.hypot(dx, dy)), 1),
        }))
    return out


class YoloMonsterDetector:
    def __init__(self, cfg, model_path, confidence, device, image_size):
        from ultralytics import YOLO

        self.model = YOLO(str(Path(model_path).resolve()))
        self.confidence = confidence
        self.device = device
        self.image_size = image_size
        self.max_monsters = cfg["perception_overlay"]["max_monsters"]
        self.ui_y_start = cfg["ui_coords"]["ui_y_start"]
        # codex 移植: 检测框平滑跟踪 + 实体坐标(速度/运动状态)
        self.tracker = DetectionTracker(max_missed=4, smoothing=0.65)
        self.coord_tracker = EntityCoordinateTracker()

    def detect(self, frame, player):
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            max_det=self.max_monsters,
            verbose=False,
        )[0]
        colors = {
            "red_snail": RED_SNAIL_COLOR,
            "blue_snail": BLUE_SNAIL_COLOR,
            "stump": STUMP_COLOR,
            "slime": (180, 255, 120),
            "green_mushroom": (60, 200, 60),
            "flower_mushroom": (255, 100, 200),
            "orange_mushroom": (0, 140, 255),
            "zombie_mushroom": (30, 170, 255),
            "thorn_mushroom": (255, 180, 40),
            "mob": (200, 200, 200),
            "monster": (200, 200, 200),
        }
        detections = []
        if result.boxes is None:
            return detections
        # Precompute the green HP-bar mask once per frame: monsters with an
        # active green HP bar over their head are the ones that have been
        # hit (real targets), so they should be preferred over idle ones.
        hp_mask = self._hp_bar_mask(frame)
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for coordinates, score, class_id in zip(boxes, scores, classes):
            class_name = str(result.names[class_id])
            if class_name not in colors:
                continue
            x1, y1, x2, y2 = coordinates
            if y1 >= self.ui_y_start:
                continue
            detection = {
                "label": class_name.replace("_", " ").upper(),
                "score": float(score),
                "box": (
                    max(0, int(round(x1))),
                    max(0, int(round(y1))),
                    max(1, int(round(x2 - x1))),
                    max(1, int(round(y2 - y1))),
                ),
                "color": colors[class_name],
                "method": "yolo",
            }
            # Green HP bar right above the monster box => an active,
            # already-hit target; prefer these. (Must be set AFTER the dict
            # is built - referencing 'detection' inside its own literal is
            # an UnboundLocalError.)
            detection["has_hp_bar"] = self._box_has_hp_bar(hp_mask, detection)
            if player and intersection_over_union(detection, player) > 0.05:
                continue
            detections.append(detection)
        detections = non_maximum_suppression(detections, threshold=0.35)
        # codex 移植: 跟踪平滑 + 实体坐标(速度/运动状态) + 相对玩家坐标
        detections = self.tracker.update(detections)
        detections = self.coord_tracker.update(
            detections, time.time(), frame.shape[1], self.ui_y_start, prefix="M")
        if player is not None:
            pc = player.get("center") or player.get("center_px")
            if pc is not None:
                detections = attach_player_relative_coordinates(
                    detections, {"center_px": pc})
        return detections

    def _hp_bar_mask(self, frame):
        """Return a boolean mask of green HP-bar pixels in the frame."""
        if frame is None:
            return None
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Maple HP bars are a vivid green (BGR ~[71,204,64]).
            lower = np.array([40, 100, 60], dtype=np.uint8)
            upper = np.array([90, 255, 200], dtype=np.uint8)
            return cv2.inRange(hsv, lower, upper)
        except Exception:
            return None

    def _box_has_hp_bar(self, hp_mask, detection):
        """True if a green HP bar sits just above the monster box."""
        if hp_mask is None:
            return False
        x, y, w, h = detection["box"]
        # Scan a strip above the box (10-24px tall, same width as the box).
        strip_y0 = max(0, y - 26)
        strip_y1 = max(0, y - 4)
        if strip_y1 <= strip_y0:
            return False
        strip = hp_mask[strip_y0:strip_y1, max(0, x - 4):min(hp_mask.shape[1], x + w + 4)]
        if strip.size == 0:
            return False
        return bool(np.count_nonzero(strip) >= max(6, int(w * 0.25)))


class MotionDetector:
    def __init__(
        self,
        cfg,
        threshold=22,
        min_area=45,
        max_area=3500,
        candidate_score=0.70,
    ):
        self.ui_y_start = cfg["ui_coords"]["ui_y_start"]
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.candidate_score = candidate_score
        self.previous = None
        self.camera_motion = False
        self.tracks = []

    def _exclusion_mask(self, shape):
        height, width = shape
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[:40, :] = 0
        mask[40:min(225, height), :min(255, width)] = 0
        if width > 1050:
            mask[40:min(130, height), 1050:] = 0
        return mask

    def detect(self, frame, player, monsters):
        gameplay = frame[: self.ui_y_start]
        gray = cv2.cvtColor(gameplay, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.previous is None or self.previous.shape != gray.shape:
            self.previous = gray
            self.camera_motion = False
            self.tracks = []
            return []

        difference = cv2.absdiff(self.previous, gray)
        self.previous = gray
        _, motion_mask = cv2.threshold(
            difference, self.threshold, 255, cv2.THRESH_BINARY
        )
        valid_mask = self._exclusion_mask(gray.shape)
        motion_mask = cv2.bitwise_and(motion_mask, valid_mask)

        valid_pixels = max(1, cv2.countNonZero(valid_mask))
        changed_fraction = cv2.countNonZero(motion_mask) / float(valid_pixels)
        self.camera_motion = changed_fraction > 0.16
        if self.camera_motion:
            for monster in monsters:
                monster["moving"] = False
            self.tracks = []
            return []

        raw_motion_mask = motion_mask.copy()
        motion_mask = cv2.morphologyEx(
            motion_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
        )
        motion_mask = cv2.dilate(
            motion_mask, np.ones((7, 7), dtype=np.uint8), iterations=2
        )

        candidates = []
        hsv = cv2.cvtColor(gameplay, cv2.COLOR_BGR2HSV)
        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if not self.min_area <= area <= self.max_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if width < 8 or height < 8 or width > 150 or height > 150:
                continue
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(gameplay.shape[1] - x, width + padding * 2)
            height = min(gameplay.shape[0] - y, height + padding * 2)
            if (
                y < 65
                or x < 18
                or x + width > gameplay.shape[1] - 18
                or width * height > 6500
            ):
                continue
            saturation = hsv[y : y + height, x : x + width, 1]
            moving_pixels = raw_motion_mask[y : y + height, x : x + width] > 0
            moving_pixel_count = np.count_nonzero(moving_pixels)
            colorful_fraction = np.count_nonzero(
                (saturation >= 55) & moving_pixels
            ) / float(max(1, moving_pixel_count))
            if colorful_fraction < 0.15:
                continue
            candidate = {
                "label": "MOTION ?",
                "score": min(
                    1.0,
                    area / float(max(1, width * height))
                    * (0.65 + colorful_fraction),
                ),
                "box": (x, y, width, height),
                "color": MOTION_COLOR,
                "method": "motion candidate",
            }
            if player and intersection_fraction(candidate, player) > 0.18:
                continue
            candidates.append(candidate)

        candidates = non_maximum_suppression(candidates, threshold=0.12)
        unmatched = []
        for monster in monsters:
            monster["moving"] = False
        for candidate in candidates:
            matched = False
            for monster in monsters:
                if intersection_fraction(candidate, monster) > 0.18:
                    monster["moving"] = True
                    matched = True
            if not matched:
                unmatched.append(candidate)
        return self._update_tracks(unmatched)[:3]

    def _update_tracks(self, candidates):
        updated = []
        matched_track_indexes = set()
        for candidate in candidates:
            x, y, width, height = candidate["box"]
            center = (x + width / 2.0, y + height / 2.0)
            best_index = None
            best_distance = 65.0
            for index, track in enumerate(self.tracks):
                if index in matched_track_indexes:
                    continue
                distance = float(
                    np.hypot(
                        center[0] - track["center"][0],
                        center[1] - track["center"][1],
                    )
                )
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is None:
                track = {
                    "candidate": candidate,
                    "center": center,
                    "origin": center,
                    "hits": 1,
                    "missed": 0,
                    "max_displacement": 0.0,
                }
            else:
                previous_track = self.tracks[best_index]
                matched_track_indexes.add(best_index)
                displacement = float(
                    np.hypot(
                        center[0] - previous_track["origin"][0],
                        center[1] - previous_track["origin"][1],
                    )
                )
                track = {
                    "candidate": candidate,
                    "center": center,
                    "origin": previous_track["origin"],
                    "hits": previous_track["hits"] + 1,
                    "missed": 0,
                    "max_displacement": max(
                        previous_track["max_displacement"], displacement
                    ),
                }
            updated.append(track)

        for index, track in enumerate(self.tracks):
            if index in matched_track_indexes:
                continue
            missed = track["missed"] + 1
            if missed <= 2:
                updated.append({**track, "missed": missed})
        self.tracks = updated
        persistent = [
            track["candidate"]
            for track in self.tracks
            if track["missed"] == 0
            and track["hits"] >= 3
            and track["max_displacement"] >= 5.0
            and track["candidate"]["score"] >= self.candidate_score
        ]
        return sorted(persistent, key=lambda item: item["score"], reverse=True)


class AdvisoryEvaluator:
    """Estimate combat-relevant geometry without issuing any input."""

    def __init__(self, cfg):
        advisory_cfg = cfg["combat_advisory"]
        self.attack_horizontal_px = float(advisory_cfg["attack_horizontal_px"])
        self.attack_vertical_px = float(advisory_cfg["attack_vertical_px"])
        self.dodge_horizontal_px = float(advisory_cfg["dodge_horizontal_px"])
        self.dodge_vertical_px = float(advisory_cfg["dodge_vertical_px"])
        self.immediate_danger_px = float(advisory_cfg["immediate_danger_px"])
        self.approach_speed_px_s = float(advisory_cfg["approach_speed_px_s"])
        self.track_match_px = float(advisory_cfg["track_match_px"])
        self.previous_tracks = []
        self.latest = self._empty_result("WAITING")
        # Target lock-on: keep attacking the SAME monster across frames so a
        # single-frame YOLO miss (common for the small melee-range targets a
        # warrior fights) does not flip the target to a far-away monster.
        # The warrior then stops "chasing the far monster" and stays on the
        # one next to it (which is what is hitting the player).
        self.locked_target = None       # {label, box, center, timestamp}
        self.lock_hold_seconds = 3.0    # tolerate this long without re-detection

    @staticmethod
    def _center(detection):
        x, y, width, height = detection["box"]
        return (x + width / 2.0, y + height / 2.0)

    @staticmethod
    def _empty_result(status):
        return {
            "status": status,
            "attack_ready": False,
            "dodge_risk": False,
            "suggested_direction": None,
            "target_label": None,
            "target_box": None,
            "distance_px": None,
            "horizontal_distance_px": None,
            "vertical_distance_px": None,
            "approach_speed_px_s": None,
        }

    def reset(self, status="WAITING"):
        self.previous_tracks = []
        self.locked_target = None
        self.latest = self._empty_result(status)
        return self.latest

    def evaluate(self, player, monsters, timestamp, camera_motion=False, facing=None, stationary=False):
        if camera_motion:
            return self.reset("PAUSED CAMERA")
        if not player:
            return self.reset("PLAYER MISSED")
        if not monsters:
            return self.reset("NO TARGET")

        player_center = self._center(player)
        current_tracks = []
        available_previous = set(range(len(self.previous_tracks)))
        candidates = []

        for monster in monsters:
            center = self._center(monster)
            horizontal_distance = abs(center[0] - player_center[0])
            vertical_distance = abs(center[1] - player_center[1])
            distance = float(np.hypot(horizontal_distance, vertical_distance))
            best_index = None
            best_match_distance = self.track_match_px
            for index in available_previous:
                previous = self.previous_tracks[index]
                if previous["label"] != monster["label"]:
                    continue
                match_distance = float(
                    np.hypot(
                        center[0] - previous["center"][0],
                        center[1] - previous["center"][1],
                    )
                )
                if match_distance < best_match_distance:
                    best_match_distance = match_distance
                    best_index = index

            approach_speed = None
            if best_index is not None:
                previous = self.previous_tracks[best_index]
                available_previous.remove(best_index)
                elapsed = timestamp - previous["timestamp"]
                if elapsed > 1e-6:
                    approach_speed = (previous["distance"] - distance) / elapsed

            current_tracks.append(
                {
                    "label": monster["label"],
                    "center": center,
                    "distance": distance,
                    "timestamp": timestamp,
                }
            )
            candidates.append(
                {
                    "monster": monster,
                    "center": center,
                    "distance": distance,
                    "horizontal_distance": horizontal_distance,
                    "vertical_distance": vertical_distance,
                    "approach_speed": approach_speed,
                }
            )

        self.previous_tracks = current_tracks

        # 定向清怪: 只在 facing 方向(前方)选目标, 背后的怪忽略。
        front_candidates = candidates
        back_count = 0
        if facing is not None:
            front_candidates = []
            for c in candidates:
                c_left = c["center"][0] < player_center[0]
                is_front = ((facing == "left" and c_left) or
                            (facing == "right" and not c_left))
                if is_front:
                    front_candidates.append(c)
                else:
                    back_count += 1
        if not front_candidates:
            # 前方没怪(可能都在背后): 返回 NO TARGET + 背后怪数(供转身)
            self.locked_target = None
            self.latest = self._empty_result("NO TARGET")
            self.latest["front_count"] = 0
            self.latest["back_count"] = back_count
            return self.latest

        # --- Target lock-on --------------------------------------------
        # Every-frame "nearest monster" selection flips the target whenever a
        # single YOLO frame misses the close monster: the warrior then runs
        # to the far monster while the close one keeps hitting the player.
        # Instead: prefer the locked target; re-detect it each frame; if it
        # is missed keep its last known box. A target that is INSIDE attack
        # range is held indefinitely (never re-locked), so the warrior keeps
        # swinging at the monster it is already hitting instead of flipping
        # to a fresh monster every other frame (which made it take damage
        # from the monsters it walked away from).
        target = None
        locked = self.locked_target
        if stationary:
            # 站桩模式: 取消锁定远处目标, 只选横向在攻击范围内的怪
            # (纵向不限制: 头顶的怪靠跳击打, 由 decide 决定直接攻击还是跳击)
            in_range = [
                c for c in front_candidates
                if c["horizontal_distance"] <= self.attack_horizontal_px
            ]
            if not in_range:
                self.locked_target = None
                self.latest = self._empty_result("NO TARGET")
                return self.latest
            target = min(in_range, key=lambda item: item["distance"])
            self.locked_target = {
                "label": target["monster"]["label"],
                "box": target["monster"]["box"],
                "center": target["center"],
                "entity_id": target["monster"].get("entity_id"),
                "timestamp": timestamp,
            }
        else:
            # 【无攻击锁定】: 每帧从当前检测到的怪里选最近的(优先已开 HP 条的),
            # 不保持旧目标坐标——YOLO 漏检/误检时不会对着旧坐标/空气一直攻击。
            hp_candidates = [
                c for c in front_candidates if c["monster"].get("has_hp_bar")
            ]
            pool = hp_candidates if hp_candidates else front_candidates
            target = min(pool, key=lambda item: item["distance"])
            self.locked_target = None

        attack_ready = (
            target["horizontal_distance"] <= self.attack_horizontal_px
            and target["vertical_distance"] <= self.attack_vertical_px
        )
        close_enough_to_dodge = (
            target["horizontal_distance"] <= self.dodge_horizontal_px
            and target["vertical_distance"] <= self.dodge_vertical_px
        )
        approaching = (
            target["approach_speed"] is not None
            and target["approach_speed"] >= self.approach_speed_px_s
        )
        dodge_risk = close_enough_to_dodge and (
            approaching
            or target["horizontal_distance"] <= self.immediate_danger_px
        )
        suggested_direction = None
        if dodge_risk:
            if target["center"][0] < player_center[0]:
                suggested_direction = "RIGHT"
            elif target["center"][0] > player_center[0]:
                suggested_direction = "LEFT"
            else:
                suggested_direction = "MOVE AWAY"

        status = "DODGE RISK" if dodge_risk else (
            "ATTACK READY" if attack_ready else "TRACKING"
        )
        self.latest = {
            "status": status,
            "attack_ready": attack_ready,
            "dodge_risk": dodge_risk,
            "suggested_direction": suggested_direction,
            "target_label": target["monster"]["label"],
            "target_box": target["monster"]["box"],
            "distance_px": target["distance"],
            "horizontal_distance_px": target["horizontal_distance"],
            "vertical_distance_px": target["vertical_distance"],
            "approach_speed_px_s": target["approach_speed"],
        }
        return self.latest


def read_vitals(monitor, frame, cfg):
    if cfg["health_monitor"].get("input_full_frame", False):
        health_frame = frame
    else:
        health_frame = frame[cfg["ui_coords"]["ui_y_start"] :]
    monitor.update_frame(health_frame)
    return monitor.get_hp_mp_exp_percent()


def draw_box(image, detection, default_color):
    x, y, width, height = detection["box"]
    color = detection.get("color", default_color)
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    if detection["label"] == "MOTION ?":
        label = detection["label"]
    else:
        moving = " MOVING" if detection.get("moving", False) else ""
        label = f"{detection['label']} {detection['score']:.2f}{moving}"
    cv2.putText(
        image,
        label,
        (x, max(18, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_meter(panel, y, label, value, color):
    safe_value = 0.0 if value is None else max(0.0, min(100.0, float(value)))
    text = f"{label:<4} {'--' if value is None else f'{safe_value:5.1f}%'}"
    cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (18, y + 10), (242, y + 27), (90, 90, 90), 1)
    cv2.rectangle(panel, (20, y + 12), (20 + int(220 * safe_value / 100.0), y + 25), color, -1)


def render(
    frame,
    player,
    monsters,
    motion_candidates,
    vitals,
    fps,
    camera_motion,
    advisory=None,
):
    height, width = frame.shape[:2]
    output = frame.copy()
    for candidate in motion_candidates:
        draw_box(output, candidate, MOTION_COLOR)
    if player:
        draw_box(output, player, PLAYER_COLOR)
        tx, ty, tw, th = player["nametag_box"]
        cv2.rectangle(output, (tx, ty), (tx + tw, ty + th), (0, 220, 0), 1)
    for monster in monsters:
        draw_box(output, monster, STUMP_COLOR)
    if advisory and advisory.get("target_box"):
        target_x, target_y, target_width, target_height = advisory["target_box"]
        target_center = (
            target_x + target_width // 2,
            target_y + target_height // 2,
        )
        if player:
            player_center = tuple(map(int, AdvisoryEvaluator._center(player)))
            line_color = (
                ADVISORY_RISK_COLOR
                if advisory["dodge_risk"]
                else ADVISORY_READY_COLOR
            )
            cv2.line(output, player_center, target_center, line_color, 2)

    panel = np.full((height, 280, 3), 28, dtype=np.uint8)
    cv2.putText(panel, "OBSERVE ONLY", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"FPS       {fps:4.1f}", (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Player    {'FOUND' if player else 'MISSED'}", (18, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.58, PLAYER_COLOR if player else (80, 80, 255), 1, cv2.LINE_AA)
    player_score = "--" if not player else f"{player['score']:.3f}"
    cv2.putText(panel, f"Confidence {player_score}", (18, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Monsters  {len(monsters)}", (18, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (205, 205, 205), 1, cv2.LINE_AA)
    moving_count = sum(1 for monster in monsters if monster.get("moving", False))
    motion_text = "CAMERA" if camera_motion else f"{moving_count}+{len(motion_candidates)}?"
    cv2.putText(panel, f"Motion    {motion_text}", (18, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MOTION_COLOR, 1, cv2.LINE_AA)

    hp, mp, exp = vitals
    draw_meter(panel, 215, "HP", hp, (40, 40, 230))
    draw_meter(panel, 275, "MP", mp, (230, 130, 30))
    draw_meter(panel, 335, "EXP", exp, (40, 210, 230))

    method_counts = {}
    for monster in monsters:
        key = monster["method"]
        method_counts[key] = method_counts.get(key, 0) + 1
    if motion_candidates:
        method_counts["motion ?"] = len(motion_candidates)
    y = 405
    cv2.putText(panel, "DETECTIONS", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    for method, count in sorted(method_counts.items()):
        y += 27
        cv2.putText(panel, f"{method}: {count}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)

    if advisory:
        y = max(y + 42, 515)
        status = advisory["status"]
        status_color = (
            ADVISORY_RISK_COLOR
            if advisory["dodge_risk"]
            else ADVISORY_READY_COLOR
            if advisory["attack_ready"]
            else ADVISORY_IDLE_COLOR
        )
        cv2.putText(panel, "ADVISORY", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        y += 29
        cv2.putText(panel, status, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
        if advisory["target_label"]:
            y += 27
            cv2.putText(panel, f"Target {advisory['target_label']}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 205, 205), 1, cv2.LINE_AA)
            y += 25
            distance = advisory["distance_px"]
            cv2.putText(panel, f"Distance {distance:.0f}px", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 205, 205), 1, cv2.LINE_AA)
            speed = advisory["approach_speed_px_s"]
            speed_text = "--" if speed is None else f"{speed:+.0f}px/s"
            y += 25
            cv2.putText(panel, f"Approach {speed_text}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (205, 205, 205), 1, cv2.LINE_AA)
        if advisory["suggested_direction"]:
            y += 27
            cv2.putText(panel, f"Suggest {advisory['suggested_direction']}", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ADVISORY_RISK_COLOR, 2, cv2.LINE_AA)

    return np.hstack((output, panel))


def main():
    parser = argparse.ArgumentParser(description="Live perception viewer with no input control.")
    parser.add_argument("--cfg", default="shanda_legacy")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--snapshot")
    parser.add_argument("--summary")
    parser.add_argument(
        "--monster-backend", choices=("heuristic", "yolo"), default="heuristic"
    )
    parser.add_argument(
        "--yolo-model",
        default="training_runs/maple_three_class_v1_balanced/weights/best.pt",
    )
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-image-size", type=int, default=960)
    parser.add_argument("--motion-detection", action="store_true")
    parser.add_argument("--motion-threshold", type=int, default=22)
    parser.add_argument("--motion-min-area", type=int, default=45)
    parser.add_argument("--motion-max-area", type=int, default=3500)
    parser.add_argument("--motion-candidate-score", type=float, default=0.70)
    parser.add_argument(
        "--combat-advisory",
        action="store_true",
        help="Show read-only attack/dodge geometry without sending input.",
    )
    args = parser.parse_args()

    cfg = override_cfg(
        load_yaml("config/config_default.yaml"),
        load_yaml(f"config/config_{args.cfg}.yaml"),
    )
    overlay_cfg = cfg["perception_overlay"]
    player_detector = PlayerDetector(cfg)
    if args.monster_backend == "yolo":
        monster_detector = YoloMonsterDetector(
            cfg,
            args.yolo_model,
            args.yolo_confidence,
            args.yolo_device,
            args.yolo_image_size,
        )
    else:
        monster_detector = MonsterDetector(cfg)
    monitor = HealthMonitor(cfg, kb_controller=None)
    motion_detector = None
    if args.motion_detection:
        motion_detector = MotionDetector(
            cfg,
            threshold=args.motion_threshold,
            min_area=args.motion_min_area,
            max_area=args.motion_max_area,
            candidate_score=args.motion_candidate_score,
        )
    advisory_evaluator = AdvisoryEvaluator(cfg) if args.combat_advisory else None
    capture = GameWindowCapturor(cfg)
    cached_monsters = []
    motion_candidates = []
    started = time.time()
    last_frame_time = started
    fps = 0.0
    frame_count = 0
    latest = None
    advisory = None

    try:
        if not args.headless:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

        while True:
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            frame_count += 1
            player = player_detector.detect(frame)
            detections_refreshed = (
                frame_count == 1
                or frame_count % overlay_cfg["monster_refresh_frames"] == 0
            )
            if detections_refreshed:
                cached_monsters = monster_detector.detect(frame, player)
            if motion_detector:
                motion_candidates = motion_detector.detect(
                    frame, player, cached_monsters
                )
            vitals = read_vitals(monitor, frame, cfg)

            now = time.time()
            camera_motion = bool(motion_detector and motion_detector.camera_motion)
            if advisory_evaluator and (detections_refreshed or camera_motion):
                advisory = advisory_evaluator.evaluate(
                    player,
                    cached_monsters,
                    now,
                    camera_motion=camera_motion,
                )
            instantaneous_fps = 1.0 / max(now - last_frame_time, 1e-6)
            fps = instantaneous_fps if fps == 0 else fps * 0.85 + instantaneous_fps * 0.15
            last_frame_time = now
            latest = render(
                frame,
                player,
                cached_monsters,
                motion_candidates,
                vitals,
                fps,
                camera_motion,
                advisory,
            )

            if not args.headless:
                cv2.imshow(WINDOW_TITLE, latest)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break
            if args.duration > 0 and now - started >= args.duration:
                break

            target_duration = 1.0 / overlay_cfg["fps_limit"]
            elapsed = time.time() - now
            if elapsed < target_duration:
                time.sleep(target_duration - elapsed)
    finally:
        capture.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    if latest is not None and args.snapshot:
        save_image(args.snapshot, latest)

    summary = {
        "observe_only": True,
        "monster_backend": args.monster_backend,
        "motion_detection": args.motion_detection,
        "combat_advisory": args.combat_advisory,
        "camera_motion_suppressed": bool(
            motion_detector and motion_detector.camera_motion
        ),
        "frames": frame_count,
        "fps": round(fps, 2),
        "player": None if player is None else {
            "box": list(map(int, player["box"])),
            "confidence": round(player["score"], 4),
        },
        "monsters": [
            {
                "label": item["label"],
                "box": list(map(int, item["box"])),
                "confidence": round(item["score"], 4),
                "method": item["method"],
                "moving": bool(item.get("moving", False)),
            }
            for item in cached_monsters
        ],
        "motion_candidates": [
            {
                "box": list(map(int, item["box"])),
                "confidence": round(item["score"], 4),
            }
            for item in motion_candidates
        ],
        "vitals": {
            "hp_percent": None if vitals[0] is None else round(float(vitals[0]), 2),
            "mp_percent": None if vitals[1] is None else round(float(vitals[1]), 2),
            "exp_percent": None if vitals[2] is None else round(float(vitals[2]), 2),
        },
        "advisory": None if advisory is None else {
            "status": advisory["status"],
            "attack_ready": advisory["attack_ready"],
            "dodge_risk": advisory["dodge_risk"],
            "suggested_direction": advisory["suggested_direction"],
            "target_label": advisory["target_label"],
            "distance_px": None if advisory["distance_px"] is None else round(advisory["distance_px"], 2),
            "horizontal_distance_px": None if advisory["horizontal_distance_px"] is None else round(advisory["horizontal_distance_px"], 2),
            "vertical_distance_px": None if advisory["vertical_distance_px"] is None else round(advisory["vertical_distance_px"], 2),
            "approach_speed_px_s": None if advisory["approach_speed_px_s"] is None else round(advisory["approach_speed_px_s"], 2),
        },
    }
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
