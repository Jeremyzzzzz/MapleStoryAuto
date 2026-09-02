"""Sprite template monster detector + runtime template collector.

Detector: two-stage template matching.
1. Coarse: every mean-filled template matched on the 0.5x frame. Peaks
   record WHICH template and scale produced them.
2. Fine: only those specific templates re-verified at native resolution
   (masked CCOEFF_NORMED) at nearby scales + class-color check.

Why templates work here: MapleStory monsters are fixed sprites. Sprites
cropped from real game frames match each other at 0.94-1.00 median CCOEFF,
while cross-class scores stay at 0.11-0.65. White-background official art
does NOT work (0.45-0.58) because it is high-res promotional art, not the
in-game sprite.

TemplateCollector: during combat, when a locked target is confidently
detected, crop its patch from the frame and save it into the template
library (with dedup). The library improves as the bot plays, and new-map
monsters (flower mushroom, green slime, green mushroom) get learned
automatically.
"""

import time
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp")
TPL_DIR = ROOT / "monster_templates_final"
SRC = ROOT / "training_data" / "maple_six_class_v5"
CLASS_NAMES = ["red_snail", "blue_snail", "stump", "green_mushroom", "orange_mushroom", "slime"]

# Module-level switch for load_library: skip live_*.png (pollution-safe by default)
INCLUDE_LIVE = False

CLASS_HSV = {
    "red_snail":      [((0, 60, 60), (22, 255, 255)), ((165, 60, 60), (180, 255, 255))],
    "blue_snail":     [((90, 50, 50), (140, 255, 255))],
    "stump":          [((5, 40, 40), (32, 255, 170))],
    "green_mushroom": [((30, 60, 60), (78, 255, 230))],
    "orange_mushroom":[((6, 80, 110), (26, 255, 255))],
    "slime":          [((55, 45, 45), (112, 255, 255))],
}
MIN_COLOR_FRAC = 0.40
COARSE_SCALES = (0.40, 0.55)
# fine scales tried per coarse-scale family: cs=0.40 -> target~0.8, cs=0.55 -> target~1.1
FINE_BY_COARSE = {
    0.40: (0.70, 0.80, 0.90),
    0.55: (0.95, 1.10, 1.25),
}


def imread_cn(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def load_library(max_per_class=10):
    lib = defaultdict(list)
    for cls_dir in TPL_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        items = []
        for p in sorted(cls_dir.glob("*.png")):
            # Skip live_*_collect templates by default: they were captured
            # from a possibly-bwrong target lock during in-combat auto
            # collection, and their larger box area used to push real
            # in-game sprite templates out of the top-N. Reload via
            # include_live=True only when debugging.
            if p.name.startswith("live_") and not INCLUDE_LIVE:
                continue
            img = imread_cn(p)
            if img is None:
                continue
            bg = np.all(img == [0, 255, 0], axis=2)
            mask = (~bg).astype(np.uint8) * 255
            if int((mask > 0).sum()) < 40:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            m = mask > 0
            mean_val = gray[m].mean()
            filled = gray.copy()
            filled[~m] = mean_val
            items.append({"gray": gray, "fill": filled, "mask": mask,
                          "h": img.shape[0], "w": img.shape[1]})
        items.sort(key=lambda t: -(t["h"] * t["w"]))
        lib[cls] = items[:max_per_class]
    return lib


def nan_safe_peaks(res, threshold, top_per_template=1, min_gap=16, max_total=30):
    if not np.isfinite(res).any():
        return []
    res = np.nan_to_num(res, nan=-2.0, posinf=-2.0, neginf=-2.0)
    pts = []
    while True:
        _, mv, _, ml = cv2.minMaxLoc(res)
        if not np.isfinite(mv) or mv < threshold:
            break
        pts.append((ml[0], ml[1], float(mv)))
        x0 = max(0, ml[0] - min_gap)
        y0 = max(0, ml[1] - min_gap)
        x1 = min(res.shape[1], ml[0] + min_gap)
        y1 = min(res.shape[0], ml[1] + min_gap)
        res[y0:y1, x0:x1] = -2.0
        if len(pts) >= max_total:
            break
    return pts


def box_color_frac(hsv, box, cls):
    """Fraction of box pixels inside class color (0..1)."""
    ranges = CLASS_HSV.get(cls)
    if not ranges:
        return 1.0
    x, y, w_, h_ = (int(v) for v in box)
    if w_ < 6 or h_ < 6:
        return 0.0
    reg = hsv[max(0, y):y + h_, max(0, x):x + w_]
    if reg.size == 0:
        return 0.0
    mask = np.zeros(reg.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(reg, np.array(lo), np.array(hi))
    return float((mask > 0).sum()) / float(reg.shape[0] * reg.shape[1])


class FastSpriteDetectorV8:
    def __init__(self, threshold_fine=0.68, threshold_coarse=0.66, max_per_class=10):
        self.lib = load_library(max_per_class)
        self.tf = threshold_fine
        self.tc = threshold_coarse
        n = sum(len(v) for v in self.lib.values())
        print(f"[FastSpriteDetectorV8] {n} templates "
              f"{ {k: len(v) for k, v in self.lib.items()} }")

    def detect(self, frame, search_box=None):
        h, w = frame.shape[:2]
        if search_box is None:
            x0, y0, x1, y1 = 0, 60, w, 540
        else:
            sx, sy, sw, sh = search_box
            x0, y0 = max(0, sx), max(0, sy)
            x1, y1 = min(w, sx + sw), min(h, sy + sh)
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = gray_full[y0:y1, x0:x1]
        g05 = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # ---- coarse: mean-filled, record (cls, tpl_idx, cs) ----
        coarse_hits = []  # (cls, tpl_idx, cs, cx, cy, score)
        for cls, tpls in self.lib.items():
            for ti, t in enumerate(tpls):
                for cs in COARSE_SCALES:
                    mg = cv2.resize(t["fill"], None, fx=cs, fy=cs, interpolation=cv2.INTER_AREA)
                    if mg.shape[0] < 8 or mg.shape[1] < 8:
                        continue
                    if g05.shape[0] < mg.shape[0] or g05.shape[1] < mg.shape[1]:
                        continue
                    try:
                        res = cv2.matchTemplate(g05, mg, cv2.TM_CCOEFF_NORMED)
                    except cv2.error:
                        continue
                    for (px, py, pscore) in nan_safe_peaks(res, self.tc, top_per_template=1):
                        cx = x0 + int((px + mg.shape[1] / 2) * 2)
                        cy = y0 + int((py + mg.shape[0] / 2) * 2)
                        coarse_hits.append((cls, ti, cs, cx, cy, pscore))

        # ---- cluster by proximity (same class) ----
        clusters = []  # list of {cls, members: [(cx,cy,tpl_idx,cs)]}
        used = set()
        for i, (cls, ti, cs, cx, cy, ps) in enumerate(coarse_hits):
            if i in used:
                continue
            used.add(i)
            members = [(cx, cy, ti, cs)]
            for j, (cls2, ti2, cs2, cx2, cy2, ps2) in enumerate(coarse_hits):
                if j in used or cls2 != cls:
                    continue
                if abs(cx2 - cx) < 70 and abs(cy2 - cy) < 70:
                    used.add(j)
                    members.append((cx2, cy2, ti2, cs2))
            clusters.append((cls, members))

        # ---- fine: verify only the templates/scales that fired ----
        dets = []
        for cls, members in clusters:
            mx = int(np.mean([m[0] for m in members]))
            my = int(np.mean([m[1] for m in members]))
            rw, rh = 130, 130
            fx0 = max(x0, mx - rw)
            fy0 = max(y0, my - rh)
            fx1 = min(x1, mx + rw)
            fy1 = min(y1, my + rh)
            reg_gray = gray_full[fy0:fy1, fx0:fx1]
            if reg_gray.shape[0] < 10 or reg_gray.shape[1] < 10:
                continue
            # dedupe (tpl_idx, scale) pairs
            pairs = set()
            for (_, _, ti, cs) in members:
                for sc in FINE_BY_COARSE.get(cs, (1.0,)):
                    pairs.add((ti, sc))
            for (ti, sc) in pairs:
                t = self.lib[cls][ti]
                tw = max(8, int(t["w"] * sc))
                th = max(8, int(t["h"] * sc))
                tg = cv2.resize(t["gray"], (tw, th), interpolation=cv2.INTER_AREA)
                tm = cv2.resize(t["mask"], (tw, th), interpolation=cv2.INTER_AREA)
                _, tm = cv2.threshold(tm, 128, 255, cv2.THRESH_BINARY)
                if int((tm > 0).sum()) < 20:
                    continue
                if reg_gray.shape[0] < th or reg_gray.shape[1] < tw:
                    continue
                try:
                    r2 = cv2.matchTemplate(reg_gray, tg, cv2.TM_CCOEFF_NORMED, mask=tm)
                except cv2.error:
                    continue
                if not np.isfinite(r2).any():
                    continue
                _, mv, _, ml = cv2.minMaxLoc(r2)
                box = (fx0 + ml[0], fy0 + ml[1], tw, th)
                if not (np.isfinite(mv) and mv >= self.tf):
                    continue
                frac = box_color_frac(hsv_full, box, cls)
                if frac < MIN_COLOR_FRAC:
                    continue
                dets.append({
                    "label": cls,
                    "box": box,
                    "score": float(mv),
                    "color_frac": frac,
                    "method": "sprite",
                })
        dets.sort(key=lambda d: -d["score"])
        kept = []
        for d in dets:
            if all(self._iou(d["box"], k["box"]) < 0.30 for k in kept):
                kept.append(d)
        return kept

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        uni = aw * ah + bw * bh - inter
        return inter / uni if uni > 0 else 0.0


# ---------------------------------------------------------------------------
# Combat-pipeline wrapper: detect(frame, player) interface + ROI from player
# ---------------------------------------------------------------------------
class SpriteMonsterDetector:
    """Wraps FastSpriteDetectorV8 with the auto_combat detector interface.

    detect(frame, player): searches a ROI around the player (or the whole
    play field when player is None).

    Same-level filtering (precision boost): instead of a tall box around the
    player, we only search a horizontal band centered on the player's vertical
    position (y within cy +/- level_band_half_h), and afterwards drop any
    detection whose FEET (box bottom) are more than level_tol_px away from the
    player's feet. MapleStory monsters standing on the same platform have
    nearly identical ground y; monsters on platforms above/below differ by
    much more, so this removes cross-level false positives cheaply.
    """

    def __init__(self, threshold_fine=0.68, threshold_coarse=0.66,
                 roi_half_w=420, level_band_half_h=110, level_tol_px=45,
                 max_per_class=10):
        self.det = FastSpriteDetectorV8(
            threshold_fine=threshold_fine,
            threshold_coarse=threshold_coarse,
            max_per_class=max_per_class)
        self.roi_half_w = roi_half_w
        self.level_band_half_h = level_band_half_h
        self.level_tol_px = level_tol_px

    def detect(self, frame, player=None):
        search_box = None
        feet_ref = None
        if player is not None and player.get("box"):
            bx, by, bw, bh = (int(v) for v in player["box"])
            cx, cy = bx + bw // 2, by + bh // 2
            # horizontal band only: skip monsters far above/below the player
            search_box = (cx - self.roi_half_w, cy - self.level_band_half_h,
                          2 * self.roi_half_w, 2 * self.level_band_half_h)
            feet_ref = by + bh  # player's feet (ground line)
        dets = self.det.detect(frame, search_box=search_box)
        if feet_ref is not None:
            kept = []
            for d in dets:
                dx, dy, dw, dh = d["box"]
                feet = dy + dh
                if abs(feet - feet_ref) <= self.level_tol_px:
                    kept.append(d)
            dets = kept
        return dets


# ---------------------------------------------------------------------------
# Runtime template collector: learn new monsters while fighting
# ---------------------------------------------------------------------------
class TemplateCollector:
    """Collects new sprite templates from live combat frames.

    When a monster is detected with high confidence (score >= min_score)
    and is the current combat target, crop its box from the frame, green-
    screen the background (dominant-color mask) and save into the library.
    Dedup: skip if it matches an existing template at >= 0.93.
    """

    def __init__(self, out_dir=None, min_score=0.80, max_per_class=40):
        self.out_dir = Path(out_dir) if out_dir else TPL_DIR
        self.min_score = min_score
        self.max_per_class = max_per_class
        self.saved = defaultdict(int)
        self._loaded = None  # lazy list of (cls, gray, mask) for dedup

    def _existing(self, cls):
        if self._loaded is None:
            self._loaded = defaultdict(list)
            sub = self.out_dir / cls
            if sub.exists():
                for p in sorted(sub.glob("*.png")):
                    img = imread_cn(p)
                    if img is None:
                        continue
                    bg = np.all(img == [0, 255, 0], axis=2)
                    mask = (~bg).astype(np.uint8) * 255
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self._loaded[cls].append({"gray": gray, "mask": mask})
        return self._loaded.get(cls, [])

    def _build_green_screen(self, patch):
        """Crop to dominant-color blob and composite onto green screen."""
        if patch is None or patch.size == 0 or min(patch.shape[:2]) < 10:
            return None, None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        center = hsv[h // 4:3 * h // 4, w // 4:3 * w // 4].reshape(-1, 3)
        if len(center) < 10:
            return None, None
        med = np.median(center, axis=0).astype(np.uint16)
        hfull = hsv.reshape(-1, 3).astype(np.int16)
        dfull = np.abs(hfull - med.astype(np.int16))
        hued = np.minimum(dfull[:, 0], 180 - dfull[:, 0])
        mask = ((hued < 35) & (dfull[:, 1] < 130) & (dfull[:, 2] < 130))
        mask = mask.reshape(h, w).astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return None, None
        best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        x, y, bw, bh = stats[best, :4]
        if bw < 10 or bh < 10 or bw * bh < 100:
            return None, None
        tight = patch[y:y + bh, x:x + bw].copy()
        tight_mask = mask[y:y + bh, x:x + bw].copy()
        comp = np.zeros((bh, bw, 3), np.uint8)
        comp[:] = (0, 255, 0)
        comp[tight_mask > 0] = tight[tight_mask > 0]
        return comp, tight_mask

    def collect(self, frame, detection):
        """frame: BGR ndarray. detection: dict with label/box/score."""
        cls = detection.get("label")
        if cls:
            cls = cls.strip().lower().replace(" ", "_")
        score = detection.get("score", 0.0)
        if not cls or score < self.min_score:
            return None
        if self.saved.get(cls, 0) >= self.max_per_class:
            return None
        x, y, w_, h_ = (int(v) for v in detection["box"])
        if w_ < 12 or h_ < 12:
            return None
        pad = 8
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame.shape[1], x + w_ + pad)
        y1 = min(frame.shape[0], y + h_ + pad)
        patch = frame[y0:y1, x0:x1].copy()
        comp, tight_mask = self._build_green_screen(patch)
        if comp is None:
            return None
        gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
        mask = (tight_mask > 0).astype(np.uint8) * 255
        # dedup against existing templates
        for ex in self._existing(cls):
            if ex["gray"].shape[0] > gray.shape[0] or ex["gray"].shape[1] > gray.shape[1]:
                continue
            res = cv2.matchTemplate(gray, ex["gray"], cv2.TM_CCOEFF_NORMED, mask=ex["mask"])
            _, mv, _, _ = cv2.minMaxLoc(res)
            if np.isfinite(mv) and mv >= 0.93:
                return None
        sub = self.out_dir / cls
        sub.mkdir(exist_ok=True)
        idx = self.saved[cls]
        out_p = sub / f"live_{idx:03d}.png"
        ok, buf = cv2.imencode(".png", comp)
        out_p.write_bytes(buf.tobytes())
        self.saved[cls] += 1
        self._loaded[cls].append({"gray": gray, "mask": mask})
        return str(out_p)


def eval_val():
    det = FastSpriteDetectorV8()
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    lats = []
    for split in ("val",):
        img_dir = SRC / "images" / split
        lab_dir = SRC / "labels" / split
        for ip in sorted(img_dir.glob("old_*")):
            img = imread_cn(ip)
            lab = lab_dir / (ip.stem + ".txt")
            if img is None or not lab.exists():
                continue
            h, w = img.shape[:2]
            gt = []
            for line in lab.read_text(encoding="ascii").strip().split("\n"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                if cid > 2:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                gt.append({"cls": CLASS_NAMES[cid],
                           "box": (int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                                   int(bw * w), int(bh * h))})
            t0 = time.perf_counter()
            dets = det.detect(img)
            lats.append(time.perf_counter() - t0)
            matched = [False] * len(gt)
            for d in dets:
                best_j, best_dist = -1, 1e9
                dcx = d["box"][0] + d["box"][2] / 2
                dcy = d["box"][1] + d["box"][3] / 2
                for j, g in enumerate(gt):
                    if matched[j] or g["cls"] != d["label"]:
                        continue
                    gcx = g["box"][0] + g["box"][2] / 2
                    gcy = g["box"][1] + g["box"][3] / 2
                    dist = ((dcx - gcx) ** 2 + (dcy - gcy) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist, best_j = dist, j
                if best_j >= 0 and best_dist < 35:
                    matched[best_j] = True
                    stats[d["label"]]["tp"] += 1
                else:
                    stats[d["label"]]["fp"] += 1
            for j, g in enumerate(gt):
                if not matched[j]:
                    stats[g["cls"]]["fn"] += 1
    print(f"\nLatency: mean={np.mean(lats)*1000:.0f}ms  max={np.max(lats)*1000:.0f}ms")
    print(f"{'class':16s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'P':>6s} {'R':>6s}")
    for cls in ("red_snail", "blue_snail", "stump"):
        s = stats[cls]
        p = s["tp"] / max(1, s["tp"] + s["fp"])
        r = s["tp"] / max(1, s["tp"] + s["fn"])
        print(f"{cls:16s} {s['tp']:4d} {s['fp']:4d} {s['fn']:4d} {p:6.3f} {r:6.3f}")


if __name__ == "__main__":
    eval_val()