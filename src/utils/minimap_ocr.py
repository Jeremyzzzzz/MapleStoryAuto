"""
Minimap map-name OCR. Kept dependency-light so it can run in the GPU venv
(which has rapidocr but not pyautogui/pygetwindow): only numpy + cv2 + rapidocr.
"""
import cv2

# OCR engine is expensive to construct (model load ~5s). Cache it at module
# level so repeated calls reuse the instance instead of reloading each time.
_OCR_ENGINE = None


def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE

_MINIMAP_UI_TEXT = frozenset({
    "小地图", "大地图", "冒险岛", "冒险岛怀旧服", "怀旧服", "地图",
})

# Names that are safe to treat as the map name once seen; used to break ties
# when OCR output flickers between similar readings (e.g. 勇士/男士).
_KNOWN_MAP_HINTS = (
    "勇士部落", "射手村", "魔法密林", "废弃都市", "明珠港", "猪海岸",
    "蘑菇山", "彩虹岛", "金银岛", "地下城", "蘑菇", "训练场",
)


def _pick_candidate(results):
    """Pick the most likely map name from raw OCR results."""
    candidates = []
    for box, text, score in results:
        text = (text or "").strip()
        if not text or score < 0.6:
            continue
        if any(hint in text for hint in _MINIMAP_UI_TEXT):
            continue
        candidates.append((len(text), score, text))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return candidates[0][2]


def get_minimap_map_name(img_frame, region=None, ocr_engine=None):
    """
    Read the map name shown in the top-left minimap via OCR.

    The minimap header usually contains the current map name (e.g.
    "勇士部落东入口"). The minimap crop is upscaled 2x before OCR (small text
    reads much better), and OCR is run twice so a single misread flicker
    (勇 -> 男) does not poison the result: if the two readings disagree, the
    one matching a known map name wins; otherwise the longer reading wins.

    Returns the map name string, or None if nothing usable is detected.
    """
    if ocr_engine is None:
        ocr_engine = _get_ocr_engine()
    if region is None:
        region = [0, 0, 285, 245]
    x, y, w, h = (int(v) for v in region)
    roi = img_frame[y:y + h, x:x + w]
    if roi.size == 0:
        return None

    # Upscale 2x: the minimap header text is small; OCR is much more reliable
    # on the enlarged crop. INTER_CUBIC keeps glyph edges smooth.
    roi = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2),
                     interpolation=cv2.INTER_CUBIC)

    readings = []
    try:
        for _ in range(2):
            res, _ = ocr_engine(roi)
            readings.append(_pick_candidate(res) if res else None)
    except Exception:
        return None

    # Filter out None readings.
    readings = [r for r in readings if r]
    if not readings:
        return None
    if len(readings) == 1:
        return readings[0]
    if readings[0] == readings[1]:
        return readings[0]

    # Disagree: prefer the reading that matches a known map name, else the
    # longer one (map names are long, UI hints are short).
    for reading in readings:
        if any(hint in reading for hint in _KNOWN_MAP_HINTS):
            return reading
    return max(readings, key=len)
