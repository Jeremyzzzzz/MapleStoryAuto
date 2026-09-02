"""
Chinese text rendering for OpenCV frames.

cv2.putText cannot draw CJK characters; this helper renders text with PIL
using a system Chinese font (Microsoft YaHei / SimHei) and pastes it onto a
BGR frame. Falls back to ASCII-only drawing when PIL or the font is missing.
"""
import os
from pathlib import Path

import cv2
import numpy as np

_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msyh.ttc"),   # Microsoft YaHei
    Path("C:/Windows/Fonts/msyhbd.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"), # SimHei
    Path("C:/Windows/Fonts/simsun.ttc"), # SimSun
    Path("/System/Library/Fonts/PingFang.ttc"),  # macOS
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
]

_pil_font = None
_pil_font_size = 0


def _get_font(size):
    global _pil_font, _pil_font_size
    if _pil_font is not None and _pil_font_size == size:
        return _pil_font
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        if path.exists():
            try:
                _pil_font = ImageFont.truetype(str(path), size)
                _pil_font_size = size
                return _pil_font
            except Exception:
                continue
    _pil_font = ImageFont.load_default()
    _pil_font_size = size
    return _pil_font


def put_text_cn(frame, text, org, font_scale=0.5, color=(255, 255, 255),
                thickness=1, line_type=cv2.LINE_AA):
    """Draw text (possibly Chinese) onto a BGR frame at org (bottom-left).

    Mirrors cv2.putText semantics for the common arguments. Returns the frame.
    """
    if not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        # Pure ASCII: use the fast OpenCV path.
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, line_type)
        return frame
    try:
        from PIL import Image, ImageDraw
    except Exception:
        # No PIL: fall back to ASCII rendering (CJK becomes question marks).
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, line_type)
        return frame

    h, w = frame.shape[:2]
    # Rough glyph size: font_scale * 32 is comparable to HERSHEY scale.
    px = max(12, int(font_scale * 32))
    font = _get_font(px)
    # Measure text.
    from PIL import ImageDraw as _ID
    tmp = Image.new("RGB", (8, 8))
    draw = _ID.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        return frame

    # Canvas for the text.
    canvas = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((2 - bbox[0], 2 - bbox[1]), text, font=font,
              fill=(int(color[2]), int(color[1]), int(color[0]), 255))
    alpha = np.array(canvas)[:, :, 3:4].astype(np.float32) / 255.0
    rgb = np.array(canvas)[:, :, :3].astype(np.float32)

    # Paste location: org is bottom-left like putText.
    x = int(org[0])
    y = int(org[1]) - th
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + tw)
    y1 = min(h, y + th)
    if x1 <= x0 or y1 <= y0:
        return frame
    ch = x1 - x0
    cw = y1 - y0
    region = frame[y0:y1, x0:x1].astype(np.float32)
    a = alpha[(y0 - y):(y0 - y + cw), (x0 - x):(x0 - x + ch)]
    fg = rgb[(y0 - y):(y0 - y + cw), (x0 - x):(x0 - x + ch)]
    frame[y0:y1, x0:x1] = (region * (1 - a) + fg * a).astype(np.uint8)
    return frame
