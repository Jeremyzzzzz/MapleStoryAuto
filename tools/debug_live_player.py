# -*- coding: utf-8 -*-
"""诊断:用挂机同款 WindowsCapture 抓一帧,分析玩家定位失败原因。"""
import sys, time, os, cv2, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / ".yolo_runtime", ROOT / ".yolo_runtime" / "win32",
           ROOT / ".yolo_runtime" / "win32" / "lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.add_dll_directory(str(ROOT / ".yolo_runtime" / "pywin32_system32"))

from tools.auto_combat import (
    load_config, PlayerDetector, NameOcrPlayerDetector, PlayerLocator,
)
from src.input.GameWindowCapturor import GameWindowCapturor
from src.utils.common import imread_cn, imwrite_cn

cfg = load_config("shanda_legacy")
cap = GameWindowCapturor(cfg)
time.sleep(2.0)
frame = cap.get_frame()
cap.stop()
if frame is None:
    print("抓帧失败")
    sys.exit(1)
h, w = frame.shape[:2]
print(f"挂机帧尺寸: {w}x{h}")
os.makedirs("probe_output", exist_ok=True)
imwrite_cn("probe_output/live_frame.png", frame)

# --- 玩家定位诊断 ---
class Tpl:
    def __init__(self, cfg):
        self.d = PlayerDetector(cfg)
    def detect(self, f):
        r = self.d.detect(f)
        return r

od = NameOcrPlayerDetector(cfg, "麻超圆")
loc = PlayerLocator(Tpl(cfg), od, refresh_frames=1, sticky_seconds=1.5)

# 模板匹配
tpl_p = loc.template_detector.detect(frame)
print(f"模板匹配: {tpl_p}")

# OCR(完整一次,含模型加载)
t0 = time.time()
ocr_p = od.detect(frame)
print(f"OCR 识别: {'成功 center=' + str(ocr_p['center']) + ' score=%.3f' % ocr_p['score'] if ocr_p else '失败'} 耗时 {time.time()-t0:.1f}s")

# 模拟 locator 流程:播种后多帧跟踪
if ocr_p:
    loc._ocr_result = ocr_p
    loc._ocr_stamp = time.time()
    p = loc.detect(frame, time.time())
    print(f"播种帧: {p['center'] if p else None}")
    # 模拟玩家移动 40px
    M = np.float32([[1, 0, 40], [0, 1, 0]])
    frame2 = cv2.warpAffine(frame, M, (w, h))
    p2 = loc.detect(frame2, time.time() + 0.1)
    print(f"移动40px跟踪: {p2['center'] if p2 else None} method={p2.get('method') if p2 else None}")
else:
    print("OCR 播种失败 → 无法跟踪")
