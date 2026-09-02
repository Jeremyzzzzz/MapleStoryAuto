# -*- coding: utf-8 -*-
"""调试:抓当前游戏窗口一帧,测试玩家检测(模板+OCR),保存截图供人工分析。"""
import sys, time, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = REPO_ROOT / ".yolo_runtime"
for _p in (REPO_ROOT, LOCAL_RUNTIME, LOCAL_RUNTIME / "win32", LOCAL_RUNTIME / "win32" / "lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.add_dll_directory(str(LOCAL_RUNTIME / "pywin32_system32"))
import cv2
import numpy as np
import pyautogui
import ctypes
from ctypes import wintypes

from tools.auto_combat import (
    load_config, PlayerDetector, NameOcrPlayerDetector, PlayerLocator,
)

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.FindWindowW.restype = wintypes.HWND
_user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
_user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
_user32.IsIconic.argtypes = [wintypes.HWND]
_user32.IsIconic.restype = wintypes.BOOL


def find_windows(title_part):
    out = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, lparam):
        length = _user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        if title_part in buf.value:
            rect = wintypes.RECT()
            _user32.GetWindowRect(hwnd, ctypes.byref(rect))
            out.append((hwnd, buf.value, (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)))
        return True

    _user32.EnumWindows(_cb, 0)
    return out


WINS = find_windows("冒险岛怀旧服")
print("找到窗口:", [(t, rect) for _, t, rect in WINS])
if not WINS:
    sys.exit("未找到游戏窗口")
hwnd, _, (left, top, width, height) = WINS[0]
if _user32.IsIconic(hwnd):
    print("窗口已最小化,无法截图")
    sys.exit(0)

img = pyautogui.screenshot(region=(left, top, width, height))
frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
os.makedirs("probe_output", exist_ok=True)
cv2.imwrite("probe_output/player_debug_frame.png", frame)
print("截图已保存 probe_output/player_debug_frame.png, 尺寸:", frame.shape)

cfg = load_config("shanda_legacy")
td = PlayerDetector(cfg)
print("PlayerDetector 模板匹配:", td.detect(frame))

print("--- 开始 OCR 识别玩家名(麻超圆),模型加载约需几秒 ---")
od = NameOcrPlayerDetector(cfg, "麻超圆")
t0 = time.time()
res = od.detect(frame)
print(f"OCR 检测耗时 {time.time()-t0:.1f}s, 结果: {res}")

# 直接跑一次完整 OCR 打印所有识别文本,便于看名字被识别成了什么
raw = od.ocr(frame[: cfg["ui_coords"]["ui_y_start"]])
if isinstance(raw, tuple):
    raw = raw[0]
print("OCR 全量识别文本:")
if raw:
    for entry in raw:
        if len(entry) >= 3:
            print(f"  '{entry[1]}'  score={entry[2]:.3f}  box={[tuple(map(int, p)) for p in entry[0]]}")
else:
    print("  (无)")
