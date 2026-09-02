# -*- coding: utf-8 -*-
"""采集玩家角色形象模板:连续抓帧,OCR 定位玩家,截取角色 box(70x90),
保存到 player_templates/ 目录(不同姿势各一张,去重)。

用法: python tools/capture_player_templates.py [帧数]
游戏角色保持可见并活动(走路/跳/攻击),姿势越多模板越全。
"""
import sys, time, os, glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / ".yolo_runtime", REPO_ROOT / ".yolo_runtime" / "win32",
           REPO_ROOT / ".yolo_runtime" / "win32" / "lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.add_dll_directory(str(REPO_ROOT / ".yolo_runtime" / "pywin32_system32"))

import cv2
import numpy as np

from tools.auto_combat import load_config, NameOcrPlayerDetector
from src.input.GameWindowCapturor import GameWindowCapturor
from src.utils.common import imread_cn, imwrite_cn

N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
INTERVAL = 2.0
PLAYER_NAME = "麻超圆"

# Verify the captured region actually contains the player (red pixel ratio
# from clothing/effects) -- a previous run captured a wooden bridge because
# OCR mis-read a worn badge as the name. Require at least one meaningful
# colour signal in the box so we never store a wrong-template.
def _has_player_signal(crop):
    if crop is None or crop.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Red clothing / wing effects (warrior gear is heavily red-toned)
    red_mask = (
        cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        | cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    )
    # Saturated non-grey colour (any distinctive armor/clothing)
    sat_mask = (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 70)
    red_ratio = float(red_mask.mean() / 255.0)
    sat_ratio = float(sat_mask.mean())
    # Player body has saturated colour; wooden bridge / sky / rock is grey.
    return (sat_ratio > 0.10 or red_ratio > 0.03), max(red_ratio, sat_ratio)


cfg = load_config("shanda_legacy")
od = NameOcrPlayerDetector(cfg, PLAYER_NAME)
cap = GameWindowCapturor(cfg)
time.sleep(2.0)

out_dir = REPO_ROOT / "player_templates"
out_dir.mkdir(exist_ok=True)
# 清空旧模板(重采)
for f in glob.glob(str(out_dir / "*.png")):
    os.remove(f)

saved = 0
skipped_signal = 0
skipped_dup = 0
for i in range(N_FRAMES):
    frame = cap.get_frame()
    if frame is None:
        time.sleep(0.5)
        continue
    player = od.detect(frame)
    if player is None:
        print(f"[{i+1}/{N_FRAMES}] 玩家未定位,跳过")
        time.sleep(INTERVAL)
        continue
    # Use nametag_box (the OCR-located name strip) to anchor the player
    # body BELOW the name. The body centre sits ~60px below the name bottom
    # (name + small gap + head/body); a 70x90 crop around that body centre
    # contains the character, not the ground. Previous "+30" anchored the
    # crop on the gap between name and head, capturing only background.
    box = player.get("nametag_box")
    if box and len(box) == 4:
        nx, ny, nw, nh = (int(v) for v in box)
        cx = nx + nw // 2
        cy = ny + nh + 60
    else:
        cx, cy = player["center"]
    x0, y0 = max(0, cx - 35), max(0, cy - 45)
    x1, y1 = min(frame.shape[1], cx + 35), min(frame.shape[0], cy + 45)
    if x1 - x0 < 40 or y1 - y0 < 60:
        print(f"[{i+1}/{N_FRAMES}] 区域过小,跳过")
        time.sleep(INTERVAL)
        continue
    tpl = frame[y0:y1, x0:x1]
    # Validate the crop actually contains the player (rejects wooden bridge).
    ok, sig = _has_player_signal(tpl)
    if not ok:
        skipped_signal += 1
        print(f"[{i+1}/{N_FRAMES}] 区域无玩家色彩特征(可能OCR截错到背景),跳过 sig={sig:.3f}")
        time.sleep(INTERVAL)
        continue
    # 去重:与已存模板差异 < 6 跳过
    dup = False
    for f in glob.glob(str(out_dir / "*.png")):
        existing = imread_cn(f)
        if existing is not None and existing.shape == tpl.shape:
            if cv2.absdiff(existing, tpl).mean() < 6.0:
                dup = True
                break
    if dup:
        skipped_dup += 1
        print(f"[{i+1}/{N_FRAMES}] 与已有模板重复,跳过 (center=({cx},{cy}))")
        time.sleep(INTERVAL)
        continue
    path = out_dir / f"{PLAYER_NAME}_{saved}.png"
    imwrite_cn(str(path), tpl)
    saved += 1
    print(f"[{i+1}/{N_FRAMES}] 已保存 {path.name} (center=({cx},{cy}), sig={sig:.3f})")
    time.sleep(INTERVAL)

cap.stop()
print(f"完成: 共保存 {saved} 张角色模板到 player_templates/ (跳过 {skipped_signal} 个无特征区域, {skipped_dup} 个重复)")