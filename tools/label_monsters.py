"""交互式怪物标注工具 — 把 monster/测试集/*.png 里的怪物精灵抠出来
并存入 monster_templates_final/<class>/ 作为模板。

操作:
  1) 左键点击怪物中心 → 输入类别编号(见下方提示) → 自动抠精灵并入库
  2) 右键点击 → 删除最近一个已标怪物(撤销)
  3) 'n' → 加载下一张场景图
  4) 's' → 跳过当前图
  5) 'q' → 退出

依赖: opencv-python
用法: /c/quant_lab/lstm_gpu_venv/Scripts/python.exe tools/label_monsters.py
"""

import os
import sys
import glob
import cv2
import numpy as np
from pathlib import Path

REPO = Path(r"C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp")
SCENE_DIR = REPO / "monster" / "测试集"
OUT_DIR = REPO / "monster_templates_final"

CLASSES = {
    "1": ("red_snail",      "红蜗牛"),
    "2": ("blue_snail",     "蓝蜗牛"),
    "3": ("green_mushroom", "绿蘑菇"),
    "4": ("slime",          "绿水灵"),
    "5": ("stump",          "树妖"),
    "6": ("flower_mushroom","花蘑菇"),
    "7": ("orange_mushroom","橙蘑菇"),
}

# 用怪物中心向外扩 box, 抠出精灵; mask = 颜色 mask
COLOR_RANGES = {
    "red_snail":      [((0, 60, 60), (22, 255, 255)), ((165, 60, 60), (180, 255, 255))],
    "blue_snail":     [((90, 50, 50), (140, 255, 255))],
    "green_mushroom": [((30, 60, 60), (78, 255, 230))],
    "slime":          [((55, 45, 45), (112, 255, 255))],
    "stump":          [((5, 40, 40), (32, 255, 170))],
    "flower_mushroom":[((150, 50, 50), (180, 255, 255)), ((0, 50, 50), (10, 255, 255))],
    "orange_mushroom":[((6, 80, 110), (26, 255, 255))],
}

# 怪物精灵合理尺寸(高/宽)
SIZE = {"red_snail": 50, "blue_snail": 40, "green_mushroom": 50,
        "slime": 55, "stump": 70, "flower_mushroom": 55, "orange_mushroom": 55}

def build_mask(hsv, cls):
    """生成怪物的彩色 mask (精灵部分)"""
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOR_RANGES.get(cls, []):
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def extract_sprite(img, hsv, cx, cy, cls):
    """以 (cx,cy) 为中心, 找最近的 mask 连通域, 抠出精灵(绿幕合成)"""
    m_full = build_mask(hsv, cls)
    # 找离点击点最近的连通域
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m_full, 8)
    best_i = -1
    best_d = 1e9
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        ccx, ccy = centroids[i]
        d = (ccx - cx) ** 2 + (ccy - cy) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    if best_i < 1:
        return None, None
    x, y, w, h, area = stats[best_i]
    # 稍微 padding
    pad = 4
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad); y1 = min(img.shape[0], y + h + pad)
    patch = img[y0:y1, x0:x1].copy()
    pmask = (m_full[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    # 抠出精灵中最大的连通 mask 块(防止抓到大片背景)
    n2, _, _, _ = cv2.connectedComponentsWithStats(pmask, 8)
    if n2 > 1:
        sizes = [cv2.connectedComponentsWithStats(pmask, 8)[3][i, cv2.CC_STAT_AREA] for i in range(1, n2)]
        keep = np.argmax(sizes) + 1
        n3, lbls, st3, _ = cv2.connectedComponentsWithStats(pmask, 8)
        pmask[:] = 0
        pmask[lbls == keep] = 255
    # 绿幕合成
    comp = np.zeros_like(patch)
    comp[:] = (0, 255, 0)  # 绿幕背景
    comp[pmask > 0] = patch[pmask > 0]
    return comp, pmask


def list_scenes():
    files = []
    for ext in ("png", "jpg", "jpeg"):
        files.extend(glob.glob(str(SCENE_DIR / f"*.{ext}")))
    return sorted(files)


def main():
    if not SCENE_DIR.exists():
        print(f"[label_monsters] 场景目录不存在: {SCENE_DIR}")
        return
    OUT_DIR.mkdir(exist_ok=True)
    scenes = list_scenes()
    if not scenes:
        print(f"[label_monsters] {SCENE_DIR} 里没有图片; 请把游戏截图放进去")
        return

    print("类别编号 (按一下数字键即选):")
    for k, (en, zh) in CLASSES.items():
        print(f"  {k}: {zh} ({en})")

    cv2.namedWindow("label_monsters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label_monsters", 1280, 720)

    history = []  # (scene_path, cls_en, save_path)
    idx = 0
    pending_click = None  # (cx, cy, cls_code)
    need_redraw = True

    def redraw(img, anns, hover=None):
        vis = img.copy()
        # 已标怪物
        for x, y, cls_en in anns:
            color = (0, 255, 255) if cls_en == "red_snail" else \
                    (255, 180, 0) if cls_en == "blue_snail" else \
                    (60, 200, 60) if cls_en == "green_mushroom" else \
                    (180, 255, 120) if cls_en == "slime" else \
                    (0, 80, 255) if cls_en == "stump" else \
                    (255, 100, 200)
            cv2.circle(vis, (x, y), 14, color, 2)
            cv2.putText(vis, cls_en[:6], (x + 18, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if hover:
            cv2.circle(vis, hover, 18, (200, 200, 200), 1)
        cv2.putText(vis, "左键=标怪, 数字键=选类别, n=下一张, u=撤销, q=退出",
                    (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return vis

    def on_mouse(event, x, y, flags, param):
        nonlocal pending_click, need_redraw
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click = (x, y, None)  # 等类别键
            need_redraw = True

    cv2.setMouseCallback("label_monsters", on_mouse)

    while idx < len(scenes):
        scene_path = scenes[idx]
        img = cv2.imdecode(np.fromfile(scene_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  无法读 {scene_path}, 跳过")
            idx += 1
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        anns = []  # 当前图标注 (cx, cy, cls_en)

        print(f"\n=== [{idx+1}/{len(scenes)}] {scene_path} ({img.shape[1]}x{img.shape[0]}) ===")
        pending_click = None
        cv2.imshow("label_monsters", redraw(img, anns))

        while True:
            k = cv2.waitKey(50) & 0xFF
            if k == ord("q"):
                cv2.destroyAllWindows()
                print(f"本次标注: 共 {len(history)} 个精灵入库")
                for sp in [h[2] for h in history]:
                    print("  -", sp)
                return
            if k == ord("n") or k == ord("s"):
                idx += 1
                break
            if k == ord("u"):
                if anns:
                    last = anns.pop()
                    print(f"  撤销最近标: {last[2]} @ ({last[0]},{last[1]})")
                    history = [h for h in history if not (h[0] == scene_path and h[2] == last[2])]
                need_redraw = True
            if pending_click and k in (ord(c) for c in CLASSES.keys()):
                cx, cy, _ = pending_click
                cls_code = chr(k)
                cls_en, cls_zh = CLASSES[cls_code]
                comp, pmask = extract_sprite(img, hsv, cx, cy, cls_en)
                if comp is None or comp.shape[0] < 8 or comp.shape[1] < 8:
                    print(f"  ({cx},{cy}) 找不到 {cls_zh} 颜色的 mask, 请确认颜色范围")
                    pending_click = None
                    continue
                cls_dir = OUT_DIR / cls_en
                cls_dir.mkdir(exist_ok=True)
                n = len(list(cls_dir.glob("scene_*.png")))
                save_path = cls_dir / f"scene_{n:03d}.png"
                ok, buf = cv2.imencode(".png", comp)
                save_path.write_bytes(buf.tobytes())
                anns.append((cx, cy, cls_en))
                history.append((scene_path, cls_en, str(save_path)))
                print(f"  ✓ {cls_zh} @ ({cx},{cy}) -> {save_path.name}  size={comp.shape[1]}x{comp.shape[0]}")
                pending_click = None
                need_redraw = True

            if need_redraw:
                cv2.imshow("label_monsters", redraw(img, anns))
                need_redraw = False

    cv2.destroyAllWindows()
    print(f"\n所有场景图处理完成. 共入库 {len(history)} 个怪物精灵")


if __name__ == "__main__":
    main()