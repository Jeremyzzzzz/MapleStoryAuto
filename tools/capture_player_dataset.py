# -*- coding: utf-8 -*-
"""采集玩家训练帧(只读, 不发送任何按键)。

用法(项目根目录):
  python tools/capture_player_dataset.py --cfg shanda_legacy --player-name 麻超圆 --out training_data/player_capture_v2 --count 200 --interval 1.0

原理:
  - ReadOnlyWindowCapture 只读截取"冒险岛怀旧服"窗口(不动游戏)。
  - 每 interval 秒抓一帧, 用当前玩家检测链路(color_anchor 蓝条+红分打分)
    定位玩家 -> 标注 YOLO 格式 box(归一化 cx,cy,w,h)。
  - 玩家检测置信度低于 min_conf 的帧跳过(标注不可靠的不进数据集)。
  - 保存: 原始帧 images/xxx.jpg + 标注 labels/xxx.txt。
  姿态覆盖: 用户按平常操作角色(走动/跳跃/边缘), 脚本多抓几组即可。
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.yolo_monster_viewer import (  # noqa: E402
    ReadOnlyWindowCapture,
    ReadOnlyPlayerDetector,
    load_config,
    resolve_gameplay_height,
    find_visible_window_title,
)


def parse_args():
    p = argparse.ArgumentParser(description="Capture player training frames (read-only).")
    p.add_argument("--cfg", default="shanda_legacy")
    p.add_argument("--player-name", default="麻超圆")
    p.add_argument("--window-title-token", default="冒险岛")
    p.add_argument("--out", default=str(REPO_ROOT / "training_data" / "player_capture_v2"))
    p.add_argument("--count", type=int, default=200, help="目标采集帧数")
    p.add_argument("--interval", type=float, default=1.0, help="采集间隔秒")
    p.add_argument("--min-conf", type=float, default=0.5,
                   help="玩家检测置信度低于此值时跳过该帧(标注不可靠)")
    p.add_argument("--box-width", type=int, default=150)
    p.add_argument("--box-height", type=int, default=120)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    overlay = cfg["perception_overlay"]
    ui_y = int(cfg["ui_coords"]["ui_y_start"])
    ref_w = cfg["ui_coords"].get("reference_width")

    safe = "".join(c if c.isalnum() else "_" for c in args.player_name)
    tpl = f"nametag/{safe}_player.png"
    detector = ReadOnlyPlayerDetector(
        tpl,
        threshold=float(overlay.get("player_match_threshold", 0.30)),
        box_size=tuple(overlay["player_box_size"]),
        center_offset_y=abs(int(cfg["nametag"]["offset"][1])),
        identity_threshold=float(overlay.get("player_identity_threshold", 0.48)),
        local_identity_threshold=float(overlay.get("player_local_identity_threshold", 0.38)),
        identity_margin=float(overlay.get("player_identity_margin", 0.015)),
        glyph_threshold=int(overlay.get("player_glyph_threshold", 130)),
        glyph_weight=float(overlay.get("player_glyph_weight", 0.70)),
        candidate_count=int(overlay.get("player_candidate_count", 16)),
        lock_radius=float(overlay.get("player_lock_radius", 180.0)),
        reacquire_misses=int(overlay.get("player_reacquire_misses", 12)),
        center_weight=float(overlay.get("player_center_weight", 0.12)),
        require_identity_seed=False,
        max_valid_x=overlay.get("player_max_valid_x"),
        max_valid_y=overlay.get("player_max_valid_y"),
        glyph_min_columns=int(overlay.get("player_glyph_min_columns", 2)),
        color_anchor_enabled=True,
        color_anchor_name_offset_y=int(overlay.get("player_color_anchor_name_offset_y", 24)),
        color_anchor_min_red_fraction=0.0,  # 标注用: 红分不打分, 只靠蓝条
        color_anchor_local_radius=float(overlay.get("player_color_anchor_local_radius", 260)),
        keep_color_anchor_misses=6,
    )

    title = find_visible_window_title(args.window_title_token)
    if not title:
        print("未找到游戏窗口(标题含: %s)" % args.window_title_token, flush=True)
        return 1
    capture = ReadOnlyWindowCapture(title)

    img_dir = Path(args.out) / "images"
    lbl_dir = Path(args.out) / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    started = time.time()
    print(f"开始采集: 窗口={title!r} 目标={args.count}帧 间隔={args.interval}s", flush=True)
    print("请在游戏中正常走动/跳跃/边缘游走, 覆盖多种姿态。", flush=True)
    try:
        time.sleep(0.5)
        while saved < args.count:
            frame = capture.get_frame()
            if frame is None:
                time.sleep(0.2)
                continue
            gh = resolve_gameplay_height(frame.shape, ui_y, ref_w)
            gameplay = frame[:gh]
            det = detector.detect(gameplay, gh)
            kept = False
            if det is not None and det.get("box"):
                conf = float(det.get("confidence", 0.0))
                if conf >= args.min_conf:
                    bx, by, bw, bh = (int(v) for v in det["box"])
                    # 裁剪到画面内
                    bx = max(0, bx); by = max(0, by)
                    bw = min(gameplay.shape[1] - bx, bw)
                    bh = min(gameplay.shape[0] - by, bh)
                    if bw > 10 and bh > 10:
                        cx = (bx + bw / 2.0) / gameplay.shape[1]
                        cy = (by + bh / 2.0) / gameplay.shape[0]
                        nx = bw / float(gameplay.shape[1])
                        ny = bh / float(gameplay.shape[0])
                        # 归一化中心/宽高(用检测框, 但画大一点的"人物全身"框)
                        # 人物框: 以检测框中心为准, 用固定 box 尺寸(全身)
                        cx_full = cx
                        cy_full = cy
                        nw = min(0.30, args.box_width / float(gameplay.shape[1]))
                        nh = min(0.35, args.box_height / float(gameplay.shape[0]))
                        label = f"0 {cx_full:.6f} {cy_full:.6f} {nw:.6f} {nh:.6f}\n"
                        img_name = f"live_{int(time.time()*1000)}_{saved:05d}.jpg"
                        import cv2
                        import numpy as np
                        cv2.imencode(".jpg", frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
                                         str(img_dir / img_name))
                        (lbl_dir / (Path(img_name).stem + ".txt")).write_text(
                            label, encoding="utf-8")
                        saved += 1
                        kept = True
                        print(f"[{saved}/{args.count}] conf={conf:.2f} box=({bx},{by},{bw},{bh}) "
                              f"skip={skipped}", flush=True)
            if not kept:
                skipped += 1
            time.sleep(args.interval)
    finally:
        capture.stop()
    print(f"完成: 保存 {saved} 帧, 跳过 {skipped} 帧 -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
