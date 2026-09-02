# -*- coding: utf-8 -*-
"""生成桌面启动 bat(GBK 编码, Windows cmd 直接可跑)。"""
import os

PROJ = r"C:\Users\Administrator\Documents\ChatGPT\冒险岛\MapleStoryAutoLevelUp"
PY = r"C:\quant_lab\lstm_gpu_venv\Scripts\python.exe"

lines = []
lines.append("@echo off")
lines.append("title 冒险岛自动打猪 - 纯点位巡航模式 (v6)")
lines.append('cd /d "{}"'.format(PROJ))
lines.append("echo ============================================")
lines.append("echo   冒险岛自动打猪 v6 (纯点位巡航 + 顺路打怪)")
lines.append("echo   玩家定位: color_anchor 色块锚点 (codex新版, no-ocr)")
lines.append("echo   热键: F1 开始录制 / F2 打普通点 / F3 打跳跃点")
lines.append("echo         F4 保存并开始巡航 / F5 清空录制 / F8 暂停恢复")
lines.append("echo   使用前请先点游戏窗口聚焦, 日志会显示 游戏聚焦=Y")
lines.append("echo ============================================")
lines.append("echo 启动中... (关闭本窗口即停止脚本)")
cmd = (
    '"{}" tools\\auto_combat.py --cfg shanda_legacy --monster-backend yolo '
    "--yolo-model training_runs\\wild_boar_real_hardneg_v4_960\\weights\\best.pt "
    "--yolo-confidence 0.10 --yolo-iou 0.70 --yolo-image-size 960 "
    "--no-color-verify --show-viz --player-name 麻超圆 --fps-limit 12 "
    "--no-ocr --monster-labels wild_boar --no-terrain "
    "--mode minimap_patrol --map-name \"野猪的领土！！\""
).format(PY)
lines.append(cmd)
lines.append("echo.")
lines.append("echo 脚本已退出, 按任意键关闭窗口")
lines.append("pause >nul")

out = r"C:\Users\Administrator\Desktop\自动打猪.bat"
with open(out, "w", encoding="gbk") as f:
    f.write("\r\n".join(lines) + "\r\n")

print("已写入:", out)
print("大小:", os.path.getsize(out))
# 验证
with open(out, encoding="gbk") as f:
    for line in f:
        if "cd /d" in line or "python.exe" in line:
            print("  ", line.rstrip())
