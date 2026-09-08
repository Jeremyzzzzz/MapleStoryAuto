@echo off
title 冒险岛自动打鳄鱼 - 纯点位巡航模式 (v11.3)
cd /d "%~dp0"
echo ============================================
echo   冒险岛自动打鳄鱼 v11.3 (点位巡航 + 顺路打怪, 沼泽地2)
echo   怪物检测: YOLO 鳄鱼(crocodile_v2_hardneg) conf=0.25
echo   玩家定位: color_anchor 色块锚点 (no-ocr)
echo   顺路打怪范围: ±400px   扣血立即反击: 已恢复
echo   其他玩家: 红点检测(已关闭挂机暂停)
echo   安全点: 每小时第18/38/58分进商城2分钟(测谎规避), 恢复路线自动走回
echo   热键: F1 开始录制 / F2 打普通点 / F3 打跳跃点
echo         F4 保存并开始巡航 / F5 清空录制 / F8 暂停恢复
echo         F6=点位置定位 F10=安全点录制 F11=恢复路线录制
echo   使用前请先点游戏窗口聚焦, 日志会显示 游戏聚焦=Y
echo ============================================
echo 启动中... (关闭本窗口即停止脚本)
echo.
echo [1/2] 检查并清理残留进程/锁...
rem 按命令行匹配杀旧 bot(匹配串 'auto'+'_combat' 拼接, 防止 powershell 匹配到自己的命令行自杀)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='auto'+'_combat'; Get-CimInstance Win32_Process -Filter 'Name=''python.exe''' | Where-Object { $_.CommandLine -match $p } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
del /Q "%TEMP%\auto_combat_ms.lock" >nul 2>&1
echo [2/2] 启动 bot...
set "PY=C:\quant_lab\lstm_gpu_venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" tools\auto_combat.py --cfg shanda_legacy --monster-backend yolo --yolo-model training_runs\crocodile_v2_hardneg_20260908\weights\best.pt --yolo-confidence 0.25 --yolo-iou 0.70 --yolo-image-size 1280 --no-color-verify --show-viz --no-ocr --no-capture --player-name 麻超圆 --fps-limit 12 --monster-labels crocodile --no-terrain --mode minimap_patrol --map-name "沼泽地2"
echo.
echo 切回野猪版: --yolo-model training_runs\wild_boar_real_hardneg_v4_960\weights\best.pt --yolo-confidence 0.07 --yolo-image-size 960 --monster-labels wild_boar --map-name "野猪的领土！！" 并把 config 中 yolo_confidence/patrol_hunt_range_px 改回、stump_model 填回树妖模型
echo 脚本已退出, 按任意键关闭窗口
pause >nul