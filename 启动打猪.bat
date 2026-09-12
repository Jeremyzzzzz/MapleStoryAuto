@echo off
title 冒险岛自动打猪 - 点位巡航模式 (v11.5)
cd /d "%~dp0"
echo ============================================
echo   冒险岛自动打猪 v11.5 (点位巡航 + 顺路打怪, 野猪的领土！！)
echo   怪物检测: YOLO 野猪(wild_boar_real_hardneg_v4_960) conf=0.07 + 树妖(warrior_stump_v2)
echo   玩家定位: color_anchor 色块锚点(no-ocr) + 名字字形 + 位置校验
echo   顺路打怪范围: ±300px   扣血反击: 已禁用(只保留被击退反击)
echo   安全点: 每小时第18/38/58分进商城2分钟(测谎规避), 恢复路线自动走回
echo   热键: F1 开始录制 / F2 打普通点 / F3 打跳跃点 / F4 保存并巡航
echo         F5=遇玩家挂机开关(面板有状态显示) / F12=清空录制
echo         F6=点位置定位 F7=保存航点 F8=暂停恢复 F9=退出
echo         F10=安全点录制 F11=恢复路线录制
echo   使用前请先点游戏窗口聚焦, 日志会显示 游戏聚焦=Y
echo ============================================
echo 启动中... (关闭本窗口即停止脚本)
echo.
echo [1/2] 检查并清理残留进程/锁...
rem 按命令行匹配杀旧 bot(匹配串 auto+_combat 拼接, 防止 powershell 匹配到自己自杀)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='auto'+'_combat'; Get-CimInstance Win32_Process -Filter 'Name=''python.exe''' | Where-Object { $_.CommandLine -match $p } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
if exist "%TEMP%\auto_combat_ms.lock" del /f /q "%TEMP%\auto_combat_ms.lock"
echo [2/2] 启动 bot...
set "PY=C:\quant_lab\lstm_gpu_venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" tools\auto_combat.py --cfg shanda_legacy --monster-backend yolo --yolo-model training_runs\wild_boar_real_hardneg_v4_960\weights\best.pt --yolo-confidence 0.07 --yolo-iou 0.70 --yolo-image-size 960 --no-color-verify --show-viz --no-ocr --no-capture --player-name 麻超圆 --fps-limit 12 --monster-labels wild_boar --no-terrain --mode minimap_patrol --map-name "野猪的领土！！"
echo.
echo 切回鳄鱼版: --yolo-model training_runs\crocodile_v2_hardneg_20260908\weights\best.pt --yolo-confidence 0.65 --yolo-image-size 1280 --monster-labels crocodile --map-name "沼泽地2" 并把 config 中 yolo_confidence 改回 0.65、patrol_hunt_range_px 改 400、stump_model 置空
echo 脚本已退出, 按任意键关闭窗口
pause >nul
