@echo off
title 冒险岛自动打猪 - 纯点位巡航模式 (v10.3)
cd /d "%~dp0"
echo ============================================
echo   冒险岛自动打猪 v10.3 (点位巡航 + 顺路打怪 + 下跳 + 安全点商城 + 恢复路线)
echo   玩家定位: color_anchor 色块锚点 (no-ocr)
echo   怪物检测: YOLO 野猪 + 树妖(木妖) 双模型
echo   其他玩家: 小地图红点检测 (R1/R2, 检测到即挂机, 消失2秒自动恢复)
echo   安全点: 定时停止打怪走进商城(T), ESC+回车返回, 重置测谎仪
echo   恢复路线: 退出商城/跌落底层后自动走回巡游线
echo   热键: F1 开始录制 / F2 打普通点 / F3 打跳跃点
echo         F4 保存并开始巡航 / F5 清空录制 / F8 暂停恢复
echo         F6=点位置定位 F10=安全点录制 F11=恢复路线录制 F12=恢复点位
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
"%PY%" tools\auto_combat.py --cfg shanda_legacy --monster-backend yolo --yolo-model training_runs\wild_boar_real_hardneg_v4_960\weights\best.pt --yolo-confidence 0.07 --yolo-iou 0.70 --yolo-image-size 960 --no-color-verify --show-viz --no-ocr --no-capture --player-name 麻超圆 --fps-limit 12 --monster-labels wild_boar --no-terrain --mode minimap_patrol --map-name "野猪的领土！！"
echo.
echo 脚本已退出, 按任意键关闭窗口
pause >nul