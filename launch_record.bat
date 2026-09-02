@echo off
echo =============================================
echo  MapleStory 怪物识别 + 录制模式 一键启动
echo =============================================
echo.

:: 激活虚拟环境
call "C:\quant_lab\lstm_gpu_venv\Scripts\activate.bat"

echo 正在启动录制可视化界面...
echo.

python tools\live_perception_viewer.py --cfg shanda_legacy --player-name 麻超圆 --monster-backend sprite --show-viz

echo.
echo 脚本已退出。
echo 请按任意键关闭此窗口...