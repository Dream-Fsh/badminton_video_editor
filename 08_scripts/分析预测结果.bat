@echo off
chcp 65001 >nul
echo 正在分析最新的预测结果...
python 08_scripts\analyze_predictions.py
if errorlevel 1 pause
echo.
echo 按任意键退出...
pause >nul
