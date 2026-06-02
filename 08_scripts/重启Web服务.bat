@echo off
echo 重启Web服务
echo =========================================
echo.

echo 步骤1: 检查配置...
echo.
echo 当前配置：
type 05_config\config.yaml | findstr /B "sliding_window_stride"
type 05_config\config.yaml | findstr /B "sample_rate" | head -1
echo.

echo 步骤2: 启动Web服务...
echo.
echo 服务将在浏览器打开 http://127.0.0.1:5000
echo.
echo 按 Ctrl+C 可以停止服务
echo.
echo =========================================

REM 启动Web服务
python 02_code\web_app.py

pause
