@echo off
chcp 65001 >nul
title 羽毛球视频智能剪辑系统

echo ================================================================
echo 羽毛球视频智能剪辑系统 - Web前端
echo ================================================================
echo.

echo 正在启动Web服务...
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 停止服务
echo.

python 08_scripts/simple_start.py

pause