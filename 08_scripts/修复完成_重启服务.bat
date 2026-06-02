@echo off
echo 配置已修复，请重启Web服务
echo =========================================
echo.

echo 错误已修复：
echo 1. 添加了 min_interval 和 smooth_window 的默认值
echo 2. 在配置文件中添加了完整的 post_processing 配置
echo.

echo 请按以下步骤操作：
echo.
echo 步骤1: 停止当前Web服务
echo    - 在运行 python web_app.py 的窗口
echo    - 按 Ctrl+C 停止服务
echo.
echo 步骤2: 重新启动服务
echo    python 02_code\web_app.py
echo.
echo 步骤3: 重新测试视频
echo    上传同样的视频，速度应该提升10-15倍
echo.
echo =========================================
echo.

pause
