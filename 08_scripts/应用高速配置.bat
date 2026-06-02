@echo off
echo 应用高速优化配置
echo ===========================================
echo.

echo 备份原配置...
if exist "05_config\config_backup.yaml" (
    echo 备份文件已存在
) else (
    copy 05_config\config.yaml 05_config\config_backup.yaml
    echo 已备份到 config_backup.yaml
)

echo.
echo 应用高速配置...
copy /Y 05_config\config_fast.yaml 05_config\config.yaml
echo 配置已更新！
echo.

echo 优化内容:
echo - YOLO采样率: 2帧→5帧 (速度提升2.5倍)
echo - 滑动步长: 8→16 (窗口数减半)
echo - 保持YOLO启用 (保证精度)
echo.
echo ===========================================
echo 现在可以重启Web服务: python 02_code\web_app.py
echo ===========================================
pause
