@echo off
echo 快速优化羽毛球视频检测速度
echo =========================================
echo.

REM 检查CUDA
echo 检查CUDA环境...
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
if errorlevel 1 (
    echo 检查失败，请手动运行: python 08_scripts/check_environment.py
)

echo.
echo 当前配置优化建议:
echo.
echo 1. 使用高速配置（立即生效）
echo    copy 05_config\config_fast.yaml 05_config\config.yaml
echo.
echo 2. 如果YOLO不是必须的，可以禁用:
echo    编辑 05_config\config.yaml
echo    athlete_detection.enabled: false
echo.
echo 3. 如果使用的是CPU，强烈建议安装CUDA版本PyTorch
echo    速度可提升10-20倍！
echo.
echo =========================================
pause
