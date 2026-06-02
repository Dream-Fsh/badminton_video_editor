@echo off
echo 正在安装CUDA版本的PyTorch...
echo ===========================================
echo.

echo 步骤1: 卸载CPU版本（已完成）
echo 步骤2: 安装CUDA 11.8版本...
echo 这可能需要几分钟时间，请耐心等待...
echo.

REM 安装CUDA版本
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo ===========================================
echo 安装完成！
echo.
echo 验证安装:
echo python 08_scripts/check_environment.py
echo.
echo 验证后，如果显示CUDA可用，重启Web服务:
echo python 02_code\web_app.py
echo.
echo ===========================================
pause
