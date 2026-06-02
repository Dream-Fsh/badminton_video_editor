@echo off
REM 羽毛球视频自动剪辑系统 - Windows安装脚本

echo ========================================
echo 羽毛球视频自动剪辑系统 - 安装程序
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.7+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] 检测到Python环境
python --version

REM 创建虚拟环境
echo.
echo [2/4] 创建虚拟环境...
if not exist .venv (
    python -m venv .venv
    echo 虚拟环境创建成功
) else (
    echo 虚拟环境已存在
)

REM 激活虚拟环境
echo.
echo [3/4] 激活虚拟环境...
call .venv\Scripts\activate.bat

REM 安装依赖
echo.
echo [4/4] 安装Python依赖包...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 接下来的步骤：
echo 1. 下载I3D预训练权重到 03_model/pretrained/
echo 2. 安装FFmpeg（如果尚未安装）
echo 3. 将视频文件放入 01_data/raw_videos/
echo 4. 运行主程序: python 02_code/main.py
echo.
echo 详细说明请查看 README.md
echo.
pause
