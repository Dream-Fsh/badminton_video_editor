@echo off
echo 羽毛球视频回合检测系统（带YOLO运动员检测）
echo ==========================================
echo.

REM 设置编码为UTF-8
chcp 65001 >nul

REM 激活虚拟环境（如果存在）
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo 已激活虚拟环境
) else (
    echo 未找到虚拟环境，使用系统Python
)

REM 检查模型文件
set MODEL_PATH=03_model\trained\best_model_20260323_224211.pth
if not exist "%MODEL_PATH%" (
    echo 错误: 模型文件不存在
    echo 路径: %MODEL_PATH%
    pause
    exit /b 1
)

echo.
echo 配置信息:
echo - 模型: %MODEL_PATH%
echo - YOLO: 已启用 (yolov8n.pt)
echo - 置信度阈值: 0.05 (低阈值捕获更多动作)
echo - 步长: 8 (平衡速度与精度)
echo.

REM 运行主程序
echo 开始预测...
python main.py

echo.
echo ==========================================
echo 预测完成！
echo 结果保存在: 04_output\predictions\
echo 可使用"分析预测结果.bat"查看详细结果
echo ==========================================
pause
