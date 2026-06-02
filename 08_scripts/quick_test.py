#!/usr/bin/env python3
"""
快速测试脚本 - 验证Web应用基本功能
"""

import os
import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)  # 切换工作目录到项目根目录
sys.path.insert(0, str(PROJECT_ROOT / '02_code'))  # 添加代码目录到路径


def test_imports():
    """测试必要的模块导入"""
    print("测试模块导入...")
    
    try:
        import flask
        print("✓ Flask")
    except ImportError:
        print("✗ Flask - 请运行: pip install flask")
        return False
    
    try:
        import cv2
        print("✓ OpenCV")
    except ImportError:
        print("✗ OpenCV - 请运行: pip install opencv-python")
        return False
    
    try:
        import numpy
        print("✓ NumPy")
    except ImportError:
        print("✗ NumPy - 请运行: pip install numpy")
        return False
    
    return True

def test_web_app():
    """测试Web应用加载"""
    print("\n测试Web应用...")
    
    try:
        from web_app import app
        print("✓ Web应用加载成功")
        return True
    except Exception as e:
        print(f"✗ Web应用加载失败: {e}")
        return False

def test_directories():
    """测试目录结构"""
    print("\n检查目录结构...")
    
    import os
    dirs = [
        'templates',
        'static',
        '01_data/raw_videos',
        '04_output',
        'temp'
    ]
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - 将自动创建")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ {dir_path} - 已创建")
    
    return True

def main():
    print("=" * 50)
    print("羽毛球视频智能剪辑系统 - 快速测试")
    print("=" * 50)
    
    success = True
    
    # 测试导入
    if not test_imports():
        success = False
    
    # 测试目录
    if not test_directories():
        success = False
    
    # 测试Web应用
    if not test_web_app():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！可以启动Web应用")
        print("运行命令: python simple_start.py")
        print("或双击: start_web.bat")
    else:
        print("❌ 测试失败，请解决上述问题后重试")
    print("=" * 50)

if __name__ == '__main__':
    main()