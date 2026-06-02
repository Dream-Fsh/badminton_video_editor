#!/usr/bin/env python3
"""
羽毛球视频智能剪辑系统 - Web前端启动脚本
启动现代化的Web界面服务
"""

import os
import sys
import webbrowser
import threading
import time
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)  # 切换工作目录到项目根目录
sys.path.insert(0, str(PROJECT_ROOT / '02_code'))  # 添加代码目录到路径

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'flask',
        'opencv-python',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("=" * 60)
        print("缺少必要的依赖包")
        print("=" * 60)
        print("\n请安装以下包:")
        for package in missing_packages:
            print(f"  pip install {package}")
        print("\n或者运行: pip install -r requirements.txt")
        print("=" * 60)
        return False
    
    return True

def check_directories():
    """检查必要目录"""
    required_dirs = [
        '01_data/raw_videos',
        '04_output',
        'temp',
        'templates',
        'static'
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ 目录结构检查完成")

def open_browser():
    """延迟打开浏览器"""
    time.sleep(2)  # 等待服务器启动
    webbrowser.open('http://localhost:5000')

def main():
    """主函数"""
    print("=" * 60)
    print("羽毛球视频智能剪辑系统 - Web前端")
    print("=" * 60)
    print()
    
    # 检查依赖
    print("检查系统依赖...")
    if not check_dependencies():
        input("\n按回车键退出...")
        return
    
    # 检查目录
    print("检查目录结构...")
    check_directories()
    
    # 启动浏览器
    print("准备启动Web服务...")
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    try:
        # 导入并启动Flask应用
        from web_app import app
        
        print("\n" + "=" * 60)
        print("🚀 Web服务已启动!")
        print("📱 访问地址: http://localhost:5000")
        print("🛑 按 Ctrl+C 停止服务")
        print("=" * 60)
        print()
        
        # 启动Flask应用
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\n\n服务已停止")
    except Exception as e:
        print(f"\n启动失败: {e}")
        print("\n请检查:")
        print("1. 端口5000是否被占用")
        print("2. 是否有足够的权限")
        print("3. 依赖包是否正确安装")
        input("\n按回车键退出...")

if __name__ == '__main__':
    main()