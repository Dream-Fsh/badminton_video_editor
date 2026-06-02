#!/usr/bin/env python3
"""
简化的Web应用启动脚本
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

def create_directories():
    """创建必要目录"""
    dirs = ['01_data/raw_videos', '04_output', 'temp']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def open_browser():
    """延迟打开浏览器"""
    time.sleep(3)
    try:
        webbrowser.open('http://localhost:5000')
    except:
        pass

def main():
    print("=" * 50)
    print("羽毛球视频智能剪辑系统")
    print("=" * 50)
    
    # 创建目录
    create_directories()
    
    # 启动浏览器 (仅在主进程中启动，避免 debug 模式下打开两次)
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
    
    try:
        from web_app import app
        
        print("\n🚀 启动Web服务...")
        print("📱 访问地址: http://localhost:5000")
        print("🛑 按 Ctrl+C 停止服务\n")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请安装必要依赖: pip install flask opencv-python")
    except Exception as e:
        print(f"启动失败: {e}")
    
    input("\n按回车键退出...")

if __name__ == '__main__':
    main()