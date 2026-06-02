#!/usr/bin/env python3
"""
Web应用测试脚本
验证所有路由和功能是否正常工作
"""

import os
import sys
import requests
import time
import threading
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)  # 切换工作目录到项目根目录
sys.path.insert(0, str(PROJECT_ROOT / '02_code'))  # 添加代码目录到路径

from web_app import app

def test_routes():
    """测试所有路由"""
    base_url = 'http://localhost:5000'
    
    # 等待服务器启动
    time.sleep(2)
    
    try:
        # 测试主页
        response = requests.get(f'{base_url}/')
        print(f"✓ 主页访问: {response.status_code}")
        
        # 测试API状态
        response = requests.get(f'{base_url}/api/status')
        print(f"✓ API状态: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  当前步骤: {data.get('current_step', 'unknown')}")
        
        print("\n🎉 所有基础路由测试通过!")
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保Web应用正在运行")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def start_test_server():
    """启动测试服务器"""
    app.run(host='localhost', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    print("=" * 50)
    print("羽毛球视频智能剪辑系统 - Web应用测试")
    print("=" * 50)
    
    # 在后台启动服务器
    server_thread = threading.Thread(target=start_test_server, daemon=True)
    server_thread.start()
    
    print("启动测试服务器...")
    
    # 运行测试
    test_routes()
    
    print("\n测试完成! 按Ctrl+C退出")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n测试结束")