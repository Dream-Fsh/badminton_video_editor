"""
羽毛球视频智能剪辑系统 - Web启动入口
使用方法: python web.py 或 python run_web.py
"""

import os
import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
os.chdir(PROJECT_ROOT)

# 添加代码目录到Python路径
code_dir = PROJECT_ROOT / '02_code'
sys.path.insert(0, str(code_dir))

# 导入并运行Web应用
if __name__ == '__main__':
    try:
        from web_app import app
        
        print("=" * 70)
        print("羽毛球视频智能剪辑系统 - Web服务")
        print("=" * 70)
        print("\n正在启动服务器...")
        print(f"工作目录: {PROJECT_ROOT}")
        print(f"模板目录: {PROJECT_ROOT / 'templates'}")
        print(f"静态文件目录: {PROJECT_ROOT / 'static'}")
        
        # 启动Flask应用
        print("\n服务地址:")
        print("  - 本地访问: http://127.0.0.1:5000")
        print("  - 局域网访问: http://0.0.0.0:5000")
        print("\n按 Ctrl+C 停止服务")
        print("=" * 70 + "\n")
        
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000
        )
        
    except ImportError as e:
        print("=" * 70)
        print("错误：无法导入Web应用模块")
        print("=" * 70)
        print(f"\n详细信息: {e}")
        print("\n请确保：")
        print("1. 已安装所有依赖包：pip install -r requirements.txt")
        print("2. Web应用文件存在：02_code/web_app.py")
        print("3. 模板目录存在：templates/")
        print("4. 静态文件目录存在：static/")
        print("\n" + "=" * 70)
        input("\n按回车键退出...")
        
    except Exception as e:
        print(f"\n启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
