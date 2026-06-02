"""
羽毛球视频自动剪辑系统 - 启动入口
本文件用于从项目根目录启动系统
实际主程序位于 02_code/main.py
"""

import os
import sys

# 添加代码目录到Python路径
code_dir = os.path.join(os.path.dirname(__file__), '02_code')
sys.path.insert(0, code_dir)

# 切换到代码目录
os.chdir(code_dir)

# 导入并运行主程序
if __name__ == '__main__':
    try:
        from main import main
        main()
    except ImportError as e:
        print("=" * 70)
        print("错误：无法导入主程序模块")
        print("=" * 70)
        print(f"\n详细信息: {e}")
        print("\n请确保：")
        print("1. 已安装所有依赖包：pip install -r requirements.txt")
        print("2. 配置文件存在：05_config/config.yaml")
        print("3. 或直接运行：cd 02_code && python main.py")
        print("\n" + "=" * 70)
        input("\n按回车键退出...")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
