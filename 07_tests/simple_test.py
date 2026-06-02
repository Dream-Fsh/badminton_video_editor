"""简单测试脚本"""
import os
import sys

# 切换到项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, '02_code'))

print("当前工作目录:", os.getcwd())
print("\n开始测试...\n")

# 测试1: 配置加载
print("[1] 测试配置加载...")
try:
    from config_loader import load_config
    config = load_config()
    print("✓ 配置加载成功")
    print(f"  视频路径: {config.get('paths', 'raw_videos')}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试2: 数据预处理器
print("\n[2] 测试数据预处理器...")
try:
    from data_preprocess import VideoPreprocessor
    preprocessor = VideoPreprocessor()
    print("✓ 初始化成功")
    print(f"  视频目录: {preprocessor.raw_videos_dir}")
    print(f"  目录存在: {os.path.exists(preprocessor.raw_videos_dir)}")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 视频编辑器
print("\n[3] 测试视频编辑器...")
try:
    from video_editor import VideoEditor
    editor = VideoEditor()
    print("✓ 初始化成功")
except Exception as e:
    print(f"✗ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")
