"""
测试修复后的功能
"""
import os
import sys
from pathlib import Path

print("=" * 70)
print("测试修复后的功能")
print("=" * 70)

# 测试1: 路径解析
print("\n[1] 测试路径解析...")
sys.path.insert(0, '02_code')

try:
    from config_loader import load_config
    config = load_config()
    
    print("✓ 配置加载成功")
    print(f"  raw_videos: {config.get('paths', 'raw_videos')}")
    print(f"  配置文件存在: {Path(config.get('paths', 'raw_videos')).exists()}")
    
except Exception as e:
    print(f"✗ 配置加载失败: {e}")

# 测试2: FFmpeg检查
print("\n[2] 测试FFmpeg检查...")
try:
    from video_editor import VideoEditor
    editor = VideoEditor()
    print("✓ 视频编辑器初始化成功")
except Exception as e:
    print(f"✗ 视频编辑器初始化失败: {e}")

# 测试3: 数据预处理器
print("\n[3] 测试数据预处理器...")
try:
    from data_preprocess import VideoPreprocessor
    preprocessor = VideoPreprocessor()
    print("✓ 数据预处理器初始化成功")
    print(f"  原始视频目录: {preprocessor.raw_videos_dir}")
    print(f"  目录存在: {Path(preprocessor.raw_videos_dir).exists()}")
except Exception as e:
    print(f"✗ 数据预处理器初始化失败: {e}")

print("\n" + "=" * 70)