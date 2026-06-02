"""测试路径配置"""
import os
import sys

# 添加代码目录到路径
sys.path.insert(0, '02_code')

print("=" * 70)
print("路径配置检查")
print("=" * 70)

# 检查当前工作目录
print(f"\n当前工作目录: {os.getcwd()}")

# 检查配置文件
config_path = "05_config/config.yaml"
print(f"\n配置文件路径: {config_path}")
print(f"配置文件存在: {os.path.exists(config_path)}")

if os.path.exists(config_path):
    # 加载配置
    from config_loader import Config
    
    # 从项目根目录加载
    config = Config(config_path)
    
    print("\n解析后的路径:")
    print(f"  raw_videos: {config.get('paths', 'raw_videos')}")
    print(f"  processed_frames: {config.get('paths', 'processed_frames')}")
    print(f"  annotations: {config.get('paths', 'annotations')}")
    print(f"  merged_labels: {config.get('paths', 'merged_labels')}")
    
    # 检查路径是否存在
    print("\n路径存在性检查:")
    for key in ['raw_videos', 'processed_frames', 'annotations']:
        path = config.get('paths', key)
        exists = os.path.exists(path)
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {key}: {exists}")
        
        # 如果是raw_videos，列出文件
        if key == 'raw_videos' and exists:
            videos = [f for f in os.listdir(path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            print(f"      找到 {len(videos)} 个视频文件")

print("\n" + "=" * 70)
