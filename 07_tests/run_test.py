"""
完整的项目测试脚本 - 发现并记录所有bug
"""
import os
import sys
import subprocess
from pathlib import Path

# 确保在项目根目录
os.chdir(Path(__file__).parent)
sys.path.insert(0, '02_code')

print("=" * 70)
print("羽毛球视频自动剪辑系统 - 完整测试")
print("=" * 70)

bugs_found = []

# 测试1: 环境检查
print("\n[测试1] 运行环境检查...")
try:
    result = subprocess.run([sys.executable, 'check_environment.py'], 
                          capture_output=True, text=True, timeout=10)
    if 'FFmpeg未安装' in result.stdout:
        bugs_found.append("Bug: FFmpeg检测失败（实际已安装）")
    print("✓ 环境检查脚本运行完成")
except Exception as e:
    bugs_found.append(f"Bug: 环境检查脚本失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试2: 配置加载
print("\n[测试2] 测试配置加载...")
try:
    from config_loader import load_config
    config = load_config()
    raw_videos = config.get('paths', 'raw_videos')
    
    if not Path(raw_videos).exists():
        bugs_found.append(f"Bug: 视频路径不存在 - {raw_videos}")
    else:
        print(f"✓ 配置加载成功，视频路径: {raw_videos}")
except Exception as e:
    bugs_found.append(f"Bug: 配置加载失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试3: 数据预处理器初始化
print("\n[测试3] 测试数据预处理器...")
try:
    from data_preprocess import VideoPreprocessor
    preprocessor = VideoPreprocessor()
    
    # 检查路径
    if not Path(preprocessor.raw_videos_dir).exists():
        bugs_found.append(f"Bug: 预处理器视频路径错误 - {preprocessor.raw_videos_dir}")
    else:
        print(f"✓ 预处理器初始化成功")
        print(f"  视频目录: {preprocessor.raw_videos_dir}")
except Exception as e:
    bugs_found.append(f"Bug: 预处理器初始化失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试4: 视频编辑器初始化
print("\n[测试4] 测试视频编辑器...")
try:
    from video_editor import VideoEditor
    editor = VideoEditor()
    print(f"✓ 视频编辑器初始化成功")
except Exception as e:
    bugs_found.append(f"Bug: 视频编辑器初始化失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试5: 检查视频文件
print("\n[测试5] 检查视频文件...")
try:
    videos_dir = Path('01_data/raw_videos')
    if videos_dir.exists():
        videos = list(videos_dir.glob('*.mp4')) + list(videos_dir.glob('*.avi'))
        print(f"✓ 找到 {len(videos)} 个视频文件")
        if len(videos) == 0:
            bugs_found.append("Bug: 视频目录为空")
    else:
        bugs_found.append("Bug: 视频目录不存在")
except Exception as e:
    bugs_found.append(f"Bug: 检查视频文件失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试6: 检查标注文件
print("\n[测试6] 检查标注文件...")
try:
    annotations_dir = Path('01_data/annotations')
    if annotations_dir.exists():
        annotations = list(annotations_dir.glob('*_annotations.json'))
        print(f"✓ 找到 {len(annotations)} 个标注文件")
        
        # 验证标注文件格式
        if len(annotations) > 0:
            import json
            sample = annotations[0]
            with open(sample, 'r', encoding='utf-8') as f:
                data = json.load(f)
                required_keys = ['video_name', 'annotations']
                for key in required_keys:
                    if key not in data:
                        bugs_found.append(f"Bug: 标注文件缺少字段 '{key}' - {sample.name}")
            print(f"  ✓ 标注文件格式验证通过")
    else:
        bugs_found.append("Bug: 标注目录不存在")
except Exception as e:
    bugs_found.append(f"Bug: 检查标注文件失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试7: 测试FFmpeg
print("\n[测试7] 测试FFmpeg...")
try:
    result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ FFmpeg可用")
    else:
        bugs_found.append("Bug: FFmpeg不可用")
except Exception as e:
    bugs_found.append(f"Bug: FFmpeg测试失败 - {e}")
    print(f"✗ 失败: {e}")

# 输出结果
print("\n" + "=" * 70)
print("测试结果汇总")
print("=" * 70)

if bugs_found:
    print(f"\n发现 {len(bugs_found)} 个问题:\n")
    for i, bug in enumerate(bugs_found, 1):
        print(f"{i}. {bug}")
else:
    print("\n✓ 所有测试通过！")

print("\n" + "=" * 70)
