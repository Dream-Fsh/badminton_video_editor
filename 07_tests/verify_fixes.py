"""
验证所有bug修复
"""
import os
import sys
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root / '02_code'))

print("=" * 70)
print("验证Bug修复")
print("=" * 70)

all_passed = True

# 测试1: 从项目根目录加载配置
print("\n[测试1] 从项目根目录加载配置...")
try:
    from config_loader import load_config
    config = load_config()
    assert Path(config.get('paths', 'raw_videos')).exists()
    print("✓ 通过")
except Exception as e:
    print(f"✗ 失败: {e}")
    all_passed = False

# 测试2: 从02_code目录加载配置
print("\n[测试2] 从02_code目录加载配置...")
try:
    os.chdir(project_root / '02_code')
    # 重新导入以测试不同工作目录
    import importlib
    import config_loader
    importlib.reload(config_loader)
    config2 = config_loader.load_config()
    assert Path(config2.get('paths', 'raw_videos')).exists()
    os.chdir(project_root)  # 切回根目录
    print("✓ 通过")
except Exception as e:
    print(f"✗ 失败: {e}")
    os.chdir(project_root)
    all_passed = False

# 测试3: 数据预处理器初始化
print("\n[测试3] 数据预处理器初始化...")
try:
    from data_preprocess import VideoPreprocessor
    preprocessor = VideoPreprocessor()
    assert Path(preprocessor.raw_videos_dir).exists()
    print("✓ 通过")
except Exception as e:
    print(f"✗ 失败: {e}")
    all_passed = False

# 测试4: 视频编辑器初始化（即使FFmpeg不存在也不应崩溃）
print("\n[测试4] 视频编辑器初始化...")
try:
    from video_editor import VideoEditor
    editor = VideoEditor()
    print("✓ 通过")
except Exception as e:
    print(f"✗ 失败: {e}")
    all_passed = False

# 测试5: 环境检查脚本
print("\n[测试5] 环境检查脚本...")
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, str(project_root / 'check_environment.py')],
        capture_output=True,
        text=True,
        timeout=15,
        input='\n'  # 自动按回车
    )
    # 检查是否有严重错误
    if 'Traceback' in result.stdout or 'Traceback' in result.stderr:
        raise Exception("环境检查脚本有错误")
    print("✓ 通过")
except Exception as e:
    print(f"✗ 失败: {e}")
    all_passed = False

# 测试6: GUI启动（不实际显示窗口）
print("\n[测试6] GUI模块导入...")
try:
    from ui import BadmintonVideoEditorGUI
    print("✓ 通过")
except Exception as e:
    print(f"✗ 失败: {e}")
    all_passed = False

# 最终结果
print("\n" + "=" * 70)
if all_passed:
    print("✅ 所有测试通过！所有bug已修复")
    print("\n系统可以正常使用:")
    print("  - 运行GUI: python start_gui.py")
    print("  - 运行命令行: cd 02_code && python main.py")
else:
    print("❌ 部分测试失败，请检查错误信息")
print("=" * 70)
