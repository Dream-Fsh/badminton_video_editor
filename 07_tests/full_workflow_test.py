"""
完整工作流程测试
模拟用户实际使用场景，发现潜在bug
"""
import os
import sys
import json
from pathlib import Path

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 设置项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, os.path.join(project_root, '02_code'))

def safe_print(msg):
    """安全打印函数，处理Windows编码问题"""
    try:
        print(msg)
    except UnicodeEncodeError:
        # 如果打印失败，尝试替换特殊字符
        safe_msg = msg.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('⚠', '[WARN]')
        safe_msg = safe_msg.replace('🎓', '[TRAIN]').replace('🎬', '[VIDEO]')
        try:
            print(safe_msg)
        except:
            # 最后的退路：仅打印ASCII字符
            print(msg.encode('ascii', 'replace').decode('ascii'))

print("=" * 70)
print("羽毛球视频自动剪辑系统 - 完整工作流程测试")
print("=" * 70)

bugs_found = []

# 测试1: 检查数据文件
safe_print("\n[测试1] 检查数据文件...")
try:
    videos_dir = Path('01_data/raw_videos')
    videos = list(videos_dir.glob('*.mp4')) + list(videos_dir.glob('*.avi'))
    safe_print(f"✓ 找到 {len(videos)} 个视频文件")
    
    if len(videos) == 0:
        bugs_found.append("警告: 没有视频文件可供处理")
    else:
        # 检查第一个视频是否可读
        test_video = videos[0]
        safe_print(f"  测试视频: {test_video.name}")
        
        import cv2
        cap = cv2.VideoCapture(str(test_video))
        if not cap.isOpened():
            bugs_found.append(f"Bug: 无法打开视频文件 - {test_video.name}")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            safe_print(f"  ✓ 视频可读: {fps:.2f} FPS, {frames} 帧")
            cap.release()
            
except Exception as e:
    bugs_found.append(f"Bug: 检查视频文件失败 - {e}")
    safe_print(f"✗ 失败: {e}")

# 测试2: 检查标注文件
safe_print("\n[测试2] 检查标注文件...")
try:
    annotations_dir = Path('01_data/annotations')
    annotations = list(annotations_dir.glob('*_annotations.json'))
    safe_print(f"✓ 找到 {len(annotations)} 个标注文件")
    
    if len(annotations) > 0:
        # 验证标注文件格式
        sample = annotations[0]
        with open(sample, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查必需字段
        required_keys = ['video_name', 'annotations']
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            bugs_found.append(f"Bug: 标注文件缺少字段 {missing_keys} - {sample.name}")
        else:
            safe_print(f"  ✓ 标注文件格式正确")
            
            # 检查标注内容
            if len(data['annotations']) == 0:
                bugs_found.append(f"警告: 标注文件为空 - {sample.name}")
            else:
                safe_print(f"  ✓ 标注数量: {len(data['annotations'])}")
                
except Exception as e:
    bugs_found.append(f"Bug: 检查标注文件失败 - {e}")
    safe_print(f"✗ 失败: {e}")

# 测试3: 测试数据预处理（不实际执行，只检查逻辑）
safe_print("\n[测试3] 测试数据预处理逻辑...")
try:
    from data_preprocess import VideoPreprocessor
    preprocessor = VideoPreprocessor()
    
    # 检查输出目录是否可写
    test_file = Path(preprocessor.frames_dir) / 'test.txt'
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text('test')
    test_file.unlink()
    safe_print("✓ 输出目录可写")
    
except Exception as e:
    bugs_found.append(f"Bug: 数据预处理检查失败 - {e}")
    safe_print(f"✗ 失败: {e}")
    print(f"✗ 失败: {e}")

# 测试4: 检查合并标注文件
print("\n[测试4] 检查合并标注文件...")
try:
    merged_labels_path = Path('01_data/annotations/merged_labels.json')
    if merged_labels_path.exists():
        with open(merged_labels_path, 'r', encoding='utf-8') as f:
            merged_data = json.load(f)
        
        if isinstance(merged_data, list):
            print(f"✓ 合并标注文件存在，包含 {len(merged_data)} 条标注")
        else:
            bugs_found.append("Bug: 合并标注文件格式错误（应为列表）")
    else:
        print("⚠ 合并标注文件不存在（需要先运行合并操作）")
        
except Exception as e:
    bugs_found.append(f"Bug: 检查合并标注失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试5: 检查预训练模型
print("\n[测试5] 检查预训练模型...")
try:
    model_path = Path('03_model/pretrained/rgb_imagenet.pt')
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ 预训练模型存在: {size_mb:.2f} MB")
        
        # 尝试加载模型
        import torch
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            print(f"  ✓ 模型可加载")
        except Exception as e:
            bugs_found.append(f"Bug: 预训练模型无法加载 - {e}")
    else:
        bugs_found.append("警告: 预训练模型不存在（训练需要）")
        
except Exception as e:
    bugs_found.append(f"Bug: 检查预训练模型失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试6: 检查输出目录
print("\n[测试6] 检查输出目录...")
try:
    output_dirs = [
        '04_output/clips',
        '04_output/predictions',
        '04_output/logs'
    ]
    
    for dir_path in output_dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            bugs_found.append(f"Bug: 无法创建目录 - {dir_path}")
            
except Exception as e:
    bugs_found.append(f"Bug: 检查输出目录失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试7: 测试视频编辑器（FFmpeg）
print("\n[测试7] 测试FFmpeg功能...")
try:
    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        version_line = result.stdout.split('\n')[0]
        print(f"✓ FFmpeg可用: {version_line[:50]}...")
    else:
        bugs_found.append("Bug: FFmpeg不可用")
        
except Exception as e:
    bugs_found.append(f"Bug: FFmpeg测试失败 - {e}")
    print(f"✗ 失败: {e}")

# 测试8: 检查配置文件参数
print("\n[测试8] 检查配置文件参数...")
try:
    from config_loader import load_config
    config = load_config()
    
    # 检查关键参数
    batch_size = config.get('training', 'batch_size')
    epochs = config.get('training', 'epochs')
    lr = config.get('training', 'learning_rate')
    
    print(f"  训练参数: batch_size={batch_size}, epochs={epochs}, lr={lr}")
    
    # 检查参数合理性
    if batch_size > 4:
        bugs_found.append(f"警告: batch_size={batch_size} 可能过大，建议设为1-2")
    
    if epochs > 50:
        bugs_found.append(f"警告: epochs={epochs} 可能过多，建议设为10-20")
        
    print("✓ 配置参数检查完成")
    
except Exception as e:
    bugs_found.append(f"Bug: 检查配置参数失败 - {e}")
    print(f"✗ 失败: {e}")

# 输出结果
print("\n" + "=" * 70)
print("测试结果汇总")
print("=" * 70)

if bugs_found:
    print(f"\n发现 {len(bugs_found)} 个问题:\n")
    for i, bug in enumerate(bugs_found, 1):
        print(f"{i}. {bug}")
    
    # 保存bug报告
    with open('test_report.txt', 'w', encoding='utf-8') as f:
        f.write("羽毛球视频自动剪辑系统 - 测试报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"发现 {len(bugs_found)} 个问题:\n\n")
        for i, bug in enumerate(bugs_found, 1):
            f.write(f"{i}. {bug}\n")
    print("\n✓ 测试报告已保存到 test_report.txt")
else:
    print("\n✓ 所有测试通过！系统可以正常使用")

print("\n" + "=" * 70)
