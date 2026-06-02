"""
最终项目状态检查
验证所有修复和整理工作
"""
import os
import sys
from pathlib import Path

# 设置项目根目录
project_root = Path(__file__).parent
os.chdir(project_root)

print("=" * 70)
print("羽毛球视频自动剪辑系统 - 最终状态检查")
print("=" * 70)

all_passed = True
issues = []

# 1. 检查目录结构
print("\n[1] 检查目录结构...")
required_dirs = {
    '01_data': '数据目录',
    '02_code': '源代码',
    '03_model': '模型文件',
    '04_output': '输出目录',
    '05_config': '配置文件',
    '06_docs': '文档',
    '07_tests': '测试脚本',
    '08_scripts': '工具脚本',
}

for dir_name, desc in required_dirs.items():
    dir_path = project_root / dir_name
    if dir_path.exists():
        print(f"  ✓ {dir_name}/ - {desc}")
    else:
        print(f"  ✗ {dir_name}/ - 不存在")
        issues.append(f"缺少目录: {dir_name}")
        all_passed = False

# 2. 检查核心文件
print("\n[2] 检查核心文件...")
core_files = {
    'README.md': '项目说明',
    'QUICKSTART.md': '快速开始',
    'requirements.txt': '依赖列表',
    'start_gui.py': 'GUI启动脚本',
    '05_config/config.yaml': '配置文件',
}

for file_path, desc in core_files.items():
    full_path = project_root / file_path
    if full_path.exists():
        print(f"  ✓ {file_path} - {desc}")
    else:
        print(f"  ✗ {file_path} - 不存在")
        issues.append(f"缺少文件: {file_path}")
        all_passed = False

# 3. 检查源代码模块
print("\n[3] 检查源代码模块...")
code_modules = [
    'config_loader.py',
    'data_preprocess.py',
    'i3d.py',
    'model_train.py',
    'model_predict.py',
    'video_editor.py',
    'ui.py',
    'main.py',
]

code_dir = project_root / '02_code'
for module in code_modules:
    module_path = code_dir / module
    if module_path.exists():
        print(f"  ✓ {module}")
    else:
        print(f"  ✗ {module} - 不存在")
        issues.append(f"缺少模块: 02_code/{module}")
        all_passed = False

# 4. 检查文档
print("\n[4] 检查文档...")
docs_dir = project_root / '06_docs'
doc_count = len(list(docs_dir.glob('*.md'))) if docs_dir.exists() else 0
print(f"  ✓ 找到 {doc_count} 个文档文件")

if doc_count < 10:
    issues.append(f"文档数量偏少: {doc_count}")

# 5. 检查工具脚本
print("\n[5] 检查工具脚本...")
scripts_dir = project_root / '08_scripts'
if scripts_dir.exists():
    scripts = list(scripts_dir.glob('*.py'))
    print(f"  ✓ 找到 {len(scripts)} 个工具脚本")
    for script in scripts:
        print(f"    - {script.name}")
else:
    issues.append("工具脚本目录不存在")
    all_passed = False

# 6. 检查数据
print("\n[6] 检查数据...")
videos_dir = project_root / '01_data' / 'raw_videos'
if videos_dir.exists():
    videos = list(videos_dir.glob('*.mp4')) + list(videos_dir.glob('*.avi'))
    print(f"  ✓ 找到 {len(videos)} 个视频文件")
else:
    print(f"  ⚠ 视频目录不存在")

annotations_dir = project_root / '01_data' / 'annotations'
if annotations_dir.exists():
    annotations = list(annotations_dir.glob('*_annotations.json'))
    print(f"  ✓ 找到 {len(annotations)} 个标注文件")
    
    merged = annotations_dir / 'merged_labels.json'
    if merged.exists():
        print(f"  ✓ 合并标注文件存在")
    else:
        print(f"  ⚠ 合并标注文件不存在")
else:
    print(f"  ⚠ 标注目录不存在")

# 7. 检查预训练模型
print("\n[7] 检查预训练模型...")
model_path = project_root / '03_model' / 'pretrained' / 'rgb_imagenet.pt'
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ 预训练模型存在 ({size_mb:.2f} MB)")
else:
    print(f"  ⚠ 预训练模型不存在")

# 8. 测试配置加载
print("\n[8] 测试配置加载...")
sys.path.insert(0, str(project_root / '02_code'))
try:
    from config_loader import load_config
    config = load_config()
    print(f"  ✓ 配置加载成功")
except Exception as e:
    print(f"  ✗ 配置加载失败: {e}")
    issues.append(f"配置加载失败: {e}")
    all_passed = False

# 9. 检查02_code目录清洁度
print("\n[9] 检查02_code目录清洁度...")
code_dir = project_root / '02_code'
unwanted_items = ['01_data', '04_output', '__pycache__']
clean = True
for item in unwanted_items:
    item_path = code_dir / item
    if item_path.exists() and item != '__pycache__':
        print(f"  ⚠ 发现不应存在的目录: {item}")
        clean = False

if clean:
    print(f"  ✓ 02_code目录结构清洁")

# 10. 检查根目录清洁度
print("\n[10] 检查根目录清洁度...")
root_files = list(project_root.glob('*.py')) + list(project_root.glob('*.md'))
expected_root_files = {
    'start_gui.py',
    'main.py',
    'organize_files.py',
    'cleanup_project.py',
    'final_check.py',
    'README.md',
    'QUICKSTART.md',
    'PROJECT_OVERVIEW.md',
}

unexpected = [f.name for f in root_files if f.name not in expected_root_files]
if unexpected:
    print(f"  ⚠ 根目录有 {len(unexpected)} 个可能需要整理的文件:")
    for f in unexpected[:5]:
        print(f"    - {f}")
else:
    print(f"  ✓ 根目录文件整洁")

# 最终结果
print("\n" + "=" * 70)
print("检查结果")
print("=" * 70)

if all_passed and len(issues) == 0:
    print("\n✅ 所有检查通过！项目状态良好")
    print("\n系统已准备就绪，可以开始使用:")
    print("  1. 运行GUI: python start_gui.py")
    print("  2. 或运行命令行: cd 02_code && python main.py")
    print("  3. 查看文档: 06_docs/如何运行.md")
else:
    print(f"\n⚠️ 发现 {len(issues)} 个问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\n建议:")
    print("  - 检查缺失的文件和目录")
    print("  - 参考 06_docs/项目结构说明.md")

print("\n" + "=" * 70)

# 生成状态报告
status_file = project_root / '项目状态.txt'
with open(status_file, 'w', encoding='utf-8') as f:
    f.write("羽毛球视频自动剪辑系统 - 项目状态报告\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"检查时间: {Path(__file__).stat().st_mtime}\n\n")
    
    f.write("目录结构:\n")
    for dir_name, desc in required_dirs.items():
        status = "✓" if (project_root / dir_name).exists() else "✗"
        f.write(f"  {status} {dir_name}/ - {desc}\n")
    
    f.write("\n核心文件:\n")
    for file_path, desc in core_files.items():
        status = "✓" if (project_root / file_path).exists() else "✗"
        f.write(f"  {status} {file_path} - {desc}\n")
    
    if issues:
        f.write("\n发现的问题:\n")
        for i, issue in enumerate(issues, 1):
            f.write(f"  {i}. {issue}\n")
    else:
        f.write("\n✅ 所有检查通过！\n")

print(f"\n✓ 状态报告已保存到: 项目状态.txt")
