"""
文件整理脚本
将项目根目录的文件整理到对应的文件夹
"""
import os
import shutil
from pathlib import Path

# 项目根目录
root = Path(__file__).parent

# 创建目录
dirs_to_create = {
    '07_tests': '测试脚本',
    '08_scripts': '工具脚本',
}

for dir_name, desc in dirs_to_create.items():
    dir_path = root / dir_name
    dir_path.mkdir(exist_ok=True)
    print(f"✓ 创建目录: {dir_name} ({desc})")

# 文件移动规则
file_moves = {
    # 测试脚本 -> 07_tests
    '07_tests': [
        'simple_test.py',
        'full_workflow_test.py',
        'test_fixes.py',
        'test_paths.py',
        'verify_fixes.py',
        'run_test.py',
        'test_result.txt',
    ],
    
    # 工具脚本 -> 08_scripts
    '08_scripts': [
        'check_environment.py',
        'create_annotation_templates.py',
        'merge_annotations.py',
        'fix_bugs.py',
    ],
    
    # 文档 -> 06_docs
    '06_docs': [
        'BUG_FIXES.md',
        'BUG_FIXES_FINAL.md',
        '修复总结.md',
        '如何运行.md',
        'GUI功能说明.md',
        'ROI功能说明.md',
        '标注工作流程.md',
        '训练指南.md',
        '项目交付清单.md',
        '项目描述_答辩版.md',
    ],
}

# 执行移动
print("\n开始整理文件...")
moved_count = 0
skipped_count = 0

for target_dir, files in file_moves.items():
    print(f"\n移动到 {target_dir}:")
    for filename in files:
        src = root / filename
        dst = root / target_dir / filename
        
        if src.exists():
            try:
                # 如果目标文件已存在，先删除
                if dst.exists():
                    dst.unlink()
                
                shutil.move(str(src), str(dst))
                print(f"  ✓ {filename}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ {filename} - 失败: {e}")
                skipped_count += 1
        else:
            print(f"  - {filename} (不存在)")
            skipped_count += 1

# 创建README文件
readme_07 = root / '07_tests' / 'README.md'
readme_07.write_text("""# 测试脚本目录

本目录包含项目的各种测试脚本。

## 测试脚本说明

- `simple_test.py` - 简单功能测试
- `full_workflow_test.py` - 完整工作流程测试
- `test_fixes.py` - Bug修复测试
- `test_paths.py` - 路径配置测试
- `verify_fixes.py` - 验证所有修复
- `run_test.py` - 运行测试套件
- `test_result.txt` - 测试结果输出

## 运行测试

```bash
# 简单测试
python 07_tests/simple_test.py

# 完整测试
python 07_tests/full_workflow_test.py

# 验证修复
python 07_tests/verify_fixes.py
```
""", encoding='utf-8')

readme_08 = root / '08_scripts' / 'README.md'
readme_08.write_text("""# 工具脚本目录

本目录包含项目的各种工具脚本。

## 工具脚本说明

- `check_environment.py` - 环境检查脚本
- `create_annotation_templates.py` - 创建标注模板
- `merge_annotations.py` - 合并标注文件
- `fix_bugs.py` - Bug修复脚本

## 使用方法

```bash
# 检查环境
python 08_scripts/check_environment.py

# 创建标注模板
python 08_scripts/create_annotation_templates.py

# 合并标注
python 08_scripts/merge_annotations.py
```
""", encoding='utf-8')

print("\n" + "=" * 60)
print(f"文件整理完成！")
print(f"  移动成功: {moved_count} 个文件")
print(f"  跳过: {skipped_count} 个文件")
print("=" * 60)

# 显示整理后的目录结构
print("\n整理后的目录结构:")
print("├── 06_docs/          # 文档")
print("├── 07_tests/         # 测试脚本")
print("├── 08_scripts/       # 工具脚本")
print("├── README.md         # 项目说明")
print("├── QUICKSTART.md     # 快速开始")
print("├── PROJECT_OVERVIEW.md  # 项目概览")
print("├── requirements.txt  # 依赖列表")
print("├── start_gui.py      # GUI启动")
print("├── start_gui.bat     # GUI启动(Windows)")
print("├── install.bat       # 安装脚本")
print("└── main.py           # 主程序")
