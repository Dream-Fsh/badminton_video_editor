"""
项目清理和整理脚本
清理不必要的文件和目录，整理项目结构
"""
import os
import shutil
from pathlib import Path

root = Path(__file__).parent

print("=" * 70)
print("项目清理和整理")
print("=" * 70)

# 1. 清理02_code目录下不应该存在的子目录
print("\n[1] 清理02_code目录...")
code_dir = root / '02_code'
unwanted_dirs = ['01_data', '04_output']

for dir_name in unwanted_dirs:
    dir_path = code_dir / dir_name
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
            print(f"  ✓ 删除: 02_code/{dir_name}")
        except Exception as e:
            print(f"  ✗ 删除失败 02_code/{dir_name}: {e}")
    else:
        print(f"  - 02_code/{dir_name} (不存在)")

# 2. 检查并移动02_code中的工具脚本
print("\n[2] 检查02_code中的工具脚本...")
tool_scripts = ['create_annotation_templates.py']

for script in tool_scripts:
    src = code_dir / script
    if src.exists():
        dst = root / '08_scripts' / script
        try:
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            print(f"  ✓ 移动: {script} -> 08_scripts/")
        except Exception as e:
            print(f"  ✗ 移动失败 {script}: {e}")

# 3. 检查根目录的文件
print("\n[3] 检查根目录文件...")
root_files = list(root.glob('*.py'))
root_files += list(root.glob('*.md'))
root_files += list(root.glob('*.txt'))
root_files += list(root.glob('*.bat'))

# 应该保留在根目录的文件
keep_in_root = {
    'start_gui.py',
    'start_gui.bat',
    'main.py',
    'install.bat',
    'README.md',
    'QUICKSTART.md',
    'PROJECT_OVERVIEW.md',
    'requirements.txt',
    'organize_files.py',
    'cleanup_project.py',
}

print(f"\n  根目录文件清单:")
for file in sorted(root_files):
    if file.name in keep_in_root:
        print(f"    ✓ {file.name} (保留)")
    else:
        print(f"    ? {file.name} (可能需要移动)")

# 4. 显示最终的目录结构
print("\n[4] 项目目录结构:")
print("""
badminton_video_editor/
├── 01_data/              # 数据目录
│   ├── raw_videos/       # 原始视频
│   ├── processed_frames/ # 处理后的帧
│   ├── annotations/      # 标注文件
│   └── dataset_split/    # 数据集划分
│
├── 02_code/              # 源代码
│   ├── config_loader.py  # 配置加载器
│   ├── data_preprocess.py # 数据预处理
│   ├── i3d.py            # I3D模型
│   ├── model_train.py    # 模型训练
│   ├── model_predict.py  # 模型预测
│   ├── video_editor.py   # 视频编辑
│   ├── ui.py             # GUI界面
│   ├── main.py           # 命令行主程序
│   ├── roi_selector.py   # ROI选择器
│   └── annotation_helper.py # 标注辅助
│
├── 03_model/             # 模型文件
│   ├── pretrained/       # 预训练模型
│   └── trained/          # 训练后的模型
│
├── 04_output/            # 输出目录
│   ├── clips/            # 剪辑片段
│   ├── predictions/      # 预测结果
│   └── logs/             # 日志文件
│
├── 05_config/            # 配置文件
│   └── config.yaml       # 系统配置
│
├── 06_docs/              # 文档
│   ├── 使用说明.md
│   ├── 技术文档.md
│   ├── 标注指南.md
│   ├── GUI使用指南.md
│   ├── BUG_FIXES.md
│   ├── BUG_FIXES_FINAL.md
│   ├── 修复总结.md
│   ├── 如何运行.md
│   ├── GUI功能说明.md
│   ├── ROI功能说明.md
│   ├── 标注工作流程.md
│   ├── 训练指南.md
│   ├── 项目交付清单.md
│   └── 项目描述_答辩版.md
│
├── 07_tests/             # 测试脚本
│   └── README.md
│
├── 08_scripts/           # 工具脚本
│   ├── check_environment.py
│   ├── create_annotation_templates.py
│   ├── merge_annotations.py
│   └── README.md
│
├── README.md             # 项目说明
├── QUICKSTART.md         # 快速开始
├── PROJECT_OVERVIEW.md   # 项目概览
├── requirements.txt      # 依赖列表
├── start_gui.py          # GUI启动脚本
├── start_gui.bat         # GUI启动脚本(Windows)
├── install.bat           # 安装脚本
└── main.py               # 主程序入口
""")

print("\n" + "=" * 70)
print("清理完成！")
print("=" * 70)

# 5. 生成项目结构文档
structure_doc = root / '06_docs' / '项目结构说明.md'
structure_doc.write_text("""# 项目结构说明

## 目录结构

```
badminton_video_editor/
├── 01_data/              # 数据目录
├── 02_code/              # 源代码
├── 03_model/             # 模型文件
├── 04_output/            # 输出目录
├── 05_config/            # 配置文件
├── 06_docs/              # 文档
├── 07_tests/             # 测试脚本
└── 08_scripts/           # 工具脚本
```

## 各目录说明

### 01_data/ - 数据目录
- `raw_videos/` - 存放原始视频文件
- `processed_frames/` - 存放提取的视频帧
- `annotations/` - 存放标注文件
- `dataset_split/` - 数据集划分信息

### 02_code/ - 源代码
核心功能模块：
- `config_loader.py` - 配置文件加载器
- `data_preprocess.py` - 数据预处理模块
- `i3d.py` - I3D模型定义
- `model_train.py` - 模型训练模块
- `model_predict.py` - 模型预测模块
- `video_editor.py` - 视频编辑模块
- `ui.py` - GUI界面模块
- `main.py` - 命令行主程序
- `roi_selector.py` - ROI区域选择器
- `annotation_helper.py` - 标注辅助工具

### 03_model/ - 模型文件
- `pretrained/` - 预训练模型（rgb_imagenet.pt）
- `trained/` - 训练后的模型文件

### 04_output/ - 输出目录
- `clips/` - 剪辑后的视频片段
- `predictions/` - 预测结果JSON文件
- `logs/` - 运行日志

### 05_config/ - 配置文件
- `config.yaml` - 系统配置文件

### 06_docs/ - 文档
- `使用说明.md` - 详细使用说明
- `技术文档.md` - 技术实现文档
- `标注指南.md` - 数据标注指南
- `GUI使用指南.md` - GUI操作指南
- `BUG_FIXES_FINAL.md` - Bug修复报告
- `修复总结.md` - 修复总结
- `如何运行.md` - 快速运行指南
- 其他文档...

### 07_tests/ - 测试脚本
包含各种测试脚本，用于验证系统功能。

### 08_scripts/ - 工具脚本
- `check_environment.py` - 环境检查工具
- `create_annotation_templates.py` - 创建标注模板
- `merge_annotations.py` - 合并标注文件

## 根目录文件

- `README.md` - 项目说明文档
- `QUICKSTART.md` - 快速开始指南
- `PROJECT_OVERVIEW.md` - 项目概览
- `requirements.txt` - Python依赖列表
- `start_gui.py` - GUI启动脚本
- `start_gui.bat` - Windows批处理启动脚本
- `install.bat` - 依赖安装脚本
- `main.py` - 主程序入口

## 使用建议

### 新用户
1. 阅读 `README.md` 了解项目
2. 阅读 `06_docs/如何运行.md` 快速开始
3. 运行 `python start_gui.py` 启动GUI

### 开发者
1. 阅读 `06_docs/技术文档.md` 了解技术细节
2. 查看 `02_code/` 目录下的源代码
3. 参考 `07_tests/` 中的测试脚本

### 数据标注
1. 阅读 `06_docs/标注指南.md`
2. 使用 `08_scripts/create_annotation_templates.py` 创建模板
3. 完成标注后使用 `08_scripts/merge_annotations.py` 合并
""", encoding='utf-8')

print(f"\n✓ 已生成项目结构文档: 06_docs/项目结构说明.md")
