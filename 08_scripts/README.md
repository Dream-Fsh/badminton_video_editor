# 工具脚本目录

本目录包含项目的各种工具脚本。

## 工具脚本说明

### 基础工具
- `check_environment.py` - 环境检查脚本
- `create_annotation_templates.py` - 创建标注模板
- `merge_annotations.py` - 合并标注文件
- `fix_bugs.py` - Bug修复脚本

### 数据集准备
- `prepare_videobadminton.py` - **VideoBadminton 数据集准备**（I3D 动作识别）
- `finetune_yolo_shuttlecock.py` - **Kaggle 羽毛球检测下载+YOLO 微调**

---

## 使用方法

### 基础工具
```bash
python 08_scripts/check_environment.py
python 08_scripts/create_annotation_templates.py
python 08_scripts/merge_annotations.py
```

### I3D 动作识别 — VideoBadminton 数据集

1. 下载数据集到 `01_data/VideoBadminton/`：
   - GitHub: https://github.com/qilimk/VideoBadminton
   - 种子下载: https://hyper.ai/en/datasets/30582
2. 运行准备脚本：
```bash
# 18分类模式（需修改模型）
python 08_scripts/prepare_videobadminton.py --mode multi

# 2分类模式（兼容现有模型）
python 08_scripts/prepare_videobadminton.py --mode binary
```

### YOLO 羽毛球检测 — Kaggle 数据集微调

1. 配置 Kaggle API（一次性）：
   - 注册 https://www.kaggle.com → Settings → API → Create New Token
   - 将 `kaggle.json` 放到 `%USERPROFILE%\.kaggle\`
2. 安装依赖：`pip install kaggle ultralytics`
3. 一键运行：
```bash
python 08_scripts/finetune_yolo_shuttlecock.py
```
脚本会自动：下载 → 整理 → 训练 → 测试 → 更新 config.yaml
