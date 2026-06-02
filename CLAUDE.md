# CLAUDE.md - 项目约束与规范

## 语言要求
- 所有回复必须使用**中文**
- 代码注释使用中文
- Git commit message 使用中文

## 项目概述
羽毛球视频智能剪辑系统：基于 I3D 深度学习模型识别发球/落地动作，自动分割回合并剪辑视频。

## 技术栈
- Python 3.12+
- Flask (Web 前端)
- PyTorch (I3D 模型)
- OpenCV / FFmpeg (视频处理)
- YOLOv8 (运动员检测)
- YAML 配置管理
- Tkinter (GUI界面)

## 项目结构
```
01_data/       # 数据（视频、帧、标注）
02_code/       # 核心源代码
03_model/      # 训练好的模型
04_output/     # 输出结果
05_config/     # YAML 配置文件
06_docs/       # 文档
07_tests/      # 测试
08_scripts/    # 脚本工具
templates/     # Web前端HTML模板
static/        # 静态资源文件
```

## 编码规范
- Web 入口：`web_app.py`，CLI 入口：`main.py`/`02_code/main.py`
- 核心代码放在 `02_code/` 目录下
- 配置文件使用 YAML 格式，放在 `05_config/`
- 测试文件放在 `07_tests/`
- Web模板放在 `templates/`，静态资源放在 `static/`
- 遵循 PEP 8 风格
- 不要随意修改模型文件（`03_model/`）和数据文件（`01_data/`）

## 启动方式
1. **Web界面**: `python web_app.py` (访问 http://localhost:5000)
2. **CLI界面**: `python main.py` 或 `cd 02_code && python main.py`
3. **环境检查**: `python 08_scripts/check_environment.py`
