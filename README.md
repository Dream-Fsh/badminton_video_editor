# 羽毛球视频智能剪辑系统

基于深度学习的羽毛球比赛视频关键动作识别与自动剪辑系统

---

## 项目简介

本项目利用深度学习与计算机视觉技术，以 **I3D 3D卷积神经网络** 为核心实现动作识别，结合 **YOLOv8** 进行运动员和羽毛球检测，使用 OpenCV 与 FFmpeg 完成视频处理，自动识别羽毛球比赛视频中 **"发球"** 和 **"球落地"** 等关键动作，实现智能回合分割与自动剪辑。

### 核心功能

#### 🎯 智能检测与剪辑
- **I3D深度学习动作识别**：精准识别发球和落地动作
- **YOLOv8运动员检测**：实时检测运动员位置和羽毛球状态
- **智能回合分割**：自动分割比赛回合，支持单打/双打模式
- **运动员身份追踪**：追踪主运动员，过滤无效回合
- **左右站位检测**：智能识别比赛开始时机

#### 🌐 现代化Web界面
- **响应式设计**：基于Flask的现代化Web前端
- **视频上传**：支持MP4、AVI、MOV、MKV、WMV格式（最大2GB）
- **实时进度显示**：异步检测，实时显示处理进度
- **回合播放器**：在线预览检测到的回合
- **视频流式播放**：支持大视频拖动进度条
- **片段下载**：一键下载指定回合的视频片段

#### 👥 完善的用户系统
- **用户注册/登录**：安全的用户认证系统
- **忘记密码**：邮箱验证码重置密码
- **管理员后台**：用户管理、操作日志、系统统计
- **操作日志**：记录所有用户操作
- **用户统计**：上传和处理视频的统计数据

#### ⚙️ 灵活的配置系统
- **高速模式**：优化的滑动窗口步长，提升检测速度
- **ROI支持**：可配置的感兴趣区域裁剪
- **多约束规则**：可调节的回合分割参数
- **YOLO约束**：可配置的运动员检测约束

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch, I3D, YOLOv8 |
| 计算机视觉 | OpenCV, NumPy |
| 视频处理 | FFmpeg |
| Web框架 | Flask, Werkzeug |
| 数据库 | SQLite |
| 邮件服务 | SMTP (smtplib) |
| 前端 | HTML5, CSS3, JavaScript, AJAX |
| 配置管理 | PyYAML |

---

## 项目结构

```
badminton_video_editor/
├── 01_data/                  # 数据目录
│   ├── raw_videos/           # 原始视频文件
│   ├── processed_frames/     # 处理后的视频帧
│   └── annotations/         # 标注数据
├── 02_code/                  # 核心源代码
│   ├── i3d.py               # I3D模型定义
│   ├── model_train.py        # 模型训练模块
│   ├── model_predict.py      # 模型预测模块（标准版）
│   ├── model_predict_optimized.py  # 模型预测模块（优化版）
│   ├── athlete_detector.py   # 运动员检测与追踪模块
│   ├── video_editor.py       # 视频剪辑模块
│   ├── data_preprocess.py    # 数据预处理模块
│   ├── annotation_helper.py  # 标注辅助工具
│   ├── web_app.py            # Web应用主程序
│   ├── auth.py               # 用户认证与数据库模块
│   ├── email_service.py      # 邮件服务模块
│   └── config_loader.py      # 配置加载模块
├── 03_model/                 # 模型文件
│   ├── pretrained/           # 预训练权重（ImageNet、Kinetics）
│   └── trained/              # 训练好的模型权重
├── 04_output/                # 输出结果
│   ├── clips/                # 剪辑好的视频片段
│   ├── predictions/          # 预测结果（JSON）
│   └── logs/                 # 处理日志
├── 05_config/                # 配置文件
│   └── config.yaml           # 主配置文件
├── 06_docs/                  # 文档
├── 07_tests/                 # 测试脚本
├── 08_scripts/               # 工具脚本
├── config/                   # 额外配置
│   └── email_config.json     # 邮件服务配置
├── data/                     # 数据库
│   └── users.db              # 用户数据库
├── templates/                # Web前端模板
│   ├── base.html             # 基础模板
│   ├── login.html            # 登录页面
│   ├── register.html         # 注册页面
│   ├── forgot_password.html  # 忘记密码页面
│   ├── upload.html           # 视频上传页面
│   ├── detection.html        # 智能检测页面
│   ├── player.html           # 回合播放器页面
│   └── admin.html            # 管理员后台页面
├── static/                   # 静态资源
│   └── style.css             # 样式文件
├── main.py                   # CLI主程序入口
├── run_web.py                # Web应用启动脚本
├── requirements.txt          # Python依赖包
└── README.md                 # 项目说明文档
```

---

## 快速开始

### 1. 环境要求

- **Python**: 3.8+
- **PyTorch**: 1.8+ (支持CUDA加速)
- **FFmpeg**: 4.0+
- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository_url>
cd badminton_video_editor

# 创建虚拟环境（推荐）
python -m venv .venv

# 激活虚拟环境
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 安装Python依赖
pip install -r requirements.txt

# 安装FFmpeg
# Ubuntu: sudo apt-get install ffmpeg
# Mac: brew install ffmpeg
# Windows: 下载 https://ffmpeg.org/download.html，添加到PATH
```

### 3. 配置邮件服务（可选）

如果需要使用忘记密码功能，请配置 `config/email_config.json`：

```json
{
  "enabled": true,
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "use_tls": true,
  "sender_email": "your_email@gmail.com",
  "sender_password": "your_app_password",
  "sender_name": "羽毛球视频剪辑系统"
}
```

**注意**：也可以使用环境变量 `EMAIL_SENDER` 和 `EMAIL_PASSWORD` 配置邮件服务。

### 4. 启动系统

**Web界面（推荐）**

```bash
python run_web.py
# 访问 http://localhost:5000
# 默认管理员账号: admin / admin123
```

**命令行界面**

```bash
python main.py
```

---

## 使用流程

### Web界面使用流程

```
1. 注册/登录系统
   └─ 新用户注册 → 登录
   └─ 忘记密码 → 邮箱验证 → 重置密码

2. 上传视频
   └─ 选择视频文件（支持拖拽上传）
   └─ 设置比赛类型（单打/双打）
   └─ 可选：设置ROI区域

3. 智能检测
   └─ 点击"开始检测"按钮
   └─ 实时查看检测进度
   └─ 系统自动识别回合

4. 查看结果
   └─ 在回合播放器中预览
   └─ 查看检测到的回合列表
   └─ 播放指定回合

5. 下载片段
   └─ 选择需要下载的回合
   └─ 一键下载视频片段
```

### 命令行使用流程

```bash
# 1. 数据预处理（视频拆帧、标注验证）
python 02_code/data_preprocess.py

# 2. 模型训练（I3D迁移学习）
python 02_code/model_train.py

# 3. 模型预测（滑动窗口推理）
python 02_code/model_predict_optimized.py

# 4. 视频剪辑（FFmpeg批量导出）
python 02_code/video_editor.py
```

---

## 核心算法

### I3D 动作识别

I3D（Inflated 3D ConvNet）基于Inception-v1架构，通过3D卷积同时处理时间和空间维度，专门用于视频动作识别。

- **输入**：16帧 × 224×224 RGB图像（可配置）
- **输出**：2类（发球 round_start / 球落地 round_end）
- **迁移学习**：使用ImageNet和Kinetics预训练权重
- **滑动窗口**：使用滑动窗口进行视频推理，支持自定义步长

### YOLOv8 运动员检测

基于YOLOv8检测运动员位置和羽毛球状态：

- **运动员检测**：YOLOv8n模型，实时检测运动员位置
- **运动员追踪**：SimplePersonTracker，跨帧追踪运动员
- **羽毛球检测**：微调的YOLOv8模型，检测羽毛球位置
- **羽毛球状态分析**：ShuttlecockTracker，分析球的运动状态
- **左右站位检测**：识别运动员是否处于比赛准备姿势

### 智能回合分割算法

结合I3D和YOLOv8的检测结果，使用多约束规则进行回合分割：

- **时间约束**：最小/最大回合时长
- **间隔约束**：回合间最小间隔，自动合并短间隔回合
- **运动员约束**：回合期间必须有指定数量的运动员在场上
- **场地约束**：运动员必须在绿色比赛场地内
- **优先级规则**：I3D检测结果优先，运动停止作为后备判定

---

## 配置说明

### 主配置文件：`05_config/config.yaml`

#### 训练配置

```yaml
training:
  batch_size: 1               # 批次大小（显存不足时设为1）
  epochs: 30                  # 训练轮数
  learning_rate: 0.001        # 学习率
  sequence_length: 16         # I3D输入帧数
  num_classes: 2              # 分类数量
  train_split: 0.8            # 训练集比例
  num_workers: 0              # 数据加载线程数（Windows设为0）
```

#### 预测配置

```yaml
prediction:
  confidence_threshold: 0.1   # 置信度阈值
  sliding_window_stride: 16   # 滑动窗口步长（越大越快）
  batch_size: 1               # 预测批次大小
  
  # YOLO运动员检测配置
  athlete_detection:
    enabled: true             # 是否启用YOLO辅助识别
    model_type: "yolov8n.pt" # YOLO模型版本
    person_class_id: 0        # COCO数据集中人的类别ID
    
    # 羽毛球检测配置
    shuttlecock_detection:
      enabled: true
      model_path: "runs/detect/03_model/shuttlecock_yolov8n/weights/best.pt"
      min_confidence: 0.3
      sample_rate: 3          # 每3帧检测一次球
      still_duration: 0.5     # 球静止多少秒判定为落地
      vanish_duration: 1.5    # 球连续消失多少秒判定为下落
    
    static_threshold: 15.0    # 位移阈值（像素）
    window_size: 10           # 判定静止的时间窗口（帧数）
    sample_rate: 2            # 每2帧检测一次
  
  # 运动员身份追踪配置
  athlete_identification:
    enabled: true
    max_detect_count: 4
    track_max_distance: 250    # 追踪匹配最大距离（像素）
    track_min_appearances: 5   # 成为主运动员的最低出现次数
  
  # 回合分割约束
  round_constraints:
    min_round_duration: 2.0    # 最小时长（秒）
    max_round_duration: 30.0   # 最大时长（秒）
    min_start_interval: 3.0    # 回合间最小间隔（秒）
    merge_interval: 2.0        # 合并间隔（秒）
    
    # 左右站位检测
    ready_stance_detection:
      enabled: true
      min_x_separation: 50.0   # 两球员最小间距（像素）
      min_static_duration: 0.3 # 需要持续静止的最短时间（秒）
      allow_minor_motion: true  # 允许轻微运动
```

#### 系统配置

```yaml
system:
  device: "auto"              # 自动选择设备（CUDA/CPU）
  seed: 42                   # 随机种子
  log_level: "INFO"          # 日志级别
```

#### 路径配置

```yaml
paths:
  raw_videos: "01_data/raw_videos"
  processed_frames: "01_data/processed_frames"
  annotations: "01_data/annotations"
  pretrained_weights: "03_model/pretrained/rgb_imagenet.pt"
  trained_models: "03_model/trained"
  output_clips: "04_output/clips"
  output_logs: "04_output/logs"
  output_predictions: "04_output/predictions"
```

### 邮件配置：`config/email_config.json`

```json
{
  "enabled": false,
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "use_tls": true,
  "sender_email": "",
  "sender_password": "",
  "sender_name": "羽毛球视频剪辑系统",
  "use_env_vars": true
}
```

**环境变量配置**（优先级更高）：

```bash
export EMAIL_SENDER="your_email@gmail.com"
export EMAIL_PASSWORD="your_app_password"
```

---

## 文档

- [使用说明](06_docs/使用说明.md) - 详细使用指南
- [技术文档](06_docs/技术文档.md) - 技术实现细节
- [标注指南](06_docs/标注指南.md) - 数据标注规范
- [训练指南](06_docs/训练指南.md) - 模型训练指南

---

## 性能指标

| 指标 | 数值 |
|------|------|
| 训练集准确率 | ~95% |
| 验证集准确率 | ~90% |
| 推理速度（CPU） | ~2 FPS |
| 推理速度（GPU） | ~10 FPS |
| 检测加速（优化版） | 4-5倍（stride=8） |
| 剪辑效率提升 | 10倍 |

---

## 常见问题

### 1. FFmpeg未安装或不可用

**错误信息**：`FFmpeg未安装或不可用`

**解决方案**：
- 安装FFmpeg并添加到系统PATH
- Windows用户可下载：https://ffmpeg.org/download.html
- Ubuntu用户：`sudo apt-get install ffmpeg`

### 2. 模型权重未找到

**错误信息**：`未找到训练好的模型权重，请先进行模型训练`

**解决方案**：
- 运行 `python 02_code/model_train.py` 训练模型
- 或手动将预训练权重放到 `03_model/trained/` 目录

### 3. 忘记密码功能不可用

**原因**：邮件服务未配置

**解决方案**：
- 配置 `config/email_config.json`
- 或使用环境变量配置邮件服务
- 开发模式下，验证码会直接显示在页面上

### 4. 视频上传失败

**可能原因**：
- 视频格式不支持（仅支持MP4、AVI、MOV、MKV、WMV）
- 视频大小超过2GB限制
- FFmpeg未安装（视频信息读取失败）

---

## 开发计划

- [ ] 支持更多动作识别（如扣杀、吊球等）
- [ ] 增加批量视频处理功能
- [ ] 优化Web界面UI/UX
- [ ] 添加视频剪辑预览功能
- [ ] 支持分布式处理（多GPU）
- [ ] 添加API接口文档
- [ ] 支持Docker部署

---

## 参考资料

- Carreira, J., & Zisserman, A. (2017). Quo vadis, action recognition? a new model and the kinetics dataset.
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Flask Documentation: https://flask.palletsprojects.com/
- PyTorch Documentation: https://pytorch.org/docs/

---

## 许可证

本项目仅用于学术研究和教育目的。

---

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件至：[您的邮箱]

---

**最后更新时间**：2026年5月
