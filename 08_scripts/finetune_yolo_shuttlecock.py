"""
Kaggle 羽毛球检测数据集下载 + YOLO 微调脚本
============================================
功能：
1. 下载 Kaggle shuttle_badminton_photos 数据集（YOLO格式，已标注）
2. 微调 YOLOv8n 得到羽毛球专用检测器
3. 保存模型供 athlete_detector.py 使用

使用前提：
  1. 安装依赖：pip install kaggle ultralytics
  2. 配置 Kaggle API：
     - 注册 Kaggle 账号 → https://www.kaggle.com
     - 进入 Settings → API → Create New Token
     - 将下载的 kaggle.json 放到 %USERPROFILE%\.kaggle\

数据集地址：
  https://www.kaggle.com/datasets/ayushsinha731/shuttle-badminton-photos

使用方法：
  python 08_scripts/finetune_yolo_shuttlecock.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============ 配置 ============
DATASET_SLUG = "ayushsinha731/shuttle-badminton-photos"
DATA_DIR = PROJECT_ROOT / "01_data" / "shuttlecock_yolo"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "03_model" / "trained"
FINAL_MODEL_NAME = "yolov8n_shuttlecock.pt"

# 训练超参数
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 640
PATIENCE = 10  # 早停轮数


def check_kaggle_cli():
    """检查 Kaggle CLI 是否可用"""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"[OK] Kaggle CLI: {result.stdout.strip()}")
            return True
        else:
            print("[WARN] kaggle 命令执行失败")
            return False
    except FileNotFoundError:
        print("[ERROR] 未安装 kaggle CLI")
        print("  安装命令: pip install kaggle")
        return False
    except Exception as e:
        print(f"[ERROR] 检查 kaggle 失败: {e}")
        return False


def check_kaggle_auth():
    """检查 Kaggle API 认证配置"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_json.exists():
        print("[ERROR] 未找到 Kaggle API 配置文件")
        print(f"  期望位置: {kaggle_json}")
        print("\n配置步骤:")
        print("  1. 访问 https://www.kaggle.com → 登录/注册")
        print("  2. 点击右上角头像 → Settings")
        print("  3. 找到 'API' 部分 → 点击 'Create New Token'")
        print("  4. 将下载的 kaggle.json 放到以下目录:")
        print(f"     {kaggle_dir}")
        return False

    print(f"[OK] Kaggle 配置文件存在: {kaggle_json}")
    return True


def download_dataset():
    """从 Kaggle 下载数据集"""
    print(f"\n{'='*60}")
    print("【步骤1】下载 Kaggle 数据集")
    print(f"{'='*60}")
    print(f"数据集: {DATASET_SLUG}")
    print(f"目标目录: {DATA_DIR}")

    if DATA_DIR.exists():
        # 检查是否已有数据
        yaml_files = list(DATA_DIR.rglob("data.yaml"))
        if yaml_files:
            print(f"[INFO] 数据集已存在，跳过下载")
            print(f"  找到: {yaml_files[0]}")
            return str(yaml_files[0].parent)

    os.makedirs(DATA_DIR, exist_ok=True)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(DATA_DIR),
        "--unzip"
    ]

    print(f"\n执行: {' '.join(cmd)}")
    print("下载中... (如果较慢，可手动下载后解压到目标目录)\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"[OK] 数据集下载完成")
        else:
            print(f"[WARN] 下载可能失败: {result.stderr}")
            # 检查是否有文件
            if any(DATA_DIR.iterdir()):
                print("[INFO] 但目录中已有文件，继续处理...")
            else:
                print(f"\n[ERROR] 下载失败！请手动操作:")
                print(f"  1. 浏览器访问: https://www.kaggle.com/datasets/{DATASET_SLUG}")
                print(f"  2. 点击 'Download' 下载 ZIP")
                print(f"  3. 解压到: {DATA_DIR}")
                return None
    except subprocess.TimeoutExpired:
        print("[WARN] 下载超时，请手动下载")
        return None
    except Exception as e:
        print(f"[ERROR] 下载出错: {e}")
        return None

    # 查找 data.yaml
    yaml_files = list(DATA_DIR.rglob("data.yaml"))
    if yaml_files:
        return str(yaml_files[0].parent)

    # 有些数据集没有 data.yaml，需要查找标注目录
    # 尝试找到 images/ 和 labels/ 目录
    for root, dirs, files in os.walk(DATA_DIR):
        if "images" in dirs and "labels" in dirs:
            print(f"[INFO] 找到 images/labels 目录: {root}")
            return root

    return str(DATA_DIR)


def prepare_dataset(dataset_dir):
    """
    检查并整理数据集目录结构
    YOLOv8 要求的结构:
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml
    """
    print(f"\n{'='*60}")
    print("【步骤2】检查数据集结构")
    print(f"{'='*60}")

    dataset_path = Path(dataset_dir)

    # 情况1: 已经有标准结构
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        data_yaml = dataset_path.parent / "data.yaml"
    if not data_yaml.exists():
        data_yaml = next(dataset_path.rglob("data.yaml"), None)

    if data_yaml and data_yaml.exists():
        print(f"[OK] 找到 data.yaml: {data_yaml}")
        with open(data_yaml, 'r') as f:
            content = f.read()
            print(f"  内容:\n{content}")
        return str(data_yaml)

    # 情况2: 没有标准结构，需要整理
    print("[INFO] 未找到 data.yaml，尝试自动整理...")

    # 查找 images 和 labels 目录
    all_images = []
    all_labels = []

    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        all_images.extend(dataset_path.rglob(f"*{ext}"))
    all_labels.extend(dataset_path.rglob("*.txt"))

    # 排除非标注的 txt 文件
    all_labels = [l for l in all_labels if l.parent.name == 'labels' or l.stem != 'README']

    if not all_images:
        print("[ERROR] 未找到任何图片文件！")
        print(f"  目录内容: {list(dataset_path.iterdir())[:10]}")
        return None

    if not all_labels:
        print("[ERROR] 未找到任何标注文件！")
        return None

    print(f"[INFO] 找到 {len(all_images)} 张图片, {len(all_labels)} 个标注")

    # 创建标准目录结构
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    val_images = dataset_path / "val" / "images"
    val_labels = dataset_path / "val" / "labels"

    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # 按文件名配对图片和标注
    label_stems = {l.stem for l in all_labels}
    paired = [(img, img.with_name(img.stem + '.txt')) for img in all_images if img.stem in label_stems]

    if not paired:
        # 尝试不同的配对逻辑
        paired = []
        for img in all_images:
            for lbl in all_labels:
                if img.stem == lbl.stem:
                    paired.append((img, lbl))
                    break

    print(f"[INFO] 成功配对: {len(paired)} 对")

    # 80/20 划分
    split_idx = int(len(paired) * 0.8)
    train_pairs = paired[:split_idx]
    val_pairs = paired[split_idx:]

    # 复制文件
    import shutil as sh
    for img, lbl in train_pairs:
        sh.copy2(img, train_images / img.name)
        sh.copy2(lbl, train_labels / lbl.name)

    for img, lbl in val_pairs:
        sh.copy2(img, val_images / img.name)
        sh.copy2(lbl, val_labels / lbl.name)

    # 推断类别数量
    classes = set()
    for _, lbl in paired:
        with open(lbl, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))

    num_classes = len(classes)

    # 生成 data.yaml
    yaml_content = f"""path: {dataset_path.as_posix()}
train: train/images
val: val/images

nc: {num_classes}
names: {list(range(num_classes))}
"""
    data_yaml = dataset_path / "data.yaml"
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)

    print(f"[OK] 数据集整理完成")
    print(f"  训练集: {len(train_pairs)} 张")
    print(f"  验证集: {len(val_pairs)} 张")
    print(f"  类别数: {num_classes}")
    print(f"  data.yaml: {data_yaml}")

    return str(data_yaml)


def train_yolo(data_yaml_path):
    """微调 YOLOv8n"""
    print(f"\n{'='*60}")
    print("【步骤3】微调 YOLOv8n 羽毛球检测器")
    print(f"{'='*60}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 未安装 ultralytics")
        print("  安装命令: pip install ultralytics")
        return None

    # 加载预训练模型
    model = YOLO("yolov8n.pt")
    print("[OK] 加载 YOLOv8n 预训练权重")

    # 开始微调
    print(f"\n训练参数:")
    print(f"  data:    {data_yaml_path}")
    print(f"  epochs:  {EPOCHS}")
    print(f"  batch:   {BATCH_SIZE}")
    print(f"  imgsz:   {IMAGE_SIZE}")
    print(f"  patience:{PATIENCE}")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        patience=PATIENCE,
        project=str(MODEL_OUTPUT_DIR),
        name="shuttlecock_finetune",
        exist_ok=True,
        verbose=True,
    )

    # 复制最佳模型到指定位置
    best_weights = Path(MODEL_OUTPUT_DIR) / "shuttlecock_finetune" / "weights" / "best.pt"
    if best_weights.exists():
        final_path = MODEL_OUTPUT_DIR / FINAL_MODEL_NAME
        shutil.copy2(best_weights, final_path)
        print(f"\n[OK] 最佳模型已保存到: {final_path}")
        return str(final_path)
    else:
        # 尝试查找 runs 目录
        runs_dir = PROJECT_ROOT / "runs"
        for d in runs_dir.rglob("best.pt"):
            final_path = MODEL_OUTPUT_DIR / FINAL_MODEL_NAME
            shutil.copy2(d, final_path)
            print(f"\n[OK] 最佳模型已保存到: {final_path}")
            return str(final_path)

    print("[WARN] 未找到训练后的权重文件")
    return None


def test_model(model_path, data_yaml_path):
    """测试微调后的模型"""
    print(f"\n{'='*60}")
    print("【步骤4】测试模型")
    print(f"{'='*60}")

    if not model_path or not os.path.exists(model_path):
        print("[WARN] 模型文件不存在，跳过测试")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        return

    model = YOLO(model_path)

    # 在验证集上测试
    results = model.val(data=data_yaml_path, imgsz=IMAGE_SIZE)
    print(f"\n验证结果:")
    print(f"  mAP50:   {results.box.map50:.4f}")
    print(f"  mAP50-95:{results.box.map:.4f}")


def update_project_config(model_path):
    """更新项目配置，使用新模型"""
    print(f"\n{'='*60}")
    print("【步骤5】更新项目配置")
    print(f"{'='*60}")

    if not model_path:
        return

    config_path = PROJECT_ROOT / "05_config" / "config.yaml"
    if not config_path.exists():
        print(f"[WARN] 配置文件不存在: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换 YOLO 模型路径
    old_model = 'model_type: "yolov8n.pt"'
    # 使用相对路径
    rel_path = os.path.relpath(model_path, PROJECT_ROOT).replace("\\", "/")
    new_model = f'model_type: "{rel_path}"'

    if old_model in content:
        content = content.replace(old_model, new_model)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] 已更新 config.yaml 中的 YOLO 模型路径")
        print(f"  旧: yolov8n.pt (COCO 通用)")
        print(f"  新: {rel_path} (羽毛球专用)")
    else:
        print(f"[INFO] 请手动修改 {config_path}")
        print(f"  将 model_type 改为: \"{rel_path}\"")


def main():
    print("=" * 60)
    print("羽毛球检测 YOLO 微调工具")
    print("=" * 60)

    # 前置检查
    if not check_kaggle_cli():
        print("\n请先安装: pip install kaggle")
        return

    if not check_kaggle_auth():
        return

    # 步骤1: 下载数据集
    dataset_dir = download_dataset()
    if not dataset_dir:
        print("\n请手动下载数据集后重新运行")
        return

    # 步骤2: 整理数据集
    data_yaml = prepare_dataset(dataset_dir)
    if not data_yaml:
        print("\n数据集整理失败")
        return

    # 步骤3: 微调 YOLO
    model_path = train_yolo(data_yaml)
    if not model_path:
        print("\n训练失败")
        return

    # 步骤4: 测试
    test_model(model_path, data_yaml)

    # 步骤5: 更新配置
    update_project_config(model_path)

    print(f"\n{'=' * 60}")
    print("全部完成！")
    print(f"{'=' * 60}")
    print(f"\n微调后的模型: {model_path}")
    print(f"使用方法:")
    print(f"  项目会自动从 config.yaml 读取模型路径")
    print(f"  如需回退: 将 config.yaml 中 model_type 改回 'yolov8n.pt'")


if __name__ == "__main__":
    main()
