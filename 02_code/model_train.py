"""
模型训练模块
功能：
1. 加载预处理后的帧数据和标注
2. 使用I3D模型进行迁移学习
3. 训练模型识别发球和球落地动作
4. 保存训练好的模型权重
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import cv2
import numpy as np
from datetime import datetime

# 兼容直接运行和模块导入
try:
    from config_loader import load_config
except ImportError:
    from .config_loader import load_config

# 加载配置
config = load_config()

# -------------------------- 配置参数 --------------------------
# 路径配置
FRAME_DIR: str = config.get('paths', 'processed_frames', default='')
LABEL_PATH: str = config.get('paths', 'merged_labels', default='')
PRETRAINED_WEIGHTS: str = config.get('paths', 'pretrained_weights', default='')
SAVE_DIR: str = config.get('paths', 'trained_models', default='')

# 训练参数
BATCH_SIZE: int = config.get('training', 'batch_size', default=2)
EPOCHS: int = config.get('training', 'epochs', default=20)
LEARNING_RATE: float = config.get('training', 'learning_rate', default=0.0001)
SEQUENCE_LENGTH: int = config.get('training', 'sequence_length', default=32)
# 模型输入尺寸
# 原设计：I3D 标准输入为 224x224（需要固定尺寸的AvgPool）
# 修改后：使用AdaptiveAvgPool，支持任意输入尺寸
# 设为None表示使用原始帧尺寸（ROI裁剪后的950×720）
MODEL_INPUT_SIZE = (224, 224)  # I3D标准输入尺寸（4GB GPU必须缩小，原始950x720会OOM）
NUM_CLASSES: int = config.get('training', 'num_classes', default=2)
TRAIN_SPLIT: float = config.get('training', 'train_split', default=0.8)
NUM_WORKERS: int = config.get('training', 'num_workers', default=0)
WEIGHT_DECAY: float = config.get('training', 'weight_decay', default=0.0001)

# 设备配置
device_config: str = config.get('system', 'device', default='auto')
if device_config == 'auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_config)
DEVICE = device

print(f"Using device: {DEVICE}")


# -------------------------- 数据集类 --------------------------
class BadmintonDataset(Dataset):
    def __init__(self, frame_dir, label_data, sequence_length=32, transform=None, is_train=True):
        """
        羽毛球动作数据集：加载帧序列和对应标签
        
        参数:
            frame_dir: 帧目录
            label_data: 标注数据
            sequence_length: 序列长度
            transform: 转换操作
            is_train: 是否为训练模式（决定是否应用数据增强）
        """
        self.frame_dir = frame_dir
        self.label_data = label_data
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_train = is_train
        
        # 加载数据增强配置（仅在训练模式）
        if self.is_train:
            aug_config = config.get('training', 'augmentation', default={})
            self.aug_enabled = True
            self.hflip_prob = aug_config.get('horizontal_flip', 0.5)
            self.random_crop = aug_config.get('random_crop', True)
            self.crop_scale = aug_config.get('crop_scale', [0.95, 1.0])
            self.color_jitter = aug_config.get('color_jitter', True)
            self.brightness = aug_config.get('brightness', 0.2)
            self.contrast = aug_config.get('contrast', 0.2)
            self.saturation = aug_config.get('saturation', 0.2)
            self.hue = aug_config.get('hue', 0.1)
            self.random_rotate = aug_config.get('random_rotate', True)
            self.rotate_degree = aug_config.get('rotate_degree', 5)
            self.gaussian_blur = aug_config.get('gaussian_blur', True)
            self.blur_prob = aug_config.get('blur_prob', 0.1)
        else:
            self.aug_enabled = False
        
        # 按视频名称组织帧路径
        self.video_frames = {}
        if os.path.exists(frame_dir):
            for video_name in os.listdir(frame_dir):
                video_path = os.path.join(frame_dir, video_name)
                if os.path.isdir(video_path):
                    frames = sorted([
                        os.path.join(video_path, f)
                        for f in os.listdir(video_path)
                        if f.lower().endswith((".jpg", ".png"))
                    ])
                    if frames:
                        self.video_frames[video_name] = frames
        
        # 生成训练样本（帧序列路径 + 标签）
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        """从标注数据中提取有效帧序列样本"""
        samples = []
        for action in self.label_data:
            video_name = action.get("video_name")
            if not video_name or video_name not in self.video_frames:
                continue
            
            video_frames = self.video_frames[video_name]
            
            # 从标注中获取动作的帧范围
            start_frame = action["start_frame"]
            end_frame = action["end_frame"]
            
            # 取动作中间帧为中心，截取连续SEQUENCE_LENGTH帧
            mid_frame = (start_frame + end_frame) // 2
            seq_start = mid_frame - self.sequence_length // 2
            seq_end = mid_frame + self.sequence_length // 2

            # 跳过越界的帧序列
            if seq_start < 0 or seq_end >= len(video_frames):
                continue

            # 获取该序列的所有帧路径
            seq_paths = video_frames[seq_start:seq_end]
            # 转换标签为数字（0/1）
            label = 0 if action["class"] == "round_start" else 1
            samples.append((seq_paths, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, frame):
        """
        对单帧应用数据增强（训练时）
        为了保持时间一致性，所有帧使用相同的随机参数
        """
        if not self.aug_enabled:
            return frame
        
        h, w = frame.shape[:2]
        
        # 水平翻转（羽毛球场地对称，合理）
        if np.random.random() < self.hflip_prob:
            frame = cv2.flip(frame, 1)
        
        # 随机裁剪（小范围）
        if self.random_crop:
            scale = np.random.uniform(self.crop_scale[0], self.crop_scale[1])
            new_h, new_w = int(h * scale), int(w * scale)
            y = np.random.randint(0, h - new_h + 1)
            x = np.random.randint(0, w - new_w + 1)
            frame = frame[y:y+new_h, x:x+new_w]
            frame = cv2.resize(frame, (w, h))
        
        # 颜色抖动（模拟不同光照）
        if self.color_jitter:
            # 亮度
            if self.brightness > 0:
                brightness_factor = np.random.uniform(1 - self.brightness, 1 + self.brightness)
                frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
            
            # 对比度
            if self.contrast > 0:
                contrast_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
                mean = np.mean(frame)
                frame = np.clip((frame - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
            
            # 饱和度（转换为HSV调整）
            if self.saturation > 0 or self.hue > 0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
                if self.saturation > 0:
                    sat_factor = np.random.uniform(1 - self.saturation, 1 + self.saturation)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
                if self.hue > 0:
                    hue_factor = np.random.uniform(-self.hue * 180, self.hue * 180)
                    hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor) % 180
                frame = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 随机旋转（小幅）
        if self.random_rotate:
            angle = np.random.uniform(-self.rotate_degree, self.rotate_degree)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # 高斯模糊（模拟运动模糊）
        if self.gaussian_blur and np.random.random() < self.blur_prob:
            ksize = np.random.choice([3, 5])
            frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        
        return frame
    
    def __getitem__(self, idx):
        """加载帧序列并转换为模型输入格式"""
        seq_paths, label = self.samples[idx]

        # 读取帧并转换为RGB
        frames = []
        for i, path in enumerate(seq_paths):
            frame = cv2.imread(path)
            if frame is None:
                raise FileNotFoundError(f"Frame not found: {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
            
            # 训练时应用数据增强（为了保持时间一致性，使用第一帧的随机种子）
            if self.is_train and i == 0:
                # 设置随机种子，确保所有帧使用相同的增强参数
                aug_seed = np.random.randint(0, 2**31 - 1)
            
            if self.is_train:
                np.random.seed(aug_seed)  # 复用相同的随机种子
                frame = self._apply_augmentation(frame)
            
            # 如果设置了目标尺寸，则resize；否则保持原始尺寸
            if MODEL_INPUT_SIZE is not None:
                if frame.shape[0] != MODEL_INPUT_SIZE[1] or frame.shape[1] != MODEL_INPUT_SIZE[0]:
                    frame = cv2.resize(frame, MODEL_INPUT_SIZE)
                
            frames.append(frame)

        # 转换为 tensor (T, H, W, 3)
        frames_np = np.stack(frames, axis=0)
        # (T, H, W, 3) -> (T, 3, H, W)
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0

        # 应用数据增强
        if self.transform:
            frames_tensor = self.transform(frames_tensor)

        # 最终形状 (3, T, H, W) 适配 I3D
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)

        return frames_tensor, torch.tensor(label, dtype=torch.long)


# -------------------------- 模型加载与定义 --------------------------
class BadmintonI3D(nn.Module):
    def __init__(self, pretrained_path: str, num_classes: int = 2):
        super().__init__()
        # 从本地i3d.py导入模型（兼容直接运行和模块导入）
        try:
            from i3d import InceptionI3d, Unit3D
        except ImportError:
            from .i3d import InceptionI3d, Unit3D
        # 初始化原始I3D模型
        self.model = InceptionI3d(num_classes=400, in_channels=3)

        # 加载预训练权重
        if os.path.exists(pretrained_path):
            self.model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
            print(f"Loaded pretrained weights: {pretrained_path}")
        else:
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

        # 修改适配二分类任务
        self.model.logits = Unit3D(in_channels=1024, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        self.model.end_points['Logits'] = self.model.logits

    def forward(self, x):
        # x shape: (batch, 3, T, H, W)
        logits = self.model(x)
        
        # I3D 可能返回 5D (B, C, T, 1, 1) 或 3D (B, C, T)
        if len(logits.shape) == 5:
            logits = logits.squeeze(-1).squeeze(-1)
            
        # 如果还有时间维度 T，则进行时间维度平均 (Global Average Pooling over time)
        if len(logits.shape) == 3:
            logits = torch.mean(logits, dim=2)
        
        # 兜底：如果维度依然不匹配 (例如 [B, C, 1, 1])，强制拉平
        if len(logits.shape) > 2:
            logits = logits.view(logits.size(0), -1)
            
        return logits


# -------------------------- 训练与验证函数 --------------------------
def train_epoch(model, train_loader, criterion, optimizer, max_grad_norm=1.0):
    """训练一个epoch并返回平均损失和准确率"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in tqdm(train_loader, desc="Training batch"):
        frames, labels = frames.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    acc = correct / total
    return avg_loss, acc


def validate_epoch(model, val_loader, criterion):
    """验证模型并返回平均损失和准确率"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc="Validation batch"):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            outputs = model(frames)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    acc = correct / total
    return avg_loss, acc


# -------------------------- 主函数 --------------------------
def main():
    # 0. 设置随机种子（保证可复现性）
    seed = config.get('system', 'seed')
    if seed is None:
        seed = 42
    else:
        seed = int(seed)  # 确保是整数类型
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")
    
    # 1. 加载标注数据
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    print(f"Loaded labels: {len(label_data)} action samples")

    # 2. 数据增强与转换
    # 注意：尺寸调整已在 Dataset.__getitem__ 中完成
    # 这里只做归一化和时间维度增强
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. 初始化数据集（训练集和验证集分开，训练集启用增强）
    full_dataset = BadmintonDataset(
        frame_dir=FRAME_DIR,
        label_data=label_data,
        sequence_length=SEQUENCE_LENGTH,
        transform=transform,
        is_train=False  # 先创建完整数据集用于划分
    )
    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty! Check frame folders and labels.")

    # 划分训练集和验证集索引
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 根据划分索引，创建带/不带增强的数据集
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]
    
    # 创建训练集（启用增强）
    train_dataset = BadmintonDataset(
        frame_dir=FRAME_DIR,
        label_data=label_data,
        sequence_length=SEQUENCE_LENGTH,
        transform=transform,
        is_train=True
    )
    train_dataset.samples = train_samples
    
    # 创建验证集（不启用增强）
    val_dataset = BadmintonDataset(
        frame_dir=FRAME_DIR,
        label_data=label_data,
        sequence_length=SEQUENCE_LENGTH,
        transform=transform,
        is_train=False
    )
    val_dataset.samples = val_samples

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    print(f"Split: train {train_size}, val {val_size}")

    # 4. 初始化模型
    model = BadmintonI3D(pretrained_path=PRETRAINED_WEIGHTS, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器
    scheduler_config = config.get('training', 'lr_scheduler')
    if scheduler_config and scheduler_config.get('enabled', False):
        scheduler_type = scheduler_config.get('type', 'step')
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=scheduler_config.get('step_size', 5),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        else:
            scheduler = None
    else:
        scheduler = None
    
    # 早停机制
    early_stop_patience = 5  # 连续 5 个 epoch 验证准确率不提升则停止
    early_stop_counter = 0

    # 5. 训练循环
    best_val_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        print(f"Train: loss = {train_loss:.4f}, acc = {train_acc:.4f}")
        print(f"Val: loss = {val_loss:.4f}, acc = {val_acc:.4f}")

        # 学习率调度
        if scheduler:
            scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            best_model_path = os.path.join(SAVE_DIR, f"best_model_{current_time}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"-> Saved best model: {best_model_path} (acc: {best_val_acc:.4f})")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered! No improvement for {early_stop_patience} epochs.")
                break

    final_model_path = os.path.join(SAVE_DIR, f"final_model_{current_time}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining finished! Best acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
