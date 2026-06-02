import os
import json
import torch
from torchvision import transforms
import sys

# 添加代码目录到路径
sys.path.append('02_code')
from model_train import BadmintonDataset
from config_loader import load_config

def test_dataset():
    config = load_config('05_config/config.yaml')
    frame_dir = config.get('paths', 'processed_frames')
    label_path = config.get('paths', 'merged_labels')
    sequence_length = config.get('training', 'sequence_length')
    
    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        return

    with open(label_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    
    print(f"Loaded {len(label_data)} labels")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = BadmintonDataset(
        frame_dir=frame_dir,
        label_data=label_data,
        sequence_length=sequence_length,
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        frames, label = dataset[0]
        print(f"Sample shape: {frames.shape}, label: {label}")
    else:
        print("Dataset is still empty!")
        # Debug why it's empty
        print(f"Frame dir: {frame_dir}")
        if os.path.exists(frame_dir):
            subdirs = os.listdir(frame_dir)
            print(f"Subdirs in frame_dir: {subdirs[:5]}...")
            for sd in subdirs[:5]:
                sd_path = os.path.join(frame_dir, sd)
                if os.path.isdir(sd_path):
                    files = os.listdir(sd_path)
                    print(f"  {sd} contains {len(files)} files")

if __name__ == "__main__":
    test_dataset()
