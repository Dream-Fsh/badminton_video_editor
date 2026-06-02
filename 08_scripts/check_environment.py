#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查运行环境
"""

import sys
import torch

def check_environment():
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    # PyTorch 信息
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"Python 版本: {sys.version}")
    
    # CUDA 信息
    print(f"\nCUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("❌ CUDA 不可用，正在使用 CPU (慢)")
    
    # 性能建议
    print("\n" + "=" * 60)
    print("性能建议")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("\n❌ 当前使用 CPU 推理，速度会很慢！")
        print("\n建议:")
        print("1. 安装 CUDA 版本的 PyTorch:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("\n2. 或者降低 YOLO 采样率:")
        print("   编辑 05_config/config.yaml")
        print("   sample_rate: 5  # 改为5或更高")
    else:
        print("\n✅ CUDA 可用！")
        print("建议:")
        print("1. 确保 batch_size 设置合适")
        print("2. 可以启用 FP16 加速:")
        print("   use_fp16: true")

if __name__ == "__main__":
    check_environment()
