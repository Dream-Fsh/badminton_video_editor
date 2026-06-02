"""
配置文件加载器
功能：加载YAML配置文件，提供配置参数访问接口
"""

import os
import yaml
import sys
from pathlib import Path

from typing import Any, Optional, Union


class Config:
    """配置类：加载并管理所有配置参数"""
    
    def __init__(self, config_path: str = "../05_config/config.yaml") -> None:
        """
        初始化配置加载器
        
        参数:
            config_path: 配置文件路径
        """
        self.config_path = self._find_config_path(config_path)
        self.config: dict = self._load_config()
        self._resolve_paths()
    
    def _find_config_path(self, config_path):
        """
        智能查找配置文件路径
        支持从项目根目录或02_code目录运行
        """
        # 如果是绝对路径，直接使用
        if os.path.isabs(config_path):
            return config_path
        
        # 尝试多个可能的路径
        possible_paths = [
            config_path,  # 原始相对路径
            os.path.join('..', config_path) if not config_path.startswith('..') else config_path,
            os.path.join('05_config', 'config.yaml'),  # 从项目根目录
            os.path.join('..', '05_config', 'config.yaml'),  # 从02_code目录
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果都找不到，返回原始路径（让后续报错）
        return config_path
    
    def _load_config(self):
        """加载YAML配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"[OK] Loaded config: {self.config_path}")
        return config
    
    def _resolve_paths(self):
        """解析相对路径为绝对路径"""
        # 获取项目根目录
        current_dir = Path.cwd()
        if current_dir.name == '02_code':
            base_dir = current_dir.parent
        else:
            base_dir = current_dir
        
        if 'paths' in self.config:
            for key, value in self.config['paths'].items():
                if isinstance(value, str) and not os.path.isabs(value):
                    # 将相对路径转换为绝对路径
                    abs_path = (base_dir / value).resolve()
                    self.config['paths'][key] = str(abs_path)
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        获取配置参数（支持多级键访问）
        """
        value: Any = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问：config['training']"""
        return self.config.get(key)


def load_config(config_path="../05_config/config.yaml"):
    """
    加载配置文件的便捷函数
    """
    return Config(config_path)


if __name__ == "__main__":
    # 测试配置加载
    config = load_config()
    print("\n========== Config Test ==========")
    print(f"Batch size: {config.get('training', 'batch_size')}")
    print(f"Learning rate: {config.get('training', 'learning_rate')}")
    print(f"Raw video path: {config.get('paths', 'raw_videos')}")
    print(f"Action classes: {config.get('action_classes')}")
    print("=" * 30)
