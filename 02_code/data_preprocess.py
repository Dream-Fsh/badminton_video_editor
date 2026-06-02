"""
数据预处理模块
功能：
1. 视频拆帧：将原始视频分解为图像帧序列
2. 标注解析：解析标注文件，生成训练数据标签
3. 数据集划分：划分训练集和验证集
"""

import os
import cv2
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# 兼容直接运行和模块导入
try:
    from config_loader import load_config
except ImportError:
    from .config_loader import load_config


class VideoPreprocessor:
    """视频预处理器：负责视频拆帧和标注处理"""
    
    def __init__(self, config_path="../05_config/config.yaml"):
        """
        初始化预处理器
        
        参数:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        
        # 加载路径配置
        self.raw_videos_dir = self.config.get('paths', 'raw_videos')
        self.frames_dir = self.config.get('paths', 'processed_frames')
        self.annotations_dir = self.config.get('paths', 'annotations')
        
        # 加载预处理配置
        self.frame_rate = self.config.get('preprocessing', 'frame_rate')
        # frame_size 如果为 null 或空，则保留原始尺寸
        config_size = self.config.get('preprocessing', 'frame_size')
        self.frame_size = tuple(config_size) if config_size else None
        self.frame_format = self.config.get('preprocessing', 'frame_format')
        self.frame_quality = self.config.get('preprocessing', 'frame_quality')
        
        # ROI 裁剪配置
        roi_config = self.config.get('preprocessing', 'roi')
        self.roi_enabled = roi_config.get('enabled', False) if roi_config else False
        self.roi_x = roi_config.get('x_offset', 0) if roi_config else 0
        self.roi_y = roi_config.get('y_offset', 0) if roi_config else 0
        self.roi_w = roi_config.get('width', 950) if roi_config else 950
        self.roi_h = roi_config.get('height', 720) if roi_config else 720
        
        # 创建必要的目录
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
    
    def extract_frames(self, video_path, output_dir=None, video_name=None):
        """
        从视频中提取帧并保存为图片
        
        参数:
            video_path: 视频文件路径
            output_dir: 输出目录（默认使用配置中的路径）
            video_name: 视频名称（用于命名帧文件）
        
        返回:
            提取的帧数量
        """
        if output_dir is None:
            output_dir = self.frames_dir
        
        if video_name is None:
            video_name = Path(video_path).stem
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n处理视频: {video_name}")
        print(f"  总帧数: {total_frames}")
        print(f"  帧率: {fps:.2f} FPS")
        if self.frame_size:
            print(f"  目标尺寸: {self.frame_size}")
        else:
            print(f"  目标尺寸: 原始视频尺寸")
        
        # 创建视频专属的帧文件夹
        video_frames_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_frames_dir, exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        
        # 逐帧读取并保存
        with tqdm(total=total_frames, desc="提取帧") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # ROI 裁剪（在调整尺寸之前）
                if self.roi_enabled:
                    h, w = frame.shape[:2]
                    # 确保裁剪区域不超出图像边界
                    x_end = min(self.roi_x + self.roi_w, w)
                    y_end = min(self.roi_y + self.roi_h, h)
                    frame = frame[self.roi_y:y_end, self.roi_x:x_end]
                
                # 调整帧尺寸（仅当设置了目标尺寸时）
                if self.frame_size:
                    frame = cv2.resize(frame, self.frame_size)
                
                # 保存帧
                frame_filename = f"{video_name}_frame_{frame_count:06d}.{self.frame_format}"
                frame_path = os.path.join(video_frames_dir, frame_filename)
                
                # 根据格式保存
                if self.frame_format.lower() == 'jpg':
                    cv2.imwrite(frame_path, frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, self.frame_quality])
                else:
                    cv2.imwrite(frame_path, frame)
                
                frame_count += 1
                saved_count += 1
                pbar.update(1)
        
        cap.release()
        print(f"[OK] 成功提取 {saved_count} 帧到: {video_frames_dir}")
        
        return saved_count
    
    def process_all_videos(self):
        """
        批量处理所有原始视频
        
        返回:
            处理的视频数量
        """
        # 自动创建目录（如果不存在）
        if not os.path.exists(self.raw_videos_dir):
            os.makedirs(self.raw_videos_dir, exist_ok=True)
            print(f"[OK] 已创建原始视频目录: {self.raw_videos_dir}")
            print(f"\n提示：请将羽毛球比赛视频放入此目录，然后重新运行")
            return 0
        
        # 获取所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = [
            f for f in os.listdir(self.raw_videos_dir)
            if Path(f).suffix.lower() in video_extensions
        ]
        
        if not video_files:
            print(f"警告: 在 {self.raw_videos_dir} 中未找到视频文件")
            return 0
        
        print(f"\n找到 {len(video_files)} 个视频文件")
        print("=" * 50)
        
        # 逐个处理视频
        for video_file in video_files:
            video_path = os.path.join(self.raw_videos_dir, video_file)
            try:
                self.extract_frames(video_path)
            except Exception as e:
                print(f"[FAIL] 处理视频失败 {video_file}: {e}")
                continue
        
        print("\n" + "=" * 50)
        print(f"[OK] 批量处理完成！共处理 {len(video_files)} 个视频")
        
        return len(video_files)
    
    def create_annotation_template(self, video_name, num_frames, output_path=None):
        """
        创建标注文件模板（JSON格式）
        
        参数:
            video_name: 视频名称
            num_frames: 视频总帧数
            output_path: 输出路径（默认保存到annotations目录）
        
        返回:
            标注文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.annotations_dir, f"{video_name}_annotations.json")
        
        # 标注模板结构
        annotation_template = {
            "video_name": video_name,
            "total_frames": num_frames,
            "frame_rate": self.frame_rate,
            "annotations": [
                {
                    "action_id": 1,
                    "class": "round_start",
                    "start_frame": 100,
                    "end_frame": 120,
                    "description": "第1回合发球"
                },
                {
                    "action_id": 2,
                    "class": "round_end",
                    "start_frame": 450,
                    "end_frame": 470,
                    "description": "第1回合球落地"
                }
                # 继续添加更多标注...
            ]
        }
        
        # 保存模板
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_template, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] 标注模板已创建: {output_path}")
        print(f"  请手动编辑此文件，添加实际的动作标注")
        
        return output_path
    
    def merge_annotations(self, output_path=None):
        """
        合并所有标注文件为一个统一的JSON文件
        
        参数:
            output_path: 输出路径（默认使用配置中的merged_labels路径）
        
        返回:
            合并后的标注数据
        """
        if output_path is None:
            output_path = self.config.get('paths', 'merged_labels')
        
        # 查找所有标注文件
        annotation_files = [
            f for f in os.listdir(self.annotations_dir)
            if f.endswith('_annotations.json')
        ]
        
        if not annotation_files:
            print(f"警告: 在 {self.annotations_dir} 中未找到标注文件")
            return []
        
        merged_data = []
        
        # 逐个读取并合并
        for ann_file in annotation_files:
            ann_path = os.path.join(self.annotations_dir, ann_file)
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 提取annotations部分
                    if 'annotations' in data:
                        for ann in data['annotations']:
                            # 添加视频名称信息
                            ann['video_name'] = data.get('video_name', '')
                            merged_data.append(ann)
            except Exception as e:
                print(f"[FAIL] 读取标注文件失败 {ann_file}: {e}")
                continue
        
        # 保存合并后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[OK] 成功合并 {len(annotation_files)} 个标注文件")
        print(f"  总标注数量: {len(merged_data)}")
        print(f"  保存路径: {output_path}")
        
        return merged_data
    
    def validate_annotations(self, annotation_path=None):
        """
        验证标注文件的有效性
        
        参数:
            annotation_path: 标注文件路径
        
        返回:
            验证结果（True/False）
        """
        if annotation_path is None:
            annotation_path = self.config.get('paths', 'merged_labels')
        
        if not os.path.exists(annotation_path):
            print(f"[FAIL] 标注文件不存在: {annotation_path}")
            return False
        
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print("[FAIL] 标注文件格式错误：应为列表格式")
                return False
            
            # 检查必要字段
            required_fields = ['class', 'start_frame', 'end_frame']
            valid_classes = ['round_start', 'round_end']
            
            for i, ann in enumerate(data):
                # 检查必要字段是否存在
                for field in required_fields:
                    if field not in ann:
                        print(f"[FAIL] 标注 {i+1} 缺少字段: {field}")
                        return False
                
                # 检查类别是否有效
                if ann['class'] not in valid_classes:
                    print(f"[FAIL] 标注 {i+1} 类别无效: {ann['class']}")
                    return False
                
                # 检查帧范围是否合理
                if ann['start_frame'] >= ann['end_frame']:
                    print(f"[FAIL] 标注 {i+1} 帧范围无效: start={ann['start_frame']}, end={ann['end_frame']}")
                    return False
            
            print(f"[OK] 标注文件验证通过！共 {len(data)} 条标注")
            return True
            
        except json.JSONDecodeError as e:
            print(f"[FAIL] JSON解析错误: {e}")
            return False
        except Exception as e:
            print(f"[FAIL] 验证失败: {e}")
            return False


def main():
    """主函数：演示数据预处理流程"""
    print("=" * 60)
    print("羽毛球视频自动剪辑系统 - 数据预处理模块")
    print("=" * 60)
    
    # 初始化预处理器
    preprocessor = VideoPreprocessor()
    
    # 步骤1: 批量提取视频帧
    print("\n【步骤1】批量提取视频帧")
    print("-" * 60)
    num_videos = preprocessor.process_all_videos()
    
    if num_videos == 0:
        print("\n提示：请将羽毛球比赛视频放入以下目录：")
        print(f"  {preprocessor.raw_videos_dir}")
        print("\n然后重新运行此脚本")
        return
    
    # 步骤2: 创建标注模板（示例）
    print("\n【步骤2】创建标注模板")
    print("-" * 60)
    print("提示：标注模板已创建在 annotations 目录")
    print("请手动编辑标注文件，标记每个视频中的关键动作：")
    print("  - round_start: 发球动作")
    print("  - round_end: 球落地动作")
    
    # 步骤3: 合并标注文件
    print("\n【步骤3】合并标注文件")
    print("-" * 60)
    merged_data = preprocessor.merge_annotations()
    
    # 步骤4: 验证标注
    print("\n【步骤4】验证标注文件")
    print("-" * 60)
    is_valid = preprocessor.validate_annotations()
    
    if is_valid:
        print("\n[OK] 数据预处理完成！可以开始模型训练")
    else:
        print("\n[FAIL] 标注文件存在问题，请检查后重试")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
