"""
批量创建标注文件模板
为所有没有标注文件的视频创建JSON模板
"""

import os
import cv2
import json
from pathlib import Path

# 配置路径
RAW_VIDEOS_DIR = "01_data/raw_videos"
ANNOTATIONS_DIR = "01_data/annotations"

def get_video_info(video_path):
    """获取视频的帧数和帧率"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ 无法打开视频: {video_path}")
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    return total_frames, fps

def create_annotation_template(video_name, total_frames, fps):
    """创建标注文件模板"""
    template = {
        "video_name": video_name,
        "total_frames": total_frames,
        "frame_rate": fps,
        "annotations": [
            {
                "action_id": 1,
                "class": "round_start",
                "start_frame": 100,
                "end_frame": 120,
                "description": "第1回合发球",
                "notes": "请修改为实际帧号"
            },
            {
                "action_id": 2,
                "class": "round_end",
                "start_frame": 450,
                "end_frame": 470,
                "description": "第1回合球落地",
                "notes": "请修改为实际帧号"
            }
        ]
    }
    return template

def main():
    print("=" * 60)
    print("批量创建标注文件模板")
    print("=" * 60)
    
    # 确保目录存在
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    
    # 获取所有视频文件
    video_files = [f for f in os.listdir(RAW_VIDEOS_DIR) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"✗ 在 {RAW_VIDEOS_DIR} 中未找到视频文件")
        return
    
    print(f"\n找到 {len(video_files)} 个视频文件")
    print("-" * 60)
    
    created_count = 0
    skipped_count = 0
    
    for video_file in sorted(video_files):
        video_name = Path(video_file).stem
        annotation_file = f"{video_name}_annotations.json"
        annotation_path = os.path.join(ANNOTATIONS_DIR, annotation_file)
        
        # 检查标注文件是否已存在
        if os.path.exists(annotation_path):
            print(f"⊙ {video_name}: 标注文件已存在，跳过")
            skipped_count += 1
            continue
        
        # 获取视频信息
        video_path = os.path.join(RAW_VIDEOS_DIR, video_file)
        total_frames, fps = get_video_info(video_path)
        
        if total_frames is None:
            continue
        
        # 创建标注模板
        template = create_annotation_template(video_name, total_frames, fps)
        
        # 保存为JSON文件
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        
        print(f"✓ {video_name}: 已创建 (帧数: {total_frames}, 帧率: {fps} FPS)")
        created_count += 1
    
    print("-" * 60)
    print(f"\n完成！")
    print(f"  新创建: {created_count} 个标注文件")
    print(f"  已跳过: {skipped_count} 个已存在的文件")
    print(f"\n提示: 请手动编辑标注文件，填写实际的动作帧号")
    print(f"标注文件位置: {ANNOTATIONS_DIR}")

if __name__ == "__main__":
    main()
