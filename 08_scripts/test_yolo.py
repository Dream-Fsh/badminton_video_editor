"""
测试 YOLO 运动员检测
"""

import sys
import cv2
from pathlib import Path

# 添加路径
sys.path.insert(0, '02_code')

from athlete_detector import AthleteDetector

def test_yolo(video_path):
    """测试 YOLO 检测"""
    print(f"\n{'='*60}")
    print(f"测试 YOLO 运动员检测")
    print(f"视频: {video_path}")
    print(f"{'='*60}\n")
    
    # 初始化检测器
    print("初始化 YOLO 检测器...")
    try:
        detector = AthleteDetector(model_type="yolov8n.pt")
        print("✓ 检测器初始化成功")
    except Exception as e:
        print(f"✗ 检测器初始化失败: {e}")
        return
    
    # 打开视频
    print(f"\n打开视频...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ 无法打开视频: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ 视频信息: {total_frames} 帧, {fps:.2f} fps")
    
    # 测试检测
    print(f"\n进行运动员检测测试...")
    print(f"采样率: 每 10 帧检测一次")
    
    frame_idx = 0
    detection_count = 0
    
    constraints = {
        'min_confidence': 0.5,
        'court_boundary': {
            'enabled': True,
            'x_min': 50,
            'x_max': 900,
            'y_min': 100,
            'y_max': 650
        },
        'max_count': 2
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 10 == 0:
            # ROI 裁剪（模拟实际使用场景）
            h, w = frame.shape[:2]
            roi_x, roi_y, roi_w, roi_h = 165, 0, 950, 720
            x_end = min(roi_x + roi_w, w)
            y_end = min(roi_y + roi_h, h)
            frame_roi = frame[roi_y:y_end, roi_x:x_end]
            
            # 检测
            players = detector.detect_athletes(frame_roi, constraints=constraints)
            
            if players:
                detection_count += 1
                print(f"  Frame {frame_idx}: 检测到 {len(players)} 个运动员")
                for i, player in enumerate(players):
                    print(f"    Player {i+1}: 置信度={player['conf']:.3f}, 中心点={player['center']}")
        
        frame_idx += 1
        if frame_idx > 200:  # 只测试前200帧
            break
    
    cap.release()
    
    print(f"\n{'='*60}")
    print(f"测试完成")
    print(f"总检测帧数: {detection_count}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # 使用示例视频
        video_path = "01_data/raw_videos/match_001.mp4"
    
    test_yolo(video_path)
