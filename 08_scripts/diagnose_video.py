"""
视频诊断工具 - 详细分析预测结果
"""

import json
import sys
from pathlib import Path
import cv2

def diagnose_video(json_path, video_path=None):
    """诊断视频检测问题"""
    print(f"\n{'='*70}")
    print(f"视频诊断分析")
    print(f"{'='*70}\n")
    
    # 读取预测结果
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    predictions = results['action_predictions']
    motion_data = results['motion_data']
    
    print(f"视频: {data['video_name']}")
    print(f"时长: {results['total_duration']:.2f} 秒")
    print(f"预测点数: {len(predictions)}")
    print(f"运动检测点: {len(motion_data)}\n")
    
    # 分析预测结果
    round_starts = [p for p in predictions if p['class_name'] == 'round_start']
    round_ends = [p for p in predictions if p['class_name'] == 'round_end']
    
    print(f"检测统计:")
    print(f"   - Round Start (发球): {len(round_starts)}")
    print(f"   - Round End (球落地): {len(round_ends)}")
    
    if round_starts:
        print(f"\nRound Start 详情:")
        for i, pred in enumerate(round_starts, 1):
            print(f"   [{i}] 时间: {pred['timestamp']:.2f}s, 置信度: {pred['confidence']:.3f}")
    
    if round_ends:
        print(f"\nRound End 详情:")
        for i, pred in enumerate(round_ends, 1):
            print(f"   [{i}] 时间: {pred['timestamp']:.2f}s, 置信度: {pred['confidence']:.3f}")
    
    # 分析motion_data
    print(f"\nYOLO 运动检测分析:")
    valid_detections = [m for m in motion_data if m.get('valid_players', False)]
    print(f"   - 总检测点: {len(motion_data)}")
    print(f"   - 有效检测点: {len(valid_detections)}")
    
    if motion_data:
        player_counts = {}
        for m in motion_data:
            count = m.get('player_count', 0)
            player_counts[count] = player_counts.get(count, 0) + 1
        
        print(f"\n   运动员数量分布:")
        for count, num in sorted(player_counts.items()):
            percentage = (num / len(motion_data)) * 100
            print(f"      - {count}人: {num}次 ({percentage:.1f}%)")
    
    # 分析问题
    print(f"\n{'='*70}")
    print(f"问题诊断:")
    print(f"{'='*70}\n")
    
    issues = []
    
    if len(round_starts) == 0:
        issues.append("未检测到任何 round_start (发球动作)")
        issues.append("   可能原因:")
        issues.append("   1. 模型置信度阈值过高")
        issues.append("   2. 视频中没有明显的发球动作")
        issues.append("   3. 发球动作被遮挡或不清晰")
        issues.append("   4. 模型需要重新训练")
    
    if len(round_ends) == 0:
        issues.append("未检测到任何 round_end (球落地动作)")
    elif len(round_ends) <= 1:
        issues.append("只检测到1个 round_end，可能漏检")
    
    if len(valid_detections) == 0:
        issues.append("YOLO 没有有效检测点")
        issues.append("   可能原因:")
        issues.append("   1. min_player_count 设置过高")
        issues.append("   2. 视频中运动员数量不足")
        issues.append("   3. 运动员检测置信度阈值过高")
        issues.append("   4. 运动员在场地边界外")
    
    if not issues:
        print("检测器工作正常，但可能没有有效回合")
        print("   建议: 检查视频内容是否包含完整回合")
    else:
        for issue in issues:
            print(issue)
    
    # 提供建议
    print(f"\n建议:")
    print(f"1. 检查置信度阈值（当前: 0.05）")
    print(f"2. 检查运动员数量约束（当前: min=1, max=2）")
    print(f"3. 使用分析工具查看预测详情:")
    print(f"   python 08_scripts/analyze_predictions.py")
    
    if video_path and Path(video_path).exists():
        print(f"\n视频文件检查:")
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"   - 分辨率: {width}x{height}")
            print(f"   - 帧率: {fps:.2f} fps")
            print(f"   - 总帧数: {total_frames}")
            print(f"   - 时长: {duration:.2f} 秒")
            
            # 读取第一帧检查内容
            ret, frame = cap.read()
            if ret:
                print(f"   - 第一帧形状: {frame.shape}")
            
            cap.release()
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        json_path = sys.argv[1]
        video_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # 使用最新的预测结果
        pred_dir = Path("04_output/predictions")
        json_files = sorted(pred_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if json_files:
            json_path = json_files[0]
            video_name = json_path.stem.replace("_predictions", "")
            video_path = Path("01_data/raw_videos") / f"{video_name}.mp4"
            if not video_path.exists():
                video_path = None
        else:
            print("❌ 没有找到预测结果文件")
            sys.exit(1)
    
    diagnose_video(json_path, video_path)
