"""
回合提取过程详细分析
分析为什么某些时间段没有被识别为有效回合
"""

import json
import sys
from pathlib import Path

def analyze_round_extraction(json_path, video_path=None):
    """分析回合提取过程"""
    print(f"\n{'='*70}")
    print(f"回合提取过程分析")
    print(f"{'='*70}\n")
    
    # 读取预测结果
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    predictions = results['action_predictions']
    motion_data = results['motion_data']
    
    print(f"视频: {data['video_name']}")
    print(f"时长: {results['total_duration']:.2f} 秒\n")
    
    # 分析I3D检测
    round_starts = [p for p in predictions if p['class_name'] == 'round_start']
    round_ends = [p for p in predictions if p['class_name'] == 'round_end']
    
    print(f"I3D 检测结果:")
    print(f"   - Round Start (发球): {len(round_starts)}")
    for i, pred in enumerate(round_starts, 1):
        print(f"     [{i}] 时间: {pred['timestamp']:.2f}s, 置信度: {pred['confidence']:.3f}")
    
    print(f"   - Round End (球落地): {len(round_ends)}")
    for i, pred in enumerate(round_ends, 1):
        print(f"     [{i}] 时间: {pred['timestamp']:.2f}s, 置信度: {pred['confidence']:.3f}")
    
    # 分析YOLO motion starts
    print(f"\nYOLO 运动检测点: {len(motion_data)}")
    
    # 找出状态变化点
    if motion_data:
        print(f"\n状态变化分析:")
        
        # 找出静止->运动的转换点
        transitions = []
        for i in range(1, len(motion_data)):
            prev = motion_data[i-1]
            curr = motion_data[i]
            
            # 静止到运动（回合可能开始）
            if prev['is_static'] and not curr['is_static']:
                transitions.append({
                    'type': 'static_to_motion',
                    'timestamp': curr['timestamp'],
                    'frame': curr['frame_idx']
                })
            
            # 运动到静止（回合可能结束）
            elif not prev['is_static'] and curr['is_static']:
                transitions.append({
                    'type': 'motion_to_static',
                    'timestamp': curr['timestamp'],
                    'frame': curr['frame_idx']
                })
        
        print(f"   状态转换点数量: {len(transitions)}")
        for i, t in enumerate(transitions[:10], 1):  # 显示前10个
            type_desc = '静止→运动 (回合开始)' if t['type'] == 'static_to_motion' else '运动→静止 (回合结束)'
            print(f"   [{i}] {type_desc}: {t['timestamp']:.2f}s")
        
        # 检查开头部分
        print(f"\n开头部分分析 (0-7秒):")
        early_motion = [m for m in motion_data if m['timestamp'] < 7.0]
        print(f"   检测点数量: {len(early_motion)}")
        
        if early_motion:
            static_count = sum(1 for m in early_motion if m['is_static'])
            motion_count = sum(1 for m in early_motion if not m['is_static'])
            print(f"   静止状态: {static_count}次 ({static_count/len(early_motion)*100:.1f}%)")
            print(f"   运动状态: {motion_count}次 ({motion_count/len(early_motion)*100:.1f}%)")
            
            # 检查是否有有效的状态转换
            early_transitions = [t for t in transitions if t['timestamp'] < 7.0]
            print(f"   状态转换: {len(early_transitions)}次")
            for t in early_transitions:
                type_desc = '静止→运动' if t['type'] == 'static_to_motion' else '运动→静止'
                print(f"     - {type_desc}: {t['timestamp']:.2f}s")
    
    # 分析过滤的回合
    print(f"\n{'='*70}")
    print(f"可能的过滤原因:")
    print(f"{'='*70}\n")
    
    print(f"1. min_round_duration = 2.0秒")
    print(f"   任何时长 < 2.0秒的回合都会被过滤")
    print(f"   日志显示: [FILTERED] Round too short: 1.57s")
    print(f"   → 开头可能有个1.57秒的短回合被过滤了\n")
    
    print(f"2. min_start_interval = 3.0秒")
    print(f"   相邻回合开始时间必须间隔 ≥ 3.0秒")
    print(f"   如果YOLO检测到多个motion starts在3秒内，会被合并\n")
    
    print(f"3. I3D未检测到round_start")
    print(f"   视频开头可能没有清晰的发球动作")
    print(f"   YOLO虽然检测到运动，但I3D没有发球动作作为确认\n")
    
    print(f"4. 有效球员检测")
    print(f"   开头部分可能valid_players=false")
    print(f"   导致motion starts不被采用\n")
    
    # 建议
    print(f"{'='*70}")
    print(f"建议:")
    print(f"{'='*70}\n")
    
    print(f"选项1: 降低min_round_duration（不推荐，会影响精度）")
    print(f"   编辑 05_config/config.yaml")
    print(f"   min_round_duration: 1.0  # 从2.0改为1.0")
    print(f"   → 可以捕获更短的回合，但可能增加误检\n")
    
    print(f"选项2: 仅依赖YOLO检测（忽略I3D的round_start）")
    print(f"   修改回合提取逻辑，优先使用YOLO motion starts")
    print(f"   → 可以检测到更多回合，但精度可能下降\n")
    
    print(f"选项3: 检查视频开头内容")
    print(f"   视频0-7秒可能没有实际比赛，只是热身/准备")
    print(f"   → 如果是这样，当前行为是正确的\n")
    
    print(f"选项4: 降低YOLO检测的valid_players约束")
    print(f"   如果开头运动员在场地外，可能被判定为无效")
    print(f"   调整 court_boundary 或 min_player_count\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
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
    
    analyze_round_extraction(json_path, video_path)
