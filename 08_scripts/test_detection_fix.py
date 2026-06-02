#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试YOLO运动检测和回合提取修复效果
"""

import sys
sys.path.insert(0, '02_code')

from model_predict_optimized import ActionPredictorFast
from pathlib import Path
import json

def test_detection(video_path):
    """测试检测效果"""
    print(f"\n{'='*70}")
    print(f"测试视频: {Path(video_path).name}")
    print(f"{'='*70}\n")
    
    # 初始化预测器
    model_path = "03_model/trained/best_model.pth"
    config_path = "05_config/config_fast.yaml"
    
    print("[1/3] 初始化模型...")
    try:
        predictor = ActionPredictorFast(model_path, config_path=config_path)
        print("✓ 模型初始化成功\n")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return
    
    # 运行预测
    print("[2/3] 运行预测...")
    try:
        results = predictor.predict_video(video_path, save_predictions=True)
        print(f"✓ 预测完成\n")
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 分析结果
    print("[3/3] 分析结果...")
    predictions = results.get('action_predictions', [])
    motion_data = results.get('motion_data', [])
    
    print(f"\n{'='*70}")
    print("检测结果摘要:")
    print(f"{'='*70}")
    print(f"I3D检测到 {len(predictions)} 个动作")
    print(f"  - round_start: {sum(1 for p in predictions if p['class_name'] == 'round_start')}")
    print(f"  - round_end: {sum(1 for p in predictions if p['class_name'] == 'round_end')}")
    print(f"YOLO检测到 {len(motion_data)} 个运动状态点")
    print(f"  - 有运动员: {sum(1 for m in motion_data if m['player_count'] > 0)}")
    print(f"  - 球检测到: {sum(1 for m in motion_data if m.get('shuttlecock_detected', False))}")
    
    # 提取回合
    print(f"\n{'='*70}")
    print("回合提取结果:")
    print(f"{'='*70}")
    try:
        rounds = predictor.extract_rounds(results)
        print(f"✓ 成功提取 {len(rounds)} 个回合\n")
        
        for i, r in enumerate(rounds[:10]):  # 显示前10个
            print(f"Round {r['round_id']:2d}: {r['start_time']:6.2f}s - {r['end_time']:6.2f}s "
                  f"({r['duration']:4.2f}s) | start: {r['start_method']}, end: {r['end_method']}")
        
        if len(rounds) > 10:
            print(f"\n... 还有 {len(rounds) - 10} 个回合未显示")
            
    except Exception as e:
        print(f"✗ 回合提取失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*70}")
    print("测试完成！")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # 使用示例视频
        video_path = "01_data/raw_videos/match_001.mp4"
        if not Path(video_path).exists():
            print(f"示例视频不存在: {video_path}")
            print("请提供视频路径: python test_detection_fix.py <video_path>")
            sys.exit(1)
    
    test_detection(video_path)
