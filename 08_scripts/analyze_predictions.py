"""
预测结果分析工具
用于诊断为什么某些回合没有被检测到
"""

import json
import sys
from pathlib import Path

def analyze_predictions(json_path):
    """分析预测结果"""
    print(f"\n{'='*60}")
    print(f"分析预测结果: {json_path}")
    print(f"{'='*60}\n")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = data['results']['action_predictions']
    
    if not predictions:
        print("❌ 没有任何预测结果！")
        return
    
    # 统计各类别
    round_starts = [p for p in predictions if p['class_name'] == 'round_start']
    round_ends = [p for p in predictions if p['class_name'] == 'round_end']
    
    print(f"✓ 总预测数量: {len(predictions)}")
    print(f"  - round_start (发球): {len(round_starts)}")
    print(f"  - round_end (球落地): {len(round_ends)}")
    
    # 显示round_start详情
    if round_starts:
        print(f"\n📍 Round Start 检测点:")
        for i, pred in enumerate(round_starts, 1):
            print(f"  [{i}] 时间: {pred['timestamp']:.2f}s, 置信度: {pred['confidence']:.3f}, 帧: {pred['center_frame']}")
    
    # 显示round_end详情
    if round_ends:
        print(f"\n📍 Round End 检测点:")
        for i, pred in enumerate(round_ends, 1):
            print(f"  [{i}] 时间: {pred['timestamp']:.2f}s, 置信度: {pred['confidence']:.3f}, 帧: {pred['center_frame']}")
    else:
        print(f"\n⚠️  没有检测到 round_end！这是问题的关键")
        print(f"   可能原因:")
        print(f"   1. 模型对球落地动作识别不敏感")
        print(f"   2. 置信度阈值设置过高")
        print(f"   3. 视频中round_end动作不明显")
    
    # 显示置信度分布
    print(f"\n📊 置信度分布:")
    confidences = [p['confidence'] for p in predictions]
    print(f"  最高: {max(confidences):.3f}")
    print(f"  最低: {min(confidences):.3f}")
    print(f"  平均: {sum(confidences)/len(confidences):.3f}")
    
    # 检查低置信度预测
    low_conf = [p for p in predictions if p['confidence'] < 0.15]
    if low_conf:
        print(f"\n⚠️  有 {len(low_conf)} 个低置信度预测 (< 0.15):")
        for p in low_conf:
            print(f"   - {p['class_name']}: {p['timestamp']:.2f}s, 置信度: {p['confidence']:.3f}")
    
    # 建议
    print(f"\n💡 建议:")
    if not round_ends:
        print(f"   1. 降低 prediction.confidence_threshold (当前可能为 0.15)")
        print(f"   2. 考虑重新训练模型，提高round_end识别能力")
        print(f"   3. 检查视频中是否有清晰的球落地画面")
    elif len(round_starts) == 1 and len(round_ends) == 0:
        print(f"   1. 视频可能只包含发球动作，没有完整回合")
        print(f"   2. 或者模型未能识别到球落地动作")
    
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # 使用最新的预测结果
        pred_dir = Path("04_output/predictions")
        json_files = sorted(pred_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if json_files:
            json_path = json_files[0]
        else:
            print("❌ 没有找到预测结果文件")
            sys.exit(1)
    
    analyze_predictions(json_path)
