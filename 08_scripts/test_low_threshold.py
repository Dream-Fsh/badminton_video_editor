"""
使用低阈值重新预测视频
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, '02_code')

from model_predict_optimized import ActionPredictorFast

def test_low_threshold(video_path, model_path):
    """使用低阈值测试预测"""
    print(f"\n{'='*70}")
    print(f"低阈值测试预测")
    print(f"{'='*70}\n")
    
    # 初始化预测器（使用默认配置，阈值已设为0.01）
    print("加载预测器...")
    predictor = ActionPredictorFast(model_path)
    
    print(f"\n视频: {Path(video_path).name}")
    print(f"阈值: 0.01 (超低)")
    print(f"步长: 2 (高密度检测)\n")
    
    # 执行预测
    print("开始预测...\n")
    predictions = predictor.predict_video(
        video_path, 
        sliding_stride=2,  # 更密集的检测
        save_predictions=True
    )
    
    # 提取回合
    rounds = predictor.extract_rounds(predictions)
    
    print(f"\n{'='*70}")
    print(f"预测完成")
    print(f"检测到的回合数: {len(rounds)}")
    print(f"{'='*70}\n")
    
    if rounds:
        print("检测到的回合:")
        for r in rounds:
            print(f"  Round {r['round_id']}: {r['start_time']:.2f}s - {r['end_time']:.2f}s")
    
    return len(rounds) > 0

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "01_data/raw_videos/match_004_d7ee8554.mp4"
    
    # 模型路径
    model_path = "03_model/trained/best_model_20260323_224211.pth"
    
    success = test_low_threshold(video_path, model_path)
    
    if success:
        print("\n✅ 检测到回合！查看预测结果文件获取详情")
    else:
        print("\n❌ 仍未检测到回合，可能是模型需要重新训练")
        print("建议: 收集更多训练数据，特别是发球动作")
