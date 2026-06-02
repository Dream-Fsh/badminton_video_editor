"""
高速预测入口
使用方法:
  python predict_fast.py <视频路径> [--stride 8] [--save]

示例:
  python predict_fast.py 01_data/raw_videos/match_001.mp4
  python predict_fast.py 01_data/raw_videos/match_001.mp4 --stride 16 --save
"""

import os
import sys
import argparse

# 添加代码目录到路径
code_dir = os.path.join(os.path.dirname(__file__), '02_code')
sys.path.insert(0, code_dir)
os.chdir(code_dir)

from model_predict_optimized import ActionPredictorFast
from config_loader import load_config

def find_best_model():
    """自动查找最佳模型"""
    config = load_config("../05_config/config.yaml")
    trained_models_dir = config.get('paths', 'trained_models')
    
    if not os.path.exists(trained_models_dir):
        return None
    
    model_files = [f for f in os.listdir(trained_models_dir) if f.endswith('.pth')]
    if not model_files:
        return None
    
    # 优先找best_model，否则找最新的
    for name in ['best_model.pth', 'final_model.pth']:
        if name in model_files:
            return os.path.join(trained_models_dir, name)
    
    return os.path.join(trained_models_dir, sorted(model_files)[-1])

def main():
    parser = argparse.ArgumentParser(
        description='羽毛球视频高速分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
速度/精度权衡建议:
  stride=2:  最高精度，最慢速度 (约10秒/窗口)
  stride=8:  平衡模式，速度提升4倍 (推荐)
  stride=16: 最高速度，速度提升8倍 (约1.2秒/窗口)
        """
    )
    parser.add_argument('video', help='视频文件路径')
    parser.add_argument('--model', help='模型路径（默认自动查找）')
    parser.add_argument('--stride', type=int, default=8, 
                       help='滑动窗口步长 (默认: 8, 可选: 2/4/8/16)')
    parser.add_argument('--save', action='store_true', help='保存预测结果到JSON')
    parser.add_argument('--rounds', action='store_true', help='同时提取回合信息')
    
    args = parser.parse_args()
    
    # 检查视频
    video_path = args.video
    if not os.path.exists(video_path):
        # 尝试在raw_videos目录查找
        config = load_config("../05_config/config.yaml")
        raw_dir = config.get('paths', 'raw_videos')
        alt_path = os.path.join(raw_dir, args.video)
        if os.path.exists(alt_path):
            video_path = alt_path
        else:
            print(f"错误: 找不到视频文件: {args.video}")
            return
    
    # 查找模型
    model_path = args.model or find_best_model()
    if not model_path or not os.path.exists(model_path):
        print("错误: 找不到模型文件，请先训练模型或指定模型路径")
        return
    
    print("="*70)
    print("羽毛球视频分析 - 高速模式")
    print("="*70)
    print(f"视频: {video_path}")
    print(f"模型: {model_path}")
    print(f"步长: {args.stride}")
    print("="*70)
    
    # 运行预测
    try:
        predictor = ActionPredictorFast(model_path)
        results = predictor.predict_video(video_path, save_predictions=args.save, 
                                         sliding_stride=args.stride)
        
        # 提取回合
        if args.rounds:
            print("\n" + "="*70)
            print("【回合提取】")
            print("="*70)
            rounds = predictor.extract_rounds(results)
            
            if rounds:
                print(f"\n共识别到 {len(rounds)} 个回合:")
                print("-"*70)
                for r in rounds:
                    print(f"回合 {r['round_id']}: {r['start_time']:.2f}s - {r['end_time']:.2f}s "
                          f"(时长: {r['duration']:.2f}s) [{r['confidence']:.1%}]")
        
        print("\n" + "="*70)
        print("分析完成!")
        print("="*70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
