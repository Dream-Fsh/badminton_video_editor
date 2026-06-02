"""
速度对比测试脚本
对比原始版本 vs 优化版本的预测速度
"""

import os
import sys
import time

# 添加代码目录到路径
code_dir = os.path.join(os.path.dirname(__file__), '02_code')
sys.path.insert(0, code_dir)

def test_original(video_path, model_path):
    """测试原始版本"""
    print("\n" + "="*70)
    print("【测试原始版本】")
    print("="*70)
    
    from model_predict import ActionPredictor
    
    start_time = time.time()
    predictor = ActionPredictor(model_path)
    load_time = time.time() - start_time
    print(f"模型加载时间: {load_time:.2f}s")
    
    start_time = time.time()
    results = predictor.predict_video(video_path, save_predictions=False)
    predict_time = time.time() - start_time
    
    print(f"\n原始版本总时间: {predict_time:.2f}s")
    return predict_time, results

def test_optimized(video_path, model_path, stride=8):
    """测试优化版本"""
    print("\n" + "="*70)
    print(f"【测试优化版本 - stride={stride}】")
    print("="*70)
    
    from model_predict_optimized import ActionPredictorFast
    
    start_time = time.time()
    predictor = ActionPredictorFast(model_path)
    load_time = time.time() - start_time
    print(f"模型加载时间: {load_time:.2f}s")
    
    start_time = time.time()
    results = predictor.predict_video(video_path, save_predictions=False, sliding_stride=stride)
    predict_time = time.time() - start_time
    
    print(f"\n优化版本总时间 (stride={stride}): {predict_time:.2f}s")
    return predict_time, results

def compare_results(results1, results2, name1="原始版本", name2="优化版本"):
    """对比两个版本的结果"""
    print("\n" + "="*70)
    print("【结果对比】")
    print("="*70)
    
    pred1 = results1.get('action_predictions', [])
    pred2 = results2.get('action_predictions', [])
    
    starts1 = [p for p in pred1 if p['class_name'] == 'round_start']
    starts2 = [p for p in pred2 if p['class_name'] == 'round_start']
    ends1 = [p for p in pred1 if p['class_name'] == 'round_end']
    ends2 = [p for p in pred2 if p['class_name'] == 'round_end']
    
    print(f"{name1}:")
    print(f"  检测点总数: {len(pred1)}")
    print(f"  round_start: {len(starts1)}")
    print(f"  round_end: {len(ends1)}")
    
    print(f"\n{name2}:")
    print(f"  检测点总数: {len(pred2)}")
    print(f"  round_start: {len(starts2)}")
    print(f"  round_end: {len(ends2)}")
    
    # 对比时间戳
    if starts1 and starts2:
        times1 = sorted([p['timestamp'] for p in starts1])
        times2 = sorted([p['timestamp'] for p in starts2])
        print(f"\n开始点时间对比:")
        print(f"  {name1}: {[f'{t:.2f}' for t in times1[:5]]}...")
        print(f"  {name2}: {[f'{t:.2f}' for t in times2[:5]]}...")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='速度对比测试')
    parser.add_argument('video', help='测试视频路径')
    parser.add_argument('--model', default='03_model/trained/best_model.pth', help='模型路径')
    parser.add_argument('--stride', type=int, default=8, help='优化版本的滑动步长')
    parser.add_argument('--skip-original', action='store_true', help='跳过原始版本测试（节省时间）')
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"错误: 视频不存在: {args.video}")
        return
    
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在: {model_path}")
        return
    
    print("\n" + "="*70)
    print("羽毛球视频分析 - 速度对比测试")
    print("="*70)
    print(f"视频: {args.video}")
    print(f"模型: {model_path}")
    
    results_original = None
    results_optimized = None
    time_original = None
    time_optimized = None
    
    # 测试原始版本
    if not args.skip_original:
        try:
            time_original, results_original = test_original(args.video, model_path)
        except Exception as e:
            print(f"\n原始版本测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n跳过原始版本测试")
    
    # 测试优化版本
    try:
        time_optimized, results_optimized = test_optimized(args.video, model_path, args.stride)
    except Exception as e:
        print(f"\n优化版本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 对比结果
    if results_original and results_optimized:
        compare_results(results_original, results_optimized)
        
        print("\n" + "="*70)
        print("【速度提升总结】")
        print("="*70)
        if time_original:
            speedup = time_original / time_optimized
            print(f"原始版本: {time_original:.2f}s")
            print(f"优化版本: {time_optimized:.2f}s")
            print(f"速度提升: {speedup:.1f}x")
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)

if __name__ == "__main__":
    # 如果没有参数，使用默认测试视频
    import sys
    if len(sys.argv) == 1:
        # 自动查找测试视频
        test_video = "d:/Projects/python/badminton_video_editor/01_data/raw_videos/match_001.mp4"
        if os.path.exists(test_video):
            print(f"使用默认测试视频: {test_video}")
            sys.argv.append(test_video)
            sys.argv.append("--skip-original")  # 默认跳过原始版，节省时间
            sys.argv.append("--stride")
            sys.argv.append("8")
        else:
            print("未找到默认测试视频，请指定视频路径")
    main()
