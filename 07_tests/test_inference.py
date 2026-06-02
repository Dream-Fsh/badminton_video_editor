"""
推理测试脚本
功能：加载训练好的最优模型，对一个示例视频进行动作识别并打印结果。
"""

import sys
import os
from pathlib import Path

# 添加代码目录到路径
sys.path.append(str(Path(__file__).parent.parent / '02_code'))

from model_predict import ActionPredictor
from config_loader import load_config

def main():
    # 1. 设置路径
    project_root = Path(__file__).parent.parent
    model_path = project_root / "03_model" / "trained" / "best_model_20260318_192512.pth"
    video_path = project_root / "01_data" / "raw_videos" / "match_001.mp4"
    config_path = project_root / "05_config" / "config.yaml"
    
    print(f"--- 启动推理测试 ---")
    print(f"模型路径: {model_path}")
    print(f"视频路径: {video_path}")
    
    # 2. 初始化预测器
    # 注意：model_predict.py 中的 ActionPredictor 构造函数会加载配置
    try:
        predictor = ActionPredictor(str(model_path), config_path=str(config_path))
    except Exception as e:
        print(f"[FAIL] 初始化预测器失败: {e}")
        return

    # 3. 执行视频识别
    if not video_path.exists():
        print(f"[FAIL] 视频文件不存在: {video_path}")
        return
        
    print(f"\n[INFO] 正在分析视频，这可能需要一点时间（取决于视频长度和CPU性能）...")
    try:
        results = predictor.predict_video(str(video_path), save_predictions=True)
    except Exception as e:
        print(f"[FAIL] 预测过程发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 展示识别到的动作
    print("\n" + "="*50)
    print(f" 识别结果汇总 - {video_path.name}")
    print("="*50)
    
    if not results:
        print("未识别到任何符合置信度阈值的动作。")
    else:
        print(f"共识别到 {len(results)} 个关键动作：")
        print(f"{'时间戳(s)':<12} | {'动作类型':<15} | {'置信度':<8}")
        print("-" * 45)
        for res in results:
            print(f"{res['timestamp']:<12.2f} | {res['class_name']:<15} | {res['confidence']:<8.2f}")
    
    print("="*50)
    print(f"\n预测结果已保存至 04_output/predictions/ 目录下。")

if __name__ == "__main__":
    main()
