"""
羽毛球视频自动剪辑系统 - 主程序
功能：整合数据预处理、模型训练、预测和视频剪辑的完整流程

作者：本科毕业设计
日期：2024
"""

import os
import sys
from pathlib import Path

# 获取项目根目录（main.py 在 02_code/ 下，需要回到上级目录）
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)  # 切换工作目录到项目根目录

# 兼容直接运行和模块导入
try:
    from config_loader import load_config
except ImportError:
    from .config_loader import load_config


def safe_print(msg):
    """安全打印函数，处理Windows编码问题"""
    try:
        print(msg)
    except UnicodeEncodeError:
        # 如果打印失败，尝试替换特殊字符
        safe_msg = msg.replace('✓', '[OK]').replace('✗', '[FAIL]').replace('⚠', '[WARN]')
        try:
            print(safe_msg)
        except:
            # 最后的退路：仅打印ASCII字符
            print(msg.encode('ascii', 'replace').decode('ascii'))

def print_banner():
    """打印系统标题"""
    safe_print("\n" + "=" * 70)
    safe_print(" " * 15 + "羽毛球视频自动剪辑系统")
    safe_print(" " * 10 + "基于I3D深度学习的动作识别与视频分割")
    safe_print("=" * 70)


def print_menu():
    """打印主菜单"""
    safe_print("\n请选择功能：")
    safe_print("  [1] 数据预处理 - 视频拆帧和标注处理")
    safe_print("  [2] 模型训练 - 训练I3D动作识别模型")
    safe_print("  [3] 模型预测 - 识别视频中的关键动作")
    safe_print("  [4] 视频剪辑 - 自动分割回合片段")
    safe_print("  [5] 完整流程 - 执行预测+剪辑")
    safe_print("  [0] 退出系统")
    safe_print("-" * 70)


def data_preprocessing():
    """数据预处理流程"""
    safe_print("\n" + "=" * 70)
    safe_print("【数据预处理】")
    safe_print("=" * 70)
    
    from data_preprocess import VideoPreprocessor
    
    preprocessor = VideoPreprocessor()
    
    safe_print("\n请选择操作：")
    safe_print("  [1] 批量提取视频帧")
    safe_print("  [2] 合并标注文件")
    safe_print("  [3] 验证标注文件")
    safe_print("  [4] 执行全部操作")
    
    choice = input("\n请输入选项 [1-4]: ").strip()
    
    if choice == '1':
        num_videos = preprocessor.process_all_videos()
        if num_videos > 0:
            safe_print(f"\n✓ 成功处理 {num_videos} 个视频")
    
    elif choice == '2':
        merged_data = preprocessor.merge_annotations()
        safe_print(f"\n✓ 成功合并标注，共 {len(merged_data)} 条")
    
    elif choice == '3':
        is_valid = preprocessor.validate_annotations()
        if is_valid:
            safe_print("\n✓ 标注文件验证通过")
        else:
            safe_print("\n✗ 标注文件存在问题")
    
    elif choice == '4':
        # 执行全部操作
        safe_print("\n步骤1: 提取视频帧")
        num_videos = preprocessor.process_all_videos()
        
        safe_print("\n步骤2: 合并标注文件")
        merged_data = preprocessor.merge_annotations()
        
        safe_print("\n步骤3: 验证标注文件")
        is_valid = preprocessor.validate_annotations()
        
        if is_valid:
            safe_print("\n✓ 数据预处理全部完成！")
        else:
            safe_print("\n✗ 标注文件存在问题，请检查")
    
    else:
        safe_print("无效选项")


def model_training():
    """模型训练流程"""
    safe_print("\n" + "=" * 70)
    safe_print("【模型训练】")
    safe_print("=" * 70)
    
    # 检查数据是否准备好
    config = load_config()
    label_path = config.get('paths', 'merged_labels')
    
    if not os.path.exists(label_path):
        safe_print("\n✗ 标注文件不存在，请先执行数据预处理")
        return
    
    print("\n开始训练模型...")
    print("提示：训练过程可能需要较长时间，请耐心等待")
    print("-" * 70)
    
    # 导入并运行训练脚本
    import model_train
    model_train.main()


def model_prediction():
    """模型预测流程"""
    print("\n" + "=" * 70)
    print("【模型预测】")
    print("=" * 70)
    
    config = load_config()
    
    # 检查是否有训练好的模型
    trained_models_dir = config.get('paths', 'trained_models')
    if not os.path.exists(trained_models_dir):
        print("\n✗ 未找到训练好的模型，请先执行模型训练")
        return
    
    model_files = [f for f in os.listdir(trained_models_dir) if f.endswith('.pth')]
    if not model_files:
        print("\n✗ 未找到训练好的模型文件，请先执行模型训练")
        return
    
    # 选择模型
    print("\n可用的模型：")
    for i, model_file in enumerate(model_files, 1):
        print(f"  [{i}] {model_file}")
    
    model_idx = input(f"\n请选择模型 [1-{len(model_files)}] (默认使用最新): ").strip()
    
    if model_idx.isdigit() and 1 <= int(model_idx) <= len(model_files):
        model_file = model_files[int(model_idx) - 1]
    else:
        model_file = sorted(model_files)[-1]  # 使用最新的
    
    model_path = os.path.join(trained_models_dir, model_file)
    print(f"\n使用模型: {model_file}")
    
    # 选择视频
    raw_videos_dir = config.get('paths', 'raw_videos')
    video_files = [
        f for f in os.listdir(raw_videos_dir)
        if Path(f).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    ]
    
    if not video_files:
        print(f"\n✗ 未找到视频文件，请将视频放入: {raw_videos_dir}")
        return
    
    print("\n可用的视频：")
    for i, video_file in enumerate(video_files, 1):
        print(f"  [{i}] {video_file}")
    print(f"  [0] 处理所有视频")
    
    video_idx = input(f"\n请选择视频 [0-{len(video_files)}]: ").strip()
    
    # 选择预测模式
    print("\n选择预测模式:")
    print("  [1] 高速模式 (批量推理，速度提升5-10倍，推荐)")
    print("  [2] 标准模式 (原始版本，精度略高但慢)")
    mode = input("请选择 [1-2] (默认: 1): ").strip() or "1"
    
    if mode == "2":
        from model_predict import ActionPredictor
        predictor = ActionPredictor(model_path)
    else:
        from model_predict_optimized import ActionPredictorFast as ActionPredictor
        predictor = ActionPredictor(model_path)
        print("\n使用高速模式 (stride=8, batch推理)")
    
    if video_idx == '0':
        # 处理所有视频
        for video_file in video_files:
            video_path = os.path.join(raw_videos_dir, video_file)
            try:
                if mode == "2":
                    full_results = predictor.predict_video(video_path)
                else:
                    full_results = predictor.predict_video(video_path, sliding_stride=8)
                # predict_video 返回完整结果（含action_predictions + motion_data + rounds）
                rounds = full_results.get('rounds', [])
                print(f"\n✓ {video_file}: 识别到 {len(rounds)} 个回合")
            except Exception as e:
                print(f"\n✗ 处理失败 {video_file}: {e}")
    
    elif video_idx.isdigit() and 1 <= int(video_idx) <= len(video_files):
        # 处理单个视频
        video_file = video_files[int(video_idx) - 1]
        video_path = os.path.join(raw_videos_dir, video_file)
        
        if mode == "2":
            full_results = predictor.predict_video(video_path)
        else:
            full_results = predictor.predict_video(video_path, sliding_stride=8)
        # predict_video 返回完整结果（含action_predictions + motion_data + rounds）
        rounds = full_results.get('rounds', [])
        
        # 显示回合信息
        print("\n识别到的回合：")
        print("-" * 70)
        for r in rounds:
            print(f"回合 {r['round_id']}: "
                  f"{r['start_time']:.2f}s - {r['end_time']:.2f}s "
                  f"(时长: {r['duration']:.2f}s, start_method={r.get('start_method','?')})")
    
    else:
        print("无效选项")


def video_editing():
    """视频剪辑流程"""
    print("\n" + "=" * 70)
    print("【视频剪辑】")
    print("=" * 70)
    
    config = load_config()
    
    # 检查是否有预测结果
    predictions_dir = config.get('paths', 'output_predictions')
    if not os.path.exists(predictions_dir):
        print("\n✗ 未找到预测结果，请先执行模型预测")
        return
    
    prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('_predictions.json')]
    if not prediction_files:
        print("\n✗ 未找到预测结果文件，请先执行模型预测")
        return
    
    print(f"\n找到 {len(prediction_files)} 个预测结果文件")
    
    from video_editor import VideoEditor
    editor = VideoEditor()
    
    # 批量剪辑
    video_dir = config.get('paths', 'raw_videos')
    num_videos = editor.batch_extract_rounds(video_dir, predictions_dir)
    
    print(f"\n✓ 成功处理 {num_videos} 个视频")


def full_pipeline():
    """完整流程：预测 + 剪辑"""
    print("\n" + "=" * 70)
    print("【完整流程】预测 + 剪辑")
    print("=" * 70)
    
    print("\n步骤1: 模型预测")
    print("-" * 70)
    model_prediction()
    
    print("\n步骤2: 视频剪辑")
    print("-" * 70)
    video_editing()
    
    print("\n✓ 完整流程执行完毕！")


def main():
    """主函数"""
    print_banner()
    
    while True:
        print_menu()
        choice = input("请输入选项 [0-5]: ").strip()
        
        try:
            if choice == '1':
                data_preprocessing()
            
            elif choice == '2':
                model_training()
            
            elif choice == '3':
                model_prediction()
            
            elif choice == '4':
                video_editing()
            
            elif choice == '5':
                full_pipeline()
            
            elif choice == '0':
                print("\n感谢使用！再见！")
                sys.exit(0)
            
            else:
                print("\n无效选项，请重新输入")
        
        except KeyboardInterrupt:
            print("\n\n操作已取消")
        
        except Exception as e:
            print(f"\n✗ 发生错误: {e}")
            import traceback
            traceback.print_exc()
        
        input("\n按回车键继续...")


if __name__ == "__main__":
    main()
