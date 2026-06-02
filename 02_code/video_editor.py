"""
视频剪辑模块
功能：
1. 根据预测结果自动分割视频回合
2. 使用FFmpeg进行高效视频剪辑
3. 批量导出回合片段
4. 绿色像素检测：剔除不含绿色比赛场地的误检回合
"""

import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from config_loader import load_config
from athlete_detector import GreenPixelDetector


class VideoEditor:
    """视频编辑器：基于FFmpeg实现视频剪辑功能"""
    
    def __init__(self, config_path="../05_config/config.yaml"):
        """
        初始化视频编辑器
        
        参数:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        
        # 加载剪辑配置
        self.pre_padding = self.config.get('video_editing', 'round_extraction', 'pre_padding')
        self.post_padding = self.config.get('video_editing', 'round_extraction', 'post_padding')
        self.min_duration = self.config.get('video_editing', 'round_extraction', 'min_round_duration')
        self.max_duration = self.config.get('video_editing', 'round_extraction', 'max_round_duration')
        
        # FFmpeg参数
        self.video_codec = self.config.get('video_editing', 'ffmpeg', 'video_codec')
        self.audio_codec = self.config.get('video_editing', 'ffmpeg', 'audio_codec')
        self.crf = self.config.get('video_editing', 'ffmpeg', 'crf')
        self.preset = self.config.get('video_editing', 'ffmpeg', 'preset')
        self.output_format = self.config.get('video_editing', 'ffmpeg', 'output_format')
        
        # 输出路径
        self.output_dir = self.config.get('paths', 'output_clips')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 绿色像素过滤配置 - 剔除不含绿色比赛场地的误检回合
        green_cfg = self.config.get('video_editing', 'green_pixel_filter', default={})
        self.green_filter_enabled = green_cfg.get('enabled', True)
        if self.green_filter_enabled:
            self.green_detector = GreenPixelDetector(
                hsv_lower=(green_cfg.get('hsv_lower_h', 35),
                           green_cfg.get('hsv_lower_s', 40),
                           green_cfg.get('hsv_lower_v', 40)),
                hsv_upper=(green_cfg.get('hsv_upper_h', 85),
                           green_cfg.get('hsv_upper_s', 255),
                           green_cfg.get('hsv_upper_v', 255)),
                min_green_ratio=green_cfg.get('min_green_ratio', 0.05),
                sample_rate=green_cfg.get('sample_rate', 30),
                required_green_frame_ratio=green_cfg.get('required_green_frame_ratio', 0.3)
            )
            print(f"  [GREEN_FILTER] 启用绿色像素过滤: HSV({green_cfg.get('hsv_lower_h',35)},{green_cfg.get('hsv_lower_s',40)},{green_cfg.get('hsv_lower_v',40)})"
                  f"-({green_cfg.get('hsv_upper_h',85)},{green_cfg.get('hsv_upper_s',255)},{green_cfg.get('hsv_upper_v',255)})")
        
        # 检查FFmpeg是否可用
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查系统是否安装FFmpeg"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("✓ FFmpeg 已安装")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("✗ 未检测到FFmpeg")
            print("  请确保FFmpeg已安装并添加到系统PATH")
            print("  下载地址: https://ffmpeg.org/download.html")
            # 不抛出异常，允许程序继续运行，但会在实际使用时报错
            return False
    
    def cut_video_segment(self, input_video, start_time, end_time, output_path):
        """
        剪辑视频片段
        
        参数:
            input_video: 输入视频路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            output_path: 输出路径
        
        返回:
            是否成功
        """
        # 添加缓冲时间 (已按用户要求修改: 前10s, 后10s)
        actual_start = max(0, start_time - self.pre_padding)
        actual_end = end_time + self.post_padding
        duration = actual_end - actual_start
        
        print(f"  正在剪辑: {start_time:.2f}s - {end_time:.2f}s (应用缓冲后: {actual_start:.2f}s - {actual_end:.2f}s, 时长: {duration:.2f}s)")
        
        # 检查时长是否合理
        if duration < self.min_duration:
            print(f"  警告: 片段时长过短 ({duration:.2f}s < {self.min_duration}s)，跳过")
            return False
        
        if duration > self.max_duration:
            print(f"  警告: 片段时长过长 ({duration:.2f}s > {self.max_duration}s)，截断")
            duration = self.max_duration
        
        # 构建FFmpeg命令
        cmd = [
            'ffmpeg',
            '-y',
            '-ss', str(actual_start),
            '-t', str(duration),
            '-i', input_video,
            '-c:v', self.video_codec,
            '-crf', str(self.crf),
            '-preset', self.preset,
            '-c:a', self.audio_codec,
            output_path
        ]
        
        try:
            # 执行FFmpeg命令（隐藏输出）
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                return True
            else:
                print(f"  FFmpeg错误: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  剪辑超时")
            return False
        except Exception as e:
            print(f"  剪辑失败: {e}")
            return False
    
    def extract_rounds_from_predictions(self, video_path, predictions_path):
        """
        根据预测结果提取所有回合片段
        
        参数:
            video_path: 原始视频路径
            predictions_path: 预测结果JSON文件路径
        
        返回:
            成功剪辑的片段数量
        """
        print(f"\n开始剪辑视频: {Path(video_path).name}")
        
        # 读取预测结果
        if not os.path.exists(predictions_path):
            raise FileNotFoundError(f"预测结果文件不存在: {predictions_path}")
        
        with open(predictions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 支持三种可能的 JSON 结构:
        # 1. 'rounds' (model_predict_optimized 生成，含YOLO增强检测) ← 优先使用
        # 2. 'results' -> 'predictions' (model_predict 生成)
        # 3. 'predictions' (旧版本)
        if 'rounds' in data and data['rounds']:
            rounds = data['rounds']
            print(f"  使用增强回合数据 (含YOLO就位检测): {len(rounds)} 个回合")
        else:
            predictions = data.get('results', data.get('predictions', []))
            if not predictions:
                print("  警告: 预测结果为空")
                return 0
            # 提取回合信息
            rounds = self._extract_rounds_from_predictions(predictions)
        
        if not rounds:
            print("  警告: 未找到有效回合 (需要同时识别到发球和落地才能组成回合)")
            return 0
        
        print(f"  找到 {len(rounds)} 个回合")
        
        # 创建视频专属输出目录
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # 逐个剪辑回合
        success_count = 0
        
        for round_info in tqdm(rounds, desc="剪辑回合"):
            round_id = round_info['round_id']
            start_time = round_info['start_time']
            end_time = round_info['end_time']
            
            # ===== 绿色像素过滤：剔除不含绿色比赛场地的误检回合 =====
            if self.green_filter_enabled:
                has_green = self.green_detector.check_video_segment(
                    video_path, start_time, end_time, verbose=True
                )
                if not has_green:
                    round_info['filtered'] = True
                    round_info['filter_reason'] = 'no_green_pixels'
                    print(f"  [FILTERED] Round {round_id}: 画面不含绿色比赛场地，已剔除")
                    continue
            
            # 输出文件名
            output_filename = f"{video_name}_round_{round_id:03d}.{self.output_format}"
            output_path = os.path.join(video_output_dir, output_filename)
            
            # 剪辑片段
            success = self.cut_video_segment(video_path, start_time, end_time, output_path)
            
            if success:
                success_count += 1
                # 记录片段信息
                round_info['output_path'] = output_path
                round_info['file_size'] = os.path.getsize(output_path)
        
        # 保存剪辑信息
        self._save_clip_info(video_output_dir, video_name, rounds)
        
        print(f"\n✓ 剪辑完成！成功: {success_count}/{len(rounds)}")
        print(f"  输出目录: {video_output_dir}")
        
        return success_count
    
    def _extract_rounds_from_predictions(self, predictions):
        """
        从预测结果中提取回合信息
        
        参数:
            predictions: 预测结果列表
        
        返回:
            回合列表
        """
        rounds = []
        round_id = 1
        
        # 分离发球和落地动作
        starts = [p for p in predictions if p['class_name'] == 'round_start']
        ends = [p for p in predictions if p['class_name'] == 'round_end']
        
        # 配对发球和落地
        for start in starts:
            # 找到该发球后的第一个落地
            matching_end = None
            for end in ends:
                if end['timestamp'] > start['timestamp']:
                    matching_end = end
                    break
            
            if matching_end:
                duration = matching_end['timestamp'] - start['timestamp']
                
                # 过滤不合理的回合
                if self.min_duration <= duration <= self.max_duration:
                    rounds.append({
                        'round_id': round_id,
                        'start_time': start['timestamp'],
                        'end_time': matching_end['timestamp'],
                        'duration': duration,
                        'start_confidence': start['confidence'],
                        'end_confidence': matching_end['confidence']
                    })
                    round_id += 1
        
        return rounds
    
    def _save_clip_info(self, output_dir, video_name, rounds):
        """
        保存剪辑信息为JSON文件
        
        参数:
            output_dir: 输出目录
            video_name: 视频名称
            rounds: 回合信息列表
        """
        info_path = os.path.join(output_dir, f"{video_name}_clips_info.json")
        
        info_data = {
            'video_name': video_name,
            'clip_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_rounds': len(rounds),
            'rounds': rounds
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, ensure_ascii=False, indent=2)
        
        print(f"  剪辑信息已保存: {info_path}")
    
    def batch_extract_rounds(self, video_dir, predictions_dir):
        """
        批量处理多个视频的回合提取
        
        参数:
            video_dir: 视频目录
            predictions_dir: 预测结果目录
        
        返回:
            处理的视频数量
        """
        print("\n" + "=" * 60)
        print("批量剪辑视频回合")
        print("=" * 60)
        
        # 查找所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = [
            f for f in os.listdir(video_dir)
            if Path(f).suffix.lower() in video_extensions
        ]
        
        if not video_files:
            print(f"未找到视频文件: {video_dir}")
            return 0
        
        print(f"\n找到 {len(video_files)} 个视频文件")
        
        total_clips = 0
        
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            video_name = Path(video_file).stem
            
            # 查找对应的预测结果
            predictions_file = f"{video_name}_predictions.json"
            predictions_path = os.path.join(predictions_dir, predictions_file)
            
            if not os.path.exists(predictions_path):
                print(f"\n✗ 未找到预测结果: {predictions_file}")
                continue
            
            # 剪辑视频
            try:
                num_clips = self.extract_rounds_from_predictions(video_path, predictions_path)
                total_clips += num_clips
            except Exception as e:
                print(f"\n✗ 处理失败 {video_file}: {e}")
                continue
        
        print("\n" + "=" * 60)
        print(f"✓ 批量剪辑完成！共生成 {total_clips} 个片段")
        print("=" * 60)
        
        return len(video_files)
    
    def merge_clips(self, clip_paths, output_path):
        """
        合并多个视频片段为一个视频
        
        参数:
            clip_paths: 片段路径列表
            output_path: 输出路径
        
        返回:
            是否成功
        """
        if not clip_paths:
            print("没有片段需要合并")
            return False
        
        print(f"\n合并 {len(clip_paths)} 个片段...")
        
        # 创建临时文件列表
        list_file = os.path.join(os.path.dirname(output_path), 'concat_list.txt')
        
        try:
            # 写入文件列表
            with open(list_file, 'w', encoding='utf-8') as f:
                for clip_path in clip_paths:
                    # FFmpeg concat需要转义路径
                    escaped_path = clip_path.replace('\\', '/').replace("'", "\\'")
                    f.write(f"file '{escaped_path}'\n")
            
            # FFmpeg合并命令
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c', 'copy',  # 直接复制流，不重新编码
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                print(f"✓ 合并成功: {output_path}")
                return True
            else:
                print(f"✗ 合并失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ 合并失败: {e}")
            return False
        finally:
            # 删除临时文件
            if os.path.exists(list_file):
                os.remove(list_file)


def main():
    """主函数：演示视频剪辑流程"""
    print("=" * 60)
    print("羽毛球视频自动剪辑系统 - 视频剪辑模块")
    print("=" * 60)
    
    # 初始化编辑器
    editor = VideoEditor()
    
    # 配置路径
    config = load_config()
    video_dir = config.get('paths', 'raw_videos')
    predictions_dir = config.get('paths', 'output_predictions')
    
    # 批量剪辑
    editor.batch_extract_rounds(video_dir, predictions_dir)
    
    print("\n✓ 所有视频剪辑完成！")


if __name__ == "__main__":
    main()
