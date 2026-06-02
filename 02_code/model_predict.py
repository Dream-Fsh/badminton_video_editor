"""
模型预测模块
功能：
1. 加载训练好的I3D模型
2. 对新视频进行滑动窗口推理
3. 识别关键动作（发球、球落地）的时间戳
4. 生成回合分割信息
"""

import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# 兼容直接运行和模块导入
try:
    from config_loader import load_config
    from i3d import InceptionI3d, Unit3D
    from athlete_detector import AthleteDetector
except ImportError:
    from .config_loader import load_config
    from .i3d import InceptionI3d, Unit3D
    from .athlete_detector import AthleteDetector


class ActionPredictor:
    """动作预测器：使用训练好的模型识别视频中的关键动作"""
    
    def __init__(self, model_path, config_path="../05_config/config.yaml"):
        """
        初始化预测器
        """
        self.config = load_config(config_path)
        self.model_path = model_path
        
        # 加载配置参数
        self.sequence_length = self.config.get('training', 'sequence_length')
        self.num_classes = self.config.get('training', 'num_classes')
        self.confidence_threshold = self.config.get('prediction', 'confidence_threshold')
        self.sliding_stride = self.config.get('prediction', 'sliding_window_stride')
        self.batch_size = self.config.get('prediction', 'batch_size')
        self.frame_rate = self.config.get('preprocessing', 'frame_rate')
        # 模型输入尺寸
        # 原设计：I3D 标准输入为 224x224
        # 修改后：使用AdaptiveAvgPool，支持任意输入尺寸
        # 设为None表示使用ROI裁剪后的原始尺寸（950×720）
        self.model_input_size = None  # 可选：设为(224, 224)强制resize，或None保持原始尺寸
        
        # YOLO 运动员检测配置
        self.yolo_enabled = self.config.get('prediction', 'athlete_detection', 'enabled', default=False)
        if self.yolo_enabled:
            yolo_model = self.config.get('prediction', 'athlete_detection', 'model_type', default='yolov8n.pt')
            shuttle_cfg = self.config.get('prediction', 'athlete_detection', 'shuttlecock_detection', default={})
            shuttle_model = shuttle_cfg.get('model_path') if shuttle_cfg.get('enabled', False) else None
            self.athlete_detector = AthleteDetector(model_type=yolo_model, shuttlecock_model=shuttle_model)
            self.static_threshold = self.config.get('prediction', 'athlete_detection', 'static_threshold', default=5.0)
            self.yolo_sample_rate = self.config.get('prediction', 'athlete_detection', 'sample_rate', default=5)
            
            # 运动员检测约束配置
            self.athlete_constraints = self.config.get('prediction', 'athlete_detection', 'constraints', default={})
            self.min_player_count = self.athlete_constraints.get('min_player_count', 2)
            self.max_player_count = self.athlete_constraints.get('max_player_count', 2)
        
        # 后处理参数
        self.min_interval = self.config.get('prediction', 'post_processing', 'min_interval')
        self.smooth_window = self.config.get('prediction', 'post_processing', 'smooth_window')
        
        # ROI 裁剪配置（与训练时保持一致）
        roi_config = self.config.get('preprocessing', 'roi')
        self.roi_enabled = roi_config.get('enabled', False) if roi_config else False
        self.roi_x = roi_config.get('x_offset', 0) if roi_config else 0
        self.roi_y = roi_config.get('y_offset', 0) if roi_config else 0
        self.roi_w = roi_config.get('width', 950) if roi_config else 950
        self.roi_h = roi_config.get('height', 720) if roi_config else 720
        
        # 设备配置
        device_config = self.config.get('system', 'device')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = self._load_model()
        
        # 动作类别映射
        self.action_classes = self.config.get('action_classes')
    
    def _load_model(self):
        """加载训练好的I3D模型"""
        print(f"\nLoading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # 定义与训练时相同的模型结构
        class BadmintonI3D(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.model = InceptionI3d(num_classes=num_classes, in_channels=3)
                # 修改适配二分类任务 (与训练时一致)
                self.model.logits = Unit3D(in_channels=1024, output_channels=num_classes,
                                     kernel_shape=[1, 1, 1],
                                     activation_fn=None,
                                     use_batch_norm=False,
                                     use_bias=True,
                                     name='logits')
                self.model.end_points['Logits'] = self.model.logits

            def forward(self, x):
                logits = self.model(x)
                if len(logits.shape) == 5:
                    logits = logits.squeeze(-1).squeeze(-1)
                if len(logits.shape) == 3:
                    logits = torch.mean(logits, dim=2)
                return logits

        model = BadmintonI3D(num_classes=self.num_classes)
        
        # 加载权重
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        
        # 设置为评估模式
        model.to(self.device)
        model.eval()
        
        print("[OK] Model loaded successfully")
        return model
    
    def preprocess_frames(self, frames):
        """
        预处理帧序列，转换为模型输入格式
        """
        if len(frames) != self.sequence_length:
            raise ValueError(f"Frame count mismatch: expected {self.sequence_length}, got {len(frames)}")
        
        # 转换为numpy数组并归一化
        frames_np = np.stack(frames, axis=0)  # (T, H, W, 3)
        frames_np = frames_np.astype(np.float32) / 255.0
        
        # ImageNet标准化
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        frames_np = (frames_np - mean) / std
        
        # 转换为tensor并调整维度 (T, H, W, 3) -> (1, 3, T, H, W)
        frames_tensor = torch.from_numpy(frames_np).float()
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # (3, T, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)  # (1, 3, T, H, W)
        
        return frames_tensor
    
    def predict_video(self, video_path, save_predictions=True):
        """
        对整个视频进行动作识别
        """
        print(f"\nPredicting video: {Path(video_path).name}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        
        # 读取所有帧
        all_frames = []
        athlete_motions = []  # 记录每一帧运动员的静止状态 (仅当启用 YOLO 时)
        
        print("  Reading video frames and analyzing motion...")
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            frame_idx = 0
            # 用于检测静止状态的历史记录
            athlete1_history = []
            athlete2_history = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 记录原始帧以供 YOLO 使用 (每隔几帧检测一次以提速)
                if self.yolo_enabled and frame_idx % self.yolo_sample_rate == 0:
                    athletes = self.athlete_detector.detect_athletes(frame, constraints=self.athlete_constraints)
                    
                    is_static = False
                    if len(athletes) >= self.min_player_count:
                        # 简单的距离追踪，确保 athlete1_history 始终对应同一个人
                        
                        # 如果是第一次检测，直接初始化
                        if not athlete1_history or not athlete2_history:
                            athlete1_history.append(athletes[0]['center'])
                            athlete2_history.append(athletes[1]['center'])
                        else:
                            # 将当前检测到的两个点与上一帧的两个点进行距离匹配
                            p1_last = athlete1_history[-1]
                            p2_last = athlete2_history[-1]
                            
                            # 计算四种组合的距离
                            d11 = np.linalg.norm(np.array(athletes[0]['center']) - np.array(p1_last))
                            d12 = np.linalg.norm(np.array(athletes[1]['center']) - np.array(p1_last))
                            
                            # 如果 athletes[1] 离 p1 更近，说明顺序反了
                            if d12 < d11:
                                athlete1_history.append(athletes[1]['center'])
                                athlete2_history.append(athletes[0]['center'])
                            else:
                                athlete1_history.append(athletes[0]['center'])
                                athlete2_history.append(athletes[1]['center'])
                        
                        # 保持历史窗口大小
                        window_size = self.config.get('prediction', 'athlete_detection', 'window_size', default=15)
                        if len(athlete1_history) > window_size:
                            athlete1_history.pop(0)
                        if len(athlete2_history) > window_size:
                            athlete2_history.pop(0)
                            
                        # 只有当两个人的历史记录都足够时，才进行"双方同时静止"判定
                        if len(athlete1_history) >= window_size and len(athlete2_history) >= window_size:
                            static1 = self.athlete_detector.analyze_motion(athlete1_history, self.static_threshold)
                            static2 = self.athlete_detector.analyze_motion(athlete2_history, self.static_threshold)
                            # 严格要求：双方都必须静止 (Static1 AND Static2)
                            is_static = static1 and static2
                    else:
                        # 如果画面中少于两人，无法判定"双方静止"，重置历史以防误判
                        athlete1_history = []
                        athlete2_history = []
                    
                    athlete_motions.append({
                        'frame_idx': frame_idx,
                        'is_static': bool(is_static),  # 强制转换为 Python 原生 bool，解决 JSON 序列化问题
                        'timestamp': frame_idx / fps,
                        'valid_players': len(athletes) >= self.min_player_count,  # 标记是否检测到足够运动员
                        'player_count': len(athletes),
                        'players': athletes
                    })
                
                # ROI 裁剪（与训练时保持一致）
                if self.roi_enabled:
                    h, w = frame.shape[:2]
                    x_end = min(self.roi_x + self.roi_w, w)
                    y_end = min(self.roi_y + self.roi_h, h)
                    frame = frame[self.roi_y:y_end, self.roi_x:x_end]
                
                # 转换为RGB供 I3D 使用
                # 如果设置了目标尺寸，则resize；否则保持原始尺寸
                if self.model_input_size is not None:
                    frame = cv2.resize(frame, self.model_input_size)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame_rgb)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # 滑动窗口预测
        predictions = []
        num_windows = (len(all_frames) - self.sequence_length) // self.sliding_stride + 1
        
        print(f"  Sliding window prediction (window={self.sequence_length}, stride={self.sliding_stride})...")
        
        with torch.no_grad():
            for i in tqdm(range(num_windows), desc="Predicting"):
                start_idx = i * self.sliding_stride
                end_idx = start_idx + self.sequence_length
                
                if end_idx > len(all_frames):
                    break
                
                frame_sequence = all_frames[start_idx:end_idx]
                input_tensor = self.preprocess_frames(frame_sequence)
                input_tensor = input_tensor.to(self.device)
                
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
                
                predictions.append({
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'center_frame': (start_idx + end_idx) // 2,
                    'predicted_class': int(predicted_class.item()),
                    'class_name': self.action_classes[int(predicted_class.item())],
                    'confidence': float(confidence.item()),
                    'timestamp': (start_idx + end_idx) / 2 / fps
                })
        
        filtered_predictions = self._post_process(predictions)
        
        # 将 YOLO 识别到的静止状态信息也存入结果
        final_results = {
            'action_predictions': filtered_predictions,
            'motion_data': athlete_motions,
            'total_duration': total_frames / fps if fps > 0 else 0
        }
        
        print(f"\n[OK] Prediction finished")
        print(f"  Original count: {len(predictions)}")
        print(f"  Filtered count: {len(filtered_predictions)}")
        
        if save_predictions:
            self._save_results(video_path, final_results)
            
        return final_results

    def extract_rounds(self, results):
        """
        智能回合提取算法 V2.0
        逻辑：
        1. YOLO 检测静→动作为回合开始触发点
        2. I3D 检测球落地(round_end)作为回合结束
        3. 如果未检测到球落地，则以下一回合开始前的静止作为结束
        """
        if isinstance(results, dict):
            predictions = results.get('action_predictions', [])
            motion_data = results.get('motion_data', [])
            total_duration = results.get('total_duration', 3600.0)
        else:
            predictions = results
            motion_data = []
            total_duration = 3600.0

        rounds = []
        round_id = 1
        
        # 加载回合约束配置，置信度
        round_constraints = self.config.get('prediction', 'round_constraints', default={})
        MIN_START_INTERVAL = round_constraints.get('min_start_interval', 3.0)
        MIN_ROUND_DURATION = round_constraints.get('min_round_duration', 2.0)
        MAX_ROUND_DURATION = round_constraints.get('max_round_duration', 30.0)
        MERGE_INTERVAL = round_constraints.get('merge_interval', 2.0)
        MAX_END_SEARCH_TIME = round_constraints.get('max_round_end_search_time', 25.0)
        PREFER_I3D_END = round_constraints.get('prefer_i3d_end', True)
        FALLBACK_TO_MOTION = round_constraints.get('fallback_to_motion_stop', True)
        confidence_weights = round_constraints.get('confidence_weights', {})
        i3d_end_confidence = confidence_weights.get('i3d_round_end', 0.9)
        motion_stop_confidence = confidence_weights.get('motion_stop', 0.7)
        estimated_confidence = confidence_weights.get('estimated', 0.5)
        
        # 提取所有 I3D 检测到的动作点
        i3d_starts = sorted([{'t': p['timestamp'], 'f': p['center_frame'], 'conf': p['confidence']} 
                             for p in predictions if p['class_name'] == 'round_start'], key=lambda x: x['t'])
        i3d_ends = sorted([{'t': p['timestamp'], 'f': p['center_frame'], 'conf': p['confidence']} 
                           for p in predictions if p['class_name'] == 'round_end'], key=lambda x: x['t'])
        
        print(f"  I3D detected: {len(i3d_starts)} round_starts, {len(i3d_ends)} round_ends")
        
        # --- 第一阶段：YOLO 检测回合开始点 ---
        round_start_points = []

        # ====== 最高优先级：左右站位规则（两球员分居两侧且静止）======
        ready_stance_cfg = round_constraints.get('ready_stance_detection', {})
        ready_stance_enabled = ready_stance_cfg.get('enabled', False)
        stance_starts = []

        if ready_stance_enabled and self.yolo_enabled and motion_data:
            min_x_separation = ready_stance_cfg.get('min_x_separation', 150.0)
            min_static_duration = ready_stance_cfg.get('min_static_duration', 1.0)
            ready_confidence = ready_stance_cfg.get('min_confidence', 0.6)

            sample_rate = self.yolo_sample_rate
            fps_val = self.frame_rate if hasattr(self, 'frame_rate') else 30
            min_static_points = max(2, int(min_static_duration * fps_val / sample_rate))

            # 调试统计
            debug_total_2players = 0
            debug_static_2players = 0
            debug_x_separations = []

            consecutive_count = 0
            stance_start_candidate = None

            for i in range(len(motion_data)):
                m = motion_data[i]
                player_count = m.get('player_count', 0)
                is_static = m.get('is_static', False)

                if player_count >= 2:
                    debug_total_2players += 1
                    players = m.get('players', [])
                    if len(players) >= 2:
                        x1 = players[0]['center'][0]
                        x2 = players[1]['center'][0]
                        x_sep = abs(x1 - x2)
                        debug_x_separations.append(x_sep)

                        if is_static:
                            debug_static_2players += 1
                            if x_sep >= min_x_separation:
                                consecutive_count += 1
                                if consecutive_count == 1:
                                    stance_start_candidate = i
                                if consecutive_count >= min_static_points:
                                    candidate_t = motion_data[stance_start_candidate]['timestamp']
                                    candidate_f = motion_data[stance_start_candidate]['frame_idx']
                                    stance_starts.append({
                                        'timestamp': candidate_t,
                                        'frame': candidate_f,
                                        'method': 'yolo_ready_stance',
                                        'confidence': ready_confidence
                                    })
                                    consecutive_count = 0
                                    stance_start_candidate = None
                                continue
                consecutive_count = 0
                stance_start_candidate = None

            # 调试输出
            if debug_total_2players > 0:
                avg_x_sep = sum(debug_x_separations) / len(debug_x_separations) if debug_x_separations else 0
                max_x_sep = max(debug_x_separations) if debug_x_separations else 0
                print(f"  [STANCE_DEBUG] 2-players frames: {debug_total_2players}, "
                      f"static frames: {debug_static_2players}, "
                      f"x_sep range: {min(debug_x_separations):.0f}-{max_x_sep:.0f}px (avg={avg_x_sep:.0f})")

            # 左右站位去重（间隔 < MIN_START_INTERVAL 的只保留第一个）
            deduped_stance = []
            last_stance_t = -MIN_START_INTERVAL * 2
            for ss in sorted(stance_starts, key=lambda x: x['timestamp']):
                if ss['timestamp'] - last_stance_t >= MIN_START_INTERVAL:
                    deduped_stance.append(ss)
                    last_stance_t = ss['timestamp']
            stance_starts = deduped_stance

            if stance_starts:
                round_start_points.extend(stance_starts)
                print(f"  [READY_STANCE] Detected {len(stance_starts)} ready-stance starts "
                      f"(min_x_sep={min_x_separation:.0f}px, min_static={min_static_duration:.1f}s)")

        # ====== 次优先级：静→动检测（补充左右站位未覆盖的回合开始）======
        if self.yolo_enabled and motion_data:
            STATIC_CONFIRM_FRAMES = 3
            MOTION_CONFIRM_FRAMES = 2

            last_start_time = -MIN_START_INTERVAL

            # 记录左右站位已覆盖的时间区域，静→动不会覆盖这些区域
            stance_time_ranges = []
            for ss in stance_starts:
                stance_time_ranges.append((ss['timestamp'] - MIN_START_INTERVAL / 2,
                                           ss['timestamp'] + MIN_START_INTERVAL / 2))

            smoothed_static = []
            for i in range(len(motion_data)):
                window_start = max(0, i - 1)
                window_end = min(len(motion_data), i + 2)
                window = motion_data[window_start:window_end]
                static_votes = sum(1 for m in window if m['is_static'])
                smoothed_static.append(static_votes > len(window) / 2)

            static_count = 0
            motion_count = 0
            in_static_state = True

            for i in range(len(motion_data)):
                curr = motion_data[i]
                is_curr_static = smoothed_static[i]

                if in_static_state:
                    if is_curr_static:
                        static_count = min(static_count + 1, STATIC_CONFIRM_FRAMES)
                        motion_count = 0
                    else:
                        motion_count += 1
                        if motion_count >= MOTION_CONFIRM_FRAMES:
                            curr_t = curr['timestamp']
                            # 检查是否与左右站位检测冲突，冲突则跳过（优先级低）
                            in_stance_zone = any(s[0] <= curr_t <= s[1] for s in stance_time_ranges)
                            if not in_stance_zone and curr.get('valid_players', True) and curr_t - last_start_time >= MIN_START_INTERVAL:
                                round_start_points.append({
                                    'timestamp': curr_t,
                                    'frame': curr['frame_idx'],
                                    'method': 'yolo_motion_start'
                                })
                                last_start_time = curr_t
                            in_static_state = False
                            static_count = 0
                            motion_count = 0
                else:
                    if not is_curr_static:
                        motion_count = min(motion_count + 1, MOTION_CONFIRM_FRAMES)
                        static_count = 0
                    else:
                        static_count += 1
                        if static_count >= STATIC_CONFIRM_FRAMES:
                            in_static_state = True
                            static_count = 0
                            motion_count = 0

            motion_count_val = len(round_start_points) - len(stance_starts)
            print(f"  YOLO detected {len(round_start_points)} total starts "
                  f"(stance: {len(stance_starts)}, motion: {motion_count_val})")

        # 按时间排序
        round_start_points.sort(key=lambda x: x['timestamp'])

        # 如果没有 YOLO 数据，使用 I3D round_start
        if not round_start_points and i3d_starts:
            for s in i3d_starts:
                round_start_points.append({
                    'timestamp': s['t'],
                    'frame': s['f'],
                    'method': 'i3d_round_start'
                })
            print(f"  Using I3D round_start as fallback ({len(round_start_points)} points)")
        
        # --- 第二阶段：为每个开始点匹配结束点 ---
        # 标记已使用的 round_end，确保每个 end 只匹配一个回合
        used_end_indices = set()
        
        for i, start_point in enumerate(round_start_points):
            start_t = start_point['timestamp']
            start_f = start_point['frame']
            
            # 寻找该回合的结束点
            end_t = None
            end_f = None
            end_method = None
            
            # 2.1 优先：查找 I3D 检测到的球落地 (round_end)
            # 每个 round_end 只能被使用一次
            if PREFER_I3D_END:
                for idx, e in enumerate(i3d_ends):
                    if idx in used_end_indices:
                        continue
                    # 在当前回合开始后、下一个回合开始前（或最大搜索时间内）寻找球落地
                    next_start_t = round_start_points[i + 1]['timestamp'] if i + 1 < len(round_start_points) else start_t + MAX_END_SEARCH_TIME
                    if start_t < e['t'] < min(start_t + MAX_END_SEARCH_TIME, next_start_t):
                        end_t = e['t']
                        end_f = e['f']
                        end_method = 'i3d_round_end'
                        used_end_indices.add(idx)
                        break
            
            # 2.2 备选：查找下一个回合开始前，或视频结束（如果启用回退）
            if end_t is None and FALLBACK_TO_MOTION:
                if i + 1 < len(round_start_points):
                    # 有下一回合，当前回合结束于下一回合开始前
                    next_start_t = round_start_points[i + 1]['timestamp']
                    # 找到运动停止的点（静下来的最后一帧）
                    for j in range(len(motion_data) - 1, -1, -1):
                        if motion_data[j]['timestamp'] < next_start_t and motion_data[j]['is_static']:
                            end_t = motion_data[j]['timestamp']
                            end_f = motion_data[j]['frame_idx']
                            end_method = 'motion_stop_before_next'
                            break
                    if end_t is None:
                        end_t = next_start_t - 0.5  # 保守估计
                        end_f = int(end_t * self.frame_rate)
                        end_method = 'estimated_before_next'
                else:
                    # 最后一个回合，结束于视频末尾或最后一个运动点
                    end_t = total_duration
                    end_f = int(end_t * self.frame_rate)
                    end_method = 'video_end'
            
            # 确保结束时间合理
            if end_t and end_t > start_t:
                # 应用缓冲：开始前 0.3s，结束后 0.5s（减小缓冲避免重叠）
                final_start = max(0, start_t - 0.3)
                final_end = min(total_duration, end_t + 0.5)
                
                # 计算持续时间
                duration = final_end - final_start
                
                # 应用持续时间约束检查
                if duration < MIN_ROUND_DURATION:
                    print(f"  [FILTERED] Round too short: {duration:.2f}s < {MIN_ROUND_DURATION}s")
                    continue
                if duration > MAX_ROUND_DURATION:
                    print(f"  [FILTERED] Round too long: {duration:.2f}s > {MAX_ROUND_DURATION}s")
                    continue
                
                # 根据结束方法确定置信度
                if end_method == 'i3d_round_end':
                    confidence = i3d_end_confidence
                elif end_method in ['motion_stop_before_next', 'motion_stop']:
                    confidence = motion_stop_confidence
                else:
                    confidence = estimated_confidence
                
                rounds.append({
                    'round_id': round_id,
                    'start_time': final_start,
                    'end_time': final_end,
                    'duration': duration,
                    'start_frame': start_f,
                    'end_frame': end_f,
                    'start_method': start_point['method'],
                    'end_method': end_method,
                    'confidence': confidence
                })
                print(f"  Round {round_id}: {final_start:.2f}s - {final_end:.2f}s "
                      f"(start: {start_point['method']}, end: {end_method})")
                round_id += 1
        
        # --- 合并相邻回合 ---
        MERGE_IF_OVERLAP = round_constraints.get('merge_if_overlap', True)
        
        if len(rounds) > 1:
            merged_rounds = []
            current = rounds[0]
            
            for next_round in rounds[1:]:
                # 计算间隔：下一回合开始 - 当前回合结束
                interval = next_round['start_time'] - current['end_time']
                
                # 间隔在合理范围内（0到MERGE_INTERVAL之间）才合并
                if 0 <= interval < MERGE_INTERVAL:
                    # 扩展当前回合的结束时间
                    current['end_time'] = next_round['end_time']
                    current['end_frame'] = next_round['end_frame']
                    current['duration'] = current['end_time'] - current['start_time']
                    current['end_method'] = f"merged({current['end_method']}+{next_round['end_method']})"
                    current['confidence'] = min(current['confidence'], next_round['confidence'])
                    print(f"  [MERGE] Round {current['round_id']} + {next_round['round_id']}: "
                          f"间隔 {interval:.2f}s < {MERGE_INTERVAL}s")
                elif interval < 0 and MERGE_IF_OVERLAP:
                    # 重叠的情况，合并并记录
                    print(f"  [MERGE-OVERLAP] Round {current['round_id']} + {next_round['round_id']}: "
                          f"重叠 {-interval:.2f}s")
                    current['end_time'] = max(current['end_time'], next_round['end_time'])
                    current['end_frame'] = max(current['end_frame'], next_round['end_frame'])
                    current['duration'] = current['end_time'] - current['start_time']
                    current['end_method'] = f"merged_overlap({current['end_method']}+{next_round['end_method']})"
                    current['confidence'] = min(current['confidence'], next_round['confidence'])
                elif interval < 0:
                    # 重叠但不合并，保留当前并切换到下一个
                    print(f"  [OVERLAP] Round {current['round_id']} + {next_round['round_id']}: "
                          f"重叠 {-interval:.2f}s (not merged)")
                    merged_rounds.append(current)
                    current = next_round
                else:
                    merged_rounds.append(current)
                    current = next_round
            
            merged_rounds.append(current)
            rounds = merged_rounds
            
            # 重新编号
            for i, r in enumerate(rounds, 1):
                r['round_id'] = i
        
        if not rounds:
            print("  [WARNING] No rounds detected!")
        else:
            print(f"  Total: {len(rounds)} rounds extracted (after merging)")

        return rounds

    def _post_process(self, predictions):
        """后处理：过滤低置信度并平滑结果.NMS"""
        # 1. 过滤低置信度
        valid = [p for p in predictions if p['confidence'] >= self.confidence_threshold]
        
        if not valid:
            return []
            
        # 2. 合并重复动作（在最小间隔内的相同动作）
        final = []
        if valid:
            valid.sort(key=lambda x: x['center_frame'])
            
            last_p = valid[0]
            current_group = [last_p]
            
            for i in range(1, len(valid)):
                curr_p = valid[i]
                if curr_p['center_frame'] - last_p['center_frame'] < self.min_interval:
                    current_group.append(curr_p)
                else:
                    # 从组中选出置信度最高的
                    best_in_group = max(current_group, key=lambda x: x['confidence'])
                    final.append(best_in_group)
                    current_group = [curr_p]
                last_p = curr_p
            
            if current_group:
                best_in_group = max(current_group, key=lambda x: x['confidence'])
                final.append(best_in_group)
                
        return final

    def _save_results(self, video_path, results):
        """保存预测结果到JSON文件"""
        video_name = Path(video_path).stem
        output_dir = Path(self.config.get('paths', 'output_predictions'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{video_name}_predictions.json"
        
        output_data = {
            'video_name': video_name,
            'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
            
        print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    # 测试预测
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "03_model/trained/best_model.pth"
        
        predictor = ActionPredictor(model_path)
        predictor.predict_video(video_path)
    else:
        print("Usage: python model_predict.py <video_path> [model_path]")
