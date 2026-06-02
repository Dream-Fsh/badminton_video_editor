"""
模型预测模块 - 高性能优化版（流式处理）
优化点：
1. 流式处理：分块读取视频，避免OOM
2. 半精度推理：FP16加速（支持的话）
3. 增大滑动步长：默认从2改为8，减少窗口数量
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
    from athlete_detector import AthleteDetector, ShuttlecockTracker, SimplePersonTracker, GreenPixelDetector
except ImportError:
    from .config_loader import load_config
    from .i3d import InceptionI3d, Unit3D
    from .athlete_detector import AthleteDetector, ShuttlecockTracker, SimplePersonTracker, GreenPixelDetector




class ActionPredictorFast:
    """高速动作预测器 - 流式处理版（避免OOM）"""
    
    def __init__(self, model_path, config_path="../05_config/config.yaml"):
        """初始化预测器"""
        self.config = load_config(config_path)
        self.model_path = model_path
        
        # 加载配置参数
        self.sequence_length = self.config.get('training', 'sequence_length')
        self.num_classes = self.config.get('training', 'num_classes')
        self.confidence_threshold = self.config.get('prediction', 'confidence_threshold')
        self.batch_size = self.config.get('prediction', 'batch_size')
        self.frame_rate = self.config.get('preprocessing', 'frame_rate')
        
        # 模型输入尺寸
        self.model_input_size = None
        
        # YOLO 运动员检测配置
        self.yolo_enabled = self.config.get('prediction', 'athlete_detection', 'enabled', default=False)
        if self.yolo_enabled:
            yolo_model = self.config.get('prediction', 'athlete_detection', 'model_type', default='yolov8n.pt')
            shuttle_cfg = self.config.get('prediction', 'athlete_detection', 'shuttlecock_detection', default={})
            shuttle_model = shuttle_cfg.get('model_path') if shuttle_cfg.get('enabled', False) else None
            self.athlete_detector = AthleteDetector(model_type=yolo_model, shuttlecock_model=shuttle_model)
            self.static_threshold = self.config.get('prediction', 'athlete_detection', 'static_threshold', default=5.0)
            self.yolo_sample_rate = self.config.get('prediction', 'athlete_detection', 'sample_rate', default=5)
            self.athlete_constraints = self.config.get('prediction', 'athlete_detection', 'constraints', default={})
            self.min_player_count = self.athlete_constraints.get('min_player_count', 2)
            self.max_player_count = self.athlete_constraints.get('max_player_count', 2)
        
        # 后处理参数 - 添加默认值
        self.min_interval = self.config.get('prediction', 'post_processing', 'min_interval', default=15)
        self.smooth_window = self.config.get('prediction', 'post_processing', 'smooth_window', default=5)
        
        # ROI 裁剪配置
        roi_config = self.config.get('preprocessing', 'roi')
        self.roi_enabled = roi_config.get('enabled', False) if roi_config else False
        self.roi_x = roi_config.get('x_offset', 0) if roi_config else 0
        self.roi_y = roi_config.get('y_offset', 0) if roi_config else 0
        self.roi_w = roi_config.get('width', 950) if roi_config else 950
        self.roi_h = roi_config.get('height', 720) if roi_config else 720
        
        # 设备配置 - 必须放在batch size检测之前
        device_config = self.config.get('system', 'device')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        # 优化参数
        self.use_fp16 = self.config.get('prediction', 'use_fp16', default=True)
        # 降低batch size防止OOM，根据显存自动调整
        if self.device.type == 'cuda':
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if gpu_mem < 6:  # < 6GB
                self.inference_batch_size = 4
            elif gpu_mem < 10:  # < 10GB
                self.inference_batch_size = 8
            else:
                self.inference_batch_size = 16
        else:
            self.inference_batch_size = 4
        self.num_workers = 0  # Windows下设为0避免多进程问题
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  FP16加速: {'启用' if self.use_fp16 else '禁用'}")
        
        # 加载模型
        self.model = self._load_model()
        self.action_classes = self.config.get('action_classes')
    
    def _load_model(self):
        """加载训练好的I3D模型"""
        # 清理显存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        print(f"\nLoading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        class BadmintonI3D(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.model = InceptionI3d(num_classes=num_classes, in_channels=3)
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
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # 启用半精度
        if self.use_fp16 and self.device.type == 'cuda':
            model = model.half()
        
        print("[OK] Model loaded successfully")
        return model
    
    def predict_video(self, video_path, save_predictions=True, sliding_stride=None, progress_callback=None):
        """
        对整个视频进行动作识别 - 流式处理版本（避免OOM）
        
        Args:
            video_path: 视频路径
            save_predictions: 是否保存结果
            sliding_stride: 滑动步长，None则使用配置值，建议8-16
            progress_callback: 进度回调函数，签名：callback(progress_percent, message)
                           - progress_percent: 0-100的进度百分比
                           - message: 当前阶段描述
        """
        print(f"\nPredicting video: {Path(video_path).name}")
        
        def update_progress(percent, message=""):
            """更新进度的内部函数"""
            if progress_callback:
                progress_callback(percent, message)
        
        # 使用传入的stride或配置值
        if sliding_stride is None:
            sliding_stride = self.config.get('prediction', 'sliding_window_stride', default=8)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Sliding stride: {sliding_stride} (窗口: {self.sequence_length})")
        
        # 计算总窗口数
        num_windows = (total_frames - self.sequence_length) // sliding_stride + 1
        print(f"  Total windows: {num_windows} (using batch_size={self.inference_batch_size})")
        
        # YOLO运动员检测数据
        athlete_motions = []
        shuttle_landings = []
        
        # ========== YOLO运动员检测（如果启用） ==========
        if self.yolo_enabled:
            update_progress(5, "正在检测运动员和羽毛球...")
            athlete_motions, shuttle_landings = self._detect_athlete_motion(video_path, fps)
            update_progress(30, "运动员检测完成")
        
        # ========== 流式处理：按块读取和处理 ==========
        # 每次处理 chunk_size 个窗口，避免内存爆炸
        chunk_size = min(500, num_windows)  # 每次最多处理500个窗口
        
        predictions = []
        print(f"  Running streaming inference (chunk_size={chunk_size})...")
        
        update_progress(30, "开始I3D识别...")
        
        with torch.no_grad():
            # 分块处理 - 显示窗口级别进度
            total_windows_processed = 0
            chunk_indices = list(range(0, num_windows, chunk_size))
            
            for chunk_idx, chunk_start in enumerate(chunk_indices):
                chunk_end = min(chunk_start + chunk_size, num_windows)
                
                # 计算进度百分比（30-90%是I3D识别阶段）
                base_progress = 30
                i3d_range = 60  # I3D识别占总进度的60%
                chunk_progress = int(base_progress + (chunk_idx / len(chunk_indices)) * i3d_range)
                
                # 显示当前 chunk 进度
                msg = f"正在识别 [{chunk_idx+1}/{len(chunk_indices)}]"
                print(f"  [{chunk_idx+1}/{len(chunk_indices)}] Processing windows {chunk_start+1}-{chunk_end}/{num_windows}")
                update_progress(chunk_progress, msg)
                
                # 读取当前chunk需要的帧
                # 第一个窗口的起始帧
                first_window_start = chunk_start * sliding_stride
                # 最后一个窗口的结束帧
                last_window_end = (chunk_end - 1) * sliding_stride + self.sequence_length
                
                # 读取帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_window_start)
                chunk_frames = []
                
                for frame_idx in range(first_window_start, min(last_window_end, total_frames)):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # ROI裁剪
                    if self.roi_enabled:
                        h, w = frame.shape[:2]
                        x_end = min(self.roi_x + self.roi_w, w)
                        y_end = min(self.roi_y + self.roi_h, h)
                        frame = frame[self.roi_y:y_end, self.roi_x:x_end]
                    
                    # Resize
                    if self.model_input_size is not None:
                        frame = cv2.resize(frame, self.model_input_size)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    chunk_frames.append(frame_rgb)
                
                # 处理当前chunk的窗口
                chunk_predictions = self._process_chunk(
                    chunk_frames, 
                    chunk_start, 
                    chunk_end, 
                    sliding_stride,
                    fps,
                    first_window_start,
                    num_windows  # 传递总窗口数用于进度显示
                )
                predictions.extend(chunk_predictions)
                total_windows_processed += len(chunk_predictions)
                
                # 释放内存
                del chunk_frames
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        cap.release()
        
        # 后处理
        update_progress(90, "正在后处理...")
        filtered_predictions = self._post_process(predictions)
        
        final_results = {
            'action_predictions': filtered_predictions,
            'motion_data': athlete_motions,
            'shuttle_landings': shuttle_landings,
            'total_duration': total_frames / fps if fps > 0 else 0
        }

        # 调用增强的回合提取（整合I3D+YOLO就位检测+YOLO运动检测）
        print(f"\n[ROUND EXTRACTION] Running enhanced round extraction...")
        enhanced_rounds = self.extract_rounds(final_results)

        # ===== 绿色像素过滤：剔除不含绿色比赛场地的误检回合（如演播室、休息区） =====
        green_cfg = self.config.get('video_editing', 'green_pixel_filter', default={})
        green_filter_enabled = green_cfg.get('enabled', True)
        if green_filter_enabled and enhanced_rounds:
            update_progress(92, "正在绿色场地验证...")
            green_detector = GreenPixelDetector(
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
            filtered_rounds = []
            for r in enhanced_rounds:
                has_green = green_detector.check_video_segment(
                    video_path, r['start_time'], r['end_time'], verbose=True
                )
                if has_green:
                    filtered_rounds.append(r)
                else:
                    print(f"  [GREEN_FILTER] Round {r['round_id']}: 画面不含绿色比赛场地，已剔除（{r['start_time']:.1f}s-{r['end_time']:.1f}s）")

            if len(filtered_rounds) < len(enhanced_rounds):
                # 重新编号
                for i, r in enumerate(filtered_rounds, 1):
                    r['round_id'] = i
                print(f"  [GREEN_FILTER] 绿色场地验证完成：{len(enhanced_rounds)}→{len(filtered_rounds)} 个回合")

            enhanced_rounds = filtered_rounds

        final_results['rounds'] = enhanced_rounds

        update_progress(95, "正在保存结果...")
        print(f"\n[OK] Prediction finished")
        print(f"  Original count: {len(predictions)}")
        print(f"  Filtered count: {len(filtered_predictions)}")
        print(f"  Enhanced rounds: {len(enhanced_rounds)}")

        if save_predictions:
            self._save_results(video_path, final_results)
        
        update_progress(100, "识别完成")
        
        return final_results  # 返回完整结果（含rounds），供 main.py 直接使用
    
    def _detect_athlete_motion(self, video_path, fps):
        """检测运动员运动状态 + 羽毛球状态（落地/消失），综合判断回合开始和结束
        新增：使用 SimplePersonTracker 为每个检测到的人员分配持久 ID，识别场地上的2名主运动员
        """
        print(f"  Running YOLO athlete+shuttlecock detection...")
        
        # 羽毛球检测配置
        shuttle_cfg = self.config.get('prediction', 'athlete_detection', 'shuttlecock_detection', default={})
        shuttle_tracking_enabled = shuttle_cfg.get('enabled', False) and self.athlete_detector.shuttlecock_model is not None
        shuttle_sample_rate = shuttle_cfg.get('sample_rate', 3)
        
        # 初始化羽毛球跟踪器
        shuttle_tracker = None
        if shuttle_tracking_enabled:
            shuttle_tracker = ShuttlecockTracker(
                still_duration=shuttle_cfg.get('still_duration', 0.5),
                still_threshold=shuttle_cfg.get('still_threshold', 8.0),
                fps=fps,
                sample_rate=shuttle_sample_rate,
                vanish_duration=shuttle_cfg.get('vanish_duration', 0.5)
            )
            print(f"  Shuttlecock tracking: enabled (still={shuttle_cfg.get('still_duration', 0.5)}s, vanish={shuttle_cfg.get('vanish_duration', 0.5)}s)")
        
        # ===== 运动员身份追踪配置 =====
        identity_cfg = self.config.get('prediction', 'athlete_identification', default={})
        identity_enabled = identity_cfg.get('enabled', True)
        max_detect_count = identity_cfg.get('max_detect_count', 4)  # 检测最多的人数（需要>2才能发现"其他人"）
        track_max_distance = identity_cfg.get('track_max_distance', 250)
        track_min_appearances = identity_cfg.get('track_min_appearances', 5)
        merge_gap_frames = identity_cfg.get('merge_gap_frames', 30)   # 合并碎片轨迹的最大间隔帧数
        merge_distance = identity_cfg.get('merge_distance', 250)       # 合并碎片轨迹的最大空间距离
        
        # 初始化人员追踪器（宽松参数减少碎片化）
        person_tracker = SimplePersonTracker(
            max_distance=track_max_distance,
            max_lost_frames=int(5.0 * fps / self.yolo_sample_rate)  # 丢失5秒后删除轨迹（增大避免过早丢失）
        )
        if identity_enabled:
            print(f"  Athlete identification: enabled (max_detect={max_detect_count}, track_dist={track_max_distance}px)")
        
        # 使用统一的采样率（取两者中更频繁的）
        unified_sample_rate = min(self.yolo_sample_rate, shuttle_sample_rate) if shuttle_tracking_enabled else self.yolo_sample_rate
        print(f"  Unified sample_rate: every {unified_sample_rate} frames")
        
        cap_yolo = cv2.VideoCapture(video_path)
        if not cap_yolo.isOpened():
            print(f"  Warning: Cannot open video for YOLO detection: {video_path}")
            return [], []
        
        athlete_motions = []
        shuttle_landings = []
        frame_idx = 0
        athlete_positions_history = []
        shuttlecock_positions = []
        
        # 运动员运动状态（持续跟踪）
        current_athletes_static = False
        current_athlete_count = 0
        
        while True:
            ret, frame = cap_yolo.read()
            if not ret:
                break
            
            # ROI裁剪
            if self.roi_enabled:
                h, w = frame.shape[:2]
                x_end = min(self.roi_x + self.roi_w, w)
                y_end = min(self.roi_y + self.roi_h, h)
                frame = frame[self.roi_y:y_end, self.roi_x:x_end]
            
            # 统一采样
            if frame_idx % unified_sample_rate == 0:
                
                # 检测运动员（如果启用了身份追踪，则检测更多人以发现"其他人"）
                detect_count = max_detect_count if identity_enabled else self.max_player_count
                players = self.athlete_detector.detect_athletes(
                    frame, 
                    constraints={
                        'min_confidence': self.athlete_constraints.get('min_player_confidence', 0.5),
                        'court_boundary': self.athlete_constraints.get('court_boundary', {}),
                        'green_court_boundary': self.athlete_constraints.get('green_court_boundary', {}),
                        'max_count': detect_count
                    }
                )
                
                # 只追踪绿色场地上的运动员，忽略场地外的人员（教练、观众等）
                if identity_enabled and players:
                    green_players = [p for p in players if p.get('in_green_court', False)]
                    if green_players:
                        green_players = person_tracker.update(green_players, frame_idx)
                
                current_athlete_count = len(green_players) if identity_enabled else len(players)
                players_in_green_court = current_athlete_count
                
                # 运动分析和后续逻辑只关心绿色场地上的运动员
                players = green_players if identity_enabled else players
                
                # 记录每帧中出现的追踪 ID（用于后续身份过滤）
                frame_track_ids = [p.get('track_id', -1) for p in players] if identity_enabled else []
                frame_green_track_ids = [p.get('track_id', -1) for p in players] if identity_enabled else []
                
                # 运动分析（只看面积最大的前2个人，用于回合开始/结束检测）
                top_players = players[:self.max_player_count] if players else []
                if top_players:
                    centers = [player['center'] for player in top_players]
                    avg_center = np.mean(centers, axis=0).tolist()
                    athlete_positions_history.append(avg_center)
                    
                    history_window = int(1.0 * fps / unified_sample_rate)
                    if len(athlete_positions_history) > history_window:
                        athlete_positions_history = athlete_positions_history[-history_window:]
                    
                    current_athletes_static = False
                    if len(athlete_positions_history) >= 3:
                        current_athletes_static = self.athlete_detector.analyze_motion(
                            athlete_positions_history, 
                            threshold=self.static_threshold
                        )
                else:
                    current_athlete_count = 0
                    players_in_green_court = 0
                    current_athletes_static = True  # 没检测到人视为静止
                
                # 羽毛球跟踪 + 组合判断回合结束
                shuttle_result_for_tracker = None
                if shuttle_tracker:
                    shuttle_result_for_tracker = self.athlete_detector.detect_shuttlecock(frame, min_confidence=0.25)
                    landing_event = shuttle_tracker.update(
                        frame_idx, shuttle_result_for_tracker,
                        athletes_static=current_athletes_static,
                        athlete_count=current_athlete_count,
                        players=top_players
                    )
                    
                    if landing_event:
                        shuttle_landings.append(landing_event)
                
                # 记录羽毛球位置（复用刚才的检测结果，避免重复推理）
                shuttlecock = shuttle_result_for_tracker if shuttle_tracker else self.athlete_detector.detect_shuttlecock(frame, min_confidence=0.3)
                ball_position = shuttlecock['center'] if shuttlecock else None
                if ball_position:
                    shuttlecock_positions.append({
                        'frame': frame_idx,
                        'timestamp': frame_idx / fps,
                        'position': ball_position,
                        'conf': shuttlecock['conf']
                    })
                
                athlete_motions.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'is_static': bool(current_athletes_static),
                    'player_count': int(current_athlete_count),
                    'players_in_green_court': int(players_in_green_court),
                    'valid_players': True,
                    'players': top_players,
                    'center_position': np.mean([p['center'] for p in top_players], axis=0).tolist() if top_players else [0, 0],
                    'shuttlecock_detected': shuttlecock is not None,
                    'shuttlecock_position': ball_position,
                    'shuttlecock_conf': shuttlecock['conf'] if shuttlecock else 0.0,
                    # 身份追踪数据
                    'track_ids': frame_track_ids,
                    'green_track_ids': frame_green_track_ids,
                })
            
            frame_idx += 1
        
        cap_yolo.release()
        
        # ===== 识别2名主运动员（在绿色场地上出现次数最多的2个追踪 ID） =====
        if identity_enabled:
            # 先合并碎片化的轨迹（同一人因遮挡等原因被分配了多个ID）
            pre_merge_count = len(person_tracker.tracks)
            id_mapping = person_tracker.merge_fragmented_tracks(
                merge_gap_frames=merge_gap_frames,
                merge_distance=merge_distance
            )
            post_merge_count = len(person_tracker.tracks)
            
            # 用合并映射重写 motion_data 中的 track_ids 和 green_track_ids
            if id_mapping:
                for m in athlete_motions:
                    m['track_ids'] = [id_mapping.get(tid, tid) for tid in m.get('track_ids', [])]
                    m['green_track_ids'] = [id_mapping.get(tid, tid) for tid in m.get('green_track_ids', [])]
                print(f"  轨迹合并: {pre_merge_count} → {post_merge_count} 个轨迹（合并了 {len(id_mapping)} 个碎片ID）")
            
            main_ids = person_tracker.get_main_athletes(n=2, min_appearances=track_min_appearances)
            self.main_athlete_ids = main_ids
            print(f"  追踪完成: 共 {len(person_tracker.tracks)} 个不同的人员轨迹")
            print(f"  主运动员 ID: {main_ids} (在绿色场地上出现最频繁的2人)")
            # 打印所有轨迹的统计
            for tid, t in sorted(person_tracker.tracks.items(), key=lambda x: x[1]['green_court_count'], reverse=True):
                marker = " <-- 主运动员" if tid in main_ids else ""
                print(f"    Track {tid}: 出现 {t['appearances']} 次, 绿色场地 {t['green_court_count']} 次{marker}")
        else:
            self.main_athlete_ids = []
        
        print(f"  YOLO检测完成: {len(athlete_motions)} 个检测点")
        print(f"  回合结束事件: {len(shuttle_landings)} 个")
        for ev in shuttle_landings:
            print(f"    [{ev['method']}] {ev['timestamp']:.1f}s (frame {ev['frame']}) - {ev['reason']}")
        
        return athlete_motions, shuttle_landings
    
    def _process_chunk(self, chunk_frames, chunk_start, chunk_end, sliding_stride, fps, frame_offset, num_windows=None):
        """处理一个chunk的窗口"""
        predictions = []
        
        # 构建窗口 - 显示每个窗口的进度
        window_indices = list(range(chunk_start, chunk_end))
        for window_idx in tqdm(window_indices, desc=f"  Windows", leave=False, ncols=80):
            start_idx = (window_idx - chunk_start) * sliding_stride
            end_idx = start_idx + self.sequence_length
            
            if end_idx > len(chunk_frames):
                break
            
            frame_sequence = chunk_frames[start_idx:end_idx]
            
            # 预处理
            try:
                frames_np = np.stack(frame_sequence, axis=0).astype(np.float32) / 255.0
                
                # ImageNet标准化
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
                frames_np = (frames_np - mean) / std
                
                # 转换为tensor (T, H, W, 3) -> (1, 3, T, H, W)
                frames_tensor = torch.from_numpy(frames_np).float().permute(3, 0, 1, 2).unsqueeze(0)
                frames_tensor = frames_tensor.to(self.device)
                
                if self.use_fp16 and self.device.type == 'cuda':
                    frames_tensor = frames_tensor.half()
                
                # 推理
                output = self.model(frames_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)
                
                # 计算实际帧索引
                actual_start_frame = window_idx * sliding_stride
                actual_center_frame = actual_start_frame + self.sequence_length // 2
                
                predictions.append({
                    'start_frame': actual_start_frame,
                    'end_frame': actual_start_frame + self.sequence_length,
                    'center_frame': actual_center_frame,
                    'predicted_class': int(predicted_class.item()),
                    'class_name': self.action_classes[int(predicted_class.item())],
                    'confidence': float(confidence.item()),
                    'timestamp': actual_center_frame / fps
                })
                
                # 清理
                del frames_tensor, output, probabilities
                
            except Exception as e:
                print(f"    警告: 处理窗口 {window_idx} 时出错: {e}")
                continue
        
        return predictions

    def extract_rounds(self, results):
        """智能回合提取算法 - 优先使用羽毛球落地检测"""
        if isinstance(results, dict):
            predictions = results.get('action_predictions', [])
            motion_data = results.get('motion_data', [])
            shuttle_landings = results.get('shuttle_landings', [])
            total_duration = results.get('total_duration', 3600.0)
        else:
            predictions = results
            motion_data = []
            shuttle_landings = []
            total_duration = 3600.0

        rounds = []
        round_id = 1
        
        round_constraints = self.config.get('prediction', 'round_constraints', default={})
        MIN_START_INTERVAL = round_constraints.get('min_start_interval', 3.0)
        MIN_ROUND_DURATION = round_constraints.get('min_round_duration', 2.0)
        MAX_ROUND_DURATION = round_constraints.get('max_round_duration', 30.0)
        MERGE_INTERVAL = round_constraints.get('merge_interval', 2.0)
        MERGE_IF_OVERLAP = round_constraints.get('merge_if_overlap', True)
        MAX_END_SEARCH_TIME = round_constraints.get('max_round_end_search_time', 25.0)
        PREFER_I3D_END = round_constraints.get('prefer_i3d_end', True)
        FALLBACK_TO_MOTION = round_constraints.get('fallback_to_motion_stop', True)
        confidence_weights = round_constraints.get('confidence_weights', {})
        i3d_end_confidence = confidence_weights.get('i3d_round_end', 0.9)
        motion_stop_confidence = confidence_weights.get('motion_stop', 0.7)
        estimated_confidence = confidence_weights.get('estimated', 0.5)
        
        i3d_starts = sorted([{'t': p['timestamp'], 'f': p['center_frame'], 'conf': p['confidence']} 
                             for p in predictions if p['class_name'] == 'round_start'], key=lambda x: x['t'])
        i3d_ends = sorted([{'t': p['timestamp'], 'f': p['center_frame'], 'conf': p['confidence']} 
                           for p in predictions if p['class_name'] == 'round_end'], key=lambda x: x['t'])
        
        print(f"  I3D detected: {len(i3d_starts)} round_starts, {len(i3d_ends)} round_ends")
        if shuttle_landings:
            print(f"  Shuttle landings detected: {len(shuttle_landings)}")
        
        # YOLO检测回合开始点
        round_start_points = []

        # ====== 最高优先级：左右站位规则（两球员分居两侧且静止）======
        ready_stance_cfg = round_constraints.get('ready_stance_detection', {})
        ready_stance_enabled = ready_stance_cfg.get('enabled', False)
        stance_starts = []

        if ready_stance_enabled and self.yolo_enabled and motion_data:
            min_x_separation = ready_stance_cfg.get('min_x_separation', 120.0)
            min_static_duration = ready_stance_cfg.get('min_static_duration', 0.5)
            ready_confidence = ready_stance_cfg.get('min_confidence', 0.6)
            min_round_motion_duration = ready_stance_cfg.get('min_round_motion_duration', 1.0)
            
            # 位置稳定性检测参数（替代纯静止检测）
            allow_minor_motion = ready_stance_cfg.get('allow_minor_motion', False)
            position_stability_window = ready_stance_cfg.get('position_stability_window', 1.0)
            max_position_drift = ready_stance_cfg.get('max_position_drift', 40.0)

            sample_rate = self.yolo_sample_rate
            fps_val = self.frame_rate if hasattr(self, 'frame_rate') else 30
            
            # 原始静态帧数要求（用于严格静止检测）
            min_static_points_strict = max(2, int(min_static_duration * fps_val / sample_rate))
            # 位置稳定性窗口的帧数（用于宽松检测）
            stability_window_frames = int(position_stability_window * fps_val / sample_rate)

            # 调试统计
            debug_total_2players = 0
            debug_static_2players = 0
            debug_stable_2players = 0
            debug_x_separations = []

            consecutive_count = 0
            stance_start_candidate = None
            
            # 位置历史追踪（用于稳定性判断）
            recent_positions_history = []

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
                        
                        # 记录当前帧的两球员平均中心位置
                        avg_center = ((x1 + x2) / 2, (players[0]['center'][1] + players[1]['center'][1]) / 2)

                        if is_static:
                            debug_static_2players += 1
                        
                        # ====== 判断是否满足"就位"条件 ======
                        is_ready_stance = False
                        
                        if allow_minor_motion:
                            # 模式2: 位置稳定性检测（允许轻微运动）
                            # 条件1: X间距足够大（左右分居）
                            x_ok = (x_sep >= min_x_separation)
                            
                            # 条件2: 位置稳定（最近N帧内整体偏移不大）
                            recent_positions_history.append(avg_center)
                            if len(recent_positions_history) > stability_window_frames:
                                recent_positions_history = recent_positions_history[-stability_window_frames:]
                            
                            pos_ok = True
                            if len(recent_positions_history) >= max(3, stability_window_frames // 2):
                                positions_arr = np.array(recent_positions_history)
                                pos_range = np.ptp(positions_arr, axis=0)
                                total_drift = np.linalg.norm(pos_range)
                                pos_ok = (total_drift <= max_position_drift)
                            
                            is_ready_stance = (x_ok and pos_ok)
                            if is_ready_stance:
                                debug_stable_2players += 1
                        else:
                            # 模式1: 严格静止检测（原始逻辑）
                            if is_static and x_sep >= min_x_separation:
                                is_ready_stance = True

                        if is_ready_stance:
                            consecutive_count += 1
                            if consecutive_count == 1:
                                stance_start_candidate = i
                            required_consecutive = max(2, min_static_points_strict)
                            if consecutive_count >= required_consecutive:
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
                if not allow_minor_motion or m.get('player_count', 0) < 2:
                    recent_positions_history.clear()

            # 调试输出
            if debug_total_2players > 0:
                avg_x_sep = sum(debug_x_separations) / len(debug_x_separations) if debug_x_separations else 0
                max_x_sep = max(debug_x_separations) if debug_x_separations else 0
                min_x_sep_val = min(debug_x_separations) if debug_x_separations else 0
                mode_str = "position_stability" if allow_minor_motion else "strict_static"
                print(f"  [STANCE_DEBUG] Mode={mode_str}, 2-players frames: {debug_total_2players}, "
                      f"static frames: {debug_static_2players}, "
                      f"stable frames: {debug_stable_2players}, "
                      f"x_sep range: {min_x_sep_val:.0f}-{max_x_sep:.0f}px (avg={avg_x_sep:.0f}, need>={min_x_separation})")

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

            # 记录左右站位已覆盖的时间点（高优先级），静→动不会覆盖这些区域
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
                            # 关键：少于2人不算有效运动开始（画面中没有运动员对战）
                            if curr.get('player_count', 0) < 2:
                                motion_count = 0
                                # 保持 static 状态，不触发运动开始
                            else:
                                # 检查是否与左右站位检测冲突，冲突则跳过（优先级低）
                                in_stance_zone = any(s[0] <= curr_t <= s[1] for s in stance_time_ranges)
                                if not in_stance_zone and curr_t - last_start_time >= MIN_START_INTERVAL:
                                    round_start_points.append({
                                        'timestamp': curr_t,
                                        'frame': curr['frame_idx'],
                                        'method': 'yolo_motion_start',
                                        'confidence': motion_stop_confidence  # 0.7，与运动停止同级
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

            print(f"  YOLO detected {len(round_start_points)} total starts "
                  f"(stance: {len(stance_starts)}, motion: {len(round_start_points) - len(stance_starts)})")
        else:
            if not self.yolo_enabled:
                print(f"  YOLO disabled in config")
            elif not motion_data:
                print(f"  No motion data collected (YOLO may have failed)")

        # 按时间排序
        round_start_points.sort(key=lambda x: x['timestamp'])
        if not round_start_points:
            if i3d_starts:
                for s in i3d_starts:
                    round_start_points.append({
                        'timestamp': s['t'],
                        'frame': s['f'],
                        'method': 'i3d_round_start',
                        'confidence': s.get('conf', i3d_end_confidence)  # 使用I3D模型实际置信度
                    })
                print(f"  [FALLBACK] Using I3D round_start ({len(round_start_points)} points)")
            else:
                print(f"  [WARNING] No round start points detected at all!")
        elif len(round_start_points) < len(i3d_starts) * 0.5:
            print(f"  [WARNING] YOLO detected only {len(round_start_points)} starts, but I3D found {len(i3d_starts)}")
            print(f"  Consider adjusting static_threshold or min_start_interval in config")
        
        # 匹配结束点
        used_end_indices = set()
        
        # 预计算：每个 ready_stance start 对应的"就位打破"时间
        # （球员从就位状态开始移动的时刻，作为回合最小结束时间）
        stance_break_times = {}
        for sp_idx, start_point in enumerate(round_start_points):
            if start_point['method'] != 'yolo_ready_stance':
                continue
            sp_t = start_point['timestamp']
            # 在 stance start 之后，搜索 motion_data 找到就位打破的时刻
            # 关键：就位打破 = 球员从静止变为运动（is_static: True→False）
            # 而非 x_sep 降到阈值以下（球员靠近不一定打破就位）
            break_t = None
            for j in range(len(motion_data)):
                m = motion_data[j]
                mt = m.get('timestamp', 0)
                if mt <= sp_t + 0.5:  # 跳过stance起点后0.5s内的帧（避免噪声）
                    continue
                pc = m.get('player_count', 0)
                is_static = m.get('is_static', False)
                players = m.get('players', [])
                # 就位打破条件：有2人且从静止变为非静止（球员开始跑动），且必须在绿色场地上
                on_green = m.get('players_in_green_court', 0) >= 2
                if pc >= 2 and not is_static and on_green and len(players) >= 2:
                    x1 = players[0]['center'][0]
                    x2 = players[1]['center'][0]
                    x_sep = abs(x1 - x2)
                    # 确认之前确实处于就位状态（x_sep>=50 或 static=True）
                    # 查看前几帧是否是static的
                    prev_were_static = True
                    lookback = max(0, j - 3)
                    for k in range(lookback, j):
                        prev_green = motion_data[k].get('players_in_green_court', 0) >= 2
                        if not (motion_data[k].get('is_static', False) and prev_green):
                            prev_were_static = False
                            break
                    if prev_were_static:
                        break_t = mt
                        break
            stance_break_times[sp_idx] = break_t
            if break_t is not None:
                print(f"  [STANCE_BREAK] start={sp_t:.2f}s → stance breaks at {break_t:.2f}s")
        
        # 预过滤：移除被其他 stance start 的 stance_break 覆盖的 start points
        # （同一个就位状态不应产生多个回合）
        filtered_start_points = []
        for i, sp in enumerate(round_start_points):
            sp_t = sp['timestamp']
            # 检查是否被前面某个 stance 的 break 时间覆盖
            covered = False
            for j in range(i):
                if round_start_points[j]['method'] == 'yolo_ready_stance':
                    break_t = stance_break_times.get(j)
                    if break_t is not None and sp_t < break_t:
                        covered = True
                        print(f"  [SKIP_DUPLICATE] start at {sp_t:.2f}s covered by stance {round_start_points[j]['timestamp']:.2f}s (breaks at {break_t:.2f}s)")
                        break
            if not covered:
                filtered_start_points.append(sp)
        
        if len(filtered_start_points) < len(round_start_points):
            print(f"  [DEDUP] Removed {len(round_start_points) - len(filtered_start_points)} duplicate starts")
            round_start_points = filtered_start_points
            # 重新计算 stance_break_times 的索引
            new_stance_break_times = {}
            new_idx = 0
            for old_idx, sp in enumerate(round_start_points):
                if sp['method'] == 'yolo_ready_stance':
                    # 找到原来的 break time
                    for old_j in stance_break_times:
                        # 通过时间匹配
                        pass  # 简化：直接保留所有 break times，因为 break 时间本身不依赖索引
            # 简化处理：重新计算 stance_break_times
            stance_break_times = {}
            for idx, sp in enumerate(round_start_points):
                if sp['method'] != 'yolo_ready_stance':
                    continue
                sp_t = sp['timestamp']
                break_t = None
                for j in range(len(motion_data)):
                    m = motion_data[j]
                    mt = m.get('timestamp', 0)
                    if mt <= sp_t + 0.5:
                        continue
                    pc = m.get('player_count', 0)
                    is_static = m.get('is_static', False)
                    players = m.get('players', [])
                    on_green = m.get('players_in_green_court', 0) >= 2
                    if pc >= 2 and not is_static and on_green and len(players) >= 2:
                        prev_were_static = True
                        lookback = max(0, j - 3)
                        for k in range(lookback, j):
                            prev_green = motion_data[k].get('players_in_green_court', 0) >= 2
                            if not (motion_data[k].get('is_static', False) or not prev_green):
                                prev_were_static = False
                                break
                        if prev_were_static:
                            break_t = mt
                            break
                stance_break_times[idx] = break_t
                if break_t is not None:
                    print(f"  [STANCE_BREAK] start={sp_t:.2f}s -> stance breaks at {break_t:.2f}s")

        for i, start_point in enumerate(round_start_points):
            start_t = start_point['timestamp']
            start_f = start_point['frame']
            
            end_t = None
            end_f = None
            end_method = None
            
            # 计算搜索范围的上界
            stance_break = stance_break_times.get(i)
            raw_next_start_t = round_start_points[i + 1]['timestamp'] if i + 1 < len(round_start_points) else start_t + MAX_END_SEARCH_TIME
            next_start_t = raw_next_start_t
            if stance_break is not None and i + 1 < len(round_start_points):
                next_sp = round_start_points[i + 1]
                # 如果下一个 start 在当前 stance break 之前（球员还在就位），跳过它
                if next_sp['timestamp'] < stance_break:
                    print(f"  [SKIP_NEXT_START] next start at {next_sp['timestamp']:.2f}s is before stance break at {stance_break:.2f}s, extending search")
                    # 继续往后找第一个有效 start
                    for k in range(i + 1, len(round_start_points)):
                        if round_start_points[k]['timestamp'] >= stance_break + 2.0:  # 至少在break后2秒
                            next_start_t = round_start_points[k]['timestamp']
                            break
                    else:
                        next_start_t = start_t + MAX_END_SEARCH_TIME
            
            # ===== 优先级1: 羽毛球落地静止（最可靠） =====
            if shuttle_landings and end_t is None:
                for landing in shuttle_landings:
                    landing_t = landing['timestamp']
                    if start_t < landing_t < min(start_t + MAX_END_SEARCH_TIME, next_start_t):
                        end_t = landing_t
                        end_f = landing['frame']
                        end_method = 'shuttlecock_landed'
                        break
            
            # ===== 优先级2: I3D round_end =====
            if PREFER_I3D_END and end_t is None:
                for idx, e in enumerate(i3d_ends):
                    if idx in used_end_indices:
                        continue
                    if start_t < e['t'] < min(start_t + MAX_END_SEARCH_TIME, next_start_t):
                        # 对 ready_stance 回合：如果 I3D end 在就位打破之前，说明是误检，跳过
                        stance_break_i = stance_break_times.get(i)
                        if stance_break_i is not None and e['t'] < stance_break_i:
                            print(f"  [STANCE_SKIP] I3D round_end at {e['t']:.2f}s before stance break at {stance_break_i:.2f}s, skipping")
                            used_end_indices.add(idx)  # 标记为已使用，防止后续重复匹配
                            continue
                        end_t = e['t']
                        end_f = e['f']
                        end_method = 'i3d_round_end'
                        used_end_indices.add(idx)
                        break
            
            # 优先级2.5: 球员离开（少于2人持续超过阈值则回合结束）
            if end_t is None:
                MIN_NO_PLAYER_GAP = 1.0  # 少于2人持续1秒即结束回合
                no_player_start = None
                for j in range(len(motion_data)):
                    m = motion_data[j]
                    mt = m.get('timestamp', 0)
                    if mt <= start_t + 0.5:  # 给开始留0.5秒缓冲
                        continue
                    search_limit = min(start_t + MAX_END_SEARCH_TIME, next_start_t)
                    if mt > search_limit:
                        break
                    if m.get('player_count', 0) < 2:
                        if no_player_start is None:
                            no_player_start = mt
                    else:
                        no_player_start = None
                    if no_player_start is not None and mt - no_player_start >= MIN_NO_PLAYER_GAP:
                        prev_j = j - 1
                        while prev_j >= 0 and motion_data[prev_j].get('player_count', 0) < 2:
                            prev_j -= 1
                        end_t = motion_data[prev_j]['timestamp'] if prev_j >= 0 else start_t + 0.5
                        end_f = motion_data[prev_j]['frame_idx'] if prev_j >= 0 else int(end_t * self.frame_rate)
                        end_method = 'player_count_below_2'
                        break
            
            # ★ 通用结束点运动验证
            # 如果结束点是 shuttle_landed 或 player_count_below_2，
            # 但结束点后 0.3-1.0s 内运动员还在运动 → 否定该结束点，让 motion_stop 重新搜索
            if end_t is not None and end_method in ('shuttlecock_landed', 'player_count_below_2'):
                still_active = False
                check_window = min(end_t + 1.0, total_duration)
                for m_check in motion_data:
                    mt = m_check.get('timestamp', 0)
                    if end_t + 0.3 <= mt <= check_window:
                        if not m_check.get('is_static', True) and m_check.get('player_count', 0) >= 1:
                            still_active = True
                            break
                if still_active:
                    print(f"  [END_REJECT] {end_method} at {end_t:.2f}s rejected: players still active 0.3-1.0s after")
                    end_t = None
                    end_f = None
                    end_method = None
                    # 让后续的 motion_stop 或 estimated 重新找结束点
            
            if end_t is None and FALLBACK_TO_MOTION:
                if i + 1 < len(round_start_points):
                    motion_skip = ready_stance_cfg.get('motion_validation_skip_start', 2.0)
                    search_start = max(start_t + 0.5, start_t + motion_skip * 0.5)
                    if stance_break is not None:
                        search_start = max(search_start, stance_break + 0.3)
                    search_limit = min(start_t + MAX_END_SEARCH_TIME, next_start_t)
                    MIN_STOP_DURATION = 0.8
                    
                    # 单遍扫描: 找到静止段 → 验证 → 失败则跳过后继续（不重新扫描）
                    static_start = None
                    skip_until = search_start
                    for j in range(len(motion_data)):
                        m = motion_data[j]
                        mt = m['timestamp']
                        if mt < skip_until:
                            continue
                        if mt > search_limit:
                            break
                        if m.get('is_static', True):
                            if static_start is None:
                                static_start = mt
                            elif mt - static_start >= MIN_STOP_DURATION:
                                test_end = static_start
                                still_active = False
                                for m2 in motion_data:
                                    m2t = m2.get('timestamp', 0)
                                    if test_end + 0.3 <= m2t <= min(test_end + 1.0, search_limit):
                                        if not m2.get('is_static', True) and m2.get('player_count', 0) >= 1:
                                            still_active = True
                                            break
                                if not still_active:
                                    end_t = test_end
                                    end_f = int(end_t * self.frame_rate)
                                    end_method = 'motion_stop'
                                    break
                                else:
                                    # 验证失败 → 跳过这段静止，继续找下一段
                                    print(f"  [END_REJECT] motion_stop at {test_end:.2f}s: play resumed")
                                    skip_until = mt + 0.3
                                    static_start = None
                        else:
                            static_start = None
                    
                    if end_t is None:
                        end_t = next_start_t - 0.5
                        end_f = int(end_t * self.frame_rate)
                        end_method = 'estimated_before_next'
                else:
                    # 最后一个round_start，没有round_end，使用合理的默认时长
                    DEFAULT_ROUND_DURATION = 15.0  # 默认回合时长15秒
                    end_t = min(start_t + DEFAULT_ROUND_DURATION, total_duration)
                    end_f = int(end_t * self.frame_rate)
                    end_method = f'default_duration_{DEFAULT_ROUND_DURATION}s'
            
            # ===== ready_stance 最小结束时间保底 =====
            # 就位检测的回合至少应延续到球员开始移动（就位打破）的时刻
            stance_break = stance_break_times.get(i)
            if stance_break is not None and end_t is not None:
                if end_t < stance_break:
                    print(f"  [STANCE_FLOOR] end_t={end_t:.2f}s < stance_break={stance_break:.2f}s, extending to stance break")
                    end_t = stance_break
                    end_f = int(end_t * self.frame_rate)
                    end_method = 'stance_break_floor'
            
            # 获取预padding配置
            start_pre_padding = round_constraints.get('start_pre_padding', 0.2)
            end_post_padding = round_constraints.get('end_post_padding', 0.8)
            
            if end_t and end_t > start_t:
                # 重要修复：ready_stance 检测到的是运动员就位时刻，即回合开始时刻
                # 不应再往前减（避免错过0:02这类就位瞬间）
                # 而 motion_start 等其他方法，检测到的是运动发生，需要往前找实际开始点
                if start_point['method'] == 'yolo_ready_stance':
                    final_start = start_t  # 不减，检测到就位=回合开始
                else:
                    final_start = max(0, start_t - start_pre_padding)
                final_end = min(total_duration, end_t + end_post_padding)
                duration = final_end - final_start
                
                if duration < MIN_ROUND_DURATION:
                    print(f"  [FILTERED] Round too short: {duration:.2f}s")
                    continue
                if duration > MAX_ROUND_DURATION:
                    print(f"  [FILTERED] Round too long: {duration:.2f}s")
                    continue
                
                # 计算回合置信度：取 start 方式与 end 方式的较小值
                # start_point 含 confidence（stance=0.6, I3D有实际值, motion默认0.7）
                start_conf = start_point.get('confidence', 0.7)
                end_conf = i3d_end_confidence if end_method == 'i3d_round_end' else (
                    motion_stop_confidence if end_method in ['motion_stop_before_next', 'motion_stop'] else (
                        0.95 if end_method == 'shuttlecock_landed' else estimated_confidence
                    )
                )
                confidence = min(start_conf, end_conf)
                
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
                print(f"  Round {round_id}: {final_start:.2f}s - {final_end:.2f}s")
                round_id += 1
        
        # 合并相邻回合（包括重叠的）
        # 关键原则：一旦只检测到一名运动员，则不算回合，不跨过少于2人的时段合并
        if len(rounds) > 1:
            merged_rounds = []
            current = rounds[0]
            
            for next_round in rounds[1:]:
                interval = next_round['start_time'] - current['end_time']
                
                should_merge = False
                if interval < 0:  # 重叠
                    if MERGE_IF_OVERLAP:
                        should_merge = True
                elif interval < MERGE_INTERVAL:
                    # 间隔小于合并阈值：检查中间是否有 player_count < 2 持续超过1秒
                    has_no_player_gap = False
                    no_player_start = None
                    for m in motion_data:
                        mt = m.get('timestamp', 0)
                        if current['end_time'] <= mt <= next_round['start_time']:
                            if m.get('player_count', 0) < 2:
                                if no_player_start is None:
                                    no_player_start = mt
                            else:
                                no_player_start = None
                            if no_player_start is not None and mt - no_player_start >= 1.0:
                                has_no_player_gap = True
                                break
                    should_merge = not has_no_player_gap
                
                if should_merge:
                    current['end_time'] = max(current['end_time'], next_round['end_time'])
                    current['end_frame'] = max(current['end_frame'], next_round['end_frame'])
                    current['duration'] = current['end_time'] - current['start_time']
                    current['end_method'] = f"merged({current['end_method']}+{next_round['end_method']})"
                    current['confidence'] = min(current['confidence'], next_round['confidence'])
                else:
                    merged_rounds.append(current)
                    current = next_round
            
            merged_rounds.append(current)
            rounds = merged_rounds
            
            for i, r in enumerate(rounds, 1):
                r['round_id'] = i
            print(f"  [MERGE] Merged to {len(rounds)} rounds after overlap merge")
        
        # ===== 保存过滤前的所有回合快照（供用户手动选择）=====
        import copy
        results['all_raw_rounds'] = copy.deepcopy(rounds)
        for r in results['all_raw_rounds']:
            r['_auto_filtered'] = False
            r['_filter_reason'] = ''
        # 保存原始渲染ID，过滤链中会重编号，用于最后匹配 all_raw_rounds
        for r in rounds:
            r['_raw_round_id'] = r['round_id']

        # ===== 回合运动验证：有效回合必须包含持续的运动片段 =====
        # 姿态检测容易把"两人静止等待"误检为回合开始，必须验证回合中真的有击球运动
        # 关键原则：所有运动都必须在绿色场地上，否则不算运动
        if rounds and motion_data:
            min_round_motion = ready_stance_cfg.get('min_round_motion_duration', 0.5)
            motion_skip_start = ready_stance_cfg.get('motion_validation_skip_start', 2.0)
            # 关键：回合中不允许出现超长静止期（如休息、非比赛场景）
            max_static_gap = ready_stance_cfg.get('max_static_gap_duration', 8.0)
            valid_rounds = []
            for r in rounds:
                # 计算回合内的运动时长（连续非静止片段）
                # 关键：跳过开始后的前 N 秒，给球员从就位到击球的时间
                validation_start = r['start_time'] + motion_skip_start
                motion_segments = []
                static_segments = []
                in_motion = False
                in_static = False
                seg_start = 0
                static_start = 0
                for m in motion_data:
                    t = m.get('timestamp', 0)
                    if validation_start <= t <= r['end_time']:
                        is_static = m.get('is_static', True)
                        # 运动必须在绿色场地上：非静止且至少有2人在绿色场地
                        is_valid_motion = (not is_static) and (m.get('players_in_green_court', 0) >= 2)
                        if is_valid_motion:
                            if not in_motion:
                                in_motion = True
                                seg_start = t
                            if in_static:
                                in_static = False
                                static_segments.append(t - static_start)
                        else:
                            if in_motion:
                                in_motion = False
                                motion_segments.append(t - seg_start)
                            if not in_static:
                                in_static = True
                                static_start = t
                if in_motion:
                    motion_segments.append(r['end_time'] - seg_start)
                if in_static:
                    static_segments.append(r['end_time'] - static_start)
                
                total_motion = sum(motion_segments)
                longest_motion = max(motion_segments) if motion_segments else 0
                longest_static = max(static_segments) if static_segments else 0
                
                # 有效回合条件：
                # 1. 最长连续运动 >= min_round_motion
                # 2. 最长连续静止 < max_static_gap（排除休息/非比赛场景）
                if longest_motion < min_round_motion:
                    print(f"  [FILTERED] Round {r['round_id']}: no sufficient motion "
                          f"(longest={longest_motion:.2f}s < {min_round_motion}s after t={validation_start:.1f}s)")
                elif longest_static > max_static_gap:
                    print(f"  [FILTERED] Round {r['round_id']}: too long static gap "
                          f"(longest_static={longest_static:.2f}s > {max_static_gap}s, likely non-match scene)")
                else:
                    valid_rounds.append(r)
            
            rounds = valid_rounds
            for i, r in enumerate(rounds, 1):
                r['round_id'] = i
            print(f"  [MOTION CHECK] {len(rounds)} rounds passed motion validation")
        
        # ===== 回合有效性验证 =====
        # 优先使用运动员身份追踪过滤（如果启用），否则回退到简单人数统计
        main_ids = getattr(self, 'main_athlete_ids', [])
        identity_enabled = len(main_ids) == 2  # 只有成功识别出2名主运动员时才使用身份过滤
        
        IN_GREEN_COURT_RATIO = round_constraints.get('in_green_court_ratio', 0.8)
        MIN_GREEN_COVERAGE = round_constraints.get('min_green_coverage', 0.03)  # 绿色场地覆盖率硬下限

        # 关键修复：YOLO就位检测(yolo_ready_stance)已保证2人左右站位，
        # 身份追踪在球员远距分离时会失败(track_id不匹配)，因此对stance检测的回合跳过身份过滤
        stance_round_indices = [i for i, r in enumerate(rounds) if r.get('start_method') == 'yolo_ready_stance']

        if identity_enabled:
            # ===== 身份追踪过滤模式：要求2名主运动员同时出现在绿色场地上 =====
            athlete1_id, athlete2_id = main_ids[0], main_ids[1]
            print(f"  [IDENTITY FILTER] 主运动员: ID={athlete1_id} 和 ID={athlete2_id}")
            
            def count_both_athletes_in_court(start_t, end_t):
                """统计时间段内2名主运动员同时在绿色场地上的帧数占比"""
                if not motion_data:
                    return 0.0
                total = 0
                both_present = 0
                for m in motion_data:
                    if start_t <= m['timestamp'] <= end_t:
                        total += 1
                        green_ids = m.get('green_track_ids', [])
                        if athlete1_id in green_ids and athlete2_id in green_ids:
                            both_present += 1
                return both_present / total if total > 0 else 0.0
            
            def has_both_athletes_near(t, window=1.0):
                """检查某时刻附近2名主运动员是否同时在绿色场地上"""
                for m in motion_data:
                    if abs(m['timestamp'] - t) <= window:
                        green_ids = m.get('green_track_ids', [])
                        if athlete1_id in green_ids and athlete2_id in green_ids:
                            return True
                return False

            def count_extra_people_frames(start_t, end_t, threshold=2):
                """统计时间段内绿色场地人数超过threshold的帧数占比"""
                if not motion_data:
                    return 0.0
                total = 0
                extra = 0
                for m in motion_data:
                    if start_t <= m['timestamp'] <= end_t:
                        total += 1
                        if m.get('players_in_green_court', 0) > threshold:
                            extra += 1
                return extra / total if total > 0 else 0.0

            def count_avg_green_people(start_t, end_t):
                """统计时间段内绿色场地人数的平均值"""
                if not motion_data:
                    return 0.0
                total = 0
                people_sum = 0
                for m in motion_data:
                    if start_t <= m['timestamp'] <= end_t:
                        total += 1
                        people_sum += m.get('players_in_green_court', 0)
                return people_sum / total if total > 0 else 0.0

            def count_two_player_frames(start_t, end_t):
                """统计时间段内 player_count==2 的帧数占比"""
                if not motion_data:
                    return 0.0
                total = 0
                two_player = 0
                for m in motion_data:
                    if start_t <= m['timestamp'] <= end_t:
                        total += 1
                        if m['player_count'] == 2:
                            two_player += 1
                return two_player / total if total > 0 else 0.0

            def count_green_coverage(start_t, end_t):
                """统计时间段内 >=2人在绿色场地的帧数占比"""
                if not motion_data:
                    return 0.0
                total = 0
                on_green = 0
                for m in motion_data:
                    if start_t <= m['timestamp'] <= end_t:
                        total += 1
                        if m.get('players_in_green_court', 0) >= 2:
                            on_green += 1
                return on_green / total if total > 0 else 0.0

            TWO_PLAYER_AT_START = round_constraints.get('two_player_at_start', True)
            # stance回合允许的最大"额外人物"占比（>2人出现在绿色场地）
            STANCE_EXTRA_PEOPLE_RATIO = round_constraints.get('stance_extra_people_ratio', 0.10)
            STANCE_MAX_AVG_PEOPLE = round_constraints.get('stance_max_avg_people', 2.5)
            STANCE_MIN_TWO_PLAYER_RATIO = round_constraints.get('stance_min_two_player_ratio', 0.20)

            if not rounds:
                print("  [WARNING] No rounds detected!")
            else:
                valid_rounds = []
                for idx, r in enumerate(rounds):
                    valid = True
                    reasons = []
                    is_stance_round = r.get('start_method') == 'yolo_ready_stance'
                    both_ratio = count_both_athletes_in_court(r['start_time'], r['end_time'])
                    start_ok = not TWO_PLAYER_AT_START or has_both_athletes_near(r['start_time'])

                    if is_stance_round:
                        # stance回合：姿态检测已验证"2人左右分居+静止"，是强信号
                        # 但无法区分"运动员比赛"和"场边两人静止"，需要额外验证
                        extra_ratio = count_extra_people_frames(r['start_time'], r['end_time'])
                        avg_people = count_avg_green_people(r['start_time'], r['end_time'])
                        two_p_ratio = count_two_player_frames(r['start_time'], r['end_time'])
                        green_cov = count_green_coverage(r['start_time'], r['end_time'])

                        if extra_ratio > STANCE_EXTRA_PEOPLE_RATIO:
                            valid = False
                            reasons.append(f"额外人物占比{extra_ratio:.0%}>{STANCE_EXTRA_PEOPLE_RATIO:.0%}")
                        elif avg_people > STANCE_MAX_AVG_PEOPLE:
                            valid = False
                            reasons.append(f"平均场地人数{avg_people:.1f}>{STANCE_MAX_AVG_PEOPLE:.1f}")
                        elif two_p_ratio < STANCE_MIN_TWO_PLAYER_RATIO:
                            valid = False
                            reasons.append(f"2人帧占比{two_p_ratio:.0%}<{STANCE_MIN_TWO_PLAYER_RATIO:.0%}")
                        elif green_cov < MIN_GREEN_COVERAGE:
                            valid = False
                            reasons.append(f"绿色场地覆盖率{green_cov:.1%}<{MIN_GREEN_COVERAGE:.0%}")
                        # 身份追踪作为参考（不硬性要求，因为追踪可能断链）
                        elif both_ratio > 0:
                            # 主运动员出现了，给过
                            pass
                    else:
                        # non-stance回合：信号较弱，使用较严格的身份过滤
                        if both_ratio < IN_GREEN_COURT_RATIO:
                            valid = False
                            reasons.append(f"双运动员在场占比{both_ratio:.0%}<{IN_GREEN_COURT_RATIO:.0%}")

                        if not start_ok:
                            valid = False
                            reasons.append("开始时2名主运动员未同时在绿色场地上")

                    if valid:
                        valid_rounds.append(r)
                    else:
                        print(f"  [FILTERED] Round {r['round_id']}: {', '.join(reasons)}")

                rounds = valid_rounds
                if rounds:
                    for i, r in enumerate(rounds, 1):
                        r['round_id'] = i
                print(f"  Total: {len(rounds)} rounds extracted (after identity filter)")
        else:
            # ===== 回退模式：简单人数统计过滤 =====
            print(f"  [COUNT FILTER] 未启用身份追踪，使用简单人数统计过滤")
            TWO_PLAYER_REQUIRED_RATIO = round_constraints.get('two_player_required_ratio', 0.5)
            TWO_PLAYER_AT_START = round_constraints.get('two_player_at_start', True)

            def count_two_player_frames(start_t, end_t):
                """统计时间段内 player_count==2 的帧数占比"""
                if not motion_data:
                    return 0.0
                total = 0
                two_player = 0
                for m in motion_data:
                    if start_t <= m['timestamp'] <= end_t:
                        total += 1
                        if m['player_count'] == 2:
                            two_player += 1
                return two_player / total if total > 0 else 0.0

            def count_green_court_frames(start_t, end_t):
                """统计时间段内 2人都在绿色场地内的帧数占比"""
                if not motion_data:
                    return 0.0
                total = 0
                in_court = 0
                for m in motion_data:
                    if start_t <= m['timestamp'] <= end_t:
                        total += 1
                        if m.get('players_in_green_court', 0) >= 2:
                            in_court += 1
                return in_court / total if total > 0 else 0.0

            def has_two_players_near(t, window=1.0):
                """检查某时刻附近是否有2名运动员"""
                for m in motion_data:
                    if abs(m['timestamp'] - t) <= window and m['player_count'] == 2:
                        return True
                return False

            if not rounds:
                print("  [WARNING] No rounds detected!")
            else:
                valid_rounds = []
                for r in rounds:
                    valid = True
                    reasons = []

                    # YOLO就位检测的回合不再完全跳过过滤
                    is_stance_round = r.get('start_method') == 'yolo_ready_stance'
                    
                    two_player_ratio = count_two_player_frames(r['start_time'], r['end_time'])
                    green_court_ratio = count_green_court_frames(r['start_time'], r['end_time'])
                    start_ok = not TWO_PLAYER_AT_START or has_two_players_near(r['start_time'])

                    if is_stance_round:
                        # stance回合宽松验证：至少开始附近有2人，或回合内有少量2人帧
                        start_has_two = has_two_players_near(r['start_time'], window=2.0)
                        stance_min_2p = TWO_PLAYER_REQUIRED_RATIO * 0.3
                        stance_min_green = IN_GREEN_COURT_RATIO * 0.3
                        if not start_has_two and (two_player_ratio < stance_min_2p or green_court_ratio < stance_min_green):
                            valid = False
                            reasons.append(f"stance回合2人占比{two_player_ratio:.1%}/{green_court_ratio:.1%}过低且开始附近不足2人")
                        # 绿色覆盖率硬下限（所有stance回合都必须满足）
                        if valid and green_court_ratio < MIN_GREEN_COVERAGE:
                            valid = False
                            reasons.append(f"绿色场地覆盖率{green_court_ratio:.1%}<{MIN_GREEN_COVERAGE:.0%}")
                    else:
                        if two_player_ratio < TWO_PLAYER_REQUIRED_RATIO:
                            valid = False
                            reasons.append(f"2人占比{two_player_ratio:.0%}<{TWO_PLAYER_REQUIRED_RATIO:.0%}")

                        if green_court_ratio < IN_GREEN_COURT_RATIO:
                            valid = False
                            reasons.append(f"在绿色场地占比{green_court_ratio:.0%}<{IN_GREEN_COURT_RATIO:.0%}")

                        if not start_ok:
                            valid = False
                            reasons.append("开始时不足2人")

                    if valid:
                        valid_rounds.append(r)
                    else:
                        print(f"  [FILTERED] Round {r['round_id']}: {', '.join(reasons)}")

                rounds = valid_rounds
                if rounds:
                    for i, r in enumerate(rounds, 1):
                        r['round_id'] = i
                print(f"  Total: {len(rounds)} rounds extracted (after filters)")

        # ===== 单人占比异常检测 =====
        # 回合中单人帧占比超过阈值且无羽毛球检测 → 可能不是真实对打
        if rounds and motion_data:
            SOLO_MAX_RATIO = round_constraints.get('single_player_max_ratio', 0.80)
            valid_rounds = []
            for r in rounds:
                total = 0
                solo = 0
                has_shuttle = False
                for m in motion_data:
                    if r['start_time'] <= m['timestamp'] <= r['end_time']:
                        total += 1
                        if m.get('player_count', 0) == 1:
                            solo += 1
                        if m.get('shuttlecock_detected', False):
                            has_shuttle = True

                solo_ratio = solo / total if total > 0 else 0.0

                if solo_ratio > SOLO_MAX_RATIO and not has_shuttle:
                    print(f"  [FILTERED] Round {r['round_id']}: 单人帧占比{solo_ratio:.0%}>{SOLO_MAX_RATIO:.0%}"
                          f" 且无羽毛球检出 → 非对打场景")
                else:
                    valid_rounds.append(r)

            if len(valid_rounds) < len(rounds):
                rounds = valid_rounds
                for i, r in enumerate(rounds, 1):
                    r['round_id'] = i
                print(f"  [SOLO CHECK] {len(rounds)} rounds passed solo-player validation")

        # ===== 标记过滤掉的回合（供前端用户选择参考）=====
        # 使用 _raw_round_id（原始ID）而非重编号后的 round_id，确保与 all_raw_rounds 匹配
        final_raw_ids = set(r['_raw_round_id'] for r in rounds)
        for raw_r in results.get('all_raw_rounds', []):
            if raw_r['round_id'] not in final_raw_ids:
                raw_r['_auto_filtered'] = True
                raw_r['_filter_reason'] = '系统自动过滤（运动不足或运动员不完整）'
        results['auto_approved_ids'] = sorted(final_raw_ids)

        # 清理内部字段，避免泄露到前端数据
        for r in rounds:
            r.pop('_raw_round_id', None)

        return rounds

    def _post_process(self, predictions):
        """后处理：过滤低置信度并平滑结果"""
        valid = [p for p in predictions if p['confidence'] >= self.confidence_threshold]
        
        if not valid:
            return []
        
        final = []
        valid.sort(key=lambda x: x['center_frame'])
        
        last_p = valid[0]
        current_group = [last_p]
        
        for i in range(1, len(valid)):
            curr_p = valid[i]
            if curr_p['center_frame'] - last_p['center_frame'] < self.min_interval:
                current_group.append(curr_p)
            else:
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
            'results': results,
            'rounds': results.get('rounds', [])
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "03_model/trained/best_model.pth"
        
        predictor = ActionPredictorFast(model_path)
        predictor.predict_video(video_path)
    else:
        print("Usage: python model_predict_optimized.py <video_path> [model_path]")
