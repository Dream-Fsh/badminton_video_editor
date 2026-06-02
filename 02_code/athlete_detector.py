"""
运动员检测与运动分析模块 (基于 YOLO)
功能：识别场地上的运动员并分析其运动状态（是否静止）
      跟踪羽毛球位置，检测球落地静止事件（回合结束）
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import torch
from typing import Tuple, Optional


class ShuttlecockTracker:
    """羽毛球状态跟踪器 - 两阶段回合结束检测
    
    阶段1: 检测球下落（落地静止 或 连续消失）
    阶段2: 球下落后，等待球再次出现在运动员手中 → 触发回合结束
    
    关键机制：
    1. 球下落后必须等球回到运动员手中才触发
    2. 事件触发后进入冷却期，冷却期内不触发新事件
    """
    
    def __init__(self, still_duration=0.5, still_threshold=8.0, fps=30, sample_rate=3,
                 vanish_duration=0.5, vanish_threshold=5):
        self.still_duration = still_duration
        self.still_threshold = still_threshold
        self.fps = fps
        self.sample_rate = sample_rate
        self.vanish_duration = vanish_duration
        
        # 球位置历史（用于检测落地静止）
        max_history = int(max(still_duration, vanish_duration) * fps / sample_rate) + 5
        self.position_history = deque(maxlen=max_history)
        
        # 球消失跟踪
        self.vanish_count = 0
        self.vanish_needed = int(vanish_duration * fps / sample_rate)
        
        # 球落地跟踪
        self.is_landed = False
        
        # ===== 两阶段状态机 =====
        self.state = 'watching'  # 'watching' | 'ball_down'
        self.ball_down_time = None      # 球下落的时间戳
        self.ball_down_frame = None     # 球下落的帧号
        self.ball_down_method = None    # 下落原因（landed/vanished）
        
        # 冷却期
        self.cooldown_until_frame = -999
        self.cooldown_seconds = 3.0
    
    def _is_ball_near_athlete(self, ball_center, players, expand_ratio=1.3):
        """判断球是否在运动员手中（检测框内或附近）"""
        if not players or not ball_center:
            return False
        bx, by = ball_center
        for p in players:
            x1, y1, x2, y2 = p['box']
            # 稍微扩大检测框，因为手可能在身体两侧
            cx1 = x1 - (x2 - x1) * (expand_ratio - 1) / 2
            cy1 = y1 - (y2 - y1) * (expand_ratio - 1) / 2
            cx2 = x2 + (x2 - x1) * (expand_ratio - 1) / 2
            cy2 = y2 + (y2 - y1) * (expand_ratio - 1) / 2
            if cx1 <= bx <= cx2 and cy1 <= by <= cy2:
                return True
        return False
    
    def update(self, frame_idx, shuttle_result, athletes_static=False, athlete_count=0, players=None):
        """
        更新一帧的检测结果，两阶段回合结束检测
        
        参数:
            frame_idx:       当前帧号
            shuttle_result:  detect_shuttlecock 返回值，None=未检测到球
            athletes_static: 运动员是否静止
            athlete_count:   检测到的运动员数量
            players:         运动员检测结果列表（含box），用于判断球是否在手中
            
        返回:
            dict or None: 如果判定回合结束，返回事件信息
        """
        # 冷却期内不触发
        if frame_idx < self.cooldown_until_frame:
            return None
        
        # ===== 阶段1: watching — 监测球是否下落 =====
        if self.state == 'watching':
            if shuttle_result is None:
                # 未检测到球 → 计数消失
                self.vanish_count += 1
                if self.vanish_count >= self.vanish_needed:
                    self.state = 'ball_down'
                    self.ball_down_time = frame_idx / self.fps
                    self.ball_down_frame = frame_idx
                    self.ball_down_method = 'shuttlecock_vanished'
            else:
                # 检测到球 → 检查是否落地静止
                self.vanish_count = 0
                center = shuttle_result['center']
                self.position_history.append({
                    'frame': frame_idx,
                    'center': center,
                    'conf': shuttle_result['conf']
                })
                
                needed_samples = int(self.still_duration * self.fps / self.sample_rate)
                if len(self.position_history) >= needed_samples:
                    recent = list(self.position_history)[-needed_samples:]
                    positions = np.array([p['center'] for p in recent])
                    std_dev = np.std(positions, axis=0)
                    
                    if np.all(std_dev < self.still_threshold) and not self.is_landed:
                        self.is_landed = True
                        self.state = 'ball_down'
                        self.ball_down_time = frame_idx / self.fps
                        self.ball_down_frame = frame_idx
                        self.ball_down_method = 'shuttlecock_landed'
                else:
                    self.is_landed = False
        
        # ===== 阶段2: ball_down — 等待球出现在运动员手中 =====
        elif self.state == 'ball_down':
            if shuttle_result is not None:
                ball_center = shuttle_result['center']
                # 判断球是否在运动员手中
                if self._is_ball_near_athlete(ball_center, players):
                    # 球回到运动员手中 → 回合结束！
                    self.cooldown_until_frame = frame_idx + int(self.cooldown_seconds * self.fps)
                    
                    event = {
                        'timestamp': frame_idx / self.fps,
                        'frame': frame_idx,
                        'method': 'round_end',
                        'ball_down_method': self.ball_down_method,
                        'ball_down_time': self.ball_down_time,
                        'reason': f'ball {self.ball_down_method.replace("shuttlecock_", "")} at {self.ball_down_time:.1f}s → picked up at {frame_idx / self.fps:.1f}s'
                    }
                    
                    # 重置到 watching 状态（保留冷却）
                    self._reset_for_next_round()
                    return event
                else:
                    # 检测到球但不在运动员手中（可能球还在地上被弹起等），继续等待
                    pass
        
        return None
    
    def _reset_for_next_round(self):
        """为下一个回合重置状态，保留冷却时间"""
        self.position_history.clear()
        self.vanish_count = 0
        self.is_landed = False
        self.state = 'watching'
        self.ball_down_time = None
        self.ball_down_frame = None
        self.ball_down_method = None
    
    def reset(self):
        """完全重置所有跟踪状态，包括冷却时间"""
        self._reset_for_next_round()
        self.cooldown_until_frame = -999


class AthleteDetector:
    def __init__(self, model_type="yolov8n.pt", device="auto", shuttlecock_model=None):
        """
        初始化 YOLO 检测器
        
        参数:
            model_type: 运动员检测模型路径
            device: 设备
            shuttlecock_model: 羽毛球检测微调模型路径（优先使用）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else ("cpu" if device == "auto" else device))
        self.model = YOLO(model_type)
        self.model.to(self.device)
        self.person_class_id = 0  # COCO 人的类别 ID
        self.shuttlecock_class_id = 32  # COCO sports ball 类别 ID（备用）
        
        # 加载微调的羽毛球检测模型（如果提供）
        self.shuttlecock_model = None
        if shuttlecock_model:
            try:
                self.shuttlecock_model = YOLO(shuttlecock_model)
                self.shuttlecock_model.to(self.device)
            except Exception as e:
                print(f"[WARN] 加载羽毛球检测模型失败: {e}，将使用 COCO sports ball 备用")
        
    def detect_athletes(self, frame, constraints=None):
        """
        检测画面中的所有人，并筛选出可能的运动员（通常是最大的两个）
        返回: list of dicts，每个包含 box, conf, center, area, in_green_court
        
        参数:
            frame: 输入图像
            constraints: 可选的约束配置字典，包含:
                - min_confidence: 最小置信度
                - court_boundary: 检测范围约束 {enabled, x_min, x_max, y_min, y_max}
                - green_court_boundary: 绿色场地边界 {enabled, x_min, x_max, y_min, y_max}
                - max_count: 最大返回人数
        """
        results = self.model(frame, classes=[self.person_class_id], verbose=False)[0]
        
        # 默认约束
        min_confidence = constraints.get('min_confidence', 0.3) if constraints else 0.3
        court_boundary = constraints.get('court_boundary', {}) if constraints else {}
        green_court_boundary = constraints.get('green_court_boundary', {}) if constraints else {}
        max_count = constraints.get('max_count', 2) if constraints else 2
        
        athletes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 置信度约束检查
            if conf < min_confidence:
                continue
            
            # 检测范围约束检查（用于筛选画面中的人）
            if court_boundary and court_boundary.get('enabled', False):
                if not (court_boundary['x_min'] <= cx <= court_boundary['x_max'] and
                        court_boundary['y_min'] <= cy <= court_boundary['y_max']):
                    continue  # 在检测范围外，跳过
            
            # 判断是否在绿色场地内（默认不在，只有边界启用且点在边界内才算在）
            in_green_court = False
            if green_court_boundary and green_court_boundary.get('enabled', False):
                in_green_court = (green_court_boundary['x_min'] <= cx <= green_court_boundary['x_max'] and
                                  green_court_boundary['y_min'] <= cy <= green_court_boundary['y_max'])
            
            # 计算面积以便排序
            area = (x2 - x1) * (y2 - y1)
            athletes.append({
                'box': [x1, y1, x2, y2],
                'conf': conf,
                'center': (cx, cy),
                'area': area,
                'in_green_court': in_green_court
            })
            
        # 按面积降序排列，取前N个作为运动员
        athletes.sort(key=lambda x: x['area'], reverse=True)
        return athletes[:max_count]

    def detect_shuttlecock(self, frame, min_confidence=0.3):
        """
        检测羽毛球位置
        返回: {'box': [x1,y1,x2,y2], 'center': (cx,cy), 'conf': 置信度} 或 None
        """
        # 优先使用微调的羽毛球检测模型
        model = self.shuttlecock_model if self.shuttlecock_model else self.model
        class_ids = [0] if self.shuttlecock_model else [self.shuttlecock_class_id]
        
        results = model(frame, classes=class_ids, verbose=False)[0]
        
        best_ball = None
        for box in results.boxes:
            conf = box.conf[0].item()
            if conf > min_confidence:
                if not best_ball or conf > best_ball['conf']:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_ball = {
                        'box': [x1, y1, x2, y2],
                        'center': ((x1+x2)/2, (y1+y2)/2),
                        'conf': conf
                    }
        return best_ball

    def analyze_motion(self, athlete_history, threshold=3.0):
        """
        根据历史位置判断运动员是否相对静止
        athlete_history: 过去几帧的中心点坐标列表 (建议传入 1 秒的帧数)
        """
        if len(athlete_history) < 2:
            return False
            
        # 算法优化：计算历史点集的波动范围 (Standard Deviation) 
        # 而不是单纯的相邻位移，这能更好地应对细微的摆动
        centers = np.array(athlete_history)
        std_dev = np.std(centers, axis=0)
        
        # 如果 X 和 Y 方向的波动都在阈值内，则认为绝对静止
        return np.all(std_dev < threshold)


class SimplePersonTracker:
    """
    运动员位置
    简单的人员追踪器 - 基于中心点距离的贪心匹配，为每个人分配持久 ID
    支持轨迹合并：将碎片化的同一人轨迹合并为一个 ID
    """

    def __init__(self, max_distance=150, max_lost_frames=10):
        """
        参数:
            max_distance: 匹配最大距离（像素），超过此距离视为新人
            max_lost_frames: 轨迹丢失多少帧后删除
        """
        self.tracks = {}  # track_id -> {'first_center', 'last_center', 'first_seen_frame', 'last_seen_frame', 'appearances', 'green_court_count'}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames

    def update(self, detections, frame_idx):
        """
        更新追踪：为每个检测框分配 track_id

        参数:
            detections: list of dicts，每个含 'center' 和 'in_green_court' 键
            frame_idx: 当前帧号

        返回:
            更新后的 detections 列表，每个多了 'track_id' 键
        """
        # 清理长时间未出现的轨迹
        lost_ids = [tid for tid, t in self.tracks.items()
                    if frame_idx - t['last_seen_frame'] > self.max_lost_frames]
        for tid in lost_ids:
            del self.tracks[tid]

        if not detections:
            return []

        # 首次检测，直接初始化
        if not self.tracks:
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                det['track_id'] = tid
                self.tracks[tid] = {
                    'first_center': det['center'],
                    'last_center': det['center'],
                    'first_seen_frame': frame_idx,
                    'last_seen_frame': frame_idx,
                    'appearances': 1,
                    'green_court_count': 1 if det.get('in_green_court', False) else 0
                }
            return detections

        # 贪心最近邻匹配
        assigned_tracks = set()
        assigned_dets = set()

        # 计算所有配对距离
        pairs = []
        for tid, track in self.tracks.items():
            for di, det in enumerate(detections):
                dist = np.linalg.norm(
                    np.array(track['last_center']) - np.array(det['center'])
                )
                pairs.append((dist, tid, di))

        pairs.sort(key=lambda x: x[0])

        for dist, tid, di in pairs:
            if dist > self.max_distance:
                break
            if tid in assigned_tracks or di in assigned_dets:
                continue
            detections[di]['track_id'] = tid
            self.tracks[tid]['last_center'] = detections[di]['center']
            self.tracks[tid]['last_seen_frame'] = frame_idx
            self.tracks[tid]['appearances'] += 1
            if detections[di].get('in_green_court', False):
                self.tracks[tid]['green_court_count'] += 1
            assigned_tracks.add(tid)
            assigned_dets.add(di)

        # 未匹配的检测 → 创建新轨迹
        for di, det in enumerate(detections):
            if di not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                det['track_id'] = tid
                self.tracks[tid] = {
                    'first_center': det['center'],
                    'last_center': det['center'],
                    'first_seen_frame': frame_idx,
                    'last_seen_frame': frame_idx,
                    'appearances': 1,
                    'green_court_count': 1 if det.get('in_green_court', False) else 0
                }

        return detections

    def merge_fragmented_tracks(self, merge_gap_frames=30, merge_distance=250):
        """合并碎片化的同一人轨迹
        
        当一个轨迹结束、另一个轨迹紧接着在附近开始时，说明是同一人被短暂丢失后重新检测到。
        
        参数:
            merge_gap_frames: 两个轨迹之间的最大间隔帧数（超过此间隔不合并）
            merge_distance: 两个轨迹交界处的最大空间距离（像素）
            
        返回:
            dict: ID 映射 {被合并的旧ID: 合并后的存活ID}
        """
        id_mapping = {}  # 被合并的ID → 存活的ID
        merged = True
        while merged:
            merged = False
            track_ids = sorted(self.tracks.keys())
            for i in range(len(track_ids)):
                if merged:
                    break
                for j in range(i + 1, len(track_ids)):
                    t1 = self.tracks[track_ids[i]]
                    t2 = self.tracks[track_ids[j]]

                    # 检查两个轨迹的时间关系：一个结束，另一个紧接着开始
                    gaps = [
                        t2['first_seen_frame'] - t1['last_seen_frame'],
                        t1['first_seen_frame'] - t2['last_seen_frame'],
                    ]
                    min_gap = min(g for g in gaps if g >= 0) if any(g >= 0 for g in gaps) else float('inf')

                    if min_gap > merge_gap_frames:
                        continue

                    # 检查空间距离
                    if gaps[0] >= 0 and gaps[0] <= merge_gap_frames:
                        dist = np.linalg.norm(np.array(t1['last_center']) - np.array(t2['first_center']))
                    elif gaps[1] >= 0 and gaps[1] <= merge_gap_frames:
                        dist = np.linalg.norm(np.array(t2['last_center']) - np.array(t1['first_center']))
                    else:
                        continue

                    if dist <= merge_distance:
                        # 合并 t2 到 t1
                        surviving_id = track_ids[i]
                        deleted_id = track_ids[j]
                        self.tracks[surviving_id]['appearances'] += t2['appearances']
                        self.tracks[surviving_id]['green_court_count'] += t2['green_court_count']
                        if t2['first_seen_frame'] < t1['first_seen_frame']:
                            self.tracks[surviving_id]['first_center'] = t2['first_center']
                            self.tracks[surviving_id]['first_seen_frame'] = t2['first_seen_frame']
                        if t2['last_seen_frame'] > t1['last_seen_frame']:
                            self.tracks[surviving_id]['last_center'] = t2['last_center']
                            self.tracks[surviving_id]['last_seen_frame'] = t2['last_seen_frame']
                        del self.tracks[deleted_id]
                        # 记录映射：被删除的ID → 存活的ID（链式查找）
                        id_mapping[deleted_id] = id_mapping.get(deleted_id, id_mapping.get(surviving_id, surviving_id))
                        # 同时更新之前已映射到 deleted_id 的记录
                        for old_id in list(id_mapping.keys()):
                            if id_mapping[old_id] == deleted_id:
                                id_mapping[old_id] = surviving_id
                        merged = True
                        break
        
        return id_mapping

    def get_main_athletes(self, n=2, min_appearances=5):
        """
        获取在绿色场地上出现次数最多的前 n 个轨迹 ID

        返回:
            list of track_id，长度最多为 n
        """
        valid = [(tid, t['green_court_count']) for tid, t in self.tracks.items()
                 if t['appearances'] >= min_appearances]
        valid.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in valid[:n]]


class GreenPixelDetector:
    """绿色像素检测器 - 使用HSV颜色空间检测画面中的绿色比赛场地

    原理：羽毛球场地通常为特定绿色（H通道约35-85），通过HSV颜色空间
    提取绿色像素区域，计算占比来判断帧画面是否为真实比赛场景。
    """

    def __init__(self,
                 hsv_lower: Tuple[int, int, int] = (35, 40, 40),
                 hsv_upper: Tuple[int, int, int] = (85, 255, 255),
                 min_green_ratio: float = 0.05,
                 sample_rate: int = 30,
                 required_green_frame_ratio: float = 0.3):
        """
        参数:
            hsv_lower: HSV下界 (H, S, V)，标准草地绿约35-85度
            hsv_upper: HSV上界 (H, S, V)
            min_green_ratio: 单帧中绿色像素占比低于此值判定该帧无绿色
            sample_rate: 采样率（每N帧检测一帧）
            required_green_frame_ratio: 采样帧中至少有比例含绿色，否则剔除回合
        """
        self.lower_green = np.array(hsv_lower, dtype=np.uint8)
        self.upper_green = np.array(hsv_upper, dtype=np.uint8)
        self.min_green_ratio = min_green_ratio
        self.sample_rate = sample_rate
        self.required_green_frame_ratio = required_green_frame_ratio

    def detect_green_ratio(self, frame: np.ndarray) -> float:
        """检测单帧画面中绿色像素占比

        参数:
            frame: BGR图像 (OpenCV默认格式)

        返回:
            绿色像素占画面总像素的比例 (0.0 ~ 1.0)
        """
        if frame is None or frame.size == 0:
            return 0.0

        # 转为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建绿色掩膜
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # 计算绿色像素占比
        total_pixels = frame.shape[0] * frame.shape[1]
        green_pixels = cv2.countNonZero(mask)

        return green_pixels / total_pixels

    def has_green(self, frame: np.ndarray) -> bool:
        """判断单帧是否包含足够的绿色元素

        参数:
            frame: BGR图像

        返回:
            True=包含绿色，False=不含绿色
        """
        ratio = self.detect_green_ratio(frame)
        return ratio >= self.min_green_ratio

    def check_video_segment(self, video_path: str,
                            start_time: float, end_time: float,
                            verbose: bool = True) -> bool:
        """检测视频片段中是否有足够多的帧包含绿色元素

        对片段进行等间隔采样，统计含绿色帧的占比，若超过阈值则保留。

        参数:
            video_path: 视频文件路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            verbose: 是否打印调试信息

        返回:
            True=包含绿色元素（保留），False=无绿色元素（剔除）
        """
        import os
        if not os.path.exists(video_path):
            if verbose:
                print(f"  [GREEN_CHECK] 视频文件不存在: {video_path}")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if verbose:
                print(f"  [GREEN_CHECK] 无法打开视频: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # 默认值

        # 计算采样帧
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        total_frames = end_frame - start_frame

        if total_frames <= 0:
            cap.release()
            return False

        # 等间隔采样，最多采样100帧避免过慢
        sample_indices = list(range(start_frame, end_frame, self.sample_rate))
        if len(sample_indices) > 100:
            step = len(sample_indices) // 100
            sample_indices = sample_indices[::step]

        green_frame_count = 0
        total_checked = 0

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if self.has_green(frame):
                green_frame_count += 1
            total_checked += 1

        cap.release()

        if total_checked == 0:
            return False

        green_ratio = green_frame_count / total_checked

        if verbose:
            print(f"  [GREEN_CHECK] 时段 {start_time:.1f}s-{end_time:.1f}s: "
                  f"采样{total_checked}帧, 含绿色{green_frame_count}帧 "
                  f"({green_ratio:.0%}), 要求含绿帧占比≥{self.required_green_frame_ratio:.0%}"
                  f" → {'✓ 保留' if green_ratio >= self.required_green_frame_ratio else '✗ 剔除'}")

        return green_ratio >= self.required_green_frame_ratio
