"""
VideoBadminton 发球数据增强脚本
===============================
功能：从 VideoBadminton 数据集中提取发球视频帧，
      与现有 round_end 标注合并，扩充 I3D 训练数据。

效果：round_start 从 33 条扩充到 ~1900 条

使用前提：
  1. VideoBadminton 数据集已下载到 01_data/VideoBadminton_Dataset/

使用方法：
  # 测试（只处理前20个）
  python 08_scripts/prepare_videobadminton.py --limit 20
  # 全量处理
  python 08_scripts/prepare_videobadminton.py
"""

import os
import sys
import json
import cv2
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 只提取发球类别
SERVE_CLASSES = ["00_Short Serve", "13_Long Serve"]

# 最少帧数要求（与 config.yaml 的 sequence_length 一致）
MIN_FRAMES = 16


def scan_serve_videos(dataset_dir):
    """只扫描发球类别的视频"""
    videos = []
    for folder_name in SERVE_CLASSES:
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"[WARN] 文件夹不存在: {folder_path}")
            continue
        for v in sorted(os.listdir(folder_path)):
            if v.endswith('.mp4'):
                videos.append(os.path.join(folder_path, v))
    return videos


def extract_frames(video_path, frames_dir, video_prefix="vb"):
    """
    从视频提取帧到 frames_dir/vb_XXX/ 目录
    返回: video_name 或 None
    """
    # 生成唯一视频名（避免与现有视频冲突）
    original_name = Path(video_path).stem
    video_name = f"vb_{original_name}"
    out_dir = os.path.join(frames_dir, video_name)

    if os.path.exists(out_dir):
        existing = [f for f in os.listdir(out_dir) if f.endswith('.jpg')]
        if len(existing) >= MIN_FRAMES:
            return video_name

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < MIN_FRAMES:
        cap.release()
        # 清理空目录
        if not os.listdir(out_dir):
            os.rmdir(out_dir)
        return None

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(out_dir, f"frame_{count:06d}.jpg")
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1
    cap.release()

    if count < MIN_FRAMES:
        shutil.rmtree(out_dir, ignore_errors=True)
        return None

    return video_name


def main():
    parser = argparse.ArgumentParser(description="VideoBadminton 发球数据增强")
    parser.add_argument("--input", "-i",
                        default="01_data/VideoBadminton_Dataset/VideoBadminton_Dataset")
    parser.add_argument("--limit", "-n", type=int, default=0,
                        help="限制处理数量（0=全部）")
    parser.add_argument("--skip-extract", action="store_true",
                        help="跳过帧提取")
    args = parser.parse_args()

    input_dir = os.path.join(PROJECT_ROOT, args.input)
    frames_dir = os.path.join(PROJECT_ROOT, "01_data/processed_frames")
    label_path = os.path.join(PROJECT_ROOT, "01_data/annotations/merged_labels.json")

    print("=" * 50)
    print("VideoBadminton 发球数据增强")
    print("=" * 50)

    # ======== 第1步：扫描发球视频 ========
    print(f"\n[1] 扫描发球视频（{', '.join(SERVE_CLASSES)}）...")
    if not os.path.exists(input_dir):
        print(f"[ERROR] 数据集目录不存在: {input_dir}")
        return

    serve_videos = scan_serve_videos(input_dir)
    if not serve_videos:
        print("[ERROR] 未找到发球视频！")
        return

    print(f"[OK] 找到 {len(serve_videos)} 个发球视频")

    # 按类别统计
    cls_count = Counter()
    for v in serve_videos:
        parent = Path(v).parent.name
        cls_count[parent] += 1
    for cls, cnt in sorted(cls_count.items()):
        print(f"    {cls}: {cnt}")

    if args.limit > 0:
        serve_videos = serve_videos[:args.limit]
        print(f"[INFO] 限制处理: {args.limit}")

    # ======== 第2步：提取帧 ========
    print(f"\n[2] 提取帧到 {frames_dir}...")
    os.makedirs(frames_dir, exist_ok=True)

    vb_video_names = []  # 成功提取的视频名

    if args.skip_extract:
        # 读取已有的 vb_ 前缀帧目录
        print("[INFO] 跳过提取，读取已有帧...")
        for d in sorted(os.listdir(frames_dir)):
            if d.startswith("vb_") and os.path.isdir(os.path.join(frames_dir, d)):
                existing = [f for f in os.listdir(os.path.join(frames_dir, d))
                           if f.endswith('.jpg')]
                if len(existing) >= MIN_FRAMES:
                    vb_video_names.append(d)
        print(f"[OK] 读取到 {len(vb_video_names)} 个已有帧目录")
    else:
        for video_path in tqdm(serve_videos, desc="提取帧"):
            vname = extract_frames(video_path, frames_dir)
            if vname:
                vb_video_names.append(vname)
        print(f"[OK] 成功提取 {len(vb_video_names)}/{len(serve_videos)} 个视频")

    if not vb_video_names:
        print("[ERROR] 没有可用的视频帧！")
        return

    # ======== 第3步：生成发球标注 ========
    print(f"\n[3] 生成发球标注...")
    serve_annotations = []
    for i, vname in enumerate(vb_video_names):
        # 获取帧数
        frame_dir = os.path.join(frames_dir, vname)
        num_frames = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])

        if num_frames < MIN_FRAMES:
            continue

        start_frame = max(0, (num_frames - MIN_FRAMES) // 2)
        end_frame = start_frame + MIN_FRAMES - 1

        serve_annotations.append({
            "action_id": i + 1,
            "class": "round_start",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "video_name": vname,
            "description": f"VideoBadminton发球 #{i+1}",
            "source": "VideoBadminton"
        })

    print(f"[OK] 生成 {len(serve_annotations)} 条发球标注")

    # ======== 第4步：合并现有标注 ========
    print(f"\n[4] 合并现有标注...")
    existing_annotations = []

    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            existing_annotations = json.load(f)
        print(f"[OK] 读取现有标注: {len(existing_annotations)} 条")

    # 保留所有现有的 round_end（球落地）标注
    round_end_annotations = [a for a in existing_annotations if a["class"] == "round_end"]
    # 保留现有的 round_start 标注
    round_start_existing = [a for a in existing_annotations if a["class"] == "round_start"]

    print(f"    现有 round_start: {len(round_start_existing)} 条")
    print(f"    现有 round_end:   {len(round_end_annotations)} 条")
    print(f"    新增 round_start: {len(serve_annotations)} 条（VideoBadminton）")

    # 合并：新增发球 + 现有球落地 + 现有发球
    merged = serve_annotations + round_end_annotations + round_start_existing

    # 重新编号
    for i, ann in enumerate(merged):
        ann["action_id"] = i + 1

    # 备份原文件
    if os.path.exists(label_path):
        backup_path = label_path.replace('.json', '_backup.json')
        shutil.copy2(label_path, backup_path)
        print(f"[INFO] 原标注已备份到: {backup_path}")

    # 保存合并后的标注
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # 统计
    final_count = Counter(a["class"] for a in merged)
    print(f"\n[OK] 合并完成！")
    print(f"    总标注数: {len(merged)}")
    print(f"    round_start: {final_count.get('round_start', 0)} 条")
    print(f"    round_end:   {final_count.get('round_end', 0)} 条")
    print(f"    标注文件: {label_path}")
    print(f"    帧目录:   {frames_dir}")

    print(f"\n{'=' * 50}")
    print(f"发球数据从 {len(round_start_existing)} -> {final_count.get('round_start', 0)} 条")
    print(f"现在可以直接运行 model_train.py 训练。")


if __name__ == "__main__":
    main()
