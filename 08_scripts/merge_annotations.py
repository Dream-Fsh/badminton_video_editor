"""
合并标注文件脚本
功能：将所有单独的标注文件合并为一个统一的merged_labels.json
"""

import os
import json
from pathlib import Path


def merge_annotations():
    """合并所有标注文件"""
    
    annotations_dir = "01_data/annotations"
    output_path = "01_data/annotations/merged_labels.json"
    
    print("=" * 70)
    print("合并标注文件")
    print("=" * 70)
    
    # 查找所有标注文件
    annotation_files = [
        f for f in os.listdir(annotations_dir)
        if f.endswith('_annotations.json') and f != 'merged_labels.json'
    ]
    
    if not annotation_files:
        print(f"\n✗ 在 {annotations_dir} 中未找到标注文件")
        return False
    
    print(f"\n找到 {len(annotation_files)} 个标注文件")
    
    merged_data = []
    total_annotations = 0
    
    # 逐个读取并合并
    for ann_file in sorted(annotation_files):
        ann_path = os.path.join(annotations_dir, ann_file)
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            video_name = data.get('video_name', '')
            annotations = data.get('annotations', [])
            
            print(f"  ✓ {ann_file:35s} - {len(annotations)} 个标注")
            
            # 提取annotations部分
            for ann in annotations:
                # 添加视频名称信息
                ann['video_name'] = video_name
                merged_data.append(ann)
                total_annotations += 1
                
        except Exception as e:
            print(f"  ✗ 读取失败 {ann_file}: {e}")
            continue
    
    # 保存合并后的数据
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 成功合并标注文件")
        print(f"  总标注数量: {total_annotations}")
        print(f"  保存路径: {output_path}")
        
        # 统计各类别数量
        class_counts = {}
        for ann in merged_data:
            cls = ann.get('class', 'unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        print(f"\n标注类别统计:")
        for cls, count in class_counts.items():
            print(f"  {cls:15s}: {count} 个")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 保存失败: {e}")
        return False


if __name__ == "__main__":
    try:
        success = merge_annotations()
        
        if success:
            print("\n" + "=" * 70)
            print("✓ 标注文件合并完成！")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("✗ 标注文件合并失败")
            print("=" * 70)
            
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    # input("\n按回车键退出...")
