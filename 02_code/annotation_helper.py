"""
标注辅助工具
功能：
1. 获取视频信息
2. 时间戳转帧号
3. 创建标注模板
4. 验证标注文件
"""

import os
import cv2
import json
from pathlib import Path


class AnnotationHelper:
    """标注辅助工具类"""
    
    def __init__(self):
        # 获取项目根目录
        self.project_root = Path(__file__).parent.parent.absolute()
        self.raw_videos_dir = str(self.project_root / "01_data" / "raw_videos")
        self.annotations_dir = str(self.project_root / "01_data" / "annotations")
    
    def get_video_info(self, video_path):
        """
        获取视频信息
        
        参数:
            video_path: 视频文件路径
        
        返回:
            字典包含视频信息
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        info = {
            'video_name': Path(video_path).stem,
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0
        }
        
        info['duration'] = info['total_frames'] / info['fps']
        
        cap.release()
        
        return info
    
    def time_to_frame(self, time_seconds, fps):
        """
        时间戳转帧号
        
        参数:
            time_seconds: 时间（秒）
            fps: 帧率
        
        返回:
            帧号
        """
        return int(time_seconds * fps)
    
    def frame_to_time(self, frame_number, fps):
        """
        帧号转时间戳
        
        参数:
            frame_number: 帧号
            fps: 帧率
        
        返回:
            时间（秒）
        """
        return frame_number / fps
    
    def create_annotation_template(self, video_path, output_path=None):
        """
        创建标注模板文件
        
        参数:
            video_path: 视频文件路径
            output_path: 输出路径（可选）
        
        返回:
            模板文件路径
        """
        # 获取视频信息
        info = self.get_video_info(video_path)
        
        # 创建模板
        template = {
            "video_name": info['video_name'],
            "total_frames": info['total_frames'],
            "frame_rate": int(info['fps']),
            "annotations": [
                {
                    "action_id": 1,
                    "class": "round_start",
                    "start_frame": 100,
                    "end_frame": 120,
                    "description": "第1回合发球（示例，请修改）"
                },
                {
                    "action_id": 2,
                    "class": "round_end",
                    "start_frame": 450,
                    "end_frame": 470,
                    "description": "第1回合球落地（示例，请修改）"
                }
            ]
        }
        
        # 确定输出路径
        if output_path is None:
            output_path = os.path.join(
                self.annotations_dir,
                f"{info['video_name']}_annotations.json"
            )
        
        # 保存模板
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 标注模板已创建: {output_path}")
        print(f"\n视频信息:")
        print(f"  名称: {info['video_name']}")
        print(f"  总帧数: {info['total_frames']}")
        print(f"  帧率: {info['fps']:.2f} FPS")
        print(f"  分辨率: {info['width']}x{info['height']}")
        print(f"  时长: {info['duration']:.2f} 秒")
        print(f"\n请编辑此文件，添加实际的动作标注")
        
        return output_path
    
    def batch_create_templates(self):
        """批量创建标注模板"""
        if not os.path.exists(self.raw_videos_dir):
            print(f"✗ 视频目录不存在: {self.raw_videos_dir}")
            return
        
        # 获取所有视频文件
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = [
            f for f in os.listdir(self.raw_videos_dir)
            if Path(f).suffix.lower() in video_extensions
        ]
        
        if not video_files:
            print(f"✗ 未找到视频文件")
            return
        
        print(f"\n找到 {len(video_files)} 个视频文件")
        print("=" * 60)
        
        for video_file in video_files:
            video_path = os.path.join(self.raw_videos_dir, video_file)
            try:
                self.create_annotation_template(video_path)
                print()
            except Exception as e:
                print(f"✗ 处理失败 {video_file}: {e}\n")
        
        print("=" * 60)
        print(f"✓ 批量创建完成！")
    
    def convert_labelme_to_frame(self, labelme_json, fps=30):
        """
        将LabelMe时间戳标注转换为帧号标注
        
        参数:
            labelme_json: LabelMe标注文件路径
            fps: 视频帧率
        
        返回:
            转换后的标注数据
        """
        with open(labelme_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        converted_annotations = []
        
        for i, annotation in enumerate(data.get('annotations', []), 1):
            start_time = annotation.get('start_time', 0)
            end_time = annotation.get('end_time', 0)
            
            converted_annotations.append({
                "action_id": i,
                "class": annotation.get('class', ''),
                "start_frame": self.time_to_frame(start_time, fps),
                "end_frame": self.time_to_frame(end_time, fps),
                "description": annotation.get('description', '')
            })
        
        output = {
            "video_name": data.get('video_name', ''),
            "total_frames": self.time_to_frame(data.get('duration', 0), fps),
            "frame_rate": fps,
            "annotations": converted_annotations
        }
        
        # 保存转换后的文件
        output_path = labelme_json.replace('.json', '_converted.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 转换完成: {output_path}")
        print(f"  转换了 {len(converted_annotations)} 个标注")
        
        return output
    
    def show_video_frame(self, video_path, frame_number):
        """
        显示视频的指定帧
        
        参数:
            video_path: 视频路径
            frame_number: 帧号
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"✗ 无法打开视频: {video_path}")
            return
        
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            # 显示帧
            cv2.imshow(f'Frame {frame_number}', frame)
            print(f"显示帧 {frame_number}，按任意键关闭")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"✗ 无法读取帧 {frame_number}")
        
        cap.release()


def main():
    """主函数：交互式标注辅助工具"""
    helper = AnnotationHelper()
    
    print("=" * 60)
    print("羽毛球视频标注辅助工具")
    print("=" * 60)
    
    while True:
        print("\n请选择功能：")
        print("  [1] 获取视频信息")
        print("  [2] 创建标注模板")
        print("  [3] 批量创建标注模板")
        print("  [4] 时间戳转帧号")
        print("  [5] 帧号转时间戳")
        print("  [6] 转换LabelMe标注")
        print("  [7] 查看视频帧")
        print("  [0] 退出")
        print("-" * 60)
        
        choice = input("请输入选项 [0-7]: ").strip()
        
        if choice == '1':
            # 获取视频信息
            video_path = input("请输入视频路径: ").strip()
            if os.path.exists(video_path):
                try:
                    info = helper.get_video_info(video_path)
                    print("\n视频信息:")
                    print(f"  名称: {info['video_name']}")
                    print(f"  总帧数: {info['total_frames']}")
                    print(f"  帧率: {info['fps']:.2f} FPS")
                    print(f"  分辨率: {info['width']}x{info['height']}")
                    print(f"  时长: {info['duration']:.2f} 秒")
                except Exception as e:
                    print(f"✗ 错误: {e}")
            else:
                print(f"✗ 文件不存在: {video_path}")
        
        elif choice == '2':
            # 创建标注模板
            video_path = input("请输入视频路径: ").strip()
            if os.path.exists(video_path):
                try:
                    helper.create_annotation_template(video_path)
                except Exception as e:
                    print(f"✗ 错误: {e}")
            else:
                print(f"✗ 文件不存在: {video_path}")
        
        elif choice == '3':
            # 批量创建标注模板
            helper.batch_create_templates()
        
        elif choice == '4':
            # 时间戳转帧号
            try:
                time_seconds = float(input("请输入时间（秒）: ").strip())
                fps = float(input("请输入帧率（默认30）: ").strip() or "30")
                frame_number = helper.time_to_frame(time_seconds, fps)
                print(f"\n时间 {time_seconds:.2f}秒 = 帧号 {frame_number}")
            except ValueError:
                print("✗ 输入格式错误")
        
        elif choice == '5':
            # 帧号转时间戳
            try:
                frame_number = int(input("请输入帧号: ").strip())
                fps = float(input("请输入帧率（默认30）: ").strip() or "30")
                time_seconds = helper.frame_to_time(frame_number, fps)
                print(f"\n帧号 {frame_number} = 时间 {time_seconds:.2f}秒")
            except ValueError:
                print("✗ 输入格式错误")
        
        elif choice == '6':
            # 转换LabelMe标注
            labelme_json = input("请输入LabelMe标注文件路径: ").strip()
            if os.path.exists(labelme_json):
                try:
                    fps = float(input("请输入帧率（默认30）: ").strip() or "30")
                    helper.convert_labelme_to_frame(labelme_json, fps)
                except Exception as e:
                    print(f"✗ 错误: {e}")
            else:
                print(f"✗ 文件不存在: {labelme_json}")
        
        elif choice == '7':
            # 查看视频帧
            video_path = input("请输入视频路径: ").strip()
            if os.path.exists(video_path):
                try:
                    frame_number = int(input("请输入帧号: ").strip())
                    helper.show_video_frame(video_path, frame_number)
                except ValueError:
                    print("✗ 帧号格式错误")
                except Exception as e:
                    print(f"✗ 错误: {e}")
            else:
                print(f"✗ 文件不存在: {video_path}")
        
        elif choice == '0':
            print("\n再见！")
            break
        
        else:
            print("✗ 无效选项")


if __name__ == "__main__":
    main()
