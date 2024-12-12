import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(test_json, test_id):
    # 读取异常分数
    with open(test_json, 'r') as f:
        scores = json.load(f)
    # 读取ground truth标签
    with open('/workspace/data/carla_local/ground_truth_demo/gt_label.json', 'r') as f:
        gt_data = json.load(f)
        labels = gt_data[test_id]
    return scores, labels

def create_curve_image(scores, labels, frame_idx, img_size, test_id):
    # 创建图像并设置大小
    plt.figure(figsize=(img_size[0]/100, img_size[1]/100))
    
    # 创建完整的x轴范围
    x_range = np.arange(len(scores))
    
    # 只绘制到当前帧的数据
    current_scores = scores[:frame_idx+1]
    current_x = x_range[:frame_idx+1]
    
    # 绘制到当前帧的异常分数曲线
    plt.plot(current_x, current_scores, 'b-', label='Anomaly Score', linewidth=1)
    
    # 绘制剩余帧的位置（可选，用虚线表示）
    if frame_idx + 1 < len(scores):
        plt.plot(x_range[frame_idx+1:], [None] * (len(scores)-frame_idx-1), 'b--', alpha=0.2)
    
    # 标记异常区域（整个范围）
    gt_array = np.array(labels)
    anomaly_regions = np.where(gt_array == 1)[0]
    if len(anomaly_regions) > 0:
        start = anomaly_regions[0]
        end = anomaly_regions[-1]
        plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Intervals')
    
    # 设置图表属性
    plt.title(f'Test video {test_id}',fontsize=6)
    plt.xlabel('Frames Sequence',fontsize=6)
    plt.ylabel('Anomaly Score',fontsize=6)
    plt.xlim(0, 100)
    plt.ylim(-5, 35)
    plt.grid(True, alpha=0.3)
    
    plt.tick_params(axis='both', which='major', labelsize=6)

    plt.legend(prop={'size': 6})
    plt.tight_layout()
    

    # 将matplotlib图像转换为numpy数组
    fig = plt.gcf()
    fig.canvas.draw()
    
    # 获取图像数据
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # 调整图像大小
    img_array = cv2.resize(img_array, img_size)
    
    # 转换颜色空间从RGB到BGR（OpenCV使用BGR）
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 清理matplotlib图像
    plt.close()
    
    return img_array

def overlay_images(large_folder, small_folder, output_folder, test_json, test_id):
    images_for_video=[]
    # 加载异常分数和标签数据
    scores, labels = load_data(test_json, test_id)
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图片文件
    large_files = sorted([os.path.join(large_folder, f) for f in os.listdir(large_folder) if f.endswith(('.jpg', '.png'))])
    small_files = sorted([os.path.join(small_folder, f) for f in os.listdir(small_folder) if f.endswith(('.jpg', '.png'))])
    
    for i in range(len(large_files)):
        filename = os.path.basename(large_files[i])
        
        # 读取大图和小图
        large_img = cv2.imread(large_files[i])
        small_img = cv2.imread(small_files[i])
        
        if large_img is None or small_img is None:
            print(f"无法读取图片: {filename}")
            continue
        
        # 计算小图的新尺寸（原尺寸的1/3）
        small_h, small_w = small_img.shape[:2]
        new_w = small_w // 3
        new_h = small_h // 3
        small_img_resized = cv2.resize(small_img, (new_w, new_h))
        
        # 生成当前帧的曲线图
        curve_img = create_curve_image(scores, labels, i, (new_w, new_h), test_id)
        
        # 放置小图到右上角
        x_offset_right = large_img.shape[1] - new_w - 10
        y_offset = 10
        
        # 放置曲线图到左上角
        x_offset_left = 10
        
        # 在右上角区域涂黑并放置小图
        large_img[y_offset:y_offset+new_h, x_offset_right:x_offset_right+new_w] = 0
        large_img[y_offset:y_offset+new_h, x_offset_right:x_offset_right+new_w] = small_img_resized
        
        # 在左上角区域涂黑并放置曲线图
        large_img[y_offset:y_offset+new_h, x_offset_left:x_offset_left+new_w] = 0
        large_img[y_offset:y_offset+new_h, x_offset_left:x_offset_left+new_w] = curve_img
        
        images_for_video.append(large_img)

        # 保存结果
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, large_img)
        print(f"处理完成: {filename}")
    return images_for_video

def create_video(images, output_video_path, fps=30):
    """
    将图片列表转换为视频
    
    Args:
        images: 包含所有帧图像的列表
        output_video_path: 输出视频文件路径
        fps: 视频帧率，默认30
    """
    if not images:
        print("没有图片可以处理")
        return
    
    # 获取第一帧的尺寸
    height, width = images[0].shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 写入每一帧
    for image in images:
        video_writer.write(image)
    
    # 释放资源
    video_writer.release()
    print(f"视频已保存到: {output_video_path}")



if __name__ == "__main__":
    scenarios_id = 0
    for scenarios_id in range(13):
        test_json= f'/workspace/eval/carla_local_ML_MemAE_SC_f4l16/anomaly_curves_best/anomaly_scores_{scenarios_id}.json'
        test_id = str(scenarios_id+1).zfill(4)
        root_dir = f'/workspace/data/carla_local/testing/frames/{test_id}/'
        large_folder = f"{root_dir}RGB_IMG"
        small_folder = f"{root_dir}ground-truth"
        output_folder = f"/workspace/results/frames/figures/scenarios_{scenarios_id+1}"

        processed_images = overlay_images(large_folder, small_folder, output_folder,test_json, test_id)

        output_video_path = f"/workspace/results/frames/video_output_{test_id}.mp4"
        create_video(processed_images, output_video_path, fps=30)