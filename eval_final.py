import json
import numpy as np
import matplotlib.pyplot as plt

# # 读取数据
# def load_data():
#     # 读取异常分数
#     with open('/workspace/eval/carla_local_ML_MemAE_SC_f4l16/anomaly_curves_best/anomaly_scores_8.json', 'r') as f:
#         scores = json.load(f)
    
#     # 读取ground truth标签
#     with open('/workspace/data/carla_local/ground_truth_demo/gt_label.json', 'r') as f:
#         gt_data = json.load(f)
#         labels = gt_data['0009']  # 获取0009的数据
    
#     return scores, labels

# def create_visualization():
#     # 设置图形大小
#     plt.figure(figsize=(10, 6))
    
#     # 绘制异常分数曲线
#     plt.plot(scores, 'b-', label='Anomaly Score', linewidth=1)
    
#     # 找到并标记异常区域
#     gt_array = np.array(labels)
#     anomaly_regions = np.where(gt_array == 1)[0]
#     if len(anomaly_regions) > 0:
#         start = anomaly_regions[0]
#         end = anomaly_regions[-1]
#         plt.axvspan(start, end, color='red', alpha=0.3, label='Anomaly Intervals')
    
#     # 设置图表属性
#     plt.title('Test video 9')
#     plt.xlabel('Frames Sequence')
#     plt.ylabel('Anomaly Score')
    
#     # 设置坐标轴范围
#     plt.xlim(0, 100)
#     plt.ylim(-5, 35)
    
#     # 添加图例
#     plt.legend()
    
#     # 调整布局
#     plt.tight_layout()
    
#     # 显示图表
#     plt.savefig('/workspace/sample.jpg',dpi=300)

# if __name__ == "__main__":
#     # 加载数据
#     scores, labels = load_data()
    
#     # 创建可视化
#     create_visualization()


import cv2
import os
from pathlib import Path

def overlay_images(large_folder, small_folder, output_folder):
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
            print(f"无法读取图片: {filename} ")
            continue
        
        # 计算小图的新尺寸（原尺寸的1/3）
        small_h, small_w = small_img.shape[:2]
        new_w = small_w // 3
        new_h = small_h // 3
        small_img_resized = cv2.resize(small_img, (new_w, new_h))
        
        # 计算放置位置（右上角）
        x_offset = large_img.shape[1] - new_w - 10  # 距离右边缘10像素
        y_offset = 10  # 距离上边缘10像素
        
        large_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = 0
        
        # 在大图上创建ROI
        roi = large_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
        
        # 创建小图的mask
        small_img_gray = cv2.cvtColor(small_img_resized, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(small_img_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        # 将ROI中的图像置黑
        large_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        
        # 将小图放在黑色区域上
        small_fg = cv2.bitwise_and(small_img_resized, small_img_resized, mask=mask)
        
        # 组合大图和小图
        dst = cv2.add(large_bg, small_fg)
        large_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = dst
        
        # 保存结果
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, large_img)
        
        print(f"处理完成: {filename}")

# 使用示例
if __name__ == "__main__":
    root_dir = '/workspace/data/carla_local/testing/frames/0009/'
    large_folder = f"{root_dir}RGB_IMG"    # 原始大图文件夹路径
    small_folder = f"{root_dir}ground-truth"     # 要叠加的小图文件夹路径
    output_folder = "/workspace/results/frames/"     # 输出文件夹路径
    
    overlay_images(large_folder, small_folder, output_folder)