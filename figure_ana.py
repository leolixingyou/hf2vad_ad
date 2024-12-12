import cv2
import os
import json
import numpy as np
import pandas as pd
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

def save_scores_to_csv():
    # 初始化存储数据的字典
    scores_dict = {}
    stats_dict = {
        'Scenario': [],
        'Max': [],
        'Min': [],
        'Mean': [],
        'Std': []
    }

    # 首先找到最大长度
    max_len = 0
    for scenarios_id in range(13):
        test_json = f'/workspace/eval/carla_local_ML_MemAE_SC_f4l16/anomaly_curves_best/anomaly_scores_{scenarios_id}.json'
        scores, _ = load_data(test_json, str(scenarios_id+1).zfill(4))
        max_len = max(max_len, len(scores))

    # 收集所有场景的数据
    for scenarios_id in range(13):
        # 构建文件路径
        test_json = f'/workspace/eval/carla_local_ML_MemAE_SC_f4l16/anomaly_curves_best/anomaly_scores_{scenarios_id}.json'
        test_id = str(scenarios_id+1).zfill(4)
        
        # 加载数据
        scores, _ = load_data(test_json, test_id)
        
        # 创建填充后的scores列表
        padded_scores = scores + [np.nan] * (max_len - len(scores))
        
        # 保存scores数据
        col_name = f'scenario_{scenarios_id+1}'
        scores_dict[col_name] = padded_scores
        
        # 计算统计信息（使用原始scores，不包括填充的值）
        stats_dict['Scenario'].append(col_name)
        stats_dict['Max'].append(np.max(scores))
        stats_dict['Min'].append(np.min(scores))
        stats_dict['Mean'].append(np.mean(scores))
        stats_dict['Std'].append(np.std(scores))

    # 创建scores的DataFrame
    scores_df = pd.DataFrame(scores_dict)
    
    # 创建统计信息的DataFrame
    stats_df = pd.DataFrame(stats_dict)

    # 保存为CSV文件
    scores_df.to_csv('scenario_scores.csv', index=False)
    stats_df.to_csv('scenario_statistics.csv', index=False)
    
    # 打印每个场景的数据长度信息
    print("\nData lengths for each scenario:")
    for col in scores_df.columns:
        valid_count = scores_df[col].count()  # 计算非NaN值的数量
        print(f"{col}: {valid_count} valid values out of {max_len} total")

if __name__ == "__main__":
    save_scores_to_csv()
    print("\nFiles saved successfully:")
    print("1. scenario_scores.csv - Contains all scores for each scenario")
    print("2. scenario_statistics.csv - Contains statistical information for each scenario")