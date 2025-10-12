#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制动作数据的7个子图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_action_data(csv_file, gt_file):
    """
    读取CSV文件并绘制动作子图，同时显示预测和GT数据，自动检测维度
    """
    # 读取预测数据（无列名）
    data_pred = pd.read_csv(csv_file, header=None)
    
    # 读取GT数据（无列名）
    data_gt = pd.read_csv(gt_file, header=None)
    
    # 自动检测维度
    n_dims = min(data_pred.shape[1], data_gt.shape[1])
    
    # 为列命名
    column_names = [f'Action_{i+1}' for i in range(n_dims)]
    data_pred.columns = column_names
    data_gt.columns = column_names
    
    # 创建时间步索引（使用较短的长度）
    min_len = min(len(data_pred), len(data_gt))
    time_steps = np.arange(min_len)
    
    # 截取到相同长度
    data_pred = data_pred.iloc[:min_len]
    data_gt = data_gt.iloc[:min_len]
    
    # 设置图像参数
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (16, 12)
    # 设置中文字体，如果没有中文字体则使用英文
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 根据维度数量动态创建子图布局
    if n_dims <= 6:
        rows, cols = 2, 3
        figsize = (16, 10)
    elif n_dims <= 9:
        rows, cols = 3, 3
        figsize = (16, 12)
    else:
        rows, cols = 4, 3
        figsize = (16, 16)
    
    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f'Action Data Visualization - {n_dims} Action Dimensions', fontsize=16, fontweight='bold')
    
    # 绘制动作子图
    for i in range(n_dims):
        row = i // cols
        col = i % cols
        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        
        # 绘制预测数据和GT数据
        ax.plot(time_steps, data_pred.iloc[:, i], linewidth=2, color=f'C{i}', 
                label='Prediction', alpha=0.8)
        ax.plot(time_steps, data_gt.iloc[:, i], linewidth=2, color='red', 
                linestyle='--', label='Ground Truth', alpha=0.8)
        
        ax.set_title(f'{column_names[i]}', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # 设置y轴范围（考虑两条线的范围）
        all_values = np.concatenate([data_pred.iloc[:, i], data_gt.iloc[:, i]])
        y_min, y_max = all_values.min(), all_values.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    # 隐藏多余的子图
    total_subplots = rows * cols
    for i in range(n_dims, total_subplots):
        row = i // cols
        col = i % cols
        if rows > 1:  # 确保是多行布局
            axes[row, col].set_visible(False)
        else:  # 单行布局
            axes[col].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = '/ML-vePFS/tangyinzhou/yinuo/ManiSkill_evaluation/action_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {output_path}")
    
    # 显示图像
    plt.show()
    
    # 打印数据统计信息
    print("\nData Statistics:")
    print("=" * 80)
    print(f"{'Dimension':<12} {'Pred Mean':<12} {'GT Mean':<12} {'Pred Std':<12} {'GT Std':<12} {'MAE':<12}")
    print("-" * 80)
    
    for i, col in enumerate(column_names):
        pred_stats = data_pred.iloc[:, i].describe()
        gt_stats = data_gt.iloc[:, i].describe()
        mae = np.mean(np.abs(data_pred.iloc[:, i] - data_gt.iloc[:, i]))
        
        print(f"{col:<12} {pred_stats['mean']:<12.6f} {gt_stats['mean']:<12.6f} "
              f"{pred_stats['std']:<12.6f} {gt_stats['std']:<12.6f} {mae:<12.6f}")
    
    # 总体MAE
    overall_mae = np.mean(np.abs(data_pred - data_gt))
    print(f"\nOverall MAE: {overall_mae:.6f}")

if __name__ == "__main__":
    csv_file = "/ML-vePFS/tangyinzhou/yinuo/ManiSkill_evaluation/debug/action.csv"
    gt_file = "/ML-vePFS/tangyinzhou/yinuo/ManiSkill_evaluation/debug/action_GT.csv"
    plot_action_data(csv_file, gt_file)
