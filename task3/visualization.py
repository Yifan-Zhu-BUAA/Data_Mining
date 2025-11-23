"""
可视化模块 - 绘制训练曲线、预测结果等
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class Visualizer:
    """可视化工具类"""
    
    def __init__(self):
        os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    
    def plot_training_history(self, train_losses, val_losses, save_path=None):
        """绘制训练历史曲线"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        
        plt.title('训练历史曲线', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        plt.close()
    
    def plot_predictions(self, y_true, y_pred, title="预测结果对比", save_path=None):
        """绘制预测结果对比"""
        plt.figure(figsize=(12, 6))
        
        # 选择前500个点进行可视化（如果数据太多）
        n_samples = min(500, len(y_true))
        indices = np.arange(n_samples)
        
        plt.plot(indices, y_true[:n_samples], 'b-', label='真实值', linewidth=1.5, alpha=0.7)
        plt.plot(indices, y_pred[:n_samples], 'r--', label='预测值', linewidth=1.5, alpha=0.7)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('样本索引', fontsize=12)
        plt.ylabel('室外温度 (OT)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果图已保存到: {save_path}")
        plt.close()
    
    def plot_prediction_scatter(self, y_true, y_pred, save_path=None):
        """绘制预测值 vs 真实值的散点图"""
        plt.figure(figsize=(8, 8))
        
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # 绘制完美预测线（y=x）
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
        
        plt.title('预测值 vs 真实值散点图', fontsize=14, fontweight='bold')
        plt.xlabel('真实值 (OT)', fontsize=12)
        plt.ylabel('预测值 (OT)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"散点图已保存到: {save_path}")
        plt.close()
    
    def plot_error_distribution(self, y_true, y_pred, save_path=None):
        """绘制误差分布图"""
        errors = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 误差直方图
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title('误差分布直方图', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('误差 (真实值 - 预测值)', fontsize=10)
        axes[0].set_ylabel('频数', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        # 误差时序图
        axes[1].plot(errors[:1000], linewidth=1, alpha=0.7)
        axes[1].set_title('误差时序图', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('样本索引', fontsize=10)
        axes[1].set_ylabel('误差', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"误差分布图已保存到: {save_path}")
        plt.close()
    
    def plot_residuals(self, y_true, y_pred, save_path=None):
        """绘制残差图"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        plt.title('残差图', fontsize=14, fontweight='bold')
        plt.xlabel('预测值', fontsize=12)
        plt.ylabel('残差 (真实值 - 预测值)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"残差图已保存到: {save_path}")
        plt.close()

# 单例模式
visualizer = Visualizer()
