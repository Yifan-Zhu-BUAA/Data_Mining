"""可视化和结果分析模块

该模块负责可视化训练过程、异常检测结果和评估指标。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib支持中文显示
# 尝试使用多种可能在系统中可用的中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'WenQuanYi Micro Hei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.metrics import roc_curve, precision_recall_curve
import torch
from torchvision.utils import make_grid

def plot_training_history(history, save_path=None):
    """绘制训练历史
    
    Args:
        history (dict): 训练历史记录
        save_path (str, optional): 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制损失曲线
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    
    # 设置图表属性
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_reconstruction_error_distribution(errors, labels, threshold=None, save_path=None):
    """绘制重建误差分布
    
    Args:
        errors (np.ndarray): 重建误差数组
        labels (np.ndarray): 真实标签数组 (0=正常, 1=异常)
        threshold (float, optional): 异常检测阈值
        save_path (str, optional): 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制正常样本和异常样本的误差分布
    sns.histplot(errors[labels == 0], bins=50, alpha=0.5, label='正常样本', kde=True)
    sns.histplot(errors[labels == 1], bins=50, alpha=0.5, label='异常样本', kde=True)
    
    # 绘制阈值线
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'阈值: {threshold:.4f}')
    
    # 设置图表属性
    plt.xlabel('重建误差')
    plt.ylabel('频率')
    plt.title('重建误差分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_roc_curve(errors, labels, save_path=None):
    """绘制ROC曲线
    
    Args:
        errors (np.ndarray): 重建误差数组
        labels (np.ndarray): 真实标签数组 (0=正常, 1=异常)
        save_path (str, optional): 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(labels, errors)
    
    # 计算AUC
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    
    # 绘制对角线（随机分类器）
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title('ROC曲线')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_precision_recall_curve(errors, labels, threshold=None, save_path=None):
    """绘制精确率-召回率曲线
    
    Args:
        errors (np.ndarray): 重建误差数组
        labels (np.ndarray): 真实标签数组 (0=正常, 1=异常)
        threshold (float, optional): 异常检测阈值
        save_path (str, optional): 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 计算精确率-召回率曲线
    precision, recall, thresholds = precision_recall_curve(labels, errors)
    
    # 绘制曲线
    plt.plot(recall, precision, color='blue', lw=2, label='精确率-召回率曲线')
    
    # 如果提供了阈值，在曲线上标记对应点
    if threshold is not None:
        # 找到最接近的阈值索引
        idx = np.argmin(np.abs(thresholds - threshold))
        if idx < len(precision) - 1 and idx < len(recall) - 1:
            plt.scatter(recall[idx], precision[idx], color='red', s=100, 
                      label=f'阈值: {threshold:.4f}\n精确率: {precision[idx]:.3f}, 召回率: {recall[idx]:.3f}')
    
    # 设置图表属性
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_confusion_matrix(conf_matrix, save_path=None):
    """绘制混淆矩阵
    
    Args:
        conf_matrix (np.ndarray): 混淆矩阵
        save_path (str, optional): 保存路径
    """
    plt.figure(figsize=(8, 6))
    
    # 绘制热力图
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
               xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
    
    # 设置图表属性
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def visualize_reconstructions(model, dataloader, device, num_samples=8, save_path=None, is_vae=False):
    """可视化重建结果
    
    Args:
        model (nn.Module): 自编码器模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 运行设备
        num_samples (int): 要可视化的样本数量
        save_path (str, optional): 保存路径
        is_vae (bool): 是否为变分自编码器
    """
    model.eval()
    
    # 获取一批数据
    data_iter = iter(dataloader)
    images, labels, _ = next(data_iter)
    
    # 选择前num_samples个样本
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # 重建图像
    with torch.no_grad():
        if is_vae:
            reconstructed, _, _ = model(images)
        else:
            reconstructed = model(images)
    
    # 将图像移至CPU并转换为numpy数组
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    
    # 创建网格图像
    original_grid = make_grid(images, nrow=4, normalize=True)
    reconstructed_grid = make_grid(reconstructed, nrow=4, normalize=True)
    
    # 绘制图像
    plt.figure(figsize=(15, 8))
    
    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_grid.permute(1, 2, 0))
    plt.title('原始图像')
    plt.axis('off')
    
    # 重建图像
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_grid.permute(1, 2, 0))
    plt.title('重建图像')
    plt.axis('off')
    
    # 添加标签信息
    plt.figtext(0.5, 0.01, f'标签: {labels.tolist()}', ha='center')
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def visualize_anomaly_heatmaps(model, dataloader, device, threshold, num_samples=4, save_path=None, is_vae=False):
    """可视化异常热力图
    
    Args:
        model (nn.Module): 自编码器模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 运行设备
        threshold (float): 异常检测阈值
        num_samples (int): 要可视化的样本数量
        save_path (str, optional): 保存路径
        is_vae (bool): 是否为变分自编码器
    """
    model.eval()
    
    # 获取一批数据
    data_iter = iter(dataloader)
    images, labels, _ = next(data_iter)
    
    # 选择前num_samples个样本
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    # 重建图像
    with torch.no_grad():
        if is_vae:
            reconstructed, _, _ = model(images)
        else:
            reconstructed = model(images)
    
    # 计算像素级误差
    pixel_errors = torch.abs(reconstructed - images)
    
    # 将图像移至CPU并转换为numpy数组
    images = images.cpu()
    reconstructed = reconstructed.cpu()
    pixel_errors = pixel_errors.cpu()
    
    # 绘制图像
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        # 原始图像
        axes[i, 0].imshow(images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # 反归一化
        axes[i, 0].set_title(f'原始图像\n真实标签: {"异常" if labels[i] == 1 else "正常"}')
        axes[i, 0].axis('off')
        
        # 重建图像
        axes[i, 1].imshow(reconstructed[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # 反归一化
        axes[i, 1].set_title('重建图像')
        axes[i, 1].axis('off')
        
        # 误差热力图
        error_map = pixel_errors[i].mean(dim=0).numpy()
        im = axes[i, 2].imshow(error_map, cmap='jet')
        axes[i, 2].set_title('误差热力图')
        axes[i, 2].axis('off')
        
        # 添加颜色条
        fig.colorbar(im, ax=axes[i, 2])
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def save_results_to_json(metrics, save_path):
    """将评估结果保存为JSON文件
    
    Args:
        metrics (dict): 评估指标
        save_path (str): 保存路径
    """
    import json
    import torch
    import numpy as np
    
    # 递归转换非JSON可序列化对象为可序列化类型
    def make_serializable(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    # 转换所有非JSON可序列化对象
    serializable_metrics = make_serializable(metrics)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存为JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_metrics, f, indent=4, ensure_ascii=False)