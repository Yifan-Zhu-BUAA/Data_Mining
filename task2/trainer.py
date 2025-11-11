"""模型训练和评估模块

该模块负责训练模型、评估性能以及进行异常检测。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

def train_autoencoder(model, train_loader, val_loader, device, num_epochs=100, lr=1e-4, weight_decay=1e-5, 
                     checkpoint_dir=None, is_vae=False):
    """训练自编码器模型
    
    Args:
        model (nn.Module): 自编码器模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        device (torch.device): 运行设备
        num_epochs (int): 训练轮数
        lr (float): 学习率
        weight_decay (float): 权重衰减
        checkpoint_dir (str, optional): 检查点保存目录
        is_vae (bool): 是否为变分自编码器
        
    Returns:
        dict: 训练历史记录
    """
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 定义损失函数
    if is_vae:
        from autoencoder import vae_loss
    else:
        criterion = nn.MSELoss()
    
    # 移至设备
    model = model.to(device)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # 最佳验证损失
    best_val_loss = float('inf')
    
    # 早停计数器
    patience = 10
    counter = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss_epoch = 0.0
        
        # 训练迭代
        for images, _, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            if is_vae:
                reconstructed, mu, logvar = model(images)
                loss = vae_loss(reconstructed, images, mu, logvar)
            else:
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 累加损失
            train_loss_epoch += loss.item() * images.size(0)
        
        # 计算平均训练损失
        train_loss_epoch /= len(train_loader.dataset)
        history['train_loss'].append(train_loss_epoch)
        
        # 验证模式
        model.eval()
        val_loss_epoch = 0.0
        
        with torch.no_grad():
            for images, _, _ in val_loader:
                images = images.to(device)
                
                # 前向传播
                if is_vae:
                    reconstructed, mu, logvar = model(images)
                    loss = vae_loss(reconstructed, images, mu, logvar)
                else:
                    reconstructed = model(images)
                    loss = criterion(reconstructed, images)
                
                # 累加损失
                val_loss_epoch += loss.item() * images.size(0)
        
        # 计算平均验证损失
        val_loss_epoch /= len(val_loader.dataset)
        history['val_loss'].append(val_loss_epoch)
        
        # 打印进度
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_epoch:.6f}, Val Loss: {val_loss_epoch:.6f}')
        
        # 保存最佳模型
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            counter = 0
            
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 加载最佳模型
    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return history

def compute_reconstruction_error(model, dataloader, device, is_vae=False):
    """计算重建误差
    
    Args:
        model (nn.Module): 自编码器模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 运行设备
        is_vae (bool): 是否为变分自编码器
        
    Returns:
        tuple: (errors, labels, paths) - 重建误差、标签和图像路径
    """
    model.eval()
    errors = []
    labels = []
    paths = []
    
    with torch.no_grad():
        for images, batch_labels, batch_paths in dataloader:
            images = images.to(device)
            
            # 前向传播
            if is_vae:
                reconstructed, _, _ = model(images)
            else:
                reconstructed = model(images)
            
            # 计算每张图像的重建误差
            batch_errors = torch.mean((reconstructed - images) ** 2, dim=[1, 2, 3]).cpu().numpy()
            
            # 保存结果
            errors.extend(batch_errors)
            labels.extend(batch_labels.numpy())
            paths.extend(batch_paths)
    
    return np.array(errors), np.array(labels), paths

def detect_anomalies(errors, threshold):
    """根据阈值检测异常
    
    Args:
        errors (np.ndarray): 重建误差数组
        threshold (float): 异常检测阈值
        
    Returns:
        np.ndarray: 预测标签数组 (0=正常, 1=异常)
    """
    return (errors > threshold).astype(int)

def find_optimal_threshold(errors, labels):
    """寻找最优阈值
    
    Args:
        errors (np.ndarray): 重建误差数组
        labels (np.ndarray): 真实标签数组 (0=正常, 1=异常)
        
    Returns:
        float: 最优阈值
    """
    # 计算精确率-召回率曲线
    precision, recall, thresholds = precision_recall_curve(labels, errors)
    
    # 计算F1分数
    f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
    
    # 找到F1分数最高的阈值
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold

def evaluate_model(errors, labels, threshold=None):
    """评估模型性能
    
    Args:
        errors (np.ndarray): 重建误差数组
        labels (np.ndarray): 真实标签数组 (0=正常, 1=异常)
        threshold (float, optional): 异常检测阈值，若为None则自动寻找最优阈值
        
    Returns:
        dict: 评估指标
    """
    # 如果没有提供阈值，自动寻找最优阈值
    if threshold is None:
        threshold = find_optimal_threshold(errors, labels)
    
    # 检测异常
    predictions = detect_anomalies(errors, threshold)
    
    # 计算评估指标
    auc_roc = roc_auc_score(labels, errors)
    f1 = f1_score(labels, predictions)
    conf_matrix = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True)
    
    # 计算精确率、召回率和准确率
    precision = report['1']['precision']
    recall = report['1']['recall']
    accuracy = report['accuracy']
    
    metrics = {
        'threshold': threshold,
        'auc_roc': auc_roc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    }
    
    return metrics