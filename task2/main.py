"""图像异常检测主程序

该程序整合了数据加载、模型训练、评估和可视化功能，实现完整的图像异常检测流程。
"""

import os
import argparse
import json
import torch
import numpy as np
import random
from datetime import datetime

# 导入自定义模块
from data_loader import get_dataloaders, get_normal_only_dataloader
# 不再需要特征提取器
from autoencoder import Autoencoder, VariationalAutoencoder
from trainer import train_autoencoder, detect_anomalies, evaluate_model, compute_reconstruction_error
from visualization import (
    plot_training_history,
    plot_reconstruction_error_distribution,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    visualize_reconstructions,
    visualize_anomaly_heatmaps,
    save_results_to_json
)

def set_seed(seed=42):
    """设置随机种子以确保可重复性
    
    Args:
        seed (int): 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """解析命令行参数
    
    Returns:
        argparse.Namespace: 命令行参数
    """
    parser = argparse.ArgumentParser(description='图像异常检测')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, 
                      default='./Image_Anomaly_Detection',
                      help='数据集路径')
    parser.add_argument('--category', type=str, default='hazelnut', 
                      choices=['hazelnut', 'zipper'],
                      help='选择的类别')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='数据加载的工作线程数')
    
    # 模型相关参数
    parser.add_argument('--model_type', type=str, default='autoencoder',
                      choices=['autoencoder', 'vae'],
                      help='模型类型')
    parser.add_argument('--latent_dim', type=int, default=128,
                      help='潜在空间维度')
    parser.add_argument('--model_path', type=str, default=None,
                      help='预训练模型路径，如果提供则直接加载模型而不进行训练')
    # 不再需要特征提取器相关参数
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='权重衰减')
    parser.add_argument('--patience', type=int, default=10,
                      help='早停耐心值')
    
    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='输出目录')
    parser.add_argument('--save_model', action='store_true',
                      help='是否保存模型')
    parser.add_argument('--visualize', action='store_true',
                      help='是否可视化结果')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    
    return parser.parse_args()

def main():
    """主函数
    
    执行完整的图像异常检测流程：数据加载、模型训练、评估和可视化
    """
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.category}_{args.model_type}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 1. 加载数据
    print(f'\n加载{args.category}类别数据...')
    # 对于自编码器异常检测，我们只使用正常样本进行训练
    train_loader = get_normal_only_dataloader(
        root_dir=args.data_dir,
        category=args.category,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 获取测试集加载器
    _, test_loader = get_dataloaders(
        root_dir=args.data_dir,
        category=args.category,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 对于验证集，我们可以使用部分测试集数据
    val_loader = test_loader
    
    # 2. 创建或加载模型
    print(f'\n创建{args.model_type}模型...')
    
    # 获取图像通道数和尺寸
    image_channels = 3
    image_size = 224
    
    # 创建自编码器或变分自编码器
    if args.model_type == 'autoencoder':
        model = Autoencoder(
            input_channels=image_channels,
            latent_dim=args.latent_dim
        )
    else:  # vae
        model = VariationalAutoencoder(
            input_channels=image_channels,
            latent_dim=args.latent_dim
        )
    
    model = model.to(device)
    
    # 是否加载预训练模型
    if args.model_path:
        # 处理相对路径，转换为绝对路径
        model_path = os.path.abspath(args.model_path)
        print(f'\n加载预训练模型: {model_path}')
        if not os.path.exists(model_path):
            print(f"错误: 找不到模型文件 '{model_path}'")
            exit(1)
        # 加载保存的检查点字典
        checkpoint = torch.load(model_path, map_location=device)
        # 提取模型状态字典
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print('从检查点提取并加载模型状态字典')
        else:
            # 尝试直接加载（兼容旧格式）
            model.load_state_dict(checkpoint)
            print('直接加载模型状态字典')
        model.eval()
        print('预训练模型加载成功！')
        # 不需要训练历史
        history = None
    else:
        # 3. 训练模型
        print('\n开始训练模型...')
        history = train_autoencoder(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            checkpoint_dir=output_dir,
            is_vae=(args.model_type == 'vae')
        )
        
        # 绘制训练历史
        if args.visualize and history:
            history_path = os.path.join(output_dir, 'training_history.png')
            plot_training_history(history, history_path)
            print(f'训练历史已保存至: {history_path}')
        
        # 保存模型
        if args.save_model:
            model_path = os.path.join(output_dir, f'{args.model_type}_best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'最佳模型已保存至: {model_path}')
    
    # 4. 在验证集上确定最佳阈值
    print('\n计算验证集上的重建误差...')
    
    # 提取验证集特征并计算重建误差
    val_errors, val_labels, val_paths = compute_reconstruction_error(
        model=model,
        dataloader=val_loader,
        device=device,
        is_vae=(args.model_type == 'vae')
    )
    
    # 寻找最佳阈值
    from trainer import find_optimal_threshold
    optimal_threshold = find_optimal_threshold(val_errors, val_labels)
    print(f'最佳阈值: {optimal_threshold:.4f}')
    
    # 5. 在测试集上评估模型
    print('\n在测试集上评估模型...')
    
    # 提取测试集特征并计算重建误差
    test_errors, test_labels, test_paths = compute_reconstruction_error(
        model=model,
        dataloader=test_loader,
        device=device,
        is_vae=(args.model_type == 'vae')
    )
    
    # 评估模型性能
    metrics = evaluate_model(
        errors=test_errors,
        labels=test_labels,
        threshold=optimal_threshold
    )
    
    # 打印评估结果
    print('\n评估结果:')
    # 安全访问评估指标，处理可能缺少的键
    print(f'AUC-ROC: {metrics["auc_roc"]:.4f}' if "auc_roc" in metrics else 'AUC-ROC: N/A')
    print(f'AUC-PR: {metrics["auc_pr"]:.4f}' if "auc_pr" in metrics else 'AUC-PR: N/A')
    print(f'准确率: {metrics["accuracy"]:.4f}' if "accuracy" in metrics else '准确率: N/A')
    print(f'精确率: {metrics["precision"]:.4f}' if "precision" in metrics else '精确率: N/A')
    print(f'召回率: {metrics["recall"]:.4f}' if "recall" in metrics else '召回率: N/A')
    print(f'F1分数: {metrics["f1_score"]:.4f}' if "f1_score" in metrics else 'F1分数: N/A')
    print(f'混淆矩阵:\n{metrics["confusion_matrix"]}')
    
    # 保存评估结果
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    save_results_to_json(metrics, results_path)
    print(f'评估结果已保存至: {results_path}')
    
    # 6. 可视化结果
    if args.visualize:
        print('\n生成可视化结果...')
        
        # 重建误差分布
        error_dist_path = os.path.join(output_dir, 'reconstruction_error_distribution.png')
        plot_reconstruction_error_distribution(
            test_errors, test_labels, optimal_threshold, error_dist_path
        )
        
        # ROC曲线
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plot_roc_curve(test_errors, test_labels, roc_path)
        
        # 精确率-召回率曲线
        pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
        plot_precision_recall_curve(test_errors, test_labels, optimal_threshold, pr_path)
        
        # 混淆矩阵
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(metrics['confusion_matrix'], cm_path)
        
        # 重建结果可视化
        recon_path = os.path.join(output_dir, 'reconstructions.png')
        visualize_reconstructions(
            model, test_loader, device, num_samples=8,
            save_path=recon_path, is_vae=(args.model_type == 'vae')
        )
        
        # 异常热力图
        heatmap_path = os.path.join(output_dir, 'anomaly_heatmaps.png')
        visualize_anomaly_heatmaps(
            model, test_loader, device, optimal_threshold, num_samples=4,
            save_path=heatmap_path, is_vae=(args.model_type == 'vae')
        )
        
        print(f'可视化结果已保存至: {output_dir}')
    
    # 7. 生成简单报告
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('图像异常检测实验报告\n')
        f.write('=' * 50 + '\n\n')
        
        f.write('实验配置\n')
        f.write('-' * 20 + '\n')
        f.write(f'数据集类别: {args.category}\n')
        f.write(f'模型类型: {args.model_type}\n')
        f.write(f'潜在空间维度: {args.latent_dim}\n')
        f.write(f'训练轮数: {args.epochs}\n')
        f.write(f'学习率: {args.lr}\n')
        f.write(f'最佳阈值: {optimal_threshold:.4f}\n\n')
        
        f.write('评估结果\n')
        f.write('-' * 20 + '\n')
        # 安全访问评估指标，处理可能缺少的键
        f.write(f'AUC-ROC: {metrics["auc_roc"]:.4f}\n' if "auc_roc" in metrics else 'AUC-ROC: N/A\n')
        f.write(f'AUC-PR: {metrics["auc_pr"]:.4f}\n' if "auc_pr" in metrics else 'AUC-PR: N/A\n')
        f.write(f'准确率: {metrics["accuracy"]:.4f}\n' if "accuracy" in metrics else '准确率: N/A\n')
        f.write(f'精确率: {metrics["precision"]:.4f}\n' if "precision" in metrics else '精确率: N/A\n')
        f.write(f'召回率: {metrics["recall"]:.4f}\n' if "recall" in metrics else '召回率: N/A\n')
        f.write(f'F1分数: {metrics["f1_score"]:.4f}\n' if "f1_score" in metrics else 'F1分数: N/A\n')
        f.write(f'\n混淆矩阵:\n{metrics["confusion_matrix"]}\n' if "confusion_matrix" in metrics else '\n混淆矩阵: N/A\n')
    
    print(f'实验报告已保存至: {report_path}')
    print('\n图像异常检测任务完成！')

if __name__ == '__main__':
    main()