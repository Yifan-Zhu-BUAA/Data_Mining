"""
主程序入口
"""

import os
import torch
import numpy as np
from datetime import datetime

# 导入自定义模块
from config import config
from data_loader import data_loader
from preprocessing import preprocessor
from model import create_model
from trainer import ModelTrainer
from evaluator import evaluator
from visualization import visualizer

def main():
    """主函数"""
    print("="*80)
    print("时间序列预测任务 - 室外温度预测")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)
    
    # 1. 加载数据
    print("\n[步骤1] 加载数据...")
    if not data_loader.load_data():
        print("数据加载失败，程序退出")
        return
    
    features, target = data_loader.get_features_and_target()
    print(f"特征数据形状: {features.shape}")
    print(f"目标数据形状: {target.shape}")
    
    # 2. 数据预处理
    print("\n[步骤2] 数据预处理和滑动窗口切分...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_dataset(features, target)
    
    # 保存标准化器
    preprocessor.save_scalers()
    
    # 3. 创建模型
    print("\n[步骤3] 创建模型...")
    model = create_model()
    print(f"模型结构:")
    print(model)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 创建训练器并准备数据
    print("\n[步骤4] 准备训练...")
    trainer = ModelTrainer(model)
    train_loader, test_loader = trainer.create_dataloaders(
        X_train, X_test, y_train, y_test
    )
    
    # 5. 训练模型
    print("\n[步骤5] 开始训练模型...")
    print(f"使用设备: {config.DEVICE}")
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
    print(f"批次大小: {config.BATCH_SIZE}, 学习率: {config.LEARNING_RATE}")
    
    trainer.train(train_loader, test_loader)
    
    # 加载最佳模型
    trainer.load_model(model)
    
    # 6. 在测试集上评估
    print("\n[步骤6] 在测试集上评估模型...")
    model.eval()
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            if config.USE_ATTENTION:
                pred, _ = model(batch_X)
            else:
                pred = model(batch_X)
            predictions.append(pred.squeeze().cpu().numpy())
            true_values.append(batch_y.cpu().numpy())
    
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(true_values)
    
    # 计算评估指标（包括准确率）
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics()
    
    # 打印测试集准确率
    print("\n" + "="*60)
    print("测试集准确率（基于相对误差阈值）")
    print("="*60)
    print(f"相对误差 < 5%  的样本比例: {metrics.get('Accuracy (5%)', 0):.2f}%")
    print(f"相对误差 < 10% 的样本比例: {metrics.get('Accuracy (10%)', 0):.2f}%")
    print(f"相对误差 < 20% 的样本比例: {metrics.get('Accuracy (20%)', 0):.2f}%")
    print("="*60)
    
    # 保存评估结果
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(config.RESULTS_DIR, 'evaluation_metrics.json')
    evaluator.save_metrics(metrics_path)
    
    # 保存预测结果
    results_df = {
        'y_true': y_true,
        'y_pred': y_pred,
        'error': y_true - y_pred
    }
    import pandas as pd
    results_df = pd.DataFrame(results_df)
    results_path = os.path.join(config.RESULTS_DIR, 'predictions.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到: {results_path}")
    
    # 7. 可视化结果
    print("\n[步骤7] 生成可视化结果...")
    
    # 训练历史
    history_path = os.path.join(config.VISUALIZATION_DIR, 'training_history.png')
    visualizer.plot_training_history(trainer.train_losses, trainer.val_losses, history_path)
    
    # 预测结果对比
    pred_path = os.path.join(config.VISUALIZATION_DIR, 'predictions.png')
    visualizer.plot_predictions(y_true, y_pred, "测试集预测结果对比", pred_path)
    
    # 散点图
    scatter_path = os.path.join(config.VISUALIZATION_DIR, 'prediction_scatter.png')
    visualizer.plot_prediction_scatter(y_true, y_pred, scatter_path)
    
    # 误差分布
    error_path = os.path.join(config.VISUALIZATION_DIR, 'error_distribution.png')
    visualizer.plot_error_distribution(y_true, y_pred, error_path)
    
    # 残差图
    residual_path = os.path.join(config.VISUALIZATION_DIR, 'residuals.png')
    visualizer.plot_residuals(y_true, y_pred, residual_path)
    
    # 8. 生成总结报告
    print("\n[步骤8] 生成总结报告...")
    report_path = os.path.join(config.RESULTS_DIR, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("时间序列预测任务 - 实验报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("实验配置:\n")
        f.write("-"*40 + "\n")
        f.write(f"窗口大小: {config.WINDOW_SIZE} (2小时)\n")
        f.write(f"特征维度: {config.INPUT_DIM}\n")
        f.write(f"隐藏层维度: {config.HIDDEN_DIM}\n")
        f.write(f"LSTM层数: {config.NUM_LAYERS}\n")
        f.write(f"是否使用注意力: {config.USE_ATTENTION}\n")
        f.write(f"批次大小: {config.BATCH_SIZE}\n")
        f.write(f"学习率: {config.LEARNING_RATE}\n")
        f.write(f"训练轮数: {len(trainer.train_losses)}\n\n")
        
        f.write("数据集信息:\n")
        f.write("-"*40 + "\n")
        f.write(f"训练集大小: {len(X_train)}\n")
        f.write(f"测试集大小: {len(X_test)}\n\n")
        
        f.write("评估结果:\n")
        f.write("-"*40 + "\n")
        for metric, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{metric:20s}: {value:10.6f}\n")
            else:
                f.write(f"{metric:20s}: {value:10.2f}\n")
        
        f.write(f"\n最佳验证损失: {trainer.best_loss:.6f}\n")
        f.write(f"最终训练损失: {trainer.train_losses[-1]:.6f}\n")
        f.write(f"最终验证损失: {trainer.val_losses[-1]:.6f}\n")
    
    print(f"实验报告已保存到: {report_path}")
    
    print("\n" + "="*80)
    print("任务完成！")
    print("="*80)
    print(f"\n结果文件位置:")
    print(f"  - 模型: {config.MODEL_SAVE_PATH}")
    print(f"  - 评估指标: {metrics_path}")
    print(f"  - 预测结果: {results_path}")
    print(f"  - 实验报告: {report_path}")
    print(f"  - 可视化: {config.VISUALIZATION_DIR}/")

if __name__ == "__main__":
    main()
