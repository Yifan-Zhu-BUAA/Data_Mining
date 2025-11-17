"""
评估指标模块 - 计算各种评估指标
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_accuracy(self, y_true, y_pred, error_thresholds=[5, 10, 20]):
        """计算准确率（基于相对误差阈值）"""
        accuracies = {}
        relative_errors = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
        
        for threshold in error_thresholds:
            accuracy = np.mean(relative_errors < threshold) * 100
            accuracies[f'Accuracy ({threshold}%)'] = accuracy
        
        return accuracies
    
    def calculate_metrics(self, y_true, y_pred):
        """计算所有评估指标"""
        # 基本指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # 平均绝对百分比误差 (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # 计算误差统计
        errors = y_true - y_pred
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # 计算准确率
        accuracies = self.calculate_accuracy(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Mean Error': mean_error,
            'Std Error': std_error,
            'Max Error': np.max(np.abs(errors)),
            'Min Error': np.min(np.abs(errors))
        }
        
        # 添加准确率指标
        metrics.update(accuracies)
        
        self.metrics = metrics
        return metrics
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n" + "="*60)
        print("模型评估结果")
        print("="*60)
        for metric, value in self.metrics.items():
            if isinstance(value, float):
                print(f"{metric:20s}: {value:10.6f}")
            else:
                print(f"{metric:20s}: {value:10.2f}")
        print("="*60)
    
    def save_metrics(self, filepath):
        """保存评估指标到文件"""
        import json
        # 将numpy类型转换为Python原生类型，以便JSON序列化
        metrics_serializable = {}
        for key, value in self.metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            else:
                metrics_serializable[key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_serializable, f, indent=4, ensure_ascii=False)
        print(f"评估指标已保存到: {filepath}")
    
    def get_comparison_dataframe(self, predictions_dict):
        """创建多个模型预测结果的比较DataFrame"""
        comparison_data = []
        for model_name, (y_true, y_pred) in predictions_dict.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['Model'] = model_name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        # 重新排列列顺序
        cols = ['Model'] + [col for col in df.columns if col != 'Model']
        df = df[cols]
        return df

# 单例模式
evaluator = ModelEvaluator()
