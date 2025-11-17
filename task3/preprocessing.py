"""
数据预处理模块 - 特征标准化和滑动窗口切分
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from config import config

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
    
    def preprocess_features(self, features):
        """特征标准化"""
        scaled_features = self.feature_scaler.fit_transform(features)
        return scaled_features
    
    def transform_features(self, features):
        """使用已训练的标准化器转换特征"""
        scaled_features = self.feature_scaler.transform(features)
        return scaled_features
    
    def create_sliding_windows(self, features, target, window_size, step=1):
        """创建滑动窗口数据集"""
        X, y = [], []
        n_samples = len(features)
        
        for i in range(0, n_samples - window_size, step):
            X.append(features[i:i+window_size])
            y.append(target[i+window_size])
        
        return np.array(X), np.array(y)
    
    def prepare_dataset(self, features, target):
        """准备完整数据集"""
        # 特征标准化
        scaled_features = self.preprocess_features(features)
        
        # 创建滑动窗口
        X, y = self.create_sliding_windows(
            scaled_features, target.values, 
            config.WINDOW_SIZE, config.PREDICTION_STEP
        )
        
        print(f"滑动窗口数据集形状: X={X.shape}, y={y.shape}")
        
        # 划分训练测试集（按时间顺序）
        split_idx = int(len(X) * (1 - config.TEST_SIZE))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"训练集: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"测试集: X_test={X_test.shape}, y_test={y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_scalers(self):
        """保存标准化器"""
        os.makedirs(os.path.dirname(config.SCALER_SAVE_PATH), exist_ok=True)
        joblib.dump(self.feature_scaler, config.SCALER_SAVE_PATH)
        print(f"特征标准化器已保存到: {config.SCALER_SAVE_PATH}")
    
    def load_scalers(self):
        """加载标准化器"""
        if os.path.exists(config.SCALER_SAVE_PATH):
            self.feature_scaler = joblib.load(config.SCALER_SAVE_PATH)
            print("特征标准化器已加载")
        else:
            print("标准化器文件不存在")

# 单例模式
preprocessor = DataPreprocessor()
