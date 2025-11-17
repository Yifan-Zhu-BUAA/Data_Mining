import pandas as pd
import numpy as np
from config import config

class WeatherDataLoader:
    """气象数据加载器"""
    
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        
    def load_data(self):
        """加载原始数据"""
        try:
            # 尝试不同编码方式读取数据
            try:
                self.data = pd.read_csv(config.DATA_PATH, encoding='utf-8')
            except UnicodeDecodeError:
                self.data = pd.read_csv(config.DATA_PATH, encoding='latin-1')
            
            print(f"数据加载成功，形状: {self.data.shape}")
            print(f"数据列名: {list(self.data.columns)}")
            
            # 处理时间戳
            self.data['date'] = pd.to_datetime(self.data['date'])
            
            # 检查并处理重复的时间戳（保留第一个）
            duplicate_count = self.data['date'].duplicated().sum()
            if duplicate_count > 0:
                print(f"发现 {duplicate_count} 个重复的时间戳，将删除重复项（保留第一个）")
                self.data = self.data.drop_duplicates(subset='date', keep='first')
            
            self.data.set_index('date', inplace=True)
            
            # 确保时间序列连续性（使用10min替代已弃用的10T）
            self.data = self.data.asfreq('10min')
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def get_features_and_target(self):
        """获取特征和目标变量"""
        if self.data is None:
            self.load_data()
        
        # 使用配置中定义的特征列
        self.features = self.data[config.FEATURE_COLUMNS].copy()
        self.target = self.data[config.TARGET_COLUMN].copy()
        
        # 检查缺失值
        self._handle_missing_values()
        
        return self.features, self.target
    
    def _handle_missing_values(self):
        """处理缺失值"""
        # 检查缺失值
        missing_features = self.features.isnull().sum()
        missing_target = self.target.isnull().sum()
        
        print("特征缺失值统计:")
        print(missing_features[missing_features > 0])
        print(f"目标变量缺失值: {missing_target}")
        
        # 向前填充缺失值（使用ffill()替代已弃用的method='ffill'）
        self.features.ffill(inplace=True)
        self.target.ffill(inplace=True)
        
        # 如果还有缺失值，用均值填充
        self.features.fillna(self.features.mean(), inplace=True)
        self.target.fillna(self.target.mean(), inplace=True)

# 单例模式
data_loader = WeatherDataLoader()