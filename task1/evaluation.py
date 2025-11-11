"""
聚类评估模块
实现多种评估指标：ARI, NMI, Silhouette Score, Adjusted Rand Index等
"""

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    silhouette_samples,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class ClusteringEvaluator:
    """聚类评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.label_encoder = LabelEncoder()
    
    def encode_labels(self, labels):
        """
        编码标签为数值
        
        Args:
            labels: 标签列表（可以是字符串）
            
        Returns:
            numpy array: 编码后的数值标签
        """
        if isinstance(labels[0], str):
            return self.label_encoder.fit_transform(labels)
        return np.array(labels)
    
    def adjusted_rand_index(self, true_labels, pred_labels):
        """
        计算调整兰德指数 (ARI)
        范围: [-1, 1]，1表示完全一致，0表示随机
        
        Args:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        Returns:
            float: ARI分数
        """
        true_labels = self.encode_labels(true_labels)
        pred_labels = self.encode_labels(pred_labels)
        
        return adjusted_rand_score(true_labels, pred_labels)
    
    def normalized_mutual_info(self, true_labels, pred_labels):
        """
        计算标准化互信息 (NMI)
        范围: [0, 1]，1表示完全一致
        
        Args:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        Returns:
            float: NMI分数
        """
        true_labels = self.encode_labels(true_labels)
        pred_labels = self.encode_labels(pred_labels)
        
        return normalized_mutual_info_score(true_labels, pred_labels)
    
    def silhouette_score_custom(self, features, pred_labels):
        """
        计算轮廓系数 (Silhouette Score)
        范围: [-1, 1]，值越大越好
        
        Args:
            features: 特征矩阵
            pred_labels: 预测标签
            
        Returns:
            float: 轮廓系数
        """
        # 移除噪声点（标签为-1的点）
        mask = pred_labels != -1
        if mask.sum() < 2:
            return -1
        
        features_clean = features[mask]
        labels_clean = pred_labels[mask]
        
        if len(np.unique(labels_clean)) < 2:
            return -1
        
        return silhouette_score(features_clean, labels_clean)
    
    def homogeneity_score_custom(self, true_labels, pred_labels):
        """
        计算同质性分数
        每个聚类只包含单一类的成员
        
        Args:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        Returns:
            float: 同质性分数
        """
        true_labels = self.encode_labels(true_labels)
        pred_labels = self.encode_labels(pred_labels)
        
        return homogeneity_score(true_labels, pred_labels)
    
    def completeness_score_custom(self, true_labels, pred_labels):
        """
        计算完整性分数
        给定类的所有成员都分配给同一个聚类
        
        Args:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        Returns:
            float: 完整性分数
        """
        true_labels = self.encode_labels(true_labels)
        pred_labels = self.encode_labels(pred_labels)
        
        return completeness_score(true_labels, pred_labels)
    
    def v_measure_score_custom(self, true_labels, pred_labels):
        """
        计算V-measure分数
        同质性和完整性的调和平均
        
        Args:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        Returns:
            float: V-measure分数
        """
        true_labels = self.encode_labels(true_labels)
        pred_labels = self.encode_labels(pred_labels)
        
        return v_measure_score(true_labels, pred_labels)
    
    def calinski_harabasz_score_custom(self, features, pred_labels):
        """
        计算Calinski-Harabasz指数（方差比标准）
        值越大越好
        
        Args:
            features: 特征矩阵
            pred_labels: 预测标签
            
        Returns:
            float: CH分数
        """
        # 移除噪声点
        mask = pred_labels != -1
        if mask.sum() < 2:
            return 0
        
        features_clean = features[mask]
        labels_clean = pred_labels[mask]
        
        if len(np.unique(labels_clean)) < 2:
            return 0
        
        return calinski_harabasz_score(features_clean, labels_clean)
    
    def davies_bouldin_score_custom(self, features, pred_labels):
        """
        计算Davies-Bouldin指数
        值越小越好
        
        Args:
            features: 特征矩阵
            pred_labels: 预测标签
            
        Returns:
            float: DB分数
        """
        # 移除噪声点
        mask = pred_labels != -1
        if mask.sum() < 2:
            return float('inf')
        
        features_clean = features[mask]
        labels_clean = pred_labels[mask]
        
        if len(np.unique(labels_clean)) < 2:
            return float('inf')
        
        return davies_bouldin_score(features_clean, labels_clean)
    
    def evaluate_all(self, features, true_labels, pred_labels, verbose=True):
        """
        计算所有评估指标
        
        Args:
            features: 特征矩阵
            true_labels: 真实标签
            pred_labels: 预测标签
            verbose: 是否打印结果
            
        Returns:
            dict: 评估指标字典
        """
        results = {}
        
        # 有监督指标（需要真实标签）
        if true_labels is not None:
            results['ARI'] = self.adjusted_rand_index(true_labels, pred_labels)
            results['NMI'] = self.normalized_mutual_info(true_labels, pred_labels)
            results['Homogeneity'] = self.homogeneity_score_custom(true_labels, pred_labels)
            results['Completeness'] = self.completeness_score_custom(true_labels, pred_labels)
            results['V-measure'] = self.v_measure_score_custom(true_labels, pred_labels)
        
        # 无监督指标（只需要特征和预测标签）
        results['Silhouette'] = self.silhouette_score_custom(features, pred_labels)
        results['Calinski-Harabasz'] = self.calinski_harabasz_score_custom(features, pred_labels)
        results['Davies-Bouldin'] = self.davies_bouldin_score_custom(features, pred_labels)
        
        if verbose:
            print("\n" + "="*50)
            print("聚类评估结果")
            print("="*50)
            for metric, score in results.items():
                if isinstance(score, float) and not np.isinf(score):
                    print(f"{metric:20s}: {score:.4f}")
                else:
                    print(f"{metric:20s}: {score}")
            print("="*50)
        
        return results
    
    def compare_methods(self, features, true_labels, pred_labels_dict, verbose=True):
        """
        比较多种聚类方法的结果
        
        Args:
            features: 特征矩阵
            true_labels: 真实标签
            pred_labels_dict: 字典，键为方法名，值为预测标签
            verbose: 是否打印结果
            
        Returns:
            pandas DataFrame: 比较结果表格
        """
        all_results = []
        
        for method_name, pred_labels in pred_labels_dict.items():
            results = self.evaluate_all(features, true_labels, pred_labels, verbose=False)
            results['Method'] = method_name
            all_results.append(results)
        
        df = pd.DataFrame(all_results)
        
        # 重新排列列顺序
        cols = ['Method'] + [col for col in df.columns if col != 'Method']
        df = df[cols]
        
        if verbose:
            print("\n" + "="*80)
            print("聚类方法比较")
            print("="*80)
            print(df.to_string(index=False))
            print("="*80)
        
        return df


if __name__ == "__main__":
    # 测试代码
    print("Testing evaluation metrics...")
    
    evaluator = ClusteringEvaluator()
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    features = np.random.rand(n_samples, n_features)
    
    # 生成真实标签和预测标签
    true_labels = np.random.randint(0, 3, n_samples)
    pred_labels = np.random.randint(0, 3, n_samples)
    
    # 测试评估
    results = evaluator.evaluate_all(features, true_labels, pred_labels)
    print("\nEvaluation results:", results)

