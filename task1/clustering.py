"""
聚类算法模块
实现多种聚类算法：K-means, DBSCAN, 层次聚类, 谱聚类等
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class ClusteringAlgorithms:
    """聚类算法集合"""
    
    def __init__(self, n_clusters=6, random_state=42):
        """
        初始化聚类算法
        
        Args:
            n_clusters: 聚类数量
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
    
    def preprocess_features(self, features, use_pca=False, n_components=None):
        """
        预处理特征：标准化和可选PCA降维
        
        Args:
            features: 特征矩阵
            use_pca: 是否使用PCA降维
            n_components: PCA主成分数量
            
        Returns:
            numpy array: 预处理后的特征
        """
        # 标准化
        features_scaled = self.scaler.fit_transform(features)
        
        if use_pca and n_components is not None:
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            features_scaled = self.pca.fit_transform(features_scaled)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return features_scaled
    
    def kmeans(self, features, n_clusters=None, init='k-means++', n_init=10, max_iter=300):
        """
        K-means聚类
        
        Args:
            features: 特征矩阵
            n_clusters: 聚类数量（如果为None，使用self.n_clusters）
            init: 初始化方法
            n_init: 运行次数
            max_iter: 最大迭代次数
            
        Returns:
            numpy array: 聚类标签
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state
        )
        
        labels = kmeans.fit_predict(features)
        
        return labels, kmeans
    
    def dbscan(self, features, eps=0.5, min_samples=5):
        """
        DBSCAN聚类（密度聚类）
        
        Args:
            features: 特征矩阵
            eps: 邻域半径
            min_samples: 最小样本数
            
        Returns:
            numpy array: 聚类标签（-1表示噪声点）
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        
        # 统计聚类结果
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        return labels, dbscan
    
    def hierarchical(self, features, n_clusters=None, linkage='ward', affinity='euclidean'):
        """
        层次聚类（凝聚聚类）
        
        Args:
            features: 特征矩阵
            n_clusters: 聚类数量
            linkage: 链接准则 ('ward', 'complete', 'average', 'single')
            affinity: 距离度量
            
        Returns:
            numpy array: 聚类标签
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        # ward方法只能使用euclidean距离
        if linkage == 'ward':
            affinity = 'euclidean'
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity
        )
        
        labels = hierarchical.fit_predict(features)
        
        return labels, hierarchical
    
    def spectral(self, features, n_clusters=None, affinity='rbf', gamma=1.0):
        """
        谱聚类
        
        Args:
            features: 特征矩阵
            n_clusters: 聚类数量
            affinity: 相似度矩阵构建方法 ('rbf', 'nearest_neighbors')
            gamma: RBF核参数
            
        Returns:
            numpy array: 聚类标签
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            gamma=gamma,
            random_state=self.random_state
        )
        
        labels = spectral.fit_predict(features)
        
        return labels, spectral
    
    def gmm(self, features, n_clusters=None, covariance_type='full'):
        """
        高斯混合模型聚类
        
        Args:
            features: 特征矩阵
            n_clusters: 聚类数量（组件数）
            covariance_type: 协方差类型
            
        Returns:
            numpy array: 聚类标签
        """
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            random_state=self.random_state,
            max_iter=100
        )
        
        labels = gmm.fit_predict(features)
        
        return labels, gmm
    
    def cluster(self, features, method='kmeans', **kwargs):
        """
        执行聚类
        
        Args:
            features: 特征矩阵
            method: 聚类方法 ('kmeans', 'dbscan', 'hierarchical', 'spectral', 'gmm')
            **kwargs: 其他参数
            
        Returns:
            tuple: (聚类标签, 模型对象)
        """
        if method == 'kmeans':
            return self.kmeans(features, **kwargs)
        elif method == 'dbscan':
            return self.dbscan(features, **kwargs)
        elif method == 'hierarchical':
            return self.hierarchical(features, **kwargs)
        elif method == 'spectral':
            return self.spectral(features, **kwargs)
        elif method == 'gmm':
            return self.gmm(features, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")


if __name__ == "__main__":
    # 测试代码
    print("Testing clustering algorithms...")
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    X = np.random.rand(n_samples, n_features)
    
    clusterer = ClusteringAlgorithms(n_clusters=3, random_state=42)
    
    # 测试K-means
    labels_kmeans, model_kmeans = clusterer.kmeans(X)
    print(f"K-means labels: {np.unique(labels_kmeans)}")
    
    # 测试层次聚类
    labels_hier, model_hier = clusterer.hierarchical(X)
    print(f"Hierarchical labels: {np.unique(labels_hier)}")

