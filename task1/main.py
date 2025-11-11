"""
图像聚类主程序
整合特征提取、聚类和评估功能
"""

import os
import json
import numpy as np
import pandas as pd
from feature_extraction import ImageFeatureExtractor, load_images_from_directory
from clustering import ClusteringAlgorithms
from evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def visualize_clustering_results(features, labels, true_labels, method_name, save_path=None):
    """
    可视化聚类结果（使用t-SNE降维）
    
    Args:
        features: 特征矩阵
        labels: 预测标签
        true_labels: 真实标签
        method_name: 方法名称
        save_path: 保存路径
    """
    from sklearn.preprocessing import LabelEncoder
    
    # 编码标签为数值
    le = LabelEncoder()
    if isinstance(true_labels[0], str):
        true_labels_encoded = le.fit_transform(true_labels)
    else:
        true_labels_encoded = np.array(true_labels)
    
    # 使用t-SNE降维到2D
    print(f"Applying t-SNE for {method_name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：真实标签
    scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=true_labels_encoded, cmap='tab10', s=20, alpha=0.6)
    axes[0].set_title(f'真实标签分布 (t-SNE)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # 右图：预测标签
    scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=labels, cmap='tab10', s=20, alpha=0.6)
    axes[1].set_title(f'{method_name} 聚类结果 (t-SNE)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()


def main():
    """主函数"""
    print("="*80)
    print("图像聚类任务")
    print("="*80)
    
    # 配置路径
    dataset_dir = "../cluster/Cluster/dataset"
    labels_file = "../cluster/Cluster/cluster_labels.json"
    output_dir = "results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载图像和标签
    print("\n[步骤1] 加载图像数据...")
    image_paths, true_labels = load_images_from_directory(dataset_dir, labels_file)
    print(f"成功加载 {len(image_paths)} 张图像")
    print(f"类别分布: {pd.Series(true_labels).value_counts().to_dict()}")
    
    # 2. 提取特征
    print("\n[步骤2] 提取图像特征...")
    
    # 选择特征提取方法（可以根据需要修改）
    feature_method = 'resnet'  # 可选: 'resnet', 'hog', 'histogram', 'combined'
    
    print(f"使用特征提取方法: {feature_method}")
    
    # 检查是否有GPU
    device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    print(f"使用设备: {device}")
    
    extractor = ImageFeatureExtractor(method=feature_method, device=device)
    features, valid_paths = extractor.extract_features_batch(image_paths, verbose=True)
    
    if features is None:
        print("特征提取失败！")
        return
    
    # 过滤有效标签
    valid_indices = [i for i, path in enumerate(image_paths) if path in valid_paths]
    true_labels_filtered = [true_labels[i] for i in valid_indices]
    
    print(f"特征矩阵形状: {features.shape}")
    
    # 保存特征
    np.save(os.path.join(output_dir, 'features.npy'), features)
    print(f"特征已保存到 {output_dir}/features.npy")
    
    # 3. 聚类
    print("\n[步骤3] 执行聚类...")
    
    n_clusters = 6  # 已知有6个类别
    clusterer = ClusteringAlgorithms(n_clusters=n_clusters, random_state=42)
    
    # 预处理特征
    features_scaled = clusterer.preprocess_features(features, use_pca=False)
    
    # 测试多种聚类方法
    clustering_methods = {
        'K-means': 'kmeans',
        'DBSCAN': 'dbscan',
        'Hierarchical': 'hierarchical',
        'Spectral': 'spectral',
        'GMM': 'gmm'
    }
    
    all_labels = {}
    all_results = {}
    
    for method_name, method_key in clustering_methods.items():
        print(f"\n执行 {method_name} 聚类...")
        
        try:
            if method_key == 'dbscan':
                # DBSCAN需要特殊处理
                labels, model = clusterer.dbscan(features_scaled, eps=0.5, min_samples=5)
            else:
                labels, model = clusterer.cluster(features_scaled, method=method_key)
            
            all_labels[method_name] = labels
            
            # 评估
            evaluator = ClusteringEvaluator()
            results = evaluator.evaluate_all(
                features_scaled, 
                true_labels_filtered, 
                labels, 
                verbose=False
            )
            all_results[method_name] = results
            
            print(f"{method_name} 完成，ARI: {results.get('ARI', 'N/A'):.4f}")
            
            # 可视化（只对前几个方法）
            if method_name in ['K-means', 'Hierarchical', 'Spectral']:
                vis_path = os.path.join(output_dir, f'visualization_{method_name}.png')
                visualize_clustering_results(
                    features_scaled, 
                    labels, 
                    true_labels_filtered, 
                    method_name,
                    vis_path
                )
        
        except Exception as e:
            print(f"{method_name} 聚类失败: {e}")
            continue
    
    # 4. 评估和比较
    print("\n[步骤4] 评估聚类结果...")
    
    evaluator = ClusteringEvaluator()
    comparison_df = evaluator.compare_methods(
        features_scaled,
        true_labels_filtered,
        all_labels,
        verbose=True
    )
    
    # 保存比较结果
    comparison_df.to_csv(os.path.join(output_dir, 'comparison_results.csv'), index=False, encoding='utf-8-sig')
    print(f"\n比较结果已保存到 {output_dir}/comparison_results.csv")
    
    # 5. 生成详细报告
    print("\n[步骤5] 生成详细报告...")
    
    report_path = os.path.join(output_dir, 'detailed_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("图像聚类详细报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("数据集信息:\n")
        f.write(f"  图像数量: {len(valid_paths)}\n")
        f.write(f"  特征维度: {features.shape[1]}\n")
        f.write(f"  类别数量: {len(set(true_labels_filtered))}\n")
        f.write(f"  类别分布: {pd.Series(true_labels_filtered).value_counts().to_dict()}\n\n")
        
        f.write("特征提取方法:\n")
        f.write(f"  {feature_method}\n\n")
        
        f.write("聚类方法比较:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("各方法详细结果:\n")
        for method_name, results in all_results.items():
            f.write(f"\n{method_name}:\n")
            for metric, score in results.items():
                if isinstance(score, float) and not np.isinf(score):
                    f.write(f"  {metric:20s}: {score:.4f}\n")
                else:
                    f.write(f"  {metric:20s}: {score}\n")
    
    print(f"详细报告已保存到 {report_path}")
    
    # 6. 找出最佳方法
    print("\n[步骤6] 最佳聚类方法分析...")
    
    # 根据ARI选择最佳方法
    best_method = None
    best_ari = -1
    
    for method_name, results in all_results.items():
        ari = results.get('ARI', -1)
        if isinstance(ari, float) and ari > best_ari:
            best_ari = ari
            best_method = method_name
    
    if best_method:
        print(f"\n最佳聚类方法（基于ARI）: {best_method}")
        print(f"ARI分数: {best_ari:.4f}")
        print(f"\n{best_method} 的完整评估结果:")
        for metric, score in all_results[best_method].items():
            if isinstance(score, float) and not np.isinf(score):
                print(f"  {metric:20s}: {score:.4f}")
    
    print("\n" + "="*80)
    print("任务完成！")
    print("="*80)


if __name__ == "__main__":
    main()

