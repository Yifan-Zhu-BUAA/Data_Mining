"""
无监督疾病判断任务 - 异常检测模型实现
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示和随机种子
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

class AnomalyDetector:
    """异常检测器基类"""

    def __init__(self, name):
        self.name = name
        self.model = None

    def fit(self, X_train):
        """训练模型"""
        raise NotImplementedError

    def predict(self, X_test):
        """预测异常分数或标签"""
        raise NotImplementedError

    def get_anomaly_score(self, X):
        """获取异常分数（分数越高越异常）"""
        raise NotImplementedError


class IsolationForestDetector(AnomalyDetector):
    """孤立森林异常检测器

    原理：
    - 异常样本更容易被孤立（在特征空间中与其他样本距离较远）
    - 通过随机切分特征空间，异常点需要更少的切分次数就能被孤立
    - 适合处理高维数据和异常检测问题

    优点：
    - 不需要假设数据分布
    - 对异常值敏感度高
    - 训练速度快
    """

    def __init__(self, contamination=0.05, n_estimators=100):
        super().__init__("Isolation Forest")
        self.contamination = contamination
        self.n_estimators = n_estimators

    def fit(self, X_train):
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train)
        return self

    def predict(self, X_test):
        # 返回预测标签: 1表示正常，-1表示异常
        pred = self.model.predict(X_test)
        # 转换为0-1标签：0表示正常，1表示异常
        return (pred == -1).astype(int)

    def get_anomaly_score(self, X):
        # decision_function返回的是anomaly score，值越小越异常
        # 取负号使得分数越高越异常
        return -self.model.decision_function(X)


class OneClassSVMDetector(AnomalyDetector):
    """单类支持向量机异常检测器

    原理：
    - 在高维空间中找到一个超平面，将正常样本包围起来
    - 使用核技巧将数据映射到高维空间
    - 新样本如果在边界外则被认为是异常

    优点：
    - 理论基础扎实
    - 适合高维数据
    - 对噪声鲁棒
    """

    def __init__(self, nu=0.05, kernel='rbf', gamma='auto'):
        super().__init__("One-Class SVM")
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X_train):
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma
        )
        self.model.fit(X_train)
        return self

    def predict(self, X_test):
        pred = self.model.predict(X_test)
        return (pred == -1).astype(int)

    def get_anomaly_score(self, X):
        return -self.model.decision_function(X)


class LOFDetector(AnomalyDetector):
    """局部离群因子异常检测器

    原理：
    - 基于局部密度的异常检测方法
    - 比较每个样本与其邻居的局部密度
    - 密度明显低于邻居的样本被认为是异常

    优点：
    - 能够发现局部异常（在全局看正常但在局部看异常的样本）
    - 不需要假设数据分布
    - 对不同密度的聚类效果好
    """

    def __init__(self, n_neighbors=20, contamination=0.05):
        super().__init__("Local Outlier Factor")
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def fit(self, X_train):
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True  # 设置为True以便在新数据上预测
        )
        self.model.fit(X_train)
        return self

    def predict(self, X_test):
        pred = self.model.predict(X_test)
        return (pred == -1).astype(int)

    def get_anomaly_score(self, X):
        return -self.model.decision_function(X)


class EllipticEnvelopeDetector(AnomalyDetector):
    """椭圆包络异常检测器

    原理：
    - 假设正常数据服从高斯分布
    - 通过鲁棒协方差估计拟合一个椭圆包络
    - 在椭圆外的样本被认为是异常

    优点：
    - 对多元正态分布数据效果好
    - 对异常值鲁棒
    - 计算效率高
    """

    def __init__(self, contamination=0.05):
        super().__init__("Elliptic Envelope")
        self.contamination = contamination

    def fit(self, X_train):
        self.model = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(X_train)
        return self

    def predict(self, X_test):
        pred = self.model.predict(X_test)
        return (pred == -1).astype(int)

    def get_anomaly_score(self, X):
        return -self.model.decision_function(X)


class GMMDetector(AnomalyDetector):
    """高斯混合模型异常检测器

    原理：
    - 假设正常数据由多个高斯分布混合而成
    - 计算每个样本的对数似然概率
    - 概率低的样本被认为是异常

    优点：
    - 能够建模复杂的数据分布
    - 提供概率解释
    - 适合多模态分布
    """

    def __init__(self, n_components=3, contamination=0.05):
        super().__init__("Gaussian Mixture Model")
        self.n_components = n_components
        self.contamination = contamination
        self.threshold = None

    def fit(self, X_train):
        self.model = GaussianMixture(
            n_components=self.n_components,
            random_state=42,
            covariance_type='full'
        )
        self.model.fit(X_train)

        # 计算训练集的分数，用于确定阈值
        train_scores = self.get_anomaly_score(X_train)
        self.threshold = np.percentile(train_scores, 100 * (1 - self.contamination))

        return self

    def predict(self, X_test):
        scores = self.get_anomaly_score(X_test)
        return (scores > self.threshold).astype(int)

    def get_anomaly_score(self, X):
        # 返回负对数似然（值越大越异常）
        return -self.model.score_samples(X)


def load_data():
    """加载数据"""
    train_data = pd.read_csv('../thyroid/thyroid/train-set.csv')
    test_data = pd.read_csv('../thyroid/thyroid/test-set.csv')

    X_train = train_data.values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values

    return X_train, X_test, y_test


def evaluate_model(y_true, y_pred, y_scores, model_name):
    """评估模型性能"""
    print(f"\n{'='*80}")
    print(f"{model_name} 评估结果")
    print(f"{'='*80}")

    # 基本指标
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n准确率 (Accuracy): {acc:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # ROC-AUC
    auc = roc_auc_score(y_true, y_scores)
    print(f"ROC-AUC: {auc:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n混淆矩阵:")
    print(f"  预测正常  预测患病")
    print(f"实际正常: {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"实际患病: {cm[1,0]:6d}   {cm[1,1]:6d}")

    # 分类报告
    print(f"\n详细分类报告:")
    print(classification_report(y_true, y_pred,
                                target_names=['正常', '患病'],
                                digits=4))

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def plot_results(results_dict, y_test):
    """可视化所有模型的结果"""

    # 1. 性能指标对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数', 'ROC-AUC']

    # 准备数据
    metric_data = {metric: [] for metric in metrics}
    for model in models:
        for metric in metrics:
            metric_data[metric].append(results_dict[model]['metrics'][metric])

    # 绘制条形图
    x = np.arange(len(models))
    width = 0.15

    ax = axes[0, 0]
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        offset = width * (i - 2)
        ax.bar(x + offset, metric_data[metric], width, label=name)

    ax.set_ylabel('分数')
    ax.set_title('各模型性能指标对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. ROC曲线
    ax = axes[0, 1]
    for model_name in models:
        y_scores = results_dict[model_name]['y_scores']
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc = results_dict[model_name]['metrics']['auc']
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机猜测')
    ax.set_xlabel('假阳性率 (FPR)')
    ax.set_ylabel('真阳性率 (TPR)')
    ax.set_title('ROC曲线对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Precision-Recall曲线
    ax = axes[1, 0]
    for model_name in models:
        y_scores = results_dict[model_name]['y_scores']
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_scores)
        ax.plot(recall_vals, precision_vals, label=model_name, linewidth=2)

    ax.set_xlabel('召回率 (Recall)')
    ax.set_ylabel('精确率 (Precision)')
    ax.set_title('Precision-Recall曲线对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. F1分数和AUC对比
    ax = axes[1, 1]
    f1_scores = [results_dict[m]['metrics']['f1'] for m in models]
    auc_scores = [results_dict[m]['metrics']['auc'] for m in models]

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, f1_scores, width, label='F1分数', alpha=0.8)
    ax.bar(x + width/2, auc_scores, width, label='ROC-AUC', alpha=0.8)

    ax.set_ylabel('分数')
    ax.set_title('F1分数和AUC对比')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n模型对比图已保存为 model_comparison.png")

    # 绘制混淆矩阵
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, model_name in enumerate(models):
        cm = results_dict[model_name]['metrics']['confusion_matrix']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['预测正常', '预测患病'],
                   yticklabels=['实际正常', '实际患病'],
                   ax=axes[idx], cbar_kws={'label': '样本数'})
        axes[idx].set_title(f'{model_name}\n混淆矩阵')

    # 隐藏多余的子图
    if len(models) < 6:
        for idx in range(len(models), 6):
            axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("混淆矩阵图已保存为 confusion_matrices.png")


def main():
    """主函数"""
    print("="*80)
    print("无监督疾病判断任务 - 异常检测")
    print("="*80)

    # 1. 加载数据
    print("\n正在加载数据...")
    X_train, X_test, y_test = load_data()
    print(f"训练集样本数: {X_train.shape[0]}, 特征维度: {X_train.shape[1]}")
    print(f"测试集样本数: {X_test.shape[0]}, 患病样本数: {y_test.sum()}")

    # 2. 初始化所有模型
    contamination = y_test.mean()  # 使用真实的异常比例
    print(f"\n设置污染率 (contamination): {contamination:.4f}")

    detectors = [
        IsolationForestDetector(contamination=contamination, n_estimators=100),
        OneClassSVMDetector(nu=contamination, kernel='rbf'),
        LOFDetector(n_neighbors=20, contamination=contamination),
        EllipticEnvelopeDetector(contamination=contamination),
        GMMDetector(n_components=3, contamination=contamination)
    ]

    # 3. 训练和评估每个模型
    results_dict = {}

    for detector in detectors:
        print(f"\n{'='*80}")
        print(f"正在训练 {detector.name}...")
        print(f"{'='*80}")

        # 训练
        detector.fit(X_train)

        # 预测
        y_pred = detector.predict(X_test)
        y_scores = detector.get_anomaly_score(X_test)

        # 评估
        metrics = evaluate_model(y_test, y_pred, y_scores, detector.name)

        # 保存结果
        results_dict[detector.name] = {
            'y_pred': y_pred,
            'y_scores': y_scores,
            'metrics': metrics
        }

    # 4. 可视化结果
    print(f"\n{'='*80}")
    print("正在生成可视化结果...")
    print(f"{'='*80}")
    plot_results(results_dict, y_test)

    # 5. 总结最佳模型
    print(f"\n{'='*80}")
    print("最佳模型总结")
    print(f"{'='*80}")

    best_f1_model = max(results_dict.items(),
                        key=lambda x: x[1]['metrics']['f1'])
    best_auc_model = max(results_dict.items(),
                         key=lambda x: x[1]['metrics']['auc'])

    print(f"\nF1分数最高的模型: {best_f1_model[0]}")
    print(f"  F1分数: {best_f1_model[1]['metrics']['f1']:.4f}")
    print(f"  ROC-AUC: {best_f1_model[1]['metrics']['auc']:.4f}")

    print(f"\nROC-AUC最高的模型: {best_auc_model[0]}")
    print(f"  F1分数: {best_auc_model[1]['metrics']['f1']:.4f}")
    print(f"  ROC-AUC: {best_auc_model[1]['metrics']['auc']:.4f}")

    # 6. 保存结果到CSV
    results_df = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy': [r['metrics']['accuracy'] for r in results_dict.values()],
        'Precision': [r['metrics']['precision'] for r in results_dict.values()],
        'Recall': [r['metrics']['recall'] for r in results_dict.values()],
        'F1-Score': [r['metrics']['f1'] for r in results_dict.values()],
        'ROC-AUC': [r['metrics']['auc'] for r in results_dict.values()]
    })
    results_df.to_csv('model_results.csv', index=False)
    print("\n模型评估结果已保存为 model_results.csv")


if __name__ == '__main__':
    main()
