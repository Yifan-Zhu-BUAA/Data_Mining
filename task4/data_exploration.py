import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
train_data = pd.read_csv('../thyroid/thyroid/train-set.csv')
test_data = pd.read_csv('../thyroid/thyroid/test-set.csv')

print("=" * 80)
print("训练集信息：")
print("=" * 80)
print(f"训练集样本数: {len(train_data)}")
print(f"训练集特征维度: {train_data.shape[1]}")
print("\n训练集前5行：")
print(train_data.head())
print("\n训练集统计信息：")
print(train_data.describe())
print("\n训练集缺失值：")
print(train_data.isnull().sum())

print("\n" + "=" * 80)
print("测试集信息：")
print("=" * 80)
print(f"测试集样本数: {len(test_data)}")
print(f"测试集特征维度: {test_data.shape[1]}")
print("\n测试集前5行：")
print(test_data.head())
print("\n测试集标签分布：")
if 'label' in test_data.columns:
    print(test_data['label'].value_counts())
    print(f"\n患病样本比例: {test_data['label'].sum() / len(test_data) * 100:.2f}%")
print("\n测试集统计信息：")
print(test_data.describe())
print("\n测试集缺失值：")
print(test_data.isnull().sum())

# 可视化数据分布
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
feature_cols = [col for col in train_data.columns if col != 'label']

for idx, col in enumerate(feature_cols):
    row = idx // 3
    col_idx = idx % 3

    axes[row, col_idx].hist(train_data[col], bins=50, alpha=0.7, label='训练集(正常)', color='blue', density=True)
    if 'label' in test_data.columns:
        test_normal = test_data[test_data['label'] == 0][col]
        test_abnormal = test_data[test_data['label'] == 1][col]
        axes[row, col_idx].hist(test_normal, bins=50, alpha=0.5, label='测试集(正常)', color='green', density=True)
        axes[row, col_idx].hist(test_abnormal, bins=50, alpha=0.5, label='测试集(患病)', color='red', density=True)

    axes[row, col_idx].set_xlabel(col)
    axes[row, col_idx].set_ylabel('密度')
    axes[row, col_idx].legend()
    axes[row, col_idx].set_title(f'{col}分布')

plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
print("\n数据分布图已保存为 data_distribution.png")

# 相关性分析
plt.figure(figsize=(10, 8))
sns.heatmap(train_data[feature_cols].corr(), annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('训练集特征相关性矩阵')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("特征相关性矩阵已保存为 correlation_matrix.png")
