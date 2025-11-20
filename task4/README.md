# 无监督疾病判断任务

## 任务概述

本任务使用无监督学习方法对甲状腺疾病进行异常检测。训练集仅包含正常样本，测试集包含正常和患病样本。

## 文件说明

```
task4/
├── data_exploration.py      # 数据探索和可视化脚本
├── anomaly_detection.py     # 异常检测模型实现和训练
├── report.tex               # LaTeX格式的实验报告
├── report.pdf               # 编译后的PDF报告
├── data_distribution.png    # 数据分布可视化
├── correlation_matrix.png   # 特征相关性矩阵
├── model_comparison.png     # 模型性能对比图
├── confusion_matrices.png   # 混淆矩阵可视化
├── model_results.csv        # 模型评估结果
└── README.md                # 本文件
```

## 环境要求

- Python 3.7+
- 依赖库：
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

## 运行步骤

### 1. 数据探索

```bash
cd task4
python data_exploration.py
```

输出：
- `data_distribution.png`：各特征在训练集和测试集中的分布
- `correlation_matrix.png`：特征相关性热力图

### 2. 模型训练和评估

```bash
python anomaly_detection.py
```

输出：
- 控制台输出：每个模型的详细评估指标
- `model_comparison.png`：模型性能对比图（包括各项指标、ROC曲线、PR曲线等）
- `confusion_matrices.png`：所有模型的混淆矩阵
- `model_results.csv`：模型评估结果汇总

### 3. 生成报告

```bash
# 使用XeLaTeX编译LaTeX文档（支持中文）
xelatex report.tex
xelatex report.tex  # 编译两次以生成完整的交叉引用
```

或使用在线LaTeX编译器（如Overleaf）上传`report.tex`文件进行编译。

## 实验结果

### 算法对比

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | ROC-AUC |
|------|--------|--------|--------|--------|---------|
| Isolation Forest | 0.9395 | 0.4433 | **0.9574** | **0.6061** | **0.9787** |
| One-Class SVM | 0.9317 | 0.4087 | 0.9043 | 0.5629 | 0.9606 |
| LOF | **0.9421** | **0.4505** | 0.8723 | 0.5942 | 0.9658 |
| Elliptic Envelope | 0.9374 | 0.4335 | 0.9362 | 0.5926 | 0.9746 |
| GMM | 0.9353 | 0.4244 | 0.9255 | 0.5819 | 0.9718 |

### 最佳模型

**Isolation Forest（孤立森林）** 在F1分数、召回率和ROC-AUC三个关键指标上表现最优：

- **召回率 95.74%**：94个患病样本中成功识别出90个，仅漏诊4个
- **F1分数 0.6061**：在精确率和召回率之间达到最佳平衡
- **ROC-AUC 0.9787**：优秀的分类能力

## 设计思路

### 1. 问题分析

这是一个**异常检测（Anomaly Detection）**问题，因为：
- 训练集仅包含正常样本（无标签或标签全为0）
- 需要识别偏离正常模式的异常样本（患病样本）
- 测试集中患病样本占比约4.86%，类别不平衡

### 2. 算法选择

实现了5种经典的无监督异常检测算法：

#### Isolation Forest（孤立森林）
- **原理**：异常样本更容易被"孤立"（在特征空间中距离较远）
- **优点**：不需要假设数据分布，对异常值敏感度高，训练速度快

#### One-Class SVM（单类支持向量机）
- **原理**：在高维空间中找到超平面包围正常样本
- **优点**：理论基础扎实，适合高维数据

#### LOF（局部离群因子）
- **原理**：基于局部密度检测异常
- **优点**：能发现局部异常，对不同密度的聚类效果好

#### Elliptic Envelope（椭圆包络）
- **原理**：假设数据服从高斯分布，拟合椭圆包络
- **优点**：对多元正态分布数据效果好，计算效率高

#### GMM（高斯混合模型）
- **原理**：用多个高斯分布的混合建模数据
- **优点**：能建模复杂的多模态分布，提供概率解释

### 3. 评估指标

- **召回率（Recall）**：最重要指标，确保尽可能少的漏诊
- **F1分数**：平衡精确率和召回率
- **ROC-AUC**：不受类别不平衡影响，评估整体分类能力
- **混淆矩阵**：详细分析各类错误

### 4. 为什么选择Isolation Forest？

1. **最高召回率**：95.74%，仅漏诊4个患病样本（4.26%）
2. **最佳F1分数和ROC-AUC**：在精确率和召回率之间达到最佳平衡
3. **算法优势**：
   - 不需要假设数据分布
   - 对异常值高度敏感
   - 训练速度快（O(n log n)）
   - 超参数少，易于调优
4. **医疗适用性**：在医疗诊断中，高召回率至关重要，即使精确率相对较低，误诊的患者也可以通过进一步检查排除

## 代码结构

### anomaly_detection.py 核心类

- `AnomalyDetector`：异常检测器基类
- `IsolationForestDetector`：孤立森林实现
- `OneClassSVMDetector`：单类SVM实现
- `LOFDetector`：LOF实现
- `EllipticEnvelopeDetector`：椭圆包络实现
- `GMMDetector`：GMM实现

每个检测器都实现了统一的接口：
- `fit(X_train)`：在训练集上训练模型
- `predict(X_test)`：预测测试集标签
- `get_anomaly_score(X)`：计算异常分数

## 医疗应用建议

1. **两阶段筛查**：
   - 第一阶段：使用Isolation Forest初步筛查（高召回率）
   - 第二阶段：对预测为患病的样本进行进一步临床检查

2. **阈值调整**：根据实际需求调整异常分数阈值
   - 需要更高召回率 → 降低阈值
   - 需要更高精确率 → 提高阈值

3. **集成多模型**：结合多个算法提高鲁棒性

## 参考资料

- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In ICDM.
- Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution. Neural computation.
- Breunig, M. M., et al. (2000). LOF: identifying density-based local outliers. In ACM SIGMOD.
