# 图像聚类任务

## 项目结构

```
task1/
├── feature_extraction.py  # 图像特征提取模块
├── clustering.py          # 聚类算法模块
├── evaluation.py          # 聚类评估模块
├── main.py               # 主程序
├── report.tex            # LaTeX报告
├── requirements.txt      # 依赖包
└── README.md            # 说明文档
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

确保数据集位于 `../cluster/Cluster/dataset/` 目录下，包含600张PNG图像。

标签文件位于 `../cluster/Cluster/cluster_labels.json`。

**数据集类别说明**：
- **cable（电缆）**：电缆横截面图像，显示多根绝缘电线
- **tile（瓷砖）**：瓷砖纹理图像，呈现颗粒状纹理
- **bottle（瓶子）**：瓶子顶部俯视图，显示瓶口和瓶颈
- **pill（药丸）**：白色椭圆形药丸图像，表面有压印文字
- **leather（皮革）**：皮革纹理图像，呈现红棕色纹理表面
- **transistor（晶体管）**：电子元件图像，显示三脚晶体管

每个类别包含100张图像，共600张图像。

## 使用方法

### 运行主程序

```bash
python main.py
```

程序将自动完成以下步骤：

1. **加载图像数据**：从指定目录加载所有图像
2. **提取特征**：使用ResNet50提取图像特征（2048维）
3. **执行聚类**：使用多种聚类算法（K-means, DBSCAN, 层次聚类, 谱聚类, GMM）
4. **评估结果**：计算多种评估指标（ARI, NMI, 轮廓系数等）
5. **生成报告**：保存结果到 `results/` 目录

### 结果文件

运行完成后，`results/` 目录将包含：

- `features.npy`: 提取的特征矩阵
- `comparison_results.csv`: 各聚类方法的比较结果
- `detailed_report.txt`: 详细的文本报告
- `visualization_*.png`: 聚类结果可视化图像

## 代码模块说明

### feature_extraction.py

图像特征提取模块，支持多种特征提取方法：

- **ResNet50**: 使用预训练的ResNet50提取深度特征（2048维）
- **HOG**: 方向梯度直方图特征
- **颜色直方图**: RGB三通道颜色直方图
- **SIFT**: 尺度不变特征变换
- **组合特征**: 多种特征的组合

### clustering.py

聚类算法模块，实现了5种经典聚类算法：

- **K-means**: 基于划分的聚类算法
- **DBSCAN**: 基于密度的聚类算法
- **层次聚类**: 凝聚式层次聚类
- **谱聚类**: 基于图论的聚类算法
- **GMM**: 高斯混合模型

### evaluation.py

聚类评估模块，实现了多种评估指标：

**有监督指标**（需要真实标签）：
- ARI (Adjusted Rand Index)
- NMI (Normalized Mutual Information)
- 同质性 (Homogeneity)
- 完整性 (Completeness)
- V-measure

**无监督指标**（不需要真实标签）：
- 轮廓系数 (Silhouette Score)
- Calinski-Harabasz指数
- Davies-Bouldin指数

### main.py

主程序，整合所有功能模块，完成完整的聚类流程。

## 参数配置

可以在 `main.py` 中修改以下参数：

```python
# 特征提取方法
feature_method = 'resnet'  # 可选: 'resnet', 'hog', 'histogram', 'combined'

# 聚类数量
n_clusters = 6

# 设备选择
device = 'cuda'  # 或 'cpu'
```

## 生成LaTeX报告

使用XeLaTeX编译报告：

```bash
xelatex report.tex
```

或者使用在线LaTeX编辑器（如Overleaf）编译。

## 注意事项

1. **GPU支持**：如果有GPU，程序会自动使用GPU加速ResNet50特征提取
2. **内存要求**：处理600张图像需要约2-4GB内存
3. **运行时间**：完整流程约需10-30分钟（取决于硬件配置）
4. **结果数值**：LaTeX报告中的表格数值为占位符，需要运行程序后手动更新

## 常见问题

### Q: 特征提取失败怎么办？

A: 检查图像路径是否正确，确保所有图像都是有效的PNG文件。

### Q: 聚类结果不理想？

A: 可以尝试：
- 调整聚类算法的参数
- 使用不同的特征提取方法
- 对特征进行PCA降维
- 尝试组合特征

### Q: 如何修改聚类数量？

A: 在 `main.py` 中修改 `n_clusters` 变量。

## 作者

数据挖掘课程作业

## 日期

2025年

