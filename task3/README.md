# 时间序列预测任务 - 室外温度预测

## 项目概述

本项目实现基于LSTM+注意力机制的时间序列预测模型，用于预测室外温度（OT）的变化。

## 项目结构

```
task3/
├── data/                    # 数据集目录
│   └── weather.csv         # 原始气象数据
├── models/                  # 模型保存目录
│   ├── best_model.pth     # 最佳模型权重
│   └── scaler.pkl         # 特征标准化器
├── results/                 # 结果输出目录
│   ├── evaluation_metrics.json  # 评估指标
│   ├── predictions.csv     # 预测结果
│   └── report.txt          # 实验报告
├── visualizations/          # 可视化文件目录
│   ├── training_history.png    # 训练历史曲线
│   ├── predictions.png     # 预测结果对比
│   ├── prediction_scatter.png  # 散点图
│   ├── error_distribution.png  # 误差分布
│   └── residuals.png       # 残差图
├── config.py               # 配置文件
├── data_loader.py          # 数据加载模块
├── preprocessing.py        # 数据预处理模块
├── model.py               # 模型定义模块
├── trainer.py             # 模型训练模块
├── evaluator.py           # 评估指标模块
├── visualization.py       # 可视化模块
├── main.py               # 主程序入口
├── requirements.txt       # 依赖包列表
└── README.md             # 项目说明
```

## 数据集说明

- **数据来源**: Weather数据集 - 德国某气象站半年内的气象数据
- **数据量**: 26200个数据点
- **时间间隔**: 每10分钟记录一次
- **特征维度**: 20个气象指标
- **目标变量**: 室外温度（OT）

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据集准备

确保数据集文件位于 `task3/data/weather.csv`

## 使用方法

### 运行主程序

```bash
cd task3
python main.py
```

程序将自动执行以下步骤：

1. **加载数据**: 从CSV文件读取气象数据
2. **数据预处理**: 特征标准化和滑动窗口切分
3. **创建模型**: 构建LSTM+注意力机制模型
4. **训练模型**: 在训练集上训练模型
5. **评估模型**: 在测试集上评估模型性能
6. **生成可视化**: 创建各种可视化图表
7. **保存结果**: 保存评估指标和预测结果

## 模型架构

- **输入**: 过去4小时（24个时间点）× 20个气象特征（已优化：从2小时增加到4小时）
- **模型**: LSTM + 注意力机制 + 深层全连接网络
- **输出**: 下一时间点的室外温度（OT）

### 模型参数

- LSTM隐藏层维度: 256（已优化：从128增加到256）
- LSTM层数: 3
- 全连接层隐藏维度: 128
- Dropout: 0.3
- 注意力机制: 启用
- 批次大小: 32
- 学习率: 0.0003（已优化：从0.0005降低到0.0003）
- 权重衰减（L2正则化）: 1e-5（新增）

## 评估指标

模型使用以下指标进行评估：

- **MSE** (均方误差)
- **MAE** (平均绝对误差)
- **RMSE** (均方根误差)
- **R²** (决定系数)
- **MAPE** (平均绝对百分比误差)
- **准确率** (基于相对误差阈值: 5%, 10%, 20%)

## 输出结果

### 1. 模型文件
- `models/best_model.pth`: 最佳模型权重
- `models/scaler.pkl`: 特征标准化器

### 2. 评估结果
- `results/evaluation_metrics.json`: 详细的评估指标
- `results/predictions.csv`: 测试集预测结果
- `results/report.txt`: 完整的实验报告

### 3. 可视化图表
- `visualizations/training_history.png`: 训练损失曲线
- `visualizations/predictions.png`: 预测结果对比
- `visualizations/prediction_scatter.png`: 预测值vs真实值散点图
- `visualizations/error_distribution.png`: 误差分布
- `visualizations/residuals.png`: 残差图

## 参数配置

可以在 `config.py` 中修改以下参数：

```python
# 数据参数
WINDOW_SIZE = 24      # 滑动窗口大小（已优化：从12增加到24，即4小时）
TEST_SIZE = 0.2       # 测试集比例

# 模型参数
HIDDEN_DIM = 256      # LSTM隐藏层维度（已优化：从128增加到256）
NUM_LAYERS = 3        # LSTM层数
FC_HIDDEN_DIM = 128   # 全连接层隐藏维度（新增）
DROPOUT = 0.3         # Dropout比率
USE_ATTENTION = True  # 是否使用注意力机制

# 训练参数
BATCH_SIZE = 32       # 批次大小
LEARNING_RATE = 0.0003 # 学习率（已优化：从0.0005降低到0.0003）
NUM_EPOCHS = 300      # 最大训练轮数（已优化：从200增加到300）
PATIENCE = 20         # 早停耐心值（已优化：从15增加到20）
WEIGHT_DECAY = 1e-5   # L2正则化权重衰减（新增）
```

## 注意事项

1. **Python版本**: 建议使用 Python 3.7-3.10 版本
2. **GPU支持**: 如果有NVIDIA GPU和CUDA支持，程序会自动使用GPU加速训练
   - CPU训练也是可行的，但速度会较慢
   - 如果没有GPU，程序会自动使用CPU
3. **内存要求**: 处理26200个数据点需要约2-4GB内存
4. **训练时间**: 
   - CPU: 约90-180分钟（由于模型复杂度和窗口大小增加）
   - GPU: 约30-60分钟（由于模型复杂度和窗口大小增加）
   - 取决于硬件配置和训练轮数
5. **时间序列特性**: 数据集按时间顺序划分，确保训练集时间早于测试集
6. **虚拟环境**: 强烈建议使用虚拟环境，避免与系统Python环境冲突

## 常见问题

### Q: 训练损失不下降怎么办？

A: 可以尝试：
- 调整学习率
- 增加LSTM层数或隐藏层维度
- 调整批次大小
- 检查数据预处理是否正确

### Q: 如何修改预测窗口大小？

A: 在 `config.py` 中修改 `WINDOW_SIZE` 参数

### Q: 如何保存和加载模型？

A: 模型会自动保存到 `models/best_model.pth`，可以手动加载用于预测

## 作者

数据挖掘课程作业

## 日期

2025年11.15

