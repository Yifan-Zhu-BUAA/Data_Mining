"""
配置文件
"""

import torch

# 模型超参数
class Config:
    # 数据路径
    DATA_PATH = "data/weather.csv"
    
    # 数据参数
    WINDOW_SIZE = 24  # 4小时窗口（24个10分钟间隔）- 增加历史信息
    PREDICTION_STEP = 1  # 预测下一步
    TEST_SIZE = 0.2  # 测试集比例
    RANDOM_STATE = 42
    
    # 特征列（20个气象指标）
    FEATURE_COLUMNS = [
        'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 
        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 
        'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m^2)', 
        'PAR (umol/m^2s)', 'max. PAR (umol/m^2s)', 'Tlog (degC)'
    ]
    TARGET_COLUMN = 'OT'  # 目标变量
    
    # 模型参数
    INPUT_DIM = len(FEATURE_COLUMNS)  # 20
    HIDDEN_DIM = 256  # 进一步增加隐藏层维度以提升模型容量
    NUM_LAYERS = 3  # LSTM层数
    DROPOUT = 0.3  # Dropout比率
    USE_ATTENTION = True
    FC_HIDDEN_DIM = 128  # 全连接层隐藏维度
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0003  # 进一步降低学习率以获得更稳定的训练
    NUM_EPOCHS = 300  # 增加最大训练轮数
    PATIENCE = 20  # 增加早停耐心值，允许更多训练轮次
    WEIGHT_DECAY = 1e-5  # L2正则化权重衰减
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 输出路径
    MODEL_SAVE_PATH = "models/best_model.pth"
    SCALER_SAVE_PATH = "models/scaler.pkl"
    RESULTS_DIR = "results"
    VISUALIZATION_DIR = "visualizations"

# 创建配置实例
config = Config()