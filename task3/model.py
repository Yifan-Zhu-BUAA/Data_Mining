import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class AttentionLayer(nn.Module):
    """注意力机制层"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        attention_weights = F.softmax(self.attention(lstm_output).squeeze(-1), dim=1)
        # attention_weights shape: (batch_size, seq_len)
        
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        # context_vector shape: (batch_size, hidden_dim)
        
        return context_vector, attention_weights

class TimeSeriesPredictor(nn.Module):
    """时间序列预测模型（LSTM + Attention）"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, fc_hidden_dim=None, use_attention=True):
        super(TimeSeriesPredictor, self).__init__()
        
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力层
        if use_attention:
            self.attention = AttentionLayer(hidden_dim)
        
        # 输出层（使用更深的网络结构）
        if fc_hidden_dim is None:
            fc_hidden_dim = hidden_dim // 2
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, 1)  # 预测单个值（OT温度）
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # LSTM前向传播
        lstm_output, (hn, cn) = self.lstm(x)
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        
        if self.use_attention:
            # 使用注意力机制
            context_vector, attention_weights = self.attention(lstm_output)
            output = self.fc(context_vector)
            return output, attention_weights
        else:
            # 使用最后一个时间步的输出
            last_output = lstm_output[:, -1, :]  # 取最后一个时间步
            output = self.fc(last_output)
            return output
    
    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))

# 创建模型实例
def create_model():
    return TimeSeriesPredictor(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        fc_hidden_dim=config.FC_HIDDEN_DIM,
        use_attention=config.USE_ATTENTION
    )