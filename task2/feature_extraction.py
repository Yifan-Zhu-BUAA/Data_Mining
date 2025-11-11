"""特征提取模块

该模块负责使用预训练的CNN模型提取图像特征。
"""

import torch
import torch.nn as nn
from torchvision import models

def get_feature_extractor(pretrained=True, model_name='resnet18'):
    """获取特征提取器
    
    Args:
        pretrained (bool): 是否使用预训练权重
        model_name (str): 模型名称 ('resnet18', 'resnet34', 'efficientnet_b0'等)
        
    Returns:
        nn.Module: 特征提取器模型
    """
    if model_name.startswith('resnet'):
        # 使用ResNet系列模型
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的ResNet模型: {model_name}")
        
        # 移除最后的全连接层
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_dim = model.fc.in_features
    
    elif model_name.startswith('efficientnet'):
        # 使用EfficientNet系列模型
        if model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的EfficientNet模型: {model_name}")
        
        # 移除最后的分类层
        feature_extractor = nn.Sequential(*list(model.children())[:-2])
        feature_dim = model.classifier[1].in_features
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    # 设置为评估模式
    feature_extractor.eval()
    
    # 冻结参数
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    return feature_extractor, feature_dim

class FeatureExtractorWithProjection(nn.Module):
    """带投影层的特征提取器
    
    将预训练CNN提取的特征投影到低维空间
    """
    
    def __init__(self, base_extractor, input_dim, hidden_dim=512, output_dim=256):
        """初始化
        
        Args:
            base_extractor (nn.Module): 基础特征提取器
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出特征维度
        """
        super(FeatureExtractorWithProjection, self).__init__()
        self.base_extractor = base_extractor
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入图像 tensor，形状为 [batch_size, 3, 224, 224]
            
        Returns:
            torch.Tensor: 提取的特征，形状为 [batch_size, output_dim]
        """
        # 提取基础特征
        with torch.no_grad():
            features = self.base_extractor(x)
            # 展平特征
            features = features.view(features.size(0), -1)
        
        # 投影到低维空间
        projected_features = self.projection(features)
        
        return projected_features

def extract_features(model, dataloader, device):
    """从数据加载器中提取特征
    
    Args:
        model (nn.Module): 特征提取模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 运行设备
        
    Returns:
        tuple: (features, labels, paths) - 特征、标签和图像路径
    """
    model.eval()
    features_list = []
    labels_list = []
    paths_list = []
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            
            # 提取特征
            features = model(images)
            
            # 保存结果
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            paths_list.extend(paths)
    
    # 合并结果
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    return features, labels, paths