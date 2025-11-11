"""自编码器模型模块

该模块实现基于自编码器的异常检测模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """基本自编码器模型"""
    
    def __init__(self, input_channels=3, latent_dim=64):
        """初始化自编码器
        
        Args:
            input_channels (int): 输入图像的通道数
            latent_dim (int): 潜在空间维度
        """
        super(Autoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 第四个卷积块
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # 全连接层压缩到潜在空间
        self.fc_encoder = nn.Sequential(
            nn.Linear(256 * 14 * 14, 1024),  # 假设输入图像大小为224x224
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        
        # 全连接层从潜在空间还原
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 14 * 14)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            # 第一个反卷积块
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 第二个反卷积块
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 第三个反卷积块
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 第四个反卷积块
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 使用Sigmoid确保输出在[0,1]范围内
        )
    
    def encode(self, x):
        """编码过程
        
        Args:
            x (torch.Tensor): 输入图像
            
        Returns:
            torch.Tensor: 潜在空间表示
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_encoder(x)
        return x
    
    def decode(self, x):
        """解码过程
        
        Args:
            x (torch.Tensor): 潜在空间表示
            
        Returns:
            torch.Tensor: 重建的图像
        """
        x = self.fc_decoder(x)
        x = x.view(x.size(0), 256, 14, 14)  # 重塑为特征图形状
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入图像
            
        Returns:
            torch.Tensor: 重建的图像
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

class VariationalAutoencoder(nn.Module):
    """变分自编码器模型"""
    
    def __init__(self, input_channels=3, latent_dim=64):
        """初始化变分自编码器
        
        Args:
            input_channels (int): 输入图像的通道数
            latent_dim (int): 潜在空间维度
        """
        super(VariationalAutoencoder, self).__init__()
        
        # 编码器结构与基本自编码器相同
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # 用于生成均值和方差的全连接层
        self.fc_mu = nn.Linear(256 * 14 * 14, latent_dim)
        self.fc_logvar = nn.Linear(256 * 14 * 14, latent_dim)
        
        # 解码器的全连接层
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256 * 14 * 14)
        )
        
        # 解码器结构与基本自编码器相同
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧
        
        Args:
            mu (torch.Tensor): 均值
            logvar (torch.Tensor): 对数方差
            
        Returns:
            torch.Tensor: 从正态分布采样的潜在变量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        """编码过程
        
        Args:
            x (torch.Tensor): 输入图像
            
        Returns:
            tuple: (mu, logvar) - 均值和对数方差
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        """解码过程
        
        Args:
            z (torch.Tensor): 潜在变量
            
        Returns:
            torch.Tensor: 重建的图像
        """
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 256, 14, 14)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (torch.Tensor): 输入图像
            
        Returns:
            tuple: (reconstructed, mu, logvar) - 重建图像、均值和对数方差
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """变分自编码器的损失函数
    
    Args:
        recon_x (torch.Tensor): 重建的图像
        x (torch.Tensor): 原始输入图像
        mu (torch.Tensor): 均值
        logvar (torch.Tensor): 对数方差
        beta (float): KL散度的权重
        
    Returns:
        torch.Tensor: 总损失
    """
    # 重建损失
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss