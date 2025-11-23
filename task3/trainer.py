import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
from config import config

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model):
        self.model = model.to(config.DEVICE)
        self.criterion = nn.MSELoss()
        # 添加权重衰减（L2正则化）
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def create_dataloaders(self, X_train, X_test, y_train, y_test):
        """创建数据加载器"""
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(config.DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).to(config.DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(config.DEVICE)
        y_test_tensor = torch.FloatTensor(y_test).to(config.DEVICE)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
        )
        
        return train_loader, test_loader
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch_X, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            # 前向传播
            if config.USE_ATTENTION:
                predictions, _ = self.model(batch_X)
            else:
                predictions = self.model(batch_X)
            
            # 计算损失
            loss = self.criterion(predictions.squeeze(), batch_y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 收集预测值和真实值用于计算准确率
            all_predictions.append(predictions.squeeze().detach().cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
        
        # 计算训练集准确率
        train_pred = np.concatenate(all_predictions)
        train_true = np.concatenate(all_targets)
        train_acc = self._calculate_accuracy(train_true, train_pred)
        
        return total_loss / len(train_loader), train_acc
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                if config.USE_ATTENTION:
                    predictions, _ = self.model(batch_X)
                else:
                    predictions = self.model(batch_X)
                
                loss = self.criterion(predictions.squeeze(), batch_y)
                total_loss += loss.item()
                
                # 收集预测值和真实值用于计算准确率
                all_predictions.append(predictions.squeeze().cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        # 计算验证集准确率
        val_pred = np.concatenate(all_predictions)
        val_true = np.concatenate(all_targets)
        val_acc = self._calculate_accuracy(val_true, val_pred)
        
        return total_loss / len(val_loader), val_acc
    
    def _calculate_accuracy(self, y_true, y_pred, error_thresholds=[5, 10, 20]):
        """计算准确率（基于相对误差阈值）"""
        relative_errors = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
        accuracies = {}
        for threshold in error_thresholds:
            accuracy = np.mean(relative_errors < threshold) * 100
            accuracies[threshold] = accuracy
        return accuracies
    
    def train(self, train_loader, val_loader):
        """完整训练过程"""
        early_stop_counter = 0
        
        for epoch in range(config.NUM_EPOCHS):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                early_stop_counter = 0
                self.save_model()
            else:
                early_stop_counter += 1
            
            # 打印进度
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}/{config.NUM_EPOCHS} | '
                      f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | '
                      f'Best Val Loss: {self.best_loss:.6f}')
                print(f'           | Train Acc(5%/10%/20%): {train_acc[5]:.2f}%/{train_acc[10]:.2f}%/{train_acc[20]:.2f}% | '
                      f'Val Acc(5%/10%/20%): {val_acc[5]:.2f}%/{val_acc[10]:.2f}%/{val_acc[20]:.2f}%')
            
            # 早停
            if early_stop_counter >= config.PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break
        
        print(f'\n训练完成，最佳验证损失: {self.best_loss:.6f}')
        # 打印最终准确率（使用最后一轮的准确率）
        _, final_train_acc = self.train_epoch(train_loader)
        _, final_val_acc = self.validate(val_loader)
        print(f'最终训练集准确率(5%/10%/20%): {final_train_acc[5]:.2f}%/{final_train_acc[10]:.2f}%/{final_train_acc[20]:.2f}%')
        print(f'最终验证集准确率(5%/10%/20%): {final_val_acc[5]:.2f}%/{final_val_acc[10]:.2f}%/{final_val_acc[20]:.2f}%')
    
    def save_model(self):
        """保存模型"""
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, config.MODEL_SAVE_PATH)
    
    def load_model(self, model):
        """加载模型"""
        checkpoint = torch.load(config.MODEL_SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f'模型已加载，最佳验证损失: {self.best_loss:.6f}')