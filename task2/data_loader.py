"""数据加载和预处理模块

该模块负责从数据集目录中加载图像数据，进行预处理，并创建数据加载器。
"""

import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class AnomalyDataset(Dataset):
    """异常检测数据集类"""
    
    def __init__(self, root_dir, category='hazelnut', split='train', transform=None):
        """初始化数据集
        
        Args:
            root_dir (str): 数据集根目录
            category (str): 类别名称 ('hazelnut' 或 'zipper')
            split (str): 数据集分割 ('train' 或 'test')
            transform (callable, optional): 数据转换函数
        """
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 加载标签文件
        labels_file = os.path.join(root_dir, 'image_anomaly_labels.json')
        with open(labels_file, 'r', encoding='utf-8') as f:
            self.all_labels = json.load(f)
        
        self._load_data()
    
    def _load_data(self):
        """加载数据路径和标签"""
        if self.split == 'train':
            # 训练集：包含good和bad子目录
            for label in ['good', 'bad']:
                dir_path = os.path.join(self.root_dir, self.category, 'train', label)
                if os.path.exists(dir_path):
                    for filename in os.listdir(dir_path):
                        if filename.endswith('.png'):
                            img_path = os.path.join(dir_path, filename)
                            self.image_paths.append(img_path)
                            self.labels.append(0 if label == 'good' else 1)
        else:  # test
            # 测试集：使用标签文件中的标注
            dir_path = os.path.join(self.root_dir, self.category, 'test')
            for filename in os.listdir(dir_path):
                if filename.endswith('.png'):
                    img_key = f"{self.category}/test/{filename}"
                    if img_key in self.all_labels:
                        img_path = os.path.join(dir_path, filename)
                        self.image_paths.append(img_path)
                        self.labels.append(0 if self.all_labels[img_key]['label'] == 'good' else 1)
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (image, label, path)
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

def get_transforms(split='train'):
    """获取数据转换
    
    Args:
        split (str): 数据集分割 ('train' 或 'test')
        
    Returns:
        transforms.Compose: 数据转换组合
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(root_dir, category='hazelnut', batch_size=16, num_workers=4):
    """获取数据加载器
    
    Args:
        root_dir (str): 数据集根目录
        category (str): 类别名称
        batch_size (int): 批次大小
        num_workers (int): 工作线程数
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # 创建训练集和测试集
    train_dataset = AnomalyDataset(
        root_dir=root_dir,
        category=category,
        split='train',
        transform=get_transforms('train')
    )
    
    test_dataset = AnomalyDataset(
        root_dir=root_dir,
        category=category,
        split='test',
        transform=get_transforms('test')
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_normal_only_dataloader(root_dir, category='hazelnut', batch_size=16, num_workers=4):
    """获取只包含正常样本的训练集加载器
    
    Args:
        root_dir (str): 数据集根目录
        category (str): 类别名称
        batch_size (int): 批次大小
        num_workers (int): 工作线程数
        
    Returns:
        DataLoader: 只包含正常样本的数据加载器
    """
    # 创建只包含正常样本的训练集
    dataset = AnomalyDataset(
        root_dir=root_dir,
        category=category,
        split='train',
        transform=get_transforms('train')
    )
    
    # 过滤出正常样本
    normal_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
    
    from torch.utils.data import Subset
    normal_dataset = Subset(dataset, normal_indices)
    
    # 创建数据加载器
    normal_loader = DataLoader(
        normal_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return normal_loader