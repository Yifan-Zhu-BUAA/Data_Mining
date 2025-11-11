"""
图像特征提取模块
支持多种特征提取方法：CNN特征（ResNet50）、HOG、颜色直方图、SIFT等
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from skimage.feature import hog
import os
import json


class ImageFeatureExtractor:
    """图像特征提取器"""
    
    def __init__(self, method='resnet', device='cpu'):
        """
        初始化特征提取器
        
        Args:
            method: 特征提取方法 ('resnet', 'hog', 'histogram', 'sift', 'combined')
            device: 计算设备 ('cpu' or 'cuda')
        """
        self.method = method
        self.device = device
        
        if method == 'resnet':
            self._init_resnet()
        elif method == 'combined':
            self._init_resnet()
    
    def _init_resnet(self):
        """初始化ResNet50模型用于特征提取"""
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # 移除最后的全连接层，只保留特征提取部分
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet.eval()
        self.resnet.to(self.device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_resnet_features(self, image_path):
        """
        使用ResNet50提取图像特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            numpy array: 特征向量 (2048维)
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.resnet(image_tensor)
                features = features.squeeze().cpu().numpy()
                # 展平特征向量
                features = features.flatten()
            
            return features
        except Exception as e:
            print(f"Error extracting ResNet features from {image_path}: {e}")
            return None
    
    def extract_hog_features(self, image_path):
        """
        提取HOG（方向梯度直方图）特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            numpy array: HOG特征向量
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.array(Image.open(image_path).convert('L'))
            
            # 调整图像大小以保持一致性
            image = cv2.resize(image, (224, 224))
            
            # 提取HOG特征
            features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=True,
                             feature_vector=True)
            
            return features
        except Exception as e:
            print(f"Error extracting HOG features from {image_path}: {e}")
            return None
    
    def extract_histogram_features(self, image_path):
        """
        提取颜色直方图特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            numpy array: 颜色直方图特征向量
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                image = np.array(Image.open(image_path).convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 调整图像大小
            image = cv2.resize(image, (224, 224))
            
            # 计算每个通道的直方图
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            # 合并特征
            features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
            
            # 归一化
            features = features / (features.sum() + 1e-7)
            
            return features
        except Exception as e:
            print(f"Error extracting histogram features from {image_path}: {e}")
            return None
    
    def extract_sift_features(self, image_path):
        """
        提取SIFT特征（使用特征袋方法）
        
        Args:
            image_path: 图像路径
            
        Returns:
            numpy array: SIFT特征向量
        """
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.array(Image.open(image_path).convert('L'))
            
            image = cv2.resize(image, (224, 224))
            
            # 创建SIFT检测器
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(image, None)
            
            if descriptors is None or len(descriptors) == 0:
                # 如果没有检测到特征点，返回零向量
                return np.zeros(128)
            
            # 使用平均描述符作为特征
            features = np.mean(descriptors, axis=0)
            
            return features
        except Exception as e:
            print(f"Error extracting SIFT features from {image_path}: {e}")
            return None
    
    def extract_combined_features(self, image_path):
        """
        提取组合特征（ResNet + HOG + 颜色直方图）
        
        Args:
            image_path: 图像路径
            
        Returns:
            numpy array: 组合特征向量
        """
        features_list = []
        
        # ResNet特征
        resnet_feat = self.extract_resnet_features(image_path)
        if resnet_feat is not None:
            features_list.append(resnet_feat)
        
        # HOG特征
        hog_feat = self.extract_hog_features(image_path)
        if hog_feat is not None:
            features_list.append(hog_feat)
        
        # 颜色直方图特征
        hist_feat = self.extract_histogram_features(image_path)
        if hist_feat is not None:
            features_list.append(hist_feat)
        
        if len(features_list) == 0:
            return None
        
        # 合并所有特征
        combined_features = np.concatenate(features_list)
        
        return combined_features
    
    def extract_features(self, image_path):
        """
        根据指定方法提取特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            numpy array: 特征向量
        """
        if self.method == 'resnet':
            return self.extract_resnet_features(image_path)
        elif self.method == 'hog':
            return self.extract_hog_features(image_path)
        elif self.method == 'histogram':
            return self.extract_histogram_features(image_path)
        elif self.method == 'sift':
            return self.extract_sift_features(image_path)
        elif self.method == 'combined':
            return self.extract_combined_features(image_path)
        else:
            raise ValueError(f"Unknown feature extraction method: {self.method}")
    
    def extract_features_batch(self, image_paths, verbose=True):
        """
        批量提取特征
        
        Args:
            image_paths: 图像路径列表
            verbose: 是否显示进度
            
        Returns:
            numpy array: 特征矩阵 (n_samples, n_features)
        """
        features_list = []
        valid_paths = []
        
        for i, path in enumerate(image_paths):
            if verbose and (i + 1) % 50 == 0:
                print(f"Processing {i + 1}/{len(image_paths)} images...")
            
            features = self.extract_features(path)
            if features is not None:
                features_list.append(features)
                valid_paths.append(path)
        
        if len(features_list) == 0:
            return None, []
        
        features_matrix = np.array(features_list)
        
        if verbose:
            print(f"Extracted features shape: {features_matrix.shape}")
        
        return features_matrix, valid_paths


def load_images_from_directory(directory, labels_file=None):
    """
    从目录加载图像路径
    
    Args:
        directory: 图像目录路径
        labels_file: 标签文件路径（可选）
        
    Returns:
        tuple: (图像路径列表, 标签列表（如果有）)
    """
    image_paths = []
    labels = []
    
    # 获取所有PNG图像
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith('.png'):
            image_paths.append(os.path.join(directory, filename))
    
    # 如果有标签文件，加载标签
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels_dict = json.load(f)
        
        for path in image_paths:
            filename = os.path.basename(path)
            if filename in labels_dict:
                labels.append(labels_dict[filename])
            else:
                labels.append(None)
    
    return image_paths, labels


if __name__ == "__main__":
    # 测试代码
    dataset_dir = "../cluster/Cluster/dataset"
    labels_file = "../cluster/Cluster/cluster_labels.json"
    
    print("Loading images...")
    image_paths, labels = load_images_from_directory(dataset_dir, labels_file)
    print(f"Loaded {len(image_paths)} images")
    
    # 测试不同特征提取方法
    methods = ['resnet', 'hog', 'histogram']
    
    for method in methods:
        print(f"\nTesting {method} feature extraction...")
        extractor = ImageFeatureExtractor(method=method)
        
        # 只测试前10张图像
        test_paths = image_paths[:10]
        features, valid_paths = extractor.extract_features_batch(test_paths, verbose=False)
        
        if features is not None:
            print(f"{method} features shape: {features.shape}")

