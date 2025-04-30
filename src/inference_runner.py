"""
图像分类模型加载和推理模块

此模块提供了ImageClassifier类，用于：
1. 加载预训练的ResNet-18模型
2. 加载和预处理图像
3. 运行模型推理
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import datetime

class ImageClassifier:
    """
    图像分类器类
    
    该类封装了图像分类模型的加载、图像预处理和推理功能。
    默认使用在ImageNet上预训练的ResNet-18模型。
    """
    
    def __init__(self, model_name='resnet18'):
        """
        初始化图像分类器，加载预训练模型
        
        参数:
            model_name (str): 模型名称，默认为'resnet18'
                              目前支持的选项: 'resnet18'
        
        异常:
            ValueError: 当提供的模型名称不受支持时抛出
        """
        # 检查模型名称是否支持
        supported_models = ['resnet18']
        if model_name not in supported_models:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {supported_models}")
        
        # 根据模型名称加载预训练模型
        if model_name == 'resnet18':
            # 加载预训练的ResNet-18模型
            # pretrained=True表示使用在ImageNet上预训练的权重
            self.model = models.resnet18(pretrained=True)
        
        # 将模型设置为评估模式，关闭Dropout等训练特有的层
        self.model.eval()
        
        # 定义图像预处理流程
        # 这些预处理步骤与模型训练时使用的步骤需要一致
        self.preprocess = transforms.Compose([
            transforms.Resize(256),              # 将图像缩放到256x256
            transforms.CenterCrop(224),          # 中心裁剪到224x224
            transforms.ToTensor(),               # 转换为张量，并将像素值从[0,255]转换到[0,1]
            transforms.Normalize(                # 标准化，使用ImageNet数据集的均值和标准差
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_and_preprocess_image(self, image_path):
        """
        加载图像并应用预处理
        
        参数:
            image_path (str): 图像文件的路径
            
        返回:
            torch.Tensor: 预处理后的图像张量，形状为(1, 3, 224, 224)
            
        异常:
            FileNotFoundError: 当图像文件不存在时抛出
            各种PIL异常: 当图像文件格式不支持或损坏时抛出
        """
        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        try:
            # 打开图像并确保是RGB格式
            image = Image.open(image_path).convert('RGB')
            
            # 应用预处理流程
            input_tensor = self.preprocess(image)
            
            # 添加批次维度，将形状从(3, 224, 224)变为(1, 3, 224, 224)
            # 模型期望的输入是一个批次的图像
            input_batch = input_tensor.unsqueeze(0)
            
            return input_batch
        except Exception as e:
            # 重新抛出异常，添加更多上下文信息
            raise Exception(f"处理图像'{image_path}'时发生错误: {str(e)}")
    
    def run_inference(self, input_tensor):
        """
        运行模型推理
        
        参数:
            input_tensor (torch.Tensor): 输入图像张量，形状为(B, 3, 224, 224)，
                                         其中B是批次大小，通常为1
            
        返回:
            torch.Tensor: 模型输出，形状为(B, 1000)，表示ImageNet 1000个类别的预测分数
            
        异常:
            RuntimeError: 当输入张量形状不正确或推理过程中出现错误时抛出
        """
        # 检查输入张量的格式
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(f"输入必须是PyTorch张量，而不是{type(input_tensor)}")
        
        # 检查输入张量的形状
        if len(input_tensor.shape) != 4 or input_tensor.shape[1] != 3:
            raise ValueError(f"输入张量形状错误: {input_tensor.shape}，预期形状应为(B, 3, H, W)")
        
        try:
            # 使用torch.no_grad()包裹推理代码，告诉PyTorch不需要计算梯度
            # 这可以减少内存使用并加速推理
            with torch.no_grad():
                output = self.model(input_tensor)
            
            return output
        except Exception as e:
            # 重新抛出异常，添加更多上下文信息
            raise RuntimeError(f"模型推理过程中发生错误: {str(e)}")
    
    def get_top_predictions(self, output, top_k=5, class_names=None):
        """
        获取前K个预测结果
        
        参数:
            output (torch.Tensor): 模型输出，形状为(1, 1000)
            top_k (int): 返回的预测数量，默认为5
            class_names (list): 类别名称列表，默认为None
            
        返回:
            list: 包含(类别索引, 概率, 类别名称)元组的列表
        """
        # 应用softmax将输出转换为概率
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # 获取前K个预测
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # 转换为Python列表
        top_probs = top_probs.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()
        
        # 确保top_indices和top_probs是列表
        if not isinstance(top_indices, list):
            top_indices = [top_indices]
            top_probs = [top_probs]
        
        # 创建结果列表
        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            # 如果提供了类别名称，则包含在结果中
            name = class_names[idx] if class_names else None
            results.append((idx, prob * 100, name))
        
        return results
        
    def get_timestamp(self):
        """
        获取当前时间戳字符串
        
        返回:
            str: 格式化的时间戳字符串
        """
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 