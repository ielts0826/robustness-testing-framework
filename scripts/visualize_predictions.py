"""
Visualize predictions

This script loads images, runs model predictions, and generates visualizations with prediction results
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from pathlib import Path

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 修改导入路径，使用相对导入
from src.inference_runner import ImageClassifier

def load_class_names(file_path='data/imagenet_classes.txt'):
    """加载ImageNet类别名称"""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"警告: 类别名称文件不存在: {file_path}")
        return None

def visualize_prediction(image_path, output_dir='results', top_k=5):
    """
    加载图像，运行预测，并生成可视化结果
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        top_k: 显示的top-k预测结果数量
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)
    
    # 加载图像
    try:
        pil_image = Image.open(image_path)
    except Exception as e:
        print(f"无法加载图像 {image_path}: {e}")
        return None
    
    # 加载分类器模型
    classifier = ImageClassifier()
    
    # 加载类别名称
    class_names = load_class_names()
    
    # 预处理图像并进行推理
    input_tensor = classifier.load_and_preprocess_image(image_path)
    output = classifier.run_inference(input_tensor)
    
    # 获取top-k预测结果
    predictions = classifier.get_top_predictions(output, top_k=top_k, class_names=class_names)
    
    # 创建可视化图像
    plt.figure(figsize=(12, 6))
    
    # 左侧显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(pil_image)
    plt.title("Input Image")
    plt.axis('off')
    
    # 右侧显示预测结果
    plt.subplot(1, 2, 2)
    
    # 创建水平条形图
    labels = []
    probs = []
    
    for i, (idx, prob, name) in enumerate(predictions):
        if name:
            label = f"{name}"
        else:
            label = f"Class {idx}"
        labels.append(label)
        probs.append(prob / 100)  # 转为0-1区间概率
    
    # 翻转列表以使最高概率在顶部
    labels.reverse()
    probs.reverse()
    
    # 条形图
    bars = plt.barh(range(len(probs)), probs, color='skyblue')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Probability')
    plt.title('Prediction Results')
    
    # 添加概率值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{probs[i]:.1%}', va='center')
    
    # 保存图像
    timestamp = classifier.get_timestamp()
    output_path = os.path.join(output_dir, f"prediction_vis_{os.path.basename(image_path).split('.')[0]}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    
    print(f"Visualization saved to: {output_path}")
    return output_path

def main():
    # 默认使用cat.jpg
    image_path = 'data/cat.jpg'
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像不存在于 {image_path}")
        return
    
    # 运行可视化
    output_path = visualize_prediction(image_path)
    
    if output_path:
        print(f"可视化预测结果已生成于: {output_path}")

if __name__ == "__main__":
    main() 