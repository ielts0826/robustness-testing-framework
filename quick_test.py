"""
快速测试脚本

此脚本用于演示如何使用ImageClassifier类加载图像并进行推理。
"""

import torch
import os
from src.inference_runner import ImageClassifier

# 类别名称映射文件路径
# 这里假设已经有了ImageNet类别映射文件，如果没有，程序会跳过名称展示
IMAGENET_CLASSES_FILE = 'data/imagenet_classes.txt'

def load_class_names():
    """加载ImageNet类别名称，如果文件不存在则返回None"""
    if os.path.exists(IMAGENET_CLASSES_FILE):
        with open(IMAGENET_CLASSES_FILE, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return None

def main():
    try:
        # 实例化图像分类器
        print("正在加载ResNet-18模型...")
        classifier = ImageClassifier(model_name='resnet18')
        print("模型加载完成")
        
        # 加载ImageNet类别名称（如果有）
        class_names = load_class_names()
        
        # 测试图像路径（需要确保数据文件夹中有这个图像）
        image_path = 'data/cat.jpg'
        
        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"错误：测试图像不存在于路径 {image_path}")
            print("请在data目录中放入一张名为cat.jpg的图片，或修改本脚本中的image_path变量")
            return
        
        print(f"加载并预处理图像：{image_path}")
        
        # 加载和预处理图像
        input_tensor = classifier.load_and_preprocess_image(image_path)
        
        # 运行推理
        print("执行推理...")
        result = classifier.run_inference(input_tensor)
        
        # 打印结果形状
        print(f"推理结果形状: {result.shape}")
        print("预期形状: torch.Size([1, 1000]) (1张图片的1000个类别预测分数)")
        
        # 获取并展示前5个预测结果
        print("\n前5个预测结果:")
        predictions = classifier.get_top_predictions(result, top_k=5, class_names=class_names)
        
        for i, (idx, prob, name) in enumerate(predictions):
            if name:
                print(f"{i+1}. {name} - {prob:.2f}%")
            else:
                print(f"{i+1}. 类别 {idx} - {prob:.2f}%")
                
        print("\n测试完成")
    except Exception as e:
        print(f"错误：{e}")

if __name__ == "__main__":
    main() 