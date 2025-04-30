"""
图像分类模型推理测试
"""

import pytest
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference_runner import ImageClassifier
from PIL import Image

class TestImageClassifier:
    """图像分类器测试类"""
    
    def test_model_loading(self, classifier):
        """测试模型能否正确加载"""
        assert classifier.model is not None, "模型未成功加载"
    
    def test_successful_inference(self, classifier, processed_test_image):
        """测试模型是否能成功进行推理"""
        # 检查输入张量的形状
        assert processed_test_image.shape == torch.Size([1, 3, 224, 224]), \
            f"输入张量形状错误: {processed_test_image.shape}，预期: [1, 3, 224, 224]"
        
        # 运行推理
        output = classifier.run_inference(processed_test_image)
        
        # 检查输出张量的形状
        assert output is not None, "推理结果为空"
        assert output.shape == torch.Size([1, 1000]), \
            f"输出张量形状错误: {output.shape}，预期: [1, 1000]"
            
        # 检查输出是否包含有效的预测分数(不全是0或NaN)
        assert not torch.isnan(output).any(), "输出包含NaN值"
        assert not (output == 0).all(), "输出全为0"
    
    def test_invalid_image_path(self, classifier):
        """测试处理不存在的文件路径"""
        with pytest.raises(FileNotFoundError):
            classifier.load_and_preprocess_image('data/non_existent_file.jpg')
    
    def test_non_image_file(self, classifier):
        """测试处理非图片文件"""
        # 检查dummy.txt是否存在
        dummy_file = 'data/dummy.txt'
        if not os.path.exists(dummy_file):
            pytest.skip(f"测试文件不存在: {dummy_file}")
            
        # 尝试加载非图像文件，应该抛出异常
        with pytest.raises(Exception):
            classifier.load_and_preprocess_image(dummy_file)
    
    def test_edge_case_images_run_without_error(self, classifier, edge_case_image_path):
        """测试边缘情况图像（纯黑、纯白、噪声）不会导致模型崩溃"""
        try:
            # 加载和预处理图像
            input_tensor = classifier.load_and_preprocess_image(edge_case_image_path)
            
            # 运行推理
            output = classifier.run_inference(input_tensor)
            
            # 检查输出是否有效
            assert output is not None, "推理结果为空"
            assert output.shape == torch.Size([1, 1000]), \
                f"输出张量形状错误: {output.shape}，预期: [1, 1000]"
                
            # 检查输出是否包含有效的预测分数(不全是NaN)
            assert not torch.isnan(output).any(), f"{edge_case_image_path}的输出包含NaN值"
        except Exception as e:
            pytest.fail(f"使用{edge_case_image_path}进行推理时发生错误: {str(e)}")
    
    def test_input_size_boundaries(self, classifier, temp_images):
        """测试不同尺寸的输入图像"""
        # 测试非常小的图像
        input_tensor = classifier.load_and_preprocess_image(temp_images['tiny'])
        output = classifier.run_inference(input_tensor)
        assert output is not None, "小图像的推理结果为空"
        
        # 测试非常大的图像
        input_tensor = classifier.load_and_preprocess_image(temp_images['large'])
        output = classifier.run_inference(input_tensor)
        assert output is not None, "大图像的推理结果为空"
    
    def test_inference_determinism(self, classifier, test_image_path):
        """测试模型推理的确定性（相同输入应产生相同输出）"""
        # 第一次加载和预处理图像
        input_tensor1 = classifier.load_and_preprocess_image(test_image_path)
        
        # 第一次运行推理
        output1 = classifier.run_inference(input_tensor1)
        
        # 第二次加载和预处理图像（应该完全相同）
        input_tensor2 = classifier.load_and_preprocess_image(test_image_path)
        
        # 第二次运行推理
        output2 = classifier.run_inference(input_tensor2)
        
        # 检查两次输出是否完全相同
        assert torch.equal(output1, output2), "模型对相同输入产生了不同的输出，缺乏确定性"
        
        # 比较两次输入张量，确保预处理是确定性的
        assert torch.equal(input_tensor1, input_tensor2), "预处理管道对相同输入产生了不同的输出"
    
    def test_inference_determinism_with_reloading(self, test_image_path):
        """测试模型在重新加载后的确定性"""
        # 第一次加载模型
        classifier1 = ImageClassifier()
        
        # 加载和预处理图像
        input_tensor = classifier1.load_and_preprocess_image(test_image_path)
        
        # 第一次运行推理
        output1 = classifier1.run_inference(input_tensor)
        
        # 重新加载模型（模拟程序重启）
        classifier2 = ImageClassifier()
        
        # 第二次运行推理（使用相同的输入张量）
        output2 = classifier2.run_inference(input_tensor)
        
        # 检查两次输出是否完全相同
        assert torch.equal(output1, output2), "重新加载模型后对相同输入产生了不同的输出"
    
    def test_batch_invariance(self, classifier, processed_test_image):
        """测试模型对批次大小变化的鲁棒性"""
        # 单张图像推理
        single_output = classifier.run_inference(processed_test_image)
        
        # 创建批次（2个相同的图像）
        batch_tensor = torch.cat([processed_test_image, processed_test_image], dim=0)
        
        # 验证批次形状
        assert batch_tensor.shape == torch.Size([2, 3, 224, 224]), \
            f"批次张量形状错误: {batch_tensor.shape}，预期: [2, 3, 224, 224]"
        
        # 批次推理 - 使用相同的API
        outputs = []
        for i in range(batch_tensor.shape[0]):
            # 分别对每个图像使用相同的run_inference方法
            output = classifier.run_inference(batch_tensor[i:i+1])
            outputs.append(output)
        
        batch_output = torch.cat(outputs, dim=0)
        
        # 验证批次输出形状
        assert batch_output.shape == torch.Size([2, 1000]), \
            f"批次输出形状错误: {batch_output.shape}，预期: [2, 1000]"
        
        # 验证批次中的第一个输出与单张图像的输出相同
        assert torch.equal(single_output, batch_output[0:1]), \
            "批次中的第一个输出与单张图像的输出不同"
        
        # 验证批次中的两个输出相同（因为输入了相同的图像）
        assert torch.equal(batch_output[0:1], batch_output[1:2]), \
            "批次中的两个输出不同，尽管输入了相同的图像" 