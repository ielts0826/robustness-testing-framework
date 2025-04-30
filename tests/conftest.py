"""
pytest配置文件，提供共享fixtures
"""

import pytest
import os
import torch
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference_runner import ImageClassifier

@pytest.fixture(scope="session")
def classifier():
    """
    提供一个ImageClassifier实例，整个测试会话期间共享
    
    这是一个共享的fixture，可以在多个测试中重用，避免重复创建分类器实例
    scope="session"表示在整个测试会话中只创建一次实例
    """
    return ImageClassifier()

@pytest.fixture
def test_image_path():
    """提供标准测试图像的路径"""
    image_path = os.path.join('data', 'cat.jpg')
    if not os.path.exists(image_path):
        pytest.skip(f"测试图片不存在: {image_path}")
    return image_path

@pytest.fixture
def processed_test_image(classifier, test_image_path):
    """提供预处理后的测试图像张量"""
    return classifier.load_and_preprocess_image(test_image_path)

@pytest.fixture(params=['data/black.jpg', 'data/white.jpg', 'data/noise.jpg'])
def edge_case_image_path(request):
    """提供边缘情况测试图像的路径"""
    image_path = request.param
    if not os.path.exists(image_path):
        pytest.skip(f"测试图片不存在: {image_path}")
    return image_path

@pytest.fixture
def temp_images(request):
    """
    创建临时测试图像并在测试后清理
    
    创建两种尺寸的图像用于测试：
    1. 非常小的图像 (16x16)
    2. 非常大的图像 (4000x3000)
    """
    temp_paths = []
    
    # 创建一个非常小的图像
    tiny_image = Image.new('RGB', (16, 16), color='blue')
    tiny_path = 'data/temp_tiny.jpg'
    tiny_image.save(tiny_path)
    temp_paths.append(tiny_path)
    
    # 创建一个非常大的图像
    large_image = Image.new('RGB', (4000, 3000), color='green')
    large_path = 'data/temp_large.jpg'
    large_image.save(large_path)
    temp_paths.append(large_path)
    
    # 返回包含临时文件路径的字典
    yield {
        'tiny': tiny_path,
        'large': large_path
    }
    
    # 清理临时文件
    for path in temp_paths:
        if os.path.exists(path):
            os.remove(path) 