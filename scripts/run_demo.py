"""
一键演示脚本

此脚本运行完整演示，包括:
1. 运行快速测试
2. 生成可视化预测结果
3. 运行测试并生成测试报告
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# 修改导入路径
from scripts.visualize_predictions import visualize_prediction
from scripts.generate_test_report import run_tests_with_report

def check_imagenet_classes_file():
    """检查并下载ImageNet类别文件（如果不存在）"""
    class_file = os.path.join(PROJECT_ROOT, 'data', 'imagenet_classes.txt')
    
    if not os.path.exists(class_file):
        print(f"未找到ImageNet类别文件，正在下载...")
        
        # 创建data目录（如果不存在）
        data_dir = os.path.join(PROJECT_ROOT, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # 提供1000个ImageNet类别名称
        url = "https://raw.githubusercontent.com/pytorch/tutorials/master/beginner_source/data/imagenet_classes.txt"
        
        try:
            import urllib.request
            urllib.request.urlretrieve(url, class_file)
            print(f"下载成功: {class_file}")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载ImageNet类别文件")

def create_sample_images():
    """创建用于测试的样例图像（如果不存在）"""
    from PIL import Image
    import numpy as np
    
    # 创建数据目录（如果不存在）
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义图像路径
    black_path = os.path.join(data_dir, 'black.jpg')
    white_path = os.path.join(data_dir, 'white.jpg')
    noise_path = os.path.join(data_dir, 'noise.jpg')
    dummy_path = os.path.join(data_dir, 'dummy.txt')
    
    # 创建黑色图像
    if not os.path.exists(black_path):
        Image.new('RGB', (224, 224), color='black').save(black_path)
        print(f"创建黑色测试图像: {black_path}")
    
    # 创建白色图像
    if not os.path.exists(white_path):
        Image.new('RGB', (224, 224), color='white').save(white_path)
        print(f"创建白色测试图像: {white_path}")
    
    # 创建噪声图像
    if not os.path.exists(noise_path):
        noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(noise).save(noise_path)
        print(f"创建噪声测试图像: {noise_path}")
    
    # 创建dummy文本文件用于测试非图像文件处理
    if not os.path.exists(dummy_path):
        with open(dummy_path, 'w') as f:
            f.write("This is a dummy text file for testing error handling.")
        print(f"创建伪图像文件: {dummy_path}")

def check_cat_image():
    """检查cat.jpg是否存在，如果不存在则提供下载建议"""
    cat_path = os.path.join(PROJECT_ROOT, 'data', 'cat.jpg')
    
    if not os.path.exists(cat_path):
        print("警告: 未找到cat.jpg测试图像。")
        print("请放置一个名为cat.jpg的图像到data目录。")
        print("或从网络下载一个示例:")
        print("例如: https://github.com/pytorch/examples/raw/main/mnist/cat.jpg")
        
        try:
            # 尝试下载猫的图像
            url = "https://github.com/pytorch/examples/raw/main/mnist/cat.jpg"
            import urllib.request
            print(f"尝试下载示例猫图像...")
            urllib.request.urlretrieve(url, cat_path)
            print(f"下载成功: {cat_path}")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载cat.jpg或使用其他测试图像")

def create_results_directory():
    """创建结果目录"""
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    return results_dir, reports_dir

def run_quick_test():
    """运行快速测试脚本"""
    print("\n" + "="*80)
    print("运行快速测试...")
    print("="*80)
    
    # 运行quick_test.py
    quick_test_path = os.path.join(PROJECT_ROOT, 'quick_test.py')
    
    # 使用subprocess运行脚本
    result = subprocess.run([sys.executable, quick_test_path], 
                           cwd=PROJECT_ROOT,
                           capture_output=True, 
                           text=True)
    
    # 打印输出
    print(result.stdout)
    
    if result.stderr:
        print("错误信息:")
        print(result.stderr)

def run_visualization():
    """运行可视化脚本"""
    print("\n" + "="*80)
    print("生成预测可视化...")
    print("="*80)
    
    # 检查测试图像
    cat_path = os.path.join(PROJECT_ROOT, 'data', 'cat.jpg')
    if not os.path.exists(cat_path):
        print(f"未找到测试图像: {cat_path}")
        return None
    
    # 运行可视化
    output_path = visualize_prediction(cat_path)
    return output_path

def run_tests():
    """运行测试并生成报告"""
    print("\n" + "="*80)
    print("运行测试并生成报告...")
    print("="*80)
    
    # 运行测试并生成报告
    html_path, txt_path = run_tests_with_report()
    return html_path, txt_path

def main():
    """主函数，运行完整演示"""
    print("\n" + "="*80)
    print(" AI模型测试框架演示 ".center(80, "="))
    print("="*80 + "\n")
    
    # 准备目录和文件
    print("准备测试环境...")
    create_results_directory()
    check_imagenet_classes_file()
    create_sample_images()
    check_cat_image()
    
    # 运行快速测试
    run_quick_test()
    
    # 生成可视化
    output_image = run_visualization()
    
    # 运行测试并生成报告
    html_report, txt_report = run_tests()
    
    # 打印总结
    print("\n" + "="*80)
    print(" 演示结果摘要 ".center(80, "="))
    print("="*80)
    
    if output_image and os.path.exists(output_image):
        print(f"✓ 可视化预测结果: {output_image}")
    else:
        print("✗ 可视化预测结果生成失败")
    
    if html_report and os.path.exists(html_report):
        print(f"✓ HTML测试报告: {html_report}")
    else:
        print("✗ HTML测试报告生成失败")
    
    if txt_report and os.path.exists(txt_report):
        print(f"✓ 文本测试报告: {txt_report}")
    else:
        print("✗ 文本测试报告生成失败")
    
    print("\n演示完成！这些文件可以用于GitHub项目展示。")

if __name__ == "__main__":
    main() 