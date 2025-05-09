"""
自动为原始图片和所有干扰图片生成可视化预测结果

此脚本会加载原始图片和所有生成的干扰图片，并为每个图片生成可视化预测结果
"""

import os
import sys
import glob
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入可视化预测脚本中的函数
from .visualize_predictions import visualize_prediction

def visualize_all_images():
    """加载并可视化原始图片和所有干扰图片的预测结果"""
    # 设置路径
    original_image_path = 'data/cat.jpg'
    perturbed_images_dir = 'data/test_images'
    output_dir = 'results/all_visualizations'
    
    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # 首先可视化原始图片
    print(f"\n处理原始图片: {original_image_path}")
    if os.path.exists(original_image_path):
        output_path = visualize_prediction(original_image_path, output_dir=output_dir)
        if output_path:
            print(f"原始图片可视化保存到: {output_path}")
    else:
        print(f"警告: 原始图片不存在 {original_image_path}")
    
    # 检查干扰图片目录是否存在
    if not os.path.exists(perturbed_images_dir):
        print(f"\n警告: 干扰图片目录不存在 {perturbed_images_dir}")
        print("请先运行 generate_test_images.py 创建干扰图片")
        return
    
    # 查找所有干扰图片
    perturbed_image_paths = glob.glob(os.path.join(perturbed_images_dir, '*.jpg'))
    perturbed_image_paths += glob.glob(os.path.join(perturbed_images_dir, '*.png'))
    
    if not perturbed_image_paths:
        print(f"\n警告: 未找到干扰图片。请先运行 generate_test_images.py")
        return
    
    # 为每个干扰图片生成可视化
    for img_path in perturbed_image_paths:
        print(f"\n处理干扰图片: {img_path}")
        output_path = visualize_prediction(img_path, output_dir=output_dir)
        if output_path:
            print(f"干扰图片可视化保存到: {output_path}")
    
    print(f"\n所有可视化结果已保存到 {output_dir} 目录")

def main():
    """主函数"""
    # 检查是否已生成干扰图片，如果没有则提示用户
    if not os.path.exists('data/test_images') or not os.listdir('data/test_images'):
        print("警告: 未检测到干扰图片。建议先运行 generate_test_images.py 生成干扰图片。")
        answer = input("是否继续? (y/n): ").strip().lower()
        if answer != 'y':
            print("已取消。请先运行 generate_test_images.py 生成干扰图片。")
            return
    
    # 执行可视化处理
    print("开始为原始图片和所有干扰图片生成可视化预测结果...")
    visualize_all_images()
    print("可视化处理完成！")

if __name__ == "__main__":
    main() 