"""
生成测试图像脚本

此脚本用于生成用于测试的各种图像：
- 纯黑图像
- 纯白图像
- 随机噪声图像
- 空文本文件（用于非图像文件测试）
"""

import os
import numpy as np
from PIL import Image

def main():
    # 确保data目录存在
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 设置图像尺寸
    width, height = 300, 300
    
    # 生成纯黑图像
    print("生成纯黑图像...")
    black_image = Image.new('RGB', (width, height), color='black')
    black_image.save(os.path.join(data_dir, 'black.jpg'))
    
    # 生成纯白图像
    print("生成纯白图像...")
    white_image = Image.new('RGB', (width, height), color='white')
    white_image.save(os.path.join(data_dir, 'white.jpg'))
    
    # 生成随机噪声图像
    print("生成随机噪声图像...")
    noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    noise_image = Image.fromarray(noise)
    noise_image.save(os.path.join(data_dir, 'noise.jpg'))
    
    # 创建一个空的文本文件
    print("创建非图像文件...")
    with open(os.path.join(data_dir, 'dummy.txt'), 'w') as f:
        f.write("这不是一个图像文件，用于测试非图像文件的处理逻辑。")
    
    print("所有测试图像和文件已创建完成！")
    print(f"文件位置：{data_dir}")
    print("生成的文件：")
    print("- black.jpg (纯黑图像)")
    print("- white.jpg (纯白图像)")
    print("- noise.jpg (随机噪声图像)")
    print("- dummy.txt (非图像文件)")

if __name__ == "__main__":
    main() 