"""
一键式脚本：生成干扰图片并为所有图片创建可视化预测结果

此脚本会:
1. 生成各种干扰图片
2. 为原始图片和所有干扰图片生成可视化预测结果
"""

import os
import sys
import time
import importlib
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_dependencies():
    """检查所需的Python依赖项是否已安装"""
    dependencies = ['torch', 'torchvision', 'PIL', 'matplotlib', 'numpy']
    missing = []
    
    for module in dependencies:
        try:
            importlib.import_module(module)
        except ImportError:
            if module == 'PIL':
                # PIL通常是通过Pillow安装的
                try:
                    importlib.import_module('pillow')
                except ImportError:
                    missing.append('Pillow')
            else:
                missing.append(module)
    
    if missing:
        print("\n警告: 以下依赖项未安装:")
        for module in missing:
            print(f"  - {module}")
        
        print("\n请激活虚拟环境并安装所需依赖项:")
        print("1. 激活虚拟环境: venv\\Scripts\\activate.bat")
        print("2. 安装依赖项: pip install -r requirements.txt")
        
        answer = input("\n是否继续尝试运行? (y/n): ").strip().lower()
        return answer == 'y'
    
    return True

def run_script(script_name):
    """导入并运行指定脚本的main函数"""
    try:
        # 动态导入脚本模块
        script_module = importlib.import_module(f"scripts.{script_name}")
        
        # 运行main函数
        if hasattr(script_module, 'main'):
            script_module.main()
        else:
            print(f"错误: {script_name}.py 中没有找到main函数")
        
        return True
    except ImportError as e:
        if "No module named" in str(e):
            missing_module = str(e).split("'")[-2]
            print(f"错误: 缺少必要的Python模块: {missing_module}")
            print("请确保已激活虚拟环境并安装了所有依赖项:")
            print("  1. 激活虚拟环境: venv\\Scripts\\activate.bat")
            print("  2. 安装依赖项: pip install -r requirements.txt")
        else:
            print(f"运行 {script_name}.py 时发生错误: {e}")
        return False
    except Exception as e:
        print(f"运行 {script_name}.py 时发生错误: {e}")
        return False

def main():
    """主函数：按顺序运行生成和可视化脚本"""
    # 设置结果目录
    results_dir = 'results/visualizations'
    Path(results_dir).mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 80)
    print("开始一键式处理：生成干扰图片并创建可视化")
    print("=" * 80)
    
    # 检查依赖项
    if not check_dependencies():
        print("\n由于缺少依赖项，已取消运行。")
        print("请激活虚拟环境并安装所需依赖项后重试。")
        return
    
    # 第1步：生成干扰图片
    print("\n步骤 1: 生成干扰图片...")
    success = run_script("generate_test_images")
    if not success:
        print("生成干扰图片失败，但仍将继续尝试可视化...")
    
    # 等待1秒，确保文件系统操作完成
    time.sleep(1)
    
    # 第2步：为所有图片生成可视化预测结果
    print("\n步骤 2: 为原始图片和所有干扰图片生成可视化预测结果...")
    visualize_success = run_script("visualize_all")
    
    print("\n" + "=" * 80)
    if visualize_success:
        print(f"处理完成！可视化结果保存在 {os.path.abspath('results/all_visualizations')} 目录")
    else:
        print("警告: 可视化步骤失败。请确保已激活虚拟环境并安装所有依赖项。")
    print("=" * 80)

if __name__ == "__main__":
    main() 