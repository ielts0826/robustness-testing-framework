"""
生成详细的测试报告

此脚本运行所有测试并生成详细的HTML和文本报告，适合在GitHub上展示
"""

import os
import sys
import pytest
import datetime
from pathlib import Path

def run_tests_with_report():
    """运行所有测试并生成HTML和文本报告"""
    # 确保项目根目录下存在reports目录
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 定义报告文件路径
    html_report = report_dir / f"test_report_{timestamp}.html"
    txt_report = report_dir / f"test_report_{timestamp}.txt"
    
    # 运行pytest并生成HTML报告
    print(f"正在运行测试并生成报告...")
    html_args = [
        "--html", str(html_report),
        "--self-contained-html",
        "-v"
    ]
    
    # 运行pytest并重定向输出到文本文件
    with open(txt_report, 'w', encoding='utf-8') as f:
        # 将原始输出保存到文本文件
        stdout_backup = sys.stdout
        sys.stdout = f
        
        # 添加测试报告标题
        print("=" * 80)
        print(f"AI模型测试框架 - 测试报告")
        print(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print("")
        
        # 运行测试
        pytest.main(["-v"])
        
        # 恢复标准输出
        sys.stdout = stdout_backup
    
    # 运行HTML报告
    pytest.main(html_args)
    
    print(f"测试报告已生成:")
    print(f"- HTML报告: {html_report}")
    print(f"- 文本报告: {txt_report}")
    
    return str(html_report), str(txt_report)

if __name__ == "__main__":
    html_path, txt_path = run_tests_with_report() 