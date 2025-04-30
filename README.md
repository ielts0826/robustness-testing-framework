# AI模型鲁棒性测试框架

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/PyTorch-1.7%2B-orange.svg" alt="PyTorch 1.7+">
  <img src="https://img.shields.io/badge/pytest-6.0%2B-green.svg" alt="pytest 6.0+">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
</div>

这个项目提供了一个自动化测试框架，用于评估预训练图像分类模型的鲁棒性、确定性和合理性。该框架主要使用PyTorch和pytest实现，可以帮助开发者和研究人员验证模型在各种情况下的稳定性和可靠性。

<div align="center">
  <img src="results/sample_prediction.png" alt="示例预测结果" width="600">
</div>

## ✨ 功能特点

- **模型加载与推理**：加载预训练的ResNet-18模型并执行图像分类
- **图像预处理**：标准化的图像加载和预处理流程
- **鲁棒性测试**：测试模型对异常输入和边缘情况的处理能力
- **确定性测试**：验证模型在相同输入下产生一致的输出
- **批次不变性测试**：确保模型在不同批次大小下表现一致
- **自动化生成测试数据**：脚本化生成各种测试用的图像
- **可视化预测结果**：生成图像分类预测结果的可视化图表
- **自动化测试报告**：生成详细的HTML和文本格式测试报告
- **一键演示**：提供一键运行全部功能的演示脚本

## 项目进度

### 第一步：项目初始化与环境设置 ✓

- 创建项目文件夹结构：
  - `data/`: 存放测试图片
  - `scripts/`: 存放自动化脚本
  - `src/`: 存放源代码
  - `tests/`: 存放测试代码
- 创建环境设置脚本 `scripts/setup_env.bat`：
  - 检查Python安装
  - 创建虚拟环境
  - 升级pip
  - 安装依赖包
- 创建`requirements.txt`文件，包含以下依赖：
  - torch: PyTorch深度学习框架
  - torchvision: 提供计算机视觉相关的工具和预训练模型
  - pytest: Python测试框架
  - Pillow: 图像处理库
  - pyyaml: YAML文件解析库

### 第二步：加载预训练模型并进行简单推理 ✓

- 创建 `src/inference_runner.py` 文件：
  - 实现 `ImageClassifier` 类，用于加载预训练的ResNet-18模型
  - 实现 `run_inference` 方法，使用no_grad模式运行模型推理
- 创建 `quick_test.py` 测试脚本：
  - 加载 `ImageClassifier` 实例
  - 创建随机张量作为模型输入
  - 运行推理并验证输出形状
  - 展示前5个预测类别

### 第三步：学习加载和预处理真实图像 ✓

- 更新 `src/inference_runner.py` 文件：
  - 导入 PIL 和 torchvision.transforms
  - 添加 `preprocess` 预处理流程：
    - Resize(256) - 缩放图像
    - CenterCrop(224) - 中心裁剪
    - ToTensor() - 转换为张量
    - Normalize() - 使用ImageNet均值和标准差标准化
  - 实现 `load_and_preprocess_image` 方法：
    - 加载图像并转换为RGB格式
    - 应用预处理流程
    - 添加批次维度
- 更新 `quick_test.py` 测试脚本：
  - 修改为使用真实图像而非随机张量
  - 增加图像路径检查
  - 添加异常处理
  - 使用softmax转换输出为概率
  - 改进结果展示格式

### 第四步：编写第一个pytest测试用例 ✓

- 创建 `tests/test_inference.py` 文件：
  - 定义 `TestImageClassifier` 测试类
  - 实现 `test_model_loading` 测试方法：验证模型能否正确加载
  - 实现 `test_successful_inference` 测试方法：
    - 验证模型能否成功对真实图像进行推理
    - 检查输入张量的形状
    - 检查输出张量的形状
    - 验证输出不含NaN且不全为0
- 创建 `tests/conftest.py` 文件：
  - 定义 `classifier` fixture，提供可重用的 `ImageClassifier` 实例
  - 减少测试代码重复

### 第五步：准备更多测试数据 ✓

- 创建 `scripts/generate_test_images.py` 脚本：
  - 生成多种类型的测试图像：
    - 纯黑图像 (`black.jpg`)
    - 纯白图像 (`white.jpg`)
    - 随机噪声图像 (`noise.jpg`)
  - 创建非图像文件 (`dummy.txt`) 用于测试非法输入处理
- 这些图像将用于以下测试：
  - 边缘情况图像测试
  - 模型鲁棒性测试
  - 非法输入处理测试

### 第六步：实现鲁棒性测试用例 ✓

- 在 `tests/test_inference.py` 中添加以下测试用例：
  - `test_invalid_image_path`：测试处理不存在的文件路径
    - 验证加载不存在的图像时会抛出 `FileNotFoundError`
  - `test_non_image_file`：测试处理非图像文件
    - 验证加载文本文件时会抛出异常
  - `test_edge_case_images_run_without_error`：测试边缘情况图像
    - 使用参数化测试处理多种特殊图像（黑、白、噪声）
    - 验证模型不会在处理这些图像时崩溃
    - 确保输出形状正确且不包含NaN值
  - `test_input_size_boundaries`：测试极端尺寸的图像
    - 测试非常小的图像（16x16像素）
    - 测试非常大的图像（4000x3000像素）
    - 验证预处理管道能正确处理这些图像

### 第七步：实现确定性测试 ✓

- 在 `tests/test_inference.py` 中添加以下测试用例：
  - `test_inference_determinism`：测试推理的确定性
    - 对相同图像进行两次加载和推理
    - 验证两次得到的输出张量完全相同
    - 验证预处理管道的确定性（相同输入产生相同的预处理输出）
  - `test_inference_determinism_with_reloading`：测试模型重新加载后的确定性
    - 使用两个独立创建的分类器实例
    - 验证两个实例对同一输入产生相同的输出
    - 确保模型加载是确定性的
  - `test_batch_invariance`：测试批次不变性
    - 验证单张图像与批次中的同一图像得到相同的结果
    - 测试批次处理与单张图像处理的一致性
    - 确保模型在不同批次大小下行为一致

### 第八步：编写完整文档 ✓

- 完善 `README.md` 文件：
  - 添加项目概述和功能特点
  - 详细的安装和使用说明
  - 文件结构说明
  - 测试用例说明
  - 开发和扩展指南

### 第九步：代码整理与优化 ✓

- 优化 `src/inference_runner.py`：
  - 添加更详细的模块和函数文档字符串
  - 支持不同模型的加载（通过参数指定）
  - 添加输入验证和错误处理
  - 实现 `get_top_predictions` 方法用于获取前K个预测结果
- 优化 `tests/conftest.py`：
  - 添加更多实用的fixtures：
    - `test_image_path`：提供标准测试图像路径
    - `processed_test_image`：提供预处理后的测试图像
    - `edge_case_image_path`：提供边缘情况测试图像，使用参数化
    - `temp_images`：在测试过程中创建临时图像并在测试后清理
- 重构 `tests/test_inference.py`：
  - 使用新的fixtures简化测试代码
  - 移除重复代码，提高可维护性
- 优化 `quick_test.py`：
  - 使用新增的 `get_top_predictions` 方法
  - 改进错误处理

## 安装与使用

### 环境要求

- Python 3.7+
- PyTorch 1.7+
- CUDA（可选，用于GPU加速）

### 安装步骤

1. 克隆仓库：
   ```
   git clone https://github.com/yourusername/ai_model_tester.git
   cd ai_model_tester
   ```

2. 运行环境设置脚本：
   ```
   cd scripts
   setup_env.bat
   ```

3. 激活虚拟环境：
   ```
   ..\venv\Scripts\activate.bat
   ```

4. 安装项目（开发模式）：
   ```
   cd ..
   pip install -e .
   ```

### 运行演示

一键运行完整演示（测试、可视化和报告生成）：
```
python scripts/run_demo.py
```

### 运行测试

在项目根目录下，运行以下命令执行所有测试：
```
pytest
```

生成测试报告：
```
python scripts/generate_test_report.py
```

### 生成预测可视化

为测试图像生成可视化预测结果：
```
python scripts/visualize_predictions.py
```

### 简单测试

要快速测试ResNet模型的推理功能，请在项目根目录下运行：
```
python quick_test.py
```

注意：确保在`data`目录中有一张名为`cat.jpg`的图片。

## 文件结构

```
ai_model_tester/
├── data/                      # 测试图像和数据
│   ├── cat.jpg                # 用户提供的测试图像
│   ├── black.jpg              # 生成的纯黑图像
│   ├── white.jpg              # 生成的纯白图像
│   ├── noise.jpg              # 生成的噪声图像
│   └── dummy.txt              # 非图像文件
├── scripts/                   # 脚本文件
│   ├── setup_env.bat          # 环境设置脚本
│   ├── generate_test_images.py # 测试图像生成脚本
│   ├── visualize_predictions.py # 预测可视化脚本
│   ├── generate_test_report.py # 测试报告生成脚本
│   └── run_demo.py            # 一键演示脚本
├── src/                       # 源代码
│   └── inference_runner.py    # 模型加载和推理实现
├── tests/                     # 测试代码
│   ├── conftest.py            # pytest配置和fixtures
│   └── test_inference.py      # 推理测试用例
├── results/                   # 结果输出目录
│   └── prediction_vis_*.png   # 生成的预测可视化
├── reports/                   # 测试报告目录
│   ├── test_report_*.html     # HTML格式测试报告
│   └── test_report_*.txt      # 文本格式测试报告
├── requirements.txt           # 项目依赖
├── setup.py                   # 项目安装配置
├── quick_test.py              # 快速测试脚本
└── README.md                  # 项目文档
```

## 测试用例说明

### 基础功能测试
- 测试模型加载
- 测试基本推理功能

### 鲁棒性测试
- 无效文件路径处理
- 非图像文件处理
- 边缘情况图像处理（黑、白、噪声）
- 极端尺寸图像处理（非常小/大）

### 确定性测试
- 相同输入多次推理
- 模型重新加载后的一致性
- 批次大小不变性

## 开发和扩展指南

### 添加新测试
1. 在`tests/test_inference.py`中添加新的测试方法
2. 使用pytest的装饰器（如`@pytest.mark.parametrize`）实现参数化测试
3. 利用`classifier` fixture减少代码重复

### 支持新的模型
1. 修改`src/inference_runner.py`中的`ImageClassifier`类
2. 更新预处理流程以匹配新模型的要求
3. 调整测试用例中的预期输出形状

### 改进测试框架
1. 添加更多边缘情况的测试数据
2. 实现对抗样本测试
3. 添加性能测试用例
4. 实现覆盖率报告

## 如何贡献

我们欢迎任何形式的贡献！

1. Fork 这个仓库
2. 创建您的特性分支：`git checkout -b feature/amazing-feature`
3. 提交您的更改：`git commit -m 'Add some amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 提交Pull Request

## 项目展示

演示脚本会生成以下展示资源：

1. **可视化预测结果**：位于`results/`目录，展示模型对图像的分类结果
2. **HTML测试报告**：位于`reports/`目录，包含详细的测试结果和统计信息
3. **文本测试报告**：位于`reports/`目录，包含原始测试输出

这些资源是GitHub项目展示的理想材料，可以在README和项目Wiki中引用。

## 许可证

该项目采用MIT许可证 - 详情请查看 [LICENSE](LICENSE) 文件

## 联系方式

项目维护者 - 您的姓名 - your.email@example.com

项目链接: [https://github.com/yourusername/ai_model_tester](https://github.com/yourusername/ai_model_tester) 