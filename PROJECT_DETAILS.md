# AI 模型鲁棒性测试框架 - 详细说明

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是一个用于测试 AI 图像分类模型（当前使用 ResNet18）鲁棒性的框架。它可以对输入图像应用多种扰动，评估模型在这些扰动下的表现，并生成详细的测试报告和可视化结果。

## 项目目标

本项目旨在提供一个易于使用和扩展的框架，用于：

1.  **评估模型稳定性**：测试模型在面对常见的图像扰动（如噪声、模糊、亮度变化等）时的预测一致性。
2.  **识别模型弱点**：发现模型容易出错的特定类型的图像或扰动。
3.  **量化鲁棒性**：通过确定性测试和扰动测试，提供模型鲁棒性的量化指标。
4.  **结果可视化**：生成直观的图表和报告，帮助理解模型的预测行为。

## 项目结构

```
ai_model_tester/
├── data/                  # 存放原始图像和类别文件
│   └── cat.jpg            # 示例输入图像
├── scripts/               # 存放主要的执行脚本
│   ├── demo.py            # 一键运行所有测试、报告和可视化的演示脚本
│   ├── generate_test_images.py # 生成带扰动的测试图像
│   ├── quick_test.py      # 对单个图像进行快速推理测试
│   ├── report_generator.py # 生成 HTML 和 Markdown 格式的测试报告
│   ├── run_all_tests.py   # 运行确定性测试和鲁棒性测试
│   ├── setup_env.bat      # (Windows) 用于创建虚拟环境和安装依赖的批处理脚本
│   └── visualize_predictions.py # 可视化单个图像的预测结果
├── src/                   # 存放核心源代码
│   ├── __init__.py
│   └── inference_runner.py # 核心类，负责加载模型、图像处理、执行推理
├── tests/                 # 存放测试逻辑脚本
│   ├── __init__.py
│   ├── test_deterministic.py # 确定性测试：验证模型对同一输入是否总产生相同输出
│   └── test_robustness.py  # 鲁棒性测试：评估模型在图像扰动下的表现
├── .gitignore             # 指定 Git 应忽略的文件和目录
├── LICENSE                # 项目许可证文件 (MIT)
├── README.md              # 项目说明文件 (简版)
├── PROJECT_DETAILS.md     # 项目说明文件 (详细版 - 本文档)
├── requirements.txt       # Python 依赖库列表
└── setup.py               # 项目打包配置文件，用于开发模式安装
```

## 核心组件

### 1. 模型 (ResNet18)

*   **模型**: 使用 PyTorch Hub 提供的预训练 `ResNet18` 模型。
*   **权重**: 在 ImageNet 数据集上预训练。
*   **作用**: 作为图像分类器，对输入图像进行预测。模型在首次运行时会自动下载。
*   **可扩展性**: `src/inference_runner.py` 可以被修改以支持加载和使用其他图像分类模型。

### 2. 脚本 (`scripts/` & `src/`)

*   `setup_env.bat`: (Windows) 自动化创建 Python 虚拟环境 (`.venv`) 并使用 `pip` 安装 `requirements.txt` 中的所有依赖项，最后以开发模式安装本项目。
*   `src/inference_runner.py`:
    *   `ImageClassifier` 类是框架的核心。
    *   负责加载 ResNet18 模型和权重。
    *   提供图像加载和预处理方法 (`load_and_preprocess_image`)。
    *   执行模型推理 (`run_inference`)。
    *   获取 Top-K 预测结果，并可选择映射到类别名称 (`get_top_predictions`)。
    *   包含用于生成扰动图像的辅助函数 (`apply_perturbations`)。
*   `scripts/quick_test.py`:
    *   一个简单的示例，演示如何使用 `ImageClassifier` 对 `data/cat.jpg` 进行单次预测。
    *   输出 Top-5 预测类别及其概率。
*   `scripts/generate_test_images.py`:
    *   读取输入图像（如 `data/cat.jpg`）。
    *   应用一系列预定义的扰动（如高斯噪声、模糊、亮度调整）。
    *   将生成的扰动图像保存到 `data/test_images/` 目录（此目录被 `.gitignore` 忽略）。
*   `scripts/run_all_tests.py`:
    *   编排测试流程。
    *   调用 `tests/test_deterministic.py` 来执行确定性测试。
    *   调用 `tests/test_robustness.py` 来对原始图像和生成的扰动图像进行预测，评估鲁棒性。
    *   将测试结果保存为 JSON 文件，供报告生成器使用。
*   `scripts/report_generator.py`:
    *   读取由 `run_all_tests.py` 生成的 JSON 测试结果。
    *   生成易于阅读的 HTML 和 Markdown 格式的测试报告，保存在 `reports/` 目录（此目录被 `.gitignore` 忽略）。
    *   报告总结了确定性测试结果和鲁棒性测试中模型预测的变化情况。
*   `scripts/visualize_predictions.py`:
    *   加载一张图像（默认为 `data/cat.jpg`）。
    *   使用 `ImageClassifier` 进行预测。
    *   生成一张包含原始图像和 Top-K 预测概率条形图的可视化图片。
    *   支持中文标签显示。
    *   将可视化结果保存到 `results/` 目录（此目录被 `.gitignore` 忽略）。
*   `scripts/demo.py`:
    *   提供一个简单的一键式入口点。
    *   依次执行：图像生成、所有测试、报告生成、预测可视化。
    *   方便快速演示整个框架的功能。

### 3. 测试 (`tests/`)

*   `test_deterministic.py`:
    *   多次加载同一图像并进行预测。
    *   验证模型对于完全相同的输入，是否每次都产生完全相同的预测输出。
    *   测试结果用于评估模型的基础稳定性。
*   `test_robustness.py`:
    *   加载原始图像和 `data/test_images/` 中的所有扰动图像。
    *   对每张图像进行预测。
    *   比较模型对原始图像和扰动图像的预测结果。
    *   评估模型在不同扰动下的预测一致性和准确性变化。

## 安装与运行

### 先决条件

*   Python 3.8 或更高版本
*   `pip` (Python 包管理器)

### 安装步骤

1.  **克隆仓库 (如果从 Git 获取)**:
    ```bash
    git clone <repository-url>
    cd ai-model-tester
    ```
2.  **设置环境 (Windows)**:
    *   双击运行 `setup_env.bat`。这将自动创建虚拟环境 `.venv`，激活它，安装所有依赖项，并以开发模式安装项目。
3.  **设置环境 (Linux/macOS - 手动)**:
    ```bash
    # 创建虚拟环境
    python -m venv .venv
    # 激活虚拟环境
    source .venv/bin/activate  # Linux/macOS
    # 或者 .venv\Scripts\activate # Windows Git Bash / PowerShell

    # 安装依赖
    pip install -r requirements.txt

    # 以开发模式安装项目 (允许在不重新安装的情况下修改源代码)
    pip install -e .
    ```

### 运行

*   **运行完整演示**:
    ```bash
    python scripts/demo.py
    ```
    这将按顺序执行所有主要步骤：生成测试图像 -> 运行测试 -> 生成报告 -> 生成可视化。

*   **运行单个脚本**:
    ```bash
    # 运行快速测试
    python scripts/quick_test.py

    # 生成扰动图像
    python scripts/generate_test_images.py

    # 运行所有测试 (确定性 + 鲁棒性)
    python scripts/run_all_tests.py

    # 生成测试报告 (需要先运行 run_all_tests.py)
    python scripts/report_generator.py

    # 生成预测可视化
    python scripts/visualize_predictions.py
    ```
    *(注意: 运行单个脚本前，请确保虚拟环境已激活)*

## 理解结果

*   **测试报告 (`reports/`)**:
    *   查看生成的 `.html` 或 `.md` 文件。
    *   报告会显示确定性测试是否通过，以及鲁棒性测试中模型对不同扰动图像的预测结果和与原始图像预测结果的对比。
*   **可视化 (`results/`)**:
    *   查看生成的 `.png` 文件。
    *   图像左侧是输入图片，右侧是模型预测的 Top-K 类别及其置信度（概率）条形图。

## 许可证

本项目采用 [MIT License](LICENSE) 授权。 