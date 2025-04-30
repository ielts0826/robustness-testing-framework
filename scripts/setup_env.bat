@echo off
echo 正在检查Python是否已安装...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python，请安装Python 3.x后再运行此脚本。
    exit /b 1
)

echo 正在创建虚拟环境...
cd ..
python -m venv venv

echo 正在升级pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo 正在安装依赖包...
pip install -r requirements.txt

echo 环境设置脚本执行完毕。
echo 请手动激活虚拟环境，运行以下命令:
echo venv\Scripts\activate.bat

exit /b 0 