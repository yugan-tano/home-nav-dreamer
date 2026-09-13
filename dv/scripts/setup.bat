@echo off
chcp 65001 >nul
echo ============================================
echo  DreamerV3 居家服务机器人 - 环境安装脚本
echo  (Windows + RTX 4060)
echo ============================================
echo.

:: Step 1: 检查 Python
echo [1/4] 检查 Python ...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python, 请先安装 Python 3.11+
    pause
    exit /b 1
)
python --version
echo.

:: Step 2: 创建虚拟环境
echo [2/4] 创建虚拟环境 ...
if not exist "%~dp0..\..\..\venv_dreamer" (
    python -m venv "%~dp0..\..\..\venv_dreamer"
    echo 虚拟环境已创建: venv_dreamer
) else (
    echo 虚拟环境已存在, 跳过创建
)
echo.

:: Step 3: 激活虚拟环境并安装依赖
echo [3/4] 安装依赖 (这可能需要5-10分钟) ...
call "%~dp0..\..\..\venv_dreamer\Scripts\activate.bat"
pip install --upgrade pip -q
pip install jax[cuda12]==0.4.33 -q
pip install -r "%~dp0..\requirements.txt" -q
pip install gradio matplotlib gymnasium pillow -q
echo 依赖安装完成
echo.

:: Step 4: 设置环境变量
echo [4/4] 配置环境变量 (RTX 4060) ...
set XLA_PYTHON_CLIENT_PREALLOCATE=false
set XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
set JAX_COMPILATION_CACHE_DIR=%USERPROFILE%\.jax_cache
set JAX_PLATFORM_NAME=cuda
set CUDA_VISIBLE_DEVICES=0
echo 环境变量已设置:
echo   PREALLOCATE = false
echo   MEM_FRACTION = 0.80
echo   CUDA_DEVICE = 0
echo.

echo ============================================
echo  安装完成!
echo ============================================
echo.
echo 下一步:
echo   1. 验证环境: scripts\quick_test.bat
echo   2. 启动演示: scripts\run_demo.bat
echo   3. 开始训练: scripts\run_train.bat
echo.
pause
