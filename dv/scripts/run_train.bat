@echo off
chcp 65001 >nul
echo ============================================
echo  DreamerV3 居家导航训练 (RTX 4060 8GB)
echo  预计时间: 50万步 ~6-8小时
echo ============================================
echo.

call "%~dp0..\..\..\venv_dreamer\Scripts\activate.bat"

:: RTX 4060 优化环境变量
set XLA_PYTHON_CLIENT_PREALLOCATE=false
set XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
set JAX_COMPILATION_CACHE_DIR=%USERPROFILE%\.jax_cache
set JAX_PLATFORM_NAME=cuda
set CUDA_VISIBLE_DEVICES=0

cd /d "%~dp0.."

echo 开始训练 (home_quick 配置: 30万步, ~3-5小时) ...
echo 训练期间可另外开终端运行 run_demo.bat 查看实时曲线
echo.

python main.py ^
  --configs home4060 home_quick ^
  --logdir "%USERPROFILE%/logdir/dreamer_home"

echo.
echo 训练完成! 日志保存在 %USERPROFILE%/logdir/dreamer_home
pause
