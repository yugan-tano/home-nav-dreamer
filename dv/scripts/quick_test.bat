@echo off
chcp 65001 >nul
echo ============================================
echo  DreamerV3 快速环境验证
echo ============================================
echo.

call "%~dp0..\..\..\venv_dreamer\Scripts\activate.bat"
set XLA_PYTHON_CLIENT_PREALLOCATE=false
set JAX_PLATFORM_NAME=cuda

echo [1/3] 检查 JAX + GPU ...
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices()); print('GPU OK!' if jax.devices()[0].platform == 'gpu' else 'CPU mode')"

echo.
echo [2/3] 测试居家环境渲染 ...
cd /d "%~dp0.."
python -c "from demo.env_home import HomeEnv; import numpy as np; env = HomeEnv(layout='small', size=64); obs, _ = env.reset(); print('Obs shape:', obs.shape); obs2, r, _, _, _ = env.step(1); print('Step reward:', r); print('环境渲染 OK!')"

echo.
echo [3/3] 验证 DreamerV3 Debug 模式 (CPU, 5000步, ~30秒) ...
python main.py --configs debug_home --logdir "%USERPROFILE%/logdir/test_verify"
if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo  全部验证通过! 可以开始正式训练.
    echo ============================================
) else (
    echo.
    echo [警告] Debug 训练有问题, 查看上方日志排查.
)

pause
