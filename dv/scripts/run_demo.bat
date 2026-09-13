@echo off
chcp 65001 >nul
echo ============================================
echo  启动 DreamerV3 演示界面
echo  浏览器打开: http://localhost:7860
echo ============================================
echo.

call "%~dp0..\..\..\venv_dreamer\Scripts\activate.bat"
set XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /d "%~dp0.."

echo 启动 Gradio 界面...
python demo/app.py --logdir "%USERPROFILE%/logdir/dreamer_home" --port 7860

pause
