@echo off
title Aircraft Classifier - Setup
cd /d "%~dp0"

echo ============================================
echo   Aircraft Classifier - First-time setup
echo ============================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python was not found on this computer.
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo IMPORTANT: during install, check the box "Add Python to PATH"
    echo Then run this file again.
    pause
    exit /b 1
)

if not exist "venv\" (
    echo Creating a private Python environment for this app...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Checking dependencies (this can take a few minutes the first time)...
python -c "import torch, torchvision, PIL" >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing torch, torchvision, pillow...
    pip install --quiet torch torchvision pillow
)

echo.
echo Starting Aircraft Classifier...
python predict_gui.py

pause
