@echo off
echo ========================================
echo Building CUDA Extension for Windows
echo ========================================
echo.

REM Set DISTUTILS_USE_SDK to avoid multiple VC env activations
set DISTUTILS_USE_SDK=1
echo DISTUTILS_USE_SDK set to: %DISTUTILS_USE_SDK%

REM Set CUDA_PATH if not already set
if not defined CUDA_PATH (
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
    echo CUDA_PATH set to: %CUDA_PATH%
) else (
    echo CUDA_PATH already set to: %CUDA_PATH%
)

REM Verify CUDA installation
echo.
echo Checking CUDA installation...
nvcc --version
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found! Make sure CUDA is installed and in PATH.
    pause
    exit /b 1
)

echo.
echo Building CUDA extension...
echo This may take 5-10 minutes on first build...
echo.

REM Clean previous builds (optional)
if exist build rmdir /s /q build
if exist cuda_ops*.pyd del /q cuda_ops*.pyd

REM Build the extension
python setup.py build_ext --inplace

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo Build successful!
    echo ========================================
    echo.
    echo Testing the extension...
    python test_cuda.py
) else (
    echo.
    echo ========================================
    echo Build failed!
    echo ========================================
    echo.
    echo Common issues:
    echo 1. Make sure you're using x64 Native Tools Command Prompt
    echo 2. Check that CUDA_PATH is set correctly
    echo 3. Verify Visual Studio Build Tools are installed
    echo 4. Make sure PyTorch with CUDA is installed
)

echo.
pause