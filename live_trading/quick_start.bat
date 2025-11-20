@echo off
REM Windows 快速启动脚本

echo ========================================
echo Binance 实时交易信号系统
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
pip show ccxt >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装 ccxt...
    pip install ccxt
)

pip show pandas >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装 pandas...
    pip install pandas
)

pip show pyyaml >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装 pyyaml...
    pip install pyyaml
)

echo.
echo [2/3] 运行测试...
python test_signal.py

echo.
echo [3/3] 启动实时信号系统...
echo.
echo 选择运行模式:
echo   1. 单次查询（查看当前信号）
echo   2. 持续运行（每小时更新）
echo   3. 快速更新（每5分钟更新）
echo.
set /p choice="请选择 (1/2/3): "

if "%choice%"=="1" (
    python binance_live_signal.py --once
) else if "%choice%"=="2" (
    python binance_live_signal.py
) else if "%choice%"=="3" (
    python binance_live_signal.py --interval 300
) else (
    echo [错误] 无效选择
)

pause
