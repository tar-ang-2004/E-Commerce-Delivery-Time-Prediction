@echo off
echo ========================================
echo   DeliveryAI - Startup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [INFO] Python found
echo [INFO] Setting up DeliveryAI application...
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo [INFO] Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
if not exist "models" mkdir models
if not exist "mlruns" mkdir mlruns
if not exist "logs" mkdir logs
if not exist "static\css" mkdir static\css
if not exist "static\js" mkdir static\js

REM Copy environment file if it doesn't exist
if not exist ".env" (
    echo [INFO] Creating environment file...
    copy .env.example .env
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo The application is ready to start.
echo.
echo To start the application:
echo   python app.py
echo.
echo To start MLflow UI (in another terminal):
echo   mlflow ui --backend-store-uri file:./mlruns
echo.
echo Application URL: http://localhost:5000
echo MLflow UI URL: http://localhost:5000 (when running)
echo.

REM Ask if user wants to start the application
set /p start_app="Start the application now? (y/n): "
if /i "%start_app%"=="y" (
    echo.
    echo [INFO] Starting DeliveryAI application...
    echo [INFO] Application available at: http://localhost:5000
    echo [INFO] Press Ctrl+C to stop the application
    echo.
    python app.py
)

pause