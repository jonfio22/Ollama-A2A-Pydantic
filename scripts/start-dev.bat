@echo off
REM Development startup script for A2A Multi-Agent System (Windows)

echo.
echo 🚀 A2A Agent Startup Script (Windows)
echo =====================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo    Run: scripts\setup.bat
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if Ollama is running
echo 🔍 Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Ollama is not running
    echo    Start with: ollama serve
    exit /b 1
)
echo ✅ Ollama is running
echo.

echo 📋 Start each agent in a separate terminal:
echo.
echo Terminal 1 (Orchestrator - port 8000):
echo   uvicorn main:orchestrator_app --port 8000 --reload
echo.
echo Terminal 2 (Analyst - port 8001):
echo   uvicorn main:analyst_app --port 8001 --reload
echo.
echo Terminal 3 (Coder - port 8002):
echo   uvicorn main:coder_app --port 8002 --reload
echo.
echo Terminal 4 (Validator - port 8003):
echo   uvicorn main:validator_app --port 8003 --reload
echo.
echo Terminal 5 (Vision - port 8004):
echo   uvicorn main:vision_app --port 8004 --reload
echo.
echo 🧪 Test with:
echo   curl http://localhost:8000/health
echo.
echo Or use Docker Compose:
echo   docker-compose up -d
echo.
pause
