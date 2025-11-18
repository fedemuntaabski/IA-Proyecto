@echo off
echo ========================================
echo Quick Draw Challenge - Starting Frontend
echo ========================================
cd frontend
echo Installing/Updating dependencies...
call npm install
echo.
echo Starting React development server...
call npm run dev
