@echo off
echo Starting Pinturillo Paint Game...
echo.

REM Start backend in a new window
start "Backend Server" cmd /k "cd /d "%~dp0backend" && python app.py"

REM Wait a moment for backend to initialize
timeout /t 3 /nobreak >nul

REM Start frontend in a new window
start "Frontend Server" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo Both servers are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Close this window when done, or close the individual server windows to stop them.
