@echo off
echo Installing backend dependencies...
echo.

cd /d "%~dp0backend"
pip install -r requirements.txt

echo.
echo Done! Now installing frontend dependencies...
echo.

cd /d "%~dp0frontend"
call npm install

echo.
echo All dependencies installed!
echo You can now run run_all.bat to start the game.
pause
