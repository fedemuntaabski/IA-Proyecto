@echo off
echo ========================================
echo Quick Draw Challenge - Starting Backend
echo ========================================
cd backend
call venv\Scripts\activate
echo Installing/Updating dependencies...
pip install -r requirements.txt
echo.
echo Starting Flask server with hand tracking...
python app.py
