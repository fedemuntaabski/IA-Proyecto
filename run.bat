@echo off
REM Launcher para Pictionary Live con Python 3.12 (compatible con MediaPipe)
echo ========================================
echo   Pictionary Live - Iniciando...
echo ========================================
echo.
echo Usando Python 3.12 para compatibilidad con MediaPipe
echo.

py -3.12 main.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: La aplicacion fallo con codigo %ERRORLEVEL%
    echo.
    pause
)
