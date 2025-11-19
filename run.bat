@echo off
REM Script para ejecutar Pictionary Live con el entorno virtual

REM Verificar si existe el entorno virtual
if not exist venv (
    echo ERROR: Entorno virtual no encontrado
    echo Por favor ejecute setup.bat primero para crear el entorno virtual
    echo.
    pause
    exit /b 1
)

REM Activar el entorno virtual y ejecutar la aplicacion
call venv\Scripts\activate.bat
python main.py %*
