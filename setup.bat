@echo off
REM Setup script para Pictionary Live
REM Crea un entorno virtual y instala las dependencias

echo ========================================
echo Pictionary Live - Configuracion Inicial
echo ========================================
echo.

REM Verificar version de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo Por favor instale Python 3.10, 3.11 o 3.12
    pause
    exit /b 1
)

echo [1/4] Verificando version de Python...
python -c "import sys; exit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ADVERTENCIA: Se recomienda Python 3.10-3.12 para MediaPipe
    echo Version actual:
    python --version
    echo.
    echo Puede continuar pero la deteccion de manos podria no funcionar.
    echo Presione Ctrl+C para cancelar o cualquier tecla para continuar...
    pause >nul
)

echo [2/4] Creando entorno virtual en 'venv'...
if exist venv (
    echo El entorno virtual ya existe. Eliminando...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: No se pudo crear el entorno virtual
    echo Asegurese de tener instalado el modulo venv
    pause
    exit /b 1
)

echo [3/4] Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: No se pudo activar el entorno virtual
    pause
    exit /b 1
)

echo [4/4] Instalando dependencias desde requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ADVERTENCIA: Hubo errores durante la instalacion
    echo Revise los mensajes anteriores para mas detalles
    echo.
) else (
    echo.
    echo ========================================
    echo INSTALACION COMPLETADA EXITOSAMENTE
    echo ========================================
    echo.
    echo Para usar la aplicacion:
    echo   1. Active el entorno virtual: venv\Scripts\activate.bat
    echo   2. Ejecute la aplicacion: python main.py
    echo.
    echo O simplemente ejecute: run.bat
    echo.
)

pause
